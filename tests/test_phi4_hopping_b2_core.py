"""Synthetic b=2 / deeper-chain checks; not physical sampling or training."""
import pytest
import torch

from recurrent_rg.phi4_aligned_qcr import (
    FEATURE_DIM,
    AlignedPhi4QCR,
    _Block2GaussianCoarseGrainer,
    _aligned_information,
    _sample_aligned_regions,
    aligned_objectives,
)
from recurrent_rg.phi4_qcr import GaussianCoarseGrainer, gaussian_bin_probabilities


@pytest.fixture(autouse=True)
def isolated_rng_and_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        yield
    torch.set_num_threads(previous)


def test_default_geometry_retains_historical_initialization_and_state_dict():
    torch.manual_seed(29)
    default = AlignedPhi4QCR(1.3, hidden=6, theory_dim=2)
    torch.manual_seed(29)
    explicit = AlignedPhi4QCR(1.3, hidden=6, theory_dim=2, block_size=3, steps=2)
    assert type(default.C) is GaussianCoarseGrainer
    assert default.block_size == 3 and default.steps == 2
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any(key.endswith(("steps", "block_size")) for key in default.state_dict())
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(1, 2, 18, 18, 1, dtype=torch.double)
    actual = default.coarse_grain(x, torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(7)
    first = default.C(x, generator)
    expected = [x, first, default.C(first, generator)]
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_b2_three_steps_cover_128_64_32_16_without_changing_q_or_r_capacity():
    model = AlignedPhi4QCR(1., hidden=6, theory_dim=1,
                          block_size=2, steps=3, patch_chunk_size=128)
    reference = AlignedPhi4QCR(1., hidden=6, theory_dim=1)
    assert model.C.net[0].in_features == 4
    assert model.Q.local_mlp[0].in_features == 9
    assert model.Q.decoder.out_features == FEATURE_DIM == 625
    assert [p.shape for p in model.Q.parameters()] == [p.shape for p in reference.Q.parameters()]
    assert [p.shape for p in model.R.parameters()] == [p.shape for p in reference.R.parameters()]
    assert list(model.C.state_dict()) == list(reference.C.state_dict())
    assert model.C.noise_ratio.item() == .1
    x = torch.randn(1, 2, 128, 128, 1, dtype=torch.double)
    levels = model.coarse_grain(x, torch.Generator().manual_seed(17))
    assert [tuple(value.shape) for value in levels] == [
        (1, 2, length, length, 1) for length in (128, 64, 32, 16)]
    repeated = model.coarse_grain(x, torch.Generator().manual_seed(17))
    for value, reference_value in zip(levels, repeated):
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)


def test_b2_nonoverlapping_row_major_blocks_and_fixed_noise():
    model = AlignedPhi4QCR(1.7, hidden=4, block_size=2, steps=1)
    x = torch.arange(2 * 2 * 8 * 8, dtype=torch.double).reshape(2, 2, 8, 8, 1) / 50.
    expected_blocks = torch.stack([
        torch.stack([x[:, :, row:row + 2, column:column + 2, 0].flatten(-2)
                     for column in range(0, 8, 2)], dim=2)
        for row in range(0, 8, 2)], dim=2)
    mean = model.C.mean_field(x)
    torch.testing.assert_close(mean, model.C.mean_blocks(expected_blocks), rtol=0, atol=0)
    generator = torch.Generator().manual_seed(31)
    noise = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
    actual = model.C(x, torch.Generator().manual_seed(31))
    torch.testing.assert_close(actual, mean + .17 * noise)
    with pytest.raises(ValueError, match=r"\[\.\.\.,4\]"):
        model.C.mean_blocks(torch.zeros(2, 9, dtype=torch.double))


def test_b2_chunking_matches_full_value_input_and_parameter_gradients():
    chunked = _Block2GaussianCoarseGrainer(1.3, hidden=6, block_chunk_size=7)
    full = _Block2GaussianCoarseGrainer(1.3, hidden=6, block_chunk_size=4096)
    full.load_state_dict(chunked.state_dict())
    x = torch.randn(2, 3, 8, 8, 1, dtype=torch.double, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    sizes = []
    handle = chunked.net[0].register_forward_pre_hook(
        lambda module, args: sizes.append(len(args[0])))
    actual = chunked.mean_field(x)
    expected = full.mean_field(reference_x)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (x, *chunked.parameters()))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (reference_x, *full.parameters()))
    for actual_gradient, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, reference, rtol=1e-11, atol=1e-13)
    with torch.no_grad():
        actual_no_grad = chunked.mean_field(x)
    handle.remove()
    assert max(sizes) <= 7
    torch.testing.assert_close(actual_no_grad, expected, rtol=1e-13, atol=1e-14)
    # A single inherited full-shape noise call, rather than one draw per
    # chunk, keeps sampling's random-number sequence independent of chunking.
    actual_generator = torch.Generator().manual_seed(43)
    reference_generator = torch.Generator().manual_seed(43)
    actual_sample = chunked(x, actual_generator)
    expected_sample = full(x, reference_generator)
    torch.testing.assert_close(actual_sample, expected_sample, rtol=1e-13, atol=1e-14)
    assert torch.equal(actual_generator.get_state(), reference_generator.get_state())


def test_b2_information_regions_use_complete_buffered_shell_twenty_sites():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.double).reshape(2, 3, 8, 8, 1)
    blocks, environment = _sample_aligned_regions(x, 7, torch.Generator().manual_seed(47), 2)
    generator = torch.Generator().manual_seed(47)
    configs = torch.randint(3, (2, 7), generator=generator)
    rows = torch.randint(8, (2, 7), generator=generator)
    columns = torch.randint(8, (2, 7), generator=generator)
    block_offsets = [(i, j) for i in range(2) for j in range(2)]
    shell_offsets = [(i, j) for i in range(-2, 4) for j in range(-2, 4)
                     if i in (-2, 3) or j in (-2, 3)]
    assert len(shell_offsets) == len(set(shell_offsets)) == 20
    assert blocks.shape == (2, 7, 4) and environment.shape == (2, 7, 20)
    for ensemble in range(2):
        for pair in range(7):
            for actual, offsets in ((blocks, block_offsets), (environment, shell_offsets)):
                expected = torch.stack([
                    x[ensemble, configs[ensemble, pair],
                      (rows[ensemble, pair] + i) % 8,
                      (columns[ensemble, pair] + j) % 8, 0]
                    for i, j in offsets])
                torch.testing.assert_close(actual[ensemble, pair], expected, rtol=0, atol=0)


def test_b2_smoothed_information_matches_formula_and_detaches_environment():
    means = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
    environment = torch.randn(2, 5, 20, dtype=torch.double, requires_grad=True)
    scale = torch.tensor(1.2, dtype=torch.double)
    actual = _aligned_information(means, environment, scale, .1, 2)
    differences = (environment.detach()[:, :, None] - environment.detach()[:, None, :]) / scale
    distances = differences.square().mean(-1)
    diagonal = torch.eye(5, dtype=torch.bool)[None]
    weights = torch.softmax((-.5 * distances).masked_fill(diagonal, -torch.inf), dim=-1)
    conditionals = weights @ gaussian_bin_probabilities(means, scale, .1)
    marginal = conditionals.mean(1, keepdim=True)
    tiny = torch.finfo(torch.double).tiny
    expected = (conditionals * (conditionals.clamp_min(tiny).log()
                               - marginal.clamp_min(tiny).log())).sum(-1).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)
    gradients = torch.autograd.grad(actual, (means, environment), allow_unused=True)
    assert bool(torch.isfinite(gradients[0]).all()) and bool(gradients[0].abs().max() > 0)
    assert gradients[1] is None


@pytest.mark.parametrize("block_size,steps,length", [(2, 1, 8), (2, 3, 8), (3, 3, 27)])
def test_deeper_objective_all_start_weights_and_independent_gradients(block_size, steps, length):
    model = AlignedPhi4QCR(1., hidden=4, theory_dim=2, local_features=3,
                          patch_chunk_size=128, block_size=block_size, steps=steps)
    x = torch.randn(1, 4, length, length, 1, dtype=torch.double)
    q_loss, c_loss, r_loss, metrics = aligned_objectives(
        model, x, torch.Generator().manual_seed(53), information_pairs=4)
    for name, loss in (("Q", q_loss), ("C", c_loss), ("R", r_loss)):
        for other in ("Q", "C", "R"):
            gradients = torch.autograd.grad(loss, list(getattr(model, other).parameters()),
                                            retain_graph=True, allow_unused=True)
            if name == other:
                assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
                assert any(bool(value.abs().max() > 0) for value in gradients)
            else:
                assert all(value is None for value in gradients)
    for level in range(steps + 1):
        assert f"reconstruction_level_{level}" in metrics
        assert f"subset_level_{level}" in metrics
    assert sum(name.startswith("closure_start_") for name in metrics) == steps * (steps + 1) // 2
    torch.testing.assert_close(q_loss, q_loss.new_tensor(
        10. * metrics["q_reconstruction"] + 20. * metrics["q_subset_micro"]
        + 5. * metrics["q_subset_coarse"]))
    levels = model.coarse_grain(x, torch.Generator().manual_seed(53))
    direct = [(.5 * (model.Q(level.detach()[:, :2])
                     + model.Q(level.detach()[:, 2:]))).detach() for level in levels]
    target_losses = []
    for target in range(1, steps + 1):
        start_losses = []
        for start in range(target):
            prediction = direct[start]
            for _ in range(target - start):
                prediction = model.advance(prediction)
            error = (prediction - direct[target]).square().sum(-1).mean()
            start_losses.append(error)
            assert metrics[f"closure_start_{start}_target_{target}"] == pytest.approx(error.item())
        target_losses.append(torch.stack(start_losses).mean())
    expected = 10. * torch.stack(target_losses).mean()
    torch.testing.assert_close(r_loss, expected)
    actual_gradient = torch.autograd.grad(r_loss, list(model.R.parameters()))
    expected_gradient = torch.autograd.grad(expected, list(model.R.parameters()))
    for actual, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kwargs", [
    {"block_size": True}, {"block_size": 4}, {"block_size": 2.},
    {"steps": 0}, {"steps": True}, {"steps": 1.},
])
def test_explicit_geometry_validation(kwargs):
    with pytest.raises(ValueError):
        AlignedPhi4QCR(1., **kwargs)


def test_b2_chain_and_shell_validation():
    model = AlignedPhi4QCR(1., hidden=4, block_size=2, steps=3)
    with pytest.raises(ValueError, match="divisible by 8"):
        model.coarse_grain(torch.randn(1, 2, 12, 12, 1, dtype=torch.double))
    with pytest.raises(ValueError, match="requires L >= 6"):
        _sample_aligned_regions(torch.randn(1, 2, 4, 4, 1, dtype=torch.double),
                                4, torch.Generator().manual_seed(59), 2)
