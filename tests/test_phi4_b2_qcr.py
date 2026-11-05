"""Independent b=2 continuous model and paired closure-ablation contracts."""
import copy
import math

import pytest
import torch

from recurrent_rg.phi4_b2_qcr import (
    BLOCK_OFFSETS_B2, CLOSURE_MODES, FEATURE_DIM, SHELL_OFFSETS_B2,
    GaussianCoarseGrainerB2, Phi4B2QCR, block_vectors_b2,
    continuous_objectives_b2, gaussian_bin_probabilities, sample_regions_b2,
    smoothed_information_b2,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(ensembles=2, samples=4, length=8):
    return torch.randn(ensembles, samples, length, length, 1, dtype=torch.double)


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


def test_blocks_are_nonoverlapping_row_major_and_cover_field():
    x = torch.arange(64, dtype=torch.double).reshape(1, 1, 8, 8, 1)
    blocks = block_vectors_b2(x)
    assert blocks.shape == (1, 1, 4, 4, 4)
    torch.testing.assert_close(blocks[0, 0, 0, 0], torch.tensor([0., 1., 8., 9.], dtype=torch.double))
    torch.testing.assert_close(blocks[0, 0, 3, 3], torch.tensor([54., 55., 62., 63.], dtype=torch.double))
    assert torch.equal(blocks.flatten().sort().values, x.flatten())
    with pytest.raises(ValueError, match="factor 2"):
        block_vectors_b2(field(length=7))


def test_fresh_model_geometry_construction_order_and_gaussian_noise():
    torch.manual_seed(91)
    model = Phi4B2QCR(scale=2., hidden=8)
    torch.manual_seed(91)
    expected_c = GaussianCoarseGrainerB2(scale=2., hidden=8)
    assert all(torch.equal(v, expected_c.state_dict()[k]) for k, v in model.C.state_dict().items())
    assert model.C.net[0].in_features == 4
    assert model.C.net[-1].out_features == 1
    assert model.Q.net[0].in_features == FEATURE_DIM == 105
    assert model.Q.net[2].out_features == 12
    assert model.Q.net[-1].out_features == model.R[0].in_features == 1
    assert all(parameter.dtype == torch.double for parameter in model.parameters())
    x = field()
    mean = model.C.mean_field(x)
    noise = torch.randn(mean.shape, dtype=torch.double, generator=generator())
    actual = model.C(x, generator=generator())
    torch.testing.assert_close(actual, mean + .2 * noise)
    assert not model.C.noise_ratio.requires_grad
    assert not model.C.scale.requires_grad
    assert torch.any(actual != actual.sign())
    assert [level.shape[2] for level in model.coarse_grain(x, generator())] == [8, 4, 2, 1]
    assert [level.shape[2] for level in model.coarse_grain(field(length=32), generator())] == [32, 16, 8, 4]
    with pytest.raises(ValueError, match="divisible by 8"):
        model.coarse_grain(field(length=12))


def test_shell_contains_all_twenty_sites_and_one_site_buffer():
    assert set(BLOCK_OFFSETS_B2) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    expected = {(i, j) for i in range(-2, 4) for j in range(-2, 4)
                if i in (-2, 3) or j in (-2, 3)}
    assert set(SHELL_OFFSETS_B2) == expected
    assert len(SHELL_OFFSETS_B2) == len(expected) == 20
    assert not (set(BLOCK_OFFSETS_B2) & expected)
    assert min(max(abs(a - i), abs(b - j))
               for a, b in SHELL_OFFSETS_B2 for i, j in BLOCK_OFFSETS_B2) == 2
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.double).reshape(2, 3, 8, 8, 1)
    blocks, shell = sample_regions_b2(x, 7, generator())
    assert blocks.shape == (2, 7, 4) and shell.shape == (2, 7, 20)
    for ensemble in range(2):
        for pair in range(7):
            origin = int(blocks[ensemble, pair, 0])
            configuration = origin // 64 - ensemble * 3
            row, column = divmod(origin % 64, 8)
            for observed, offsets in ((blocks, BLOCK_OFFSETS_B2), (shell, SHELL_OFFSETS_B2)):
                expected_values = torch.stack([x[ensemble, configuration, (row + a) % 8,
                                                 (column + b) % 8, 0] for a, b in offsets])
                torch.testing.assert_close(observed[ensemble, pair], expected_values)
    blocks2, shell2 = sample_regions_b2(x, 7, generator())
    assert torch.equal(blocks, blocks2) and torch.equal(shell, shell2)
    # The exact 6-site torus is allowed: every shell location remains unique.
    assert sample_regions_b2(field(length=6), 2, generator())[1].shape[-1] == 20


def test_information_proxy_matches_explicit_full_twenty_coordinate_kernel():
    torch.manual_seed(22)
    means = torch.randn(2, 7, dtype=torch.double) * .3
    environment = torch.randn(2, 7, 20, dtype=torch.double)
    scale, bandwidth = 1.3, .7
    distance = ((environment[:, :, None] - environment[:, None, :]) / scale).square().mean(-1)
    weights = torch.exp(-distance / (2 * bandwidth ** 2))
    weights[:, torch.arange(7), torch.arange(7)] = 0
    weights /= weights.sum(-1, keepdim=True)
    probability = gaussian_bin_probabilities(means, scale)
    conditional = weights @ probability
    marginal = conditional.mean(1, keepdim=True)
    tiny = torch.finfo(torch.double).tiny
    expected = (conditional * (conditional.clamp_min(tiny).log()
                               - marginal.clamp_min(tiny).log())).sum(-1).mean()
    actual = smoothed_information_b2(means, environment, scale, bandwidth=bandwidth)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-12)
    # Alter each coordinate independently: all twenty participate.
    for coordinate in range(20):
        changed = environment.clone()
        changed[:, 0, coordinate] += 10
        assert abs(smoothed_information_b2(means, changed, scale, bandwidth=bandwidth).item()
                   - actual.item()) > 1e-8


def test_information_is_loo_constant_zero_and_detaches_environment_scale():
    environment = torch.randn(2, 12, 20, dtype=torch.double, requires_grad=True)
    means = torch.full((2, 12), .13, dtype=torch.double, requires_grad=True)
    scale = torch.tensor(1., dtype=torch.double, requires_grad=True)
    proxy = smoothed_information_b2(means, environment, scale)
    assert abs(proxy.item()) < 1e-14
    proxy.backward()
    assert means.grad is not None and environment.grad is None and scale.grad is None
    means = torch.tensor([[-.6, .6]], dtype=torch.double)
    probability = gaussian_bin_probabilities(means, 1.)
    marginal = probability.mean(1, keepdim=True)
    tiny = torch.finfo(torch.double).tiny
    expected = (probability * (probability.clamp_min(tiny).log()
                               - marginal.clamp_min(tiny).log())).sum(-1).mean()
    torch.testing.assert_close(smoothed_information_b2(means, torch.zeros(1, 2, 20, dtype=torch.double), 1.), expected)


@pytest.mark.parametrize("branch", ["Q", "C", "R"])
@pytest.mark.parametrize("mode", CLOSURE_MODES)
def test_objective_gradients_are_strictly_independent(branch, mode):
    torch.manual_seed(9)
    model = Phi4B2QCR(scale=1.1, hidden=8)
    q_loss, c_loss, r_loss, metrics = continuous_objectives_b2(model, field(), generator(), 8, mode)
    {"Q": q_loss, "C": c_loss, "R": r_loss}[branch].backward()
    for name in ("Q", "C", "R"):
        gradients = [p.grad for p in getattr(model, name).parameters()]
        if name == branch:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
            assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())


def test_exact_objective_coefficients_and_six_anchored_errors():
    torch.manual_seed(11)
    model = Phi4B2QCR(scale=.8, hidden=8)
    q, c, r, m = continuous_objectives_b2(model, field(), generator(), 8)
    assert m["reconstruction"] == pytest.approx(sum(m[f"reconstruction_level_{k}"] for k in range(4)) / 4)
    assert m["subset_coarse"] == pytest.approx(sum(m[f"subset_level_{k}"] for k in (1, 2, 3)) / 3)
    assert q.item() == pytest.approx(10 * m["reconstruction"] + 20 * m["subset_micro"] + 5 * m["subset_coarse"])
    assert c.item() == pytest.approx(-2 * m["information_proxy"] + 2 * m["variance_floor"] + m["uniform_amplitude"])
    assert m["closure_one_step"] == pytest.approx(sum(m[f"closure_error_{k}_{k + 1}"] for k in range(3)) / 3)
    assert m["closure_rollout"] == pytest.approx(sum(m[f"closure_error_0_{k}"] for k in (1, 2, 3)) / 3)
    expected = (10 / 3) * (m["closure_error_0_1"]
                           + (m["closure_error_0_2"] + m["closure_error_1_2"]) / 2
                           + (m["closure_error_0_3"] + m["closure_error_1_3"] + m["closure_error_2_3"]) / 3)
    assert r.item() == pytest.approx(expected)
    assert m["all_start_anchor"] == m["closure_all_start_anchor"] == r.item()
    assert m["variance_floor"] == pytest.approx(sum(m[f"variance_floor_level_{k}"] for k in (1, 2, 3)) / 3)


def test_closure_modes_differ_but_q_c_losses_gradients_and_random_stream_do_not():
    torch.manual_seed(91)
    original = Phi4B2QCR(scale=1., hidden=8)
    x = field()
    observations = []
    for mode in CLOSURE_MODES:
        model = copy.deepcopy(original)
        rng = generator()
        losses = continuous_objectives_b2(model, x, rng, 8, mode)
        (losses[0] + losses[1] + losses[2]).backward()
        observations.append((losses, model, rng.get_state()))
    (old_losses, old, old_rng), (new_losses, new, new_rng) = observations
    assert torch.equal(old_rng, new_rng)
    assert old_losses[0].item() == new_losses[0].item()
    assert old_losses[1].item() == new_losses[1].item()
    assert old_losses[2].item() != pytest.approx(new_losses[2].item(), abs=1e-10)
    assert old_losses[2].item() == pytest.approx(5 * old_losses[3]["closure_one_step"]
                                                + 5 * old_losses[3]["closure_rollout"])
    for name in ("Q", "C"):
        for old_parameter, new_parameter in zip(getattr(old, name).parameters(), getattr(new, name).parameters()):
            assert torch.equal(old_parameter.grad, new_parameter.grad)
    assert any(not torch.equal(a.grad, b.grad) for a, b in zip(old.R.parameters(), new.R.parameters()))


@pytest.mark.parametrize("mode", CLOSURE_MODES)
def test_objective_rollouts_keep_derivatives_through_all_r_calls(mode):
    torch.manual_seed(91)
    model = Phi4B2QCR(scale=1., hidden=4)
    model.R = torch.nn.Linear(1, 1, bias=False, dtype=torch.double)
    model.R.weight.data.fill_(.2)
    x = field()
    # Replay the identical C draws and exactly the direct half-ensemble estimator.
    levels = model.coarse_grain(x, generator())
    direct = [.5 * (model.Q(model.features(level[:, :2]))
                    + model.Q(model.features(level[:, 2:]))) for level in levels]
    a = 1.2
    weighted_derivative = 0.
    for start in range(3):
        for target in range(start + 1, 4):
            horizon = target - start
            error = a ** horizon * direct[start].detach() - direct[target].detach()
            derivative = (2 * error * horizon * a ** (horizon - 1) * direct[start].detach()).sum(-1).mean().item()
            weight = ((10 / 3) / target if mode == "all_start_anchor"
                      else (5 / 3) * ((target == start + 1) + (start == 0)))
            weighted_derivative += weight * derivative
    _, _, r_loss, _ = continuous_objectives_b2(model, x, generator(), 8, mode)
    r_loss.backward()
    assert model.R.weight.grad.item() == pytest.approx(weighted_derivative, rel=1e-12, abs=1e-14)


def test_noise_does_not_satisfy_variance_floor_or_information():
    model = Phi4B2QCR(scale=1., hidden=5)
    with torch.no_grad():
        for parameter in model.C.parameters():
            parameter.zero_()
    x = field()
    _, _, _, metrics = continuous_objectives_b2(model, x, generator(), 8)
    levels = model.coarse_grain(x, generator())
    expected = sum(level.flatten(1).std(dim=1, unbiased=False).square().mean().item()
                   for level in levels[:-1]) / 3
    assert metrics["variance_floor"] == pytest.approx(expected)
    assert metrics["variance_floor"] > .1
    assert metrics["uniform_amplitude"] == pytest.approx(1.)
    assert abs(metrics["information_proxy"]) < 1e-14


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -1}, {"scale": float("nan")},
                                   {"scale": 1, "hidden": 0}, {"scale": 1, "theory_dim": 0}])
def test_invalid_model_arguments(kwargs):
    with pytest.raises(ValueError):
        Phi4B2QCR(**kwargs)


def test_invalid_objective_regions_and_mean_shapes():
    model = Phi4B2QCR(scale=1., hidden=4)
    with pytest.raises(ValueError, match="even sample"):
        continuous_objectives_b2(model, field(samples=3))
    with pytest.raises(ValueError, match="closure_mode"):
        continuous_objectives_b2(model, field(), closure_mode="unknown")
    with pytest.raises(ValueError, match="At least two"):
        sample_regions_b2(field(), 1)
    with pytest.raises(ValueError, match="L >= 6"):
        sample_regions_b2(field(length=4))
    with pytest.raises(ValueError, match="full-shell"):
        smoothed_information_b2(torch.zeros(1, 3), torch.zeros(1, 3, 24), 1.)
    with pytest.raises(ValueError, match=r"\[\.\.\.,4\]"):
        model.C.mean_blocks(torch.zeros(2, 9))
