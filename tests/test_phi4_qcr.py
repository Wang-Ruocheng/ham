"""Continuous-field recipe tests, independent of binary QCR defaults."""
import math

import pytest
import torch
from torch.nn import functional as F

from recurrent_rg.phi4_qcr import (
    BLOCK_OFFSETS, FEATURE_DIM, PAIR_OFFSETS, SHELL_OFFSETS,
    Phi4QCR, block_vectors, continuous_objectives,
    gaussian_bin_probabilities, sample_regions, smoothed_information,
    soft_local_features,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(ensembles=2, samples=4, length=9):
    return torch.randn(ensembles, samples, length, length, 1, dtype=torch.double)


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


def test_gaussian_bins_have_infinite_tails_and_unit_mass():
    means = torch.tensor([-100., -3., -.1, 0., 3., 100.], dtype=torch.double)
    probability = gaussian_bin_probabilities(means, scale=1.)
    assert probability.shape == (6, 16)
    assert torch.all(probability >= 0.)
    torch.testing.assert_close(probability.sum(-1), torch.ones(6, dtype=torch.double), atol=1e-15, rtol=0)
    assert probability[0, 0] == probability[-1, -1] == 1.
    torch.testing.assert_close(probability[1], probability[4].flip(0))
    edge = torch.tensor([-2.625], dtype=torch.double)
    assert gaussian_bin_probabilities(edge, 1.)[0, 0] == .5


def test_information_constant_is_zero_and_only_means_are_differentiable():
    environment = torch.randn(2, 12, 24, dtype=torch.double, requires_grad=True)
    constant = torch.full((2, 12), .13, dtype=torch.double, requires_grad=True)
    proxy = smoothed_information(constant, environment, 1.)
    assert abs(proxy.item()) < 1e-14
    proxy.backward()
    assert constant.grad is not None
    assert environment.grad is None
    means = torch.linspace(-.6, .6, 12, dtype=torch.double)[None].repeat(2, 1).requires_grad_()
    nonconstant = smoothed_information(means, environment, 1.)
    assert nonconstant.item() >= -1e-14
    nonconstant.backward()
    assert means.grad is not None and means.grad.abs().sum() > 0
    assert environment.grad is None


def test_information_uses_loo_and_all_environment_coordinates():
    # Two leave-one-out rows each choose only the other example, regardless of
    # environment proximity. Analytic Gaussian bin Shannon information follows.
    means = torch.tensor([[-.6, .6]], dtype=torch.double)
    environment = torch.zeros(1, 2, 24, dtype=torch.double)
    probability = gaussian_bin_probabilities(means, 1.)
    marginal = probability.mean(1, keepdim=True)
    tiny = torch.finfo(torch.double).tiny
    expected = (probability * (probability.clamp_min(tiny).log() - marginal.clamp_min(tiny).log())).sum(-1).mean()
    torch.testing.assert_close(smoothed_information(means, environment, 1.), expected)
    environment[:, 1, -1] = 1e6
    torch.testing.assert_close(smoothed_information(means, environment, 1.), expected)


def test_information_matches_explicit_full_vector_kernel():
    torch.manual_seed(22)
    means = torch.randn(2, 7, dtype=torch.double) * .3
    environment = torch.randn(2, 7, 24, dtype=torch.double)
    scale, bandwidth = 1.3, .7
    distance = ((environment[:, :, None] - environment[:, None, :]) / scale).square().mean(-1)
    weights = torch.exp(-distance / (2 * bandwidth ** 2))
    weights[:, torch.arange(7), torch.arange(7)] = 0
    weights /= weights.sum(-1, keepdim=True)
    probability = gaussian_bin_probabilities(means, scale)
    conditional = weights @ probability
    marginal = conditional.mean(1, keepdim=True)
    tiny = torch.finfo(torch.double).tiny
    expected = (conditional * (conditional.clamp_min(tiny).log() - marginal.clamp_min(tiny).log())).sum(-1).mean()
    actual = smoothed_information(means, environment, scale, bandwidth=bandwidth)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-12)
    changed = environment.clone()
    changed[:, 0, -1] += 10
    assert abs(smoothed_information(means, changed, scale, bandwidth=bandwidth).item() - actual.item()) > 1e-7


def test_full_shell_has_buffer_and_region_layout_is_correct():
    assert len(SHELL_OFFSETS) == len(set(SHELL_OFFSETS)) == 24
    assert set(BLOCK_OFFSETS) == {(i, j) for i in range(3) for j in range(3)}
    assert all(i in (-2, 4) or j in (-2, 4) for i, j in SHELL_OFFSETS)
    assert not (set(SHELL_OFFSETS) & set(BLOCK_OFFSETS))
    x = torch.arange(2 * 3 * 9 * 9, dtype=torch.double).reshape(2, 3, 9, 9, 1)
    v, environment = sample_regions(x, pairs=7, generator=generator())
    assert v.shape == (2, 7, 9) and environment.shape == (2, 7, 24)
    for ensemble in range(2):
        for pair in range(7):
            origin = int(v[ensemble, pair, 0])
            config = origin // 81 - ensemble * 3
            i, j = divmod(origin % 81, 9)
            expected_v = torch.stack([x[ensemble, config, (i + a) % 9, (j + b) % 9, 0]
                                      for a, b in BLOCK_OFFSETS])
            expected_e = torch.stack([x[ensemble, config, (i + a) % 9, (j + b) % 9, 0]
                                      for a, b in SHELL_OFFSETS])
            torch.testing.assert_close(v[ensemble, pair], expected_v)
            torch.testing.assert_close(environment[ensemble, pair], expected_e)
    v2, e2 = sample_regions(x, 7, generator=generator())
    assert torch.equal(v, v2) and torch.equal(environment, e2)


def test_soft_features_group_masses_and_oriented_probabilities():
    x = field()
    features = soft_local_features(x, 1.)
    assert features.shape == (2, FEATURE_DIM) and FEATURE_DIM == 105
    torch.testing.assert_close(features.sum(-1), torch.ones(2, dtype=torch.double))
    assert torch.all(features >= 0)
    starts = ((0, 5), (5, 30), (30, 55), (55, 80), (80, 105))
    for start, stop in starts:
        torch.testing.assert_close(features[:, start:stop].sum(-1), torch.full((2,), .2, dtype=torch.double))
    probability = torch.softmax(-.5 * ((x[..., 0, None] - torch.arange(-2, 3, dtype=torch.double)) / .5).square(), -1)
    for index, offset in enumerate(PAIR_OFFSETS):
        shifted = torch.roll(probability, tuple(-v for v in offset), (2, 3))
        expected = (probability[..., :, None] * shifted[..., None, :]).mean((1, 2, 3)).flatten(1) / 5.
        torch.testing.assert_close(features[:, 5 + 25 * index:30 + 25 * index], expected)


def test_soft_features_pool_halves_exactly_and_preserve_input_derivative():
    x = field(samples=6).requires_grad_()
    f = soft_local_features(x, 1.)
    halves = .5 * (soft_local_features(x[:, :3], 1.) + soft_local_features(x[:, 3:], 1.))
    torch.testing.assert_close(f, halves, atol=2e-16, rtol=2e-14)
    f[:, 0].sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_nonoverlapping_blocks_and_gaussian_continuous_output():
    torch.manual_seed(91)
    model = Phi4QCR(scale=2., hidden=8)
    x = torch.arange(81, dtype=torch.double).reshape(1, 1, 9, 9, 1)
    blocks = block_vectors(x)
    assert blocks.shape == (1, 1, 3, 3, 9)
    torch.testing.assert_close(blocks[0, 0, 0, 0], torch.tensor([0., 1., 2., 9., 10., 11., 18., 19., 20.], dtype=torch.double))
    expected = model.C.mean_field(x)
    noise = torch.randn(expected.shape, dtype=torch.double, generator=generator())
    actual = model.C(x, generator=generator())
    torch.testing.assert_close(actual, expected + .2 * noise)
    assert torch.any(actual != actual.sign())
    assert all(p.dtype == torch.double for p in model.parameters())
    assert model.C.noise_std.item() == .2
    assert not model.C.noise_ratio.requires_grad and not model.C.scale.requires_grad


def test_model_architecture_random_kernel_and_fixed_geometry():
    torch.manual_seed(13)
    a = Phi4QCR(scale=1., hidden=9)
    torch.manual_seed(14)
    b = Phi4QCR(scale=1., hidden=9)
    assert not torch.equal(a.C.net[0].weight, b.C.net[0].weight)
    assert a.C.net[0].in_features == 9
    assert a.Q.net[0].in_features == 105 and a.Q.net[2].out_features == 12
    assert a.Q.net[6].out_features == 1 and a.R[0].in_features == 1
    levels = a.coarse_grain(field(length=36), generator())
    assert [level.shape[2] for level in levels] == [36, 12, 4]
    with pytest.raises(ValueError, match="divisible by 9"):
        a.coarse_grain(field(length=12))


def test_decoder_is_column_normalized_in_forward_and_after_projection():
    model = Phi4QCR(scale=1., hidden=7, theory_dim=3)
    torch.testing.assert_close(model.Q.normalized_decoder_weight().norm(dim=0), torch.ones(3, dtype=torch.double))
    with torch.no_grad():
        model.Q.decoder.weight.mul_(11.)
    g = torch.randn(4, 3, dtype=torch.double)
    decoded = model.Q.decode(g)
    expected = F.linear(g, model.Q.normalized_decoder_weight(), model.Q.decoder.bias)
    torch.testing.assert_close(decoded, expected)
    model.Q.normalize_decoder_()
    torch.testing.assert_close(model.Q.decoder.weight.norm(dim=0), torch.ones(3, dtype=torch.double))


@pytest.mark.parametrize("branch", ["Q", "C", "R"])
def test_objectives_have_strict_independent_parameter_gradients(branch):
    torch.manual_seed(9)
    model = Phi4QCR(scale=1.1, hidden=8)
    q_loss, c_loss, r_loss, metrics = continuous_objectives(model, field(), generator(), 8)
    {"Q": q_loss, "C": c_loss, "R": r_loss}[branch].backward()
    for name in ("Q", "C", "R"):
        gradients = [p.grad for p in getattr(model, name).parameters()]
        if name == branch:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
            assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())


def test_losses_use_uniform_scale_weights_and_declared_coefficients():
    torch.manual_seed(11)
    model = Phi4QCR(scale=.8, hidden=8)
    q, c, r, metrics = continuous_objectives(model, field(), generator(), 8)
    m = metrics
    assert m["reconstruction"] == pytest.approx(sum(m[f"reconstruction_level_{k}"] for k in range(3)) / 3)
    assert m["subset_coarse"] == pytest.approx((m["subset_level_1"] + m["subset_level_2"]) / 2)
    assert q.item() == pytest.approx(10 * m["reconstruction"] + 20 * m["subset_micro"] + 5 * m["subset_coarse"])
    for name in ("one_step", "rollout"):
        assert m[f"closure_{name}"] == pytest.approx(.5 * sum(m[f"closure_{name}_level_{k}"] for k in (1, 2)))
    assert r.item() == pytest.approx(5 * m["closure_one_step"] + 5 * m["closure_rollout"])
    assert c.item() == pytest.approx(-2 * m["information_proxy"] + 2 * m["variance_floor"] + m["uniform_amplitude"])


def test_multistep_r_gradient_includes_intermediate_r_application():
    model = Phi4QCR(scale=1., hidden=4)
    model.R = torch.nn.Linear(1, 1, bias=False, dtype=torch.double)
    with torch.no_grad():
        model.R.weight.fill_(.2)
    g = torch.tensor([[.7]], dtype=torch.double)
    target = torch.tensor([[.1]], dtype=torch.double)
    prediction = model.advance(model.advance(g))
    loss = (prediction - target).square().sum()
    loss.backward()
    expected = 2 * ((1.2 ** 2) * .7 - .1) * (2 * 1.2 * .7)
    assert model.R.weight.grad.item() == pytest.approx(expected)


def test_noise_variance_cannot_satisfy_c_variance_floor():
    model = Phi4QCR(scale=1., hidden=5)
    with torch.no_grad():
        for p in model.C.parameters():
            p.zero_()
    x = field()
    _, _, _, metrics = continuous_objectives(model, x, generator(), 8)
    levels = model.coarse_grain(x, generator())
    expected = .5 * sum(level.flatten(1).std(dim=1, unbiased=False).square().mean().item()
                         for level in levels[:-1])
    assert metrics["variance_floor"] == pytest.approx(expected)
    assert metrics["variance_floor"] > .1
    assert metrics["uniform_amplitude"] == pytest.approx(1.)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -1}, {"scale": float("nan")},
                                   {"scale": 1, "hidden": 0}, {"scale": 1, "theory_dim": 0}])
def test_invalid_model_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Phi4QCR(**kwargs)


def test_invalid_objective_inputs_are_rejected():
    model = Phi4QCR(scale=1., hidden=4)
    with pytest.raises(ValueError, match="even sample"):
        continuous_objectives(model, field(samples=3))
    with pytest.raises(ValueError, match="At least two"):
        sample_regions(field(), pairs=1)
    with pytest.raises(ValueError, match="L >= 7"):
        sample_regions(field(length=6))
    with pytest.raises(ValueError, match="full-shell"):
        smoothed_information(torch.zeros(1, 3), torch.zeros(1, 3, 23), 1.)
