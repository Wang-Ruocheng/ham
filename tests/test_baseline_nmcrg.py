"""Synthetic CPU checks for the independent NMCRG method adaptation."""

import itertools

import pytest
import torch

from recurrent_rg.baselines.nmcrg import PeriodicConvRBM


def _states(sites):
    return torch.tensor(list(itertools.product((-1.0, 1.0), repeat=sites)), dtype=torch.float64)


def _model(width=2, stride=2):
    model = PeriodicConvRBM(width, stride, dtype=torch.float64, generator=torch.Generator().manual_seed(91))
    with torch.no_grad():
        model.weight.copy_(torch.linspace(-0.37, 0.43, width * width).reshape(1, 1, width, width))
    return model


def test_zero_bias_small_uniform_initialization_and_explicit_generator():
    model = PeriodicConvRBM(dtype=torch.float64, generator=torch.Generator().manual_seed(91))
    duplicate = PeriodicConvRBM(dtype=torch.float64, generator=torch.Generator().manual_seed(91))
    assert tuple(dict(model.named_parameters())) == ("weight",)
    assert not tuple(model.named_buffers())
    assert model.weight.shape == (1, 1, 9, 9)
    assert model.weight.dtype == torch.float64
    assert model.weight.device.type == "cpu"
    assert torch.all(model.weight >= -1e-5)
    assert torch.all(model.weight <= 1e-5)
    torch.testing.assert_close(model.weight, duplicate.weight, rtol=0, atol=0)


def test_conditional_trace_normalization_and_spin_flip_symmetry():
    model = _model(2, 1)
    visible = _states(4).reshape(-1, 2, 2)
    fields = model.hidden_fields(visible).flatten(1)
    hidden = _states(4)
    log_prob = fields @ hidden.T - torch.logaddexp(fields, -fields).sum(-1, keepdim=True)
    torch.testing.assert_close(log_prob.exp().sum(-1), torch.ones(16, dtype=torch.float64))
    torch.testing.assert_close(model.hidden_fields(-visible), -model.hidden_fields(visible))
    hidden_means = log_prob.exp() @ hidden
    torch.testing.assert_close(hidden_means, torch.tanh(fields))


def test_visible_conditional_trace_matches_adjoint_logistic():
    model = _model()
    hidden = _states(1).reshape(-1, 1, 1)
    fields = model.visible_fields(hidden).flatten(1)
    visible = _states(4)
    log_prob = fields @ visible.T - torch.logaddexp(fields, -fields).sum(-1, keepdim=True)
    torch.testing.assert_close(log_prob.exp().sum(-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(log_prob.exp() @ visible, torch.tanh(fields))


@pytest.mark.parametrize("width,stride,size", [(8, 2, 16), (9, 3, 12), (3, 3, 9), (2, 2, 2), (5, 2, 4), (9, 3, 3)])
def test_periodic_fields_are_bilinear_adjoints(width, stride, size):
    model = _model(width, stride)
    rng = torch.Generator().manual_seed(92)
    visible = torch.randn((2, size, size), dtype=torch.float64, generator=rng, requires_grad=True)
    hidden = torch.randn((2, size // stride, size // stride), dtype=torch.float64, generator=rng)
    energy_vh = (model.hidden_fields(visible) * hidden).sum()
    energy_hv = (visible * model.visible_fields(hidden)).sum()
    torch.testing.assert_close(energy_vh, energy_hv, rtol=1e-12, atol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(energy_vh, visible, retain_graph=True)[0], model.visible_fields(hidden))
    # The opposite adjoint gradient must also retain the weight graph.
    grad_first = torch.autograd.grad(energy_vh, model.weight, retain_graph=True)[0]
    grad_second = torch.autograd.grad(energy_hv, model.weight)[0]
    torch.testing.assert_close(grad_first, grad_second, rtol=1e-12, atol=1e-11)


def test_periodic_wrap_matches_explicit_indices_and_translation():
    model = _model(9, 3)
    visible = torch.arange(144, dtype=torch.float64).reshape(1, 12, 12)
    fields = model.hidden_fields(visible)
    expected = torch.zeros_like(fields)
    for row in range(4):
        for col in range(4):
            expected[0, row, col] = sum(
                model.weight[0, 0, a, b] * visible[0, (3 * row + a) % 12, (3 * col + b) % 12]
                for a in range(9) for b in range(9)
            )
    torch.testing.assert_close(fields, expected)
    shifted = torch.roll(visible, (3, -6), dims=(-2, -1))
    torch.testing.assert_close(model.hidden_fields(shifted), torch.roll(fields, (1, -2), dims=(-2, -1)))


@pytest.mark.parametrize("prefix", [(), (2,), (2, 3)])
def test_coarse_grain_preserves_prefix_and_scales(prefix):
    model = PeriodicConvRBM(dtype=torch.float64)
    visible = torch.ones((*prefix, 108, 108), dtype=torch.float64, requires_grad=True)
    first = model.coarse_grain(visible, generator=torch.Generator().manual_seed(11))
    second = model.coarse_grain(first, generator=torch.Generator().manual_seed(12))
    assert first.shape == (*prefix, 36, 36)
    assert second.shape == (*prefix, 12, 12)
    assert set(first.unique().tolist()) <= {-1.0, 1.0}
    assert not first.requires_grad
    assert not second.requires_grad
    assert model.weight.grad is None


def test_sampler_reproducibility_and_generator_isolation():
    model = _model(8, 2)
    visible = torch.ones((2, 16, 16), dtype=torch.float64)
    global_before = torch.random.get_rng_state().clone()
    a = model.sample_hidden(visible, generator=torch.Generator().manual_seed(94))
    b = model.sample_hidden(visible, generator=torch.Generator().manual_seed(94))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
    back = model.sample_visible(a, generator=torch.Generator().manual_seed(95))
    assert back.shape == visible.shape
    assert set(back.unique().tolist()) <= {-1.0, 1.0}
    assert not back.requires_grad


def test_free_energy_matches_finite_hidden_state_sum():
    model = _model(2, 1)
    visible = _states(4).reshape(-1, 2, 2)
    fields = model.hidden_fields(visible).flatten(1)
    hidden = _states(4)
    energies = fields @ hidden.T
    exact = -torch.logsumexp(energies, dim=1).mean() / fields.shape[1]
    torch.testing.assert_close(model.free_energy(visible), exact)
    with torch.no_grad():
        model.weight.fill_(1000)
    assert torch.isfinite(model.free_energy(visible))


def test_negative_phase_sign_matches_exact_finite_state_kl_gradient():
    model = _model(2, 1)
    all_visible = _states(4).reshape(-1, 2, 2)
    data = all_visible[[0, 3, 5, 11, 15]]
    fields = model.hidden_fields(all_visible).flatten(1)
    log_visible_weight = torch.logsumexp(fields @ _states(4).T, dim=1)
    log_partition = torch.logsumexp(log_visible_weight, dim=0)
    true_nll = model.free_energy(data) + log_partition / fields.shape[1]
    true_gradient = torch.autograd.grad(true_nll, model.weight, retain_graph=True)[0]
    probability = (log_visible_weight - log_partition).exp().detach()
    negative_free_energy = -(probability * torch.logaddexp(fields, -fields).mean(-1)).sum()
    surrogate = model.free_energy(data) - negative_free_energy
    surrogate_gradient = torch.autograd.grad(surrogate, model.weight)[0]
    torch.testing.assert_close(true_gradient, surrogate_gradient)


def test_cd3_continues_the_chain_and_disables_sampling_gradients(monkeypatch):
    model = _model(1, 1)
    visible = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    seen = []
    generator = torch.Generator().manual_seed(91)

    def sample_hidden(current, *, generator):
        assert not torch.is_grad_enabled()
        assert not current.requires_grad
        seen.append(current.clone())
        return current.clone()

    def sample_visible(current, *, generator):
        assert not torch.is_grad_enabled()
        return -current

    monkeypatch.setattr(model, "sample_hidden", sample_hidden)
    monkeypatch.setattr(model, "sample_visible", sample_visible)
    loss = model.cd_loss(visible, steps=3, generator=generator)
    assert len(seen) == 3
    for actual, expected in zip(seen, (visible, -visible, visible)):
        torch.testing.assert_close(actual, expected)
    assert loss.requires_grad


def test_cd_empirical_negative_gradient_is_detached_but_recomputed(monkeypatch):
    model = _model()
    visible = _states(4)[[2, 3, 6, 10]].reshape(-1, 2, 2).requires_grad_(True)
    negative = torch.ones_like(visible, requires_grad=True)
    monkeypatch.setattr(model, "sample_hidden", lambda current, **kwargs: current[..., :1, :1])
    monkeypatch.setattr(model, "sample_visible", lambda current, **kwargs: negative)
    actual_loss = model.cd_loss(visible, steps=3)
    expected_loss = model.free_energy(visible) - model.free_energy(negative.detach())
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_gradient = torch.autograd.grad(actual_loss, model.weight, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_loss, model.weight)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    positive_only_gradient = torch.autograd.grad(model.free_energy(visible), model.weight)[0]
    assert not torch.allclose(actual_gradient, positive_only_gradient)
    actual_loss.backward()
    assert negative.grad is None


@pytest.mark.parametrize("width,stride", [(0, 1), (2, 0), (2, 3), (3.5, 1), (3, True)])
def test_invalid_geometry_rejected(width, stride):
    with pytest.raises(ValueError):
        PeriodicConvRBM(width, stride)


def test_invalid_input_and_cd_steps_rejected():
    model = _model(3, 3)
    for bad in (torch.ones((2, 5, 5), dtype=torch.float64), torch.ones((2, 6, 5), dtype=torch.float64)):
        with pytest.raises(ValueError):
            model.hidden_fields(bad)
    with pytest.raises(TypeError):
        model.hidden_fields(torch.ones((2, 6, 6), dtype=torch.float32))
    for bad_steps in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.cd_loss(torch.ones((2, 6, 6), dtype=torch.float64), steps=bad_steps)
