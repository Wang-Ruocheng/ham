"""Synthetic contracts for shared autonomous R calibration; no physical data."""
import pytest
import torch
from torch import nn

from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, conditioned_copy,
    fit_input_normalizer, function_preserving_widen,
)


def sample_direct():
    return torch.tensor([[[.2, -.5], [.7, .1], [.4, -.8]],
                         [[-.3, .8], [.2, -.4], [-.1, .5]],
                         [[.9, .6], [-.2, .3], [.6, -.7]]], dtype=torch.float64)


def test_normalizer_uses_only_frozen_train_input_scales_with_population_std():
    direct = sample_direct()[:, None].repeat(1, 2, 1, 1)
    direct[:, 1] += .2
    center, scale = fit_input_normalizer(direct)
    expected = direct[..., :2, :].reshape(-1, 2)
    torch.testing.assert_close(center, expected.mean(0))
    torch.testing.assert_close(scale, expected.std(0, unbiased=False))
    direct[..., 2, :] = 1e9
    other = fit_input_normalizer(direct)
    for first, second in zip((center, scale), other):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not center.requires_grad and not scale.requires_grad


def test_normalizer_rejects_zero_variance_and_nonfinite_or_trainable_encodings():
    for direct in (torch.zeros(3, 3, 2, dtype=torch.float64),
                   sample_direct().requires_grad_(),
                   sample_direct() * float("nan")):
        with pytest.raises(ValueError):
            fit_input_normalizer(direct)


def test_conditioning_preserves_function_jacobian_source_and_buffers():
    torch.manual_seed(345)
    raw = FrozenResidualMap(hidden=7)
    before = {k: v.clone() for k, v in raw.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    scaled = conditioned_copy(raw, [.17, -2.4], [.002, 7.1])
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    g = sample_direct().reshape(-1, 2)
    torch.testing.assert_close(raw(g), scaled(g), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(torch.autograd.functional.jacobian(raw, g[0]),
                               torch.autograd.functional.jacobian(scaled, g[0]),
                               rtol=1e-13, atol=1e-13)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert {"center", "scale"}.issubset(scaled.state_dict())
    assert {"center", "scale"}.issubset(dict(scaled.named_buffers()))
    assert not {"center", "scale"} & dict(scaled.named_parameters()).keys()
    scaled.train()
    scaled(sample_direct().reshape(-1, 2))
    torch.testing.assert_close(scaled.center, torch.tensor([.17, -2.4], dtype=torch.float64))


def test_wider_map_preserves_function_and_jacobian_while_extra_units_can_learn():
    torch.manual_seed(78)
    raw = FrozenResidualMap(hidden=5)
    before = {k: v.clone() for k, v in raw.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    wide = function_preserving_widen(raw, 11, seed=991)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    g = sample_direct().reshape(-1, 2)
    torch.testing.assert_close(raw(g), wide(g), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(torch.autograd.functional.jacobian(raw, g[0]),
                               torch.autograd.functional.jacobian(wide, g[0]),
                               rtol=1e-13, atol=1e-13)
    wide(g).square().mean().backward()
    assert torch.count_nonzero(wide.net[-1].weight.grad[:, raw.hidden:]) > 0
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    again = function_preserving_widen(raw, 11, seed=991)
    for a, b in zip(wide.parameters(), again.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(1.3, dtype=torch.float64))

    def advance(self, g):
        return self.factor * g


def test_closure_keeps_original_coordinate_sum_weights_and_recurrent_gradients():
    direct, model = sample_direct(), Affine()
    loss, terms = all_start_closure_loss(model, direct)
    p01, p12 = model.factor * direct[:, 0], model.factor * direct[:, 1]
    p02 = model.factor * p01
    e01 = (p01 - direct[:, 1]).square().sum(-1).mean()
    e12 = (p12 - direct[:, 2]).square().sum(-1).mean()
    e02 = (p02 - direct[:, 2]).square().sum(-1).mean()
    torch.testing.assert_close(loss, 5 * e01 + 2.5 * e12 + 2.5 * e02)
    terms["e02"].backward()
    expected = (4 * (model.factor.detach().square() * direct[:, 0] - direct[:, 2])
                * model.factor.detach() * direct[:, 0]).sum(-1).mean()
    torch.testing.assert_close(model.factor.grad, expected)
    assert direct.grad is None
    with pytest.raises(ValueError, match="frozen"):
        all_start_closure_loss(model, direct.requires_grad_())


@pytest.mark.parametrize("kwargs", [dict(scale=[1., 0.]), dict(center=[float("nan"), 0.]),
                                   dict(hidden=0), dict(theory_dim=0)])
def test_map_rejects_invalid_architecture_or_internal_units(kwargs):
    with pytest.raises(ValueError):
        FrozenResidualMap(**kwargs)


def test_conditioned_source_cannot_be_mistaken_for_raw_network():
    source = FrozenResidualMap(center=[.1, .2])
    with pytest.raises(ValueError, match="original-coordinate"):
        conditioned_copy(source, [0., 0.], [1., 1.])
    with pytest.raises(ValueError, match="original-coordinate"):
        function_preserving_widen(source, 256, seed=91)
