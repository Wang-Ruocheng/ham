"""Synthetic CPU-only contracts for single-factor b=2 validation models."""
import copy
import json

import pytest
import torch

from recurrent_rg.phi4_b2_qcr import GaussianCoarseGrainerB2, Phi4B2QCR
from recurrent_rg.phi4_b2_validation import (
    VARIANTS, CubicTheoryEncoder, OddGaussianCoarseGrainerB2,
    ValidationModel, continuous_objectives_b2, module_fingerprint,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rng(seed=17):
    return torch.Generator().manual_seed(seed)


def field():
    return torch.randn(2, 4, 8, 8, 1, dtype=torch.double, generator=rng(18))


def seeded(variant):
    torch.manual_seed(91)
    return ValidationModel(scale=1.3, hidden=8, variant=variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_initial_common_parameters_and_rng_match_historical_model(variant):
    torch.manual_seed(91)
    historical = Phi4B2QCR(scale=1.3, hidden=8)
    historical_rng = torch.get_rng_state().clone()
    model = seeded(variant)
    assert torch.equal(torch.get_rng_state(), historical_rng)
    assert model.initial_common_sha256 == {
        "C": module_fingerprint(historical.C),
        "Q_net": module_fingerprint(historical.Q.net),
        "Q_affine_decoder": module_fingerprint(historical.Q.decoder),
        "R": module_fingerprint(historical.R),
    }
    for key, value in historical.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    assert all(p.requires_grad and p.dtype == torch.double for p in model.parameters())
    assert type(model.C) is (OddGaussianCoarseGrainerB2 if variant == "odd_C"
                            else GaussianCoarseGrainerB2)
    assert isinstance(model.Q, CubicTheoryEncoder) == (variant == "cubic_decoder")


def test_cubic_starts_as_exact_affine_function_with_exact_common_q_gradient():
    baseline, cubic = seeded("baseline"), seeded("cubic_decoder")
    x = field()
    q_base = continuous_objectives_b2(baseline, x, rng(), 8)[0]
    q_cubic = continuous_objectives_b2(cubic, x, rng(), 8)[0]
    assert torch.equal(q_base, q_cubic)
    q_base.backward()
    q_cubic.backward()
    for name, parameter in baseline.Q.named_parameters():
        assert torch.equal(parameter.grad, dict(cubic.Q.named_parameters())[name].grad), name
    for parameter in (cubic.Q.quadratic_weight, cubic.Q.cubic_weight):
        assert parameter.shape == (105, 1)
        assert torch.equal(parameter, torch.zeros_like(parameter))
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    g = torch.linspace(-.7, .8, 9, dtype=torch.double)[:, None]
    assert torch.equal(baseline.Q.decode(g), cubic.Q.decode(g))
    assert sum(p.numel() for p in cubic.Q.parameters()) - sum(p.numel() for p in baseline.Q.parameters()) == 210


def test_cubic_formula_and_normalization_do_not_rescale_high_order_coefficients():
    model = seeded("cubic_decoder")
    g = torch.tensor([[-.7], [.3]], dtype=torch.double)
    with torch.no_grad():
        model.Q.quadratic_weight.copy_(torch.linspace(-1, 1, 105, dtype=torch.double)[:, None])
        model.Q.cubic_weight.fill_(.21)
        model.Q.decoder.weight.mul_(4)
    before = (model.Q.quadratic_weight.detach().clone(), model.Q.cubic_weight.detach().clone())
    model.Q.normalize_decoder_()
    assert torch.equal(model.Q.quadratic_weight, before[0])
    assert torch.equal(model.Q.cubic_weight, before[1])
    torch.testing.assert_close(model.Q.decoder.weight.norm(dim=0), torch.ones(1, dtype=torch.double))
    expected = (g @ model.Q.normalized_decoder_weight().T + model.Q.decoder.bias
                + g.square() @ before[0].T + g.pow(3) @ before[1].T)
    torch.testing.assert_close(model.Q.decode(g), expected)


def test_odd_c_mean_is_exactly_odd_and_mean_field_dispatches():
    model = seeded("odd_C")
    original = seeded("baseline")
    blocks = torch.randn(31, 4, dtype=torch.double, generator=rng())
    expected = .5 * (original.C.mean_blocks(blocks) - original.C.mean_blocks(-blocks))
    assert torch.equal(model.C.mean_blocks(blocks), expected)
    assert torch.equal(model.C.mean_blocks(-blocks), -model.C.mean_blocks(blocks))
    assert torch.equal(model.C.mean_blocks(torch.zeros_like(blocks)), torch.zeros(31, 1, dtype=torch.double))
    x = field()
    assert torch.equal(model.C.mean_field(-x), -model.C.mean_field(x))
    assert torch.equal(model.C.noise_std, original.C.noise_std)
    # No data-dependent amplitude normalization or clipping is introduced.
    doubled = model.C.mean_blocks(2 * blocks)
    assert not torch.equal(doubled, model.C.mean_blocks(blocks))


def test_odd_c_has_exact_antithetic_noise_Z2_distribution_contract(monkeypatch):
    model = seeded("odd_C")
    x = field()
    shape = model.C.mean_field(x).shape
    epsilon = torch.randn(shape, dtype=torch.double, generator=rng())
    signs = iter((1., -1.))

    def antithetic_randn(requested_shape, **kwargs):
        assert requested_shape == shape
        assert kwargs["dtype"] == torch.double and kwargs["device"].type == "cpu"
        return next(signs) * epsilon

    monkeypatch.setattr(torch, "randn", antithetic_randn)
    positive = model.C(x, generator=rng())
    negative = model.C(-x, generator=rng())
    assert torch.equal(negative, -positive)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("branch", ("Q", "C", "R"))
def test_unchanged_objectives_have_strictly_independent_gradients(variant, branch):
    model = seeded(variant)
    losses = continuous_objectives_b2(model, field(), rng(), 8, "all_start_anchor")
    dict(zip(("Q", "C", "R"), losses[:3]))[branch].backward()
    for name in ("Q", "C", "R"):
        gradients = [p.grad for p in getattr(model, name).parameters()]
        if name == branch:
            assert any(g is not None and g.abs().sum() > 0 for g in gradients)
            assert all(g is None or torch.isfinite(g).all() for g in gradients)
        else:
            assert all(g is None for g in gradients)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_start_objective_keeps_uniform_targets_and_all_six_anchors(variant):
    model = seeded(variant)
    _, _, loss, m = continuous_objectives_b2(model, field(), rng(), 8)
    expected = (10 / 3) * (
        m["closure_error_0_1"]
        + (m["closure_error_0_2"] + m["closure_error_1_2"]) / 2
        + (m["closure_error_0_3"] + m["closure_error_1_3"] + m["closure_error_2_3"]) / 3)
    assert loss.item() == pytest.approx(expected)
    receipt = model.validation_receipt()
    assert receipt["closure_scale_weights"] is None
    assert receipt["Q_closure_feedback"] is False


def test_odd_c_parameters_receive_nonzero_gradients_and_remain_randomly_trainable():
    model = seeded("odd_C")
    before = copy.deepcopy(model.C.state_dict())
    _, loss, _, _ = continuous_objectives_b2(model, field(), rng(), 8)
    loss.backward()
    # The final additive bias cancels by construction; this is not frozen C.
    active = [p for p in model.C.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(active) >= 3
    optimizer = torch.optim.Adam(model.C.parameters(), lr=.001)
    optimizer.step()
    assert any(not torch.equal(value, model.C.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("variant", VARIANTS)
def test_receipt_is_json_safe_and_names_only_one_changed_factor(variant):
    receipt = seeded(variant).validation_receipt()
    json.dumps(receipt, allow_nan=False)
    assert receipt["single_factor"] is True
    assert receipt["extra_decoder_parameters"] == (210 if variant == "cubic_decoder" else 0)
    assert receipt["C_mean_exactly_odd"] == (variant == "odd_C")
    assert receipt["C_per_ensemble_amplitude_normalization"] is False
    assert receipt["exact_Shannon_MI"] is False
    assert receipt["labels_used_for_training"] is False
    assert receipt["default_recipe_modified"] is False


@pytest.mark.parametrize("kwargs", [{"variant": "combined"}, {"theory_dim": 2},
                                   {"theory_dim": 0}, {"scale": 0}])
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        ValidationModel(**{"scale": 1., **kwargs})
