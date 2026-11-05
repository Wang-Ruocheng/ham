"""Synthetic CPU-only contracts for the frozen r/u endpoint diagnostic.

No physical fields, checkpoints, source results, GPU, or training runs are used.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from experiments import phi4_ru_frozen_r_diagnostic as diagnostic


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class AffineByScale(nn.Module):
    """Nonidentical transitions make a detached or wrong recurrent step visible."""

    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
        self.second = nn.Parameter(torch.tensor(-0.7, dtype=torch.float64))
        self.register_buffer("offsets", torch.tensor([[0.4, -0.1], [-0.2, 0.3]],
                                                      dtype=torch.float64))

    def advance(self, g, transition=0):
        factor = self.first if transition == 0 else self.second
        return factor * g + self.offsets[transition]


def frozen_direct():
    return torch.tensor([
        [[0.2, -0.5], [0.7, 0.1], [0.4, -0.8]],
        [[-0.3, 0.8], [0.2, -0.4], [-0.1, 0.5]],
        [[0.9, 0.6], [-0.2, 0.3], [0.6, -0.7]],
    ], dtype=torch.float64)


def test_closure_uses_exact_all_start_weights_and_original_coordinate_norm():
    model, direct = AffineByScale(), frozen_direct()
    original = direct.clone()
    loss, terms = diagnostic.closure_loss(model, direct)
    p01 = model.first * direct[:, 0] + model.offsets[0]
    p12 = model.second * direct[:, 1] + model.offsets[1]
    p02 = model.second * p01 + model.offsets[1]
    expected = {
        "e01": (p01 - direct[:, 1]).square().sum(-1).mean(),
        "e12": (p12 - direct[:, 2]).square().sum(-1).mean(),
        "e02": (p02 - direct[:, 2]).square().sum(-1).mean(),
    }
    assert set(terms) == set(expected)
    for key in expected:
        torch.testing.assert_close(terms[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(loss, 5 * expected["e01"] + 2.5 * expected["e12"]
                               + 2.5 * expected["e02"], rtol=0, atol=0)
    assert terms["e01"].item() == pytest.approx(2 * (p01 - direct[:, 1]).square().mean().item())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert direct.grad is None and not direct.requires_grad
    torch.testing.assert_close(direct, original, rtol=0, atol=0)


def test_two_step_term_backpropagates_through_the_first_map():
    model, direct = AffineByScale(), frozen_direct()
    _, terms = diagnostic.closure_loss(model, direct)
    p01 = model.first.detach() * direct[:, 0] + model.offsets[0]
    error = model.second.detach() * p01 + model.offsets[1] - direct[:, 2]
    expected_first = (2 * error * model.second.detach() * direct[:, 0]).sum(-1).mean()
    expected_second = (2 * error * p01).sum(-1).mean()
    terms["e02"].backward()
    assert expected_first.abs() > 1e-3
    torch.testing.assert_close(model.first.grad, expected_first)
    torch.testing.assert_close(model.second.grad, expected_second)


@pytest.mark.parametrize("shape", [(3, 2), (2, 4, 2), (2, 3, 1), (2, 3, 3)])
def test_closure_rejects_wrong_cache_geometry(shape):
    with pytest.raises(ValueError, match="frozen"):
        diagnostic.closure_loss(AffineByScale(), torch.zeros(shape, dtype=torch.float64))


def test_closure_rejects_trainable_cache_instead_of_silently_updating_q():
    direct = frozen_direct().requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        diagnostic.closure_loss(AffineByScale(), direct)
    assert direct.grad is None


def test_internal_conditioning_preserves_physical_map_and_its_jacobian():
    torch.manual_seed(733)
    raw = diagnostic.ResidualMap(7, np.zeros(2), np.ones(2))
    initial = {name: value.clone() for name, value in raw.state_dict().items()}
    center, scale = np.array([0.17, -2.4]), np.array([0.002, 7.1])
    conditioned = diagnostic.conditioned_copy(raw, center, scale)
    g = torch.tensor([[-0.3, 0.5], [0.0, -2.5], [0.21, 1.1]], dtype=torch.float64)
    assert conditioned.hidden == raw.hidden == 7
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in conditioned.parameters())
    torch.testing.assert_close(raw.advance(g), conditioned.advance(g), rtol=1e-13, atol=1e-13)
    point = g[0].clone().requires_grad_()
    torch.testing.assert_close(torch.autograd.functional.jacobian(raw.advance, point),
                               torch.autograd.functional.jacobian(conditioned.advance, point),
                               rtol=1e-13, atol=1e-13)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    for first, second in zip(raw.parameters(), conditioned.parameters()):
        assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("center,scale", [
    ([0.0], [1.0, 1.0]), ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 0.0], [1.0, -1.0]), ([float("nan"), 0.0], [1.0, 1.0]),
    ([0.0, 0.0], [1.0, float("inf")]),
])
def test_residual_map_rejects_invalid_internal_units(center, scale):
    with pytest.raises(ValueError):
        diagnostic.ResidualMap(4, np.array(center), np.array(scale))


def test_conditioned_copy_rejects_already_conditioned_source():
    source = diagnostic.ResidualMap(4, np.array([0.3, 0.0]), np.ones(2))
    with pytest.raises(ValueError, match="original-coordinate"):
        diagnostic.conditioned_copy(source, np.zeros(2), np.ones(2))


def test_scale_specific_prediction_uses_first_then_second_map_without_mutation():
    torch.manual_seed(29)
    first = diagnostic.ResidualMap(5, np.array([0.1, -0.2]), np.array([0.2, 0.7]))
    second = diagnostic.ResidualMap(5, np.array([-0.3, 0.4]), np.array([0.8, 0.3]))
    model = diagnostic.ScaleSpecificMap(first, second)
    direct = frozen_direct().numpy()
    original = direct.copy()

    def physical_map(mapping, values):
        return values + mapping.scale * mapping.net((values - mapping.center) / mapping.scale)

    with torch.no_grad():
        g = torch.as_tensor(direct)
        p01 = physical_map(first, g[:, 0])
        p12 = physical_map(second, g[:, 1])
        p02 = physical_map(second, p01)
        reversed_path = physical_map(first, physical_map(second, g[:, 0]))
    assert not torch.allclose(p02, reversed_path, rtol=1e-6, atol=1e-6)
    predicted = diagnostic.predict(model, direct)
    assert predicted["one_step"].shape == (3, 2, 2)
    assert predicted["rollout"].shape == (3, 3, 2)
    np.testing.assert_allclose(predicted["one_step"], torch.stack([p01, p12], dim=1).numpy())
    np.testing.assert_allclose(predicted["rollout"], torch.stack([g[:, 0], p01, p02], dim=1).numpy())
    np.testing.assert_array_equal(predicted["rollout"][:, 0], direct[:, 0])
    np.testing.assert_array_equal(direct, original)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):
        model.advance(g[:, 0], 2)
    with pytest.raises(ValueError):
        first.advance(g[:, 0], 2)


def test_metrics_use_fixed_train_population_std_not_test_or_pooled_coordinates():
    base = np.array([[-3.0, 4.0], [-1.0, 2.0], [1.0, 7.0], [3.0, 1.0]])
    train = np.stack([base, base * [2.0, 0.5] + [10.0, -4.0],
                      base * [0.3, 3.0] + [-2.0, 50.0]], axis=1)
    test = train[:3] * 20.0 + np.array([700.0, -500.0])
    errors = {"0_to_1": np.array([0.3, -0.4]), "1_to_2": np.array([0.1, 0.7]),
              "0_to_2": np.array([-0.2, 0.6])}
    predictions = {
        split: dict(one_step=np.stack([g[:, 1] + errors["0_to_1"],
                                       g[:, 2] + errors["1_to_2"]], axis=1),
                    rollout=np.stack([g[:, 0], g[:, 1] + errors["0_to_1"],
                                      g[:, 2] + errors["0_to_2"]], axis=1))
        for split, g in (("train", train), ("test", test))
    }
    rows = diagnostic.evaluate_metrics(train, test, predictions)
    assert len(rows) == 6
    assert {(row["split"], row["path"]) for row in rows} == {
        (split, path) for split in ("train", "test") for path in errors}
    for row in rows:
        start, target = {"0_to_1": (0, 1), "1_to_2": (1, 2), "0_to_2": (0, 2)}[row["path"]]
        g = train if row["split"] == "train" else test
        rmse = np.sqrt(np.mean(errors[row["path"]] ** 2))
        train_std = np.sqrt(np.mean(np.var(train[:, target], axis=0, ddof=0)))
        identity = np.sqrt(np.mean((g[:, start] - g[:, target]) ** 2))
        assert row["rmse"] == pytest.approx(rmse)
        assert row["nrmse"] == pytest.approx(rmse / train_std)
        assert row["identity_rmse"] == pytest.approx(identity)
        assert row["relative_displacement_error"] == pytest.approx(rmse / identity)


def test_degenerate_metric_denominators_are_not_reported_as_zero_error():
    direct = np.zeros((3, 3, 2), dtype=float)
    predictions = {split: dict(one_step=np.ones((3, 2, 2)), rollout=np.ones((3, 3, 2)))
                   for split in ("train", "test")}
    for row in diagnostic.evaluate_metrics(direct, direct, predictions):
        assert row["rmse"] == 1.0
        assert row["nrmse"] is None and row["relative_displacement_error"] is None


def test_internal_normalizer_does_not_use_target_only_scale_two():
    direct = frozen_direct().numpy()
    expected = direct[:, :2].reshape(-1, 2)
    center, scale = diagnostic._normalizer(direct)
    np.testing.assert_allclose(center, expected.mean(axis=0))
    np.testing.assert_allclose(scale, expected.std(axis=0, ddof=0))
    changed = direct.copy()
    changed[:, 2] = 1e9
    other_center, other_scale = diagnostic._normalizer(changed)
    np.testing.assert_array_equal(center, other_center)
    np.testing.assert_array_equal(scale, other_scale)
