"""Synthetic endpoint contracts; no source fields, GPU or physical experiment."""
import numpy as np
import pytest
import torch
from torch import nn

from experiments import phi4_ru_r_calibration as calibration


class AffineMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.tensor([[1.2, .3], [-.1, .8]], dtype=torch.float64))
        self.register_buffer("bias", torch.tensor([.2, -.4], dtype=torch.float64))

    def advance(self, g):
        return g @ self.matrix.T + self.bias


def direct_fixture(repeats=4, chains=6):
    rng = np.random.default_rng(123)
    return rng.normal(size=(repeats, chains, 3, 2))


def test_predict_preserves_repeat_chain_axes_and_actual_recurrent_composition():
    direct, model = direct_fixture(), AffineMap()
    result = calibration.predict(model, direct)
    matrix, bias = model.matrix.detach().numpy(), model.bias.numpy()
    first = direct[..., 0, :] @ matrix.T + bias
    assert result["one_step"].shape == (4, 6, 2, 2)
    assert result["rollout"].shape == (4, 6, 3, 2)
    np.testing.assert_array_equal(result["rollout"][..., 0, :], direct[..., 0, :])
    np.testing.assert_allclose(result["one_step"][..., 0, :], first)
    np.testing.assert_allclose(result["one_step"][..., 1, :], direct[..., 1, :] @ matrix.T + bias)
    np.testing.assert_allclose(result["rollout"][..., 2, :], first @ matrix.T + bias)
    assert not np.allclose(first @ matrix.T + bias, direct[..., 0, :] @ matrix.T + 2 * bias)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_metrics_average_coordinates_repeats_and_chains_with_fixed_reference_stds():
    direct = direct_fixture()
    errors = {
        "0_to_1": np.arange(48).reshape(4, 6, 2) / 10.,
        "1_to_2": np.arange(48).reshape(4, 6, 2) / -13.,
        "0_to_2": np.arange(48).reshape(4, 6, 2)[::-1] / 17.,
    }
    predictions = {
        "one_step": np.stack((direct[..., 1, :] + errors["0_to_1"],
                               direct[..., 2, :] + errors["1_to_2"]), axis=-2),
        "rollout": np.stack((direct[..., 0, :], direct[..., 1, :],
                              direct[..., 2, :] + errors["0_to_2"]), axis=-2),
    }
    stds = [2.7, 9.3]
    rows = calibration.endpoint_metrics(direct, predictions, stds)
    for row, (path, start, target, _, _) in zip(rows, calibration.PATHS):
        error = errors[path]
        expected = np.sqrt(np.mean(error ** 2))
        assert row["rmse"] == pytest.approx(expected)
        assert row["nrmse"] == pytest.approx(expected / stds[target - 1])
        np.testing.assert_allclose(row["repeat_rmse"], np.sqrt(np.mean(error ** 2, axis=(1, 2))))
        np.testing.assert_allclose(row["coordinate_rmse"], np.sqrt(np.mean(error ** 2, axis=(0, 1))))
        assert row["identity_rmse"] == pytest.approx(np.sqrt(np.mean((direct[..., start, :] - direct[..., target, :]) ** 2)))
        assert row["physical_chains"] == 6 and row["C_and_subset_repeats"] == 4
        assert row["repeats_are_not_independent_physical_chains"] is True
    # Repeating measurements changes their count but cannot improve the score.
    doubled = calibration.endpoint_metrics(np.concatenate((direct, direct)),
        {key: np.concatenate((value, value)) for key, value in predictions.items()}, stds)
    for before, after in zip(rows, doubled):
        assert before["rmse"] == after["rmse"]
        assert before["nrmse"] == after["nrmse"]
        assert before["physical_chains"] == after["physical_chains"]


def test_native_metric_uses_saved_affine_decoder_and_native_target_units():
    direct = direct_fixture()
    predictions = calibration.predict(AffineMap(), direct)
    weight = np.array([[.2, .7], [-.3, 1.1], [.8, -.4]])
    bias = np.array([.1, .3, -.5])
    targets = direct @ weight.T + bias + np.array([.03, -.01, .07])
    rows = calibration.endpoint_metrics(direct, predictions, [3., 7.],
                                        feature_target=targets, decoder=(weight, bias))
    for row, (_, start, target, kind, index) in zip(rows, calibration.PATHS):
        expected = predictions[kind][..., index, :] @ weight.T + bias - targets[..., target, :]
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(expected ** 2)))
        assert row["nrmse"] == pytest.approx(row["rmse"] / [3., 7.][target - 1])
        expected_identity = direct[..., start, :] @ weight.T + bias - targets[..., target, :]
        assert row["identity_rmse"] == pytest.approx(np.sqrt(np.mean(expected_identity ** 2)))


class UnusableLabels:
    def __array__(self, *args, **kwargs):
        raise AssertionError("No coupling label may be used by these helpers")


def test_bootstrap_is_source_paired_by_condition_keeps_all_repeats_and_ignores_labels():
    direct = direct_fixture(repeats=3)
    cells = np.array([10, 30, 10, 20, 30, 20])
    # Unequal cell/repeat errors expose accidental chain or draw bootstraps.
    error = np.arange(36).reshape(3, 6, 2) / 11. + .1
    source = {
        "one_step": np.stack((direct[..., 1, :] + error,
                               direct[..., 2, :] + 2 * error), axis=-2),
        "rollout": np.stack((direct[..., 0, :], direct[..., 1, :],
                              direct[..., 2, :] + 3 * error), axis=-2),
    }
    better = {
        "one_step": np.stack((direct[..., 1, :] + .5 * error,
                               direct[..., 2, :] + error), axis=-2),
        "rollout": np.stack((direct[..., 0, :], direct[..., 1, :],
                              direct[..., 2, :] + 1.5 * error), axis=-2),
    }
    arrays = {"eval_test_direct": direct, "test_cell_ids": cells,
              "train_labels": UnusableLabels(), "test_labels": UnusableLabels(),
              "fit_direct": np.full_like(direct, 1e12)}
    predictions = {"source_seed0": {"test": source}, "scaled256_seed91": {"test": better}}
    count = 87
    result = calibration.bootstrap_differences(arrays, predictions, count=count)
    sampled = np.random.default_rng(20261030).integers(3, size=(count, 3))
    unique = np.unique(cells)
    for row, (_, _, target, kind, index) in zip(result["by_model"]["scaled256_seed91"], calibration.PATHS):
        def by_condition(prediction):
            squared = (prediction[kind][..., index, :] - direct[..., target, :]) ** 2
            return np.array([squared[:, cells == cell].mean() for cell in unique])
        a, b = by_condition(better), by_condition(source)
        expected = np.sqrt(a[sampled].mean(1)) - np.sqrt(b[sampled].mean(1))
        assert row["rmse_difference"] == pytest.approx(np.sqrt(a.mean()) - np.sqrt(b.mean()))
        np.testing.assert_allclose(row["interval95"], np.quantile(expected, [.025, .975]))
    assert result["C_repeats_kept_together"] and result["paired_across_arms"]
    # The fit cache and all coupling labels are irrelevant to TEST evaluation.
    arrays["fit_direct"] *= -1e6
    arrays["train_labels"], arrays["test_labels"] = None, None
    assert calibration.bootstrap_differences(arrays, predictions, count=count) == result


def test_full_jacobian_statistics_include_identity_and_do_not_accumulate_parameter_gradients():
    model = AffineMap()
    points = direct_fixture()[..., :2, :]
    info, jac = calibration.jacobian_statistics(model, points)
    matrix = model.matrix.detach().numpy()
    expected = np.linalg.svd(matrix, compute_uv=False)[0]
    assert jac.shape == (4 * 6 * 2, 2, 2)
    np.testing.assert_allclose(jac, np.broadcast_to(matrix, jac.shape))
    for key in ("largest_singular_value_median", "largest_singular_value_p95", "largest_singular_value_max"):
        assert info[key] == pytest.approx(expected)
    assert info["count"] == 48
    assert info["full_DR_includes_identity"] and info["values_are_not_fixed_point_eigenvalues"]
    assert model.matrix.grad is None


def test_repeat_stability_tracks_paired_error_noise_without_inflating_physical_count():
    direct = direct_fixture()
    predictions = calibration.predict(AffineMap(), direct)
    rows = calibration.repeat_stability(direct, predictions)
    rms = lambda value: np.sqrt(np.mean(np.var(value, axis=0, ddof=0)))
    for row, (_, start, target, kind, index) in zip(rows, calibration.PATHS):
        output = predictions[kind][..., index, :]
        assert row["input_repeat_rms"] == pytest.approx(rms(direct[..., start, :]))
        assert row["prediction_repeat_rms"] == pytest.approx(rms(output))
        assert row["paired_error_repeat_rms"] == pytest.approx(rms(output - direct[..., target, :]))
    constant = np.repeat(direct[:1], 4, axis=0)
    for row in calibration.repeat_stability(constant, calibration.predict(AffineMap(), constant)):
        assert row["prediction_to_input_repeat_rms"] is None
