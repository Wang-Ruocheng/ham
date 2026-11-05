import copy

import numpy as np
import pytest

from scripts.plot_phi4_b2_saved_q_diagnostic import (
    KAPPAS, correlation, decoder_geometry_summary, descriptive_summary, grouped_statistics,
    rank_average_ties, reconstruction_summary,
)


def toy_arrays():
    values = np.repeat(KAPPAS, 4)[:, None, None] + np.arange(4)[None, :, None] * .1
    values += np.tile(np.asarray([-.003, -.001, .001, .003]), 15)[:, None, None]
    target = np.zeros((60, 4, 625))
    target[..., 0] = .5
    target[..., 1] = .5
    target[:, :, 0] += np.repeat(KAPPAS, 4)[:, None] * .1
    target[:, :, 1] -= np.repeat(KAPPAS, 4)[:, None] * .1
    return dict(direct_full_g=values, direct_g=values - .01,
                one_step_g=values[:, 1:] - .01,
                cell_ids=np.repeat(np.arange(15), 4),
                labels=np.column_stack((np.repeat(KAPPAS, 4), np.ones(60), np.zeros(60))),
                splits=np.repeat(["test" if i % 3 == 1 else "train" for i in range(15)], 4),
                target_features=target, reconstructed_features=target.copy())


def test_four_scales_group_physical_chains_without_conflation():
    arrays = toy_arrays()
    rows = grouped_statistics(arrays)
    assert len(rows) == 15
    assert np.asarray(rows[0]["pooled_g_by_chain"]).shape == (4, 4)
    assert rows[1]["split"] == "test"
    np.testing.assert_allclose(rows[0]["mean"], KAPPAS[0] + np.arange(4) * .1)
    np.testing.assert_allclose(rows[0]["chain_sd_ddof1"], np.std([-.003, -.001, .001, .003], ddof=1))


def test_reject_missing_chain_or_scale_and_condition_metadata():
    arrays = toy_arrays()
    arrays["direct_full_g"] = arrays["direct_full_g"][:59]
    with pytest.raises(ValueError):
        grouped_statistics(arrays)
    arrays = toy_arrays()
    arrays["splits"][0] = "test"
    with pytest.raises(ValueError):
        grouped_statistics(arrays)
    arrays = toy_arrays()
    arrays["labels"][0, 0] += .001
    with pytest.raises(ValueError):
        grouped_statistics(arrays)


def test_tie_ranks_and_correlations():
    np.testing.assert_array_equal(rank_average_ties([3, 1, 1, 4]), [3, 1.5, 1.5, 4])
    assert correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert correlation([1, 1, 1], [3, 2, 1]) is None
    with pytest.raises(ValueError):
        rank_average_ties([np.nan])


def test_signed_observed_gaps_do_not_assume_coordinate_orientation():
    arrays = toy_arrays()
    rows = grouped_statistics(arrays)
    metrics = descriptive_summary(rows, arrays)
    assert all(row["strictly_increasing_condition_means"] for row in metrics["by_scale"])
    assert all(row["signed_high_min_minus_low_max"] > 0 for row in metrics["by_scale"])
    assert all(row["spearman_condition_means"] == pytest.approx(1) for row in metrics["by_scale"])
    np.testing.assert_allclose(metrics["pooled_vs_half_rmse_by_scale"], .01)
    arrays["direct_full_g"] *= -1
    reversed_metrics = descriptive_summary(grouped_statistics(arrays), arrays)
    assert all(row["strictly_decreasing_condition_means"] for row in reversed_metrics["by_scale"])
    assert all(row["signed_low_min_minus_high_max"] > 0 for row in reversed_metrics["by_scale"])


def test_nonmonotonic_microscopic_coordinate_is_detected_separately():
    arrays = toy_arrays()
    arrays["direct_full_g"][4:8, 0] += 1
    metrics = descriptive_summary(grouped_statistics(arrays), arrays)
    assert not metrics["by_scale"][0]["strictly_increasing_condition_means"]
    assert metrics["by_scale"][0]["negative_adjacent_differences"] == 1
    assert metrics["by_scale"][1]["strictly_increasing_condition_means"]


def test_reconstruction_uses_train_only_constant_and_no_new_decoder():
    arrays = toy_arrays()
    result = reconstruction_summary(arrays)
    assert result["splits"]["test"]["learned_rmse_by_scale"] == [0] * 4
    assert np.asarray(result["learned_rmse_by_physical_chain_and_scale"]).shape == (60, 4)
    original_train = copy.deepcopy(result["splits"]["train"])
    test = arrays["splits"] == "test"
    arrays["target_features"][test, :, 0] += .01
    result = reconstruction_summary(arrays)
    assert result["splits"]["train"] == original_train


def test_decoder_geometry_recovers_existing_unit_direction_without_parameter_fit():
    arrays = toy_arrays()
    direction = np.zeros(625)
    direction[0] = 1.
    arrays["target_features"] = arrays["direct_g"] * direction
    arrays["reconstructed_features"] = arrays["target_features"].copy()
    result = decoder_geometry_summary(arrays)
    assert result["saved_decoder_norm"] == pytest.approx(1.)
    assert result["saved_decoder_abs_cosine_to_global_TRAIN_PC1"] == pytest.approx(1.)
    assert result["TRAIN_actual_reconstruction_squared_L2"] == 0.
    np.testing.assert_allclose(result["fixed_decoder_optimum_g_by_physical_chain_and_scale"],
                               arrays["direct_g"][..., 0])
    arrays["reconstructed_features"][0, 0, 1] += .01
    with pytest.raises(ValueError, match="shared unit-norm affine"):
        decoder_geometry_summary(arrays)
