import numpy as np
import pytest

from scripts.plot_phi4_saved_q_scales import grouped_statistics, descriptive_summary


def synthetic():
    kappa = np.repeat(.50 + .03*np.arange(13), 4)
    chain_offset = np.tile(np.arange(4)*.001, 13)
    g = kappa[:, None] + np.arange(3)[None, :]*.1 + chain_offset[:, None]
    return dict(direct_full_g=g[..., None], direct_g=(g+.002)[..., None],
        one_step_g=(g[:, 1:]+.003)[..., None], cell_ids=np.repeat(np.arange(13), 4),
        labels=np.column_stack((kappa, np.ones(52), np.zeros(52))), splits=np.repeat("train", 52))


def test_four_chain_means_and_descriptive_error_not_field_count():
    a = synthetic()
    rows = grouped_statistics(a)
    assert len(rows) == 13
    assert rows[0]["physical_chains"] == 4
    assert np.allclose(rows[0]["mean"], [.5015, .6015, .7015])
    assert np.allclose(rows[0]["chain_sd_ddof1"], np.std(np.arange(4)*.001, ddof=1))
    m = descriptive_summary(rows, a)
    assert m["means_strictly_increasing_with_kappa_by_scale"] == [True]*3
    assert np.isclose(m["max_absolute_pooled_vs_two_half_difference"], .002)
    assert np.allclose(m["formal_two_half_one_step_rmse_by_transition"], .001)


def test_row_order_does_not_change_grouped_results():
    a = synthetic()
    order = np.random.default_rng(7).permutation(52)
    shuffled = grouped_statistics({k: v[order] for k, v in a.items()})
    original = grouped_statistics(a)
    for left, right in zip(shuffled, original):
        for key in ("cell", "kappa", "split", "physical_chains"):
            assert left[key] == right[key]
        for key in ("mean", "chain_sd_ddof1", "mean_empirical_delta", "half_encoder_mean"):
            assert np.allclose(left[key], right[key], atol=1e-14, rtol=1e-14)
        assert np.allclose(sorted(left["pooled_g_by_chain"]), sorted(right["pooled_g_by_chain"]))


def test_missing_chain_is_not_silently_dropped():
    a = synthetic()
    with pytest.raises(ValueError, match="four physical chains"):
        grouped_statistics({k: v[1:] for k, v in a.items()})


def test_nonfinite_q_output_rejected():
    a = synthetic()
    a["direct_full_g"][0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        grouped_statistics(a)


def test_scale_shape_mismatch_rejected():
    a = synthetic()
    a["direct_full_g"] = a["direct_full_g"][:, :2]
    with pytest.raises(ValueError, match="three scales"):
        grouped_statistics(a)
