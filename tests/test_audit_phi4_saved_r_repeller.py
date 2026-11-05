import numpy as np
import pytest

from scripts.audit_phi4_saved_r_repeller import closure_comparison, saved_r_query


def test_comparison_keeps_three_distinct_branches_and_actual_weights():
    direct = np.asarray([[[1.], [2.], [4.]], [[2.], [3.], [5.]]])
    one = np.asarray([[[2.], [5.]], [[3.], [6.]]])
    rollout = np.asarray([[[1.], [2.], [4.]], [[2.], [3.], [5.]]])
    result = closure_comparison(dict(direct_g=direct, one_step_g=one,
        rollout_g=rollout, splits=np.asarray(["train", "test"])))
    for split in ("train", "test"):
        rows = result[split]
        assert [r["objective_weight"] for r in rows] == [5, 2.5, 2.5]
        assert [r["rmse"] for r in rows] == [0, 1, 0]
        assert [r["identity_rmse"] for r in rows] == [1, 2, 3]
        assert rows[1]["mse_ratio_to_identity"] == .25


def test_saved_query_uses_full_R_not_residual_and_does_not_search_roots():
    grid = np.linspace(.5, .8, 301)[:, None]
    result = saved_r_query(dict(flow_grid=grid, flow_next=1.25*grid-.148), .64)
    assert np.isclose(result["R_g"], .652)
    assert np.isclose(result["drift"], .012)
    assert np.isclose(result["saved_grid_finite_difference_derivative"], 1.25)
    assert result["root_search_performed"] is False


def test_query_does_not_extrapolate():
    arrays = dict(flow_grid=np.asarray([[.6], [.7]]), flow_next=np.asarray([[.61], [.72]]))
    with pytest.raises(ValueError, match="inside"):
        saved_r_query(arrays, .8)
