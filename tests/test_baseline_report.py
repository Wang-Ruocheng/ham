"""Synthetic numeric-only figure checks; never load a model or physical cache."""
from copy import deepcopy
import json

import numpy as np
import pytest

from recurrent_rg.baselines.report import (
    ARM_FILES, COMPARISON_FILES, _C_mi, _R_component, _comparison_value, _display_name, _grouped,
    _history_points, _native_loss,
    _variance_note,
    _root_positions, _validate_arrays, write_arm_report, write_comparison_report,
)


def arrays(test=True):
    rng = np.random.default_rng(91)
    n, draws = 6, 2
    g = rng.normal(size=(n, draws, 3, 5))
    r = g[:, :, :2] + .1 * g[:, :, :2] ** 2
    rollout = g.copy()
    rollout[:, :, 1:] += .02
    p = rng.uniform(.5, 1., size=(n, draws, 3, 16))
    p /= p.sum(-1, keepdims=True)
    return dict(g=g, r_one=r, rollout=rollout, labels=np.array([.2, .2, .4, .4, .6, .6]),
        split=np.array([0, 0, 1, 1, 1, 1]) if test else np.zeros(n, dtype=int),
        p16=p, decoded_direct=p + .001, decoded_rollout=p + .003,
        pca_fraction=np.array([.98, .01, .005, .003, .002]))


def metrics():
    return dict(seed=91, test=dict(latent=dict(rollout_nrmse=.12),
        motif16=dict(rollout_rmse=.003)), model_specific_note="保留原始指标")


def test_grouped_R_is_mean_of_actual_R_not_R_of_mean():
    data = _validate_arrays(arrays())
    mask = data["split"] == 1
    labels, grouped = _grouped(data, mask)
    for index, value in enumerate(labels):
        selected = mask & (data["labels"] == value)
        expected = data["r_one"][selected].mean(axis=(0, 1))
        np.testing.assert_array_equal(grouped["r_one"][index], expected)
        not_actual = grouped["g"][index, :2] + .1 * grouped["g"][index, :2] ** 2
        assert not np.allclose(expected, not_actual)


@pytest.mark.parametrize("key,bad", [
    ("r_one", np.ones((6, 2, 3, 5))),
    ("pca_fraction", np.array([.5, .1, .2, .1, .1])),
    ("split", np.ones(6) * 2),
    ("labels", np.ones((6, 1))),
    ("p16", np.ones((6, 2, 3, 16))),
    ("decoded_direct", np.full((6, 2, 3, 16), np.nan)),
])
def test_array_contract_rejects_ambiguous_or_nonfinite_inputs(key, bad):
    data = arrays()
    data[key] = bad
    with pytest.raises(ValueError):
        _validate_arrays(data)


def test_no_root_means_no_invented_projection_and_explicit_roots_are_only_read():
    assert _root_positions(None) == []
    assert _root_positions([]) == []
    assert _root_positions([dict(g=[.2, 0, 0, 0, 0]), dict(pc1=.3)]) == [(1, .2), (2, .3)]
    for invalid in ([{}], [dict(g=[.1])], [dict(pc1=[.1])], dict(g=[0] * 5)):
        with pytest.raises(ValueError):
            _root_positions(invalid)


def test_optional_R_of_conditional_C_mean_is_per_chain_not_all_chain_R_mean():
    data = arrays()
    data["mean_R"] = data["r_one"].mean(axis=1)
    per_chain_mean = data["g"][:, :, :2].mean(axis=1)
    data["r_of_C_mean"] = per_chain_mean + .1 * per_chain_mean ** 2
    checked = _validate_arrays(data)
    labels, grouped = _grouped(checked, checked["split"] == 1)
    for index, label in enumerate(labels):
        selected = (checked["split"] == 1) & (checked["labels"] == label)
        np.testing.assert_allclose(grouped["r_of_C_mean"][index], data["r_of_C_mean"][selected].mean(0))
        assert not np.allclose(grouped["r_of_C_mean"][index], grouped["mean_R"][index])
    bad = deepcopy(data)
    bad["mean_R"] += 1.
    with pytest.raises(ValueError, match="mean_R"):
        _validate_arrays(bad)
    bad = deepcopy(data)
    bad["r_of_C_mean"] = np.ones((6, 2, 1))
    with pytest.raises(ValueError):
        _validate_arrays(bad)


def test_retained_latent_variance_and_original_512_motif_variance_have_distinct_denominators():
    data = arrays()
    data["input_motif_pca_selected_fraction"] = .8 * data["pca_fraction"]
    data["retained_input_variance_fraction"] = np.asarray(.8)
    checked = _validate_arrays(data)
    assert "五维 g 总方差归一化" in _variance_note(checked)
    assert "80%" in _variance_note(checked)
    assert np.isclose(checked["pca_fraction"].sum(), 1.)
    assert np.isclose(checked["input_motif_pca_selected_fraction"].sum(), .8)
    bad = deepcopy(data)
    bad["pca_fraction"] = bad["input_motif_pca_selected_fraction"].copy()
    with pytest.raises(ValueError, match="归一化"):
        _validate_arrays(bad)
    bad = deepcopy(data)
    bad["retained_input_variance_fraction"] = np.asarray(.9)
    with pytest.raises(ValueError, match="保留率"):
        _validate_arrays(bad)
    del bad["retained_input_variance_fraction"]
    with pytest.raises(ValueError, match="成对"):
        _validate_arrays(bad)


def test_zero_retained_variance_remains_all_zero_not_a_uniform_distribution():
    data = arrays()
    data["pca_fraction"] = np.zeros(5)
    data["input_motif_pca_selected_fraction"] = np.zeros(5)
    data["retained_input_variance_fraction"] = np.asarray(0.)
    checked = _validate_arrays(data)
    np.testing.assert_array_equal(checked["pca_fraction"], np.zeros(5))
    assert "全零" in _variance_note(checked) and "保留0%" in _variance_note(checked)
    data["pca_fraction"] = np.ones(5) / 5
    with pytest.raises(ValueError, match="零塌缩"):
        _validate_arrays(data)


def test_native_C_and_common_R_history_components_do_not_mix():
    history = [dict(stage="C", step=0, native_loss=-.5, mi_estimate=.4, loss=999.),
               dict(stage="C", step=10, native_loss=-.7, mi_estimate=.6),
               dict(stage="R", step=0, loss=.3, one_step=.1, rollout=.2),
               dict(stage="R", step=10, loss=.1, one_step=.04, rollout=.06)]
    np.testing.assert_array_equal(_history_points(history, _native_loss), [[0, -.5], [10, -.7]])
    np.testing.assert_array_equal(_history_points(history, _C_mi), [[0, .4], [10, .6]])
    np.testing.assert_array_equal(_history_points(history, lambda row: _R_component(row, "loss")), [[0, .3], [10, .1]])
    assert _R_component(history[0], "loss") is None


def _font_or_skip():
    from scripts.report_ising_zh import select_chinese_font
    try:
        select_chinese_font()
    except RuntimeError:
        pytest.skip("No preinstalled Chinese font; no download/install during tests")


def test_arm_report_writes_chinese_pngs_preserves_metrics_and_inputs(tmp_path):
    _font_or_skip()
    data, supplied_metrics = arrays(), metrics()
    before = {key: value.copy() for key, value in data.items()}
    metrics_before = deepcopy(supplied_metrics)
    history = [dict(step=0, train_loss=.4, test_loss=.5),
               dict(step=10, train_loss=.2, test_loss=.3)]
    output = tmp_path / "arm"
    files = write_arm_report(output, data, supplied_metrics, history)
    assert files == list(ARM_FILES)
    assert all((output / name).is_file() for name in files)
    assert all((output / name).stat().st_size > 1000 for name in files if name.endswith(".png"))
    receipt = json.loads((output / "arm_report.json").read_text())
    assert receipt["metrics"] == supplied_metrics == metrics_before
    assert receipt["root_marker_count"] == 0
    assert receipt["fixed_points"] is None
    assert receipt["plotting_split"] == "test"
    assert receipt["model_loaded"] is False and receipt["inference_performed"] is False
    assert receipt["PCA_refitted"] is False and receipt["roots_recomputed"] is False
    assert "不是R(平均g)" in receipt["map_definition"]
    assert json.loads((output / "history.json").read_text()) == history
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    with pytest.raises(FileExistsError):
        write_arm_report(output, data, supplied_metrics, history)


def test_train_only_report_and_empty_history_do_not_fabricate_test_or_roots(tmp_path):
    _font_or_skip()
    data = arrays(test=False)
    fixed = [dict(g=[.2, 0., 0., 0., 0.], residual=1e-10)]
    write_arm_report(tmp_path / "train", data, {"train": {}}, [], fixed_points=fixed)
    receipt = json.loads((tmp_path / "train/arm_report.json").read_text())
    assert receipt["plotting_split"] == "train"
    assert receipt["root_marker_count"] == 1 and receipt["fixed_points"] == fixed
    assert not (tmp_path / "train/metrics.json").exists()


def test_two_panel_native_history_and_C_mean_diagnostics_render_without_refitting(tmp_path):
    _font_or_skip()
    data = arrays()
    data["mean_R"] = data["r_one"].mean(1)
    mean_g = data["g"][:, :, :2].mean(1)
    data["r_of_C_mean"] = mean_g + .1 * mean_g ** 2
    data["input_motif_pca_selected_fraction"] = .8 * data["pca_fraction"]
    data["retained_input_variance_fraction"] = np.asarray(.8)
    history = [dict(stage="C", step=0, native_loss=-.5, mi_estimate=.4),
               dict(stage="C", step=10, native_loss=-.7, mi_estimate=.6),
               dict(stage="R", step=0, loss=.3, one_step=.1, rollout=.2),
               dict(stage="R", step=10, loss=.1, one_step=.04, rollout=.06)]
    write_arm_report(tmp_path / "two_stage", data, metrics(), history)
    receipt = json.loads((tmp_path / "two_stage/arm_report.json").read_text())
    assert receipt["conditional_mean_R_diagnostic_available"] is True
    assert "每条物理链" in receipt["conditional_mean_R_definition"]
    assert "不是R作用于同K全链" in receipt["conditional_mean_R_definition"]
    assert receipt["array_shapes"]["r_of_C_mean"] == [6, 2, 5]
    assert "80%" in receipt["variance_fraction_definition"]
    assert (tmp_path / "two_stage/损失曲线.png").stat().st_size > 1000


def test_comparison_extracts_last_not_best_and_falls_back_to_train():
    values = dict(test=dict(latent=dict(rollout_nrmse=[.1, .7])))
    assert _comparison_value(values, "latent", "rollout_nrmse") == .7
    assert _comparison_value(dict(train=dict(motif16=dict(rollout_rmse=.2))), "motif16", "rollout_rmse") == .2
    assert _comparison_value(dict(test={}), "motif16", "rollout_rmse") is None
    assert "蒙卡" in _display_name("nmcrg", 1)
    assert "互信息" in _display_name("rsmine", 2)
    assert "冻结" in _display_name("qcr_c_pca", 3)
    assert _display_name("unknown_method", 4) == "方法4"


def test_comparison_numeric_report_has_explicit_cross_C_caution_and_no_metrics_overwrite(tmp_path):
    _font_or_skip()
    summaries = dict(nmcrg=metrics(), rsmine=dict(train=dict(latent=dict(rollout_nrmse=.3))))
    before = deepcopy(summaries)
    reference = metrics()
    files = write_comparison_report(tmp_path / "compare", summaries, reference)
    assert files == list(COMPARISON_FILES)
    receipt = json.loads((tmp_path / "compare/comparison_report.json").read_text())
    assert receipt["summaries"] == summaries == before
    assert receipt["reference"] == reference
    assert receipt["cross_method_latent_ranking_valid"] is False
    assert receipt["missing_values_are_zero"] is False
    assert "不同粗粒化" in receipt["comparison_caution"]
    assert not (tmp_path / "compare/metrics.json").exists()
    assert (tmp_path / "compare/方法比较.png").stat().st_size > 1000


def test_report_rejects_nonfinite_metrics_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError):
        write_arm_report(tmp_path / "bad", arrays(), {"metric": float("nan")}, [])
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError):
        write_comparison_report(tmp_path / "badcompare", {"method": {"x": object()}})
    assert not (tmp_path / "badcompare").exists()
