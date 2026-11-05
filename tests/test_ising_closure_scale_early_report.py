"""Synthetic public reporting tests; no trained model or physical cache opened."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import report_ising_closure_scale_early as report
from test_ising_closure_scale_late_report import single_fixture
from test_ising_closure_scale_report import write_json


def triple_fixture(root, steps=6):
    single_fixture(root)
    (root / "run_late").rename(root / "reference_late")
    shutil.copytree(root / "reference_late", root / "run_early")
    path = root / "run_early/manifest.json"
    manifest = json.loads(path.read_text())
    weights = [1.8, .2]
    manifest["config"]["loss"]["closure_scale_weights"] = weights
    manifest["config"]["data"]["cache"] = "/new/unchanged_data/cache.pt"
    manifest["closure_scale_weighting"]["weights"] = weights
    manifest["closure_scale_weighting"]["normalized_weights"] = [.9, .1]
    write_json(path, manifest)
    path = root / "run_early/summary.json"
    summary = json.loads(path.read_text())
    summary["config"] = manifest["config"]
    summary["contract"]["closure_scale_weighting"] = manifest["closure_scale_weighting"]
    write_json(path, summary)
    return root


def test_three_way_report_keeps_reference_roles_and_original_metadata(tmp_path):
    triple_fixture(tmp_path)
    originals = {arm: (tmp_path / arm / "manifest.json").read_bytes() for arm in ("reference_control", "reference_late")}
    result = report.report_run(tmp_path, expected_steps=6, make_plots=False, execution_smoke=True)
    assert result["status"] == "complete" and result["workflow_complete"]
    assert result["new_training_arms"] == ["early"]
    assert result["qualified_reference_comparison_available"]
    assert not result["formal_fresh_paired_comparison_available"]
    assert result["execution_smoke"] and not result["scientific_result"]
    assert result["thermal_nu"] is None and not result["root_search_performed"]
    assert result["dataset_receipt"]["samples_are_iid"] is False
    for arm in originals:
        assert (tmp_path / arm / "manifest.json").read_bytes() == originals[arm]
        assert result["arms"][arm]["execution_role"] == "previously_completed_reference_not_retrained"
    reference = result["arms"]["reference_control"]
    assert not reference["initial_fingerprints_available"]
    assert "closure_scale_weighting" not in reference["original_manifest"]
    assert reference["closure_per_C_draw"]["test"][2]["rmse"] is None
    assert result["arms"]["early"]["closure_per_C_draw"]["test"][2]["rmse"] is not None
    assert len(result["pairwise_coordinate_comparison"]) == 3
    assert all(v["saved_coordinate_values_equal"] for v in result["pairwise_coordinate_comparison"].values())
    assert result["arms"]["early"]["physical_chain_count"] == dict(train=4, test=4)
    assert result["arms"]["early"]["C_draw_count_per_chain"] == dict(train=2, test=2)
    with np.load(tmp_path / "report/evaluation_arrays.npz", allow_pickle=False) as z:
        assert "early__rollout_draws_test" in z and "reference_control__rollout_draws_test" not in z
    status = json.loads((tmp_path / "report/status.json").read_text())
    assert status["execution_smoke"] and not status["scientific_result"]
    assert status["report_sha256"] == report.public_sha(tmp_path / "report/report.json")


def test_fingerprint_difference_not_conflated_with_equal_saved_coordinates(tmp_path):
    triple_fixture(tmp_path)
    path = tmp_path / "run_early/report/metrics.json"
    metrics = json.loads(path.read_text()); metrics["Q_sha256"] = "different"
    write_json(path, metrics)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    pair = result["pairwise_coordinate_comparison"]["reference_late__early"]
    assert pair["saved_coordinate_values_equal"] and not pair["shared_QC_fingerprints"]
    assert not pair["shared_coordinate_evaluation_verified"]


@pytest.mark.parametrize("mutation,match", [
    ("weight", "尺度权重"), ("data", "数据指纹"), ("config", "配置不同"),
    ("labels", "无标签"), ("future", "未来物理"), ("source", "全新初始化"),
    ("initial", "初始指纹"), ("history", "非有限"), ("array_sha", "指纹不符"),
])
def test_mutated_contract_rejected(tmp_path, mutation, match):
    triple_fixture(tmp_path)
    manifest_path, summary_path = tmp_path / "run_early/manifest.json", tmp_path / "run_early/summary.json"
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    if mutation == "weight": manifest["config"]["loss"]["closure_scale_weights"] = [.2, 1.8]
    elif mutation == "data": manifest["data_sha256"] = "different"
    elif mutation == "config": manifest["config"]["training"]["learning_rate"] = .01
    elif mutation == "labels": manifest["labels_used_in_training"] = True
    elif mutation == "future": manifest["q_future_physics"]["weight"] = .1
    elif mutation == "source": manifest["source"] = "previous.pt"
    elif mutation == "initial":
        manifest["initial_module_sha256"]["R"] = "different"
        summary["contract"]["initial_module_sha256"] = manifest["initial_module_sha256"]
    elif mutation == "history": summary["history"][-1]["train_commutator"] = float("nan")
    elif mutation == "array_sha":
        path = tmp_path / "run_early/report/metrics.json"
        metrics = json.loads(path.read_text()); metrics["encodings_sha256"] = "changed"; write_json(path, metrics)
    summary["config"] = copy.deepcopy(manifest["config"])
    write_json(manifest_path, manifest); write_json(summary_path, summary)
    with pytest.raises(ValueError, match=match): report.build_report(tmp_path, expected_steps=6)


def test_missing_reference_and_early_stop_remain_honest(tmp_path):
    triple_fixture(tmp_path)
    (tmp_path / "reference_late/report/metrics.json").unlink()
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["new_training_complete"] and not result["qualified_reference_comparison_available"]
    path = tmp_path / "run_early/summary.json"
    summary = json.loads(path.read_text()); summary["actual_epochs"] = 3
    summary["history"][-1]["epoch"] = 3; summary["stopped_by_wall_limit"] = True; write_json(path, summary)
    path = tmp_path / "run_early/report/metrics.json"
    metrics = json.loads(path.read_text()); metrics["actual_steps"] = 3; write_json(path, metrics)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["status"] == "incomplete" and not result["workflow_complete"]


def test_each_scale_retained_even_at_old_stride_alias_shape():
    direct = np.arange(324 * 5 * 3, dtype=float).reshape(324, 5, 3, 1)
    arrays = dict(direct_draws_test=direct, R_g_draws_test=direct + 1.)
    g, mapped, levels, per_draw = report.stratified_flow_points(arrays)
    assert per_draw and len(g) == 240
    assert np.unique(levels, return_counts=True)[1].tolist() == [80, 80, 80]
    np.testing.assert_array_equal(mapped - g, np.ones(240))
    for level in range(3): assert np.all(g[levels == level] % 3 == level)
    with pytest.raises(ValueError): report.stratified_flow_points(arrays, max_points_per_scale=0)


def test_identity_bounds_include_omitted_and_mean_markers():
    direct = np.zeros((4, 3, 1))
    arrays = dict(direct_test=direct, R_g_test=direct + 1, R_of_mean_g_test=direct - 9)
    arrays["direct_draws_test"] = np.zeros((4, 2, 3, 1))
    arrays["R_g_draws_test"] = np.zeros((4, 2, 3, 1)); arrays["R_g_draws_test"][1, 1, 2, 0] = 17
    assert report.flow_identity_bounds(arrays) == (-9., 17.)


def test_three_way_metrics_keep_fixed_reference_denominator(tmp_path):
    triple_fixture(tmp_path)
    result, panels = report.build_report(tmp_path, expected_steps=6)
    for arm in report.ARMS:
        rows = result["arms"][arm]["closure_of_C_means"]["test"]
        assert [(v["source"], v["target"]) for v in rows] == [(108, 36), (36, 12), (108, 12)]
        assert [v["fixed_uniform_train_target_std"] for v in rows] == [v["train_target_std"] for v in rows]
    assert report.stratified_flow_points(panels["early"])[3]


def test_chinese_render_and_no_overwrite_or_model_readers(tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try: font = select_chinese_font()
    except RuntimeError: pytest.skip("No installed Chinese font")
    triple_fixture(tmp_path)
    report.report_run(tmp_path, expected_steps=6, font=font, execution_smoke=True)
    assert all((tmp_path / "report" / name).stat().st_size > 1000 for name in report.FIGURES)
    with pytest.raises(FileExistsError): report.report_run(tmp_path, expected_steps=6, make_plots=False)
    source = Path(report.__file__).read_text()
    assert "import torch" not in source and "load_run(" not in source and "brentq" not in source
