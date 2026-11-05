"""Single new arm with untouched historical public reference, no inference."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_ising_closure_scale_late as report
from test_ising_closure_scale_report import fixture_run, write_json


def single_fixture(root):
    fixture_run(root)
    (root / "run_uniform").rename(root / "reference_control")
    path = root / "reference_control/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("closure_scale_weighting"); manifest.pop("initial_module_sha256")
    manifest["config"]["loss"].pop("closure_scale_weights")
    manifest["q_future_physics"]["mode"] = "paired_control"
    manifest["config"]["data"]["cache"] = "/previous/completed/cache.pt"
    write_json(path, manifest)
    path = root / "reference_control/summary.json"
    summary = json.loads(path.read_text()); summary["config"] = manifest["config"]
    summary["contract"].pop("closure_scale_weighting"); summary["contract"].pop("initial_module_sha256")
    write_json(path, summary)
    path = root / "reference_control/report/encodings.npz"
    with np.load(path) as z: arrays = {key: z[key] for key in z.files if not key.startswith("rollout_draws_")}
    np.savez_compressed(path, **arrays)
    metrics_path = root / "reference_control/report/metrics.json"
    metrics = json.loads(metrics_path.read_text()); metrics["encodings_sha256"] = report.public_sha(path)
    write_json(metrics_path, metrics)
    return root


def test_single_new_training_completion_and_honest_reference_metadata(tmp_path):
    single_fixture(tmp_path)
    original = (tmp_path / "reference_control/manifest.json").read_bytes()
    result = report.report_run(tmp_path, expected_steps=6, make_plots=False)
    assert result["status"] == "complete" and result["new_training_complete"]
    assert result["qualified_reference_comparison_available"]
    assert not result["formal_fresh_paired_comparison_available"]
    assert result["coordinate_comparison"]["shared_coordinate_evaluation_verified"]
    assert not result["coordinate_comparison"]["same_fresh_initial_fingerprints_claimed"]
    reference = result["arms"]["reference_control"]
    assert not reference["initial_fingerprints_available"]
    assert "closure_scale_weighting" not in reference["original_manifest"]
    assert reference["execution_role"] == "previously_completed_reference_not_retrained"
    assert (tmp_path / "reference_control/manifest.json").read_bytes() == original
    assert reference["closure_per_C_draw"]["test"][2]["rmse"] is None
    assert result["arms"]["late"]["closure_per_C_draw"]["test"][2]["rmse"] is not None
    assert result["thermal_nu"] is None and not result["root_search_performed"]
    status = json.loads((tmp_path / "report/status.json").read_text())
    assert status["workflow_complete"] and not status["formal_fresh_paired_comparison_available"]


def test_QC_difference_qualifies_not_fake_coordinate_match(tmp_path):
    single_fixture(tmp_path)
    path = tmp_path / "run_late/report/metrics.json"
    metrics = json.loads(path.read_text()); metrics["Q_sha256"] = "different"
    write_json(path, metrics)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["new_training_complete"]
    assert not result["coordinate_comparison"]["shared_coordinate_evaluation_verified"]


def test_incomplete_reference_does_not_become_fake_execution_stage(tmp_path):
    single_fixture(tmp_path)
    (tmp_path / "reference_control/report/metrics.json").unlink()
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["status"] == "complete" and result["new_training_complete"]
    assert not result["qualified_reference_comparison_available"]


def test_different_config_or_mutated_reference_arrays_rejected(tmp_path):
    single_fixture(tmp_path)
    path = tmp_path / "reference_control/report/metrics.json"
    metrics = json.loads(path.read_text()); metrics["encodings_sha256"] = "changed"
    write_json(path, metrics)
    with pytest.raises(ValueError, match="指纹"):
        report.build_report(tmp_path, expected_steps=6)


def test_chinese_report_renders_and_refuses_overwrite(tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try: font = select_chinese_font()
    except RuntimeError: pytest.skip("No installed Chinese font")
    single_fixture(tmp_path)
    report.report_run(tmp_path, expected_steps=6, font=font)
    assert all((tmp_path / "report" / name).stat().st_size > 1000 for name in report.FIGURES)
    with pytest.raises(FileExistsError): report.report_run(tmp_path, expected_steps=6, make_plots=False)
    source = Path(report.__file__).read_text()
    assert "import torch" not in source and "load_run(" not in source and "brentq" not in source
