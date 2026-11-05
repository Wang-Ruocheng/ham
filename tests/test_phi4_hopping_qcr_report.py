"""Synthetic public-array reports, never physical or training evidence."""
import json

import numpy as np
import pytest

from scripts import report_phi4_hopping_qcr as report


def fixture_run(tmp_path, dim=1):
    root = tmp_path / f"run{dim}"
    root.mkdir()
    rng = np.random.default_rng(15)
    direct = rng.normal(size=(12, 3, dim)) * .1
    target = rng.uniform(size=(12, 3, 625))
    target /= target.sum(-1, keepdims=True)
    labels = np.column_stack((np.repeat([.5, .68, .86], 4), np.ones(12), np.zeros(12)))
    arrays = dict(direct_g=direct, one_step_g=direct[:, 1:] + .01,
        rollout_g=np.concatenate((direct[:, :1], direct[:, 1:] + .02), axis=1),
        labels=labels, splits=np.repeat(["train", "test", "train"], 4), cell_ids=np.repeat(np.arange(3), 4),
        target_features=target, reconstructed_features=target + .0001)
    if dim == 1:
        arrays.update(flow_grid=np.linspace(-.3, .3, 21)[:, None], flow_next=np.linspace(-.2, .4, 21)[:, None])
    np.savez(root / "evaluation_arrays.npz", **arrays)
    manifest = dict(recipe=report.RECIPE, schema_version=5, options=dict(steps=2),
        labels_used_in_training=False, field_representation="phi",
        evaluation_label_columns=["kappa", "quartic_lambda", "h"])
    summary = dict(recipe=report.RECIPE, status="complete", labels_used_in_training=False,
        actual_steps=2, thermal_nu=None,
        optimizer_steps={name: dict(min=2, max=2, parameters=4) for name in ("Q", "C", "R")})
    history = [dict(step=step, q_loss=1. / step, c_loss=-.1, r_loss=.2, reconstruction=.3,
                    information_proxy=.2, closure_one_step=.1, closure_rollout=.2) for step in (1, 2)]
    for name, value in (("manifest", manifest), ("summary", summary), ("history", history)):
        (root / f"{name}.json").write_text(json.dumps(value))
    return root, arrays


def test_public_loader_has_explicit_new_recipe_and_no_model_io(tmp_path, monkeypatch):
    root, _ = fixture_run(tmp_path)
    import torch
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("report loaded checkpoint"))
    manifest, summary, history, arrays, pins = report.load_public_run(root)
    assert manifest["recipe"] == report.RECIPE
    assert summary["actual_steps"] == history[-1]["step"] == 2
    assert arrays["target_features"].shape == (12, 3, 625)
    assert set(pins) == set(report.INPUT_FILES)


@pytest.mark.parametrize("artifact,key,value", [
    ("manifest", "recipe", "phi4_continuous_gaussian_soft_statistics_v1"),
    ("manifest", "evaluation_label_columns", ["r", "u", "h"]),
    ("manifest", "field_representation", "psi"),
    ("manifest", "labels_used_in_training", True),
    ("summary", "actual_steps", 1),
    ("summary", "thermal_nu", 1.),
    ("summary", "status", "running"),
])
def test_mixed_or_incomplete_public_contract_rejected(tmp_path, artifact, key, value):
    root, _ = fixture_run(tmp_path)
    path = root / f"{artifact}.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.load_public_run(root)


@pytest.mark.parametrize("mutation", ["joint_mass", "split", "labels", "nonfinite", "flow_order", "clock"])
def test_corrupt_evaluation_rejected(tmp_path, mutation):
    root, arrays = fixture_run(tmp_path)
    if mutation == "joint_mass": arrays["target_features"] *= 2.
    elif mutation == "split": arrays["splits"][0] = "test"
    elif mutation == "labels": arrays["labels"][4:8] = arrays["labels"][0]
    elif mutation == "nonfinite": arrays["direct_g"][0, 0, 0] = np.nan
    elif mutation == "flow_order": arrays["flow_grid"] = arrays["flow_grid"][::-1]
    elif mutation == "clock":
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        summary["optimizer_steps"]["C"]["min"] = 1
        path.write_text(json.dumps(summary))
    np.savez(root / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError):
        report.load_public_run(root)


@pytest.mark.parametrize("dim", [1, 2])
def test_chinese_report_saves_three_required_plots_without_mutating_inputs(tmp_path, dim):
    root, _ = fixture_run(tmp_path, dim)
    before = {name: report._sha(root / name) for name in report.INPUT_FILES}
    result = report.render_report(root)
    assert not result["fixed_point_search_performed"]
    assert result["thermal_nu"] is None and result["critical_kappa"] is None
    assert result["source_sha256"] == before
    assert all((root / "report" / name).stat().st_size > 1000 for name in result["plots"])
    assert (root / "report" / "report.json").is_file()
    assert before == {name: report._sha(root / name) for name in report.INPUT_FILES}
    with pytest.raises(FileExistsError): report.render_report(root)


def test_report_cannot_overwrite_source_directory(tmp_path):
    root, _ = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        report.render_report(root, root)
