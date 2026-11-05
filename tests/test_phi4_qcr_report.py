"""Synthetic public-array tests: no sampler, model load, GPU, or real data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_qcr as report


def _fixture(root, *, constant=False):
    root.mkdir()
    rng = np.random.default_rng(31)
    arrays = {}
    for split, r, cells in (("train", [-1., -1., 0., 0.], ["a", "a", "b", "b"]),
                            ("test", [-.5, -.5], ["c", "c"])):
        direct = np.zeros((len(r), 3, 1)) if constant else rng.normal(size=(len(r), 3, 1))
        arrays[f"{split}_direct"] = direct
        arrays[f"{split}_one_step"] = direct[:, 1:] + .1
        arrays[f"{split}_rollout"] = direct.copy()
        arrays[f"{split}_rollout"][:, 1:] += .2
        arrays[f"{split}_labels"] = np.column_stack((r, np.ones(len(r)), np.zeros(len(r))))
        arrays[f"{split}_cell_ids"] = np.array(cells)
    g = np.linspace(-1, 1, 101)[:, None]
    arrays.update(flow_g=g, flow_R=.7 * g + .03, flow_displacement=-.3 * g + .03)
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    metadata = {"manifest.json": {"labels_used_in_training": False},
                "summary.json": {"actual_steps": 5, "lattice_sizes": [36, 12, 4], "thermal_nu": None},
                "status.json": {"status": "complete"},
                "history_progress.json": {"history": [{"step": step, "q_reconstruction": 1 / step,
                    "q_subset_micro": .2 / step, "q_subset_coarse": .1 / step, "c_information": .3,
                    "c_variance": .1, "c_uniform": .2, "r_one_step": .1 / step, "r_rollout": .2 / step}
                    for step in (1, 5)]}}
    for name, payload in metadata.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return arrays


def _rewrite_arrays(root, arrays):
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)


def _strict_fixture(root):
    arrays = _fixture(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(schema_version=5, recipe=report.RECIPE, options={"steps": 5},
                    lattice_sizes=[36, 12, 4], theory_dim=1)
    summary = json.loads((root / "summary.json").read_text())
    summary.update(status="complete", recipe=report.RECIPE, latest_checkpoint_step=5, theory_dim=1,
        optimizer_steps={name: {"min": 5, "max": 5, "parameters": 3} for name in ("Q", "C", "R")},
        numerical_fixed_points=[dict(g=.1, residual=1e-13, R_derivative=.7, classification="attracting",
            physical_critical_point_certified=False, thermal_nu=None)],
        evaluation_direct_semantics="synthetic matching half-ensemble mean")
    history = json.loads((root / "history_progress.json").read_text())
    aliases = dict(q_reconstruction="reconstruction", q_subset_micro="subset_micro", q_subset_coarse="subset_coarse",
        c_information="information_proxy", c_variance="variance_floor", c_uniform="uniform_amplitude",
        r_one_step="closure_one_step", r_rollout="closure_rollout")
    history["history"] = [{aliases.get(key, key): value for key, value in row.items()} for row in history["history"]]
    history.update(latest_step=5, latest_checkpoint_step=5)
    for name, payload in (("manifest.json", manifest), ("summary.json", summary), ("history_progress.json", history)):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    status = dict(status="complete", workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        actual_steps=5, latest_checkpoint_step=5, recipe=report.RECIPE,
        summary_sha256=report.sha256(root / "summary.json"),
        evaluation_arrays_sha256=report.sha256(root / "evaluation_arrays.npz"))
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return arrays


def _refresh_receipt(root):
    status = json.loads((root / "status.json").read_text())
    status.update(summary_sha256=report.sha256(root / "summary.json"),
                  evaluation_arrays_sha256=report.sha256(root / "evaluation_arrays.npz"))
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")


def test_public_loader_and_fixed_train_normalization(tmp_path):
    root = tmp_path / "run"
    arrays = _fixture(root)
    loaded = report.load_public_run(root)
    assert loaded.arrays["flow_g"].shape == (101,)
    metrics = report.closure_metrics(loaded.arrays)
    assert len(metrics) == 8
    for row in metrics:
        assert row["absolute_rmse"] == pytest.approx(.1 if row["branch"] == "one_step" else .2)
        expected = np.std(arrays["train_direct"][:, row["target_scale"], :], ddof=0)
        assert row["train_target_population_std"] == pytest.approx(expected)
        assert row["fixed_train_nrmse"] == pytest.approx(row["absolute_rmse"] / expected)
    assert all(row["source_scale"] == 0 for row in metrics if row["branch"] == "microscopic_rollout")


def test_zero_train_variance_is_null_not_epsilon_or_infinity(tmp_path):
    root = tmp_path / "run"
    _fixture(root, constant=True)
    rows = report.closure_metrics(report.load_public_run(root).arrays)
    assert all(row["fixed_train_nrmse"] is None for row in rows)


@pytest.mark.parametrize("mutation, message", [
    (lambda a: a["test_cell_ids"].__setitem__(0, "a"), "overlap"),
    (lambda a: a["test_labels"].__setitem__(slice(None), [-1., 1., 0.]), "conditions overlap"),
    (lambda a: a["flow_displacement"].__iadd__(.01), "R\\(g\\)-g"),
    (lambda a: a["flow_g"].__setitem__(1, a["flow_g"][0]), "strictly increasing"),
    (lambda a: a["train_rollout"].__setitem__((0, 0, 0), 999), "begin at"),
    (lambda a: a["train_direct"].__setitem__((0, 0, 0), np.nan), "finite numeric"),
])
def test_inconsistent_public_arrays_rejected(tmp_path, mutation, message):
    root = tmp_path / "run"
    arrays = _fixture(root)
    mutation(arrays)
    _rewrite_arrays(root, arrays)
    with pytest.raises(ValueError, match=message):
        report.load_public_run(root)


def test_label_cell_identity_checked(tmp_path):
    root = tmp_path / "run"
    arrays = _fixture(root)
    arrays["train_labels"][1, 0] = .7
    _rewrite_arrays(root, arrays)
    with pytest.raises(ValueError, match="different physical parameters"):
        report.load_public_run(root)


def test_thermal_exponent_and_supervision_rejected(tmp_path):
    root = tmp_path / "run"
    _fixture(root)
    path = root / "summary.json"
    payload = json.loads(path.read_text())
    payload["thermal_nu"] = 1.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="thermal_nu"):
        report.load_public_run(root)
    payload["thermal_nu"] = None
    payload["labels_used_in_training"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupervised"):
        report.load_public_run(root)


def test_all_grid_sign_changes_and_exact_zero_preserved_no_certification():
    zeros = report.interpolated_flow_zeros(np.arange(7.), np.array([1., -1., 0., 1., -2., -1., 1.]))
    assert [row["g"] for row in zeros] == pytest.approx([.5, 2., 3 + 1 / 3, 5.5])
    assert all(row["thermal_nu"] is None and not row["full_root_certified"] for row in zeros)


def test_history_order_checked_and_missing_metrics_not_fabricated():
    assert report.history_rows({"history": [{"step": 1}, {"step": 4}]}) == [{"step": 1}, {"step": 4}]
    with pytest.raises(ValueError, match="strictly increasing"):
        report.history_rows({"history": [{"step": 2}, {"step": 2}]})


def test_report_is_new_source_unchanged_model_free_and_public(tmp_path):
    root = tmp_path / "run"
    _fixture(root)
    # Deliberately unreadable fake checkpoint: the report has no reason to open it.
    checkpoint = root / "training_step_5.pt"
    checkpoint.write_bytes(b"not a model")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    output = tmp_path / "report"
    provenance, metrics = report.write_report(root, output)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()} == before
    assert provenance["public_sources_unchanged"] is True
    assert provenance["model_loaded"] is False
    assert provenance["GPU_used"] is False
    assert metrics["thermal_nu"] is metrics["critical_r"] is None
    assert set(provenance["public_files"]) == {"R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png",
                                               "report.json", "metrics.json", "status.json"}
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(output / name) == digest
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        report.write_report(root, output)


def test_final_report_refuses_running_status(tmp_path):
    root = tmp_path / "run"
    _fixture(root)
    (root / "status.json").write_text('{"status": "running"}')
    with pytest.raises(ValueError, match="completed"):
        report.load_public_run(root)


def test_real_interface_raw_losses_and_numerical_roots_are_preserved(tmp_path):
    root = tmp_path / "run"
    _strict_fixture(root)
    loaded = report.load_public_run(root)
    assert "reconstruction" in loaded.history["history"][0]
    provenance, metrics = report.write_report(root, tmp_path / "report")
    assert metrics["saved_numerical_roots"] == loaded.summary["numerical_fixed_points"]
    assert metrics["evaluation_direct_semantics"] == "synthetic matching half-ensemble mean"
    assert provenance["public_sources_unchanged"]


@pytest.mark.parametrize("field,value,message", [("workflow_complete", False, "successfully complete"),
    ("returncode", 1, "successfully complete"), ("actual_steps", 4, "endpoint disagree"),
    ("recipe", "wrong", "recipe identities disagree"),
    ("summary_sha256", "wrong", "summary_sha256 disagrees")])
def test_real_completed_receipt_must_match(tmp_path, field, value, message):
    root = tmp_path / "run"
    _strict_fixture(root)
    payload = json.loads((root / "status.json").read_text())
    payload[field] = value
    (root / "status.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        report.load_public_run(root)


def test_real_optimizer_parameter_counts_and_artifact_fingerprints_checked(tmp_path):
    root = tmp_path / "run"
    _strict_fixture(root)
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary["optimizer_steps"]["C"]["min"] = 4
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary_sha256"):
        report.load_public_run(root)
    _refresh_receipt(root)
    with pytest.raises(ValueError, match="C Adam parameter counters"):
        report.load_public_run(root)


def test_optional_pooled_feature_encoding_is_separate_from_closure(tmp_path):
    root = tmp_path / "run"
    arrays = _fixture(root)
    for split in ("train", "test"):
        arrays[f"{split}_direct_full"] = arrays[f"{split}_direct"] + 100
    _rewrite_arrays(root, arrays)
    loaded = report.load_public_run(root)
    assert report.closure_metrics(loaded.arrays)[0]["absolute_rmse"] == pytest.approx(.1)


def test_scalar_r_figure_refuses_mixing_other_couplings(tmp_path):
    root = tmp_path / "run"
    arrays = _fixture(root)
    arrays["test_labels"][:, 1] = 2
    _rewrite_arrays(root, arrays)
    with pytest.raises(ValueError, match="fixed u and h"):
        report.load_public_run(root)
