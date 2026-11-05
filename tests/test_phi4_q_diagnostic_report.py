"""Public-statistics-only report contract for isolated continuous-field Q tests."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts import report_phi4_q_diagnostic as report


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _metrics(prediction, target, std, *, e=False, factor=1):
    error = prediction - target
    l2 = np.mean(np.sum(error * error, axis=-1), axis=0)
    rmse = np.sqrt(np.mean(error * error, axis=(0, 2)))
    subset = np.array([.0002, .0001, .0001]) * factor
    full = float(10 * l2.mean() + 20 * subset[0] + 5 * subset[1:].mean())
    return dict(reconstruction_rmse_by_scale=rmse.tolist(),
        reconstruction_nrmse_by_scale=(rmse / std).tolist(),
        reconstruction_l2_by_scale=l2.tolist(), subset_l2_by_scale=subset.tolist(),
        q_full_objective=full, optimization_objective=float(10 * l2.mean()) if e else full)


def _fixture(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    rng = np.random.default_rng(91)
    arrays = dict(train_target=rng.normal(size=(8, 3, 5)), test_target=rng.normal(size=(4, 3, 5)))
    std = np.sqrt(np.mean(np.var(arrays["train_target"], axis=0, ddof=0), axis=-1))
    arrays["train_std_by_scale"] = std
    for split, cells, r in (("train", np.repeat([0, 2, 4, 6], 2), np.repeat([-.5, -.3, -.1, .1], 2)),
                            ("test", np.repeat([1, 3], 2), np.repeat([-.4, -.2], 2))):
        arrays[f"eval_{split}_cell_ids"] = cells
        arrays[f"eval_{split}_chain_ids"] = np.arange(len(cells))
        arrays[f"eval_{split}_labels"] = np.column_stack((r, np.full(len(r), .2), np.zeros(len(r))))
    arms = {}
    for index, arm in enumerate(report.ARMS):
        arms[arm] = dict(metrics={}, diagnostic=dict(decoder_alignment_abs_pc1=.7,
            decoder_explained_fraction=.6, latent_stats={}))
        for split in ("train", "test"):
            target = arrays[f"{split}_target"]
            reconstruction = .85 * target + index * .005
            arrays[f"{arm}_{split}_reconstruction"] = reconstruction
            arrays[f"{arm}_{split}_oracle_reconstruction"] = .9 * target
            arrays[f"{arm}_{split}_direct"] = target[..., :1] + index
            arrays[f"{arm}_{split}_direct_full"] = target[..., :1] + index + .001
            arms[arm]["metrics"][split] = _metrics(reconstruction, target, std, e=arm == "E")
    baselines = {}
    train_pooled = arrays["train_target"].reshape(-1, 5)
    center = train_pooled.mean(axis=0)
    _, _, right = np.linalg.svd(train_pooled - center, full_matrices=False)
    for name, rank in (("constant", 0), ("PCA1", 1), ("PCA3", 3)):
        baselines[name] = dict(metrics={}, reconstruction_only=True)
        basis = right[:rank]
        for split in ("train", "test"):
            target = arrays[f"{split}_target"]
            reconstruction = center + ((target - center) @ basis.T) @ basis
            arrays[f"{name}_{split}_reconstruction"] = reconstruction
    summary = dict(schema_version=1, status="complete", actual_steps=3000,
        labels_used_in_training=False, source_unchanged=True, arms=arms,
        baselines=baselines, source_quality=dict(equilibrium_certified=False,
            chains_with_any_observable_ess_below32=11, independent_chain_count=48))
    _write(training / "summary.json", summary)
    rows = [dict(step=step, arms={arm: dict(train=arms[arm]["metrics"]["train"],
        test=arms[arm]["metrics"]["test"]) for arm in report.ARMS[1:]}) for step in (0, 100, 3000)]
    _write(training / "history_progress.json", dict(history=rows))
    np.savez_compressed(training / "evaluation_arrays.npz", **arrays)
    _refresh_status(training)
    # Files present beside public artifacts may never be opened by the report.
    (training / "checkpoint.pt").write_bytes(b"FORBIDDEN MODEL")
    (training / "configurations.npz").write_bytes(b"FORBIDDEN PHYSICAL DATA")
    return training


def _refresh_status(training):
    _write(training / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False,
        **{field: report.sha256(training / filename) for field, filename in (
            ("summary_sha256", "summary.json"), ("history_progress_sha256", "history_progress.json"),
            ("evaluation_arrays_sha256", "evaluation_arrays.npz"))}))


def _mutate_summary(training, change):
    data = json.loads((training / "summary.json").read_text())
    change(data)
    _write(training / "summary.json", data)
    _refresh_status(training)


def test_public_arrays_reconstruction_and_fixed_train_normalization(tmp_path):
    training = _fixture(tmp_path)
    evidence = report.load_diagnostic(training)
    expected = np.sqrt(np.mean(np.var(evidence["arrays"]["train_target"], axis=0), axis=-1))
    assert np.allclose(evidence["train_std_by_scale"], expected)
    for arm in report.ARMS:
        actual = evidence["metrics"][arm]["test"]
        assert np.allclose(actual["reconstruction_nrmse_by_scale"],
                           np.array(actual["reconstruction_rmse_by_scale"]) / expected)
        assert "decoder_oracle" in actual
        variance = actual["latent_variance_decomposition"]
        np.testing.assert_allclose(variance["total_variance_by_scale"],
            np.array(variance["between_condition_variance_by_scale"]) + variance["within_condition_variance_by_scale"])
        np.testing.assert_allclose(actual["pooled_vs_mean_half_Q_rms_by_scale"], .001, atol=1e-14)
    e = evidence["metrics"]["E"]["test"]
    assert e["optimization_objective"] < e["q_full_objective"]


def test_report_no_models_sampling_or_new_flow_and_hashes(tmp_path, monkeypatch):
    training = _fixture(tmp_path)
    before = {name: report.sha256(training / name) for name in report.INPUT_FILES}
    loaded = []
    original_load = np.load

    def public_load(path, *args, **kwargs):
        loaded.append(str(path))
        assert str(path).endswith("evaluation_arrays.npz")
        assert kwargs.get("allow_pickle") is False
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", public_load)
    monkeypatch.setitem(sys.modules, "torch", None)
    output = tmp_path / "report"
    receipt, metrics = report.write_report(training, output)
    assert len(loaded) == 1
    assert {name: report.sha256(training / name) for name in report.INPUT_FILES} == before
    assert receipt["sources_sha256_before"] == receipt["sources_sha256_after"] == before
    for key in ("model_loaded", "checkpoint_loaded", "physical_data_loaded", "GPU_used",
                "inference_performed", "physical_sampling_performed", "MI_computed",
                "root_search_performed", "new_R_flow_available"):
        assert receipt[key] is False
    assert metrics["thermal_nu"] is None
    assert metrics["source_quality"]["chains_with_any_observable_ess_below32"] == 11
    assert all("R_vs_g" not in name for name in receipt["public_files"])
    assert len([name for name in receipt["public_files"] if name.endswith(".png")]) == 4
    for name, digest in receipt["outputs_sha256"].items():
        assert report.sha256(output / name) == digest
    assert "report.json" not in receipt["outputs_sha256"]
    assert set(receipt["public_files"]) == {*receipt["outputs_sha256"], "report.json", "status.json"}
    with pytest.raises(FileExistsError):
        report.write_report(training, output)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(training, training)


def test_reject_incomplete_or_changed_public_sources(tmp_path):
    training = _fixture(tmp_path)
    status = json.loads((training / "status.json").read_text())
    status["returncode"] = 1
    _write(training / "status.json", status)
    with pytest.raises(ValueError, match="completed endpoint"):
        report.load_diagnostic(training)
    _refresh_status(training)
    value = json.loads((training / "summary.json").read_text())
    value["source_unchanged"] = False
    _write(training / "summary.json", value)
    with pytest.raises(ValueError, match="mismatch"):
        report.load_diagnostic(training)
    _refresh_status(training)
    with pytest.raises(ValueError, match="Frozen source"):
        report.load_diagnostic(training)


def test_reject_supervision_missing_endpoint_and_nonmonotone_history(tmp_path):
    training = _fixture(tmp_path)
    _mutate_summary(training, lambda data: data.update(labels_used_in_training=True))
    with pytest.raises(ValueError, match="unsupervised"):
        report.load_diagnostic(training)
    _mutate_summary(training, lambda data: data.update(labels_used_in_training=False))
    history = json.loads((training / "history_progress.json").read_text())
    history["history"][-1]["step"] = 2999
    _write(training / "history_progress.json", history)
    _refresh_status(training)
    with pytest.raises(ValueError, match="fixed endpoint"):
        report.load_diagnostic(training)


def test_reject_inconsistent_normalization_reconstruction_and_e_objective(tmp_path):
    training = _fixture(tmp_path)
    _mutate_summary(training, lambda data: data["arms"]["A"]["metrics"]["test"].update(
        reconstruction_nrmse_by_scale=[0, 0, 0]))
    with pytest.raises(ValueError, match="disagrees with public arrays"):
        report.load_diagnostic(training)
    # Restore from public, then deliberately relabel E's optimization objective.
    data = json.loads((training / "summary.json").read_text())
    with np.load(training / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        data["arms"]["A"]["metrics"]["test"] = _metrics(arrays["A_test_reconstruction"],
            arrays["test_target"], arrays["train_std_by_scale"])
    e = data["arms"]["E"]["metrics"]["test"]
    e["optimization_objective"] = e["q_full_objective"]
    _write(training / "summary.json", data)
    _refresh_status(training)
    with pytest.raises(ValueError, match="distinguish E"):
        report.load_diagnostic(training)


def test_reject_split_condition_overlap(tmp_path):
    training = _fixture(tmp_path)
    with np.load(training / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["eval_test_labels"][:2] = arrays["eval_train_labels"][0]
    np.savez_compressed(training / "evaluation_arrays.npz", **arrays)
    _refresh_status(training)
    with pytest.raises(ValueError, match="conditions overlap"):
        report.load_diagnostic(training)
