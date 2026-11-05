"""Synthetic, public-artifact-only Gaussian calibration tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_gaussian_scalar_qcr as report


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    conditions = [(0, .02, "train"), (1, .1, "test"), (2, 2., "train")]
    shards = []
    for cell, r, split in conditions:
        directory = data / f"c{cell:03d}"
        diagnostics = dict(exact_covariance_by_displacement=[[2., 1.], [1., .5]],
            empirical_covariance_by_displacement=[[2.1, .9], [.8, .4]],
            exact_structure_factor=[[1., .5], [.5, .2]],
            empirical_structure_factor=[[1.1, .4], [.6, .3]],
            mean_field=dict(mean_z_score=.5), twice_action_chi_squared=dict(mean_z_score=-.7),
            point_variance=dict(exact=2., empirical=2.1), warnings=[])
        _write(directory / "diagnostics.json", diagnostics)
        sidecar = dict(status="complete", r=r, u=0., h=0., cell_index=cell, size=36, split=split,
            diagnostics="diagnostics.json", diagnostics_sha256=report.sha256(directory / "diagnostics.json"))
        _write(directory / "sidecar.json", sidecar)
        shards.append(dict(sidecar=f"c{cell:03d}/sidecar.json",
                           sidecar_sha256=report.sha256(directory / "sidecar.json"),
                           path=f"c{cell:03d}/data.npz"))
        # A forbidden fake archive must never be parsed by this report.
        (directory / "data.npz").write_bytes(b"not a physical archive")
    manifest = dict(schema=report.DATA_SCHEMA, status="complete", sample_semantics=report.SAMPLE_SEMANTICS,
        options=dict(dimension=2, r_values=[row[1] for row in conditions], u_values=[0.], h=0.,
                     sizes=[36], groups=2, chains=2, samples=128), shards=shards)
    _write(data / "manifest.json", manifest)
    arrays = {}
    for split in ("train", "test"):
        selected = [row for row in conditions if row[2] == split]
        r = np.repeat([row[1] for row in selected], 2)
        ids = np.repeat([f"c{row[0]:03d}" for row in selected], 2)
        direct = (np.log(r)[:, None, None] + np.arange(3)[None, :, None] * .3)
        direct += np.tile([-.01, .01], len(selected))[:, None, None]
        target = np.concatenate([direct * .03 + .2, direct * -.02 + .3], axis=-1)
        arrays.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, 1:] + .01,
            f"{split}_rollout": direct.copy(), f"{split}_labels": np.column_stack((r, r * 0, r * 0)),
            f"{split}_cell_ids": ids, f"{split}_feature_target": target,
            f"{split}_feature_reconstruction": target + .0001,
            f"{split}_one_step_feature_prediction": target[:, 1:] + .0002,
            f"{split}_rollout_feature_prediction": target.copy()})
        arrays[f"{split}_rollout"][:, 1:] += .02
        arrays[f"{split}_rollout_feature_prediction"][:, 1:] += .0003
    g = np.linspace(-5, 2, 101)
    arrays.update(flow_g=g, flow_R=g + .3, flow_displacement=g * 0 + .3)
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)
    data_manifest_sha = report.sha256(data / "manifest.json")
    _write(run / "manifest.json", dict(labels_used_in_training=False, data_manifest_sha256=data_manifest_sha))
    _write(run / "summary.json", dict(actual_steps=3000, lattice_sizes=[36, 12, 4], thermal_nu=None,
                                      data_manifest_sha256=data_manifest_sha))
    _write(run / "status.json", dict(status="complete"))
    _write(run / "history_progress.json", dict(history=[dict(step=3000)]))
    (run / "training_step_3000.pt").write_bytes(b"not a model")
    return data, run, arrays


def _resave(run, arrays):
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)


def _hash_tree(path):
    return {str(file.relative_to(path)): report.sha256(file) for file in path.rglob("*") if file.is_file()}


def test_reconstruction_baseline_and_fixed_train_normalization(tmp_path):
    data, run, arrays = _fixture(tmp_path)
    flags = report._optional_feature_arrays(arrays)
    q = report.reconstruction_metrics(arrays, flags)
    assert q["available"]
    assert len(q["rows"]) == 6
    for row in q["rows"]:
        scale, split = row["scale"], row["split"]
        reference = arrays["train_feature_target"][:, scale]
        truth = arrays[f"{split}_feature_target"][:, scale]
        expected = np.sqrt(np.mean((truth - reference.mean(axis=0)) ** 2))
        assert row["same_scale_train_mean_constant"]["absolute_rmse"] == pytest.approx(expected)
        assert row["reconstruction"]["absolute_rmse"] == pytest.approx(.0001)
        assert row["reconstruction"]["mean_squared_l2"] == pytest.approx(2e-8)
        assert row["better_than_constant"]
        assert row["reconstruction"]["train_target_population_std"] == pytest.approx(
            np.sqrt(np.mean(np.var(reference, axis=0, ddof=0))))


def test_closure_and_decoder_arrays_only_not_recomputed(tmp_path):
    _, _, arrays = _fixture(tmp_path)
    flags = report._optional_feature_arrays(arrays)
    for features, expected in ((False, [.01, .02]), (True, [.0002, .0003])):
        metrics = report.prediction_metrics(arrays, features=features, available=flags)
        assert len(metrics["rows"]) == 8
        for row in metrics["rows"]:
            assert row["model"]["absolute_rmse"] == pytest.approx(expected[row["branch"] != "one_step"])
            assert row["better_than_identity"]
            suffix = "feature_target" if features else "direct"
            assert row["model"]["train_target_population_std"] == pytest.approx(
                np.sqrt(np.mean(np.var(arrays[f"train_{suffix}"][:, row["target_scale"]], axis=0, ddof=0))))


def test_zero_variance_denominator_is_null_not_epsilon():
    target = np.ones((4, 3))
    errors = report._errors(target + 1, target, target)
    assert errors["absolute_rmse"] == 1.
    assert errors["train_target_population_std"] == 0.
    assert errors["fixed_train_nrmse"] is None


def test_missing_public_features_not_fabricated(tmp_path):
    _, _, arrays = _fixture(tmp_path)
    for key in list(arrays):
        if "feature" in key:
            del arrays[key]
    flags = report._optional_feature_arrays(arrays)
    assert not report.reconstruction_metrics(arrays, flags)["available"]
    assert not report.prediction_metrics(arrays, features=True, available=flags)["available"]


def test_partial_or_inconsistent_saved_feature_arrays_rejected(tmp_path):
    _, _, arrays = _fixture(tmp_path)
    original = arrays.pop("test_feature_reconstruction")
    with pytest.raises(ValueError, match="Both splits"):
        report._optional_feature_arrays(arrays)
    arrays["test_feature_reconstruction"] = original[:, :-1]
    with pytest.raises(ValueError, match="physical group, scale, feature"):
        report._optional_feature_arrays(arrays)
    arrays["test_feature_reconstruction"] = original + np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        report._optional_feature_arrays(arrays)


def test_rank_correlation_is_per_parameter_cell_and_tie_safe(tmp_path):
    _, _, arrays = _fixture(tmp_path)
    payload = report.rank_correlations(arrays)
    assert len(payload["rows"]) == 6
    for row in payload["rows"]:
        if row["split"] == "train":
            assert row["number_of_parameter_cells"] == 2
            assert row["cell_mean_r_g_spearman"] == pytest.approx(1.)
        else:
            assert row["cell_mean_r_g_spearman"] is None
    assert report._ranks([2., 1., 2.]).tolist() == [1.5, 0., 1.5]
    arrays["train_direct"][:] = 0
    assert report.rank_correlations(arrays)["rows"][0]["cell_mean_r_g_spearman"] is None


def test_read_only_report_complete_and_hashes_public_files(tmp_path):
    data, run, _ = _fixture(tmp_path)
    before_data, before_run = _hash_tree(data), _hash_tree(run)
    provenance, calibration = report.write_report(data, run, tmp_path / "calibration")
    assert _hash_tree(data) == before_data
    assert _hash_tree(run) == before_run
    assert provenance["public_sources_unchanged"]
    assert not any(provenance[key] for key in ("model_loaded", "physical_data_loaded", "GPU_used",
        "inference_performed", "root_search_performed"))
    assert calibration["theoretical_gaussian_nu_background"] == .5
    assert calibration["thermal_nu"] is calibration["critical_r"] is None
    assert len(calibration["sampling_diagnostics"]["conditions"]) == 3
    assert calibration["sampling_diagnostics"]["conditions"][0]["diagnostics"]["covariance_by_displacement"]["absolute_rmse"] > 0
    assert set(provenance["public_files"]) == {"calibration.json", "diagnostics.png", "report.json", "status.json"}
    for filename, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "calibration" / filename) == digest
    assert len(provenance["sources_sha256_before"]["data"]) == 7
    with pytest.raises(FileExistsError):
        report.write_report(data, run, tmp_path / "calibration")


@pytest.mark.parametrize("key,value,message", [
    ("schema", "hmc", "completed exact-Gaussian"),
    ("status", "running", "completed exact-Gaussian"),
    ("sample_semantics", "markov_chains", "independent IID"),
])
def test_wrong_data_identity_or_state_rejected(tmp_path, key, value, message):
    data, _, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest[key] = value
    _write(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        report.load_public_data(data)


@pytest.mark.parametrize("r_values", [[0., .1, 2.], [-.02, .1, 2.], [.02, .1, .1]])
def test_nonpositive_or_duplicate_gaussian_mass_rejected(tmp_path, r_values):
    data, _, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["options"]["r_values"] = r_values
    _write(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="distinct positive"):
        report.load_public_data(data)


def test_interacting_data_and_split_mismatch_rejected(tmp_path):
    data, run, arrays = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["options"]["u_values"] = [1.]
    _write(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="u=0"):
        report.load_public_data(data)
    manifest["options"]["u_values"] = [0.]
    _write(data / "manifest.json", manifest)
    arrays["test_labels"][:, 0] = .11
    _resave(run, arrays)
    with pytest.raises(ValueError, match="labels disagree"):
        report.write_report(data, run, tmp_path / "report")


@pytest.mark.parametrize("source", ["manifest.json", "summary.json"])
@pytest.mark.parametrize("mismatch", ["missing", "different"])
def test_training_data_manifest_link_is_required_and_exact(tmp_path, source, mismatch):
    data, run, _ = _fixture(tmp_path)
    value = json.loads((run / source).read_text())
    if mismatch == "missing":
        del value["data_manifest_sha256"]
    else:
        value["data_manifest_sha256"] = "0" * 64
    _write(run / source, value)
    with pytest.raises(ValueError, match="data manifest SHA256 disagrees"):
        report.write_report(data, run, tmp_path / "report")


def test_same_parameter_grid_different_seed_cannot_be_misattached(tmp_path):
    data, run, _ = _fixture(tmp_path)
    value = json.loads((data / "manifest.json").read_text())
    value["options"]["seed"] = 999999
    _write(data / "manifest.json", value)
    with pytest.raises(ValueError, match="data manifest SHA256 disagrees"):
        report.write_report(data, run, tmp_path / "report")


def test_sidecar_and_diagnostic_fingerprints_enforced(tmp_path):
    data, _, _ = _fixture(tmp_path)
    path = data / "c000" / "diagnostics.json"
    values = json.loads(path.read_text())
    values["mean_field"]["mean_z_score"] = 999
    _write(path, values)
    with pytest.raises(ValueError, match="diagnostics SHA256"):
        report.load_public_data(data)
    path = data / "c000" / "sidecar.json"
    values = json.loads(path.read_text())
    values["r"] = 999
    _write(path, values)
    with pytest.raises(ValueError, match="sidecar SHA256"):
        report.load_public_data(data)


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/gaussian.json", "data.npz", "model.pt"])
def test_only_in_root_public_json_can_be_opened(tmp_path, relative):
    with pytest.raises(ValueError, match="Only in-root"):
        report._safe_public_json(tmp_path, relative)


def test_report_rejects_sources_as_output(tmp_path):
    data, run, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(data, run, data)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(data, run, tmp_path)

