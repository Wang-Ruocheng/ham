"""Synthetic public JSON/array tests; no HMC, physical files, model, or GPU."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_u02_qcr as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    r_values = np.linspace(-.5, .3, 9).tolist()
    shards = []
    for cell, r in enumerate(r_values):
        split = "test" if cell % 3 == 1 else "train"
        directory = data / "conditions" / f"L36_c{cell:04d}"
        chains = [dict(chain_index=chain, initial="hot" if chain < 2 else "cold_positive" if chain == 2 else "cold_negative",
            production_acceptance=.91, burn_in_acceptance=.9, saved_magnetization_sign_changes=0,
            diagnostics=dict(action_density=dict(samples=128, mean=1., std=.5,
                lag1=.8, ess=16., iat_saved_intervals=4.)),
            warnings=["ESS below 32 saved draws", "No saved magnetization sign change"],
            equilibrium_certified=False) for chain in range(4)]
        sidecar = dict(schema=report.DATA_SCHEMA, status="complete", cell_index=cell,
            size=36, dimension=2, r=r, u=.2, h=0., split=split, chains=chains,
            independent_chains=4, samples_per_chain=128, data_sha256="synthetic-not-opened",
            summary=dict(hot_cold_comparison=dict(action_density=dict(hot_minus_cold=.4,
                between_chain_standard_error=.1, descriptive_three_se_warning=True))),
            labels_used_in_training=False, equilibrium_certified=False)
        _write(directory / "sidecar.json", sidecar)
        (directory / "data.npz").write_bytes(b"forbidden physical fixture")
        shards.append(dict(sidecar=str((directory / "sidecar.json").relative_to(data)),
            sidecar_sha256=report.sha256(directory / "sidecar.json"),
            cell_index=cell, size=36, split=split, sha256="synthetic-not-opened",
            path=str((directory / "data.npz").relative_to(data))))
    manifest = dict(schema=report.DATA_SCHEMA, status="complete", options=dict(r_values=r_values,
        u_values=[.2], h=0., dimension=2, sizes=[36], chains=4, samples=128,
        test_every=3, test_offset=1, seed=91), shards=shards, warning_count=81,
        labels_used_in_training=False, equilibrium_certified=False)
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)
    arrays = {}
    for split in ("train", "test"):
        selected = [row for row in shards if row["split"] == split]
        r = np.repeat([r_values[row["cell_index"]] for row in selected], 4)
        ids = np.repeat([row["cell_index"] for row in selected], 4)
        direct = r[:, None, None] + np.arange(3)[None, :, None] * .3
        direct += np.tile([-.01, 0., .01, .02], len(selected))[:, None, None]
        target = np.concatenate([direct * .03 + .2, -.02 * direct + .3], axis=-1)
        arrays.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, 1:] + .01,
            f"{split}_rollout": direct.copy(), f"{split}_labels": np.column_stack((r, r * 0 + .2, r * 0)),
            f"{split}_cell_ids": ids, f"{split}_feature_target": target,
            f"{split}_feature_reconstruction": target + .0001,
            f"{split}_one_step_feature_prediction": target[:, 1:] + .0002,
            f"{split}_rollout_feature_prediction": target.copy()})
        arrays[f"{split}_rollout"][:, 1:] += .02
        arrays[f"{split}_rollout_feature_prediction"][:, 1:] += .0003
    g = np.linspace(-1, 2, 101)
    arrays.update(flow_g=g, flow_R=g + .3, flow_displacement=g * 0 + .3)
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)
    digest = report.sha256(data / "manifest.json")
    _write(run / "manifest.json", dict(labels_used_in_training=False, data_manifest_sha256=digest))
    _write(run / "summary.json", dict(actual_steps=3000, lattice_sizes=[36, 12, 4],
                                     thermal_nu=None, data_manifest_sha256=digest))
    _write(run / "status.json", dict(status="complete"))
    _write(run / "history_progress.json", dict(history=[dict(step=3000)]))
    (run / "training_step_3000.pt").write_bytes(b"not a checkpoint")
    return data, run, arrays


def _refresh_data_status(data):
    _write(data / "status.json", dict(schema=report.DATA_SCHEMA, status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False, completed_conditions=9,
        manifest_sha256=report.sha256(data / "manifest.json"), warning_count=81,
        equilibrium_certified=False))


def _change_manifest(data, key, value):
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["options"][key] = value
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)


def _hash_tree(path):
    return {str(file.relative_to(path)): report.sha256(file) for file in path.rglob("*") if file.is_file()}


def test_hmc_warning_evidence_preserved_not_failed(tmp_path):
    data, _, _ = _fixture(tmp_path)
    _, manifest, status, sidecars, fingerprints = report.load_public_data(data)
    assert status["status"] == "complete"
    assert len(fingerprints) == 11
    summary = report.sampling_summary(manifest, sidecars)
    assert not summary["equilibrium_certified"]
    assert summary["preserved_warning_count"] == summary["declared_warning_count"] == 81
    assert summary["warning_count_matches_manifest"]
    assert summary["production_acceptance_range"] == [.91, .91]
    assert summary["ess_range"] == [16., 16.]
    assert summary["lag1_range"] == [.8, .8]
    assert summary["iat_saved_intervals_range"] == [4., 4.]
    assert summary["chains_without_saved_magnetization_sign_change"] == summary["total_chains"] == 36
    assert len(summary["hot_cold_comparison"]) == 9


def test_metrics_correct_and_chinese_report_is_read_only(tmp_path, monkeypatch):
    data, run, _ = _fixture(tmp_path)
    before_data, before_run = _hash_tree(data), _hash_tree(run)
    original_load = np.load
    loads = []
    def guarded_load(path, *args, **kwargs):
        assert str(path).endswith("run/evaluation_arrays.npz")
        loads.append(str(path))
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    provenance, calibration = report.write_report(data, run, tmp_path / "calibration")
    assert len(loads) == 1
    assert _hash_tree(data) == before_data
    assert _hash_tree(run) == before_run
    assert provenance["public_sources_unchanged"]
    assert not any(provenance[key] for key in ("model_loaded", "physical_data_loaded", "GPU_used",
        "inference_performed", "root_search_performed", "equilibrium_certified"))
    assert calibration["thermal_nu"] is calibration["critical_r"] is None
    assert calibration["sampling_diagnostics"]["preserved_warning_count"] == 81
    assert len(calibration["sampling_sidecars"]) == 9
    assert calibration["sample_semantics"] == report.SAMPLE_SEMANTICS
    for row in calibration["q_reconstruction"]["rows"]:
        assert row["reconstruction"]["absolute_rmse"] == pytest.approx(.0001)
        assert row["better_than_constant"]
    for row in calibration["latent_closure"]["rows"]:
        assert row["model"]["absolute_rmse"] == pytest.approx(.01 if row["branch"] == "one_step" else .02)
        assert row["better_than_identity"]
    assert set(provenance["public_files"]) == {"calibration.json", "diagnostics.png", "report.json", "status.json"}
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "calibration" / name) == digest
    assert "自由高斯" not in "\n".join(calibration["limitations"])
    assert "u=0.2" in calibration["limitations"][0]
    with pytest.raises(FileExistsError):
        report.write_report(data, run, tmp_path / "calibration")


@pytest.mark.parametrize("key,value", [("u_values", [1.]), ("h", .1), ("sizes", [108]), ("dimension", 3)])
def test_wrong_physical_identity_rejected(tmp_path, key, value):
    data, _, _ = _fixture(tmp_path)
    _change_manifest(data, key, value)
    with pytest.raises(ValueError, match="fixed u=0.2"):
        report.load_public_data(data)


@pytest.mark.parametrize("values", [[-.4, .3], [-.5, .2], [-.5, .3, .3], [-.5, .3, .1]])
def test_wrong_r_range_or_duplicate_conditions_rejected(tmp_path, values):
    data, _, _ = _fixture(tmp_path)
    _change_manifest(data, "r_values", values)
    with pytest.raises(ValueError, match="distinct increasing"):
        report.load_public_data(data)


@pytest.mark.parametrize("source", ["manifest.json", "summary.json"])
def test_training_data_link_must_match_both_sources(tmp_path, source):
    data, run, _ = _fixture(tmp_path)
    value = json.loads((run / source).read_text())
    value["data_manifest_sha256"] = "wrong"
    _write(run / source, value)
    with pytest.raises(ValueError, match="data manifest SHA256"):
        report.write_report(data, run, tmp_path / "calibration")


def test_same_grid_wrong_seed_data_cannot_be_misattached(tmp_path):
    data, run, _ = _fixture(tmp_path)
    _change_manifest(data, "seed", 123456)
    with pytest.raises(ValueError, match="data manifest SHA256"):
        report.write_report(data, run, tmp_path / "calibration")


def test_missing_parameter_cell_rejected(tmp_path):
    data, _, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["shards"].pop()
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)
    with pytest.raises(ValueError, match="Incomplete HMC parameter"):
        report.load_public_data(data)


def test_sidecar_fingerprint_and_split_identity_enforced(tmp_path):
    data, _, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    path = data / manifest["shards"][0]["sidecar"]
    value = json.loads(path.read_text())
    value["split"] = "test"
    _write(path, value)
    with pytest.raises(ValueError, match="sidecar SHA256"):
        report.load_public_data(data)
    manifest["shards"][0]["sidecar_sha256"] = report.sha256(path)
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)
    with pytest.raises(ValueError, match="whole-cell split"):
        report.load_public_data(data)


def test_data_execution_must_reach_complete(tmp_path):
    data, _, _ = _fixture(tmp_path)
    value = json.loads((data / "status.json").read_text())
    value["workflow_complete"] = False
    _write(data / "status.json", value)
    with pytest.raises(ValueError, match="successfully complete"):
        report.load_public_data(data)


def test_run_must_include_every_chain_in_each_cell(tmp_path):
    data, run, arrays = _fixture(tmp_path)
    for key in list(arrays):
        if key.startswith("test_"):
            arrays[key] = arrays[key][1:]
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError, match="omits or duplicates"):
        report.write_report(data, run, tmp_path / "calibration")


def test_parameter_cell_identity_is_not_replaced_by_per_chain_ids(tmp_path):
    data, run, arrays = _fixture(tmp_path)
    arrays["train_cell_ids"][:4] = 99
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError, match="Run cell IDs disagree"):
        report.write_report(data, run, tmp_path / "calibration")


def test_missing_feature_predictions_are_not_fabricated(tmp_path):
    data, run, arrays = _fixture(tmp_path)
    for key in list(arrays):
        if "feature" in key:
            del arrays[key]
    np.savez_compressed(run / "evaluation_arrays.npz", **arrays)
    _, calibration = report.write_report(data, run, tmp_path / "calibration")
    assert not calibration["q_reconstruction"]["available"]
    assert not calibration["native_statistic_prediction"]["available"]
    assert calibration["latent_closure"]["available"]


def test_warning_count_disagreement_does_not_fabricate_or_fail(tmp_path):
    data, _, _ = _fixture(tmp_path)
    _, manifest, _, sidecars, _ = report.load_public_data(data)
    manifest["warning_count"] = 123456
    result = report.sampling_summary(manifest, sidecars)
    assert not result["warning_count_matches_manifest"]
    assert result["declared_warning_count"] == 123456
    assert result["preserved_warning_count"] == 81


def test_output_cannot_replace_sources(tmp_path):
    data, run, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(data, run, tmp_path)

