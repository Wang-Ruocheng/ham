"""Synthetic public-only hybrid report contracts; never sample or load a model."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_hybrid_qcr as report
from scripts import report_phi4_u02_qcr as base
from scripts.report_phi4_qcr import RECIPE


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    r_values = np.linspace(-.5, .3, 12).tolist()
    pins = {name: "a" * 64 for name in report.PINNED_SOURCES}
    totals = {name: 0 for name in report.WORK_FIELDS}
    shards = []
    for cell, r in enumerate(r_values):
        split = "test" if cell % 3 == 1 else "train"
        work = dict(hmc_trajectories=2048, hybrid_cycles=2048, force_evaluations=2048 * 13,
            force_site_evaluations=2048 * 13 * 36**2, wolff_cluster_updates=1024,
            lazy_cluster_skips=1024, total_cluster_sites=2048, total_cluster_changed_sites=2048,
            total_cluster_neighbor_visits=8192, total_cluster_bond_tests=4096,
            total_cluster_activated_bonds=1024)
        chains = [dict(schema=report.DATA_SCHEMA, chain_index=chain,
            initial="hot" if chain < 2 else "cold_positive" if chain == 2 else "cold_negative",
            production_acceptance=.9, saved_magnetization_sign_changes=20,
            diagnostics=dict(action_density=dict(samples=128, mean=1., std=.5,
                lag1=.8, ess=16., iat_saved_intervals=4.)), warnings=["ESS below 32 saved draws"],
            cluster_probability=.5, cluster_schedule="fixed_lazy_(1-p)I+pW_after_each_HMC",
            hmc_only_control=False, production_adaptation=False, symmetry_augmentation=False,
            artificial_global_sign_flips=0, equilibrium_certified=False, **work) for chain in range(4)]
        condition_work = {name: 4 * value for name, value in work.items()}
        for name, value in condition_work.items():
            totals[name] += value
        sidecar = dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
            status="complete", cell_index=cell, size=36, dimension=2, r=r, u=.2, h=0., split=split,
            chains=chains, independent_chains=4, samples_per_chain=128,
            data_sha256="not-a-physical-archive", cluster_probability=.5, source_pins=pins,
            symmetry_augmentation=False, production_adaptation=False, artificial_global_sign_flips=0,
            labels_used_in_training=False, equilibrium_certified=False,
            summary=dict(cluster_work=condition_work, hot_cold_comparison=dict(action_density=dict(
                hot_minus_cold=.4, between_chain_standard_error=.1, descriptive_three_se_warning=True))))
        path = data / "conditions" / f"L36_c{cell:04d}" / "sidecar.json"
        _write(path, sidecar)
        (path.parent / "data.npz").write_bytes(b"forbidden physical data")
        shards.append(dict(sidecar=str(path.relative_to(data)), sidecar_sha256=report.sha256(path),
            cell_index=cell, size=36, split=split, sha256=sidecar["data_sha256"],
            path=str((path.parent / "data.npz").relative_to(data))))
    quality = dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
        status="complete", actual_work=totals, warning_count=60, equilibrium_certified=False,
        warnings_retained_without_resampling=True, quality_is_execution_gate=False)
    _write(data / "quality_summary.json", quality)
    manifest = dict(schema=report.DATA_SCHEMA, status="complete", sample_semantics=report.SAMPLE_SEMANTICS,
        options=dict(r_values=r_values, u_values=[.2], h=0., sizes=[36], dimension=2,
            chains=4, samples=128, burn_in=1024, spacing=8, step_size=.08, leapfrog_steps=12,
            seed=91, test_every=3, test_offset=1, cluster_probability=.5), cluster_probability=.5,
        shards=shards, actual_work=totals, total_hmc_trajectories=totals["hmc_trajectories"],
        total_force_site_evaluations=totals["force_site_evaluations"], source_pins=pins, source_pins_match=True,
        quality_summary="quality_summary.json", quality_summary_sha256=report.sha256(data / "quality_summary.json"),
        warning_count=60, labels_used_in_training=False, symmetry_augmentation=False,
        artificial_global_sign_flips=0, production_adaptation=False, equilibrium_certified=False)
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)
    arrays = {}
    for split in ("train", "test"):
        selected = [row for row in shards if row["split"] == split]
        r = np.repeat([r_values[row["cell_index"]] for row in selected], 4)
        direct = r[:, None, None] + np.arange(3)[None, :, None] * .3
        direct += np.tile([-.01, 0., .01, .02], len(selected))[:, None, None]
        features = np.concatenate([direct * .03 + .2, direct * -.02 + .3], axis=-1)
        arrays.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, 1:] + .01,
            f"{split}_rollout": direct.copy(), f"{split}_labels": np.column_stack((r, r * 0 + .2, r * 0)),
            f"{split}_cell_ids": np.repeat([row["cell_index"] for row in selected], 4),
            f"{split}_feature_target": features, f"{split}_feature_reconstruction": features + .0001,
            f"{split}_one_step_feature_prediction": features[:, 1:] + .0002,
            f"{split}_rollout_feature_prediction": features.copy()})
        arrays[f"{split}_rollout"][:, 1:] += .02
        arrays[f"{split}_rollout_feature_prediction"][:, 1:] += .0003
    g = np.linspace(-1, 2, 101)
    arrays.update(flow_g=g, flow_R=g + .3, flow_displacement=g * 0 + .3)
    np.savez(run / "evaluation_arrays.npz", **arrays)
    digest = report.sha256(data / "manifest.json")
    _write(run / "manifest.json", dict(schema_version=5, recipe=RECIPE,
        options=dict(seed=91, steps=3000), data_manifest_sha256=digest, labels_used_in_training=False,
        lattice_sizes=[36, 12, 4], theory_dim=1, source_sha256={"model.py": "b" * 64}))
    _write(run / "summary.json", dict(status="complete", recipe=RECIPE, actual_steps=3000,
        latest_checkpoint_step=3000, lattice_sizes=[36, 12, 4], theory_dim=1, thermal_nu=None,
        data_manifest_sha256=digest, labels_used_in_training=False,
        optimizer_steps={name: dict(min=3000, max=3000, parameters=3) for name in ("Q", "C", "R")}))
    _write(run / "history_progress.json", dict(latest_step=3000, latest_checkpoint_step=3000,
        history=[dict(step=1), dict(step=3000)]))
    _refresh_run_status(run)
    (run / "training_step_3000.pt").write_bytes(b"not a checkpoint")
    return data, run, arrays


def _refresh_data_status(data):
    manifest = json.loads((data / "manifest.json").read_text())
    _write(data / "status.json", dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
        status="complete", workflow_complete=True, returncode=0, error=None,
        requires_user_action=False, completed_conditions=12, actual_work=manifest["actual_work"],
        manifest_sha256=report.sha256(data / "manifest.json"), equilibrium_certified=False))


def _refresh_run_status(run):
    _write(run / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, recipe=RECIPE, actual_steps=3000,
        latest_checkpoint_step=3000, summary_sha256=report.sha256(run / "summary.json"),
        evaluation_arrays_sha256=report.sha256(run / "evaluation_arrays.npz")))


def _rewrite_sidecar(data, callback):
    manifest = json.loads((data / "manifest.json").read_text())
    path = data / manifest["shards"][0]["sidecar"]
    sidecar = json.loads(path.read_text())
    callback(sidecar)
    _write(path, sidecar)
    manifest["shards"][0]["sidecar_sha256"] = report.sha256(path)
    _write(data / "manifest.json", manifest)
    _refresh_data_status(data)


def test_hybrid_identity_actual_work_and_warning_preservation(tmp_path):
    data, _, _ = _fixture(tmp_path)
    _, manifest, _, sidecars, _, quality, work = report.load_public_data(data)
    assert manifest["schema"] == report.DATA_SCHEMA
    assert quality["warning_count"] == 60
    assert len(sidecars) == 12 and len(work["per_chain"]) == 48
    assert work["actual_work"]["hmc_trajectories"] == 98304
    assert work["actual_work"]["wolff_cluster_updates"] == 49152
    assert work["actual_work"]["lazy_cluster_skips"] == 49152
    assert work["actual_work"]["force_site_evaluations"] == 1656225792
    assert work["artificial_global_sign_flips"] == 0
    assert not work["physical_npz_loaded_for_work_validation"]


def test_default_hmc_loader_remains_strict_without_global_monkeypatch(tmp_path):
    data, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="completed phi4 HMC"):
        base.load_public_data(data)
    assert base.DATA_SCHEMA == "euclidean_phi4_hmc_v1"
    assert base.SAMPLE_SEMANTICS != report.SAMPLE_SEMANTICS
    with pytest.raises(ValueError, match="sample semantics"):
        base.load_public_data(data, expected_schema=report.DATA_SCHEMA)


def test_report_read_only_chinese_hybrid_and_existing_adam_endpoint(tmp_path, monkeypatch):
    data, run, _ = _fixture(tmp_path)
    before = {str(path): report.sha256(path) for parent in (data, run) for path in parent.rglob("*") if path.is_file()}
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert str(path).endswith("run/evaluation_arrays.npz")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    provenance, calibration = report.write_report(data, run, tmp_path / "calibration")
    assert {str(path): report.sha256(path) for parent in (data, run) for path in parent.rglob("*") if path.is_file()} == before
    assert calibration["thermal_nu"] is calibration["critical_r"] is None
    assert calibration["sampling_diagnostics"]["sample_semantics"] == report.SAMPLE_SEMANTICS
    assert calibration["sampling_diagnostics"]["preserved_warning_count"] == 60
    assert all(row["min"] == row["max"] == 3000 for row in calibration["optimizer_steps"].values())
    assert calibration["training_source_sha256"] == {"model.py": "b" * 64}
    assert not any(provenance[key] for key in ("model_loaded", "physical_data_loaded", "GPU_used",
        "inference_performed", "root_search_performed", "symmetry_augmentation", "equilibrium_certified"))
    assert set(provenance["public_files"]) == {"calibration.json", "diagnostics.png", "report.json", "status.json"}
    assert "R_vs_g.png" in calibration["standard_report_companion"]
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "calibration" / name) == digest
    assert any("惰性" in text for text in calibration["limitations"])
    with pytest.raises(FileExistsError):
        report.write_report(data, run, tmp_path / "calibration")


@pytest.mark.parametrize("key,value", [("cluster_probability", 1.), ("symmetry_augmentation", True),
    ("artificial_global_sign_flips", 1), ("production_adaptation", True)])
def test_forbidden_schedule_or_postprocessing_rejected(tmp_path, key, value):
    data, _, _ = _fixture(tmp_path)
    _rewrite_sidecar(data, lambda sidecar: sidecar["chains"][0].update({key: value}))
    with pytest.raises(ValueError, match="fixed lazy kernel"):
        report.load_public_data(data)


@pytest.mark.parametrize("key", ["hmc_trajectories", "lazy_cluster_skips", "total_cluster_neighbor_visits",
                                "total_cluster_activated_bonds"])
def test_actual_hmc_and_cluster_work_is_not_an_expected_fraction(tmp_path, key):
    data, _, _ = _fixture(tmp_path)
    _rewrite_sidecar(data, lambda sidecar: sidecar["chains"][0].__setitem__(key, sidecar["chains"][0][key] + 1))
    with pytest.raises(ValueError, match="Actual HMC/cluster work"):
        report.load_public_data(data)


def test_quality_summary_fingerprint_mismatch_rejected(tmp_path):
    data, _, _ = _fixture(tmp_path)
    path = data / "quality_summary.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="quality-summary SHA256"):
        report.load_public_data(data)


def test_source_pin_mismatch_rejected(tmp_path):
    data, _, _ = _fixture(tmp_path)
    _rewrite_sidecar(data, lambda sidecar: sidecar["source_pins"].__setitem__("recurrent_rg/data/phi4.py", "wrong"))
    with pytest.raises(ValueError, match="schedule/source/augmentation"):
        report.load_public_data(data)


@pytest.mark.parametrize("source", ["manifest.json", "summary.json"])
def test_run_to_data_link_required_in_both_sources(tmp_path, source):
    data, run, _ = _fixture(tmp_path)
    path = run / source
    value = json.loads(path.read_text())
    value["data_manifest_sha256"] = "wrong"
    _write(path, value)
    _refresh_run_status(run)
    with pytest.raises(ValueError, match="data manifest SHA256"):
        report.write_report(data, run, tmp_path / "calibration")


def test_adam_parameter_clock_and_seed_must_reach_formal_endpoint(tmp_path):
    data, run, _ = _fixture(tmp_path)
    path = run / "summary.json"
    value = json.loads(path.read_text())
    value["optimizer_steps"]["R"]["min"] = 2999
    _write(path, value)
    _refresh_run_status(run)
    with pytest.raises(ValueError, match="Adam parameter counters"):
        report.write_report(data, run, tmp_path / "calibration")


def test_output_must_not_replace_source(tmp_path):
    data, run, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(data, run, tmp_path)


def test_default_and_custom_suptitles_survive_panel_title_loop(tmp_path, monkeypatch):
    import matplotlib.figure
    titles = []
    original = matplotlib.figure.Figure.suptitle
    def capture(self, text, *args, **kwargs):
        titles.append(text)
        return original(self, text, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", capture)
    payload = dict(q_reconstruction=dict(available=False), latent_closure=dict(available=False),
        native_statistic_prediction=dict(available=False), r_information=dict(rows=[
            dict(split=split, scale=scale, absolute_spearman=.5)
            for split in ("train", "test") for scale in range(3)]))
    plt = report._plotting(None)
    for mode in ("default", "custom"):
        directory = tmp_path / mode
        directory.mkdir()
        base.plot_diagnostics(payload, directory, plt,
            **({"title": "自定义：固定 HMC＋惰性嵌入簇"} if mode == "custom" else {}))
    assert titles[0] == "二维 φ⁴ QCR 校准：固定 u=0.2、h=0\n保留 HMC 混合警告；不认证热化、相变或临界指数"
    assert titles[1] == "自定义：固定 HMC＋惰性嵌入簇"

