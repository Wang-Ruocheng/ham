"""The continuation report consumes only immutable public endpoint evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_rg_reference_continue as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(run, steps):
    _write(run / "status.json", dict(status="complete", recipe=report.generic.RECIPE,
        workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        actual_steps=steps, latest_checkpoint_step=steps,
        summary_sha256=report.generic.sha256(run / "summary.json"),
        evaluation_arrays_sha256=report.generic.sha256(run / "evaluation_arrays.npz")))


def _arrays(continued=False):
    result = {}
    for split in ("train", "test"):
        cells = np.asarray([i for i in range(13) if (i % 3 == 0) == (split == "test")])
        r = np.repeat(report.reference.R_VALUES[cells], 4)
        n = len(r)
        direct = r[:, None, None] + .1 * np.arange(3)[None, :, None]
        direct += np.tile([-.01, 0., .01, .02], len(cells))[:, None, None]
        if continued:
            direct = direct * 1.2 + .3
        result.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, 1:] + .01,
            f"{split}_rollout": direct.copy(), f"{split}_labels": np.column_stack((r, np.ones(n), np.zeros(n))),
            f"{split}_cell_ids": np.repeat(cells, 4)})
        result[f"{split}_rollout"][:, 1:] += .02
        target = .02 + .003 * r[:, None, None] + .001 * np.arange(3)[None, :, None] + .0001 * np.arange(105)[None, None, :]
        if continued:
            target[:, 1:] *= .9
        result.update({f"{split}_feature_target": target,
            f"{split}_feature_reconstruction": target + .0002,
            f"{split}_one_step_feature_prediction": target[:, 1:] + .0003,
            f"{split}_rollout_feature_prediction": target + .0004})
    grid = np.linspace(-2., 1., 50)
    result.update(flow_g=grid, flow_R=.7 * grid, flow_displacement=-.3 * grid)
    return result


def _fixture(root):
    source, continued = root / "source", root / "continued"
    old = source / "run"
    old.mkdir(parents=True)
    continued.mkdir()
    manifest = dict(schema_version=5, recipe=report.generic.RECIPE,
        options=dict(seed=91, steps=3000, system_batch=8, sample_batch=64, information_pairs=128,
            threads=1, output=str(old), device="cuda:2"),
        data_manifest_sha256="d" * 64, labels_used_in_training=False, lattice_sizes=[36, 12, 4],
        theory_dim=1, block_size=3, stride=3, closure_objective="all_start_anchor", closure_scale_weights=None,
        q_initialization="train_mean_bias", learning_rate=.001, scale=1.3,
        optimizer="three independent fresh Adam", initial_state_sha256="a" * 64,
        training_start_state_sha256="a" * 64, training_start_module_sha256={key: "b" * 64 for key in ("Q", "C", "R")})
    summary = dict(status="complete", recipe=report.generic.RECIPE, actual_steps=3000,
        latest_checkpoint_step=3000, data_manifest_sha256="d" * 64, lattice_sizes=[36, 12, 4],
        theory_dim=1, thermal_nu=None, optimizer_steps={key: dict(min=3000, max=3000, parameters=2) for key in ("Q", "C", "R")},
        final_state_sha256="f" * 64, final_module_sha256={key: "e" * 64 for key in ("Q", "C", "R")},
        numerical_fixed_points=[dict(g=0., thermal_nu=None, physical_critical_point_certified=False)])
    history = dict(latest_step=3000, latest_checkpoint_step=3000, history=[dict(step=step,
        reconstruction=.1 / step, subset_micro=.02 / step, subset_coarse=.01 / step,
        information_proxy=.3, variance_floor=.01, uniform_amplitude=.02,
        closure_one_step=.1 / step, closure_rollout=.2 / step) for step in (1, 3000)])
    for name, value in (("manifest.json", manifest), ("summary.json", summary), ("history_progress.json", history)):
        _write(old / name, value)
    np.savez(old / "evaluation_arrays.npz", **_arrays())
    _refresh(old, 3000)
    (old / "training_step_3000.pt").write_bytes(b"never load a checkpoint")
    prov = dict(source_steps=3000, final_state_sha256=summary["final_state_sha256"],
        final_module_sha256=summary["final_module_sha256"], checkpoint_sha256="c" * 64,
        data_manifest_sha256="d" * 64, public_files_sha256={
            "run/" + name: report.generic.sha256(old / name) for name in report.generic.INPUT_FILES[:4]})
    m = copy.deepcopy(manifest)
    m["options"].update(steps=6000, output=str(continued))
    m.update(source_steps=3000, additional_steps=3000, profile_only=False, scientific_result=True,
        q_reinitialized=False, data_resampled=False, loss_changed=False, source_provenance=prov,
        initial_state_sha256=summary["final_state_sha256"], training_start_state_sha256=summary["final_state_sha256"],
        training_start_module_sha256=summary["final_module_sha256"], optimizer="resumed Adam",
        source_q_initialization=manifest["q_initialization"], source_q_initialization_receipt=None,
        q_initialization="checkpoint_restore_no_calibration",
        q_initialization_receipt=dict(mode="checkpoint_restore_no_calibration", calibration_rerun=False,
            source_steps=3000, source_checkpoint_sha256=prov["checkpoint_sha256"]),
        resumed_optimizer_state_sha256={key: "1" * 64 for key in ("Q", "C", "R")},
        resumed_rng_sha256={key: "2" * 64 for key in ("batch_rng", "field_rng")})
    s = copy.deepcopy(summary)
    s.update(actual_steps=6000, latest_checkpoint_step=6000, source_steps=3000, additional_steps=3000,
        profile_only=False, scientific_result=True, q_reinitialized=False, data_resampled=False,
        loss_changed=False, fresh_adam=False, source_provenance=prov,
        initial_state_sha256=summary["final_state_sha256"],
        optimizer_steps={key: dict(min=6000, max=6000, parameters=2) for key in ("Q", "C", "R")})
    h = copy.deepcopy(history)
    h.update(latest_step=6000, latest_checkpoint_step=6000)
    for step in (3001, 6000):
        h["history"].append(dict(h["history"][-1], step=step))
    for name, value in (("manifest.json", m), ("summary.json", s), ("history_progress.json", h)):
        _write(continued / name, value)
    np.savez(continued / "evaluation_arrays.npz", **_arrays(continued=True))
    _refresh(continued, 6000)
    (continued / "training_step_6000.pt").write_bytes(b"never load a checkpoint")
    physics = source / "report"
    physics.mkdir()
    _write(physics / "physics_summary.json", dict(actual_steps=3000, reference=report.reference.REFERENCE,
        labels_used_in_training=False, equilibrium_certified=False,
        sampling_diagnostics=dict(preserved_warning_count=13), sampling_quality_summary=dict(warning_count=13)))
    (physics / "physics_vs_r.png").write_bytes(b"original public PNG is copied, not rendered")
    np.savez(physics / "physics_arrays.npz", r=report.reference.R_VALUES)
    _write(physics / "report.json", dict(status="complete", public_sources_unchanged=True,
        public_files=list(report.COPIED_PHYSICS), outputs_sha256={name: report.generic.sha256(physics / name) for name in report.COPIED_PHYSICS}))
    return source, continued


def test_public_comparison_own_coordinates_and_baselines(tmp_path):
    source, continued = _fixture(tmp_path)
    _, old, new = report.load_comparison(source, continued)
    metric = report.endpoint_metrics(new)
    assert metric["actual_steps"] == 6000
    assert metric["saved_numerical_root_count"] == 1
    assert metric["root_search_performed"] is False
    for category in ("latent_closure", "native_statistic_prediction"):
        assert metric[category]["available"]
        assert len(metric[category]["rows"]) == 8
        for row in metric[category]["rows"]:
            assert set(("model", "identity_source_statistic", "same_scale_train_mean_constant")) <= row.keys()
            assert row["model"]["fixed_train_nrmse"] == pytest.approx(
                row["model"]["absolute_rmse"] / row["model"]["train_target_population_std"])
    a = report.endpoint_metrics(old)["latent_closure"]["rows"][0]["model"]["train_target_population_std"]
    b = metric["latent_closure"]["rows"][0]["model"]["train_target_population_std"]
    assert b == pytest.approx(1.2 * a)


def test_report_only_public_files_no_new_roots_and_chinese(tmp_path, monkeypatch):
    source, continued = _fixture(tmp_path)
    before = {str(path): path.read_bytes() for root in (source, continued) for path in root.rglob("*") if path.is_file()}
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert Path(path) in (source / "run/evaluation_arrays.npz", continued / "evaluation_arrays.npz")
        assert kwargs.get("allow_pickle") is False
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    def forbidden(*args, **kwargs):
        pytest.fail("No new root interpolation or generic report inference")
    monkeypatch.setattr(report.generic, "interpolated_flow_zeros", forbidden)
    monkeypatch.setattr(report.generic, "write_report", forbidden)
    import matplotlib.figure
    saved_text, save = [], matplotlib.figure.Figure.savefig
    def captured(figure, *args, **kwargs):
        from matplotlib.text import Text
        saved_text.extend(artist.get_text() for artist in figure.findobj(match=Text))
        return save(figure, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", captured)
    output = tmp_path / "report"
    receipt, comparison = report.write_report(source, continued, output)
    assert before == {str(path): path.read_bytes() for root in (source, continued) for path in root.rglob("*") if path.is_file()}
    assert set(receipt["public_files"]) == {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert receipt["public_files"] == report.return_files(output)
    assert all(report.generic.sha256(output / name) == digest for name, digest in receipt["outputs_sha256"].items())
    for name in report.COPIED_PHYSICS:
        assert (output / "source_physics" / name).read_bytes() == (source / "report" / name).read_bytes()
    assert comparison["microscopic_feature_targets_identical"] == dict(train=True, test=True)
    assert comparison["latent_coordinates_aligned"] is False
    assert comparison["fair_across_C_physical_prediction_ranking_certified"] is False
    assert comparison["original_sampling_diagnostics"]["preserved_warning_count"] == 13
    assert comparison["critical_r"] is comparison["thermal_nu"] is None
    assert all(receipt[name] is False for name in ("model_loaded", "physical_data_loaded", "GPU_used",
        "inference_performed", "root_search_performed", "saved_grid_zero_interpolation_performed"))
    assert "原端点／精确续训分界" in saved_text
    assert "文献无限体积蒙特卡罗参考" in saved_text
    with pytest.raises(FileExistsError, match="fresh"):
        report.write_report(source, continued, output)


@pytest.mark.parametrize("mutation,message", [
    (lambda m, s, h: m.__setitem__("loss_changed", True), "without new data/loss/init"),
    (lambda m, s, h: s.__setitem__("fresh_adam", True), "restore Adam"),
    (lambda m, s, h: m["options"].__setitem__("sample_batch", 128), "training policy changed"),
    (lambda m, s, h: m.__setitem__("learning_rate", .0001), "unchanged b3"),
    (lambda m, s, h: m.__setitem__("resumed_rng_sha256", {}), "training RNG"),
    (lambda m, s, h: m["q_initialization_receipt"].__setitem__("calibration_rerun", True), "without initialization or recalibration"),
    (lambda m, s, h: h["history"][0].__setitem__("reconstruction", 4.), "exact source prefix"),
    (lambda m, s, h: s["optimizer_steps"]["C"].__setitem__("max", 3000), "Adam parameter counters"),
])
def test_changed_continuation_contract_rejected(tmp_path, mutation, message):
    source, continued = _fixture(tmp_path)
    values = [json.loads((continued / name).read_text()) for name in ("manifest.json", "summary.json", "history_progress.json")]
    mutation(*values)
    for name, value in zip(("manifest.json", "summary.json", "history_progress.json"), values):
        _write(continued / name, value)
    _refresh(continued, 6000)
    with pytest.raises(ValueError, match=message):
        report.load_comparison(source, continued)


def test_tampered_original_physics_rejected(tmp_path):
    source, continued = _fixture(tmp_path)
    (source / "report/physics_vs_r.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="physics file SHA256"):
        report.write_report(source, continued, tmp_path / "report")


def test_output_overlap_and_microscopic_target_change_rejected(tmp_path):
    source, continued = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        report.write_report(source, continued, continued / "report")
    arrays = _arrays(continued=True)
    arrays["test_feature_target"][0, 0, 0] += .01
    np.savez(continued / "evaluation_arrays.npz", **arrays)
    _refresh(continued, 6000)
    with pytest.raises(ValueError, match="microscopic physical targets"):
        report.write_report(source, continued, tmp_path / "report")


def test_source_and_feature_symlink_escape_rejected(tmp_path):
    source, continued = _fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes((source / "report/physics_summary.json").read_bytes())
    (source / "report/physics_summary.json").unlink()
    (source / "report/physics_summary.json").symlink_to(outside)
    with pytest.raises(ValueError, match="in-root public"):
        report.write_report(source, continued, tmp_path / "report")
