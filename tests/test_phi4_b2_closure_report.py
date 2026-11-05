"""Synthetic public-artifact checks, without models or physical field I/O."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_b2_closure_comparison as report


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(root):
    status = json.loads((root / "status.json").read_text())
    status.update({key: report.sha256(root / filename) for key, filename in (
        ("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
        ("history_progress_sha256", "history_progress.json"),
        ("evaluation_arrays_sha256", "evaluation_arrays.npz"))})
    _write(root / "status.json", status)


def _fixture(root):
    root.mkdir()
    arrays, endpoints = {}, {}
    rng = np.random.default_rng(914)
    for split, r, ids in (("train", [-.5, -.5, .3, .3], [0, 0, 1, 1]),
                          ("test", [-.2, -.2, .1, .1], [2, 2, 3, 3])):
        direct = rng.normal(size=(len(r), 4, 1))
        targets = rng.uniform(.002, .04, size=(len(r), 4, 105))
        for arm, error in (("legacy", .2), ("all_start", .1)):
            local = dict(direct=direct.copy(), direct_full=direct + .001,
                one_step=direct[:, 1:] + error,
                rollout=np.concatenate((direct[:, :1], direct[:, 1:] + 2 * error), axis=1),
                labels=np.column_stack((r, np.full(len(r), .2), np.zeros(len(r)))), cell_ids=np.asarray(ids),
                feature_target=targets.copy(), feature_reconstruction=targets + .003,
                one_step_feature_prediction=targets[:, 1:] + error * .01,
                rollout_feature_prediction=targets + error * .03)
            arrays.update({f"{arm}_{split}_{key}": value for key, value in local.items()})
            endpoints.setdefault(arm, {}).update({f"{split}_{key}": value for key, value in local.items()})
    for arm, error in (("legacy", .2), ("all_start", .1)):
        local = endpoints[arm]
        labels = np.concatenate([local[f"{split}_labels"] for split in ("train", "test")])
        direct = np.concatenate([local[f"{split}_direct"] for split in ("train", "test")])
        grid_labels = np.unique(labels, axis=0)
        means = np.array([direct[np.all(labels == label, axis=1)].mean(axis=0) for label in grid_labels])
        r_of_mean = means[:, :3] * (1 + error)
        mean_r = r_of_mean + .001
        g = np.linspace(-2, 2, 40)
        drift = -error * g
        grid = dict(flow_g=g, flow_R=g + drift, flow_displacement=drift,
            grid_labels=grid_labels, grid_mean_direct=means, grid_R_of_mean=r_of_mean,
            grid_mean_R=mean_r, grid_jensen_difference=mean_r - r_of_mean,
            grid_displacement=r_of_mean - means[:, :3])
        arrays.update({f"{arm}_{key}": value for key, value in grid.items()})
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    modules = dict(Q="1" * 64, C="2" * 64, R="3" * 64)
    manifest = dict(schema_version=5, recipe=report.RECIPE, theory_dim=1,
        lattice_sizes=list(report.SIZES), options=dict(steps=8), closure_modes=report.MODES.copy(),
        block_size=2, stride=2, Q_closure_feedback=False, warm_start=False,
        C_pretraining=False, kernel_selection=False, learning_rate=.001,
        closure_weights=dict(commutator=5, rollout_weight=5, total_all_start=10),
        closure_scale_weights=None,
        labels_used_in_training=False, paired_schedule_sha256="4" * 64, data_manifest_sha256="5" * 64)
    receipts = {arm: dict(actual_steps=8, closure_objective=report.MODES[arm],
        optimizer_steps={module: dict(min=8, max=8, parameters=3) for module in ("Q", "C", "R")},
        initial_module_sha256=modules.copy(), final_module_sha256=dict(Q="6" * 64, C="7" * 64, R=str(index + 8) * 64),
        numerical_fixed_points=[dict(g=0. if arm == "all_start" else 100., classification="attracting",
            R_derivative=.9, thermal_nu=None, physical_critical_point_certified=False)])
        for index, arm in enumerate(report.NAMES)}
    summary = dict(status="complete", recipe=report.RECIPE, arms=receipts,
        actual_steps=8, actual_steps_by_arm={arm: 8 for arm in report.NAMES},
        closure_modes=report.MODES.copy(), initial_module_sha256=modules.copy(), C_Q_pairing_verified=True,
        paired_schedule_sha256=manifest["paired_schedule_sha256"], data_manifest_sha256=manifest["data_manifest_sha256"],
        critical_r=None, thermal_nu=None, equilibrium_certified=False, data_warnings=["保留原采样质量警告"])
    history = dict(latest_step=8, history=[dict(step=step, elapsed_seconds=step,
        arms={arm: dict(q_loss=.1 / step, c_loss=-.2 / step, reconstruction=.001 / step,
            subset_micro=.0001, subset_coarse=.0002, info_proxy=.2, variance_floor=.001,
            uniform_amplitude=.001, r_loss=error / step, closure_one_step=.001 / step,
            closure_rollout=.003 / step, closure_all_start_anchor=.002 / step)
            for arm, error in (("legacy", .2), ("all_start", .1))}) for step in (1, 8)])
    _write(root / "manifest.json", manifest)
    _write(root / "summary.json", summary)
    _write(root / "history_progress.json", history)
    _write(root / "status.json", dict(recipe=report.RECIPE, status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False))
    _refresh(root)
    return root


def _mutate(root, filename, update):
    path = root / filename
    value = json.loads(path.read_text())
    update(value)
    _write(path, value)
    _refresh(root)


def _mutate_arrays(root, update):
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    update(arrays)
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    _refresh(root)


def test_four_scale_metrics_common_coordinates_and_all_horizons(tmp_path):
    public = report.load_public_comparison(_fixture(tmp_path / "run"))
    metrics = {arm: report.endpoint_metrics(public, arm) for arm in report.NAMES}
    for arm in report.NAMES:
        assert len(metrics[arm]["latent_closure"]["rows"]) == 12
        assert len(metrics[arm]["native_statistic_prediction"]["rows"]) == 12
        assert len(metrics[arm]["q_reconstruction"]["rows"]) == 8
    for row in metrics["all_start"]["latent_closure"]["rows"]:
        assert row["model"]["absolute_rmse"] == pytest.approx(.1 if row["branch"] == "one_step" else .2)
        assert row["model"]["train_target_population_std"] == row["identity_source_statistic"]["train_target_population_std"]
        assert row["model"]["train_target_population_std"] == row["same_scale_train_mean_constant"]["train_target_population_std"]
    for row in metrics["all_start"]["native_statistic_prediction"]["rows"]:
        assert row["model"]["absolute_rmse"] == pytest.approx(.001 if row["branch"] == "one_step" else .003)
    paired = report.paired_comparison(public, metrics)
    assert len(paired["rows"]) == 24
    assert paired["raw_latent_error_cross_arm_comparison_valid"] is True
    assert paired["all_direct_coordinates_exactly_equal"] is True
    for row in paired["rows"]:
        assert row["relative_rmse_change"] == pytest.approx(-.5)
        assert row["mean_all_start_minus_legacy_mse"] < 0
    assert {row["target_scale"] for row in paired["rows"]} == {1, 2, 3}


def test_all_saved_roots_retained_with_support_not_new_root_search(tmp_path):
    public = report.load_public_comparison(_fixture(tmp_path / "run"))
    old = report.endpoint_metrics(public, "legacy")["saved_numerical_roots_with_support"]
    new = report.endpoint_metrics(public, "all_start")["saved_numerical_roots_with_support"]
    assert old[0]["saved_root"]["g"] == 100.
    assert old[0]["within_training_input_range"] is False
    assert new[0]["within_training_input_range"] is True
    assert len(new[0]["support_by_scale"]) == 8
    assert new[0]["critical_r"] is new[0]["thermal_nu"] is None
    assert new[0]["physical_critical_point_certified"] is False


def test_report_public_only_chinese_plots_and_source_immutability(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "run")
    (root / "training_step_8.pt").write_bytes(b"must never be loaded")
    (root / "physical_fields.npz").write_bytes(b"must never be opened")
    before = {path.name: report.sha256(path) for path in root.iterdir()}
    plt = report._plotting(None)
    saved, close = [], plt.close
    def capture(fig=None):
        if hasattr(fig, "findobj"):
            from matplotlib.text import Text
            saved.extend(text.get_text() for text in fig.findobj(Text) if text.get_text())
        close(fig)
    monkeypatch.setattr(plt, "close", capture)
    provenance, comparison = report.write_report(root, tmp_path / "report")
    assert {path.name: report.sha256(path) for path in root.iterdir()} == before
    assert provenance["model_loaded"] is provenance["physical_data_loaded"] is provenance["cache_loaded"] is False
    assert provenance["GPU_used"] is provenance["inference_performed"] is provenance["root_search_performed"] is False
    assert provenance["MI_reestimated"] is False
    assert comparison["critical_r"] is comparison["thermal_nu"] is None
    assert comparison["data_warnings"] == ["保留原采样质量警告"]
    assert comparison["latent_coordinates_identified_across_arms"] is True
    assert comparison["closure_error_weights"]["legacy"]["e13"] == 0
    assert comparison["closure_error_weights"]["all_start"]["e13"] == pytest.approx(10 / 9)
    assert set(provenance["public_files"]) == {"R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png",
        "comparison.json", "report.json", "status.json"}
    assert any("生成参数 r" in value for value in saved)
    assert any("位移 R(g)−g" == value for value in saved)
    assert any("L=4" in value for value in saved)
    assert any("C 粗粒化目标" in value for value in saved)
    for filename, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "report" / filename) == digest
    with pytest.raises(FileExistsError):
        report.write_report(root, tmp_path / "report")


@pytest.mark.parametrize("key", ["Q", "C"])
def test_different_final_Q_C_rejected(tmp_path, key):
    root = _fixture(tmp_path / "run")
    _mutate(root, "summary.json", lambda obj: obj["arms"]["all_start"]["final_module_sha256"].__setitem__(key, "a" * 64))
    with pytest.raises(ValueError, match="Final Q/C"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("suffix", ["direct", "direct_full", "feature_target", "feature_reconstruction", "labels", "cell_ids"])
def test_different_paired_public_Q_C_arrays_rejected(tmp_path, suffix):
    root = _fixture(tmp_path / "run")
    def mutate(arrays):
        key = f"all_start_test_{suffix}"
        if suffix == "cell_ids":
            arrays[key] += 100
        elif suffix == "labels":
            arrays[key][:, 0] += .01
        else:
            arrays[key] += .001
    _mutate_arrays(root, mutate)
    with pytest.raises(ValueError):
        report.load_public_comparison(root)


@pytest.mark.parametrize("module", ["Q", "C", "R"])
def test_wrong_adam_counters_rejected(tmp_path, module):
    root = _fixture(tmp_path / "run")
    _mutate(root, "summary.json", lambda obj: obj["arms"]["legacy"]["optimizer_steps"][module].__setitem__("max", 7))
    with pytest.raises(ValueError, match="Adam counters"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("filename,key", [("summary.json", "actual_steps"), ("manifest.json", "recipe"),
    ("summary.json", "C_Q_pairing_verified"), ("summary.json", "paired_schedule_sha256"),
    ("summary.json", "data_manifest_sha256"), ("manifest.json", "labels_used_in_training"),
    ("summary.json", "thermal_nu"), ("summary.json", "critical_r")])
def test_bad_execution_or_scientific_receipts_rejected(tmp_path, filename, key):
    root = _fixture(tmp_path / "run")
    value = True if key == "labels_used_in_training" else .2 if key in ("critical_r", "thermal_nu") else "wrong"
    _mutate(root, filename, lambda obj: obj.__setitem__(key, value))
    with pytest.raises(ValueError):
        report.load_public_comparison(root)


def test_modes_cannot_silently_become_equal(tmp_path):
    root = _fixture(tmp_path / "run")
    for filename in ("manifest.json", "summary.json"):
        _mutate(root, filename, lambda obj: obj["closure_modes"].__setitem__("legacy", "all_start_anchor"))
    with pytest.raises(ValueError, match="different legacy/all-start"):
        report.load_public_comparison(root)


def test_initial_modules_must_match(tmp_path):
    root = _fixture(tmp_path / "run")
    _mutate(root, "summary.json", lambda obj: obj["arms"]["legacy"]["initial_module_sha256"].__setitem__("R", "a" * 64))
    with pytest.raises(ValueError, match="start from identical"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("key", ["Q_closure_feedback", "warm_start", "C_pretraining", "kernel_selection"])
def test_no_feedback_or_pretraining_can_be_silently_enabled(tmp_path, key):
    root = _fixture(tmp_path / "run")
    _mutate(root, "manifest.json", lambda obj: obj.__setitem__(key, True))
    with pytest.raises(ValueError, match="no Q closure feedback"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("weights", [[1, 1, 1], None])
def test_equal_target_scale_weights_supported(tmp_path, weights):
    root = _fixture(tmp_path / "run")
    _mutate(root, "manifest.json", lambda obj: obj.__setitem__("closure_scale_weights", weights))
    report.load_public_comparison(root)


def test_nonuniform_weights_are_not_this_ablation(tmp_path):
    root = _fixture(tmp_path / "run")
    _mutate(root, "manifest.json", lambda obj: obj.__setitem__("closure_scale_weights", [1.8, .6, .6]))
    with pytest.raises(ValueError, match="equal target-scale"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("kind", ["flow", "features", "history", "summary"])
def test_nonfinite_values_rejected(tmp_path, kind):
    root = _fixture(tmp_path / "run")
    if kind in ("flow", "features"):
        key = "legacy_flow_R" if kind == "flow" else "all_start_train_feature_target"
        _mutate_arrays(root, lambda arrays: arrays[key].__setitem__((0,) * arrays[key].ndim, np.nan))
    elif kind == "history":
        _mutate(root, "history_progress.json", lambda obj: obj["history"][0]["arms"]["all_start"].__setitem__("r_loss", float("nan")))
    else:
        _mutate(root, "summary.json", lambda obj: obj.__setitem__("resource_seconds", float("nan")))
    with pytest.raises(ValueError, match="finite"):
        report.load_public_comparison(root)


def test_history_Q_C_pairing_cannot_be_lost(tmp_path):
    root = _fixture(tmp_path / "run")
    _mutate(root, "history_progress.json", lambda obj: obj["history"][0]["arms"]["legacy"].__setitem__("c_loss", .5))
    with pytest.raises(ValueError, match="Paired history c_loss"):
        report.load_public_comparison(root)


@pytest.mark.parametrize("key", ["grid_displacement", "grid_jensen_difference"])
def test_jensen_and_displacement_semantics_checked(tmp_path, key):
    root = _fixture(tmp_path / "run")
    _mutate_arrays(root, lambda arrays: arrays[f"legacy_{key}"].__setitem__((0, 0, 0), 100.))
    with pytest.raises(ValueError, match="displacement|Jensen"):
        report.load_public_comparison(root)


def test_endpoint_sha256_receipts_required(tmp_path):
    root = _fixture(tmp_path / "run")
    obj = json.loads((root / "status.json").read_text())
    obj["summary_sha256"] = "0" * 64
    _write(root / "status.json", obj)
    with pytest.raises(ValueError, match="SHA256"):
        report.load_public_comparison(root)


def test_cli_alias(tmp_path, capsys):
    root = _fixture(tmp_path / "run")
    assert report.main(["--run", str(root), "--output", str(tmp_path / "report")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
