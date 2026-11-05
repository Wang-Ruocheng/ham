"""Synthetic public-array tests: no physical configuration, model or GPU I/O."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_scalar_closure_ablation as report


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
    arrays = {}
    rng = np.random.default_rng(612)
    for split, r, ids in (("train", [-.5, -.5, .3, .3], [0, 0, 1, 1]),
                          ("test", [-.1, -.1], [2, 2])):
        direct = rng.normal(size=(len(r), 3, 1))
        targets = rng.uniform(.002, .04, size=(len(r), 3, 105))
        for name, scale, offset, feature_error in (("source", 1, 0, .004), ("A", 2, 7, .003), ("B", 3, -5, .001)):
            values = direct * scale + offset
            local = dict(direct=values, direct_full=values + scale * .001,
                one_step=values[:, 1:] + scale * .1,
                rollout=np.concatenate((values[:, :1], values[:, 1:] + scale * .2), axis=1),
                labels=np.column_stack((r, np.full(len(r), .2), np.zeros(len(r)))), cell_ids=np.asarray(ids),
                feature_target=targets.copy(), feature_reconstruction=targets + feature_error,
                one_step_feature_prediction=targets[:, 1:] + feature_error * 2,
                rollout_feature_prediction=targets + feature_error * 3)
            arrays.update({f"{name}_{split}_{key}": value for key, value in local.items()})
    for name, offset in (("source", 0), ("A", 7), ("B", -5)):
        g = np.linspace(-4, 4, 20) + offset
        drift = -.1 * (g - offset)
        arrays.update({f"{name}_flow_g": g, f"{name}_flow_R": g + drift,
                       f"{name}_flow_displacement": drift})
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    modules = dict(Q="q0", C="c0", R="r0")
    pins = dict(data_manifest="data", source_checkpoint="source", source_summary="summary")
    manifest = dict(schema_version=5, theory_dim=1, options=dict(steps=8),
        labels_used_in_training=False, source_pins=pins, paired_schedule_sha256="schedule")
    receipts = {name: dict(actual_steps=8, C_updates=0,
        optimizer_steps={key: dict(min=8, max=8, parameters=3) for key in ("Q", "R")},
        initial_module_sha256=modules.copy(), final_module_sha256=dict(Q=f"q{name}", C="c0", R=f"r{name}"),
        numerical_fixed_points=[dict(g=100., classification="repelling", R_derivative=1.1,
            thermal_nu=None, physical_critical_point_certified=False)]) for name in ("A", "B")}
    receipts["source"] = dict(actual_steps=0, C_updates=0, initial_module_sha256=modules.copy(),
        final_module_sha256=modules.copy(), numerical_fixed_points=[])
    summary = dict(status="complete", arms=receipts, frozen_source_verified=True,
        source_pins=pins, paired_schedule_sha256="schedule", critical_r=None, thermal_nu=None,
        data_warnings=["原有采样质量警告"])
    history = dict(latest_step=8, history=[dict(step=step, elapsed_seconds=step,
        arms={name: dict(q_loss=.01 / step, q_base=.01 / step, reconstruction=.001 / step,
            subset_micro=.0001, subset_coarse=.0002, q_guard=0.,
            q_compatible_closure=.0003 if name == "B" else 0.,
            r_loss=.002 / step, closure_one_step=.001 / step, closure_rollout=.003 / step)
              for name in ("A", "B")}) for step in (1, 8)])
    _write(root / "manifest.json", manifest)
    _write(root / "summary.json", summary)
    _write(root / "history_progress.json", history)
    _write(root / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False))
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


def test_metrics_native_pairing_and_coordinate_rescaling(tmp_path):
    root = _fixture(tmp_path / "run")
    public = report.load_public_ablation(root)
    a, b = (report.endpoint_metrics(public, name) for name in ("A", "B"))
    rows_a, rows_b = a["latent_closure"]["rows"], b["latent_closure"]["rows"]
    for ra, rb in zip(rows_a, rows_b):
        assert rb["model"]["absolute_rmse"] == pytest.approx(ra["model"]["absolute_rmse"] * 1.5)
        assert rb["model"]["fixed_train_nrmse"] == pytest.approx(ra["model"]["fixed_train_nrmse"])
    assert a["dispersion"]["rows"][0]["std_ratio_to_source"] == pytest.approx(2)
    assert b["dispersion"]["rows"][0]["std_ratio_to_source"] == pytest.approx(3)
    for row in b["q_reconstruction"]["rows"]:
        assert row["reconstruction"]["absolute_rmse"] == pytest.approx(.001)
        assert "same_scale_train_mean_constant" in row
    for row in b["native_statistic_prediction"]["rows"]:
        assert row["model"]["absolute_rmse"] == pytest.approx(.002 if row["branch"] == "one_step" else .003)
        assert row["identity_source_statistic"]["train_target_population_std"] == row["model"]["train_target_population_std"]
    pairing = report._paired_native(public)
    assert pairing["all_native_targets_exactly_equal"] is True
    for row in pairing["reconstruction_B_minus_A"]:
        assert row["mean_B_minus_A_mse"] == pytest.approx(.001**2 - .003**2)


def test_full_report_reads_no_checkpoint_keeps_sources_and_chinese_figures(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "run")
    (root / "training_step_8.pt").write_bytes(b"not a valid checkpoint; never read")
    before = {path.name: report.sha256(path) for path in root.iterdir()}
    plot = report._plotting(None)
    saved = []
    original_close = plot.close
    def capture_close(fig=None):
        if hasattr(fig, "findobj"):
            from matplotlib.text import Text
            saved.extend(text.get_text() for text in fig.findobj(Text) if text.get_text())
        original_close(fig)
    monkeypatch.setattr(plot, "close", capture_close)
    provenance, comparison = report.write_report(root, tmp_path / "report")
    assert {path.name: report.sha256(path) for path in root.iterdir()} == before
    assert provenance["model_loaded"] is provenance["physical_data_loaded"] is provenance["cache_loaded"] is False
    assert provenance["GPU_used"] is provenance["inference_performed"] is provenance["root_search_performed"] is False
    assert provenance["MI_reestimated"] is False
    assert comparison["critical_r"] is comparison["thermal_nu"] is None
    assert comparison["data_warnings"] == ["原有采样质量警告"]
    assert comparison["latent_raw_error_cross_arm_ranking_performed"] is False
    assert set(provenance["public_files"]) == {"R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png",
        "comparison.json", "report.json", "status.json"}
    assert any("生成参数 r" in value for value in saved)
    assert any("位移 R(g)−g" == value for value in saved)
    assert any("表示闭合兼容" == value for value in saved)
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "report" / name) == digest
    with pytest.raises(FileExistsError):
        report.write_report(root, tmp_path / "report")


@pytest.mark.parametrize("split,suffix", [("train", "cell_ids"), ("test", "labels"),
    ("train", "feature_target"), ("test", "feature_target")])
def test_changed_paired_sources_rejected(tmp_path, split, suffix):
    root = _fixture(tmp_path / "run")
    def mutate(arrays):
        key = f"B_{split}_{suffix}"
        arrays[key] = arrays[key].copy()
        if suffix == "cell_ids":
            arrays[key] += 10
        elif suffix == "labels":
            arrays[key][:, 0] += .01
        else:
            arrays[key][:, 1, 0] += .00001
    _mutate_arrays(root, mutate)
    with pytest.raises(ValueError, match="Paired"):
        report.load_public_ablation(root)


@pytest.mark.parametrize("kind", ["cell_ids", "labels"])
def test_split_overlap_rejected(tmp_path, kind):
    root = _fixture(tmp_path / "run")
    def mutate(arrays):
        for name in report.NAMES:
            if kind == "cell_ids":
                arrays[f"{name}_test_cell_ids"][:] = 0
            else:
                arrays[f"{name}_test_labels"][:] = arrays[f"{name}_train_labels"][0]
    _mutate_arrays(root, mutate)
    with pytest.raises(ValueError, match="overlap"):
        report.load_public_ablation(root)


@pytest.mark.parametrize("module,field,value", [("Q", "min", 7), ("R", "max", 7), ("R", "parameters", 0)])
def test_bad_adam_counters_rejected(tmp_path, module, field, value):
    root = _fixture(tmp_path / "run")
    _mutate(root, "summary.json", lambda obj: obj["arms"]["B"]["optimizer_steps"][module].__setitem__(field, value))
    with pytest.raises(ValueError, match="Adam counters"):
        report.load_public_ablation(root)


@pytest.mark.parametrize("kind", ["C_changed", "C_updates", "initial_Q_changed", "source_changed", "schedule_changed", "not_verified"])
def test_frozen_and_initial_receipts_rejected(tmp_path, kind):
    root = _fixture(tmp_path / "run")
    def mutate(obj):
        if kind == "C_changed":
            obj["arms"]["A"]["final_module_sha256"]["C"] = "changed"
        elif kind == "C_updates":
            obj["arms"]["A"]["C_updates"] = 1
        elif kind == "initial_Q_changed":
            obj["arms"]["B"]["initial_module_sha256"]["Q"] = "different"
        elif kind == "source_changed":
            obj["arms"]["source"]["final_module_sha256"]["Q"] = "different"
        elif kind == "schedule_changed":
            obj["paired_schedule_sha256"] = "different"
        else:
            obj["frozen_source_verified"] = False
    _mutate(root, "summary.json", mutate)
    with pytest.raises(ValueError):
        report.load_public_ablation(root)


def test_only_existing_roots_are_forwarded_and_support_checked(tmp_path):
    root = _fixture(tmp_path / "run")
    public = report.load_public_ablation(root)
    # All saved flow grids cross zero; root search/interpolation is forbidden.
    assert report.endpoint_metrics(public, "source")["saved_numerical_roots_with_support"] == []
    roots = report.endpoint_metrics(public, "A")["saved_numerical_roots_with_support"]
    assert roots[0]["saved_root"]["g"] == 100.
    assert roots[0]["within_training_input_range"] is False
    assert not any(row["within_saved_range"] for row in roots[0]["support_by_scale"])
    assert roots[0]["physical_critical_point_certified"] is False
    _mutate(root, "summary.json", lambda obj: obj["arms"]["A"]["numerical_fixed_points"][0].__setitem__("thermal_nu", 1.))
    with pytest.raises(ValueError, match="roots"):
        report.load_public_ablation(root)


def test_sha_nonfinite_and_incomplete_rejected(tmp_path):
    root = _fixture(tmp_path / "run")
    _mutate(root, "status.json", lambda obj: obj.__setitem__("status", "running"))
    with pytest.raises(ValueError, match="completed"):
        report.load_public_ablation(root)
    _mutate(root, "status.json", lambda obj: obj.__setitem__("status", "complete"))
    _mutate_arrays(root, lambda arrays: arrays["A_train_direct"].__setitem__((0, 0, 0), np.nan))
    with pytest.raises(ValueError, match="finite"):
        report.load_public_ablation(root)
    _mutate_arrays(root, lambda arrays: arrays["A_train_direct"].__setitem__((0, 0, 0), 1.))
    status = json.loads((root / "status.json").read_text())
    status["summary_sha256"] = "incorrect"
    _write(root / "status.json", status)
    with pytest.raises(ValueError, match="SHA256"):
        report.load_public_ablation(root)
