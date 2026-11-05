"""Synthetic receipt/array tests: no physical data, models, GPU, or root search."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_b2_validation as report


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(root):
    status = json.loads((root / "status.json").read_text())
    status.update({key: report.sha256(root / filename) for key, filename in (
        ("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
        ("history_progress_sha256", "history_progress.json"),
        ("evaluation_arrays_sha256", "evaluation_arrays.npz"))})
    _write(root / "status.json", status)


def _source(root):
    # Reuse the established synthetic public artifact, not its loader assumptions.
    spec = importlib.util.spec_from_file_location("old_report_fixture", Path(__file__).with_name("test_phi4_b2_closure_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._fixture(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["options"]["steps"] = 3000
    summary = json.loads((root / "summary.json").read_text())
    summary["actual_steps"] = 3000
    summary["actual_steps_by_arm"] = {name: 3000 for name in ("legacy", "all_start")}
    for receipt in summary["arms"].values():
        receipt["actual_steps"] = 3000
        for count in receipt["optimizer_steps"].values():
            count.update(min=3000, max=3000)
    summary["data_warning_count"] = 35
    history = json.loads((root / "history_progress.json").read_text())
    history["latest_step"] = history["latest_checkpoint_step"] = 3000
    history["history"][-1]["step"] = 3000
    for name, payload in (("manifest.json", manifest), ("summary.json", summary), ("history_progress.json", history)):
        _write(root / name, payload)
    _refresh(root)
    return report.load_public_comparison(root)


def _stage(root, stage, source):
    root.mkdir()
    names = report.STAGE_ARMS[stage]
    baseline = source.endpoints["all_start"]
    source_initial = source.summary["initial_module_sha256"]
    source_final = source.summary["arms"]["all_start"]["final_module_sha256"]
    planned, receipts, arrays = {}, {}, {}
    for name in names:
        local = {key: value.copy() for key, value in baseline.items()}
        for split in ("train", "test"):
            count = local[f"{split}_direct"].shape[0]
            local[f"{split}_field_mean"] = np.linspace(-.05, .05, count)[:, None] + .01 * np.arange(4)[None, :]
            local[f"{split}_field_rms"] = np.broadcast_to(1 + .1 * np.arange(4), (count, 4)).copy()
            local[f"{split}_field_mirror_l1"] = .01 * np.arange(count)[:, None] + .1 * np.arange(4)[None, :]
        initial = dict(source_final if stage == "r_tail" else source_initial)
        final = dict(source_final)
        if name == "cubic_decoder":
            initial["Q"] = "b" * 64
            final["Q"] = "c" * 64
            latent = ("direct", "direct_full", "one_step", "rollout")
            for split in ("train", "test"):
                for suffix in latent:
                    local[f"{split}_{suffix}"] *= 2
            for key in local:
                if key.startswith("flow_") or key.startswith("grid_") and key != "grid_labels":
                    local[key] *= 2
        elif name == "odd_C":
            final["Q"], final["C"] = "d" * 64, "e" * 64
            for key in local:
                if "feature_" in key:
                    local[key] += .05
            for split in ("train", "test"):
                local[f"{split}_field_mean"][:, 1:] *= .1
                local[f"{split}_field_rms"][:, 1:] *= .99
                local[f"{split}_field_mirror_l1"][:, 1:] *= .4
        else:
            final["R"] = "f" * 64
            for split in ("train", "test"):
                for prediction, target in (("one_step", "direct"), ("rollout", "direct"),
                    ("one_step_feature_prediction", "feature_target"), ("rollout_feature_prediction", "feature_target")):
                    truth = local[f"{split}_{target}"]
                    if prediction.startswith("one_step"):
                        truth = truth[:, 1:]
                    local[f"{split}_{prediction}"] = .5 * (local[f"{split}_{prediction}"] + truth)
        arrays.update({f"{name}_{key}": value for key, value in local.items()})
        planned[name] = dict(variant=name, initial_module_sha256=initial,
            initial_common_module_sha256=dict(source_final if stage == "r_tail" else source_initial), parameters={})
        counts = {module: 0 if stage == "r_tail" and module in ("Q", "C") else 3000 for module in ("Q", "C", "R")}
        receipts[name] = dict(actual_steps=3000, source_QCR_steps=3000, final_module_sha256=final,
            initial_module_sha256=initial,
            initial_common_module_sha256=dict(source_final if stage == "r_tail" else source_initial),
            closure_objective="all_start_anchor",
            optimizer_steps={module: dict(min=value, max=value, parameters=3) for module, value in counts.items()},
            **{f"{module}_updates": value for module, value in counts.items()}, critical_r=None, thermal_nu=None)
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    manifest = dict(schema_version=5, recipe=report.RECIPE, stage=stage, options=dict(steps=3000),
        lattice_sizes=list(report.SIZES), theory_dim=1, block_size=2, stride=2,
        labels_used_in_training=False, source_public_sha256=source.fingerprints,
        source_checkpoint_sha256="a" * 64, paired_schedule_sha256=source.manifest["paired_schedule_sha256"],
        closure_objective="all_start_anchor", learning_rate=.001, closure_scale_weights=None,
        closure_weights=dict(commutator=5, rollout_weight=5, total_all_start=10),
        data_manifest_sha256=source.manifest["data_manifest_sha256"], arms=planned)
    summary = dict(recipe=report.RECIPE, stage=stage, status="complete", labels_used_in_training=False,
        data_manifest_sha256=source.manifest["data_manifest_sha256"], arms=receipts,
        actual_steps=3000, actual_steps_by_arm={name: 3000 for name in names},
        source_public_sha256=source.fingerprints, paired_schedule_sha256=source.manifest["paired_schedule_sha256"],
        source_inputs_unchanged=True, gradient_routing_verified=True,
        cubic_C_target_source_equal=True, r_tail_QC_source_equal=True,
        critical_r=None, thermal_nu=None, equilibrium_certified=False)
    history = dict(latest_step=3000, latest_checkpoint_step=3000,
        history=[dict(step=step, arms={name: dict(q_loss=.1 / step, c_loss=-.2 / step,
            r_loss=.3 / step, reconstruction=.001, information_proxy=.2) for name in names}) for step in (1, 3000)])
    status = dict(recipe=report.RECIPE, stage=stage, status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False)
    for name, payload in (("manifest.json", manifest), ("summary.json", summary), ("history_progress.json", history), ("status.json", status)):
        _write(root / name, payload)
    _refresh(root)
    return root


def _fixture(tmp_path):
    source = _source(tmp_path / "source")
    return source.root, _stage(tmp_path / "joint", "joint", source), _stage(tmp_path / "r_tail", "r_tail", source)


def _mutate(root, filename, change, refresh=True):
    payload = json.loads((root / filename).read_text())
    change(payload)
    _write(root / filename, payload)
    if refresh:
        _refresh(root)


def _mutate_arrays(root, change):
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    change(arrays)
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    _refresh(root)


def test_complete_single_factor_contract_and_safe_comparisons(tmp_path):
    args = _fixture(tmp_path)
    validation = report.load_validation(*args)
    metrics = {name: report._metrics(arrays) for name, arrays in validation.endpoints.items()}
    comparison = report.comparisons(validation, metrics)
    assert set(metrics) == set(report.NAMES)
    assert len(comparison["cubic_decoder"]["native_prediction_rows"]) == 12
    assert len(comparison["cubic_decoder"]["native_reconstruction_rows"]) == 8
    assert comparison["cubic_decoder"]["g_coordinates_identified"] is False
    assert comparison["odd_C"]["same_native_target_comparison_valid"] is False
    assert comparison["odd_C"]["native_targets_exactly_equal"] is False
    assert comparison["r_tail"]["source_QCR_steps"] == 3000
    assert comparison["r_tail"]["additional_Q_updates"] == 0
    c_diagnostic = report.C_diagnostics(validation)
    assert c_diagnostic["original_C_and_R_tail_exactly_equal"] is True
    assert c_diagnostic["additional_baseline_inference_performed"] is False
    assert c_diagnostic["additional_conditional_draws"] == 0
    changes = [row for row in c_diagnostic["paired_changes"] if row["scale"] > 0]
    assert all(row["relative_mean_change"] == pytest.approx(-.9)
        for row in changes if row["metric"] == "absolute_chain_field_mean")
    assert all(row["relative_mean_change"] == pytest.approx(-.6)
        for row in changes if row["metric"] == "field_mirror_l1")
    for row in comparison["r_tail"]["latent_rows"]:
        assert row["relative_change"] == pytest.approx(-.5)
    for arm in metrics.values():
        for row in arm["native_statistic_prediction"]["rows"]:
            assert row["model"]["train_target_population_std"] == row["identity_source_statistic"]["train_target_population_std"]
            assert row["model"]["train_target_population_std"] == row["same_scale_train_mean_constant"]["train_target_population_std"]


@pytest.mark.parametrize("name,kind", [("cubic_decoder", "latent_closure"), ("odd_C", "latent_closure"), ("odd_C", "native_statistic_prediction")])
def test_invalid_cross_coordinate_or_changed_target_ranking_rejected(tmp_path, name, kind):
    validation = report.load_validation(*_fixture(tmp_path))
    metrics = {arm: report._metrics(arrays) for arm, arrays in validation.endpoints.items()}
    with pytest.raises(ValueError, match="forbidden"):
        report._paired_rows(validation, metrics, name, kind)


@pytest.mark.parametrize("filename", ["manifest.json", "summary.json", "history_progress.json", "evaluation_arrays.npz"])
def test_public_sha_missing_or_stale_rejected(tmp_path, filename):
    source, joint, tail = _fixture(tmp_path)
    key = {"manifest.json": "manifest_sha256", "summary.json": "summary_sha256",
        "history_progress.json": "history_progress_sha256", "evaluation_arrays.npz": "evaluation_arrays_sha256"}[filename]
    _mutate(joint, "status.json", lambda obj: obj.pop(key), refresh=False)
    with pytest.raises(ValueError, match="SHA256"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("module", ["Q", "C", "R"])
def test_bad_joint_optimizer_counters_rejected(tmp_path, module):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "summary.json", lambda obj: obj["arms"]["cubic_decoder"]["optimizer_steps"][module].__setitem__("max", 2999))
    with pytest.raises(ValueError, match="optimizer counts"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("module", ["Q", "C"])
def test_frozen_optimizer_updates_rejected(tmp_path, module):
    source, joint, tail = _fixture(tmp_path)
    _mutate(tail, "summary.json", lambda obj: obj["arms"]["r_tail"].__setitem__(f"{module}_updates", 1))
    with pytest.raises(ValueError, match="optimizer counts"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("module", ["Q", "C"])
def test_frozen_module_hash_change_rejected(tmp_path, module):
    source, joint, tail = _fixture(tmp_path)
    _mutate(tail, "summary.json", lambda obj: obj["arms"]["r_tail"]["final_module_sha256"].__setitem__(module, "a" * 64))
    with pytest.raises(ValueError, match="frozen source"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("suffix", ["direct", "direct_full", "feature_target", "feature_reconstruction"])
def test_tail_source_arrays_are_exactly_frozen(tmp_path, suffix):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(tail, lambda arrays: arrays[f"r_tail_test_{suffix}"].__iadd__(.001))
    with pytest.raises(ValueError):
        report.load_validation(source, joint, tail)


def test_cubic_changed_native_target_rejected(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(joint, lambda arrays: arrays["cubic_decoder_test_feature_target"].__iadd__(.001))
    with pytest.raises(ValueError, match="same-target"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("stage", ["joint", "r_tail"])
def test_bad_initial_common_receipt_rejected(tmp_path, stage):
    source, joint, tail = _fixture(tmp_path)
    root, arm = (joint, "odd_C") if stage == "joint" else (tail, "r_tail")
    _mutate(root, "manifest.json", lambda obj: obj["arms"][arm]["initial_common_module_sha256"].__setitem__("R", "a" * 64))
    _mutate(root, "summary.json", lambda obj: obj["arms"][arm]["initial_common_module_sha256"].__setitem__("R", "a" * 64))
    with pytest.raises(ValueError, match="Common initialization"):
        report.load_validation(source, joint, tail)


def test_source_public_fingerprint_pin_required(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "manifest.json", lambda obj: obj["source_public_sha256"].pop("status.json"))
    with pytest.raises(ValueError, match="source public SHA"):
        report.load_validation(source, joint, tail)


def test_physical_data_fingerprint_change_rejected(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "summary.json", lambda obj: obj.__setitem__("data_manifest_sha256", "a" * 64))
    with pytest.raises(ValueError, match="Physical data"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("key", ["critical_r", "thermal_nu"])
def test_no_new_critical_or_exponent_estimate(tmp_path, key):
    source, joint, tail = _fixture(tmp_path)
    _mutate(tail, "summary.json", lambda obj: obj.__setitem__(key, .5))
    with pytest.raises(ValueError, match="cannot certify"):
        report.load_validation(source, joint, tail)


def test_no_new_numerical_root_search(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "summary.json", lambda obj: obj["arms"]["odd_C"].__setitem__("numerical_fixed_points", [dict(g=0.)]))
    with pytest.raises(ValueError, match="must not search"):
        report.load_validation(source, joint, tail)


def test_missing_explicit_source_step_count_rejected(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(tail, "summary.json", lambda obj: obj["arms"]["r_tail"].pop("source_QCR_steps"))
    with pytest.raises(ValueError, match="separate"):
        report.load_validation(source, joint, tail)


def test_odd_C_initial_Q_cannot_silently_change(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "manifest.json", lambda obj: obj["arms"]["odd_C"]["initial_module_sha256"].__setitem__("Q", "a" * 64))
    _mutate(joint, "summary.json", lambda obj: obj["arms"]["odd_C"]["initial_module_sha256"].__setitem__("Q", "a" * 64))
    with pytest.raises(ValueError, match="original Q/C/R parameter"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("key,value", [("closure_objective", "one_step_and_rollout"),
    ("learning_rate", .002), ("closure_scale_weights", [1, 1, 2])])
def test_additional_training_factor_change_rejected(tmp_path, key, value):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "manifest.json", lambda obj: obj.__setitem__(key, value))
    with pytest.raises(ValueError, match="preserve all-start"):
        report.load_validation(source, joint, tail)


def test_minibatch_schedule_change_rejected(tmp_path):
    source, joint, tail = _fixture(tmp_path)
    _mutate(joint, "summary.json", lambda obj: obj.__setitem__("paired_schedule_sha256", "a" * 64))
    with pytest.raises(ValueError, match="minibatch schedule"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("suffix", ["field_mean", "field_rms", "field_mirror_l1"])
def test_C_diagnostic_missing_array_rejected(tmp_path, split, suffix):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(joint, lambda arrays: arrays.pop(f"odd_C_{split}_{suffix}"))
    with pytest.raises(ValueError, match="Missing saved C diagnostic"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("suffix", ["field_mean", "field_rms", "field_mirror_l1"])
def test_same_C_fixed_evaluation_must_have_identical_diagnostics(tmp_path, suffix):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(tail, lambda arrays: arrays[f"r_tail_test_{suffix}"][:, 1:].__iadd__(.001))
    with pytest.raises(ValueError, match="Same C fixed-evaluation"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("suffix", ["field_mean", "field_rms", "field_mirror_l1"])
def test_unchanged_raw_microscopic_C_diagnostic_required(tmp_path, suffix):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(joint, lambda arrays: arrays[f"odd_C_train_{suffix}"][:, 0].__iadd__(.001))
    with pytest.raises(ValueError, match="Microscopic physical field"):
        report.load_validation(source, joint, tail)


@pytest.mark.parametrize("value", [float("nan"), -1., 2.01])
def test_invalid_C_mirror_metric_rejected(tmp_path, value):
    source, joint, tail = _fixture(tmp_path)
    _mutate_arrays(joint, lambda arrays: arrays["odd_C_test_field_mirror_l1"].__setitem__((0, 3), value))
    with pytest.raises(ValueError, match="C diagnostics|mirror-L1"):
        report.load_validation(source, joint, tail)


def test_public_only_report_chinese_plots_and_source_immutability(tmp_path, monkeypatch):
    args = _fixture(tmp_path)
    for root in args:
        (root / "training_step_3000.pt").write_bytes(b"do not load")
        (root / "physical_fields.npz").write_bytes(b"do not load")
    before = [{p.name: report.sha256(p) for p in root.iterdir()} for root in args]
    plt = report._plotting(None)
    original_close, text = plt.close, []
    def capture(fig=None):
        if hasattr(fig, "findobj"):
            from matplotlib.text import Text
            text.extend(item.get_text() for item in fig.findobj(Text) if item.get_text())
        original_close(fig)
    monkeypatch.setattr(plt, "close", capture)
    provenance, comparison = report.write_report(*args, tmp_path / "report")
    assert before == [{p.name: report.sha256(p) for p in root.iterdir()} for root in args]
    for key in ("model_loaded", "physical_data_loaded", "cache_loaded", "GPU_used", "inference_performed", "root_search_performed", "MI_reestimated"):
        assert provenance[key] is False
    assert comparison["data_warning_count"] == 35
    assert comparison["critical_r"] is comparison["thermal_nu"] is None
    assert comparison["cross_coordinate_latent_ranking_performed"] is False
    assert comparison["odd_C_same_physical_target_ranking_performed"] is False
    assert comparison["source_saved_numerical_roots_with_support"][0]["saved_root"]["g"] == 0.
    assert any("位移 R(g)−g" == value for value in text)
    assert any("仅追加 R 收尾" in value for value in text)
    assert comparison["C_diagnostics"]["additional_conditional_draws"] == 0
    assert set(provenance["public_files"]) == {"R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png", "native_error.png", "C_symmetry.png", "comparison.json", "report.json", "status.json"}
    for filename, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "report" / filename) == digest
    with pytest.raises(FileExistsError):
        report.write_report(*args, tmp_path / "report")


def test_source_root_with_training_subdirectory_supported(tmp_path):
    parent = tmp_path / "source"
    parent.mkdir()
    source = _source(parent / "training")
    joint = _stage(tmp_path / "joint", "joint", source)
    tail = _stage(tmp_path / "tail", "r_tail", source)
    assert report.load_validation(parent, joint, tail).source.root == source.root


@pytest.mark.parametrize("key", ["labels_used_in_training", "workflow_complete", "returncode"])
def test_bad_unsupervised_execution_contract_rejected(tmp_path, key):
    source, joint, tail = _fixture(tmp_path)
    if key == "labels_used_in_training":
        _mutate(joint, "manifest.json", lambda obj: obj.__setitem__(key, True))
    else:
        _mutate(joint, "status.json", lambda obj: obj.__setitem__(key, False if key == "workflow_complete" else 1), refresh=False)
    with pytest.raises(ValueError):
        report.load_validation(source, joint, tail)
