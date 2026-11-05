"""Synthetic CPU-only controls; never read physical fields or use a GPU."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_b2_frozen_r_diagnostic as diagnostic
from scripts import report_phi4_b2_closure_comparison as source_report


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(root):
    status = json.loads((root / "status.json").read_text())
    status.update({key: diagnostic.sha256_file(root / filename) for key, filename in (
        ("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
        ("history_progress_sha256", "history_progress.json"), ("evaluation_arrays_sha256", "evaluation_arrays.npz"))})
    _write(root / "status.json", status)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    torch.manual_seed(914)
    models = {name: diagnostic.make_r() for name in ("legacy", "all_start")}
    for model in models.values():
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(.1)
    arrays, endpoints = {}, {name: {} for name in models}
    rng = np.random.default_rng(51)
    for split, cells in (("train", np.arange(8)), ("test", np.arange(8, 12))):
        ids = np.repeat(cells, 4)
        base = np.repeat(np.linspace(-1, 1, len(cells)), 4) + np.tile([-.02, -.01, .01, .02], len(cells))
        direct = np.stack([base * (1 - .05 * k) + .04 * k for k in range(4)], axis=1)[..., None]
        labels = np.column_stack((np.repeat(np.linspace(-.5, .3, 12)[cells], 4),
                                  np.full(len(ids), .2), np.zeros(len(ids))))
        targets = rng.uniform(.002, .04, (len(ids), 4, 105))
        for arm, model in models.items():
            predicted = diagnostic.predictions(model, direct)
            local = dict(direct=direct.copy(), direct_full=direct + .001, **predicted,
                labels=labels, cell_ids=ids, feature_target=targets.copy(),
                feature_reconstruction=targets + .002,
                one_step_feature_prediction=targets[:, 1:] + .003,
                rollout_feature_prediction=targets + .004)
            endpoints[arm].update({f"{split}_{key}": value for key, value in local.items()})
            arrays.update({f"{arm}_{split}_{key}": value for key, value in local.items()})
    for arm, model in models.items():
        local = endpoints[arm]
        labels = np.concatenate([local[f"{split}_labels"] for split in ("train", "test")])
        direct = np.concatenate([local[f"{split}_direct"] for split in ("train", "test")])
        one = np.concatenate([local[f"{split}_one_step"] for split in ("train", "test")])
        grid_labels = np.unique(labels, axis=0)
        means = np.array([direct[np.all(labels == label, axis=1)].mean(axis=0) for label in grid_labels])
        mean_r = np.array([one[np.all(labels == label, axis=1)].mean(axis=0) for label in grid_labels])
        with torch.no_grad():
            r_of_mean = diagnostic.advance(model, torch.as_tensor(means[:, :3])).numpy()
            grid = np.linspace(-2, 2, 40)
            mapped = diagnostic.advance(model, torch.as_tensor(grid[:, None])).numpy()[:, 0]
        shared = dict(flow_g=grid, flow_R=mapped, flow_displacement=mapped - grid,
            grid_labels=grid_labels, grid_mean_direct=means, grid_mean_R=mean_r,
            grid_R_of_mean=r_of_mean, grid_displacement=r_of_mean - means[:, :3],
            grid_jensen_difference=mean_r - r_of_mean)
        arrays.update({f"{arm}_{key}": value for key, value in shared.items()})
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    initial = dict(Q="1" * 64, C="2" * 64, R="3" * 64)
    manifest = dict(schema_version=5, recipe=source_report.RECIPE, theory_dim=1,
        lattice_sizes=list(diagnostic.SIZES), options=dict(steps=3000), closure_modes=source_report.MODES,
        block_size=2, stride=2, Q_closure_feedback=False, warm_start=False, C_pretraining=False,
        kernel_selection=False, learning_rate=.001, closure_scale_weights=None,
        closure_weights=dict(commutator=5, rollout_weight=5, total_all_start=10),
        labels_used_in_training=False, paired_schedule_sha256="4" * 64, data_manifest_sha256="5" * 64,
        q_initialization_receipt=dict(fields_per_ensemble=128), data_warning_count=35)
    receipts = {arm: dict(actual_steps=3000, closure_objective=source_report.MODES[arm],
        optimizer_steps={name: dict(min=3000, max=3000, parameters=6) for name in ("Q", "C", "R")},
        initial_module_sha256=initial, final_module_sha256=dict(Q="6" * 64, C="7" * 64,
            R=diagnostic.state_fingerprint(model)), numerical_fixed_points=[])
        for arm, model in models.items()}
    summary = dict(status="complete", recipe=source_report.RECIPE, arms=receipts,
        actual_steps=3000, actual_steps_by_arm={arm: 3000 for arm in models},
        closure_modes=source_report.MODES, initial_module_sha256=initial, C_Q_pairing_verified=True,
        paired_schedule_sha256=manifest["paired_schedule_sha256"], data_manifest_sha256=manifest["data_manifest_sha256"],
        thermal_nu=None, critical_r=None, equilibrium_certified=False)
    _write(root / "manifest.json", manifest)
    _write(root / "summary.json", summary)
    _write(root / "history_progress.json", dict(latest_step=3000, history=[dict(step=3000,
        arms={arm: dict(q_loss=.1, c_loss=.2, r_loss=.3) for arm in models})]))
    _write(root / "status.json", dict(recipe=source_report.RECIPE, status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False))
    _refresh(root)
    extracted = dict(recipe=diagnostic.EXTRACTION_RECIPE,
        source_manifest_sha256=diagnostic.sha256_file(root / "manifest.json"),
        source_summary_sha256=diagnostic.sha256_file(root / "summary.json"), arms={})
    for arm, model in models.items():
        folder = root / arm
        folder.mkdir()
        checkpoint = folder / "training_step_3000.pt"
        # Intentionally not a loadable checkpoint: any attempt to torch.load the
        # original full model will fail, whereas byte fingerprinting is allowed.
        checkpoint.write_bytes(b"synthetic full QCR checkpoint, must never deserialize")
        extracted["arms"][arm] = dict(state_dict=copy.deepcopy(model.state_dict()),
            source_checkpoint_path=str(checkpoint), source_checkpoint_sha256=diagnostic.sha256_file(checkpoint),
            step=3000, closure_objective=source_report.MODES[arm])
    extraction_path = tmp_path / "R_only.pt"
    torch.save(extracted, extraction_path)
    return root, extraction_path, extracted


def test_only_r_architecture_and_correct_closure_formula():
    model = diagnostic.make_r()
    assert list(model.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"]
    assert model[0].weight.shape == (96, 1) and model[2].weight.shape == (96, 96)
    direct = torch.randn(8, 4, 1, dtype=torch.float64)
    legacy, a = diagnostic.closure_loss(model, direct, "one_step_and_rollout")
    all_start, b = diagnostic.closure_loss(model, direct, "all_start_anchor")
    assert legacy.item() == pytest.approx(5 * (a["one_step"] + a["micro_rollout"]))
    assert all_start.item() == b["all_start"]
    assert a["one_step"] == b["one_step"] and a["micro_rollout"] == b["micro_rollout"]
    all_start.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not direct.requires_grad


def test_source_public_fingerprints_and_r_reproduction(source, monkeypatch):
    root, extraction_path, _ = source
    original = diagnostic.torch.load
    loaded = []
    def only_extracted(path, **kwargs):
        loaded.append(path)
        assert path == extraction_path
        assert kwargs == dict(map_location="cpu", weights_only=True)
        return original(path, **kwargs)
    monkeypatch.setattr(diagnostic.torch, "load", only_extracted)
    public, models, receipt, fingerprints = diagnostic.load_sources(root, extraction_path)
    assert loaded == [extraction_path]
    assert public.endpoints["legacy"]["train_direct"].shape == (32, 4, 1)
    assert set(models) == {"legacy", "all_start"} and len(fingerprints) == 8
    assert all(row["saved_one_step_max_absolute_error"] == 0 for row in receipt.values())
    assert all(row["full_checkpoint_deserialized"] is False for row in receipt.values())


@pytest.mark.parametrize("kind", ["wrong_provenance", "extra_Q", "R_hash", "source_checkpoint_changed", "saved_output"])
def test_rejects_invalid_source_or_extraction(source, kind):
    root, path, extracted = source
    if kind == "wrong_provenance":
        extracted["source_manifest_sha256"] = "0" * 64
    elif kind == "extra_Q":
        extracted["Q_state_dict"] = {"something": torch.zeros(1)}
    elif kind == "R_hash":
        extracted["arms"]["legacy"]["state_dict"]["0.bias"].add_(1)
    elif kind == "source_checkpoint_changed":
        (root / "legacy" / "training_step_3000.pt").write_bytes(b"changed")
    elif kind == "saved_output":
        with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["legacy_test_one_step"][0, 0, 0] += 1e-5
        np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
        _refresh(root)
    torch.save(extracted, path)
    with pytest.raises(ValueError):
        diagnostic.load_sources(root, path)


def test_metrics_use_same_fixed_train_std_and_identity_for_all_endpoints():
    train = np.arange(32 * 4, dtype=float).reshape(32, 4, 1) / 100
    test = np.arange(16 * 4, dtype=float).reshape(16, 4, 1) / 70
    predictions = {split: dict(one_step=direct[:, 1:] + .1,
        rollout=np.concatenate((direct[:, :1], direct[:, 1:] + .2), axis=1))
        for split, direct in (("train", train), ("test", test))}
    rows = diagnostic.evaluate_metrics(train, test, predictions)
    assert len(rows) == 12
    for row in rows:
        assert row["model"]["absolute_rmse"] == pytest.approx(.1 if row["branch"] == "one_step" else .2)
        assert row["model"]["train_target_population_std"] == train[:, row["target_scale"]].std(ddof=0)
        assert row["identity"]["train_target_population_std"] == row["model"]["train_target_population_std"]


def test_pchip_fit_uses_no_labels_no_extrapolation_and_reports_coverage():
    x = np.repeat([-.8, -.3, .2, .7], 2)
    train = np.stack([x + .1 * k for k in range(4)], axis=1)[..., None]
    y = np.array([-1., -.4, 0., .8])
    test = np.stack([y + .1 * k for k in range(4)], axis=1)[..., None]
    receipt, arrays = diagnostic.pchip_control(train, test, np.repeat(np.arange(4), 2), np.arange(4, 8))
    assert receipt["coupling_labels_used"] is False and receipt["extrapolation"] is False
    assert receipt["recursive_rollout"] is False and receipt["unified_autonomous_R"] is False
    for row in receipt["rows"]:
        if row["split"] == "test":
            assert row["coverage_fraction"] == .5
            assert row["outside_training_range_indices"] == [0, 3]
            assert row["model"]["absolute_rmse"] < 1e-14
    assert np.isnan(arrays["pchip_test_scale0_prediction"][[0, 3]]).all()


@pytest.mark.parametrize("kwargs", [dict(steps=30), dict(batch_size=16), dict(seed=1), dict(optimizer_seed=91),
                                   dict(learning_rate=.01), dict(cpu_budget_seconds=301), dict(directory_budget_bytes=101 * 1024 ** 2)])
def test_protocol_or_budget_expansion_not_allowed(kwargs):
    with pytest.raises(ValueError):
        diagnostic.Options("source", "r.pt", "out", **kwargs).validate()


def test_short_synthetic_integration_strictly_cpu_and_paired(source, tmp_path, monkeypatch):
    root, extracted, _ = source
    monkeypatch.setattr(diagnostic, "CONTROL_STEPS", 4)
    plots = []
    monkeypatch.setattr(diagnostic, "_plots", lambda *args: plots.append(args[0]))
    output = tmp_path / "diagnostic"
    result = diagnostic.run(diagnostic.Options(str(root), str(extracted), str(output), steps=4))
    assert result["status"] == "complete" and result["GPU_seconds"] == 0
    assert result["actual_steps_by_arm"] == {name: 4 for name in diagnostic.CONTROL_NAMES}
    assert result["latest_checkpoint_step_by_arm"] == {name: 4 for name in diagnostic.CONTROL_NAMES}
    assert all(row == dict(min=4, max=4, parameters=6) for row in result["optimizer_steps"].values())
    assert result["source_fingerprints_unchanged"] is True and plots == [output]
    status = json.loads((output / "status.json").read_text())
    assert status["returncode"] == 0 and status["workflow_complete"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["Q_loaded"] is manifest["C_loaded"] is False
    assert manifest["Q_updates"] == manifest["C_updates"] == 0
    assert manifest["cached_g_used"] is True and manifest["original_training_target_distribution_reproduced"] is False
    assert result["thermal_nu"] is result["critical_r"] is None
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["train_direct"].shape == (32, 4, 1)
        for name in diagnostic.CONTROL_NAMES:
            assert arrays[f"{name}_flow_g"].shape == (256,)
            assert np.allclose(arrays[f"{name}_flow_R"] - arrays[f"{name}_flow_g"], arrays[f"{name}_flow_displacement"])
    with pytest.raises(FileExistsError):
        diagnostic.run(diagnostic.Options(str(root), str(extracted), str(output), steps=4))


def test_budget_hold_preserves_source_and_does_not_restart(source, tmp_path, monkeypatch):
    root, extracted, _ = source
    output = tmp_path / "held"
    result = diagnostic.run(diagnostic.Options(str(root), str(extracted), str(output), cpu_budget_seconds=1e-9))
    assert result["status"] == "budget_hold"
    status = json.loads((output / "status.json").read_text())
    assert status["returncode"] == 3 and status["workflow_complete"] is False
    assert status["requires_user_action"] is True
    assert status["actual_steps_by_arm"] == {name: 0 for name in diagnostic.CONTROL_NAMES}


def test_chinese_plots_render_on_synthetic_control(source, tmp_path, monkeypatch):
    root, extracted, _ = source
    monkeypatch.setattr(diagnostic, "CONTROL_STEPS", 1)
    output = tmp_path / "rendered"
    result = diagnostic.run(diagnostic.Options(str(root), str(extracted), str(output), steps=1))
    assert result["status"] == "complete"
    assert all((output / name).stat().st_size > 1000 for name in (
        "R_vs_g.png", "loss.png", "metrics.png", "coupling_by_scale.png"))
    for row in result["per_transition_interpolation"]["rows"]:
        assert set(row["unified_R_on_same_covered_chains"]) == set(diagnostic.LABELS)


def test_source_mutation_after_optimization_fails_without_repair(source, tmp_path, monkeypatch):
    root, extracted, _ = source
    monkeypatch.setattr(diagnostic, "CONTROL_STEPS", 1)
    checkpoint = root / "legacy" / "training_step_3000.pt"
    monkeypatch.setattr(diagnostic, "_plots", lambda *args: checkpoint.write_bytes(b"changed during diagnostic"))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="changed during diagnostic"):
        diagnostic.run(diagnostic.Options(str(root), str(extracted), str(output), steps=1))
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["returncode"] == 1
    assert status["actual_steps_by_arm"] == {name: 1 for name in diagnostic.CONTROL_NAMES}
    assert status["latest_checkpoint_step_by_arm"] == {name: 1 for name in diagnostic.CONTROL_NAMES}
