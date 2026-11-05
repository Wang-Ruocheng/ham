"""Synthetic public artifacts only: reporting never loads a trained model."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_ising_closure_scale_ablation as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(root, steps=6):
    write_json(root / "dataset_receipt.json", dict(warnings=["有限物理链警告"], samples_are_iid=False))
    arrays = {}
    for split in ("train", "test"):
        direct = np.arange(12, dtype=float).reshape(4, 3, 1) / 10
        draws = np.stack((direct - .1, direct + .1), axis=1)
        mapped = draws ** 2 + .2
        rollout = np.stack((draws[:, :, 0], draws[:, :, 0] ** 2 + .2,
                            (draws[:, :, 0] ** 2 + .2) ** 2 + .2), axis=2)
        arrays.update({"direct_" + split: draws.mean(1), "R_g_" + split: mapped.mean(1),
            "R_of_mean_g_" + split: direct ** 2 + .2, "rollout_" + split: rollout.mean(1),
            "direct_draws_" + split: draws, "R_g_draws_" + split: mapped,
            "rollout_draws_" + split: rollout,
            "couplings_" + split: np.column_stack((np.ones(4) * (.42 if split == "train" else .44), np.zeros((4, 6)))),
            "group_ids_" + split: np.repeat([0, 1], 2)})
    for arm in report.ARMS:
        weights = report.PROFILES[arm]
        cfg = dict(schema_version=5, data=dict(cache="unopened.pt"),
            model=dict(block_factor=3, dimension=2, theory_dim=1, kernel_initialization="random",
                kernel_sampling="bernoulli", kernel_layout="nonoverlapping", fixed_kernel=False, affine_reconstruction=True),
            training=dict(epochs=steps, max_rg_steps=2, gradient_mode="qc_r_separate", seed=91),
            loss=dict(closure_scale_weights=weights))
        contract = dict(weights=weights, normalized_weights=(np.asarray(weights)/sum(weights)).tolist(),
            target_levels=[1, 2], application=["one_step_commutator", "microscopic_start_rollout"],
            labels_used=False, gradient_routing="R only; no closure gradients to Q or C")
        manifest = dict(config=cfg, labels_used_in_training=False, data_sha256="a"*64,
            initial_module_sha256={"Q": "q", "C": "c", "R": "r"}, closure_scale_weighting=contract,
            q_future_physics=dict(weight=0.), auxiliary_future_parameters=0)
        history = []
        for epoch in (1, steps):
            row = dict(epoch=epoch)
            for split in ("train", "test"):
                for key in ("separation", "anchor", "coarse_consistency", "rsmi", "kernel_noncollapse", "kernel_sign",
                            "commutator", "rollout", "one_step_level_1", "one_step_level_2", "rollout_level_1", "rollout_level_2"):
                    row[split + "_" + key] = .1 / epoch
            history.append(row)
        run = root / ("run_" + arm)
        write_json(run / "manifest.json", manifest)
        write_json(run / "summary.json", dict(config=cfg, actual_epochs=steps, requested_epochs=steps,
            contract=dict(closure_scale_weighting=contract, initial_module_sha256=manifest["initial_module_sha256"]), history=history))
        (run / "report").mkdir()
        np.savez_compressed(run / "report/encodings.npz", **arrays)
        write_json(run / "report/metrics.json", dict(status="complete", actual_steps=steps, scales=[108, 36, 12],
            model_sha256_before="b"*64, model_sha256_after="b"*64, Q_sha256="q"*64, C_sha256="c"*64,
            encodings_sha256=report.public_sha(run / "report/encodings.npz")))
    return root


def test_complete_public_report_preserves_metric_units_and_null_exponent(tmp_path):
    fixture_run(tmp_path)
    result = report.report_run(tmp_path, expected_steps=6, make_plots=False)
    assert result["status"] == "complete" and result["formal_comparison_available"]
    assert result["coordinate_comparison"]["shared_coordinate_evaluation_verified"]
    assert result["coordinate_comparison"]["same_initial_module_fingerprints"]
    assert result["thermal_nu"] is None and not result["root_search_performed"]
    assert not result["weighted_objectives_compared_as_physical_improvement"]
    assert result["dataset_receipt"]["warnings"] == ["有限物理链警告"]
    for arm in report.ARMS:
        values = result["arms"][arm]["closure_per_C_draw"]["test"]
        assert [(row["source"], row["target"]) for row in values] == [(108, 36), (36, 12), (108, 12)]
        assert all(row["available"] for row in values)
        assert all(row["fixed_uniform_nrmse"] == row["nrmse"] for row in values)
    assert all(Path(name).suffix in (".json", ".npz") for name in result["input_sha256"])
    status = json.loads((tmp_path / "report/status.json").read_text())
    assert status["workflow_complete"] and status["report_sha256"] == report.public_sha(tmp_path / "report/report.json")
    with np.load(tmp_path / "report/evaluation_arrays.npz", allow_pickle=False) as z:
        assert "uniform__rollout_draws_test" in z


def test_unweighted_path_metrics_use_matching_train_target_scale():
    train = np.array([[[0.], [0.], [0.]], [[2.], [4.], [8.]]])
    test = np.zeros_like(train)
    arrays = dict(direct_train=train, direct_test=test, R_g_test=np.ones_like(test), rollout_test=3*np.ones_like(test))
    values = report.closure_summary(arrays, "test", 2*train)
    assert [row["rmse"] for row in values] == [1., 1., 3.]
    assert [row["train_target_std"] for row in values] == [2., 4., 4.]
    assert [row["nrmse"] for row in values] == [.5, .25, .75]
    assert [row["fixed_uniform_nrmse"] for row in values] == [.25, .125, .375]


def test_per_draw_loss_not_mistaken_for_loss_of_draw_means(tmp_path):
    fixture_run(tmp_path)
    _, panels = report.build_report(tmp_path, expected_steps=6)
    arrays = panels["uniform"]
    draw = report.closure_summary(arrays, "test", arrays["direct_train"], per_draw=True)
    mean = report.closure_summary(arrays, "test", arrays["direct_train"])
    assert all(a["mse"] > b["mse"] for a, b in zip(draw, mean))
    arrays.pop("rollout_draws_test")
    draw = report.closure_summary(arrays, "test", arrays["direct_train"], per_draw=True)
    assert draw[0]["available"] and draw[1]["available"]
    assert not draw[2]["available"] and draw[2]["rmse"] is None


def test_changed_QC_fingerprint_is_disclosed_not_concealed(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "run_late/report/metrics.json"
    metrics = json.loads(path.read_text()); metrics["Q_sha256"] = "different"
    write_json(path, metrics)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["formal_comparison_available"]
    assert not result["coordinate_comparison"]["shared_coordinate_evaluation_verified"]
    assert result["coordinate_comparison"]["identical_direct_arrays"]


def test_changed_initial_R_never_claims_paired_ablation(tmp_path):
    fixture_run(tmp_path)
    for name in ("manifest", "summary"):
        path = tmp_path / "run_late" / (name + ".json")
        value = json.loads(path.read_text())
        contract = value if name == "manifest" else value["contract"]
        contract["initial_module_sha256"]["R"] = "another_initial_R"
        write_json(path, value)
    with pytest.raises(ValueError, match="相同的全新"):
        report.build_report(tmp_path, expected_steps=6)


def test_early_endpoint_and_wrong_epoch_or_auxiliary_loss(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "run_late/summary.json"
    summary = json.loads(path.read_text()); summary["actual_epochs"] = 3
    summary["history"][-1]["epoch"] = 3; summary["stopped_by_training_limit"] = True
    write_json(path, summary)
    metrics_path = tmp_path / "run_late/report/metrics.json"
    metrics = json.loads(metrics_path.read_text()); metrics["actual_steps"] = 3; write_json(metrics_path, metrics)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["status"] == "incomplete" and not result["formal_comparison_available"]
    metrics["actual_steps"] = 4; write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="采集检查点"):
        report.build_report(tmp_path, expected_steps=6)
    metrics["actual_steps"] = 3; write_json(metrics_path, metrics)
    path = tmp_path / "run_late/manifest.json"
    manifest = json.loads(path.read_text()); manifest["auxiliary_future_parameters"] = 1; write_json(path, manifest)
    with pytest.raises(ValueError, match="未来物理"):
        report.build_report(tmp_path, expected_steps=6)


def test_missing_R_or_changed_data_arrays_rejected(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "run_late/report/encodings.npz"
    with np.load(path) as z: arrays = {key: z[key] for key in z.files if key != "R_g_test"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="不能用下一尺度"):
        report.build_report(tmp_path, expected_steps=6)


def test_no_model_readers_and_no_overwrite(tmp_path):
    source = Path(report.__file__).read_text()
    assert "import torch" not in source and "load_run(" not in source
    assert "brentq" not in source and "solve_root" not in source
    with pytest.raises(ValueError, match="模型与物理缓存"):
        report.public_sha(tmp_path / "unopened.pt")
    fixture_run(tmp_path)
    report.report_run(tmp_path, expected_steps=6, make_plots=False)
    with pytest.raises(FileExistsError):
        report.report_run(tmp_path, expected_steps=6, make_plots=False)


def test_complete_chinese_plot_render(tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try:
        font = select_chinese_font()
    except RuntimeError:
        pytest.skip("No installed Chinese font; tests never download fonts")
    fixture_run(tmp_path)
    result = report.report_run(tmp_path, expected_steps=6, font=font)
    for name in report.FIGURES:
        assert (tmp_path / "report" / name).stat().st_size > 1000
    assert result["status"] == "complete"
