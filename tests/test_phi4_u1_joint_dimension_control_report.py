import json

import numpy as np
import pytest

from scripts import report_phi4_u1_joint_dimension_control as report


def fixture(tmp_path, different=False):
    root = tmp_path/"joint"
    root.mkdir()
    code = root/"frozen_source.py"
    code.write_text("# Synthetic frozen source fingerprint only.\n")
    rng = np.random.default_rng(44)
    shared = {}
    for split, ids in (("train", np.repeat(np.arange(8), 4)), ("test", np.repeat(np.arange(8, 13), 4))):
        shared[f"{split}_cell_ids"] = ids
        shared[f"{split}_labels"] = np.column_stack((-.5+.01*ids, np.ones(len(ids)), np.zeros(len(ids))))
        shared[f"{split}_feature_target"] = rng.dirichlet(np.ones(105), (len(ids), 3))
    for dim in (1, 2):
        folder = root/f"dg{dim}"
        folder.mkdir()
        arrays = {key: value.copy() for key, value in shared.items()}
        if different and dim == 2:
            arrays["test_feature_target"][0, 1, 0] += 1e-8
        for split in ("train", "test"):
            target = arrays[f"{split}_feature_target"]
            count = len(target)
            arrays[f"{split}_direct"] = rng.normal(0., .03, (count, 3, dim))
            arrays[f"{split}_feature_reconstruction"] = target+.001
            arrays[f"{split}_one_step_feature_prediction"] = target[:, 1:]+.002/dim
            arrays[f"{split}_rollout_feature_prediction"] = target+.002/dim
        train = arrays["train_direct"][:, :2].reshape(-1, dim)
        lower, upper = train.min(0), train.max(0)
        padding = .1*np.maximum(upper-lower, 1e-6)
        if dim == 1:
            values = np.concatenate([arrays[f"{split}_direct"][:, :2].reshape(-1) for split in ("train", "test")])
            pad = .1*max(np.ptp(values), 1e-6)
            grid = np.linspace(values.min()-pad, values.max()+pad, 256)[:, None]
        else:
            grid = np.stack(np.meshgrid(*[np.linspace(a, b, 33) for a, b in zip(lower-padding, upper+padding)],
                                       indexing="xy"), -1).reshape(-1, 2)
        arrays.update(flow_g=grid, flow_R=grid+.001, flow_displacement=np.full_like(grid, .001))
        initial, final = "a"*64, "b"*64
        initialization = dict(constructor_module_sha256=dict(C=initial), training_start_module_sha256=dict(C=initial))
        manifest = dict(options=dict(steps=3000, seed=91), theory_dim=dim,
            lattice_sizes=[36, 12, 4], block_size=3, stride=3, q_initialization="train_mean_bias",
            q_initialization_receipt=initialization, random_construction_module_sha256=dict(C=initial),
            training_start_module_sha256=dict(C=initial), closure_objective="all_start_anchor",
            closure_scale_weights=None, labels_used_in_training=False,
            source_sha256={str(code): report.sha256(code)}, data_manifest_sha256="d"*64,
            data_shards=[dict(path="unchanged")], evaluation_seed=123)
        summary = dict(actual_steps=3000, theory_dim=dim, frozen_evaluation_verified=True,
            labels_used_in_training=False, final_module_sha256=dict(C=final), numerical_fixed_points=[],
            data_manifest_sha256="d"*64,
            optimizer_steps={name: dict(min=3000, max=3000, parameters=2) for name in ("Q", "C", "R")})
        history = dict(latest_step=3000, history=[dict(step=step, q_loss=.01, c_loss=-.2, r_loss=.03)
                                                for step in (1, 100, 3000)])
        for name, value in (("manifest.json", manifest), ("summary.json", summary), ("history_progress.json", history)):
            report.write_json(folder/name, value)
        np.savez(folder/"evaluation_arrays.npz", **arrays)
        report.write_json(folder/"status.json", dict(status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, actual_steps=3000,
            summary_sha256=report.sha256(folder/"summary.json"),
            evaluation_arrays_sha256=report.sha256(folder/"evaluation_arrays.npz")))
    return root


def test_feature_three_paths_mean_baseline_and_coordinate_RMS(tmp_path):
    root = fixture(tmp_path)
    run = report.load_run(root/"dg1", 1, {})
    metrics, normalizers = report.feature_metrics(run["arrays"])
    assert [row["path"] for row in metrics["test"]] == ["0_to_1", "1_to_2", "0_to_2"]
    assert all(row["prediction"]["rmse"] == pytest.approx(.002) for row in metrics["test"])
    assert all(row["target_Q_reconstruction"]["rmse"] == pytest.approx(.001) for row in metrics["test"])
    assert all(row["TRAIN_mean"]["nrmse"] == pytest.approx(1.) for row in metrics["train"])
    assert len(normalizers) == 3


def test_task_mismatch_is_reported_without_dimension_causal_claim(tmp_path):
    root = fixture(tmp_path, different=True)
    runs = {dim: report.load_run(root/f"dg{dim}", dim, {}) for dim in (1, 2)}
    comparison = report.task_comparison(runs)
    assert comparison["C_training_start_SHA_equal"] and comparison["C_final_SHA_equal"]
    assert comparison["feature_targets"]["test"]["max_abs_difference"] == pytest.approx(1e-8)
    assert not comparison["identical_coarse_graining_tasks"]
    assert not comparison["directly_comparable_common_target_errors"]
    assert not comparison["Q_initialization_strictly_paired"]


def test_status_SHA_and_parameter_clock_gates(tmp_path):
    root = fixture(tmp_path)
    folder = root/"dg1"
    summary = json.loads((folder/"summary.json").read_text())
    summary["optimizer_steps"]["C"]["max"] = 2999
    report.write_json(folder/"summary.json", summary)
    with pytest.raises(ValueError, match="端点"):
        report.load_run(folder, 1, {})
    status = json.loads((folder/"status.json").read_text())
    status["summary_sha256"] = report.sha256(folder/"summary.json")
    report.write_json(folder/"status.json", status)
    with pytest.raises(ValueError, match="参数时钟"):
        report.load_run(folder, 1, {})


def test_complete_stored_array_report_four_chinese_figures_and_inventory(tmp_path):
    root = fixture(tmp_path)
    output = root/"report"
    result = report.run(root, output)
    assert result["status"] == "complete"
    assert result["task_comparison"]["identical_coarse_graining_tasks"]
    assert not result["Q_C_R_inference_performed"] and not result["checkpoint_loaded"]
    assert not result["new_root_search"] and result["thermal_nu"] is None
    assert "TRAIN和TEST" in result["original_flow_bounds_scope"]["1"]
    assert all((output/name).stat().st_size > 1000 for name in
               ("feature_comparison.png", "flow.png", "coupling_by_scale.png", "loss.png"))
    inventory = json.loads((output/"public_files.json").read_text())
    assert inventory["status.json"] == report.sha256(output/"status.json")
    with pytest.raises(FileExistsError):
        report.run(root, output)
