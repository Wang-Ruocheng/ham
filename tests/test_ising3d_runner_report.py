import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_ising3d import build_plan, stages, run
from scripts.report_ising3d import closure_metrics, root_candidates


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/ising3d_b2_g1_seed91.json").read_text())


def test_formal_plan_is_explicit_and_unsupervised():
    cfg = config()
    plan = build_plan(cfg)
    assert plan["scales"] == [[16] * 3, [8] * 3, [4] * 3]
    assert plan["fixed_training_updates"] == 6000
    assert plan["physical"]["coupling_count"] == 24
    assert plan["physical"]["chains"] == 4
    assert plan["labels_used_in_training"] is False
    assert cfg["loss"]["closure_scale_weights"] is None
    for _, command, _ in stages(Path("/new"), plan, "cuda:2"):
        assert "--allow-nondefault-recipe" not in command
        assert "--max-training-seconds" not in command
        assert "--resume-checkpoint" not in command


def test_cpu_smoke_does_not_pass_cuda_cache_flag():
    cfg = config()
    cfg["training"]["max_rg_steps"] = 1
    cfg["evaluation"]["max_rg_steps"] = 1
    commands = stages(Path("/new"), build_plan(cfg, True), "cpu")
    assert "--mi-cache-dir" not in commands[3][1]
    assert "--cache-dir" in commands[2][1]


def test_new_output_only(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path, device="cpu", smoke=True)


def test_metrics_use_train_target_scale_and_mean_C_values():
    train = np.array([[[0.], [1.], [2.]], [[2.], [3.], [6.]]])
    arrays = {}
    for split in ("train", "test"):
        arrays[f"direct_{split}"] = train.copy()
        arrays[f"R_g_{split}"] = np.concatenate((train[:, 1:] + 1, train[:, -1:]), axis=1)
        arrays[f"rollout_{split}"] = train.copy()
    metrics = closure_metrics(arrays)
    assert metrics["TRAIN_target_std_by_scale"] == [1., 2.]
    assert metrics["test"]["one_step_RMSE"] == [1., 1.]
    assert metrics["test"]["one_step_NRMSE"] == [1., .5]
    assert metrics["test"]["rollout_RMSE"] == [0., 0.]


def test_root_candidates_are_not_selected_by_theory():
    class Model:
        def advance_theory(self, x):
            return 2 * x
    grid = np.linspace(-1, 1, 101)[:, None]
    direct = np.array([[[-.5], [-.2]], [[.5], [.2]]])
    arrays = {"direct_train": direct, "direct_test": direct,
        "K_train": np.array([.1, .3]), "K_test": np.array([.1, .3])}
    result = root_candidates(Model(), grid, 2 * grid, arrays, "cpu")
    assert len(result["candidates"]) == 1
    root = result["candidates"][0]
    assert root["g"] == 0.
    assert root["R_derivative"] == 2.
    assert root["candidate_nu"] == 1.
    assert root["k0_interpolated_K"] == .2
    assert root["displacement_residual"] == 0.
    assert root["thermal_identification_certified"] is False
    assert result["thermal_nu"] is None
