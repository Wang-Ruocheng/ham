import copy
import json
from pathlib import Path

import pytest

from scripts.run_spin_glass3d import build_plan, stages, run, GPU_UUID


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/spin_glass3d_b2_overlap_g1_seed91.json").read_text())


def test_formal_plan_and_commands_are_fixed_unsupervised():
    cfg = config()
    plan = build_plan(cfg)
    assert plan["scales"] == [[16] * 3, [8] * 3, [4] * 3]
    assert plan["physical"]["disorders"] == 9
    assert plan["physical"]["replicas"] == 4
    assert plan["physical"]["samples"] == 512
    assert plan["sampling"]["sweeps_per_temperature_per_replica"] == 16384
    assert plan["sampling"]["single_spin_visits"] == 48_318_382_080
    assert plan["fixed_training_updates"] == 6000
    assert plan["labels_used_in_training"] is False
    assert plan["evaluation"]["thermal_nu"] is None
    assert plan["resources"]["gpu_uuid"] == GPU_UUID
    commands = stages(Path("/new"), plan, "cuda:0")
    assert len(commands) == 6
    for _, command, _ in commands:
        assert "--resume-checkpoint" not in command
        assert "--allow-nondefault-recipe" not in command
        assert "--max-training-seconds" not in command
        assert "cuda:2" not in command
    assert "--epochs" not in commands[4][1]
    assert commands[3][1][commands[3][1].index("--epochs") + 1] == "5"


def test_smoke_has_no_cuda_only_cache_argument():
    cfg = config()
    cfg["training"].update(epochs=2, max_rg_steps=1)
    cfg["evaluation"].update(max_rg_steps=1)
    plan = build_plan(cfg, smoke=True)
    commands = stages(Path("/new"), plan, "cpu")
    assert len(commands) == 5
    assert plan["scientific_result"] is False
    assert "--mi-cache-dir" not in commands[3][1]


def test_nonuniform_and_wrong_endpoint_are_rejected():
    cfg = config()
    changed = copy.deepcopy(cfg)
    changed["loss"]["closure_scale_weights"] = [.2, 1.8]
    with pytest.raises(ValueError):
        build_plan(changed)
    changed = copy.deepcopy(cfg)
    changed["training"]["epochs"] = 100
    with pytest.raises(ValueError):
        build_plan(changed)


def test_existing_run_cannot_be_restarted_or_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path, device="cpu", smoke=True)


def test_formal_cpu_run_is_not_allowed(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / "new", device="cpu")
