"""The new-data run preserves the QCR recipe and omits removed time caps."""
import copy
import json
from pathlib import Path

import pytest

from scripts import run_spin_glass2d_houdayer as runner


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / runner.CONFIG_PATH).read_text())


def test_new_config_matches_old_model_loss_and_training_recipe():
    old = json.loads((ROOT / "configs/spin_glass2d_overlap_g1_seed91.json").read_text())
    current = config()
    for section in ("model", "loss", "training", "evaluation"):
        assert current[section] == old[section]
    assert "four jointly Houdayer" in current["data"]["physical_model"]
    assert current["schema_version"] == 5


def test_plan_fixed_work_no_administrative_time_memory_caps():
    plan = runner.build_plan(config(), ROOT / runner.REFERENCE)
    physical = plan["physical"]
    assert physical["replicas"] == 4
    assert physical["samples"] == 512
    assert physical["burn_in"] == 8192
    assert physical["shape"] == [36, 36]
    assert plan["physical_preflight"]["total_single_spin_visits"] == 15_288_238_080
    assert all(plan["administrative_limits"][name] is None for name in (
        "cpu_seconds", "gpu_seconds", "training_seconds", "memory_bytes", "output_bytes"))
    assert plan["training"]["epochs"] == 6000
    assert len(plan["expected_bonds"]) == 9
    assert not plan["changes_from_old_sampling"]["strict_single_variable_ablation"]


def test_stage_commands_use_new_sampler_adapter_and_no_time_cutoff(tmp_path):
    plan = runner.build_plan(config(), ROOT / runner.REFERENCE)
    stages = runner.stages(tmp_path, "cuda:2", plan)
    assert [stage[0] for stage in stages] == ["physical_sampling", "prepare_overlap", "train", "report"]
    assert "scripts.generate_spin_glass_houdayer" in stages[0][1]
    assert "scripts.prepare_spin_glass_houdayer_qcr" in stages[1][1]
    train = stages[2][1]
    assert "experiments.rsmi_separate_qcr" in train
    assert "--max-training-seconds" not in train
    assert all("--allow-" not in arg for arg in train)
    assert "cuda:2" in train
    assert stages[0][2] is False and stages[2][2] is True


def test_smoke_plan_is_not_scientific(tmp_path):
    current = config()
    current["training"].update(max_rg_steps=1, epochs=2, system_batch=4, sample_batch=8)
    current["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    plan = runner.build_plan(current, ROOT / runner.REFERENCE, smoke=True)
    assert plan["execution_smoke"] and not plan["scientific_result"]
    assert plan["physical"]["shape"] == [12, 12]
    assert plan["expected_bonds"] == []
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_existing_root_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "keep"
    sentinel.write_text("user data")
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, ROOT / runner.REFERENCE, device="cpu", smoke=True)
    assert sentinel.read_text() == "user data"


def test_cpu_formal_not_permitted(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        runner.run(tmp_path / "new", ROOT / runner.REFERENCE, device="cpu")
    assert not (tmp_path / "new").exists()


def test_changed_frozen_source_fails(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    fingerprints = {"source.py": runner.sha(path)}
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen source"):
        runner.verify_code(tmp_path, fingerprints)
