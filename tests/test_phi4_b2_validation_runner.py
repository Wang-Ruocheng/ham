"""Mocked orchestration tests: no sampling, model loading, training or GPU calls."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_b2_validation as runner

ORIGINAL_UNCHANGED = runner.unchanged


@pytest.fixture
def completed_source(tmp_path, monkeypatch):
    source, old_code = tmp_path / "completed_source", tmp_path / "old_code"
    source.mkdir()
    (old_code / "recurrent_rg").mkdir(parents=True)
    (old_code / "recurrent_rg/old.py").write_text("# immutable fixture\n")
    original = dict(code_root=str(old_code), source_sha256=runner.source_hashes(old_code))
    runner.save(source / "plan.json", original)
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr(runner, "SOURCE_PLAN_SHA", runner.sha(source / "plan.json"))
    (source / "training/all_start").mkdir(parents=True)
    (source / "training/all_start/training_step_3000.pt").write_bytes(b"not opened by runner")
    shards = []
    for index in range(12):
        folder = source / "data/conditions" / f"c{index:03d}"
        folder.mkdir(parents=True)
        (folder / "data.npz").write_bytes(b"raw fixture " + bytes([index]))
        runner.save(folder / "sidecar.json", dict(cell=index))
        shards.append(dict(cell_index=index, path=str((folder / "data.npz").relative_to(source / "data")),
            sha256=runner.sha(folder / "data.npz"), sidecar=str((folder / "sidecar.json").relative_to(source / "data")),
            sidecar_sha256=runner.sha(folder / "sidecar.json")))
    runner.save(source / "data/manifest.json", dict(status="complete", options=dict(sizes=[32], chains=4, samples=128), shards=shards))
    runner.save(source / "data/status.json", dict(status="complete", warning_count=35))
    runner.save(source / "sampling_quality.json", dict(issues=["original warning"]))
    runner.save(source / "training/manifest.json", dict(recipe="phi4_b2_closure_objective_comparison_v1",
        lattice_sizes=[32, 16, 8, 4], labels_used_in_training=False,
        data_manifest_sha256=runner.sha(source / "data/manifest.json")))
    runner.save(source / "training/summary.json", dict(actual_steps_by_arm=dict(legacy=3000, all_start=3000), C_Q_pairing_verified=True))
    public = {str(path.relative_to(source)): runner.sha(path) for path in source.rglob("*.json")}
    runner.save(source / "job.json", dict(status="complete", workflow_complete=True, error=None,
        stages=[dict(returncode=0) for _ in range(5)], public_files=public))
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return source


@pytest.fixture
def prepared(completed_source, tmp_path, monkeypatch):
    root = tmp_path / "new_validation"
    runner.prepare(root, source=completed_source)
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    return root


def test_prepare_pins_read_only_dataset_and_single_factor_contract(prepared, completed_source):
    plan = runner.read(prepared / "plan.json")
    assert plan["recipe"] == runner.RECIPE
    assert plan["source"] == str(completed_source)
    assert not (prepared / "data").exists()
    assert plan["physical_data"]["warning_count"] == 35
    assert plan["physical_data"]["independent_physical_chains"] == 48
    assert plan["physical_data"]["resampled"] is False
    assert plan["controls"]["joint_arms"] == ["cubic_decoder", "odd_C"]
    assert plan["controls"]["joint_steps_per_arm"] == 3000
    assert plan["controls"]["r_tail_source_steps"] == 3000
    assert plan["controls"]["fixed_encoding_cache_used"] is False
    assert plan["controls"]["Q_closure_feedback"] is False
    assert plan["resource_policy"]["gpu_seconds"] == 3600
    assert plan["resource_policy"]["cpu_seconds"] == 7200
    assert len([name for name in plan["immutable_inputs"] if name.endswith("data.npz")]) == 12
    assert str(completed_source / "training/all_start/training_step_3000.pt") in plan["immutable_inputs"]


@pytest.mark.parametrize("case", ["plan", "public", "raw", "checkpoint", "code"])
def test_source_input_fingerprints_cover_all_immutable_inputs(completed_source, case):
    before = runner.source_inputs(completed_source)
    targets = dict(plan=completed_source / "plan.json", public=completed_source / "sampling_quality.json",
        raw=completed_source / "data/conditions/c000/data.npz",
        checkpoint=completed_source / "training/all_start/training_step_3000.pt",
        code=Path(runner.read(completed_source / "plan.json")["code_root"]) / "recurrent_rg/old.py")
    targets[case].write_bytes(targets[case].read_bytes() + b" ")
    if case == "checkpoint":
        assert runner.source_inputs(completed_source) != before
    else:
        with pytest.raises(RuntimeError):
            runner.source_inputs(completed_source)


def test_source_checks_reject_incomplete_workflow(completed_source):
    job = runner.read(completed_source / "job.json")
    job["stages"][2]["returncode"] = 1
    runner.save(completed_source / "job.json", job)
    with pytest.raises(RuntimeError, match="successfully complete"):
        runner.source_inputs(completed_source)


def test_safe_file_rejects_traversal_and_nested_symlink(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/value.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "alias").symlink_to(tmp_path / "outside", target_is_directory=True)
    for name in ("../outside/value.json", "/etc/passwd", "alias/value.json"):
        with pytest.raises(RuntimeError):
            runner.safe_file(root, name)


def test_never_overwrite_mask_or_retarget(prepared, completed_source, tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared, source=completed_source)
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / "bad", source=completed_source, gpu_index=1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="unmasked"):
        runner.prepare(tmp_path / "masked", source=completed_source)


def test_only_fixed_three_training_stages_and_read_only_report(prepared):
    plan = runner.read(prepared / "plan.json")
    stages = runner.commands(prepared, plan)
    assert [row[0] for row in stages] == ["profile", "joint", "r_tail", "report"]
    assert [row[2] for row in stages] == [True, True, True, False]
    for name, command, _ in stages[:3]:
        assert command[command.index("-m")+1] == "experiments.phi4_b2_validation"
        assert command[command.index("--stage")+1] == name
        assert command[command.index("--source")+1] == plan["source"]
        assert "--steps" not in command and "--seed" not in command
        assert command[command.index("--max-cuda-bytes")+1] == str(8*1024**3)
    assert "scripts.report_phi4_b2_validation" in stages[-1][1]
    assert not any("sampling" in part or "prepare_phi4" in part for _, command, _ in stages for part in command)


@pytest.mark.parametrize("stage", ["profile", "joint", "r_tail"])
def test_training_command_round_trips_real_cli_without_running_training(prepared, monkeypatch, stage):
    from experiments import phi4_b2_validation as trainer
    captured = []
    monkeypatch.setattr(trainer, "run", lambda options: captured.append(options) or dict(status="complete"))
    plan = runner.read(prepared / "plan.json")
    command = runner.training_command(prepared, plan, stage, remaining_cpu_seconds=6100., remaining_gpu_seconds=3100.)
    assert trainer.main(command[command.index("-m")+2:]) == 0
    options = captured[0]
    assert options.stage == stage and options.source == plan["source"]
    assert options.output == str(prepared / stage)
    assert options.device == "cuda:2" and options.expected_gpu_uuid == runner.GPU_UUID
    assert options.steps == (20 if stage == "profile" else 3000)
    assert options.seed == (191 if stage == "profile" else 91)
    assert options.cpu_budget_seconds == 6100. and options.gpu_budget_seconds == 3100.


def profile_receipt():
    return dict(status="complete", workflow_complete=True, returncode=0, profile_only=True,
        steps_per_arm=20, formal_checkpoint_reused=False, gpu_uuid=runner.GPU_UUID,
        initialization_seconds=2., joint_update_seconds=3., r_tail_update_seconds=1., peak_cuda_bytes=4096)


def test_projection_separates_initialization_and_each_update_workload():
    result = runner.profile_projection(profile_receipt(), gpu_elapsed=10.)
    assert result["projected_joint_gpu_seconds_conservative"] == pytest.approx(1.25*(2+3*150))
    assert result["projected_r_tail_gpu_seconds_conservative"] == pytest.approx(1.25*(2+150))
    assert result["projected_total_gpu_seconds_conservative"] == pytest.approx(10+1.25*(4+600)+300+80)
    assert result["budget_passed"] and not result["profile_parameters_reused_for_formal"]


@pytest.mark.parametrize("field,value", [("initialization_seconds", -1), ("joint_update_seconds", 0),
    ("r_tail_update_seconds", float("nan")), ("peak_cuda_bytes", 8*1024**3+1),
    ("steps_per_arm", 19), ("formal_checkpoint_reused", True), ("returncode", 1)])
def test_profile_rejects_invalid_evidence(field, value):
    row = profile_receipt()
    row[field] = value
    with pytest.raises(RuntimeError):
        runner.profile_projection(row, gpu_elapsed=10.)


def test_gpu_admission_requires_two_free_checks_at_least_ten_seconds_apart(prepared, monkeypatch):
    plan, job = runner.read(prepared / "plan.json"), runner.read(prepared / "job.json")
    ready = dict(gpu_uuid=runner.GPU_UUID, host_ready=True, gpu_ready=True)
    snapshot = Mock(side_effect=[dict(ready, gpu_ready=False), ready, ready])
    sleeps = []
    monkeypatch.setattr(runner, "resource_snapshot", snapshot)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner.admission(prepared, plan, job, "joint", True)
    assert snapshot.call_count == 3 and sleeps == [30, 10]


def formal_receipt(stage):
    names = ("cubic_decoder", "odd_C") if stage == "joint" else ("r_tail",)
    counts = dict(Q=3000, C=3000, R=3000) if stage == "joint" else dict(Q=0, C=0, R=3000)
    return dict(status="complete", actual_steps=3000, gpu_uuid=runner.GPU_UUID, peak_cuda_bytes=4096,
        labels_used_in_training=False, gradient_routing_verified=True,
        arms={name: dict(actual_steps=3000, optimizer_steps={key: dict(min=count,max=count,parameters=3)
            for key,count in counts.items()}) for name in names},
        Q_C_frozen_verified=True, stochastic_raw_physical_batches=True, fixed_encoding_cache_used=False,
        source_steps=3000)


@pytest.mark.parametrize("field,value", [("Q_C_frozen_verified", False), ("stochastic_raw_physical_batches", False),
    ("fixed_encoding_cache_used", True), ("source_steps", 6000), ("labels_used_in_training", True)])
def test_r_tail_must_be_raw_stochastic_and_frozen(field, value):
    row = formal_receipt("r_tail")
    row[field] = value
    with pytest.raises(RuntimeError):
        runner.validate_training_receipt(row, dict(gpu_uuid=runner.GPU_UUID), "r_tail")


def test_formal_optimizer_counts_must_match_each_arm():
    row = formal_receipt("joint")
    row["arms"]["odd_C"]["optimizer_steps"]["C"]["min"] = 2999
    with pytest.raises(RuntimeError, match="step counts"):
        runner.validate_training_receipt(row, dict(gpu_uuid=runner.GPU_UUID), "joint")


def fake_execution(root, monkeypatch, *, fail_stage=None, budget_stage=None, expensive_profile=False):
    calls = []
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    def popen(command, **kwargs):
        module = command[command.index("-m")+1]
        name = command[command.index("--stage")+1] if module.startswith("experiments") else "report"
        output = Path(command[command.index("--output")+1])
        output.mkdir()
        value = profile_receipt() if name == "profile" else formal_receipt(name) if name != "report" else {}
        if expensive_profile and name == "profile":
            value["joint_update_seconds"] = 40.
        status = "budget_hold" if name == budget_stage else "failed" if name == fail_stage else "complete"
        runner.save(output / "status.json", dict(status=status, workflow_complete=status == "complete", returncode=0 if status == "complete" else 3 if status == "budget_hold" else 1))
        if name != "report":
            runner.save(output / ("profile.json" if name == "profile" else "summary.json"), value)
        else:
            runner.save(output / "report.json", dict(status="complete"))
            (output / "evaluation_arrays.npz").write_bytes(b"statistics only fixture")
        calls.append((name, command, kwargs))
        return Mock(pid=200+len(calls), wait=lambda: 3 if name == budget_stage else 1 if name == fail_stage else 0)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return calls


def test_successful_execution_is_fresh_read_only_and_publishes_no_pt(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "complete" and job["workflow_complete"] and job["returncode"] == 0
    assert [row[0] for row in calls] == ["profile", "joint", "r_tail", "report"]
    assert [row[2]["env"].get("CUDA_VISIBLE_DEVICES") for row in calls] == [None, None, None, ""]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert "report/evaluation_arrays.npz" in job["public_files"]
    assert not any(name.endswith(".pt") or name.startswith("data/") for name in job["public_files"])
    with pytest.raises(RuntimeError, match="Already executed"):
        runner.execute(prepared)


def test_profile_budget_hold_prevents_both_formal_stages(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, expensive_profile=True)
    runner.execute(prepared)
    assert [row[0] for row in calls] == ["profile"]
    job = runner.read(prepared / "job.json")
    assert job["status"] == "budget_hold" and job["requires_user_action"]
    assert not (prepared / "joint").exists() and not (prepared / "r_tail").exists()


def test_training_hold_preserves_partial_no_retry(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, budget_stage="joint")
    runner.execute(prepared)
    assert [row[0] for row in calls] == ["profile", "joint"]
    assert runner.read(prepared / "job.json")["status"] == "budget_hold"


def test_training_failure_prevents_later_work(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, fail_stage="joint")
    with pytest.raises(RuntimeError, match="no automatic repair"):
        runner.execute(prepared)
    assert [row[0] for row in calls] == ["profile", "joint"]
    job = runner.read(prepared / "job.json")
    assert job["status"] == "failed" and job["requires_user_action"]
    assert runner.read(prepared / "execution.json")["status"] == "failed"


def test_public_inventory_rejects_raw_arrays_and_symlinks(prepared):
    (prepared / "joint").mkdir()
    (prepared / "joint/data.npz").write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="statistical NPZ"):
        runner.final_public_files(prepared)


def test_nice_applied_once_only(prepared, monkeypatch):
    fake_execution(prepared, monkeypatch)
    nice = Mock()
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 0)
    monkeypatch.setattr(runner.os, "nice", nice)
    runner.execute(prepared)
    nice.assert_called_once_with(10)


def test_source_change_blocks_child_launch(prepared, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner, "unchanged", Mock(side_effect=RuntimeError("Source changed")))
    with pytest.raises(RuntimeError, match="Source changed"):
        runner.execute(prepared)
    spawn.assert_not_called()


def test_immutable_code_inventory_checked_on_validation_snapshot(prepared, monkeypatch):
    plan = runner.read(prepared / "plan.json")
    monkeypatch.setattr(runner, "source_hashes", lambda code: {})
    with pytest.raises(RuntimeError, match="validation source code changed"):
        ORIGINAL_UNCHANGED(plan)
