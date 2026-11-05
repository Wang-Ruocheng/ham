"""No SSH/GPU jobs are launched: runner contract and admission are mocked."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_gaussian_scalar_qcr as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "new_experiment"
    runner.prepare(root)
    return root


def profile_inputs():
    summary = dict(actual_steps=20,
        timing=dict(training_seconds=14., evaluation_seconds=4., wall_seconds=38.))
    history = {"history": [dict(step=1, elapsed_seconds=10.), dict(step=20, elapsed_seconds=13.8)]}
    return summary, history


def complete_summary(steps):
    return dict(actual_steps=steps, device={"uuid": runner.GPU_UUID},
        optimizer_steps={name: {"min": steps, "max": steps} for name in ("Q", "C", "R")},
        labels_used_in_training=False, frozen_evaluation_verified=True, peak_cuda_bytes=1234)


def test_prepare_freezes_exact_independent_experiment(prepared):
    plan = runner.read(prepared / "plan.json")
    assert plan["device"] == "cuda:1"
    assert plan["gpu_uuid"] == runner.GPU_UUID
    assert plan["physical"]["options"]["u_values"] == [0.]
    assert "iid" in plan["physical"]["sample_semantics"]
    assert len(plan["physical"]["options"]["r_values"]) == 12
    assert plan["training"]["lattice_sizes"] == [36, 12, 4]
    assert plan["training"]["closure_objective"] == "all_start_anchor"
    assert plan["training"]["closure_scale_weights"] is None
    assert not plan["training"]["profile_checkpoint_used_for_formal"]
    assert runner.read(prepared / "job.json")["status"] == "prepared"
    assert runner.source_hashes(plan["code_root"]) == plan["source_sha256"]


def test_existing_outputs_are_never_overwritten(prepared):
    before = (prepared / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare(prepared)
    assert (prepared / "plan.json").read_bytes() == before


def test_prepare_rejects_wrong_host_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "other-host")
    target = tmp_path / "never_created"
    with pytest.raises(ValueError, match="physical-device"):
        runner.prepare(target)
    assert not target.exists()


def test_prepare_rejects_gpu_index_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.prepare(tmp_path / "masked")


def test_source_changes_are_rejected(tmp_path):
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "scripts" / "test.py"
    source.write_text("version = 1\n")
    plan = {"code_root": str(tmp_path), "source_sha256": runner.source_hashes(tmp_path)}
    runner.unchanged(plan)
    source.write_text("version = 2\n")
    with pytest.raises(RuntimeError, match="Frozen source changed"):
        runner.unchanged(plan)


def test_source_inventory_additions_are_rejected(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "first.py").write_text("version = 1\n")
    plan = {"code_root": str(tmp_path), "source_sha256": runner.source_hashes(tmp_path)}
    (tmp_path / "scripts" / "second.py").write_text("version = 1\n")
    with pytest.raises(RuntimeError, match="inventory"):
        runner.unchanged(plan)


def test_profile_excludes_initialization_from_steady_projection():
    summary, history = profile_inputs()
    receipt = runner.profile_projection(summary, history, profile_wall_seconds=40., gpu_elapsed=40.)
    assert receipt["profile_initialization_seconds"] == 20.
    assert receipt["profile_process_startup_seconds"] == 2.
    assert receipt["profile_first_update_seconds"] == 10.
    assert receipt["profile_steady_seconds_per_update"] == pytest.approx(.2)
    # 20 seconds initialization is counted once, not extrapolated 150 times.
    assert receipt["projected_total_gpu_seconds_conservative"] == pytest.approx(
        40 + 1.5 * (20 + 2 + 10 + 2999 * .2 + 4) + 300)
    assert receipt["budget_passed"]
    assert receipt["formal_fixed_steps"] == 3000
    assert receipt["profile_is_not_formal_result"]


def test_slow_profile_holds_budget_without_reducing_steps():
    summary, history = profile_inputs()
    history["history"][-1]["elapsed_seconds"] = 29.
    summary["timing"]["training_seconds"] = 30.
    result = runner.profile_projection(summary, history, 50., 50.)
    assert not result["budget_passed"]
    assert result["formal_fixed_steps"] == 3000
    assert result["profile_steady_seconds_per_update"] == 1.


@pytest.mark.parametrize("change", ["missing_endpoint", "nan", "non_increasing"])
def test_invalid_profile_rejected(change):
    summary, history = profile_inputs()
    if change == "missing_endpoint":
        history["history"][-1]["step"] = 19
    elif change == "nan":
        summary["timing"]["evaluation_seconds"] = float("nan")
    else:
        history["history"][-1]["elapsed_seconds"] = 9.
    with pytest.raises(RuntimeError):
        runner.profile_projection(summary, history, 40., 40.)


def test_profile_and_formal_use_distinct_seeds_and_outputs(tmp_path):
    profile = runner.training_command(tmp_path, profile=True)
    formal = runner.training_command(tmp_path, profile=False, remaining_gpu_seconds=1700.)
    assert profile[profile.index("--seed") + 1] == "191"
    assert formal[formal.index("--seed") + 1] == "91"
    assert formal[formal.index("--steps") + 1] == "3000"
    assert formal[formal.index("--max-training-seconds") + 1] == "1700.0"
    assert "--source" not in formal
    assert "--resume" not in formal
    assert formal[formal.index("--expected-gpu-uuid") + 1] == runner.GPU_UUID
    assert profile[profile.index("--output") + 1] != formal[formal.index("--output") + 1]


def test_pipeline_has_no_hmc_or_sampling_preflight(tmp_path):
    stages = runner.planned_commands(tmp_path)
    assert [name for name, _, _ in stages] == ["exact_data", "training_preflight", "training", "report", "gaussian_calibration_report"]
    assert [name for name, _, gpu in stages if gpu] == ["training_preflight", "training"]
    assert "scripts.prepare_gaussian_scalar" in stages[0][1]
    assert "--execute" in stages[0][1]
    assert "scripts.report_gaussian_scalar_qcr" in stages[-1][1]


def test_gpu_admission_requires_two_empty_checks_ten_seconds_apart(prepared, monkeypatch):
    ready = dict(gpu_uuid=runner.GPU_UUID, host_ready=True, gpu_ready=True)
    occupied = dict(gpu_uuid=runner.GPU_UUID, host_ready=True, gpu_ready=False)
    snapshots = iter([ready, occupied, ready, ready])
    sleeps = []
    monkeypatch.setattr(runner, "resource_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(runner.time, "sleep", lambda duration: sleeps.append(duration))
    job = runner.read(prepared / "job.json")
    runner.admission(prepared, job, "training", True)
    assert sleeps == [10, 30, 10]
    assert len(job["resource_checks"]) == 4


def test_wrong_physical_uuid_is_not_auto_migrated(prepared, monkeypatch):
    monkeypatch.setattr(runner, "resource_snapshot", lambda: dict(gpu_uuid="GPU-other"))
    with pytest.raises(RuntimeError, match="UUID"):
        runner.admission(prepared, runner.read(prepared / "job.json"), "training", True)


def test_training_summary_endpoint_freeze_and_device_checks():
    summary = complete_summary(3000)
    runner.validate_training_summary(summary, profile=False)
    for key, value in (("actual_steps", 20), ("labels_used_in_training", True),
                       ("frozen_evaluation_verified", False), ("peak_cuda_bytes", runner.GPU_PEAK_BYTES + 1)):
        bad = copy.deepcopy(summary)
        bad[key] = value
        with pytest.raises(RuntimeError):
            runner.validate_training_summary(bad, profile=False)
    bad = copy.deepcopy(summary)
    bad["device"]["uuid"] = "GPU-wrong"
    with pytest.raises(RuntimeError, match="UUID"):
        runner.validate_training_summary(bad, profile=False)
    bad = copy.deepcopy(summary)
    bad["optimizer_steps"]["Q"]["min"] = 2999
    with pytest.raises(RuntimeError, match="optimizer"):
        runner.validate_training_summary(bad, profile=False)


def test_existing_execution_receipt_prevents_second_parent(prepared):
    runner.save(prepared / "execution.json", {"parent_pid": 123}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.execute(prepared)
    assert runner.read(prepared / "job.json")["status"] == "prepared"


def test_launch_is_exclusive_and_does_not_add_second_nice(prepared, monkeypatch):
    popen = Mock(return_value=Mock(pid=4321))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    receipt = runner.launch(prepared)
    assert receipt["parent_pid"] == 4321
    assert popen.call_count == 1
    command = popen.call_args.args[0]
    assert "nice" not in command
    with pytest.raises(FileExistsError):
        runner.launch(prepared)
    assert popen.call_count == 1


def test_public_whitelist_excludes_tensors_checkpoints_and_profile(prepared):
    for folder in ("run", "report", "calibration_report", "profile_run", "data/conditions/c000"):
        (prepared / folder).mkdir(parents=True, exist_ok=True)
    for name in ("run/summary.json", "run/evaluation_arrays.npz", "report/flow.png",
                 "calibration_report/metrics.json", "run/training_step_3000.pt",
                 "profile_run/summary.json", "data/conditions/c000/data.pt",
                 "data/conditions/c000/sidecar.json"):
        (prepared / name).write_bytes(b"public or forbidden fixture")
    values = runner.final_public_files(prepared)
    assert "run/summary.json" in values
    assert "data/conditions/c000/sidecar.json" in values
    assert "calibration_report/metrics.json" in values
    assert not any(name.endswith(".pt") or name.startswith("profile_run/") for name in values)
    assert "job.json" not in values  # avoid recursive hash/self-reference


def test_hold_is_not_training_failure_or_restart():
    job = {"stages": [{"name": "training_preflight", "returncode": 0}]}
    runner.hold(job, "projection failed")
    assert job["status"] == "budget_hold"
    assert job["workflow_complete"] is False
    assert job["requires_user_action"]
    assert len(job["stages"]) == 1


def install_fake_execution(prepared, monkeypatch, *, failing_stage=None):
    """Exercise state accounting using children that only write tiny JSON fixtures."""
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    original_projection = runner.profile_projection
    # Profile wall time is mocked because fixtures finish faster than a GPU timer.
    monkeypatch.setattr(runner, "profile_projection", lambda s, h, w, g:
                        original_projection(s, h, 40., 40.))
    launched = []

    def fake_popen(command, **kwargs):
        module = command[command.index("-m") + 1]
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        if module == "scripts.prepare_gaussian_scalar":
            name = "exact_data"
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True,
                        returncode=0, cpu_seconds=.1))
            runner.save(output / "plan.json", runner.read(prepared / "plan.json")["physical"])
            runner.save(output / "manifest.json", {})
        elif module == "experiments.phi4_qcr":
            profile = command[command.index("--steps") + 1] == "20"
            name = "training_preflight" if profile else "training"
            summary = complete_summary(20 if profile else 3000)
            summary["timing"] = profile_inputs()[0]["timing"]
            runner.save(output / "summary.json", summary)
            runner.save(output / "history_progress.json", profile_inputs()[1])
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0))
        else:
            name = "gaussian_calibration_report" if "gaussian_scalar" in module else "report"
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0))
        if name == failing_stage:
            runner.save(output / "status.json", dict(status="failed", workflow_complete=False, returncode=1))
        launched.append(name)
        return Mock(pid=100 + len(launched), wait=lambda: 1 if name == failing_stage else 0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    return launched


def test_complete_execute_has_five_zero_returncodes_and_public_receipt(prepared, monkeypatch):
    launched = install_fake_execution(prepared, monkeypatch)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "complete"
    assert job["workflow_complete"] is True
    assert len(launched) == 5
    assert all(stage["returncode"] == 0 for stage in job["stages"])
    assert job["actual_nice"] == 10
    runner.os.nice.assert_not_called()
    assert "execution.json" in job["public_files"]
    assert runner.read(prepared / "execution.json")["status"] == "complete"


def test_failed_child_is_preserved_without_retry_or_further_launch(prepared, monkeypatch):
    launched = install_fake_execution(prepared, monkeypatch, failing_stage="training")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "failed"
    assert job["requires_user_action"]
    assert launched == ["exact_data", "training_preflight", "training"]
    assert (prepared / "data" / "manifest.json").is_file()
    assert job["stages"][-1]["returncode"] == 1
    assert not (prepared / "report").exists()


def test_nice_is_lowered_only_once_to_target(prepared, monkeypatch):
    install_fake_execution(prepared, monkeypatch)
    priorities = iter([0, 10])
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: next(priorities))
    runner.execute(prepared)
    runner.os.nice.assert_called_once_with(10)
