"""No remote launch, production sampling or GPU allocation in these checks."""
import json

import pytest

from scripts import run_phi4_qcr_large as runner


def option(command, name):
    return command[command.index(name) + 1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "new_large"
    plan = runner.prepare(root)
    return root, plan


def test_large_plan_preserves_recipe_and_all_work_counts(prepared):
    root, plan = prepared
    physical, training = plan["physical"], plan["training"]
    assert training["lattice_sizes"] == [108, 36, 12]
    assert training["steps"] == 3000 and training["theory_dim"] == 1
    assert training["closure_scale_weights"] == [1, 1]
    assert training["labels_used"] is False and training["warm_start"] is False
    assert physical["total_hmc_trajectories"] == 90112
    assert physical["total_force_site_evaluations"] == 26_276_659_200
    assert physical["options"]["step_size"] == .04
    assert physical["options"]["leapfrog_steps"] == 24
    assert physical["options"]["samples"] == 128 and physical["options"]["chains"] == 4
    assert physical["options"]["max_force_site_evaluations"] == 27_000_000_000
    assert plan["comparison_changes"]["trajectory_length_to"] == pytest.approx(.96)
    assert plan["resource_policy"]["gpu_seconds"] == 7200
    assert plan["resource_policy"]["cpu_seconds"] == 14400
    assert runner.load_plan(root)[0] == json.loads(json.dumps(plan))


def test_all_physical_options_and_integrator_are_passed_explicitly(prepared):
    root, plan = prepared
    command = runner.physical_command(root, plan)
    for name in plan["physical"]["options"]:
        assert "--" + name.replace("_", "-") in command
    assert option(command, "--sizes") == "108"
    assert option(command, "--step-size") == "0.04"
    assert option(command, "--leapfrog-steps") == "24"
    assert option(command, "--max-force-site-evaluations") == "27000000000"
    assert option(command, "--burn-in") == "1024" and option(command, "--spacing") == "8"


def test_fresh_profile_and_formal_commands_fixed_endpoint_gpu3(prepared):
    root, plan = prepared
    formal = runner.training_command(root, plan, profile=False, max_seconds=6543.2)
    pilot = runner.training_command(root, plan, profile=True)
    assert option(formal, "--steps") == "3000" and option(pilot, "--steps") == "20"
    assert option(formal, "--seed") == "91" and option(pilot, "--seed") == "191"
    assert option(formal, "--max-training-seconds") == "6543.2"
    assert option(formal, "--device") == "cuda:3"
    assert option(formal, "--expected-gpu-uuid") == runner.GPU_UUID
    assert option(formal, "--output") != option(pilot, "--output")
    assert option(formal, "--system-batch") == "8" and option(formal, "--sample-batch") == "64"
    assert "--resume" not in formal and "--allow-legacy-training" not in formal


def test_plan_tampering_is_not_accepted(prepared):
    root, plan = prepared
    plan["training"]["steps"] = 10
    runner.save(root / "plan.json", plan)
    with pytest.raises(RuntimeError, match="plan fingerprint"):
        runner.load_plan(root)


def test_source_tampering_is_not_accepted(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("# initial\n")
    plan = dict(code_root=str(tmp_path), source_sha256={"code.py": runner.sha(path)})
    runner.unchanged(plan)
    path.write_text("# changed\n")
    with pytest.raises(RuntimeError, match="Frozen source"):
        runner.unchanged(plan)


def test_existing_output_and_wrong_host_are_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "unapproved-host")
    with pytest.raises(ValueError, match="thinklab"):
        runner.prepare(tmp_path / "new")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path)


@pytest.mark.parametrize("kwargs", [{"size": 107}, {"size": 18}, {"cpu_seconds": 0},
    {"gpu_seconds": float("nan")}, {"cpu_seconds": 100, "gpu_seconds": 200},
    {"force_site_cap": 2_000_000_000}, {"directory_limit": 1000}])
def test_invalid_geometry_and_insufficient_caps_fail_before_create(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()


def test_gpu_needs_two_ready_checks_at_least_ten_seconds_apart(prepared, monkeypatch):
    root, plan = prepared
    values = iter([dict(gpu_uuid=runner.GPU_UUID, gpu_ready=False, host_ready=True),
                   dict(gpu_uuid=runner.GPU_UUID, gpu_ready=True, host_ready=True),
                   dict(gpu_uuid=runner.GPU_UUID, gpu_ready=True, host_ready=True)])
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: next(values))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    job = {}
    runner.admission(root, plan, job, "training", gpu=True)
    assert sleeps == [30, 10]
    assert len(job["resource_checks"]) == 3


def test_wrong_gpu_uuid_is_not_remapped(prepared, monkeypatch):
    root, plan = prepared
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: dict(gpu_uuid="GPU-other"))
    with pytest.raises(RuntimeError, match="UUID"):
        runner.admission(root, plan, {}, "training", gpu=True)


def test_warm_and_both_cold_states_acceptance_gate_without_adaptation(prepared, monkeypatch):
    root, plan = prepared
    # The parent deliberately hides CUDA from this CPU-only child.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.load_plan(root)
    calls = []
    def fake_sample(r, u, **kwargs):
        calls.append((r, u, kwargs))
        return dict(hmc_trajectories=1088, cpu_seconds=.5, wall_seconds=1,
            burn_in_acceptance=.8, production_acceptance=0 if kwargs["initial"] == "cold_positive" else .8)
    monkeypatch.setattr("recurrent_rg.data.phi4.sample_hmc_chain", fake_sample)
    receipt = runner.sampler_benchmark(root)
    assert len(calls) == 9
    assert {kwargs["initial"] for _, _, kwargs in calls} == {"hot", "cold_positive", "cold_negative"}
    assert all(kwargs["step_size"] == .04 and kwargs["leapfrog_steps"] == 24 for _, _, kwargs in calls)
    assert all(kwargs["burn_in"] == 1024 and kwargs["samples"] == 8 and kwargs["spacing"] == 8
               for _, _, kwargs in calls)
    assert len({kwargs["seed"] for _, _, kwargs in calls}) == 9
    assert receipt["acceptance_passed"] is False and receipt["budget_passed"] is True
    assert receipt["physical_production_started"] is False
    assert receipt["sampler_parameters_changed_automatically"] is False
    assert not (root / "data").exists()
    assert json.loads((root / "sampler_preflight.json").read_text())["rows"] == receipt["rows"]


def test_cpu_only_child_may_not_use_a_nonempty_cuda_mask(prepared, monkeypatch):
    root, plan = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.load_plan(root, cpu_only=True)


@pytest.mark.parametrize("acceptance_passed,budget_passed,expected_status", [
    (False, True, "sampling_hold"), (True, False, "budget_hold")])
def test_parent_never_starts_production_after_preflight_hold(
        prepared, monkeypatch, acceptance_passed, budget_passed, expected_status):
    root, plan = prepared
    calls = []
    class Child:
        pid = 1234
        def __init__(self, command, **kwargs):
            calls.append(command)
            assert "--sampler-benchmark" in command
            runner.save(root / "sampler_preflight.json", dict(
                acceptance_passed=acceptance_passed, budget_passed=budget_passed), exclusive=True)
        def wait(self):
            return 0
    monkeypatch.setattr(runner.os, "nice", lambda value: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    runner.execute(root)
    job = runner.read(root / "job.json")
    assert job["status"] == expected_status
    assert job["requires_user_action"] is True and job["workflow_complete"] is False
    assert len(calls) == 1 and not (root / "data").exists()
    with pytest.raises(ValueError, match="no restart"):
        runner.execute(root)


def test_settings_cannot_be_overridden_at_execution(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--execute", "--output", str(tmp_path), "--size", "108"])


def test_new_runner_does_not_modify_original_defaults():
    from scripts import run_phi4_qcr as original
    from recurrent_rg.data.phi4 import Phi4Options
    assert original.GPU == "cuda:2" and original.GPU_SECONDS == 1800
    assert Phi4Options().sizes == (36,) and Phi4Options().step_size == .08
    assert Phi4Options().leapfrog_steps == 12
