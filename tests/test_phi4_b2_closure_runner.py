"""Tiny mocked orchestration checks; never perform physical sampling or GPU work."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_b2_closure_comparison as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "fresh_b2"
    runner.prepare(root, gpu_index=2, expected_gpu_uuid=runner.GPU_UUID)
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    return root


def test_new_physical_plan_and_pairing_contract(prepared):
    plan = runner.read(prepared / "plan.json")
    physical, training = plan["physical"], plan["training"]
    assert physical["options"]["sizes"] == [32]
    assert physical["options"]["r_values"] == pytest.approx([-.5 + i*.8/11 for i in range(12)])
    assert physical["options"]["u_values"] == [.2]
    assert physical["options"]["cluster_probability"] == .5
    assert physical["options"]["burn_in"] == 1024
    assert physical["options"]["spacing"] == 8
    assert physical["options"]["chains"] == 4
    assert physical["configurations"] == 6144
    assert physical["total_hmc_trajectories"] == 98304
    assert plan["total_planned_hmc_trajectories_including_preflight"] == 99264
    assert plan["total_planned_force_site_evaluations_including_preflight"] == 99264*13*1024
    assert training["lattice_sizes"] == [32, 16, 8, 4]
    assert training["arms"] == {"legacy": "one_step_and_rollout", "all_start": "all_start_anchor"}
    assert training["steps_per_arm"] == 3000
    assert not training["q_closure_feedback"]
    assert training["independent_gradients"] and training["Q_C_losses_checked_every_step"]
    assert training["Q_C_fingerprints_checked_at_progress_and_endpoint"]
    assert plan["resource_policy"]["minimum_host_available_gib"] == 36
    assert plan["sampling_quality_policy"]["allow_quality_warnings"]


def test_never_overwrite_retarget_or_mask(prepared, tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared, gpu_index=2, expected_gpu_uuid=runner.GPU_UUID)
    with pytest.raises(ValueError, match="pinned"):
        runner.prepare(tmp_path / "wrong", gpu_index=1, expected_gpu_uuid=runner.GPU_UUID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.prepare(tmp_path / "mask", gpu_index=2, expected_gpu_uuid=runner.GPU_UUID)


def test_cli_requires_explicit_physical_target(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.main(["--prepare", "--output", str(tmp_path / "unset")])
    assert error.value.code == 2


def test_sampling_command_round_trips_unchanged_pipeline(prepared, monkeypatch):
    from scripts import prepare_phi4_hybrid as cli
    from recurrent_rg.data.phi4_hybrid_pipeline import generation_plan
    captured = []
    monkeypatch.setattr(cli, "generate_dataset", lambda out, options:
        captured.append(options) or dict(schema="fixture", shards=[1], warning_count=0))
    plan = runner.read(prepared / "plan.json")
    command = runner.sampling_command(prepared, plan)
    cli.main(command[command.index("scripts.prepare_phi4_hybrid")+1:])
    assert json.loads(json.dumps(generation_plan(captured[0]))) == plan["physical"]


def test_five_stages_and_fresh_profile_formal_seeds(prepared):
    stages = runner.planned_commands(prepared, runner.read(prepared / "plan.json"))
    assert [name for name, _, _ in stages] == ["sampler_preflight", "sampling", "training_preflight", "paired_training", "report"]
    for index, steps, seed in ((2, "20", "191"), (3, "3000", "91")):
        command = stages[index][1]
        assert command[command.index("--steps")+1] == steps
        assert command[command.index("--seed")+1] == seed
        assert command[command.index("--device")+1] == "cuda:2"
        assert command[command.index("--expected-gpu-uuid")+1] == runner.GPU_UUID
        assert ("--profile-only" in command) == (index == 2)
        assert not {"--resume", "--source", "--allow-nondefault-recipe"}.intersection(command)


def pilot(*args, **kwargs):
    cycles = kwargs["burn_in"] + kwargs["samples"] * kwargs["spacing"]
    return dict(seed=kwargs["seed"], hmc_trajectories=cycles, force_evaluations=cycles*13,
        force_site_evaluations=cycles*13*kwargs["size"]**2, cpu_seconds=.1, wall_seconds=.2,
        production_acceptance=.9, burn_in_acceptance=.8, max_absolute_delta_h=.3,
        cluster_probability=.5, wolff_cluster_updates=cycles//2, total_cluster_neighbor_visits=1200)


def test_sampler_preflight_uses_only_train_endpoints_exact_settings(prepared, monkeypatch):
    import recurrent_rg.data.phi4_hybrid as sampler
    mock = Mock(side_effect=pilot)
    monkeypatch.setattr(sampler, "sample_hybrid_chain", mock)
    result = runner.sampler_preflight(prepared)
    assert mock.call_count == 6
    assert [call.args[0] for call in mock.call_args_list] == [-.5]*3 + [.3]*3
    assert [call.kwargs["initial"] for call in mock.call_args_list] == ["hot", "cold_positive", "cold_negative"]*2
    assert all(call.kwargs["size"] == 32 and call.kwargs["step_size"] == .08
        and call.kwargs["leapfrog_steps"] == 12 and call.kwargs["cluster_probability"] == .5
        for call in mock.call_args_list)
    assert result["status"] == "complete"
    assert result["actual_preflight_work"]["hmc_trajectories"] == 960
    assert result["actual_preflight_work"]["force_site_evaluations"] == 960*13*1024
    assert result["projected_production_cpu_seconds_conservative"] == pytest.approx(.1/160*98304*2)
    assert not (prepared / "data").exists()


@pytest.mark.parametrize("case", ["acceptance", "cpu", "numeric"])
def test_preflight_holds_without_adaptation(prepared, monkeypatch, case):
    import recurrent_rg.data.phi4_hybrid as sampler
    def changed(*args, **kwargs):
        if case == "numeric":
            raise FloatingPointError("Nonfinite force")
        row = pilot(*args, **kwargs)
        row["production_acceptance" if case == "acceptance" else "cpu_seconds"] = .59 if case == "acceptance" else 5.
        return row
    monkeypatch.setattr(sampler, "sample_hybrid_chain", changed)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == ("budget_hold" if case == "cpu" else "invalid_sampling_hold")
    assert result["sampler_parameters_not_adapted"]
    assert not (prepared / "data").exists()
    if case == "numeric":
        assert result["actual_preflight_work"] is None
        assert result["actual_total_work_known"] is False


def profile_receipt():
    return dict(status="complete", workflow_complete=True, profile_only=True,
        steps_per_arm=20, arms=2, seed=191, C_Q_pairing_verified=True,
        gradient_routing_verified=True, gpu_uuid=runner.GPU_UUID, update_seconds=2.,
        initialization_seconds=3., evaluation_seconds=0., peak_cuda_bytes=4096)


def test_projection_separates_initialization_and_two_arm_updates():
    result = runner.profile_projection(profile_receipt(), actual_profile_stage_seconds=10., gpu_elapsed=10.)
    assert result["projected_formal_gpu_seconds_conservative"] == pytest.approx(1.25*(3+2*150))
    assert result["projected_total_gpu_seconds_conservative"] == pytest.approx(10+1.25*303+300+40)
    assert result["budget_passed"]
    assert not result["profile_parameters_used_for_formal"]


@pytest.mark.parametrize("field,value", [("update_seconds", float("nan")), ("update_seconds", 0),
    ("initialization_seconds", -1), ("peak_cuda_bytes", runner.GPU_PEAK_BYTES+1),
    ("steps_per_arm", 19), ("profile_only", False)])
def test_projection_rejects_invalid_receipts(field, value):
    row = profile_receipt()
    row[field] = value
    with pytest.raises(RuntimeError):
        runner.profile_projection(row, actual_profile_stage_seconds=10., gpu_elapsed=10.)


def test_gpu_admission_requires_two_unoccupied_checks_ten_seconds_apart(prepared, monkeypatch):
    plan, job = runner.read(prepared / "plan.json"), runner.read(prepared / "job.json")
    ready = dict(gpu_uuid=runner.GPU_UUID, gpu_ready=True, host_ready=True)
    checks, waits = Mock(side_effect=[dict(ready, gpu_ready=False), ready, ready]), []
    monkeypatch.setattr(runner, "resource_snapshot", checks)
    monkeypatch.setattr(runner.time, "sleep", waits.append)
    runner.admission(prepared, plan, job, "paired_training", True)
    assert checks.call_count == 3 and waits == [30, 10]


def fake_data(root, plan, *, issue=None):
    from recurrent_rg.data.phi4 import OBSERVABLES
    directory = root / "data/conditions/L32_c0000"
    directory.mkdir(parents=True)
    chain = dict(chain_index=0, production_acceptance=.9, max_absolute_delta_h=.3,
        saved_magnetization_sign_changes=5, warnings=[],
        diagnostics={name: {"mean": .1, "ess": 90., "warning": None} for name in OBSERVABLES})
    if issue == "ess":
        chain["diagnostics"]["action_density"]["ess"] = 10
    if issue == "zero_acceptance":
        chain["production_acceptance"] = 0.
    sidecar = dict(cell_index=0, r=-.5, u=.2, h=0., chains=[chain],
        summary={"hot_cold_comparison": {"action_density": {"descriptive_three_se_warning": issue == "hot_cold"}}})
    runner.save(directory / "sidecar.json", sidecar)
    runner.save(root / "data/plan.json", plan["physical"])
    runner.save(root / "data/status.json", dict(status="complete", workflow_complete=True,
        returncode=0, cpu_seconds=.1, warning_count=int(issue is not None)))
    public = ["plan.json", "status.json", "manifest.json", "conditions/L32_c0000/sidecar.json"]
    runner.save(root / "data/manifest.json", dict(status="complete", warning_count=int(issue is not None),
        public_files=public, shards=[dict(cell_index=0, sidecar="conditions/L32_c0000/sidecar.json",
        sidecar_sha256=runner.sha(directory / "sidecar.json"))]))


def formal_receipt():
    return dict(status="complete", workflow_complete=True, actual_steps=3000,
        actual_steps_by_arm={"legacy": 3000, "all_start": 3000}, gpu_uuid=runner.GPU_UUID,
        peak_cuda_bytes=4096, pairing_verified=True, C_Q_pairing_verified=True,
        gradient_routing_verified=True, labels_used_in_training=False,
        closure_modes={"legacy": "one_step_and_rollout", "all_start": "all_start_anchor"})


def fake_execution(root, monkeypatch, *, issue=None, preflight="complete", fail_stage=None,
                   expensive_profile=False, pairing_failure=False):
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    plan, calls = runner.read(root / "plan.json"), []
    def popen(command, **kwargs):
        module = command[command.index("-m")+1]
        output = Path(command[command.index("--output")+1])
        if module == runner.MODULE:
            name = "sampler_preflight"
            runner.save(root / "sampler_preflight.json", dict(status=preflight,
                acceptance_passed=preflight != "invalid_sampling_hold", numerical_error=None,
                budget_passed=preflight != "budget_hold"))
        elif module == "scripts.prepare_phi4_hybrid":
            name = "sampling"
            fake_data(root, plan, issue=issue)
        elif module == "experiments.phi4_b2_closure_comparison":
            profile = "--profile-only" in command
            name = "training_preflight" if profile else "paired_training"
            output.mkdir()
            value = profile_receipt() if profile else formal_receipt()
            if profile and expensive_profile:
                value["update_seconds"] = 20.
            if not profile and pairing_failure:
                value["C_Q_pairing_verified"] = False
            runner.save(output / ("profile.json" if profile else "summary.json"), value)
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0))
        else:
            name = "report"
            output.mkdir()
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True))
        if name == fail_stage:
            runner.save(output / "status.json", dict(status="failed", workflow_complete=False, error="test failure"))
        calls.append((name, command, kwargs))
        return Mock(pid=100+len(calls), wait=lambda: int(name == fail_stage))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return calls


@pytest.mark.parametrize("status", ["budget_hold", "invalid_sampling_hold"])
def test_sampler_hold_prevents_production_and_gpu(prepared, monkeypatch, status):
    calls = fake_execution(prepared, monkeypatch, preflight=status)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == status and job["requires_user_action"]
    assert [name for name, *_ in calls] == ["sampler_preflight"]


@pytest.mark.parametrize("issue", [None, "ess", "hot_cold"])
def test_authorized_warnings_survive_and_pipeline_finishes(prepared, monkeypatch, issue):
    calls = fake_execution(prepared, monkeypatch, issue=issue)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "complete" and job["workflow_complete"]
    assert [name for name, *_ in calls] == ["sampler_preflight", "sampling", "training_preflight", "paired_training", "report"]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert job["actual_training_steps_per_arm"] == {"legacy": 3000, "all_start": 3000}
    assert job["sampling_quality_warning_continuation_authorized"] and not job["equilibrium_certified"]
    quality = runner.read(prepared / "sampling_quality.json")
    assert bool(quality["issues"]) == bool(issue)
    assert quality["continuation_authorized"]
    for name, digest in job["public_files"].items():
        assert (prepared / name).suffix in (".json", ".png", ".npz")
        assert runner.sha(prepared / name) == digest
    assert "data/conditions/L32_c0000/sidecar.json" in job["public_files"]
    with pytest.raises(ValueError, match="Already executed"):
        runner.execute(prepared)


def test_zero_acceptance_stops_before_gpu(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, issue="zero_acceptance")
    runner.execute(prepared)
    assert runner.read(prepared / "job.json")["status"] == "invalid_sampling_hold"
    assert len(calls) == 2


def test_gpu_projection_hold_never_reduces_steps_or_runs_formal(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, expensive_profile=True)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "budget_hold" and not job["workflow_complete"]
    assert len(calls) == 3
    assert not (prepared / "training").exists()


def test_pairing_failure_is_explicit_and_prevents_report(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, pairing_failure=True)
    with pytest.raises(RuntimeError, match="pairing"):
        runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "failed" and job["requires_user_action"]
    assert len(calls) == 4


def test_nonzero_child_preserves_outputs_and_never_retries(prepared, monkeypatch):
    calls = fake_execution(prepared, monkeypatch, fail_stage="paired_training")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "failed" and job["stages"][-1]["returncode"] == 1
    assert len(calls) == 4 and (prepared / "data/manifest.json").exists()


def test_whitelist_excludes_physical_npz_and_all_checkpoints(prepared):
    plan = runner.read(prepared / "plan.json")
    fake_data(prepared, plan)
    (prepared / "training").mkdir()
    (prepared / "training/training_step_3000.pt").write_bytes(b"not public")
    (prepared / "data/conditions/L32_c0000/data.npz").write_bytes(b"physical data")
    files = runner.final_public_files(prepared)
    assert not any(name.endswith(".pt") or name.endswith("data.npz") for name in files)
    manifest = runner.read(prepared / "data/manifest.json")
    manifest["public_files"].append("conditions/L32_c0000/data.npz")
    runner.save(prepared / "data/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="non-JSON"):
        runner.final_public_files(prepared)


def test_formal_receipt_rejects_other_gpu_and_labels(prepared):
    plan = runner.read(prepared / "plan.json")
    row = formal_receipt()
    row["gpu_uuid"] = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"
    with pytest.raises(RuntimeError, match="physical GPU"):
        runner.validate_training_receipt(row, plan, profile=False)
    row = formal_receipt()
    row["labels_used_in_training"] = True
    with pytest.raises(RuntimeError, match="label-free"):
        runner.validate_training_receipt(row, plan, profile=False)
