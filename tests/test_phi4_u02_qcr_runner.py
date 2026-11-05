"""u=.2 orchestration contracts; no SSH, physical sampling, or GPU calls."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_u02_qcr as runner


UUID = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "independent_u02"
    runner.prepare(root, gpu_index=1, expected_gpu_uuid=UUID)
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    return root


def test_fixed_physics_and_complete_work_accounting(prepared):
    plan = runner.read(prepared / "plan.json")
    p = plan["physical"]
    assert p["options"]["u_values"] == [.2]
    assert p["options"]["r_values"] == pytest.approx([-.5 + i * .8 / 11 for i in range(12)])
    assert p["options"]["sizes"] == [36]
    assert p["options"]["burn_in"] == 1024
    assert p["options"]["spacing"] == 8
    assert p["options"]["leapfrog_steps"] == 12
    assert p["options"]["step_size"] == .08
    assert p["total_hmc_trajectories"] == 98304
    assert p["total_force_site_evaluations"] == 1_656_225_792
    assert plan["total_planned_hmc_trajectories_including_preflight"] == 98304 + 9 * 56
    assert plan["total_planned_force_site_evaluations_including_preflight"] == (98304 + 9 * 56) * 13 * 1296
    assert plan["total_planned_force_site_evaluations_including_preflight"] <= 2_000_000_000
    assert plan["device"] == "cuda:1"
    assert plan["gpu_uuid"] == UUID
    assert plan["training"]["closure_objective"] == "all_start_anchor"
    assert plan["training"]["closure_scale_weights"] is None
    assert plan["training"]["steps"] == 3000
    assert plan["training"]["seed"] == 91
    assert plan["training"]["profile_seed"] == 191


def test_prepare_device_is_required_and_existing_directory_preserved(tmp_path, prepared, monkeypatch):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared, gpu_index=1, expected_gpu_uuid=UUID)
    for index, uuid in ((-1, UUID), (True, UUID), (1, None), (1, "unknown")):
        with pytest.raises(ValueError):
            runner.prepare(tmp_path / "bad", gpu_index=index, expected_gpu_uuid=uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.prepare(tmp_path / "masked", gpu_index=1, expected_gpu_uuid=UUID)


def test_cli_prepare_requires_explicit_device(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.main(["--prepare", "--output", str(tmp_path / "missing_target")])
    assert error.value.code == 2


def test_hmc_command_exactly_reconstructs_physical_options(prepared, monkeypatch):
    import scripts.prepare_phi4 as data_cli
    captured = []
    monkeypatch.setattr(data_cli, "generate_dataset", lambda output, options: captured.append(options) or
                        {"shards": [1], "warning_count": 0})
    plan = runner.read(prepared / "plan.json")
    command = runner.sampling_command(prepared, plan)
    data_cli.main(command[command.index("scripts.prepare_phi4") + 1:])
    from recurrent_rg.data.phi4 import generation_plan
    import json
    # JSON canonicalization mirrors the saved parent and data-plan receipts.
    assert json.loads(json.dumps(generation_plan(captured[0]))) == plan["physical"]


def test_stage_sequence_and_fresh_training_commands(prepared):
    plan = runner.read(prepared / "plan.json")
    stages = runner.planned_commands(prepared, plan)
    assert [row[0] for row in stages] == ["sampler_preflight", "physical_sampling", "training_preflight", "training", "report", "phi4_u02_report"]
    assert stages[0][1][3] == runner.MODULE
    assert [name for name, _, gpu in stages if gpu] == ["training_preflight", "training"]
    profile = stages[2][1]
    formal = stages[3][1]
    assert profile[profile.index("--seed") + 1] == "191"
    assert formal[formal.index("--seed") + 1] == "91"
    assert formal[formal.index("--steps") + 1] == "3000"
    assert "--resume" not in formal and "--source" not in formal
    assert formal[formal.index("--expected-gpu-uuid") + 1] == UUID


def pilot_row(*args, **kwargs):
    return dict(seed=kwargs["seed"], hmc_trajectories=56, force_evaluations=56 * 13,
        cpu_seconds=.1, wall_seconds=.2, production_acceptance=.9, burn_in_acceptance=.8,
        max_absolute_delta_h=.5)


def test_fixed_nine_preflight_covers_hot_and_both_cold_starts(prepared, monkeypatch):
    import recurrent_rg.data.phi4 as data
    sampler = Mock(side_effect=pilot_row)
    monkeypatch.setattr(data, "sample_hmc_chain", sampler)
    result = runner.sampler_preflight(prepared)
    assert sampler.call_count == 9
    assert [call.args[0] for call in sampler.call_args_list] == [-.5] * 3 + [-.33158] * 3 + [.3] * 3
    assert [call.kwargs["initial"] for call in sampler.call_args_list] == ["hot", "cold_positive", "cold_negative"] * 3
    assert all(call.args[1] == .2 and call.kwargs["step_size"] == .08 and
               call.kwargs["leapfrog_steps"] == 12 for call in sampler.call_args_list)
    assert result["status"] == "complete"
    assert result["acceptance_passed"] and result["budget_passed"]
    assert result["actual_preflight_force_site_evaluations"] == 9 * 56 * 13 * 1296
    assert result["actual_total_work_known"] is True
    assert result["projected_production_cpu_seconds_conservative"] == pytest.approx(.1 / 56 * 98304 * 2)
    assert result["other_cpu_reserve_seconds"] == 600
    assert result["remaining_gpu_cpu_reserve_seconds"] == 1800
    assert result["short_pilots_do_not_certify_equilibrium"]
    assert result["sampler_parameters_not_adapted"]
    assert not (prepared / "data").exists()


def test_low_preflight_acceptance_stops_before_production_without_adjustment(prepared, monkeypatch):
    import recurrent_rg.data.phi4 as data
    def rejected(*args, **kwargs):
        result = pilot_row(*args, **kwargs)
        result["production_acceptance"] = .59
        return result
    monkeypatch.setattr(data, "sample_hmc_chain", rejected)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == "invalid_sampling_hold"
    assert len(result["rows"]) == 9
    assert not result["physical_production_started"]
    assert not (prepared / "data").exists()


def test_numerical_preflight_failure_is_scientific_hold(prepared, monkeypatch):
    import recurrent_rg.data.phi4 as data
    monkeypatch.setattr(data, "sample_hmc_chain", Mock(side_effect=FloatingPointError("fixed HMC force overflow")))
    result = runner.sampler_preflight(prepared)
    assert result["status"] == "invalid_sampling_hold"
    assert "FloatingPointError" in result["numerical_error"]
    assert result["completed_fixed_pilots"] == 0
    assert result["budget_passed"] is False
    assert result["actual_total_work_known"] is False
    assert result["actual_preflight_trajectories"] is None
    assert result["actual_preflight_force_site_evaluations"] is None
    assert result["completed_pilots_work_lower_bound"]["hmc_trajectories"] == 0


def test_failed_second_pilot_does_not_claim_partial_work_is_known(prepared, monkeypatch):
    import recurrent_rg.data.phi4 as data
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise FloatingPointError("force failed after an unknown partial trajectory")
        return pilot_row(*args, **kwargs)
    monkeypatch.setattr(data, "sample_hmc_chain", interrupted)
    result = runner.sampler_preflight(prepared)
    assert result["completed_fixed_pilots"] == 1
    assert result["actual_total_work_known"] is False
    assert result["actual_preflight_trajectories"] is None
    assert result["completed_pilots_work_lower_bound"] == dict(hmc_trajectories=56, force_site_evaluations=56 * 13 * 1296)


def test_slow_preflight_budget_hold_preserves_fixed_production(prepared, monkeypatch):
    import recurrent_rg.data.phi4 as data
    def slow(*args, **kwargs):
        result = pilot_row(*args, **kwargs)
        result["cpu_seconds"] = 2.
        return result
    monkeypatch.setattr(data, "sample_hmc_chain", slow)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == "budget_hold"
    assert result["acceptance_passed"] is True
    assert result["planned_production_trajectories"] == 98304
    assert not (prepared / "data").exists()


def write_fake_data(root, plan, *, quality_issue=False, zero_acceptance=False):
    from recurrent_rg.data.phi4 import OBSERVABLES
    data = root / "data"
    condition = data / "conditions" / "L36_c0000"
    condition.mkdir(parents=True)
    chain = dict(chain_index=0, production_acceptance=0. if zero_acceptance else .9,
        max_absolute_delta_h=.4, initial="hot", saved_magnetization_sign_changes=4,
        warnings=[], diagnostics={name: {"ess": 100, "warning": None, "mean": .1} for name in OBSERVABLES})
    if quality_issue:
        chain["diagnostics"]["action_density"]["ess"] = 8
        chain["warnings"] = ["action_density: ESS below 32 saved draws"]
    sidecar = dict(cell_index=0, r=-.5, u=.2, h=0., chains=[chain],
        summary={"hot_cold_comparison": {"action_density": {"descriptive_three_se_warning": False}}})
    runner.save(condition / "sidecar.json", sidecar)
    runner.save(data / "plan.json", plan["physical"])
    runner.save(data / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
                cpu_seconds=.1, warning_count=int(quality_issue)))
    runner.save(data / "manifest.json", dict(status="complete", warning_count=int(quality_issue),
        shards=[dict(cell_index=0, sidecar="conditions/L36_c0000/sidecar.json",
                     sidecar_sha256=runner.sha(condition / "sidecar.json"))]))


def test_quality_gate_retains_all_diagnostics_and_nonzero_acceptance_issue(prepared):
    plan = runner.read(prepared / "plan.json")
    write_fake_data(prepared, plan, quality_issue=True)
    validity = runner.sampling_validity(prepared, plan)
    quality = runner.sampling_quality(prepared)
    assert not validity["severe"]
    assert quality["status"] == "sampling_quality_hold"
    assert quality["conditions"][0]["chains"][0]["diagnostics"]["action_density"]["ess"] == 8
    assert {issue["kind"] for issue in quality["issues"]} >= {"existing_ess_below32_or_unavailable", "existing_chain_warning"}
    assert quality["training_started"] is False
    assert quality["equilibrium_certified"] is False


@pytest.mark.parametrize("case", ["low_acceptance", "unavailable_ess", "no_sign_change", "hot_cold"])
def test_each_existing_quality_diagnostic_holds_before_training(prepared, case):
    plan = runner.read(prepared / "plan.json")
    write_fake_data(prepared, plan)
    sidecar_path = prepared / "data/conditions/L36_c0000/sidecar.json"
    sidecar = runner.read(sidecar_path)
    chain = sidecar["chains"][0]
    if case == "low_acceptance":
        chain["production_acceptance"] = .59
    elif case == "unavailable_ess":
        chain["diagnostics"]["action_density"]["ess"] = None
    elif case == "no_sign_change":
        chain["saved_magnetization_sign_changes"] = 0
    else:
        sidecar["summary"]["hot_cold_comparison"]["action_density"]["descriptive_three_se_warning"] = True
    runner.save(sidecar_path, sidecar)
    manifest_path = prepared / "data/manifest.json"
    manifest = runner.read(manifest_path)
    manifest["shards"][0]["sidecar_sha256"] = runner.sha(sidecar_path)
    runner.save(manifest_path, manifest)
    assert runner.sampling_quality(prepared)["status"] == "sampling_quality_hold"


def test_zero_acceptance_is_distinct_severe_sampling_hold(prepared):
    plan = runner.read(prepared / "plan.json")
    write_fake_data(prepared, plan, zero_acceptance=True)
    result = runner.sampling_validity(prepared, plan)
    assert result["status"] == "invalid_sampling_hold"
    assert len(result["severe"]) == 1


def test_missing_one_or_all_eight_observable_diagnostics_cannot_pass(prepared):
    plan = runner.read(prepared / "plan.json")
    write_fake_data(prepared, plan)
    sidecar_path = prepared / "data/conditions/L36_c0000/sidecar.json"
    for remove_all in (False, True):
        sidecar = runner.read(sidecar_path)
        if remove_all:
            sidecar["chains"][0]["diagnostics"] = {}
        else:
            del sidecar["chains"][0]["diagnostics"]["action_density"]
        runner.save(sidecar_path, sidecar)
        manifest_path = prepared / "data/manifest.json"
        manifest = runner.read(manifest_path)
        manifest["shards"][0]["sidecar_sha256"] = runner.sha(sidecar_path)
        runner.save(manifest_path, manifest)
        quality = runner.sampling_quality(prepared)
        assert quality["status"] == "sampling_quality_hold"
        assert any(issue["kind"] == "diagnostic_observable_inventory_mismatch" for issue in quality["issues"])


def install_fake_parent(root, monkeypatch, *, preflight_status="complete", quality_issue=False, zero_acceptance=False):
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    monkeypatch.setattr(runner.common, "profile_projection", lambda *args: dict(budget_passed=True,
                        projected_formal_gpu_seconds_conservative=900.))
    plan = runner.read(root / "plan.json")
    launched = []
    def popen(command, **kwargs):
        module = command[command.index("-m") + 1]
        output = Path(command[command.index("--output") + 1])
        if module == runner.MODULE:
            name = "sampler_preflight"
            runner.save(root / "sampler_preflight.json", dict(status=preflight_status,
                acceptance_passed=preflight_status != "invalid_sampling_hold",
                numerical_error=None, budget_passed=preflight_status != "budget_hold"))
        elif module == "scripts.prepare_phi4":
            name = "physical_sampling"
            write_fake_data(root, plan, quality_issue=quality_issue, zero_acceptance=zero_acceptance)
        elif module == "experiments.phi4_qcr":
            profile = command[command.index("--steps") + 1] == "20"
            name, steps = ("training_preflight", 20) if profile else ("training", 3000)
            output.mkdir()
            summary = dict(actual_steps=steps, device={"uuid": UUID}, peak_cuda_bytes=1024,
                optimizer_steps={name: {"min": steps, "max": steps} for name in ("Q", "C", "R")},
                labels_used_in_training=False, frozen_evaluation_verified=True)
            runner.save(output / "summary.json", summary)
            runner.save(output / "history_progress.json", {})
        else:
            name = "phi4_u02_report" if "u02" in module else "report"
            output.mkdir()
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True))
        launched.append(name)
        return Mock(pid=100 + len(launched), wait=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return launched


@pytest.mark.parametrize("status", ["budget_hold", "invalid_sampling_hold"])
def test_parent_does_not_start_production_when_preflight_holds(prepared, monkeypatch, status):
    launched = install_fake_parent(prepared, monkeypatch, preflight_status=status)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == status
    assert job["requires_user_action"]
    assert launched == ["sampler_preflight"]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert not (prepared / "data").exists()


def test_parent_quality_hold_never_starts_gpu_profile(prepared, monkeypatch):
    launched = install_fake_parent(prepared, monkeypatch, quality_issue=True)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "sampling_quality_hold"
    assert not job["workflow_complete"]
    assert launched == ["sampler_preflight", "physical_sampling"]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert (prepared / "sampling_quality.json").is_file()
    assert not (prepared / "profile_run").exists()


def test_parent_severe_sampling_hold_never_starts_gpu_profile(prepared, monkeypatch):
    launched = install_fake_parent(prepared, monkeypatch, zero_acceptance=True)
    runner.execute(prepared)
    assert runner.read(prepared / "job.json")["status"] == "invalid_sampling_hold"
    assert launched == ["sampler_preflight", "physical_sampling"]


def test_parent_success_has_six_stages_and_no_second_nice(prepared, monkeypatch):
    launched = install_fake_parent(prepared, monkeypatch)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "complete" and job["workflow_complete"]
    assert len(launched) == 6
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert job["actual_training_steps"] == 3000
    assert "sampling_quality.json" in job["public_files"]
    assert all(not name.startswith("profile_run/") and not name.endswith(".pt")
               and not (name.startswith("data/") and name.endswith(".npz")) for name in job["public_files"])
    runner.os.nice.assert_not_called()


def test_exclusive_launch_uses_new_module_and_no_migration(prepared, monkeypatch):
    child = Mock(return_value=Mock(pid=1234))
    monkeypatch.setattr(runner.subprocess, "Popen", child)
    runner.launch(prepared)
    assert runner.MODULE in child.call_args.args[0]
    with pytest.raises(FileExistsError):
        runner.launch(prepared)
    assert child.call_count == 1
    runner.save(prepared / "execution.json", {"status": "retained lock"}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.execute(prepared)


def test_gpu_gate_two_empty_checks_and_uuid_mismatch(prepared, monkeypatch):
    plan = runner.read(prepared / "plan.json")
    rows = iter([dict(gpu_uuid=UUID, gpu_ready=True, host_ready=True)] * 2)
    sleeps = []
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: next(rows))
    monkeypatch.setattr(runner.time, "sleep", lambda duration: sleeps.append(duration))
    runner.admission(prepared, plan, runner.read(prepared / "job.json"), "training", True)
    assert sleeps == [10]
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: dict(gpu_uuid="GPU-different"))
    with pytest.raises(RuntimeError, match="UUID"):
        runner.admission(prepared, plan, runner.read(prepared / "job.json"), "training", True)
