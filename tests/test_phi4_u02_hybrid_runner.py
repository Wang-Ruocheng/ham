"""Fresh hybrid-parent contracts with tiny mocks; never run physical/GPU work."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_u02_hybrid_qcr as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "fresh_hybrid"
    runner.prepare(root, gpu_index=1, expected_gpu_uuid=runner.GPU_UUID)
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    return root


def test_fixed_physics_work_ledger_and_no_legacy_global_mutation(prepared):
    from scripts import run_phi4_u02_qcr as old
    plan = runner.read(prepared / "plan.json")
    p = plan["physical"]
    assert p["schema"] == "euclidean_phi4_hmc_embedded_wolff_v1"
    assert p["options"]["u_values"] == [.2]
    assert p["options"]["r_values"] == pytest.approx([-.5 + i * .8 / 11 for i in range(12)])
    assert p["options"]["cluster_probability"] == .5
    assert p["options"]["burn_in"] == 1024 and p["options"]["spacing"] == 8
    assert p["total_hmc_trajectories"] == 98304
    assert plan["sampler_preflight"]["r_values"] == [-.5, .3]
    assert plan["sampler_preflight"]["expected_pilots"] == 6
    assert plan["sampler_preflight"]["burn_in"] == 128
    assert plan["sampler_preflight"]["calibration_uses_only_TRAIN_endpoints"]
    assert plan["total_planned_force_site_evaluations_including_preflight"] == 1_672_399_872
    ledger = plan["recorded_physical_work_ledger"]
    assert ledger["recorded_total_force_sites"] == 1_993_953_024
    assert ledger["remaining_recorded_force_site_reserve"] == 6_046_976
    assert plan["total_planned_hmc_trajectories_including_preflight"] == 98304 + 960
    assert old.MODULE == "scripts.run_phi4_u02_qcr"
    assert old.physical_options().u_values == (.2,)
    assert not hasattr(old.physical_options(), "cluster_probability")
    assert plan["training"]["steps"] == 3000
    assert plan["training"]["closure_objective"] == "all_start_anchor"
    assert plan["training"]["closure_scale_weights"] is None


def test_prepare_rejects_other_gpu_mask_and_overwrite(prepared, tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared, gpu_index=1, expected_gpu_uuid=runner.GPU_UUID)
    with pytest.raises(ValueError, match="pinned"):
        runner.prepare(tmp_path / "other", gpu_index=2, expected_gpu_uuid=runner.GPU_UUID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="CUDA mask"):
        runner.prepare(tmp_path / "masked", gpu_index=1, expected_gpu_uuid=runner.GPU_UUID)


def test_new_prepare_cli_requires_explicit_physical_target(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.main(["--prepare", "--output", str(tmp_path / "unset")])
    assert error.value.code == 2


def test_sampling_cli_reproduces_exact_frozen_hybrid_options(prepared, monkeypatch):
    import scripts.prepare_phi4_hybrid as data_cli
    captured = []
    monkeypatch.setattr(data_cli, "generate_dataset", lambda output, options:
                        captured.append(options) or {"schema": "fixture", "shards": [1], "warning_count": 0})
    plan = runner.read(prepared / "plan.json")
    command = runner.sampling_command(prepared, plan)
    data_cli.main(command[command.index("scripts.prepare_phi4_hybrid")+1:])
    from recurrent_rg.data.phi4_hybrid_pipeline import generation_plan
    assert json.loads(json.dumps(generation_plan(captured[0]))) == plan["physical"]


def test_new_module_stage_sequence_and_fresh_optimizers(prepared):
    plan = runner.read(prepared / "plan.json")
    stages = runner.planned_commands(prepared, plan)
    assert [name for name, _, _ in stages] == ["sampler_preflight", "physical_sampling", "training_preflight", "training", "report", "phi4_hybrid_report"]
    assert runner.MODULE in stages[0][1]
    assert "scripts.prepare_phi4_hybrid" in stages[1][1]
    assert "scripts.report_phi4_hybrid_qcr" in stages[-1][1]
    for index, seed in ((2, "191"), (3, "91")):
        command = stages[index][1]
        assert command[command.index("--seed")+1] == seed
        assert command[command.index("--device")+1] == "cuda:1"
        assert command[command.index("--expected-gpu-uuid")+1] == runner.GPU_UUID
        assert not {"--resume", "--source"}.intersection(command)
    assert stages[3][1][stages[3][1].index("--steps")+1] == "3000"


def pilot(*args, **kwargs):
    work = kwargs["burn_in"] + kwargs["samples"] * kwargs["spacing"]
    return dict(seed=kwargs["seed"], hmc_trajectories=work, force_evaluations=13*work,
        force_site_evaluations=13*work*1296, cpu_seconds=.1, wall_seconds=.2,
        production_acceptance=.9, burn_in_acceptance=.8, max_absolute_delta_h=.4,
        cluster_probability=.5, wolff_cluster_updates=work//2,
        total_cluster_sites=200, total_cluster_neighbor_visits=800, total_cluster_bond_tests=300)


def test_six_train_only_hybrid_pilots_include_cluster_time_and_work(prepared, monkeypatch):
    import recurrent_rg.data.phi4_hybrid as kernel
    sampler = Mock(side_effect=pilot)
    monkeypatch.setattr(kernel, "sample_hybrid_chain", sampler)
    result = runner.sampler_preflight(prepared)
    assert sampler.call_count == 6
    assert [call.args[0] for call in sampler.call_args_list] == [-.5]*3 + [.3]*3
    assert [call.kwargs["initial"] for call in sampler.call_args_list] == ["hot", "cold_positive", "cold_negative"]*2
    assert all(call.kwargs["burn_in"] == 128 and call.kwargs["samples"] == 8 and
               call.kwargs["spacing"] == 4 and call.kwargs["cluster_probability"] == .5 for call in sampler.call_args_list)
    assert result["status"] == "complete"
    assert result["throughput_includes_actual_lazy_cluster_cost"]
    assert result["actual_preflight_work"]["hmc_trajectories"] == 960
    assert result["actual_preflight_work"]["force_site_evaluations"] == 16_174_080
    assert result["actual_preflight_work"]["wolff_cluster_updates"] == 480
    assert result["projected_production_cpu_seconds_conservative"] == pytest.approx(.1/160*98304*2)
    assert result["short_pilots_do_not_certify_equilibrium"]
    assert not (prepared / "data").exists()


@pytest.mark.parametrize("case", ["acceptance", "cpu", "numeric"])
def test_preflight_holds_without_automatic_adjustment_or_production(prepared, monkeypatch, case):
    import recurrent_rg.data.phi4_hybrid as kernel
    def altered(*args, **kwargs):
        if case == "numeric":
            raise FloatingPointError("fixed force overflow")
        row = pilot(*args, **kwargs)
        row["production_acceptance" if case == "acceptance" else "cpu_seconds"] = .59 if case == "acceptance" else 5.
        return row
    monkeypatch.setattr(kernel, "sample_hybrid_chain", altered)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == ("budget_hold" if case == "cpu" else "invalid_sampling_hold")
    assert result["sampler_parameters_not_adapted"]
    assert not result["physical_production_started"]
    assert not (prepared / "data").exists()
    if case == "numeric":
        assert result["actual_total_work_known"] is False
        assert result["actual_preflight_work"] is None


def fake_data(root, plan, *, issue=None):
    from recurrent_rg.data.phi4 import OBSERVABLES
    directory = root / "data/conditions/L36_c0000"
    directory.mkdir(parents=True)
    chain = dict(chain_index=0, production_acceptance=.9, max_absolute_delta_h=.4,
        saved_magnetization_sign_changes=8, warnings=[],
        diagnostics={name: {"mean": .1, "ess": 90., "warning": None} for name in OBSERVABLES})
    if issue == "ess":
        chain["diagnostics"]["action_density"]["ess"] = 10
    elif issue == "acceptance":
        chain["production_acceptance"] = .59
    elif issue == "zero_acceptance":
        chain["production_acceptance"] = 0.
    elif issue == "missing":
        chain["diagnostics"] = {}
    elif issue == "no_sign_change":
        chain["saved_magnetization_sign_changes"] = 0
    comparison = {"action_density": {"descriptive_three_se_warning": issue == "hot_cold"}}
    sidecar = dict(cell_index=0, r=-.5, u=.2, h=0., chains=[chain], summary={"hot_cold_comparison": comparison})
    runner.save(directory / "sidecar.json", sidecar)
    runner.save(root / "data/plan.json", plan["physical"])
    runner.save(root / "data/status.json", dict(status="complete", workflow_complete=True, returncode=0,
                cpu_seconds=.1, warning_count=int(issue is not None)))
    runner.save(root / "data/manifest.json", dict(status="complete", warning_count=int(issue is not None),
        shards=[dict(cell_index=0, sidecar="conditions/L36_c0000/sidecar.json", sidecar_sha256=runner.sha(directory / "sidecar.json"))]))
    runner.save(root / "data/quality_summary.json", dict(warning_count=int(issue is not None)))


def install_fake_execution(root, monkeypatch, *, issue=None, preflight="complete", failed_stage=None):
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    monkeypatch.setattr(runner.common, "profile_projection", lambda *args:
        dict(budget_passed=True, projected_formal_gpu_seconds_conservative=900.))
    plan = runner.read(root / "plan.json")
    calls = []
    def popen(command, **kwargs):
        module = command[command.index("-m")+1]
        output = Path(command[command.index("--output")+1])
        if module == runner.MODULE:
            name = "sampler_preflight"
            runner.save(root / "sampler_preflight.json", dict(status=preflight,
                acceptance_passed=preflight != "invalid_sampling_hold", numerical_error=None,
                budget_passed=preflight != "budget_hold"))
        elif module == "scripts.prepare_phi4_hybrid":
            name = "physical_sampling"
            fake_data(root, plan, issue=issue)
        elif module == "experiments.phi4_qcr":
            profile = command[command.index("--steps")+1] == "20"
            name, steps = ("training_preflight", 20) if profile else ("training", 3000)
            output.mkdir()
            runner.save(output / "summary.json", dict(actual_steps=steps, device={"uuid": runner.GPU_UUID},
                peak_cuda_bytes=1024, optimizer_steps={n: {"min": steps, "max": steps} for n in ("Q", "C", "R")},
                labels_used_in_training=False, frozen_evaluation_verified=True))
            runner.save(output / "history_progress.json", {})
        else:
            name = "phi4_hybrid_report" if "hybrid" in module else "report"
            output.mkdir()
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True))
        if name == failed_stage:
            runner.save(output / "status.json", dict(status="failed", workflow_complete=False,
                        error="deliberate execution test failure"))
        calls.append((name, command, kwargs))
        return Mock(pid=100+len(calls), wait=lambda: int(name == failed_stage))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return calls


@pytest.mark.parametrize("status", ["budget_hold", "invalid_sampling_hold"])
def test_preflight_hold_prevents_production_and_gpu(prepared, monkeypatch, status):
    calls = install_fake_execution(prepared, monkeypatch, preflight=status)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == status and job["requires_user_action"]
    assert [row[0] for row in calls] == ["sampler_preflight"]
    assert not (prepared / "data").exists()


@pytest.mark.parametrize("issue", ["ess", "acceptance", "missing", "no_sign_change", "hot_cold"])
def test_unchanged_strict_quality_gates_precede_all_gpu_work(prepared, monkeypatch, issue):
    calls = install_fake_execution(prepared, monkeypatch, issue=issue)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "sampling_quality_hold"
    assert not job["workflow_complete"]
    assert [row[0] for row in calls] == ["sampler_preflight", "physical_sampling"]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert not (prepared / "profile_run").exists()


def test_severe_sampling_invalidity_is_not_training_failure(prepared, monkeypatch):
    calls = install_fake_execution(prepared, monkeypatch, issue="zero_acceptance")
    runner.execute(prepared)
    assert runner.read(prepared / "job.json")["status"] == "invalid_sampling_hold"
    assert len(calls) == 2


def test_fresh_complete_pipeline_has_six_stages_fixed_endpoint_and_whitelist(prepared, monkeypatch):
    calls = install_fake_execution(prepared, monkeypatch)
    runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "complete" and job["workflow_complete"]
    assert len(calls) == 6 and all(row["returncode"] == 0 for row in job["stages"])
    assert job["actual_training_steps"] == 3000
    assert job["actual_nice"] == 10
    runner.os.nice.assert_not_called()
    assert "data/quality_summary.json" in job["public_files"]
    assert "sampling_quality.json" in job["public_files"]
    assert not any(name.endswith(".pt") or name.startswith("profile_run/") or
                   (name.startswith("data/") and name.endswith(".npz")) for name in job["public_files"])
    for name, command, kwargs in calls:
        assert kwargs["env"]["NUMEXPR_NUM_THREADS"] == "1"
        if name not in ("training", "training_preflight"):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_execution_failure_is_preserved_no_retry_and_no_report(prepared, monkeypatch):
    calls = install_fake_execution(prepared, monkeypatch, failed_stage="training")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runner.execute(prepared)
    job = runner.read(prepared / "job.json")
    assert job["status"] == "failed" and job["requires_user_action"]
    assert len(calls) == 4
    assert (prepared / "data/manifest.json").is_file()
    assert not (prepared / "report").exists()


def test_exclusive_launch_new_module_unmasked_gpu_and_thread_environment(prepared, monkeypatch):
    child = Mock(return_value=Mock(pid=1234))
    monkeypatch.setattr(runner.subprocess, "Popen", child)
    runner.launch(prepared)
    assert runner.MODULE in child.call_args.args[0]
    assert "nice" not in child.call_args.args[0]
    assert "CUDA_VISIBLE_DEVICES" not in child.call_args.kwargs["env"]
    assert all(child.call_args.kwargs["env"][name] == "1" for name in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    with pytest.raises(FileExistsError):
        runner.launch(prepared)
    assert child.call_count == 1
    runner.save(prepared / "execution.json", {"status": "retained durable lock"}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.execute(prepared)
