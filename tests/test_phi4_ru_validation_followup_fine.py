"""细积分父程序的计划、旧成本、预检及无重试合同；不做物理采样。"""
from pathlib import Path

import pytest

from scripts import run_phi4_ru_validation_followup_fine as runner


def previous_fixture(tmp_path):
    base = runner.base
    root = tmp_path / "previous"
    (root / "data/L108_u000_rep0").mkdir(parents=True)
    base.save(root / "plan.json", dict(output=str(root)))
    job = dict(status="failed", workflow_complete=False, returncode=1,
        current="L108_u000_rep0", error="RuntimeError: Zero or nonfinite HMC acceptance; preserving all files",
        cpu_seconds=runner.PREVIOUS_CPU_SECONDS, plan_sha256=base.sha(root / "plan.json"),
        parent_pid=100, stages=[dict(name="L108_u000_rep0", status="complete", returncode=0, pid=101)])
    base.save(root / "job.json", job)
    base.save(root / "status.json", {key: job[key] for key in
        ("status", "workflow_complete", "returncode", "current", "error")})
    base.save(root / "data/L108_u000_rep0/status.json", dict(status="complete",
        workflow_complete=True, returncode=0, error=None))
    return root


def plan_fixture(tmp_path):
    previous = previous_fixture(tmp_path)
    return runner.build_plan(tmp_path / "fine", tmp_path / "source", tmp_path / "validation", previous)


def pilot_fixture(plan, cpu=10.):
    rows = []
    for condition in plan["pilot"]["conditions"]:
        for initial in plan["pilot"]["initial_states"]:
            rows.append(dict(condition_index=condition["condition_index"], initial=initial,
                r=condition["r"], u=condition["u"], size=108, seed=condition["seeds"][initial],
                gate_passed=True, finite=True, cpu_seconds=cpu,
                hmc_trajectories=runner.PILOT_CYCLES_PER_CHAIN))
    return dict(status="complete", returncode=0, production_allowed=True,
        source_pins_match=True, source_pins_before=plan["pilot"]["source_pins"],
        source_pins_after=plan["pilot"]["source_pins"], chains=rows, cpu_seconds=18*cpu)


def test_plan_preserves_production_design_and_changes_only_large_integrator(tmp_path):
    plan = plan_fixture(tmp_path)
    assert not Path(plan["output"]).exists()
    assert len(plan["subsets"]) == 13
    assert plan["independent_physical_chains"] == 156
    assert plan["unique_size_parameter_conditions"] == 24
    assert plan["total_hmc_trajectories"] == 156 * 2048
    assert plan["total_force_site_evaluations"] == (120 * 36**2 * 13 + 36 * 108**2 * 25) * 2048
    assert plan["previous_failed_experiment"]["previous_cpu_seconds"] == 437.393331365
    assert plan["pilot"]["independent_pilot_chains"] == 18
    assert plan["pilot"]["cycles_per_chain"] == 1152
    assert plan["optimizer_updates"] == dict(Q=0, C=0, R=0)
    assert plan["budget_projection"]["L108_sampling_cpu_seconds"] is None
    for row in plan["subsets"]:
        options = row["physical"]["options"]
        assert options["burn_in"] == 1024 and options["spacing"] == 8
        assert options["samples"] == 128 and options["chains"] == 4
        assert (options["step_size"], options["leapfrog_steps"]) == ((.04, 24) if row["size"] == 108 else (.08, 12))
        assert row["physical"]["total_force_site_evaluations"] <= 8_000_000_000
        if row["size"] == 108:
            assert row["physical"]["total_force_site_evaluations"] > 7_000_000_000
    assert plan["integration_change"]["historical_child_cap_unchanged"] == 7_000_000_000


def test_child_commands_use_fixed_fine_wrappers_and_no_resume(tmp_path, monkeypatch):
    from scripts import prepare_phi4_ru_fine as cli
    from recurrent_rg.data.phi4_ru_fine_pipeline import generation_plan
    plan = plan_fixture(tmp_path)
    commands, seen = runner.stage_commands(plan), []
    monkeypatch.setattr(cli, "generate_dataset", lambda output, options:
        seen.append(options) or dict(shards=[], warning_count=0))
    assert commands[0][0] == runner.PILOT_NAME
    assert "scripts.phi4_ru_fine_sampler_pilot" in commands[0][1]
    for (_, command), expected in zip(commands[1:14], plan["subsets"]):
        cli.main(command[command.index("scripts.prepare_phi4_ru_fine") + 1:])
        assert runner.base.json.loads(runner.base.json.dumps(generation_plan(seen[-1]))) == expected["physical"]
    assert commands[-1][0] == "frozen_evaluation"
    assert "experiments.phi4_ru_followup_fine_evaluate" in commands[-1][1]
    assert "--resume" not in " ".join(token for _, command in commands for token in command)


def test_previous_work_receipt_is_exact_and_stopped_child_required(tmp_path):
    previous = previous_fixture(tmp_path)
    receipt = runner.previous_contract(previous)
    assert receipt["invalid_physical_chains_excluded"] == 12
    assert not receipt["old_data_used_in_evaluation"]
    runner.base.save(previous / "data/L108_u000_rep0/status.json",
        dict(status="running", workflow_complete=False, returncode=None))
    with pytest.raises(RuntimeError, match="child status"):
        runner.previous_contract(previous)


def test_actual_process_tree_ignores_reused_pid_and_new_previous_argument(tmp_path):
    previous = tmp_path / "previous"
    proc = tmp_path / "proc"
    proc.mkdir()
    def record(pid, ppid, command):
        folder = proc / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes("\0".join(command).encode())
        (folder / "stat").write_text(f"{pid} (python process) S {ppid} 0 0 0")
    record(100, 1, ["unrelated", "--output", str(tmp_path / "unrelated")])
    record(200, 1, ["python", "-m", "fine", "--previous-root", str(previous), "--output", str(tmp_path / "fine")])
    record(300, 1, ["python", "-m", "old", "--output", str(previous / "data/old")])
    record(301, 300, ["worker"])
    assert [row["pid"] for row in runner.previous_live_processes(previous, proc_root=proc)] == [300, 301]


def test_measured_projection_uses_slowest_full_outer_cpu_and_counts_old_work(tmp_path):
    plan = plan_fixture(tmp_path)
    summary = pilot_fixture(plan)
    summary["chains"][-1]["cpu_seconds"] = 40.
    pilot = Path(plan["output"]) / "pilot"
    pilot.mkdir(parents=True)
    runner.base.save(pilot / "summary.json", summary)
    measured = runner.measured_projection(plan, summary)
    assert measured["L108_sampling_cpu_seconds"] == pytest.approx(40 / 1152 * 2048 * 36 * 1.5)
    remaining = runner.remaining_cpu_seconds(plan, {runner.PILOT_NAME}, measured)
    expected = measured["L108_sampling_cpu_seconds"] + plan["budget_projection"]["L36_sampling_cpu_seconds"] + 770.46
    assert remaining == pytest.approx(expected)
    assert runner.PREVIOUS_CPU_SECONDS + 1100 + remaining > 7200
    with pytest.raises(RuntimeError, match="lacks the measured"):
        runner.remaining_cpu_seconds(plan, {runner.PILOT_NAME})
    summary["chains"][-1] = summary["chains"][0]
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.measured_projection(plan, summary)


def execution_fixture(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    root, source = Path(plan["output"]), Path(plan["source_root"])
    (source / "run").mkdir(parents=True)
    (source / "data").mkdir()
    for relative in ("run/training_step_3000.pt", "run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        (source / relative).write_bytes(b"fixture")
    monkeypatch.setattr(runner, "build_plan", lambda *args: plan)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.base.HOST)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    for name in runner.base.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner, "previous_live_processes", lambda *args: [])
    monkeypatch.setattr(runner.base, "verify_public", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner.base, "verify_source_and_models", lambda *args: {})
    monkeypatch.setattr(runner.base, "verify_frozen", lambda *args: None)
    monkeypatch.setattr(runner.base, "resource_snapshot", lambda plan: dict(host_ready=True))
    return root, source, Path(plan["validation"]), Path(plan["previous_root"]), plan


def test_pilot_quality_hold_never_starts_production_or_retries(tmp_path, monkeypatch):
    root, source, validation, previous, plan = execution_fixture(tmp_path, monkeypatch)
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(command)
            (root / "pilot").mkdir()
            runner.base.save(root / "pilot/status.json", dict(status="preflight_hold", returncode=3, error=None))
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def poll(self): return 3
        def wait(self): return 3
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(runner.SamplerQualityHold):
        runner.run(root, source, validation, previous)
    job = runner.base.read(root / "job.json")
    assert job["status"] == "sampler_quality_hold" and job["requires_user_action"]
    assert len(calls) == 1 and job["stages"][0]["status"] == "preflight_hold"
    assert job["cumulative_cpu_seconds"] >= 437.393331365
    with pytest.raises(FileExistsError):
        runner.run(root, source, validation, previous)
    assert len(calls) == 1


def test_successful_pilot_with_excess_projection_holds_before_production(tmp_path, monkeypatch):
    root, source, validation, previous, plan = execution_fixture(tmp_path, monkeypatch)
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(command)
            (root / "pilot").mkdir()
            runner.base.save(root / "pilot/summary.json", pilot_fixture(plan, cpu=100.))
            runner.base.save(root / "pilot/status.json", dict(status="complete", returncode=0, error=None))
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def poll(self): return 0
        def wait(self): return 0
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(runner.base.BudgetHold, match="conservative remaining"):
        runner.run(root, source, validation, previous)
    job = runner.base.read(root / "job.json")
    assert job["status"] == "budget_hold" and len(calls) == 1
    assert job["stages"][0]["status"] == "complete"
    assert (root / "budget_receipt.json").exists()
    assert job["projected_total_cpu_seconds"] > 7200
