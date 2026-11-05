"""直接启动只豁免 CPU load，旧等待成本及物理/预算合同仍严格保留。"""
from pathlib import Path

import pytest

from scripts import run_phi4_ru_validation_followup_direct as runner


def receipt_fixture(tmp_path):
    base, fine = runner.base, runner.fine
    previous = tmp_path / "previous"
    (previous / "data/L108_u000_rep0").mkdir(parents=True)
    base.save(previous / "plan.json", dict(output=str(previous)))
    job = dict(status="failed", workflow_complete=False, returncode=1,
        current="L108_u000_rep0", error="RuntimeError: Zero or nonfinite HMC acceptance; preserving all files",
        cpu_seconds=fine.PREVIOUS_CPU_SECONDS, plan_sha256=base.sha(previous / "plan.json"),
        parent_pid=100, stages=[dict(name="L108_u000_rep0", status="complete", returncode=0, pid=101)])
    base.save(previous / "job.json", job)
    base.save(previous / "status.json", {key: job[key] for key in
        ("status", "workflow_complete", "returncode", "current", "error")})
    base.save(previous / "data/L108_u000_rep0/status.json", dict(status="complete",
        workflow_complete=True, returncode=0, error=None))
    waiting = tmp_path / "waiting"
    (waiting / "data").mkdir(parents=True)
    base.save(waiting / "plan.json", dict(output=str(waiting), previous_root=str(previous), schema=fine.SCHEMA))
    job = dict(status="waiting_resources", current=fine.PILOT_NAME, stages=[], workflow_complete=False,
        returncode=None, error=None, parent_pid=200, cpu_seconds=4.763063476,
        plan_sha256=base.sha(waiting / "plan.json"))
    base.save(waiting / "job.json", job)
    base.save(waiting / "status.json", {key: job[key] for key in
        ("status", "current", "workflow_complete", "returncode", "error", "cpu_seconds")})
    before = dict(status="waiting_resources", current=fine.PILOT_NAME, stages=[], actual_child_pids=[])
    after = {}
    for name in ("job.json", "status.json", "plan.json"):
        key = name.removesuffix(".json") + "_sha256"
        before[key] = after[key] = base.sha(waiting / name)
    receipt = dict(schema=runner.STOP_SCHEMA, waiting_root=str(waiting), previous_failed_root=str(previous),
        parent_pid=200, expected_module=runner.EXPECTED_MODULE, before=before, after=after,
        stop=dict(stopped_before_snapshot=True, signal="SIGSTOP,SIGTERM,SIGCONT", exit_confirmed=True,
            remaining_related_processes=[], stopped_at="2026-10-30", cpu_seconds=6,
            observed_cpu_seconds=4.84, accounting="ceil(observed+1s termination reserve)"))
    path = tmp_path / "stop_receipt.json"
    base.save(path, receipt)
    return previous, waiting, path, receipt


def plan_fixture(tmp_path):
    previous, waiting, path, receipt = receipt_fixture(tmp_path)
    plan = runner.build_plan(tmp_path / "direct", tmp_path / "source", tmp_path / "validation", previous,
        ignore_cpu_load=True, waiting_controller_stop_receipt=path)
    return previous, waiting, path, receipt, plan


def test_direct_plan_changes_only_admission_and_adds_all_previous_costs(tmp_path):
    previous, _, path, _, plan = plan_fixture(tmp_path)
    frozen = runner.fine.build_plan(plan["output"], plan["source_root"], plan["validation"], previous)
    assert plan["subsets"] == frozen["subsets"] and plan["pilot"] == frozen["pilot"]
    assert plan["optimizer_updates"] == dict(Q=0, C=0, R=0)
    assert plan["resource_policy"]["minimum_host_available_gib"] == 36
    assert plan["resource_policy"]["io_full_avg10_percent_max"] == 20
    assert plan["resource_policy"]["cpu_seconds"] == 7200
    assert plan["budget_projection"]["previous_actual_cpu_seconds"] == pytest.approx(443.393331365)
    assert plan["budget_projection"]["superseded_waiting_cpu_seconds"] == 6
    assert str(path) in plan["superseded_waiting_controller"]["fingerprints"]
    for name in ("scripts/run_phi4_ru_validation_followup_fine.py",
                 "scripts/run_phi4_ru_validation_followup_direct.py"):
        assert name in plan["sampler_source_pins"]
    assert not Path(plan["output"]).exists()


@pytest.mark.parametrize("mutate", [
    lambda r: r["before"].update(stages=[dict(name="fine_sampler_pilot")]),
    lambda r: r["stop"].update(stopped_before_snapshot=False),
    lambda r: r["stop"].update(remaining_related_processes=[dict(pid=123)]),
    lambda r: r["stop"].update(cpu_seconds=5),
])
def test_stop_contract_rejects_started_stage_unsafe_stop_and_undercharging(tmp_path, mutate):
    previous, _, path, receipt = receipt_fixture(tmp_path)
    mutate(receipt)
    runner.base.save(path, receipt)
    with pytest.raises(RuntimeError):
        runner.waiting_controller_contract(path, previous)


def test_stop_contract_rejects_physical_output_or_changed_waiting_files(tmp_path):
    previous, waiting, path, _ = receipt_fixture(tmp_path)
    (waiting / "pilot").mkdir()
    with pytest.raises(RuntimeError, match="physical-stage"):
        runner.waiting_controller_contract(path, previous)


@pytest.mark.parametrize("memory,io,expected", [
    (40 * 1024**3, 0., True), (35 * 1024**3, 0., False), (40 * 1024**3, 20., False),
])
def test_resource_override_ignores_high_load_but_keeps_memory_and_io(tmp_path, monkeypatch, memory, io, expected):
    *_, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(runner.base, "resource_snapshot", lambda _: dict(
        mem_available_bytes=memory, cpu_count=24, load1=55., io_full_avg10_percent=io, host_ready=False))
    observed = runner.resource_snapshot(plan)
    assert observed["load_gate_bypassed"] and not observed["original_host_ready"]
    assert observed["host_ready"] is expected


def test_no_implicit_override(tmp_path):
    previous, _, path, _ = receipt_fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        runner.build_plan(tmp_path / "direct", tmp_path / "source", tmp_path / "validation", previous,
            ignore_cpu_load=False, waiting_controller_stop_receipt=path)


def test_quality_hold_does_not_start_production_and_records_cumulative_charged_cost(tmp_path, monkeypatch):
    previous, _, path, _, plan = plan_fixture(tmp_path)
    root, source = Path(plan["output"]), Path(plan["source_root"])
    (source / "run").mkdir(parents=True)
    (source / "data").mkdir()
    for relative in ("run/training_step_3000.pt", "run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        (source / relative).write_bytes(b"fixture")
    monkeypatch.setattr(runner, "build_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.base.HOST)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    for name in runner.base.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner.fine, "previous_live_processes", lambda *args: [])
    monkeypatch.setattr(runner.base, "verify_public", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner.base, "verify_source_and_models", lambda *args: {})
    monkeypatch.setattr(runner.base, "verify_frozen", lambda *args: None)
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: dict(host_ready=True))
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
    with pytest.raises(runner.fine.SamplerQualityHold):
        runner.run(root, source, plan["validation"], previous, ignore_cpu_load=True,
            waiting_controller_stop_receipt=path)
    job = runner.base.read(root / "job.json")
    assert job["status"] == "sampler_quality_hold" and job["requires_user_action"]
    assert len(calls) == 1 and job["previous_cpu_seconds"] == pytest.approx(443.393331365)
    assert job["cumulative_cpu_seconds"] >= 443.393331365
