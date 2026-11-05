"""用户明确要求直接开始：仅跳过 CPU load admission，保留所有物理与预算门控。

独立新目录、新父程序；不修改旧 fine/base 实现。旧等待父程序必须已在无子
阶段的情况下安全退出，其实际 CPU 继续计入同一 7200 秒总预算。
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import run_phi4_ru_validation_followup_fine as fine

base = fine.base
SCHEMA = "phi4_ru_independent_physical_validation_fine_direct_v1"
STOP_SCHEMA = "phi4_ru_waiting_controller_superseded_v1"
EXPECTED_MODULE = "scripts.run_phi4_ru_validation_followup_fine"


def waiting_controller_contract(receipt_path, previous_root):
    """Validate immutable receipts only; execution separately checks real processes."""
    path = Path(receipt_path).resolve()
    receipt = base.read(path)
    if receipt.get("schema") != STOP_SCHEMA:
        raise RuntimeError("Waiting-controller stop receipt schema differs")
    waiting = Path(receipt["waiting_root"]).resolve()
    if Path(receipt["previous_failed_root"]).resolve() != Path(previous_root).resolve():
        raise RuntimeError("Waiting-controller receipt refers to another failed experiment")
    if receipt.get("expected_module") != EXPECTED_MODULE:
        raise RuntimeError("Waiting-controller identity differs from fine parent")
    before, after, stop = (receipt[key] for key in ("before", "after", "stop"))
    if (before.get("status") != "waiting_resources" or before.get("current") != fine.PILOT_NAME
            or before.get("stages") != [] or before.get("actual_child_pids") != []
            or stop.get("stopped_before_snapshot") is not True
            or stop.get("signal") != "SIGSTOP,SIGTERM,SIGCONT"
            or stop.get("exit_confirmed") is not True
            or stop.get("remaining_related_processes") != [] or not stop.get("stopped_at")):
        raise RuntimeError("Waiting parent was not safely stopped before any physical stage")
    cpu = stop.get("cpu_seconds")
    if (not isinstance(cpu, (int, float)) or isinstance(cpu, bool)
            or not math.isfinite(cpu) or not 0 <= cpu <= 60):
        raise RuntimeError("Waiting-controller charged CPU must be finite and at most 60 seconds")
    pins = {str(path): base.sha(path)}
    for name in ("job.json", "status.json", "plan.json"):
        key = name.removesuffix(".json") + "_sha256"
        actual = base.sha(waiting / name)
        if before.get(key) != actual or after.get(key) != actual:
            raise RuntimeError("Superseded waiting files changed at the stage boundary")
        pins[str(waiting / name)] = actual
    job, status, plan = (base.read(waiting / name) for name in ("job.json", "status.json", "plan.json"))
    if (job.get("status") != "waiting_resources" or job.get("current") != fine.PILOT_NAME
            or job.get("stages") != [] or job.get("workflow_complete") is not False
            or job.get("returncode") is not None or job.get("error") is not None
            or job.get("parent_pid") != receipt.get("parent_pid")
            or job.get("plan_sha256") != base.sha(waiting / "plan.json")
            or Path(plan.get("output", "")).resolve() != waiting
            or Path(plan.get("previous_root", "")).resolve() != Path(previous_root).resolve()
            or plan.get("schema") != fine.SCHEMA
            or not isinstance(job.get("cpu_seconds"), (int, float))
            or not 0 <= job["cpu_seconds"] <= cpu + 1e-9):
        raise RuntimeError("Superseded parent is not an untouched, unstarted waiting experiment")
    for key in ("status", "current", "workflow_complete", "returncode", "error", "cpu_seconds"):
        if status.get(key) != job.get(key):
            raise RuntimeError("Superseded parent job/status disagree")
    observed = stop.get("observed_cpu_seconds")
    if (not isinstance(observed, (int, float)) or isinstance(observed, bool)
            or not math.isfinite(observed) or observed < 0
            or stop.get("accounting") != "ceil(observed+1s termination reserve)"
            or cpu != math.ceil(max(job["cpu_seconds"], observed) + 1.)):
        raise RuntimeError("Waiting-controller conservative CPU accounting receipt differs")
    if (waiting / "pilot").exists() or any((waiting / "data").iterdir()):
        raise RuntimeError("Superseded waiting output already contains physical-stage material")
    return dict(root=str(waiting), parent_pid=receipt["parent_pid"], charged_cpu_seconds=float(cpu),
        observed_cpu_seconds=float(observed), accounting=stop["accounting"],
        stop_receipt=str(path), stopped_at=stop["stopped_at"], fingerprints=pins,
        physical_stages_started=0, physical_samples_reused=0)


def build_plan(output, source_root, validation, previous_root, *, ignore_cpu_load,
               waiting_controller_stop_receipt, code_root=None):
    if ignore_cpu_load is not True:
        raise ValueError("This separate direct-start entry requires explicit --ignore-cpu-load")
    plan = fine.build_plan(output, source_root, validation, previous_root, code_root=code_root)
    waiting = waiting_controller_contract(waiting_controller_stop_receipt, previous_root)
    output_path, waiting_path = Path(plan["output"]), Path(waiting["root"])
    if output_path == waiting_path or output_path in waiting_path.parents or waiting_path in output_path.parents:
        raise ValueError("New direct output must be distinct from superseded waiting output")
    prior_cpu = fine.PREVIOUS_CPU_SECONDS + waiting["charged_cpu_seconds"]
    plan.update(schema=SCHEMA, superseded_waiting_controller=waiting)
    plan["sampler_source_pins"]["scripts/run_phi4_ru_validation_followup_direct.py"] = base.sha(
        Path(plan["code_root"]) / "scripts/run_phi4_ru_validation_followup_direct.py")
    policy = plan["resource_policy"]
    policy.update(load_gate_user_override=True, ignored_admission_metric="load1 only",
        override_authorization="用户要求：直接开始；不更改内存、I/O、物理质量或累计预算")
    projection = plan["budget_projection"]
    projection.update(previous_failed_cpu_seconds=fine.PREVIOUS_CPU_SECONDS,
        superseded_waiting_cpu_seconds=waiting["charged_cpu_seconds"], previous_actual_cpu_seconds=prior_cpu,
        pilot_admission_passed=prior_cpu + projection["initial_remaining_excluding_unmeasured_L108"] <= base.CPU_BUDGET_SECONDS)
    return plan


def resource_snapshot(plan):
    snapshot = base.resource_snapshot(plan)
    policy = plan["resource_policy"]
    if policy.get("load_gate_user_override") is not True:
        raise RuntimeError("Direct admission is missing the explicit load-only override")
    snapshot.update(original_host_ready=snapshot["host_ready"], load_gate_user_override=True,
        load_gate_bypassed=snapshot["load1"] > policy["load_per_cpu_max"] * snapshot["cpu_count"],
        host_ready=(snapshot["mem_available_bytes"] >= policy["minimum_host_available_gib"] * 1024**3
            and snapshot["io_full_avg10_percent"] < policy["io_full_avg10_percent_max"]))
    return snapshot


def run(output, source_root, validation, previous_root, *, ignore_cpu_load,
        waiting_controller_stop_receipt):
    started_cpu, started_wall = base.used_cpu_seconds(), time.monotonic()
    root = Path(output).resolve()
    plan = build_plan(root, source_root, validation, previous_root, ignore_cpu_load=ignore_cpu_load,
        waiting_controller_stop_receipt=waiting_controller_stop_receipt)
    prior_cpu = plan["budget_projection"]["previous_actual_cpu_seconds"]
    if socket.gethostname() != base.HOST:
        raise RuntimeError(f"Physical execution is pinned to {base.HOST}")
    if any(os.environ.get(name) != "1" for name in base.THREAD_VARIABLES):
        raise RuntimeError("Launch with all recorded CPU-thread variables set to1")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("This CPU-only followup must not expose CUDA devices")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    for frozen in (previous_root, plan["superseded_waiting_controller"]["root"]):
        active = fine.previous_live_processes(frozen)
        if active:
            raise RuntimeError(f"Previous output still has active processes; preserving them: {active}")
    input_pins = base.verify_public(validation)
    input_pins.update(base.verify_source_and_models(source_root, validation))
    input_pins.update(plan["previous_failed_experiment"]["fingerprints"])
    input_pins.update(plan["superseded_waiting_controller"]["fingerprints"])
    for relative in ("run/training_step_3000.pt", "run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        path = Path(source_root).resolve() / relative
        input_pins[str(path)] = base.sha(path)
    if root.exists():
        allowed = {Path(validation).resolve()} if root == Path(validation).resolve().parent else set()
        if any(path.resolve() not in allowed for path in root.iterdir()):
            raise FileExistsError("Existing direct-start output is never overwritten or resumed")
    else:
        root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "data").mkdir()
    plan.update(prepared_at=base.now(), frozen_input_sha256=input_pins,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    base.save(root / "plan.json", plan, exclusive=True)
    job = dict(schema=SCHEMA, status="prepared", workflow_complete=False, current=None,
        stages=[], returncode=None, error=None, requires_user_action=False,
        parent_pid=os.getpid(), started_at=base.now(), actual_nice=plan["actual_nice"],
        plan_sha256=base.sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.,
        previous_cpu_seconds=prior_cpu, cumulative_cpu_seconds=prior_cpu,
        load_gate_user_override=True, superseded_waiting_cpu_seconds=plan["superseded_waiting_controller"]["charged_cpu_seconds"])
    measured = None

    def update(state=None, **more):
        if state is not None:
            job["status"] = state
        actual = base.used_cpu_seconds() - started_cpu
        job.update(more, cpu_seconds=actual, cumulative_cpu_seconds=prior_cpu + actual,
            wall_seconds=time.monotonic() - started_wall)
        base.save(root / "job.json", job)
        base.save(root / "status.json", {key: job.get(key) for key in (
            "status", "workflow_complete", "current", "returncode", "error", "requires_user_action",
            "cpu_seconds", "previous_cpu_seconds", "cumulative_cpu_seconds", "wall_seconds", "gpu_seconds",
            "load_gate_user_override", "superseded_waiting_cpu_seconds")})

    def budget_check(*, before_stage=False):
        actual = prior_cpu + base.used_cpu_seconds() - started_cpu
        if actual >= base.CPU_BUDGET_SECONDS:
            raise base.BudgetHold("Same phase-2 cumulative CPU budget reached; all completed work preserved")
        if base.directory_bytes(root) > base.OUTPUT_BUDGET_BYTES:
            raise base.BudgetHold("Direct-start output budget reached; no stage is retried or reduced")
        if before_stage:
            completed = {row["name"] for row in job["stages"] if row["status"] == "complete"}
            remaining = fine.remaining_cpu_seconds(plan, completed, measured)
            job["remaining_conservative_cpu_seconds"] = remaining
            job["projected_total_cpu_seconds"] = actual + remaining
            if actual + remaining > base.CPU_BUDGET_SECONDS:
                raise base.BudgetHold("All previous actual CPU + all new actual CPU + conservative remaining exceeds 7200 seconds")

    update()
    try:
        if not plan["budget_projection"]["pilot_admission_passed"]:
            raise base.BudgetHold("Fixed pilot admission plus already-known remaining work does not fit budget")
        for name, command in fine.stage_commands(plan):
            base.verify_frozen(plan, input_pins)
            budget_check(before_stage=True)
            while True:
                snapshot = resource_snapshot(plan)
                job.setdefault("resource_checks", []).append(snapshot)
                job["resource_checks"] = job["resource_checks"][-30:]
                if snapshot["host_ready"]:
                    break
                update("waiting_resources", current=name)
                time.sleep(plan["resource_policy"]["resource_wait_seconds"])
                budget_check(before_stage=True)
            base.verify_frozen(plan, input_pins)
            budget_check(before_stage=True)
            update("running", current=name)
            before_cpu, before_wall = base.used_cpu_seconds(), time.monotonic()
            stage = dict(name=name, command=command, status="running", returncode=None,
                started_at=base.now(), log=f"logs/{name}.log")
            job["stages"].append(stage)
            with (root / stage["log"]).open("x", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=plan["code_root"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                        stdout=log, stderr=subprocess.STDOUT) as child:
                    stage["pid"] = child.pid
                    while child.poll() is None:
                        live = base.live_child_cpu(child.pid)
                        job["live_child_cpu_seconds"] = live
                        job["budget_pressure_while_child_running"] = (
                            prior_cpu + base.used_cpu_seconds() - started_cpu + live >= base.CPU_BUDGET_SECONDS
                            or base.directory_bytes(root) > base.OUTPUT_BUDGET_BYTES)
                        update()
                        time.sleep(plan["resource_policy"]["poll_seconds"])
                    code = child.wait()
            stage.update(returncode=code, status="complete" if code == 0 else "failed", ended_at=base.now(),
                cpu_seconds=base.used_cpu_seconds() - before_cpu, wall_seconds=time.monotonic() - before_wall)
            update()
            folder = (root / "pilot" if name == fine.PILOT_NAME else root / "evaluation" if name == "frozen_evaluation"
                else root / next(row["relative_output"] for row in plan["subsets"] if row["name"] == name))
            child_status = base.read(folder / "status.json") if (folder / "status.json").exists() else {}
            stage["child_status"] = child_status
            if code != 0:
                message = f"Stage {name} exited {code}; error={child_status.get('error')}; see {stage['log']}"
                if child_status.get("status") == "budget_hold":
                    stage["status"] = "budget_hold"
                    raise base.BudgetHold(message)
                if name == fine.PILOT_NAME and child_status.get("status") == "preflight_hold":
                    stage["status"] = "preflight_hold"
                    raise fine.SamplerQualityHold(message)
                raise RuntimeError(message)
            if name == fine.PILOT_NAME:
                base.verify_public(folder, required={"plan.json", "summary.json", "status.json"})
                measured = fine.measured_projection(plan, base.read(folder / "summary.json"))
                measured.update(at=base.now(), previous_cpu_seconds=prior_cpu,
                    cumulative_actual_cpu_seconds=prior_cpu + base.used_cpu_seconds() - started_cpu)
                base.save(root / "budget_receipt.json", measured, exclusive=True)
                job["measured_budget_projection"] = measured
                budget_check(before_stage=True)
            elif name == "frozen_evaluation":
                base.verify_public(folder, required={"manifest.json", "summary.json", "status.json"})
            else:
                base.save(root / "data_manifest.json", fine.aggregate_data(plan))
            budget_check()
        base.verify_frozen(plan, input_pins)
        manifest = fine.aggregate_data(plan)
        if (not manifest["complete"] or manifest["independent_physical_chains"] != 156
                or manifest["unique_size_parameter_conditions"] != 24):
            raise RuntimeError("Final fine-integrator chain inventory differs from fixed plan")
        base.save(root / "data_manifest.json", manifest)
        update("complete", workflow_complete=True, current=None, returncode=0, ended_at=base.now())
    except Exception as exc:
        terminal = ("budget_hold" if isinstance(exc, base.BudgetHold) else
            "sampler_quality_hold" if isinstance(exc, fine.SamplerQualityHold) else "failed")
        update(terminal, workflow_complete=False, returncode=1 if terminal == "failed" else 3,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, ended_at=base.now())
        raise
    finally:
        files = {str(path.relative_to(root)): base.sha(path) for path in root.rglob("*")
            if path.is_file() and path.name != "public_files.json"
            and path.suffix.lower() in (".json", ".png", ".npz") and path.name != "data.npz"}
        base.save(root / "public_files.json", files)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("output", "source-root", "validation", "previous-root", "waiting-controller-stop-receipt"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--ignore-cpu-load", action="store_true", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for variable in base.THREAD_VARIABLES:
        os.environ[variable] = "1"
    values = (args.output, args.source_root, args.validation, args.previous_root)
    keywords = dict(ignore_cpu_load=args.ignore_cpu_load,
        waiting_controller_stop_receipt=args.waiting_controller_stop_receipt)
    result = run(*values, **keywords) if args.execute else build_plan(*values, **keywords)
    print(base.json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
