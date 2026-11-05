"""细化大尺寸积分的独立物理验证；先预检与累计预算门控，再顺序生产。

原失败目录及冻结源码保持不变。旧坏链不进入评价，旧消耗继续计入同一
7200 CPU 秒预算；预检链不进入生产，运行中的子阶段不被终止或自动重试。
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

from scripts import run_phi4_ru_validation_followup as base


SCHEMA = "phi4_ru_independent_physical_validation_fine_v2"
PREVIOUS_CPU_SECONDS = 437.393331365
PILOT_NAME = "fine_sampler_pilot"
PILOT_CYCLES_PER_CHAIN = 1152
PRODUCTION_CYCLES_PER_CHAIN = 2048
COLLECTION_CPU_RESERVE_SECONDS = 170.46
EVALUATION_CPU_RESERVE_SECONDS = 600.


class SamplerQualityHold(RuntimeError):
    pass


def previous_contract(previous_root):
    """Read/pin the exact completed-child / failed-parent event, not old PID liveness."""
    root = Path(previous_root).resolve()
    job, status, plan = (base.read(root / name) for name in ("job.json", "status.json", "plan.json"))
    if (job.get("status") != "failed" or job.get("workflow_complete") is not False
            or job.get("returncode") != 1 or job.get("current") != "L108_u000_rep0"
            or "Zero or nonfinite HMC acceptance" not in str(job.get("error"))):
        raise RuntimeError("Previous root does not match the preserved cold-chain sampling failure")
    for key in ("status", "workflow_complete", "returncode", "current", "error"):
        if status.get(key) != job.get(key):
            raise RuntimeError("Previous job and status disagree")
    stages = job.get("stages", [])
    if (len(stages) != 1 or stages[0].get("name") != "L108_u000_rep0"
            or stages[0].get("status") != "complete" or stages[0].get("returncode") != 0):
        raise RuntimeError("Previous physical child has not been recorded as completed")
    if (not math.isclose(job.get("cpu_seconds", -1), PREVIOUS_CPU_SECONDS, rel_tol=0., abs_tol=1e-9)
            or job.get("plan_sha256") != base.sha(root / "plan.json")
            or Path(plan.get("output", "")).resolve() != root):
        raise RuntimeError("Previous work receipt or frozen plan fingerprint changed")
    child_status_path = root / "data/L108_u000_rep0/status.json"
    child = base.read(child_status_path)
    if (child.get("status") != "complete" or child.get("workflow_complete") is not True
            or child.get("returncode") != 0 or child.get("error") is not None):
        raise RuntimeError("Previous child status is not complete with returncode zero")
    pins = {str(root / name): base.sha(root / name) for name in ("job.json", "status.json", "plan.json")}
    pins[str(child_status_path)] = base.sha(child_status_path)
    return dict(root=str(root), previous_cpu_seconds=job["cpu_seconds"],
        invalid_physical_chains_excluded=12, old_data_used_in_evaluation=False,
        fingerprints=pins, previous_parent_pid=job.get("parent_pid"),
        previous_child_pids=[row.get("pid") for row in stages])


def previous_live_processes(previous_root, *, proc_root="/proc"):
    """Inspect actual output arguments and their descendants; reused old PIDs are irrelevant."""
    root, proc = Path(previous_root).resolve(), Path(proc_root)
    records, directly_related = {}, set()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            command = [token for token in command if token]
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            pid, ppid = int(entry.name), int(fields[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        records[pid] = dict(pid=pid, parent_pid=ppid, state=fields[0], command=command)
        for index, token in enumerate(command[:-1]):
            if token != "--output":
                continue
            output = Path(command[index + 1]).resolve()
            if output == root or root in output.parents:
                directly_related.add(pid)
    related = set(directly_related)
    while True:
        expanded = related | {pid for pid, record in records.items() if record["parent_pid"] in related}
        if expanded == related:
            break
        related = expanded
    return [records[pid] for pid in sorted(related) if records[pid]["state"] != "Z"]


def build_plan(output, source_root, validation, previous_root, *, code_root=None):
    """Read-only plan, including the preserved old failure and a fixed conditional gate."""
    from recurrent_rg.data.phi4_ru_fine_pipeline import Phi4FineOptions, generation_plan, source_pins
    from scripts.phi4_ru_fine_sampler_pilot import build_plan as pilot_plan

    code = Path(code_root or Path(__file__).resolve().parents[1]).resolve()
    output, source_root, validation, previous_root = map(lambda value: Path(value).resolve(),
        (output, source_root, validation, previous_root))
    for frozen in (source_root, previous_root):
        if output == frozen or frozen in output.parents or output in frozen.parents:
            raise ValueError("New output must remain outside and distinct from frozen experiment roots")
    if output == code or output in code.parents:
        raise ValueError("Output must not contain the implementation checkout")
    prior = previous_contract(previous_root)
    subsets = []
    for row in base.fixed_subsets():
        large = row["size"] == 108
        options = Phi4FineOptions(sizes=(row["size"],), r_values=tuple(row["r_values"]),
            u_values=(row["u"],), chains=4, samples=128, burn_in=1024, spacing=8,
            step_size=.04 if large else .08, leapfrog_steps=24 if large else 12,
            seed=row["seed"], test_every=3, test_offset=1)
        # Retain JSON list conventions to match child manifests exactly.
        generated = base.json.loads(base.json.dumps(generation_plan(options)))
        subsets.append(dict(row, relative_output=f"data/{row['name']}", physical=generated))
    pilot = pilot_plan(output / "pilot")
    pins = dict(source_pins())
    pins.update(pilot["source_pins"])
    names = ("recurrent_rg/data/phi4_ru_pipeline.py",
        "scripts/run_phi4_ru_validation_followup.py", "scripts/run_phi4_ru_validation_followup_fine.py",
        "scripts/prepare_phi4_ru_fine.py", "experiments/phi4_ru_followup_fine_evaluate.py",
        "experiments/phi4_ru_followup_evaluate.py", "experiments/phi4_ru_r_cache.py",
        "experiments/phi4_ru_r_calibration.py", "recurrent_rg/phi4_r_calibration.py",
        "experiments/phi4_qcr.py", "recurrent_rg/phi4_qcr.py", "scripts/report_phi4_qcr.py")
    for relative in names:
        path = code / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen continuation implementation missing: {relative}")
        pins[relative] = base.sha(path)
    small_sampling = base.SOURCE_CPU_SECONDS / base.SOURCE_CHAIN_COUNT * 120 * 1.75
    initial_remaining = (pilot["cpu_cap_seconds"] + small_sampling
        + COLLECTION_CPU_RESERVE_SECONDS + EVALUATION_CPU_RESERVE_SECONDS)
    return dict(schema_version=5, schema=SCHEMA, output=str(output), source_root=str(source_root),
        validation=str(validation), previous_root=str(previous_root), code_root=str(code), host=base.HOST,
        previous_failed_experiment=prior, subsets=subsets, pilot=pilot,
        sampler_source_pins=pins, independent_physical_chains=156,
        unique_size_parameter_conditions=24, microscopic_lattice_sizes=[36, 108],
        observed_lattice_sizes=[[36, 12, 4], [108, 36, 12]],
        source_checkpoint=str(source_root / "run/training_step_3000.pt"),
        source_state_sha256=base.SOURCE_STATE_SHA256,
        frozen_R_model_keys=[f"selected_seed{seed}" for seed in (91, 92, 93)],
        fair_baseline_keys=[f"adam6000_seed{seed}" for seed in (91, 92, 93)],
        total_hmc_trajectories=sum(row["physical"]["total_hmc_trajectories"] for row in subsets),
        total_force_site_evaluations=sum(row["physical"]["total_force_site_evaluations"] for row in subsets),
        projected_sampling_output_bytes=sum(row["physical"]["projected_output_bytes"] for row in subsets),
        integration_change=dict(L108=dict(step_size=.04, leapfrog_steps=24),
            L36=dict(step_size=.08, leapfrog_steps=12), trajectory_length=.96,
            physical_seed_policy="Same production seed namespaces as the invalid version; corrected integrator, fresh output, no old chains mixed in",
            new_child_force_site_cap=8_000_000_000, historical_child_cap_unchanged=7_000_000_000),
        budget_projection=dict(previous_actual_cpu_seconds=PREVIOUS_CPU_SECONDS,
            pilot_cpu_cap_seconds=pilot["cpu_cap_seconds"],
            L36_sampling_cpu_seconds=small_sampling, L36_sampling_safety_multiplier=1.75,
            L108_sampling_cpu_seconds=None, L108_sampling_safety_multiplier=1.5,
            L108_projection_formula="max(pilot per-chain outer CPU / 1152) * 2048 * 36 * 1.5",
            collection_cpu_seconds=COLLECTION_CPU_RESERVE_SECONDS,
            evaluation_and_report_cpu_reserve_seconds=EVALUATION_CPU_RESERVE_SECONDS,
            initial_remaining_excluding_unmeasured_L108=initial_remaining,
            pilot_admission_passed=PREVIOUS_CPU_SECONDS + initial_remaining <= base.CPU_BUDGET_SECONDS,
            production_admission_requires_pilot_gate_and_measured_budget=True,
            larger_size_throughput_not_measured=True, projection_is_not_wall_time=True),
        resource_policy=dict(cpu_seconds=base.CPU_BUDGET_SECONDS,
            directory_bytes=base.OUTPUT_BUDGET_BYTES, cpu_threads=1, nice=10, gpu_seconds=0.,
            minimum_host_available_gib=36., load_per_cpu_max=1.25, io_full_avg10_percent_max=20.,
            poll_seconds=5., resource_wait_seconds=30., wall_time_is_not_cpu_budget=True,
            cpu_budget_semantics="Same phase-2 cumulative stage-boundary budget: old failed CPU + all new outer CPU + conservative remaining; never terminate an active child",
            atomic_child_overrun_possible=True),
        stages=[PILOT_NAME] + [row["name"] for row in subsets] + ["frozen_evaluation"],
        all_new_data_evaluation_only=True, labels_used_in_R_optimization=False,
        upstream_generator_train_test_markers_ignored_for_followup_evaluation=True,
        optimizer_updates=dict(Q=0, C=0, R=0), gpu_used=False,
        warnings_retained_without_resampling=True, equilibrium_certified=False,
        automatic_retry=False, automatic_restart=False, child_termination=False,
        qualification="细积分修正版沿用相同生产seed；旧坏链及18条pilot均排除。固定Q/C与预选R，结果仍受有限链、有限尺寸与热化警告限制。")


def stage_commands(plan):
    commands = [(PILOT_NAME, [sys.executable, "-u", "-m", "scripts.phi4_ru_fine_sampler_pilot",
        "--output", str(Path(plan["output"]) / "pilot"), "--execute"])]
    for row in plan["subsets"]:
        command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_ru_fine", "--execute",
            "--output", str(Path(plan["output"]) / row["relative_output"])]
        for key, value in row["physical"]["options"].items():
            command.append("--" + key.replace("_", "-"))
            command.extend(str(item) for item in value) if isinstance(value, (tuple, list)) else command.append(str(value))
        commands.append((row["name"], command))
    commands.append(("frozen_evaluation", [sys.executable, "-u", "-m",
        "experiments.phi4_ru_followup_fine_evaluate", "--source-root", plan["source_root"],
        "--data-root", plan["output"], "--validation", plan["validation"],
        "--output", str(Path(plan["output"]) / "evaluation")]))
    return commands


def measured_projection(plan, pilot_summary):
    """No averaging away a slow condition; every fixed pilot chain must have passed."""
    expected = plan["pilot"]
    rows = pilot_summary.get("chains", [])
    if (pilot_summary.get("status") != "complete" or pilot_summary.get("returncode") != 0
            or pilot_summary.get("production_allowed") is not True or len(rows) != 18
            or pilot_summary.get("source_pins_match") is not True
            or pilot_summary.get("source_pins_before") != expected["source_pins"]
            or pilot_summary.get("source_pins_after") != expected["source_pins"]):
        raise SamplerQualityHold("Fixed L108 fine-integrator preflight did not pass all 18 chains")
    actual_identities = set()
    per_cycle = []
    for row in rows:
        index, initial = row["condition_index"], row["initial"]
        if not isinstance(index, int) or not 0 <= index < 6 or initial not in expected["initial_states"]:
            raise RuntimeError("Pilot condition identity differs from fixed plan")
        condition = expected["conditions"][index]
        cpu = row["cpu_seconds"]
        if (row.get("gate_passed") is not True or row.get("finite") is not True
                or row.get("hmc_trajectories") != PILOT_CYCLES_PER_CHAIN
                or row.get("seed") != condition["seeds"][initial]
                or row.get("r") != condition["r"] or row.get("u") != condition["u"]
                or row.get("size") != 108 or not math.isfinite(cpu) or cpu <= 0):
            raise RuntimeError("Pilot work, quality, seed or outer CPU receipt is inconsistent")
        actual_identities.add((index, initial))
        per_cycle.append(cpu / PILOT_CYCLES_PER_CHAIN)
    if len(actual_identities) != 18:
        raise RuntimeError("Pilot receipt contains duplicate or missing physical chains")
    worst = max(per_cycle)
    return dict(L108_outer_cpu_seconds_per_cycle=worst,
        L108_sampling_cpu_seconds=worst * PRODUCTION_CYCLES_PER_CHAIN * 36 * 1.5,
        L108_sampling_safety_multiplier=1.5, pilot_cpu_seconds=pilot_summary["cpu_seconds"],
        pilot_chains_excluded_from_production=18, prior_invalid_chains_excluded=12,
        pilot_summary_sha256=base.sha(Path(plan["output"]) / "pilot/summary.json"))


def remaining_cpu_seconds(plan, completed, measured=None):
    projection = plan["budget_projection"]
    remaining = 0.
    if PILOT_NAME not in completed:
        remaining += projection["pilot_cpu_cap_seconds"]
    small = sum(12 for row in plan["subsets"] if row["size"] == 36 and row["name"] not in completed)
    remaining += projection["L36_sampling_cpu_seconds"] * small / 120
    large = sum(12 for row in plan["subsets"] if row["size"] == 108 and row["name"] not in completed)
    if measured is not None:
        remaining += measured["L108_sampling_cpu_seconds"] * large / 36
    elif PILOT_NAME in completed and large:
        raise RuntimeError("L108 production lacks the measured preflight budget projection")
    if "frozen_evaluation" not in completed:
        remaining += projection["collection_cpu_seconds"] + projection["evaluation_and_report_cpu_reserve_seconds"]
    return remaining


def aggregate_data(plan):
    result = base.aggregate_data(plan)
    result.update(schema=SCHEMA, previous_invalid_chains_excluded=12,
        pilot_chains_excluded=18, old_data_used_in_evaluation=False)
    return result


def run(output, source_root, validation, previous_root):
    started_cpu, started_wall = base.used_cpu_seconds(), time.monotonic()
    root = Path(output).resolve()
    plan = build_plan(root, source_root, validation, previous_root)
    if socket.gethostname() != base.HOST:
        raise RuntimeError(f"Physical execution is pinned to {base.HOST}")
    if any(os.environ.get(name) != "1" for name in base.THREAD_VARIABLES):
        raise RuntimeError("Launch with all recorded CPU-thread variables set to1")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("This CPU-only followup must not expose CUDA devices")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    active = previous_live_processes(previous_root)
    if active:
        raise RuntimeError(f"Previous output still has actual active processes; preserving them: {active}")
    input_pins = base.verify_public(validation)
    input_pins.update(base.verify_source_and_models(source_root, validation))
    input_pins.update(plan["previous_failed_experiment"]["fingerprints"])
    for relative in ("run/training_step_3000.pt", "run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        path = Path(source_root).resolve() / relative
        input_pins[str(path)] = base.sha(path)
    if root.exists():
        allowed = {Path(validation).resolve()} if root == Path(validation).resolve().parent else set()
        if any(path.resolve() not in allowed for path in root.iterdir()):
            raise FileExistsError("Existing fine-integrator output is never overwritten or resumed")
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
        previous_cpu_seconds=PREVIOUS_CPU_SECONDS, cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS)
    measured = None

    def update(state=None, **more):
        if state is not None:
            job["status"] = state
        actual = base.used_cpu_seconds() - started_cpu
        job.update(more, cpu_seconds=actual, cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS + actual,
            wall_seconds=time.monotonic() - started_wall)
        base.save(root / "job.json", job)
        base.save(root / "status.json", {key: job.get(key) for key in (
            "status", "workflow_complete", "current", "returncode", "error", "requires_user_action",
            "cpu_seconds", "previous_cpu_seconds", "cumulative_cpu_seconds", "wall_seconds", "gpu_seconds")})

    def budget_check(*, before_stage=False):
        actual = PREVIOUS_CPU_SECONDS + base.used_cpu_seconds() - started_cpu
        if actual >= base.CPU_BUDGET_SECONDS:
            raise base.BudgetHold("Same phase-2 cumulative CPU budget reached; all completed work preserved")
        if base.directory_bytes(root) > base.OUTPUT_BUDGET_BYTES:
            raise base.BudgetHold("Fine-integrator output budget reached; no stage is retried or reduced")
        if before_stage:
            completed = {row["name"] for row in job["stages"] if row["status"] == "complete"}
            remaining = remaining_cpu_seconds(plan, completed, measured)
            job["remaining_conservative_cpu_seconds"] = remaining
            job["projected_total_cpu_seconds"] = actual + remaining
            if actual + remaining > base.CPU_BUDGET_SECONDS:
                raise base.BudgetHold("Previous actual CPU + all new actual CPU + conservative remaining exceeds 7200 seconds")

    update()
    try:
        if not plan["budget_projection"]["pilot_admission_passed"]:
            raise base.BudgetHold("Fixed pilot admission plus already-known remaining work does not fit budget")
        for name, command in stage_commands(plan):
            base.verify_frozen(plan, input_pins)
            budget_check(before_stage=True)
            while True:
                snapshot = base.resource_snapshot(plan)
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
                            PREVIOUS_CPU_SECONDS + base.used_cpu_seconds() - started_cpu + live >= base.CPU_BUDGET_SECONDS
                            or base.directory_bytes(root) > base.OUTPUT_BUDGET_BYTES)
                        update()
                        time.sleep(plan["resource_policy"]["poll_seconds"])
                    code = child.wait()
            stage.update(returncode=code, status="complete" if code == 0 else "failed", ended_at=base.now(),
                cpu_seconds=base.used_cpu_seconds() - before_cpu, wall_seconds=time.monotonic() - before_wall)
            update()
            folder = (root / "pilot" if name == PILOT_NAME else root / "evaluation" if name == "frozen_evaluation"
                else root / next(row["relative_output"] for row in plan["subsets"] if row["name"] == name))
            child_status = base.read(folder / "status.json") if (folder / "status.json").exists() else {}
            stage["child_status"] = child_status
            if code != 0:
                message = f"Stage {name} exited {code}; error={child_status.get('error')}; see {stage['log']}"
                if child_status.get("status") == "budget_hold":
                    stage["status"] = "budget_hold"
                    raise base.BudgetHold(message)
                if name == PILOT_NAME and child_status.get("status") == "preflight_hold":
                    stage["status"] = "preflight_hold"
                    raise SamplerQualityHold(message)
                raise RuntimeError(message)
            if name == PILOT_NAME:
                base.verify_public(folder, required={"plan.json", "summary.json", "status.json"})
                measured = measured_projection(plan, base.read(folder / "summary.json"))
                measured.update(at=base.now(), previous_cpu_seconds=PREVIOUS_CPU_SECONDS,
                    cumulative_actual_cpu_seconds=PREVIOUS_CPU_SECONDS + base.used_cpu_seconds() - started_cpu)
                base.save(root / "budget_receipt.json", measured, exclusive=True)
                job["measured_budget_projection"] = measured
                budget_check(before_stage=True)
            elif name == "frozen_evaluation":
                base.verify_public(folder, required={"manifest.json", "summary.json", "status.json"})
            else:
                base.save(root / "data_manifest.json", aggregate_data(plan))
            budget_check()
        base.verify_frozen(plan, input_pins)
        manifest = aggregate_data(plan)
        if (not manifest["complete"] or manifest["independent_physical_chains"] != 156
                or manifest["unique_size_parameter_conditions"] != 24):
            raise RuntimeError("Final fine-integrator chain inventory differs from fixed plan")
        base.save(root / "data_manifest.json", manifest)
        update("complete", workflow_complete=True, current=None, returncode=0, ended_at=base.now())
    except Exception as exc:
        terminal = ("budget_hold" if isinstance(exc, base.BudgetHold) else
            "sampler_quality_hold" if isinstance(exc, SamplerQualityHold) else "failed")
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
    for argument in ("output", "source-root", "validation", "previous-root"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for variable in base.THREAD_VARIABLES:
        os.environ[variable] = "1"
    values = (args.output, args.source_root, args.validation, args.previous_root)
    result = run(*values) if args.execute else build_plan(*values)
    print(base.json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
