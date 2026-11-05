"""新独立 u=.2 HMC＋lazy embedded-Wolff QCR；从头运行，不恢复旧门控实验。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import run_phi4_u02_qcr as legacy


MODULE = "scripts.run_phi4_u02_hybrid_qcr"
HOST = "thinklab-105-225"
GPU_INDEX = 1
GPU_UUID = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"
STEPS, PROFILE_STEPS = 3000, 20
CPU_SECONDS, GPU_SECONDS = 7200, 1800
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024 ** 3, 1024 ** 3
FORCE_SITE_CAP = 2_000_000_000
PRIOR_HELD_PILOT_FORCE_SITES = 8_491_392
PRIOR_DIAGNOSTIC_FORCE_SITES = 313_061_760
common = legacy.common
now, read, save, sha = legacy.now, legacy.read, legacy.save, legacy.sha
source_hashes, unchanged = legacy.source_hashes, legacy.unchanged
directory_bytes, cpu_seconds = legacy.directory_bytes, legacy.cpu_seconds
admission, training_command = legacy.admission, legacy.training_command
validate_training_summary, sampling_validity = legacy.validate_training_summary, legacy.sampling_validity
sampling_quality, hold = legacy.sampling_quality, legacy.hold


def validate_target(index, uuid):
    legacy.validate_device(index, uuid)
    if index != GPU_INDEX or uuid != GPU_UUID:
        raise ValueError("This independent experiment is pinned to physical GPU1 and its recorded UUID")


def physical_options():
    from recurrent_rg.data.phi4_hybrid_pipeline import Phi4HybridOptions
    values = legacy.physical_options()
    from dataclasses import asdict
    return Phi4HybridOptions(**asdict(values), cluster_probability=.5)


def prepare(root, *, gpu_index, expected_gpu_uuid):
    from recurrent_rg.data.phi4_hybrid_pipeline import generation_plan
    legacy.validate_host()
    validate_target(gpu_index, expected_gpu_uuid)
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing output is never overwritten or resumed")
    if root == code or root in code.parents:
        raise ValueError("Experiment output must not contain or replace the code snapshot")
    physical = generation_plan(physical_options())
    preflight = dict(r_values=[-.5, .3], initial_states=["hot", "cold_positive", "cold_negative"],
        samples=8, burn_in=128, spacing=4, seed_base=202610270300, expected_pilots=6,
        cluster_probability=.5, throughput_safety_factor=2., minimum_production_acceptance=.6,
        calibration_uses_only_TRAIN_endpoints=True, heldout_conditions_used=False,
        short_pilots_do_not_certify_equilibrium=True)
    preflight_trajectories = 6 * (128 + 8 * 4)
    preflight_forces = preflight_trajectories * 13 * 36 ** 2
    total_forces = physical["total_force_site_evaluations"] + preflight_forces
    recorded_total = total_forces + PRIOR_HELD_PILOT_FORCE_SITES + PRIOR_DIAGNOSTIC_FORCE_SITES
    if total_forces > FORCE_SITE_CAP or recorded_total > FORCE_SITE_CAP:
        raise ValueError("New fixed production/pilot and recorded historical diagnostic work exceed force cap")
    plan = dict(schema_version=5, recipe="phi4_u02_hmc_lazy_wolff_qcr_v1", prepared_at=now(),
        code_root=str(code), output=str(root), host=HOST, gpu_index=GPU_INDEX,
        device=f"cuda:{GPU_INDEX}", gpu_uuid=GPU_UUID, source_sha256=source_hashes(code),
        physical=physical, sampler_preflight=preflight,
        total_planned_hmc_trajectories_including_preflight=physical["total_hmc_trajectories"] + preflight_trajectories,
        total_planned_force_site_evaluations_including_preflight=total_forces,
        recorded_physical_work_ledger=dict(original_held_preflight_force_sites=PRIOR_HELD_PILOT_FORCE_SITES,
            independent_sampler_diagnostic_force_sites=PRIOR_DIAGNOSTIC_FORCE_SITES,
            new_formal_root_planned_force_sites=total_forces, recorded_total_force_sites=recorded_total,
            force_site_cap=FORCE_SITE_CAP, remaining_recorded_force_site_reserve=FORCE_SITE_CAP-recorded_total,
            scope="Named physical pilot/diagnostic/production work only; tiny unit tests are execution checks, not part of this scientific work ledger"),
        training=dict(steps=STEPS, profile_steps=PROFILE_STEPS, theory_dim=1, hidden=96,
            lattice_sizes=[36, 12, 4], system_batch=8, sample_batch=64, seed=91, profile_seed=191,
            optimizer="three independent fresh Adam", learning_rate=.001, schedule="constant",
            independent_gradients=True, closure_objective="all_start_anchor", closure_scale_weights=None,
            labels_used=False, warm_start=False, profile_checkpoint_used_for_formal=False, pca_initialization=False),
        resource_policy=dict(cpu_threads=1, nice=10, gpu_peak_bytes=GPU_PEAK_BYTES,
            gpu_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS, directory_bytes=DIRECTORY_BYTES,
            force_site_evaluations=FORCE_SITE_CAP, minimum_host_available_gib=12, gpu_free_mib=12288,
            gpu_compute_processes=0, io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, sampling_cpu_projection_multiplier=2.,
            other_cpu_reserve_seconds=600, occupied_resources="wait without interference or migration"),
        authorization="User explicitly authorized a new formal u=.2 hybrid-sampler unsupervised QCR training experiment",
        scientific_limits=["全新独立实验，不恢复或覆盖旧预检门控实验及采样诊断",
            "固定 HMC .08×12 后以概率 .5 作 embedded-Ising Wolff；保留拒绝态，无额外全局 Z2 增强",
            "预检只使用 TRAIN 两端点且延长预热至128，不按 seed/ESS 自动调参",
            "生产质量门控保留原 ESS32、接受率.6、热冷三SE、符号变化及八观测量要求；问题时不启动 GPU",
            "有限链诊断通过不认证平衡；频繁符号翻转不证明能量/偶观测量已混合",
            "连续场 C 与 Q/R loss 不变；一维闭合与数值根不认证相变或临界指数；thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now()), exclusive=True)
    return plan


def sampling_command(root, plan):
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_hybrid", "--execute", "--output", str(root / "data")]
    for name, value in plan["physical"]["options"].items():
        command.append("--" + name.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def planned_commands(root, plan):
    return [
        ("sampler_preflight", [sys.executable, "-u", "-m", MODULE, "--sampler-preflight", "--output", str(root)], False),
        ("physical_sampling", sampling_command(root, plan), False),
        ("training_preflight", training_command(root, plan, profile=True), True),
        ("training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr", "--run", str(root / "run"),
            "--output", str(root / "report")], False),
        ("phi4_hybrid_report", [sys.executable, "-u", "-m", "scripts.report_phi4_hybrid_qcr", "--data", str(root / "data"),
            "--run", str(root / "run"), "--output", str(root / "calibration_report")], False)]


def sampler_preflight(root):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    if socket.gethostname() != plan["host"]:
        raise ValueError("Preflight must use the frozen host")
    unchanged(plan)
    opts, spec = plan["physical"]["options"], plan["sampler_preflight"]
    cpu_started, wall_started = time.process_time(), time.monotonic()
    rows, numerical_error = [], None
    try:
        for r in spec["r_values"]:
            for initial in spec["initial_states"]:
                row = sample_hybrid_chain(r, .2, h=0., size=36, dimension=2,
                    samples=spec["samples"], burn_in=spec["burn_in"], spacing=spec["spacing"],
                    step_size=opts["step_size"], leapfrog_steps=opts["leapfrog_steps"],
                    cluster_probability=opts["cluster_probability"], seed=spec["seed_base"] + len(rows), initial=initial)
                record = {key: row[key] for key in ("seed", "hmc_trajectories", "force_evaluations",
                    "force_site_evaluations", "cpu_seconds", "wall_seconds", "production_acceptance",
                    "burn_in_acceptance", "max_absolute_delta_h", "cluster_probability", "wolff_cluster_updates",
                    "total_cluster_sites", "total_cluster_neighbor_visits", "total_cluster_bond_tests")}
                record.update(r=r, u=.2, initial=initial, cluster_opportunities=row["hmc_trajectories"],
                              execution_only_not_scientific_data=True)
                if any(not math.isfinite(float(record[key])) for key in
                       ("cpu_seconds", "wall_seconds", "production_acceptance", "max_absolute_delta_h")):
                    raise FloatingPointError("Nonfinite hybrid preflight diagnostics")
                if record["cluster_probability"] != .5:
                    raise RuntimeError("Hybrid preflight changed the frozen lazy cluster probability")
                rows.append(record)
                print(json.dumps(dict(event="phi4_hybrid_preflight_chain_complete", **record)), flush=True)
    except ArithmeticError as error:
        numerical_error = f"{type(error).__name__}: {error}"
    actual_cpu = time.process_time() - cpu_started
    complete = len(rows) == spec["expected_pilots"]
    acceptance_passed = complete and all(row["production_acceptance"] >= .6 for row in rows)
    projection = (max(row["cpu_seconds"] / row["hmc_trajectories"] for row in rows)
        * plan["physical"]["total_hmc_trajectories"] * 2.) if rows and numerical_error is None else None
    total_projection = projection + actual_cpu + GPU_SECONDS + 600 if projection is not None else None
    budget_passed = total_projection is not None and total_projection <= CPU_SECONDS
    valid = numerical_error is None and acceptance_passed
    status = "invalid_sampling_hold" if not valid else "complete" if budget_passed else "budget_hold"
    lower = dict(hmc_trajectories=sum(row["hmc_trajectories"] for row in rows),
        force_site_evaluations=sum(row["force_site_evaluations"] for row in rows),
        wolff_cluster_updates=sum(row["wolff_cluster_updates"] for row in rows),
        cluster_neighbor_visits=sum(row["total_cluster_neighbor_visits"] for row in rows))
    receipt = dict(status=status, rows=rows, physical_production_started=False,
        completed_fixed_pilots=len(rows), expected_fixed_pilots=spec["expected_pilots"], numerical_error=numerical_error,
        acceptance_passed=acceptance_passed, minimum_production_acceptance=.6,
        preflight_cpu_seconds=actual_cpu, preflight_wall_seconds=time.monotonic()-wall_started,
        projected_production_cpu_seconds_conservative=projection, projected_total_cpu_seconds_conservative=total_projection,
        throughput_includes_actual_lazy_cluster_cost=True, throughput_safety_factor=2.,
        remaining_gpu_cpu_reserve_seconds=GPU_SECONDS, other_cpu_reserve_seconds=600,
        cpu_budget_seconds=CPU_SECONDS, budget_passed=budget_passed,
        planned_production_trajectories=plan["physical"]["total_hmc_trajectories"],
        actual_total_work_known=numerical_error is None,
        actual_preflight_work=lower if numerical_error is None else None,
        completed_pilots_work_lower_bound=lower, calibration_uses_only_TRAIN_endpoints=True,
        short_pilots_do_not_certify_equilibrium=True, sampler_parameters_not_adapted=True,
        requires_user_action=status != "complete")
    unchanged(plan)
    save(root / "sampler_preflight.json", receipt, exclusive=True)
    return receipt


def final_public_files(root):
    files = legacy.final_public_files(root)
    quality = root / "data" / "quality_summary.json"
    if quality.is_file():
        files[str(quality.relative_to(root))] = sha(quality)
    return files


def execute(root):
    legacy.validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if plan["host"] != HOST or plan["device"] != f"cuda:{GPU_INDEX}":
        raise ValueError("Physical target differs from the frozen plan")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Workflow already executed; no restart")
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST, started_at=now(),
                     plan_sha256=sha(root / "plan.json"))
    save(root / "execution.json", execution, exclusive=True)
    original_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if original_nice < 10:
        os.nice(10-original_nice)
    started, cpu_started, gpu_elapsed = time.monotonic(), cpu_seconds(), 0.
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        for name, command, gpu in planned_commands(root, plan):
            unchanged(plan)
            if directory_bytes(root) > DIRECTORY_BYTES or cpu_seconds()-cpu_started >= CPU_SECONDS or (gpu and gpu_elapsed >= GPU_SECONDS):
                hold(job, "Actual accumulated resources reached the fixed budget; no work reduction or deletion")
                break
            admission(root, plan, job, name, gpu)
            if gpu:
                command[command.index("--max-training-seconds")+1] = str(GPU_SECONDS-gpu_elapsed)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = root / "logs" / f"{name}.log"
            stage_started, stage_cpu = time.monotonic(), cpu_seconds()
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, status="running", pid=child.pid, command=command,
                    started_at=now(), gpu=gpu, log=str(log))
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(root / "job.json", job)
                returncode = child.wait()
            stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                ended_at=now(), wall_seconds=time.monotonic()-stage_started, cpu_seconds=cpu_seconds()-stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds()-cpu_started,
                       directory_bytes=directory_bytes(root))
            save(root / "job.json", job)
            if returncode:
                folder = ("profile_run" if name == "training_preflight" else "run" if name == "training" else
                    "data" if name == "physical_sampling" else "calibration_report" if name == "phi4_hybrid_report" else "report")
                path = root / folder / "status.json"
                status = read(path) if path.is_file() else {}
                if status.get("status") == "budget_hold":
                    hold(job, f"{name} reached its fixed resource budget; outputs preserved")
                    break
                if name == "physical_sampling" and any(word in status.get("error", "").lower()
                        for word in ("floatingpointerror", "overflowerror", "nonfinite", "non-finite")):
                    hold(job, f"Physical sampling numerical instability: {status.get('error')}; no training",
                         status="invalid_sampling_hold")
                    break
                raise RuntimeError(f"{name} returned {returncode}; no automatic retry")
            unchanged(plan)
            if name == "sampler_preflight":
                receipt = read(root / "sampler_preflight.json")
                job["sampler_preflight"] = receipt
                if not receipt["acceptance_passed"] or receipt.get("numerical_error"):
                    hold(job, "Fixed hybrid sampler preflight invalid; production not started or adapted",
                         status="invalid_sampling_hold")
                    break
                if not receipt["budget_passed"]:
                    hold(job, "Conservative hybrid sampling CPU projection exceeds budget; production not started")
                    break
            elif name == "physical_sampling":
                status = read(root / "data" / "status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Physical hybrid sampling did not finish successfully")
                validity = sampling_validity(root, plan)
                save(root / "sampling_validity.json", validity, exclusive=True)
                job.update(sampling_warning_count=status.get("warning_count", 0), sampling_actual_cpu_seconds=status["cpu_seconds"])
                if validity["severe"]:
                    hold(job, "Production has zero acceptance/nonfinite evidence; no training or repair", status="invalid_sampling_hold")
                    break
                quality = sampling_quality(root)
                save(root / "sampling_quality.json", quality, exclusive=True)
                job["sampling_quality_issue_count"] = len(quality["issues"])
                if quality["issues"]:
                    hold(job, "Existing sampling quality diagnostics require user direction; no GPU profile/training",
                         status="sampling_quality_hold")
                    break
                if cpu_seconds()-cpu_started+GPU_SECONDS+600 > CPU_SECONDS:
                    hold(job, "Actual hybrid sampling CPU plus remaining conservative reserve exceeds CPU budget")
                    break
            elif name in ("training_preflight", "training"):
                profile = name == "training_preflight"
                folder = root / ("profile_run" if profile else "run")
                summary = read(folder / "summary.json")
                validate_training_summary(summary, plan, profile=profile)
                if profile:
                    receipt = common.profile_projection(summary, read(folder / "history_progress.json"), stage["wall_seconds"], gpu_elapsed)
                    save(root / "gpu_preflight.json", receipt, exclusive=True)
                    job["gpu_projection"] = receipt
                    if not receipt["budget_passed"]:
                        hold(job, "GPU projection exceeds 1800 seconds; fixed steps not reduced")
                        break
                    if cpu_seconds()-cpu_started+receipt["projected_formal_gpu_seconds_conservative"]+600 > CPU_SECONDS:
                        hold(job, "Actual CPU plus conservative remaining work exceeds 7200 seconds")
                        break
                else:
                    job["actual_training_steps"] = STEPS
            else:
                path = root / ("calibration_report" if name == "phi4_hybrid_report" else "report") / "status.json"
                status = read(path)
                if status.get("status") != "complete" or status.get("workflow_complete") is not True:
                    raise RuntimeError(f"{name} public report is incomplete")
            if gpu_elapsed > GPU_SECONDS or cpu_seconds()-cpu_started > CPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
                hold(job, "Actual resource accounting exceeded its fixed budget; completed artifacts preserved")
                break
        else:
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, error=f"{type(error).__name__}: {error}",
                   requires_user_action=True, ended_at=now())
        raise
    finally:
        job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds()-cpu_started,
                   wall_seconds=time.monotonic()-started, directory_bytes=directory_bytes(root))
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"], ended_at=now(),
            cpu_seconds=job["cpu_seconds"], gpu_stage_wall_seconds=gpu_elapsed, wall_seconds=job["wall_seconds"],
            directory_bytes=job["directory_bytes"], actual_nice=job["actual_nice"], error=job["error"],
            requires_user_action=job["requires_user_action"])
        save(root / "execution.json", execution)
        if job["status"] == "complete":
            job["public_files"] = final_public_files(root)
        save(root / "job.json", job)


def launch(root):
    legacy.validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if job["status"] != "prepared" or job["stages"] or (root / "execution.json").exists():
        raise ValueError("Workflow already executed; no restart")
    unchanged(plan)
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--output", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
            plan_sha256=sha(root / "plan.json"), new_experiment=True, resumed=False)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute", "sampler-preflight"):
        modes.add_argument("--"+mode, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--expected-gpu-uuid")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.gpu_index is None or args.expected_gpu_uuid is None:
            parser.error("--prepare requires explicit --gpu-index and --expected-gpu-uuid")
        plan = prepare(args.output, gpu_index=args.gpu_index, expected_gpu_uuid=args.expected_gpu_uuid)
        value = dict(status="prepared", output=plan["output"], source_files=len(plan["source_sha256"]))
    elif args.launch:
        value = launch(args.output)
    elif args.sampler_preflight:
        value = sampler_preflight(args.output)
    else:
        execute(args.output)
        value = read(args.output / "job.json")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if args.sampler_preflight or value["status"] not in (
        "failed", "budget_hold", "invalid_sampling_hold", "sampling_quality_hold") else 3


if __name__ == "__main__":
    sys.exit(main())
