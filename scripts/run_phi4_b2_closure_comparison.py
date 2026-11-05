"""全新 b=2 三步 φ⁴ 闭环定义对照：固定采样、配对训练、只读报告。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from scripts import run_phi4_u02_qcr as support


MODULE = "scripts.run_phi4_b2_closure_comparison"
HOST = "thinklab-105-225"
GPU_INDEX = 2
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
STEPS, PROFILE_STEPS = 3000, 20
CPU_SECONDS, GPU_SECONDS = 7200, 1800
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024 ** 3, 1024 ** 3
REPORT_RESERVE_SECONDS, GPU_ADMISSION_RESERVE_SECONDS = 300, 40
now, read, save, sha = support.now, support.read, support.save, support.sha
source_hashes, unchanged = support.source_hashes, support.unchanged
directory_bytes, cpu_seconds = support.directory_bytes, support.cpu_seconds
sampling_validity, hold = support.sampling_validity, support.hold


def validate_host():
    if socket.gethostname() != HOST:
        raise ValueError(f"This frozen experiment is for {HOST} only")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("A CUDA mask must not reinterpret the frozen physical GPU index")


def validate_target(index, uuid):
    support.validate_device(index, uuid)
    if index != GPU_INDEX or uuid != GPU_UUID:
        raise ValueError("This experiment is pinned to physical GPU2 and its frozen UUID")


def physical_options():
    from recurrent_rg.data.phi4_hybrid_pipeline import Phi4HybridOptions
    return Phi4HybridOptions(sizes=(32,))


def prepare(root, *, gpu_index, expected_gpu_uuid):
    from recurrent_rg.data.phi4_hybrid_pipeline import generation_plan
    validate_host()
    validate_target(gpu_index, expected_gpu_uuid)
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing experiment outputs are never overwritten or resumed")
    if root == code or root in code.parents:
        raise ValueError("Output must not contain or replace the frozen source snapshot")
    physical = generation_plan(physical_options())
    pilots = dict(r_values=[-.5, .3], initial_states=["hot", "cold_positive", "cold_negative"],
        size=32, samples=8, burn_in=128, spacing=4, seed_base=202610270700,
        expected_pilots=6, throughput_safety_factor=2., minimum_production_acceptance=.6,
        calibration_uses_only_TRAIN_endpoints=True, heldout_conditions_used=False,
        short_pilots_do_not_certify_equilibrium=True)
    pilot_cycles = 6 * (128 + 8 * 4)
    force_sites = physical["total_force_site_evaluations"] + pilot_cycles * 13 * 32 ** 2
    if force_sites > physical["options"]["max_force_site_evaluations"]:
        raise ValueError("Production plus fixed pilots exceed the force-site cap")
    plan = dict(schema_version=5, recipe="phi4_b2_closure_objective_comparison_v1",
        prepared_at=now(), host=HOST, code_root=str(code), output=str(root),
        gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID, device=f"cuda:{GPU_INDEX}",
        source_sha256=source_hashes(code), physical=physical, sampler_preflight=pilots,
        total_planned_hmc_trajectories_including_preflight=physical["total_hmc_trajectories"] + pilot_cycles,
        total_planned_force_site_evaluations_including_preflight=force_sites,
        training=dict(steps_per_arm=STEPS, profile_steps_per_arm=PROFILE_STEPS,
            arms={"legacy": "one_step_and_rollout", "all_start": "all_start_anchor"},
            lattice_sizes=[32, 16, 8, 4], block_size=2, stride=2, theory_dim=1,
            hidden=96, seed=91, profile_seed=191, system_batch=8, sample_batch=64,
            learning_rate=.001, schedule="constant", labels_used=False,
            warm_start=False, pca_initialization=False, q_initialization="train_mean_bias",
            closure_scale_weights=None, commutator=5., rollout_weight=5.,
            independent_gradients=True, q_closure_feedback=False,
            same_random_initial_state=True, paired_minibatches_and_C_draws=True,
            Q_C_losses_checked_every_step=True,
            Q_C_fingerprints_checked_at_progress_and_endpoint=True, fresh_optimizers_per_arm=True,
            profile_checkpoint_used_for_formal=False),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS,
            gpu_seconds=GPU_SECONDS, gpu_peak_bytes=GPU_PEAK_BYTES, directory_bytes=DIRECTORY_BYTES,
            minimum_host_available_gib=36, gpu_free_mib=12288, gpu_compute_processes=0,
            load_per_cpu_max=1.25, io_full_avg10_percent_max=20,
            two_gpu_checks_seconds_apart=10, sampling_cpu_projection_multiplier=2.,
            formal_gpu_projection_multiplier=1.25, other_cpu_reserve_seconds=600,
            report_reserve_seconds=REPORT_RESERVE_SECONDS,
            gpu_admission_reserve_seconds=GPU_ADMISSION_RESERVE_SECONDS,
            occupied_resources="wait without interfering, deleting, renicing, or migrating"),
        sampling_quality_policy=dict(allow_quality_warnings=True,
            authorization="用户此前明确允许保留既有 ESS／热冷质量警告，继续既定探索性训练",
            severe_nonfinite_or_zero_acceptance="stop before GPU work",
            automatic_resampling=False, automatic_repair=False, equilibrium_certified=False),
        authorization="用户明确授权 b=2、32→16→8→4 的早期一步＋多步与 all-start 独立对照",
        scientific_limits=["新 L32 数据、新 b2 核；不覆盖任何历史 L36/b3 数据、模型或报告",
            "只比较闭环定义：随机且可训练 C、Q 表示目标及独立梯度路由保持相同",
            "有限链警告全部保留，不认证平衡；不按 ESS、流场或不动点补样、换 seed 或改参数",
            "20步 profile 仅作执行与预算诊断，不作为正式科学结果或正式模型的热启动",
            "单 seed、一维表示及终点 L4 的限制保留；critical_r=null，thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False,
        current=None, stages=[], error=None, requires_user_action=False, prepared_at=now()), exclusive=True)
    return plan


def resource_snapshot(plan):
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io_full = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    query = subprocess.check_output(["nvidia-smi", "-i", str(plan["gpu_index"]),
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    actual_uuid, free_mib = query[0].strip(), int(query[1])
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    pids = [int(row.split(",")[1]) for row in rows if actual_uuid in row]
    policy = plan["resource_policy"]
    host_ready = (available >= policy["minimum_host_available_gib"] * 1024 ** 3
        and load <= policy["load_per_cpu_max"] * cpus and io_full < policy["io_full_avg10_percent_max"])
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=actual_uuid, gpu_free_mib=free_mib,
        gpu_compute_pids=pids, host_ready=host_ready,
        gpu_ready=host_ready and actual_uuid == plan["gpu_uuid"]
            and free_mib >= policy["gpu_free_mib"] and not pids)


def admission(root, plan, job, stage, gpu):
    consecutive = 0
    while True:
        row = resource_snapshot(plan)
        if row["gpu_uuid"] != plan["gpu_uuid"]:
            raise RuntimeError("Physical GPU UUID differs from the frozen plan")
        consecutive = consecutive + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if consecutive >= (2 if gpu else 1):
            return
        time.sleep(10 if consecutive else 30)


def sampling_command(root, plan):
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_hybrid", "--execute", "--output", str(root / "data")]
    for name, value in plan["physical"]["options"].items():
        command.append("--" + name.replace("_", "-"))
        command.extend(str(item) for item in value) if isinstance(value, (list, tuple)) else command.append(str(value))
    return command


def training_command(root, plan, *, profile, remaining_cpu_seconds=CPU_SECONDS, remaining_gpu_seconds=GPU_SECONDS):
    command = [sys.executable, "-u", "-m", "experiments.phi4_b2_closure_comparison",
        "--data", str(root / "data"), "--output", str(root / ("profile" if profile else "training")),
        "--device", plan["device"], "--expected-gpu-uuid", plan["gpu_uuid"],
        "--steps", str(PROFILE_STEPS if profile else STEPS),
        "--seed", "191" if profile else "91",
        "--cpu-budget-seconds", str(remaining_cpu_seconds), "--gpu-budget-seconds", str(remaining_gpu_seconds)]
    if profile:
        command.append("--profile-only")
    return command


def planned_commands(root, plan):
    return [
        ("sampler_preflight", [sys.executable, "-u", "-m", MODULE, "--sampler-preflight", "--output", str(root)], False),
        ("sampling", sampling_command(root, plan), False),
        ("training_preflight", training_command(root, plan, profile=True), True),
        ("paired_training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_b2_closure_comparison",
            "--training", str(root / "training"), "--output", str(root / "report")], False)]


def sampler_preflight(root):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    if socket.gethostname() != plan["host"]:
        raise ValueError("Sampler preflight must use the frozen host")
    unchanged(plan)
    opts, spec = plan["physical"]["options"], plan["sampler_preflight"]
    cpu_started, wall_started = time.process_time(), time.monotonic()
    rows, numerical_error = [], None
    try:
        for r in spec["r_values"]:
            for initial in spec["initial_states"]:
                result = sample_hybrid_chain(r, .2, h=0., size=spec["size"], dimension=2,
                    samples=spec["samples"], burn_in=spec["burn_in"], spacing=spec["spacing"],
                    step_size=opts["step_size"], leapfrog_steps=opts["leapfrog_steps"],
                    cluster_probability=opts["cluster_probability"], seed=spec["seed_base"] + len(rows), initial=initial)
                record = {key: result[key] for key in ("seed", "hmc_trajectories", "force_evaluations",
                    "force_site_evaluations", "cpu_seconds", "wall_seconds", "production_acceptance",
                    "burn_in_acceptance", "max_absolute_delta_h", "cluster_probability",
                    "wolff_cluster_updates", "total_cluster_neighbor_visits")}
                record.update(r=r, u=.2, initial=initial, execution_only_not_scientific_data=True)
                if any(not math.isfinite(float(record[key])) for key in
                       ("cpu_seconds", "wall_seconds", "production_acceptance", "max_absolute_delta_h")):
                    raise FloatingPointError("Nonfinite fixed sampler preflight diagnostics")
                expected_cycles = spec["burn_in"] + spec["samples"] * spec["spacing"]
                if (record["cluster_probability"] != .5 or record["hmc_trajectories"] != expected_cycles
                        or record["force_site_evaluations"] != expected_cycles * 13 * spec["size"] ** 2):
                    raise RuntimeError("Fixed sampler settings or work counters changed")
                rows.append(record)
                print(json.dumps(dict(event="sampler_preflight_chain_complete", **record)), flush=True)
    except ArithmeticError as error:
        numerical_error = f"{type(error).__name__}: {error}"
    actual_cpu = time.process_time() - cpu_started
    complete = len(rows) == spec["expected_pilots"]
    acceptance_passed = complete and all(row["production_acceptance"] >= .6 for row in rows)
    projection = (max(row["cpu_seconds"] / row["hmc_trajectories"] for row in rows)
        * plan["physical"]["total_hmc_trajectories"] * spec["throughput_safety_factor"]
        if rows and numerical_error is None else None)
    total_projection = projection + actual_cpu + GPU_SECONDS + 600 if projection is not None else None
    budget_passed = total_projection is not None and total_projection <= CPU_SECONDS
    valid = numerical_error is None and acceptance_passed
    status = "invalid_sampling_hold" if not valid else "complete" if budget_passed else "budget_hold"
    lower = {key: sum(row[key] for row in rows) for key in
        ("hmc_trajectories", "force_site_evaluations", "wolff_cluster_updates", "total_cluster_neighbor_visits")}
    receipt = dict(status=status, workflow_complete=status == "complete", rows=rows,
        physical_production_started=False, completed_fixed_pilots=len(rows), expected_fixed_pilots=6,
        numerical_error=numerical_error, acceptance_passed=acceptance_passed,
        preflight_cpu_seconds=actual_cpu, preflight_wall_seconds=time.monotonic()-wall_started,
        projected_production_cpu_seconds_conservative=projection,
        projected_total_cpu_seconds_conservative=total_projection, cpu_budget_seconds=CPU_SECONDS,
        budget_passed=budget_passed, actual_total_work_known=numerical_error is None,
        actual_preflight_work=lower if numerical_error is None else None,
        completed_pilots_work_lower_bound=lower, sampler_parameters_not_adapted=True,
        calibration_uses_only_TRAIN_endpoints=True, short_pilots_do_not_certify_equilibrium=True,
        requires_user_action=status != "complete")
    unchanged(plan)
    save(root / "sampler_preflight.json", receipt, exclusive=True)
    return receipt


def sampling_quality(root):
    receipt = support.sampling_quality(root)
    receipt.update(status="complete_with_warnings" if receipt["issues"] else "complete",
        continuation_authorized=True, training_started=False, GPU_profile_started=False,
        qualification="既有 ESS／热冷等质量警告完整保留；按用户授权仅作探索性训练，不认证平衡，不自动补样。")
    return receipt


def profile_projection(profile, *, actual_profile_stage_seconds, gpu_elapsed):
    if (profile.get("status") != "complete" or profile.get("workflow_complete") is not True
            or profile.get("profile_only") is not True or profile.get("steps_per_arm") != PROFILE_STEPS
            or profile.get("arms") != 2 or profile.get("seed") != 191
            or profile.get("C_Q_pairing_verified") is not True
            or profile.get("gradient_routing_verified") is not True):
        raise RuntimeError("Independent two-arm profile receipt is incomplete")
    values = [profile.get(key) for key in ("update_seconds", "initialization_seconds", "peak_cuda_bytes")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("Nonfinite or missing profile resource counters")
    updates, initialization, peak = values
    if updates <= 0:
        raise RuntimeError("Profile update timing must be strictly positive")
    if peak > GPU_PEAK_BYTES:
        raise RuntimeError("Profile own GPU peak exceeds the fixed 8GiB cap")
    # Two arms are profiled and trained together. Keep one-time setup separate.
    formal = 1.25 * (initialization + updates / (2 * PROFILE_STEPS) * (2 * STEPS))
    total = gpu_elapsed + formal + REPORT_RESERVE_SECONDS + GPU_ADMISSION_RESERVE_SECONDS
    return dict(status="complete", budget_passed=total <= GPU_SECONDS,
        actual_profile_stage_seconds=actual_profile_stage_seconds,
        actual_accumulated_gpu_stage_seconds=gpu_elapsed,
        measured_update_seconds=updates, measured_steps_per_arm=PROFILE_STEPS,
        measured_initialization_seconds=initialization, throughput_safety_factor=1.25,
        projected_formal_gpu_seconds_conservative=formal,
        projected_total_gpu_seconds_conservative=total, gpu_budget_seconds=GPU_SECONDS,
        report_reserve_seconds=REPORT_RESERVE_SECONDS,
        gpu_admission_reserve_seconds=GPU_ADMISSION_RESERVE_SECONDS,
        peak_cuda_bytes=peak, profile_is_execution_check_not_scientific_result=True,
        profile_parameters_used_for_formal=False)


def validate_training_receipt(value, plan, *, profile):
    device = value.get("device")
    uuid = value.get("gpu_uuid") or (device.get("uuid") if isinstance(device, dict) else None)
    if str(uuid).lower().removeprefix("gpu-") != plan["gpu_uuid"].lower().removeprefix("gpu-"):
        raise RuntimeError("Actual trainer physical GPU differs from the frozen plan")
    if value.get("peak_cuda_bytes", GPU_PEAK_BYTES + 1) > GPU_PEAK_BYTES:
        raise RuntimeError("Own CUDA peak exceeds the fixed 8GiB cap")
    if profile:
        return
    if (value.get("status") != "complete"
            or value.get("actual_steps") != STEPS
            or value.get("actual_steps_by_arm") != {"legacy": STEPS, "all_start": STEPS}
            or value.get("C_Q_pairing_verified") is not True
            or value.get("pairing_verified") is not True
            or value.get("gradient_routing_verified") is not True
            or value.get("labels_used_in_training") is not False
            or value.get("closure_modes") != plan["training"]["arms"]):
        raise RuntimeError("Formal endpoint, pairing, gradient routing or label-free contract mismatch")


def final_public_files(root):
    """Strict public extensions; never enumerate raw physical or model arrays."""
    root = Path(root).resolve()
    result = {}
    for name in ("plan.json", "execution.json", "launch.json", "sampler_preflight.json",
                 "sampling_validity.json", "sampling_quality.json", "gpu_preflight.json"):
        path = root / name
        if path.is_file():
            result[name] = sha(path)
    manifest = root / "data/manifest.json"
    if manifest.is_file():
        for name in read(manifest).get("public_files", []):
            path = (root / "data" / name).resolve()
            if not path.is_relative_to(root / "data") or path.suffix != ".json" or not path.is_file():
                raise RuntimeError("Physical public inventory contains a non-JSON or unsafe reference")
            result[str(path.relative_to(root))] = sha(path)
    for directory in ("profile", "training", "report"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink():
                raise RuntimeError("Public outputs must not include symlinks")
            if path.is_file() and path.suffix in (".json", ".png", ".npz"):
                if path.name in ("data.npz", "configurations.npz"):
                    raise RuntimeError("Raw physical configuration archives are not public")
                result[str(path.relative_to(root))] = sha(path)
    return result


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if plan["host"] != HOST or plan["device"] != f"cuda:{GPU_INDEX}":
        raise ValueError("Physical target differs from the frozen plan")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Already executed; no restart, resume, or overwrite")
    unchanged(plan)
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST,
        started_at=now(), plan_sha256=sha(root / "plan.json"))
    save(root / "execution.json", execution, exclusive=True)
    initial_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if initial_nice < 10:
        os.nice(10 - initial_nice)
    started, cpu_started, gpu_elapsed = time.monotonic(), cpu_seconds(), 0.
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(),
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        for name, command, gpu in planned_commands(root, plan):
            unchanged(plan)
            if (directory_bytes(root) > DIRECTORY_BYTES or cpu_seconds()-cpu_started >= CPU_SECONDS
                    or (gpu and gpu_elapsed >= GPU_SECONDS)):
                hold(job, "Fixed resource budget reached; outputs preserved, no work reduction")
                break
            admission(root, plan, job, name, gpu)
            if gpu:
                command[command.index("--cpu-budget-seconds") + 1] = str(CPU_SECONDS-(cpu_seconds()-cpu_started)-REPORT_RESERVE_SECONDS)
                command[command.index("--gpu-budget-seconds") + 1] = str(GPU_SECONDS-gpu_elapsed-REPORT_RESERVE_SECONDS)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = root / "logs" / f"{name}.log"
            stage_started, stage_cpu = time.monotonic(), cpu_seconds()
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, status="running", pid=child.pid, command=command,
                    gpu=gpu, log=str(log), started_at=now())
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(root / "job.json", job)
                returncode = child.wait()
            stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                ended_at=now(), wall_seconds=time.monotonic()-stage_started,
                cpu_seconds=cpu_seconds()-stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds()-cpu_started,
                directory_bytes=directory_bytes(root))
            save(root / "job.json", job)
            if returncode:
                folder = {"sampling": "data", "training_preflight": "profile",
                    "paired_training": "training", "report": "report"}.get(name)
                child_status = read(root / folder / "status.json") if folder and (root / folder / "status.json").is_file() else {}
                if child_status.get("status") == "budget_hold":
                    hold(job, f"{name} reached its fixed resource budget; no retry")
                    break
                if name == "sampling" and any(word in str(child_status.get("error", "")).lower()
                        for word in ("floatingpointerror", "overflowerror", "nonfinite", "non-finite")):
                    hold(job, "Physical sampling numerical instability; no GPU work or repair",
                        status="invalid_sampling_hold")
                    break
                raise RuntimeError(f"{name} returned {returncode}; no automatic retry")
            unchanged(plan)
            if name == "sampler_preflight":
                receipt = read(root / "sampler_preflight.json")
                job["sampler_preflight"] = receipt
                if not receipt.get("acceptance_passed") or receipt.get("numerical_error"):
                    hold(job, "Sampler preflight invalid; no production or repair", status="invalid_sampling_hold")
                    break
                if not receipt.get("budget_passed"):
                    hold(job, "Fixed sampling CPU projection exceeds budget; production not started")
                    break
            elif name == "sampling":
                status = read(root / "data/status.json")
                if status.get("workflow_complete") is not True or status.get("returncode") != 0:
                    raise RuntimeError("Physical sampling completion receipt is inconsistent")
                validity = sampling_validity(root, plan)
                save(root / "sampling_validity.json", validity, exclusive=True)
                if validity["severe"]:
                    hold(job, "Zero acceptance/nonfinite physical evidence; no GPU work", status="invalid_sampling_hold")
                    break
                quality = sampling_quality(root)
                save(root / "sampling_quality.json", quality, exclusive=True)
                job.update(sampling_warning_count=status.get("warning_count", 0),
                    sampling_quality_issue_count=len(quality["issues"]),
                    sampling_quality_warning_continuation_authorized=True, equilibrium_certified=False)
                if cpu_seconds()-cpu_started + GPU_SECONDS + 600 > CPU_SECONDS:
                    hold(job, "Actual sampling CPU plus remaining conservative reserve exceeds budget")
                    break
            elif name == "training_preflight":
                value = read(root / "profile/profile.json")
                validate_training_receipt(value, plan, profile=True)
                projection = profile_projection(value, actual_profile_stage_seconds=stage["wall_seconds"], gpu_elapsed=gpu_elapsed)
                save(root / "gpu_preflight.json", projection, exclusive=True)
                job["gpu_projection"] = projection
                if not projection["budget_passed"]:
                    hold(job, "Paired GPU projection exceeds budget; fixed steps not shortened")
                    break
                if cpu_seconds()-cpu_started + projection["projected_formal_gpu_seconds_conservative"] + 600 > CPU_SECONDS:
                    hold(job, "Actual CPU plus remaining conservative training/report projection exceeds budget")
                    break
            elif name == "paired_training":
                status = read(root / "training/status.json")
                if (status.get("status") != "complete" or status.get("workflow_complete") is not True
                        or status.get("returncode") != 0):
                    raise RuntimeError("Formal training status is incomplete")
                validate_training_receipt(read(root / "training/summary.json"), plan, profile=False)
                job["actual_training_steps_per_arm"] = {"legacy": STEPS, "all_start": STEPS}
            else:
                status = read(root / "report/status.json")
                if status.get("status") != "complete" or status.get("workflow_complete") is not True:
                    raise RuntimeError("Formal report is incomplete")
            if (gpu_elapsed > GPU_SECONDS or cpu_seconds()-cpu_started > CPU_SECONDS
                    or directory_bytes(root) > DIRECTORY_BYTES):
                hold(job, "Actual resource accounting exceeded budget; completed artifacts preserved")
                break
        else:
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, requires_user_action=True,
            error=f"{type(error).__name__}: {error}", ended_at=now())
        raise
    finally:
        job.update(cpu_seconds=cpu_seconds()-cpu_started, gpu_stage_wall_seconds=gpu_elapsed,
            wall_seconds=time.monotonic()-started, directory_bytes=directory_bytes(root))
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"], ended_at=now(),
            cpu_seconds=job["cpu_seconds"], gpu_stage_wall_seconds=gpu_elapsed,
            wall_seconds=job["wall_seconds"], directory_bytes=job["directory_bytes"],
            actual_nice=job["actual_nice"], error=job["error"], requires_user_action=job["requires_user_action"])
        save(root / "execution.json", execution)
        if job["status"] == "complete":
            job["public_files"] = final_public_files(root)
        save(root / "job.json", job)


def launch(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if job["status"] != "prepared" or job["stages"] or (root / "execution.json").exists():
        raise ValueError("Already executed; no restart or overwrite")
    unchanged(plan)
    with (root / "launch.json").open("x") as receipt, (root / "logs/parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--output", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
            plan_sha256=sha(root / "plan.json"), new_experiment=True, resumed=False)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute", "sampler-preflight"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--expected-gpu-uuid")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.gpu_index is None or args.expected_gpu_uuid is None:
            parser.error("--prepare requires --gpu-index and --expected-gpu-uuid")
        value = prepare(args.output, gpu_index=args.gpu_index, expected_gpu_uuid=args.expected_gpu_uuid)
        value = dict(status="prepared", output=value["output"], source_files=len(value["source_sha256"]))
    elif args.launch:
        value = launch(args.output)
    elif args.sampler_preflight:
        value = sampler_preflight(args.output)
    else:
        execute(args.output)
        value = read(args.output / "job.json")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if args.sampler_preflight or value["status"] in ("prepared", "launched", "complete") else 3


if __name__ == "__main__":
    sys.exit(main())
