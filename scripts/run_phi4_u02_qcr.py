"""独立 u=0.2 标量场 QCR：固定 HMC、资源门控、从头训练；不覆盖或重启。"""
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

import numpy as np

from scripts import run_gaussian_scalar_qcr as common


HOST = "thinklab-105-225"
MODULE = "scripts.run_phi4_u02_qcr"
STEPS, PROFILE_STEPS = 3000, 20
CPU_SECONDS, GPU_SECONDS = 7200, 1800
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024 ** 3, 1024 ** 3
FORCE_SITE_CAP = 2_000_000_000
now, read, save, sha = common.now, common.read, common.save, common.sha
source_hashes, unchanged = common.source_hashes, common.unchanged
directory_bytes, cpu_seconds = common.directory_bytes, common.cpu_seconds


def validate_host():
    if socket.gethostname() != HOST:
        raise ValueError(f"This physical-device plan is for {HOST} only")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Do not reinterpret physical GPU indices through a CUDA mask")


def validate_device(index, uuid):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("An explicit nonnegative physical GPU index is required")
    if not isinstance(uuid, str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise ValueError("An explicit physical GPU UUID is required")


def physical_options():
    from recurrent_rg.data.phi4 import Phi4Options
    return Phi4Options(sizes=(36,), dimension=2,
        r_values=tuple(float(value) for value in np.linspace(-.5, .3, 12)),
        u_values=(.2,), h=0., chains=4, samples=128, burn_in=1024, spacing=8,
        step_size=.08, leapfrog_steps=12, seed=91, test_every=3, test_offset=1,
        max_force_site_evaluations=FORCE_SITE_CAP, max_output_bytes=DIRECTORY_BYTES)


def prepare(root, *, gpu_index, expected_gpu_uuid):
    from recurrent_rg.data.phi4 import generation_plan
    validate_host()
    validate_device(gpu_index, expected_gpu_uuid)
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing experiment output is never overwritten or resumed")
    if root == code or root in code.parents:
        raise ValueError("Experiment output must not contain or replace the code snapshot")
    physical = generation_plan(physical_options())
    preflight = dict(r_values=[-.5, -.33158, .3],
        initial_states=["hot", "cold_positive", "cold_negative"], samples=8, burn_in=24,
        spacing=4, seed_base=202610270200, throughput_safety_factor=2.,
        minimum_production_acceptance=.6, short_pilots_do_not_certify_equilibrium=True,
        reference_critical_r_is_evaluation_only=True)
    preflight_trajectories = len(preflight["r_values"]) * len(preflight["initial_states"]) * (
        preflight["burn_in"] + preflight["samples"] * preflight["spacing"])
    preflight_force_sites = preflight_trajectories * 13 * 36 ** 2
    if physical["total_force_site_evaluations"] + preflight_force_sites > FORCE_SITE_CAP:
        raise ValueError("Production plus preflight exceeds the fixed force-site cap")
    plan = dict(schema_version=5, recipe="phi4_u02_r_scan_qcr_calibration_v1", prepared_at=now(),
        code_root=str(code), output=str(root), host=HOST, gpu_index=gpu_index,
        device=f"cuda:{gpu_index}", gpu_uuid=expected_gpu_uuid,
        source_sha256=source_hashes(code), physical=physical, sampler_preflight=preflight,
        total_planned_hmc_trajectories_including_preflight=physical["total_hmc_trajectories"] + preflight_trajectories,
        total_planned_force_site_evaluations_including_preflight=physical["total_force_site_evaluations"] + preflight_force_sites,
        training=dict(steps=STEPS, profile_steps=PROFILE_STEPS, theory_dim=1, hidden=96,
            lattice_sizes=[36, 12, 4], system_batch=8, sample_batch=64, seed=91, profile_seed=191,
            optimizer="three independent fresh Adam", learning_rate=.001, schedule="constant",
            independent_gradients=True, closure_objective="all_start_anchor", closure_scale_weights=None,
            labels_used=False, warm_start=False, profile_checkpoint_used_for_formal=False,
            pca_initialization=False),
        resource_policy=dict(cpu_threads=1, nice=10, gpu_peak_bytes=GPU_PEAK_BYTES,
            gpu_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS, directory_bytes=DIRECTORY_BYTES,
            force_site_evaluations=FORCE_SITE_CAP, minimum_host_available_gib=12,
            gpu_free_mib=12288, gpu_compute_processes=0, io_full_avg10_percent_max=20,
            load_per_cpu_max=1.25, two_gpu_checks_seconds_apart=10,
            production_cpu_projection_multiplier=2., preflight_other_cpu_reserve_seconds=600,
            occupied_resources="wait without intervention; no migration"),
        authorization="User explicitly approved unsupervised u=.2, r in [-.5,.3] training after the Gaussian calibration",
        scientific_limits=["固定 u=.2、h=0 的有限尺寸 HMC 数据；r 只用于生成和事后评价",
            "生产链质量问题会在 GPU profile 前科学门控，保留全部证据，不自动加样、换种子或调整 HMC 参数",
            "既有接受率<.6、ESS<32/不可用、热冷三SE警告或无符号翻转需用户明确决定，不能默许不平衡训练",
            "预检接受率小于 .6 或数值失稳时不启动生产；生产零接受率或非有限数值时不强训",
            "连续场 C 的信息代理保留有限样本与带宽偏差；本次不修改 loss",
            "一维 g 与有限尺度闭合不认证临界固定点、热方向或临界指数；thermal_nu=null",
            "独立 seed191 20 步 profile 非科学结果；正式 seed91 从新随机初始化开始"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now()), exclusive=True)
    return plan


def resource_snapshot(plan):
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io_full = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    query = subprocess.check_output(["nvidia-smi", "-i", str(plan["gpu_index"]),
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    actual_uuid, free_mib = query[0].strip(), int(query[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    pids = [int(line.split(",")[1]) for line in processes if actual_uuid in line]
    host_ready = available >= 12 * 1024 ** 3 and load <= 1.25 * cpus and io_full < 20
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=actual_uuid, gpu_free_mib=free_mib,
        gpu_compute_pids=pids, host_ready=host_ready,
        gpu_ready=host_ready and actual_uuid == plan["gpu_uuid"] and free_mib >= 12288 and not pids)


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
    values = plan["physical"]["options"]
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4", "--execute", "--output", str(root / "data")]
    for name, value in values.items():
        command.append("--" + name.replace("_", "-"))
        command.extend(str(item) for item in value) if isinstance(value, (list, tuple)) else command.append(str(value))
    return command


def training_command(root, plan, *, profile, remaining_gpu_seconds=GPU_SECONDS):
    return [sys.executable, "-u", "-m", "experiments.phi4_qcr", "--data", str(root / "data"),
        "--output", str(root / ("profile_run" if profile else "run")), "--device", plan["device"],
        "--steps", str(PROFILE_STEPS if profile else STEPS), "--system-batch", "8", "--sample-batch", "64",
        "--seed", "191" if profile else "91", "--threads", "1", "--log-every", "10" if profile else "100",
        "--checkpoint-every", "20" if profile else "1000", "--information-pairs", "128",
        "--max-training-seconds", str(remaining_gpu_seconds), "--max-cuda-bytes", str(GPU_PEAK_BYTES),
        "--expected-gpu-uuid", plan["gpu_uuid"]]


def planned_commands(root, plan):
    return [
        ("sampler_preflight", [sys.executable, "-u", "-m", MODULE, "--sampler-preflight", "--output", str(root)], False),
        ("physical_sampling", sampling_command(root, plan), False),
        ("training_preflight", training_command(root, plan, profile=True), True),
        ("training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr", "--run", str(root / "run"),
            "--output", str(root / "report")], False),
        ("phi4_u02_report", [sys.executable, "-u", "-m", "scripts.report_phi4_u02_qcr", "--data", str(root / "data"),
            "--run", str(root / "run"), "--output", str(root / "calibration_report")], False)]


def sampler_preflight(root):
    from recurrent_rg.data.phi4 import sample_hmc_chain
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    if socket.gethostname() != plan["host"]:
        raise ValueError("Sampler preflight must use the frozen host")
    unchanged(plan)
    cpu_started, wall_started = time.process_time(), time.monotonic()
    opts, spec = plan["physical"]["options"], plan["sampler_preflight"]
    rows, numerical_error = [], None
    try:
        for r in spec["r_values"]:
            for initial in spec["initial_states"]:
                row = sample_hmc_chain(r, .2, h=0., size=36, dimension=2,
                    samples=spec["samples"], burn_in=spec["burn_in"], spacing=spec["spacing"],
                    step_size=opts["step_size"], leapfrog_steps=opts["leapfrog_steps"],
                    seed=spec["seed_base"] + len(rows), initial=initial)
                record = dict(r=r, u=.2, initial=initial, seed=row["seed"],
                    hmc_trajectories=row["hmc_trajectories"], force_evaluations=row["force_evaluations"],
                    cpu_seconds=row["cpu_seconds"], wall_seconds=row["wall_seconds"],
                    production_acceptance=row["production_acceptance"], burn_in_acceptance=row["burn_in_acceptance"],
                    max_absolute_delta_h=row["max_absolute_delta_h"], execution_only_not_scientific_data=True)
                if any(not math.isfinite(float(record[key])) for key in
                       ("cpu_seconds", "wall_seconds", "production_acceptance", "max_absolute_delta_h")):
                    raise FloatingPointError("Nonfinite HMC preflight diagnostics")
                rows.append(record)
                print(json.dumps(dict(event="phi4_u02_preflight_chain_complete", **record)), flush=True)
    except ArithmeticError as error:
        numerical_error = f"{type(error).__name__}: {error}"
    elapsed_cpu = time.process_time() - cpu_started
    complete = len(rows) == len(spec["r_values"]) * len(spec["initial_states"])
    acceptance_passed = complete and all(row["production_acceptance"] >= .6 for row in rows)
    projection = (max(row["cpu_seconds"] / row["hmc_trajectories"] for row in rows)
                  * plan["physical"]["total_hmc_trajectories"] * 2.) if rows and numerical_error is None else None
    total_projection = projection + elapsed_cpu + GPU_SECONDS + 600 if projection is not None else None
    valid = numerical_error is None and acceptance_passed
    budget_passed = total_projection is not None and total_projection <= CPU_SECONDS
    status = "invalid_sampling_hold" if not valid else "complete" if budget_passed else "budget_hold"
    completed_trajectories = sum(row["hmc_trajectories"] for row in rows)
    completed_force_sites = sum(row["force_evaluations"] * 36 ** 2 for row in rows)
    receipt = dict(status=status, rows=rows, physical_production_started=False,
        completed_fixed_pilots=len(rows), expected_fixed_pilots=9, numerical_error=numerical_error,
        acceptance_passed=acceptance_passed, minimum_production_acceptance=.6,
        preflight_cpu_seconds=elapsed_cpu, preflight_wall_seconds=time.monotonic() - wall_started,
        projected_production_cpu_seconds_conservative=projection,
        projected_total_cpu_seconds_conservative=total_projection,
        throughput_safety_factor=2., remaining_gpu_cpu_reserve_seconds=GPU_SECONDS,
        other_cpu_reserve_seconds=600, cpu_budget_seconds=CPU_SECONDS, budget_passed=budget_passed,
        planned_production_trajectories=plan["physical"]["total_hmc_trajectories"],
        actual_total_work_known=numerical_error is None,
        actual_preflight_trajectories=completed_trajectories if numerical_error is None else None,
        actual_preflight_force_site_evaluations=completed_force_sites if numerical_error is None else None,
        completed_pilots_work_lower_bound=dict(hmc_trajectories=completed_trajectories,
                                               force_site_evaluations=completed_force_sites),
        short_pilots_do_not_certify_equilibrium=True, sampler_parameters_not_adapted=True,
        requires_user_action=status != "complete")
    unchanged(plan)
    save(root / "sampler_preflight.json", receipt, exclusive=True)
    return receipt


def sampling_validity(root, plan):
    """Read sidecars only; distinguish severe invalidity from quality evidence."""
    data = root / "data"
    manifest = read(data / "manifest.json")
    if read(data / "plan.json") != plan["physical"]:
        raise RuntimeError("Executed physical data plan differs from the frozen plan")
    rows, severe = [], []
    for shard in manifest["shards"]:
        path = (data / shard["sidecar"]).resolve()
        if not path.is_relative_to(data.resolve()) or path.suffix != ".json":
            raise ValueError("Invalid physical sidecar reference")
        if sha(path) != shard["sidecar_sha256"]:
            raise RuntimeError("Physical sidecar SHA differs from its manifest")
        sidecar = read(path)
        for chain in sidecar["chains"]:
            rate = chain.get("production_acceptance")
            delta = chain.get("max_absolute_delta_h")
            if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
                    or not isinstance(delta, (int, float)) or not math.isfinite(delta)):
                severe.append(dict(condition=shard["cell_index"], chain=chain.get("chain_index"),
                                   production_acceptance=rate, max_absolute_delta_h=delta))
        rows.append(dict(cell_index=shard["cell_index"], r=sidecar["r"],
            production_acceptance=[chain["production_acceptance"] for chain in sidecar["chains"]],
            warning_count=sum(len(chain.get("warnings", [])) for chain in sidecar["chains"])))
    return dict(status="invalid_sampling_hold" if severe else "complete", severe=severe, rows=rows,
        warning_count=manifest["warning_count"], quality_gate_is_separate=True,
        equilibrium_certified=False, automatic_repair=False)


def sampling_quality(root):
    """Gate existing sampler diagnostics before any GPU call; never alter data."""
    from recurrent_rg.data.phi4 import OBSERVABLES
    data = root / "data"
    manifest = read(data / "manifest.json")
    conditions, issues = [], []
    for shard in manifest["shards"]:
        path = (data / shard["sidecar"]).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path) != shard["sidecar_sha256"]:
            raise RuntimeError("Quality gate sidecar provenance differs from the data manifest")
        sidecar = read(path)
        condition_issues = []
        for chain in sidecar["chains"]:
            index = chain["chain_index"]
            rate = chain.get("production_acceptance")
            if rate is None or not math.isfinite(rate) or rate < .6:
                condition_issues.append(dict(chain_index=index, kind="production_acceptance_below_existing_0.6",
                                             value=rate))
            diagnostics = chain.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                condition_issues.append(dict(chain_index=index, kind="diagnostic_mapping_unavailable"))
                diagnostics = {}
            if set(diagnostics) != set(OBSERVABLES):
                condition_issues.append(dict(chain_index=index, kind="diagnostic_observable_inventory_mismatch",
                    expected=list(OBSERVABLES), actual=sorted(diagnostics)))
            for name, diagnostic in diagnostics.items():
                if not isinstance(diagnostic, dict):
                    condition_issues.append(dict(chain_index=index, kind="existing_ess_below32_or_unavailable",
                                                 observable=name, ess=None))
                    continue
                ess = diagnostic.get("ess")
                if ess is None or not isinstance(ess, (int, float)) or not math.isfinite(ess) or ess < 32:
                    condition_issues.append(dict(chain_index=index, kind="existing_ess_below32_or_unavailable",
                                                 observable=name, ess=ess))
                if diagnostic.get("warning"):
                    condition_issues.append(dict(chain_index=index, kind="existing_series_warning",
                                                 observable=name, warning=diagnostic["warning"]))
            if sidecar["h"] == 0 and not chain.get("saved_magnetization_sign_changes", 0):
                condition_issues.append(dict(chain_index=index, kind="no_saved_magnetization_sign_change",
                                             value=chain.get("saved_magnetization_sign_changes")))
            for warning in chain.get("warnings", []):
                condition_issues.append(dict(chain_index=index, kind="existing_chain_warning", warning=warning))
        for name, comparison in sidecar["summary"].get("hot_cold_comparison", {}).items():
            if comparison.get("descriptive_three_se_warning"):
                condition_issues.append(dict(kind="existing_hot_cold_three_se_warning", observable=name,
                                             comparison=comparison))
        conditions.append(dict(cell_index=shard["cell_index"], r=sidecar["r"], u=sidecar["u"], h=sidecar["h"],
            sidecar=shard["sidecar"], sidecar_sha256=shard["sidecar_sha256"],
            chains=sidecar["chains"], summary=sidecar["summary"], issues=condition_issues))
        issues.extend(dict(cell_index=shard["cell_index"], **issue) for issue in condition_issues)
    return dict(status="sampling_quality_hold" if issues else "complete", issues=issues,
        conditions=conditions, source_warning_count=manifest["warning_count"],
        data_manifest_sha256=sha(data / "manifest.json"),
        training_started=False, GPU_profile_started=False, equilibrium_certified=False,
        no_Z2_symmetrization=True, sampler_parameters_not_changed=True, automatic_resampling=False,
        qualification="有限链诊断通过也不是平衡认证；任一既有质量问题需要用户决定，不能自动视为可训练。")


def hold(job, reason, *, status="budget_hold"):
    job.update(status=status, workflow_complete=False, requires_user_action=True,
               error=reason, ended_at=now())


def validate_training_summary(summary, plan, *, profile):
    expected = PROFILE_STEPS if profile else STEPS
    actual_uuid = str(summary["device"].get("uuid", "")).lower().removeprefix("gpu-")
    if actual_uuid != plan["gpu_uuid"].lower().removeprefix("gpu-"):
        raise RuntimeError("Torch physical GPU UUID differs from the frozen plan")
    optimizers = summary.get("optimizer_steps", {})
    if (summary.get("actual_steps") != expected or summary.get("labels_used_in_training") is not False
            or summary.get("frozen_evaluation_verified") is not True or set(optimizers) != {"Q", "C", "R"}
            or any(row.get("min") != expected or row.get("max") != expected for row in optimizers.values())):
        raise RuntimeError("Independent optimizer endpoint or frozen label-free evaluation mismatch")
    if summary.get("peak_cuda_bytes", GPU_PEAK_BYTES + 1) > GPU_PEAK_BYTES:
        raise RuntimeError("Own CUDA peak exceeds the frozen 8GiB budget")


def final_public_files(root):
    files = common.final_public_files(root)
    for name in ("sampler_preflight.json", "sampling_validity.json", "sampling_quality.json"):
        path = root / name
        if path.is_file():
            files[name] = sha(path)
    return files


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    validate_device(plan["gpu_index"], plan["gpu_uuid"])
    if plan["host"] != HOST or plan["device"] != f"cuda:{plan['gpu_index']}":
        raise ValueError("Physical target differs from its frozen plan")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Workflow already executed; no restart")
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST, started_at=now(),
                     plan_sha256=sha(root / "plan.json"))
    save(root / "execution.json", execution, exclusive=True)
    original_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if original_nice < 10:
        os.nice(10 - original_nice)
    started, cpu_started, gpu_elapsed = time.monotonic(), cpu_seconds(), 0.
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        for name, command, gpu in planned_commands(root, plan):
            unchanged(plan)
            if directory_bytes(root) > DIRECTORY_BYTES or cpu_seconds() - cpu_started >= CPU_SECONDS or (gpu and gpu_elapsed >= GPU_SECONDS):
                hold(job, "Actual accumulated resources reached the fixed budget; no work reduction or deletion")
                break
            admission(root, plan, job, name, gpu)
            if gpu:
                command[command.index("--max-training-seconds") + 1] = str(GPU_SECONDS - gpu_elapsed)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                MPLCONFIGDIR=str(root / "matplotlib_cache"))
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
                ended_at=now(), wall_seconds=time.monotonic() - stage_started, cpu_seconds=cpu_seconds() - stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds() - cpu_started,
                       directory_bytes=directory_bytes(root))
            save(root / "job.json", job)
            if returncode:
                folder = ("profile_run" if name == "training_preflight" else "run" if name == "training" else
                          "data" if name == "physical_sampling" else "calibration_report" if name == "phi4_u02_report" else "report")
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
                    hold(job, "Fixed-integrator preflight invalid; production not started, no automatic adjustment",
                         status="invalid_sampling_hold")
                    break
                if not receipt["budget_passed"]:
                    hold(job, "Conservative sampling CPU projection exceeds budget; production not started")
                    break
            elif name == "physical_sampling":
                status = read(root / "data" / "status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Physical sampling did not finish successfully")
                validity = sampling_validity(root, plan)
                save(root / "sampling_validity.json", validity, exclusive=True)
                job["sampling_warning_count"] = status.get("warning_count", 0)
                job["sampling_actual_cpu_seconds"] = status["cpu_seconds"]
                if validity["severe"]:
                    hold(job, "Production has zero-acceptance/nonfinite evidence; no training or automatic repair",
                         status="invalid_sampling_hold")
                    break
                quality = sampling_quality(root)
                save(root / "sampling_quality.json", quality, exclusive=True)
                job["sampling_quality_issue_count"] = len(quality["issues"])
                if quality["issues"]:
                    hold(job, "Existing sampling quality diagnostics require user direction; GPU profile/training not started",
                         status="sampling_quality_hold")
                    break
                if cpu_seconds() - cpu_started + GPU_SECONDS + 600 > CPU_SECONDS:
                    hold(job, "Actual sampling CPU plus conservative remaining reserve exceeds CPU budget")
                    break
            elif name in ("training_preflight", "training"):
                profile = name == "training_preflight"
                folder = root / ("profile_run" if profile else "run")
                summary = read(folder / "summary.json")
                validate_training_summary(summary, plan, profile=profile)
                if profile:
                    receipt = common.profile_projection(summary, read(folder / "history_progress.json"),
                                                        stage["wall_seconds"], gpu_elapsed)
                    save(root / "gpu_preflight.json", receipt, exclusive=True)
                    job["gpu_projection"] = receipt
                    if not receipt["budget_passed"]:
                        hold(job, "GPU projection exceeds 1800 seconds; fixed steps not reduced")
                        break
                    if cpu_seconds() - cpu_started + receipt["projected_formal_gpu_seconds_conservative"] + 600 > CPU_SECONDS:
                        hold(job, "Actual CPU plus conservative remaining work exceeds 7200 seconds")
                        break
                else:
                    job["actual_training_steps"] = STEPS
            else:
                path = root / ("calibration_report" if name == "phi4_u02_report" else "report") / "status.json"
                status = read(path)
                if status.get("status") != "complete" or status.get("workflow_complete") is not True:
                    raise RuntimeError(f"{name} public report is incomplete")
            if gpu_elapsed > GPU_SECONDS or cpu_seconds() - cpu_started > CPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
                hold(job, "Actual resource accounting exceeded its fixed budget; completed artifacts preserved")
                break
        else:
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, error=f"{type(error).__name__}: {error}",
                   requires_user_action=True, ended_at=now())
        raise
    finally:
        job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds() - cpu_started,
                   wall_seconds=time.monotonic() - started, directory_bytes=directory_bytes(root))
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"], ended_at=now(),
            cpu_seconds=job["cpu_seconds"], gpu_stage_wall_seconds=gpu_elapsed, wall_seconds=job["wall_seconds"],
            directory_bytes=job["directory_bytes"], actual_nice=job["actual_nice"], error=job["error"],
            requires_user_action=job["requires_user_action"])
        save(root / "execution.json", execution)
        if job["status"] == "complete":
            job["public_files"] = final_public_files(root)
        save(root / "job.json", job)


def launch(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if job["status"] != "prepared" or job["stages"] or (root / "execution.json").exists():
        raise ValueError("Workflow already executed; no restart")
    unchanged(plan)
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--output", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
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
    # A completed preflight decision is rc0 even if it scientifically holds work.
    return 0 if args.sampler_preflight or value["status"] not in (
        "failed", "budget_hold", "invalid_sampling_hold", "sampling_quality_hold") else 3


if __name__ == "__main__":
    sys.exit(main())
