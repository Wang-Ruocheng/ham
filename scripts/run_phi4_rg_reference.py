"""独立固定 u=1 的 φ⁴ 参考扫描；默认仅准备，全新目录，不恢复或重试。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from scripts import run_gaussian_scalar_qcr as common


MODULE = "scripts.run_phi4_rg_reference"
HOST = "thinklab-105-225"
GPU_INDEX = 2
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
CPU_SECONDS, GPU_SECONDS = 7200, 3600
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024**3, 1024**3
FORCE_SITE_CAP = 2_000_000_000
STEPS, PROFILE_STEPS = 3000, 20
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
now, read, save, sha = common.now, common.read, common.save, common.sha
source_hashes, unchanged = common.source_hashes, common.unchanged
directory_bytes, cpu_seconds = common.directory_bytes, common.cpu_seconds


def default_config():
    rc = -1.2724682
    return dict(schema_version=5, experiment="phi4_u1_rg_reference_b3_g1_seed91",
        recipe="phi4_continuous_gaussian_soft_statistics_v1",
        reference=dict(r_critical=rc, r_uncertainty=.0000048, u=1., h=0.,
            use="data generation and evaluation only; never training supervision"),
        data=dict(sizes=[36], dimension=2, r_values=[rc - .4 + .8*i/12 for i in range(13)],
            u_values=[1.], h=0., chains=4, samples=128, burn_in=1024, spacing=8,
            step_size=.08, leapfrog_steps=12, seed=20261028, test_every=3, test_offset=0,
            cluster_probability=.5),
        training=dict(steps=STEPS, seed=91, system_batch=8, sample_batch=64,
            information_pairs=128, q_initialization="train_mean_bias", learning_rate=.001,
            threads=1, log_every=100, checkpoint_every=1000),
        model=dict(block_size=3, stride=3, d_g=1, lattice_sizes=[36, 12, 4], hidden=96,
            decoder="shared_affine", C="random_trainable_probabilistic_gaussian", dtype="float64"),
        loss=dict(closure_objective="all_start_anchor", closure_scale_weights=None,
            commutator=5., rollout_weight=5., Q_reconstruction_weight=10, Q_subset_weights=[20, 5],
            continuous_C_information="Gaussian-kernel-smoothed buffered full-shell proxy, not discrete empirical-histogram MI",
            continuous_C_uniform_preservation="uniform-amplitude MSE, not discrete color NLL",
            gradient_routing="simultaneous independent Q-only/C-only/R-only",
            future_observable_loss=False, prediction_prediction_loss=False, coupling_supervision=False))


def _validate_config(config):
    """This entry point has one scientific design, not arbitrary training knobs."""
    fixed = default_config()
    if config.get("schema_version") != 5:
        raise ValueError("Require schema_version 5")
    if config.get("recipe", fixed["recipe"]) != fixed["recipe"]:
        raise ValueError("Fixed continuous recipe differs")
    for section in ("reference", "data", "training", "model", "loss"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing {section} mapping")
        for key, expected in fixed[section].items():
            if section == "reference" and key == "use":
                continue
            actual = config[section].get(key)
            if key == "r_values":
                valid = (isinstance(actual, (list, tuple)) and len(actual) == 13
                         and all(isinstance(v, (int, float)) and math.isfinite(v)
                                 and abs(v-e) < 1e-12 for v, e in zip(actual, expected)))
            else:
                valid = actual == expected and not (isinstance(actual, bool) and not isinstance(expected, bool))
            if not valid:
                raise ValueError(f"Fixed reference contract differs: {section}.{key}")
    # Additional documentary metadata is allowed; hidden implementation switches are not.
    allowed = {"schema_version", "experiment", "name", "recipe", "reference", "data", "training", "model", "loss", "evaluation"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration sections")
    for section in ("training", "model", "loss"):
        if set(config[section]) - set(fixed[section]):
            raise ValueError(f"Unknown {section} options")


def build_plan(root, code_root, config=None):
    """Pure planning: no output directory, sampling, CUDA, or host access."""
    from recurrent_rg.data.phi4_hybrid_pipeline import Phi4HybridOptions, generation_plan
    config = copy.deepcopy(default_config() if config is None else config)
    _validate_config(config)
    root, code = Path(root).resolve(), Path(code_root).resolve()
    if root == code or root in code.parents:
        raise ValueError("Output must not contain or replace the code root")
    values = dict(config["data"])
    caps = dict(max_force_site_evaluations=FORCE_SITE_CAP, max_output_bytes=DIRECTORY_BYTES,
                max_memory_bytes=512*1024**2, max_trajectories=200_000,
                max_cluster_updates=2_000_000, max_cluster_neighbor_visits=1_000_000_000)
    for key, cap in caps.items():
        if key in values and values[key] != cap:
            raise ValueError(f"Fixed data cap differs: {key}")
        values[key] = cap
    for key in ("sizes", "r_values", "u_values"):
        values[key] = tuple(values[key])
    physical = json.loads(json.dumps(generation_plan(Phi4HybridOptions(**values))))
    pilot_cells = [1, 5, 7, 11]
    pilot = dict(cell_indices=pilot_cells,
        r_values=[values["r_values"][cell] for cell in pilot_cells],
        initial_states=["hot", "cold_positive", "cold_negative"], samples=8, burn_in=128,
        spacing=4, seed_base=202610280100, expected_pilots=12, throughput_safety_factor=2.,
        calibration_uses_only_TRAIN=True, production_data_reused=False)
    pilot_work = pilot["expected_pilots"]*(pilot["burn_in"] + pilot["samples"]*pilot["spacing"])
    total_forces = physical["total_force_site_evaluations"] + pilot_work*13*36**2
    if total_forces > FORCE_SITE_CAP:
        raise ValueError("New production plus fixed pilot exceeds force-site cap")
    return dict(schema_version=5, recipe="phi4_u1_rg_reference_hybrid_qcr_v1", prepared_at=now(),
        code_root=str(code), output=str(root), host=HOST, gpu_index=GPU_INDEX,
        device=f"cuda:{GPU_INDEX}", gpu_uuid=GPU_UUID, config=config,
        source_sha256=source_hashes(code), physical=physical, sampler_preflight=pilot,
        reference=copy.deepcopy(config["reference"]),
        evaluation=dict(config.get("evaluation", {}), extrapolation_test_cell_indices=[0, 12],
            reference_used_for_training=False, reference_used_for_generation_and_evaluation=True),
        total_planned_hmc_trajectories_including_preflight=physical["total_hmc_trajectories"]+pilot_work,
        total_planned_force_site_evaluations_including_preflight=total_forces,
        training=dict(**config["training"], profile_steps=PROFILE_STEPS, profile_seed=191,
            lattice_sizes=[36,12,4], theory_dim=1, hidden=96, dtype="float64",
            independent_gradients=True, closure_objective="all_start_anchor", closure_scale_weights=None,
            labels_used=False, warm_start=False, pca_initialization=False,
            profile_checkpoint_used_for_formal=False),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS,
            gpu_seconds=GPU_SECONDS, gpu_peak_bytes=GPU_PEAK_BYTES, directory_bytes=DIRECTORY_BYTES,
            force_site_evaluations=FORCE_SITE_CAP, minimum_host_available_gib=36,
            gpu_free_mib=12288, gpu_compute_processes=0, load_per_cpu_max=1.25,
            io_full_avg10_percent_max=20, two_gpu_checks_seconds_apart=10,
            formal_projection_multiplier=1.5, report_reserve_seconds=300,
            gpu_budget_semantics="GPU admission waits plus full profile/formal child wall times, including initialization and final evaluation",
            other_cpu_reserve_seconds=600, occupied_resources="wait without intervention"),
        qualification="exploratory_all_sampling_warnings_retained",
        scientific_limits=["独立固定 u=1、h=0 参考扫描；参考耦合仅用于生成与事后评价",
            "固定 HMC 与 lazy Wolff；质量警告全部保留，不认证平衡",
            "非有限数值或零生产接受率硬停止；不重试、恢复、调参、补样或修复",
            "随机可训练高斯 C 与既有连续场信息代理；无三次、奇函数或 R-only 分支",
            "单 seed、有限体积和一维闭合不认证临界固定点或临界指数"])


def prepare(root, *, code_root=None, config=None):
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing output is never overwritten or resumed")
    plan = build_plan(root, code_root or Path(__file__).resolve().parents[1], config)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "config.json", plan["config"], exclusive=True)
    plan["config_sha256"] = sha(root / "config.json")
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, returncode=None,
        public_files=["config.json", "job.json", "plan.json", "public_files.json", "status.json"],
        plan_sha256=sha(root / "plan.json")), exclusive=True)
    save(root / "status.json", dict(status="prepared", workflow_complete=False, returncode=None), exclusive=True)
    save(root / "public_files.json", {name:sha(root/name) for name in
        ("config.json", "job.json", "plan.json", "status.json")}, exclusive=True)
    return plan


def load_plan(root):
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if (sha(root / "plan.json") != job["plan_sha256"]
            or sha(root / "config.json") != plan["config_sha256"]
            or plan["output"] != str(root) or plan["host"] != HOST
            or plan["device"] != "cuda:2" or plan["gpu_uuid"] != GPU_UUID):
        raise RuntimeError("Frozen plan/config/physical target differs")
    _validate_config(plan["config"])
    unchanged(plan)
    return plan, job


def validate_host(*, cpu_only=False):
    if socket.gethostname() != HOST:
        raise ValueError(f"Frozen execution host must be {HOST}")
    if "CUDA_VISIBLE_DEVICES" in os.environ and not (cpu_only and os.environ["CUDA_VISIBLE_DEVICES"] == ""):
        raise ValueError("Do not reinterpret physical GPUs through a CUDA mask")


def resource_snapshot(plan):
    p = plan["resource_policy"]
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])*1024
    io_full = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    query = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, free = query[0].strip(), int(query[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    pids = [int(row.split(",")[1]) for row in processes if uuid in row]
    host_ready = (available >= p["minimum_host_available_gib"]*1024**3
                  and load <= p["load_per_cpu_max"]*cpus and io_full < p["io_full_avg10_percent_max"])
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=uuid, gpu_free_mib=free, gpu_compute_pids=pids,
        host_ready=host_ready, gpu_ready=host_ready and uuid == GPU_UUID and free >= p["gpu_free_mib"] and not pids)


def admission(root, plan, job, stage, gpu):
    started = time.monotonic()
    consecutive = 0
    checks = 0
    receipt = dict(stage=stage, gpu=gpu, checks=0, started_at=now(), status="waiting",
        minimum_idle_check_separation_seconds=10 if gpu else None)
    job.setdefault("admissions", []).append(receipt)
    while True:
        if gpu and job.get("gpu_stage_wall_seconds", 0.)+time.monotonic()-started >= GPU_SECONDS:
            receipt.update(status="budget_hold", wall_seconds=time.monotonic()-started, ended_at=now())
            raise BudgetHold("GPU admission wait reached cumulative GPU wall-time budget")
        row = resource_snapshot(plan)
        checks += 1
        receipt.update(checks=checks, wall_seconds=time.monotonic()-started)
        if row["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU2 UUID differs from frozen plan")
        consecutive = consecutive+1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if consecutive >= (2 if gpu else 1):
            receipt.update(status="admitted", wall_seconds=time.monotonic()-started, admitted_at=now())
            save(root / "job.json", job)
            return receipt
        time.sleep(plan["resource_policy"]["two_gpu_checks_seconds_apart"] if consecutive else 30)


def sampling_command(root, plan):
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_hybrid", "--execute", "--output", str(root / "data")]
    for key, value in plan["physical"]["options"].items():
        command.append("--"+key.replace("_", "-"))
        command.extend(str(x) for x in value) if isinstance(value, (list, tuple)) else command.append(str(value))
    return command


def training_command(root, plan, *, profile, remaining_gpu_seconds=GPU_SECONDS):
    t = plan["training"]
    command = [sys.executable, "-u", "-m", "experiments.phi4_qcr", "--data", str(root / "data"),
        "--output", str(root / ("profile_run" if profile else "run")), "--device", plan["device"],
        "--steps", str(t["profile_steps"] if profile else t["steps"]),
        "--seed", str(t["profile_seed"] if profile else t["seed"]), "--threads", "1",
        "--log-every", "10" if profile else "100", "--checkpoint-every", "20" if profile else "1000",
        "--max-training-seconds", str(remaining_gpu_seconds), "--max-cuda-bytes", str(GPU_PEAK_BYTES),
        "--expected-gpu-uuid", GPU_UUID, "--q-initialization", t["q_initialization"]]
    for key in ("system_batch", "sample_batch", "information_pairs"):
        command.extend(["--"+key.replace("_", "-"), str(t[key])])
    return command


def planned_commands(root, plan):
    return [
        ("sampler_preflight", [sys.executable, "-u", "-m", MODULE, "--sampler-preflight", "--output", str(root)], False),
        ("physical_sampling", sampling_command(root, plan), False),
        ("training_preflight", training_command(root, plan, profile=True), True),
        ("training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_rg_reference", "--run", str(root / "run"),
            "--data", str(root / "data"), "--output", str(root / "report")], False)]


def sampler_preflight(root):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    validate_host(cpu_only=True)
    root = Path(root).resolve()
    plan, _ = load_plan(root)
    options, pilot = plan["physical"]["options"], plan["sampler_preflight"]
    started, rows, error = time.process_time(), [], None
    try:
        for r in pilot["r_values"]:
            for initial in pilot["initial_states"]:
                row = sample_hybrid_chain(r, 1., h=0., size=36, dimension=2,
                    samples=pilot["samples"], burn_in=pilot["burn_in"], spacing=pilot["spacing"],
                    step_size=options["step_size"], leapfrog_steps=options["leapfrog_steps"],
                    cluster_probability=.5, seed=pilot["seed_base"]+len(rows), initial=initial)
                record = {key: row[key] for key in ("cpu_seconds", "wall_seconds", "hmc_trajectories",
                    "force_site_evaluations", "production_acceptance", "max_absolute_delta_h")}
                if (any(not math.isfinite(float(value)) for value in record.values())
                        or record["production_acceptance"] <= 0):
                    raise FloatingPointError("Nonfinite or zero-acceptance fixed sampler pilot")
                rows.append(dict(record, r=r, initial=initial, warnings=row.get("warnings", []),
                    execution_only_not_scientific_data=True))
    except ArithmeticError as exc:
        error = f"{type(exc).__name__}: {exc}"
    projection = (max(row["cpu_seconds"]/row["hmc_trajectories"] for row in rows)
        *plan["physical"]["total_hmc_trajectories"]*2) if len(rows) == pilot["expected_pilots"] and not error else None
    actual_cpu = time.process_time()-started
    budget = projection is not None and projection+actual_cpu+GPU_SECONDS+600 <= CPU_SECONDS
    receipt = dict(status="invalid_sampling_hold" if error else "complete" if budget else "budget_hold",
        rows=rows, numerical_error=error, physical_production_started=False,
        completed_fixed_pilots=len(rows), expected_fixed_pilots=pilot["expected_pilots"],
        calibration_uses_only_TRAIN=True, production_data_reused=False,
        projected_production_cpu_seconds_conservative=projection, preflight_cpu_seconds=actual_cpu,
        budget_passed=budget, quality_warnings_retained=True, equilibrium_certified=False)
    save(root / "sampler_preflight.json", receipt, exclusive=True)
    return receipt


def sampling_validity(root, plan):
    """Sidecar-only hard validity; ordinary quality warnings stay exploratory."""
    data = root / "data"
    manifest = read(data / "manifest.json")
    if read(data / "plan.json") != plan["physical"] or manifest.get("status") != "complete":
        raise RuntimeError("Executed physical data differs from frozen plan")
    severe, warnings = [], []
    if len(manifest["shards"]) != plan["physical"]["conditions"]:
        raise RuntimeError("Incomplete physical condition inventory")
    for shard in manifest["shards"]:
        path = (data / shard["sidecar"]).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path) != shard["sidecar_sha256"]:
            raise RuntimeError("Physical sidecar provenance differs")
        sidecar = read(path)
        for chain in sidecar["chains"]:
            rate, delta = chain.get("production_acceptance"), chain.get("max_absolute_delta_h")
            if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
                    or not isinstance(delta, (int, float)) or not math.isfinite(delta)):
                severe.append(dict(cell_index=shard["cell_index"], chain_index=chain.get("chain_index")))
            warnings.extend(chain.get("warnings", []))
    return dict(status="invalid_sampling_hold" if severe else "complete", severe=severe,
        warnings=warnings, warning_count=manifest.get("warning_count", len(warnings)),
        quality_warnings_retained=True, equilibrium_certified=False, automatic_repair=False)


def validate_training_summary(summary, plan, *, profile):
    expected = PROFILE_STEPS if profile else STEPS
    uuid = str(summary["device"].get("uuid", "")).lower().removeprefix("gpu-")
    clocks = summary.get("optimizer_steps", {})
    if (uuid != GPU_UUID.lower().removeprefix("gpu-") or summary.get("actual_steps") != expected
            or summary.get("labels_used_in_training") is not False or summary.get("frozen_evaluation_verified") is not True
            or set(clocks) != {"Q", "C", "R"}
            or any(row.get("min") != expected or row.get("max") != expected for row in clocks.values())
            or summary.get("peak_cuda_bytes", GPU_PEAK_BYTES+1) > GPU_PEAK_BYTES):
        raise RuntimeError("Independent label-free training endpoint/device/peak differs")


def profile_projection(summary, history, profile_wall_seconds, gpu_elapsed):
    # Reuse the timing decomposition, then apply this independent run's budget.
    receipt = common.profile_projection(summary, history, profile_wall_seconds, gpu_elapsed)
    receipt["remaining_gpu_admission_reserve_seconds"] = 10.
    receipt["projected_total_gpu_seconds_conservative"] += 10.
    receipt.update(gpu_budget_seconds=GPU_SECONDS,
        budget_passed=receipt["projected_total_gpu_seconds_conservative"] <= GPU_SECONDS)
    return receipt


def process_snapshot(pid):
    """Read only our own direct child's counters, never other users' processes."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"),
            rss_bytes=int(fields[21])*os.sysconf("SC_PAGE_SIZE"), at=now())
    except FileNotFoundError:
        return dict(pid=pid, cpu_seconds=0., rss_bytes=0, at=now(), exited=True)


def _wait_child(child, root, job, *, cpu_start, stage_start, gpu_elapsed, gpu):
    while child.poll() is None:
        snapshot = process_snapshot(child.pid)
        job.setdefault("process_resource_checks", []).append(snapshot)
        job["process_resource_checks"] = job["process_resource_checks"][-60:]
        exceeded = (cpu_seconds()-cpu_start+snapshot["cpu_seconds"] >= CPU_SECONDS
            or directory_bytes(root) > DIRECTORY_BYTES
            or (gpu and gpu_elapsed+time.monotonic()-stage_start >= GPU_SECONDS))
        save(root / "job.json", job)
        if exceeded:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return child.returncode, True
        time.sleep(1)
    return child.wait(), False


def final_public_files(root):
    files = [root/name for name in ("config.json", "plan.json", "launch.json", "execution.json", "sampler_preflight.json",
        "sampling_validity.json", "gpu_preflight.json", "status.json")]
    files += [root/"data"/name for name in ("plan.json", "manifest.json", "status.json", "quality_summary.json")]
    files += [p for folder in ("run", "report") for p in (root/folder).rglob("*")
              if p.is_file() and p.suffix in (".json", ".npz", ".png")]
    files += list((root/"data"/"conditions").glob("*/sidecar.json"))
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(files)) if p.is_file()}


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = load_plan(root)
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Workflow already executed; no retry, resume, or repair")
    save(root/"execution.json", dict(status="starting", parent_pid=os.getpid(), started_at=now(),
        plan_sha256=sha(root/"plan.json")), exclusive=True)
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 10:
        os.nice(10-current_nice)
    started, cpu_start, gpu_elapsed, admission_elapsed = time.monotonic(), cpu_seconds(), 0., 0.
    job.update(parent_pid=os.getpid(), started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    try:
        for name, command, gpu in planned_commands(root, plan):
            load_plan(root)
            if (cpu_seconds()-cpu_start >= CPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES
                    or (gpu and gpu_elapsed >= GPU_SECONDS)):
                raise BudgetHold("Cumulative resource budget reached")
            admission_started = time.monotonic()
            try:
                admission(root, plan, job, name, gpu)
            finally:
                waited = time.monotonic()-admission_started
                if gpu:
                    gpu_elapsed += waited
                    admission_elapsed += waited
                    job.update(gpu_stage_wall_seconds=gpu_elapsed, gpu_admission_wall_seconds=admission_elapsed)
            if gpu and gpu_elapsed >= GPU_SECONDS:
                raise BudgetHold("GPU admission wait reached cumulative GPU budget")
            load_plan(root)
            if gpu:
                command[command.index("--max-training-seconds")+1] = str(GPU_SECONDS-gpu_elapsed)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       MPLCONFIGDIR=str(root/"matplotlib_cache"))
            env.update({key: "1" for key in THREAD_VARIABLES})
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            tick, stage_cpu = time.monotonic(), cpu_seconds()
            with (root/"logs"/f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, command=command, gpu=gpu, pid=child.pid, status="running", started_at=now(),
                    admission_wall_seconds=waited)
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(root/"job.json", job)
                save(root/"status.json",dict(status="running",workflow_complete=False,current=name,
                    returncode=None,requires_user_action=False,error=None))
                code, budget_stopped = _wait_child(child, root, job, cpu_start=cpu_start,
                    stage_start=tick, gpu_elapsed=gpu_elapsed, gpu=gpu)
            stage.update(returncode=code, status="complete" if code == 0 else "failed", ended_at=now(),
                wall_seconds=time.monotonic()-tick, cpu_seconds=cpu_seconds()-stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds()-cpu_start)
            save(root/"job.json", job)
            if budget_stopped:
                raise BudgetHold("Own child stopped at cumulative resource budget; outputs preserved")
            if code:
                folder = {"physical_sampling":"data", "training_preflight":"profile_run", "training":"run", "report":"report"}.get(name)
                child_status = read(root/folder/"status.json") if folder and (root/folder/"status.json").is_file() else {}
                if child_status.get("status") == "budget_hold":
                    raise BudgetHold(f"{name} reached its fixed compute budget")
                raise RuntimeError(f"{name} returned {code}; no automatic retry")
            if name == "sampler_preflight":
                receipt = read(root/"sampler_preflight.json")
                if receipt["status"] != "complete":
                    raise BudgetHold("Fixed sampler pilot did not pass", status=receipt["status"])
            elif name == "physical_sampling":
                status = read(root/"data"/"status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Physical sampling endpoint incomplete")
                validity = sampling_validity(root, plan)
                save(root/"sampling_validity.json", validity, exclusive=True)
                if validity["severe"]:
                    raise BudgetHold("Zero acceptance or nonfinite production evidence", status="invalid_sampling_hold")
                if cpu_seconds()-cpu_start+GPU_SECONDS+600 > CPU_SECONDS:
                    raise BudgetHold("Sampling plus remaining CPU reserve exceeds budget")
            elif name in ("training_preflight", "training"):
                profile = name == "training_preflight"
                folder = root/("profile_run" if profile else "run")
                summary = read(folder/"summary.json")
                validate_training_summary(summary, plan, profile=profile)
                if profile:
                    projection = profile_projection(summary, read(folder/"history_progress.json"), stage["wall_seconds"], gpu_elapsed)
                    save(root/"gpu_preflight.json", projection, exclusive=True)
                    if (not projection["budget_passed"] or cpu_seconds()-cpu_start
                            +projection["projected_formal_gpu_seconds_conservative"]+300 > CPU_SECONDS):
                        raise BudgetHold("Independent profile projection exceeds cumulative budget")
            elif name == "report":
                if not read(root/"report"/"status.json").get("workflow_complete"):
                    raise RuntimeError("Reference report endpoint incomplete")
            load_plan(root)
        else:
            if cpu_seconds()-cpu_start > CPU_SECONDS or gpu_elapsed > GPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
                raise BudgetHold("Completed stages exceeded cumulative resource budget")
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except Exception as exc:
        job.update(status=exc.status if isinstance(exc, BudgetHold) else "failed", workflow_complete=False,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, ended_at=now())
    finally:
        job.update(cpu_seconds=cpu_seconds()-cpu_start, gpu_stage_wall_seconds=gpu_elapsed,
            gpu_admission_wall_seconds=admission_elapsed,
            wall_seconds=time.monotonic()-started, directory_bytes=directory_bytes(root))
        job["returncode"] = 0 if job["workflow_complete"] else 3 if job["status"].endswith("hold") else 1
        save(root/"status.json", {key:job[key] for key in ("status", "workflow_complete", "error", "requires_user_action", "returncode")})
        save(root/"execution.json", dict(status=job["status"], parent_pid=os.getpid(),
            workflow_complete=job["workflow_complete"], cpu_seconds=job["cpu_seconds"],
            returncode=job["returncode"],gpu_stage_wall_seconds=gpu_elapsed,
            gpu_admission_wall_seconds=admission_elapsed, actual_nice=job["actual_nice"], ended_at=now()))
        files = final_public_files(root)
        job["public_files"] = sorted([*files, "job.json", "public_files.json"])
        save(root/"job.json", job)
        files["job.json"] = sha(root/"job.json")
        save(root/"public_files.json", files)
    return job


class BudgetHold(RuntimeError):
    def __init__(self, message, *, status="budget_hold"):
        super().__init__(message)
        self.status = status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sampler-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help="只验证冻结计划；不采样、不训练、不创建目录")
    args = parser.parse_args(argv)
    if args.sampler_preflight:
        result = sampler_preflight(args.output)
    elif args.smoke:
        if args.execute:
            parser.error("--smoke cannot be combined with --execute")
        result = build_plan(args.output, Path(__file__).resolve().parents[1], read(args.config) if args.config else None)
    else:
        if args.execute and args.output.exists():
            if args.config:
                parser.error("Execution of a prepared root reads only its frozen config")
            result = execute(args.output)
        else:
            result = prepare(args.output, config=read(args.config) if args.config else None)
            if args.execute:
                result = execute(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.execute and result.get("status") != "complete":
        raise SystemExit(3)
    return result


if __name__ == "__main__":
    main()
