"""独立的大尺寸连续场 phi4 对照；不覆盖、恢复或修改 L36 实验。"""
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

from scripts.run_phi4_qcr import directory_bytes, now, read, save, sha, unchanged


HOST = "thinklab-105-225"
GPU = "cuda:3"
GPU_UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"
MODULE = "scripts.run_phi4_qcr_large"
STEPS = 3000
PROFILE_STEPS = 20
SIZE = 108
STEP_SIZE = .04
LEAPFROG_STEPS = 24
CPU_SECONDS = 14400
GPU_SECONDS = 7200
DIRECTORY_BYTES = 2 * 1024**3
FORCE_SITE_CAP = 27_000_000_000


def source_hashes(code):
    """Pin executable Python, not macOS AppleDouble archive metadata."""
    return {str(path.relative_to(code)): sha(path)
            for folder in ("recurrent_rg", "scripts", "experiments")
            for path in sorted((code / folder).rglob("*.py"))
            if not path.name.startswith("._")}


def prepare(root, *, size=SIZE, step_size=STEP_SIZE, leapfrog_steps=LEAPFROG_STEPS,
            cpu_seconds=CPU_SECONDS, gpu_seconds=GPU_SECONDS,
            directory_limit=DIRECTORY_BYTES, force_site_cap=FORCE_SITE_CAP):
    from recurrent_rg.data.phi4 import Phi4Options, generation_plan
    if socket.gethostname() != HOST:
        raise ValueError("This physical-device plan is for thinklab-105-225 only")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Do not reinterpret physical GPU indices through a CUDA mask")
    if isinstance(size, bool) or not isinstance(size, int) or size < 36 or size % 9:
        raise ValueError("Require a side >=36 divisible by9 for two block-3 RG steps")
    for value, label in ((cpu_seconds, "CPU"), (gpu_seconds, "GPU"),
                         (directory_limit, "directory")):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} budget must be finite and positive")
    if cpu_seconds < gpu_seconds:
        raise ValueError("CPU budget must include the conservative GPU-stage CPU reserve")
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing output is never overwritten or resumed")
    physical = generation_plan(Phi4Options(sizes=(size,), step_size=step_size,
        leapfrog_steps=leapfrog_steps, max_force_site_evaluations=force_site_cap))
    if physical["projected_output_bytes"] > directory_limit:
        raise ValueError("Projected physical data alone exceeds experiment directory budget")
    plan = dict(schema_version=5, recipe="phi4_continuous_gaussian_soft_statistics_v1",
        experiment="larger_size_control_with_disclosed_HMC_integrator_refinement",
        code_root=str(code), output=str(root), host=HOST, device=GPU, gpu_uuid=GPU_UUID,
        prepared_at=now(), source_sha256=source_hashes(code), physical=physical,
        training=dict(steps=STEPS, profile_steps=PROFILE_STEPS, theory_dim=1, hidden=96,
            lattice_sizes=[size, size // 3, size // 9], system_batch=8, sample_batch=64,
            seed=91, profile_seed=191, optimizer="three independent fresh Adam",
            learning_rate=.001, schedule="constant", independent_gradients=True,
            closure_scale_weights=[1, 1], labels_used=False, warm_start=False,
            pca_initialization=False, information_pairs=128),
        sampler_preflight=dict(r_values=[-2., -.75, .5],
            initial_states=["hot", "cold_positive", "cold_negative"],
            samples=8, burn_in=1024, spacing=8, seed_base=2026102300,
            minimum_production_acceptance=.6, throughput_safety_factor=2.,
            equilibrium_certified=False, production_data_reused=False),
        resource_policy=dict(cpu_threads=1, nice=10, gpu_peak_bytes=8 * 1024**3,
            gpu_seconds=float(gpu_seconds), cpu_seconds=float(cpu_seconds),
            directory_bytes=int(directory_limit), minimum_host_available_gib=12,
            gpu_free_mib=12288, gpu_compute_processes=0,
            io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, occupied_resources="wait without intervention",
            gpu_profile_safety_factor=1.5, gpu_final_reserve_seconds=300),
        authorization="User approved larger-size continuous-field experiment and fixed HMC .04/24 refinement",
        comparison_changes=dict(lattice_sizes_from=[36, 12, 4],
            lattice_sizes_to=[size, size // 3, size // 9], physical_device_from="cuda:2",
            physical_device_to=GPU, step_size_from=.08, step_size_to=float(step_size),
            leapfrog_steps_from=12, leapfrog_steps_to=int(leapfrog_steps),
            trajectory_length_from=.96,
            trajectory_length_to=float(step_size * leapfrog_steps),
            physical_seed_key_includes_size=True,
            model_loss_optimizer_and_training_endpoint_unchanged=True),
        scientific_limits=["正测度欧氏实场，不是实时复路径积分或符号问题求解",
            "本次有已披露的 HMC 数值步长细化与物理 GPU 差异，不冒称逐比特尺寸单变量对照",
            "保持原核平滑分箱信息代理，未修复其有限样本自点排除偏差；不是精确 MI",
            "有限链未认证平衡；临界慢化与全部 ESS/热冷/符号混合警告保留，不自动加样",
            "预检只检查数值接受率、吞吐与预算，不是正式科学数据或平衡认证",
            "L12 末尺度仍是有限体积，不保证闭包、正确相结构或普适指数；thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now(),
        plan_sha256=sha(root / "plan.json")), exclusive=True)
    return plan


def load_plan(root, *, cpu_only=False):
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if sha(root / "plan.json") != job.get("plan_sha256"):
        raise RuntimeError("Frozen plan fingerprint differs from the prepared job")
    if (plan.get("host") != HOST or plan.get("device") != GPU
            or plan.get("gpu_uuid") != GPU_UUID or plan.get("output") != str(root)):
        raise RuntimeError("Frozen host/device/output differs from this physical-device plan")
    if socket.gethostname() != HOST:
        raise ValueError("Wrong host; no relocation or restart")
    if "CUDA_VISIBLE_DEVICES" in os.environ and not (cpu_only and os.environ["CUDA_VISIBLE_DEVICES"] == ""):
        raise ValueError("Do not reinterpret physical GPU indices through a CUDA mask")
    unchanged(plan)
    return plan, job


def resource_snapshot(plan):
    policy = plan["resource_policy"]
    memory = Path("/proc/meminfo").read_text()
    available = int(re.search(r"MemAvailable:\s+(\d+)", memory)[1]) * 1024
    pressure = Path("/proc/pressure/io").read_text()
    io_full = float(re.search(r"full avg10=([\d.]+)", pressure)[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    device_index = plan["device"].split(":")[1]
    query = subprocess.check_output(["nvidia-smi", "-i", device_index,
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    actual_uuid, free_mib = query[0].strip(), int(query[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    gpu_pids = [int(line.split(",")[1]) for line in processes if actual_uuid in line]
    host_ready = (available >= policy["minimum_host_available_gib"] * 1024**3
        and load <= policy["load_per_cpu_max"] * cpus
        and io_full < policy["io_full_avg10_percent_max"])
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=actual_uuid, gpu_free_mib=free_mib,
        gpu_compute_pids=gpu_pids, host_ready=host_ready,
        gpu_ready=host_ready and actual_uuid == plan["gpu_uuid"]
            and free_mib >= policy["gpu_free_mib"] and not gpu_pids)


def admission(root, plan, job, stage, gpu):
    ready = 0
    while True:
        row = resource_snapshot(plan)
        if row["gpu_uuid"] != plan["gpu_uuid"]:
            raise RuntimeError("Physical GPU3 UUID differs from the frozen plan")
        ready = ready + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(plan["resource_policy"]["two_gpu_checks_seconds_apart"] if ready else 30)


def sampler_benchmark(root):
    from recurrent_rg.data.phi4 import sample_hmc_chain
    root = Path(root).resolve()
    plan, _ = load_plan(root, cpu_only=True)
    physical, preflight, policy = (plan["physical"]["options"],
        plan["sampler_preflight"], plan["resource_policy"])
    rows = []
    for r in preflight["r_values"]:
        for initial in preflight["initial_states"]:
            seed = preflight["seed_base"] + len(rows)
            row = sample_hmc_chain(r, 1., size=physical["sizes"][0], dimension=2,
                samples=preflight["samples"], burn_in=preflight["burn_in"],
                spacing=preflight["spacing"], step_size=physical["step_size"],
                leapfrog_steps=physical["leapfrog_steps"], seed=seed, initial=initial)
            rows.append(dict(r=r, initial=initial, seed=seed,
                trajectories=row["hmc_trajectories"], cpu_seconds=row["cpu_seconds"],
                wall_seconds=row["wall_seconds"], burn_in_acceptance=row["burn_in_acceptance"],
                production_acceptance=row["production_acceptance"],
                preflight_not_scientific_data=True))
            print(json.dumps(dict(event="sampler_preflight_chain_complete", **rows[-1])), flush=True)
    projection = (max(row["cpu_seconds"] / row["trajectories"] for row in rows)
        * plan["physical"]["total_hmc_trajectories"] * preflight["throughput_safety_factor"])
    preflight_cpu = sum(row["cpu_seconds"] for row in rows)
    acceptance_passed = all(row["production_acceptance"] >= preflight["minimum_production_acceptance"]
                            for row in rows)
    receipt = dict(status="complete", physical_production_started=False, rows=rows,
        projected_sampling_cpu_seconds_conservative=projection,
        preflight_cpu_seconds=preflight_cpu, extra_cpu_reserve_seconds=policy["gpu_seconds"],
        cpu_budget_seconds=policy["cpu_seconds"],
        budget_passed=projection + preflight_cpu + policy["gpu_seconds"] <= policy["cpu_seconds"],
        acceptance_passed=acceptance_passed,
        minimum_production_acceptance=preflight["minimum_production_acceptance"],
        sampler_parameters_changed_automatically=False,
        short_pilots_do_not_certify_equilibrium=True)
    unchanged(plan)
    save(root / "sampler_preflight.json", receipt, exclusive=True)
    return receipt


def physical_command(root, plan):
    """Pass every pinned option explicitly; the sampler's defaults cannot drift."""
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4", "--execute",
               "--output", str(root / "data")]
    for name, value in plan["physical"]["options"].items():
        command.append("--" + name.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def training_command(root, plan, *, profile, max_seconds=None):
    training, policy = plan["training"], plan["resource_policy"]
    seconds = policy["gpu_seconds"] if max_seconds is None else max_seconds
    return [sys.executable, "-u", "-m", "experiments.phi4_qcr", "--data", str(root / "data"),
        "--output", str(root / ("profile_run" if profile else "run")), "--device", plan["device"],
        "--steps", str(training["profile_steps"] if profile else training["steps"]),
        "--system-batch", str(training["system_batch"]), "--sample-batch", str(training["sample_batch"]),
        "--seed", str(training["profile_seed"] if profile else training["seed"]), "--threads", "1",
        "--log-every", "10" if profile else "100", "--checkpoint-every", "20" if profile else "1000",
        "--information-pairs", str(training["information_pairs"]), "--max-training-seconds", str(seconds),
        "--max-cuda-bytes", str(policy["gpu_peak_bytes"]), "--expected-gpu-uuid", plan["gpu_uuid"]]


def public_files(root):
    files = [path for folder in (root / "run", root / "report")
             for path in folder.iterdir() if path.is_file() and path.suffix in (".json", ".npz", ".png")]
    files += [root / name for name in ("plan.json", "launch.json", "execution.json",
                                      "sampler_preflight.json", "gpu_preflight.json")]
    files += [root / "data" / name for name in ("plan.json", "manifest.json", "status.json")]
    files += sorted((root / "data" / "conditions").glob("*/sidecar.json"))
    return {str(path.relative_to(root)): sha(path) for path in sorted(files)}


def execute(root):
    root = Path(root).resolve()
    plan, job = load_plan(root)
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Workflow already executed; no restart")
    save(root / "execution.json", dict(parent_pid=os.getpid(), at=now(),
        plan_sha256=job["plan_sha256"], new_experiment=True, resumed=False), exclusive=True)
    os.nice(plan["resource_policy"]["nice"])
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(),
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    commands = [
        ("sampler_preflight", [sys.executable, "-u", "-m", MODULE,
            "--sampler-benchmark", "--output", str(root)], False),
        ("physical_sampling", physical_command(root, plan), False),
        ("training_preflight", training_command(root, plan, profile=True), True),
        ("training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr",
            "--run", str(root / "run"), "--output", str(root / "report")], False)]
    policy, gpu_elapsed = plan["resource_policy"], 0.
    try:
        for name, command, gpu in commands:
            load_plan(root)
            if directory_bytes(root) > policy["directory_bytes"]:
                job.update(status="budget_hold", requires_user_action=True,
                    error="Experiment directory exceeds its frozen budget; no automatic deletion")
                break
            admission(root, plan, job, name, gpu)
            if name == "training":
                command = training_command(root, plan, profile=False,
                    max_seconds=max(1., policy["gpu_seconds"] - gpu_elapsed))
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log, started = root / "logs" / f"{name}.log", time.monotonic()
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
                ended_at=now(), wall_seconds=time.monotonic() - started)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job["gpu_stage_wall_seconds"] = gpu_elapsed
            save(root / "job.json", job)
            if returncode:
                child_status = root / ("profile_run" if name == "training_preflight" else "run") / "status.json"
                if child_status.is_file() and read(child_status).get("status") == "budget_hold":
                    job.update(status="budget_hold", requires_user_action=True,
                        error=f"{name} reached its fixed resource budget; outputs preserved")
                    break
                raise RuntimeError(f"{name} returned {returncode}; no automatic retry")
            load_plan(root)
            if name == "sampler_preflight":
                receipt = read(root / "sampler_preflight.json")
                if not receipt["acceptance_passed"]:
                    job.update(status="sampling_hold", requires_user_action=True,
                        error="Fixed-integrator warm/cold acceptance preflight not passed; production not started")
                    break
                if not receipt["budget_passed"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="CPU preflight budget not passed; production not started")
                    break
            if name == "physical_sampling":
                status = read(root / "data" / "status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Sampling did not finish successfully")
                job["sampling_warnings"] = status.get("warning_count")
                job["sampling_actual_cpu_seconds"] = status["cpu_seconds"]
                pilot_cpu = read(root / "sampler_preflight.json")["preflight_cpu_seconds"]
                if status["cpu_seconds"] + pilot_cpu + policy["gpu_seconds"] > policy["cpu_seconds"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="Actual sampling CPU plus remaining conservative reserve exceeds CPU budget")
                    break
            if name == "training_preflight":
                summary = read(root / "profile_run" / "summary.json")
                actual_uuid = str(summary["device"].get("uuid", "")).lower().removeprefix("gpu-")
                if actual_uuid != plan["gpu_uuid"].lower().removeprefix("gpu-"):
                    raise RuntimeError("Torch logical GPU UUID mismatch in training preflight")
                if summary.get("actual_steps") != plan["training"]["profile_steps"]:
                    raise RuntimeError("Training preflight endpoint differs from20")
                seconds = summary["timing"]["training_seconds"]
                projection = (seconds / plan["training"]["profile_steps"] * plan["training"]["steps"]
                    * policy["gpu_profile_safety_factor"] + policy["gpu_final_reserve_seconds"] + gpu_elapsed)
                receipt = dict(projected_total_gpu_seconds_conservative=projection,
                    gpu_budget_seconds=policy["gpu_seconds"], profile_is_not_formal_result=True,
                    budget_passed=projection <= policy["gpu_seconds"])
                save(root / "gpu_preflight.json", receipt, exclusive=True)
                if not receipt["budget_passed"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="GPU preflight projection exceeds frozen budget; fixed steps were not reduced")
                    break
            if name == "training":
                summary = read(root / "run" / "summary.json")
                steps = plan["training"]["steps"]
                if (summary.get("actual_steps") != steps
                        or summary.get("labels_used_in_training") is not False
                        or summary.get("frozen_evaluation_verified") is not True
                        or any(row.get("min") != steps or row.get("max") != steps
                               for row in summary.get("optimizer_steps", {}).values())
                        or set(summary.get("optimizer_steps", {})) != {"Q", "C", "R"}):
                    raise RuntimeError("Formal endpoint, independent optimizer clock or frozen label-free evaluation mismatch")
                job["actual_training_steps"] = steps
            if name == "report":
                status = read(root / "report" / "status.json")
                if status.get("status") != "complete" or status.get("workflow_complete") is not True:
                    raise RuntimeError("Final public report is incomplete")
        else:
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
            job["public_files"] = public_files(root)
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, error=f"{type(error).__name__}: {error}",
            requires_user_action=True, ended_at=now())
        raise
    finally:
        save(root / "job.json", job)


def launch(root):
    root = Path(root).resolve()
    plan, job = load_plan(root)
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Not newly prepared; no restart")
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--output", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
            plan_sha256=sha(root / "plan.json"), new_experiment=True, resumed=False,
            device=plan["device"], gpu_uuid=plan["gpu_uuid"])
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute", "sampler-benchmark"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--leapfrog-steps", type=int)
    parser.add_argument("--cpu-seconds", type=float)
    parser.add_argument("--gpu-seconds", type=float)
    parser.add_argument("--directory-limit", type=int)
    parser.add_argument("--force-site-cap", type=int)
    args = parser.parse_args(argv)
    options = {key: value for key, value in vars(args).items()
        if key in {"size", "step_size", "leapfrog_steps", "cpu_seconds", "gpu_seconds", "directory_limit", "force_site_cap"}
        and value is not None}
    if options and not args.prepare:
        parser.error("Physical settings and budgets are prepare-only; execution reads the frozen plan")
    if args.sampler_benchmark:
        result = sampler_benchmark(args.output)
    elif args.prepare:
        plan = prepare(args.output, **options)
        result = dict(status="prepared", output=plan["output"], source_files=len(plan["source_sha256"]))
    elif args.launch:
        result = launch(args.output)
    else:
        execute(args.output)
        result = read(args.output / "job.json")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
