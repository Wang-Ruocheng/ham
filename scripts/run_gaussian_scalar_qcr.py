"""独立高斯标量场 QCR 校准父程序；只新建、门控、执行，不覆盖或恢复。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time


HOST = "thinklab-105-225"
GPU = "cuda:1"
GPU_INDEX = "1"
GPU_UUID = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"
STEPS = 3000
PROFILE_STEPS = 20
CPU_SECONDS = 7200
GPU_SECONDS = 1800
GPU_PEAK_BYTES = 8 * 1024 ** 3
DIRECTORY_BYTES = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(content)
    else:
        temporary = path.with_suffix(path.suffix + ".writing")
        temporary.write_text(content)
        temporary.replace(path)


def source_hashes(code):
    code = Path(code)
    return {str(path.relative_to(code)): sha(path)
            for folder in ("recurrent_rg", "scripts", "experiments")
            for path in sorted((code / folder).rglob("*.py"))}


def unchanged(plan):
    actual = source_hashes(Path(plan["code_root"]))
    if set(actual) != set(plan["source_sha256"]):
        raise RuntimeError("Frozen source changed: Python source inventory differs")
    for name, expected in plan["source_sha256"].items():
        if actual[name] != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def validate_host():
    if socket.gethostname() != HOST:
        raise ValueError(f"This frozen physical-device plan is for {HOST} only")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Do not reinterpret the physical GPU through a CUDA mask")


def prepare(root):
    validate_host()
    from recurrent_rg.data.gaussian_scalar import GaussianScalarOptions, generation_plan
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing output is never overwritten or resumed")
    if root == code or root in code.parents:
        raise ValueError("Experiment output must not contain or replace its code snapshot")
    plan = dict(schema_version=5, recipe="gaussian_scalar_u0_qcr_calibration_v1",
        code_root=str(code), output=str(root), host=HOST, device=GPU, gpu_uuid=GPU_UUID,
        prepared_at=now(), source_sha256=source_hashes(code),
        physical=generation_plan(GaussianScalarOptions()),
        training=dict(steps=STEPS, profile_steps=PROFILE_STEPS, theory_dim=1, hidden=96,
            lattice_sizes=[36, 12, 4], system_batch=8, sample_batch=64, seed=91,
            profile_seed=191, optimizer="three independent fresh Adam", learning_rate=.001,
            schedule="constant", independent_gradients=True, closure_objective="all_start_anchor",
            closure_scale_weights=None, labels_used=False, warm_start=False,
            profile_checkpoint_used_for_formal=False, pca_initialization=False),
        resource_policy=dict(cpu_threads=1, nice=10, gpu_peak_bytes=GPU_PEAK_BYTES,
            gpu_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS, directory_bytes=DIRECTORY_BYTES,
            minimum_host_available_gib=12, gpu_free_mib=12288, gpu_compute_processes=0,
            io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, occupied_resources="wait without intervention"),
        authorization="User approved the u=0, dimg=1 small calibration and restored the SSH connection",
        scientific_limits=["正质量周期边界高斯场可独立精确采样；没有 HMC 预热或生产链",
            "连续场 kernel 与信息代理沿用现有分支；有限样本信息偏差保留",
            "r>0 单侧有限尺寸数据不能认证临界固定点或双侧相变",
            "单训练 seed、dimg=1 不保证物理闭合；thermal_nu=null",
            "20 步 profile 是独立执行诊断，非正式科学结果；正式 seed91 全新初始化"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now()), exclusive=True)
    return plan


def resource_snapshot():
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io_full = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    query = subprocess.check_output(["nvidia-smi", "-i", GPU_INDEX,
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    actual_uuid, free_mib = query[0].strip(), int(query[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    pids = [int(line.split(",")[1]) for line in processes if actual_uuid in line]
    host_ready = available >= 12 * 1024 ** 3 and load <= 1.25 * cpus and io_full < 20
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=actual_uuid, gpu_free_mib=free_mib,
        gpu_compute_pids=pids, host_ready=host_ready,
        gpu_ready=host_ready and actual_uuid == GPU_UUID and free_mib >= 12288 and not pids)


def admission(root, job, stage, gpu):
    consecutive_ready = 0
    while True:
        row = resource_snapshot()
        if row["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU1 UUID differs from the frozen plan")
        consecutive_ready = consecutive_ready + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if consecutive_ready >= (2 if gpu else 1):
            return
        time.sleep(10 if consecutive_ready else 30)


def training_command(root, *, profile, remaining_gpu_seconds=GPU_SECONDS):
    return [sys.executable, "-u", "-m", "experiments.phi4_qcr", "--data", str(root / "data"),
        "--output", str(root / ("profile_run" if profile else "run")), "--device", GPU,
        "--steps", str(PROFILE_STEPS if profile else STEPS), "--system-batch", "8",
        "--sample-batch", "64", "--seed", "191" if profile else "91", "--threads", "1",
        "--log-every", "10" if profile else "100", "--checkpoint-every", "20" if profile else "1000",
        "--information-pairs", "128", "--max-training-seconds", str(remaining_gpu_seconds),
        "--max-cuda-bytes", str(GPU_PEAK_BYTES), "--expected-gpu-uuid", GPU_UUID]


def planned_commands(root):
    return [
        ("exact_data", [sys.executable, "-u", "-m", "scripts.prepare_gaussian_scalar", "--execute",
            "--output", str(root / "data")], False),
        ("training_preflight", training_command(root, profile=True), True),
        ("training", training_command(root, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr", "--run", str(root / "run"),
            "--output", str(root / "report")], False),
        ("gaussian_calibration_report", [sys.executable, "-u", "-m", "scripts.report_gaussian_scalar_qcr",
            "--data", str(root / "data"), "--run", str(root / "run"),
            "--output", str(root / "calibration_report")], False)]


def profile_projection(summary, history, profile_wall_seconds, gpu_elapsed):
    """Separate one-time initialization and first update from steady-state updates."""
    rows = history["history"]
    first, last = rows[0], rows[-1]
    if (first["step"] != 1 or last["step"] != PROFILE_STEPS
            or summary.get("actual_steps") != PROFILE_STEPS):
        raise RuntimeError("The independent 20-step profile is incomplete")
    training = float(summary["timing"]["training_seconds"])
    evaluation = float(summary["timing"]["evaluation_seconds"])
    summary_wall = float(summary["timing"]["wall_seconds"])
    first_update = float(first["elapsed_seconds"])
    steady = (float(last["elapsed_seconds"]) - first_update) / (last["step"] - 1)
    initialization = max(0., summary_wall - training - evaluation)
    process_startup = max(0., profile_wall_seconds - summary_wall)
    values = [training, evaluation, summary_wall, first_update, steady, initialization, process_startup,
              profile_wall_seconds, gpu_elapsed]
    if any(not math.isfinite(value) or value < 0 for value in values) or steady <= 0:
        raise RuntimeError("Invalid or non-increasing profile timing")
    formal = 1.5 * (initialization + process_startup + first_update
                    + (STEPS - 1) * steady + evaluation) + 300.
    projected_total = gpu_elapsed + formal
    return dict(status="complete", profile_is_not_formal_result=True,
        profile_optimizer_seed=191, formal_optimizer_seed=91, profile_wall_seconds=profile_wall_seconds,
        profile_initialization_seconds=initialization, profile_process_startup_seconds=process_startup,
        profile_first_update_seconds=first_update, profile_steady_seconds_per_update=steady,
        profile_full_evaluation_seconds=evaluation, formal_fixed_steps=STEPS,
        conservative_multiplier=1.5, final_writing_reserve_seconds=300,
        projected_formal_gpu_seconds_conservative=formal,
        actual_gpu_stage_wall_seconds=gpu_elapsed,
        projected_total_gpu_seconds_conservative=projected_total,
        gpu_budget_seconds=GPU_SECONDS, budget_passed=projected_total <= GPU_SECONDS,
        projection_is_not_a_wall_time_guarantee=True)


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*")
               if path.is_file() and not path.is_symlink())


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def hold(job, reason):
    job.update(status="budget_hold", workflow_complete=False, requires_user_action=True,
               error=reason, ended_at=now())


def validate_training_summary(summary, *, profile):
    expected_steps = PROFILE_STEPS if profile else STEPS
    if str(summary["device"].get("uuid", "")).lower().removeprefix("gpu-") != GPU_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("Torch physical GPU UUID differs from the frozen plan")
    if summary.get("actual_steps") != expected_steps:
        raise RuntimeError(f"Optimizer endpoint differs from {expected_steps}")
    optimizers = summary.get("optimizer_steps", {})
    if (summary.get("labels_used_in_training") is not False
            or summary.get("frozen_evaluation_verified") is not True
            or set(optimizers) != {"Q", "C", "R"}
            or any(row.get("min") != expected_steps or row.get("max") != expected_steps
                   for row in optimizers.values())):
        raise RuntimeError("Independent optimizer clock or frozen label-free evaluation mismatch")
    if summary.get("peak_cuda_bytes", GPU_PEAK_BYTES + 1) > GPU_PEAK_BYTES:
        raise RuntimeError("Training CUDA peak exceeds the frozen budget")


def final_public_files(root):
    files = [path for folder in ("run", "report", "calibration_report")
             for path in (root / folder).rglob("*")
             if path.is_file() and path.suffix in (".json", ".npz", ".png")]
    files += [root / name for name in ("plan.json", "launch.json", "execution.json", "gpu_preflight.json")]
    files += [root / "data" / name for name in ("plan.json", "manifest.json", "status.json")]
    for name in ("sidecar.json", "diagnostics.json", "status.json"):
        files += sorted((root / "data" / "conditions").glob("*/" + name))
    # Data tensors/checkpoints and all profile artifacts are deliberately excluded.
    return {str(path.relative_to(root)): sha(path) for path in sorted(set(files)) if path.is_file()}


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Workflow already executed; no restart")
    if plan["host"] != HOST or plan["device"] != GPU or plan["gpu_uuid"] != GPU_UUID:
        raise ValueError("Physical target differs from this runner")
    # Exclusive receipt is also a durable single-parent lock, retained after exit.
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST, started_at=now(),
                     plan_sha256=sha(root / "plan.json"))
    save(root / "execution.json", execution, exclusive=True)
    original_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if original_nice < 10:
        os.nice(10 - original_nice)
    started, cpu_started = time.monotonic(), cpu_seconds()
    gpu_elapsed = 0.
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        for name, command, gpu in planned_commands(root):
            unchanged(plan)
            if directory_bytes(root) > DIRECTORY_BYTES:
                hold(job, "Experiment directory exceeds 1GiB; no automatic deletion")
                break
            if cpu_seconds() - cpu_started >= CPU_SECONDS:
                hold(job, "Actual accumulated CPU exceeds 7200 seconds; fixed work was not reduced")
                break
            if gpu and gpu_elapsed >= GPU_SECONDS:
                hold(job, "Actual accumulated GPU exceeds 1800 seconds; fixed work was not reduced")
                break
            admission(root, job, name, gpu)
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
                ended_at=now(), wall_seconds=time.monotonic() - stage_started,
                cpu_seconds=cpu_seconds() - stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds() - cpu_started,
                       directory_bytes=directory_bytes(root))
            save(root / "job.json", job)
            if returncode:
                child_path = root / ("profile_run" if name == "training_preflight" else
                                     "run" if name == "training" else
                                     "calibration_report" if name == "gaussian_calibration_report" else
                                     "data" if name == "exact_data" else "report") / "status.json"
                if child_path.is_file() and read(child_path).get("status") == "budget_hold":
                    hold(job, f"{name} reached its fixed resource budget; outputs preserved")
                    break
                raise RuntimeError(f"{name} returned {returncode}; no automatic retry")
            unchanged(plan)
            if name == "exact_data":
                status = read(root / "data" / "status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Exact Gaussian data did not finish successfully")
                if read(root / "data" / "plan.json") != plan["physical"]:
                    raise RuntimeError("Executed data plan differs from the frozen physical plan")
                job["data_warning_count"] = status.get("warning_count", 0)
                job["sampling_actual_cpu_seconds"] = status["cpu_seconds"]
                if cpu_seconds() - cpu_started + GPU_SECONDS + 600 > CPU_SECONDS:
                    hold(job, "Actual data CPU plus conservative remaining reserve exceeds CPU budget")
                    break
            elif name in ("training_preflight", "training"):
                profile = name == "training_preflight"
                folder = root / ("profile_run" if profile else "run")
                summary = read(folder / "summary.json")
                validate_training_summary(summary, profile=profile)
                if profile:
                    receipt = profile_projection(summary, read(folder / "history_progress.json"),
                                                 stage["wall_seconds"], gpu_elapsed)
                    save(root / "gpu_preflight.json", receipt, exclusive=True)
                    job["gpu_projection"] = receipt
                    if not receipt["budget_passed"]:
                        hold(job, "GPU projection exceeds 1800 seconds; no sampling or step reduction")
                        break
                    if cpu_seconds() - cpu_started + receipt["projected_formal_gpu_seconds_conservative"] + 600 > CPU_SECONDS:
                        hold(job, "Actual CPU plus conservative remaining work exceeds 7200 seconds")
                        break
                else:
                    job["actual_training_steps"] = STEPS
            else:
                folder = "calibration_report" if name == "gaussian_calibration_report" else "report"
                status = read(root / folder / "status.json")
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
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"],
            ended_at=now(), cpu_seconds=job["cpu_seconds"], gpu_stage_wall_seconds=gpu_elapsed,
            wall_seconds=job["wall_seconds"], directory_bytes=job["directory_bytes"],
            actual_nice=job["actual_nice"], error=job["error"],
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
    value = dict(status="launching", at=now(), output=str(root), plan_sha256=sha(root / "plan.json"),
                 new_experiment=True, resumed=False)
    # Exclusive creation prevents two simultaneous launches even before the child starts.
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_gaussian_scalar_qcr", "--execute",
            "--output", str(root)], cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        value.update(status="launched", parent_pid=parent.pid)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare:
        plan = prepare(args.output)
        value = dict(status="prepared", output=plan["output"], source_files=len(plan["source_sha256"]))
    elif args.launch:
        value = launch(args.output)
    else:
        execute(args.output)
        value = read(args.output / "job.json")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if value["status"] not in ("failed", "budget_hold") else 3


if __name__ == "__main__":
    sys.exit(main())
