"""新建、门控并执行连续场 phi4 小实验；绝不恢复或覆盖历史实验。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


HOST = "thinklab-105-225"
GPU = "cuda:2"
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
STEPS = 3000
PROFILE_STEPS = 20
CPU_SECONDS = 7200
GPU_SECONDS = 1800
DIRECTORY_BYTES = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


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
    return {str(path.relative_to(code)): sha(path)
            for folder in ("recurrent_rg", "scripts", "experiments")
            for path in sorted((code / folder).rglob("*.py"))}


def unchanged(plan):
    code = Path(plan["code_root"])
    for name, digest in plan["source_sha256"].items():
        if sha(code / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")


def prepare(root):
    from recurrent_rg.data.phi4 import Phi4Options, generation_plan
    if socket.gethostname() != HOST:
        raise ValueError("This physical-device plan is for thinklab-105-225 only")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Do not reinterpret physical GPU indices through a CUDA mask")
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing output is never overwritten or resumed")
    plan = dict(schema_version=5, recipe="phi4_continuous_gaussian_soft_statistics_v1",
        code_root=str(code), output=str(root), host=HOST, device=GPU, gpu_uuid=GPU_UUID,
        prepared_at=now(), source_sha256=source_hashes(code),
        physical=generation_plan(Phi4Options()),
        training=dict(steps=STEPS, profile_steps=PROFILE_STEPS, theory_dim=1, hidden=96,
            lattice_sizes=[36, 12, 4], system_batch=8, sample_batch=64, seed=91,
            optimizer="three independent fresh Adam", learning_rate=.001, schedule="constant",
            independent_gradients=True, closure_scale_weights=[1, 1], labels_used=False,
            warm_start=False, pca_initialization=False),
        resource_policy=dict(cpu_threads=1, nice=10, gpu_peak_bytes=8 * 1024 ** 3,
            gpu_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS, directory_bytes=DIRECTORY_BYTES,
            minimum_host_available_gib=12, gpu_free_mib=12288, gpu_compute_processes=0,
            io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, occupied_resources="wait without intervention"),
        authorization="User approved a new continuous-field recipe and starting a small dimg=1 experiment",
        scientific_limits=["正测度欧氏实场，不是实时复路径积分或符号问题求解",
            "核平滑分箱信息代理不是旧二值精确直方图 MI，也不是 InfoNCE",
            "有限链未认证平衡；所有警告保留；不会自动加样、换 seed 或重跑",
            "单 r 族与 dimg=1 不保证闭包、正确相结构或普适指数；thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now()), exclusive=True)
    return plan


def resource_snapshot():
    import re
    memory = Path("/proc/meminfo").read_text()
    available = int(re.search(r"MemAvailable:\s+(\d+)", memory)[1]) * 1024
    pressure = Path("/proc/pressure/io").read_text()
    io_full = float(re.search(r"full avg10=([\d.]+)", pressure)[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    query = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=uuid,memory.free",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
    actual_uuid, free_mib = query[0].strip(), int(query[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"], text=True).splitlines()
    gpu_pids = [int(line.split(",")[1]) for line in processes if actual_uuid in line]
    host_ready = available >= 12 * 1024 ** 3 and load <= 1.25 * cpus and io_full < 20
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io_full, gpu_uuid=actual_uuid, gpu_free_mib=free_mib,
        gpu_compute_pids=gpu_pids, host_ready=host_ready,
        gpu_ready=host_ready and actual_uuid == GPU_UUID and free_mib >= 12288 and not gpu_pids)


def admission(root, job, stage, gpu):
    ready = 0
    while True:
        row = resource_snapshot()
        if row["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU2 UUID differs from the plan")
        ready = ready + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 30)


def sampler_benchmark(output):
    from recurrent_rg.data.phi4 import sample_hmc_chain
    rows = []
    for index, r in enumerate((-2., -.75, .5)):
        row = sample_hmc_chain(r, 1., size=36, dimension=2, samples=8, burn_in=24,
            spacing=4, step_size=.08, leapfrog_steps=12, seed=2026102200 + index)
        rows.append(dict(r=r, trajectories=row["hmc_trajectories"],
            cpu_seconds=row["cpu_seconds"], wall_seconds=row["wall_seconds"],
            production_acceptance=row["production_acceptance"],
            preflight_not_scientific_data=True))
    projection = max(row["cpu_seconds"] / row["trajectories"] for row in rows) * 90112 * 2
    receipt = dict(status="complete", physical_production_started=False, rows=rows,
        projected_sampling_cpu_seconds_conservative=projection,
        extra_cpu_reserve_seconds=1800, cpu_budget_seconds=CPU_SECONDS,
        budget_passed=projection + 1800 <= CPU_SECONDS,
        short_pilots_do_not_certify_equilibrium=True)
    save(output, receipt, exclusive=True)
    return receipt


def training_command(root, *, profile):
    return [sys.executable, "-u", "-m", "experiments.phi4_qcr", "--data", str(root / "data"),
        "--output", str(root / ("profile_run" if profile else "run")), "--device", GPU,
        "--steps", str(PROFILE_STEPS if profile else STEPS), "--system-batch", "8",
        "--sample-batch", "64", "--seed", "191" if profile else "91", "--threads", "1",
        "--log-every", "10" if profile else "100", "--checkpoint-every", "20" if profile else "1000",
        "--information-pairs", "128", "--max-training-seconds", "1800",
        "--expected-gpu-uuid", GPU_UUID]


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*")
               if path.is_file() and not path.is_symlink())


def execute(root):
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if socket.gethostname() != HOST or job["status"] != "prepared" or job["stages"]:
        raise ValueError("Wrong host or workflow already executed; no restart")
    os.nice(10)
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    commands = [
        ("sampler_preflight", [sys.executable, "-u", "-m", "scripts.run_phi4_qcr",
            "--sampler-benchmark", "--output", str(root / "sampler_preflight.json")], False),
        ("physical_sampling", [sys.executable, "-u", "-m", "scripts.prepare_phi4",
            "--execute", "--output", str(root / "data")], False),
        ("training_preflight", training_command(root, profile=True), True),
        ("training", training_command(root, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr",
            "--run", str(root / "run"), "--output", str(root / "report")], False)]
    gpu_elapsed = 0.
    try:
        for name, command, gpu in commands:
            unchanged(plan)
            if directory_bytes(root) > DIRECTORY_BYTES:
                job.update(status="budget_hold", requires_user_action=True,
                    error="Experiment directory exceeds its 1GiB budget; no automatic deletion")
                break
            admission(root, job, name, gpu)
            if name == "training":
                # This is an execution budget, not a change to the fixed model,
                # optimizer schedule, sampling plan or 3000-step endpoint.
                command[command.index("--max-training-seconds") + 1] = str(max(1., GPU_SECONDS - gpu_elapsed))
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = root / "logs" / f"{name}.log"
            started = time.monotonic()
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
            unchanged(plan)
            if name == "sampler_preflight" and not read(root / "sampler_preflight.json")["budget_passed"]:
                job.update(status="budget_hold", requires_user_action=True, error="CPU preflight budget not passed")
                break
            if name == "physical_sampling":
                status = read(root / "data" / "status.json")
                if not status.get("workflow_complete") or status.get("returncode") != 0:
                    raise RuntimeError("Sampling did not finish successfully")
                job["sampling_warnings"] = status.get("warning_count")
                job["sampling_actual_cpu_seconds"] = status["cpu_seconds"]
                if status["cpu_seconds"] + GPU_SECONDS > CPU_SECONDS:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="Actual sampling CPU plus remaining conservative reserve exceeds CPU budget")
                    break
            if name == "training_preflight":
                summary = read(root / "profile_run" / "summary.json")
                actual_uuid = str(summary["device"].get("uuid", "")).lower().removeprefix("gpu-")
                if actual_uuid != GPU_UUID.lower().removeprefix("gpu-"):
                    raise RuntimeError("Torch logical GPU UUID mismatch in training preflight")
                if summary.get("actual_steps") != PROFILE_STEPS:
                    raise RuntimeError("Training preflight endpoint differs from 20")
                seconds = summary["timing"]["training_seconds"]
                projection = seconds / PROFILE_STEPS * STEPS * 1.5 + 300 + gpu_elapsed
                receipt = dict(projected_total_gpu_seconds_conservative=projection,
                    gpu_budget_seconds=GPU_SECONDS, profile_is_not_formal_result=True,
                    budget_passed=projection <= GPU_SECONDS)
                save(root / "gpu_preflight.json", receipt, exclusive=True)
                if not receipt["budget_passed"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="GPU preflight projection exceeds 1800 seconds; fixed steps were not reduced")
                    break
            if name == "training":
                summary = read(root / "run" / "summary.json")
                if summary.get("actual_steps") != STEPS:
                    raise RuntimeError("Formal optimizer endpoint differs from 3000")
                if (summary.get("labels_used_in_training") is not False
                        or summary.get("frozen_evaluation_verified") is not True
                        or any(row.get("min") != STEPS or row.get("max") != STEPS
                               for row in summary.get("optimizer_steps", {}).values())
                        or set(summary.get("optimizer_steps", {})) != {"Q", "C", "R"}):
                    raise RuntimeError("Independent optimizer clock or frozen label-free evaluation mismatch")
                job["actual_training_steps"] = STEPS
            if name == "report":
                status = read(root / "report" / "status.json")
                if status.get("status") != "complete" or status.get("workflow_complete") is not True:
                    raise RuntimeError("Final public report is incomplete")
        else:
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
            files = [path for folder in (root / "run", root / "report")
                     for path in folder.iterdir() if path.is_file() and path.suffix in (".json", ".npz", ".png")]
            files += [root / "plan.json", root / "launch.json", root / "sampler_preflight.json", root / "gpu_preflight.json"]
            files += [root / "data" / name for name in ("plan.json", "manifest.json", "status.json")]
            files += sorted((root / "data" / "conditions").glob("*/sidecar.json"))
            job["public_files"] = {str(path.relative_to(root)): sha(path) for path in sorted(files)}
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, error=f"{type(error).__name__}: {error}",
            requires_user_action=True, ended_at=now())
        raise
    finally:
        save(root / "job.json", job)


def launch(root):
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if socket.gethostname() != HOST or job["status"] != "prepared":
        raise ValueError("Wrong host or not prepared; no restart")
    unchanged(plan)
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_phi4_qcr", "--execute",
            "--output", str(root)], cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
            plan_sha256=sha(root / "plan.json"), new_experiment=True, resumed=False)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute", "sampler-benchmark"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sampler_benchmark:
        result = sampler_benchmark(args.output)
    elif args.prepare:
        plan = prepare(args.output)
        result = dict(status="prepared", output=plan["output"], source_files=len(plan["source_sha256"]))
    elif args.launch:
        result = launch(args.output)
    else:
        execute(args.output)
        result = read(args.output / "job.json")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
