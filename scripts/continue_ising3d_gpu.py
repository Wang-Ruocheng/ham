"""User-authorized GPU-only handoff of an unstarted 3D Ising training stage.

This external execution adapter never edits the frozen scientific code/plan,
regenerates data, rebuilds MI, resumes weights, or changes the recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time

GPU_INDEX = 0
GPU_UUID = "GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42"
HANDOFF = "resource_handoff_gpu0_v1"
CPU_STAGES = ("physical_sampling", "prepare_data", "full_shell_MI")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, state=stat[0], start_ticks=stat[19],
            argv=(directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))
    except FileNotFoundError:
        return None


def children(pid):
    return (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text().split()


def validate_waiting(root, job):
    if (job.get("status") != "waiting_resources" or job.get("current") != "gpu_preflight"
            or job.get("error") is not None or job.get("requires_user_action")):
        raise ValueError("Only an error-free waiter before the first GPU stage can be handed off")
    stages = job.get("stages", [])
    if tuple(s.get("name") for s in stages) != CPU_STAGES or any(
            s.get("status") != "complete" or s.get("returncode") != 0 for s in stages):
        raise ValueError("Require precisely the three completed CPU stages, no GPU stage")
    if any((root / name).exists() for name in ("gpu_preflight", "run", "report")):
        raise ValueError("GPU outputs already exist; never restart or overwrite them")


def verify_science(root, expected=None):
    plan = read(root / "plan.json")
    for relative, digest in plan["source_fingerprints"].items():
        path = (root / "code" / relative).resolve()
        if not path.is_relative_to(root / "code") or sha(path) != digest:
            raise RuntimeError("Frozen scientific source mismatch")
    receipt = read(root / "mi_receipt.json")
    config = read(root / "config.json")
    if receipt.get("config_sha256") != sha(root / "config.json"):
        raise RuntimeError("Configuration changed after MI preprocessing")
    if config["training"]["epochs"] != 6000 or plan["fixed_training_updates"] != 6000:
        raise ValueError("The user-authorized endpoint remains exactly 6000 updates")
    cache = Path(config["data"]["cache"]).resolve()
    if cache != root / "data/ensembles.pt" or receipt.get("status") != "complete":
        raise ValueError("Unexpected or incomplete original data/MI source")
    if sha(cache) != receipt["data_sha256"] or read(root / "data/evaluation_metadata.json")["cache_sha256"] != receipt["data_sha256"]:
        raise RuntimeError("Original training data fingerprint mismatch")
    fingerprints = {"config.json": sha(root / "config.json"), "plan.json": sha(root / "plan.json"),
        "data/ensembles.pt": receipt["data_sha256"], "mi_receipt.json": sha(root / "mi_receipt.json")}
    for split in ("train", "test"):
        path = Path(receipt["splits"][split]["cache"]["path"]).resolve()
        if not path.is_relative_to(root / "mi_cache"):
            raise ValueError("MI cache path leaves this experiment")
        fingerprints[str(path.relative_to(root))] = sha(path)
    if expected is not None and fingerprints != expected:
        raise RuntimeError("Data, MI, config or plan changed during resource handoff")
    return fingerprints


def prepare(root):
    root = Path(root).resolve()
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Require the verified 225 host")
    directory = root / HANDOFF
    if directory.exists():
        raise FileExistsError("Handoff already prepared; inspect its receipt instead of repeating")
    job = read(root / "job.json")
    validate_waiting(root, job)
    fingerprints = verify_science(root)
    pid = job["parent_pid"]
    identity = process_identity(pid)
    expected = ["-m", "scripts.run_ising3d", "--root", str(root), "--device", "cuda:2"]
    if identity is None or identity["argv"][1:] != expected or children(pid):
        raise RuntimeError("Waiter identity/children differ; do not stop any process")
    opener = getattr(os, "pidfd_open", None)
    descriptor = opener(pid) if callable(opener) and hasattr(signal, "pidfd_send_signal") else None
    def deliver(sig):
        current = process_identity(pid)
        if current is None or current["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("Process identity changed; do not signal a reused PID")
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(pid, sig)
    deliver(signal.SIGSTOP)
    terminate = False
    try:
        # Close the check/spawn race before sending TERM: only the same,
        # child-free, still-waiting parent may be terminated.
        current = None
        for _ in range(50):
            current = process_identity(pid)
            if current is None or current["state"] in ("T", "t"):
                break
            time.sleep(.02)
        if current is not None and current["state"] not in ("T", "t"):
            raise RuntimeError("Parent did not quiesce; do not terminate or replace it")
        if current is None or current["start_ticks"] != identity["start_ticks"] or children(pid):
            raise RuntimeError("Parent changed or spawned a child; preserve it")
        job = read(root / "job.json")
        validate_waiting(root, job)
        directory.mkdir()
        shutil.copy2(Path(__file__), directory / "continue_ising3d_gpu.py")
        write(directory / "original_job.json", job)
        record = dict(schema_version=1, status="prepared", authority="User explicitly approved GPU0 handoff before training",
            old_parent=identity, old_device="cuda:2", device="cuda:0", gpu_uuid=GPU_UUID,
            original_fingerprints=fingerprints, scientific_code_unchanged=True,
            physical_sampling_updates=0, MI_rebuilds=0, trained_updates_before_handoff=0,
            original_CPU_stages_preserved=True, adapter_sha256=sha(directory / "continue_ising3d_gpu.py"),
            prepared_at=datetime.now(timezone.utc).isoformat())
        write(directory / "plan.json", record)
        deliver(signal.SIGTERM)
        terminate = True
    finally:
        # A pending TERM executes after CONT; on a validation failure this
        # simply restores the untouched original waiter instead.
        final_identity = process_identity(pid)
        if final_identity is not None and final_identity["start_ticks"] == identity["start_ticks"]:
            deliver(signal.SIGCONT)
        if descriptor is not None:
            os.close(descriptor)
    for _ in range(50):
        state = process_identity(pid)
        if state is None or state["state"] == "Z":
            break
        if state["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("PID reused during handoff; abort without signalling it")
        time.sleep(.1)
    else:
        raise RuntimeError("Original waiter did not exit; do not launch another scheduler")
    if not terminate:
        raise RuntimeError("Waiter was not stopped")
    record.update(status="handoff_ready", old_waiter_stopped=True)
    write(directory / "plan.json", record)
    job.update(status="handoff_ready", parent_pid=None, device="cuda:0",
        resource_handoff=dict(plan=str(directory / "plan.json"), from_device="cuda:2", to_device="cuda:0",
            old_parent_pid=pid, original_job=str(directory / "original_job.json"), scientific_plan_unmodified=True))
    write(root / "job.json", job)
    return record


def snapshot():
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Wrong host")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    selected = next(line.split(",") for line in query.splitlines() if int(line.split(",")[0]) == GPU_INDEX)
    if selected[1].strip() != GPU_UUID:
        raise RuntimeError("GPU0 physical UUID mismatch")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    active = [line for line in compute.splitlines() if GPU_UUID in line]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    row = dict(checked_at=time.time(), gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID, active=active,
        free_mib=int(selected[2]), mem_available_gib=available / 1024**3, load1=os.getloadavg()[0],
        cpu_count=os.cpu_count(), io_full_avg10=io)
    row["ready"] = not active and row["free_mib"] >= 12288 and row["mem_available_gib"] >= 36 and row["load1"] <= 1.25 * row["cpu_count"] and io < 20
    return row


def running_old(record):
    old = record["old_parent"]
    actual = process_identity(old["pid"])
    return actual is not None and actual["start_ticks"] == old["start_ticks"] and actual["state"] != "Z"


def launch(root):
    root = Path(root).resolve()
    directory = root / HANDOFF
    record = read(directory / "plan.json")
    if record.get("status") != "handoff_ready" or running_old(record):
        raise RuntimeError("Old scheduler is not safely stopped")
    verify_science(root, record["original_fingerprints"])
    if (directory / "launch.json").exists() or (directory / "parent.log").exists():
        raise FileExistsError("Existing launch receipt/log; no duplicate launch")
    adapter = directory / "continue_ising3d_gpu.py"
    if sha(adapter) != record["adapter_sha256"]:
        raise RuntimeError("Execution adapter fingerprint changed")
    command = ["nice", "-n", "10", sys.executable, str(adapter), "--root", str(root), "--run"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with (directory / "parent.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = dict(parent_pid=child.pid, command=command, started_at=datetime.now(timezone.utc).isoformat(),
        device="cuda:0", gpu_uuid=GPU_UUID, data_and_scientific_code_unchanged=True)
    write(directory / "launch.json", receipt)
    return receipt


def cpu_seconds():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def run(root):
    root = Path(root).resolve()
    directory = root / HANDOFF
    record = read(directory / "plan.json")
    if running_old(record) or sha(Path(__file__)) != record["adapter_sha256"]:
        raise RuntimeError("Old scheduler active or adapter changed")
    fd = os.open(directory / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    verify_science(root, record["original_fingerprints"])
    job = read(root / "job.json")
    if job["status"] != "handoff_ready":
        raise ValueError("Expected prepared, never-started GPU handoff")
    job.update(parent_pid=os.getpid(), device="cuda:0", status="waiting_resources", current="gpu_preflight",
        handoff_started_at=datetime.now(timezone.utc).isoformat(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    write(root / "job.json", job)
    # Import only the original immutable execution command builder. Scientific
    # subprocesses execute from root/code; this adapter is never on their path.
    sys.path.insert(0, str(root / "code"))
    original = importlib.import_module("scripts.run_ising3d")
    plan = read(root / "plan.json")
    config = read(root / "config.json")
    started, cpu_start = time.monotonic(), cpu_seconds()
    prior_cpu, prior_gpu = job["CPU_core_seconds"], job["GPU_stage_seconds"]
    try:
        commands = [stage for stage in original.stages(root, plan, "cuda:0") if stage[2]]
        if [stage[0] for stage in commands] != ["gpu_preflight", "train", "report"]:
            raise ValueError("Unexpected GPU-only execution stages")
        for name, command, _ in commands:
            original.verify_code(root / "code", plan["source_fingerprints"])
            job.update(status="waiting_resources", current=name)
            consecutive = 0
            while True:
                row = snapshot()
                job.setdefault("resource_checks", []).append(row)
                job["resource_checks"] = job["resource_checks"][-360:]
                consecutive = consecutive + 1 if row["ready"] else 0
                write(root / "job.json", job)
                if consecutive == 2:
                    break
                time.sleep(10)
            target = root / ("run" if name == "train" else name)
            if target.exists():
                raise FileExistsError("Stage output exists; do not restart")
            log_path = root / "logs" / (name + ".log")
            stage = dict(name=name, status="running", command=command, gpu_stage=True, device="cuda:0",
                gpu_uuid=GPU_UUID, started_at=datetime.now(timezone.utc).isoformat(), log=str(log_path))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            env.pop("CUDA_VISIBLE_DEVICES", None)
            stage_start = time.monotonic()
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running", current=name)
                write(root / "job.json", job)
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed", elapsed_seconds=time.monotonic() - stage_start)
            job["GPU_stage_seconds"] = prior_gpu + sum(s.get("elapsed_seconds", 0) for s in job["stages"] if s.get("gpu_stage"))
            job["CPU_core_seconds"] = prior_cpu + cpu_seconds() - cpu_start
            write(root / "job.json", job)
            if rc:
                raise RuntimeError(f"Stage {name} exited {rc}; preserve files, no automatic restart")
            if name in ("gpu_preflight", "train"):
                path = root / ("gpu_preflight" if name == "gpu_preflight" else "run") / "summary.json"
                summary = read(path)
                required = 5 if name == "gpu_preflight" else config["training"]["epochs"]
                if summary["actual_epochs"] != required:
                    raise RuntimeError("Stage ended before its fixed update count")
                if name == "gpu_preflight":
                    steady = sorted(summary["timing"].get("steady_step_seconds", []))
                    job["GPU_execution_preflight"] = dict(scientific_result=False, timing=summary["timing"],
                        raw_update_only_projection_seconds=None if not steady else steady[len(steady)//2] * config["training"]["epochs"],
                        projection_excludes="Logging/evaluation/checkpoint overhead and resource waits")
                else:
                    job["training_timing"] = summary["timing"]
        verify_science(root, record["original_fingerprints"])
        report = read(root / "report/report.json")
        if report["status"] != "complete" or not report["model_state_unchanged"]:
            raise RuntimeError("Frozen-model report incomplete or changed parameters")
        job.update(status="complete", workflow_complete=True, current=None,
            finished_at=datetime.now(timezone.utc).isoformat(),
            public_files=original.public_files(root) + [f"{HANDOFF}/{name}" for name in ("plan.json", "original_job.json", "launch.json")])
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        job.update(CPU_core_seconds=prior_cpu + cpu_seconds() - cpu_start,
            handoff_elapsed_seconds=time.monotonic() - started)
        write(root / "job.json", job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--launch", action="store_true")
    choice.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root) if args.prepare else launch(args.root) if args.launch else run(args.root)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
