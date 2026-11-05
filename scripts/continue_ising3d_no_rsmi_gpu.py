"""Explicitly authorized idle-GPU handoff of an unstarted no-RSMI experiment.

An external execution adapter only: scientific code, config, plan, data and
initialization remain frozen. Never stop a training child or another user's job.
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


DEVICE = "cuda:3"
GPU_UUIDS = {
    0: "GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42",
    1: "GPU-18fac97c-11ee-4624-d298-9074e8d7028c",
    2: "GPU-713b07d0-118c-f939-c0a8-565390215e01",
    3: "GPU-a167decd-ce49-6c8f-6bcf-136fde596683",
}
HANDOFF = "resource_handoff_idle_gpu_v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


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
            or job.get("error") is not None or job.get("requires_user_action")
            or job.get("actual_training_updates") != 0 or job.get("stages") != []
            or job.get("GPU_stage_seconds") != 0):
        raise ValueError("Only a child-free error-free waiter before all GPU work can move")
    if any((root / name).exists() for name in ("gpu_preflight", "run", "report", "comparison")):
        raise ValueError("Stage output exists; preserve it, never restart or overwrite")


def verify_science(root, expected=None):
    root = Path(root).resolve()
    plan, config = read(root / "plan.json"), read(root / "config.json")
    for relative, digest in plan["source_fingerprints"].items():
        path = (root / "code" / relative).resolve()
        if not path.is_relative_to(root / "code") or sha(path) != digest:
            raise RuntimeError("Frozen scientific source mismatch")
    cache = Path(config["data"]["cache"]).resolve()
    if (config["training"]["epochs"] != 6000 or plan["fixed_training_updates"] != 6000
            or cache != Path(plan["data_cache"]).resolve() or sha(cache) != plan["data_sha256"]
            or read(root / "data/evaluation_metadata.json")["cache_sha256"] != plan["data_sha256"]
            or plan.get("MI_computed") is not False or plan.get("MI_cache_read") is not False
            or plan.get("command_changes", {}).get("rsmi_weight") != [2, 0]):
        raise ValueError("Unexpected paired no-MI science/data/endpoint contract")
    fingerprints = {name: sha(root / name) for name in
        ("config.json", "plan.json", "data/evaluation_metadata.json")}
    fingerprints["external_label_free_cache"] = plan["data_sha256"]
    baseline = Path(plan["baseline"])
    for relative, key in (("run/manifest.json", "baseline_manifest_sha256"),
                          ("report/report.json", "baseline_report_sha256")):
        if sha(baseline / relative) != plan[key]:
            raise RuntimeError("Frozen baseline public artifact changed")
        fingerprints["baseline/" + relative] = plan[key]
    if expected is not None and fingerprints != expected:
        raise RuntimeError("Scientific config, plan, metadata or data changed during handoff")
    return fingerprints


def prepare(root):
    root = Path(root).resolve()
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Require verified 225 host")
    directory = root / HANDOFF
    if directory.exists():
        raise FileExistsError("Existing handoff; inspect it instead of repeating")
    job = read(root / "job.json")
    validate_waiting(root, job)
    fingerprints = verify_science(root)
    pid = job["parent_pid"]
    identity = process_identity(pid)
    expected = ["-m", "scripts.run_ising3d_no_rsmi", "--root", str(root), "--run"]
    if identity is None or identity["argv"][1:] != expected or children(pid):
        raise RuntimeError("Waiter identity or children differ; do not signal it")
    opener = getattr(os, "pidfd_open", None)
    descriptor = opener(pid) if callable(opener) and hasattr(signal, "pidfd_send_signal") else None
    def deliver(sig):
        current = process_identity(pid)
        if current is None or current["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("PID identity changed; do not signal a reused PID")
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(pid, sig)
    deliver(signal.SIGSTOP)
    try:
        for _ in range(50):
            current = process_identity(pid)
            if current is None or current["state"] in ("T", "t"):
                break
            time.sleep(.02)
        current = process_identity(pid)
        if (current is None or current["state"] not in ("T", "t")
                or current["start_ticks"] != identity["start_ticks"] or children(pid)):
            raise RuntimeError("Waiter did not quiesce without children; preserve it")
        job = read(root / "job.json")
        validate_waiting(root, job)
        directory.mkdir()
        shutil.copy2(Path(__file__), directory / "continue_ising3d_no_rsmi_gpu.py")
        write(directory / "original_job.json", job)
        record = dict(schema_version=1, status="prepared",
            authority="User explicitly authorized using an idle GPU before any GPU work",
            old_parent=identity, old_device="cuda:2", device="auto_idle", gpu_uuid=None,
            allowed_GPU_UUIDs=GPU_UUIDS, selection="two >=10s apart idle checks on the same card, then pin all stages to that card",
            original_fingerprints=fingerprints, scientific_code_unchanged=True,
            physical_sampling_updates=0, MI_computations=0, trained_updates_before_handoff=0,
            adapter_sha256=sha(directory / "continue_ising3d_no_rsmi_gpu.py"),
            prepared_at=datetime.now(timezone.utc).isoformat())
        write(directory / "plan.json", record)
        deliver(signal.SIGTERM)
    finally:
        # On validation failure CONT restores the original waiter; after TERM
        # it allows the pending termination to run. No child/group is signalled.
        final = process_identity(pid)
        if final is not None and final["start_ticks"] == identity["start_ticks"]:
            deliver(signal.SIGCONT)
        if descriptor is not None:
            os.close(descriptor)
    for _ in range(50):
        state = process_identity(pid)
        if state is None or state["state"] == "Z":
            break
        if state["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("PID reused during handoff; never signal it")
        time.sleep(.1)
    else:
        raise RuntimeError("Old waiter has not exited; do not start another scheduler")
    record.update(status="handoff_ready", old_waiter_stopped=True)
    write(directory / "plan.json", record)
    job.update(status="handoff_ready", parent_pid=None, device="auto_idle",
        resource_handoff=dict(plan=str(directory / "plan.json"), from_device="cuda:2", to_device="auto_idle",
            old_parent_pid=pid, original_job=str(directory / "original_job.json"), scientific_plan_unmodified=True))
    write(root / "job.json", job)
    return record


def all_snapshots():
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Wrong execution host")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    rows = []
    for line in query.splitlines():
        selected = line.split(",")
        index, uuid = int(selected[0]), selected[1].strip()
        if GPU_UUIDS.get(index) != uuid:
            raise RuntimeError("Physical GPU index/UUID mismatch")
        active = [entry for entry in compute.splitlines() if uuid in entry]
        row = dict(checked_at=time.time(), gpu_index=index, gpu_uuid=uuid,
            free_mib=int(selected[2]), active=active, mem_available_gib=available / 1024**3,
            load1=os.getloadavg()[0], cpu_count=os.cpu_count(), io_full_avg10=io)
        row["ready"] = (not active and row["free_mib"] >= 12288 and row["mem_available_gib"] >= 36
            and row["load1"] <= 1.25 * row["cpu_count"] and io < 20)
        rows.append(row)
    if {row["gpu_index"] for row in rows} != set(GPU_UUIDS):
        raise RuntimeError("Expected the complete verified GPU inventory")
    return rows


def snapshot(index):
    return next(row for row in all_snapshots() if row["gpu_index"] == index)


def select_idle_gpu(root, job):
    counters = {index: 0 for index in GPU_UUIDS}
    first_ready = {}
    while True:
        rows = all_snapshots()
        for row in rows:
            index = row["gpu_index"]
            if row["ready"]:
                counters[index] += 1
                first_ready.setdefault(index, row["checked_at"])
            else:
                counters[index] = 0
                first_ready.pop(index, None)
        job.setdefault("resource_checks", []).extend(dict(row, required="initial idle GPU selection") for row in rows)
        job["resource_checks"] = job["resource_checks"][-360:]
        write(root / "job.json", job)
        eligible = [row for row in rows if counters[row["gpu_index"]] >= 2
                    and row["checked_at"] - first_ready[row["gpu_index"]] >= 10]
        if eligible:
            selected = max(eligible, key=lambda row: (row["free_mib"], row["gpu_index"]))
            selection = dict(device="cuda:" + str(selected["gpu_index"]), **selected,
                first_idle_check_at=first_ready[selected["gpu_index"]],
                policy="Pin this GPU for every stage; no further automatic migration")
            write(root / HANDOFF / "selection.json", selection)
            job.update(device=selection["device"])
            job["resource_handoff"]["to_device"] = selection["device"]
            write(root / "job.json", job)
            return selection
        time.sleep(10)


def running_old(record):
    old = record["old_parent"]
    actual = process_identity(old["pid"])
    return actual is not None and actual["start_ticks"] == old["start_ticks"] and actual["state"] != "Z"


def launch(root):
    root = Path(root).resolve()
    directory = root / HANDOFF
    record = read(directory / "plan.json")
    if record.get("status") != "handoff_ready" or running_old(record):
        raise RuntimeError("Old scheduler not safely stopped")
    verify_science(root, record["original_fingerprints"])
    adapter = directory / "continue_ising3d_no_rsmi_gpu.py"
    if sha(adapter) != record["adapter_sha256"]:
        raise RuntimeError("Execution adapter fingerprint changed")
    if (directory / "launch.json").exists() or (directory / "parent.log").exists():
        raise FileExistsError("Existing launch receipt/log; never duplicate launch")
    command = ["nice", "-n", "10", sys.executable, str(adapter), "--root", str(root), "--run"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with (directory / "parent.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = dict(parent_pid=child.pid, command=command, started_at=datetime.now(timezone.utc).isoformat(),
        device="auto_idle", gpu_uuid=None, data_and_scientific_code_unchanged=True)
    write(directory / "launch.json", receipt)
    return receipt


def stage_commands(original, root, device=DEVICE):
    stages = original.commands(root)
    if [stage[0] for stage in stages] != ["gpu_preflight", "train", "report", "comparison"]:
        raise ValueError("Unexpected immutable stage sequence")
    mapped = []
    for name, command, gpu in stages:
        command = list(command)
        if gpu:
            index = command.index("--device") + 1
            if command[index] != "cuda:2":
                raise ValueError("Unexpected original device argument")
            command[index] = device
        mapped.append((name, command, gpu))
    return mapped


def cpu_seconds():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def validate_stage_receipt(original, root, name):
    return original.validate_training_receipt(root, "run" if name == "train" else name,
        5 if name == "gpu_preflight" else 6000)


def run(root):
    root = Path(root).resolve()
    directory = root / HANDOFF
    record = read(directory / "plan.json")
    if running_old(record) or sha(Path(__file__)) != record["adapter_sha256"]:
        raise RuntimeError("Old scheduler active or adapter changed")
    descriptor = os.open(directory / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    verify_science(root, record["original_fingerprints"])
    job = read(root / "job.json")
    if job["status"] != "handoff_ready" or job["stages"] or job["actual_training_updates"] != 0:
        raise ValueError("Only the never-started GPU handoff may execute")
    job.update(parent_pid=os.getpid(), device="auto_idle", status="waiting_resources", current="gpu_preflight",
        handoff_started_at=datetime.now(timezone.utc).isoformat(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    write(root / "job.json", job)
    sys.path.insert(0, str(root / "code"))
    original = importlib.import_module("scripts.run_ising3d_no_rsmi")
    plan = read(root / "plan.json")
    prior_cpu, prior_gpu = job["CPU_core_seconds"], job["GPU_stage_seconds"]
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        selection = select_idle_gpu(root, job)
        device, gpu_index, gpu_uuid = selection["device"], selection["gpu_index"], selection["gpu_uuid"]
        for name, command, gpu in stage_commands(original, root, device):
            verify_science(root, record["original_fingerprints"])
            job.update(status="waiting_resources", current=name)
            consecutive = 0
            while True:
                check = snapshot(gpu_index)
                ready = check["ready"] if gpu else (check["mem_available_gib"] >= 36
                    and check["load1"] <= 1.25 * check["cpu_count"] and check["io_full_avg10"] < 20)
                job.setdefault("resource_checks", []).append(dict(check, required="gpu" if gpu else "host"))
                job["resource_checks"] = job["resource_checks"][-360:]
                consecutive = consecutive + 1 if ready else 0
                write(root / "job.json", job)
                if consecutive >= (2 if gpu else 1):
                    break
                time.sleep(10)
            if job["GPU_stage_seconds"] >= plan["max_GPU_stage_seconds"]:
                job.update(status="budget_hold", requires_user_action=True, error="GPU stage budget reached before next stage")
                break
            target = root / ("run" if name == "train" else name)
            if target.exists():
                raise FileExistsError("Stage output exists; do not restart")
            log_path = root / "logs" / (name + ".log")
            stage = dict(name=name, command=command, status="running", gpu_stage=gpu,
                device=device if gpu else "cpu", gpu_uuid=gpu_uuid if gpu else None,
                started_at=datetime.now(timezone.utc).isoformat(), log=str(log_path))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if gpu:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            tick = time.monotonic()
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running", current=name)
                write(root / "job.json", job)
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed",
                elapsed_seconds=time.monotonic() - tick, ended_at=datetime.now(timezone.utc).isoformat())
            if gpu:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            write(root / "job.json", job)
            if rc:
                raise RuntimeError(f"Stage {name} exited {rc}; preserve files, no automatic restart")
            if name in ("gpu_preflight", "train"):
                summary = validate_stage_receipt(original, root, name)
                job[name + "_timing"] = summary.get("timing")
                if name == "train":
                    job["actual_training_updates"] = 6000
            if name == "report":
                result = read(root / "report/report.json")
                if (result.get("status") != "complete" or not result.get("model_state_unchanged")
                        or result.get("MI_shuffle_diagnostic") is not None):
                    raise RuntimeError("Incomplete, mutating, or MI-enabled report")
            if sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) > plan["max_directory_bytes"]:
                raise RuntimeError("Directory budget exceeded; preserve all files")
        else:
            verify_science(root, record["original_fingerprints"])
            job.update(status="complete", workflow_complete=True, current=None,
                finished_at=datetime.now(timezone.utc).isoformat(), public_files=original.public_files(root)
                + [f"{HANDOFF}/{name}" for name in ("plan.json", "original_job.json", "launch.json", "selection.json")])
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        job.update(CPU_core_seconds=prior_cpu + cpu_seconds() - cpu_start,
            elapsed_seconds=time.monotonic() - started, handoff_elapsed_seconds=time.monotonic() - started,
            prior_GPU_stage_seconds=prior_gpu)
        write(root / "job.json", job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root) if args.prepare else launch(args.root) if args.launch else run(args.root)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
