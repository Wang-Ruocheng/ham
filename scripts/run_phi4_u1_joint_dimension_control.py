"""Conditional fresh trainable-C u=1 QCR comparison, dg=1 versus dg=2.

This exclusive parent reuses the original physical dataset and launches the
unchanged continuous QCR CLIs once each. Two idle pinned GPUs may run in
parallel. A failed child never causes a sibling kill, retry or migration.
"""
from __future__ import annotations

import argparse
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

from scripts.run_phi4_scalar_r_validation import now, save, sha


HOST = "thinklab-105-225"
BASE = Path("/mnt/nas-new/home/yange/ham")
SOURCE = BASE / "results/phi4_u1_rg_reference_b3_g1_seed91_vm225_20261028"
PRIOR_ROOTS = tuple(BASE / f"results/phi4_u1_fixed_C_dimensions_qseed{seed}_vm225_20261030"
                    for seed in (91, 92, 93))
DATA_SHA = "d554ac421c9a14ef0b94701ff34741bd39236938de17fbc14d60437f7228085b"
GPU_ASSIGNMENTS = {
    1: dict(index=2, uuid="GPU-713b07d0-118c-f939-c0a8-565390215e01"),
    2: dict(index=3, uuid="GPU-a167decd-ce49-6c8f-6bcf-136fde596683"),
}
CPU_SECONDS, GPU_SECONDS = 7200., 3600.
CHILD_SECONDS, PEAK_BYTES, DIRECTORY_BYTES = 1200., 8*1024**3, 512*1024**2
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PUBLIC_CHILDREN = ("manifest.json", "summary.json", "status.json", "history_progress.json", "evaluation_arrays.npz")


def prerequisite_gate(roots=PRIOR_ROOTS):
    """Verify all three completed diagnostics and their paired feature gate."""
    evidence, pins = {}, {}
    if len(roots) != 3:
        raise ValueError("Require exactly three predeclared Q-seed diagnostics")
    for q_seed, directory in zip((91, 92, 93), roots):
        root = Path(directory).resolve()
        inventory = json.loads((root / "public_files.json").read_text())
        pins[str(root / "public_files.json")] = sha(root / "public_files.json")
        for name in ("job.json", "plan.json", "training/summary.json", "report/analysis.json"):
            path = root / name
            if name not in inventory or sha(path) != inventory[name]:
                raise ValueError("Prerequisite public artifact SHA mismatch")
            pins[str(path)] = inventory[name]
        job = json.loads((root / "job.json").read_text())
        summary = json.loads((root / "training/summary.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        if (job.get("status") != "complete" or job.get("workflow_complete") is not True
                or job.get("returncode") != 0 or job.get("error") is not None
                or job.get("requires_user_action") is not False or not job.get("stages")
                or any(s.get("returncode") != 0 for s in job["stages"])
                or summary.get("status") != "complete" or summary.get("workflow_complete") is not True
                or summary.get("returncode") != 0 or plan.get("q_seed") != q_seed):
            raise ValueError("All three fixed-C Q-seed diagnostics must complete successfully")
        ratios = []
        for seed in (91, 92, 93):
            losses = []
            for dim in (1, 2):
                rows = summary["feature_metrics"][f"dg{dim}_seed{seed}_selected"]["validation"]
                if [r["path"] for r in rows] != ["0_to_1", "1_to_2", "0_to_2"]:
                    raise ValueError("Prerequisite feature paths differ")
                loss = sum(weight*row["prediction"]["chain_mean_rmse"]**2
                           for weight, row in zip((5., 2.5, 2.5), rows))
                if not math.isfinite(loss) or loss < 0:
                    raise ValueError("Invalid prerequisite common-feature validation loss")
                losses.append(loss)
            if losses[0] <= 0:
                raise ValueError("Cannot establish improvement over zero comparator error")
            ratios.append(losses[1]/losses[0])
        median, improved = sorted(ratios)[1], sum(x < 1 for x in ratios)
        evidence[str(q_seed)] = dict(ratios=ratios, median_ratio=median, improved_pairs=improved,
            passed=median <= .8 and improved >= 2)
    if not all(value["passed"] for value in evidence.values()):
        raise ValueError("Fresh trainable-C follow-up gate is not met")
    return evidence, pins


def host_resources():
    return dict(at=now(), mem_available_bytes=int(re.search(r"MemAvailable:\s+(\d+)",
        Path("/proc/meminfo").read_text())[1])*1024, load1=os.getloadavg()[0],
        cpus=os.cpu_count(), io_full_avg10=float(re.search(r"full avg10=([\d.]+)",
        Path("/proc/pressure/io").read_text())[1]))


def host_ready(row):
    return row["mem_available_bytes"] >= 36*1024**3 and row["load1"] <= 30 and row["io_full_avg10"] < 20


def gpu_resources():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)
    active = {}
    for line in processes.splitlines():
        if line.strip():
            uuid, pid = [x.strip() for x in line.split(",")]
            active.setdefault(uuid.lower(), []).append(int(pid))
    result = {}
    for line in raw.splitlines():
        index, uuid, free = [x.strip() for x in line.split(",")]
        result[int(index)] = dict(index=int(index), uuid=uuid, free_bytes=int(free)*1024**2,
                                  compute_pids=active.get(uuid.lower(), []))
    return result


def gpu_ready(rows):
    for assignment in GPU_ASSIGNMENTS.values():
        row = rows.get(assignment["index"])
        if row is None or row["uuid"].lower() != assignment["uuid"].lower():
            raise RuntimeError("Pinned physical GPU UUID mismatch")
        if row["compute_pids"] or row["free_bytes"] < 12*1024**3:
            return False
    return True


def idle_pair_receipt(first, second, elapsed):
    if elapsed < 10 or not host_ready(first["host"]) or not host_ready(second["host"]):
        return False
    return gpu_ready(first["gpus"]) and gpu_ready(second["gpus"])


def training_command(root, dim):
    if dim not in GPU_ASSIGNMENTS:
        raise ValueError("Only the two predeclared dimensions are allowed")
    assignment = GPU_ASSIGNMENTS[dim]
    return [sys.executable, "-u", "-m", "experiments.phi4_qcr" if dim == 1 else "experiments.phi4_ru_qcr",
        "--data", str(SOURCE / "data"), "--output", str(Path(root) / f"dg{dim}"),
        "--device", "cuda:0", "--expected-gpu-uuid", assignment["uuid"],
        "--steps", "3000", "--seed", "91", "--system-batch", "8", "--sample-batch", "64",
        "--threads", "1", "--information-pairs", "128", "--log-every", "100",
        "--checkpoint-every", "1000", "--max-training-seconds", str(CHILD_SECONDS),
        "--max-cuda-bytes", str(PEAK_BYTES), "--q-initialization", "train_mean_bias"]


def child_environment(dim):
    result = os.environ.copy()
    result.update({key: "1" for key in THREAD_ENV})
    result.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU_ASSIGNMENTS[dim]["uuid"])
    return result


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def unchanged(pins):
    if any(sha(path) != digest for path, digest in pins.items()):
        raise RuntimeError("Pinned code, data manifest or prerequisite artifact changed")


def live_cpu(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
        return (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
    except (FileNotFoundError, ProcessLookupError):
        return 0.


def poll_child_status(stage, path):
    """Treat an in-flight or absent JSON write as unknown until a later poll.

    Child return codes remain authoritative. Keep the last successfully read
    snapshot while recording that the newest attempted read is pending.
    """
    try:
        value = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        stage["child_status_read_pending"] = True
        return False
    if not isinstance(value, dict):
        stage["child_status_read_pending"] = True
        return False
    stage.update(child_status=value, child_status_read_pending=False)
    return True


def validate_child(root, dim):
    folder = Path(root) / f"dg{dim}"
    status = json.loads((folder / "status.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or summary.get("actual_steps") != 3000 or summary.get("theory_dim") != dim
            or manifest.get("data_manifest_sha256") != DATA_SHA
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("q_initialization") != "train_mean_bias"
            or manifest.get("options", {}).get("seed") != 91
            or summary.get("peak_cuda_bytes", 0) > PEAK_BYTES
            or status.get("summary_sha256") != sha(folder / "summary.json")
            or status.get("evaluation_arrays_sha256") != sha(folder / "evaluation_arrays.npz")
            or summary.get("data_manifest_sha256") != DATA_SHA
            or str(summary.get("device", {}).get("uuid", "")).lower().removeprefix("gpu-")
               != GPU_ASSIGNMENTS[dim]["uuid"].lower().removeprefix("gpu-")
            or set(summary.get("optimizer_steps", {})) != {"Q", "C", "R"}
            or any(row.get("min") != 3000 or row.get("max") != 3000 or row.get("parameters", 0) <= 0
                   for row in summary["optimizer_steps"].values())):
        raise RuntimeError(f"dg{dim} completed child contract differs")
    return dict(status=status, actual_steps=summary["actual_steps"],
                initial_module_sha256=manifest.get("training_start_module_sha256"),
                final_module_sha256=summary.get("final_module_sha256"))


def run(output):
    if socket.gethostname() != HOST or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Require pinned225 host and one inherited nice10 layer")
    root, code = Path(output).resolve(), Path(__file__).resolve().parents[1]
    if (root == code or root in code.parents or root == SOURCE or root in SOURCE.parents
            or SOURCE in root.parents or any(root == p or root in p.parents or p in root.parents for p in PRIOR_ROOTS)):
        raise ValueError("Exclusive output must be separate from immutable sources")
    gate, input_pins = prerequisite_gate()
    data_manifest = SOURCE / "data/manifest.json"
    if sha(data_manifest) != DATA_SHA:
        raise ValueError("Original u=1 physical data manifest SHA mismatch")
    input_pins[str(data_manifest)] = DATA_SHA
    code_pins = {str(p): sha(p) for folder in ("experiments", "recurrent_rg", "scripts")
                 for p in (code / folder).rglob("*.py")}
    pins = code_pins | input_pins
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    report_command = [sys.executable, "-u", "-m", "scripts.report_phi4_u1_joint_dimension_control",
                      "--root", str(root), "--output", str(root / "report")]
    commands = {dim: training_command(root, dim) for dim in (1, 2)}
    save(root / "plan.json", dict(schema_version=5, created_at=now(), code_root=str(code), output=str(root),
        source=str(SOURCE), data_manifest_sha256=DATA_SHA, input_sha256=input_pins, source_sha256=code_pins,
        prerequisite_gate=gate, GPU_assignments=GPU_ASSIGNMENTS, child_commands=commands, report_command=report_command,
        design=dict(fresh_QCR=True, trainable_random_C=True, theory_dims=[1, 2], seed=91, steps=3000,
            optimizer="independent fresh Adam, lr0.001", system_batch=8, sample_batch=64, information_pairs=128,
            Q_initialization="TRAIN mean decoder bias; random encoder and decoder directions",
            lattice_sizes=[36, 12, 4], block_size=3, stride=3, closure_objective="all_start_anchor",
            closure_scale_weights=None, closure_weights=[5., 2.5, 2.5], new_physical_sampling=False,
            continuous_C_recipe="-2 smoothed full-shell information proxy +2 variance floor +uniform amplitude MSE",
            old_source_checkpoint_loaded=False, labels_used_for_training=False, exact_historical_reproduction=False),
        resources=dict(CPU_seconds=CPU_SECONDS, GPU_card_seconds=GPU_SECONDS, child_max_training_seconds=CHILD_SECONDS,
            GPU_peak_bytes=PEAK_BYTES, directory_bytes=DIRECTORY_BYTES, CPU_threads_per_child=1, nice=10,
            min_memory_GiB=36, max_load1=30, max_IO_full_avg10=20, idle_check_spacing_seconds=10,
            simultaneous_children=2, no_compute_processes_on_assigned_GPU=True),
        qualification="固定u=1数据上的新随机可训练C维度对照；连续场历史配方；保留全部采样警告"))
    started, parent_cpu = time.monotonic(), time.process_time()
    child_cpu_base = resource.getrusage(resource.RUSAGE_CHILDREN)
    live, logs = {}, {}
    gpu_preflight, gpu_elapsed = 0., {}
    job = dict(status="prepared", workflow_complete=False, current=None, returncode=None, error=None,
        requires_user_action=False, parent_pid=os.getpid(), started_at=now(), stages=[],
        plan_sha256=sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.)

    def account():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = time.process_time()-parent_cpu + child.ru_utime-child_cpu_base.ru_utime+child.ru_stime-child_cpu_base.ru_stime
        cpu += sum(live_cpu(process.pid) for process in live.values() if process.poll() is None)
        gpu = gpu_preflight+sum(gpu_elapsed.values())
        gpu += sum(time.monotonic()-stage["monotonic_start"] for stage in job["stages"]
                   if stage["name"] in ("dg1", "dg2") and stage.get("status") == "running")
        job.update(cpu_seconds=cpu, gpu_seconds=gpu, directory_bytes=directory_bytes(root))
        return cpu, gpu

    def checkpoint():
        account()
        save(root / "job.json", job)

    checkpoint()
    try:
        job.update(status="waiting_resources", current="two_GPU_idle_gate")
        while True:
            first = dict(host=host_resources(), gpus=gpu_resources())
            job["resources"] = first
            checkpoint()
            if not host_ready(first["host"]) or not gpu_ready(first["gpus"]):
                time.sleep(30)
                continue
            check_start = time.monotonic()
            time.sleep(10)
            second = dict(host=host_resources(), gpus=gpu_resources())
            elapsed = time.monotonic()-check_start
            gpu_preflight += 2*elapsed
            job["GPU_idle_checks"] = dict(first=first, second=second, spacing_seconds=elapsed,
                conservative_preflight_card_seconds=gpu_preflight)
            checkpoint()
            if idle_pair_receipt(first, second, elapsed):
                break
        unchanged(pins)
        cpu, gpu = account()
        if cpu+2*CHILD_SECONDS+300 > CPU_SECONDS or gpu+2*CHILD_SECONDS+300 > GPU_SECONDS:
            job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                       error="Conservative remaining-stage projection exceeds registered budget")
            checkpoint()
            return 3
        # Both resources passed together; create exactly one process per dimension.
        for dim in (1, 2):
            name = f"dg{dim}"
            log = (root / "logs" / f"{name}.log").open("xb")
            logs[name] = log
            process = subprocess.Popen(commands[dim], cwd=code, env=child_environment(dim), stdout=log, stderr=subprocess.STDOUT)
            live[name] = process
            job["stages"].append(dict(name=name, command=commands[dim], status="running", pid=process.pid,
                returncode=None, started_at=now(), monotonic_start=time.monotonic(), GPU=GPU_ASSIGNMENTS[dim]))
        job.update(status="running", current=["dg1", "dg2"])
        checkpoint()
        while any(stage["status"] == "running" for stage in job["stages"]):
            for stage in job["stages"]:
                if stage["status"] != "running":
                    continue
                process = live[stage["name"]]
                rc = process.poll()
                if rc is None:
                    detail = root / stage["name"] / "status.json"
                    poll_child_status(stage, detail)
                    continue
                gpu_elapsed[stage["name"]] = time.monotonic()-stage["monotonic_start"]
                stage.update(status="complete" if rc == 0 else "failed", returncode=rc, ended_at=now(),
                             conservative_GPU_card_seconds=gpu_elapsed[stage["name"]])
                logs[stage["name"]].close()
            job["current"] = [s["name"] for s in job["stages"] if s["status"] == "running"]
            if any(s["returncode"] not in (None, 0) for s in job["stages"]):
                job.update(status="waiting_running_sibling_after_failure", requires_user_action=True)
            checkpoint()
            if job["current"]:
                time.sleep(5)
        failures = [stage for stage in job["stages"] if stage["returncode"] != 0]
        if failures:
            job.update(status="failed", current=None, returncode=failures[0]["returncode"],
                requires_user_action=True, error="One or more fresh QCR children failed; all siblings preserved and awaited",
                ended_at=now())
            checkpoint()
            return job["returncode"]
        for dim in (1, 2):
            job[f"dg{dim}_receipt"] = validate_child(root, dim)
        unchanged(pins)
        cpu, gpu = account()
        if cpu+120 > CPU_SECONDS or gpu > GPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
            job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                       error="Actual conservative resources leave no registered report budget")
            checkpoint()
            return 3
        report_stage = dict(name="report", command=report_command, status="running", returncode=None, started_at=now())
        job["stages"].append(report_stage)
        job.update(status="running", current="report")
        with (root / "logs/report.log").open("xb") as log:
            process = subprocess.Popen(report_command, cwd=code, env={**os.environ, **{k: "1" for k in THREAD_ENV},
                "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT)
            report_stage["pid"] = process.pid
            checkpoint()
            rc = process.wait()
        report_stage.update(status="complete" if rc == 0 else "failed", returncode=rc, ended_at=now())
        if rc:
            raise RuntimeError(f"Fresh dimension report exited {rc}; completed QCR children retained")
        unchanged(pins)
        cpu, gpu = account()
        if cpu > CPU_SECONDS or gpu > GPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
            job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                       error="Completed artifacts exceed registered conservative budget")
            checkpoint()
            return 3
        public = {}
        for dim in (1, 2):
            for filename in PUBLIC_CHILDREN:
                path = root / f"dg{dim}" / filename
                public[f"dg{dim}/{filename}"] = sha(path)
        report_inventory = json.loads((root / "report/public_files.json").read_text())
        for name, expected in report_inventory.items():
            path = (root / "report" / name).resolve()
            if (root / "report").resolve() not in path.parents or path.suffix not in (".json", ".png", ".npz") or sha(path) != expected:
                raise RuntimeError("Invalid public report artifact")
            public[f"report/{name}"] = expected
        job.update(status="complete", workflow_complete=True, current=None, returncode=0, error=None,
                   requires_user_action=False, ended_at=now())
        checkpoint()
        public.update({name: sha(root / name) for name in ("job.json", "plan.json")})
        save(root / "public_files.json", public)
        return 0
    except BaseException as exc:
        # Preserve and await any already launched sibling, even if Popen or a
        # read-only parent check failed after the first child was dispatched.
        pending = [p for p in live.values() if p.poll() is None]
        if pending:
            job.update(status="waiting_running_sibling_after_parent_error", requires_user_action=True,
                       error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            for process in pending:
                process.wait()
        job.update(status="failed", workflow_complete=False, returncode=1, error=f"{type(exc).__name__}: {exc}",
            requires_user_action=True, live_child_pids=[p.pid for p in live.values() if p.poll() is None], ended_at=now())
        checkpoint()
        raise
    finally:
        for log in logs.values():
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
