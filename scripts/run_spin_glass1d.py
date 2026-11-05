"""Bounded fresh 1D EA overlap learning, frozen code and default Q/C/R recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time


GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
TEMPERATURES = [.25, .35, .45, .6, .8, 1., 1.3, 1.7, 2.2, 3.]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def freeze_code(source, output):
    output.mkdir()
    fingerprints = {}
    for package in ("recurrent_rg", "experiments", "scripts"):
        for origin in sorted((source / package).rglob("*.py")):
            relative = origin.relative_to(source)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            fingerprints[str(relative)] = sha(target)
    for relative in ("AGENTS.md", "configs/spin_glass1d_overlap_g1_seed91.json"):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        fingerprints[relative] = sha(target)
    write_json(output / "fingerprints.json", fingerprints)
    return fingerprints


def verify_code(code, fingerprints):
    if any(sha(code / relative) != expected for relative, expected in fingerprints.items()):
        raise RuntimeError("Frozen code fingerprint changed; do not repair or rerun automatically")


def gpu_snapshot():
    if socket.gethostname() != "thinklab-105-225":
        raise RuntimeError("Formal CUDA run is pinned to thinklab-105-225")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(",") for line in query.strip().splitlines()]
    selected = next(row for row in rows if int(row[0]) == 2)
    if selected[1].strip() != GPU_UUID:
        raise RuntimeError("GPU2 UUID mismatch")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    active = [line for line in compute.strip().splitlines() if GPU_UUID in line]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    pressure = Path("/proc/pressure/io").read_text().splitlines()
    full = next(line for line in pressure if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    row = dict(checked_at=time.time(), gpu_uuid=GPU_UUID, free_mib=int(selected[2]),
        active=active, mem_available_gib=available / 1024**3,
        load1=os.getloadavg()[0], cpu_count=os.cpu_count(), io_full_avg10=io)
    row["ready"] = (not active and row["free_mib"] >= 12288 and available >= 36 * 1024**3
        and row["load1"] <= 1.25 * row["cpu_count"] and io < 20)
    return row


def resource_gate(root, job):
    consecutive = 0
    for _ in range(360):
        check = gpu_snapshot()
        job.setdefault("resource_checks", []).append(check)
        consecutive = consecutive + 1 if check["ready"] else 0
        write_json(root / "job.json", job)
        if consecutive == 2:
            return
        time.sleep(10)
    raise RuntimeError("Resource gate timed out without launching GPU work")


def run(root, device="cuda:2", smoke=False):
    root = Path(root).resolve()
    source = Path(__file__).resolve().parents[1]
    if device not in ("cpu", "cuda:2") or (device == "cpu" and not smoke):
        raise ValueError("Formal run uses cuda:2; CPU is for execution-smoke only")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    code = root / "code"
    fingerprints = freeze_code(source, code)
    config = json.loads((code / "configs/spin_glass1d_overlap_g1_seed91.json").read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, system_batch=4, sample_batch=8,
                                  max_rg_steps=2, log_every=1, checkpoint_every=2)
        config["evaluation"].update(max_rg_steps=2, samples=8, systems=4)
    write_json(root / "config.json", config)
    plan = dict(schema_version=1, experiment="1D EA overlap QCR g_dim=1",
        execution_smoke=smoke, scientific_result=not smoke,
        physical=dict(shape=[16 if smoke else 128], disorders=3 if smoke else 9,
            temperatures=[.7, 2.] if smoke else TEMPERATURES, real_replicas=2,
            samples=8 if smoke else 64, burn_in=8 if smoke else 1024, spacing=1 if smoke else 8),
        training=dict(config=config, optimizer="fresh Adam", labels_used=False,
            scale_weights="uniform within both closure branches"),
        budget=dict(max_gpu_stage_seconds=900, training_only_seconds=600,
            CPU_core_seconds=3600, output_bytes=1024**3, threads=1,
            enforcement="Admission and post-stage checks, not hard execution deadlines; no automatic kill or continuation",
            gpu_accounting="Wall seconds of CUDA stages including setup, training/evaluation and report; resource waits excluded",
            training_time_limit="Trainer checks its 600-second update-loop limit; a partial run is budget_hold, never resumed automatically",
            nice="inherited; launch parent once at nice10"),
        source_fingerprints=fingerprints,
        qualification="Finite positive-T exploratory 1D overlap learning; T_c=0; no root/exponent search; no equilibrium certification")
    write_json(root / "plan.json", plan)
    job = dict(status="running", workflow_complete=False, parent_pid=os.getpid(),
        host=socket.gethostname(), device=device, stages=[], current=None,
        GPU_stage_seconds=0., CPU_core_seconds=0., error=None, requires_user_action=False)
    write_json(root / "job.json", job)
    py = sys.executable
    physical = plan["physical"]
    stages = [
        ("physical_sampling", [py, "-m", "scripts.generate_spin_glass", "--output", str(root / "physical_data"),
            "--dimension", "1", "--length", str(physical["shape"][0]), "--temperatures",
            *map(str, physical["temperatures"]), "--disorders", str(physical["disorders"]),
            "--replicas", "2", "--samples", str(physical["samples"]),
            "--burn-in", str(physical["burn_in"]), "--spacing", str(physical["spacing"]), "--seed", "20261019"], False),
        ("prepare_overlap", [py, "-m", "scripts.prepare_spin_glass_qcr", "--source", str(root / "physical_data"),
            "--output", str(root / "data")], False),
        ("train", [py, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json"),
            "--output", str(root / "run"), "--device", device, "--skip-final-report",
            "--max-training-seconds", "600"], True),
        ("report", [py, "-m", "scripts.report_spin_glass1d", "--root", str(root), "--device", device,
            "--draws", "2" if smoke else "5"], True),
    ]
    try:
        for name, command, gpu in stages:
            verify_code(code, fingerprints)
            if job["CPU_core_seconds"] >= plan["budget"]["CPU_core_seconds"] or job["GPU_stage_seconds"] >= 900:
                job.update(status="budget_hold", requires_user_action=True)
                break
            if gpu and device.startswith("cuda"):
                job.update(status="waiting_resources", current=name)
                resource_gate(root, job)
            row = dict(name=name, command=command, status="running", started_at=time.time(),
                       log=str(root / "logs" / f"{name}.log"), gpu_stage=gpu)
            job["stages"].append(row)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MPLCONFIGDIR=str(root / "matplotlib_cache"))
            started = time.monotonic()
            with Path(row["log"]).open("w") as log:
                process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, env=env)
                row["pid"] = process.pid
                job.update(status="running", current=name)
                write_json(root / "job.json", job)
                returncode = process.wait()
            row.update(returncode=returncode, elapsed_seconds=time.monotonic() - started,
                       status="complete" if returncode == 0 else "failed")
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            job["CPU_core_seconds"] = usage.ru_utime + usage.ru_stime
            if gpu and device.startswith("cuda"):
                job["GPU_stage_seconds"] += row["elapsed_seconds"]
            job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            write_json(root / "job.json", job)
            if returncode:
                raise RuntimeError(f"{name} exited {returncode}; preserve files and inspect its log")
            if job["directory_bytes"] > plan["budget"]["output_bytes"] or job["GPU_stage_seconds"] > 900:
                job.update(status="budget_hold", requires_user_action=True,
                           error="Actual execution/output budget reached; no automatic continuation")
                break
            if name == "train":
                summary = json.loads((root / "run/summary.json").read_text())
                if summary["actual_epochs"] != config["training"]["epochs"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="Training stopped at its time cap; no automatic continuation")
                    break
        else:
            verify_code(code, fingerprints)
            report = json.loads((root / "report/report.json").read_text())
            if report.get("status") != "complete":
                raise RuntimeError("Report not complete")
            public = ["job.json", "plan.json", "config.json", "data/evaluation_metadata.json",
                      "physical_data/manifest.json", "physical_data/plan.json", "physical_data/status.json",
                      "physical_data/sampling.json", "run/manifest.json", "run/summary.json"]
            public += [str(path.relative_to(root)) for path in (root / "report").iterdir()
                       if path.suffix in (".json", ".npz", ".png")]
            job.update(status="complete", workflow_complete=True, current=None,
                       public_files=public, thermal_nu=None)
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        write_json(root / "job.json", job)
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda:2"], default="cuda:2")
    p.add_argument("--execution-smoke", action="store_true")
    args = p.parse_args()
    job = run(args.root, args.device, args.execution_smoke)
    print(json.dumps(job, ensure_ascii=False))
    if job["status"] == "failed":
        raise SystemExit(1)
    if job["status"] == "budget_hold":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
