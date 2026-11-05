"""Fresh 2D EA overlap learning with frozen code and the default Q/C/R recipe.

The execution-smoke is a two-update implementation check, never physical evidence.
All time budgets are admission/post-stage soft checks, not hard execution limits.
"""
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
CONFIG_PATH = "configs/spin_glass2d_overlap_g1_seed91.json"
TEMPERATURES = [.5 * 7. ** (index / 19.) for index in range(20)]
PHYSICAL_SEED = 20261019
MAX_GPU_STAGE_SECONDS = 2500
TRAINING_ONLY_SECONDS = 1800
MAX_CPU_CORE_SECONDS = 7200
MAX_OUTPUT_BYTES = 1024**3
MAX_SPIN_VISITS = 1_500_000_000


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_code(source, output):
    """Copy source only into a new immutable experiment-specific checkout."""
    output.mkdir()
    fingerprints = {}
    for package in ("recurrent_rg", "experiments", "scripts"):
        for origin in sorted((source / package).rglob("*.py")):
            relative = origin.relative_to(source)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            fingerprints[str(relative)] = sha(target)
    for relative in ("AGENTS.md", CONFIG_PATH):
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
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines()
                if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    row = dict(checked_at=time.time(), gpu_uuid=GPU_UUID, free_mib=int(selected[2]),
        active=active, mem_available_gib=available / 1024**3,
        load1=os.getloadavg()[0], cpu_count=os.cpu_count(), io_full_avg10=io)
    row["ready"] = (not active and row["free_mib"] >= 12288 and available >= 36 * 1024**3
        and row["load1"] <= 1.25 * row["cpu_count"] and io < 20)
    return row


class ResourceHold(RuntimeError):
    """No GPU child was launched; user can decide how to proceed."""


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
    raise ResourceHold("Resource gate timed out without launching GPU work; no automatic continuation")


def build_plan(root, config, fingerprints, smoke=False):
    """Preflight physical work and all configured geometry before allocation."""
    from recurrent_rg.config import RunConfig
    from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, generation_plan

    RunConfig.from_dict(config).validate()
    physical = dict(shape=[12, 12] if smoke else [36, 36], disorders=3 if smoke else 9,
        temperatures=[1., 3.] if smoke else TEMPERATURES, real_replicas=2,
        samples=8 if smoke else 64, burn_in=8 if smoke else 2048, spacing=1 if smoke else 16,
        exchange_interval=1, seed=PHYSICAL_SEED, distribution="bimodal",
        bond_scale=1., positive_probability=.5, field=0.,
        test_every=3, test_offset=1)
    options = GenerationOptions(shape=tuple(physical["shape"]),
        temperatures=tuple(physical["temperatures"]), disorders=physical["disorders"],
        replicas=physical["real_replicas"], samples=physical["samples"],
        burn_in=physical["burn_in"], spacing=physical["spacing"], seed=PHYSICAL_SEED,
        max_spin_proposals=MAX_SPIN_VISITS, max_output_bytes=MAX_OUTPUT_BYTES,
        max_memory_bytes=512 * 1024**2)
    projection = generation_plan(options)
    minimum_side = physical["shape"][0]
    for _ in range(config["training"]["max_rg_steps"]):
        if minimum_side % 3:
            raise ValueError("Every requested 2D RG scale must be divisible by block factor 3")
        minimum_side //= 3
    if minimum_side < max(4, config["model"]["width"]):
        raise ValueError("Final scale is smaller than the minimum side 4 / Q local-motif width")
    return dict(schema_version=1, experiment="2D EA overlap QCR g_dim=1",
        execution_smoke=smoke, scientific_result=not smoke,
        physical=physical, physical_preflight=projection,
        training=dict(config=config, optimizer="fresh Adam", labels_used=False,
            scale_weights="uniform within both closure branches"),
        budget=dict(max_gpu_stage_seconds=MAX_GPU_STAGE_SECONDS,
            training_only_seconds=TRAINING_ONLY_SECONDS,
            CPU_core_seconds=MAX_CPU_CORE_SECONDS, output_bytes=MAX_OUTPUT_BYTES,
            max_single_spin_visits=MAX_SPIN_VISITS, threads=1,
            enforcement="Admission and post-stage soft checks, not hard execution deadlines; no automatic kill or continuation",
            gpu_accounting="Wall seconds of CUDA stages including setup, training/evaluation and report; resource waits excluded",
            training_time_limit="Trainer checks its 1800-second update-loop limit; a partial run is budget_hold, never resumed automatically",
            nice="inherited; launch parent once at nice10"),
        source_fingerprints=fingerprints,
        qualification="Finite positive-T exploratory 2D zero-field bimodal EA overlap learning; T_c=0 expected in the thermodynamic limit; no root/exponent search; no equilibrium certification")


def build_stages(root, device, plan, smoke=False):
    py, physical = sys.executable, plan["physical"]
    return [
        ("physical_sampling", [py, "-m", "scripts.generate_spin_glass", "--output", str(root / "physical_data"),
            "--dimension", "2", "--length", str(physical["shape"][0]), "--temperatures",
            *map(str, physical["temperatures"]), "--disorders", str(physical["disorders"]),
            "--replicas", "2", "--samples", str(physical["samples"]),
            "--burn-in", str(physical["burn_in"]), "--spacing", str(physical["spacing"]),
            "--exchange-interval", str(physical["exchange_interval"]),
            "--distribution", physical["distribution"], "--bond-scale", str(physical["bond_scale"]),
            "--positive-probability", str(physical["positive_probability"]), "--field", str(physical["field"]),
            "--test-every", str(physical["test_every"]), "--test-offset", str(physical["test_offset"]),
            "--seed", str(PHYSICAL_SEED), "--max-spin-proposals", str(MAX_SPIN_VISITS),
            "--max-output-gib", "1", "--max-memory-gib", "0.5"], False),
        ("prepare_overlap", [py, "-m", "scripts.prepare_spin_glass_qcr", "--source", str(root / "physical_data"),
            "--output", str(root / "data")], False),
        ("train", [py, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json"),
            "--output", str(root / "run"), "--device", device, "--skip-final-report",
            "--max-training-seconds", str(TRAINING_ONLY_SECONDS)], True),
        ("report", [py, "-m", "scripts.report_spin_glass2d", "--root", str(root), "--device", device,
            "--draws", "2" if smoke else "5"], True),
    ]


def public_files(root):
    """Explicit safe public artifact types; never physical shards or model PTs."""
    public = ["job.json", "plan.json", "config.json", "data/evaluation_metadata.json",
              "physical_data/manifest.json", "physical_data/plan.json", "physical_data/status.json",
              "physical_data/sampling.json", "run/manifest.json", "run/summary.json"]
    public += [str(path.relative_to(root)) for path in sorted((root / "report").iterdir())
               if path.is_file() and (path.suffix in (".json", ".png") or path.name == "encodings.npz")]
    return public


def cpu_core_seconds():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    parent = resource.getrusage(resource.RUSAGE_SELF)
    return children.ru_utime + children.ru_stime + parent.ru_utime + parent.ru_stime


def run(root, device="cuda:2", smoke=False):
    root = Path(root).resolve()
    source = Path(__file__).resolve().parents[1]
    if device not in ("cpu", "cuda:2") or (device == "cpu" and not smoke):
        raise ValueError("Formal run uses cuda:2; CPU is for execution-smoke only")
    if device.startswith("cuda") and socket.gethostname() != "thinklab-105-225":
        raise ValueError("CUDA run is pinned to thinklab-105-225 before any physical sampling")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    code = root / "code"
    fingerprints = freeze_code(source, code)
    config = json.loads((code / CONFIG_PATH).read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, system_batch=4, sample_batch=8,
                                  max_rg_steps=1, log_every=1, checkpoint_every=2)
        config["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    write_json(root / "config.json", config)
    job = dict(status="running", workflow_complete=False, parent_pid=os.getpid(),
        host=socket.gethostname(), device=device, stages=[], current=None,
        execution_smoke=smoke, scientific_result=not smoke,
        GPU_stage_seconds=0., CPU_core_seconds=0., error=None, requires_user_action=False)
    write_json(root / "job.json", job)
    cpu_start = cpu_core_seconds()
    try:
        plan = build_plan(root, config, fingerprints, smoke)
        write_json(root / "plan.json", plan)
        for name, command, gpu in build_stages(root, device, plan, smoke):
            verify_code(code, fingerprints)
            if job["CPU_core_seconds"] >= MAX_CPU_CORE_SECONDS or job["GPU_stage_seconds"] >= MAX_GPU_STAGE_SECONDS:
                job.update(status="budget_hold", requires_user_action=True,
                           error="Actual execution budget reached before the next stage; no automatic continuation")
                break
            if gpu and device.startswith("cuda"):
                job.update(status="waiting_resources", current=name)
                resource_gate(root, job)
            row = dict(name=name, command=command, status="running", started_at=time.time(),
                       log=str(root / "logs" / f"{name}.log"), gpu_stage=gpu and device.startswith("cuda"))
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
            job["CPU_core_seconds"] = cpu_core_seconds() - cpu_start
            if row["gpu_stage"]:
                job["GPU_stage_seconds"] += row["elapsed_seconds"]
            job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            write_json(root / "job.json", job)
            if returncode:
                raise RuntimeError(f"{name} exited {returncode}; preserve files and inspect its log")
            if (job["directory_bytes"] > MAX_OUTPUT_BYTES or job["GPU_stage_seconds"] > MAX_GPU_STAGE_SECONDS
                    or job["CPU_core_seconds"] > MAX_CPU_CORE_SECONDS):
                job.update(status="budget_hold", requires_user_action=True,
                           error="Actual execution/output budget reached; no automatic continuation")
                break
            if name == "train":
                summary = json.loads((root / "run/summary.json").read_text())
                if summary["actual_epochs"] != config["training"]["epochs"]:
                    job.update(status="budget_hold", requires_user_action=True,
                        error="Training stopped before configured updates; no automatic continuation")
                    break
        else:
            verify_code(code, fingerprints)
            report = json.loads((root / "report/report.json").read_text())
            if report.get("status") != "complete":
                raise RuntimeError("Report not complete")
            job.update(status="complete", workflow_complete=True, current=None,
                       public_files=public_files(root), thermal_nu=None)
    except ResourceHold as exc:
        job.update(status="resource_hold", error=str(exc), requires_user_action=True)
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        write_json(root / "job.json", job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:2"], default="cuda:2")
    parser.add_argument("--execution-smoke", action="store_true")
    args = parser.parse_args()
    job = run(args.root, args.device, args.execution_smoke)
    print(json.dumps(job, ensure_ascii=False))
    if job["status"] == "failed":
        raise SystemExit(1)
    if job["status"] == "budget_hold":
        raise SystemExit(2)
    if job["status"] == "resource_hold":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
