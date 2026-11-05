"""Fresh default QCR training on new Houdayer--PT data, with no time/RSS cap.

The user explicitly removed the exploratory time/memory caps and requested
training. Work remains finite and fixed; resource sharing is still respected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time

from scripts.run_spin_glass2d import GPU_UUID, TEMPERATURES, gpu_snapshot, sha, write_json


CONFIG_PATH = "configs/spin_glass2d_houdayer_overlap_g1_seed91.json"
REFERENCE = "results/spin_glass2d_overlap_g1_seed91_vm225_20261019_monitor/v1/physical_data/manifest.json"


def cpu_seconds():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def freeze_code(source, target):
    target.mkdir()
    fingerprints = {}
    for package in ("recurrent_rg", "scripts", "experiments"):
        for original in sorted((source / package).rglob("*.py")):
            relative = original.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            fingerprints[str(relative)] = sha(destination)
    for relative in ("AGENTS.md", CONFIG_PATH):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        fingerprints[relative] = sha(destination)
    write_json(target / "fingerprints.json", fingerprints)
    return fingerprints


def verify_code(code, fingerprints):
    if any(sha(code / path) != expected for path, expected in fingerprints.items()):
        raise RuntimeError("Frozen source changed; preserve outputs, no repair/restart")


def build_plan(config, reference, smoke=False):
    from recurrent_rg.config import RunConfig
    from recurrent_rg.data.spin_glass_pipeline import GenerationOptions
    from recurrent_rg.data.spin_glass_houdayer_pipeline import generation_plan

    RunConfig.from_dict(config).validate()
    old = json.loads(Path(reference).read_text())
    if old.get("status") != "complete" or old["options"]["shape"] != [36, 36]:
        raise ValueError("Require completed historical L36 manifest for bond identity")
    options = GenerationOptions(shape=(12, 12) if smoke else (36, 36),
        temperatures=(.8, 2., 3.5) if smoke else tuple(TEMPERATURES),
        disorders=3 if smoke else 9, replicas=4, samples=8 if smoke else 512,
        burn_in=8 if smoke else 8192, spacing=1 if smoke else 16,
        exchange_interval=1, seed=20261019)
    physical = generation_plan(options, cluster_interval=1, trace_interval=2 if smoke else 16, initial="hot")
    if config["model"]["dimension"] != 2 or config["model"]["block_factor"] != 3:
        raise ValueError("Require default 2D block-three geometry")
    return dict(schema_version=1, experiment="2D EA new Houdayer+PT data; fresh unsupervised QCR g1",
        execution_smoke=smoke, scientific_result=not smoke, physical_preflight=physical,
        physical=dict(shape=list(options.shape), temperatures=list(options.temperatures),
            disorders=options.disorders, replicas=4, samples=options.samples, burn_in=options.burn_in,
            spacing=options.spacing, exchange_interval=1, cluster_interval=1,
            trace_interval=2 if smoke else 16, seed=20261019, initial="hot"),
        historical_manifest_sha256=sha(reference),
        expected_bonds=[] if smoke else [dict(disorder_id=row["disorder_id"], bond_sha256=row["bond_sha256"])
            for row in old["shards"]],
        training=dict(epochs=config["training"]["epochs"], fresh_initialization=True,
            optimizer="fresh Adam", seed=91, learning_rate=.001, labels_used=False,
            saved_samples_per_ensemble=options.samples, training_sample_batch=config["training"]["sample_batch"],
            formal_evaluation_samples=config["evaluation"]["samples"],
            formal_evaluation_subset="First64postburnsavedconfigurations per J,T in formal run, same evaluation protocol as old run; not all512",
            scale_weights="uniform in both one-step and microscopic-start multistep branches",
            sampler_model_input="Only fixed overlap pair 0/1; the other two replicas improve sampling, not ensemble count"),
        changes_from_old_sampling=dict(sampler="lazy local + random-pair Houdayer + PT", replicas=[2, 4],
            saved_samples=[64, 512], burn_in=[2048, 8192], spacing=[16, 16],
            initial="all random, not hot/cold independent convergence groups",
            strict_single_variable_ablation=False),
        administrative_limits=dict(cpu_seconds=None, gpu_seconds=None, training_seconds=None,
            memory_bytes=None, output_bytes=None,
            authority="User: 不用管内存和时间限制，直接训练吧; fixed work still not automatically extended"),
        resource_policy=dict(threads=1, nice=10, gpu="cuda:2", gpu_uuid=GPU_UUID,
            gpu_requires_two_idle_checks_10s_apart=True, never_interfere_other_jobs=True),
        equilibrium_certified=False, thermal_nu=None, root_search=False)


def stages(root, device, plan):
    physical = plan["physical"]
    return [
        ("physical_sampling", [sys.executable, "-m", "scripts.generate_spin_glass_houdayer",
            "--output", str(root / "physical_data"), "--dimension", "2", "--length", str(physical["shape"][0]),
            "--temperatures", *map(str, physical["temperatures"]), "--disorders", str(physical["disorders"]),
            "--replicas", "4", "--samples", str(physical["samples"]), "--burn-in", str(physical["burn_in"]),
            "--spacing", str(physical["spacing"]), "--exchange-interval", "1", "--cluster-interval", "1",
            "--trace-interval", str(physical["trace_interval"]), "--seed", "20261019", "--initial", "hot"], False),
        ("prepare_overlap", [sys.executable, "-m", "scripts.prepare_spin_glass_houdayer_qcr",
            "--source", str(root / "physical_data"), "--output", str(root / "data")], False),
        ("train", [sys.executable, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json"),
            "--output", str(root / "run"), "--device", device, "--skip-final-report"], True),
        ("report", [sys.executable, "-m", "scripts.report_spin_glass2d", "--root", str(root),
            "--device", device, "--draws", "2" if plan["execution_smoke"] else "5"], True),
    ]


def wait_for_gpu(root, job):
    consecutive = 0
    while True:
        snapshot = gpu_snapshot()
        job.setdefault("resource_checks", []).append(snapshot)
        consecutive = consecutive + 1 if snapshot["ready"] else 0
        write_json(root / "job.json", job)
        if consecutive == 2:
            return
        time.sleep(10)


def public_files(root):
    public = ["job.json", "plan.json", "config.json", "reference_manifest.json",
        "data/evaluation_metadata.json", "physical_data/manifest.json", "physical_data/plan.json",
        "physical_data/status.json", "physical_data/sampling.json", "run/manifest.json", "run/summary.json"]
    public.extend(str(path.relative_to(root)) for path in sorted((root / "report").iterdir())
        if path.is_file() and (path.suffix in (".json", ".png") or path.name == "encodings.npz"))
    return public


def run(root, reference, *, device="cuda:2", smoke=False):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    source = Path(__file__).resolve().parents[1]
    if device not in ("cpu", "cuda:2") or (device == "cpu" and not smoke):
        raise ValueError("Formal training is pinned to cuda:2; CPU only for execution smoke")
    if device.startswith("cuda") and socket.gethostname() != "thinklab-105-225":
        raise ValueError("Formal GPU host must be thinklab-105-225")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    fingerprints = freeze_code(source, root / "code")
    shutil.copy2(reference, root / "reference_manifest.json")
    config = json.loads((root / "code" / CONFIG_PATH).read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, system_batch=4, sample_batch=8, max_rg_steps=1,
            log_every=1, checkpoint_every=2)
        config["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    write_json(root / "config.json", config)
    plan = build_plan(config, reference, smoke)
    plan["source_fingerprints"] = fingerprints
    write_json(root / "plan.json", plan)
    job = dict(status="running", workflow_complete=False, parent_pid=os.getpid(),
        host=socket.gethostname(), device=device, actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
        execution_smoke=smoke, scientific_result=not smoke, stages=[], current=None,
        GPU_stage_seconds=0., CPU_core_seconds=0., error=None, requires_user_action=False,
        administrative_time_memory_limits_disabled=True)
    write_json(root / "job.json", job)
    cpu_start, started = cpu_seconds(), time.monotonic()
    try:
        for name, command, gpu in stages(root, device, plan):
            verify_code(root / "code", fingerprints)
            if gpu and device.startswith("cuda"):
                job.update(status="waiting_resources", current=name)
                wait_for_gpu(root, job)
            stage = dict(name=name, command=command, status="running", started_at=time.time(),
                gpu_stage=gpu and device.startswith("cuda"), log=str(root / "logs" / f"{name}.log"))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if stage["gpu_stage"]:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            stage_started = time.monotonic()
            with Path(stage["log"]).open("x") as log:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running", current=name)
                write_json(root / "job.json", job)
                returncode = child.wait()
            stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                elapsed_seconds=time.monotonic() - stage_started)
            job["CPU_core_seconds"] = cpu_seconds() - cpu_start
            if stage["gpu_stage"]:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            write_json(root / "job.json", job)
            if returncode:
                raise RuntimeError(f"Stage {name} exited {returncode}; preserve files, no restart")
            if name == "physical_sampling" and not smoke:
                manifest = json.loads((root / "physical_data/manifest.json").read_text())
                actual = {row["disorder_id"]: row["bond_sha256"] for row in manifest["shards"]}
                expected = {row["disorder_id"]: row["bond_sha256"] for row in plan["expected_bonds"]}
                if actual != expected:
                    raise RuntimeError("Historical J fingerprints differ; no training started")
                job["historical_bond_fingerprints_match"] = True
            if name == "train":
                summary = json.loads((root / "run/summary.json").read_text())
                if summary["actual_epochs"] != config["training"]["epochs"]:
                    raise RuntimeError("Training ended before fixed updates; no automatic continuation")
        verify_code(root / "code", fingerprints)
        report = json.loads((root / "report/report.json").read_text())
        if report.get("status") != "complete" or not report.get("model_state_unchanged"):
            raise RuntimeError("Formal frozen-model report is incomplete or changed model")
        job.update(status="complete", workflow_complete=True, current=None,
            public_files=public_files(root), thermal_nu=None, equilibrium_certified=False,
            finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        job.update(CPU_core_seconds=cpu_seconds() - cpu_start, elapsed_seconds=time.monotonic() - started,
            directory_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
            measured_max_child_rss_native=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            rss_native_units="KiB on Linux; bytes on macOS")
        write_json(root / "job.json", job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=Path(REFERENCE))
    parser.add_argument("--device", default="cuda:2", choices=("cuda:2", "cpu"))
    parser.add_argument("--execution-smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.reference, device=args.device, smoke=args.execution_smoke)
    print(json.dumps({key: result.get(key) for key in ("status", "workflow_complete", "current", "error")}, ensure_ascii=False))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
