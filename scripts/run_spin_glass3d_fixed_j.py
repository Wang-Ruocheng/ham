"""Fresh single-J default-RSMI experiment, reusing frozen physical data only.

Training/model code is copied from the completed baseline's immutable snapshot.
Only this runner, the fixed-J adapter, and explicit report split handling differ.
No resampling, warm start, automatic retry, migration, or historical overwrite.
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

from scripts.run_spin_glass2d import sha, write_json


HOST = "thinklab-105-225"
DEVICE = "cuda:2"
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
OVERRIDES = ("run_spin_glass3d_fixed_j.py", "prepare_spin_glass3d_fixed_j.py",
             "report_spin_glass3d.py", "report_spin_glass1d.py")


def read(path):
    return json.loads(Path(path).read_text())


def cpu_seconds():
    return sum(getattr(resource.getrusage(who), name) for who in
               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for name in ("ru_utime", "ru_stime"))


def validate_baseline(baseline):
    job = read(baseline / "job.json")
    if job.get("status") != "complete" or job.get("workflow_complete") is not True:
        raise ValueError("Require a complete immutable 3D SG baseline")
    config = read(baseline / "config.json")
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    validate_recipe(build_parser().parse_args(["--config", "/unused", "--output", "/unused/run"]), cfg)
    if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim,
            cfg.training.epochs, cfg.training.seed, cfg.training.system_batch, cfg.training.sample_batch,
            cfg.training.threads, cfg.training.learning_rate) != (3, 2, 2, 1, 6000, 91, 16, 64, 1, .001):
        raise ValueError("Unexpected original default science contract")
    if cfg.loss.closure_scale_weights is not None and len(set(cfg.loss.closure_scale_weights)) != 1:
        raise ValueError("Require uniform closure scale weights")
    source = read(baseline / "data/evaluation_metadata.json")
    if sha(config["data"]["cache"]) != source["cache_sha256"]:
        raise ValueError("Frozen baseline cache fingerprint changed")
    for relative, digest in read(baseline / "code/fingerprints.json").items():
        if sha(baseline / "code" / relative) != digest:
            raise RuntimeError("Historical frozen source fingerprint mismatch")
    return config, source


def snapshot():
    if socket.gethostname() != HOST:
        raise RuntimeError("Wrong execution host")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    row = next(line.split(",") for line in query.splitlines() if int(line.split(",")[0]) == 2)
    if row[1].strip() != GPU_UUID:
        raise RuntimeError("Physical GPU2 identity mismatch; never migrate")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    active = [line for line in compute.splitlines() if line.split(",")[0].strip() == GPU_UUID]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), gpu_index=2, gpu_uuid=GPU_UUID,
        free_mib=int(row[2]), active=active, mem_available_gib=available / 1024**3,
        cpu_count=os.cpu_count(), load1=os.getloadavg()[0], io_full_avg10=io,
        disk_free_gib=shutil.disk_usage(Path(__file__).resolve().parent).free / 1024**3)
    result["host_ready"] = (available >= 36 * 1024**3 and result["load1"] <= 1.25 * result["cpu_count"]
        and io < 20 and result["disk_free_gib"] >= 20)
    result["gpu_ready"] = result["host_ready"] and not active and result["free_mib"] >= 12288
    return result


def admit(root, job, gpu):
    consecutive = 0
    while True:
        row = snapshot()
        ready = row["gpu_ready" if gpu else "host_ready"]
        job.setdefault("resource_checks", []).append(dict(row, required="gpu" if gpu else "host"))
        job["resource_checks"] = job["resource_checks"][-360:]
        consecutive = consecutive + 1 if ready else 0
        write_json(root / "job.json", job)
        if consecutive >= (2 if gpu else 1):
            return
        time.sleep(10)


def stage_commands(root, baseline):
    common = [sys.executable, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json")]
    return [
        ("prepare_fixed_J", [sys.executable, "-m", "scripts.prepare_spin_glass3d_fixed_j",
            "--source-root", str(baseline), "--output", str(root / "data"), "--disorder-id", "0"], False),
        ("full_shell_MI", [sys.executable, "-m", "scripts.prepare_qcr_mi_cache", "--config", str(root / "config.json"),
            "--output", str(root / "mi_receipt.json"), "--cache-dir", str(root / "mi_cache")], False),
        ("gpu_preflight", common + ["--output", str(root / "gpu_preflight"), "--device", DEVICE,
            "--epochs", "5", "--skip-final-report", "--mi-cache-dir", str(root / "mi_cache"),
            "--benchmark-warmup", "1"], True),
        ("train", common + ["--output", str(root / "run"), "--device", DEVICE,
            "--skip-final-report", "--mi-cache-dir", str(root / "mi_cache"), "--benchmark-warmup", "5"], True),
        ("report", [sys.executable, "-m", "scripts.report_spin_glass3d", "--root", str(root),
            "--device", DEVICE, "--draws", "5"], True),
    ]


def verify_code(root, fingerprints):
    if any(sha(root / "code" / name) != digest for name, digest in fingerprints.items()):
        raise RuntimeError("Frozen code changed; no automatic repair or restart")


def run(root, baseline):
    raw_root = Path(root)
    if raw_root.exists() or raw_root.is_symlink():
        raise FileExistsError(raw_root)
    root, baseline = raw_root.resolve(), Path(baseline).resolve()
    if socket.gethostname() != HOST:
        raise ValueError("Formal training requires verified host225")
    config, metadata = validate_baseline(baseline)
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    config["data"]["physical_model"] += "；固定J0、整温度索引留出；其余模型与损失沿用原冻结版"
    config["evaluation"]["systems"] = 4
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    shutil.copytree(baseline / "code", root / "code")
    for name in OVERRIDES:
        shutil.copy2(Path(__file__).with_name(name), root / "code/scripts" / name)
    fingerprints = {str(path.relative_to(root / "code")): sha(path)
        for path in sorted((root / "code").rglob("*"))
        if path.is_file() and path.suffix in (".py", ".json", ".md") and path.name != "fingerprints.json"}
    write_json(root / "code/fingerprints.json", fingerprints)
    bond = {row["bond_sha256"] for split in ("train", "test") for row in metadata[split] if row["disorder_id"] == 0}
    if len(bond) != 1:
        raise ValueError("Fixed J0 missing or ambiguous")
    plan = dict(schema_version=1, scientific_result=True, baseline=str(baseline),
        experiment="三维EA固定J0、带完整RSMI、g1、b2的温度留出QCR探索",
        fixed_disorder_id=0, fixed_bond_sha256=next(iter(bond)), physical_sampling_updates=0,
        source_cache_sha256=metadata["cache_sha256"], source_config_sha256=sha(baseline / "config.json"),
        source_report_sha256=sha(baseline / "report/report.json"),
        source_fingerprints=fingerprints, core_training_source_unchanged=True,
        science_overrides=list(OVERRIDES), fixed_training_updates=6000, optimizer_seed=91,
        train_temperature_indices=[i for i in range(20) if i % 5 != 2], test_temperature_indices=[2, 7, 12, 17],
        MI_null_original_temperature_indices=[0, 10, 19],
        split="whole_temperature_within_fixed_disorder", training_samples_per_ensemble=512,
        train_ensembles=16, test_ensembles=4, batch_ensembles=16, batch_configurations=64,
        QCR_initialization="fresh random, no checkpoint loading, fresh Adam .001; preflight weights not reused",
        MI="-2 exact empirical full-shell MI; scale0 buffer1, joint152-spins, no projection/InfoNCE",
        C="random trainable Bernoulli b2 +2 per-ensemble variance floor ratio1 +1 uniform-color NLL",
        Q="same cube256 affine reconstruction10 + subset consistency20/5",
        R="one-step5 + microscopic-start multistep5; uniform target-scale weights",
        gradient_routing="Q representation only; C MI/variance/color only; R closure only",
        scales=[[16] * 3, [8] * 3, [4] * 3], labels_used_in_training=False,
        execution=dict(host=HOST, device=DEVICE, gpu_uuid=GPU_UUID, threads=1, nice=10,
            two_idle_checks_10_seconds_apart=True, never_interfere_other_jobs=True),
        qualification="Single fixedJ, single size and optimizer seed; same PT ladder temperatures are not independent physical chains. Original thermalization unverified. Descriptive roots/nu only, thermal_nu null.")
    write_json(root / "config.json", config)
    write_json(root / "plan.json", plan)
    protected = {str(path): sha(path) for path in (baseline / "config.json", baseline / "job.json",
        baseline / "report/report.json", baseline / "data/evaluation_metadata.json", Path(metadata["source_dataset"]) / "manifest.json")}
    protected[str(Path(read(baseline / "config.json")["data"]["cache"]))] = metadata["cache_sha256"]
    job = dict(status="running", workflow_complete=False, current=None, stages=[], parent_pid=os.getpid(),
        host=HOST, device=DEVICE, gpu_uuid=GPU_UUID, started_at=datetime.now(timezone.utc).isoformat(),
        error=None, requires_user_action=False, GPU_stage_seconds=0., CPU_core_seconds=0.,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0), actual_training_updates=0,
        physical_sampling_updates=0, fixed_disorder_id=0, protected_source_sha256=protected)
    write_json(root / "job.json", job)
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        for name, command, gpu in stage_commands(root, baseline):
            verify_code(root, fingerprints)
            job.update(status="waiting_resources", current=name)
            admit(root, job, gpu)
            stage = dict(name=name, status="running", command=command, gpu_stage=gpu,
                started_at=datetime.now(timezone.utc).isoformat(), log=str(root / "logs" / (name + ".log")))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            env["PYTHONPATH"] = str(root / "code")
            if gpu:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            began = time.monotonic()
            with Path(stage["log"]).open("x") as stream:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=stream, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running")
                write_json(root / "job.json", job)
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed",
                elapsed_seconds=time.monotonic() - began, ended_at=datetime.now(timezone.utc).isoformat())
            if gpu:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            job["CPU_core_seconds"] = cpu_seconds() - cpu_start
            write_json(root / "job.json", job)
            if rc:
                raise RuntimeError(f"{name} exited {rc}; preserve partial evidence, no retry")
            if name in ("gpu_preflight", "train"):
                directory = "gpu_preflight" if name == "gpu_preflight" else "run"
                summary = read(root / directory / "summary.json")
                expected = 5 if name == "gpu_preflight" else 6000
                if summary["actual_epochs"] != expected:
                    raise RuntimeError("Wrong actual update endpoint")
                if name == "train":
                    manifest = read(root / "run/manifest.json")
                    if manifest.get("rsmi_weight") != 2 or manifest.get("estimator") != "histogram":
                        raise RuntimeError("Wrong formal MI contract")
                    job["actual_training_updates"] = 6000
        verify_code(root, fingerprints)
        if any(sha(path) != digest for path, digest in protected.items()):
            raise RuntimeError("Protected historical source changed")
        report = read(root / "report/report.json")
        if report["status"] != "complete" or not report["model_state_unchanged"]:
            raise RuntimeError("Report did not finish read-only")
        public = ["job.json", "plan.json", "config.json", "mi_receipt.json", "data/evaluation_metadata.json",
            "data/status.json", "run/manifest.json", "run/summary.json", "run/history_progress.json",
            "gpu_preflight/manifest.json", "gpu_preflight/summary.json"]
        public += [str(path.relative_to(root)) for path in sorted((root / "report").iterdir())
                   if path.is_file() and path.suffix in (".json", ".png", ".npz")]
        job.update(status="complete", workflow_complete=True, current=None,
            public_files=public, finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start)
        write_json(root / "job.json", job)
    return job


def launch(root, baseline):
    raw_root = Path(root)
    if raw_root.exists() or raw_root.is_symlink():
        raise FileExistsError(raw_root)
    root, baseline = raw_root.resolve(), Path(baseline).resolve()
    validate_baseline(baseline)
    logs = root.parent.parent / "logs"
    logs.mkdir(exist_ok=True)
    logfile, receipt = logs / (root.name + ".log"), logs / (root.name + ".launch.json")
    if logfile.exists() or receipt.exists():
        raise FileExistsError("Existing launch receipt; inspect instead of launching twice")
    with (logs / (root.name + ".launch.lock")).open("x") as stream:
        stream.write(str(os.getpid()))
    command = ["nice", "-n", str(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0))), sys.executable,
        str(Path(__file__).resolve()), "--root", str(root), "--baseline", str(baseline), "--run"]
    with logfile.open("x") as stream:
        child = subprocess.Popen(command, cwd=baseline / "code", env=dict(os.environ,
            PYTHONPATH=str(baseline / "code"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    record = dict(parent_pid=child.pid, command=command, root=str(root), baseline=str(baseline),
        log=str(logfile), device=DEVICE, gpu_uuid=GPU_UUID, host=socket.gethostname(),
        started_at=datetime.now(timezone.utc).isoformat(), launcher_sha256=sha(Path(__file__)),
        no_automatic_restart=True)
    write_json(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = launch(args.root, args.baseline) if args.launch else run(args.root, args.baseline)
    print(json.dumps({key: result.get(key) for key in
        ("status", "workflow_complete", "current", "parent_pid", "device", "root", "error")}, ensure_ascii=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
