"""Independent raw-T supervised-Q ablation on the immutable multi-J cache.

Only the data-label/split boundary and reporting differ from the baseline.
The original Q/C/R trainer, model, losses and initialization are copied intact.
No physical sampling, warm start, kernel search, retry, migration or overwrite.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
WEIGHT = 10.0
UPDATES = 6000
SOURCE_DATA_SHA = "f4c666196f03191467957680005728bc4e8205d442b840869d9d40c27976316f"
OVERRIDES = (
    "recurrent_rg/data/ensembles.py",
    "scripts/run_spin_glass3d_temperature_supervised.py",
    "scripts/prepare_spin_glass3d_temperature_supervised.py",
    "scripts/report_spin_glass3d_temperature_supervised.py",
    "scripts/report_spin_glass3d.py",
    "scripts/report_spin_glass1d.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def cpu_seconds():
    return sum(getattr(resource.getrusage(who), name) for who in
               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for name in ("ru_utime", "ru_stime"))


def verify_code(root, fingerprints):
    if any(sha(root / "code" / name) != digest for name, digest in fingerprints.items()):
        raise RuntimeError("Frozen code changed; no automatic repair or restart")


def validate_baseline(baseline):
    baseline = Path(baseline)
    job, config = read(baseline / "job.json"), read(baseline / "config.json")
    if job.get("status") != "complete" or job.get("workflow_complete") is not True:
        raise ValueError("Require the complete immutable multi-J baseline")
    manifest, summary = read(baseline / "run/manifest.json"), read(baseline / "run/summary.json")
    if manifest.get("labels_used_in_training") is not False or summary.get("actual_epochs") != UPDATES:
        raise ValueError("Require completed unsupervised 6000-update baseline")
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    validate_recipe(build_parser().parse_args(["--config", "/unused", "--output", "/unused/run"]), cfg)
    if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim,
            cfg.training.epochs, cfg.training.seed, cfg.training.system_batch, cfg.training.sample_batch,
            cfg.training.threads, cfg.training.learning_rate, cfg.training.max_rg_steps) != (
                3, 2, 2, 1, UPDATES, 91, 16, 64, 1, .001, 2):
        raise ValueError("Unexpected original scientific contract")
    weights = cfg.loss.closure_scale_weights
    if weights is not None and len(set(weights)) != 1:
        raise ValueError("Require uniform closure target-scale weights")
    metadata = read(baseline / "data/evaluation_metadata.json")
    if metadata.get("labels_used_in_training") is not False or metadata.get("cache_sha256") != SOURCE_DATA_SHA:
        raise ValueError("Unexpected label-free physical source")
    if sha(config["data"]["cache"]) != SOURCE_DATA_SHA:
        raise ValueError("Baseline cache fingerprint changed")
    if [sorted({r["disorder_id"] for r in metadata[s]}) for s in ("train", "test")] != [
            [0, 2, 3, 5, 6, 8], [1, 4, 7]]:
        raise ValueError("Whole-disorder source split changed")
    for relative, digest in read(baseline / "code/fingerprints.json").items():
        if sha(baseline / "code" / relative) != digest:
            raise RuntimeError("Historical frozen source fingerprint mismatch")
    receipt = read(baseline / "mi_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("data_sha256") != SOURCE_DATA_SHA:
        raise ValueError("Require complete source data-only MI statistics")
    return deepcopy(config), metadata


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
    result = dict(checked_at=now(), gpu_index=2, gpu_uuid=GPU_UUID, free_mib=int(row[2]), active=active,
        mem_available_gib=available / 1024**3, cpu_count=os.cpu_count(), load1=os.getloadavg()[0],
        io_full_avg10=io, disk_free_gib=shutil.disk_usage(Path(__file__).resolve().parent).free / 1024**3)
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
    common = [sys.executable, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json"),
        "--q-coupling-weight", str(WEIGHT), "--skip-final-report", "--mi-cache-dir", str(root / "mi_cache")]
    return [
        ("prepare_temperature_labels", [sys.executable, "-m", "scripts.prepare_spin_glass3d_temperature_supervised",
            "--source-root", str(baseline), "--output", str(root / "data")], False),
        ("reuse_exact_MI_statistics", [sys.executable, "-m", "scripts.run_spin_glass3d_temperature_supervised",
            "--root", str(root), "--baseline", str(baseline), "--prepare-mi"], False),
        ("gpu_preflight", common + ["--output", str(root / "gpu_preflight"), "--device", DEVICE,
            "--epochs", "5", "--benchmark-warmup", "1"], True),
        ("train", common + ["--output", str(root / "run"), "--device", DEVICE,
            "--benchmark-warmup", "5"], True),
        ("report", [sys.executable, "-m", "scripts.report_spin_glass3d_temperature_supervised", "--root", str(root),
            "--device", DEVICE, "--draws", "5", "--baseline", str(baseline)], True),
    ]


def reuse_mi(root, baseline):
    """Rebind data-only statistics only after exact configuration equality.

    The labels change the cache-file fingerprint, but do not change a single
    input spin, group, order, region or MI count. Source caches are read-only.
    """
    root, baseline = Path(root), Path(baseline)
    receipt_path = root / "mi_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
    from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics, _cache_identity, _load_cache
    from recurrent_rg.rsmi import region_offsets
    from recurrent_rg.runner import prepared_data
    torch.set_num_threads(1)
    old_cfg, new_cfg = RunConfig.load(baseline / "config.json"), RunConfig.load(root / "config.json")
    old_sha, new_sha = sha(old_cfg.data.cache), sha(new_cfg.data.cache)
    if old_sha != SOURCE_DATA_SHA:
        raise ValueError("MI source fingerprint changed")
    old_data, new_data = prepared_data(old_cfg), prepared_data(new_cfg)
    original = read(baseline / "mi_receipt.json")
    _, environment = region_offsets(3, block=2, buffer=1)
    receipt = dict(status="running", data_sha256=new_sha, source_data_sha256=old_sha,
        source=str(baseline), labels_change_only=True, model_initialized=False, optimizer_updates=0,
        sampling_draws=0, new_MI_histograms_computed=False, device="cpu", splits={})
    started = time.monotonic()
    for split, old, new in zip(("train", "test"), old_data, new_data):
        if (not torch.equal(old.configurations, new.configurations)
                or not torch.equal(old.group_ids, new.group_ids)):
            raise ValueError("MI reuse requires identical input configurations, order and groups")
        kwargs = dict(block=2)
        old_key = lattice_statistics_cache_key(old_cfg, old_sha, split, old.configurations, environment,
            binary_lattice_mi_statistics.__name__, kwargs)
        new_key = lattice_statistics_cache_key(new_cfg, new_sha, split, new.configurations, environment,
            binary_lattice_mi_statistics.__name__, kwargs)
        if {k:v for k,v in old_key.items() if k != "data_sha256"} != {
                k:v for k,v in new_key.items() if k != "data_sha256"}:
            raise ValueError("MI region, dtype, geometry or pooling changed")
        identity, key = _cache_identity(old_key)
        source = Path(original["splits"][split]["cache"]["path"])
        if original["splits"][split]["cache"]["key"] != key:
            raise ValueError("MI source receipt identity mismatch")
        source_sha = sha(source)
        statistics = _load_cache(source, identity, key)
        _, record = load_or_build_mi_statistics(root / "mi_cache", new_key, lambda: statistics)
        if sha(source) != source_sha:
            raise RuntimeError("Historical MI source changed")
        receipt["splits"][split] = dict(cache=record, source_path=str(source), source_sha256=source_sha,
            source_key=key, configurations_identical=True, groups_identical=True,
            diagnostics=statistics.diagnostics, source_unchanged=True)
    if sha(old_cfg.data.cache) != old_sha or sha(new_cfg.data.cache) != new_sha:
        raise RuntimeError("Input cache changed during MI reuse")
    receipt.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_json(receipt_path, receipt)
    return receipt


def public_files(root):
    public = ["job.json", "plan.json", "config.json", "mi_receipt.json", "data/evaluation_metadata.json",
        "data/status.json", "run/manifest.json", "run/summary.json", "run/history_progress.json",
        "gpu_preflight/manifest.json", "gpu_preflight/summary.json"]
    return public + [str(p.relative_to(root)) for p in sorted((root / "report").iterdir())
        if p.is_file() and not p.is_symlink() and p.suffix in (".json", ".png", ".npz")]


def run(root, baseline):
    raw_root = Path(root)
    if raw_root.exists() or raw_root.is_symlink():
        raise FileExistsError(raw_root)
    root, baseline = raw_root.resolve(), Path(baseline).resolve()
    if socket.gethostname() != HOST:
        raise ValueError("Formal experiment requires verified host225")
    if root == baseline or baseline in root.parents or root in baseline.parents:
        raise ValueError("New root must be separate from historical baseline")
    config, metadata = validate_baseline(baseline)
    config["data"].update(cache=str(root / "data/ensembles.pt"), coupling_names=["T"],
        physical_model="三维零场 EA ±J 重叠场；原多 J 数据、整 J 留出；原始 T 仅监督微观 Q，J 不进入模型或损失")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    shutil.copytree(baseline / "code", root / "code")
    deployment = Path(__file__).resolve().parents[1]
    for name in OVERRIDES:
        target = root / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(deployment / name, target)
    fingerprints = {str(p.relative_to(root / "code")): sha(p) for p in sorted((root / "code").rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json", ".md") and p.name != "fingerprints.json"}
    write_json(root / "code/fingerprints.json", fingerprints)
    trainer = "experiments/rsmi_separate_qcr.py"
    if sha(root / "code" / trainer) != sha(baseline / "code" / trainer):
        raise RuntimeError("The paired trainer must remain byte-identical")
    plan = dict(schema_version=1, scientific_result=True, experiment="三维多 J 温度监督 Q 的显式配对消融",
        baseline=str(baseline), source_cache_sha256=SOURCE_DATA_SHA, source_fingerprints=fingerprints,
        science_overrides=list(OVERRIDES), core_trainer_byte_identical=True,
        physical_sampling_updates=0, train_disorders=[0, 2, 3, 5, 6, 8], test_disorders=[1, 4, 7],
        train_ensembles=len(metadata["train"]), test_ensembles=len(metadata["test"]),
        saved_samples_per_ensemble=512, temperature_count_per_disorder=20,
        split="whole quenched disorder; same temperature ladder on unseen TEST J is intentional",
        labels_used_in_training=True, q_supervision=dict(weight=WEIGHT, scale=0,
            target="raw T, full microscopic batch; no beta/standardization/Tc/exponent targets",
            fitting_split="TRAIN only", gradient="Q only; coarse Q has no temperature target; C/R receive no label loss"),
        fixed_training_updates=UPDATES, optimizer_seed=91, fresh_initialization=True,
        QCR_initialization="fresh random Q/C/R, fresh Adam .001; no source weights or preflight weights reused",
        Q="original affine cube256 reconstruction10 + subset consistency20/5 + 10*micro temperature MSE",
        C="original random Bernoulli b2; -2 full-shell empirical MI +2 variance-floor ratio1 +1 uniform-color NLL",
        R="original one-step5 + microscopic-start rollout5; uniform target-scale weights; R gradients only",
        MI="same exact original statistics; source inputs verified identical before rebinding to label-cache SHA",
        scales=[[16] * 3, [8] * 3, [4] * 3], batch_ensembles=16, batch_configurations=64,
        gpu_preflight=dict(updates=5, scientific_result=False, weights_reused=False),
        resources=dict(host=HOST, device=DEVICE, gpu_uuid=GPU_UUID, threads=1, intended_nice=10,
            two_idle_checks_10_seconds_apart=True, never_interfere_other_jobs=True),
        qualification="Temperature-supervised representation ablation, not unsupervised learning. Original sampling remains unverified; single size/optimizer seed/few J. All roots descriptive, thermal_nu null.")
    write_json(root / "config.json", config)
    write_json(root / "plan.json", plan)
    protected = {str(p): sha(p) for p in (baseline / "config.json", baseline / "job.json",
        baseline / "report/report.json", baseline / "data/evaluation_metadata.json", baseline / "run/checkpoint_last.pt")}
    protected[str(Path(read(baseline / "config.json")["data"]["cache"]))] = SOURCE_DATA_SHA
    job = dict(status="running", workflow_complete=False, parent_pid=os.getpid(), current=None, stages=[],
        host=HOST, device=DEVICE, gpu_uuid=GPU_UUID, started_at=now(), error=None, requires_user_action=False,
        GPU_stage_seconds=0., CPU_core_seconds=0., actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
        actual_training_updates=0, physical_sampling_updates=0, labels_used_in_training=True,
        protected_source_sha256=protected)
    write_json(root / "job.json", job)
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        for name, command, gpu in stage_commands(root, baseline):
            verify_code(root, fingerprints)
            job.update(status="waiting_resources", current=name)
            admit(root, job, gpu)
            stage = dict(name=name, status="running", command=command, gpu_stage=gpu,
                started_at=now(), log=str(root / "logs" / (name + ".log")))
            job["stages"].append(stage)
            env = dict(os.environ, PYTHONPATH=str(root / "code"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if gpu:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            began = time.monotonic()
            with Path(stage["log"]).open("x") as stream:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running")
                write_json(root / "job.json", job)
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed",
                elapsed_seconds=time.monotonic() - began, ended_at=now())
            if gpu:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            job["CPU_core_seconds"] = cpu_seconds() - cpu_start
            write_json(root / "job.json", job)
            if rc:
                raise RuntimeError(f"{name} exited {rc}; preserve partial evidence, no retry")
            if name in ("gpu_preflight", "train"):
                directory = "gpu_preflight" if name == "gpu_preflight" else "run"
                summary, manifest = read(root / directory / "summary.json"), read(root / directory / "manifest.json")
                if summary["actual_epochs"] != (5 if name == "gpu_preflight" else UPDATES):
                    raise RuntimeError("Wrong actual update endpoint")
                if (manifest.get("labels_used_in_training") is not True
                        or manifest.get("q_supervision", {}).get("weight") != WEIGHT
                        or manifest.get("q_supervision", {}).get("coupling_names") != ["T"]
                        or manifest.get("rsmi_weight") != 2 or manifest.get("estimator") != "histogram"
                        or not all(r["hit"] for r in manifest["execution_optimizations"]["mi_statistics_cache"].values())):
                    raise RuntimeError("Temperature supervision or unchanged MI contract mismatch")
                if name == "train":
                    job["actual_training_updates"] = UPDATES
        verify_code(root, fingerprints)
        if any(sha(p) != digest for p, digest in protected.items()):
            raise RuntimeError("Protected historical source changed")
        report = read(root / "report/report.json")
        if (report.get("status") != "complete" or report.get("labels_used_in_training") is not True
                or not report.get("model_state_unchanged")):
            raise RuntimeError("Supervised report incomplete or changed the model")
        job.update(status="complete", workflow_complete=True, current=None,
            public_files=public_files(root), finished_at=now())
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start)
        write_json(root / "job.json", job)
    return job


def launch(root, baseline):
    root, baseline = Path(root).absolute(), Path(baseline).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if socket.gethostname() != HOST:
        raise RuntimeError("Wrong launch host")
    validate_baseline(baseline)
    logs = root.parent.parent / "logs"
    logs.mkdir(exist_ok=True)
    logfile, receipt = logs / (root.name + ".log"), logs / (root.name + ".launch.json")
    if logfile.exists() or receipt.exists():
        raise FileExistsError("Existing launch receipt; inspect instead of launching twice")
    with (logs / (root.name + ".launch.lock")).open("x") as stream:
        stream.write(str(os.getpid()))
    adjustment = max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0))
    command = ["nice", "-n", str(adjustment), sys.executable, str(Path(__file__).resolve()),
        "--root", str(root), "--baseline", str(baseline), "--run"]
    with logfile.open("x") as stream:
        child = subprocess.Popen(command, cwd=baseline / "code", env=dict(os.environ,
            PYTHONPATH=str(baseline / "code"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    record = dict(status="launched", parent_pid=child.pid, command=command, root=str(root), baseline=str(baseline),
        log=str(logfile), device=DEVICE, gpu_uuid=GPU_UUID, host=socket.gethostname(), started_at=now(),
        launcher_sha256=sha(Path(__file__)), no_automatic_restart=True)
    write_json(receipt, record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--prepare-mi", action="store_true")
    args = p.parse_args(argv)
    result = (launch(args.root, args.baseline) if args.launch else
              reuse_mi(args.root, args.baseline) if args.prepare_mi else run(args.root, args.baseline))
    print(json.dumps({k:result.get(k) for k in ("status", "workflow_complete", "current", "parent_pid", "root", "error")},
        ensure_ascii=False))
    if result.get("status") == "failed":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
