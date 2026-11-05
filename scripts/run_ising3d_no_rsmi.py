"""Paired fresh 3D Ising ablation: C uses variance floor and color only.

Reuse a completed baseline's label-free cache, never its model or optimizer.
No physical sampling, MI statistics, critic, or MI diagnostic is performed.
Each GPU stage waits for two idle admissions; no other process is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from scripts.run_ising3d import cpu_seconds, freeze_code, verify_code, wait_resources
from scripts.run_spin_glass2d import GPU_UUID, sha, write_json


DATA_SHA256 = "61c9458f633bc7b1842533d8190fff5b213da674ceccebf00ed81d65c2e1c566"
INITIAL_KEYS = ("random_construction_module_sha256", "pre_affine_initialization_module_sha256",
                "training_start_module_sha256")
MODEL_FILES = ("recurrent_rg/model_nd.py", "recurrent_rg/checkpoints.py",
               "recurrent_rg/execution.py", "recurrent_rg/encoding.py",
               "recurrent_rg/adapters.py", "recurrent_rg/data/statistics.py",
               "recurrent_rg/distribution_encoder.py", "recurrent_rg/metrics.py",
               "recurrent_rg/learning_rate.py")


def read(path):
    return json.loads(Path(path).read_text())


def paired_plan(baseline, source):
    """Validate the exact completed baseline before any new output or process."""
    from recurrent_rg.config import RunConfig
    baseline, source = Path(baseline).resolve(), Path(source).resolve()
    job, old_plan = read(baseline / "job.json"), read(baseline / "plan.json")
    manifest, summary = read(baseline / "run/manifest.json"), read(baseline / "run/summary.json")
    report = read(baseline / "report/report.json")
    if (job.get("status") != "complete" or job.get("workflow_complete") is not True
            or any(row.get("returncode") != 0 for row in job["stages"])
            or summary.get("actual_epochs") != 6000 or report.get("status") != "complete"):
        raise ValueError("Require the completed 6000-update baseline and its formal report")
    config = read(baseline / "config.json")
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    if ((cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim,
            cfg.model.hidden, cfg.training.epochs, cfg.training.seed) != (3, 2, 2, 1, 96, 6000, 91)
            or cfg.data.coupling_names or cfg.model.fixed_kernel
            or cfg.model.kernel_initialization != "random"
            or not cfg.model.affine_reconstruction or cfg.loss.closure_scale_weights is not None
            or cfg.loss.kernel_noncollapse_mode != "variance_floor_ensemble"
            or cfg.loss.kernel_variance_ratio != 1 or cfg.loss.kernel_sign_weight != 1
            or cfg.loss.weights.kernel_noncollapse != 2
            or manifest.get("rsmi_weight") != 2 or manifest.get("independent_c") is not True
            or manifest.get("labels_used_in_training") is not False):
        raise ValueError("Baseline is not the expected paired default 3D Ising experiment")
    if RunConfig.from_dict(manifest["config"]).to_dict() != cfg.to_dict():
        raise ValueError("Baseline manifest and actual root config differ")
    cache = Path(cfg.data.cache).resolve()
    metadata = read(baseline / "data/evaluation_metadata.json")
    if (cache != baseline / "data/ensembles.pt" or sha(cache) != DATA_SHA256
            or manifest["data_sha256"] != DATA_SHA256 or metadata["cache_sha256"] != DATA_SHA256
            or metadata["lattice_shape"] != [16, 16, 16]):
        raise ValueError("Baseline label-free data fingerprint or geometry differs")
    unchanged = {}
    for relative in MODEL_FILES:
        expected = old_plan["source_fingerprints"][relative]
        if sha(source / relative) != expected:
            raise ValueError(f"Paired model/execution source changed: {relative}")
        unchanged[relative] = expected
    initial = {key: manifest[key] for key in INITIAL_KEYS}
    if any(set(value) != {"Q", "C", "R"} for value in initial.values()):
        raise ValueError("Missing baseline Q/C/R initialization fingerprints")
    plan = dict(schema_version=1, experiment="三维 Ising：去掉 RSMI 的配对消融",
        baseline=str(baseline), baseline_manifest_sha256=sha(baseline / "run/manifest.json"),
        baseline_report_sha256=sha(baseline / "report/report.json"), data_cache=str(cache),
        data_sha256=DATA_SHA256, data_reused_read_only=True, physical_sampling=False,
        MI_computed=False, MI_cache_read=False, critic_created=False,
        config_changes=[], command_changes={"rsmi_weight": [2, 0], "allow_nondefault_recipe": True},
        fixed_training_updates=6000, optimizer_seed=91, fresh_random_model=True,
        baseline_checkpoint_loaded=False, optimizer_warm_start=False,
        C_objective="2 * per-ensemble variance floor (ratio1) + 1 * uniform-color NLL",
        Q_objective="unchanged shared affine reconstruction and subset consistency",
        R_objective="unchanged 5 one-step + 5 microscopic-start all-horizon closure, uniform scales",
        labels_used_in_training=False, initial_module_sha256=initial,
        unchanged_model_execution_sources=unchanged,
        device="cuda:2", gpu_uuid=GPU_UUID, baseline_device=manifest["execution_device"],
        device_difference="Same RTX3090 model, different physical card; not a bitwise GPU reproduction",
        execution_preflight=dict(updates=5, scientific_result=False,
            initialization="fresh model; formal model reinitialized from the same seed, no weights reused"),
        admission="two checks >=10 seconds apart: no compute processes, >=12GiB GPU free, >=36GiB host available, load1<=1.25*cores, IO full avg10<20%",
        max_GPU_stage_seconds=3600, max_directory_bytes=2 * 1024**3,
        no_automatic_restart=True, no_automatic_migration=True,
        qualification="单 seed 有限尺寸配对消融；不同学习坐标下 RMSE/各自标准差 NRMSE 不构成同坐标严格胜负，候选斜率不是认证热指数")
    return config, plan


def prepare(root, baseline):
    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    source = Path(__file__).resolve().parents[1]
    config, plan = paired_plan(baseline, source)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "data").mkdir()
    fingerprints = freeze_code(source, root / "code")
    plan["source_fingerprints"] = fingerprints
    # Public sidecar only: no model, raw physical data, or MI cache is copied.
    shutil.copy2(baseline / "data/evaluation_metadata.json", root / "data/evaluation_metadata.json")
    write_json(root / "config.json", config)
    write_json(root / "plan.json", plan)
    write_json(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], device="cuda:2", baseline=str(baseline), error=None,
        requires_user_action=False, GPU_stage_seconds=0., CPU_core_seconds=0.,
        actual_training_updates=0, created_at=datetime.now(timezone.utc).isoformat()))
    return plan


def commands(root):
    training = [sys.executable, "-m", "experiments.rsmi_separate_qcr", "--config", str(root / "config.json"),
        "--device", "cuda:2", "--rsmi-weight", "0", "--allow-nondefault-recipe", "--skip-final-report"]
    return [
        ("gpu_preflight", training + ["--output", str(root / "gpu_preflight"), "--epochs", "5", "--benchmark-warmup", "1"], True),
        ("train", training + ["--output", str(root / "run"), "--benchmark-warmup", "5"], True),
        ("report", [sys.executable, "-m", "scripts.report_ising3d", "--root", str(root),
            "--device", "cuda:2", "--draws", "5", "--skip-mi-diagnostic"], True),
        ("comparison", [sys.executable, "-m", "scripts.compare_ising3d_no_rsmi",
            "--baseline", str(read(root / "plan.json")["baseline"]), "--candidate", str(root),
            "--output", str(root / "comparison")], False),
    ]


def validate_training_receipt(root, name, expected_updates):
    plan, manifest = read(root / "plan.json"), read(root / name / "manifest.json")
    summary = read(root / name / "summary.json")
    if summary["actual_epochs"] != expected_updates:
        raise RuntimeError("Stage did not finish its fixed endpoint; preserve output, no automatic continuation")
    if (manifest.get("rsmi_weight") != 0 or manifest.get("rsmi_enabled") is not False
            or manifest.get("estimator") is not None or manifest.get("mi_evaluated") is not False
            or manifest.get("critic_parameters") != 0
            or manifest.get("data_sha256") != DATA_SHA256
            or manifest.get("independent_c") is not True or manifest.get("labels_used_in_training") is not False):
        raise RuntimeError("No-MI independent-gradient/data contract mismatch")
    if any(manifest[key] != plan["initial_module_sha256"][key] for key in INITIAL_KEYS):
        raise RuntimeError("Initial Q/C/R fingerprints differ from baseline; do not call it paired")
    return summary


def public_files(root):
    fixed = ["job.json", "plan.json", "config.json", "data/evaluation_metadata.json",
             "gpu_preflight/manifest.json", "gpu_preflight/summary.json", "run/manifest.json", "run/summary.json",
             "run/history_progress.json"]
    for directory in ("report", "comparison"):
        fixed += [str(path.relative_to(root)) for path in sorted((root / directory).iterdir())
                  if path.is_file() and path.suffix in (".json", ".png", ".npz")]
    return fixed


def run(root):
    root = Path(root).resolve()
    job, plan = read(root / "job.json"), read(root / "plan.json")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Run once only; existing running/completed/failed state is never restarted")
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Formal execution is pinned to thinklab-105-225")
    job.update(status="running", parent_pid=os.getpid(), host=socket.gethostname(),
        started_at=datetime.now(timezone.utc).isoformat(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        for name, command, gpu in commands(root):
            verify_code(root / "code", plan["source_fingerprints"])
            if sha(plan["data_cache"]) != DATA_SHA256:
                raise RuntimeError("Baseline data fingerprint changed; no repair or rerun")
            job.update(status="waiting_resources", current=name)
            wait_resources(root, job, gpu)
            if job["GPU_stage_seconds"] >= plan["max_GPU_stage_seconds"]:
                job.update(status="budget_hold", requires_user_action=True,
                    error="GPU stage budget reached before next stage; no reduced steps or continuation")
                break
            stage = dict(name=name, status="running", command=command,
                gpu_stage=gpu, started_at=datetime.now(timezone.utc).isoformat(),
                log=str(root / "logs" / f"{name}.log"))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if gpu:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            tick = time.monotonic()
            with Path(stage["log"]).open("x") as log:
                child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running", current=name)
                write_json(root / "job.json", job)
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed",
                elapsed_seconds=time.monotonic() - tick, ended_at=datetime.now(timezone.utc).isoformat())
            if gpu:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            write_json(root / "job.json", job)
            if rc:
                raise RuntimeError(f"Stage {name} exited {rc}; no automatic restart")
            if name in ("gpu_preflight", "train"):
                result = validate_training_receipt(root, name, 5 if name == "gpu_preflight" else 6000)
                job[name + "_timing"] = result.get("timing")
                if name == "train":
                    job["actual_training_updates"] = 6000
            if name == "report":
                result = read(root / "report/report.json")
                if (result.get("status") != "complete" or not result.get("model_state_unchanged")
                        or result.get("MI_shuffle_diagnostic") is not None):
                    raise RuntimeError("Formal no-MI report is incomplete or computed MI")
            size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            if size > plan["max_directory_bytes"]:
                raise RuntimeError("Directory budget exceeded; preserve output for user decision")
        else:
            verify_code(root / "code", plan["source_fingerprints"])
            job.update(status="complete", workflow_complete=True, current=None,
                finished_at=datetime.now(timezone.utc).isoformat(), public_files=public_files(root))
    except Exception as exc:
        job.update(status="failed", requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start)
        write_json(root / "job.json", job)
    return job


def launch(root):
    root = Path(root).resolve()
    if socket.gethostname() != "thinklab-105-225" or read(root / "job.json")["status"] != "prepared":
        raise ValueError("Launch requires the newly prepared root on verified 225")
    receipt = root / "logs/launch.json"
    logfile = root / "logs/parent.log"
    if receipt.exists() or logfile.exists():
        raise FileExistsError("Existing launch receipt or log; inspect actual process, never duplicate launch")
    command = ["nice", "-n", "10", sys.executable, "-m", "scripts.run_ising3d_no_rsmi", "--root", str(root), "--run"]
    with logfile.open("x") as log:
        child = subprocess.Popen(command, cwd=root / "code", stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = dict(parent_pid=child.pid, command=command, root=str(root), log=str(logfile),
        host=socket.gethostname(), launched_at=datetime.now(timezone.utc).isoformat(), no_automatic_restart=True)
    write_json(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare and args.baseline is None:
        parser.error("--prepare requires --baseline")
    value = prepare(args.root, args.baseline) if args.prepare else launch(args.root) if args.launch else run(args.root)
    print(json.dumps(value, ensure_ascii=False))
    if value.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
