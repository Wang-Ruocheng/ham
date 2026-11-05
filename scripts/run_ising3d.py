"""New-output-only 3D ferromagnetic Ising b2, scalar default QCR experiment.

CPU sampling and exact full-shell MI preparation precede GPU admission.
No historical run is modified, resumed, or used as a model initialization.
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

from scripts.run_spin_glass2d import GPU_UUID, gpu_snapshot, sha, write_json

CONFIG_PATH = "configs/ising3d_b2_g1_seed91.json"


def freeze_code(source, target):
    target.mkdir()
    fingerprints = {}
    for package in ("recurrent_rg", "scripts", "experiments"):
        for origin in sorted((source / package).rglob("*.py")):
            relative = origin.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            fingerprints[str(relative)] = sha(destination)
    for relative in ("AGENTS.md", CONFIG_PATH):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        fingerprints[relative] = sha(destination)
    write_json(target / "fingerprints.json", fingerprints)
    return fingerprints


def verify_code(code, fingerprints):
    if any(sha(code / relative) != expected for relative, expected in fingerprints.items()):
        raise RuntimeError("Frozen source fingerprint changed; preserve output, no automatic repair")


def cpu_seconds():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def build_plan(config, smoke=False):
    from recurrent_rg.config import RunConfig, ScaleConfig
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) != (3, 2, 2, 1):
        raise ValueError("Require explicit 3D anchored-cube Q width2, b2, scalar g")
    physical = dict(size=8 if smoke else 16, coupling_count=6 if smoke else 24,
        coupling_min=.12, coupling_max=.36, chains=2 if smoke else 4,
        samples=8 if smoke else 128, burn_in_clusters=8 if smoke else 4096,
        clusters_between=2 if smoke else 64, seed=20261019)
    scales = ScaleConfig((physical["size"],) * 3, cfg.training.max_rg_steps, 2).shapes()
    if cfg.data.coupling_names or cfg.model.fixed_kernel or cfg.model.kernel_initialization != "random":
        raise ValueError("No labels or pretrained/fixed C allowed")
    return dict(schema_version=1, experiment="三维零场单最近邻铁磁 Ising，b2，一维无监督 QCR",
        execution_smoke=smoke, scientific_result=not smoke, physical=physical,
        scales=[list(shape) for shape in scales], physical_seed=20261019, optimizer_seed=91,
        fixed_training_updates=cfg.training.epochs, labels_used_in_training=False,
        GPU_execution_preflight=dict(updates=0 if smoke else 5, scientific_result=False,
            initialization="Separate fresh model/output; formal model is freshly initialized again; no preflight weights or optimizer reused"),
        representation=dict(Q_window="anchored 2x2x2, all translations, 8 input spins",
            target="all 256 oriented cube motif probabilities; shared affine decoder at all scales",
            decoder_column_norm=1, full_rank_forced=False, PCA_initialization=False,
            C="random trainable Bernoulli nonoverlapping 2x2x2 stride2",
            MI="exact empirical full-shell joint histogram, buffer1, 152 environment spins; no shell projection",
            gradient_routing="Q representation only; C -2 MI+2 per-ensemble variance+color; R closure only",
            scale_weights="uniform within both closure branches"),
        sampling_convention="Fixed Wolff cluster proposals, not full-lattice sweeps; raw diagnostics before Z2 flips",
        sharing=dict(cpu_threads=1, inherited_nice=10, gpu="cuda:2", gpu_uuid=GPU_UUID,
            admission="two idle GPU checks >=10s apart; free>=12GiB, host available>=36GiB, load<=1.25*cores, IO full<20%",
            others="never kill, renice, migrate, or interfere with existing jobs"),
        no_automatic_restart=True, equilibrium_certified=False, exponent_certified=False)


def stages(root, plan, device):
    p = plan["physical"]
    generate = [sys.executable, "-m", "scripts.generate_ising3d_wolff", "--output", str(root / "physical_data")]
    for key, value in p.items():
        generate += ["--" + key.replace("_", "-"), str(value)]
    commands = [
        ("physical_sampling", generate, False),
        ("prepare_data", [sys.executable, "-m", "scripts.prepare_ising3d_qcr",
            "--source", str(root / "physical_data"), "--output", str(root / "data")], False),
        ("full_shell_MI", [sys.executable, "-m", "scripts.prepare_qcr_mi_cache",
            "--config", str(root / "config.json"), "--output", str(root / "mi_receipt.json"),
            "--cache-dir", str(root / "mi_cache")], False),
        ("train", [sys.executable, "-m", "experiments.rsmi_separate_qcr",
            "--config", str(root / "config.json"), "--output", str(root / "run"),
            "--device", device, "--skip-final-report",
            "--benchmark-warmup", "5"], True),
        ("report", [sys.executable, "-m", "scripts.report_ising3d", "--root", str(root),
            "--device", device, "--draws", "2" if plan["execution_smoke"] else "5"], True),
    ]
    if device.startswith("cuda"):
        commands[3][1].extend(["--mi-cache-dir", str(root / "mi_cache")])
        if not plan["execution_smoke"]:
            commands.insert(3, ("gpu_preflight", [sys.executable, "-m", "experiments.rsmi_separate_qcr",
                "--config", str(root / "config.json"), "--output", str(root / "gpu_preflight"),
                "--device", device, "--epochs", "5", "--skip-final-report",
                "--mi-cache-dir", str(root / "mi_cache"), "--benchmark-warmup", "1"], True))
    return commands


def wait_resources(root, job, gpu):
    checks = 0
    while True:
        snapshot = gpu_snapshot()
        ready = snapshot["ready"] if gpu else (snapshot["mem_available_gib"] >= 36
            and snapshot["load1"] <= 1.25 * snapshot["cpu_count"] and snapshot["io_full_avg10"] < 20)
        job.setdefault("resource_checks", []).append(dict(snapshot, required="gpu" if gpu else "host"))
        # Keep the receipt bounded during long shared-host resource waits.
        job["resource_checks"] = job["resource_checks"][-360:]
        checks = checks + 1 if ready else 0
        write_json(root / "job.json", job)
        if checks >= (2 if gpu else 1):
            return
        time.sleep(10)


def public_files(root):
    fixed = ["job.json", "plan.json", "config.json", "mi_receipt.json", "data/evaluation_metadata.json",
        "physical_data/plan.json", "physical_data/manifest.json", "physical_data/status.json",
        "physical_data/sampling.json", "run/manifest.json", "run/summary.json"]
    if (root / "gpu_preflight/summary.json").exists():
        fixed += ["gpu_preflight/summary.json", "gpu_preflight/manifest.json"]
    return fixed + [str(path.relative_to(root)) for path in sorted((root / "report").iterdir())
        if path.is_file() and path.suffix in (".json", ".png", ".npz")]


def run(root, *, device="cuda:2", smoke=False):
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if device not in ("cpu", "cuda:2") or (device == "cpu" and not smoke):
        raise ValueError("Formal GPU is cuda:2; CPU only allowed for implementation smoke")
    if device.startswith("cuda") and socket.gethostname() != "thinklab-105-225":
        raise ValueError("GPU host must be thinklab-105-225")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    source = Path(__file__).resolve().parents[1]
    fingerprints = freeze_code(source, root / "code")
    config = json.loads((root / "code" / CONFIG_PATH).read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, max_rg_steps=1, system_batch=4, sample_batch=8,
            checkpoint_every=2, log_every=1)
        config["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    plan = build_plan(config, smoke)
    plan["source_fingerprints"] = fingerprints
    write_json(root / "config.json", config)
    write_json(root / "plan.json", plan)
    job = dict(status="running", workflow_complete=False, parent_pid=os.getpid(), host=socket.gethostname(),
        started_at=datetime.now(timezone.utc).isoformat(), current=None, stages=[], device=device,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0), execution_smoke=smoke, scientific_result=not smoke,
        error=None, requires_user_action=False, GPU_stage_seconds=0., CPU_core_seconds=0.)
    write_json(root / "job.json", job)
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        for name, command, gpu in stages(root, plan, device):
            verify_code(root / "code", fingerprints)
            if device.startswith("cuda"):
                job.update(status="waiting_resources", current=name)
                wait_resources(root, job, gpu)
            stage = dict(name=name, status="running", command=command, gpu_stage=gpu and device.startswith("cuda"),
                started_at=datetime.now(timezone.utc).isoformat(), log=str(root / "logs" / f"{name}.log"))
            job["stages"].append(stage)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
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
                rc = child.wait()
            stage.update(returncode=rc, status="complete" if rc == 0 else "failed",
                elapsed_seconds=time.monotonic() - stage_started, ended_at=datetime.now(timezone.utc).isoformat())
            if stage["gpu_stage"]:
                job["GPU_stage_seconds"] += stage["elapsed_seconds"]
            job["CPU_core_seconds"] = cpu_seconds() - cpu_start
            write_json(root / "job.json", job)
            if rc:
                raise RuntimeError(f"Stage {name} exited {rc}; no restart or overwrite")
            if name == "gpu_preflight":
                summary = json.loads((root / "gpu_preflight/summary.json").read_text())
                if summary["actual_epochs"] != 5:
                    raise RuntimeError("GPU preflight did not complete its five implementation-check updates")
                timing = summary["timing"]
                steady = sorted(timing.get("steady_step_seconds", []))
                job["GPU_execution_preflight"] = dict(scientific_result=False,
                    timing=timing, formal_model_not_started=True,
                    raw_update_only_projection_seconds=None if not steady else steady[len(steady)//2] * config["training"]["epochs"],
                    projection_excludes="Full evaluation/logging/checkpoint overhead and shared-resource waits; estimate, not deadline")
            if name == "train":
                summary = json.loads((root / "run/summary.json").read_text())
                if summary["actual_epochs"] != config["training"]["epochs"]:
                    raise RuntimeError("Training ended before the fixed endpoint; no automatic continuation")
                job["training_timing"] = summary.get("timing")
        verify_code(root / "code", fingerprints)
        report = json.loads((root / "report/report.json").read_text())
        if report.get("status") != "complete" or not report.get("model_state_unchanged"):
            raise RuntimeError("Incomplete or mutating frozen-model report")
        job.update(status="complete", workflow_complete=True, current=None,
            public_files=public_files(root), finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        job.update(status="failed", requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start,
            directory_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))
        write_json(root / "job.json", job)
    return job


def launch(root, *, device="cuda:2", smoke=False):
    """Detach once with a public launch receipt; a lost SSH is not a retry cue."""
    root = Path(root).resolve()
    source = Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if socket.gethostname() != "thinklab-105-225":
        raise ValueError("Detached experiment launch requires the verified 225 host")
    # Validate the intended recipe before creating any process or launch file.
    config = json.loads((source / CONFIG_PATH).read_text())
    if smoke:
        config["training"]["max_rg_steps"] = 1
        config["evaluation"]["max_rg_steps"] = 1
    build_plan(config, smoke)
    directory = root.parent.parent / "logs"
    directory.mkdir(exist_ok=True)
    logfile = directory / (root.name + ".log")
    receipt = directory / (root.name + ".launch.json")
    if logfile.exists() or receipt.exists():
        raise FileExistsError("Existing launch receipt/log; inspect actual process, never duplicate launch")
    command = ["nice", "-n", "10", sys.executable, "-m", "scripts.run_ising3d",
        "--root", str(root), "--device", device]
    if smoke:
        command.append("--execution-smoke")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with logfile.open("x") as output:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = dict(parent_pid=child.pid, command=command, host=socket.gethostname(),
        source=str(source), root=str(root), log=str(logfile), started_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(source / CONFIG_PATH), launcher_sha256=sha(Path(__file__)),
        execution_smoke=smoke, scientific_result=not smoke,
        no_automatic_restart=True, note="Launch receipt alone is not completion; job.json and actual parent/child processes are authoritative")
    write_json(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:2"), default="cuda:2")
    parser.add_argument("--execution-smoke", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Detach a new run once, saving a public launch receipt")
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(args.root, device=args.device, smoke=args.execution_smoke), ensure_ascii=False))
        return
    result = run(args.root, device=args.device, smoke=args.execution_smoke)
    print(json.dumps({k: result.get(k) for k in ("status", "workflow_complete", "current", "error")}, ensure_ascii=False))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
