"""New-only 3D EA overlap QCR run; fixed work, frozen code, no restart.

Sampling/MI run on one CPU thread. GPU0 admission is separate and never
interferes with other users. A smoke run checks execution, not physics.
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


CONFIG_PATH = "configs/spin_glass3d_b2_overlap_g1_seed91.json"
GPU_UUID = "GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42"
HOST = "thinklab-105-225"
TEMPERATURES = [.5 * 7. ** (index / 19.) for index in range(20)]


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


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
        raise RuntimeError("Frozen source fingerprint mismatch; preserve all outputs, no repair/restart")


def resource_snapshot():
    if socket.gethostname() != HOST:
        raise RuntimeError(f"Formal run requires {HOST}")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    selected = next(line.split(",") for line in query.splitlines() if int(line.split(",")[0]) == 0)
    if selected[1].strip() != GPU_UUID:
        raise RuntimeError("Physical GPU0 UUID mismatch; do not migrate")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    active = [line for line in compute.splitlines() if line.split(",")[0].strip() == GPU_UUID]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    snapshot = dict(checked_at=datetime.now(timezone.utc).isoformat(), gpu_index=0,
        gpu_uuid=GPU_UUID, free_mib=int(selected[2]), active=active,
        mem_available_gib=available / 1024**3, load1=os.getloadavg()[0],
        cpu_count=os.cpu_count(), io_full_avg10=io,
        disk_free_gib=shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 1024**3)
    snapshot["host_ready"] = (available >= 36 * 1024**3
        and snapshot["load1"] <= 1.25 * snapshot["cpu_count"] and io < 20
        and snapshot["disk_free_gib"] >= 20)
    snapshot["gpu_ready"] = snapshot["host_ready"] and not active and snapshot["free_mib"] >= 12288
    return snapshot


def wait_resources(root, job, gpu):
    consecutive = 0
    while True:
        snapshot = resource_snapshot()
        ready = snapshot["gpu_ready" if gpu else "host_ready"]
        job.setdefault("resource_checks", []).append(dict(snapshot, required="gpu" if gpu else "host"))
        job["resource_checks"] = job["resource_checks"][-360:]
        consecutive = consecutive + 1 if ready else 0
        write_json(root / "job.json", job)
        if consecutive >= (2 if gpu else 1):
            return
        time.sleep(10)


def build_plan(config, smoke=False):
    from recurrent_rg.config import RunConfig, ScaleConfig
    from experiments.rsmi_separate_qcr import build_parser as training_parser, validate_recipe
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    validate_recipe(training_parser().parse_args(["--config", "/unused/config.json", "--output", "/unused/run"]), cfg)
    if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) != (3, 2, 2, 1):
        raise ValueError("Require 3D cube Q width2, b2, gdim1")
    if (cfg.data.coupling_names or cfg.model.fixed_kernel or cfg.model.kernel_initialization != "random"
            or cfg.model.kernel_sampling != "bernoulli" or not cfg.model.affine_reconstruction):
        raise ValueError("Require fresh unsupervised default probabilistic C/affine-Q recipe")
    weights = cfg.loss.closure_scale_weights
    if weights is not None and len(set(weights)) != 1:
        raise ValueError("No nonuniform closure-scale ablation was authorized")
    if not smoke and cfg.training.epochs != 6000:
        raise ValueError("Formal endpoint is exactly 6000 fresh updates")
    if cfg.training.threads != 1 or cfg.training.learning_rate != .001:
        raise ValueError("Require one numerical thread and fresh Adam learning rate .001")
    physical = dict(length=8 if smoke else 16,
        temperatures=[.8, 1.5, 3.5] if smoke else TEMPERATURES,
        disorders=3 if smoke else 9, replicas=4, samples=8 if smoke else 512,
        burn_in=8 if smoke else 8192, spacing=1 if smoke else 16,
        exchange_interval=1, cluster_interval=1, trace_interval=2 if smoke else 16,
        seed=20261019, initial="hot", icm_max_temperature=1.)
    from recurrent_rg.data.spin_glass_pipeline import GenerationOptions
    from recurrent_rg.data.spin_glass3d_pipeline import generation_plan, NUMERIC_LIMIT
    physical_preflight = generation_plan(GenerationOptions(
        shape=(physical["length"],) * 3, temperatures=tuple(physical["temperatures"]),
        disorders=physical["disorders"], replicas=physical["replicas"], samples=physical["samples"],
        burn_in=physical["burn_in"], spacing=physical["spacing"], exchange_interval=1,
        seed=physical["seed"], max_spin_proposals=NUMERIC_LIMIT, max_output_bytes=NUMERIC_LIMIT,
        max_memory_bytes=NUMERIC_LIMIT, max_sites=NUMERIC_LIMIT),
        cluster_interval=1, trace_interval=physical["trace_interval"], initial="hot",
        icm_max_temperature=physical["icm_max_temperature"])
    scales = ScaleConfig((physical["length"],) * 3, cfg.training.max_rg_steps, 2).shapes()
    return dict(schema_version=1, experiment="三维零场 Edwards–Anderson ±J 自旋玻璃重叠场无监督 QCR",
        execution_smoke=smoke, scientific_result=not smoke, physical=physical, physical_preflight=physical_preflight,
        scales=[list(shape) for shape in scales], physical_seed=20261019, optimizer_seed=91,
        fixed_training_updates=cfg.training.epochs, labels_used_in_training=False,
        disorder=dict(distribution="bimodal", bond_scale=1., positive_probability=.5, field=0.,
            split="entire disorder realization J; index%3==1 held out", train_count=2 if smoke else 6,
            test_count=1 if smoke else 3, inherited_historical_bonds=False),
        sampling=dict(algorithm="lazy single-site Metropolis + alternating PT + low-T isoenergetic clusters",
            ICM_temperature_gate="T<=1.0 in |J|=1 units; sampler setting, not a learned/assumed critical point",
            sweeps_per_temperature_per_replica=physical["burn_in"] + physical["samples"] * physical["spacing"],
            single_spin_visits=physical["disorders"] * physical["replicas"] * len(physical["temperatures"])
                * physical["length"]**3 * (physical["burn_in"] + physical["samples"] * physical["spacing"]),
            initialization="one group of independent random spin initializations per J; not independent hot/cold convergence groups",
            overlap="q_x=s_x^0*s_x^1 at matching J,T,saved time; one predeclared pair only",
            replica_pairs_are_not_independent=True, automatic_extension=False,
            equilibrium_certified=False),
        representation=dict(Q_window="anchored 2x2x2; 8 spins, all translations",
            target="256 oriented cube probabilities, shared affine decoder at every scale",
            C="random trainable Bernoulli nonoverlapping 2x2x2 stride2",
            MI="exact empirical full-shell joint histogram; buffer1; 152 environment spins, no projection",
            MI_qualification="Finite-sample singleton/upward bias may be severe; report matched shuffled-null subset; high MI is not a physical-information certificate",
            MI_null_subset="first TRAIN J; temperature indices 0, N_T//2, N_T-1; first64 saved configurations per cell; fixed before sampling, not chosen by critical temperature",
            gradient_routing="Q representation only; C -2 MI +2 per-ensemble variance-floor ratio1 +1 uniform-color NLL; R closure only",
            scale_weights="uniform in one-step and microscopic-start multistep branches",
            supervision="No T, J, critical point, exponent or theory slope in training inputs/loss"),
        evaluation=dict(samples_per_ensemble=cfg.evaluation.samples,
            subset="first64 postburn saved configurations per J,T in formal report, not all512",
            draws=2 if smoke else 5, mean_R_and_R_mean_are_distinct=True,
            roots="retain all bracketed candidates within observed one-step source envelope; not exhaustive or certified thermal directions",
            thermal_nu=None, single_size=True, training_seeds=1),
        GPU_execution_preflight=dict(updates=0 if smoke else 5, scientific_result=False,
            weights_reused=False, formal_model="separate fresh initialization and fresh Adam .001"),
        resources=dict(threads=1, intended_nice=10, device="cuda:0", gpu_uuid=GPU_UUID,
            host=HOST, gpu_requires_two_idle_checks_10s_apart=True, free_gpu_mib=12288,
            available_host_gib=36, minimum_free_disk_gib=20, never_interfere_other_jobs=True,
            fixed_finite_work=True, no_automatic_restart=True),
        qualification="Exploratory single-size, few-disorder, single-training-seed run; fixed sampling does not certify equilibrium or a universal exponent")


def stages(root, plan, device):
    generate = [sys.executable, "-m", "scripts.generate_spin_glass3d", "--output", str(root / "physical_data")]
    for key, value in plan["physical"].items():
        generate += ["--" + key.replace("_", "-")]
        generate += list(map(str, value)) if isinstance(value, list) else [str(value)]
    result = [
        ("physical_sampling", generate, False),
        ("prepare_overlap", [sys.executable, "-m", "scripts.prepare_spin_glass3d_qcr",
            "--source", str(root / "physical_data"), "--output", str(root / "data")], False),
        ("full_shell_MI", [sys.executable, "-m", "scripts.prepare_qcr_mi_cache",
            "--config", str(root / "config.json"), "--output", str(root / "mi_receipt.json"),
            "--cache-dir", str(root / "mi_cache")], False),
        ("train", [sys.executable, "-m", "experiments.rsmi_separate_qcr",
            "--config", str(root / "config.json"), "--output", str(root / "run"),
            "--device", device, "--skip-final-report", "--benchmark-warmup", "5"], True),
        ("report", [sys.executable, "-m", "scripts.report_spin_glass3d", "--root", str(root),
            "--device", device, "--draws", "2" if plan["execution_smoke"] else "5"], True),
    ]
    if device.startswith("cuda"):
        result[3][1].extend(["--mi-cache-dir", str(root / "mi_cache")])
        if not plan["execution_smoke"]:
            result.insert(3, ("gpu_preflight", [sys.executable, "-m", "experiments.rsmi_separate_qcr",
                "--config", str(root / "config.json"), "--output", str(root / "gpu_preflight"),
                "--device", device, "--epochs", "5", "--skip-final-report",
                "--mi-cache-dir", str(root / "mi_cache"), "--benchmark-warmup", "1"], True))
    return result


def public_files(root):
    result = ["job.json", "plan.json", "config.json", "mi_receipt.json", "data/evaluation_metadata.json", "data/status.json",
        "physical_data/plan.json", "physical_data/manifest.json", "physical_data/status.json",
        "physical_data/sampling.json", "run/manifest.json", "run/summary.json"]
    if (root / "gpu_preflight/summary.json").exists():
        result += ["gpu_preflight/manifest.json", "gpu_preflight/summary.json"]
    return result + [str(path.relative_to(root)) for path in sorted((root / "report").iterdir())
        if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".png", ".npz")]


def run(root, *, device="cuda:0", smoke=False):
    root = Path(root).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if device not in ("cpu", "cuda:0") or (device == "cpu" and not smoke):
        raise ValueError("Formal run is pinned to cuda:0; CPU is only for execution smoke")
    if device.startswith("cuda") and socket.gethostname() != HOST:
        raise ValueError(f"Formal host must be {HOST}")
    source = Path(__file__).resolve().parents[1]
    config = json.loads((source / CONFIG_PATH).read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, max_rg_steps=1, system_batch=4, sample_batch=8,
            checkpoint_every=2, log_every=1)
        config["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    plan = build_plan(config, smoke)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    fingerprints = freeze_code(source, root / "code")
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
                raise RuntimeError(f"Stage {name} exited {rc}; no restart, migration or overwrite")
            if name == "physical_sampling":
                sampling = json.loads((root / "physical_data/sampling.json").read_text())
                job["sampling_qualification"] = {key: sampling.get(key) for key in
                    ("warning_count", "equilibrium_certified", "independent_initialization_comparison_performed")}
            if name in ("gpu_preflight", "train"):
                directory = "gpu_preflight" if name == "gpu_preflight" else "run"
                summary = json.loads((root / directory / "summary.json").read_text())
                expected = 5 if name == "gpu_preflight" else config["training"]["epochs"]
                if summary["actual_epochs"] != expected:
                    raise RuntimeError(f"{name} did not reach fixed endpoint; no automatic continuation")
                timing = summary.get("timing", {})
                job[name + "_timing"] = {key: value for key, value in timing.items()
                    if key not in ("step_seconds_all", "steady_step_seconds")}
        verify_code(root / "code", fingerprints)
        report = json.loads((root / "report/report.json").read_text())
        if report.get("status") != "complete" or not report.get("model_state_unchanged"):
            raise RuntimeError("Formal report incomplete or changed model state")
        job.update(status="complete", workflow_complete=True, current=None,
            public_files=public_files(root), finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        job.update(status="failed", requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start,
            directory_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))
        write_json(root / "job.json", job)
    return job


def launch(root, *, device="cuda:0", smoke=False):
    root = Path(root).absolute()
    source = Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if socket.gethostname() != HOST or device != "cuda:0":
        raise ValueError("Detached launch requires the verified 225 host and cuda:0")
    config = json.loads((source / CONFIG_PATH).read_text())
    build_plan(config, smoke)
    directory = root.parent.parent / "logs"
    directory.mkdir(exist_ok=True)
    logfile = directory / (root.name + ".log")
    receipt = directory / (root.name + ".launch.json")
    lock = directory / (root.name + ".launch.lock")
    if logfile.exists() or receipt.exists():
        raise FileExistsError("Existing launch log/receipt; inspect actual process, never duplicate launch")
    with lock.open("x") as stream:
        stream.write(f"launcher_pid={os.getpid()}\n")
    niceness = max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0))
    command = ["nice", "-n", str(niceness), sys.executable, "-m", "scripts.run_spin_glass3d",
        "--root", str(root), "--device", device]
    if smoke:
        command.append("--execution-smoke")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with logfile.open("x") as output:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = dict(parent_pid=child.pid, command=command, host=socket.gethostname(), source=str(source),
        root=str(root), log=str(logfile), started_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(source / CONFIG_PATH), launcher_sha256=sha(Path(__file__)),
        device=device, gpu_uuid=GPU_UUID, no_automatic_restart=True,
        execution_smoke=smoke, scientific_result=not smoke,
        note="Receipt alone does not prove launch/completion; inspect root job and actual parent/child identities")
    write_json(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--execution-smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(args.root, device=args.device, smoke=args.execution_smoke), ensure_ascii=False))
        return
    result = run(args.root, device=args.device, smoke=args.execution_smoke)
    print(json.dumps({key: result.get(key) for key in ("status", "workflow_complete", "current", "error")}, ensure_ascii=False))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
