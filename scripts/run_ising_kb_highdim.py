"""Fresh, label-free b3 Ising K/B experiments in latent dimensions 5 and 10.

Reuse the historical physical cache, not any historical weights. Own child
sessions only are monitored; occupied resources cause a wait, never a restart
of this or the unrelated stopped B0/dimension100 experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

DATA_SHA = "ffc776bf6c456610e3176dfe4b328b5a4cbcd6dcb5b90f900399d591d44b5c13"
DIMENSIONS = (5, 10)
UPDATES = 12000
HOST = "thinklab-105-225"
GPU = "cuda:2"
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
REPORT_SECONDS = 900
ENTRY_AVAILABLE_GIB = 18.0


def native_cuda_environment():
    """Never silently reinterpret physical GPU2 through an inherited mask."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("Inherited CUDA_VISIBLE_DEVICES is forbidden for this physical-GPU2 experiment")
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def verify_cuda_device(root):
    """Metadata only, after GPU admission; no tensors, inference or updates."""
    import torch
    native_cuda_environment()
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA ordering must be set before importing/initializing CUDA")
    properties = torch.cuda.get_device_properties(GPU)
    actual = str(getattr(properties, "uuid", ""))
    if actual.removeprefix("GPU-").lower() != GPU_UUID.removeprefix("GPU-").lower():
        raise RuntimeError(f"Torch {GPU} UUID {actual!r} does not match physical GPU2")
    receipt = dict(status="complete", device=GPU, gpu_uuid=GPU_UUID,
                   torch_uuid=actual, name=properties.name, metadata_only=True,
                   training_updates=0, tensor_allocations=0, at=now())
    save(Path(root) / "cuda_verification.json", receipt, exclusive=True)
    return receipt


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as f:
            f.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def build_config(cache, theory_dim, *, smoke=False):
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    if type(theory_dim) is not int or theory_dim not in DIMENSIONS:
        raise ValueError("This authorization is exactly dimg=5 and dimg=10")
    template = Path(__file__).resolve().parents[1] / "configs/ising2d_KB_g2_affineQ_joint_1200_seed91.json"
    config = read(template)
    config["data"].update(cache=str(Path(cache).resolve()), physical_model="ising2d",
                          coupling_names=["K", "B"], spin_flip_augment=False,
                          spin_flip_symmetric=False, train_pool_replicas=1, test_pool_replicas=1)
    config["model"].update(theory_dim=theory_dim, dimension=2, block_factor=3,
                           hidden=96, width=3, affine_reconstruction=True,
                           reconstruction_hidden=0, theory_head="nonlinear",
                           kernel_initialization="random", fixed_kernel=False,
                           kernel_sampling="bernoulli", binary_lookup=True,
                           kernel_layout="nonoverlapping")
    config["loss"].update(closure_scale_weights=None, rollout_weight=5.0,
                          kernel_sign_weight=1.0, kernel_variance_ratio=1.0,
                          kernel_noncollapse_mode="variance_floor_ensemble",
                          coordinate_reduction="sum", closure_metric="euclidean")
    config["loss"]["weights"].update(anchor=2.0, coarse_consistency=5.0,
                                       separation=10.0, commutator=5.0, kernel_noncollapse=2.0)
    config["training"].update(epochs=2 if smoke else UPDATES, max_rg_steps=2,
                              system_batch=16, sample_batch=64, seed=91, threads=2,
                              learning_rate=.001, lr_schedule="multistep",
                              lr_milestones=[6000, 9000], lr_gamma=.3,
                              log_every=1 if smoke else 200,
                              checkpoint_every=1 if smoke else 1000,
                              gradient_mode="qc_r_separate", train_modules=["Q", "C", "R"])
    config["evaluation"].update(max_rg_steps=2, samples=64, systems=24)
    config["schema_version"] = 5
    parsed = RunConfig.from_dict(config)
    parsed.validate()
    validate_recipe(build_parser().parse_args(["--config", "unused", "--output", "unused"]), parsed)
    return parsed.to_dict()


def compare_configs(config5, config10):
    def changes(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                raise ValueError("Configuration fields differ")
            return [v for key in sorted(a) for v in changes(a[key], b[key], f"{prefix}.{key}".strip("."))]
        return [] if a == b else [prefix]
    result = changes(config5, config10)
    if (result != ["model.theory_dim"] or config5["model"]["theory_dim"] != 5
            or config10["model"]["theory_dim"] != 10):
        raise ValueError("Only model.theory_dim=5/10 may differ between arms")
    return result


def validate_audit(audit):
    K, B = audit.get("K", []), audit.get("B", [])
    expected_k = [round(.2 + .03 * i, 8) for i in range(17)]
    expected_b = [round(-.01 + .001 * i, 8) for i in range(21)]
    if (K != expected_k or B != expected_b or audit.get("sites") != 36
            or audit.get("draws") != 64 or audit.get("replicas") != 2
            or audit.get("seed") != 91 or audit.get("grid") != "positive-uniform-B"
            or audit.get("train_shape") != [474, 64, 36, 36, 1]
            or audit.get("test_shape") != [240, 64, 36, 36, 1]
            or audit.get("equilibration_sweeps") != 512 or audit.get("sweeps_between") != 16):
        raise ValueError("Historical K/B physical sampling protocol changed")
    rows = audit.get("ensembles", [])
    cells, seeds, counts = defaultdict(list), set(), Counter()
    for row in rows:
        k, b, rep = row["K"], row["B"], row["replica"]
        if k not in K or b not in B or type(rep) is not int or rep not in (0, 1):
            raise ValueError("Unknown physical chain/cell")
        if type(row["test"]) is not bool or row["seed"] in seeds:
            raise ValueError("Duplicate physical seed or invalid split")
        ki, bi = K.index(k), B.index(b)
        if row["seed"] != 91 * 100000 + 1000 * ki + 2 * bi + rep:
            raise ValueError("Physical chain seed does not match its K/B cell and replica")
        if row["test"] != ((ki + abs(bi - 10)) % 3 == 1):
            raise ValueError("Physical parameter-cell split differs from the historical rule")
        seeds.add(row["seed"])
        cells[(k, b)].append(row)
        counts[row["test"]] += 1
    if len(rows) != 714 or len(cells) != 357 or counts != {False: 474, True: 240}:
        raise ValueError("Incomplete K/B cells or split counts")
    for (k, b), pair in cells.items():
        if (len(pair) != 2 or {r["replica"] for r in pair} != {0, 1}
                or len({r["test"] for r in pair}) != 1
                or pair[0]["test"] != cells[(k, round(-b, 8))][0]["test"]):
            raise ValueError("Physical replicas/sign-related cells leak across splits")
    return dict(K=K, B=B, cells=357, independent_physical_chains=714, chains_per_cell=2,
                samples_per_chain=64, train_shape=audit["train_shape"], test_shape=audit["test_shape"],
                lattice_sizes=[36, 12, 4], equilibration_sweeps=512, sweeps_between=16,
                physical_sampling_started=False,
                caution="Historical finite-chain data, not newly certified equilibrated; two chains/cell limit uncertainty inference")


def report_timeout(returncode, elapsed):
    return returncode == 124 or (returncode == 137 and elapsed >= REPORT_SECONDS)


def prepare(root, cache, audit):
    from scripts.run_ising_highdim import source_hashes
    from recurrent_rg.config import RunConfig
    from recurrent_rg.runner import prepared_data
    root, cache, audit = Path(root).resolve(), Path(cache).resolve(), Path(audit).resolve()
    native_cuda_environment()
    project = Path(__file__).resolve().parents[1]
    if socket.gethostname() != HOST or root.exists() or root.is_symlink():
        raise ValueError("Wrong host or non-new output; no overwrite/restart")
    if sha(cache) != DATA_SHA:
        raise ValueError("Historical physical cache fingerprint mismatch")
    data = validate_audit(read(audit))
    configs = [build_config(cache, dim) for dim in DIMENSIONS]
    compare_configs(*configs)
    # Full bounded binary/shape and whole-cell split validation, no MI/model.
    train, test = prepared_data(RunConfig.from_dict(configs[0]), mmap=True)
    if (list(train.configurations.shape) != data["train_shape"]
            or list(test.configurations.shape) != data["test_shape"]):
        raise ValueError("Cache tensors do not match their physical audit")
    for split, ds in [(False, train), (True, test)]:
        expected = Counter((r["K"], r["B"]) for r in read(audit)["ensembles"] if r["test"] == split)
        observed = Counter(tuple(float(v) for v in row) for row in ds.couplings.reshape(len(ds.configurations), 2))
        if observed != expected:
            raise ValueError("Actual label cells differ from physical sampling audit")
    del train, test
    root.mkdir(parents=True, exist_ok=False)
    for name in ("configs", "logs", "profiles"):
        (root / name).mkdir()
    arms = []
    for dim, config in zip(DIMENSIONS, configs):
        name = f"dim{dim}"
        (root / name).mkdir()
        formal = root / "configs" / f"{name}.json"
        smoke = root / "configs" / f"smoke_{name}.json"
        save(formal, config, exclusive=True)
        save(smoke, build_config(cache, dim, smoke=True), exclusive=True)
        arms.append(dict(name=name, theory_dim=dim, config_path=str(formal), smoke_config_path=str(smoke),
                         run=str(root / name / "run"), report=str(root / name / "report")))
    plan = dict(schema_version=1, experiment="二维Ising双参数KB高维无监督QCR", output=str(root),
                code_root=str(project), data_cache=str(cache), data_audit=str(audit), data_sha256=DATA_SHA,
                data_audit_sha256=sha(audit), physical_sampling=data, arms=arms,
                source_sha256=source_hashes(project),
                configuration_sha256={str(f): sha(f) for f in (root / "configs").glob("*.json")},
                protocol=dict(dimensions=list(DIMENSIONS), updates=UPDATES, batch_ensembles=16, samples=64,
                              seed=91, block_factor=3, hidden=96, lattice_sizes=[36, 12, 4],
                              independent_C=True, labels_used_in_training=False, fresh_Q_C_R_Adam=True,
                              spin_flip_augment=False, spin_flip_symmetric=False, uniform_scale_weights=True,
                              source_weights_used=False, physical_sampling_updates=0, extra_loss=False,
                              endpoint_selection="Fixed12000, no TEST/theory/plateau selection",
                              comparison_caution="Within KB only theory_dim changes; B0 b2 L108 is NOT a matched dim-only control",
                              report_max_seconds=840, report_wall_cap_seconds=REPORT_SECONDS,
                              report_repeats=5, geometric_analysis_not_physical_dimension_certification=True,
                              report_no_critical_exponents=True),
                resources=dict(host=HOST, gpu=GPU, gpu_uuid=GPU_UUID, entry_available_gib=ENTRY_AVAILABLE_GIB,
                               host_hard_floor_gib=8, own_anonymous_cap_gib=6, own_gpu_cap_gib=14,
                               gpu_free_gib=16, gpu_checks=2, check_spacing_seconds=10,
                               max_load_per_cpu=1.25, max_io_full_avg10=20, min_disk_free_gib=20,
                               wait_without_interference=True, nice=10, automatic_retry=False))
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
                                  parent_pid=None, stages=[], completed_arms=[], error=None,
                                  requires_user_action=False, labels_used_in_training=False,
                                  physical_sampling_updates=0), exclusive=True)
    return plan


def unchanged(plan):
    from scripts.run_ising_highdim import source_hashes
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen execution/scientific source changed")
    for path, expected in plan["configuration_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Frozen config changed")
    if sha(plan["data_audit"]) != plan["data_audit_sha256"] or sha(plan["data_cache"]) != DATA_SHA:
        raise RuntimeError("Frozen physical data identity changed")


def admission(root, job, stage, *, gpu):
    from scripts import run_ising_highdim_low_ram as guard
    ready = 0
    while True:
        row = guard.snapshot(ENTRY_AVAILABLE_GIB)
        row["requires_gpu"] = gpu
        ready = ready + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-120:]
        save(root / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 60)


def validate_training(root, plan, arm, *, smoke):
    expected_steps = 2 if smoke else UPDATES
    run = root / "smoke" / arm["name"] / "run" if smoke else Path(arm["run"])
    manifest, summary = read(run / "manifest.json"), read(run / "summary.json")
    config = read(arm["smoke_config_path"] if smoke else arm["config_path"])
    required = dict(training_driver="rsmi_separate_qcr", precision="torch.float64", estimator="histogram",
                    environment_mode="full_shell", buffer=1, independent_c=True,
                    labels_used_in_training=False, rsmi_weight=2.0, execution_device=GPU, source=None)
    if (any(manifest.get(k) != v for k, v in required.items())
            or manifest.get("config") != config or summary.get("config") != config
            or manifest.get("data_sha256") != DATA_SHA or manifest.get("q_supervision")
            or manifest.get("resumed_from") or manifest.get("frozen_C")
            or manifest.get("data_driven_initialization")
            or summary.get("actual_epochs") != expected_steps
            or summary.get("requested_epochs") != expected_steps
            or not summary.get("final_report_skipped")
            or any(summary.get(k) for k in ["stopped_by_wall_limit", "stopped_by_training_limit"])):
        raise RuntimeError("Fresh default recipe/endpoint/label-free provenance mismatch")
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy
    policy = LearningRatePolicy(RunConfig.from_dict(config).training)
    if summary.get("lr_scheduler_state") != policy.checkpoint_state(expected_steps):
        raise RuntimeError("Optimizer update clock/LR mismatch")
    history = summary.get("history", [])
    expected_history = sorted({1, expected_steps, *range(config["training"]["log_every"], expected_steps + 1,
                                                      config["training"]["log_every"])})
    if [row.get("epoch") for row in history] != expected_history:
        raise RuntimeError("Logged update history incomplete")
    if any(not math.isclose(row.get("learning_rate", -1), policy.rate_for_update(row["epoch"]),
                            rel_tol=1e-12, abs_tol=0) for row in history):
        raise RuntimeError("Logged learning rate differs from the fixed update schedule")
    if any(not math.isfinite(v) for row in history for v in row.values() if isinstance(v, (float, int))):
        raise RuntimeError("Nonfinite training metric")
    final = run / f"training_step_{expected_steps}.pt"
    if not final.is_file():
        raise RuntimeError("Missing exact endpoint checkpoint")
    mi = manifest.get("execution_optimizations", {}).get("mi_statistics_cache", {})
    if set(mi) != {"train", "test"} or any(r.get("cache_version") != 1 for r in mi.values()):
        raise RuntimeError("Missing reusable static full-shell MI statistics")
    if not smoke and any(not r.get("hit") for r in mi.values()):
        raise RuntimeError("Formal arms must reuse smoke-built data-only MI cache")
    low_ram = manifest.get("execution_optimizations", {}).get("low_ram_execution", {})
    timing = summary.get("timing", {})
    if low_ram.get("enabled") is not True or low_ram.get("source_dtype_unchanged") is not True:
        raise RuntimeError("Low-RAM execution policy was not active")
    if (timing.get("enabled") is not True or timing.get("updates") != expected_steps
            or any(type(timing.get(key)) is not int or not 0 < timing[key] <= 14 * 1024**3
                   for key in ("cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes"))):
        raise RuntimeError("CUDA allocator lifetime peaks missing or exceed own 14 GiB cap")
    return dict(status="complete", actual_updates=expected_steps, fresh=True, labels_used_in_training=False,
                checkpoint=str(final), checkpoint_sha256=sha(final), manifest_sha256=sha(run / "manifest.json"),
                final_learning_rate=history[-1].get("learning_rate"), mi_cache=mi,
                low_ram_execution=low_ram, timing=timing,
                module_initialization_provenance=manifest.get("module_initialization_provenance"))


def execute(root):
    from scripts import run_ising_highdim_low_ram as guard
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    native_cuda_environment()
    if socket.gethostname() != HOST or job["status"] != "prepared" or job["stages"]:
        raise RuntimeError("Wrong host or workflow already executed; no restart")
    os.nice(10)
    job.update(parent_pid=os.getpid(), started_at=now(), host=socket.gethostname(), device=GPU,
               status="running", actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        phases = [("verify", plan["arms"][0])]
        phases += [("smoke", a) for a in plan["arms"]]
        phases += [(kind, arm) for arm in plan["arms"] for kind in ("train", "report")]
        for kind, arm in phases:
            name = "cuda_verification" if kind == "verify" else f"{kind}_{arm['name']}"
            unchanged(plan)
            gpu = kind != "report"
            admission(root, job, name, gpu=gpu)
            unchanged(plan)
            if kind == "verify":
                command = [sys.executable, "-u", "-m", "scripts.run_ising_kb_highdim",
                           "--verify-device", "--output", str(root)]
            elif kind == "report":
                command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "900s", sys.executable,
                           "-u", "-m", "scripts.report_ising_kb_highdim", "--run", arm["run"], "--output", arm["report"],
                           "--checkpoint", str(UPDATES), "--threads", "1", "--sampling-repeats", "5",
                           "--max-seconds", "840", "--progress-path", str(root / arm["name"] / "report_progress.json")]
            else:
                run = root / "smoke" / arm["name"] / "run" if kind == "smoke" else Path(arm["run"])
                command = [sys.executable, "-u", "-m", "experiments.rsmi_separate_qcr", "--config",
                           arm["smoke_config_path"] if kind == "smoke" else arm["config_path"],
                           "--output", str(run), "--device", GPU, "--mi-cache-dir", str(root / "mi_cache"),
                           "--skip-final-report", "--benchmark-warmup", "0"]
            env = native_cuda_environment()
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
                env[key] = "2" if gpu else "1"
            env.update(PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            started = time.monotonic()
            log = root / "logs" / f"{name}.log"
            with log.open("x") as output:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdout=output,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                stage = dict(name=name, status="running", started_at=now(), command=command, pid=child.pid,
                             log=str(log), gpu=gpu, profile=str(root / "profiles" / f"{name}.json"))
                job.update(status="running", current=name)
                job["stages"].append(stage)
                save(root / "job.json", job)
                try:
                    guard.monitor(child, gpu=gpu, profile_path=stage["profile"])
                except BaseException as error:
                    stage["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    stage.update(returncode=child.poll(), elapsed_seconds=time.monotonic() - started,
                                 ended_at=now(), status="complete" if child.poll() == 0 else "failed")
                    save(root / "job.json", job)
            if child.returncode:
                if kind == "report" and report_timeout(child.returncode, stage["elapsed_seconds"]):
                    job.update(status="analysis_budget_hold", requires_user_action=True,
                               error=f"{name} exceeded report wall budget; partial outputs preserved")
                    save(root / "job.json", job)
                    return
                if kind == "report" and (root / arm["name"] / "report_progress.json").exists():
                    if read(root / arm["name"] / "report_progress.json").get("status") == "analysis_budget_hold":
                        job.update(status="analysis_budget_hold", requires_user_action=True,
                                   error=f"{name} reached analysis budget; not a training failure")
                        save(root / "job.json", job)
                        return
                raise RuntimeError(f"{name} returncode={child.returncode}; no retry")
            unchanged(plan)
            if kind == "verify":
                receipt = read(root / "cuda_verification.json")
                if receipt.get("gpu_uuid") != GPU_UUID or receipt.get("training_updates") != 0:
                    raise RuntimeError("Missing exact CUDA device identity")
            elif gpu:
                receipt = validate_training(root, plan, arm, smoke=kind == "smoke")
            else:
                marker = read(Path(arm["report"]) / "status.json")
                if (marker.get("status") != "complete" or not marker.get("report_complete")
                        or marker.get("checkpoint_epoch") != UPDATES or marker.get("theory_dim") != arm["theory_dim"]):
                    raise RuntimeError("Formal report incomplete")
                allowed = marker.get("public_files", [])
                if not allowed or any(Path(n).name != n or Path(n).suffix not in (".json", ".png", ".npz") for n in allowed):
                    raise RuntimeError("Invalid public report whitelist")
                receipt = dict(status="complete", path=arm["report"],
                               files={n: sha(Path(arm["report"]) / n) for n in allowed})
                job["completed_arms"].append(arm["name"])
            job.setdefault("receipts", {})[name] = receipt
            save(root / "job.json", job)
        job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, requires_user_action=True,
                   error=f"{type(error).__name__}: {error}", ended_at=now())
        raise
    finally:
        save(root / "job.json", job)


def launch(root):
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    env = native_cuda_environment()
    if socket.gethostname() != HOST or job["status"] != "prepared":
        raise RuntimeError("Wrong host or not prepared; no restart")
    unchanged(plan)
    # Exclusive publication prevents two launch requests from starting two parents.
    with (root / "launch.json").open("x") as receipt, (root / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_ising_kb_highdim",
                                   "--execute", "--output", str(root)], cwd=plan["code_root"],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=dict(env, PYTHONUNBUFFERED="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
                     plan_sha256=sha(root / "plan.json"), source_sha256=plan["source_sha256"],
                     fresh_experiment=True, existing_B0_dimension100_resumed=False)
        receipt.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-device", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--audit", type=Path)
    args = p.parse_args()
    if args.verify_device:
        print(json.dumps(verify_cuda_device(args.output), ensure_ascii=False))
    elif args.prepare:
        if args.cache is None or args.audit is None:
            p.error("Preparation needs the existing physical cache and audit")
        plan = prepare(args.output, args.cache, args.audit)
        print(json.dumps(dict(status="prepared", output=plan["output"], dimensions=list(DIMENSIONS),
                              source_files=len(plan["source_sha256"]), data_sha256=DATA_SHA), ensure_ascii=False))
    elif args.launch:
        print(json.dumps(launch(args.output), ensure_ascii=False))
    else:
        execute(args.output)


if __name__ == "__main__":
    main()
