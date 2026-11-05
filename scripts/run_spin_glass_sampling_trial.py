"""Bounded, sampling-only PT / Houdayer-PT comparison; never starts QCR training.

Every case is a new independently seeded run. Historical data and code are not
modified, and the formal run uses its own source snapshot. Budget checks happen
between cases, not by killing a running Markov chain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import numpy as np


REFERENCE = "results/spin_glass2d_overlap_g1_seed91_vm225_20261019_monitor/v1/physical_data/manifest.json"
MAX_CPU_SECONDS = 3600
MAX_OUTPUT_BYTES = 1024**3
MASTER_SEED = 2026101902
HISTORICAL_SAMPLING_SECONDS = 97.15057079680264
HISTORICAL_SITE_VISITS = 1_433_272_320


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
        allow_nan=False, default=lambda value: value.tolist() if isinstance(
            value, (np.ndarray, np.generic)) else str(value)) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def build_plan(reference, smoke=False):
    """Predeclare geometry, identities, work, and comparisons, before sampling."""
    reference = Path(reference).resolve()
    old = json.loads(reference.read_text())
    if old.get("status") != "complete" or old["options"]["shape"] != [36, 36]:
        raise ValueError("Require the completed historical L36 sampling manifest")
    shape = [6, 6] if smoke else [36, 36]
    temperatures = [0.5, 1., 2., 3.5] if smoke else old["options"]["temperatures"]
    replicas, samples = (4, 16) if smoke else (4, 512)
    burn_in, spacing = (16, 2) if smoke else (8192, 16)
    disorder_ids = [0, 1, 7]
    cases, case_seeds = [], set()
    reference_rows = {row["disorder_id"]: row for row in old["shards"]}
    for disorder in disorder_ids:
        # Fixed, alternating algorithm/start order; no performance-based choice.
        order = [("pt", "hot"), ("houdayer_pt", "cold"),
                 ("houdayer_pt", "hot"), ("pt", "cold")]
        if disorder % 2:
            order = list(reversed(order))
        for algorithm, initial in order:
            key = f"j{disorder:03d}_{algorithm}_{initial}"
            digest = hashlib.sha256(f"EA-Houdayer-trial-v1:{MASTER_SEED}:{key}:{shape}".encode()).digest()
            seed = int.from_bytes(digest[:4], "little")
            if seed in case_seeds:
                raise ValueError("Case seed collision; no automatic reseeding")
            case_seeds.add(seed)
            cases.append(dict(key=key, algorithm=algorithm, initial=initial,
                disorder_id=disorder, sampling_seed=seed,
                bond_seed=reference_rows[disorder]["bond_seed"],
                expected_bond_sha256=None if smoke else reference_rows[disorder]["bond_sha256"]))
    visits = len(cases) * replicas * len(temperatures) * int(np.prod(shape)) * (burn_in + samples * spacing)
    projected_arrays = len(cases) * replicas * len(temperatures) * samples * (int(np.prod(shape)) + 8)
    if projected_arrays * 1.1 + 20 * 1024**2 > MAX_OUTPUT_BYTES:
        raise ValueError("Preflight output projection exceeds budget")
    return dict(schema_version=1, experiment="2D EA sampling-only PT versus Houdayer+PT",
        execution_smoke=smoke, scientific_result=not smoke,
        shape=shape, temperatures=temperatures, replicas=replicas,
        samples=samples, burn_in=burn_in, spacing=spacing, exchange_interval=1,
        cluster_interval=1, trace_interval=16 if not smoke else 2,
        distribution="bimodal", bond_scale=1., positive_probability=.5, field=0.,
        master_seed=MASTER_SEED, cases=cases, disorder_ids=disorder_ids,
        selection="Convenience/stress subset: old IDs 0,1,7; ID7 had weak prior low-T ESS. Not a representative disorder average.",
        reference_manifest_sha256=sha(reference), reference_manifest_name=str(reference),
        old_run_comparison="Historical short run is context only; fresh PT controls match new length, replica count, initializations, and kernel schedule.",
        total_single_spin_visits=visits, projected_raw_array_bytes=projected_arrays,
        total_houdayer_pair_events_upper_bound=sum(c["algorithm"] == "houdayer_pt" for c in cases)
            * len(temperatures) * (replicas // 2) * (burn_in + samples * spacing),
        budgets=dict(cpu_core_seconds=MAX_CPU_SECONDS, output_bytes=MAX_OUTPUT_BYTES,
            memory_bytes=1024**3, gpu_seconds=0, threads=1, nice=10,
            memory_projection_basis="Pre-formal smoke_v1 compiler/import peak555512KiB exceeded512MiB;1GiB soft RSS bound includes runtime/compiler and one streamed case; no sampling-scale change",
            initial_projected_cpu_seconds=visits / HISTORICAL_SITE_VISITS * HISTORICAL_SAMPLING_SECONDS * 2. + 120,
            initial_projection_basis="Historical L36 sampling wall97.15057079680264s/1433272320 visits, doubled for new cluster/trace overhead, plus120s process/report reserve; estimate not guarantee",
            enforcement="Admission and between-case soft CPU/output gates, not a hard kill deadline; no automatic extension, resume, or reduction"),
        comparison_protocol=dict(primary_observables=["energy_density", "pair_mean_q_squared"],
            aggregate_pairs_before_time_analysis=True, replica_pairs_independent=False,
            starts="Independent random and all-+1 ordered groups for each algorithm and J; 'cold' is an API name, not a low-energy/ground-state claim",
            statistical_claim="Descriptive finite-chain evidence only; no automatic equilibrium certification or acceleration claim",
            effective_sample_rate="ESS of group-level q^2 time series divided by whole sampling wall time including burn-in; JIT reported separately",
            round_trips="Temperature-label transport only; under cluster exchange the labels do not follow intact microscopic configurations"),
        training_updates=0, model_inference_calls=0, equilibrium_certified=False)


def cpu_seconds():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def resource_snapshot():
    memory, io = None, None
    if Path("/proc/meminfo").exists():
        rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory = int(rows["MemAvailable"].split()[0]) * 1024
    if Path("/proc/pressure/io").exists():
        full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
        io = float(next(item.split("=")[1] for item in full.split() if item.startswith("avg10=")))
    result = dict(at=now(), hostname=socket.gethostname(), cpu_count=os.cpu_count(),
        load1=os.getloadavg()[0], mem_available_bytes=memory, io_full_avg10=io)
    result["ready"] = (result["load1"] <= 1.25 * result["cpu_count"]
        and (memory is None or memory >= 4 * 1024**3) and (io is None or io < 20.))
    return result


def validate_runtime():
    required = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
    required["CUDA_VISIBLE_DEVICES"] = ""
    observed = {name: os.environ.get(name) for name in required}
    if observed != required:
        raise RuntimeError(f"Launch requires single CPU thread and disabled CUDA: {required}; observed {observed}")
    return observed


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def freeze_code(source, target):
    target.mkdir()
    files = {}
    for package in ("recurrent_rg", "scripts"):
        for original in sorted((source / package).rglob("*.py")):
            relative = original.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            files[str(relative)] = sha(destination)
    shutil.copy2(source / "AGENTS.md", target / "AGENTS.md")
    files["AGENTS.md"] = sha(target / "AGENTS.md")
    write_json(target / "fingerprints.json", files)
    return files


def verify_code(code, fingerprints):
    bad = [relative for relative, expected in fingerprints.items() if sha(code / relative) != expected]
    if bad:
        raise RuntimeError(f"Frozen source fingerprint mismatch: {bad}; do not repair or restart")


def run_case(root, key):
    from recurrent_rg.data.spin_glass import energy, generate_bonds
    from recurrent_rg.data.spin_glass_houdayer import sample_houdayer_parallel_tempering
    from recurrent_rg.data.spin_glass_pipeline import _bond_sha

    runtime_environment = validate_runtime()
    plan = json.loads((root / "plan.json").read_text())
    case = next(row for row in plan["cases"] if row["key"] == key)
    receipt_path = root / "cases" / f"{key}.json"
    raw_path = root / "cases" / f"{key}.npz"
    if receipt_path.exists() or raw_path.exists():
        raise FileExistsError("Case already has output; no resume or overwrite")
    fingerprints = json.loads((root / "code" / "fingerprints.json").read_text())
    verify_code(root / "code", fingerprints)
    bonds = generate_bonds(plan["shape"], case["bond_seed"], distribution="bimodal", scale=1., positive_probability=.5)
    bond_sha = _bond_sha(bonds)
    if case["expected_bond_sha256"] is not None and bond_sha != case["expected_bond_sha256"]:
        raise RuntimeError("Historical bond fingerprint differs; no sampling started")
    # Tiny execution-only warmup compiles both branches once per new process.
    # It never contributes a scientific sample or a diagnostic point.
    warm_started, warm_cpu = time.monotonic(), cpu_seconds()
    tiny_bonds = np.ones((2, 3, 3), np.float64)
    sample_houdayer_parallel_tempering(tiny_bonds, [.8, 2.], samples=1, burn_in=1,
        spacing=1, seed=123, replicas=4, algorithm=case["algorithm"],
        initial="hot", exchange_interval=1, cluster_interval=1, trace_interval=1)
    warm_seconds, warm_cpu_seconds = time.monotonic() - warm_started, cpu_seconds() - warm_cpu
    started, start_cpu = time.monotonic(), cpu_seconds()
    print(json.dumps(dict(event="case_started", key=key, at=now(), pid=os.getpid()), ensure_ascii=False), flush=True)
    result = sample_houdayer_parallel_tempering(bonds, plan["temperatures"],
        samples=plan["samples"], burn_in=plan["burn_in"], spacing=plan["spacing"],
        seed=case["sampling_seed"], replicas=plan["replicas"], algorithm=case["algorithm"],
        initial=case["initial"], field=plan["field"], exchange_interval=plan["exchange_interval"],
        cluster_interval=plan["cluster_interval"], trace_interval=plan["trace_interval"])
    elapsed, used_cpu = time.monotonic() - started, cpu_seconds() - start_cpu
    for diagnostic in ("max_cluster_joint_energy_residual", "max_energy_tracking_drift"):
        if not np.isfinite(result[diagnostic]) or result[diagnostic] > 1e-8:
            raise RuntimeError(f"Kernel energy audit failed: {diagnostic}={result[diagnostic]}")
    # Validate saved energies independently of incremental kernel bookkeeping.
    maximum_energy_error = 0.
    for t in range(len(plan["temperatures"])):
        recomputed = energy(result["configurations"][t], bonds, field=plan["field"])
        maximum_energy_error = max(maximum_energy_error, float(np.max(np.abs(recomputed - result["energies"][t]))))
    if maximum_energy_error > 1e-8:
        raise RuntimeError(f"Saved energy mismatch: {maximum_energy_error}")
    array_payload = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    array_payload["bonds"] = bonds
    with raw_path.open("xb") as stream:
        np.savez(stream, **array_payload)
    receipt = dict(**case, status="complete", returncode=0, npz_path=str(raw_path.relative_to(root)),
        npz_sha256=sha(raw_path), npz_bytes=raw_path.stat().st_size,
        bond_sha256=bond_sha, replicas=plan["replicas"], samples=plan["samples"],
        burn_in=plan["burn_in"], spacing=plan["spacing"],
        sampling_wall_seconds=elapsed, sampling_cpu_seconds=used_cpu,
        peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_units="KiB on Linux; bytes on macOS",
        jit_execution_check_wall_seconds=warm_seconds, jit_execution_check_cpu_seconds=warm_cpu_seconds,
        saved_energy_max_absolute_error=maximum_energy_error,
        trace_aggregation="Energy mean across replicas; q_squared mean across all unordered replica pairs at each trace time; pairs are not independent",
        runtime_environment=runtime_environment,
        actual_nlwp=len(list(Path("/proc/self/task").iterdir())) if Path("/proc/self/task").exists() else None,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
        diagnostics={name: value for name, value in result.items()
                     if not isinstance(value, np.ndarray) and name not in ("shape",)},
        source_unchanged=True, training_updates=0, equilibrium_certified=False)
    verify_code(root / "code", fingerprints)
    write_json(receipt_path, receipt)
    print(json.dumps(dict(event="case_complete", key=key, seconds=elapsed), ensure_ascii=False), flush=True)
    return receipt


def run_trial(root, reference, smoke=False):
    runtime_environment = validate_runtime()
    plan = build_plan(reference, smoke)
    if not smoke and socket.gethostname() != "thinklab-105-225":
        raise RuntimeError("Formal sampling trial is pinned to thinklab-105-225")
    if plan["budgets"]["initial_projected_cpu_seconds"] > MAX_CPU_SECONDS:
        raise RuntimeError("Initial conservative CPU projection exceeds budget; no output created")
    source = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir()
    (root / "logs").mkdir()
    fingerprints = freeze_code(source, root / "code")
    write_json(root / "plan.json", plan)
    shutil.copy2(reference, root / "reference_manifest.json")
    job = dict(status="running", workflow_complete=False, started_at=now(),
        pid=os.getpid(), hostname=socket.gethostname(), stages=[], resource_checks=[],
        training_updates=0, model_inference_calls=0, equilibrium_certified=False,
        requires_user_action=False, error=None, runtime_environment=runtime_environment,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
        launch_source_sha256=sha(Path(__file__)),
        frozen_runner_sha256=fingerprints["scripts/run_spin_glass_sampling_trial.py"])
    write_json(root / "job.json", job)
    initial_cpu, started = cpu_seconds(), time.monotonic()
    try:
        for index, case in enumerate(plan["cases"]):
            verify_code(root / "code", fingerprints)
            check = resource_snapshot()
            job["resource_checks"].append(check)
            used = cpu_seconds() - initial_cpu
            completed_cpu = [row["case_cpu_seconds"] for row in job["stages"] if "case_cpu_seconds" in row]
            projected = (used + max(completed_cpu) * (len(plan["cases"]) - index) * 1.5 + 60
                if completed_cpu else plan["budgets"]["initial_projected_cpu_seconds"])
            if not check["ready"] or used > MAX_CPU_SECONDS or projected > MAX_CPU_SECONDS or directory_bytes(root) > MAX_OUTPUT_BYTES:
                job.update(status="budget_hold" if check["ready"] else "resource_hold", current=None,
                    requires_user_action=True, projected_cpu_seconds=projected,
                    error="Between-case resource/budget gate; no further cases launched, no automatic resume")
                break
            stage = dict(name=case["key"], status="running", started_at=now())
            job["stages"].append(stage)
            job["current"] = case["key"]
            write_json(root / "job.json", job)
            command = [sys.executable, "-m", "scripts.run_spin_glass_sampling_trial",
                "--output", str(root), "--case-key", case["key"]]
            with (root / "logs" / f"{case['key']}.log").open("x") as stream:
                before_child = cpu_seconds()
                child = subprocess.Popen(command, cwd=root / "code", stdout=stream, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                write_json(root / "job.json", job)
                returncode = child.wait()
            stage.update(returncode=returncode, finished_at=now(), status="complete" if returncode == 0 else "failed")
            if returncode:
                raise RuntimeError(f"Sampling case {case['key']} returned {returncode}; no automatic restart")
            receipt = json.loads((root / "cases" / f"{case['key']}.json").read_text())
            stage["case_cpu_seconds"] = cpu_seconds() - before_child
            peak_bytes = receipt["peak_rss_native_units"] * (1024 if sys.platform.startswith("linux") else 1)
            if peak_bytes > plan["budgets"]["memory_bytes"]:
                job.update(status="budget_hold", current=None, requires_user_action=True,
                    error="Measured case RSS exceeded soft memory budget; samples preserved, no further cases")
                break
            job["cpu_core_seconds"] = cpu_seconds() - initial_cpu
            job["elapsed_seconds"] = time.monotonic() - started
            write_json(root / "job.json", job)
        else:
            if cpu_seconds() - initial_cpu + 60 > MAX_CPU_SECONDS or directory_bytes(root) > MAX_OUTPUT_BYTES:
                job.update(status="budget_hold", current=None, requires_user_action=True,
                    error="Final admission budget gate; samples preserved; no automatic report or continuation")
                return job
            verify_code(root / "code", fingerprints)
            job["current"] = "report"
            write_json(root / "job.json", job)
            with (root / "logs" / "report.log").open("x") as stream:
                returncode = subprocess.call([sys.executable, "-m", "scripts.report_spin_glass_sampling_trial",
                    "--root", str(root)], cwd=root / "code", stdout=stream, stderr=subprocess.STDOUT)
            job["stages"].append(dict(name="report", status="complete" if returncode == 0 else "failed", returncode=returncode))
            if returncode:
                raise RuntimeError(f"Report returned {returncode}; preserve all samples, no automatic regeneration")
            verify_code(root / "code", fingerprints)
            max_child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1024 if sys.platform.startswith("linux") else 1)
            job["measured_max_child_rss_bytes"] = max_child_rss
            if (cpu_seconds() - initial_cpu > MAX_CPU_SECONDS or directory_bytes(root) > MAX_OUTPUT_BYTES
                    or max_child_rss > plan["budgets"]["memory_bytes"]):
                job.update(status="budget_hold", workflow_complete=False, current=None, requires_user_action=True,
                    error="Post-report CPU/output/RSS budget exceeded; completed artifacts preserved")
            else:
                job.update(status="complete", workflow_complete=True, current=None, finished_at=now())
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
        raise
    finally:
        job.update(cpu_core_seconds=cpu_seconds() - initial_cpu, elapsed_seconds=time.monotonic() - started,
            directory_bytes=directory_bytes(root))
        write_json(root / "job.json", job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(REFERENCE))
    parser.add_argument("--execution-smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--case-key")
    args = parser.parse_args()
    if args.case_key:
        run_case(args.output.resolve(), args.case_key)
    elif args.plan_only:
        print(json.dumps(build_plan(args.reference, args.execution_smoke), indent=2, ensure_ascii=False))
    else:
        run_trial(args.output.resolve(), args.reference.resolve(), args.execution_smoke)


if __name__ == "__main__":
    main()
