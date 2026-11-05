"""Run the authorized four-arm, four-fold frozen-QC scalar-R diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time


HOST = "thinklab-105-225"
CACHE = Path("/mnt/nas-new/home/yange/ham/results/phi4_u1_scalar_R_validation_seed91_vm225_20261030/cache")
CPU_BUDGET = 5400.0
OUTPUT_BUDGET = 768 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resources():
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    return dict(at=now(), mem_available_bytes=available, io_full_avg10=io,
                load1=os.getloadavg()[0], cpus=os.cpu_count())


def run(output):
    if socket.gethostname() != HOST:
        raise ValueError("The diagnostic is pinned to the verified existing225 host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU diagnostic requires an empty CUDA_VISIBLE_DEVICES")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Launch through nice10 once; children inherit priority")
    root, code = Path(output).resolve(), Path(__file__).resolve().parents[1]
    if root == code or root in code.parents or root == CACHE or root in CACHE.parents or CACHE in root.parents:
        raise ValueError("Output must be separate from source and immutable cache")
    cache_status = json.loads((CACHE / "status.json").read_text())
    if (cache_status.get("status") != "complete" or cache_status.get("workflow_complete") is not True
            or cache_status.get("returncode") != 0):
        raise ValueError("Completed source cache required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    pins = {str(p): sha(p) for directory in ("experiments", "recurrent_rg", "scripts")
            for p in (code / directory).rglob("*.py")}
    cache_pins = {str(CACHE / name): sha(CACHE / name)
                  for name in ("cache.npz", "manifest.json", "receipt.json", "status.json", "public_files.json")}
    commands = [
        ("four_arm_condition_CV", [sys.executable, "-u", "-m", "experiments.phi4_scalar_r_ablation",
             "--cache", str(CACHE), "--output", str(root / "training"),
             "--cpu-budget-seconds", "3600"]),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_scalar_r_ablation",
             "--training", str(root / "training"), "--cache", str(CACHE),
             "--output", str(root / "report")]),
    ]
    plan = dict(schema_version=5, created_at=now(), code_root=str(code), cache=str(CACHE),
        output=str(root), source_sha256=pins, cache_sha256=cache_pins,
        stages=[dict(name=name, command=command) for name, command in commands],
        design=dict(arms={"A_shared96": dict(heads=1, hidden=96),
                          "B_separate96": dict(heads=2, hidden=96),
                          "C_shared136": dict(heads=1, hidden=136),
                          "D_shared16": dict(heads=1, hidden=16)},
            R_seeds=[91, 92, 93], folds=4, independent_R_fits=48,
            split_seed=20261030, fit_conditions_per_fold=6, validation_conditions_per_fold=2,
            Adam_updates_per_fit=6000, Adam_lr=.001, LBFGS_max_iterations=1000,
            LBFGS_block_iterations=100, patience_blocks=3, relative_improvement=.001,
            closure_objective="all_start_anchor", closure_weights=[5., 2.5, 2.5],
            scale_separated_composition="R1(R0(g0))", common_fit_only_normalizer=True,
            Q_updates=0, C_updates=0, new_physical_sampling=False, extra_QC_inference=False,
            labels_used_by_optimizer=False, TEST_used_for_selection=False,
            TEST_has_previously_been_inspected=True, QC_has_seen_R_validation_conditions=True,
            original_sampling_warning_count=13, equilibrium_certified=False,
            condition_validation_is_endpoint_selection_not_blind_generalization=True,
            follow_up_gate=dict(primary="validation chain-mean weighted closure loss; no TEST or root-count gate",
                representation_probe="Both B/A and B/C median paired ratios <=0.8, and B better in >=9 of12 fold-seed pairs for each comparison",
                capacity_signal="D/A median paired ratio <=0.8 and D better in >=9 of12 fold-seed pairs",
                threshold_qualification="预先固定的工程筛选门槛，不是显著性或物理定理"),
            scope="第一轮诊断；后续表示或采样实验依据结果另建独立根执行"),
        resources=dict(device="cpu", GPU_seconds=0, CPU_threads=1, nice=10,
            CPU_budget_seconds=CPU_BUDGET, output_budget_bytes=OUTPUT_BUDGET,
            min_MemAvailable_GiB=36, max_load_per_CPU=1.25, max_IO_full_avg10=20))
    save(root / "plan.json", plan)
    started = time.process_time()
    def cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() - started + child.ru_utime + child.ru_stime
    job = dict(status="prepared", workflow_complete=False, returncode=None, error=None,
        requires_user_action=False, parent_pid=os.getpid(), started_at=now(), current=None,
        stages=[], plan_sha256=sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.)
    save(root / "job.json", job)
    try:
        for name, command in commands:
            job.update(status="waiting_resources", current=name)
            while True:
                row = resources()
                job.update(resources=row, cpu_seconds=cpu())
                save(root / "job.json", job)
                if (row["mem_available_bytes"] >= 36 * 1024**3
                        and row["load1"] <= 1.25 * row["cpus"] and row["io_full_avg10"] < 20):
                    break
                time.sleep(30)
            if any(sha(path) != expected for path, expected in {**pins, **cache_pins}.items()):
                raise RuntimeError("Frozen deployment or cache fingerprint changed")
            if cpu() > CPU_BUDGET - 450:
                job.update(status="budget_hold", requires_user_action=True, returncode=3,
                           error="Insufficient remaining declared CPU budget")
                save(root / "job.json", job)
                return 3
            stage = dict(name=name, status="running", started_at=now(), command=command, returncode=None)
            job["stages"].append(stage)
            with (root / "logs" / (name + ".log")).open("xb") as log:
                process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = process.pid
                job.update(status="running")
                save(root / "job.json", job)
                rc = process.wait()
            stage.update(status="complete" if rc == 0 else "failed", ended_at=now(), returncode=rc)
            job.update(cpu_seconds=cpu())
            save(root / "job.json", job)
            if rc:
                detail = root / ("training" if name == "four_arm_condition_CV" else "report") / "status.json"
                child = json.loads(detail.read_text()) if detail.exists() else {}
                job.update(status=child.get("status", "failed"), returncode=rc,
                    error=child.get("error") or f"{name} exited {rc}", requires_user_action=True)
                save(root / "job.json", job)
                return rc
        if (cpu() > CPU_BUDGET
                or sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > OUTPUT_BUDGET):
            raise RuntimeError("Completed diagnostic exceeded its declared resource budget")
        if any(sha(path) != expected for path, expected in {**pins, **cache_pins}.items()):
            raise RuntimeError("Frozen deployment or cache changed during diagnostic")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
                   ended_at=now(), cpu_seconds=cpu())
        save(root / "job.json", job)
        public = {}
        for folder in ("training", "report"):
            inventory = json.loads((root / folder / "public_files.json").read_text())
            for name, expected in inventory.items():
                path = (root / folder / name).resolve()
                if (path.parent != root / folder or path.suffix not in (".json", ".npz", ".png")
                        or sha(path) != expected):
                    raise RuntimeError("Invalid public artifact inventory")
                public[f"{folder}/{name}"] = expected
        public.update({name: sha(root / name) for name in ("job.json", "plan.json")})
        save(root / "public_files.json", public)
        print(json.dumps(job, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        job.update(status="failed", workflow_complete=False, returncode=1, error=str(exc),
                   requires_user_action=True, ended_at=now(), cpu_seconds=cpu())
        save(root / "job.json", job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))


if __name__ == "__main__":
    main()
