"""Conditional fixed-C dimension diagnostic after the four-arm closure test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts.run_phi4_scalar_r_ablation import HOST, now, resources, save, sha


BASE = Path("/mnt/nas-new/home/yange/ham")
SOURCE = BASE / "results/phi4_u1_rg_reference_b3_g1_seed91_vm225_20261028"
SCALAR_CACHE = BASE / "results/phi4_u1_scalar_R_validation_seed91_vm225_20261030/cache"
FIRST_ROUND = BASE / "results/phi4_u1_scalar_R4_cv_seeds91_93_vm225_20261030"
FIRST_Q_PAIR = BASE / "results/phi4_u1_fixed_C_dimensions_qseed91_vm225_20261030"
CPU_BUDGET = 3000.
OUTPUT_BUDGET = 512 * 1024**2


def completed(root):
    job = json.loads((root / "job.json").read_text())
    inventory = json.loads((root / "public_files.json").read_text())
    if (job.get("status") != "complete" or job.get("workflow_complete") is not True
            or job.get("returncode") != 0 or job.get("error") is not None
            or any(stage.get("returncode") != 0 for stage in job["stages"])):
        raise ValueError("Prior authorized diagnostic must complete successfully")
    pins = {str(root / "public_files.json"): sha(root / "public_files.json")}
    for name in ("job.json", "plan.json", "training/summary.json", "report/analysis.json"):
        path = root / name
        if sha(path) != inventory[name]:
            raise ValueError("Prior diagnostic public SHA mismatch")
        pins[str(path)] = inventory[name]
    return json.loads((root / "training/summary.json").read_text()), pins


def representation_gate():
    summary, pins = completed(FIRST_ROUND)
    scores, evidence = summary["closure_scores"], {}
    for comparator in ("A_shared96", "C_shared136"):
        ratios = []
        for fold in range(4):
            for seed in (91, 92, 93):
                prefix = f"fold{fold}_"
                first = scores[f"{prefix}B_separate96_seed{seed}_selected"]["validation"]["chain_mean_loss"]
                second = scores[f"{prefix}{comparator}_seed{seed}_selected"]["validation"]["chain_mean_loss"]
                if second <= 0:
                    raise ValueError("Cannot establish ratio against a zero-error comparator")
                ratios.append(first / second)
        ordered = sorted(ratios)
        median = .5 * (ordered[5] + ordered[6])
        improved = sum(value < 1 for value in ratios)
        evidence[comparator] = dict(ratios=ratios, median_ratio=median, improved_pairs=improved,
                                   passed=median <= .8 and improved >= 9)
    if not all(value["passed"] for value in evidence.values()):
        raise ValueError("Predeclared representation follow-up gate was not met")
    return evidence, pins


def replication_gate():
    summary, pins = completed(FIRST_Q_PAIR)
    ratios = []
    for seed in (91, 92, 93):
        losses = []
        for dimension in (1, 2):
            rows = summary["feature_metrics"][f"dg{dimension}_seed{seed}_selected"]["validation"]
            losses.append(sum(weight * row["prediction"]["chain_mean_rmse"]**2
                              for weight, row in zip((5., 2.5, 2.5), rows)))
        if losses[0] <= 0:
            raise ValueError("Cannot establish common-feature improvement over zero error")
        ratios.append(losses[1] / losses[0])
    median = sorted(ratios)[1]
    evidence = dict(ratios=ratios, median_ratio=median, improved_R_seed_pairs=sum(x < 1 for x in ratios),
        passed=median <= .8 and sum(x < 1 for x in ratios) >= 2,
        qualification="预先固定的共同特征验证误差筛选门槛；不是统计显著性认证")
    if not evidence["passed"]:
        raise ValueError("Paired-Q replication gate was not met")
    return evidence, pins


def run(output, q_seed, features=None):
    if socket.gethostname() != HOST or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use the verified225 host with GPU visibility disabled")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10 or q_seed not in (91, 92, 93):
        raise ValueError("Require exactly nice10 and a predeclared Q seed")
    evidence, input_pins = representation_gate()
    replication = None
    if q_seed != 91:
        replication, extra = replication_gate()
        input_pins.update(extra)
    root, code = Path(output).resolve(), Path(__file__).resolve().parents[1]
    feature_root = Path(features).resolve() if features else root / "features"
    if (root == code or root in code.parents or root == SOURCE or SOURCE in root.parents
            or root in SOURCE.parents or (features and (root == feature_root or root in feature_root.parents
                                                       or feature_root in root.parents))):
        raise ValueError("New output must be separate from immutable sources and external features")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    code_pins = {str(p): sha(p) for folder in ("experiments", "recurrent_rg", "scripts")
                 for p in (code / folder).rglob("*.py")}
    commands = []
    if not features:
        commands.append(("paired_features", "features", [sys.executable, "-u", "-m",
            "experiments.phi4_fixed_c_features", "--source", str(SOURCE), "--scalar-cache", str(SCALAR_CACHE),
            "--output", str(feature_root), "--cpu-budget-seconds", "600"]))
    else:
        from experiments.phi4_fixed_c_dimension import verify_features
        _, feature_pins = verify_features(feature_root)
        input_pins.update(feature_pins)
    commands.extend([
        ("paired_Q_dimensions", "training", [sys.executable, "-u", "-m", "experiments.phi4_fixed_c_dimension",
             "--features", str(feature_root), "--output", str(root / "training"), "--q-seed", str(q_seed),
             "--cpu-budget-seconds", "1200"]),
        ("report", "report", [sys.executable, "-u", "-m", "scripts.report_phi4_fixed_c_dimension",
             "--training", str(root / "training"), "--features", str(feature_root), "--output", str(root / "report")])])
    plan = dict(schema_version=5, created_at=now(), source=str(SOURCE), output=str(root),
        code_root=str(code), features=str(feature_root), q_seed=q_seed,
        source_sha256=code_pins, prior_input_sha256=input_pins,
        stages=[dict(name=name, command=cmd) for name, _, cmd in commands],
        representation_gate=evidence, replication_gate=replication,
        design=dict(theory_dims=[1, 2], Q_steps_each=3000, R_seeds=[91, 92, 93],
            R_Adam_steps_each=6000, R_LBFGS_max_iterations=1000,
            Q_objective_weights=[10., 20., 5.], closure_weights=[5., 2.5, 2.5],
            frozen_C=True, C_updates=0, paired_Q_initialization_and_batches=True,
            feature_encoding_repeats=4, new_physical_sampling=False,
            explicit_dimension_ablation=True, exact_historical_online_reproduction=False,
            labels_used_by_optimizer=False, TEST_used_for_selection=False,
            primary_comparison="common105-feature decoded prediction and Q reconstruction errors",
            replication_gate="validation common-feature weighted MSE dg2/dg1 median across3 R seeds<=0.8 and >=2/3 pairs improved",
            gate_qualification="工程筛选门槛；不认证物理闭合或显著性"),
        resources=dict(device="cpu", CPU_threads=1, nice=10, GPU_seconds=0,
            CPU_budget_seconds=CPU_BUDGET, output_budget_bytes=OUTPUT_BUDGET))
    save(root / "plan.json", plan)
    started = time.process_time()
    def cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() - started + child.ru_utime + child.ru_stime
    job = dict(status="prepared", workflow_complete=False, current=None, returncode=None, error=None,
        requires_user_action=False, parent_pid=os.getpid(), started_at=now(), stages=[],
        plan_sha256=sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.)
    save(root / "job.json", job)
    try:
        for name, folder, command in commands:
            job.update(status="waiting_resources", current=name)
            while True:
                row = resources()
                job.update(resources=row, cpu_seconds=cpu())
                save(root / "job.json", job)
                if (row["mem_available_bytes"] >= 36 * 1024**3 and row["load1"] <= 1.25 * row["cpus"]
                        and row["io_full_avg10"] < 20):
                    break
                time.sleep(30)
            if any(sha(path) != digest for path, digest in (code_pins | input_pins).items()):
                raise RuntimeError("Frozen code or prerequisite diagnostic changed")
            if cpu() > CPU_BUDGET - 450:
                job.update(status="budget_hold", returncode=3, requires_user_action=True,
                           error="Insufficient declared CPU budget before next stage")
                save(root / "job.json", job)
                return 3
            stage = dict(name=name, command=command, status="running", started_at=now(), returncode=None)
            job["stages"].append(stage)
            with (root / "logs" / f"{name}.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                job.update(status="running")
                save(root / "job.json", job)
                rc = child.wait()
            stage.update(status="complete" if rc == 0 else "failed", returncode=rc, ended_at=now())
            job.update(cpu_seconds=cpu())
            save(root / "job.json", job)
            if rc:
                path = root / folder / "status.json"
                state = json.loads(path.read_text()) if path.exists() else {}
                job.update(status=state.get("status", "failed"), returncode=rc,
                    error=state.get("error") or f"{name} exited {rc}", requires_user_action=True)
                save(root / "job.json", job)
                return rc
        if (cpu() > CPU_BUDGET or sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > OUTPUT_BUDGET
                or any(sha(path) != digest for path, digest in (code_pins | input_pins).items())):
            raise RuntimeError("Final CPU/output budget or immutable input validation failed")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
                   ended_at=now(), cpu_seconds=cpu())
        save(root / "job.json", job)
        public = {}
        for folder in (("features",) if not features else ()) + ("training", "report"):
            inventory = json.loads((root / folder / "public_files.json").read_text())
            for name, expected in inventory.items():
                path = (root / folder / name).resolve()
                if path.parent != root / folder or path.suffix not in (".json", ".npz", ".png") or sha(path) != expected:
                    raise RuntimeError("Invalid public diagnostic artifact")
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
    parser.add_argument("--q-seed", type=int, choices=(91, 92, 93), default=91)
    parser.add_argument("--features")
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.q_seed, args.features))


if __name__ == "__main__":
    main()
