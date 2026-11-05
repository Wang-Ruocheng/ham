"""固定细积分的 L=108 热／冷启动预检；默认仅打印计划，不生成生产数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


SCHEMA = "phi4_ru_fine_sampler_pilot_v1"
SEED_NAMESPACE = 2026103004
CPU_CAP_SECONDS = 1200.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BANDS = ((.5, -.7625, -.63125), (.75, -1.025, -.89375),
         (1., -1.2875, -1.15625))
INITIALS = ("hot", "cold_positive", "cold_negative")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/phi4_ru_fine_sampler_pilot.py", "recurrent_rg/data/phi4.py",
             "recurrent_rg/data/phi4_hybrid.py", "recurrent_rg/data/spin_glass_pipeline.py")
    return {name: sha(root / name) for name in names}


def seed_for(condition_index, initial):
    if initial not in INITIALS or not 0 <= condition_index < 6:
        raise ValueError("Unknown fixed pilot condition or initialization")
    payload = json.dumps([SCHEMA, SEED_NAMESPACE, int(condition_index), initial])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "little")


def build_plan(output=None):
    """No writes, sampling, model loads, random draws or NumPy import."""
    conditions = []
    for u, low, high in BANDS:
        for side, r in (("left", low), ("right", high)):
            index = len(conditions)
            conditions.append(dict(condition_index=index, r=r, u=u, h=0., side=side,
                seeds={initial: seed_for(index, initial) for initial in INITIALS}))
    cycles, count = 1024 + 32 * 4, len(conditions) * len(INITIALS)
    return dict(schema_version=5, schema=SCHEMA,
        output=None if output is None else str(Path(output).resolve()),
        seed_namespace=SEED_NAMESPACE, seed_derivation="SHA256(schema,namespace,condition_index,initial), first 128 bits little-endian",
        size=108, dimension=2, samples=32, burn_in=1024, spacing=4,
        step_size=.04, leapfrog_steps=24, trajectory_length=.96, cluster_probability=.5,
        initial_states=list(INITIALS), conditions=conditions, independent_pilot_chains=count,
        cycles_per_chain=cycles, total_hmc_trajectories=count * cycles,
        total_force_site_evaluations=count * cycles * 25 * 108**2,
        maximum_cluster_neighbor_visits=count * cycles * 4 * 108**2,
        cpu_cap_seconds=CPU_CAP_SECONDS, cpu_threads=1, gpu_used=False,
        cpu_budget_semantics="Outer process CPU timer includes kernel diagnostics; admission at chain boundaries; a running chain is never killed",
        gate=dict(minimum_production_hmc_acceptance=.6,
                  cold_minimum_saved_mean_field_squared_std=1e-6, finite_required=True,
                  ess_is_hard_gate=False, equilibrium_certified=False),
        source_pins=source_pins(), production_data_saved=False,
        raw_configuration_arrays_saved=False, automatic_retry=False, automatic_step_selection=False,
        training_updates=dict(Q=0, C=0, R=0),
        qualification="接受率与冷初态幅度变化只检查积分可用性，不认证热化；ESS及所有原始诊断警告保留。18条独立pilot不进入生产。")


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def finite_json(value):
    """Retain optional diagnostics as null while rejecting actual nonfinite values."""
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_json(item) for item in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def summarize_chain(result, condition, initial, *, cpu_seconds, wall_seconds):
    import numpy as np

    cycles, burn_in = 1152, 1024
    accepted = np.asarray(result["accepted"])
    delta = np.asarray(result["delta_h"], dtype=float)
    raw = np.asarray(result["raw_observables"], dtype=float)
    names = list(result["raw_observable_names"])
    if (accepted.dtype != np.bool_ or accepted.shape != (cycles,)
            or delta.shape != (cycles,) or raw.shape != (32, len(names))
            or "mean_field_squared" not in names):
        raise RuntimeError("Frozen kernel returned inconsistent pilot dimensions")
    if not (np.isfinite(delta).all() and np.isfinite(raw).all()
            and np.isfinite(np.asarray(result["configurations"])).all()
            and finite_json(result["diagnostics"])):
        raise FloatingPointError("Nonfinite pilot field, raw observable or diagnostic")
    expected_sites = cycles * 25 * 108**2
    if (result["hmc_trajectories"] != cycles or result["hybrid_cycles"] != cycles
            or result["force_evaluations"] != cycles * 25
            or result["force_site_evaluations"] != expected_sites
            or result["initial"] != initial
            or result["seed"] != condition["seeds"][initial]
            or result.get("cluster_probability") != .5):
        raise RuntimeError("Frozen kernel returned inconsistent work/seed/schedule receipt")
    rate = float(accepted[burn_in:].mean())
    burn_rate = float(accepted[:burn_in].mean())
    if (abs(rate - result["production_acceptance"]) > 1e-14
            or abs(burn_rate - result["burn_in_acceptance"]) > 1e-14):
        raise RuntimeError("HMC acceptance receipt disagrees with actual acceptances")
    metrics = {name: dict(mean=float(raw[:, i].mean()), std=float(raw[:, i].std(ddof=0)),
        minimum=float(raw[:, i].min()), maximum=float(raw[:, i].max()),
        diagnostics=result["diagnostics"][name]) for i, name in enumerate(names)}
    amplitude_std = metrics["mean_field_squared"]["std"]
    reasons = []
    if rate < .6:
        reasons.append("production_hmc_acceptance_below_0.6")
    if initial != "hot" and amplitude_std <= 1e-6:
        reasons.append("cold_saved_amplitude_std_not_above_1e-6")
    positions = np.flatnonzero(accepted)
    row = dict(condition_index=condition["condition_index"], r=condition["r"], u=condition["u"],
        h=0., size=108, initial=initial, seed=condition["seeds"][initial],
        production_acceptance=rate, burn_in_acceptance=burn_rate,
        hmc_accepted_total=int(accepted.sum()), hmc_accepted_burn_in=int(accepted[:burn_in].sum()),
        hmc_accepted_production=int(accepted[burn_in:].sum()),
        first_accepted_hmc_cycle=None if not len(positions) else int(positions[0] + 1),
        first_accepted_hmc_cycle_is_one_based=True,
        mean_field_squared_std=amplitude_std, max_absolute_delta_h=float(np.max(np.abs(delta))),
        delta_h=dict(mean=float(delta.mean()), minimum=float(delta.min()), maximum=float(delta.max())),
        raw_metrics=metrics, diagnostics=result["diagnostics"], warnings=list(result.get("warnings", [])),
        cpu_seconds=float(cpu_seconds), wall_seconds=float(wall_seconds),
        kernel_reported_cpu_seconds=float(result["cpu_seconds"]),
        kernel_reported_wall_seconds=float(result["wall_seconds"]),
        gate_passed=not reasons, gate_reasons=reasons, finite=True,
        equilibrium_certified=False)
    for key in ("hmc_trajectories", "hybrid_cycles", "force_evaluations", "force_site_evaluations",
                "wolff_cluster_updates", "total_cluster_sites", "total_cluster_changed_sites",
                "total_cluster_neighbor_visits", "total_cluster_bond_tests", "total_cluster_activated_bonds",
                "saved_magnetization_sign_changes"):
        row[key] = int(result[key])
    if not finite_json(row):
        raise FloatingPointError("Nonfinite outer pilot timing or summary")
    return row


def run(output):
    # Set all numerical runtimes before importing the frozen NumPy kernel.
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain

    root = Path(output).resolve()
    plan = build_plan(root)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", plan)
    before = plan["source_pins"]
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows, error, terminal = [], None, "complete"
    current = None
    try:
        for condition in plan["conditions"]:
            for initial in INITIALS:
                if time.process_time() - started_cpu >= plan["cpu_cap_seconds"]:
                    terminal, error = "budget_hold", "Pilot CPU cap reached at a chain boundary"
                    break
                current = dict(condition_index=condition["condition_index"], initial=initial)
                save(root / "status.json", dict(status="running", workflow_complete=False,
                    returncode=None, production_allowed=False, current=current,
                    completed_chains=len(rows), cpu_seconds=time.process_time() - started_cpu))
                wall, cpu = time.perf_counter(), time.process_time()
                result = sample_hybrid_chain(condition["r"], condition["u"], size=108, dimension=2,
                    samples=32, burn_in=1024, spacing=4, step_size=.04, leapfrog_steps=24,
                    seed=condition["seeds"][initial], initial=initial, h=0., cluster_probability=.5)
                row = summarize_chain(result, condition, initial,
                    cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
                rows.append(row)
                del result
                save(root / "progress.json", dict(chains=rows, completed_chains=len(rows),
                    cpu_seconds=time.process_time() - started_cpu, production_allowed=False))
                if time.process_time() - started_cpu > plan["cpu_cap_seconds"]:
                    terminal, error = "budget_hold", "Pilot CPU cap exceeded by the completed atomic chain"
                    break
            if terminal == "budget_hold":
                break
        if terminal == "complete" and not all(row["gate_passed"] for row in rows):
            terminal = "preflight_hold"
    except Exception as exc:
        terminal, error = "failed", f"{type(exc).__name__}: {exc}"
    after = source_pins()
    if before != after:
        terminal, error = "failed", "Frozen pilot/kernel source fingerprint changed"
    allowed = terminal == "complete" and len(rows) == plan["independent_pilot_chains"]
    code = 0 if allowed else 1 if terminal == "failed" else 3
    summary = dict(schema=SCHEMA, status=terminal, workflow_complete=terminal in ("complete", "preflight_hold"),
        returncode=code, error=error, current=current, production_allowed=allowed,
        stage_accepted=allowed, stageaccepted=allowed, chains=rows,
        completed_chains=len(rows), planned_chains=plan["independent_pilot_chains"],
        cpu_seconds=time.process_time() - started_cpu, wall_seconds=time.perf_counter() - started_wall,
        cpu_cap_seconds=plan["cpu_cap_seconds"],
        total_force_site_evaluations=sum(row["force_site_evaluations"] for row in rows),
        total_hmc_trajectories=sum(row["hmc_trajectories"] for row in rows),
        completed_work_is_lower_bound=terminal == "failed",
        source_pins_before=before, source_pins_after=after, source_pins_match=before == after,
        gate_failures=[dict(condition_index=row["condition_index"], initial=row["initial"],
            reasons=row["gate_reasons"]) for row in rows if not row["gate_passed"]],
        warnings=[dict(condition_index=row["condition_index"], initial=row["initial"],
            warnings=row["warnings"]) for row in rows if row["warnings"]],
        gpu_used=False, production_data_saved=False, equilibrium_certified=False,
        automatic_retry=False, automatic_step_selection=False,
        qualification=plan["qualification"])
    save(root / "summary.json", summary)
    save(root / "status.json", {key: summary[key] for key in
        ("schema", "status", "workflow_complete", "returncode", "error", "production_allowed",
         "stage_accepted", "completed_chains", "cpu_seconds", "source_pins_match")})
    files = {path.name: sha(path) for path in root.iterdir() if path.suffix == ".json"}
    save(root / "public_files.json", dict(schema=SCHEMA, files=files))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.output is None:
        parser.error("--execute requires --output")
    result = run(args.output) if args.execute else build_plan(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result.get("returncode", 0)


if __name__ == "__main__":
    raise SystemExit(main())
