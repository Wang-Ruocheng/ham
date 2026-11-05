"""Fresh correlated phi4 datasets from the separately frozen lazy hybrid kernel.

This wrapper never edits or monkeypatches the historical HMC implementation or
the hybrid kernel. All three implementation sources are SHA256 pinned before
generation and checked again on completion. Configurations alone are consumed
by the existing continuous-field loader; coupling labels and every diagnostic
remain provenance/evaluation data. No overwrite, resume, adaptive tuning,
automatic resampling, sector balancing, or additional Z2 augmentation occurs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import phi4 as base
from . import phi4_hybrid as kernel


SCHEMA = kernel.SCHEMA
SAMPLE_SEMANTICS = "independent_correlated_hybrid_markov_chains"
ARRAY_MEMBERS = (
    "configurations", "raw_observables", "accepted", "delta_h",
    "cluster_sizes", "cluster_changed_sites", "cluster_neighbor_visits",
    "cluster_bond_tests", "cluster_activated_bonds", "cluster_attempted",
    "trajectory_magnetization",
)
WORK_FIELDS = (
    "hmc_trajectories", "hybrid_cycles", "force_evaluations", "force_site_evaluations",
    "wolff_cluster_updates", "lazy_cluster_skips", "total_cluster_sites",
    "total_cluster_changed_sites", "total_cluster_neighbor_visits",
    "total_cluster_bond_tests", "total_cluster_activated_bonds",
)
QUALIFICATION = (
    kernel.QUALIFICATION
    + "四个独立相关Markov链初态为2hot、冷正、冷负；此初态安排不是平衡扇区权重认证。"
    "每条链仍为独立训练系综，保存样本不称iid；sampling/cluster原始轨迹只作诊断。"
    "实际r/u/h耦合由manifest记录，仅用于生成与事后评价，不作为训练监督。"
    "所有既有ESS/接受率/热冷/变号警告保留，不因警告自动换seed、补样、删链或重跑。"
)


@dataclass(frozen=True)
class Phi4HybridOptions(base.Phi4Options):
    sizes: tuple[int, ...] = (36,)
    dimension: int = 2
    r_values: tuple[float, ...] = tuple(float(r) for r in np.linspace(-.5, .3, 12))
    u_values: tuple[float, ...] = (.2,)
    h: float = 0.
    chains: int = 4
    samples: int = 128
    burn_in: int = 1024
    spacing: int = 8
    step_size: float = .08
    leapfrog_steps: int = 12
    seed: int = 91
    test_every: int = 3
    test_offset: int = 1
    cluster_probability: float = .5
    max_force_site_evaluations: int = 2_000_000_000
    max_cluster_updates: int = 2_000_000
    max_cluster_neighbor_visits: int = 1_000_000_000
    max_output_bytes: int = 1024 ** 3
    max_memory_bytes: int = 512 * 1024 ** 2

    def validate(self):
        super().validate()
        if base._finite(self.h, "h") != 0.:
            raise ValueError("This hybrid dataset requires h=0; no ghost-spin extension")
        if base._finite(self.cluster_probability, "cluster_probability") != .5:
            raise ValueError("Dataset generation fixes lazy cluster_probability=.5; kernel p=0/1 controls are not production options")
        for name in ("max_cluster_updates", "max_cluster_neighbor_visits"):
            base._integer(getattr(self, name), name, 1)
        if self.max_force_site_evaluations > 2_000_000_000:
            raise ValueError("This run's force-site cap cannot exceed 2,000,000,000")
        if self.max_output_bytes > 1024 ** 3:
            raise ValueError("This small independent dataset is capped at <=1 GiB")
        return self


def source_pins():
    return {
        "recurrent_rg/data/phi4.py": base.sha256_file(base.__file__),
        "recurrent_rg/data/phi4_hybrid.py": base.sha256_file(kernel.__file__),
        "recurrent_rg/data/phi4_hybrid_pipeline.py": base.sha256_file(__file__),
    }


def generation_plan(options):
    """Pure resource preflight; worst-case cluster work uses every cycle, not pN."""
    options.validate()
    cells = len(options.r_values) * len(options.u_values)
    cycles = options.burn_in + options.samples * options.spacing
    volumes = [int(size) ** options.dimension for size in options.sizes]
    total_cycles = cells * len(volumes) * options.chains * cycles
    force_calls = total_cycles * (options.leapfrog_steps + 1)
    force_sites = cells * options.chains * cycles * (options.leapfrog_steps + 1) * sum(volumes)
    maximum_cluster_sites = cells * options.chains * cycles * sum(volumes)
    maximum_cluster_neighbor_visits = maximum_cluster_sites * 2 * options.dimension
    # Each undirected periodic edge is tested at most once by one Wolff update.
    maximum_cluster_bond_tests = maximum_cluster_sites * options.dimension
    maximum_activated_bonds = cells * options.chains * cycles * sum(v - 1 for v in volumes)
    # Every cycle: accepted bool + deltaH f64 + 5 cluster i64 arrays + attempted
    # bool + raw trajectory magnetization f64 = 58 bytes. Saved obs use f64.
    shard_bytes = [options.chains * (options.samples * (v + len(base.OBSERVABLES)) * 8
                                    + cycles * 58) for v in volumes]
    projected_output = cells * sum(value + options.chains * 65536 + 262144
                                   for value in shard_bytes) + 1048576
    projected_working = (3 * max(shard_bytes) + 128 * max(volumes) + options.chains * 65536
                         + 4096 * cells * len(volumes) + 8 * 1024 ** 2)
    for value, cap, name in (
        (total_cycles, options.max_trajectories, "HMC trajectory"),
        (force_sites, options.max_force_site_evaluations, "force-site evaluation"),
        (total_cycles, options.max_cluster_updates, "maximum cluster update"),
        (maximum_cluster_neighbor_visits, options.max_cluster_neighbor_visits, "maximum cluster neighbor visit"),
        (projected_output, options.max_output_bytes, "output byte"),
        (projected_working, options.max_memory_bytes, "working-memory byte")):
        if value > cap:
            raise ValueError(f"Projected {name} budget exceeded: {value} > {cap}")
    pins = source_pins()
    return dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS, options=asdict(options),
        sampler="fixed HMC-Metropolis followed by lazy (I+embedded-Ising Wolff)/2",
        cluster_probability=.5, action="S=.5*sum_(x,a)(phi[x+e_a]-phi[x])^2+sum_x(.5*r*phi^2+.25*u*phi^4)",
        boundary="periodic", dtype="float64", cpu_threads_requested=1,
        parameter_cells=cells, conditions=cells * len(volumes),
        trajectories_per_chain=cycles, total_hmc_trajectories=total_cycles,
        total_hybrid_cycles=total_cycles, total_force_evaluations=force_calls,
        total_force_site_evaluations=force_sites, maximum_cluster_updates=total_cycles,
        expected_cluster_updates=total_cycles * .5,
        maximum_cluster_sites=maximum_cluster_sites,
        maximum_cluster_neighbor_visits=maximum_cluster_neighbor_visits,
        maximum_cluster_bond_tests=maximum_cluster_bond_tests,
        maximum_cluster_activated_bonds=maximum_activated_bonds,
        configurations=cells * len(volumes) * options.chains * options.samples,
        projected_output_bytes=projected_output, projected_working_bytes=projected_working,
        work_estimate_is_not_wall_time=True, split_unit="entire (r,u,h) cell across all sizes and chains",
        train_cell_indices=[i for i in range(cells) if i % options.test_every != options.test_offset],
        test_cell_indices=[i for i in range(cells) if i % options.test_every == options.test_offset],
        initialization="2 hot, 1 cold_positive, 1 cold_negative for four chains; no augmentation",
        axes=["independent_correlated_chain", "saved_sample", "*spatial", "channel=1"],
        saved_sample_spacing_unit="fixed hybrid cycle, not local sweep or independent draw",
        raw_array_members=list(ARRAY_MEMBERS), training_member="configurations",
        labels_used_in_training=False, couplings_in_configuration_arrays=False,
        training_updates=0, equilibrium_certified=False, artificial_global_sign_flips=0,
        source_sha256=pins["recurrent_rg/data/phi4_hybrid_pipeline.py"], source_pins=pins,
        qualification=QUALIFICATION)


def chain_seed(options, size, cell, chain):
    message = [SCHEMA, int(options.seed), int(options.dimension), int(size),
               float(options.r_values[cell // len(options.u_values)]),
               float(options.u_values[cell % len(options.u_values)]), 0., int(chain)]
    return int.from_bytes(hashlib.sha256(json.dumps(message).encode()).digest()[:16], "little")


def _validate_chain_result(result, *, size, options):
    cycles = options.burn_in + options.samples * options.spacing
    volume = size ** options.dimension
    result["lazy_cluster_skips"] = cycles - int(result["wolff_cluster_updates"])
    expected = {
        "hmc_trajectories": cycles, "hybrid_cycles": cycles,
        "force_evaluations": cycles * (options.leapfrog_steps + 1),
        "force_site_evaluations": cycles * (options.leapfrog_steps + 1) * volume,
    }
    if (result.get("schema") != SCHEMA or result.get("cluster_probability") != .5
            or result.get("production_adaptation") is not False
            or result.get("symmetry_augmentation") is not False
            or result.get("artificial_global_sign_flips") != 0):
        raise RuntimeError("Hybrid kernel identity/schedule differs from the frozen dataset plan")
    if any(result.get(name) != value for name, value in expected.items()):
        raise RuntimeError("HMC cycle/force counters disagree with actual fixed settings")
    config = result["configurations"]
    if (config.shape != (options.samples, *((size,) * options.dimension), 1)
            or config.dtype != np.float64 or not np.isfinite(config).all()):
        raise RuntimeError("Invalid hybrid configuration array")
    for name in ARRAY_MEMBERS:
        array = result[name]
        if not isinstance(array, np.ndarray) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid/nonfinite sampler array: {name}")
        if name not in ("configurations", "raw_observables") and array.shape != (cycles,):
            raise RuntimeError(f"Sampler array length differs from hybrid cycles: {name}")
    attempted = result["cluster_attempted"]
    if attempted.dtype != np.bool_ or result["accepted"].dtype != np.bool_:
        raise RuntimeError("Sampler acceptance/attempt masks must be bool")
    if (result["raw_observables"].shape != (options.samples, len(base.OBSERVABLES))
            or result["raw_observables"].dtype != np.float64):
        raise RuntimeError("Invalid raw-observable trajectory")
    if int(attempted.sum()) != result["wolff_cluster_updates"]:
        raise RuntimeError("Wolff attempts counter mismatch")
    for array, total in (
        ("cluster_sizes", "total_cluster_sites"),
        ("cluster_changed_sites", "total_cluster_changed_sites"),
        ("cluster_neighbor_visits", "total_cluster_neighbor_visits"),
        ("cluster_bond_tests", "total_cluster_bond_tests"),
        ("cluster_activated_bonds", "total_cluster_activated_bonds")):
        values = result[array]
        if values.dtype != np.int64 or np.any(values < 0) or np.any(values[~attempted] != 0):
            raise RuntimeError(f"Invalid actual/skip cluster work: {array}")
        if int(values.sum()) != result[total]:
            raise RuntimeError(f"Cluster work sum mismatch: {array}")
    sizes = result["cluster_sizes"]
    if np.any(sizes[attempted] < 1) or np.any(sizes > volume):
        raise RuntimeError("Invalid Wolff cluster size")
    if (np.any(result["cluster_changed_sites"] > sizes)
            or np.any(result["cluster_neighbor_visits"] != 2 * options.dimension * sizes)
            or np.any(result["cluster_bond_tests"] > volume * options.dimension)
            or np.any(result["cluster_activated_bonds"][attempted] != sizes[attempted] - 1)):
        raise RuntimeError("Actual Wolff work exceeds nearest-neighbor cluster bounds")
    selected = result["trajectory_magnetization"][options.burn_in + options.spacing - 1::options.spacing]
    if not np.array_equal(selected, result["raw_observables"][:, 1]):
        raise RuntimeError("Saved magnetization differs from the raw post-hybrid trajectory")


def generate_dataset(output, options):
    """Write a fresh dataset; all diagnostics survive warnings, never resample."""
    plan = generation_plan(options)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base._write_json(output / "plan.json", plan)
    (output / "conditions").mkdir()
    wall, cpu = time.perf_counter(), time.process_time()
    shards, qualities, warning_count, current = [], [], 0, None
    totals = {name: 0 for name in WORK_FIELDS}
    public_files = ["plan.json", "manifest.json", "status.json", "quality_summary.json"]
    try:
        for size in options.sizes:
            for cell in range(plan["parameter_cells"]):
                r = float(options.r_values[cell // len(options.u_values)])
                u = float(options.u_values[cell % len(options.u_values)])
                split = "test" if cell % options.test_every == options.test_offset else "train"
                current = f"L{size}_c{cell:04d}"
                directory = output / "conditions" / current
                directory.mkdir()
                arrays, rows = [], []
                for chain in range(options.chains):
                    initial = ("hot" if chain < options.chains // 2 else
                               "cold_positive" if chain % 2 == 0 else "cold_negative")
                    progress = dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS,
                        status="running", workflow_complete=False, current=current,
                        chain_index=chain, completed_conditions=len(shards), error=None,
                        requires_user_action=False, equilibrium_certified=False)
                    base._write_json(output / "status.json", progress)
                    base._write_json(directory / "status.json", progress)
                    result = kernel.sample_hybrid_chain(r, u, size=size, dimension=options.dimension,
                        samples=options.samples, burn_in=options.burn_in, spacing=options.spacing,
                        step_size=options.step_size, leapfrog_steps=options.leapfrog_steps,
                        seed=chain_seed(options, size, cell, chain), initial=initial, h=0.,
                        cluster_probability=.5)
                    _validate_chain_result(result, size=size, options=options)
                    arrays.append({name: result.pop(name) for name in ARRAY_MEMBERS})
                    rows.append(dict(chain_index=chain, **result))
                    for name in WORK_FIELDS:
                        totals[name] += int(result[name])
                    if (totals["force_site_evaluations"] > options.max_force_site_evaluations
                            or totals["wolff_cluster_updates"] > options.max_cluster_updates
                            or totals["total_cluster_neighbor_visits"] > options.max_cluster_neighbor_visits):
                        raise RuntimeError("Actual work exceeded the frozen resource cap; no restart or sample reduction")
                    warning_count += len(result["warnings"])
                    print(json.dumps(dict(event="phi4_hybrid_chain_complete", condition=current,
                        chain_index=chain, production_acceptance=result["production_acceptance"],
                        actual_wolff_updates=result["wolff_cluster_updates"],
                        warnings=len(result["warnings"])), allow_nan=False), flush=True)
                data_path = directory / "data.npz"
                stacked = {name: np.stack([array[name] for array in arrays]) for name in ARRAY_MEMBERS}
                np.savez(data_path, **stacked)
                summary = base._condition_summary(stacked["raw_observables"], rows, size, options.dimension)
                summary["cluster_work"] = {name: sum(int(row[name]) for row in rows) for name in WORK_FIELDS}
                summary["magnetization_sign_change_is_not_even_mode_mixing"] = True
                hot_cold_warnings = [name for name, value in summary["hot_cold_comparison"].items()
                                     if value["descriptive_three_se_warning"]]
                warning_count += len(hot_cold_warnings)
                quality = dict(cell_index=cell, size=size, r=r, u=u, h=0., split=split,
                    chain_warning_count=sum(len(row["warnings"]) for row in rows),
                    hot_cold_warning_observables=hot_cold_warnings,
                    chain_warnings=[dict(chain_index=row["chain_index"], warnings=row["warnings"]) for row in rows],
                    production_acceptance=[row["production_acceptance"] for row in rows],
                    saved_magnetization_sign_changes=[row["saved_magnetization_sign_changes"] for row in rows],
                    minimum_chain_ess={name: min(float(row["diagnostics"][name]["ess"] or 0.) for row in rows)
                                       for name in base.OBSERVABLES},
                    actual_work=summary["cluster_work"], equilibrium_certified=False)
                qualities.append(quality)
                sidecar = dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS, status="complete",
                    cell_index=cell, size=size, dimension=options.dimension, r=r, u=u, h=0., split=split,
                    chains=rows, raw_observable_names=list(base.OBSERVABLES), summary=summary,
                    independent_chains=options.chains, samples_per_chain=options.samples,
                    cluster_probability=.5, source_pins=plan["source_pins"],
                    raw_array_members=list(ARRAY_MEMBERS), training_member="configurations",
                    data_sha256=base.sha256_file(data_path), symmetry_augmentation=False,
                    artificial_global_sign_flips=0, production_adaptation=False,
                    labels_used_in_training=False, equilibrium_certified=False, qualification=QUALIFICATION)
                sidecar_path = directory / "sidecar.json"
                base._write_json(sidecar_path, sidecar)
                condition_status = directory / "status.json"
                base._write_json(condition_status, dict(schema=SCHEMA, status="complete", workflow_complete=True,
                    returncode=0, error=None, requires_user_action=False, equilibrium_certified=False))
                shards.append(dict(cell_index=cell, size=size, split=split,
                    path=str(data_path.relative_to(output)), sha256=sidecar["data_sha256"],
                    sidecar=str(sidecar_path.relative_to(output)), sidecar_sha256=base.sha256_file(sidecar_path),
                    bytes=data_path.stat().st_size))
                public_files.extend(str(path.relative_to(output)) for path in (sidecar_path, condition_status))
                del arrays, stacked
        if source_pins() != plan["source_pins"]:
            raise RuntimeError("Pinned sampler implementation source changed during generation")
        if (totals["hmc_trajectories"] != plan["total_hmc_trajectories"]
                or totals["force_site_evaluations"] != plan["total_force_site_evaluations"]):
            raise RuntimeError("Final actual HMC work differs from the fixed plan")
        quality_path = output / "quality_summary.json"
        base._write_json(quality_path, dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS,
            status="complete", warning_count=warning_count, conditions=qualities, actual_work=totals,
            warnings_retained_without_resampling=True, quality_is_execution_gate=False,
            equilibrium_certified=False, qualification=QUALIFICATION))
        manifest = dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS, status="complete",
            options=asdict(options), cluster_probability=.5, shards=shards,
            public_files=public_files, plan_sha256=base.sha256_file(output / "plan.json"),
            source_sha256=plan["source_sha256"], source_pins=plan["source_pins"], source_pins_match=True,
            quality_summary="quality_summary.json", quality_summary_sha256=base.sha256_file(quality_path),
            total_hmc_trajectories=totals["hmc_trajectories"],
            total_force_site_evaluations=totals["force_site_evaluations"], actual_work=totals,
            warning_count=warning_count, labels_used_in_training=False,
            symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
            equilibrium_certified=False, qualification=QUALIFICATION)
        base._write_json(output / "manifest.json", manifest)
        actual_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if actual_bytes + 65536 > options.max_output_bytes:
            raise RuntimeError("Actual dataset output exceeded frozen directory cap; files preserved")
        base._write_json(output / "status.json", dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS,
            status="complete", workflow_complete=True, current=None, completed_conditions=len(shards),
            returncode=0, error=None, requires_user_action=False, equilibrium_certified=False,
            warning_count=warning_count, actual_work=totals, actual_output_bytes_before_final_status=actual_bytes,
            manifest_sha256=base.sha256_file(output / "manifest.json"),
            wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu))
        return manifest
    except Exception as exc:
        failure = dict(schema=SCHEMA, status="failed", workflow_complete=False, returncode=1,
            current=current, completed_conditions=len(shards), error=f"{type(exc).__name__}: {exc}",
            requires_user_action=True, actual_work=totals, equilibrium_certified=False)
        base._write_json(output / "status.json", failure)
        if current is not None:
            base._write_json(output / "conditions" / current / "status.json", failure)
        raise
