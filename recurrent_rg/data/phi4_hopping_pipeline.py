"""Zero-field, fixed-lambda hopping scan in the original physical phi units.

This is a separate data format, not an r/u grid with renamed labels. Historical
samplers and datasets are immutable. The kernel samples psi=sqrt(kappa)*phi
and restores phi before any configuration or diagnostic is saved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import phi4 as base
from . import phi4_hybrid
from . import phi4_hybrid_pipeline as counts
from . import phi4_hopping as kernel


SCHEMA = "euclidean_phi4_hopping_hmc_embedded_wolff_v1"
PARAMETERIZATION = "hopping_kappa_quartic_lambda"
LABEL_COLUMNS = ("kappa", "quartic_lambda", "h")
ARRAY_MEMBERS = counts.ARRAY_MEMBERS
WORK_FIELDS = counts.WORK_FIELDS
SAMPLE_SEMANTICS = "independent_correlated_hybrid_markov_chains"
ACTION = "S=-kappa*sum_forward_edges(phi_x*phi_y)+sum_x(phi_x^2+quartic_lambda*(phi_x^2-1)^2)"
QUALIFICATION = (
    "固定正quartic_lambda、零外场扫描正kappa；每条最近邻边只计一次，无额外因子2。"
    "内部psi=sqrt(kappa)*phi仅为采样坐标变换，保存构型及原始诊断均还原为phi单位。"
    "固定HMC-Metropolis加lazy embedded-Ising Wolff；簇翻转不证明幅度或偶统计热化。"
    "保留所有ESS/接受率/热冷/变号警告，不自动补样、删链、换seed或重跑。"
    "kappa/lambda/h只用于生成和事后评价，训练只读取configurations。"
    "有限尺寸不认证平衡或无限体积相变；文献临界值仅作事后参考。"
)


@dataclass(frozen=True)
class Phi4HoppingOptions:
    sizes: tuple[int, ...] = (36,)
    dimension: int = 2
    kappa_values: tuple[float, ...] = tuple(float(x) for x in np.linspace(.50, .86, 13))
    quartic_lambda: float = 1.
    h: float = 0.
    chains: int = 4
    samples: int = 128
    burn_in: int = 1024
    spacing: int = 8
    step_size: float = .08
    leapfrog_steps: int = 12
    seed: int = 20261031
    test_every: int = 3
    test_offset: int = 0
    cluster_probability: float = .5
    max_trajectories: int = 200_000
    max_force_site_evaluations: int = 2_000_000_000
    max_cluster_updates: int = 2_000_000
    max_cluster_neighbor_visits: int = 1_000_000_000
    max_output_bytes: int = 1024**3
    max_memory_bytes: int = 512*1024**2

    def validate(self):
        if not isinstance(self.kappa_values, (list, tuple)) or not self.kappa_values:
            raise ValueError("kappa_values must be a nonempty sequence")
        kappas = [base._finite(x, "kappa") for x in self.kappa_values]
        if any(x <= 0 for x in kappas) or len(set(kappas)) != len(kappas):
            raise ValueError("Require distinct positive kappa values; kappa=0 has a singular sampling-coordinate map")
        if base._finite(self.quartic_lambda, "quartic_lambda") <= 0:
            raise ValueError("quartic_lambda must be strictly positive; no massless unconfined model")
        if base._finite(self.h, "h") != 0:
            raise ValueError("This zero-field Ising comparison requires h=0")
        if self.chains != 4:
            raise ValueError("This controlled dataset fixes four independent chains per condition")
        # Reuse only validation/work-count arithmetic. These stand-in r/u
        # values are never sampled, written as physical labels, or trained on.
        _count_options(self).validate()
        if self.samples % 2:
            raise ValueError("QCR evaluation requires two equal saved-sample halves")
        if self.dimension == 2 and any(side < 9 or side % 9 for side in self.sizes):
            raise ValueError("New 2D data require sizes divisible by 9 for two default b=3 transitions")
        for kappa in kappas:
            kernel.gradient_parameters(kappa, self.quartic_lambda, self.dimension)
        return self


def _count_options(options):
    common = {name: getattr(options, name) for name in counts.Phi4HybridOptions.__dataclass_fields__
              if name not in ("r_values", "u_values")}
    return counts.Phi4HybridOptions(**common, r_values=tuple(options.kappa_values),
                                   u_values=(options.quartic_lambda,))


def source_pins():
    return {
        "recurrent_rg/data/phi4.py": base.sha256_file(base.__file__),
        "recurrent_rg/data/phi4_hybrid.py": base.sha256_file(phi4_hybrid.__file__),
        "recurrent_rg/data/phi4_hybrid_pipeline.py": base.sha256_file(counts.__file__),
        "recurrent_rg/data/phi4_hopping.py": base.sha256_file(kernel.__file__),
        "recurrent_rg/data/phi4_hopping_pipeline.py": base.sha256_file(__file__),
    }


def _semantics():
    return dict(schema_version=5, schema=SCHEMA, parameterization=PARAMETERIZATION,
        evaluation_label_columns=list(LABEL_COLUMNS), field_representation="phi",
        internal_sampling_field="psi=sqrt(kappa)*phi", sample_semantics=SAMPLE_SEMANTICS)


def generation_plan(options):
    """Pure preflight: no RNG, sampling, model construction, or output writes."""
    options.validate()
    projection = counts.generation_plan(_count_options(options))
    retained = ("parameter_cells", "conditions", "trajectories_per_chain", "total_hmc_trajectories",
        "total_hybrid_cycles", "total_force_evaluations", "total_force_site_evaluations",
        "maximum_cluster_updates", "expected_cluster_updates", "maximum_cluster_sites",
        "maximum_cluster_neighbor_visits", "maximum_cluster_bond_tests", "maximum_cluster_activated_bonds",
        "configurations", "train_cell_indices", "test_cell_indices")
    result = {name: projection[name] for name in retained}
    # Transforming psi->phi needs a temporary field-sized array and diagnostic
    # scratch space in addition to the historical hybrid projection.
    working = projection["projected_working_bytes"] + options.samples*max(options.sizes)**options.dimension*8 + 1024**2
    projected_output = projection["projected_output_bytes"] + options.chains*len(options.kappa_values)*65536 + 1048576
    if working > options.max_memory_bytes or projected_output > options.max_output_bytes:
        raise ValueError("Hopping unit-conversion memory/output projection exceeds its registered cap")
    reference = (dict(kappa_critical=.680601, uncertainty=.000011, quartic_lambda=1., h=0.,
        dimension=2, url="https://arxiv.org/pdf/1807.05123", location="Equation (1), Table I",
        convention="nearest-neighbor coefficient -kappa, each undirected edge once; no factor 2",
        qualification="Published infinite-volume numerical reference, not finite-L truth or a training label")
        if options.dimension == 2 and float(options.quartic_lambda) == 1. else None)
    return dict(**_semantics(), **result, options=asdict(options), action=ACTION,
        boundary="periodic", lattice_spacing=1., probability="exp(-S)/Z", dtype="float64",
        sampler="unchanged psi-coordinate HMC-Metropolis followed by fixed lazy (I+Wolff)/2",
        cluster_probability=.5, cpu_threads_requested=1,
        projected_output_bytes=projected_output, projected_working_bytes=working,
        working_memory_projection_semantics="arrays and sampler workspace; excludes Python/native runtime baseline; not an RSS guarantee",
        work_estimate_is_not_wall_time=True,
        split_unit="entire (kappa,quartic_lambda,h) cell across all sizes and chains",
        axes=["independent_correlated_chain", "saved_sample", "*spatial", "channel=1"],
        initialization=projection["initialization"],
        step_size_coordinate="psi; not the physical phi coordinate",
        equivalent_phi_HMC_mass="kappa", momentum_refresh="standard Gaussian in psi coordinates",
        sampling_coordinate_maps=[kernel.gradient_parameters(k, options.quartic_lambda,
                                        options.dimension) for k in options.kappa_values],
        raw_array_members=list(ARRAY_MEMBERS), training_member="configurations",
        couplings_in_configuration_arrays=False, labels_used_in_training=False,
        training_updates=0, equilibrium_certified=False, production_adaptation=False,
        artificial_global_sign_flips=0, reference=reference, reference_used_for_training=False,
        qcr_2d_b3_geometry_compatible=options.dimension == 2,
        legacy_r_u_loader_compatible=False, loader="recurrent_rg.data.phi4_hopping_loader.load_dataset",
        source_pins=source_pins(), qualification=QUALIFICATION)


def chain_seed(options, size, cell, chain):
    """Parameter-keyed streams: list reordering cannot replace a chain's RNG."""
    message = [SCHEMA, int(options.seed), int(options.dimension), int(size),
               float(options.kappa_values[cell]), float(options.quartic_lambda), 0., int(chain)]
    return int.from_bytes(hashlib.sha256(json.dumps(message).encode()).digest()[:16], "little")


def _validate_chain(result, options, size):
    if result.get("schema") != SCHEMA or result.get("kernel_schema") != phi4_hybrid.SCHEMA:
        raise RuntimeError("Hopping sampler or its underlying frozen kernel identity differs")
    result["lazy_cluster_skips"] = options.burn_in+options.samples*options.spacing-int(result["wolff_cluster_updates"])
    # The historical checker is reused solely for shared array/work invariants.
    # Its schema guard is explicitly supplied the already verified kernel ID.
    counts._validate_chain_result({**result, "schema": result["kernel_schema"]}, size=size, options=options)


def generate_dataset(output, options):
    """Explicit fresh execution only. Never resume, overwrite, or tune a chain."""
    plan = generation_plan(options)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/"conditions").mkdir()
    base._write_json(root/"plan.json", plan)
    wall, cpu = time.perf_counter(), time.process_time()
    totals = {name: 0 for name in WORK_FIELDS}
    shards, qualities, warning_count, current = [], [], 0, None
    public = ["plan.json", "manifest.json", "status.json", "quality_summary.json"]
    try:
        for size in options.sizes:
            for cell, kappa in enumerate(options.kappa_values):
                kappa = float(kappa)
                split = "test" if cell % options.test_every == options.test_offset else "train"
                current = f"L{size}_c{cell:04d}"
                directory = root/"conditions"/current
                directory.mkdir()
                arrays, rows = [], []
                for chain in range(options.chains):
                    initial = ("hot" if chain < options.chains//2 else
                               "cold_positive" if chain % 2 == 0 else "cold_negative")
                    progress = dict(**_semantics(), status="running", workflow_complete=False,
                        current=current, chain_index=chain, completed_conditions=len(shards),
                        error=None, returncode=None, requires_user_action=False, equilibrium_certified=False)
                    base._write_json(root/"status.json", progress)
                    base._write_json(directory/"status.json", progress)
                    result = kernel.sample_chain(kappa, options.quartic_lambda, size=size,
                        dimension=options.dimension, samples=options.samples, burn_in=options.burn_in,
                        spacing=options.spacing, step_size=options.step_size, leapfrog_steps=options.leapfrog_steps,
                        seed=chain_seed(options, size, cell, chain), initial=initial, h=0., cluster_probability=.5)
                    _validate_chain(result, options, size)
                    arrays.append({name: result.pop(name) for name in ARRAY_MEMBERS})
                    rows.append(dict(chain_index=chain, **result))
                    for name in WORK_FIELDS:
                        totals[name] += int(result[name])
                    if (totals["force_site_evaluations"] > options.max_force_site_evaluations
                            or totals["wolff_cluster_updates"] > options.max_cluster_updates
                            or totals["total_cluster_neighbor_visits"] > options.max_cluster_neighbor_visits):
                        raise RuntimeError("Actual sampling work exceeded its registered cap; no restart or sample reduction")
                    warning_count += len(result["warnings"])
                    print(json.dumps(dict(event="phi4_hopping_chain_complete", condition=current,
                        chain_index=chain, production_acceptance=result["production_acceptance"],
                        warnings=len(result["warnings"])), allow_nan=False), flush=True)
                data_path = directory/"data.npz"
                stacked = {name: np.stack([value[name] for value in arrays]) for name in ARRAY_MEMBERS}
                np.savez(data_path, **stacked)
                summary = base._condition_summary(stacked["raw_observables"], rows, size, options.dimension)
                summary["cluster_work"] = {name: sum(int(row[name]) for row in rows) for name in WORK_FIELDS}
                summary["magnetization_sign_change_is_not_even_mode_mixing"] = True
                hot_cold = [name for name, value in summary["hot_cold_comparison"].items()
                            if value["descriptive_three_se_warning"]]
                warning_count += len(hot_cold)
                quality = dict(cell_index=cell, size=size, kappa=kappa,
                    quartic_lambda=float(options.quartic_lambda), h=0., split=split,
                    chain_warning_count=sum(len(row["warnings"]) for row in rows),
                    hot_cold_warning_observables=hot_cold,
                    chain_warnings=[dict(chain_index=row["chain_index"], warnings=row["warnings"]) for row in rows],
                    production_acceptance=[row["production_acceptance"] for row in rows],
                    saved_magnetization_sign_changes=[row["saved_magnetization_sign_changes"] for row in rows],
                    minimum_chain_ess={name: min(float(row["diagnostics"][name]["ess"] or 0.) for row in rows)
                                       for name in base.OBSERVABLES},
                    actual_work=summary["cluster_work"], equilibrium_certified=False)
                qualities.append(quality)
                sidecar = dict(**_semantics(), status="complete", cell_index=cell, size=size,
                    dimension=options.dimension, kappa=kappa, quartic_lambda=float(options.quartic_lambda),
                    h=0., split=split, sampling_coordinate_map=kernel.gradient_parameters(kappa,
                        options.quartic_lambda, options.dimension), chains=rows,
                    raw_observable_names=list(base.OBSERVABLES), summary=summary,
                    independent_chains=options.chains, samples_per_chain=options.samples,
                    cluster_probability=.5, source_pins=plan["source_pins"], raw_array_members=list(ARRAY_MEMBERS),
                    training_member="configurations", data_sha256=base.sha256_file(data_path),
                    symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
                    labels_used_in_training=False, equilibrium_certified=False, qualification=QUALIFICATION)
                sidecar_path = directory/"sidecar.json"
                base._write_json(sidecar_path, sidecar)
                status_path = directory/"status.json"
                base._write_json(status_path, dict(**_semantics(), status="complete", workflow_complete=True,
                    returncode=0, error=None, requires_user_action=False, equilibrium_certified=False))
                shards.append(dict(cell_index=cell, size=size, split=split,
                    path=str(data_path.relative_to(root)), sha256=sidecar["data_sha256"],
                    sidecar=str(sidecar_path.relative_to(root)), sidecar_sha256=base.sha256_file(sidecar_path),
                    bytes=data_path.stat().st_size))
                public.extend(str(path.relative_to(root)) for path in (sidecar_path, status_path))
                del arrays, stacked
                current = None
        base._write_json(root/"status.json", dict(**_semantics(), status="finalizing", workflow_complete=False,
            current=None, completed_conditions=len(shards), returncode=None, error=None,
            requires_user_action=False, equilibrium_certified=False))
        if source_pins() != plan["source_pins"]:
            raise RuntimeError("Pinned hopping sampler source changed during generation")
        if (totals["hmc_trajectories"] != plan["total_hmc_trajectories"]
                or totals["force_site_evaluations"] != plan["total_force_site_evaluations"]):
            raise RuntimeError("Actual fixed work differs from the hopping plan")
        quality_path = root/"quality_summary.json"
        base._write_json(quality_path, dict(**_semantics(), status="complete", warning_count=warning_count,
            conditions=qualities, actual_work=totals, warnings_retained_without_resampling=True,
            quality_is_execution_gate=False, equilibrium_certified=False, qualification=QUALIFICATION))
        manifest = dict(**_semantics(), status="complete", options=asdict(options), action=ACTION,
            cluster_probability=.5, shards=shards, public_files=public,
            plan_sha256=base.sha256_file(root/"plan.json"), source_pins=plan["source_pins"], source_pins_match=True,
            quality_summary="quality_summary.json", quality_summary_sha256=base.sha256_file(quality_path),
            actual_work=totals, total_hmc_trajectories=totals["hmc_trajectories"],
            total_force_site_evaluations=totals["force_site_evaluations"], warning_count=warning_count,
            training_member="configurations", couplings_in_configuration_arrays=False, labels_used_in_training=False,
            symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
            equilibrium_certified=False, qualification=QUALIFICATION)
        base._write_json(root/"manifest.json", manifest)
        actual_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if actual_bytes+65536 > options.max_output_bytes:
            raise RuntimeError("Actual output exceeds its frozen cap; completed artifacts preserved")
        base._write_json(root/"status.json", dict(**_semantics(), status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, current=None, completed_conditions=len(shards),
            warning_count=warning_count, actual_work=totals, actual_output_bytes_before_final_status=actual_bytes,
            manifest_sha256=base.sha256_file(root/"manifest.json"), equilibrium_certified=False,
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))
        return manifest
    except Exception as exc:
        failure = dict(**_semantics(), status="failed", workflow_complete=False, returncode=1,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, current=current,
            failure_scope="sampling_condition" if current is not None else "global_finalization",
            completed_conditions=len(shards), actual_work=totals, equilibrium_certified=False)
        base._write_json(root/"status.json", failure)
        if current is not None:
            base._write_json(root/"conditions"/current/"status.json", failure)
        raise
