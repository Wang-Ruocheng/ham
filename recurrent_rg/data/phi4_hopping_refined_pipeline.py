"""Isolated .04 x 24 HMC refinement, preserving the frozen physical kernel.

The historical hybrid wrapper has a hard two-billion force-site ceiling.  This
new orchestration extends only that ceiling to four billion, as explicitly
approved for the doubled integrator resolution.  No historical implementation
is changed or monkeypatched.  Physical seeds, action, arrays, diagnostics and
the strict hopping-loader schema retain their existing meaning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import json

import numpy as np

from . import phi4 as base
from . import phi4_hybrid_pipeline as counts
from . import phi4_hopping as kernel
from . import phi4_hopping_pipeline as historical


SCHEMA = historical.SCHEMA
ARRAY_MEMBERS = historical.ARRAY_MEMBERS
WORK_FIELDS = historical.WORK_FIELDS
QUALIFICATION = historical.QUALIFICATION + (
    "本次仅将固定psi坐标HMC步长0.08细化为0.04、积分步数12改为24，轨迹长度仍0.96。"
    "为此显式扩展力计算工作量保护线至40亿；其余资源保护、物理seed与采样规则不变。"
)
FROZEN_SOURCE_PINS = {
    "recurrent_rg/data/phi4.py": "49f7ad37c7c8b5780674583bdb5e848f47f11fa7b98f65cb53b28eb3fac8b4ce",
    "recurrent_rg/data/phi4_hybrid.py": "376efca351ba46c5cde736f15997f8545688590e4723b200db14b4cd4548c34c",
    "recurrent_rg/data/phi4_hybrid_pipeline.py": "0364fda6bf51b0d4865c014c29b084cc8e492e93b31d9994b7b1c6843b0561ea",
    "recurrent_rg/data/phi4_hopping.py": "66e1e94b9d38ee19acd9b3b7e0cf02878ecf08220959c4c441ff0206facc446a",
    "recurrent_rg/data/phi4_hopping_pipeline.py": "f6f759ed33a4a85dc7ce02d8dc54a56c8b2f4e9babbee1d360ed9d702458f029",
}


@dataclass(frozen=True)
class _RefinedCountingOptions(counts.Phi4HybridOptions):
    """Same historical semantic validation, with one explicit cap extension."""

    def validate(self):
        base.Phi4Options.validate(self)
        if base._finite(self.h, "h") != 0.:
            raise ValueError("This hybrid dataset requires h=0; no ghost-spin extension")
        if base._finite(self.cluster_probability, "cluster_probability") != .5:
            raise ValueError("Dataset generation fixes lazy cluster_probability=.5")
        for name in ("max_cluster_updates", "max_cluster_neighbor_visits"):
            base._integer(getattr(self, name), name, 1)
        if self.max_force_site_evaluations > 4_000_000_000:
            raise ValueError("Refined force-site cap cannot exceed 4,000,000,000")
        if self.max_output_bytes > 1024**3:
            raise ValueError("This small independent dataset is capped at <=1 GiB")
        return self


@dataclass(frozen=True)
class Phi4HoppingRefinedOptions(historical.Phi4HoppingOptions):
    step_size: float = .04
    leapfrog_steps: int = 24
    max_force_site_evaluations: int = 4_000_000_000

    def validate(self):
        if not isinstance(self.kappa_values, (list, tuple)) or not self.kappa_values:
            raise ValueError("kappa_values must be a nonempty sequence")
        kappas = [base._finite(x, "kappa") for x in self.kappa_values]
        if any(x <= 0 for x in kappas) or len(set(kappas)) != len(kappas):
            raise ValueError("Require distinct positive kappa values")
        if base._finite(self.quartic_lambda, "quartic_lambda") <= 0:
            raise ValueError("quartic_lambda must be strictly positive")
        if base._finite(self.h, "h") != 0:
            raise ValueError("This zero-field Ising comparison requires h=0")
        if self.chains != 4:
            raise ValueError("This controlled dataset fixes four independent chains per condition")
        _count_options(self).validate()
        if base._finite(self.step_size, "step_size") != .04 or self.leapfrog_steps != 24:
            raise ValueError("This isolated refinement fixes step_size=.04 and leapfrog_steps=24")
        if self.samples % 2:
            raise ValueError("QCR evaluation requires two equal saved-sample halves")
        if self.dimension == 2 and any(side < 9 or side % 9 for side in self.sizes):
            raise ValueError("New 2D data require sizes divisible by 9")
        for kappa in kappas:
            kernel.gradient_parameters(kappa, self.quartic_lambda, self.dimension)
        return self


def _count_options(options):
    common = {name: getattr(options, name) for name in counts.Phi4HybridOptions.__dataclass_fields__
              if name not in ("r_values", "u_values")}
    return _RefinedCountingOptions(**common, r_values=tuple(options.kappa_values),
                                   u_values=(options.quartic_lambda,))


def source_pins():
    pins = historical.source_pins()
    if pins != FROZEN_SOURCE_PINS:
        raise RuntimeError("A frozen physical sampling source fingerprint differs")
    project = Path(__file__).resolve().parents[2]
    pins.update({
        "recurrent_rg/data/phi4_hopping_refined_pipeline.py": base.sha256_file(__file__),
        "scripts/prepare_phi4_hopping_refined.py": base.sha256_file(
            project / "scripts/prepare_phi4_hopping_refined.py"),
    })
    return pins


def generation_plan(options):
    """Pure preflight: no RNG, sampling, model construction or output writes."""
    options.validate()
    projection = counts.generation_plan(_count_options(options))
    retained = ("parameter_cells", "conditions", "trajectories_per_chain", "total_hmc_trajectories",
        "total_hybrid_cycles", "total_force_evaluations", "total_force_site_evaluations",
        "maximum_cluster_updates", "expected_cluster_updates", "maximum_cluster_sites",
        "maximum_cluster_neighbor_visits", "maximum_cluster_bond_tests", "maximum_cluster_activated_bonds",
        "configurations", "train_cell_indices", "test_cell_indices")
    result = {name: projection[name] for name in retained}
    working = projection["projected_working_bytes"] + options.samples*max(options.sizes)**options.dimension*8 + 1024**2
    projected_output = projection["projected_output_bytes"] + options.chains*len(options.kappa_values)*65536 + 1048576
    if working > options.max_memory_bytes or projected_output > options.max_output_bytes:
        raise ValueError("Hopping unit-conversion memory/output projection exceeds its registered cap")
    reference = (dict(kappa_critical=.680601, uncertainty=.000011, quartic_lambda=1., h=0.,
        dimension=2, url="https://arxiv.org/pdf/1807.05123", location="Equation (1), Table I",
        convention="nearest-neighbor coefficient -kappa, each undirected edge once; no factor 2",
        qualification="Published infinite-volume numerical reference, not finite-L truth or a training label")
        if options.dimension == 2 and float(options.quartic_lambda) == 1. else None)
    return dict(**historical._semantics(), **result, options=asdict(options), action=historical.ACTION,
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
        sampler_refinement=dict(old_step_size=.08, old_leapfrog_steps=12,
            step_size=.04, leapfrog_steps=24, trajectory_length=.96,
            physical_chain_seed_policy="unchanged historical.chain_seed",
            maximum_force_site_evaluations=4_000_000_000,
            changes_physical_target=False, changes_sample_count=False,
            changes_kernel_implementation=False, changes_integrator_proposal=True),
        nonzero_production_acceptance_is_execution_gate=True,
        source_pins=source_pins(), qualification=QUALIFICATION)


chain_seed = historical.chain_seed


def generate_dataset(output, options):
    """Fresh generation only; original kernel and seeds, no retries or tuning."""
    plan = generation_plan(options)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "conditions").mkdir()
    base._write_json(root / "plan.json", plan)
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
                directory = root / "conditions" / current
                directory.mkdir()
                arrays, rows = [], []
                for chain in range(options.chains):
                    if source_pins() != plan["source_pins"]:
                        raise RuntimeError("Pinned sampling source changed before a chain")
                    initial = ("hot" if chain < options.chains//2 else
                               "cold_positive" if chain % 2 == 0 else "cold_negative")
                    progress = dict(**historical._semantics(), status="running", workflow_complete=False,
                        current=current, chain_index=chain, completed_conditions=len(shards),
                        error=None, returncode=None, requires_user_action=False, equilibrium_certified=False)
                    base._write_json(root / "status.json", progress)
                    base._write_json(directory / "status.json", progress)
                    result = kernel.sample_chain(kappa, options.quartic_lambda, size=size,
                        dimension=options.dimension, samples=options.samples, burn_in=options.burn_in,
                        spacing=options.spacing, step_size=options.step_size, leapfrog_steps=options.leapfrog_steps,
                        seed=chain_seed(options, size, cell, chain), initial=initial, h=0., cluster_probability=.5)
                    historical._validate_chain(result, options, size)
                    if source_pins() != plan["source_pins"]:
                        raise RuntimeError("Pinned sampling source changed during a chain")
                    arrays.append({name: result.pop(name) for name in ARRAY_MEMBERS})
                    rows.append(dict(chain_index=chain, **result))
                    for name in WORK_FIELDS:
                        totals[name] += int(result[name])
                    if (totals["force_site_evaluations"] > options.max_force_site_evaluations
                            or totals["wolff_cluster_updates"] > options.max_cluster_updates
                            or totals["total_cluster_neighbor_visits"] > options.max_cluster_neighbor_visits):
                        raise RuntimeError("Actual sampling work exceeded its registered cap; no restart or sample reduction")
                    if result["production_acceptance"] <= 0.:
                        raise RuntimeError("Zero production HMC acceptance; no automatic adjustment or retry")
                    warning_count += len(result["warnings"])
                    print(json.dumps(dict(event="phi4_hopping_chain_complete", condition=current,
                        chain_index=chain, production_acceptance=result["production_acceptance"],
                        warnings=len(result["warnings"])), allow_nan=False), flush=True)
                data_path = directory / "data.npz"
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
                sidecar = dict(**historical._semantics(), status="complete", cell_index=cell, size=size,
                    dimension=options.dimension, kappa=kappa, quartic_lambda=float(options.quartic_lambda),
                    h=0., split=split, sampling_coordinate_map=kernel.gradient_parameters(kappa,
                        options.quartic_lambda, options.dimension), chains=rows,
                    raw_observable_names=list(base.OBSERVABLES), summary=summary,
                    independent_chains=options.chains, samples_per_chain=options.samples,
                    cluster_probability=.5, source_pins=plan["source_pins"], raw_array_members=list(ARRAY_MEMBERS),
                    training_member="configurations", data_sha256=base.sha256_file(data_path),
                    symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
                    labels_used_in_training=False, equilibrium_certified=False, qualification=QUALIFICATION)
                sidecar_path = directory / "sidecar.json"
                base._write_json(sidecar_path, sidecar)
                status_path = directory / "status.json"
                base._write_json(status_path, dict(**historical._semantics(), status="complete", workflow_complete=True,
                    returncode=0, error=None, requires_user_action=False, equilibrium_certified=False))
                shards.append(dict(cell_index=cell, size=size, split=split,
                    path=str(data_path.relative_to(root)), sha256=sidecar["data_sha256"],
                    sidecar=str(sidecar_path.relative_to(root)), sidecar_sha256=base.sha256_file(sidecar_path),
                    bytes=data_path.stat().st_size))
                public.extend(str(path.relative_to(root)) for path in (sidecar_path, status_path))
                del arrays, stacked
                current = None
        base._write_json(root / "status.json", dict(**historical._semantics(), status="finalizing", workflow_complete=False,
            current=None, completed_conditions=len(shards), returncode=None, error=None,
            requires_user_action=False, equilibrium_certified=False))
        if source_pins() != plan["source_pins"]:
            raise RuntimeError("Pinned hopping sampler source changed during generation")
        if (totals["hmc_trajectories"] != plan["total_hmc_trajectories"]
                or totals["force_site_evaluations"] != plan["total_force_site_evaluations"]):
            raise RuntimeError("Actual fixed work differs from the hopping plan")
        quality_path = root / "quality_summary.json"
        base._write_json(quality_path, dict(**historical._semantics(), status="complete", warning_count=warning_count,
            conditions=qualities, actual_work=totals, warnings_retained_without_resampling=True,
            quality_is_execution_gate=False, nonzero_production_acceptance_is_execution_gate=True,
            equilibrium_certified=False, qualification=QUALIFICATION))
        manifest = dict(**historical._semantics(), status="complete", options=asdict(options), action=historical.ACTION,
            cluster_probability=.5, shards=shards, public_files=public,
            plan_sha256=base.sha256_file(root / "plan.json"), source_pins=plan["source_pins"], source_pins_match=True,
            quality_summary="quality_summary.json", quality_summary_sha256=base.sha256_file(quality_path),
            actual_work=totals, total_hmc_trajectories=totals["hmc_trajectories"],
            total_force_site_evaluations=totals["force_site_evaluations"], warning_count=warning_count,
            sampler_refinement=plan["sampler_refinement"],
            training_member="configurations", couplings_in_configuration_arrays=False, labels_used_in_training=False,
            symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
            equilibrium_certified=False, qualification=QUALIFICATION)
        base._write_json(root / "manifest.json", manifest)
        actual_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if actual_bytes+65536 > options.max_output_bytes:
            raise RuntimeError("Actual output exceeds its frozen cap; completed artifacts preserved")
        base._write_json(root / "status.json", dict(**historical._semantics(), status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, current=None, completed_conditions=len(shards),
            warning_count=warning_count, actual_work=totals, actual_output_bytes_before_final_status=actual_bytes,
            manifest_sha256=base.sha256_file(root / "manifest.json"), equilibrium_certified=False,
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))
        return manifest
    except Exception as exc:
        failure = dict(**historical._semantics(), status="failed", workflow_complete=False, returncode=1,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, current=current,
            failure_scope="sampling_condition" if current is not None else "global_finalization",
            completed_conditions=len(shards), actual_work=totals, equilibrium_certified=False)
        base._write_json(root / "status.json", failure)
        if current is not None:
            base._write_json(root / "conditions" / current / "status.json", failure)
        raise
