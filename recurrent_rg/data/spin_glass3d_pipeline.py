"""Isolated finite zero-field 3D EA ICM/PT dataset with audited provenance.

The storage envelope remains edwards_anderson_pt_v1 for the existing validated
loader; the sampler schema identifies new 3D temperature-selective minority
clusters. One independent initialization group per disorder is deliberate.
Four real replicas are coupled by ICM, not four independent physical chains.
Raw spins/energies/overlap signs are preserved; no post-hoc Z2 augmentation.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .spin_glass import generate_bonds
from .spin_glass3d import sample_icm_parallel_tempering
from .spin_glass_pipeline import (
    GenerationOptions,
    SCHEMA,
    _bond_sha,
    _integer,
    _json,
    _seed,
    _sha,
    generation_plan as _historical_generation_plan,
    sampling_diagnostics,
    series_diagnostics,
)


SAMPLER_SCHEMA = "isoenergetic_cluster_parallel_tempering_3d_v1"
NUMERIC_LIMIT = int(np.iinfo(np.int64).max)
QUALIFICATION = (
    "Finite correlated ICM/PT MCMC data, not certified equilibrated. "
    "Real-replica trajectories within a group are coupled by cluster moves; "
    "replica pairs are not independent observations. Temperature transport "
    "labels are not intact configurations after cluster moves. No automatic "
    "extension, reseeding, training, or critical-point fit."
)


def _effective_options(options):
    if not isinstance(options, GenerationOptions):
        raise ValueError("options must be a GenerationOptions instance")
    # The user explicitly removed the old administrative work/memory/time
    # budgets.  Keep only representability bounds, recording the effective
    # options so the ordinary loader can independently validate the envelope.
    return replace(
        options,
        max_spin_proposals=NUMERIC_LIMIT,
        max_output_bytes=NUMERIC_LIMIT,
        max_memory_bytes=NUMERIC_LIMIT,
        max_sites=int(np.iinfo(np.intp).max),
    )


def _sampling_seed(master, disorder, shape):
    previous = _seed(master, "thermal_ladders", disorder, shape)
    encoded = json.dumps([SAMPLER_SCHEMA, int(previous), int(disorder)]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def generation_plan(
    options, *, cluster_interval=1, trace_interval=16, initial="hot", algorithm="icm_pt",
    icm_max_temperature=1.0,
):
    """Project fixed work; old administrative caps are explicitly disabled.

    Integer/index/finite-value validation remains mandatory.  Removing an
    administrative budget is not permission to overflow numerical counters or
    to adaptively extend a requested chain.  Estimated bytes are descriptive,
    not a measured RSS guarantee or an admission threshold.
    """
    options = _effective_options(options)
    options.validate()
    if len(options.shape) != 3:
        raise ValueError("3D ICM data generation requires a 3D lattice")
    if options.field != 0.0:
        raise ValueError("3D minority ICM requires exactly zero field")
    if isinstance(icm_max_temperature, (bool, np.bool_)) or not np.isscalar(icm_max_temperature):
        raise ValueError("icm_max_temperature must be a positive finite scalar")
    icm_max_temperature = float(icm_max_temperature)
    if not math.isfinite(icm_max_temperature) or icm_max_temperature <= 0.0:
        raise ValueError("icm_max_temperature must be a positive finite scalar")
    cluster_interval = _integer(cluster_interval, "cluster_interval", 1)
    trace_interval = _integer(trace_interval, "trace_interval", 1)
    if max(cluster_interval, trace_interval) > NUMERIC_LIMIT:
        raise ValueError("sampling intervals exceed int64 range")
    if initial not in ("hot", "cold"):
        raise ValueError("initial must be hot or cold; independent start groups require separate calls")
    if algorithm not in ("pt", "icm_pt"):
        raise ValueError("algorithm must be pt or icm_pt")
    plan = _historical_generation_plan(options)
    sweeps = plan["sweeps_per_temperature_per_real_replica"]
    n, nt, nr = plan["sites"], len(options.temperatures), int(options.replicas)
    if sweeps >= NUMERIC_LIMIT or sweeps * n * (nr // 2) > NUMERIC_LIMIT:
        raise ValueError("requested sampling/cluster counters exceed int64 range")
    trace_count = 1 + sweeps // trace_interval + int(sweeps % trace_interval != 0)
    extra_shard_bytes = 8 * nt * (n + 1) + 8 * trace_count * (1 + 2 * nt) + 8 * nt * 16 + 4096
    extra_projected = int(options.disorders) * extra_shard_bytes
    if plan["projected_output_bytes"] + extra_projected > NUMERIC_LIMIT:
        raise ValueError("projected storage exceeds numeric index range")
    if plan["projected_working_bytes"] + 4 * extra_shard_bytes > NUMERIC_LIMIT:
        raise ValueError("projected working arrays exceed numeric index range")
    sampling_seeds = [_sampling_seed(options.seed, i, options.shape) for i in range(options.disorders)]
    if len(set(sampling_seeds)) != options.disorders:
        raise ValueError("Derived new sampling seeds collide; no sampling or automatic reseed")
    sampler = dict(
        schema=SAMPLER_SCHEMA,
        algorithm=algorithm,
        initial=initial,
        replicas=nr,
        real_replicas_coupled=algorithm == "icm_pt",
        cluster_interval=cluster_interval,
        exchange_interval=options.exchange_interval,
        icm_max_temperature=icm_max_temperature,
        icm_active_temperatures=[float(T) <= icm_max_temperature for T in options.temperatures],
        icm_temperature_selection="fixed before sampling; not chosen using learned parameters, Kc, or diagnostics",
        field=0.0,
        gauge_flip_convention="if raw negative overlap exceeds N/2, flip a randomly selected copy for construction and undo the same global flip afterwards",
        saved_overlap_signs_unfolded=True,
        winding_definition="nonzero torus winding from inconsistent unwrapped BFS coordinates; a single periodic edge crossing is not winding",
        cluster_winding_is_diagnostic_only=True,
        algorithm_reference="Zhu, Ochoa, Katzgraber, arXiv:1501.05630; zero-field gauge-conjugated minority component",
        pairing="uniform random disjoint matching independently at each temperature and cluster event",
        target_joint_distribution="product of identical-Hamiltonian Boltzmann distributions at each temperature (stationary target, not equilibrium certification)",
        cluster_preserves_paired_overlap=True,
        cluster_joint_energy_conserved=True,
        cluster_acceptance="unconditional for every nonempty disagreement component",
        walker_label_caveat="ICM moves mix configurations; labels do not retain intact configuration identity",
        kernel_source_sha256=_sha(Path(__file__).with_name("spin_glass3d.py")),
    )
    plan.update(
        sampler=sampler,
        sampler_schema=SAMPLER_SCHEMA,
        sampler_algorithm=algorithm,
        sampler_options=dict(cluster_interval=cluster_interval, trace_interval=trace_interval,
                             initial=initial, algorithm=algorithm, icm_max_temperature=icm_max_temperature),
        projected_output_bytes=plan["projected_output_bytes"] + extra_projected,
        projected_working_bytes=plan["projected_working_bytes"] + 4 * extra_shard_bytes,
        cluster_pair_attempts_per_temperature=(sweeps // cluster_interval) * (nr // 2)
            if algorithm == "icm_pt" else 0,
        cluster_pair_attempts_by_temperature=[
            (sweeps // cluster_interval) * (nr // 2)
            if algorithm == "icm_pt" and float(T) <= icm_max_temperature else 0
            for T in options.temperatures
        ],
        trace_count_per_disorder=trace_count,
        initialization=f"{initial}: all real replicas {'random independent initial spins' if initial == 'hot' else 'all +1'}",
        replica_trajectory_dependence="ICM couples replica trajectories; neither R replicas nor R*(R-1)/2 pairs count as independent chains",
        cluster_update="Only at fixed T<=cutoff: random disjoint replica matching; choose a uniform site in the minority overlap domain and flip its entire nonzero-bond six-neighbor component in both copies, using an undone zero-field gauge flip when needed",
        cluster_order="local sweeps -> optional paired clusters -> alternating PT exchange -> diagnostics/save",
        cluster_energy_audit="Recompute full individual Hamiltonians before and after every nonempty cluster; record maximal joint-energy residual",
        round_trip_definition="Per temperature transport label: first endpoint -> other endpoint -> original endpoint; not intact configuration identity after clusters",
        seed_layout="Historical _seed(master,quenched_bonds,J,shape) unchanged; thermal SHA256(isoenergetic_cluster_parallel_tempering_3d_v1, old thermal seed, J); separate initialization comparisons require additional independently seeded groups (not generated here)",
        sampling_seeds=sampling_seeds,
        administrative_budget_caps_enabled=False,
        numeric_representability_checks_enabled=True,
        automatic_extension=False,
        qualification=QUALIFICATION,
    )
    return plan


def _group_diagnostics(result, shape, temperatures):
    diagnostic = sampling_diagnostics(result, shape, temperatures)
    spins, energies = result["configurations"], result["energies"]
    nt, nr, ns = spins.shape[:3]
    n = math.prod(shape)
    group_means = []
    for temperature in range(nt):
        q2_sum = np.zeros(ns, dtype=np.float64)
        pairs = 0
        for first in range(nr):
            for second in range(first + 1, nr):
                q = (spins[temperature, first] * spins[temperature, second]).reshape(ns, n).mean(1)
                q2_sum += q * q
                pairs += 1
        group_means.append(dict(
            temperature=float(temperatures[temperature]),
            energy_density_replica_mean=series_diagnostics(energies[temperature].mean(0) / n),
            q2_pair_mean=series_diagnostics(q2_sum / pairs),
            independent_sample_count_not_inferred=True,
        ))
    diagnostic.update(
        sampler_schema=SAMPLER_SCHEMA,
        sampler_algorithm=result["algorithm"],
        group_mean_series=group_means,
        replica_trajectories_coupled_by_clusters=result["algorithm"] == "icm_pt",
        independent_physical_group_count=1,
        independent_initialization_comparison_performed=False,
        initial=result["initial"],
        icm_max_temperature=result["icm_max_temperature"],
        icm_active_temperatures=result["icm_active_temperatures"],
        gauge_flip_count=result["gauge_flip_count"],
        raw_overlap_signs_preserved=True,
        gauge_flip_convention=result["gauge_flip_convention"],
        winding_definition=result["winding_definition"],
        cluster_statistics=[dict(
            temperature=float(T), icm_active=bool(result["icm_active_temperatures"][t]),
            attempts=int(result["cluster_attempts"][t]),
            nonempty_updates=int(result["cluster_updates"][t]),
            gauge_flips=int(result["gauge_flip_count"][t]),
            raw_disagreement_fraction=(float(result["raw_disagreement_sum"][t]) /
                (n * result["cluster_attempts"][t])) if result["cluster_attempts"][t] else None,
            minority_disagreement_fraction=(float(result["minority_disagreement_sum"][t]) /
                (n * result["cluster_attempts"][t])) if result["cluster_attempts"][t] else None,
            mean_cluster_fraction=(float(result["cluster_size_sum"][t]) /
                (n * result["cluster_updates"][t])) if result["cluster_updates"][t] else None,
            max_cluster_fraction=float(result["cluster_size_max"][t]) / n,
            winding_axis_counts=result["cluster_winding_count"][t],
            winding_any_count=int(result["cluster_winding_any_count"][t]),
            winding_fraction_nonempty=(float(result["cluster_winding_any_count"][t]) /
                result["cluster_updates"][t]) if result["cluster_updates"][t] else None,
        ) for t, T in enumerate(temperatures)],
        round_trip_label_caveat=result["walker_label_caveat"],
        cluster_attempts=result["cluster_attempts"],
        cluster_updates=result["cluster_updates"],
        cluster_size_sum=result["cluster_size_sum"],
        cluster_size_max=result["cluster_size_max"],
        max_cluster_joint_energy_residual=float(result["max_cluster_joint_energy_residual"]),
        max_energy_tracking_drift=float(result["max_energy_tracking_drift"]),
        trace_summary=dict(
            count=len(result["trace_sweeps"]),
            first_sweep=int(result["trace_sweeps"][0]),
            last_sweep=int(result["trace_sweeps"][-1]),
            includes_initialization_and_burn_in=True,
            no_stationary_ess_inferred_from_burn_in_trace=True,
            energy_density_initial=result["trace_energy_density"][:, 0],
            energy_density_final=result["trace_energy_density"][:, -1],
            q2_initial=result["trace_q2"][:, 0],
            q2_final=result["trace_q2"][:, -1],
        ),
        quality_certified=False,
        qualification=QUALIFICATION,
    )
    return diagnostic


def generate_dataset(
    output, options, *, cluster_interval=1, trace_interval=16, initial="hot", algorithm="icm_pt",
    icm_max_temperature=1.0,
):
    """Write a new finite dataset, preserving partial data on any exception.

    No existing destination is overwritten or resumed.  All-hot is deliberate:
    separate hot/cold ensembles would need separate independent generator
    groups; alternating initial states inside a coupled group is not such a
    comparison.  This function never loads a QCR model or starts training.
    """
    plan = generation_plan(options, cluster_interval=cluster_interval, trace_interval=trace_interval,
                           initial=initial, algorithm=algorithm, icm_max_temperature=icm_max_temperature)
    options = GenerationOptions(**plan["options"])
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "shards").mkdir()
    _json(root / "plan.json", plan)
    started = time.monotonic()
    status = dict(schema=SCHEMA, sampler_schema=SAMPLER_SCHEMA, status="running",
                  workflow_complete=False, started_at=datetime.now(timezone.utc).isoformat(),
                  completed_disorders=0, training_updates=0, equilibrium_certified=False,
                  administrative_budget_caps_enabled=False)
    _json(root / "status.json", status)
    manifest = dict(schema=SCHEMA, sampler=plan["sampler"], sampler_schema=SAMPLER_SCHEMA, sampler_algorithm=algorithm,
                    sampler_options=plan["sampler_options"], status="running",
                    plan_sha256=_sha(root / "plan.json"), options=asdict(options), shards=[],
                    training_updates=0, labels_used_in_training=False,
                    replica_trajectories_coupled_by_clusters=algorithm == "icm_pt",
                    administrative_budget_caps_enabled=False, qualification=QUALIFICATION)
    diagnostics, bond_splits = [], {}
    try:
        for disorder in range(options.disorders):
            status["current_disorder"] = disorder
            _json(root / "status.json", status)
            bond_seed = _seed(options.seed, "quenched_bonds", disorder, options.shape)
            sampling_seed = plan["sampling_seeds"][disorder]
            bonds = generate_bonds(options.shape, bond_seed, distribution=options.distribution,
                                   scale=options.bond_scale, mean=options.bond_mean,
                                   positive_probability=options.positive_probability)
            split = "test" if disorder % options.test_every == options.test_offset else "train"
            fingerprint = _bond_sha(bonds)
            if fingerprint in bond_splits and bond_splits[fingerprint] != split:
                raise ValueError("Identical quenched bonds across TRAIN/TEST; preserving partial data, no reseed")
            bond_splits[fingerprint] = split
            result = sample_icm_parallel_tempering(
                bonds, options.temperatures, samples=options.samples, burn_in=options.burn_in,
                spacing=options.spacing, seed=sampling_seed, replicas=options.replicas,
                algorithm=algorithm, exchange_interval=options.exchange_interval,
                cluster_interval=cluster_interval, field=options.field, initial=initial,
                trace_interval=trace_interval, icm_max_temperature=icm_max_temperature, max_output_bytes=NUMERIC_LIMIT,
            )
            residual = float(result["max_cluster_joint_energy_residual"])
            drift = float(result["max_energy_tracking_drift"])
            energy_scale = max(1., float(np.sum(np.abs(bonds))) + abs(options.field) * plan["sites"])
            if (not np.isfinite(residual) or not np.isfinite(drift)
                    or max(residual, drift) > 1e-9 * energy_scale):
                raise FloatingPointError("Cluster conservation or tracked-energy audit failed; no automatic repair")
            relative = f"shards/disorder_{disorder:06d}.npz"
            target = root / relative
            keys = (
                "configurations", "energies", "swap_attempts", "swap_accepts", "local_attempts",
                "local_accepts", "round_trips", "temperature_visits", "site_visits",
                "cluster_attempts", "cluster_updates", "cluster_size_sum", "cluster_size_max",
                "cluster_size_histogram", "trace_sweeps", "trace_energy_density", "trace_q2",
                "gauge_flip_count", "raw_disagreement_sum", "minority_disagreement_sum",
                "cluster_winding_count", "cluster_winding_any_count", "icm_active_temperatures",
            )
            arrays = {key: result[key] for key in keys}
            arrays.update(
                bonds=bonds, temperatures=np.asarray(options.temperatures, np.float64),
                disorder_id=np.array(disorder, np.int64), bond_seed=np.array(bond_seed, np.uint32),
                sampling_seed=np.array(sampling_seed, np.uint32),
                max_cluster_joint_energy_residual=np.array(residual, np.float64),
                max_energy_tracking_drift=np.array(drift, np.float64),
                icm_max_temperature=np.array(icm_max_temperature, np.float64),
                sampler_algorithm=np.array(algorithm), sampler_schema=np.array(SAMPLER_SCHEMA),
            )
            temporary = target.with_suffix(".npz.writing")
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(target)
            row = dict(disorder_id=disorder, split=split, path=relative, sha256=_sha(target),
                       bytes=target.stat().st_size, bond_sha256=fingerprint, bond_seed=bond_seed,
                       sampling_seed=sampling_seed, sampler_schema=SAMPLER_SCHEMA,
                       max_cluster_joint_energy_residual=residual, max_energy_tracking_drift=drift)
            manifest["shards"].append(row)
            diagnostics.append(dict(disorder_id=disorder, split=split,
                                    **_group_diagnostics(result, options.shape, options.temperatures)))
            _json(root / "manifest.json", manifest)
            _json(root / "sampling.json", dict(schema=SCHEMA, sampler_schema=SAMPLER_SCHEMA,
                                               status="running", disorders=diagnostics,
                                               qualification=QUALIFICATION))
            status.update(completed_disorders=disorder + 1, current_disorder=None,
                          elapsed_seconds=time.monotonic() - started)
            _json(root / "status.json", status)
            del arrays, result, bonds
        _json(root / "sampling.json", dict(
            schema=SCHEMA, sampler_schema=SAMPLER_SCHEMA, sampler_algorithm=algorithm,
            disorders=diagnostics,
            disorder_average=[dict(temperature=float(T), q2=float(np.mean([
                row["temperatures"][index]["q2_pair_mean"] for row in diagnostics])))
                for index, T in enumerate(options.temperatures)],
            warning_count=sum(len(row["warnings"]) for row in diagnostics),
            aggregation="Within-group dependent pair means within J first, then equal-weight disorder mean",
            independent_initialization_comparison_performed=False,
            equilibrium_certified=False, qualification=QUALIFICATION,
        ))
        source_files = (Path(__file__), Path(__file__).with_name("spin_glass3d.py"),
                        Path(__file__).with_name("spin_glass.py"),
                        Path(__file__).with_name("spin_glass_pipeline.py"))
        manifest.update(status="complete", sampling_sha256=_sha(root / "sampling.json"),
                        distinct_bond_realizations=len(bond_splits),
                        max_cluster_joint_energy_residual=max(row["max_cluster_joint_energy_residual"] for row in diagnostics),
                        max_energy_tracking_drift=max(row["max_energy_tracking_drift"] for row in diagnostics),
                        code_sha256={source.name: _sha(source) for source in source_files})
        _json(root / "manifest.json", manifest)
        status.update(status="complete", workflow_complete=True,
                      elapsed_seconds=time.monotonic() - started,
                      manifest_sha256=_sha(root / "manifest.json"))
        _json(root / "status.json", status)
    except Exception as exc:
        status.update(status="failed", workflow_complete=False,
                      error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started,
                      requires_user_action=True)
        _json(root / "status.json", status)
        raise
    return manifest

