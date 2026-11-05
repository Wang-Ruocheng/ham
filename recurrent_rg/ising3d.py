"""New-only, zero-field three-dimensional ferromagnetic Wolff data pipeline.

Work is specified in *cluster proposals*, never lattice sweeps.  Every chain
has a private thermal seed and an independent augmentation seed.  Observables
are recorded before optional global Z2 augmentation; finite-chain diagnostics
are descriptive evidence, not an equilibrium certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import time

import numpy as np
from numba import njit

from .data.spin_glass_pipeline import series_diagnostics


SCHEMA = "ising3d_wolff_v1"
QUALIFICATION = (
    "三维周期零场最近邻铁磁 Ising；固定次数 Wolff 簇提议，不将其称为 sweep。"
    "各物理链具有分离的热随机种子；一半随机初态、一半有序初态。"
    "原始能量及磁化诊断先于逐构型全局 Z2 增强；有限链诊断不认证平衡。"
    "K 仅用于生成和事后评价，不写入训练耦合标签。"
)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(result)


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
        allow_nan=False, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x)) + "\n")
    temporary.replace(path)


def periodic_neighbors(size):
    """Six neighbor entries per site, including parallel edges when size=2."""
    size = _integer(size, "size", 2)
    sites = np.arange(size ** 3, dtype=np.int64).reshape((size,) * 3)
    return np.ascontiguousarray(np.stack([
        np.roll(sites, shift, axis=axis).ravel()
        for axis in range(3) for shift in (-1, 1)
    ], axis=1))


def energy_3d(spins):
    """H/J=-sum_(x,a) s_x s_(x+e_a), three positive-direction bonds/site."""
    spins = np.asarray(spins)
    if spins.ndim < 3 or len(set(spins.shape[-3:])) != 1 or spins.shape[-1] < 2:
        raise ValueError("spins must end in a cubic lattice of side >=2")
    if not np.all((spins == -1) | (spins == 1)):
        raise ValueError("spins must be exactly -1 or +1")
    return -sum(np.sum(spins * np.roll(spins, -1, axis=axis),
        axis=(-3, -2, -1), dtype=np.int64) for axis in (-3, -2, -1))


@njit(cache=True)
def _cluster_update(spins, neighbors, probability, stack):
    start = np.random.randint(spins.size)
    target = spins[start]
    # Flip at insertion: membership can then be tested without an O(N) clear.
    spins[start] = -target
    stack[0] = start
    pending, count = 1, 1
    while pending:
        pending -= 1
        site = stack[pending]
        for direction in range(6):
            other = neighbors[site, direction]
            if spins[other] == target and np.random.random() < probability:
                spins[other] = -target
                stack[pending] = other
                pending += 1
                count += 1
    return count


@njit(cache=True)
def _sample_chain(size, coupling, samples, burn_in, spacing, seed, hot, cold_sign, neighbors):
    np.random.seed(seed)
    sites = size ** 3
    spins = np.full(sites, cold_sign, dtype=np.int8)
    if hot:
        for site in range(sites):
            spins[site] = 2 * np.random.randint(0, 2) - 1
    stack = np.empty(sites, dtype=np.int64)
    configurations = np.empty((samples, sites), dtype=np.int8)
    observables = np.empty((samples, 4), dtype=np.float64)
    cluster_sizes = np.empty(burn_in + samples * spacing, dtype=np.int64)
    probability = -math.expm1(-2. * coupling)
    position = 0
    for _ in range(burn_in):
        cluster_sizes[position] = _cluster_update(spins, neighbors, probability, stack)
        position += 1
    for sample in range(samples):
        for _ in range(spacing):
            cluster_sizes[position] = _cluster_update(spins, neighbors, probability, stack)
            position += 1
        configurations[sample] = spins
        magnetization, energy = 0, 0
        for site in range(sites):
            magnetization += spins[site]
            for axis in range(3):
                energy -= spins[site] * spins[neighbors[site, 2 * axis]]
        m = magnetization / sites
        observables[sample, 0] = energy / sites
        observables[sample, 1] = m
        observables[sample, 2] = m * m
        observables[sample, 3] = abs(m)
    return configurations.reshape((samples, size, size, size)), observables, cluster_sizes


def sample_wolff_chain(coupling, *, size, samples, burn_in_clusters, clusters_between,
                       seed, initial="hot", cold_sign=1, augmentation_seed=None,
                       max_output_bytes=None):
    """One independent chain; augmentation never changes its dynamics/diagnostics."""
    size = _integer(size, "size", 2)
    samples = _integer(samples, "samples", 1)
    burn_in_clusters = _integer(burn_in_clusters, "burn_in_clusters", 0)
    clusters_between = _integer(clusters_between, "clusters_between", 1)
    seed = _integer(seed, "seed", 0)
    if seed >= 2 ** 32:
        raise ValueError("seed must fit uint32")
    if np.ndim(coupling) or np.iscomplexobj(coupling):
        raise ValueError("coupling must be a finite nonnegative scalar")
    coupling = float(coupling)
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("Wolff requires finite nonnegative ferromagnetic coupling")
    if initial not in ("hot", "cold") or cold_sign not in (-1, 1):
        raise ValueError("initial must be hot/cold and cold_sign must be +/-1")
    if augmentation_seed is not None:
        augmentation_seed = _integer(augmentation_seed, "augmentation_seed", 0)
        if augmentation_seed >= 2 ** 32:
            raise ValueError("augmentation_seed must fit uint32")
    clusters = burn_in_clusters + samples * clusters_between
    sites = size ** 3
    projected_bytes = samples * sites + samples * 4 * 8 + clusters * 8
    if max(projected_bytes, sites * 6 * 8) > np.iinfo(np.intp).max:
        raise ValueError("Requested arrays exceed platform index range")
    if max_output_bytes is not None and projected_bytes > _integer(max_output_bytes, "max_output_bytes", 1):
        raise ValueError("Projected chain output exceeds explicit memory preflight limit")
    wall, cpu = time.perf_counter(), time.process_time()
    configurations, raw_observables, cluster_sizes = _sample_chain(size, coupling, samples,
        burn_in_clusters, clusters_between, seed, initial == "hot", int(cold_sign),
        periodic_neighbors(size))
    # Separate RNG namespace: enabling/disabling augmentation leaves raw chain intact.
    flips = np.ones(samples, dtype=np.int8)
    if augmentation_seed is not None:
        flips = np.where(np.random.default_rng(augmentation_seed).integers(0, 2, samples), -1, 1).astype(np.int8)
        configurations *= flips[:, None, None, None]
    elapsed, cpu_seconds = time.perf_counter() - wall, time.process_time() - cpu
    names = ("energy_density", "magnetization", "magnetization_squared", "absolute_magnetization")
    diagnostics = {name: series_diagnostics(raw_observables[:, i]) for i, name in enumerate(names)}
    warnings = [dict(observable=name, reason=values["warning"]) for name, values in diagnostics.items() if values.get("warning")]
    warnings.extend(dict(observable=name, reason="ESS below 32 saved configurations")
        for name, values in diagnostics.items() if values.get("ess") is not None and values["ess"] < 32)
    return dict(configurations=configurations, raw_observables=raw_observables,
        raw_observable_names=names, cluster_sizes=cluster_sizes, augmentation_flips=flips,
        diagnostics=diagnostics, warnings=warnings, cluster_proposals=clusters,
        cluster_visited_sites=int(cluster_sizes.sum()), neighbor_inspections=int(6 * cluster_sizes.sum()),
        wall_seconds=elapsed, cpu_seconds=cpu_seconds, equilibrium_certified=False,
        diagnostics_before_global_z2=True)


@dataclass(frozen=True)
class Wolff3DOptions:
    size: int = 16
    couplings: tuple[float, ...] = tuple(np.linspace(.12, .36, 24))
    chains: int = 4
    samples: int = 128
    burn_in_clusters: int = 4096
    clusters_between: int = 64
    seed: int = 20261019
    augment_global_z2: bool = True
    test_every: int = 3
    test_offset: int = 1
    max_working_bytes: int | None = None

    def validate(self):
        for field, minimum in (("size", 2), ("chains", 2), ("samples", 4),
                ("burn_in_clusters", 0), ("clusters_between", 1), ("seed", 0), ("test_every", 2), ("test_offset", 0)):
            _integer(getattr(self, field), field, minimum)
        if self.chains % 2 or self.samples % 2:
            raise ValueError("chains and saved sample count must be even")
        couplings = np.asarray(self.couplings, dtype=float)
        if couplings.ndim != 1 or len(couplings) < 3 or not np.isfinite(couplings).all() or (couplings < 0).any():
            raise ValueError("Require at least 3 finite nonnegative couplings")
        if not np.all(np.diff(couplings) > 0):
            raise ValueError("couplings must be strictly increasing")
        if self.test_offset >= self.test_every or self.seed >= 2 ** 32:
            raise ValueError("Invalid split offset or uint32 seed")
        if type(self.augment_global_z2) is not bool:
            raise ValueError("augment_global_z2 must be bool")
        if not any(i % self.test_every == self.test_offset for i in range(len(couplings))):
            raise ValueError("Split must retain nonempty train and test coupling sets")
        if self.max_working_bytes is not None:
            _integer(self.max_working_bytes, "max_working_bytes", 1)
        return self


def chain_seed(options, coupling_index, chain_index, purpose):
    """Bijective uint32 tags per chain/purpose within one frozen physical plan."""
    if purpose not in ("thermal", "global_z2"):
        raise ValueError("Unknown seed namespace")
    message = json.dumps([SCHEMA, int(options.seed), int(options.size),
        [float(k).hex() for k in options.couplings]], separators=(",", ":"))
    base = int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], "little")
    tag = 2 * (coupling_index * options.chains + chain_index) + (purpose == "global_z2")
    return int(((base + tag) * 0x9E3779B1) % 2 ** 32)


def generation_plan(options):
    options.validate()
    nk, sites = len(options.couplings), options.size ** 3
    chain_clusters = options.burn_in_clusters + options.samples * options.clusters_between
    # One condition is streamed; explicit bound includes serialization/validation copies.
    condition_bytes = options.chains * (options.samples * sites + options.samples * 40 + chain_clusters * 8)
    working = 4 * condition_bytes + sites * 64 + 2 ** 20
    if max(condition_bytes, working) > np.iinfo(np.intp).max or 2 * nk * options.chains >= 2 ** 32:
        raise ValueError("Requested physical plan exceeds numeric index/seed range")
    if options.max_working_bytes is not None and working > options.max_working_bytes:
        raise ValueError("Projected working memory exceeds explicit preflight limit")
    seeds = [chain_seed(options, i, c, p) for i in range(nk) for c in range(options.chains) for p in ("thermal", "global_z2")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed namespace collision")
    return dict(schema=SCHEMA, options=asdict(options), dimension=3, sites=sites,
        hamiltonian="H/J=-sum_(x,a=1..3) s_x*s_(x+e_a); K=J/(k_B*T), zero field, J>0",
        sampler="Wolff single-cluster proposals; p_add=1-exp(-2*K); six periodic neighbors",
        cluster_proposals_per_chain=chain_clusters,
        total_cluster_proposals=nk * options.chains * chain_clusters,
        cluster_visited_sites_upper_bound=nk * options.chains * chain_clusters * sites,
        projected_condition_array_bytes=condition_bytes, projected_working_bytes=working,
        projected_output_bytes=nk * (condition_bytes + 262144) + 262144,
        train_coupling_indices=[i for i in range(nk) if i % options.test_every != options.test_offset],
        test_coupling_indices=[i for i in range(nk) if i % options.test_every == options.test_offset],
        split_unit="entire coupling index including all independent physical chains",
        initialization="first half hot, second half cold with alternating +1/-1 ordered signs",
        seed_layout="SHA256(plan namespace) base plus bijective uint32 chain/purpose tags",
        labels_used_in_training=False, equilibrium_certified=False, qualification=QUALIFICATION,
        kernel_source_sha256=sha256_file(Path(__file__)))


def _hot_cold_comparison(chain_rows):
    result = {}
    for name in ("energy_density", "magnetization_squared", "absolute_magnetization"):
        hot = np.asarray([row["diagnostics"][name]["mean"] for row in chain_rows if row["initial"] == "hot"])
        cold = np.asarray([row["diagnostics"][name]["mean"] for row in chain_rows if row["initial"] == "cold"])
        difference = float(hot.mean() - cold.mean())
        # Descriptive between-chain SE; never replace physical chains by saved draws.
        se = None if min(len(hot), len(cold)) < 2 else float(np.sqrt(hot.var(ddof=1) / len(hot) + cold.var(ddof=1) / len(cold)))
        result[name] = dict(hot_chain_means=hot.tolist(), cold_chain_means=cold.tolist(),
            hot_minus_cold=difference, independent_chain_mean_standard_error=se,
            standardized_difference=None if se is None or se < 1e-15 else difference / se,
            descriptive_three_se_warning=bool(se is not None and abs(difference) > max(3 * se, 1e-14)),
            equilibrium_certified=False)
    return result


def generate_dataset(output, options):
    """Create a distinct physical root; never resume, overwrite, or add samples."""
    output = Path(output).resolve()
    plan = generation_plan(options)
    output.mkdir(parents=True, exist_ok=False)
    (output / "conditions").mkdir()
    write_json(output / "plan.json", plan)
    wall, cpu = time.perf_counter(), time.process_time()
    shards, conditions, warnings = [], [], []
    total_visited, current = 0, None
    try:
        for i, coupling in enumerate(options.couplings):
            current = f"c{i:03d}"
            condition = output / "conditions" / current
            condition.mkdir()
            arrays, raws, clusters, flips, chain_rows = [], [], [], [], []
            for chain in range(options.chains):
                initial = "hot" if chain < options.chains // 2 else "cold"
                cold_sign = 1 if (chain - options.chains // 2) % 2 == 0 else -1
                status = dict(schema=SCHEMA, status="running", workflow_complete=False,
                    current=current, coupling_index=i, K=float(coupling), chain_index=chain,
                    completed_conditions=len(shards), completed_chains_in_condition=chain,
                    error=None, requires_user_action=False, equilibrium_certified=False)
                write_json(output / "status.json", status)
                write_json(condition / "status.json", status)
                thermal = chain_seed(options, i, chain, "thermal")
                augment = chain_seed(options, i, chain, "global_z2")
                result = sample_wolff_chain(coupling, size=options.size, samples=options.samples,
                    burn_in_clusters=options.burn_in_clusters, clusters_between=options.clusters_between,
                    seed=thermal, initial=initial, cold_sign=cold_sign,
                    augmentation_seed=augment if options.augment_global_z2 else None)
                arrays.append(result.pop("configurations"))
                raws.append(result.pop("raw_observables"))
                clusters.append(result.pop("cluster_sizes"))
                flips.append(result.pop("augmentation_flips"))
                row = dict(chain_index=chain, initial=initial, cold_sign=cold_sign,
                    thermal_seed=thermal, augmentation_seed=augment, **result)
                chain_rows.append(row)
                total_visited += result["cluster_visited_sites"]
                warnings.extend(dict(coupling_index=i, chain_index=chain, **warning) for warning in result["warnings"])
                print(json.dumps(dict(event="physical_chain_complete", condition=current,
                    chain_index=chain, wall_seconds=result["wall_seconds"], cluster_proposals=result["cluster_proposals"]), allow_nan=False), flush=True)
            data_path = condition / "data.npz"
            np.savez(data_path, configurations=np.stack(arrays), raw_observables=np.stack(raws),
                cluster_sizes=np.stack(clusters), augmentation_flips=np.stack(flips))
            split = "test" if i % options.test_every == options.test_offset else "train"
            sidecar = dict(schema=SCHEMA, status="complete", coupling_index=i, K=float(coupling),
                split=split, size=options.size, dimension=3, chains=chain_rows,
                samples_per_chain=options.samples, data_sha256=sha256_file(data_path),
                raw_observable_names=list(result["raw_observable_names"]),
                raw_diagnostics_before_global_z2=True, global_z2_augmentation=options.augment_global_z2,
                hot_cold_comparison=_hot_cold_comparison(chain_rows), equilibrium_certified=False,
                qualification=QUALIFICATION)
            warnings.extend(dict(coupling_index=i, observable=name,
                reason="Descriptive hot/cold chain-mean difference exceeds 3 between-chain SE; not a formal equilibrium test")
                for name, item in sidecar["hot_cold_comparison"].items() if item["descriptive_three_se_warning"])
            write_json(condition / "sidecar.json", sidecar)
            write_json(condition / "status.json", dict(status="complete", workflow_complete=True,
                returncode=0, error=None, equilibrium_certified=False))
            shards.append(dict(coupling_index=i, K=float(coupling), split=split,
                path=str(data_path.relative_to(output)), sha256=sidecar["data_sha256"],
                sidecar=str((condition / "sidecar.json").relative_to(output)),
                sidecar_sha256=sha256_file(condition / "sidecar.json"), bytes=data_path.stat().st_size))
            conditions.append(dict(coupling_index=i, K=float(coupling), split=split,
                hot_cold_comparison=sidecar["hot_cold_comparison"], chains=chain_rows))
        sampling = dict(schema=SCHEMA, conditions=conditions, warnings=warnings, warning_count=len(warnings),
            total_cluster_proposals=plan["total_cluster_proposals"], cluster_visited_sites=total_visited,
            neighbor_inspections=6 * total_visited, wall_seconds=time.perf_counter() - wall,
            cpu_seconds=time.process_time() - cpu, equilibrium_certified=False, qualification=QUALIFICATION)
        write_json(output / "sampling.json", sampling)
        manifest = dict(schema=SCHEMA, status="complete", options=asdict(options), dimension=3,
            plan_sha256=sha256_file(output / "plan.json"), sampling_sha256=sha256_file(output / "sampling.json"),
            kernel_source_sha256=plan["kernel_source_sha256"], shards=shards, equilibrium_certified=False)
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", dict(schema=SCHEMA, status="complete", workflow_complete=True,
            completed_conditions=len(shards), current=None, returncode=0, error=None, requires_user_action=False,
            manifest_sha256=sha256_file(output / "manifest.json"), equilibrium_certified=False,
            wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu))
        return manifest
    except Exception as exc:
        failure = dict(schema=SCHEMA, status="failed", workflow_complete=False,
            current=current, completed_conditions=len(shards), error=f"{type(exc).__name__}: {exc}",
            requires_user_action=True, equilibrium_certified=False)
        write_json(output / "status.json", failure)
        if current is not None and (output / "conditions" / current).is_dir():
            write_json(output / "conditions" / current / "status.json", failure)
        raise
