"""Independent Euclidean scalar-field data; deliberately not a binary QCR cache.

S = 1/2 sum_(x,a) (phi[x+e_a]-phi[x])**2
    + sum_x (r phi[x]**2/2 + u phi[x]**4/4 - h phi[x]).
Each forward periodic edge is counted once, lattice spacing and k_B T are one.
HMC uses fresh unit Gaussian momenta, reversible leapfrog, and a Metropolis
correction. No adaptation, clipping, symmetry augmentation, or automatic retry
is performed. Finite-chain diagnostics cannot certify equilibration.
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

from .spin_glass_pipeline import series_diagnostics


SCHEMA = "euclidean_phi4_hmc_v1"
OBSERVABLES = (
    "action_density", "magnetization", "absolute_magnetization",
    "magnetization_squared", "magnetization_fourth", "mean_field_squared",
    "structure_factor_zero", "structure_factor_minimum_momentum",
)
QUALIFICATION = (
    "实标量欧氏周期格点场，正概率测度 exp(-S)；固定参数 HMC，非实时路径积分。"
    "有限相关链；接受率、自相关、ESS 和热/冷初态比较不认证平衡。"
    "不自动延长、换种子或重跑；不把势阱、双峰或慢隧穿当作相变。"
    "r/u/h 只用于生成与评价；构型分片不含耦合训练标签。"
    "连续实场不能直接交给当前二值自旋 QCR 训练入口。"
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


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _parameters(r, u, h):
    r, u, h = (_finite(value, name) for value, name in ((r, "r"), (u, "u"), (h, "h")))
    if u < 0 or (u == 0 and r <= 0):
        raise ValueError("Require u>0, or the normalizable Gaussian limit u=0 with r>0")
    return r, u, h


def _field(phi):
    if np.iscomplexobj(phi):
        raise ValueError("Require a real scalar field")
    phi = np.asarray(phi, dtype=np.float64)
    if not 1 <= phi.ndim <= 3 or any(side < 3 for side in phi.shape):
        raise ValueError("Require 1..3 spatial dimensions, each of side >=3, without channel axis")
    if not np.isfinite(phi).all():
        raise ValueError("Field contains nonfinite values")
    return phi


def _action_only(phi, r, u, h):
    with np.errstate(over="raise", invalid="raise"):
        square = phi * phi
        action = np.sum(.5 * r * square + .25 * u * square * square - h * phi)
        for axis in range(phi.ndim):
            forward = np.roll(phi, -1, axis=axis)
            action += .5 * np.sum((phi - forward) ** 2)
    if not np.isfinite(action):
        raise FloatingPointError("Nonfinite action; fixed HMC parameters were not changed")
    return float(action)


def _gradient(phi, r, u, h):
    with np.errstate(over="raise", invalid="raise"):
        gradient = r * phi + u * phi ** 3 - h
        for axis in range(phi.ndim):
            forward = np.roll(phi, -1, axis=axis)
            gradient = gradient + 2 * phi - forward - np.roll(phi, 1, axis=axis)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite force; fixed HMC parameters were not changed")
    return gradient


def _action_gradient(phi, r, u, h):
    return _action_only(phi, r, u, h), _gradient(phi, r, u, h)


def action_and_gradient(phi, r, u, h=0.):
    """Action and exact field derivative; input is never modified."""
    return _action_gradient(_field(phi), *_parameters(r, u, h))


def leapfrog(phi, momentum, r, u, step_size, n_steps, h=0.):
    """Return unflipped (q,p); time reversal uses (q,-p). No in-place mutation."""
    phi = _field(phi)
    if np.iscomplexobj(momentum):
        raise ValueError("Momentum must be real")
    momentum = np.asarray(momentum, dtype=np.float64)
    if momentum.shape != phi.shape or not np.isfinite(momentum).all():
        raise ValueError("Momentum must be finite and match the field")
    r, u, h = _parameters(r, u, h)
    epsilon = _finite(step_size, "step_size")
    if epsilon <= 0:
        raise ValueError("step_size must be positive")
    steps = _integer(n_steps, "n_steps", 1)
    return _leapfrog(phi, momentum, r, u, h, epsilon, steps)


def _leapfrog(phi, momentum, r, u, h, epsilon, steps):
    q, p = phi.copy(), momentum.copy()
    force = _gradient(q, r, u, h)
    p -= .5 * epsilon * force
    for step in range(steps):
        q += epsilon * p
        force = _gradient(q, r, u, h)
        p -= (.5 if step == steps - 1 else 1.) * epsilon * force
    return q, p


def _observations(phi, r, u, h):
    volume = phi.size
    m = float(phi.mean())
    transformed = np.fft.fftn(phi)
    minimum = []
    for axis in range(phi.ndim):
        index = [0] * phi.ndim
        index[axis] = 1
        minimum.append(float(abs(transformed[tuple(index)]) ** 2 / volume))
    return np.array((
        _action_only(phi, r, u, h) / volume, m, abs(m), m*m, m**4,
        np.mean(phi * phi), abs(transformed.flat[0]) ** 2 / volume,
        np.mean(minimum),
    ), dtype=np.float64)


def sample_hmc_chain(r, u, *, size, dimension=2, samples, burn_in, spacing,
                     step_size, leapfrog_steps, seed, initial="hot", h=0.):
    """One independent chain; work unit is an HMC trajectory, never a sweep."""
    r, u, h = _parameters(r, u, h)
    size = _integer(size, "size", 3)
    dimension = _integer(dimension, "dimension", 1)
    if dimension > 3:
        raise ValueError("dimension must be 1, 2, or 3")
    samples = _integer(samples, "samples", 4)
    burn_in = _integer(burn_in, "burn_in", 0)
    spacing = _integer(spacing, "spacing", 1)
    steps = _integer(leapfrog_steps, "leapfrog_steps", 1)
    epsilon = _finite(step_size, "step_size")
    if epsilon <= 0:
        raise ValueError("step_size must be positive")
    seed = _integer(seed, "seed", 0)
    if initial not in ("hot", "cold_positive", "cold_negative"):
        raise ValueError("Unknown initial state")
    shape = (size,) * dimension
    rng = np.random.default_rng(seed)
    if initial == "hot":
        phi = rng.normal(size=shape)
    else:
        amplitude = math.sqrt(-r / u) if r < 0 and u > 0 else 1.
        phi = np.full(shape, amplitude if initial == "cold_positive" else -amplitude)
    trajectories = burn_in + samples * spacing
    configurations = np.empty((samples, *shape, 1), dtype=np.float64)
    raw = np.empty((samples, len(OBSERVABLES)), dtype=np.float64)
    accepted = np.zeros(trajectories, dtype=np.bool_)
    delta_h = np.empty(trajectories, dtype=np.float64)
    wall, cpu = time.perf_counter(), time.process_time()
    for trajectory in range(trajectories):
        momentum = rng.normal(size=shape)
        before = _action_only(phi, r, u, h) + .5 * np.sum(momentum ** 2)
        proposal, final_momentum = _leapfrog(phi, momentum, r, u, h, epsilon, steps)
        after = _action_only(proposal, r, u, h) + .5 * np.sum(final_momentum ** 2)
        difference = float(after - before)
        if not math.isfinite(difference):
            raise FloatingPointError("Nonfinite HMC Hamiltonian difference")
        delta_h[trajectory] = difference
        # Final momentum flip is implicit: kinetic energy is even and momenta
        # are fully refreshed. Rejected states remain in the saved chain.
        if math.log(max(float(rng.random()), np.finfo(float).tiny)) < -difference:
            phi = proposal
            accepted[trajectory] = True
        production = trajectory - burn_in + 1
        if production > 0 and production % spacing == 0:
            index = production // spacing - 1
            configurations[index, ..., 0] = phi
            raw[index] = _observations(phi, r, u, h)
    diagnostics = {name: series_diagnostics(raw[:, i]) for i, name in enumerate(OBSERVABLES)}
    warnings = [f"{name}: {row.get('warning') or 'ESS below 32 saved draws'}"
                for name, row in diagnostics.items()
                if row.get("warning") or row.get("ess") is None or row["ess"] < 32]
    rate = float(accepted[burn_in:].mean())
    if rate < .6:
        warnings.append("Production HMC acceptance below 0.6; no automatic step-size adjustment")
    changes = int(np.count_nonzero(np.sign(raw[1:, 1]) != np.sign(raw[:-1, 1])))
    if h == 0 and changes == 0:
        warnings.append("No saved magnetization sign change; symmetry-sector mixing unverified")
    return dict(configurations=configurations, raw_observables=raw, accepted=accepted,
        delta_h=delta_h, raw_observable_names=list(OBSERVABLES), diagnostics=diagnostics,
        production_acceptance=rate, burn_in_acceptance=float(accepted[:burn_in].mean()) if burn_in else None,
        max_absolute_delta_h=float(np.max(np.abs(delta_h))),
        saved_magnetization_sign_changes=changes, hmc_trajectories=trajectories,
        force_evaluations=trajectories * (steps + 1), initial=initial, seed=seed,
        warnings=warnings, equilibrium_certified=False, symmetry_augmentation=False,
        wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)


@dataclass(frozen=True)
class Phi4Options:
    sizes: tuple[int, ...] = (36,)
    dimension: int = 2
    r_values: tuple[float, ...] = tuple(-2. + .25 * i for i in range(11))
    u_values: tuple[float, ...] = (1.,)
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
    max_trajectories: int = 2_000_000
    max_force_site_evaluations: int = 2_000_000_000
    max_output_bytes: int = 2 * 1024**3
    max_memory_bytes: int = 512 * 1024**2

    def validate(self):
        for name, minimum in (("dimension", 1), ("chains", 4), ("samples", 4),
                ("burn_in", 0), ("spacing", 1), ("leapfrog_steps", 1), ("seed", 0),
                ("test_every", 2), ("test_offset", 0), ("max_trajectories", 1),
                ("max_force_site_evaluations", 1), ("max_output_bytes", 1), ("max_memory_bytes", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.dimension > 3 or self.chains % 4:
            raise ValueError("Require dimension=1..3 and chains divisible by 4 for balanced starts")
        if not isinstance(self.sizes, (list, tuple)) or not self.sizes:
            raise ValueError("sizes must be a nonempty sequence")
        for side in self.sizes:
            _integer(side, "size", 3)
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("Duplicate sizes")
        for name in ("r_values", "u_values"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            normalized = [_finite(value, name) for value in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"Duplicate {name}")
        for r in self.r_values:
            for u in self.u_values:
                _parameters(r, u, self.h)
        if _finite(self.step_size, "step_size") <= 0:
            raise ValueError("step_size must be positive")
        if self.test_offset >= self.test_every:
            raise ValueError("test_offset must be smaller than test_every")
        cells = len(self.r_values) * len(self.u_values)
        tests = (1 + (cells - 1 - self.test_offset) // self.test_every
                 if self.test_offset < cells else 0)
        if not 0 < tests < cells:
            raise ValueError("Whole-parameter-cell split must have both train and test")
        return self


def generation_plan(options):
    """Pure preflight; budgets are checked before RNG allocation or filesystem writes."""
    options.validate()
    cells = len(options.r_values) * len(options.u_values)
    per_chain = options.burn_in + options.samples * options.spacing
    volumes = [int(size) ** options.dimension for size in options.sizes]
    trajectories = cells * len(volumes) * options.chains * per_chain
    forces = cells * options.chains * per_chain * (options.leapfrog_steps + 1) * sum(volumes)
    shard_bytes = [options.chains * (options.samples * (volume + len(OBSERVABLES)) * 8
                   + per_chain * 9) for volume in volumes]
    # Diagnostics scale with chain count, even for tiny/short physical arrays.
    # Also reserve Python/JSON objects for the retained condition receipts.
    diagnostic_disk_per_condition = options.chains * 32768
    output_bytes = cells * sum(value + diagnostic_disk_per_condition + 262144
                               for value in shard_bytes) + 262144
    memory_bytes = (3 * max(shard_bytes) + 64 * max(volumes) + options.chains * 65536
                    + 4096 * cells * len(volumes) + 64 * cells + 4 * 1024**2)
    for value, cap, label in ((trajectories, options.max_trajectories, "HMC trajectory"),
            (forces, options.max_force_site_evaluations, "force-site evaluation"),
            (output_bytes, options.max_output_bytes, "output byte"),
            (memory_bytes, options.max_memory_bytes, "working-memory byte")):
        if value > cap:
            raise ValueError(f"Projected {label} budget exceeded: {value} > {cap}")
    return dict(schema=SCHEMA, options=asdict(options), dimension=options.dimension,
        action="S=.5 sum_(x,a)(phi[x+e_a]-phi[x])^2+sum_x(.5*r*phi^2+.25*u*phi^4-h*phi)",
        lattice_spacing=1., probability="exp(-S)/Z", dtype="float64", boundary="periodic",
        sampler="fixed-parameter HMC; fresh Gaussian momenta, leapfrog, Metropolis correction",
        parameter_cells=cells, conditions=cells * len(volumes),
        trajectories_per_chain=per_chain, total_hmc_trajectories=trajectories,
        total_force_site_evaluations=forces, projected_output_bytes=output_bytes,
        projected_working_bytes=memory_bytes, work_estimate_is_not_wall_time=True,
        split_unit="entire (r,u,h) cell, including every size and independent chain",
        train_cell_indices=[i for i in range(cells) if i % options.test_every != options.test_offset],
        test_cell_indices=[i for i in range(cells) if i % options.test_every == options.test_offset],
        initialization="half hot, quarter cold positive, quarter cold negative; no augmentation",
        axes=["independent_chain", "saved_sample", "*spatial", "channel=1"],
        couplings_in_configuration_arrays=False, labels_used_in_training=False, training_updates=0,
        current_binary_qcr_compatible=False, equilibrium_certified=False, qualification=QUALIFICATION,
        source_sha256=sha256_file(Path(__file__)))


def chain_seed(options, size, cell, chain):
    """Stable independent streams, keyed by physical parameters rather than list order."""
    r = options.r_values[cell // len(options.u_values)]
    u = options.u_values[cell % len(options.u_values)]
    message = [SCHEMA, int(options.seed), int(options.dimension), int(size),
               float(r), float(u), float(options.h), int(chain)]
    return int.from_bytes(hashlib.sha256(json.dumps(message).encode()).digest()[:16], "little")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _condition_summary(raw, rows, size, dimension):
    means = raw.mean(axis=(0, 1))
    volume = size ** dimension
    binder = 1. - float(means[4] / (3 * means[3] ** 2)) if means[3] > 0 else None
    comparison = {}
    for name in ("action_density", "absolute_magnetization", "magnetization_squared"):
        hot = [row["diagnostics"][name]["mean"] for row in rows if row["initial"] == "hot"]
        cold = [row["diagnostics"][name]["mean"] for row in rows if row["initial"] != "hot"]
        se = math.sqrt(np.var(hot, ddof=1) / len(hot) + np.var(cold, ddof=1) / len(cold))
        difference = float(np.mean(hot) - np.mean(cold))
        comparison[name] = dict(hot_chain_means=hot, cold_chain_means=cold,
            hot_minus_cold=difference, between_chain_standard_error=se,
            descriptive_three_se_warning=bool(abs(difference) > max(3 * se, 1e-14)))
    return dict(binder_1_minus_m4_over_3m2sq=binder,
        susceptibility_connected=volume * max(0., float(means[3] - means[1] ** 2)),
        structure_factor_zero=float(means[6]), structure_factor_minimum_momentum=float(means[7]),
        hot_cold_comparison=comparison, equilibrium_certified=False,
        qualification="诊断性有限链点估计，无误差条；不构成相变位置或热化认证。")


def generate_dataset(output, options):
    """New physical dataset only; stream conditions, never overwrite or resume."""
    plan = generation_plan(options)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    (output / "conditions").mkdir()
    wall, cpu = time.perf_counter(), time.process_time()
    shards, warning_count, current = [], 0, None
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
                    progress = dict(schema=SCHEMA, status="running", workflow_complete=False,
                        current=current, chain_index=chain, completed_conditions=len(shards),
                        error=None, requires_user_action=False, equilibrium_certified=False)
                    _write_json(output / "status.json", progress)
                    _write_json(directory / "status.json", progress)
                    result = sample_hmc_chain(r, u, h=options.h, size=size,
                        dimension=options.dimension, samples=options.samples, burn_in=options.burn_in,
                        spacing=options.spacing, step_size=options.step_size,
                        leapfrog_steps=options.leapfrog_steps,
                        seed=chain_seed(options, size, cell, chain), initial=initial)
                    arrays.append({name: result.pop(name) for name in
                        ("configurations", "raw_observables", "accepted", "delta_h")})
                    rows.append(dict(chain_index=chain, **result))
                    warning_count += len(result["warnings"])
                    print(json.dumps(dict(event="phi4_chain_complete", condition=current,
                        chain_index=chain, production_acceptance=result["production_acceptance"],
                        warnings=len(result["warnings"])), allow_nan=False), flush=True)
                data_path = directory / "data.npz"
                stacked = {name: np.stack([array[name] for array in arrays]) for name in arrays[0]}
                np.savez(data_path, **stacked)
                summary = _condition_summary(stacked["raw_observables"], rows, size, options.dimension)
                warning_count += sum(item["descriptive_three_se_warning"]
                                     for item in summary["hot_cold_comparison"].values())
                sidecar = dict(schema=SCHEMA, status="complete", cell_index=cell, size=size,
                    dimension=options.dimension, r=r, u=u, h=float(options.h), split=split,
                    chains=rows, raw_observable_names=list(OBSERVABLES), summary=summary,
                    independent_chains=options.chains, samples_per_chain=options.samples,
                    data_sha256=sha256_file(data_path), symmetry_augmentation=False,
                    labels_used_in_training=False, equilibrium_certified=False,
                    qualification=QUALIFICATION)
                _write_json(directory / "sidecar.json", sidecar)
                _write_json(directory / "status.json", dict(status="complete", workflow_complete=True,
                    returncode=0, error=None, equilibrium_certified=False))
                shards.append(dict(cell_index=cell, size=size, split=split,
                    path=str(data_path.relative_to(output)), sha256=sidecar["data_sha256"],
                    sidecar=str((directory / "sidecar.json").relative_to(output)),
                    sidecar_sha256=sha256_file(directory / "sidecar.json"), bytes=data_path.stat().st_size))
                del arrays, stacked
        manifest = dict(schema=SCHEMA, status="complete", options=asdict(options), shards=shards,
            plan_sha256=sha256_file(output / "plan.json"), source_sha256=plan["source_sha256"],
            total_hmc_trajectories=plan["total_hmc_trajectories"], warning_count=int(warning_count),
            labels_used_in_training=False, current_binary_qcr_compatible=False,
            equilibrium_certified=False, qualification=QUALIFICATION)
        _write_json(output / "manifest.json", manifest)
        _write_json(output / "status.json", dict(schema=SCHEMA, status="complete", workflow_complete=True,
            current=None, completed_conditions=len(shards), returncode=0, error=None,
            requires_user_action=False, manifest_sha256=sha256_file(output / "manifest.json"),
            warning_count=int(warning_count), equilibrium_certified=False,
            wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu))
        return manifest
    except Exception as exc:
        failure = dict(schema=SCHEMA, status="failed", workflow_complete=False, returncode=1,
            current=current, completed_conditions=len(shards), error=f"{type(exc).__name__}: {exc}",
            requires_user_action=True, equilibrium_certified=False)
        _write_json(output / "status.json", failure)
        if current is not None:
            _write_json(output / "conditions" / current / "status.json", failure)
        raise
