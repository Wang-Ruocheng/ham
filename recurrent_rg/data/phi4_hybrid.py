"""Fixed HMC plus embedded-Ising Wolff updates for the h=0 scalar theory.

This is a new sampler, not a change to historical ``data.phi4``.  For fixed
rho_i=abs(phi_i), write phi_i=rho_i*s_i.  The nearest-neighbor term is
sum_edges(.5*rho_i**2+.5*rho_j**2-rho_i*rho_j*s_i*s_j), and the even on-site
potential does not depend on s.  Thus the conditional signs are a zero-field
ferromagnetic Ising model with J_ij=rho_i*rho_j.  A uniformly seeded Wolff
cluster joins equal signs with p_ij=1-exp(-2*J_ij), then flips every cluster
sign.  This kernel satisfies conditional detailed balance; amplitudes are
unchanged.  HMC updates amplitudes with its own Metropolis correction.

References: Brower and Tamayo, "Embedded dynamics for phi4 theory", PRL 62,
1087 (1989), https://doi.org/10.1103/PhysRevLett.62.1087; Wolff, "Collective
Monte Carlo Updating for Spin Systems", PRL 62, 361 (1989),
https://doi.org/10.1103/PhysRevLett.62.361.  The exact factor of two above is
derived from this repository's once-per-forward-edge action, not imported from
another convention.

Every fixed hybrid cycle is one HMC proposal (including a possible rejection)
followed by the fixed lazy kernel (1-p)I+pW, with p=.5 by default.  This
prevents deterministic full-cluster alternation from aliasing even save spacing.
The laziness is fixed before sampling, not an outcome-driven adaptation.
Each kernel preserves exp(-S); their
fixed-order composition is stationary but is not claimed globally reversible.
There is no separate/global random Z2 augmentation.  Cluster flips can change
magnetization signs while even/amplitude modes remain slow, so sign changes
alone never certify equilibration.  Production never adapts or repeats seeds.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np

from . import phi4 as base


SCHEMA = "euclidean_phi4_hmc_embedded_wolff_v1"
OBSERVABLES = base.OBSERVABLES
QUALIFICATION = (
    "周期实标量场h=0；固定HMC-Metropolis后采用固定概率的lazy embedded-Ising Wolff单簇核。"
    "无额外全局随机Z2增强，不丢HMC拒绝态；簇核仅更新条件符号，HMC更新幅度。"
    "簇翻转造成磁化变号不证明幅度、能量或偶统计已热化；有限链ESS/热冷比较不认证平衡。"
    "生产参数不自适应，不自动补样、换seed或重跑。"
)


@lru_cache(maxsize=32)
def _neighbors(shape):
    lattice = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    result = np.stack([np.roll(lattice, shift, axis=axis).reshape(-1)
                       for axis in range(len(shape)) for shift in (-1, 1)], axis=1)
    result.setflags(write=False)
    return result


def bond_probability(left, right):
    """Activation probability for a single aligned edge; zero for unlike signs."""
    left, right = base._finite(left, "left"), base._finite(right, "right")
    if (left > 0) != (right > 0) or left == 0 or right == 0:
        return 0.
    # Python float overflow to inf is harmless here: -expm1(-inf)=1.
    return -math.expm1(-2. * abs(left) * abs(right))


def cluster_step(phi, rng):
    """Return ``(new_field, work_receipt)`` without mutating the input field.

    ``phi`` has 1..3 spatial axes, every side >=3, and no channel axis.  ``rng``
    is a NumPy Generator (or a deterministic test double with integers/random).
    No parameters are needed because every even on-site potential is invariant.
    This operation is only the h=0 kernel; use ``sample_hybrid_chain`` to check
    the global model's normalizability and source h before sampling.
    """
    phi = base._field(phi)
    if not callable(getattr(rng, "integers", None)) or not callable(getattr(rng, "random", None)):
        raise ValueError("rng must provide integers() and random()")
    flat = phi.reshape(-1)
    neighbors = _neighbors(phi.shape)
    seed = int(rng.integers(flat.size))
    if not 0 <= seed < flat.size:
        raise ValueError("rng returned an invalid seed site")
    marked = np.zeros(flat.size, dtype=np.bool_)
    marked[seed] = True
    cluster = [seed]
    cursor = visits = tests = activated = 0
    while cursor < len(cluster):
        current = cluster[cursor]
        cursor += 1
        value = float(flat[current])
        for adjacent in neighbors[current]:
            adjacent = int(adjacent)
            visits += 1
            if marked[adjacent]:
                continue
            other = float(flat[adjacent])
            if value == 0. or other == 0. or (value > 0.) != (other > 0.):
                continue
            tests += 1
            probability = -math.expm1(-2. * abs(value) * abs(other))
            if float(rng.random()) < probability:
                marked[adjacent] = True
                cluster.append(adjacent)
                activated += 1
    result = phi.copy()
    result.reshape(-1)[marked] *= -1.
    changed = int(np.count_nonzero(flat[marked]))
    return result, dict(seed_site=seed, cluster_size=len(cluster),
        changed_sites=changed, neighbor_visits=visits, bond_tests=tests,
        bond_random_draws=tests, activated_bonds=activated,
        amplitude_changed=False, always_accepted_conditional_sign_update=True)


def sample_hybrid_chain(r, u, *, size, dimension=2, samples, burn_in, spacing,
                        step_size, leapfrog_steps, seed, initial="hot", h=0.,
                        cluster_probability=.5):
    """HMC-compatible output, plus actual cluster work and raw saved diagnostics.

    ``spacing`` counts fixed hybrid cycles, never local sweeps or clusters per
    site. ``accepted`` and ``delta_h`` describe HMC only. Saved configurations
    are after the lazy cluster kernel in the selected cycles, including cycles
    whose HMC proposal was rejected. The cluster attempt decision is independent
    of HMC acceptance. p=1 requests the original non-lazy Wolff composition;
    p=0 is an explicitly requested HMC-only control, not the production default.
    """
    r, u, h = base._parameters(r, u, h)
    if h != 0.:
        raise ValueError("Embedded sign-cluster sampler requires h=0; ghost-spin extension is not implemented")
    size = base._integer(size, "size", 3)
    dimension = base._integer(dimension, "dimension", 1)
    if dimension > 3:
        raise ValueError("dimension must be 1, 2, or 3")
    samples = base._integer(samples, "samples", 4)
    burn_in = base._integer(burn_in, "burn_in", 0)
    spacing = base._integer(spacing, "spacing", 1)
    steps = base._integer(leapfrog_steps, "leapfrog_steps", 1)
    epsilon = base._finite(step_size, "step_size")
    if epsilon <= 0:
        raise ValueError("step_size must be positive")
    probability = base._finite(cluster_probability, "cluster_probability")
    if not 0. <= probability <= 1.:
        raise ValueError("cluster_probability must be in [0,1]; zero is an explicit HMC-only control")
    seed = base._integer(seed, "seed", 0)
    if initial not in ("hot", "cold_positive", "cold_negative"):
        raise ValueError("Unknown initial state")
    shape = (size,) * dimension
    rng = np.random.default_rng(seed)
    if initial == "hot":
        phi = rng.normal(size=shape)
    else:
        amplitude = math.sqrt(-r / u) if r < 0 and u > 0 else 1.
        phi = np.full(shape, amplitude if initial == "cold_positive" else -amplitude)
    cycles = burn_in + samples * spacing
    configurations = np.empty((samples, *shape, 1), dtype=np.float64)
    raw = np.empty((samples, len(OBSERVABLES)), dtype=np.float64)
    accepted = np.zeros(cycles, dtype=np.bool_)
    delta_h = np.empty(cycles, dtype=np.float64)
    cluster_sizes = np.empty(cycles, dtype=np.int64)
    cluster_changed_sites = np.empty(cycles, dtype=np.int64)
    cluster_neighbor_visits = np.empty(cycles, dtype=np.int64)
    cluster_bond_tests = np.empty(cycles, dtype=np.int64)
    cluster_activated_bonds = np.empty(cycles, dtype=np.int64)
    cluster_attempted = np.zeros(cycles, dtype=np.bool_)
    trajectory_magnetization = np.empty(cycles, dtype=np.float64)
    wall, cpu = time.perf_counter(), time.process_time()
    for cycle in range(cycles):
        momentum = rng.normal(size=shape)
        before = base._action_only(phi, r, u, h) + .5 * np.sum(momentum ** 2)
        proposal, final_momentum = base._leapfrog(phi, momentum, r, u, h, epsilon, steps)
        after = base._action_only(proposal, r, u, h) + .5 * np.sum(final_momentum ** 2)
        difference = float(after - before)
        if not math.isfinite(difference):
            raise FloatingPointError("Nonfinite HMC Hamiltonian difference; fixed hybrid parameters were not changed")
        delta_h[cycle] = difference
        if math.log(max(float(rng.random()), np.finfo(float).tiny)) < -difference:
            phi = proposal
            accepted[cycle] = True
        attempt = probability == 1. or (probability > 0. and float(rng.random()) < probability)
        cluster_attempted[cycle] = attempt
        if attempt:
            phi, receipt = cluster_step(phi, rng)
            cluster_sizes[cycle] = receipt["cluster_size"]
            cluster_changed_sites[cycle] = receipt["changed_sites"]
            cluster_neighbor_visits[cycle] = receipt["neighbor_visits"]
            cluster_bond_tests[cycle] = receipt["bond_tests"]
            cluster_activated_bonds[cycle] = receipt["activated_bonds"]
        else:
            cluster_sizes[cycle] = cluster_changed_sites[cycle] = cluster_neighbor_visits[cycle] = 0
            cluster_bond_tests[cycle] = cluster_activated_bonds[cycle] = 0
        trajectory_magnetization[cycle] = float(phi.mean())
        production = cycle - burn_in + 1
        if production > 0 and production % spacing == 0:
            index = production // spacing - 1
            configurations[index, ..., 0] = phi
            raw[index] = base._observations(phi, r, u, h)
    diagnostics = {name: base.series_diagnostics(raw[:, index])
                   for index, name in enumerate(OBSERVABLES)}
    warnings = [f"{name}: {row.get('warning') or 'ESS below 32 saved draws'}"
                for name, row in diagnostics.items()
                if row.get("warning") or row.get("ess") is None or row["ess"] < 32]
    rate = float(accepted[burn_in:].mean())
    if rate < .6:
        warnings.append("Production HMC acceptance below 0.6; cluster flips do not repair amplitude rejection")
    changes = int(np.count_nonzero(np.sign(raw[1:, 1]) != np.sign(raw[:-1, 1])))
    if changes == 0:
        warnings.append("No saved magnetization sign change; symmetry-sector mixing unverified")
    return dict(schema=SCHEMA, configurations=configurations, raw_observables=raw,
        accepted=accepted, delta_h=delta_h, raw_observable_names=list(OBSERVABLES),
        diagnostics=diagnostics, production_acceptance=rate,
        burn_in_acceptance=float(accepted[:burn_in].mean()) if burn_in else None,
        max_absolute_delta_h=float(np.max(np.abs(delta_h))),
        saved_magnetization_sign_changes=changes, hmc_trajectories=cycles,
        hybrid_cycles=cycles, wolff_cluster_updates=int(cluster_attempted.sum()),
        cluster_probability=probability, cluster_schedule="fixed_lazy_(1-p)I+pW_after_each_HMC",
        hmc_only_control=probability == 0.,
        force_evaluations=cycles * (steps + 1), force_site_evaluations=cycles * (steps + 1) * phi.size,
        cluster_sizes=cluster_sizes, cluster_changed_sites=cluster_changed_sites,
        cluster_neighbor_visits=cluster_neighbor_visits, cluster_bond_tests=cluster_bond_tests,
        cluster_activated_bonds=cluster_activated_bonds,
        cluster_attempted=cluster_attempted, trajectory_magnetization=trajectory_magnetization,
        total_cluster_sites=int(cluster_sizes.sum()), total_cluster_changed_sites=int(cluster_changed_sites.sum()),
        total_cluster_neighbor_visits=int(cluster_neighbor_visits.sum()),
        total_cluster_bond_tests=int(cluster_bond_tests.sum()),
        total_cluster_activated_bonds=int(cluster_activated_bonds.sum()),
        mean_production_cluster_fraction=float(cluster_sizes[burn_in:].mean() / phi.size),
        mean_attempted_cluster_fraction=(float(cluster_sizes[cluster_attempted].mean() / phi.size)
                                         if cluster_attempted.any() else None),
        initial=initial, seed=seed, warnings=warnings, equilibrium_certified=False,
        symmetry_augmentation=False, artificial_global_sign_flips=0,
        production_adaptation=False, qualification=QUALIFICATION,
        wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)
