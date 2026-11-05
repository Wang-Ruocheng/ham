"""Nonzero-field hopping phi4: full-target HMC plus field-corrected Wolff.

The zero-field Wolff proposal q satisfies detailed balance for exp(-S0).
For S=S0-h*sum(phi), q(x,y)/q(y,x) cancels S0 in MH. Thus
a(x,y)=min(1,exp(h*(sum(y)-sum(x)))).
In psi=sqrt(kappa)*phi coordinates h_psi=h/sqrt(kappa).
Rejected cluster proposals remain in the chain. No sign augmentation.
The fixed composition is stationary, not necessarily globally reversible.
Historical h=0 samplers remain unchanged. At h=0 the extra MH draw is omitted.
"""
from __future__ import annotations
import math
import time
import numpy as np
from . import phi4 as base
from . import phi4_hybrid as hybrid
from . import phi4_hopping as hopping

SCHEMA = "euclidean_phi4_kappa_h_hmc_field_corrected_wolff_v1"
OBSERVABLES = base.OBSERVABLES
ACTION = "S=-kappa*sum_forward_edges(phi_x*phi_y)+sum_x(phi_x^2+lambda*(phi_x^2-1)^2-h*phi_x)"
QUALIFICATION = ("非零外场完整目标HMC＋零场Wolff提议的外场MH校正；拒绝态保留。"
    "每条近邻边只计一次，保存原始物理phi单位；无正负对称增强、无自适应、无补样。"
    "有限链ESS/热冷比较不认证平衡；非零外场不要求磁化反复变号。")

def gradient_parameters(kappa, quartic_lambda, dimension, h=0.):
    mapping = hopping.gradient_parameters(kappa, quartic_lambda, dimension)
    h = base._finite(h, "h")
    mapping.update(h=h / mapping["field_scale"], physical_h=h)
    return mapping

def action_and_gradient(phi, kappa, quartic_lambda, h=0.):
    h = base._finite(h, "h")
    value, grad = hopping.action_and_gradient(phi, kappa, quartic_lambda)
    return value-h*np.sum(phi), grad-h

def field_cluster_step(phi, h, rng):
    h = base._finite(h, "h")
    proposed, receipt = hybrid.cluster_step(phi, rng)
    log_accept = min(0., float(h*np.sum(proposed-phi)))
    accepted = (h == 0. or math.log(max(float(rng.random()), np.finfo(float).tiny)) < log_accept)
    receipt.update(field_accepted=bool(accepted), field_log_acceptance=log_accept,
                   always_accepted_conditional_sign_update=(h == 0.))
    if not accepted:
        receipt["changed_sites"] = 0
    return (proposed if accepted else phi.copy()), receipt

def _sample_gradient_chain(r, u, *, size, dimension=2, samples, burn_in, spacing,
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
    field_accepted = np.zeros(cycles, dtype=np.bool_)
    field_log_acceptance = np.zeros(cycles)
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
            phi, receipt = field_cluster_step(phi, h, rng)
            field_accepted[cycle] = receipt["field_accepted"]
            field_log_acceptance[cycle] = receipt["field_log_acceptance"]
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
    if h == 0. and changes == 0:
        warnings.append("No saved magnetization sign change; symmetry-sector mixing unverified")
    return dict(schema=SCHEMA, field_accepted=field_accepted,
        field_log_acceptance=field_log_acceptance,
        cluster_field_acceptance=(float(field_accepted[cluster_attempted].mean()) if cluster_attempted.any() else None),
        configurations=configurations, raw_observables=raw,
        accepted=accepted, delta_h=delta_h, raw_observable_names=list(OBSERVABLES),
        diagnostics=diagnostics, production_acceptance=rate,
        burn_in_acceptance=float(accepted[:burn_in].mean()) if burn_in else None,
        max_absolute_delta_h=float(np.max(np.abs(delta_h))),
        saved_magnetization_sign_changes=changes, hmc_trajectories=cycles,
        hybrid_cycles=cycles, wolff_cluster_updates=int(cluster_attempted.sum()),
        cluster_probability=probability, cluster_schedule="fixed_lazy_(1-p)I+p_MH_corrected_W_after_each_HMC",
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

def sample_chain(kappa, quartic_lambda, *, h=0., **kwargs):
    dimension = kwargs.get("dimension", 2)
    mapping = gradient_parameters(kappa, quartic_lambda, dimension, h)
    result = _sample_gradient_chain(mapping["r"], mapping["u"], h=mapping["h"], **kwargs)
    result["configurations"] /= mapping["field_scale"]
    result["trajectory_magnetization"] /= mapping["field_scale"]
    result["raw_observables"] = hopping._physical_observations(result["raw_observables"], mapping)
    result["diagnostics"] = {name:base.series_diagnostics(result["raw_observables"][:,i])
                             for i,name in enumerate(OBSERVABLES)}
    result.update(mapping=mapping, action=ACTION, physical_field_coordinate="phi",
        integrator_coordinate="psi", symmetry_augmentation=False,
        lazy_cluster_skips=int(result["hybrid_cycles"]-result["wolff_cluster_updates"]))
    return result

