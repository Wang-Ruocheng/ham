"""Isolated, single-thread PT / Houdayer--PT comparison sampler.

This module deliberately does not change the historical sampler.  Both arms
use the same lazy, random-permutation single-site Metropolis kernel and the
same alternating adjacent-temperature exchange convention.  The extra kernel
flips a complete nonzero-bond connected component of ``q = s_a * s_b == -1`` in both real
replicas at the *same* temperature and disorder.  Its joint energy change is
zero, including an arbitrary uniform magnetic field.

Houdayer-coupled real replicas are not independent trajectories.  Independent
hot/cold groups must be separate calls with independently derived seeds.
Temperature walker labels survive exchanges, but after a cluster move they no
longer label intact configurations; their round trips are only transport
diagnostics, not an equilibration certificate.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .spin_glass import (
    _bonds,
    _energy_flat,
    _finite_scalar,
    _integer,
    _real_array,
    energy,
    make_periodic_lattice,
)


@njit(cache=True)
def _flip_disagreement_component(a, b, bonds, forward, backward, chosen_site, queue, marked):
    """Flip the whole disagreement component at a specified site in place."""
    marked[:] = False
    if a[chosen_site] == b[chosen_site]:
        return 0
    queue[0] = chosen_site
    marked[chosen_site] = True
    count = 1
    head = 0
    while head < count:
        site = queue[head]
        head += 1
        for axis in range(forward.shape[0]):
            neighbor = forward[axis, site]
            if (bonds[axis, site] != 0.0 and not marked[neighbor]
                    and a[neighbor] != b[neighbor]):
                marked[neighbor] = True
                queue[count] = neighbor
                count += 1
            neighbor = backward[axis, site]
            if (bonds[axis, neighbor] != 0.0 and not marked[neighbor]
                    and a[neighbor] != b[neighbor]):
                marked[neighbor] = True
                queue[count] = neighbor
                count += 1
    for index in range(count):
        site = queue[index]
        a[site] = -a[site]
        b[site] = -b[site]
    return count


def houdayer_cluster_update(spins_a, spins_b, bonds, *, seed, field=0.0) -> dict:
    """One non-mutating, auditable cluster update on a pair of 2D states.

    Select a site uniformly among disagreements, then flip its complete
    nonzero-bond nearest-neighbor component in both copies.  Zero bonds do not
    join components.  The overlap is unchanged, so
    the reverse proposal selects the same component with the same probability.
    At no disagreements the update is the identity.  This helper is for small
    correctness checks; the sampler calls the compiled kernel directly.
    """
    bonds, shape = _bonds(bonds)
    if len(shape) != 2:
        raise ValueError("Houdayer comparison currently requires a 2D lattice")
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    field = _finite_scalar(field, "field")
    a = _real_array(spins_a, "spins_a")
    b = _real_array(spins_b, "spins_b")
    if a.shape != shape or b.shape != shape:
        raise ValueError(f"both spin states must have lattice shape {shape}")
    if not np.all((a == -1) | (a == 1)) or not np.all((b == -1) | (b == 1)):
        raise ValueError("spins must contain only -1 and +1")
    a = np.asarray(a, dtype=np.int8).copy().ravel()
    b = np.asarray(b, dtype=np.int8).copy().ravel()
    before = np.asarray(energy(np.stack([a, b]).reshape(2, *shape), bonds, field))
    disagreements = np.flatnonzero(a != b)
    queue = np.empty(a.size, dtype=np.int64)
    marked = np.empty(a.size, dtype=np.bool_)
    count = 0
    if disagreements.size:
        chosen_site = int(np.random.default_rng(seed).choice(disagreements))
        forward, backward = make_periodic_lattice(shape)
        count = _flip_disagreement_component(
            a, b, bonds.reshape(2, a.size), forward, backward, chosen_site, queue, marked,
        )
    after_states = np.stack([a, b]).reshape(2, *shape)
    after = np.asarray(energy(after_states, bonds, field))
    return {
        "spins_a": after_states[0],
        "spins_b": after_states[1],
        "cluster_indices": queue[:count].copy(),
        "cluster_size": count,
        "energies_before": before,
        "energies_after": after,
        "joint_energy_residual": float(after.sum() - before.sum()),
    }


@njit(cache=True)
def _record_trace(spins, tracked, bonds, forward, field, trace_energy, trace_q2, index):
    """Independent full-energy check and all-unordered-pair q² statistic."""
    replicas, temperatures, size = spins.shape
    max_drift = 0.0
    for temperature in range(temperatures):
        total_energy = 0.0
        for replica in range(replicas):
            exact = _energy_flat(spins[replica, temperature], bonds, forward, field)
            max_drift = max(max_drift, abs(exact - tracked[replica, temperature]))
            tracked[replica, temperature] = exact
            total_energy += exact
        trace_energy[temperature, index] = total_energy / (replicas * size)
        total_q2 = 0.0
        pairs = 0
        for first in range(replicas):
            for second in range(first + 1, replicas):
                overlap_sum = 0.0
                for site in range(size):
                    overlap_sum += spins[first, temperature, site] * spins[second, temperature, site]
                total_q2 += (overlap_sum / size) ** 2
                pairs += 1
        trace_q2[temperature, index] = total_q2 / pairs
    return max_drift


@njit(cache=True)
def _sample_group(bonds, forward, backward, beta, samples, burn_in, spacing,
                  seed, replicas, exchange_interval, cluster_interval, field,
                  ordered, use_clusters, trace_interval, trace_count):
    np.random.seed(seed)
    temperature_count = beta.size
    size = forward.shape[1]
    spins = np.ones((replicas, temperature_count, size), dtype=np.int8)
    if not ordered:
        for replica in range(replicas):
            for temperature in range(temperature_count):
                for site in range(size):
                    spins[replica, temperature, site] = 2 * np.random.randint(0, 2) - 1
    energies = np.empty((replicas, temperature_count), dtype=np.float64)
    for replica in range(replicas):
        for temperature in range(temperature_count):
            energies[replica, temperature] = _energy_flat(
                spins[replica, temperature], bonds, forward, field,
            )
    configurations = np.empty((temperature_count, replicas, samples, size), dtype=np.int8)
    saved_energies = np.empty((temperature_count, replicas, samples), dtype=np.float64)
    swap_attempts = np.zeros((replicas, temperature_count - 1), dtype=np.int64)
    swap_accepts = np.zeros_like(swap_attempts)
    local_attempts = np.zeros((replicas, temperature_count), dtype=np.int64)
    local_accepts = np.zeros_like(local_attempts)
    walkers = np.empty((replicas, temperature_count), dtype=np.int64)
    visits = np.zeros((replicas, temperature_count, temperature_count), dtype=np.int64)
    round_trips = np.zeros((replicas, temperature_count), dtype=np.int64)
    origin = np.full((replicas, temperature_count), -1, dtype=np.int64)
    reached_opposite = np.zeros((replicas, temperature_count), dtype=np.bool_)
    for replica in range(replicas):
        for temperature in range(temperature_count):
            walkers[replica, temperature] = temperature
            visits[replica, temperature, temperature] = 1
        origin[replica, 0] = 0
        origin[replica, temperature_count - 1] = temperature_count - 1
    cluster_attempts = np.zeros(temperature_count, dtype=np.int64)
    cluster_updates = np.zeros(temperature_count, dtype=np.int64)
    cluster_size_sum = np.zeros(temperature_count, dtype=np.int64)
    cluster_size_max = np.zeros(temperature_count, dtype=np.int64)
    cluster_histogram = np.zeros((temperature_count, size + 1), dtype=np.int64)
    queue = np.empty(size, dtype=np.int64)
    marked = np.empty(size, dtype=np.bool_)
    trace_sweeps = np.empty(trace_count, dtype=np.int64)
    trace_energy = np.empty((temperature_count, trace_count), dtype=np.float64)
    trace_q2 = np.empty_like(trace_energy)
    trace_sweeps[0] = 0
    max_drift = _record_trace(
        spins, energies, bonds, forward, field, trace_energy, trace_q2, 0,
    )
    max_cluster_residual = 0.0
    exchange_event = 0
    sample_index = 0
    trace_index = 1
    total_sweeps = burn_in + samples * spacing
    for sweep in range(total_sweeps):
        for replica in range(replicas):
            for temperature in range(temperature_count):
                order = np.random.permutation(size)
                for order_index in range(size):
                    site = order[order_index]
                    if np.random.random() >= 0.5:
                        continue
                    local_attempts[replica, temperature] += 1
                    local_field = field
                    for axis in range(bonds.shape[0]):
                        previous = backward[axis, site]
                        local_field += bonds[axis, site] * spins[replica, temperature, forward[axis, site]]
                        local_field += bonds[axis, previous] * spins[replica, temperature, previous]
                    delta = 2.0 * spins[replica, temperature, site] * local_field
                    if delta <= 0.0 or np.random.random() < np.exp(-beta[temperature] * delta):
                        spins[replica, temperature, site] = -spins[replica, temperature, site]
                        energies[replica, temperature] += delta
                        local_accepts[replica, temperature] += 1

        if use_clusters and (sweep + 1) % cluster_interval == 0:
            for temperature in range(temperature_count):
                # The state-independent random matching is itself reversible.
                pairing = np.random.permutation(replicas)
                for pair_index in range(replicas // 2):
                    first = pairing[2 * pair_index]
                    second = pairing[2 * pair_index + 1]
                    a = spins[first, temperature]
                    b = spins[second, temperature]
                    cluster_attempts[temperature] += 1
                    disagreement_count = 0
                    for site in range(size):
                        if a[site] != b[site]:
                            disagreement_count += 1
                    count = 0
                    if disagreement_count:
                        target = np.random.randint(disagreement_count)
                        chosen_site = 0
                        seen = 0
                        for site in range(size):
                            if a[site] != b[site]:
                                if seen == target:
                                    chosen_site = site
                                    break
                                seen += 1
                        # Audit every actual cluster using independently
                        # recomputed Hamiltonians, not a presumed zero delta.
                        before_a = _energy_flat(a, bonds, forward, field)
                        before_b = _energy_flat(b, bonds, forward, field)
                        max_drift = max(max_drift, abs(before_a - energies[first, temperature]))
                        max_drift = max(max_drift, abs(before_b - energies[second, temperature]))
                        count = _flip_disagreement_component(
                            a, b, bonds, forward, backward, chosen_site, queue, marked,
                        )
                        after_a = _energy_flat(a, bonds, forward, field)
                        after_b = _energy_flat(b, bonds, forward, field)
                        max_cluster_residual = max(
                            max_cluster_residual, abs((after_a + after_b) - (before_a + before_b)),
                        )
                        energies[first, temperature] = after_a
                        energies[second, temperature] = after_b
                        cluster_updates[temperature] += 1
                        cluster_size_sum[temperature] += count
                        cluster_size_max[temperature] = max(cluster_size_max[temperature], count)
                    cluster_histogram[temperature, count] += 1

        if (sweep + 1) % exchange_interval == 0:
            for replica in range(replicas):
                for left in range(exchange_event % 2, temperature_count - 1, 2):
                    right = left + 1
                    swap_attempts[replica, left] += 1
                    log_ratio = (beta[left] - beta[right]) * (
                        energies[replica, left] - energies[replica, right]
                    )
                    if log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio):
                        for site in range(size):
                            temporary_spin = spins[replica, left, site]
                            spins[replica, left, site] = spins[replica, right, site]
                            spins[replica, right, site] = temporary_spin
                        temporary_energy = energies[replica, left]
                        energies[replica, left] = energies[replica, right]
                        energies[replica, right] = temporary_energy
                        temporary_walker = walkers[replica, left]
                        walkers[replica, left] = walkers[replica, right]
                        walkers[replica, right] = temporary_walker
                        swap_accepts[replica, left] += 1
            exchange_event += 1
        for replica in range(replicas):
            for temperature in range(temperature_count):
                walker = walkers[replica, temperature]
                visits[replica, walker, temperature] += 1
                if temperature == 0 or temperature == temperature_count - 1:
                    if origin[replica, walker] == -1:
                        origin[replica, walker] = temperature
                    elif temperature != origin[replica, walker]:
                        reached_opposite[replica, walker] = True
                    elif reached_opposite[replica, walker]:
                        round_trips[replica, walker] += 1
                        reached_opposite[replica, walker] = False
        if (sweep + 1) % trace_interval == 0 or sweep + 1 == total_sweeps:
            trace_sweeps[trace_index] = sweep + 1
            max_drift = max(max_drift, _record_trace(
                spins, energies, bonds, forward, field, trace_energy, trace_q2, trace_index,
            ))
            trace_index += 1
        if sweep >= burn_in and (sweep - burn_in + 1) % spacing == 0:
            for temperature in range(temperature_count):
                for replica in range(replicas):
                    configurations[temperature, replica, sample_index] = spins[replica, temperature]
                    exact = _energy_flat(spins[replica, temperature], bonds, forward, field)
                    max_drift = max(max_drift, abs(exact - energies[replica, temperature]))
                    energies[replica, temperature] = exact
                    saved_energies[temperature, replica, sample_index] = exact
            sample_index += 1
    return (
        configurations, saved_energies, swap_attempts, swap_accepts,
        local_attempts, local_accepts, round_trips, visits, cluster_attempts,
        cluster_updates, cluster_size_sum, cluster_size_max, cluster_histogram,
        max_cluster_residual, max_drift, trace_sweeps, trace_energy, trace_q2,
    )


def sample_houdayer_parallel_tempering(
    bonds, temperatures, *, samples, burn_in, spacing, seed, replicas=4,
    algorithm="houdayer_pt", exchange_interval=1, cluster_interval=1,
    field=0.0, initial="hot", trace_interval=16, max_output_bytes=512 * 1024**2,
) -> dict:
    """Run one independent group, with either ordinary PT or Houdayer--PT.

    Configurations are ``int8[T,R,S,*shape]`` and energies ``float64[T,R,S]``.
    Traces include initialization, each ``trace_interval`` sweeps, and the final
    sweep.  Trace q² is averaged over all unordered *within-group* real-replica
    pairs; these are not independent observations.  Every sweep includes one
    lazy local sweep at every temperature/replica, optionally random-pair
    Houdayer clusters, and then the usual alternating PT exchange event.

    Counts include burn-in.  Cluster histogram bin zero counts matching pairs
    with no disagreement; no rejection is applied to a nonempty cluster.
    Cluster energies are recomputed before and after every nonempty update.
    ``max_energy_tracking_drift`` audits local energy accumulation before any
    reset.  No equilibration, effective-sample, or independent-replica claim is
    inferred from these implementation checks or temperature round trips.

    The explicit allocation bound includes outputs and estimated working
    arrays, and is checked before neighbor/configuration allocations.  It is
    an allocation guard, not a process RSS or runtime resource guarantee.
    """
    bonds, shape = _bonds(bonds)
    if len(shape) != 2:
        raise ValueError("Houdayer comparison currently requires a 2D lattice")
    temperatures = _real_array(temperatures, "temperatures")
    if (temperatures.ndim != 1 or temperatures.size < 2
            or np.any(temperatures <= 0) or np.any(np.diff(temperatures) <= 0)):
        raise ValueError("temperatures must be a positive strictly increasing 1D array of length >= 2")
    temperatures = np.ascontiguousarray(temperatures)
    with np.errstate(over="ignore", divide="ignore"):
        beta = 1.0 / temperatures
    if not np.all(np.isfinite(beta)):
        raise ValueError("temperatures are too small for finite inverse temperatures")
    samples = _integer(samples, "samples", 1)
    burn_in = _integer(burn_in, "burn_in", 0)
    spacing = _integer(spacing, "spacing", 1)
    replicas = _integer(replicas, "replicas", 2)
    limit = np.iinfo(np.int64).max
    exchange_interval = _integer(exchange_interval, "exchange_interval", 1, limit)
    cluster_interval = _integer(cluster_interval, "cluster_interval", 1, limit)
    trace_interval = _integer(trace_interval, "trace_interval", 1, limit)
    max_output_bytes = _integer(max_output_bytes, "max_output_bytes", 1)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    field = _finite_scalar(field, "field")
    if algorithm not in ("pt", "houdayer_pt"):
        raise ValueError("algorithm must be 'pt' or 'houdayer_pt'")
    if initial not in ("hot", "cold"):
        raise ValueError("initial must be 'hot' or 'cold'; use separate calls for independent starts")
    size = math.prod(shape)
    count = temperatures.size
    total_sweeps = burn_in + samples * spacing
    trace_count = 1 + total_sweeps // trace_interval + int(total_sweeps % trace_interval != 0)
    if (total_sweeps >= limit or total_sweeps * size > limit
            or total_sweeps * (replicas // 2) * size > limit):
        raise ValueError("requested sweep or cluster count exceeds int64 counter range")
    elements = count * replicas * samples * size
    if elements > np.iinfo(np.intp).max:
        raise ValueError("requested configurations exceed the platform index range")
    estimated_bytes = (
        elements + 8 * count * replicas * samples
        + 8 * count * (size + 1) + 8 * trace_count * (1 + 2 * count)
        + replicas * count * size + 8 * replicas * count * count
        + 8 * replicas * count * 12 + 8 * count * 8
        + size * (8 * (2 * len(shape) + 3) + 2)
    )
    if estimated_bytes > max_output_bytes:
        raise ValueError(f"requested allocations require about {estimated_bytes} bytes; guard is {max_output_bytes}")
    with np.errstate(over="ignore", invalid="ignore"):
        bound = np.sum(np.abs(bonds), dtype=np.float64) + abs(field) * size
    if not np.isfinite(bound) or bound > np.finfo(np.float64).max / 8:
        raise ValueError("bond/field magnitudes exceed safe float64 energy arithmetic")
    forward, backward = make_periodic_lattice(shape)
    result = _sample_group(
        bonds.reshape(len(shape), size), forward, backward, beta, samples,
        burn_in, spacing, seed, replicas, exchange_interval, cluster_interval,
        field, initial == "cold", algorithm == "houdayer_pt", trace_interval,
        trace_count,
    )
    keys = (
        "configurations", "energies", "swap_attempts", "swap_accepts",
        "local_attempts", "local_accepts", "round_trips", "temperature_visits",
        "cluster_attempts", "cluster_updates", "cluster_size_sum", "cluster_size_max",
        "cluster_size_histogram", "max_cluster_joint_energy_residual",
        "max_energy_tracking_drift", "trace_sweeps", "trace_energy_density", "trace_q2",
    )
    output = dict(zip(keys, result))
    output["configurations"] = output["configurations"].reshape(count, replicas, samples, *shape)
    if not np.all(np.isfinite(output["energies"])):
        raise FloatingPointError("sampling produced nonfinite energies")
    output.update({
        "site_visits": np.full((replicas, count), size * total_sweeps, dtype=np.int64),
        "temperatures": temperatures.copy(), "algorithm": algorithm, "seed": seed,
        "shape": shape, "dimension": len(shape), "samples": samples, "burn_in": burn_in,
        "spacing": spacing, "replicas": replicas, "exchange_interval": exchange_interval,
        "cluster_interval": cluster_interval, "trace_interval": trace_interval,
        "total_sweeps": total_sweeps, "field": field, "initial": initial,
        "estimated_allocation_bytes": estimated_bytes,
        "proposal_flip_probability": 0.5,
        "exchange_schedule": "alternating even/odd adjacent edges, even first",
        "cluster_schedule": "random disjoint real-replica matching independently at every temperature/event",
        "round_trip_definition": "first endpoint -> opposite endpoint -> first endpoint, indexed by temperature transport label",
        "walker_label_caveat": "Houdayer moves mix configurations; labels do not retain intact configuration identity",
        "trace_q2_definition": "mean squared global overlap over all unordered within-group replica pairs; not independent observations",
        "equilibration_certified": False, "independent_equilibrium_certified": False,
    })
    return output
