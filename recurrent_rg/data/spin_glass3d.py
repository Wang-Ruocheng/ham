"""Isolated zero-field 3D EA PT with temperature-selective ICM.

The lazy Metropolis and alternating PT convention match the validated generic
EA kernels. Below a fixed temperature cutoff, a random matching of real
replicas receives a complete minority-overlap component update. When raw
negative overlap is the majority, one copy is globally flipped for cluster
construction and flipped back afterwards. This gauge-conjugate operation
preserves the raw overlap sign and the joint Boltzmann weight, rather than
folding saved physical data into q >= 0. This requires exactly zero field.

Clusters record torus winding, not merely a boundary-edge crossing. A winding
cluster is not rejected or truncated: these are diagnostic counters only.
Neither coupled replicas, transport round trips nor ESS certify equilibrium.
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
def _flip_disagreement_component(a, b, bonds, forward, backward, chosen_site, queue,
                                  marked, coordinates, winding):
    """Build an entire nonzero-bond component, audit torus winding, then flip."""
    marked[:] = False
    winding[:] = False
    if a[chosen_site] == b[chosen_site]:
        return 0
    queue[0] = chosen_site
    marked[chosen_site] = True
    coordinates[chosen_site, :] = 0
    count = 1
    head = 0
    while head < count:
        site = queue[head]
        head += 1
        for axis in range(3):
            for direction in range(2):
                sign = 1 if direction == 0 else -1
                neighbor = forward[axis, site] if direction == 0 else backward[axis, site]
                edge = bonds[axis, site] if direction == 0 else bonds[axis, neighbor]
                if edge == 0.0 or a[neighbor] == b[neighbor]:
                    continue
                if not marked[neighbor]:
                    marked[neighbor] = True
                    for component in range(3):
                        coordinates[neighbor, component] = coordinates[site, component]
                    coordinates[neighbor, axis] += sign
                    queue[count] = neighbor
                    count += 1
                else:
                    for component in range(3):
                        expected = coordinates[site, component] + (sign if component == axis else 0)
                        if coordinates[neighbor, component] != expected:
                            winding[component] = True
    for index in range(count):
        site = queue[index]
        a[site] = -a[site]
        b[site] = -b[site]
    return count


@njit(cache=True)
def _minority_cluster(a, b, bonds, forward, backward, queue, marked, coordinates, winding):
    """Uniform minority-component proposal, with a reversible zero-field gauge."""
    size = a.size
    disagreement = 0
    for site in range(size):
        disagreement += int(a[site] != b[site])
    gauge_copy = -1
    minority_count = disagreement
    if disagreement > size // 2:
        gauge_copy = np.random.randint(2)
        target = a if gauge_copy == 0 else b
        for site in range(size):
            target[site] = -target[site]
        minority_count = size - disagreement
    winding[:] = False
    count = 0
    if minority_count:
        selected = np.random.randint(minority_count)
        seen = 0
        chosen = 0
        for site in range(size):
            if a[site] != b[site]:
                if seen == selected:
                    chosen = site
                    break
                seen += 1
        count = _flip_disagreement_component(
            a, b, bonds, forward, backward, chosen, queue, marked, coordinates, winding,
        )
    if gauge_copy != -1:
        target = a if gauge_copy == 0 else b
        for site in range(size):
            target[site] = -target[site]
    return count, disagreement, minority_count, gauge_copy != -1


def icm_cluster_update(spins_a, spins_b, bonds, *, seed, field=0.0) -> dict:
    """Nonmutating audited zero-field minority-overlap cluster move in 3D.

    The result is unfolded: raw q is preserved pointwise, including its sign.
    With a negative-overlap majority the cluster belongs to the original
    positive-overlap domain. A single boundary crossing is not torus winding.
    """
    bonds, shape = _bonds(bonds)
    if len(shape) != 3:
        raise ValueError("ICM requires a 3D lattice")
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    if _finite_scalar(field, "field") != 0.0:
        raise ValueError("3D minority ICM requires exactly zero field")
    a = _real_array(spins_a, "spins_a")
    b = _real_array(spins_b, "spins_b")
    if a.shape != shape or b.shape != shape:
        raise ValueError(f"both spin states must have lattice shape {shape}")
    if not np.all((a == -1) | (a == 1)) or not np.all((b == -1) | (b == 1)):
        raise ValueError("spins must contain only -1 and +1")
    a, b = a.astype(np.int8).copy().ravel(), b.astype(np.int8).copy().ravel()
    before = np.asarray(energy(np.stack([a, b]).reshape(2, *shape), bonds))
    forward, backward = make_periodic_lattice(shape)
    queue = np.empty(a.size, dtype=np.int64)
    marked = np.empty(a.size, dtype=np.bool_)
    coordinates = np.empty((a.size, 3), dtype=np.int64)
    winding = np.zeros(3, dtype=np.bool_)
    count, disagreement, minority, gauge = _seeded_cluster(
        a, b, bonds.reshape(3, a.size), forward, backward, seed, queue, marked,
        coordinates, winding,
    )
    states = np.stack([a, b]).reshape(2, *shape)
    after = np.asarray(energy(states, bonds))
    return dict(
        spins_a=states[0], spins_b=states[1], cluster_indices=queue[:count].copy(),
        cluster_size=count, energies_before=before, energies_after=after,
        joint_energy_residual=float(after.sum() - before.sum()),
        raw_disagreement_count=disagreement, minority_disagreement_count=minority,
        gauge_flip_used=bool(gauge), gauge_flip_undone=True,
        winding_axes=winding.copy(), winding_any=bool(winding.any()),
    )


@njit(cache=True)
def _seeded_cluster(a, b, bonds, forward, backward, seed, queue, marked, coordinates, winding):
    np.random.seed(seed)
    return _minority_cluster(a, b, bonds, forward, backward, queue, marked, coordinates, winding)


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
                  ordered, use_clusters, trace_interval, trace_count, icm_active):
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
    coordinates = np.empty((size, 3), dtype=np.int64)
    winding = np.zeros(3, dtype=np.bool_)
    gauge_flip_count = np.zeros(temperature_count, dtype=np.int64)
    raw_disagreement_sum = np.zeros(temperature_count, dtype=np.int64)
    minority_disagreement_sum = np.zeros(temperature_count, dtype=np.int64)
    cluster_winding_count = np.zeros((temperature_count, 3), dtype=np.int64)
    cluster_winding_any_count = np.zeros(temperature_count, dtype=np.int64)
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
                if not icm_active[temperature]:
                    continue
                # The state-independent random matching is itself reversible.
                pairing = np.random.permutation(replicas)
                for pair_index in range(replicas // 2):
                    first = pairing[2 * pair_index]
                    second = pairing[2 * pair_index + 1]
                    a = spins[first, temperature]
                    b = spins[second, temperature]
                    cluster_attempts[temperature] += 1
                    before_a = _energy_flat(a, bonds, forward, field)
                    before_b = _energy_flat(b, bonds, forward, field)
                    max_drift = max(max_drift, abs(before_a - energies[first, temperature]))
                    max_drift = max(max_drift, abs(before_b - energies[second, temperature]))
                    count, disagreement, minority, gauge = _minority_cluster(
                        a, b, bonds, forward, backward, queue, marked, coordinates, winding,
                    )
                    gauge_flip_count[temperature] += int(gauge)
                    raw_disagreement_sum[temperature] += disagreement
                    minority_disagreement_sum[temperature] += minority
                    after_a = _energy_flat(a, bonds, forward, field)
                    after_b = _energy_flat(b, bonds, forward, field)
                    max_cluster_residual = max(
                        max_cluster_residual, abs(after_a + after_b - before_a - before_b),
                    )
                    energies[first, temperature] = after_a
                    energies[second, temperature] = after_b
                    if count:
                        cluster_updates[temperature] += 1
                        cluster_size_sum[temperature] += count
                        cluster_size_max[temperature] = max(cluster_size_max[temperature], count)
                        any_winding = False
                        for axis in range(3):
                            cluster_winding_count[temperature, axis] += int(winding[axis])
                            any_winding = any_winding or winding[axis]
                        cluster_winding_any_count[temperature] += int(any_winding)
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
        gauge_flip_count, raw_disagreement_sum, minority_disagreement_sum,
        cluster_winding_count, cluster_winding_any_count,
    )


def sample_icm_parallel_tempering(
    bonds, temperatures, *, samples, burn_in, spacing, seed, replicas=4,
    algorithm="icm_pt", exchange_interval=1, cluster_interval=1,
    field=0.0, initial="hot", trace_interval=16, icm_max_temperature=1.0,
    max_output_bytes=512 * 1024**2,
) -> dict:
    """Run one finite group, with ordinary PT or zero-field temperature-gated ICM.

    Configurations int8[T,R,S,*shape] and energies float64[T,R,S] remain
    unfolded physical states. ICM is applied only when T <= fixed cutoff.
    Local lazy sweeps, complete clusters, then alternating adjacent exchanges
    are fixed before sampling. No diagnostic rejection, truncation, extension
    or equilibrium certificate is inferred from ESS or transport labels.
    """
    bonds, shape = _bonds(bonds)
    if len(shape) != 3:
        raise ValueError("ICM requires a 3D lattice")
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
    if field != 0.0:
        raise ValueError("3D minority ICM requires exactly zero field")
    icm_max_temperature = _finite_scalar(icm_max_temperature, "icm_max_temperature")
    if icm_max_temperature <= 0.0:
        raise ValueError("icm_max_temperature must be positive")
    if algorithm not in ("pt", "icm_pt"):
        raise ValueError("algorithm must be 'pt' or 'icm_pt'")
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
        + size * (8 * (2 * len(shape) + 6) + 2) + 8 * count * 8
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
        field, initial == "cold", algorithm == "icm_pt", trace_interval,
        trace_count, np.asarray(temperatures <= icm_max_temperature),
    )
    keys = (
        "configurations", "energies", "swap_attempts", "swap_accepts",
        "local_attempts", "local_accepts", "round_trips", "temperature_visits",
        "cluster_attempts", "cluster_updates", "cluster_size_sum", "cluster_size_max",
        "cluster_size_histogram", "max_cluster_joint_energy_residual",
        "max_energy_tracking_drift", "trace_sweeps", "trace_energy_density", "trace_q2",
        "gauge_flip_count", "raw_disagreement_sum", "minority_disagreement_sum",
        "cluster_winding_count", "cluster_winding_any_count",
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
        "icm_max_temperature": icm_max_temperature,
        "icm_active_temperatures": temperatures <= icm_max_temperature,
        "gauge_flip_convention": "Undo the same global spin reversal after minority cluster construction; saved physical states and overlap signs are unfolded",
        "winding_definition": "Nonzero torus winding detected by inconsistent unwrapped BFS coordinates; no clusters rejected",
        "total_sweeps": total_sweeps, "field": field, "initial": initial,
        "estimated_allocation_bytes": estimated_bytes,
        "proposal_flip_probability": 0.5,
        "exchange_schedule": "alternating even/odd adjacent edges, even first",
        "cluster_schedule": "random disjoint real-replica matching independently at every temperature/event",
        "round_trip_definition": "first endpoint -> opposite endpoint -> first endpoint, indexed by temperature transport label",
        "walker_label_caveat": "ICM moves mix configurations; labels do not retain intact configuration identity",
        "trace_q2_definition": "mean squared global overlap over all unordered within-group replica pairs; not independent observations",
        "equilibration_certified": False, "independent_equilibrium_certified": False,
    })
    return output

