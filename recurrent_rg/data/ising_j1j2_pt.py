"""Auditable local-Metropolis parallel tempering for square-lattice J1--J2.

The convention is H = E_nn + rho E_diagonal, with E_nn = -sum_NN s_i s_j
and E_diagonal = +sum_diagonal s_i s_j.  Each undirected bond occurs once.
No ferromagnetic-only cluster shortcut, global sign folding, adaptive burn-in,
or automatic resampling is used.  A ladder group, not a temperature slot or
saved configuration, is the independent sampling unit.

Every sweep visits every site once in four noninteracting sublattices.  At a
visited site a fixed probability 1/2 proposes a flip; the usual Metropolis
test then accepts or rejects it.  This lazy kernel avoids periodic high-T
flips and preserves exactly the requested Boltzmann law.  PT exchanges use
state-index maps rather than copying lattice arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

try:
    from numba import njit
except ImportError:  # The same scalar kernel is available for small diagnostics.
    njit = None


def _jit(function):
    return njit(cache=True)(function) if njit is not None else function


def _integer(value, name, minimum, maximum=2**63 - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _rho(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("rho must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("rho must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("rho must be finite and nonnegative")
    return result


def seed_for_group(master_seed, size, rho, purpose, group):
    """Stable 64-bit namespace; pilot groups and production groups must differ."""
    master_seed = _integer(master_seed, "master_seed", 0, 2**64 - 1)
    size = _integer(size, "size", 4)
    group = _integer(group, "group", 0)
    rho = _rho(rho)
    if not isinstance(purpose, str) or not purpose or len(purpose) > 256:
        raise ValueError("purpose must be a nonempty string of at most 256 characters")
    # hex is stable and lossless, including when a parameter came from JSON.
    token = json.dumps(
        ["ising_j1j2_pt_v1", master_seed, size, rho.hex(), purpose, group],
        ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _spins_array(spins):
    spins = np.asarray(spins)
    if spins.ndim < 2 or spins.shape[-1] != spins.shape[-2] or spins.shape[-1] < 4:
        raise ValueError("spins must end in a square lattice of size >= 4")
    if not np.all((spins == -1) | (spins == 1)):
        raise ValueError("spins must contain only -1 and +1")
    return spins.astype(np.int64, copy=False)


def energy_components(spins):
    """Return extensive (negative NN, positive diagonal) energies, no mutation."""
    spins = _spins_array(spins)
    nn = -np.sum(spins * (np.roll(spins, -1, -2) + np.roll(spins, -1, -1)), axis=(-2, -1))
    diagonal = np.sum(
        spins * (np.roll(np.roll(spins, -1, -2), -1, -1)
                 + np.roll(np.roll(spins, -1, -2), 1, -1)), axis=(-2, -1),
    )
    return nn, diagonal


def configuration_energy(spins, rho):
    """H=-sum_NN ss+rho sum_diagonal ss; J1=1, k_B=1."""
    nn, diagonal = energy_components(spins)
    return nn + _rho(rho) * diagonal


def local_delta_energy(spins, row, column, rho):
    """Exact energy increase from flipping one site, without changing input."""
    spins = _spins_array(spins)
    if spins.ndim != 2:
        raise ValueError("local_delta_energy requires one lattice")
    size = spins.shape[0]
    row = _integer(row, "row", 0, size - 1)
    column = _integer(column, "column", 0, size - 1)
    up, down = (row - 1) % size, (row + 1) % size
    left, right = (column - 1) % size, (column + 1) % size
    nn = spins[up, column] + spins[down, column] + spins[row, left] + spins[row, right]
    diag = spins[up, left] + spins[up, right] + spins[down, left] + spins[down, right]
    return float(2 * spins[row, column] * (nn - _rho(rho) * diag))


def proposal_log_swap_ratio(temperature_a, temperature_b, energy_a, energy_b):
    """log[p(after exchange)/p(before)] for two fixed-temperature slots."""
    values = [temperature_a, temperature_b, energy_a, energy_b]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("swap inputs must be finite")
    if float(temperature_a) <= 0 or float(temperature_b) <= 0:
        raise ValueError("temperatures must be positive")
    return (1.0 / float(temperature_a) - 1.0 / float(temperature_b)) * (
        float(energy_a) - float(energy_b)
    )


def _geometry(size):
    sites = np.arange(size * size, dtype=np.int64).reshape(size, size)
    neighbors = np.stack([
        np.roll(sites, 1, 0), np.roll(sites, -1, 0),
        np.roll(sites, 1, 1), np.roll(sites, -1, 1),
        np.roll(np.roll(sites, 1, 0), 1, 1),
        np.roll(np.roll(sites, 1, 0), -1, 1),
        np.roll(np.roll(sites, -1, 0), 1, 1),
        np.roll(np.roll(sites, -1, 0), -1, 1),
    ]).reshape(8, size * size)
    # All NN and diagonal edges join different members of this four-coloring.
    order = np.concatenate([sites[x::2, y::2].ravel() for x in range(2) for y in range(2)])
    sign_x = np.repeat(1 - 2 * (np.arange(size) % 2), size).astype(np.int64)
    sign_y = np.tile(1 - 2 * (np.arange(size) % 2), size).astype(np.int64)
    return neighbors, order, sign_x, sign_y


@_jit
def _run(spins, neighbors, order, sign_x, sign_y, beta, rho, rng,
         burn_in, samples, spacing, swap_interval, trace_steps):
    n_temp, n_sites = spins.shape
    mapping = np.arange(n_temp)
    enn = np.zeros(n_temp, dtype=np.int64)
    ediag = np.zeros(n_temp, dtype=np.int64)
    magnetization = np.zeros(n_temp, dtype=np.int64)
    stripe_x = np.zeros(n_temp, dtype=np.int64)
    stripe_y = np.zeros(n_temp, dtype=np.int64)
    for replica in range(n_temp):
        for site in range(n_sites):
            spin = int(spins[replica, site])
            enn[replica] -= spin * (int(spins[replica, neighbors[1, site]])
                                    + int(spins[replica, neighbors[3, site]]))
            ediag[replica] += spin * (int(spins[replica, neighbors[6, site]])
                                      + int(spins[replica, neighbors[7, site]]))
            magnetization[replica] += spin
            stripe_x[replica] += spin * sign_x[site]
            stripe_y[replica] += spin * sign_y[site]
    configs = np.empty((n_temp, samples, n_sites), dtype=np.int8)
    saved_nn = np.empty((n_temp, samples), dtype=np.int64)
    saved_diagonal = np.empty_like(saved_nn)
    saved_m = np.empty((n_temp, samples), dtype=np.float64)
    saved_sx = np.empty_like(saved_m)
    saved_sy = np.empty_like(saved_m)
    trace_values = np.empty((n_temp, trace_steps.size, 4), dtype=np.float64)
    trace_labels = np.empty((n_temp, trace_steps.size), dtype=np.int64)
    swap_attempts = np.zeros(max(0, n_temp - 1), dtype=np.int64)
    swap_accepts = np.zeros_like(swap_attempts)
    local_attempts = np.zeros(n_temp, dtype=np.int64)
    local_accepts = np.zeros(n_temp, dtype=np.int64)
    temperature_visits = np.zeros((n_temp, n_temp), dtype=np.int64)
    round_trips = np.zeros(n_temp, dtype=np.int64)
    origins = np.full(n_temp, -1, dtype=np.int64)
    opposite_seen = np.zeros(n_temp, dtype=np.bool_)
    if n_temp > 1:
        origins[0] = 0
        origins[n_temp - 1] = n_temp - 1
    trace_index = 0
    saved_index = 0
    exchange_epoch = 0
    total_steps = burn_in + samples * spacing
    for step in range(total_steps + 1):
        if step > 0:
            for temperature in range(n_temp):
                replica = mapping[temperature]
                for ordered_site in range(n_sites):
                    site = order[ordered_site]
                    if rng.random() >= 0.5:
                        continue
                    local_attempts[temperature] += 1
                    spin = int(spins[replica, site])
                    nn = (int(spins[replica, neighbors[0, site]])
                          + int(spins[replica, neighbors[1, site]])
                          + int(spins[replica, neighbors[2, site]])
                          + int(spins[replica, neighbors[3, site]]))
                    diagonal = (int(spins[replica, neighbors[4, site]])
                                + int(spins[replica, neighbors[5, site]])
                                + int(spins[replica, neighbors[6, site]])
                                + int(spins[replica, neighbors[7, site]]))
                    delta_nn = 2 * spin * nn
                    delta_diagonal = -2 * spin * diagonal
                    delta = delta_nn + rho * delta_diagonal
                    if delta <= 0 or rng.random() < math.exp(-beta[temperature] * delta):
                        spins[replica, site] = -spin
                        enn[replica] += delta_nn
                        ediag[replica] += delta_diagonal
                        magnetization[replica] -= 2 * spin
                        stripe_x[replica] -= 2 * spin * sign_x[site]
                        stripe_y[replica] -= 2 * spin * sign_y[site]
                        local_accepts[temperature] += 1
            if step % swap_interval == 0 and n_temp > 1:
                # Alternating disjoint pairs; for two temperatures attempt every epoch.
                parity = exchange_epoch % 2 if n_temp > 2 else 0
                for temperature in range(parity, n_temp - 1, 2):
                    first, second = mapping[temperature], mapping[temperature + 1]
                    log_ratio = (beta[temperature] - beta[temperature + 1]) * (
                        (enn[first] - enn[second]) + rho * (ediag[first] - ediag[second])
                    )
                    swap_attempts[temperature] += 1
                    if log_ratio >= 0 or rng.random() < math.exp(log_ratio):
                        mapping[temperature], mapping[temperature + 1] = second, first
                        swap_accepts[temperature] += 1
                exchange_epoch += 1
        # Count visits at initial time and each exchange epoch, not every site.
        if step == 0 or step % swap_interval == 0:
            for temperature in range(n_temp):
                replica = mapping[temperature]
                temperature_visits[replica, temperature] += 1
                if n_temp > 1 and (temperature == 0 or temperature == n_temp - 1):
                    if origins[replica] == -1:
                        origins[replica] = temperature
                    elif temperature != origins[replica]:
                        opposite_seen[replica] = True
                    elif opposite_seen[replica]:
                        round_trips[replica] += 1
                        opposite_seen[replica] = False
        if trace_index < trace_steps.size and step == trace_steps[trace_index]:
            for temperature in range(n_temp):
                replica = mapping[temperature]
                trace_values[temperature, trace_index, 0] = (
                    enn[replica] + rho * ediag[replica]
                ) / n_sites
                trace_values[temperature, trace_index, 1] = (magnetization[replica] / n_sites) ** 2
                trace_values[temperature, trace_index, 2] = (stripe_x[replica] / n_sites) ** 2
                trace_values[temperature, trace_index, 3] = (stripe_y[replica] / n_sites) ** 2
                trace_labels[temperature, trace_index] = replica
            trace_index += 1
        if step > burn_in and (step - burn_in) % spacing == 0:
            for temperature in range(n_temp):
                replica = mapping[temperature]
                configs[temperature, saved_index] = spins[replica]
                saved_nn[temperature, saved_index] = enn[replica]
                saved_diagonal[temperature, saved_index] = ediag[replica]
                saved_m[temperature, saved_index] = magnetization[replica] / n_sites
                saved_sx[temperature, saved_index] = stripe_x[replica] / n_sites
                saved_sy[temperature, saved_index] = stripe_y[replica] / n_sites
            saved_index += 1
    return (configs, saved_nn, saved_diagonal, saved_m, saved_sx, saved_sy,
            trace_values, trace_labels, swap_attempts, swap_accepts,
            local_attempts, local_accepts, temperature_visits, round_trips)


def series_diagnostics(values):
    """Finite-series initial-positive-pair IAT estimate; never thermalization proof."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("diagnostic series must be finite, one-dimensional and nonempty")
    count = values.size
    centered = values - values.mean()
    variance = float(np.dot(centered, centered) / count)
    common = dict(count=int(count), mean=float(values.mean()), variance=variance)
    if count < 4 or variance <= 0:
        return dict(**common, lag1=None, iat=None, ess=None,
                    status="constant_or_too_short_not_mixing_evidence")
    # O(n log n), so diagnostics remain bounded even for long pilots.
    n_fft = 1 << (2 * count - 1).bit_length()
    transformed = np.fft.rfft(centered, n=n_fft)
    covariance = np.fft.irfft(transformed * transformed.conjugate(), n=n_fft)[:count]
    correlations = covariance / covariance[0]
    tau = 1.0
    for first in range(1, count - 1, 2):
        pair = float(correlations[first] + correlations[first + 1])
        if pair <= 0:
            break
        tau += 2.0 * pair
    tau = min(float(count), max(1.0, tau))
    return dict(**common, lag1=float(correlations[1]), iat=tau,
                ess=float(count / tau), status="finite_series_estimate_only")


def sample_ladder(size, rho, temperatures, seed, burn_in, samples, spacing,
                  initial="hot", trace_interval=32, swap_interval=1,
                  max_trace_points=4096, use_numba=True, max_output_bytes=2**30):
    """Sample one independent PT group without writing files or mutating inputs.

    Temperatures are strictly increasing and all positive.  Output ``configs``
    is [temperature, saved sample, row, column], with raw spins in {-1,+1}.
    Components and all energies are extensive; trace energies are per site.
    A saved sample follows ``spacing`` complete lazy sweeps after burn-in.
    ``cpu_seconds`` includes JIT initialization on first call; wall seconds are
    likewise measured, not extrapolated.  Warm-up costs must not be discarded
    from resource accounting even when subsequent throughput is benchmarked.
    """
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    size = _integer(size, "size", 4)
    if size % 2:
        raise ValueError("periodic four-sublattice sampling requires even size")
    rho = _rho(rho)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    burn_in = _integer(burn_in, "burn_in", 0)
    samples = _integer(samples, "samples", 1)
    spacing = _integer(spacing, "spacing", 1)
    trace_interval = _integer(trace_interval, "trace_interval", 1)
    swap_interval = _integer(swap_interval, "swap_interval", 1)
    max_trace_points = _integer(max_trace_points, "max_trace_points", 2)
    max_output_bytes = _integer(max_output_bytes, "max_output_bytes", 1)
    if initial not in {"hot", "ferro", "stripe_x", "stripe_y"}:
        raise ValueError("initial must be hot, ferro, stripe_x, or stripe_y")
    if not isinstance(use_numba, (bool, np.bool_)):
        raise ValueError("use_numba must be boolean")
    temperatures = np.asarray(temperatures, dtype=np.float64).copy()
    if (temperatures.ndim != 1 or not 1 <= temperatures.size <= 4096
            or not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0)
            or np.any(np.diff(temperatures) <= 0)):
        raise ValueError("temperatures must be finite, positive and strictly increasing")
    n_temp = temperatures.size
    total_steps = burn_in + samples * spacing
    if total_steps > 2**63 - 1 or n_temp * total_steps > 2**63 - 1:
        raise ValueError("sweep count is too large")
    if not math.isfinite(rho * (2 * size * size)):
        raise ValueError("rho times the extensive interaction count must remain finite")
    with np.errstate(over="ignore", divide="ignore"):
        beta = 1.0 / temperatures
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0):
        raise ValueError("inverse temperatures must remain finite")
    # Audit at most eight saved configurations at once.  The conservative bound
    # includes returned arrays, live state, geometry and int64 audit temporaries.
    audit_batch = min(samples, 8)
    estimated_bytes = (n_temp * samples * (size * size + 80)
                       + n_temp * max_trace_points * 48 + n_temp * n_temp * 8
                       + n_temp * size * size * 16 + size * size * (96 + audit_batch * 96))
    if estimated_bytes > max_output_bytes:
        raise ValueError(f"estimated peak arrays {estimated_bytes} exceed max_output_bytes")
    effective_interval = max(trace_interval, math.ceil(total_steps / (max_trace_points - 1)))
    trace_steps = np.arange(0, total_steps + 1, effective_interval, dtype=np.int64)
    if trace_steps[-1] != total_steps:
        trace_steps = np.append(trace_steps, np.int64(total_steps))
    neighbors, order, sign_x, sign_y = _geometry(size)
    rng = np.random.default_rng(seed)
    if initial == "hot":
        spins = (2 * rng.integers(0, 2, size=(n_temp, size * size), dtype=np.int8) - 1)
    else:
        state = np.ones(size * size, dtype=np.int8)
        if initial == "stripe_x":
            state[:] = sign_x
        elif initial == "stripe_y":
            state[:] = sign_y
        spins = np.tile(state, (n_temp, 1))
    kernel = _run
    if not use_numba and hasattr(_run, "py_func"):
        kernel = _run.py_func
    result = kernel(spins, neighbors, order, sign_x, sign_y, beta,
                    rho, rng, burn_in, samples, spacing, swap_interval, trace_steps)
    (configs, saved_nn, saved_diagonal, saved_m, saved_sx, saved_sy,
     trace_values, trace_labels, swap_attempts, swap_accepts,
     local_attempts, local_accepts, temperature_visits, round_trips) = result
    configs = configs.reshape(n_temp, samples, size, size)
    energies = saved_nn + rho * saved_diagonal
    # Recompute all saved energies independently, rather than auditing one old PID/state.
    component_drift, energy_drift = 0, 0.0
    for temperature in range(n_temp):
        for first in range(0, samples, audit_batch):
            stop = min(samples, first + audit_batch)
            recomputed_nn, recomputed_diagonal = energy_components(configs[temperature, first:stop])
            component_drift = max(
                component_drift,
                int(np.max(np.abs(recomputed_nn - saved_nn[temperature, first:stop]))),
                int(np.max(np.abs(recomputed_diagonal - saved_diagonal[temperature, first:stop]))),
            )
            energy_drift = max(energy_drift, float(np.max(np.abs(
                recomputed_nn + rho * recomputed_diagonal - energies[temperature, first:stop]
            ))))
    diagnostics = []
    for temperature in range(n_temp):
        diagnostics.append({
            "temperature": float(temperatures[temperature]),
            "energy_per_site": series_diagnostics(energies[temperature] / (size * size)),
            "magnetization_squared": series_diagnostics(saved_m[temperature] ** 2),
            "stripe_x_squared": series_diagnostics(saved_sx[temperature] ** 2),
            "stripe_y_squared": series_diagnostics(saved_sy[temperature] ** 2),
        })
    return {
        "schema_version": 1, "model": "square_ising_j1j2", "size": size,
        "rho": rho, "temperatures": temperatures, "seed": seed,
        "initial": initial, "configs": configs, "energies": energies,
        "energy_nn": saved_nn, "energy_diagonal": saved_diagonal,
        "magnetization": saved_m, "stripe_x": saved_sx, "stripe_y": saved_sy,
        "swap_attempts": swap_attempts, "swap_accepts": swap_accepts,
        "local_attempts": local_attempts, "local_accepts": local_accepts,
        "visited_sites": n_temp * total_steps * size * size,
        "temperature_visits": temperature_visits, "round_trips": round_trips,
        "trace": {
            "sweeps": trace_steps, "effective_interval": effective_interval,
            "energy_per_site": trace_values[:, :, 0],
            "magnetization_squared": trace_values[:, :, 1],
            "stripe_x_squared": trace_values[:, :, 2],
            "stripe_y_squared": trace_values[:, :, 3],
            "temperature_to_replica": trace_labels,
            "includes_burn_in": True,
        },
        "diagnostics": diagnostics,
        "max_component_tracking_drift": component_drift,
        "max_energy_tracking_drift": energy_drift,
        "total_sweeps": n_temp * total_steps,
        "sweeps_per_temperature": total_steps,
        "burn_in": burn_in, "samples": samples, "spacing": spacing,
        "swap_interval": swap_interval, "local_proposal_probability": 0.5,
        "independent_unit": "whole_temperature_ladder_group",
        "temperatures_within_group_are_correlated": True,
        "equilibration_certified": False,
        "numba_used": bool(use_numba and njit is not None),
        "estimated_peak_array_bytes": estimated_bytes,
        "cpu_seconds": float(time.process_time() - cpu_start),
        "wall_seconds": float(time.perf_counter() - wall_start),
    }
