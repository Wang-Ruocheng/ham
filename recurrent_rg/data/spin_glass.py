"""CPU parallel tempering for periodic Edwards--Anderson Ising systems.

The Hamiltonian is ``H = -sum[a,x] J[a,x] s[x] s[x+e_a] - h sum[x] s[x]``.
Every positively oriented periodic bond occurs once.  Axis lengths must be at
least three, so there is no length-two parallel-edge convention to resolve.

A real replica is an independent *whole temperature ladder*.  Exchanges move
walkers within that ladder, never between real replicas.  Sampling at fixed
temperatures does not make the resulting observations independent or establish
equilibration; this module does not extend runs or certify convergence.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numba import njit


def _integer(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        bounds = f"at least {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bounds}")
    return int(result)


def _shape(shape) -> tuple[int, ...]:
    try:
        result = tuple(_integer(length, "axis length", 3) for length in shape)
    except TypeError as exc:
        raise ValueError("shape must be a nonempty sequence of integer axis lengths") from exc
    if not result:
        raise ValueError("shape must contain at least one axis")
    if math.prod(result) > np.iinfo(np.intp).max:
        raise ValueError("lattice size exceeds the platform index range")
    return result


def _finite_scalar(value, name: str) -> float:
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain finite real values")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _bonds(bonds) -> tuple[np.ndarray, tuple[int, ...]]:
    values = _real_array(bonds, "bonds")
    if values.ndim < 2 or values.shape[0] != values.ndim - 1:
        raise ValueError("bonds must have shape [D, *shape] with D lattice axes")
    shape = _shape(values.shape[1:])
    return np.ascontiguousarray(values), shape


def make_periodic_lattice(shape) -> tuple[np.ndarray, np.ndarray]:
    """Return C-order forward/backward neighbor indices, each ``int64[D,N]``."""
    shape = _shape(shape)
    size = math.prod(shape)
    sites = np.arange(size, dtype=np.int64).reshape(shape)
    forward = np.empty((len(shape), size), dtype=np.int64)
    backward = np.empty_like(forward)
    for axis in range(len(shape)):
        forward[axis] = np.roll(sites, -1, axis=axis).ravel()
        backward[axis] = np.roll(sites, 1, axis=axis).ravel()
    return forward, backward


def generate_bonds(
    shape,
    seed,
    distribution: str = "bimodal",
    scale: float = 1.0,
    mean: float = 0.0,
    positive_probability: float = 0.5,
) -> np.ndarray:
    """Generate one quenched disorder realization as ``float64[D,*shape]``.

    Bimodal bonds are ``mean + scale`` with ``positive_probability`` and
    ``mean - scale`` otherwise; the probability refers to the sign of the
    displacement, not necessarily the final bond.  Gaussian ``scale`` is the
    standard deviation.  A zero scale is allowed for either distribution.
    """
    shape = _shape(shape)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    scale = _finite_scalar(scale, "scale")
    mean = _finite_scalar(mean, "mean")
    positive_probability = _finite_scalar(positive_probability, "positive_probability")
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    if not 0 <= positive_probability <= 1:
        raise ValueError("positive_probability must be in [0, 1]")
    if distribution not in ("bimodal", "gaussian"):
        raise ValueError("distribution must be 'bimodal' or 'gaussian'")
    rng = np.random.default_rng(seed)
    bond_shape = (len(shape), *shape)
    with np.errstate(over="ignore", invalid="ignore"):
        if distribution == "bimodal":
            positive = rng.random(bond_shape) < positive_probability
            result = mean + scale * np.where(positive, 1.0, -1.0)
        else:
            result = rng.normal(mean, scale, size=bond_shape)
    if not np.all(np.isfinite(result)):
        raise ValueError("bond generation overflowed; reduce mean or scale")
    return np.asarray(result, dtype=np.float64)


def energy(spins, bonds, field: float = 0.0):
    """Evaluate ``H`` for spins of shape ``[...,*shape]`` without mutation.

    Returns one float64 energy for each leading configuration index, or a
    scalar float64 when a single lattice is supplied.  Spins must be exactly
    ``-1`` or ``+1``; no rescaling or rounding is performed.
    """
    bonds, shape = _bonds(bonds)
    field = _finite_scalar(field, "field")
    spins = _real_array(spins, "spins")
    dimension = len(shape)
    if spins.ndim < dimension or spins.shape[-dimension:] != shape:
        raise ValueError(f"spins must have trailing lattice shape {shape}")
    if not np.all((spins == -1) | (spins == 1)):
        raise ValueError("spins must contain only -1 and +1")
    axes = tuple(range(spins.ndim - dimension, spins.ndim))
    with np.errstate(over="ignore", invalid="ignore"):
        result = -field * np.sum(spins, axis=axes, dtype=np.float64)
        for axis in range(dimension):
            neighbor = np.roll(spins, -1, axis=spins.ndim - dimension + axis)
            result = result - np.sum(bonds[axis] * spins * neighbor, axis=axes, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("energy overflowed; reduce bond or field magnitudes")
    return result


def swap_log_acceptance(beta_left, beta_right, E_left, E_right) -> float:
    """Untruncated log Metropolis ratio for exchanging adjacent walkers.

    The accepted probability is ``min(1, exp(result))``.  The energy includes
    the field and is the same physical Hamiltonian at both temperatures.
    """
    return (float(beta_left) - float(beta_right)) * (float(E_left) - float(E_right))


@njit(cache=True)
def _energy_flat(spins, bonds, forward, field):
    result = 0.0
    for site in range(spins.size):
        result -= field * spins[site]
        for axis in range(bonds.shape[0]):
            result -= bonds[axis, site] * spins[site] * spins[forward[axis, site]]
    return result


@njit(cache=True)
def _sample_ladder(bonds, forward, backward, beta, samples, burn_in, spacing,
                   seed, exchange_interval, field, ordered):
    """One real replica; all random draws use this ladder's private seed."""
    np.random.seed(seed)
    temperature_count = beta.size
    size = forward.shape[1]
    spins = np.ones((temperature_count, size), dtype=np.int8)
    if not ordered:
        for temperature in range(temperature_count):
            for site in range(size):
                spins[temperature, site] = 2 * np.random.randint(0, 2) - 1
    energies = np.empty(temperature_count, dtype=np.float64)
    for temperature in range(temperature_count):
        energies[temperature] = _energy_flat(spins[temperature], bonds, forward, field)
    configurations = np.empty((temperature_count, samples, size), dtype=np.int8)
    saved_energies = np.empty((temperature_count, samples), dtype=np.float64)
    swap_attempts = np.zeros(temperature_count - 1, dtype=np.int64)
    swap_accepts = np.zeros(temperature_count - 1, dtype=np.int64)
    local_attempts = np.zeros(temperature_count, dtype=np.int64)
    local_accepts = np.zeros(temperature_count, dtype=np.int64)
    walkers = np.arange(temperature_count, dtype=np.int64)
    visits = np.zeros((temperature_count, temperature_count), dtype=np.int64)
    round_trips = np.zeros(temperature_count, dtype=np.int64)
    origin = np.full(temperature_count, -1, dtype=np.int64)
    reached_opposite = np.zeros(temperature_count, dtype=np.bool_)
    origin[0] = 0
    origin[temperature_count - 1] = temperature_count - 1
    for temperature in range(temperature_count):
        visits[temperature, temperature] = 1

    total_sweeps = burn_in + samples * spacing
    exchange_event = 0
    sample_index = 0
    for sweep in range(total_sweeps):
        for temperature in range(temperature_count):
            # A fresh random permutation visits every site exactly once.
            order = np.random.permutation(size)
            for order_index in range(size):
                site = order[order_index]
                # Fixed laziness removes whole-sweep period-two degeneracies
                # (e.g. J=h=0), while preserving the Boltzmann stationary law.
                if np.random.random() >= 0.5:
                    continue
                local_attempts[temperature] += 1
                local_field = field
                for axis in range(bonds.shape[0]):
                    previous = backward[axis, site]
                    local_field += bonds[axis, site] * spins[temperature, forward[axis, site]]
                    local_field += bonds[axis, previous] * spins[temperature, previous]
                delta = 2.0 * spins[temperature, site] * local_field
                if delta <= 0.0 or np.random.random() < np.exp(-beta[temperature] * delta):
                    spins[temperature, site] = -spins[temperature, site]
                    energies[temperature] += delta
                    local_accepts[temperature] += 1

        if (sweep + 1) % exchange_interval == 0:
            for left in range(exchange_event % 2, temperature_count - 1, 2):
                right = left + 1
                swap_attempts[left] += 1
                log_ratio = (beta[left] - beta[right]) * (energies[left] - energies[right])
                if log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio):
                    # Exchange state, energy, and walker identity together.
                    for site in range(size):
                        temporary_spin = spins[left, site]
                        spins[left, site] = spins[right, site]
                        spins[right, site] = temporary_spin
                    temporary_energy = energies[left]
                    energies[left] = energies[right]
                    energies[right] = temporary_energy
                    temporary_walker = walkers[left]
                    walkers[left] = walkers[right]
                    walkers[right] = temporary_walker
                    swap_accepts[left] += 1
            exchange_event += 1

        for temperature in range(temperature_count):
            walker = walkers[temperature]
            visits[walker, temperature] += 1
            if temperature == 0 or temperature == temperature_count - 1:
                if origin[walker] == -1:
                    origin[walker] = temperature
                elif temperature != origin[walker]:
                    reached_opposite[walker] = True
                elif reached_opposite[walker]:
                    round_trips[walker] += 1
                    reached_opposite[walker] = False

        if sweep >= burn_in and (sweep - burn_in + 1) % spacing == 0:
            for temperature in range(temperature_count):
                configurations[temperature, sample_index] = spins[temperature]
                # Recompute at saves so output energies do not accumulate drift.
                energies[temperature] = _energy_flat(spins[temperature], bonds, forward, field)
                saved_energies[temperature, sample_index] = energies[temperature]
            sample_index += 1

    return (configurations, saved_energies, swap_attempts, swap_accepts,
            local_attempts, local_accepts, round_trips, visits)


def sample_parallel_tempering(
    bonds,
    temperatures,
    *,
    samples,
    burn_in,
    spacing,
    seed,
    replicas=2,
    exchange_interval=1,
    field=0.0,
    initial="mixed",
) -> dict:
    """Sample independent CPU PT ladders sharing one quenched bond array.

    Temperatures must be finite, positive, strictly increasing and contain at
    least two entries.  ``replicas`` counts independently seeded whole ladders,
    each containing one walker per temperature.  ``initial='mixed'`` starts
    even-numbered real replicas hot (random independent spins) and odd-numbered
    replicas cold (all +1); ``hot``/``cold`` apply to all real replicas.

    A sweep visits every site once, in a fresh random order at each temperature.
    Each visit proposes a flip with fixed probability 1/2, then uses the
    standard Metropolis acceptance; otherwise the spin stays unchanged.  This
    lazy kernel remains irreducible and aperiodic for finite energies, including
    the all-zero Hamiltonian.  Exchange events occur every ``exchange_interval``
    sweeps.  Events alternate edges (0,1),(2,3),... and (1,2),(3,4),... starting
    with even edges; for two temperatures the odd event has no edge.  After
    ``burn_in`` sweeps, each save follows another ``spacing`` full sweeps.
    Thus the total is ``burn_in + samples*spacing`` sweeps.

    Configurations have shape ``[T,R,S,*shape]`` and energies ``[T,R,S]``.
    Attempt/acceptance counters include burn-in.  ``local_attempts`` counts
    actual flip proposals, ``local_accepts`` accepted flips, and
    ``site_visits[R,T]`` all site visits, including visits with no proposal.
    ``temperature_visits[R,w,T]``
    counts each walker's initial position and position after every full sweep.
    ``round_trips[R,w]`` counts completed first-endpoint -> other-endpoint ->
    first-endpoint journeys, where a walker's first endpoint is its first ever
    visited endpoint; intermediate walkers have no origin until reaching one.

    No equilibration or independence claim is made for hot, cold, or mixed
    starts.  Inputs are never modified, and no automatic extra sampling occurs.
    """
    bonds, shape = _bonds(bonds)
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
    replicas = _integer(replicas, "replicas", 1)
    exchange_interval = _integer(exchange_interval, "exchange_interval", 1, np.iinfo(np.int64).max)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    field = _finite_scalar(field, "field")
    if initial not in ("mixed", "hot", "cold"):
        raise ValueError("initial must be 'mixed', 'hot', or 'cold'")
    size = math.prod(shape)
    temperature_count = temperatures.size
    total_sweeps = burn_in + samples * spacing
    counter_limit = np.iinfo(np.int64).max
    if total_sweeps >= counter_limit or total_sweeps * size > counter_limit:
        raise ValueError("requested sweep count exceeds int64 counter range")
    configuration_count = temperature_count * replicas * samples * size
    if configuration_count > np.iinfo(np.intp).max:
        raise ValueError("requested configurations exceed the platform index range")
    # Finite individual J values need not give a representable total energy.
    with np.errstate(over="ignore", invalid="ignore"):
        energy_bound = np.sum(np.abs(bonds), dtype=np.float64) + abs(field) * size
    if not np.isfinite(energy_bound) or energy_bound > np.finfo(np.float64).max / 4:
        raise ValueError("bond/field magnitudes exceed safe float64 energy arithmetic")

    forward, backward = make_periodic_lattice(shape)
    flattened_bonds = bonds.reshape(len(shape), size)
    configurations = np.empty((temperature_count, replicas, samples, *shape), dtype=np.int8)
    energies = np.empty((temperature_count, replicas, samples), dtype=np.float64)
    swap_attempts = np.empty((replicas, temperature_count - 1), dtype=np.int64)
    swap_accepts = np.empty_like(swap_attempts)
    local_attempts = np.empty((replicas, temperature_count), dtype=np.int64)
    local_accepts = np.empty_like(local_attempts)
    round_trips = np.empty((replicas, temperature_count), dtype=np.int64)
    visits = np.empty((replicas, temperature_count, temperature_count), dtype=np.int64)
    replica_seeds = np.asarray([
        child.generate_state(1, dtype=np.uint32)[0]
        for child in np.random.SeedSequence(seed).spawn(replicas)
    ], dtype=np.uint32)
    if len(np.unique(replica_seeds)) != replicas:
        raise ValueError("Derived real-replica seeds collide; no sampling started")
    replica_initializations = []
    for replica in range(replicas):
        ordered = initial == "cold" or (initial == "mixed" and replica % 2 == 1)
        replica_initializations.append("cold" if ordered else "hot")
        result = _sample_ladder(
            flattened_bonds, forward, backward, beta, samples, burn_in, spacing,
            int(replica_seeds[replica]), exchange_interval, field, ordered,
        )
        configurations[:, replica] = result[0].reshape(temperature_count, samples, *shape)
        energies[:, replica] = result[1]
        swap_attempts[replica] = result[2]
        swap_accepts[replica] = result[3]
        local_attempts[replica] = result[4]
        local_accepts[replica] = result[5]
        round_trips[replica] = result[6]
        visits[replica] = result[7]
        del result
    if not np.all(np.isfinite(energies)):
        raise FloatingPointError("sampling produced nonfinite energies")
    return {
        "configurations": configurations,
        "energies": energies,
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "local_attempts": local_attempts,
        "local_accepts": local_accepts,
        "site_visits": np.full((replicas, temperature_count), size * total_sweeps, dtype=np.int64),
        "round_trips": round_trips,
        "temperature_visits": visits,
        "temperatures": temperatures.copy(),
        "seed": seed,
        "replica_seeds": replica_seeds,
        "shape": shape,
        "dimension": len(shape),
        "samples": samples,
        "burn_in": burn_in,
        "spacing": spacing,
        "replicas": replicas,
        "exchange_interval": exchange_interval,
        "total_sweeps": total_sweeps,
        "field": field,
        "initial": initial,
        "replica_initializations": replica_initializations,
        "round_trip_definition": "first endpoint -> opposite endpoint -> first endpoint, indexed by walker",
        "temperature_visit_definition": "initial position plus end of each sweep, including burn-in",
        "exchange_schedule": "alternating even/odd adjacent edges, even first",
        "proposal_flip_probability": 0.5,
        "equilibration_certified": False,
        "independent_equilibrium_certified": False,
    }
