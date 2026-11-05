"""Fixed-budget, B=0 finite-coupling data for the authorized frozen-Q/C audit.

The target log weight is K0 * O_nn + dot(delta6, O_even6), using exactly the
six extensive even operators in diagnose_ising_mcrg. A nearest-neighbor Wolff
proposal reversible for K0 is Metropolis-corrected by exp(delta6 * Delta O).
Every 32 proposals (including rejections) one target single-spin MH sweep is
performed. Pilots, benchmarks and production have disjoint seed namespaces.
No old physical data, model, optimizer, training label, or MI cache is opened.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from numba import njit
import torch


EVEN_OPERATOR_NAMES = ('nn_axial', 'diagonal', 'plaquette', 'axial_distance2',
                       'rectangle_2x1', 'axial_distance3')
OPERATOR_DEFINITIONS = (
    'sum_x s(x)[s(x+e1)+s(x+e2)]',
    'sum_x s(x)[s(x+e1+e2)+s(x+e1-e2)]',
    'sum_x s(x)s(x+e1)s(x+e2)s(x+e1+e2)',
    'sum_x s(x)[s(x+2e1)+s(x+2e2)]',
    'sum_x s(x)s(x+2e1)s(x+e2)s(x+2e1+e2) + rotated term',
    'sum_x s(x)[s(x+3e1)+s(x+3e2)]',
)
# Each translation of each template is counted once, including both oriented
# bonds/rectangles. Periodic duplicate sites at tiny oracle sizes are retained.
TEMPLATES = (
    (((0, 0), (1, 0)), ((0, 0), (0, 1))),
    (((0, 0), (1, 1)), ((0, 0), (1, -1))),
    (((0, 0), (1, 0), (0, 1), (1, 1)),),
    (((0, 0), (2, 0)), ((0, 0), (0, 2))),
    (((0, 0), (2, 0), (0, 1), (2, 1)),
     ((0, 0), (1, 0), (0, 2), (1, 2))),
    (((0, 0), (3, 0)), ((0, 0), (0, 3))),
)
FORMAL_BUDGET = dict(replicas=12, samples=128, equil=4096, gap=128, pilot_samples=512)
SMOKE_BUDGET = dict(replicas=2, samples=8, equil=16, gap=2, pilot_samples=16)
SEED_PURPOSES = dict(pilot_hot=1, pilot_cold=2, production=3, global_sign=4,
                     response_C=5, benchmark_hot=6, benchmark_cold=7)
QUALITY_THRESHOLDS = dict(hot_cold_abs_z_max=3., ess_fraction_min=.25, abs_lag1_max=.30)
SWEEP_INTERVAL = 32
CPU_BUDGET_HOURS = 24.
BENCHMARK_SAFETY_FACTOR = 2.
EXPERIMENT = 'frozen_QC_finite_even_coupling_response_coverage_v1'


def _bool_smoke(smoke):
    if not isinstance(smoke, bool):
        raise ValueError('smoke must be a bool')


def condition_specs(smoke=False):
    """The immutable 14 training-domain + 54 held-out physical conditions."""
    _bool_smoke(smoke)
    result = []

    def add(logical_size, k, role, family, delta):
        index = len(result)
        size = {108: 16, 54: 8}[logical_size] if smoke else logical_size
        result.append(dict(index=index, condition_index=index, id=f'c{index:03d}',
            condition_id=f'c{index:03d}', size=size, logical_size=logical_size,
            depth=2 if logical_size == 108 else 1, target_size=4 if smoke else 27,
            K0=float(k), role=role, family=family, unit_delta6=[float(v) for v in delta]))

    def axis(logical_size, k, role, family, indices, amplitude):
        for j in indices:
            for sign in (-1., 1.):
                delta = np.zeros(6)
                delta[j] = sign * amplitude
                add(logical_size, k, role, family, delta)

    for k in (.42, .46):
        add(108, k, 'train_domain', 'zero', np.zeros(6))
        axis(108, k, 'train_domain', 'axis_unit', range(3), 1.)
    for side in (108, 54):
        add(side, .44, 'heldout_domain', 'zero', np.zeros(6))
        axis(side, .44, 'heldout_domain', 'axis_unit', range(3), 1.)
        axis(side, .44, 'heldout_domain', 'axis_double', range(3), 2.)
        for signs in itertools.product((-1., 1.), repeat=3):
            add(side, .44, 'heldout_domain', 'mixed_unit', [v / math.sqrt(3) for v in signs] + [0.] * 3)
        axis(side, .44, 'heldout_domain', 'unaugmented_axis', range(3, 6), 1.)
    return result


def namespaced_seed(seed, size, purpose, condition_index, replica=0):
    if isinstance(purpose, str):
        purpose = SEED_PURPOSES[purpose]
    values = (seed, size, purpose, condition_index, replica)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
        raise ValueError('seed namespace components must be nonnegative integers')
    return int(np.random.SeedSequence([int(v) for v in values]).generate_state(1, dtype=np.uint32)[0])


def validate_seed_namespace(seed, smoke=False):
    budget = SMOKE_BUDGET if smoke else FORMAL_BUDGET
    values = [namespaced_seed(seed, point['size'], purpose, point['index'], replica)
              for point in condition_specs(smoke) for purpose in SEED_PURPOSES
              for replica in range(budget['replicas'])]
    if len(set(values)) != len(values):
        raise ValueError('uint32 seed namespace collision; no sampling was started')
    return len(values)


def total_sampling_budget(smoke=False):
    _bool_smoke(smoke)
    b = SMOKE_BUDGET if smoke else FORMAL_BUDGET
    n = len(condition_specs(smoke))
    prod = b['equil'] + b['samples'] * b['gap']
    pilot = b['equil'] + b['pilot_samples'] * b['gap']
    return dict(conditions=n, production_chains=n * b['replicas'], pilot_chains=2 * n,
                production_configurations=n * b['replicas'] * b['samples'],
                pilot_configurations=2 * n * b['pilot_samples'],
                cluster_proposals=n * (b['replicas'] * prod + 2 * pilot),
                local_sweeps=n * (b['replicas'] * (prod // SWEEP_INTERVAL) + 2 * (pilot // SWEEP_INTERVAL)),
                pilot_chains_reused_from_calibration=4,
                local_sweep_definition='Visit every spin exactly once, forward or reverse with equal probability',
                cluster_proposal_definition='One NN Wolff cluster proposed; rejected proposals count too; not lattice sweeps')


@lru_cache(maxsize=16)
def interaction_structure(size, active=None):
    """Prebuild monomial incidence tables, retaining periodic multiplicities.

    site_terms contains a term only when a single flip changes its sign; terms
    containing the same tiny-torus site twice are therefore not overcounted.
    Arrays are internal read-only geometry, never physical sampling data.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError('size must be an integer >=2')
    if active is None:
        active = (True,) * 6
    if len(active) != 6:
        raise ValueError('active operator mask must have six components')
    term_sites, degrees, operators = [], [], []
    incidence = [[] for _ in range(size * size)]
    for op, templates in enumerate(TEMPLATES):
        if not active[op]:
            continue
        for template in templates:
            for i in range(size):
                for j in range(size):
                    sites = [((i + a) % size) * size + (j + b) % size for a, b in template]
                    term = len(term_sites)
                    term_sites.append(sites + [0] * (4 - len(sites)))
                    degrees.append(len(sites))
                    operators.append(op)
                    for site in set(sites):
                        if sites.count(site) % 2:
                            incidence[site].append(term)
    sites_array = np.asarray(term_sites, dtype=np.int32).reshape(-1, 4)
    degree_array = np.asarray(degrees, dtype=np.int8)
    operator_array = np.asarray(operators, dtype=np.int8)
    width = max((len(row) for row in incidence), default=0)
    site_terms = np.zeros((size * size, width), dtype=np.int32)
    site_counts = np.asarray([len(row) for row in incidence], dtype=np.int8)
    for index, row in enumerate(incidence):
        site_terms[index, :len(row)] = row
    return sites_array, degree_array, operator_array, site_terms, site_counts


@njit(cache=True)
def _operator_batch(x, structure):
    sites, degree, operators, _, _ = structure
    result = np.zeros((x.shape[0], 6), dtype=np.float64)
    for sample in range(x.shape[0]):
        flat = x[sample].reshape(-1)
        for term in range(len(degree)):
            product = 1
            for slot in range(degree[term]):
                product *= flat[sites[term, slot]]
            result[sample, operators[term]] += product
    return result


def operator_values_numpy(spins):
    x = np.asarray(spins)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2] or not np.all(np.abs(x) == 1):
        raise ValueError('Expected finite binary configurations [draw,L,L]')
    return _operator_batch(np.ascontiguousarray(x, dtype=np.int8), interaction_structure(x.shape[-1]))


@njit(cache=True)
def _single_spin_delta(flat, site, structure):
    sites, degree, operators, site_terms, site_counts = structure
    result = np.zeros(6, dtype=np.float64)
    for slot in range(site_counts[site]):
        term = site_terms[site, slot]
        product = 1
        for j in range(degree[term]):
            product *= flat[sites[term, j]]
        result[operators[term]] -= 2 * product
    return result


def single_spin_operator_delta(spins, site, structure=None):
    x = np.asarray(spins, dtype=np.int8)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not 0 <= site < x.size:
        raise ValueError('Invalid square configuration or spin index')
    return _single_spin_delta(x.reshape(-1), site, structure or interaction_structure(x.shape[0]))


@njit(cache=True)
def _cluster_delta(flat_proposed, marks, stamp, selected, count, structure, seen_terms):
    sites, degree, operators, site_terms, site_counts = structure
    result = np.zeros(6, dtype=np.float64)
    for i in range(count):
        site = selected[i]
        for slot in range(site_counts[site]):
            term = site_terms[site, slot]
            if seen_terms[term] == stamp:
                continue
            seen_terms[term] = stamp
            parity = 0
            product = 1
            for j in range(degree[term]):
                spin_site = sites[term, j]
                product *= flat_proposed[spin_site]
                if marks[spin_site] == stamp:
                    parity = 1 - parity
            if parity:
                # Here flat_proposed already contains the flipped cluster.
                result[operators[term]] += 2 * product
    return result


def cluster_operator_delta(proposed_spins, cluster_sites, structure=None):
    x = np.asarray(proposed_spins, dtype=np.int8)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError('Expected a square proposed configuration')
    structure = structure or interaction_structure(x.shape[0])
    selected = np.asarray(cluster_sites, dtype=np.int64)
    if len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= x.size):
        raise ValueError('Invalid cluster sites')
    marks = np.zeros(x.size, dtype=np.int64)
    marks[selected] = 1
    if len(selected) > x.size // 2:
        selected = np.flatnonzero(marks == 0).astype(np.int64)
    return _cluster_delta(x.reshape(-1), marks, 1, selected, len(selected), structure,
                          np.zeros(len(structure[1]), dtype=np.int64))


@njit(cache=True)
def _target_sweep(flat, coupling, structure):
    """Random forward/reverse systematic MH: each site exactly once."""
    count = 0
    n = len(flat)
    reverse = np.random.random() < .5
    sites, degree, operators, site_terms, site_counts = structure
    for position in range(n):
        site = n - 1 - position if reverse else position
        log_ratio = 0.
        for slot in range(site_counts[site]):
            term = site_terms[site, slot]
            product = 1
            for j in range(degree[term]):
                product *= flat[sites[term, j]]
            log_ratio -= 2 * product * coupling[operators[term]]
        if log_ratio >= 0. or np.log(np.random.random()) < log_ratio:
            flat[site] *= -1
            count += 1
    return count


@njit(cache=True)
def _chain_kernel(k, delta, size, draws, equil, gap, seed, cold, local_structure, perturb_structure):
    np.random.seed(seed)
    n = size * size
    flat = np.ones(n, dtype=np.int8)
    if not cold:
        for i in range(n):
            flat[i] = 1 if np.random.random() < .5 else -1
    cluster = np.empty(n, dtype=np.int64)
    complement = np.empty(n, dtype=np.int64)
    marks = np.zeros(n, dtype=np.int64)
    seen_terms = np.zeros(len(perturb_structure[1]), dtype=np.int64)
    result = np.empty((draws, size, size), dtype=np.int8)
    p = -np.expm1(-2 * k)
    coupling = delta.copy()
    coupling[0] += k
    has_perturbation = False
    for j in range(6):
        has_perturbation = has_perturbation or delta[j] != 0.
    accepted, sweeps, local_accepted, cluster_sites_proposed = 0, 0, 0, 0
    steps = equil + draws * gap
    for step in range(steps):
        stamp = step + 1
        start = np.random.randint(n)
        target = flat[start]
        flat[start] = -target
        marks[start] = stamp
        cluster[0] = start
        head, tail = 0, 1
        while head < tail:
            z = cluster[head]
            head += 1
            i, j = z // size, z % size
            neighbors = (((i - 1) % size) * size + j, ((i + 1) % size) * size + j,
                         i * size + (j - 1) % size, i * size + (j + 1) % size)
            for neighbor in neighbors:
                if flat[neighbor] == target and np.random.random() < p:
                    flat[neighbor] = -target
                    marks[neighbor] = stamp
                    cluster[tail] = neighbor
                    tail += 1
        cluster_sites_proposed += tail
        log_ratio = 0.
        if has_perturbation and tail != n:
            selected = cluster
            count = tail
            # All terms have even parity: a cluster and its complement flip the
            # same monomial signs. Use the smaller incidence set, not a full
            # operator rescan for every (possibly almost-global) proposal.
            if tail > n // 2:
                count = 0
                for site in range(n):
                    if marks[site] != stamp:
                        complement[count] = site
                        count += 1
                selected = complement
            difference = _cluster_delta(flat, marks, stamp, selected, count, perturb_structure, seen_terms)
            for j in range(6):
                log_ratio += delta[j] * difference[j]
        if log_ratio >= 0. or np.log(np.random.random()) < log_ratio:
            accepted += 1
        else:
            for i in range(tail):
                flat[cluster[i]] *= -1
        if stamp % SWEEP_INTERVAL == 0:
            local_accepted += _target_sweep(flat, coupling, local_structure)
            sweeps += 1
        if step >= equil and (step - equil + 1) % gap == 0:
            index = (step - equil) // gap
            for i in range(size):
                for j in range(size):
                    result[index, i, j] = flat[i * size + j]
    return result, accepted, sweeps, local_accepted, cluster_sites_proposed


def run_chain(k, delta6, size, draws, equil, gap, seed, cold=False):
    """One fixed-length chain; no accept-count or ESS dependent extension."""
    delta = np.asarray(delta6, dtype=np.float64)
    if not np.isfinite(k) or k <= 0 or delta.shape != (6,) or not np.isfinite(delta).all():
        raise ValueError('A positive finite K0 and six finite perturbations are required')
    for name, value, minimum in (('size', size, 2), ('draws', draws, 1), ('equil', equil, 0),
                                  ('gap', gap, 1), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if seed >= 2 ** 32 or not isinstance(cold, bool):
        raise ValueError('Expected uint32 seed and bool cold')
    perturb = interaction_structure(int(size), tuple(bool(v) for v in delta != 0.))
    active = delta != 0.
    active[0] = True
    local = interaction_structure(int(size), tuple(bool(v) for v in active))
    x, accepted, sweeps, local_accepted, proposed_sites = _chain_kernel(
        float(k), delta, int(size), int(draws), int(equil), int(gap), int(seed), cold, local, perturb)
    proposals = int(equil + draws * gap)
    local_proposals = int(sweeps * size * size)
    receipt = dict(cluster_proposals=proposals, cluster_accepted=int(accepted),
                   local_sweeps=int(sweeps), local_spin_proposals=local_proposals,
                   local_spin_accepted=int(local_accepted), cluster_sites_proposed=int(proposed_sites),
                   cluster_acceptance=float(accepted / proposals),
                   local_acceptance=float(local_accepted / local_proposals) if local_proposals else None)
    return x, receipt


def _series_stats(z):
    z = np.asarray(z, dtype=np.float64)
    centered = z - z.mean()
    variance = float(np.mean(centered ** 2))
    rho = [float(np.mean(centered[:-lag] * centered[lag:]) / variance) if variance > 0 else 0.
           for lag in range(1, min(65, len(z) // 2))]
    tau = 1.
    for index in range(0, len(rho) - 1, 2):
        pair = rho[index] + rho[index + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return dict(mean=float(z.mean()), variance_ddof0=variance, lag1=rho[0] if rho else 0.,
                ess_estimate=float(len(z) / tau), tau_int_saved_draws=float(tau),
                se_estimate=float(np.sqrt(variance * tau / len(z))), zero_observed_variance=variance == 0.)


def raw_diagnostics(x):
    """All six raw even densities and raw signed/absolute/m² before Z2 draws."""
    op = operator_values_numpy(x) / x.shape[-1] ** 2
    result = {name: _series_stats(op[:, j]) for j, name in enumerate(EVEN_OPERATOR_NAMES)}
    m = np.mean(x, axis=(-2, -1), dtype=np.float64)
    result.update(magnetization=_series_stats(m), magnetization_squared=_series_stats(m ** 2),
                  absolute_magnetization=_series_stats(np.abs(m)))
    return result


def _hot_cold_comparison(hot, cold):
    result = {}
    for name in hot:
        variance = hot[name]['se_estimate'] ** 2 + cold[name]['se_estimate'] ** 2
        difference = abs(hot[name]['mean'] - cold[name]['mean'])
        result[name] = dict(abs_difference=float(difference),
            abs_z=float(difference / np.sqrt(variance)) if variance > 0 else None, zero_combined_se=variance == 0.)
    return result


def quality_report(pilot, ensembles, *, samples, pilot_samples):
    warnings = []
    comparisons = _hot_cold_comparison(pilot['hot'], pilot['cold'])
    for name, comparison in comparisons.items():
        # Raw signed magnetization is retained, but its sector-sensitive mean
        # does not become an equilibration gate in a zero-field finite chain.
        if name != 'magnetization' and (comparison['abs_z'] is None or comparison['abs_z'] > 3.):
            warnings.append(dict(kind='pilot_hot_cold', observable=name, **comparison))
    production_ranges, pilot_ranges = {}, {}
    for name in pilot['hot']:
        rows = [r['raw_diagnostics'][name] for r in ensembles]
        production_ranges[name] = dict(ess_min=min(r['ess_estimate'] for r in rows),
            ess_max=max(r['ess_estimate'] for r in rows), lag1_min=min(r['lag1'] for r in rows),
            lag1_max=max(r['lag1'] for r in rows))
        rows = [pilot[initial][name] for initial in ('hot', 'cold')]
        pilot_ranges[name] = dict(ess_min=min(r['ess_estimate'] for r in rows),
            ess_max=max(r['ess_estimate'] for r in rows), lag1_min=min(r['lag1'] for r in rows),
            lag1_max=max(r['lag1'] for r in rows))
    for initial in ('hot', 'cold'):
        for name, row in pilot[initial].items():
            if row['zero_observed_variance'] or row['ess_estimate'] < .25 * pilot_samples or abs(row['lag1']) > .30:
                warnings.append(dict(kind='pilot_autocorrelation', initial=initial, observable=name, **row))
    for ensemble in ensembles:
        for name, row in ensemble['raw_diagnostics'].items():
            if row['zero_observed_variance'] or row['ess_estimate'] < .25 * samples or abs(row['lag1']) > .30:
                warnings.append(dict(kind='production_autocorrelation', replica=ensemble['replica'],
                                     observable=name, **row))
    return dict(status='warning' if warnings else 'passed_finite_chain_checks',
        thresholds=QUALITY_THRESHOLDS.copy(), hot_cold_comparison=comparisons,
        production_ranges=production_ranges, pilot_ranges=pilot_ranges, warnings=warnings,
        action='Report only; never automatically resample, extend, rerun, or choose another seed',
        limitation='Finite-chain diagnostics do not prove equilibrium, independent draws or unbiased expectations')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _publish_new_file(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=f'.{path.name}.',
                                          suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Fails rather than clobbering any existing output.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload):
    return (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _atomic_status(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.status.',
                                          suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _new_output(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _checked_chain(k, delta, size, draws, b, seed, cold=False):
    x, kernel = run_chain(k, delta, size, draws, b['equil'], b['gap'], seed, cold)
    x = np.asarray(x)
    if x.shape != (draws, size, size) or x.dtype != np.int8 or not np.all(np.abs(x) == 1):
        raise ValueError('Sampler must return int8 binary configurations with the fixed requested shape')
    expected = b['equil'] + draws * b['gap']
    if kernel['cluster_proposals'] != expected or kernel['local_sweeps'] != expected // 32:
        raise ValueError('Sampler fixed proposal/sweep budget does not match its receipt')
    return x, kernel


def benchmark_budget(epsilon6, seed, smoke=False):
    """Fixed, non-scientific throughput precheck, never an adaptive pilot.

    Actual nonzero positive/negative double-amplitude corrections for ALL six
    axes are benchmarked, including both rectangle orientations and distance3;
    a six-dimensional mixed correction exercises the full incidence table.
    Both hot and cold initializations at both TRAIN K0 values are used. The
    worst L108 CPU time per proposal,
    including its scheduled local sweeps, is applied to ALL remaining sizes,
    then doubled. This deliberately overestimates smaller L54 work.
    """
    _bool_smoke(smoke)
    epsilon = np.asarray(epsilon6, dtype=np.float64)
    size = 16 if smoke else 108
    cases = [('zero', np.zeros(6))]
    for j in range(6):
        for sign in (-1., 1.):
            delta = np.zeros(6)
            delta[j] = sign * 2 * epsilon[j]
            cases.append((f'{EVEN_OPERATOR_NAMES[j]}_{sign:+.0f}_2eps', delta))
    for sign in (-1., 1.):
        cases.append((f'mixed6_{sign:+.0f}_2eps', sign * 2 * epsilon / np.sqrt(6)))
    warm, measured = ((16, 32) if smoke else (128, 256))
    receipts = []
    for k_index, k in enumerate((.42, .46)):
        for case_index, (name, delta) in enumerate(cases):
            for cold in (False, True):
                purpose = 'benchmark_cold' if cold else 'benchmark_hot'
                current_seed = namespaced_seed(seed, size, purpose, k_index * len(cases) + case_index)
                # Warm compilation/geometry preparation is recorded but excluded
                # from throughput; total preflight CPU is added separately below.
                compile_start = time.process_time()
                run_chain(k, delta, size, 1, 0, 1, current_seed, cold)
                compile_cpu = time.process_time() - compile_start
                cpu, wall = time.process_time(), time.perf_counter()
                x, kernel = run_chain(k, delta, size, 1, warm, measured, current_seed, cold)
                # Include all-six-operator statistics, not just the fast transition.
                operator_values_numpy(x)
                cpu_elapsed = time.process_time() - cpu
                wall_elapsed = time.perf_counter() - wall
                receipts.append(dict(name=name, initial='cold' if cold else 'hot', K0=k, size=size,
                    seed=current_seed, delta6=delta.tolist(), cpu_seconds=cpu_elapsed, wall_seconds=wall_elapsed,
                    warmup_compile_cpu_seconds=compile_cpu, kernel=kernel,
                    cpu_seconds_per_cluster_proposal_including_sweeps=cpu_elapsed / kernel['cluster_proposals']))
    worst = max(r['cpu_seconds_per_cluster_proposal_including_sweeps'] for r in receipts)
    budget = total_sampling_budget(smoke)
    b = SMOKE_BUDGET if smoke else FORMAL_BUDGET
    already_pilot = 4 * (b['equil'] + b['pilot_samples'] * b['gap'])
    remaining = budget['cluster_proposals'] - already_pilot
    benchmark_cpu = sum(r['cpu_seconds'] + r['warmup_compile_cpu_seconds'] for r in receipts)
    estimate = BENCHMARK_SAFETY_FACTOR * remaining * worst
    return dict(passed=None, cpu_budget_hours=CPU_BUDGET_HOURS,
        projected_remaining_cpu_seconds_with_margin=float(estimate), benchmark_cpu_seconds=benchmark_cpu,
        safety_factor=BENCHMARK_SAFETY_FACTOR, worst_cpu_seconds_per_cluster_proposal_including_sweeps=worst,
        cases=receipts, non_scientific=True, smoke=smoke, benchmark_budget_not_in_scientific_production=True,
        fixed_benchmark_cases=60, measured_cluster_proposals_per_case=warm + measured,
        total_scientific_sampling_budget=budget, remaining_cluster_proposals_after_calibration=remaining,
        estimate_method='Worst measured L108 per-proposal CPU cost including scheduled sweeps applied to both sizes, times 2',
        limitation='Short throughput precheck is not an equilibrium or mixing certificate; no sampler budget is adapted')


def calibrate(output, *, seed=20261017, smoke=False):
    """Four fixed TRAIN zero-coupling pilots, then a throughput-only gate."""
    _bool_smoke(smoke)
    # Check existing target before any work, including collision/benchmark work.
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    namespace_count = validate_seed_namespace(seed, smoke)
    b = (SMOKE_BUDGET if smoke else FORMAL_BUDGET).copy()
    output = _new_output(output)
    status_path = output / 'status.json'
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    status = dict(schema_version=1, status='running', workflow_complete=False, smoke=smoke,
                  seed=seed, stage='calibration_pilots', completed_pilot_chains=0,
                  total_pilot_chains=4, training_updates=0, MI_estimates=0)
    _atomic_status(status_path, status)
    try:
        pilots = []
        for index in (0, 7):
            point = condition_specs(smoke)[index]
            record = dict(condition_index=index, condition_id=point['id'], K0=point['K0'], size=point['size'])
            for cold in (False, True):
                initial = 'cold' if cold else 'hot'
                current_seed = namespaced_seed(seed, point['size'], f'pilot_{initial}', index)
                status.update(current=dict(condition_index=index, initial=initial, seed=current_seed))
                _atomic_status(status_path, status)
                x, kernel = _checked_chain(point['K0'], np.zeros(6), point['size'], b['pilot_samples'], b, current_seed, cold)
                op = operator_values_numpy(x)
                record[initial] = dict(seed=current_seed, raw_diagnostics=raw_diagnostics(x), kernel=kernel,
                    operator_variance_extensive=np.var(op, axis=0, ddof=0).tolist())
                status['completed_pilot_chains'] += 1
                _atomic_status(status_path, status)
            record['hot_cold_comparison'] = _hot_cold_comparison(record['hot']['raw_diagnostics'], record['cold']['raw_diagnostics'])
            pilots.append(record)
            print(f'幅度校准 TRAIN K0={point["K0"]:.2f} 热/冷 pilot 完成', flush=True)
        sigma = np.sqrt(np.mean([r[initial]['operator_variance_extensive']
                                 for r in pilots for initial in ('hot', 'cold')], axis=0))
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('TRAIN pilot variance does not support six finite nonzero perturbation amplitudes')
        epsilon = .5 / sigma
        status.update(stage='fixed_throughput_precheck', current=None)
        _atomic_status(status_path, status)
        gate = benchmark_budget(epsilon.tolist(), seed, smoke)
        cpu_so_far = time.process_time() - started_cpu
        if 'projected_remaining_cpu_seconds_with_margin' in gate:
            projected = (cpu_so_far + gate['projected_remaining_cpu_seconds_with_margin']) / 3600.
            gate.update(projected_cpu_hours_with_margin=projected, preflight_cpu_seconds=cpu_so_far,
                        passed=bool(projected <= CPU_BUDGET_HOURS))
        elif 'passed' not in gate:
            raise ValueError('Benchmark did not return a budget gate')
        provenance = dict(experiment=EXPERIMENT, smoke=smoke, seed=seed, sampling_budget=b,
            calibration_conditions=[0, 7], amplitude_formula='epsilon_j=0.5/sqrt(mean of four within-pilot Var_ddof0(O_j))',
            pilot_seeds=[r[initial]['seed'] for r in pilots for initial in ('hot', 'cold')],
            operator_definitions=list(OPERATOR_DEFINITIONS), sigma6=sigma.tolist(), epsilon6=epsilon.tolist(),
            no_model_or_exponent_used=True, same_physical_epsilon_at_both_sizes=True)
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result = dict(schema_version=1, experiment=EXPERIMENT, status='complete', workflow_complete=True,
            smoke=smoke, scientific_result=not smoke, seed=seed, sigma6=sigma.tolist(), epsilon6=epsilon.tolist(),
            operator_names=list(EVEN_OPERATOR_NAMES), operator_definitions=list(OPERATOR_DEFINITIONS),
            variance_convention='Within each fixed pilot ddof0, average four variances, then square root; no between-K variance',
            pilot=pilots, pilot_in_production=False, pilot_chains_reused_later=4,
            budget_gate=gate, total_sampling_budget=total_sampling_budget(smoke),
            training_updates=0, MI_estimates=0, seed_collision_checked_count=namespace_count,
            provenance=provenance, metadata_fingerprint=fingerprint,
            elapsed_seconds=time.perf_counter() - started_wall, cpu_seconds=time.process_time() - started_cpu,
            nonadaptive='Amplitudes fixed once from TRAIN pilots; no Q/R/nu or held-out feedback; failed budget gate stops production')
        path = output / 'calibration.json'
        _publish_new_file(path, lambda handle: handle.write(_json_bytes(result)))
        status.update(status='complete', workflow_complete=True, stage='complete', current=None,
            calibration_sha256=file_sha256(path), metadata_fingerprint=fingerprint, budget_gate=gate,
            elapsed_seconds=time.perf_counter() - started_wall)
        _atomic_status(status_path, status)
        return result
    except Exception as error:
        status.update(status='failed', workflow_complete=False, error=dict(type=type(error).__name__, message=str(error)),
                      elapsed_seconds=time.perf_counter() - started_wall)
        _atomic_status(status_path, status)
        raise


def _read_calibration(directory, seed, smoke):
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    if status.get('status') != 'complete' or not status.get('workflow_complete'):
        raise ValueError('Calibration is incomplete')
    path = directory / 'calibration.json'
    sha = file_sha256(path)
    if sha != status.get('calibration_sha256'):
        raise ValueError('Calibration SHA does not match its completion receipt')
    result = json.loads(path.read_text())
    if result.get('seed') != seed:
        raise ValueError('Calibration seed mismatch')
    if result.get('smoke') is not smoke:
        raise ValueError('Calibration smoke mode mismatch')
    if result.get('experiment') != EXPERIMENT:
        raise ValueError('Calibration experiment mismatch')
    if not result.get('budget_gate', {}).get('passed'):
        raise ValueError('Calibration CPU budget gate did not pass; production is forbidden')
    epsilon = np.asarray(result.get('epsilon6'), dtype=np.float64)
    if epsilon.shape != (6,) or not np.isfinite(epsilon).all() or np.any(epsilon <= 0):
        raise ValueError('Invalid calibration epsilon6')
    provenance = result['provenance']
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if fingerprint != result.get('metadata_fingerprint') or fingerprint != status.get('metadata_fingerprint'):
        raise ValueError('Calibration metadata fingerprint mismatch')
    return result, sha


def sample(calibration, condition_index, output, *, seed=20261017, smoke=False):
    """Publish one immutable condition; production never contains pilot draws."""
    _bool_smoke(smoke)
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    points = condition_specs(smoke)
    if isinstance(condition_index, bool) or not isinstance(condition_index, int) or not 0 <= condition_index < len(points):
        raise ValueError('condition index must be in [0,67]')
    namespace_count = validate_seed_namespace(seed, smoke)
    calibration_dir = Path(calibration)
    calibration, calibration_sha = _read_calibration(calibration_dir, seed, smoke)
    point = points[condition_index]
    b = (SMOKE_BUDGET if smoke else FORMAL_BUDGET).copy()
    epsilon = np.asarray(calibration['epsilon6'], dtype=np.float64)
    delta = epsilon * np.asarray(point['unit_delta6'], dtype=np.float64)
    output = _new_output(output)
    status_path = output / 'status.json'
    started = time.perf_counter()
    status = dict(schema_version=1, status='running', workflow_complete=False, smoke=smoke,
        index=condition_index, id=point['id'], condition_index=condition_index, condition_id=point['id'], size=point['size'],
        role=point['role'], family=point['family'], stage='pilot', completed_pilot_chains=0,
        total_pilot_chains=2, completed_production_chains=0, total_production_chains=b['replicas'],
        training_updates=0, MI_estimates=0)
    _atomic_status(status_path, status)
    try:
        reused = condition_index in (0, 7)
        pilot = dict(reused_from_calibration=reused, seeds={}, kernel={})
        if reused:
            original = next(r for r in calibration['pilot'] if r['condition_index'] == condition_index)
            for initial in ('hot', 'cold'):
                pilot[initial] = original[initial]['raw_diagnostics']
                pilot['seeds'][initial] = original[initial]['seed']
                pilot['kernel'][initial] = original[initial]['kernel']
            status['completed_pilot_chains'] = 2
            status['new_pilot_chains'] = 0
        else:
            for cold in (False, True):
                initial = 'cold' if cold else 'hot'
                current_seed = namespaced_seed(seed, point['size'], f'pilot_{initial}', condition_index)
                status.update(current=dict(stage='pilot', initial=initial, seed=current_seed))
                _atomic_status(status_path, status)
                x, kernel = _checked_chain(point['K0'], delta, point['size'], b['pilot_samples'], b, current_seed, cold)
                pilot[initial], pilot['seeds'][initial], pilot['kernel'][initial] = raw_diagnostics(x), current_seed, kernel
                status['completed_pilot_chains'] += 1
                _atomic_status(status_path, status)
            status['new_pilot_chains'] = 2
        pilot['hot_cold_comparison'] = _hot_cold_comparison(pilot['hot'], pilot['cold'])
        print(f'{point["id"]} L={point["size"]} 有限耦合热/冷 pilot 完成（复用={reused}）', flush=True)
        shape = (b['replicas'], b['samples'], point['size'], point['size'])
        configurations = np.empty(shape, dtype=np.int8)
        chain_seeds = np.empty(b['replicas'], dtype=np.int64)
        sign_seeds = np.empty(b['replicas'], dtype=np.int64)
        ensembles = []
        status['stage'] = 'production'
        for replica in range(b['replicas']):
            chain_seed = namespaced_seed(seed, point['size'], 'production', condition_index, replica)
            sign_seed = namespaced_seed(seed, point['size'], 'global_sign', condition_index, replica)
            status.update(current=dict(stage='production', replica=replica, seed=chain_seed))
            _atomic_status(status_path, status)
            x, kernel = _checked_chain(point['K0'], delta, point['size'], b['samples'], b, chain_seed)
            diagnostic = raw_diagnostics(x)  # Must precede the exact B=0 sign augmentation.
            ensembles.append(dict(replica=replica, chain_seed=chain_seed, global_flip_seed=sign_seed,
                                  raw_diagnostics=diagnostic, kernel=kernel))
            signs = np.random.default_rng(sign_seed).choice(np.array([-1, 1], np.int8), size=(b['samples'], 1, 1))
            configurations[replica] = x * signs
            chain_seeds[replica], sign_seeds[replica] = chain_seed, sign_seed
            status['completed_production_chains'] += 1
            _atomic_status(status_path, status)
            print(f'{point["id"]} L={point["size"]} 独立生产链 {replica + 1}/{b["replicas"]}', flush=True)
        quality = quality_report(pilot, ensembles, samples=b['samples'], pilot_samples=b['pilot_samples'])
        provenance = dict(experiment=EXPERIMENT, condition=point, seed=seed, smoke=smoke,
            budget=b, delta6=delta.tolist(), epsilon6=epsilon.tolist(), calibration_sha256=calibration_sha,
            chain_seeds=chain_seeds.tolist(), global_flip_seeds=sign_seeds.tolist(),
            pilot_seeds=pilot['seeds'], operator_definitions=list(OPERATOR_DEFINITIONS),
            sampler='NN reversible Wolff proposal + exact perturbation MH + full target MH sweep every 32 proposals')
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        all_kernel = [r['kernel'] for r in ensembles] + [pilot['kernel'][initial] for initial in ('hot', 'cold')]
        new_kernel = [r['kernel'] for r in ensembles] + ([] if reused else [pilot['kernel'][initial] for initial in ('hot', 'cold')])
        report = dict(schema_version=1, experiment=EXPERIMENT, status='complete', workflow_complete=True,
            scientific_result=not smoke, smoke=smoke, seed=seed, condition=point,
            id=point['id'], index=condition_index, condition_id=point['id'], condition_index=condition_index,
            role=point['role'], family=point['family'], unit_delta6=point['unit_delta6'],
            size=point['size'], logical_size=point['logical_size'], depth=point['depth'], max_rg_steps=point['depth'],
            target_size=point['target_size'], block_factor=2,
            lattice_sizes=[point['size'] // 2 ** i for i in range(point['depth'] + 1)],
            K0=point['K0'], delta6=delta.tolist(), epsilon6=epsilon.tolist(),
            parameter_role='Physical generation/evaluation metadata only; forbidden as network inputs or targets',
            target_log_weight='K0*O_nn + dot(delta6,O_even6); all operators extensive, both rectangle orientations',
            operator_names=list(EVEN_OPERATOR_NAMES), operator_definitions=list(OPERATOR_DEFINITIONS),
            sampler=provenance['sampler'], cluster_acceptance_rule='min(1,exp(dot(delta6,O_proposed-O_current)))',
            local_acceptance_rule='min(1,exp(K0*Delta O_nn + dot(delta6,Delta O_even6)))',
            sweep_interval_cluster_proposals=32, sweep_order='equal-probability forward/reverse; visits each spin once',
            sampler_stationarity='Every component preserves the full target; scheduled composition need not itself be reversible',
            sampling_budget=b, **b, pilot=pilot, ensembles=ensembles, pilot_in_production=False,
            pilot_generated_configurations_here=0 if reused else 2 * b['pilot_samples'],
            production_saved_configurations=b['replicas'] * b['samples'],
            configuration_shape=list(shape), configuration_dtype='torch.int8', quality=quality,
            total_cluster_proposals_including_reused_pilots=sum(r['cluster_proposals'] for r in all_kernel),
            new_cluster_proposals_here=sum(r['cluster_proposals'] for r in new_kernel),
            new_local_sweeps_here=sum(r['local_sweeps'] for r in new_kernel),
            global_sign_symmetrization='Exact independent B=0 Z2 sign, applied only after raw diagnostics; pilots excluded',
            seed_derivation='uint32 SeedSequence([seed,size,purpose,condition_index,replica])',
            seed_purposes=SEED_PURPOSES.copy(), seed_collision_checked_count=namespace_count,
            independent_physical_chains=True, C_repeats_are_not_independent_physical_chains=True,
            train_replica_rule='replica<1 for smoke, replica<8 formal, only in train_domain; labels never consumed by model',
            no_adaptive_sampling=True, training_updates=0, MI_estimates=0,
            calibration_sha256=calibration_sha, provenance=provenance, metadata_fingerprint=fingerprint)
        payload = dict(schema_version=1, configurations=torch.from_numpy(configurations),
            chain_seeds=torch.from_numpy(chain_seeds), global_flip_seeds=torch.from_numpy(sign_seeds),
            condition_index=condition_index, condition_id=point['id'], size=point['size'],
            logical_size=point['logical_size'], depth=point['depth'], max_rg_steps=point['depth'],
            target_size=point['target_size'], block_factor=2, seed=seed, smoke=smoke,
            metadata_fingerprint=fingerprint, training_updates=0, MI_estimates=0)
        status.update(stage='publishing', current=None)
        _atomic_status(status_path, status)
        json.dumps(report, allow_nan=False)
        data_path = output / 'data.pt'
        _publish_new_file(data_path, lambda handle: torch.save(payload, handle))
        report.update(data_sha256=file_sha256(data_path), elapsed_seconds=time.perf_counter() - started)
        sidecar_path = output / 'sidecar.json'
        _publish_new_file(sidecar_path, lambda handle: handle.write(_json_bytes(report)))
        status.update(status='complete', workflow_complete=True, stage='complete', current=None,
            data_sha256=report['data_sha256'], sidecar_sha256=file_sha256(sidecar_path), metadata_fingerprint=fingerprint,
            configuration_shape=list(shape), quality_status=quality['status'], warning_count=len(quality['warnings']),
            elapsed_seconds=time.perf_counter() - started)
        _atomic_status(status_path, status)
        return report
    except Exception as error:
        status.update(status='failed', workflow_complete=False, error=dict(type=type(error).__name__, message=str(error)),
                      elapsed_seconds=time.perf_counter() - started)
        _atomic_status(status_path, status)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    calibration = subcommands.add_parser('calibrate', help='Four fixed TRAIN pilots and non-scientific throughput gate')
    sampling = subcommands.add_parser('sample', help='One fixed physical condition, no retry/resume/extension')
    for command in (calibration, sampling):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--seed', type=int, default=20261017)
        command.add_argument('--smoke', action='store_true')
    sampling.add_argument('--calibration', type=Path, required=True)
    sampling.add_argument('--condition-index', type=int, required=True)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop('command')
    return calibrate(**args) if command == 'calibrate' else sample(**args)


if __name__ == '__main__':
    main()
