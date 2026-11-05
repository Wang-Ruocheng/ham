"""Independent B=0 Ising caches for the explicitly authorized b=2 size audit.

This diagnostic-only cache never trains Q/C/R and never estimates MI.  The
historical Wolff kernel is unchanged.  The two initial sizes are independent
simulations, not two members of the same coarse-graining trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from scripts.prepare_ising2d_lr_data import _publish_new_file, file_sha256
from scripts.prepare_ising2d_qcr import chain, stats


K_VALUES = (.43619047619047624, .44068679350977147, .45142857142857146)
FORMAL_BUDGET = dict(replicas=12, samples=128, equil=4096, gap=128, pilot_samples=512)
SMOKE_BUDGET = dict(replicas=2, samples=8, equil=16, gap=2, pilot_samples=16)
SEED_PURPOSES = dict(pilot_hot=1, pilot_cold=2, production=3, global_sign=4, response_C=5)
QUALITY_THRESHOLDS = dict(hot_cold_abs_z_max=3., ess_fraction_min=.25, abs_lag1_max=.30)


def namespaced_seed(seed, size, purpose, K_index, replica=0):
    """A uint32 stream identifier, with size and sampling purpose separated.

    ``response_C`` is reserved for the later fixed-model diagnostic collector;
    this preparation script does not invoke C or use that stream.
    """
    if isinstance(purpose, str):
        purpose = SEED_PURPOSES[purpose]
    values = (seed, size, purpose, K_index, replica)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError('Seed namespace components must be nonnegative integers')
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def _options(size, seed, replicas, samples, equil, gap, pilot_samples, smoke):
    values = dict(size=size, seed=seed, replicas=replicas, samples=samples,
                  equil=equil, gap=gap, pilot_samples=pilot_samples)
    if not isinstance(smoke, bool):
        raise ValueError('smoke must be a bool')
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} must be a valid positive integer (seed may be zero)')
    budget = {name: values[name] for name in FORMAL_BUDGET}
    if smoke:
        if size not in (8, 16):
            raise ValueError('Independent smoke data must use size 8 or 16, ending at 4')
        # The public function retains its formal defaults; --smoke alone
        # explicitly selects the one predeclared, independent smoke budget.
        if budget == FORMAL_BUDGET:
            budget = SMOKE_BUDGET.copy()
        if budget != SMOKE_BUDGET:
            raise ValueError('Smoke budget is fixed at 2 replicas, 8 draws, equil 16, gap 2, pilot 16')
        target = 4
    else:
        if size not in (54, 108):
            raise ValueError('Formal matched-size data require independent initial size 54 or 108')
        if budget != FORMAL_BUDGET:
            raise ValueError('Formal sampling budget is fixed; reduced budgets require --smoke')
        target = 27
    depth = int(math.log2(size // target))
    if target * 2 ** depth != size:
        raise ValueError('Explicit b=2 trajectory must end exactly at the target size')
    return dict(size=size, seed=seed, **budget, smoke=smoke,
                block_factor=2, max_rg_steps=depth, target_size=target,
                lattice_sizes=[size // 2 ** i for i in range(depth + 1)])


def _atomic_status(path, payload):
    """Replace only our status file inside a directory created exclusively."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.status.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _series_stats(z):
    """The historical finite-chain paired-positive autocorrelation estimator."""
    z = np.asarray(z, dtype=np.float64)
    centered = z - z.mean()
    variance = float(np.mean(centered ** 2))
    rho = [float(np.mean(centered[:-lag] * centered[lag:]) / variance)
           if variance > 0 else 0. for lag in range(1, min(65, len(z) // 2))]
    tau = 1.
    for index in range(0, len(rho) - 1, 2):
        pair = rho[index] + rho[index + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return dict(mean=float(z.mean()), lag1=rho[0] if rho else 0.,
                ess_estimate=float(len(z) / tau), tau_int_saved_draws=float(tau),
                se_estimate=float(np.sqrt(variance * tau / len(z))),
                zero_observed_variance=variance == 0.)


def raw_diagnostics(x):
    """Retain unchanged energy/m² stats and add signed/absolute magnetization."""
    result = stats(x)
    for record in result.values():
        record['tau_int_saved_draws'] = float(len(x) / record['ess_estimate'])
        record['zero_observed_variance'] = record['se_estimate'] == 0.
    magnetization = x.mean(axis=(1, 2), dtype=np.float64)
    result['magnetization'] = _series_stats(magnetization)
    result['absolute_magnetization'] = _series_stats(np.abs(magnetization))
    # H=-K*sum_nn s_i*s_j; this energy-like observable is sum_nn/N,
    # i.e. minus the exchange energy per site when J=1.
    return result


def _checked_chain(k, options, draws, seed, cold=False):
    x = np.asarray(chain(k, options['size'], draws, options['equil'], options['gap'], seed, cold))
    if x.shape != (draws, options['size'], options['size']):
        raise ValueError('Sampler returned an unexpected configuration shape')
    if not np.all((x == -1) | (x == 1)):
        raise ValueError('Sampler must return finite binary spins in {-1, +1}')
    return x.astype(np.int8, copy=False)


def _hot_cold_z(hot, cold):
    result = {}
    for name in hot:
        # Signed magnetization is sector sensitive and is reported, not used
        # as an equilibration gate at B=0.
        variance = hot[name]['se_estimate'] ** 2 + cold[name]['se_estimate'] ** 2
        delta = abs(hot[name]['mean'] - cold[name]['mean'])
        result[name] = dict(abs_difference=float(delta),
                            abs_z=float(delta / np.sqrt(variance)) if variance > 0 else None,
                            zero_combined_se=variance == 0.)
    return result


def _quality(report):
    warnings = []
    for pilot in report['pilot']:
        for name, comparison in pilot['hot_cold_comparison'].items():
            if name == 'magnetization':
                continue
            if comparison['abs_z'] is None or comparison['abs_z'] > QUALITY_THRESHOLDS['hot_cold_abs_z_max']:
                warnings.append(dict(kind='pilot_hot_cold', K_index=pilot['K_index'], observable=name,
                                     **comparison))
    ranges, pilot_ranges = {}, {}
    for name in report['ensembles'][0]['raw_diagnostics']:
        records = [row['raw_diagnostics'][name] for row in report['ensembles']]
        ess = [row['ess_estimate'] for row in records]
        lag1 = [row['lag1'] for row in records]
        ranges[name] = dict(ess_min=float(min(ess)), ess_max=float(max(ess)),
                            lag1_min=float(min(lag1)), lag1_max=float(max(lag1)))
        pilot_records = [row[initial][name] for row in report['pilot'] for initial in ('hot', 'cold')]
        pilot_ess = [row['ess_estimate'] for row in pilot_records]
        pilot_lag1 = [row['lag1'] for row in pilot_records]
        pilot_ranges[name] = dict(ess_min=float(min(pilot_ess)), ess_max=float(max(pilot_ess)),
                                  lag1_min=float(min(pilot_lag1)), lag1_max=float(max(pilot_lag1)))
    for row in report['pilot']:
        for initial in ('hot', 'cold'):
            for name, diagnostic in row[initial].items():
                if (diagnostic['zero_observed_variance']
                        or diagnostic['ess_estimate'] < QUALITY_THRESHOLDS['ess_fraction_min'] * report['pilot_samples']
                        or abs(diagnostic['lag1']) > QUALITY_THRESHOLDS['abs_lag1_max']):
                    warnings.append(dict(kind='pilot_autocorrelation', K_index=row['K_index'], initial=initial,
                                         observable=name, **diagnostic))
    for row in report['ensembles']:
        for name, diagnostic in row['raw_diagnostics'].items():
            if (diagnostic['zero_observed_variance']
                    or diagnostic['ess_estimate'] < QUALITY_THRESHOLDS['ess_fraction_min'] * report['samples']
                    or abs(diagnostic['lag1']) > QUALITY_THRESHOLDS['abs_lag1_max']):
                warnings.append(dict(kind='production_autocorrelation', K_index=row['K_index'],
                                     replica=row['replica'], observable=name,
                                     ess_estimate=diagnostic['ess_estimate'], lag1=diagnostic['lag1'],
                                     zero_observed_variance=diagnostic['zero_observed_variance']))
    return dict(status='warning' if warnings else 'passed_finite_chain_checks',
                thresholds=QUALITY_THRESHOLDS.copy(), production_ranges=ranges, pilot_ranges=pilot_ranges,
                warnings=warnings,
                action='Report only; never automatically resample, extend, rerun, or choose another seed',
                limitation='Finite-chain ESS/IAT and hot/cold checks do not prove independence or equilibration')


def prepare_size(output, *, size, seed=20261016, replicas=12, samples=128,
                 equil=4096, gap=128, pilot_samples=512, smoke=False):
    """Create one exclusive diagnostic-only directory and return its sidecar.

    ``smoke=True`` selects the fixed small budget when formal defaults are
    supplied.  Pilot chains are discarded after diagnostics, never inserted
    into the saved production ensemble.  No historical cache is opened.
    """
    options = _options(size, seed, replicas, samples, equil, gap, pilot_samples, smoke)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    # Check collisions across both independent sizes and all reserved purposes
    # before creating an output or drawing any configurations.
    seed_sizes = (8, 16) if smoke else (54, 108)
    all_seeds = [namespaced_seed(seed, side, purpose, index, replica)
                 for side in seed_sizes for purpose in SEED_PURPOSES.values()
                 for index in range(len(K_VALUES)) for replica in range(options['replicas'])]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Namespaced uint32 seed collision; no sampling was started')
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / 'status.json'
    started = time.perf_counter()
    status = dict(schema_version=1, status='running', workflow_complete=False,
                  smoke=smoke, size=size, seed=seed, stage='pilot',
                  completed_pilot_chains=0, total_pilot_chains=2 * len(K_VALUES),
                  completed_production_chains=0, total_production_chains=len(K_VALUES) * options['replicas'],
                  training_updates=0, MI_estimates=0)
    _atomic_status(status_path, status)
    report = dict(schema_version=1, experiment='independent_initial_sizes_same_terminal_b2_v1',
                  diagnostic_only=True, smoke=smoke, scientific_result=not smoke,
                  sampler='Wolff fixed cluster count, numba v2',
                  sampler_implementation='Unchanged scripts.prepare_ising2d_qcr.chain/stats',
                  physical_model='Periodic square Ising; positive K; B=0',
                  **{key: options[key] for key in ('size', 'seed', 'replicas', 'samples', 'equil', 'gap',
                      'pilot_samples', 'block_factor', 'max_rg_steps', 'target_size', 'lattice_sizes')},
                  depth=options['max_rg_steps'],
                  K_values=list(K_VALUES), K_role='Generation and evaluation only; no training labels',
                  independent_sizes=True, training_updates=0, MI_estimates=0,
                  seed_derivation='uint32 SeedSequence([seed, size, purpose, K_index, replica])',
                  seed_purposes=SEED_PURPOSES.copy(),
                  seed_collision_checked_sizes=list(seed_sizes),
                  sampling_spin_flip=dict(enabled=True, B=0.,
                      scope='Independent global sign per stored configuration, after raw diagnostics',
                      justification='Exact zero-field global Z2 symmetry; even observables unchanged',
                      raw_signed_magnetization_retained_in_sidecar=True),
                  autocorrelation_convention=dict(
                      estimator='Historical paired-positive autocorrelation sum, max lag min(64, n//2-1)',
                      tau_convention='tau_int_saved_draws=1+2*sum(rho); ESS=n/tau',
                      energy_observable='bond_sum_per_site=sum_nn(s_i*s_j)/L²; minus energy/site for J=1',
                      no_pooling='Each independent physical chain is diagnosed separately',
                      threshold_scope='Descriptive predeclared checks; no data-dependent budget adaptation'),
                  data_format='schema1 dict: int8 configurations [K, replica, draw, L, L], float64 K',
                  production_initial_state='hot', pilot_in_production=False,
                  pilot=[], ensembles=[])
    try:
        for index, k in enumerate(K_VALUES):
            record = dict(K_index=index, K=k, samples=options['pilot_samples'], seeds={})
            for cold in (False, True):
                label = 'cold' if cold else 'hot'
                current_seed = namespaced_seed(seed, size, f'pilot_{label}', index)
                status.update(current=dict(stage='pilot', K_index=index, K=k, initial=label,
                                           seed=current_seed))
                _atomic_status(status_path, status)
                x = _checked_chain(k, options, options['pilot_samples'], current_seed, cold)
                record[label] = raw_diagnostics(x)
                record['seeds'][label] = current_seed
                status['completed_pilot_chains'] += 1
                _atomic_status(status_path, status)
            record['hot_cold_comparison'] = _hot_cold_z(record['hot'], record['cold'])
            report['pilot'].append(record)
            print(f'质检 L={size} K={k:.12f} 热/冷初态完成', flush=True)
        shape = (len(K_VALUES), options['replicas'], options['samples'], size, size)
        configurations = np.empty(shape, dtype=np.int8)
        chain_seeds = np.empty(shape[:2], dtype=np.int64)
        sign_seeds = np.empty(shape[:2], dtype=np.int64)
        status['stage'] = 'production'
        for index, k in enumerate(K_VALUES):
            for replica in range(options['replicas']):
                current_seed = namespaced_seed(seed, size, 'production', index, replica)
                sign_seed = namespaced_seed(seed, size, 'global_sign', index, replica)
                status.update(current=dict(stage='production', K_index=index, K=k,
                                           replica=replica, seed=current_seed))
                _atomic_status(status_path, status)
                x = _checked_chain(k, options, options['samples'], current_seed)
                diagnostic = raw_diagnostics(x)
                report['ensembles'].append(dict(K_index=index, K=k, replica=replica,
                    chain_seed=current_seed, global_flip_seed=sign_seed, raw_diagnostics=diagnostic))
                signs = np.random.default_rng(sign_seed).choice(np.array([-1, 1], dtype=np.int8),
                                                                size=(options['samples'], 1, 1))
                configurations[index, replica] = x * signs
                chain_seeds[index, replica] = current_seed
                sign_seeds[index, replica] = sign_seed
                status['completed_production_chains'] += 1
                _atomic_status(status_path, status)
                print(f'独立采样 L={size} K={k:.12f} 系综={replica + 1}/{options["replicas"]}', flush=True)
        report['quality'] = _quality(report)
        report.update(configuration_shape=list(shape), configuration_dtype='torch.int8',
                      production_saved_configurations=int(np.prod(shape[:3])),
                      pilot_generated_configurations=2 * len(K_VALUES) * options['pilot_samples'],
                      pilot_saved_configurations=0,
                      equilibration_cluster_updates_per_chain=options['equil'],
                      gap_cluster_updates=options['gap'],
                      total_cluster_updates=(len(K_VALUES) * options['replicas'] *
                          (options['equil'] + options['samples'] * options['gap']) +
                          2 * len(K_VALUES) * (options['equil'] + options['pilot_samples'] * options['gap'])))
        provenance = dict(sampling_options=options, K_values=list(K_VALUES),
                          chain_seeds=chain_seeds.tolist(), global_flip_seeds=sign_seeds.tolist(),
                          sampler='Wolff fixed cluster count, numba v2',
                          seed_derivation=report['seed_derivation'],
                          experiment=report['experiment'])
        metadata_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        report.update(metadata_fingerprint=metadata_fingerprint, provenance=provenance)
        payload = dict(schema_version=1, configurations=torch.from_numpy(configurations),
                       K=torch.tensor(K_VALUES, dtype=torch.float64),
                       chain_seeds=torch.from_numpy(chain_seeds), global_flip_seeds=torch.from_numpy(sign_seeds),
                       size=size, seed=seed, replicas=options['replicas'], samples=options['samples'],
                       block_factor=2, max_rg_steps=options['max_rg_steps'],
                       depth=options['max_rg_steps'], metadata_fingerprint=metadata_fingerprint,
                       target_size=options['target_size'], smoke=smoke,
                       diagnostic_only=True, training_updates=0, MI_estimates=0)
        status.update(stage='publishing', current=None)
        _atomic_status(status_path, status)
        json.dumps(report, allow_nan=False)
        data_path = output / 'data.pt'
        _publish_new_file(data_path, lambda stream: torch.save(payload, stream))
        report.update(data_sha256=file_sha256(data_path), elapsed_seconds=time.perf_counter() - started,
                      status='complete', workflow_complete=True)
        serialized = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')
        sidecar_path = output / 'sidecar.json'
        _publish_new_file(sidecar_path, lambda stream: stream.write(serialized))
        status.update(status='complete', workflow_complete=True, stage='complete', current=None,
                      data_sha256=report['data_sha256'], sidecar_sha256=file_sha256(sidecar_path),
                      metadata_fingerprint=metadata_fingerprint,
                      quality_status=report['quality']['status'],
                      warning_count=len(report['quality']['warnings']),
                      elapsed_seconds=time.perf_counter() - started)
        _atomic_status(status_path, status)
        return report
    except Exception as error:
        status.update(status='failed', workflow_complete=False,
                      error=dict(type=type(error).__name__, message=str(error)),
                      elapsed_seconds=time.perf_counter() - started)
        _atomic_status(status_path, status)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--smoke', action='store_true')
    for name, value in FORMAL_BUDGET.items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=value)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return prepare_size(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
