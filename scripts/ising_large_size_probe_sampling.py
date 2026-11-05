"""Small, new-output-only physical sample for a frozen large-size QCR probe.

The six throughput cases are execution checks, never scientific samples.  The
two production chains per K start hot/cold; they are *not* independent pilots
and cannot certify equilibrium.  Nothing here constructs a model or optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


EXPERIMENT = 'frozen_b3_large_size_small_sample_probe_v1'
SEED_NAMESPACE = 0x4C535031
DEFAULT_SEED = 20261018
CONDITIONS = (.42, .44, .46)
SAMPLES = 64
REPLICAS = 2
EQUIL = 4096
GAP = 128
BENCHMARK_EQUIL = 64
BENCHMARK_GAP = 128
CPU_BUDGET_SECONDS = 4. * 3600.
CPU_RESERVE_SECONDS = 120.
SAFETY_FACTOR = 2.
SWEEP_INTERVAL = 32
PURPOSES = {'benchmark': 1, 'production': 2, 'global_sign': 3, 'setup_warmup': 4}
OBSERVABLE_NAMES = ('nn_axial', 'diagonal', 'plaquette', 'axial_distance2',
                    'rectangle_2x1', 'axial_distance3', 'magnetization',
                    'absolute_magnetization', 'magnetization_squared')


def _kernel_module():
    # Reuse the frozen transition kernel, not a second NN-only implementation.
    # Its module imports tensor libraries but no model/data/MI is loaded here.
    from scripts import ising_finite_perturbation_data
    return ising_finite_perturbation_data


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _validate_size_seed(size, seed):
    size, seed = _integer(size, 'size', 27), _integer(seed, 'seed')
    if size % 27 or seed >= 2 ** 32:
        raise ValueError('size must be divisible by 27 and seed must fit uint32')
    return size, seed


def namespaced_seed(seed, size, purpose, condition_index, replica):
    seed, size = _integer(seed, 'seed'), _integer(size, 'size', 2)
    condition_index = _integer(condition_index, 'condition_index')
    replica = _integer(replica, 'replica')
    if purpose not in PURPOSES or condition_index >= len(CONDITIONS) or replica >= REPLICAS:
        raise ValueError('Unknown seed purpose, condition, or replica')
    return int(np.random.SeedSequence([seed, SEED_NAMESPACE, size, PURPOSES[purpose],
                                      condition_index, replica]).generate_state(1, dtype=np.uint32)[0])


def validate_seed_namespace(seed, size):
    values = [namespaced_seed(seed, size, purpose, condition, replica)
              for purpose in PURPOSES for condition in range(len(CONDITIONS))
              for replica in range(REPLICAS)]
    if len(values) != len(set(values)):
        raise ValueError('Seed collision; no chain was started')
    return len(values)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _publish_new(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=f'.{path.name}.',
                                          suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication, never replace a prior result.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _status(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.status.',
                                          suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _new_stage(root, relative):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Probe root must not be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    output = root / relative
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'No overwrite or restart permitted: {output}')
    if any(parent.is_symlink() for parent in output.parents if parent != root.parent):
        raise ValueError('Output ancestors must not be symlinks')
    output.mkdir(parents=True, exist_ok=False)
    return output


def _read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _source_receipt():
    module = _kernel_module()
    return {'sampler_file_sha256': file_sha256(Path(__file__)),
            'frozen_kernel_file': str(Path(module.__file__).resolve()),
            'frozen_kernel_file_sha256': file_sha256(module.__file__),
            'kernel': 'scripts.ising_finite_perturbation_data.run_chain',
            'target': 'exp(K0 * O_nn), delta6 identically zero, B=0',
            'local_sweep_interval_cluster_proposals': SWEEP_INTERVAL,
            'cluster_proposal_is_not_lattice_sweep': True,
            'training_updates': 0, 'MI_estimates': 0,
            'model_loaded': False, 'old_physical_data_opened': False}


def raw_observable_series(spins):
    """Nine raw density series, one configuration at a time, O(L²) scratch.

    Roll expressions match all six frozen even-monomial definitions, including
    *both* rectangle orientations.  Do not build the all-six incidence table at
    L1080 just to calculate diagnostics.
    """
    x = np.asarray(spins)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2] or x.shape[0] < 1:
        raise ValueError('Expected nonempty [draw,L,L] square configurations')
    if x.dtype != np.int8 or not np.all((x == -1) | (x == 1)):
        raise ValueError('Expected binary int8 configurations')
    result = np.empty((len(x), len(OBSERVABLE_NAMES)), dtype=np.float64)
    for index, s in enumerate(x):
        # Negative roll shifts mean s(x+a), with periodic boundaries.
        shift = lambda a, b: np.roll(s, (-a, -b), axis=(0, 1))
        e1, e2, diagonal = shift(1, 0), shift(0, 1), shift(1, 1)
        mean = lambda value: np.mean(value, dtype=np.float64)
        result[index, 0] = mean(s * e1) + mean(s * e2)
        result[index, 1] = mean(s * diagonal) + mean(s * shift(1, -1))
        result[index, 2] = mean(s * e1 * e2 * diagonal)
        result[index, 3] = mean(s * shift(2, 0)) + mean(s * shift(0, 2))
        result[index, 4] = (mean(s * shift(2, 0) * e2 * shift(2, 1))
                            + mean(s * e1 * shift(0, 2) * shift(1, 2)))
        result[index, 5] = mean(s * shift(3, 0)) + mean(s * shift(0, 3))
        m = mean(s)
        result[index, 6:] = (m, abs(m), m * m)
    return {name: result[:, j] for j, name in enumerate(OBSERVABLE_NAMES)}


def raw_diagnostics(spins):
    module = _kernel_module()
    return {name: module._series_stats(z) for name, z in raw_observable_series(spins).items()}


def production_quality(chains):
    """Reuse fixed diagnostic thresholds without falsely labelling pilots.

    The old helper has a mandatory ``pilot`` parameter.  Feed the *same two
    production chains* to obtain its fixed hot/cold algorithm, then remove the
    duplicated pilot-autocorrelation warnings and rename pilot-only fields.
    """
    module = _kernel_module()
    hot, cold = (chain['raw_diagnostics'] for chain in chains)
    quality = module.quality_report({'hot': hot, 'cold': cold}, chains,
                                    samples=SAMPLES, pilot_samples=SAMPLES)
    warnings = []
    for warning in quality['warnings']:
        if warning['kind'] == 'pilot_autocorrelation':
            continue  # Exactly duplicated by production_autocorrelation below.
        warning = dict(warning)
        if warning['kind'] == 'pilot_hot_cold':
            warning['kind'] = 'production_hot_cold'
        warnings.append(warning)
    return dict(status='warning' if warnings else 'passed_finite_chain_checks',
                thresholds=quality['thresholds'], production_ranges=quality['production_ranges'],
                hot_cold_comparison=module._hot_cold_comparison(hot, cold), warnings=warnings,
                hot_cold_source='The same two production chains, not additional independent pilots',
                independent_pilot_chains=0, production_chains=REPLICAS,
                diagnostic_algorithm='Frozen _series_stats / _hot_cold_comparison / quality_report',
                action='Record only; never automatically resample, extend, rerun, or change seed',
                limitation='Two finite chains cannot certify equilibrium, independence, confidence intervals, or absence of bias')


def _checked_chain(k, size, draws, equil, gap, seed, cold):
    x, receipt = _kernel_module().run_chain(k, np.zeros(6, dtype=np.float64), size,
                                          draws, equil, gap, seed, cold)
    x = np.asarray(x)
    if x.shape != (draws, size, size) or x.dtype != np.int8 or not np.all((x == -1) | (x == 1)):
        raise ValueError('Frozen sampler returned an invalid shape/dtype/spin value')
    proposals = equil + draws * gap
    if receipt['cluster_proposals'] != proposals or receipt['local_sweeps'] != proposals // SWEEP_INTERVAL:
        raise ValueError('Frozen sampler violated the fixed proposal/sweep budget')
    return x, receipt


def benchmark(root, *, size=1080, seed=DEFAULT_SEED):
    size, seed = _validate_size_seed(size, seed)
    output = _new_stage(root, 'benchmark')
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    status = dict(schema_version=1, experiment=EXPERIMENT, status='running', workflow_complete=False,
                  stage='throughput_benchmark', size=size, seed=seed, non_scientific=True,
                  completed_cases=0, training_updates=0, MI_estimates=0)
    _status(output / 'status.json', status)
    try:
        checked_count = validate_seed_namespace(seed, size)
        setup_import_start = time.process_time()
        sources = _source_receipt()
        setup_import_cpu = time.process_time() - setup_import_start
        # Initialize frozen NN geometry and JIT/cache once, separately from the
        # six 192-proposal throughput cases.  One proposal is non-scientific and
        # its seed differs from every benchmark and production chain.
        setup_seed = namespaced_seed(seed, size, 'setup_warmup', 0, 0)
        status.update(stage='non_scientific_setup_warmup', current=dict(seed=setup_seed))
        _status(output / 'status.json', status)
        cpu, wall = time.process_time(), time.perf_counter()
        setup_spins, setup_kernel = _checked_chain(CONDITIONS[0], size, 1, 0, 1, setup_seed, False)
        raw_diagnostics(setup_spins)
        setup_cpu, setup_wall = time.process_time() - cpu, time.perf_counter() - wall
        setup_receipt = dict(seed=setup_seed, K0=CONDITIONS[0], size=size, draws=1, equil=0, gap=1,
                             kernel=setup_kernel, cpu_seconds=setup_cpu, wall_seconds=setup_wall,
                             source_import_cpu_seconds=setup_import_cpu, non_scientific=True,
                             retained_or_reused=False)
        future_setup_cpu = setup_cpu + setup_import_cpu
        del setup_spins
        status.update(stage='throughput_benchmark', current=None, setup_warmup=setup_receipt)
        _status(output / 'status.json', status)
        cases, projections = [], []
        for index, k in enumerate(CONDITIONS):
            for replica, cold in enumerate((False, True)):
                current_seed = namespaced_seed(seed, size, 'benchmark', index, replica)
                status['current'] = dict(condition_index=index, K0=k, initial='cold' if cold else 'hot')
                _status(output / 'status.json', status)
                cpu, wall = time.process_time(), time.perf_counter()
                x, kernel = _checked_chain(k, size, 1, BENCHMARK_EQUIL, BENCHMARK_GAP, current_seed, cold)
                raw_diagnostics(x)  # Include diagnostic cost; discard all benchmark spins.
                elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
                cases.append(dict(condition_index=index, K0=k, size=size, replica=replica,
                                  initial='cold' if cold else 'hot', seed=current_seed,
                                  cpu_seconds=elapsed_cpu, wall_seconds=elapsed_wall, kernel=kernel,
                                  cpu_seconds_per_cluster_proposal_including_sweeps=elapsed_cpu / kernel['cluster_proposals']))
                status['completed_cases'] += 1
                _status(output / 'status.json', status)
                del x
            worst = max(case['cpu_seconds_per_cluster_proposal_including_sweeps']
                        for case in cases if case['condition_index'] == index)
            production_proposals = REPLICAS * (EQUIL + SAMPLES * GAP)
            projections.append(dict(condition_index=index, K0=k,
                                    worst_cpu_seconds_per_cluster_proposal_including_sweeps=worst,
                                    production_cluster_proposals=production_proposals,
                                    projected_cpu_seconds=worst * production_proposals * SAFETY_FACTOR,
                                    process_setup_cpu_seconds=future_setup_cpu))
        actual_cpu = time.process_time() - started_cpu
        setup_total = len(CONDITIONS) * future_setup_cpu
        projected = (actual_cpu + sum(row['projected_cpu_seconds'] for row in projections)
                     + setup_total + CPU_RESERVE_SECONDS)
        passed = bool(projected <= CPU_BUDGET_SECONDS)
        gate = dict(passed=passed, cpu_budget_seconds=CPU_BUDGET_SECONDS, cpu_reserve_seconds=CPU_RESERVE_SECONDS,
                    safety_factor=SAFETY_FACTOR, benchmark_actual_cpu_seconds=actual_cpu,
                    projected_sample_process_setup_cpu_seconds=setup_total,
                    process_setup_cpu_seconds=future_setup_cpu,
                    projected_total_cpu_seconds=projected, conditions=projections,
                    estimate_method='For each K: max(hot,cold) CPU/proposal × 24576 production proposals × 2; add actual preflight CPU, one measured import/geometry/JIT setup per future sample process, and 120-second overhead reserve')
        receipt = dict(schema_version=1, experiment=EXPERIMENT,
                       status='complete' if passed else 'budget_hold', workflow_complete=True,
                       size=size, seed=seed, non_scientific=True, scientific_samples_reused=False,
                       benchmark_equil=BENCHMARK_EQUIL, benchmark_gap=BENCHMARK_GAP, benchmark_draws=1,
                       setup_warmup=setup_receipt, fixed_cases=6, cases=cases, budget_gate=gate, sources=sources,
                       seed_collision_checked_count=checked_count, training_updates=0, MI_estimates=0,
                       cpu_seconds=time.process_time() - started_cpu,
                       elapsed_seconds=time.perf_counter() - started_wall,
                       limitation='A short throughput benchmark is not a mixing or equilibrium test; production budget is not adapted')
        _publish_new(output / 'receipt.json', lambda handle: handle.write(_bytes(receipt)))
        status.update(status=receipt['status'], workflow_complete=True, current=None,
                      budget_gate=gate, requires_user_action=not passed,
                      receipt_sha256=file_sha256(output / 'receipt.json'),
                      cpu_seconds=receipt['cpu_seconds'], elapsed_seconds=receipt['elapsed_seconds'])
        _status(output / 'status.json', status)
        return receipt
    except Exception as error:
        status.update(status='failed', error=dict(type=type(error).__name__, message=str(error)),
                      requires_user_action=True, cpu_seconds=time.process_time() - started_cpu,
                      elapsed_seconds=time.perf_counter() - started_wall)
        _status(output / 'status.json', status)
        raise


def _verified_benchmark(root, size, seed):
    root = Path(root)
    receipt, status = _read(root / 'benchmark/receipt.json'), _read(root / 'benchmark/status.json')
    if status.get('receipt_sha256') != file_sha256(root / 'benchmark/receipt.json'):
        raise ValueError('Benchmark receipt fingerprint mismatch')
    if (receipt.get('experiment') != EXPERIMENT or receipt.get('size') != size or receipt.get('seed') != seed
            or receipt.get('status') != 'complete' or not receipt.get('workflow_complete')
            or not receipt.get('budget_gate', {}).get('passed') or status.get('status') != 'complete'):
        raise ValueError('A completed, passing, same-size/seed benchmark is required; no sampling was started')
    current_sources = _source_receipt()
    for field in ('frozen_kernel_file_sha256', 'sampler_file_sha256'):
        if receipt['sources'][field] != current_sources[field]:
            raise ValueError('Frozen sampler/kernel fingerprint changed after benchmark')
    return receipt


def _condition_gate(root, condition_index, benchmark_receipt):
    used = float(benchmark_receipt['cpu_seconds'])
    for previous in range(condition_index):
        directory = Path(root) / 'data' / f'c{previous:03d}'
        receipt, status = _read(directory / 'sidecar.json'), _read(directory / 'status.json')
        if (receipt.get('status') != 'complete' or status.get('status') != 'complete'
                or status.get('sidecar_sha256') != file_sha256(directory / 'sidecar.json')):
            raise ValueError('Earlier sequential sampling stage is incomplete or fingerprint mismatch')
        used += float(receipt['cpu_seconds'])
    remaining = sum(row['projected_cpu_seconds'] + row['process_setup_cpu_seconds']
                    for row in benchmark_receipt['budget_gate']['conditions']
                    if row['condition_index'] >= condition_index)
    projected = used + remaining + CPU_RESERVE_SECONDS
    return dict(passed=projected <= CPU_BUDGET_SECONDS, cpu_budget_seconds=CPU_BUDGET_SECONDS,
                actual_completed_cpu_seconds=used, projected_remaining_cpu_seconds=remaining,
                cpu_reserve_seconds=CPU_RESERVE_SECONDS, projected_total_cpu_seconds=projected)


def sample(root, *, condition_index, size=1080, seed=DEFAULT_SEED):
    size, seed = _validate_size_seed(size, seed)
    index = _integer(condition_index, 'condition_index')
    if index >= len(CONDITIONS):
        raise ValueError('Only condition indices 0, 1, 2 are allowed')
    root = Path(root)
    # Do not create even a failed stage if the preflight or predecessor is invalid.
    benchmark_receipt = _verified_benchmark(root, size, seed)
    gate = _condition_gate(root, index, benchmark_receipt)
    output = _new_stage(root, Path('data') / f'c{index:03d}')
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    status = dict(schema_version=1, experiment=EXPERIMENT, status='running', workflow_complete=False,
                  condition_index=index, condition_id=f'c{index:03d}', K0=CONDITIONS[index],
                  size=size, seed=seed, completed_production_chains=0, training_updates=0, MI_estimates=0,
                  budget_gate=gate)
    _status(output / 'status.json', status)
    if not gate['passed']:
        result = dict(status, status='budget_hold', workflow_complete=True, requires_user_action=True,
                      production_started=False, cpu_seconds=time.process_time() - started_cpu,
                      elapsed_seconds=time.perf_counter() - started_wall)
        _publish_new(output / 'sidecar.json', lambda handle: handle.write(_bytes(result)))
        result['sidecar_sha256'] = file_sha256(output / 'sidecar.json')
        _status(output / 'status.json', result)
        return result
    try:
        sources, chains, stored = _source_receipt(), [], []
        for replica, cold in enumerate((False, True)):
            current_seed = namespaced_seed(seed, size, 'production', index, replica)
            sign_seed = namespaced_seed(seed, size, 'global_sign', index, replica)
            status['current'] = dict(replica=replica, initial='cold' if cold else 'hot', seed=current_seed)
            _status(output / 'status.json', status)
            cpu, wall = time.process_time(), time.perf_counter()
            x, kernel = _checked_chain(CONDITIONS[index], size, SAMPLES, EQUIL, GAP, current_seed, cold)
            raw_series = raw_observable_series(x)  # Raw sector/time dependence, before independent Z2 signs.
            diagnostics = {name: _kernel_module()._series_stats(z) for name, z in raw_series.items()}
            series = {name: z.tolist() for name, z in raw_series.items()}
            rng = np.random.default_rng(sign_seed)
            signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=SAMPLES)
            x *= signs[:, None, None]
            stored.append(x)
            chains.append(dict(replica=replica, initial='cold' if cold else 'hot', seed=current_seed,
                               global_sign_seed=sign_seed, global_signs=signs.tolist(),
                               raw_diagnostics=diagnostics, raw_observable_series=series, kernel=kernel,
                               cpu_seconds=time.process_time() - cpu, elapsed_seconds=time.perf_counter() - wall))
            status['completed_production_chains'] += 1
            _status(output / 'status.json', status)
        quality = production_quality(chains)
        data = np.stack(stored, axis=0)
        path = output / 'spins.npy'
        _publish_new(path, lambda handle: np.save(handle, data, allow_pickle=False))
        result = dict(schema_version=1, experiment=EXPERIMENT, status='complete', workflow_complete=True,
                      condition_index=index, condition_id=f'c{index:03d}', K0=CONDITIONS[index], size=size,
                      seed=seed, target_delta6=[0.] * 6, B=0.,
                      sampling_budget=dict(replicas=REPLICAS, samples=SAMPLES, equil=EQUIL, gap=GAP,
                                           independent_pilot_chains=0),
                      spin_shape=list(data.shape), spin_dtype='int8', spins_file='spins.npy',
                      spins_sha256=file_sha256(path), global_Z2_flipped=True,
                      raw_diagnostics_before_global_Z2=True, raw_series_before_global_Z2=True,
                      chain_axis='Independent physical chain; replica 0 hot, replica 1 cold',
                      chains=chains, quality=quality, sources=sources, budget_gate=gate,
                      benchmark_samples_reused=False, training_updates=0, MI_estimates=0,
                      scientific_claim='Exploratory small sample only; no significance, equilibrium, or critical-exponent claim',
                      cpu_seconds=time.process_time() - started_cpu,
                      elapsed_seconds=time.perf_counter() - started_wall)
        _publish_new(output / 'sidecar.json', lambda handle: handle.write(_bytes(result)))
        status.update(status='complete', workflow_complete=True, current=None, quality=quality,
                      spins_sha256=result['spins_sha256'], sidecar_sha256=file_sha256(output / 'sidecar.json'),
                      cpu_seconds=result['cpu_seconds'], elapsed_seconds=result['elapsed_seconds'],
                      requires_user_action=False)
        _status(output / 'status.json', status)
        return result
    except Exception as error:
        status.update(status='failed', error=dict(type=type(error).__name__, message=str(error)),
                      requires_user_action=True, cpu_seconds=time.process_time() - started_cpu,
                      elapsed_seconds=time.perf_counter() - started_wall)
        _status(output / 'status.json', status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1080)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('benchmark')
    sample_parser = subparsers.add_parser('sample')
    sample_parser.add_argument('--condition-index', type=int, required=True)
    args = parser.parse_args(argv)
    kwargs = dict(size=args.size, seed=args.seed)
    result = benchmark(args.root, **kwargs) if args.command == 'benchmark' else sample(
        args.root, condition_index=args.condition_index, **kwargs)
    print(json.dumps({key: result[key] for key in ('status', 'workflow_complete', 'cpu_seconds', 'elapsed_seconds')},
                     allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
