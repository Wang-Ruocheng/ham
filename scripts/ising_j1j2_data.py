"""CPU-only sampling, quality receipts and label-free cache assembly for J1--J2.

Every ladder repeat is an independent physical source.  Symmetry augmentation
is applied only after raw diagnostics and never creates independent samples or
repairs poor thermalization.  No model, GPU, MI, or historical data is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from recurrent_rg.data.ising_j1j2_pt import sample_ladder, seed_for_group, series_diagnostics


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f'Refusing symlink output: {path}')
    encoded = json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(encoded)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('x') as stream:
            stream.write(encoded)
        temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _validate_retained_grid(p, temps, ids):
    """Validate the optional new retained-K contract before any output I/O.

    Older frozen plans have only saved temperature indices and keep their
    historical meaning.  For the uniform-K campaign, auxiliary PT slots must
    form the exact complement of the twelve retained data conditions.
    """
    fields = ('training_coupling_grid', 'retained_K_values',
              'retained_temperature_values', 'auxiliary_temperature_indices')
    if not any(key in p for key in fields):
        return
    if any(key not in p for key in fields) or p['training_coupling_grid'] != 'uniform_K':
        raise ValueError('Incomplete or unsupported retained training coupling grid')
    retained_k = np.asarray(p['retained_K_values'], dtype=np.float64)
    retained_t = np.asarray(p['retained_temperature_values'], dtype=np.float64)
    auxiliary = np.asarray(p['auxiliary_temperature_indices'])
    expected_k = np.linspace(.25, 2.5, 12, dtype=np.float64)[::-1]
    if (len(ids) != 12 or np.any(np.diff(ids) <= 0)
            or retained_k.shape != (12,) or not np.array_equal(retained_k, expected_k)
            or retained_t.shape != (12,) or np.any(~np.isfinite(retained_t))
            or not np.array_equal(retained_t, 1. / expected_k)
            or not np.array_equal(temps[ids], retained_t)
            or not np.allclose(1. / temps[ids], retained_k, rtol=2e-16, atol=0.)):
        raise ValueError('Retained conditions must be the twelve uniform K values on [.25, 2.5]')
    expected_auxiliary = np.array([i for i in range(len(temps)) if i not in set(ids.tolist())])
    if (auxiliary.ndim != 1 or auxiliary.dtype.kind not in 'iu'
            or not np.array_equal(auxiliary, expected_auxiliary)):
        raise ValueError('Auxiliary temperature indices must be exactly the unsaved PT slots')


def _grid_metadata(p):
    """Public physical labels and slot roles, never inserted into training PT."""
    retained_t = [p['temperatures'][i] for i in p['saved_temperature_indices']]
    return dict(training_coupling_grid=p.get('training_coupling_grid', 'historical_temperature_subset'),
        coupling_definition='K=J1/T, J1=1',
        retained_K_values=p.get('retained_K_values', [1. / t for t in retained_t]),
        retained_temperature_values=retained_t,
        retained_temperature_indices=p['saved_temperature_indices'],
        total_sampling_temperature_count=len(p['temperatures']),
        retained_condition_count=len(retained_t),
        auxiliary_temperature_count=len(p['temperatures']) - len(retained_t),
        auxiliary_temperatures_are_sampling_only=True,
        auxiliary_rows_in_training_cache=0,
        physical_labels_external_only=True)


def _physical(plan):
    p = plan['physical']
    required = ('size', 'rhos', 'train_rhos', 'heldout_rhos', 'temperatures',
        'saved_temperature_indices', 'groups', 'train_groups', 'samples',
        'burn_in', 'spacing', 'master_seed', 'initials')
    if any(key not in p for key in required):
        raise ValueError('Incomplete physical sampling plan')
    temps = np.asarray(p['temperatures'], dtype=np.float64)
    ids = np.asarray(p['saved_temperature_indices'])
    if (temps.ndim != 1 or len(temps) < 2 or np.any(~np.isfinite(temps))
            or np.any(temps <= 0) or np.any(np.diff(temps) <= 0)):
        raise ValueError('Require increasing positive finite temperatures')
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(temps))
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError('Saved temperature indices must be unique valid integers')
    _validate_retained_grid(p, temps, ids)
    if (p['groups'] != 12 or p['train_groups'] != 8
            or p['initials'] != ['hot', 'ferro', 'stripe_x', 'stripe_y']):
        raise ValueError('This campaign requires 12 repeats, split8/4, and four fixed initial states')
    if (not set(p['train_rhos']).isdisjoint(p['heldout_rhos'])
            or set(p['rhos']) != set(p['train_rhos']) | set(p['heldout_rhos'])
            or len(set(p['rhos'])) != len(p['rhos'])):
        raise ValueError('Train and heldout rho sets must partition unique rho values')
    if p['size'] < 4 or p['size'] % 2:
        raise ValueError('The lazy four-sublattice kernel needs an even size >=4')
    return p


def _run(p, *, rho, purpose, group, initial, burn_in, samples, spacing, size=None):
    size = p['size'] if size is None else size
    seed = seed_for_group(p['master_seed'], size, rho, purpose, group)
    return sample_ladder(size, rho, p['temperatures'], seed, burn_in, samples,
                         spacing, initial=initial)


def _diagnostics(result):
    records = []
    for index, row in enumerate(result['diagnostics']):
        item = dict(row)
        item['rotation_invariant_order'] = series_diagnostics(
            result['magnetization'][index] ** 2 + result['stripe_x'][index] ** 2
            + result['stripe_y'][index] ** 2)
        records.append(item)
    return records


def _receipt(result, *, purpose, plan_sha256, rho_index, group):
    if (result['max_component_tracking_drift'] != 0
            or result['max_energy_tracking_drift'] > 1e-8):
        raise ValueError('Independent recomputation found an energy tracking drift')
    attempts = np.asarray(result['swap_attempts'])
    accepts = np.asarray(result['swap_accepts'])
    rates = np.divide(accepts, attempts, out=np.zeros_like(accepts, dtype=float), where=attempts > 0)
    return dict(schema_version=1, status='complete', workflow_complete=True,
        completed_at=_now(), purpose=purpose, physical_plan_sha256=plan_sha256,
        rho_index=rho_index, rho=result['rho'], group=group, size=result['size'],
        seed=result['seed'], initial=result['initial'], temperatures=result['temperatures'],
        samples=result['samples'], spacing=result['spacing'], burn_in=result['burn_in'],
        total_sweeps=result['total_sweeps'], sweeps_per_temperature=result['sweeps_per_temperature'],
        visited_sites=result['visited_sites'], cpu_seconds=result['cpu_seconds'],
        wall_seconds=result['wall_seconds'], numba_used=result['numba_used'],
        max_component_tracking_drift=result['max_component_tracking_drift'],
        max_energy_tracking_drift=result['max_energy_tracking_drift'],
        swap_attempts=attempts, swap_accepts=accepts, exchange_acceptance=rates,
        min_exchange_acceptance=float(rates.min()), local_attempts=result['local_attempts'],
        local_accepts=result['local_accepts'], round_trips=result['round_trips'],
        temperature_visits=result['temperature_visits'], diagnostics=_diagnostics(result),
        raw_diagnostics_before_symmetry_augmentation=True,
        independent_unit='whole_temperature_ladder_group',
        temperatures_within_group_are_correlated=True, equilibration_certified=False,
        trace_saved='unaugmented raw arrays in physical shard; pilot diagnostics only')


def _individual_quality(receipt, criteria):
    failures, warnings = [], []
    if receipt['max_component_tracking_drift'] != 0 or receipt['max_energy_tracking_drift'] > 1e-8:
        failures.append(dict(kind='energy_tracking_drift'))
    if receipt['min_exchange_acceptance'] < criteria['min_exchange']:
        failures.append(dict(kind='low_exchange', actual=receipt['min_exchange_acceptance'],
                             threshold=criteria['min_exchange']))
    for index, row in enumerate(receipt['diagnostics']):
        for observable in ('energy_per_site', 'rotation_invariant_order'):
            d = row[observable]
            if d['ess'] is None:
                warnings.append(dict(kind='constant_or_short_not_mixing_evidence',
                                     temperature_index=index, observable=observable))
            elif d['ess'] < criteria['min_ess']:
                failures.append(dict(kind='low_ess', temperature_index=index,
                    observable=observable, actual=d['ess'], threshold=criteria['min_ess']))
    return failures, warnings


def _hot_cold(records, criteria, *, production):
    """Compare raw group means, using whole-repeat uncertainty in production."""
    checks, failures = [], []
    for rho in sorted({r['rho'] for r in records}):
        subset = [r for r in records if r['rho'] == rho]
        hot = [r for r in subset if r['initial'] == 'hot']
        ordered_names = ('ferro',) if rho < .5 else ('stripe_x', 'stripe_y')
        cold = [r for r in subset if r['initial'] in ordered_names]
        if not hot or not cold:
            failures.append(dict(kind='missing_hot_or_ordered_group', rho=rho))
            continue
        for index in range(len(subset[0]['temperatures'])):
            for name in ('energy_per_site', 'rotation_invariant_order'):
                a = [r['diagnostics'][index][name] for r in hot]
                b = [r['diagnostics'][index][name] for r in cold]
                ma, mb = np.array([d['mean'] for d in a]), np.array([d['mean'] for d in b])
                gap = abs(float(ma.mean() - mb.mean()))
                if production:
                    variance = (float(ma.var(ddof=1)) / len(ma) if len(ma) > 1 else 0.)
                    variance += (float(mb.var(ddof=1)) / len(mb) if len(mb) > 1 else 0.)
                else:
                    # A pilot has one hot ladder: this is a finite-chain check,
                    # not an independent-repeat uncertainty certification.
                    variance = sum(d['variance'] / max(1., d['ess'] or 1.) for d in a) / len(a) ** 2
                    variance += sum(d['variance'] / max(1., d['ess'] or 1.) for d in b) / len(b) ** 2
                se = math.sqrt(max(0., variance))
                z = gap / se if se > 0 else None
                zero_se_disagreement = se == 0 and gap > 0
                bad = (gap > criteria['hot_cold_abs_threshold']
                       and (zero_se_disagreement or
                            (z is not None and z > criteria['hot_cold_z_threshold'])))
                row = dict(rho=rho, temperature_index=index,
                    temperature=subset[0]['temperatures'][index], observable=name,
                    hot_mean=float(ma.mean()), ordered_mean=float(mb.mean()),
                    gap=gap, standard_error=se, z=z,
                    zero_standard_error_with_nonzero_gap=zero_se_disagreement, failed=bad,
                    hot_groups=len(a), ordered_groups=len(b),
                    uncertainty='whole_repeat_means' if production else 'finite_chain_ESS_estimate')
                checks.append(row)
                if bad:
                    failures.append(dict(kind='hot_cold_disagreement', **row))
    return checks, failures


def benchmark(plan, root):
    """Short executable check and measured throughput, not scientific output."""
    p, root = _physical(plan), Path(root)
    directory = root / 'data' / 'benchmark'
    if (directory / 'receipt.json').exists() or (directory / 'status.json').exists():
        raise FileExistsError('Benchmark is exclusive and cannot be rerun automatically')
    directory.mkdir(parents=True, exist_ok=True)
    _json(directory / 'status.json', dict(status='running', current='numba_warmup', started_at=_now()))
    warm = sample_ladder(6, 0., [.5, 1., 2., 4.],
        seed_for_group(p['master_seed'], 6, 0., 'benchmark_warmup', 0),
        1, 2, 1, initial='hot')
    rows = []
    for index, rho in enumerate(plan['pilot']['rhos']):
        _json(directory / 'status.json', dict(status='running', current=f'rho={rho}',
            completed=len(rows), total=len(plan['pilot']['rhos'])), exclusive=False)
        result = _run(p, rho=rho, purpose='benchmark', group=index, initial='hot',
                      burn_in=16, samples=8, spacing=2)
        record = _receipt(result, purpose='benchmark', plan_sha256=_fingerprint(p),
                          rho_index=p['rhos'].index(rho), group=index)
        record['cpu_seconds_per_visited_site'] = result['cpu_seconds'] / result['visited_sites']
        _json(directory / f'r{index:02d}.json', record)
        rows.append(_plain(record))
    per_site = max(row['cpu_seconds_per_visited_site'] for row in rows)
    production_sites = (len(p['rhos']) * p['groups'] * len(p['temperatures']) * p['size'] ** 2
                        * (p['burn_in'] + p['samples'] * p['spacing']))
    result = dict(status='complete', workflow_complete=True, scientific_result=False,
        warmup_cpu_seconds=warm['cpu_seconds'], warmup_wall_seconds=warm['wall_seconds'],
        cpu_seconds=warm['cpu_seconds'] + sum(row['cpu_seconds'] for row in rows),
        throughput_rows=rows, measured_worst_cpu_seconds_per_visited_site=per_site,
        projected_production_cpu_seconds=per_site * production_sites,
        projection_safety_factor=2., conservative_production_cpu_seconds=2 * per_site * production_sites,
        projection_caution='short hot-start timing is not a guaranteed wall time or equilibrium result',
        report_path=str(directory / 'receipt.json'), physical_plan_sha256=_fingerprint(p))
    _json(directory / 'receipt.json', result)
    _json(directory / 'status.json', dict(status='complete', workflow_complete=True), exclusive=False)
    return result


def pilot(plan, root):
    """Fixed pilot ladders; a failed scientific gate never triggers more sampling."""
    p, root = _physical(plan), Path(root)
    criteria = plan['pilot']
    directory = root / 'data' / 'pilot'
    if (directory / 'receipt.json').exists() or (directory / 'status.json').exists():
        raise FileExistsError('Pilot cannot overwrite or retry previous execution')
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [(rho, initial) for rho in criteria['rhos']
             for initial in (('hot', 'ferro') if rho < .5 else ('hot', 'stripe_x', 'stripe_y'))]
    _json(directory / 'status.json', dict(status='running', completed=0, total=len(tasks)))
    rows, failures, warnings = [], [], []
    for index, (rho, initial) in enumerate(tasks):
        _json(directory / 'status.json', dict(status='running', completed=index,
            total=len(tasks), current=dict(rho=rho, initial=initial)), exclusive=False)
        result = _run(p, rho=rho, purpose='pilot', group=index, initial=initial,
            burn_in=criteria['burn_in'], samples=criteria['samples'], spacing=criteria['spacing'])
        record = _plain(_receipt(result, purpose='pilot', plan_sha256=_fingerprint(p),
                           rho_index=p['rhos'].index(rho), group=index))
        _json(directory / f"p{index:02d}_r{record['rho_index']:02d}_{initial}.json", record)
        failed, warned = _individual_quality(record, criteria)
        failures.extend(dict(pilot=index, **row) for row in failed)
        warnings.extend(dict(pilot=index, **row) for row in warned)
        rows.append(record)
    comparisons, disagreements = _hot_cold(rows, criteria, production=False)
    failures.extend(disagreements)
    status = 'scientific_stop' if failures else 'complete'
    result = dict(status=status, workflow_complete=True, scientific_gate_passed=not failures,
        requires_user_action=bool(failures), cpu_seconds=sum(row['cpu_seconds'] for row in rows),
        ladders=len(rows), failures=failures, warnings=warnings, hot_cold=comparisons,
        equilibrium_certified=False, automatic_resampling=False,
        physical_plan_sha256=_fingerprint(p), report_path=str(directory / 'receipt.json'))
    _json(directory / 'receipt.json', result)
    _json(directory / 'status.json', result, exclusive=False)
    return result


def sample_group(plan, root, rho_index, group):
    p, root = _physical(plan), Path(root)
    if (isinstance(rho_index, bool) or not isinstance(rho_index, int)
            or not 0 <= rho_index < len(p['rhos']) or isinstance(group, bool)
            or not isinstance(group, int) or not 0 <= group < p['groups']):
        raise ValueError('Invalid production rho/repeat identity')
    directory = root / 'data' / 'shards'
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f'r{rho_index:02d}_g{group:02d}'
    data, sidecar, status = (prefix.with_suffix(suffix) for suffix in ('.pt', '.sidecar.json', '.status.json'))
    if any(path.exists() or path.is_symlink() for path in (data, sidecar, status)):
        raise FileExistsError('Production shard cannot be overwritten or retried')
    _json(status, dict(status='running', rho_index=rho_index, group=group, started_at=_now()))
    initial = p['initials'][group % len(p['initials'])]
    raw = _run(p, rho=p['rhos'][rho_index], purpose='production', group=group,
               initial=initial, burn_in=p['burn_in'], samples=p['samples'], spacing=p['spacing'])
    record = _receipt(raw, purpose='production', plan_sha256=_fingerprint(p),
                      rho_index=rho_index, group=group)
    selected = raw['configs'][p['saved_temperature_indices']].copy()
    aug_seed = seed_for_group(p['master_seed'], p['size'], p['rhos'][rho_index],
                              'production_symmetry', group)
    generator = np.random.default_rng(aug_seed)
    signs = 2 * generator.integers(0, 2, selected.shape[:2], dtype=np.int8) - 1
    rotations = generator.integers(0, 4, selected.shape[:2], dtype=np.int8)
    for t in range(len(selected)):
        for sample in range(p['samples']):
            selected[t, sample] = signs[t, sample] * np.rot90(selected[t, sample], int(rotations[t, sample]))
    import torch
    payload = dict(schema_version=1, configurations=torch.from_numpy(selected),
        raw={key: torch.from_numpy(raw[key]) for key in
             ('energies', 'energy_nn', 'energy_diagonal', 'magnetization', 'stripe_x', 'stripe_y')},
        raw_trace={key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                   for key, value in raw['trace'].items()},
        symmetry_signs=torch.from_numpy(signs), symmetry_rotations=torch.from_numpy(rotations))
    with data.open('xb') as stream:
        torch.save(payload, stream)
    record.update(data_path=str(data.relative_to(root)), data_sha256=_sha(data),
        saved_temperature_indices=p['saved_temperature_indices'],
        saved_temperatures=[p['temperatures'][i] for i in p['saved_temperature_indices']],
        saved_K_values=[1. / p['temperatures'][i] for i in p['saved_temperature_indices']],
        temperature_grid=_grid_metadata(p),
        configuration_shape=list(selected.shape), dtype='torch.int8',
        symmetry_augmentation=dict(seed=aug_seed, operations=['global_Z2', 'quarter_turn_rotation'],
            independent_samples_added=0, fixes_thermalization=False,
            raw_diagnostics_precede_augmentation=True))
    _json(sidecar, record)
    _json(status, dict(status='complete', workflow_complete=True, returncode=0,
        data_sha256=record['data_sha256'], sidecar_sha256=_sha(sidecar)), exclusive=False)
    return _plain(dict(status='complete', workflow_complete=True,
        cpu_seconds=raw['cpu_seconds'], rho_index=rho_index, group=group,
        data_path=str(data), sidecar_path=str(sidecar), data_sha256=record['data_sha256']))


def _production_records(plan, root):
    p, root = _physical(plan), Path(root)
    rows = []
    for ri in range(len(p['rhos'])):
        for group in range(p['groups']):
            prefix = root / 'data' / 'shards' / f'r{ri:02d}_g{group:02d}'
            sidecar, status = prefix.with_suffix('.sidecar.json'), prefix.with_suffix('.status.json')
            receipt, state = json.loads(sidecar.read_text()), json.loads(status.read_text())
            if (state.get('status') != 'complete' or state.get('returncode') != 0
                    or state.get('sidecar_sha256') != _sha(sidecar)
                    or receipt.get('status') != 'complete' or receipt.get('purpose') != 'production'
                    or receipt.get('physical_plan_sha256') != _fingerprint(p)
                    or receipt.get('rho_index') != ri or receipt.get('group') != group
                    or receipt.get('rho') != p['rhos'][ri]
                    or receipt.get('size') != p['size']
                    or receipt.get('samples') != p['samples']
                    or receipt.get('burn_in') != p['burn_in']
                    or receipt.get('spacing') != p['spacing']
                    or receipt.get('saved_temperature_indices') != p['saved_temperature_indices']
                    or receipt.get('raw_diagnostics_before_symmetry_augmentation') is not True
                    or receipt.get('initial') != p['initials'][group % 4]
                    or receipt.get('seed') != seed_for_group(p['master_seed'], p['size'], p['rhos'][ri], 'production', group)
                    or receipt.get('temperatures') != p['temperatures']
                    or receipt.get('data_sha256') != state.get('data_sha256')):
                raise ValueError('Incomplete or inconsistent production sidecar/status identity')
            rows.append(receipt)
    return p, rows


def quality_gate(plan, root):
    """JSON-only production quality aggregation; no configuration/model reads."""
    root = Path(root)
    target = root / 'data' / 'quality_gate.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError('Quality gate is exclusive')
    p, rows = _production_records(plan, root)
    failures, warnings = [], []
    for row in rows:
        failed, warned = _individual_quality(row, plan['pilot'])
        identity = dict(rho_index=row['rho_index'], group=row['group'])
        failures.extend(dict(**identity, **item) for item in failed)
        warnings.extend(dict(**identity, **item) for item in warned)
    comparisons, disagreements = _hot_cold(rows, plan['pilot'], production=True)
    failures.extend(disagreements)
    result = dict(status='scientific_stop' if failures else 'complete', workflow_complete=True,
        scientific_gate_passed=not failures, requires_user_action=bool(failures),
        physical_groups=len(rows), configurations=len(rows) * len(p['saved_temperature_indices']) * p['samples'],
        cpu_seconds=sum(row['cpu_seconds'] for row in rows),
        failures=failures, warnings=warnings, hot_cold=comparisons,
        equilibrium_certified=False, raw_diagnostics_before_symmetry_augmentation=True,
        independent_unit='whole_temperature_ladder_group', automatic_resampling=False,
        physical_plan_sha256=_fingerprint(p), report_path=str(target))
    _json(target, result)
    return result


def assemble(plan, root):
    """Assemble only the new verified physical shards, without physical labels."""
    import torch
    from recurrent_rg.tensor_io import load_cpu_tensors
    p, records = _production_records(plan, root)
    root = Path(root)
    config_path = root / 'configs' / 'dg5_seed91.json'
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError('Registered dg5_seed91 config is required before cache assembly')
    target, meta = root / 'data' / 'training.pt', root / 'data' / 'metadata.json'
    if any(path.exists() or path.is_symlink() for path in (target, meta)):
        raise FileExistsError('Training cache and metadata cannot be overwritten')
    arrays, identities, rows = ({key: [] for key in ('train', 'test')} for _ in range(3))
    for record in records:
        ri, group, rho = record['rho_index'], record['group'], record['rho']
        path = root / record['data_path']
        if path.resolve() != (root / 'data' / 'shards' / f'r{ri:02d}_g{group:02d}.pt').resolve() or path.is_symlink():
            raise ValueError('Physical shard path differs from declared identity')
        if _sha(path) != record['data_sha256']:
            raise ValueError('Physical shard fingerprint mismatch')
        payload = load_cpu_tensors(path, mmap=True)
        x = payload['configurations']
        expected = (len(p['saved_temperature_indices']), p['samples'], p['size'], p['size'])
        if x.dtype != torch.int8 or tuple(x.shape) != expected or not bool(((x == -1) | (x == 1)).all()):
            raise ValueError('New physical shard has invalid binary geometry')
        partition = ('heldout_rho' if rho in p['heldout_rhos'] else
                     'train' if group < p['train_groups'] else 'same_rho_new_group')
        split = 'train' if partition == 'train' else 'test'
        for slot, temperature_index in enumerate(p['saved_temperature_indices']):
            arrays[split].append(x[slot].unsqueeze(-1))
            identities[split].append([ri, temperature_index, group])
            rows[split].append(dict(rho_index=ri, rho=rho, temperature_index=temperature_index,
                temperature=p['temperatures'][temperature_index],
                K=1. / p['temperatures'][temperature_index], group=group,
                condition_id=f'r{ri:02d}_t{temperature_index:03d}', partition=partition,
                source_path=record['data_path'], source_sha256=record['data_sha256'],
                raw_initial=record['initial']))
        del payload
    data = dict(schema_version=1)
    for split in ('train', 'test'):
        data[split] = dict(configurations=torch.stack(arrays[split]), couplings=None,
                           group_ids=torch.tensor(identities[split], dtype=torch.int64))
    with target.open('xb') as stream:
        torch.save(data, stream)
    metadata = dict(schema_version=1, data_sha256=_sha(target), physical_model='ising2d_j1j2',
        physical_plan_sha256=_fingerprint(p), labels_used_in_training=False,
        temperature_grid=_grid_metadata(p),
        physical_labels_in_training_cache=False, train=rows['train'], test=rows['test'],
        train_shape=list(data['train']['configurations'].shape),
        test_shape=list(data['test']['configurations'].shape),
        group_columns=['rho_index', 'temperature_index', 'independent_repeat'],
        independent_unit='whole_temperature_ladder_group', temperatures_within_group_are_correlated=True,
        symmetry_operations_are_not_independent_samples=True, equilibrium_certified=False)
    _json(meta, metadata)
    from experiments.prepare_ising_j1j2 import validate_label_free_cache
    boundary = validate_label_free_cache(config_path, meta)
    return dict(status='complete', workflow_complete=True, data_path=str(target),
        metadata_path=str(meta), data_sha256=metadata['data_sha256'],
        train_ensembles=len(rows['train']), test_ensembles=len(rows['test']),
        labels_used_in_training=False, model_initialized=False, optimizer_updates=0,
        label_boundary_validation=boundary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('benchmark', 'pilot', 'sample', 'assemble', 'quality'), required=True)
    parser.add_argument('--rho-index', type=int)
    parser.add_argument('--group', type=int)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.mode == 'sample':
        result = sample_group(plan, args.root, args.rho_index, args.group)
    else:
        result = dict(benchmark=benchmark, pilot=pilot, assemble=assemble, quality=quality_gate)[args.mode](plan, args.root)
    print(json.dumps(_plain(result), ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
