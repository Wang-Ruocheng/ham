"""Explicitly authorized, non-equilibrium J1--J2 diagnostic data only.

This separate adapter does not weaken the frozen equilibrium campaign.  It
retains every quality failure while allowing only this diagnostic to proceed.
Energy tracking, binary geometry, fingerprints, and split integrity stay hard
requirements.  Physical parameters remain outside the label-free training PT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recurrent_rg.data.ising_j1j2_pt import sample_ladder, seed_for_group
from scripts.ising_j1j2_data import (
    _fingerprint, _grid_metadata, _hot_cold, _individual_quality, _json, _now,
    _plain, _receipt, _sha, _validate_retained_grid,
)


PURPOSE = 'unthermalized_diagnostic'
SYMMETRY_PURPOSE = 'unthermalized_diagnostic_symmetry'
GROUP_IDS = [0, 1, 8, 9]
TRAIN_GROUP_IDS = [0, 1]
TEST_GROUP_IDS = [8, 9]
RHOS = [0., .35, .7, 1.]


def validate_plan(plan):
    """Pin the one authorized small diagnostic, before sampling or output I/O."""
    declaration = plan.get('diagnostic', {})
    if (declaration.get('diagnostic_only') is not True
            or declaration.get('quality_override_authorized') is not True
            or declaration.get('equilibrium_certified') is not False
            or not isinstance(declaration.get('user_authorization'), str)
            or not declaration['user_authorization'].strip()):
        raise ValueError('Explicit user-authorized non-equilibrium diagnostic declaration required')
    p = plan.get('physical', {})
    pinned = dict(size=108, rhos=RHOS, train_rhos=RHOS, heldout_rhos=[],
        groups=4, train_groups=2, group_ids=GROUP_IDS,
        train_group_ids=TRAIN_GROUP_IDS, test_group_ids=TEST_GROUP_IDS,
        samples=128, burn_in=4096, spacing=32, master_seed=20261026)
    if any(p.get(key) != value or isinstance(p.get(key), bool)
           for key, value in pinned.items()):
        raise ValueError('Diagnostic physical parameters differ from the authorized fixed plan')
    for key in ('size', 'groups', 'train_groups', 'samples', 'burn_in', 'spacing', 'master_seed'):
        if type(p[key]) is not int:
            raise ValueError('Diagnostic sizes, counts, and seed must be exact integers')
    if any(type(group) is not int for key in ('group_ids', 'train_group_ids', 'test_group_ids')
           for group in p[key]):
        raise ValueError('Diagnostic whole-ladder identities must be exact integers')
    temperatures = np.asarray(p.get('temperatures', []), dtype=np.float64)
    indices = np.asarray(p.get('saved_temperature_indices', []))
    from scripts.run_ising_j1j2 import coupling_grid
    expected_grid = coupling_grid()
    if any(p.get(key) != value for key, value in expected_grid.items()):
        raise ValueError('Diagnostic retains the original 83-rung PT ladder and twelve uniform K points')
    _validate_retained_grid(p, temperatures, indices)
    criteria = plan.get('pilot', {})
    expected_criteria = dict(min_exchange=.01, min_ess=16,
        hot_cold_abs_threshold=.05, hot_cold_z_threshold=4.)
    if any(criteria.get(key) != value or isinstance(criteria.get(key), bool)
           for key, value in expected_criteria.items()):
        raise ValueError('Quality thresholds must be retained unchanged as diagnostic warnings')
    return p


def initial_for_group(rho, group):
    if group in (0, 8):
        return 'hot'
    if group in (1, 9):
        return 'ferro' if rho < .5 else 'stripe_x'
    raise ValueError('Unknown diagnostic whole-ladder group identity')


def _declaration(plan):
    return dict(diagnostic_only=True, quality_override_authorized=True,
        equilibrium_certified=False, user_authorization=plan['diagnostic']['user_authorization'],
        quality_failure_blocks_training=False,
        interpretation='Non-equilibrium finite-chain diagnostic; not an equilibrium phase or RG certification')


def sample_group(plan, root, rho_index, group):
    """Save one independent ladder; physics-quality failures are not discarded."""
    p, root = validate_plan(plan), Path(root)
    if (type(rho_index) is not int or not 0 <= rho_index < len(p['rhos'])
            or type(group) is not int or group not in p['group_ids']):
        raise ValueError('Invalid diagnostic rho/whole-ladder group identity')
    directory = root / 'data' / 'shards'
    prefix = directory / f'r{rho_index:02d}_g{group:02d}'
    data, sidecar, status = (prefix.with_suffix(suffix)
        for suffix in ('.pt', '.sidecar.json', '.status.json'))
    if any(path.exists() or path.is_symlink() for path in (data, sidecar, status)):
        raise FileExistsError('Diagnostic shard is exclusive; no overwrite or automatic retry')
    _json(status, dict(status='running', rho_index=rho_index, group=group,
                       started_at=_now(), **_declaration(plan)))
    rho, initial = p['rhos'][rho_index], initial_for_group(p['rhos'][rho_index], group)
    seed = seed_for_group(p['master_seed'], p['size'], rho, PURPOSE, group)
    raw = sample_ladder(p['size'], rho, p['temperatures'], seed,
        p['burn_in'], p['samples'], p['spacing'], initial=initial)
    record = _receipt(raw, purpose=PURPOSE, plan_sha256=_fingerprint(p),
                      rho_index=rho_index, group=group)
    expected = (len(p['temperatures']), p['samples'], p['size'], p['size'])
    if (raw['configs'].dtype != np.int8 or raw['configs'].shape != expected
            or not np.all((raw['configs'] == -1) | (raw['configs'] == 1))):
        raise ValueError('Non-binary or incorrect diagnostic spin geometry is a hard failure')
    if (raw['size'] != p['size'] or raw['rho'] != rho or raw['seed'] != seed
            or raw['initial'] != initial or raw['samples'] != p['samples']
            or raw['burn_in'] != p['burn_in'] or raw['spacing'] != p['spacing']
            or not np.array_equal(raw['temperatures'], p['temperatures'])):
        raise ValueError('Sampler return identity differs from frozen diagnostic plan')
    failures, warnings = _individual_quality(record, plan['pilot'])
    selected = raw['configs'][p['saved_temperature_indices']].copy()
    augmentation_seed = seed_for_group(p['master_seed'], p['size'], rho,
                                       SYMMETRY_PURPOSE, group)
    generator = np.random.default_rng(augmentation_seed)
    signs = 2 * generator.integers(0, 2, selected.shape[:2], dtype=np.int8) - 1
    rotations = generator.integers(0, 4, selected.shape[:2], dtype=np.int8)
    for slot in range(len(selected)):
        for index in range(p['samples']):
            selected[slot, index] = signs[slot, index] * np.rot90(
                selected[slot, index], int(rotations[slot, index]))
    import torch
    payload = dict(schema_version=1, configurations=torch.from_numpy(selected),
        raw={key: torch.from_numpy(raw[key]) for key in
             ('energies', 'energy_nn', 'energy_diagonal', 'magnetization', 'stripe_x', 'stripe_y')},
        raw_trace={key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                   for key, value in raw['trace'].items()},
        symmetry_signs=torch.from_numpy(signs), symmetry_rotations=torch.from_numpy(rotations))
    with data.open('xb') as stream:
        torch.save(payload, stream)
    record.update(**_declaration(plan), data_path=str(data.relative_to(root)),
        data_sha256=_sha(data), saved_temperature_indices=p['saved_temperature_indices'],
        saved_temperatures=[p['temperatures'][i] for i in p['saved_temperature_indices']],
        saved_K_values=[1. / p['temperatures'][i] for i in p['saved_temperature_indices']],
        temperature_grid=_grid_metadata(p), configuration_shape=list(selected.shape),
        dtype='torch.int8', quality_failures=failures, quality_warnings=warnings,
        symmetry_augmentation=dict(seed=augmentation_seed,
            operations=['global_Z2', 'quarter_turn_rotation'], independent_samples_added=0,
            fixes_thermalization=False, raw_diagnostics_precede_augmentation=True))
    _json(sidecar, record)
    _json(status, dict(status='complete', workflow_complete=True, returncode=0,
        data_sha256=record['data_sha256'], sidecar_sha256=_sha(sidecar),
        quality_failure_count=len(failures), **_declaration(plan)), exclusive=False)
    return _plain(dict(status='complete', workflow_complete=True,
        cpu_seconds=raw['cpu_seconds'], rho_index=rho_index, group=group,
        data_path=str(data), sidecar_path=str(sidecar), data_sha256=record['data_sha256'],
        quality_failure_count=len(failures), **_declaration(plan)))


def _records(plan, root):
    p, root = validate_plan(plan), Path(root)
    records = []
    for ri, rho in enumerate(p['rhos']):
        for group in p['group_ids']:
            prefix = root / 'data' / 'shards' / f'r{ri:02d}_g{group:02d}'
            sidecar, status = prefix.with_suffix('.sidecar.json'), prefix.with_suffix('.status.json')
            if sidecar.is_symlink() or status.is_symlink():
                raise ValueError('Diagnostic sidecar/status symlinks are not accepted')
            row, state = json.loads(sidecar.read_text()), json.loads(status.read_text())
            expected = dict(status='complete', purpose=PURPOSE,
                physical_plan_sha256=_fingerprint(p), rho_index=ri, group=group,
                rho=rho, size=p['size'], samples=p['samples'], burn_in=p['burn_in'],
                spacing=p['spacing'], temperatures=p['temperatures'],
                saved_temperature_indices=p['saved_temperature_indices'],
                initial=initial_for_group(rho, group),
                seed=seed_for_group(p['master_seed'], p['size'], rho, PURPOSE, group),
                raw_diagnostics_before_symmetry_augmentation=True,
                diagnostic_only=True, quality_override_authorized=True,
                equilibrium_certified=False,
                user_authorization=plan['diagnostic']['user_authorization'])
            if (any(row.get(key) != value for key, value in expected.items())
                    or state.get('status') != 'complete' or state.get('returncode') != 0
                    or state.get('sidecar_sha256') != _sha(sidecar)
                    or state.get('data_sha256') != row.get('data_sha256')
                    or row.get('max_component_tracking_drift') != 0
                    or not np.isfinite(row.get('max_energy_tracking_drift', np.inf))
                    or row['max_energy_tracking_drift'] > 1e-8):
                raise ValueError('Incomplete, inconsistent, or invalid diagnostic shard receipt')
            records.append(row)
    return p, records


def _quality_result(plan, p, records):
    failures, warnings = [], []
    for row in records:
        failed, warned = _individual_quality(row, plan['pilot'])
        identity = dict(rho_index=row['rho_index'], group=row['group'])
        failures.extend(dict(**identity, **item) for item in failed)
        warnings.extend(dict(**identity, **item) for item in warned)
    comparisons, disagreements = _hot_cold(records, plan['pilot'], production=True)
    failures.extend(disagreements)
    return _plain(dict(status='complete', workflow_complete=True,
        quality_passed=not failures, scientific_gate_passed=not failures,
        quality_failure_count=len(failures), failures=failures, warnings=warnings,
        failed_criteria=sorted({row['kind'] for row in failures}), hot_cold=comparisons,
        physical_groups=len(records),
        configurations=len(records) * len(p['saved_temperature_indices']) * p['samples'],
        physical_plan_sha256=_fingerprint(p),
        sidecar_data_sha256={row['data_path']: row['data_sha256'] for row in records},
        raw_diagnostics_before_symmetry_augmentation=True,
        automatic_resampling=False, independent_unit='whole_temperature_ladder_group',
        **_declaration(plan)))


def quality_warning(plan, root):
    """JSON-only quality aggregation. Failed criteria remain visible, nonblocking."""
    root = Path(root)
    p, records = _records(plan, root)
    result = _quality_result(plan, p, records)
    _json(root / 'data' / 'quality_warning.json', result)
    return _plain(result)


def assemble(plan, root):
    """Assemble verified diagnostic shards; physical labels stay in JSON only."""
    import torch
    from recurrent_rg.tensor_io import load_cpu_tensors, tensor_chunks
    p, records = _records(plan, root)
    root = Path(root)
    config_path = root / 'configs' / 'dg5_seed91.json'
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError('Registered dg5_seed91 config required before diagnostic assembly')
    target, meta = root / 'data' / 'training.pt', root / 'data' / 'metadata.json'
    if any(path.exists() or path.is_symlink() for path in (target, meta)):
        raise FileExistsError('Diagnostic cache and metadata cannot be overwritten')
    quality_path = root / 'data' / 'quality_warning.json'
    if quality_path.is_symlink():
        raise ValueError('Quality warning symlink is not accepted')
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        if quality != _quality_result(plan, p, records):
            raise ValueError('Diagnostic quality warning is not bound to current shards')
    else:
        quality = quality_warning(plan, root)
    arrays, identities, rows = ({key: [] for key in ('train', 'test')} for _ in range(3))
    for record in records:
        ri, group, rho = record['rho_index'], record['group'], record['rho']
        expected_path = root / 'data' / 'shards' / f'r{ri:02d}_g{group:02d}.pt'
        path = root / record['data_path']
        if path != expected_path or path.is_symlink():
            raise ValueError('Diagnostic physical shard path differs from declared identity')
        if _sha(path) != record['data_sha256']:
            raise ValueError('Diagnostic physical shard fingerprint mismatch')
        payload = load_cpu_tensors(path, mmap=True)
        x = payload['configurations']
        expected_shape = (len(p['saved_temperature_indices']), p['samples'], p['size'], p['size'])
        if (x.dtype != torch.int8 or tuple(x.shape) != expected_shape
                or not all(bool(((chunk == -1) | (chunk == 1)).all()) for chunk in tensor_chunks(x))):
            raise ValueError('Diagnostic shard has invalid binary geometry')
        split = 'train' if group in p['train_group_ids'] else 'test'
        partition = 'train' if split == 'train' else 'same_rho_new_group'
        for slot, temperature_index in enumerate(p['saved_temperature_indices']):
            arrays[split].append(x[slot].unsqueeze(-1))
            identities[split].append([ri, temperature_index, group])
            rows[split].append(dict(rho_index=ri, rho=rho,
                temperature_index=temperature_index, temperature=p['temperatures'][temperature_index],
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
    metadata = dict(schema_version=1, data_sha256=_sha(target),
        physical_model='ising2d_j1j2', physical_plan_sha256=_fingerprint(p),
        labels_used_in_training=False, physical_labels_in_training_cache=False,
        temperature_grid=_grid_metadata(p), train=rows['train'], test=rows['test'],
        train_shape=list(data['train']['configurations'].shape),
        test_shape=list(data['test']['configurations'].shape),
        group_columns=['rho_index', 'temperature_index', 'independent_repeat'],
        train_group_ids=p['train_group_ids'], test_group_ids=p['test_group_ids'],
        independent_unit='whole_temperature_ladder_group', temperatures_within_group_are_correlated=True,
        symmetry_operations_are_not_independent_samples=True,
        quality_warning_path='data/quality_warning.json', quality_warning_sha256=_sha(quality_path),
        quality_passed=quality['quality_passed'], quality_failure_count=quality['quality_failure_count'],
        failed_criteria=quality['failed_criteria'], **_declaration(plan))
    _json(meta, metadata)
    from experiments.prepare_ising_j1j2 import validate_label_free_cache
    boundary = validate_label_free_cache(config_path, meta)
    from recurrent_rg.data.ensembles import load_ensembles
    from scripts.report_ising_j1j2 import validate_metadata
    train, test = load_ensembles(target)
    report_boundary = validate_metadata(metadata, dict(train=train, test=test),
        actual_sha256=metadata['data_sha256'], manifest_sha256=metadata['data_sha256'])
    return dict(status='complete', workflow_complete=True,
        data_path=str(target), metadata_path=str(meta), data_sha256=metadata['data_sha256'],
        train_ensembles=len(rows['train']), test_ensembles=len(rows['test']),
        labels_used_in_training=False, model_initialized=False, optimizer_updates=0,
        quality_failure_count=quality['quality_failure_count'],
        label_boundary_validation=boundary, report_boundary_validation=report_boundary,
        **_declaration(plan))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('sample', 'assemble', 'quality'), required=True)
    parser.add_argument('--rho-index', type=int)
    parser.add_argument('--group', type=int)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.mode == 'sample':
        result = sample_group(plan, args.root, args.rho_index, args.group)
    else:
        result = {'assemble': assemble, 'quality': quality_warning}[args.mode](plan, args.root)
    print(json.dumps(_plain(result), ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
