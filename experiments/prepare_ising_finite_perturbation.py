"""冻结既有 Q/C 的有限物理扰动统计采集；训练缓存不含物理耦合标签。

The physical-generation sidecars are read only for validation and a separate
evaluation mapping.  The R trainer receives detached encodings, paired score
responses and opaque group IDs, never K, delta, epsilon or generation seeds.
No physical sample, MI estimate or optimizer is created by this collector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.ising_response_loss_ablation import state_sha256, write_json
from experiments.prepare_ising_physical_response import (
    EVEN_OPERATORS, FD_HEAD_H, FD_TILT_T, head_value_jacobian,
    local_feature_means_device, q_feature_table, validate_derivatives,
)
from recurrent_rg.checkpoints import qcr_state_dict
from scripts.diagnose_ising_dimg_response import full_r_finite_difference
from scripts.diagnose_ising_mcrg import (
    OPERATOR_DEFINITIONS, OPERATOR_NAMES, c_draw, c_probability_table,
    covariance, full_r_jacobian, model_sha256, operator_values, sha256_file,
)


CACHE_FIELDS = frozenset((
    'schema_version', 'direct', 'vin', 'vout', 'score_std', 'train_mask',
    'baseline_mask', 'condition_index', 'replica_index', 'condition_opaque_id',
    'c_draw_seeds', 'size', 'logical_size', 'depth', 'smoke',
))
FLOAT_FIELDS = ('direct', 'vin', 'vout', 'score_std')
CUDA_MEMORY_CAP_BYTES = 8 * 1024 ** 3


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Require regular, non-symlink source artifact: {path}')
    return path


def _new_output(output, protected):
    original = Path(output)
    if original.is_symlink():
        raise FileExistsError(f'Refusing symbolic-link output: {original}')
    output = original.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing overwrite, retry or implicit resume: {output}')
    for item in protected:
        source = Path(item).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Output must be separate from historical model and new physical data')
    output.mkdir(parents=True, exist_ok=False)
    return output


def c_seed(seed, logical_size, condition_index, replica, draw, transition):
    """Independent per-size/per-condition C uniforms, without coupling access."""
    values = (seed, logical_size, condition_index, replica, draw, transition)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('C stream coordinates must be nonnegative integers')
    text = 'finite-perturbation-C-v1|' + '|'.join(map(str, values))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') & ((1 << 63)-1)


def _cuda_memory(device):
    if device.type != 'cuda':
        return dict(peak_allocated_bytes=0, peak_reserved_bytes=0,
                    memory_cap_bytes=CUDA_MEMORY_CAP_BYTES, uses_cuda=False)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    if max(allocated, reserved) > CUDA_MEMORY_CAP_BYTES:
        raise MemoryError('Collector exceeded the predeclared 8 GiB CUDA allocation/reservation cap')
    return dict(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                memory_cap_bytes=CUDA_MEMORY_CAP_BYTES, uses_cuda=True)


def validate_device(device):
    """Never remap CUDA; check that the requested ordinal is the gated GPU UUID."""
    device = torch.device(device)
    if str(device) not in ('cpu', 'cuda:2', 'cuda:3'):
        raise ValueError('Only CPU or explicitly authorized physical cuda:2/cuda:3 may be used')
    if device.type == 'cpu':
        return device, dict(device='cpu', GPU_mapping_checked=False, uses_cuda=False)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        entries = [part.strip() for part in visible.split(',')]
        if entries != list(map(str, range(len(entries)))):
            raise ValueError('CUDA_VISIBLE_DEVICES remapping is forbidden; environment was not modified')
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError('Require the explicit authorized CUDA device to exist')
    physical = subprocess.run(['nvidia-smi', f'--id={device.index}', '--query-gpu=uuid',
        '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout.strip()
    reported = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
    if reported is None:
        raise ValueError('Torch must expose a GPU UUID to verify the physical CUDA identity')
    def canonical(value):
        return str(value).lower().removeprefix('gpu-').replace('-', '').strip()
    if (len(physical.splitlines()) != 1 or not physical.startswith('GPU-')
            or canonical(reported) != canonical(physical)):
        raise ValueError('Torch CUDA ordinal UUID differs from the nvidia-smi physical GPU UUID')
    return device, dict(device=str(device), GPU_mapping_checked=True, uses_cuda=True,
        physical_gpu_index=device.index, physical_gpu_uuid=physical, torch_gpu_uuid=str(reported),
        CUDA_VISIBLE_DEVICES_preserved=visible)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def collect_ensemble(model, spins, table, probabilities, *, depth, seeds, spotcheck=False):
    """Pair input/output responses using the CURRENT input extensive scores."""
    if (spins.ndim != 3 or len(spins) < 2 or depth not in (1, 2)
            or spins.shape[-1] != spins.shape[-2] or spins.shape[-1] % 2 ** depth
            or spins.dtype != torch.float64 or seeds.ndim != 2
            or tuple(seeds.shape[1:]) != (depth,)):
        raise ValueError('Require float64 paired [sample,L,L], depth1/2 and [Cdraw,depth] seeds')
    feature0 = local_feature_means_device(spins, table)
    op0 = operator_values(spins)[:, :EVEN_OPERATORS]
    head = model.theory_encoder.head
    rows = {key: [] for key in FLOAT_FIELDS}
    public = {key: [] for key in ('physical_even_density', 'source_r_of_g',
        'source_full_DR', 'source_r_fd_jacobian', 'source_all_start_horizon_pred')}
    checks = []
    for draw in range(len(seeds)):
        levels = [spins]
        for step in range(depth):
            generator = torch.Generator(device=spins.device).manual_seed(int(seeds[draw, step]))
            side = levels[-1].shape[-1] // 2
            uniform = torch.rand((len(spins), side, side), device=spins.device,
                                  dtype=torch.float64, generator=generator)
            levels.append(c_draw(levels[-1], probabilities, uniform))
        features = [feature0] + [local_feature_means_device(level, table) for level in levels[1:]]
        operators = [op0] + [operator_values(level)[:, :EVEN_OPERATORS] for level in levels[1:]]
        gs, head_jacobians = zip(*(head_value_jacobian(head, feature.mean(0)) for feature in features))
        direct = torch.stack(gs)
        incoming, outgoing, deviations = [], [], []
        for step in range(depth):
            score = operators[step]
            vin = head_jacobians[step] @ covariance(features[step], score, ddof=0)
            vout = head_jacobians[step+1] @ covariance(features[step+1], score, ddof=0)
            incoming.append(vin.detach()); outgoing.append(vout.detach())
            deviations.append((score-score.mean(0)).square().mean(0).sqrt())
            if spotcheck and draw == 0:
                for side, feature, jacobian, response in (
                        ('in', features[step], head_jacobians[step], vin),
                        ('out', features[step+1], head_jacobians[step+1], vout)):
                    checks.append(dict(transition=step, side=side,
                        **validate_derivatives(head, feature, score, jacobian, response)))
        rows['direct'].append(direct)
        rows['vin'].append(torch.stack(incoming))
        rows['vout'].append(torch.stack(outgoing))
        rows['score_std'].append(torch.stack(deviations))
        # Source R snapshots are evaluation-only public arrays, not new labels.
        with torch.no_grad():
            public['source_r_of_g'].append(model.advance_theory(direct))
            prediction = torch.zeros((depth+1, depth+1, 3), device=spins.device, dtype=torch.float64)
            for start in range(depth+1):
                state = direct[start]
                prediction[start, start] = state
                for end in range(start+1, depth+1):
                    state = model.advance_theory(state)
                    prediction[start, end] = state
            public['source_all_start_horizon_pred'].append(prediction)
        jacobians = torch.stack([full_r_jacobian(model, g) for g in direct])
        finite = torch.stack([full_r_finite_difference(model, g, h=FD_HEAD_H) for g in direct])
        error = (jacobians-finite).abs().max()
        if error > 1e-7 and error > 1e-4 * jacobians.abs().max():
            raise FloatingPointError('Complete source R finite-difference verification failed')
        public['source_full_DR'].append(jacobians)
        public['source_r_fd_jacobian'].append(finite)
        public['physical_even_density'].append(torch.stack([
            op.mean(0) / level.shape[-1] ** 2 for op, level in zip(operators, levels)]))
    result = {key: torch.stack(values).detach().cpu() for key, values in rows.items()}
    references = {key: torch.stack(values).detach().cpu() for key, values in public.items()}
    if any(not torch.isfinite(value).all() for value in (*result.values(), *references.values())):
        raise FloatingPointError('Nonfinite encoding, paired response or source R snapshot')
    return result, references, checks


def validate_condition(data_root, spec, *, smoke):
    """Read only new configurations; K/delta values remain in the sidecar mapping."""
    identifier = spec.get('condition_id', spec.get('id'))
    if (not isinstance(identifier, str) or len(identifier) != 4
            or identifier[0] != 'c' or not identifier[1:].isdigit()):
        raise ValueError('Require the opaque fixed c000 condition-directory naming')
    folder = Path(data_root)/'conditions'/identifier
    if folder.is_symlink():
        raise ValueError('Refusing symlink condition data')
    sidecar_path, status_path = _regular(folder/'sidecar.json'), _regular(folder/'status.json')
    data_path = _regular(folder/'data.pt')
    sidecar, status = json.loads(sidecar_path.read_text()), json.loads(status_path.read_text())
    for receipt in (sidecar, status):
        if receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True:
            raise ValueError('Finite physical sampling condition is not complete')
    if (not _sha(sidecar.get('data_sha256'))
            or status.get('data_sha256') != sidecar['data_sha256']
            or status.get('sidecar_sha256') != sha256_file(sidecar_path)):
        raise ValueError('Finite-data fingerprint small receipts disagree')
    for key in ('condition_index', 'condition_id', 'role', 'family', 'size', 'logical_size', 'depth'):
        expected = spec.get(key, spec.get('id') if key == 'condition_id' else None)
        if sidecar.get(key) != expected:
            raise ValueError(f'Condition sidecar/predeclared {key} mismatch')
    if sidecar.get('condition') != spec:
        raise ValueError('Generation-only condition metadata differs from the fixed specification')
    if sidecar.get('smoke') is not smoke:
        raise ValueError('Smoke and formal sampling receipts cannot be mixed')
    payload = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('Require newly sampled schema1 physical condition payload')
    spins = payload['configurations']
    size, depth = spec['size'], spec['depth']
    if (type(size) is not int or type(depth) is not int or depth not in (1, 2)
            or size != (4 if smoke else 27) * 2 ** depth):
        raise ValueError('Require predeclared independent b2 logical L54/L108 geometry')
    nrep, samples = (2, 8) if smoke else (12, 128)
    if (not isinstance(spins, torch.Tensor) or spins.dtype != torch.int8
            or tuple(spins.shape) != (nrep, samples, size, size)):
        raise ValueError('Finite data require int8 [independent replica,samples,L,L]')
    if sidecar.get('configuration_shape') not in (None, list(spins.shape)):
        raise ValueError('Configuration tensor/sidecar shape mismatch')
    fingerprint = sidecar.get('metadata_fingerprint')
    canonical_hash = hashlib.sha256(json.dumps(sidecar.get('provenance'), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (not _sha(fingerprint) or status.get('metadata_fingerprint') != fingerprint
            or payload.get('metadata_fingerprint') != fingerprint or canonical_hash != fingerprint):
        raise ValueError('Condition payload/sidecar/status metadata fingerprints disagree')
    return spins, sidecar, dict(condition_id=identifier,
        data_sha256_from_receipt=sidecar['data_sha256'],
        sidecar_sha256=status['sidecar_sha256'], metadata_fingerprint=fingerprint,
        status_sha256=sha256_file(status_path),
        physical_data_rehashed=False)


def validate_training_cache(cache):
    """Strict allowlist: numerical generation labels cannot enter trainer files."""
    if not isinstance(cache, dict) or set(cache) != CACHE_FIELDS:
        raise ValueError('Training cache fields must match the label-free allowlist exactly')
    depth, size = cache['depth'], cache['size']
    if cache['schema_version'] != 1 or depth not in (1, 2) or cache['logical_size'] not in (54, 108):
        raise ValueError('Invalid label-free cache geometry')
    direct = cache['direct']
    if not isinstance(direct, torch.Tensor) or direct.ndim != 4:
        raise ValueError('Require direct[ensemble,Cdraw,level,3]')
    ensembles, draws = direct.shape[:2]
    shapes = dict(direct=(ensembles, draws, depth+1, 3),
        vin=(ensembles, draws, depth, 3, 6), vout=(ensembles, draws, depth, 3, 6),
        score_std=(ensembles, draws, depth, 6))
    for key, shape in shapes.items():
        value = cache[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float64 or value.requires_grad
                or not torch.isfinite(value).all()):
            raise ValueError(f'Require finite detached float64 {key} with declared shape')
    for key in ('train_mask', 'baseline_mask'):
        if cache[key].dtype != torch.bool or tuple(cache[key].shape) != (ensembles,):
            raise ValueError('Require boolean ensemble masks')
    for key in ('condition_index', 'replica_index'):
        if cache[key].dtype != torch.int64 or tuple(cache[key].shape) != (ensembles,):
            raise ValueError('Require int64 opaque ensemble indices')
    identifiers = cache['condition_opaque_id']
    if (not isinstance(identifiers, list) or len(identifiers) != ensembles
            or any(not isinstance(value, str) or len(value) != 4 or value[0] != 'c'
                   or not value[1:].isdigit() for value in identifiers)):
        raise ValueError('Only fixed opaque condition IDs may accompany training arrays')
    if (cache['c_draw_seeds'].dtype != torch.int64
            or tuple(cache['c_draw_seeds'].shape) != (ensembles, draws, depth)
            or torch.any(cache['score_std'] < 0) or type(size) is not int):
        raise ValueError('Invalid score deviations or conditional C stream receipt')
    if cache['logical_size'] == 54 and cache['train_mask'].any():
        raise ValueError('Independent logical L54 is evaluation-only')
    return cache


def _save_torch(path, payload):
    pending = Path(str(path)+'.tmp')
    with pending.open('xb') as handle:
        torch.save(payload, handle)
    pending.replace(path)


def _validate_source(source_run, expected_checkpoint_sha256):
    if not _sha(expected_checkpoint_sha256):
        raise ValueError('Explicit lowercase source checkpoint SHA256 is mandatory')
    paths = [source_run/name for name in ('training_step_12000.pt', 'manifest.json', 'summary.json')]
    hashes = {str(_regular(path)): sha256_file(path) for path in paths}
    if hashes[str(paths[0])] != expected_checkpoint_sha256:
        raise ValueError('Numbered 12000 source checkpoint SHA256 mismatch')
    from scripts.report_ising_coupling_representation import load_representation_run, validate_formal_run
    loaded = load_representation_run(source_run, 12000)
    cfg, manifest, summary, representation = validate_formal_run(source_run, loaded, 12000)
    if (representation['arm'] != 'control' or cfg.model.theory_dim != 3 or cfg.model.hidden != 96
            or summary.get('actual_epochs') != 12000):
        raise ValueError('Require the frozen phase2 original-representation dimg3 hidden96 endpoint')
    optimizer = loaded.payload.get('optimizer', {})
    groups = optimizer.get('param_groups', [])
    parameter_ids = [index for group in groups for index in group.get('params', [])]
    if (len(groups) != 1 or not parameter_ids
            or len(parameter_ids) != len(set(parameter_ids))
            or any(float(optimizer.get('state', {}).get(index, {}).get('step', -1)) != 12000
                   for index in parameter_ids)):
        raise ValueError('Source per-parameter Adam updates must all equal actual12000')
    return loaded, hashes, dict(source_epoch=12000, source_training_actual_updates=12000,
        source_checkpoint_sha256=expected_checkpoint_sha256,
        source_training_data_sha256_from_receipt=manifest['data_sha256'],
        source_data_opened=False, source_data_rehashed=False,
        source_optimizer_each_parameter_updates=12000, source_representation=representation)


def collect(source_run, data_root, output, expected_checkpoint_sha256, *, device='cpu',
            seed=20261017, smoke=False):
    if Path(source_run).is_symlink() or Path(data_root).is_symlink():
        raise ValueError('Source model and physical-data roots must not be symlinks')
    source_run, data_root = Path(source_run).resolve(), Path(data_root).resolve()
    output = _new_output(output, (source_run, data_root))
    started = time.monotonic()
    completed = 0
    write_json(output/'status.json', dict(status='running', workflow_complete=False,
        no_training=True, no_optimizer=True, no_MI=True, no_new_physical_sampling=True))
    model, before_full, before_qc, before_r = None, None, None, None
    try:
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError('Require explicit nonnegative integer collection seed')
        device, device_receipt = validate_device(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(1)
        loaded, source_hashes, provenance = _validate_source(source_run, expected_checkpoint_sha256)
        model = loaded.model.to(device=device, dtype=torch.float64).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        before_full = model_sha256(model)
        before_qc = state_sha256(qcr_state_dict(model))
        before_r = state_sha256(qcr_state_dict(model), residual=True)
        table = q_feature_table(model)
        _, probabilities = c_probability_table(model)
        from scripts.ising_finite_perturbation_data import condition_specs
        specs = list(condition_specs(smoke=smoke))
        if len(specs) != 68:
            raise ValueError('Require the fixed complete 68-condition physical design')
        if ([row['condition_index'] for row in specs] != list(range(68))
                or sum(row['logical_size'] == 108 for row in specs) != 41
                or sum(row['logical_size'] == 54 for row in specs) != 27):
            raise ValueError('Condition grouping/order differs from the predeclared 41+27 design')
        repeats = 1 if smoke else 5
        grouped = {size: {key: [] for key in (*FLOAT_FIELDS, 'train_mask', 'baseline_mask',
            'condition_index', 'replica_index', 'condition_opaque_id', 'c_draw_seeds')}
            for size in (108, 54)}
        public = {size: {} for size in (108, 54)}
        mappings, data_receipts, spotchecks = [], [], []
        condition_timings = []
        for spec in specs:
            _synchronize(device)
            condition_started = time.monotonic()
            raw, sidecar, data_receipt = validate_condition(data_root, spec, smoke=smoke)
            size, depth, index = spec['size'], spec['depth'], spec['condition_index']
            logical_size, identifier = spec['logical_size'], spec['condition_id']
            mappings.append(dict(sidecar, unit_delta6=spec['unit_delta6'],
                                 unit_delta=spec['unit_delta6']))
            data_receipts.append(data_receipt)
            bucket, public_bucket = grouped[logical_size], public[logical_size]
            for replica in range(len(raw)):
                seeds = torch.tensor([[c_seed(seed, logical_size, index, replica, draw, step)
                    for step in range(depth)] for draw in range(repeats)], dtype=torch.int64)
                row, reference, checks = collect_ensemble(model,
                    raw[replica].to(device=device, dtype=torch.float64), table, probabilities,
                    depth=depth, seeds=seeds, spotcheck=replica == 0 and index in (0, 41))
                for key, value in row.items():
                    bucket[key].append(value)
                train = spec['role'] == 'train_domain' and replica < (1 if smoke else 8)
                baseline = spec['role'] == 'train_domain' and spec['family'] == 'zero'
                bucket['train_mask'].append(train)
                bucket['baseline_mask'].append(baseline)
                bucket['condition_index'].append(index)
                bucket['replica_index'].append(replica)
                bucket['condition_opaque_id'].append(identifier)
                bucket['c_draw_seeds'].append(seeds)
                for key, value in reference.items():
                    public_bucket.setdefault(key, []).append(value)
                if checks:
                    spotchecks.append(dict(condition_opaque_id=identifier, replica_index=replica,
                                           derivatives=checks))
            completed += 1
            _synchronize(device)
            condition_wall = time.monotonic()-condition_started
            condition_timings.append(dict(condition_opaque_id=identifier,
                logical_size=logical_size, physical_ensembles=len(raw), Cdraws=repeats,
                elapsed_wall_seconds=condition_wall))
            memory = _cuda_memory(device)
            status = dict(status='running', workflow_complete=False, pid=os.getpid(),
                completed_conditions=completed, total_conditions=68, condition_opaque_id=identifier,
                elapsed_seconds=time.monotonic()-started,
                condition_elapsed_wall_seconds=condition_wall,
                elapsed_gpu_card_seconds=time.monotonic()-started if device.type == 'cuda' else 0.,
                projected_remaining_wall_seconds_completed_mean=(68-completed)*sum(
                    row['elapsed_wall_seconds'] for row in condition_timings)/completed,
                projected_remaining_wall_seconds_worst_completed=(68-completed)*max(
                    row['elapsed_wall_seconds'] for row in condition_timings),
                timing_is_descriptive_not_adaptive=True,
                timing_drift_policy='仅记录并报告预算偏移；不据此改变数据、采样或固定训练终点。', **memory)
            write_json(output/'status.json', status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            del raw
        caches, npz = {}, {}
        for logical_size, bucket in grouped.items():
            subset = [row for row in specs if row['logical_size'] == logical_size]
            cache = {key: torch.stack(bucket[key]) for key in (*FLOAT_FIELDS, 'c_draw_seeds')}
            for key in ('train_mask', 'baseline_mask'):
                cache[key] = torch.tensor(bucket[key], dtype=torch.bool)
            for key in ('condition_index', 'replica_index'):
                cache[key] = torch.tensor(bucket[key], dtype=torch.int64)
            cache.update(schema_version=1, size=subset[0]['size'], logical_size=logical_size,
                depth=subset[0]['depth'], smoke=smoke, condition_opaque_id=bucket['condition_opaque_id'])
            validate_training_cache(cache)
            filename = 'training_L108.pt' if logical_size == 108 else 'evaluation_L54.pt'
            _save_torch(output/filename, cache)
            caches[filename] = sha256_file(output/filename)
            prefix = f'L{logical_size}_'
            for key, value in cache.items():
                npz[prefix+key] = value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, values in public[logical_size].items():
                npz[prefix+key] = torch.stack(values).numpy()
        npz_path = output/'collected_statistics.npz'
        with npz_path.open('xb') as handle:
            np.savez_compressed(handle, **npz)
        after_full = model_sha256(model)
        after_qc = state_sha256(qcr_state_dict(model))
        after_r = state_sha256(qcr_state_dict(model), residual=True)
        if ((before_full, before_qc, before_r) != (after_full, after_qc, after_r)
                or any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters())):
            raise RuntimeError('Frozen source Q/C/R parameters, state or gradients changed')
        if any(sha256_file(path) != fingerprint for path, fingerprint in source_hashes.items()):
            raise RuntimeError('Pinned source checkpoint or small receipt changed during collection')
        for item in data_receipts:
            folder = data_root/'conditions'/item['condition_id']
            if (sha256_file(_regular(folder/'sidecar.json')) != item['sidecar_sha256']
                    or sha256_file(_regular(folder/'status.json')) != item['status_sha256']):
                raise RuntimeError('Physical generation small receipt changed during collection')
        metadata = dict(schema_version=1, status='complete', workflow_complete=True,
            generation_mapping_evaluation_only=True, conditions=mappings, data_receipts=data_receipts,
            operator_names=list(OPERATOR_NAMES[:6]), operator_definitions=list(OPERATOR_DEFINITIONS[:6]),
            covariance_ddof=0, FD_head_h=FD_HEAD_H, FD_tilt_t=FD_TILT_T,
            source_full_DR_includes_identity=True, source_full_DR_finite_difference_h=FD_HEAD_H,
            train_replicas=list(range(1 if smoke else 8)), physical_chains_per_condition=2 if smoke else 12,
            Cdraws_are_conditionally_repeated_not_independent_physical_ensembles=True,
            generation_labels_must_not_be_loaded_by_trainer=True,
            no_thermal_mode_certification=True, thermal_nu=None)
        write_json(output/'metadata.json', metadata)
        receipt = dict(schema_version=1, status='complete', workflow_complete=True, **provenance,
            smoke=smoke, seed=seed, device=str(device), threads=1, c_repeats=repeats,
            source_run=str(source_run), source_sha256=source_hashes,
            source_state_sha256=before_full, model_sha256_before=before_full, model_sha256_after=after_full,
            source_frozen_QC_sha256_before=before_qc, source_frozen_QC_sha256_after=after_qc,
            frozen_QC_sha256_before=before_qc, frozen_QC_sha256_after=after_qc,
            initial_R_sha256=before_r, source_R_sha256=before_r, source_R_sha256_after=after_r,
            all_parameter_grads_none=True, all_parameters_frozen=True,
            no_K_labels_in_training_cache=True, labels_accessed_in_training_cache=False,
            no_training=True, no_optimizer=True, no_MI=True, no_new_physical_sampling=True,
            train_cache_allowed_fields=sorted(CACHE_FIELDS), cache_sha256=caches,
            public_statistics_sha256=sha256_file(npz_path), metadata_sha256=sha256_file(output/'metadata.json'),
            completed_conditions=completed, total_conditions=68, derivative_spotchecks=spotchecks,
            data_small_source_sha256={str(data_root/'conditions'/item['condition_id']/filename):item[key]
                for item in data_receipts
                for filename, key in (('sidecar.json', 'sidecar_sha256'), ('status.json', 'status_sha256'))},
            device_identity=device_receipt,
            condition_timings=condition_timings,
            elapsed_gpu_card_seconds=time.monotonic()-started if device.type == 'cuda' else 0.,
            elapsed_seconds=time.monotonic()-started, **_cuda_memory(device))
        write_json(output/'receipt.json', receipt)
        write_json(output/'status.json', receipt)
        return receipt
    except Exception as error:
        failed = dict(status='failed', workflow_complete=False, completed_conditions=completed,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            no_training=True, no_optimizer=True, no_MI=True, no_new_physical_sampling=True,
            retry_policy='保留失败目录及既有结果；不覆盖、不自动重跑。')
        if model is not None:
            failed.update(model_sha256_before=before_full, model_sha256_after=model_sha256(model),
                all_parameter_grads_none=all(parameter.grad is None for parameter in model.parameters()))
        write_json(output/'status.json', failed)
        raise


def benchmark(source_run, output, expected_checkpoint_sha256, *, device='cpu', seed=20261017):
    """Fixed synthetic execution timing only; no physical distributions are sampled."""
    if Path(source_run).is_symlink():
        raise ValueError('Source model root must not be a symlink')
    source_run = Path(source_run).resolve()
    output = _new_output(output, (source_run,))
    started = time.monotonic()
    model, before_full = None, None
    write_json(output/'status.json', dict(status='running', workflow_complete=False,
        benchmark_only=True, scientific_result=False, no_physical_sampling=True,
        no_training=True, no_MI=True, no_optimizer=True))
    try:
        if type(seed) is not int or seed != 20261017:
            raise ValueError('Execution benchmark seed is fixed20261017')
        torch.set_num_threads(1)
        device, device_receipt = validate_device(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        loaded, source_hashes, provenance = _validate_source(source_run, expected_checkpoint_sha256)
        model = loaded.model.to(device=device, dtype=torch.float64).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        before_full = model_sha256(model)
        before_qc = state_sha256(qcr_state_dict(model))
        before_r = state_sha256(qcr_state_dict(model), residual=True)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        iid = (2*torch.randint(0, 2, (128, 108, 108), generator=generator)-1).to(
            device=device, dtype=torch.float64)
        table = q_feature_table(model)
        _, probabilities = c_probability_table(model)
        seeds = torch.tensor([[c_seed(seed, 108, 0, 0, 0, step) for step in range(2)]],
                             dtype=torch.int64)
        times, cpu_times, cases = [], [], []
        for name in ('iid_binary', 'all_plus_binary_histogram_contention'):
            synthetic = iid if name == 'iid_binary' else torch.ones_like(iid)
            case_times, case_cpu_times = [], []
            for repeat in range(5):
                _synchronize(device)
                wall, cpu = time.monotonic(), time.process_time()
                row, references, checks = collect_ensemble(model, synthetic, table, probabilities,
                    depth=2, seeds=seeds, spotcheck=True)
                _synchronize(device)
                elapsed, elapsed_cpu = time.monotonic()-wall, time.process_time()-cpu
                _cuda_memory(device)
                if repeat:
                    times.append(elapsed)
                    cpu_times.append(elapsed_cpu)
                    case_times.append(elapsed)
                    case_cpu_times.append(elapsed_cpu)
                # Discard synthetic statistics; only timing is published.
                del row, references, checks
            cases.append(dict(case=name, warmup_iterations=1, measured_iterations=4,
                measured_single_draw_wall_seconds=case_times,
                measured_single_draw_cpu_seconds=case_cpu_times))
            del synthetic
        after_full = model_sha256(model)
        after_qc = state_sha256(qcr_state_dict(model))
        after_r = state_sha256(qcr_state_dict(model), residual=True)
        if ((before_full, before_qc, before_r) != (after_full, after_qc, after_r)
                or any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters())):
            raise RuntimeError('Frozen source Q/C/R changed during execution timing')
        if any(sha256_file(path) != fingerprint for path, fingerprint in source_hashes.items()):
            raise RuntimeError('Pinned source checkpoint or small receipt changed during execution timing')
        projected = max(times) * 816 * 5 * 2
        result = dict(status='complete', workflow_complete=True, benchmark_only=True,
            scientific_result=False, synthetic_inputs_only=True, physical_data_opened=False,
            no_physical_sampling=True, no_training=True, no_optimizer=True, no_MI=True,
            source_run=str(source_run), **provenance, source_sha256=source_hashes,
            seed=seed, threads=1, device_identity=device_receipt, device=str(device),
            model_sha256_before=before_full, model_sha256_after=after_full,
            source_frozen_QC_sha256_before=before_qc, source_frozen_QC_sha256_after=after_qc,
            source_R_sha256_before=before_r, source_R_sha256_after=after_r,
            all_parameter_grads_none=True, all_parameters_frozen=True,
            synthetic_shape=[128, 108, 108], depth=2, Cdraws_per_iteration=1,
            cases=cases, warmup_iterations=2, measured_iterations=8,
            warmup_iterations_per_case=1, measured_iterations_per_case=4,
            measured_single_draw_wall_seconds=times, measured_single_draw_cpu_seconds=cpu_times,
            worst_single_draw_wall_seconds=max(times), worst_single_draw_cpu_seconds=max(cpu_times),
            projected_physical_ensembles=816, projected_Cdraws=5, safety_factor=2,
            benchmark_module_sha256=sha256_file(Path(__file__)),
            projected_collection_gpu_seconds_with_margin=projected if device.type == 'cuda' else 0.,
            projected_collection_wall_seconds_with_margin=projected,
            projected_collection_cpu_seconds_with_margin=(projected if device.type == 'cpu'
                else max(cpu_times)*816*5*2),
            timing_estimate='Worst of eight synchronized L108 depth2 single-draw collections over IID and all-plus histogram-contention inputs ×816 physical ensembles ×5 Cdraws ×2; includes derivative/source-R checks; smaller L54 overestimated.',
            CPU_device_is_not_GPU_budget=device.type == 'cpu',
            saved_synthetic_data=False, saved_model=False,
            elapsed_gpu_card_seconds=time.monotonic()-started if device.type == 'cuda' else 0.,
            elapsed_seconds=time.monotonic()-started, **_cuda_memory(device))
        write_json(output/'benchmark.json', result)
        write_json(output/'status.json', result)
        return result
    except Exception as error:
        result = dict(status='failed', workflow_complete=False, benchmark_only=True,
            scientific_result=False, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-started, no_physical_sampling=True,
            no_training=True, no_MI=True, no_optimizer=True,
            retry_policy='保留失败目录；不覆盖、不自动重跑。')
        if model is not None:
            result.update(model_sha256_before=before_full, model_sha256_after=model_sha256(model),
                all_parameter_grads_none=all(parameter.grad is None for parameter in model.parameters()))
        write_json(output/'status.json', result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:2', 'cuda:3'), default='cpu')
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = vars(parser.parse_args(argv))
    is_benchmark = args.pop('benchmark')
    if is_benchmark:
        if args.pop('data_root') is not None or args.pop('smoke'):
            parser.error('--benchmark does not accept physical --data-root or --smoke')
        benchmark(**args)
    else:
        if args['data_root'] is None:
            parser.error('normal collection requires --data-root')
        collect(**args)


if __name__ == '__main__':
    main()
