"""冻结无监督 Q/C，按原系综保存无标签物理响应；不训练或生成物理样本。

Each transition uses extensive scores on its CURRENT input configurations for
both sides of the pair.  Population covariance is the exact derivative of the
empirical exponential reweighting, not a difference along the K grid.  All
source ensembles are visited in source order without opening coupling labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from experiments.ising_response_loss_ablation import state_sha256, write_json
from recurrent_rg.binary_pooling import binary_pattern_table
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from scripts.diagnose_ising_dimg_response import validate_source
from scripts.diagnose_ising_mcrg import (
    EXPECTED_DATA_SHA256, OPERATOR_DEFINITIONS, OPERATOR_NAMES,
    c_draw, c_probability_table, covariance, model_sha256, operator_values,
    sha256_file,
)


SPLIT_COUNTS = dict(train=516, test=252)
SIDES = (108, 54, 27)
EVEN_OPERATORS = 6
CACHE_FILENAME = 'response_cache.pt'
FD_HEAD_H = 1e-5
FD_TILT_T = 1e-4


def mmap_configurations(path):
    """Get configuration tensors only; no coupling value is indexed or scanned.

    torch.load necessarily reads the archive metadata, but mmap leaves the
    physical storage lazy and no coupling array is subsequently accessed.
    """
    payload = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(payload, (tuple, list)) and len(payload) == 4:
        result = dict(train=payload[0], test=payload[2])
    elif isinstance(payload, dict) and payload.get('schema_version') == 1:
        result = {split: payload[split]['configurations'] for split in SPLIT_COUNTS}
    else:
        raise ValueError('Require the existing supported physical cache; no sampling fallback')
    for split, value in result.items():
        if (not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (SPLIT_COUNTS[split], 128, 108, 108, 1)):
            raise ValueError(f'Require the original {split} configuration tensor shape')
    return result


def q_feature_table(model):
    """Frozen 512-pattern feature table, on the same device as Q."""
    encoder = model.theory_encoder
    if tuple(encoder.geometry.widths) != (3, 3):
        raise ValueError('Q motif width must remain 3, independent of C block size 2')
    reference = next(encoder.parameters())
    if reference.dtype != torch.float64:
        raise ValueError('Physical response collection requires float64 Q')
    powers, patterns = binary_pattern_table(9, reference)
    with torch.no_grad():
        values = encoder.local_mlp(patterns).detach()
    return dict(powers=powers, values=values,
                offsets=tuple(encoder.geometry.patch_offsets()))


def local_feature_means_device(spins, table):
    """Exact per-configuration Q feature means, GPU-safe pattern histogram.

    The offsets as well as motif powers live on spins.device.  No giant
    [128,108,108,hidden] intermediate and no feature autograd graph is created.
    """
    if (spins.ndim != 3 or spins.device != table['values'].device
            or spins.dtype != torch.float64 or table['values'].dtype != spins.dtype):
        raise ValueError('Require same-device float64 [configuration,L,L] inputs and feature table')
    with torch.no_grad():
        indices = torch.zeros_like(spins, dtype=torch.long)
        for power, (a, b) in zip(table['powers'], table['offsets']):
            indices += (torch.roll(spins, shifts=(-a, -b), dims=(-2, -1)) > 0).long() * power
        flat = indices.flatten(1)
        patterns = table['values'].shape[0]
        offsets = torch.arange(len(spins), device=spins.device)[:, None] * patterns
        counts = torch.bincount((flat + offsets).flatten(), minlength=len(spins) * patterns)
        frequencies = counts.reshape(len(spins), patterns).to(spins.dtype) / flat.shape[1]
        return (frequencies @ table['values']).detach()


def head_value_jacobian(head, pooled):
    """Complete nonlinear Q head Jacobian; parameter gradients remain absent."""
    with torch.enable_grad():
        point = pooled.detach().clone().requires_grad_(True)
        value = head(point)
        jacobian = torch.stack([torch.autograd.grad(value[j], point, retain_graph=True)[0]
                                for j in range(value.numel())])
    return value.detach(), jacobian.detach()


def finite_difference_head(head, pooled, h=FD_HEAD_H):
    columns = []
    with torch.no_grad():
        for coordinate in range(pooled.numel()):
            delta = torch.zeros_like(pooled)
            delta[coordinate] = h
            columns.append((head(pooled + delta) - head(pooled - delta)) / (2 * h))
    return torch.stack(columns, dim=-1)


def physical_response(head, features, score):
    """Exact empirical derivative: D head(mean f) Cov_population(f,S)."""
    if (features.ndim != 2 or score.ndim != 2 or len(features) != len(score)
            or features.device != score.device or features.dtype != torch.float64
            or score.dtype != torch.float64 or len(features) < 2):
        raise ValueError('Require paired same-device float64 configuration features and scores')
    g, jacobian = head_value_jacobian(head, features.mean(0))
    response = jacobian @ covariance(features, score, ddof=0)
    std = (score - score.mean(0)).square().mean(0).sqrt()
    return g, response.detach(), std.detach(), jacobian


def finite_difference_response(head, features, score, t=FD_TILT_T):
    """Exponential-tilt finite difference holding saved configurations/C fixed."""
    centered = score - score.mean(0)
    std = centered.square().mean(0).sqrt()
    dimension = head(features.mean(0)).numel()
    result = features.new_zeros((dimension, score.shape[1]))
    with torch.no_grad():
        for direction in range(score.shape[1]):
            if std[direction] <= 1e-12:
                continue
            epsilon = t / std[direction]
            plus = torch.softmax(epsilon * centered[:, direction], dim=0) @ features
            minus = torch.softmax(-epsilon * centered[:, direction], dim=0) @ features
            result[:, direction] = (head(plus) - head(minus)) / (2 * epsilon)
    return result


def _difference_record(exact, finite):
    error = finite - exact
    return dict(max_absolute_error=float(error.abs().max()),
                relative_frobenius_error=float(torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(exact).clamp_min(1e-15)))


def validate_derivatives(head, features, score, jacobian, response):
    head_fd = finite_difference_head(head, features.mean(0))
    response_fd = finite_difference_response(head, features, score)
    if any(not torch.isfinite(value).all() for value in (jacobian, response, head_fd, response_fd)):
        raise FloatingPointError('Nonfinite derivative spotcheck')
    head_check = _difference_record(jacobian, head_fd)
    response_check = _difference_record(response, response_fd)
    for name, row in (('head', head_check), ('physical_response', response_check)):
        if row['relative_frobenius_error'] > 1e-4 and row['max_absolute_error'] > 1e-7:
            raise FloatingPointError(f'{name} finite-difference verification failed: {row}')
    return dict(head=head_check, physical_response=response_check,
                head_h=FD_HEAD_H, tilt_t=FD_TILT_T)


def draw_seed(seed, split, index, draw):
    """Independent declared C stream per source ensemble and draw; no K access."""
    return seed + (0 if split == 'train' else 1_000_000_000) + index * 10_000 + draw


def collect_ensemble(model, spins, table, probabilities, seeds, *, spotcheck=False):
    """Keep all configurations paired; C draws are not physical replicas."""
    if tuple(spins.shape) != (128, 108, 108):
        raise ValueError('Require one unpooled 128-configuration L108 physical ensemble')
    feature0 = local_feature_means_device(spins, table)
    op0 = operator_values(spins)  # seven extensive scores; only six even enter cache
    head = model.theory_encoder.head
    direct, incoming, outgoing, deviations, checks = [], [], [], [], []
    for draw, seed in enumerate(seeds):
        generator = torch.Generator(device=spins.device).manual_seed(seed)
        levels = [spins]
        for _ in range(2):
            uniform = torch.rand((len(spins), levels[-1].shape[-2] // 2,
                                  levels[-1].shape[-1] // 2), device=spins.device,
                                 dtype=torch.float64, generator=generator)
            levels.append(c_draw(levels[-1], probabilities, uniform))
        features = [feature0] + [local_feature_means_device(level, table) for level in levels[1:]]
        operators = [op0] + [operator_values(level) for level in levels[1:]]
        g_values, jacobians = zip(*[head_value_jacobian(head, feature.mean(0)) for feature in features])
        vin, vout, std = [], [], []
        for step in range(2):
            # Use the CURRENT input score for BOTH derivatives.  In particular,
            # step 1 is not reweighted with microscopic step-0 operators.
            score = operators[step][:, :EVEN_OPERATORS]
            value_in = jacobians[step] @ covariance(features[step], score, ddof=0)
            value_out = jacobians[step + 1] @ covariance(features[step + 1], score, ddof=0)
            vin.append(value_in.detach()); vout.append(value_out.detach())
            std.append((score - score.mean(0)).square().mean(0).sqrt())
            if spotcheck and draw == 0:
                for side, feature, jacobian, response in (
                        ('in', features[step], jacobians[step], value_in),
                        ('out', features[step + 1], jacobians[step + 1], value_out)):
                    checks.append(dict(transition=step, side=side,
                        **validate_derivatives(head, feature, score, jacobian, response)))
        direct.append(torch.stack(g_values)); incoming.append(torch.stack(vin))
        outgoing.append(torch.stack(vout)); deviations.append(torch.stack(std))
    result = dict(direct=torch.stack(direct).detach().cpu(),
                  vin=torch.stack(incoming).detach().cpu(),
                  vout=torch.stack(outgoing).detach().cpu(),
                  score_std=torch.stack(deviations).detach().cpu())
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite paired physical response cache')
    return result, checks


def _validate_completed_workflow(source_run, dimension):
    root = source_run.parent.parent
    path = root / 'job.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Require completed dimg-ablation workflow receipt')
    job = json.loads(path.read_text())
    row = job.get('training', {}).get(f'dim{dimension}', {})
    if (job.get('status') != 'complete' or job.get('workflow_complete') is not True
            or row.get('status') != 'complete' or row.get('actual_epochs') != 12000
            or row.get('manifest_sha256') != sha256_file(source_run / 'manifest.json')):
        raise ValueError('Source workflow/arm is not a verified complete 12000 endpoint')
    return path


def collect(source_run, output, *, device='cuda:2', c_repeats=5, seed=20261016,
            smoke=False, expected_checkpoint_sha256=None):
    """Write an exclusive TRAIN/TEST response cache for the frozen source Q/C.

    This function creates no optimizer, does not call R, and never reads K/Kc.
    TEST output is evaluation-only; normalization and fitting belong to TRAIN.
    """
    source_input, output_input = Path(source_run), Path(output)
    if source_input.is_symlink():
        raise ValueError('Require an explicit regular source directory')
    source_run, output = source_input.resolve(), output_input.resolve()
    if output_input.is_symlink() or output.exists():
        raise FileExistsError(f'Refusing overwrite or implicit resume: {output}')
    if source_run == output or source_run in output.parents or output in source_run.parents:
        raise ValueError('New output must be separate from historical source run')
    if (type(c_repeats) is not int or not 1 <= c_repeats < 10_000
            or type(seed) is not int or not 0 <= seed < 2**62):
        raise ValueError('Require a positive fixed C count and nonnegative integer seed')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit CPU/CUDA devices are supported')
    if device.type == 'cuda':
        if device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError('Require an available explicitly numbered CUDA device')
    checkpoint = source_run / 'training_step_12000.pt'
    for path in (checkpoint, source_run / 'manifest.json', source_run / 'summary.json'):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Require a regular saved source artifact: {path}')
    checkpoint_hash = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError('Numbered source checkpoint fingerprint mismatch')
    loaded = load_run(source_run, 12000)
    dimension = int(loaded.config['model']['theory_dim'])
    if dimension not in (1, 3):
        raise ValueError('This predeclared stage-one experiment uses only dimg1 and dimg3')
    data = Path(loaded.config['data']['cache']).resolve()
    if not data.is_file() or data.is_symlink():
        raise ValueError('Require existing regular physical data cache')
    summary = json.loads((source_run / 'summary.json').read_text())
    manifest = validate_source(source_run, data, loaded, summary)
    workflow = _validate_completed_workflow(source_run, dimension)
    source_hashes = {str(path): sha256_file(path) for path in
                     (checkpoint, source_run / 'manifest.json', source_run / 'summary.json', workflow)}
    dataset = mmap_configurations(data)
    count = {split: (2 if smoke else total) for split, total in SPLIT_COUNTS.items()}
    repeats = 1 if smoke else c_repeats
    torch.set_num_threads(1)
    model = loaded.model.to(device=device, dtype=torch.float64).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_full, before_qc = model_sha256(model), state_sha256(qcr_state_dict(model))
    table = q_feature_table(model)
    _, probabilities = c_probability_table(model)
    probability_hash = state_sha256({'probabilities': probabilities})
    metadata = dict(source_run=str(source_run), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=checkpoint_hash, source_epoch=12000, theory_dim=dimension,
        source_sha256=source_hashes, source_data=str(data),
        source_data_file_bytes=data.stat().st_size,
        source_data_sha256_from_checkpoint=loaded.payload['metadata']['data_sha256'],
        source_data_sha256_from_manifest=manifest['data_sha256'], source_data_rehashed=False,
        source_data_expected_sha256=EXPECTED_DATA_SHA256,
        source_config=loaded.config, source_labels_used_in_training=False,
        labels_used_in_training=False, labels_accessed=False, K_label_access=False, Kc_selection=False,
        accessed_dataset_fields=['train/configurations', 'test/configurations'],
        all_ensembles_in_original_order=not smoke, split_counts=count,
        physical_configurations_per_ensemble=128, scales=list(SIDES), c_repeats=repeats,
        requested_c_repeats=c_repeats, seed=seed, device=str(device), threads=1,
        precision='torch.float64', smoke=smoke,
        model_sha256_before=before_full, frozen_QC_sha256_before=before_qc,
        frozen_QC_sha256=before_qc,
        source_R_sha256=state_sha256(model.state_dict(), residual=True),
        C_probability_table_sha256=probability_hash,
        operator_names=list(OPERATOR_NAMES[:EVEN_OPERATORS]),
        operator_definitions=list(OPERATOR_DEFINITIONS[:EVEN_OPERATORS]),
        seven_operator_helper=True, training_operators='six even extensive scores only',
        covariance_ddof=0, covariance_normalization='population 1/128, within each independent physical ensemble',
        response_definition='Dhead(mean f) Cov_population(per-configuration features, current-scale extensive score)',
        response_pairing='same current-level score reweights the input and its paired next-level configurations',
        head_derivative='exact autograd with frozen parameters; central finite-difference spotchecks',
        Q_encoding='head of full128 mean local features; differs from online mean of half-subset Q values',
        C_noise='independent declared diagnostic draws; two successive C steps share each draw generator',
        resampling_unit='physical ensemble, not configuration or C draw; C draws condition on the same physical data',
        no_training=True, no_optimizer=True, no_R_inference=True, no_MI=True,
        no_new_physical_sampling=True, no_QC_parameter_changes=True,
        TEST_policy='cache for evaluation only; never fit or normalize on TEST',
        limitation='Single trained seed and finite 108/54/27; empirical reweighting derivatives are noisy and do not certify thermal eigenmodes.')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = dict(train=0, test=0)
    write_json(output / 'plan.json', metadata)
    write_json(output / 'status.json', dict(status='running', pid=os.getpid(), **metadata,
                                           completed_ensembles=completed))
    cache, checks = {}, {}
    try:
        for split in SPLIT_COUNTS:
            tensors = dict(direct=torch.empty((count[split], repeats, 3, dimension), dtype=torch.float64),
                vin=torch.empty((count[split], repeats, 2, dimension, EVEN_OPERATORS), dtype=torch.float64),
                vout=torch.empty((count[split], repeats, 2, dimension, EVEN_OPERATORS), dtype=torch.float64),
                score_std=torch.empty((count[split], repeats, 2, EVEN_OPERATORS), dtype=torch.float64))
            seeds = torch.empty((count[split], repeats), dtype=torch.int64)
            for index in range(count[split]):
                row_seeds = [draw_seed(seed, split, index, draw) for draw in range(repeats)]
                spins = dataset[split][index, ..., 0].detach().to(device=device, dtype=torch.float64)
                row, validation = collect_ensemble(model, spins, table, probabilities, row_seeds,
                                                    spotcheck=index == 0)
                for key, value in row.items():
                    tensors[key][index].copy_(value)
                seeds[index] = torch.tensor(row_seeds, dtype=torch.int64)
                if validation:
                    checks[split] = validation
                completed[split] += 1
                status = dict(status='running', pid=os.getpid(), theory_dim=dimension,
                    current_split=split, completed_ensembles=dict(completed),
                    total_ensembles=count, c_repeats=repeats,
                    elapsed_seconds=time.monotonic() - started)
                write_json(output / 'status.json', status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
            cache.update({f'{split}_{key}': value for key, value in tensors.items()})
            cache[f'{split}_source_indices'] = torch.arange(count[split], dtype=torch.int64)
            cache[f'{split}_c_draw_seeds'] = seeds
        after_full, after_qc = model_sha256(model), state_sha256(qcr_state_dict(model))
        if (before_full != after_full or before_qc != after_qc
                or any(parameter.grad is not None for parameter in model.parameters())):
            raise RuntimeError('Frozen source model state/parameter gradients changed')
        if any(sha256_file(path) != fingerprint for path, fingerprint in source_hashes.items()):
            raise RuntimeError('A source checkpoint/receipt changed during collection')
        metadata.update(model_sha256_after=after_full, frozen_QC_sha256_after=after_qc,
                        derivative_spotchecks=checks,
                        elapsed_seconds=time.monotonic() - started,
                        shapes={key: list(value.shape) for key, value in cache.items()})
        cache['metadata'] = metadata
        pending = output / (CACHE_FILENAME + '.tmp')
        with pending.open('xb') as handle:
            torch.save(cache, handle)
        pending.replace(output / CACHE_FILENAME)
        receipt = dict(status='complete', workflow_complete=True, **metadata,
            response_cache=str(output / CACHE_FILENAME),
            response_cache_sha256=sha256_file(output / CACHE_FILENAME))
        write_json(output / 'receipt.json', receipt)
        write_json(output / 'status.json', dict(status='complete', workflow_complete=True,
            completed_ensembles=completed, split_counts=count, theory_dim=dimension,
            elapsed_seconds=time.monotonic() - started,
            response_cache_sha256=receipt['response_cache_sha256']))
        return receipt
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', workflow_complete=False,
            completed_ensembles=completed, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic() - started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--c-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--smoke', action='store_true')
    collect(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
