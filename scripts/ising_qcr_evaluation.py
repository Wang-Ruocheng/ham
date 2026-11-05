"""Versioned Q estimators for future b3 collections and JSON-only reports.

No model or tensor library is imported by the metadata helpers. Historical
unmarked collections retain their original full-ensemble interpretation.
"""
from __future__ import annotations

import random


HALF_SUBSET_ESTIMATOR = 'mean_of_two_disjoint_Q_halves_v1'
FULL_ENSEMBLE_ESTIMATOR = 'Q_of_full_ensemble_v1'
HALF_SPLIT_SEED = 2026101724


def encoding_contract(samples, *, split_seed=HALF_SPLIT_SEED):
    if type(samples) is not int or samples < 2 or samples % 2:
        raise ValueError('Evaluation requires a positive even configuration count')
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError('Require a fixed nonnegative half-split seed')
    partition = random.Random(split_seed).sample(range(samples), samples)
    return dict(version=1, estimator=HALF_SUBSET_ESTIMATOR, samples=samples,
        half_samples=samples // 2, sample_partition=partition, split_seed=split_seed,
        partition_rule='One fixed disjoint partition reused across chains, scales and C draws; no global RNG consumed',
        definition='0.5 * (Q(first half) + Q(second half)), matching the training closure estimator',
        qualification='Same estimator function; one fixed partition is not an average over training random partitions',
        diagnostic_estimator=FULL_ENSEMBLE_ESTIMATOR,
        Q_gap='Q(full ensemble) minus mean of half-ensemble Q values; not the R/C-draw Jensen gap')


def collection_estimator(metrics):
    """Missing metadata means legacy full-Q, never an invented corrected run."""
    version = metrics.get('schema_version', 1)
    if type(version) is not int or version < 1:
        raise ValueError('无效的采集 schema 版本')
    if version >= 2 and 'encoding_estimator' not in metrics:
        raise ValueError('新版采集缺少编码估计器；不能回退成历史口径')
    name = metrics.get('encoding_estimator', FULL_ENSEMBLE_ESTIMATOR)
    if name not in (HALF_SUBSET_ESTIMATOR, FULL_ENSEMBLE_ESTIMATOR):
        raise ValueError('未知的编码估计器；不能静默解释历史数组')
    if name == HALF_SUBSET_ESTIMATOR:
        contract = metrics.get('encoding_protocol', {})
        samples = metrics.get('samples')
        if (not isinstance(contract, dict) or type(samples) is not int or samples < 2 or samples % 2
                or type(contract.get('version')) is not int or contract.get('version') != 1
                or contract.get('estimator') != name
                or contract.get('samples') != samples or contract.get('half_samples') != samples // 2):
            raise ValueError('半子集编码协议或样本数缺失／不一致')
        partition = contract.get('sample_partition')
        if (not isinstance(partition, list) or len(partition) != samples
                or any(type(index) is not int for index in partition)
                or sorted(partition) != list(range(samples))):
            raise ValueError('半子集编码需要完整、无重复的固定划分')
    return name


def require_matching_estimators(metrics):
    """Refuse mixed finite-sample coordinates before computing comparisons."""
    metrics = list(metrics)
    names = [collection_estimator(row) for row in metrics]
    if len(set(names)) > 1:
        raise ValueError('编码估计器不同：不能混比历史全体 Q 与训练口径的半子集均值')
    samples = [row['samples'] for row in metrics if row.get('samples') is not None]
    if samples and any(value != samples[0] for value in samples[1:]):
        raise ValueError('编码估计器的样本数不同，不能直接比较')
    if names and names[0] == HALF_SUBSET_ESTIMATOR:
        signatures = [(row['samples'], row['encoding_protocol']['sample_partition']) for row in metrics]
        if any(value != signatures[0] for value in signatures[1:]):
            raise ValueError('半子集编码的样本数或固定划分不同，不能直接比较')
    return names[0] if names else None


def validate_encoding_arrays(metrics, arrays):
    """Check new estimator claims against saved diagnostics, without inference."""
    name = collection_estimator(metrics)
    if name == FULL_ENSEMBLE_ESTIMATOR:
        return name
    import numpy as np
    count = metrics.get('C_draws')
    if type(count) is not int or count < 1:
        raise ValueError('半子集评价必须记录正整数 C 重复数')
    for split in ('train', 'test'):
        reference = arrays.get(f'direct_draws_{split}')
        if (reference is None or reference.ndim != 4 or min(reference.shape) < 1
                or reference.shape[1] != count):
            raise ValueError('逐 C 编码形状或实际 C 重复数与元数据不符')
        for key in ('direct', 'R_g', 'rollout'):
            primary = arrays.get(f'{key}_draws_{split}')
            full = arrays.get(f'full_ensemble_{key}_draws_{split}')
            if (primary is None or full is None or primary.shape != reference.shape
                    or full.shape != reference.shape):
                raise ValueError('半子集评价必须保留同形的全体编码／R／rollout 诊断')
            for prefix, value in (('', primary), ('full_ensemble_', full)):
                mean = arrays.get(f'{prefix}{key}_{split}')
                if mean is None or mean.shape != value.mean(1).shape:
                    raise ValueError('缺少编码估计器对应的 C 均值')
                np.testing.assert_allclose(mean, value.mean(1), rtol=1e-9, atol=1e-11)
        gap = arrays.get(f'Q_full_minus_half_draws_{split}')
        expected = arrays[f'full_ensemble_direct_draws_{split}'] - arrays[f'direct_draws_{split}']
        if gap is None or gap.shape != expected.shape:
            raise ValueError('缺少 Q 全体／半子集差值诊断')
        np.testing.assert_allclose(gap, expected, rtol=1e-9, atol=1e-11)
        mean = arrays.get(f'Q_full_minus_half_{split}')
        if mean is None or mean.shape != gap.mean(1).shape:
            raise ValueError('缺少 Q 估计器差值的 C 均值')
        np.testing.assert_allclose(mean, gap.mean(1), rtol=1e-9, atol=1e-11)
    return name


def collect_saved_encodings(model, datasets, device, *, draw_count=5, batch_size=8):
    """One C trajectory per draw; retain both Q estimators without new C noise.

    The caller owns no_grad and RNG isolation. The fixed Q partition consumes
    no training/C RNG; full-ensemble diagnostics reuse these exact C samples.
    """
    import torch
    from recurrent_rg.encoding import half_subset_theory
    from recurrent_rg.future_physics import binary2d_future_observables

    datasets = list(datasets)
    if (not datasets or type(draw_count) is not int or draw_count < 1
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Require datasets and positive collection counts')
    samples = datasets[0][1].configurations.shape[1]
    contract = encoding_contract(samples)
    if any(dataset.configurations.shape[1] != samples for _, dataset in datasets):
        raise ValueError('TRAIN/TEST must use the same Q estimator sample count')
    partition = torch.tensor(contract['sample_partition'], dtype=torch.long, device=device)
    arrays = {}
    for split, dataset in datasets:
        chunks = {key: [] for key in ('direct', 'R_g', 'rollout', 'full_ensemble_direct',
                                     'full_ensemble_R_g', 'full_ensemble_rollout', 'observables')}
        for start in range(0, len(dataset.configurations), batch_size):
            batch = dataset.configurations[start:start + batch_size].to(device=device, dtype=torch.float64)
            draws = {key: [] for key in chunks}
            for _ in range(draw_count):
                levels = model.coarse_grain(batch)
                g = half_subset_theory(model, [level.index_select(1, partition) for level in levels])
                full = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
                for prefix, direct in (('', g), ('full_ensemble_', full)):
                    rollout = [direct[:, 0]]
                    for _ in range(len(levels) - 1):
                        rollout.append(model.advance_theory(rollout[-1]))
                    draws[prefix + 'direct'].append(direct.cpu())
                    draws[prefix + 'R_g'].append(model.advance_theory(direct).cpu())
                    draws[prefix + 'rollout'].append(torch.stack(rollout, 1).cpu())
                draws['observables'].append(torch.stack(
                    [binary2d_future_observables(level) for level in levels], 1).cpu())
            for key in chunks:
                chunks[key].append(torch.stack(draws[key], 1))
        values = {key: torch.cat(parts) for key, parts in chunks.items()}
        for key, value in values.items():
            arrays[f'{key}_draws_{split}'] = value.numpy()
            arrays[f'{key}_{split}'] = value.mean(1).numpy()
        for prefix in ('', 'full_ensemble_'):
            mean = values[prefix + 'direct'].mean(1).to(device)
            arrays[f'{prefix}R_of_mean_g_{split}'] = model.advance_theory(mean).cpu().numpy()
        gap = values['full_ensemble_direct'] - values['direct']
        arrays[f'Q_full_minus_half_draws_{split}'] = gap.numpy()
        arrays[f'Q_full_minus_half_{split}'] = gap.mean(1).numpy()
        arrays[f'couplings_{split}'] = dataset.couplings.numpy()
        arrays[f'group_ids_{split}'] = dataset.group_ids.numpy()
    return arrays, contract
