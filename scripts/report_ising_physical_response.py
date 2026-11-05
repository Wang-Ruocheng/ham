"""冻结 Q/C 的物理响应 R 消融报告；只读缓存，绝不重新推理 Q/C。

K 标签只从原正式报告读入以评价曲线、根的逆 K 分支及 Kc 邻域。
响应目标来自独立物理构型的保存统计，不以理论倍率或指数挑选根。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.ising_physical_response_ablation import evaluate as driver_evaluate
from experiments.ising_response_loss_ablation import state_sha256
from recurrent_rg.checkpoints import load_run
from scripts.report_ising_dimg_ablation import (
    KC, closure_details, coupling_statistics, find_roots, json_value, mapped,
    plot_maps, representation_statistics, write_json,
)
from scripts.report_ising_zh import chinese_figures


ARM_LABELS = {'source': '原始 R', 'control': '仅闭合续训', 'response': '闭合＋物理响应'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cached_split(cache, split):
    """保存的物理系综轴和五次 C 轴始终分开，禁止把它们当独立物理样本。"""
    nested = cache.get(split, {})
    values = {}
    for name in ('direct', 'vin', 'vout', 'score_std'):
        raw = cache.get(split + '_' + name, nested.get(name))
        if raw is None:
            raise ValueError(f'Missing saved {split}_{name}')
        values[name] = torch.as_tensor(raw, dtype=torch.float64).detach().cpu()
    direct, vin, vout, score = (values[name] for name in ('direct', 'vin', 'vout', 'score_std'))
    if (direct.ndim != 4 or direct.shape[1:3] != (5, 3) or direct.shape[-1] not in (1, 2, 3)
            or not len(direct) or vin.shape != (*direct.shape[:2], 2, direct.shape[-1], 6)
            or vout.shape != vin.shape or score.shape != (*direct.shape[:2], 2, 6)):
        raise ValueError('Need saved [ensemble,5 C draws,3 scales,d] and paired six-operator responses')
    if any(not torch.isfinite(value).all() for value in values.values()) or (score < 0).any():
        raise ValueError('Saved response arrays must be finite and score deviations nonnegative')
    return values


def evaluation_labels(source_report, cache, split, count):
    """只读原报告标签；执行检查子集必须以显式原始行索引对齐。"""
    with np.load(Path(source_report) / 'encodings.npz', allow_pickle=False) as encodings:
        labels = np.asarray(encodings[split + '_labels']).reshape(-1)
    indices = cache.get(split + '_source_indices')
    if indices is None and isinstance(cache.get('source_indices'), dict):
        indices = cache['source_indices'].get(split)
    if indices is not None:
        indices = np.asarray(indices)
        if (indices.ndim != 1 or len(indices) != count or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(labels)) or len(np.unique(indices)) != len(indices)):
            raise ValueError('Invalid saved source ensemble row indices')
        labels = labels[indices]
    if len(labels) != count or not np.isfinite(labels).all():
        raise ValueError('Saved cache order/count differs from original report; explicit indices required')
    return labels


def saved_trajectories(model, direct):
    """仅调用 R；direct 已由保存 Q/C 计算，不读取物理构型。"""
    direct = np.asarray(direct, dtype=np.float64)
    rollout = [direct[:, 0]]
    for _ in range(direct.shape[1] - 1):
        rollout.append(mapped(model, rollout[-1]))
    return dict(direct=direct, one_step=mapped(model, direct[:, :-1]),
                rollout=np.stack(rollout, axis=1))


def response_diagnostics(model, values, normalization):
    """完整 DR（包括恒等项）对保存物理响应的作用；不构造伪热谱。"""
    direct, vin, vout, score = (values[name] for name in ('direct', 'vin', 'vout', 'score_std'))
    shape = vin.shape
    points = direct[:, :, :-1].reshape(-1, direct.shape[-1])
    directions = vin.reshape(-1, direct.shape[-1], 6)
    prediction = []
    for operator in range(6):
        _, derivative = torch.autograd.functional.jvp(model.advance_theory, points,
            directions[:, :, operator], create_graph=False, strict=False)
        prediction.append(derivative.detach())
    predicted = torch.stack(prediction, dim=-1).reshape(shape)
    inverse_std = torch.where(score > 1e-12, 1. / score.clamp_min(1e-12), torch.zeros_like(score))
    normalized_error = (predicted - vout) * inverse_std.unsqueeze(-2)
    normalized_target = vout * inverse_std.unsqueeze(-2)
    error = normalized_error.numpy()
    target = normalized_target.numpy()
    # Exactly the training convention: SUM coordinates, MEAN ensembles/draws/operators.
    squared = np.mean(np.sum(error ** 2, axis=3), axis=(0, 1, 3))
    target_power = np.sum(target ** 2, axis=(0, 1, 3, 4))
    error_power = np.sum(error ** 2, axis=(0, 1, 3, 4))
    relative = [float(np.sqrt(a / b)) if b > 1e-30 else None for a, b in zip(error_power, target_power)]
    per_ensemble_target = np.sum(target ** 2, axis=(1, 3, 4))
    per_ensemble_error = np.sum(error ** 2, axis=(1, 3, 4))
    per_ensemble = np.sqrt(np.divide(per_ensemble_error, per_ensemble_target,
        out=np.full_like(per_ensemble_error, np.nan), where=per_ensemble_target > 1e-30))
    normalization = np.asarray(normalization, dtype=float)
    if normalization.shape != (2,) or not np.isfinite(normalization).all() or np.any(normalization <= 0):
        raise ValueError('Need two positive fixed TRAIN response normalization values')
    return dict(response_mse_by_scale=squared.tolist(), response_normalized=float(np.mean(squared / normalization)),
        response_relative_norm_by_scale=relative, response_relative_norm_per_physical_ensemble=per_ensemble.tolist(),
        response_relative_train_reference_by_scale=np.sqrt(squared/normalization).tolist(),
        score_zero_fraction_by_scale=np.mean(score.numpy() <= 1e-12, axis=(0, 1, 3)).tolist(),
        physical_ensembles=len(direct), C_draws=direct.shape[1],
        definition='先按各系综物理扰动得分标准差归一，坐标求和、其余轴均值；完整 DR·vin 比较保存 vout，不是根的热谱。',
        relative_norm_definition='relative_norm 的分母是该评价 split 保存目标的范数；另列 train_reference 才使用 TRAIN 预定标尺。',
        normalization_source='仅 TRAIN 源模型预定标尺，TEST 不参与标尺或权重标定。',
        confidence_interval=None,
        caution='五次 C 抽样不当成五个独立物理系综；此处不伪造训练 seed、有限尺寸或根指数置信区间。')


def augment_inverse_K(roots):
    for root in roots['roots']:
        available = []
        for row in root['curve_projections_by_scale']:
            row['exact_inverse_K_errors'] = [dict(K=value, signed_error=value-KC,
                absolute_error=abs(value-KC), relative_absolute_error=abs(value-KC)/KC)
                for value in row['exact_inverse_K_branches']]
            available.extend(row['exact_inverse_K_branches'])
        root['inverse_K_all_branch_span'] = max(available)-min(available) if available else None
        root['inverse_K_caution'] = '保留各尺度全部分支；多分支总跨度不是唯一根对应的尺度漂移。最近投影不当成精确逆 K。'
    return roots


def critical_diagnostics(model, direct, labels, knots, means, critical):
    """R(mean g)、mean R(g) 和下一尺度均值分别保留，均只调用 R。"""
    mean_R = np.stack([mapped(model, direct[labels == K]).mean(0) for K in knots])
    critical_mean_R = np.stack([[np.interp(KC, knots, mean_R[:, scale, coordinate])
        for coordinate in range(direct.shape[-1])] for scale in range(3)])
    R_mean = mapped(model, critical)
    return dict(Kc=KC, evaluation_only=True, encoded_g_by_scale=critical.tolist(),
        R_of_encoded_mean=R_mean.tolist(), mean_R_of_encoded=critical_mean_R.tolist(),
        R_of_mean_minus_next_mean=(R_mean[:-1]-critical[1:]).tolist(),
        mean_R_minus_next_mean=(critical_mean_R[:-1]-critical[1:]).tolist(),
        jensen_mean_R_minus_R_mean=(critical_mean_R-R_mean).tolist(),
        direct_scale_drift=np.diff(critical, axis=0).tolist(),
        R_drift=(R_mean-critical).tolist(),
        caution='Kc 紫菱形是前向编码，不是 R 根；mean R 先逐系综调用 R 再均值，Kc 处仅对已保存 K 点插值。')


def repeated_closure(model, split, train):
    direct = split['direct'].numpy()
    reference = train['direct'].numpy()
    per_draw = [closure_details(model, saved_trajectories(model, direct[:, index]), reference[:, index])
                for index in range(5)]
    aggregate = closure_details(model, saved_trajectories(model, direct.reshape(-1, 3, direct.shape[-1])),
                              reference.reshape(-1, 3, reference.shape[-1]))
    return dict(all_five_draws=aggregate, by_C_draw=per_draw,
        caution='合并抽样轴只计算点估计；并未把抽样数当独立物理系综数或指数不确定度。')


def history_rows(loaded, folder):
    rows = loaded.payload.get('history')
    if rows is None and (folder / 'history_progress.json').is_file():
        raw = json.loads((folder / 'history_progress.json').read_text())
        rows = raw if isinstance(raw, list) else raw.get('history', [])
    return rows or []


def plot_arm_diagnostics(closure, response, history, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for split, style, label in (('train', '-', '训练集'), ('test', '--', '测试集（仅评价）')):
        values = closure[split]['all_five_draws']['all_start_horizon']
        axes[0].plot(range(3), [row['nrmse'] for row in values], 'o'+style, label=label)
        axes[1].plot([1, 2], response[split]['response_relative_norm_by_scale'], 'o'+style, label=label)
    axes[0].set(title='五次 C 抽样的闭环点估计', xlabel='路径：108→54、108→27、54→27', ylabel='训练目标标准差归一误差', xticks=range(3))
    axes[1].set(title='完整 R 导数的物理响应误差', xlabel='变换编号', ylabel='物理扰动得分归一后的相对范数误差', xticks=[1, 2])
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output / 'response_comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for split, style, label in (('train', '-', '训练集'), ('test', '--', '测试集（仅评价）')):
        for key, title in (('closure', '加权闭环'), ('response_normalized', '标尺归一响应'),
                           ('one_step_mse', '单步均方误差'), ('rollout_mse', '多步均方误差')):
            rows = [row for row in history if split+'_'+key in row]
            if rows:
                ax = axes[1 if key == 'response_normalized' else 0]
                ax.plot([row.get('epoch', row.get('update', row.get('step'))) for row in rows],
                    [row[split+'_'+key] for row in rows], style, label=label+'：'+title)
    axes[0].set(title='R 续训闭环记录；Q/C 保持不变', xlabel='新增 R 更新次数', ylabel='记录值', yscale='symlog')
    axes[1].set(title='独立物理响应记录；不监督指数', xlabel='新增 R 更新次数', ylabel='训练集预定标尺归一响应', yscale='symlog')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle('原 Q 重构与 C 的 MI/方差/颜色目标未更新，也未在本报告重算')
    fig.savefig(output / 'loss_comparison.png', dpi=150); plt.close(fig)


def arm_report(name, loaded, cache_splits, labels, plan, output, *, updates):
    output.mkdir(parents=True, exist_ok=False)
    direct = {split: value['direct'][:, 0].numpy() for split, value in cache_splits.items()}
    knots, means, critical = coupling_statistics(direct['train'], direct['test'], labels['train'], labels['test'])
    roots = augment_inverse_K(find_roots(loaded.model, direct['train'], knots, means, block_factor=2))
    closure = {split: repeated_closure(loaded.model, values, cache_splits['train']) for split, values in cache_splits.items()}
    response = {split: response_diagnostics(loaded.model, values, plan['normalization']) for split, values in cache_splits.items()}
    driver_metrics = {}
    # The trainer's analytic JVP and this report's autograd JVP are independently checked.
    from experiments.ising_physical_response_ablation import FrozenResidualR
    only_R = FrozenResidualR(loaded.payload['model']) if 'model' in loaded.payload else None
    if only_R is not None:
        for split, values in cache_splits.items():
            driver_metrics[split] = driver_evaluate(only_R,
                *[values[key] for key in ('direct', 'vin', 'vout', 'score_std')],
                torch.as_tensor(plan['normalization'], dtype=torch.float64),
                plan['response_weight'] if name == 'response' else 0.)
            if not np.allclose(driver_metrics[split]['response_mse_by_scale'],
                               response[split]['response_mse_by_scale'], rtol=1e-9, atol=1e-12):
                raise ValueError('Analytic trainer response JVP differs from independent report autograd JVP')
    all_direct = np.concatenate((direct['train'], direct['test']))
    all_labels = np.concatenate((labels['train'], labels['test']))
    history = history_rows(loaded, loaded.checkpoint.parent) if name != 'source' else []
    result = dict(schema_version=1, arm=name, arm_label=ARM_LABELS[name], source_epoch=plan.get('source_epoch', 12000),
        additional_R_updates=0 if name == 'source' else updates, theory_dim=direct['train'].shape[-1], block_factor=2,
        lattice_sizes=[108, 54, 27], checkpoint=str(loaded.checkpoint),
        labels_used_in_training=False, labels_used_in_report=True,
        Q_C_inference_performed=False, physical_sampling_performed=False, MI_estimated=False,
        first_C_draw_rule='预定第 0 次保存抽样用于根、K 曲线和几何支持；不按结果选抽样。',
        closure=closure, physical_response=response, driver_metrics=driver_metrics,
        roots=roots['roots'], fixed_points=roots,
        critical_reference=critical_diagnostics(loaded.model, all_direct, all_labels, knots, means, critical),
        train_representation=representation_statistics(direct['train']),
        response_weight=plan['response_weight'] if name == 'response' else 0.,
        source_Q_C_objectives='原 Q 目标与 C 的 MI/方差/颜色目标及权重不变；本阶段 Q/C 冻结，无对应新训练更新。',
        selection=dict(predeclared_updates=updates, best_checkpoint_selection=False, all_found_roots_retained=True,
            all_eigenvalues_retained=True, exhaustive_search_claim=False, selected_thermal_mode=None),
        limitations=['单 seed、有限 108/54/27、两次粗粒化、仅 B0；不独立识别磁指数。',
            '有条件 ln2/lnλ 是算术换算，不把 K 路径或几何支持当作热模认证。',
            '冻结 Q/C 的 R 消融只能检验 R 的拟合瓶颈，不能自动修正表示或有限尺度偏差。'])
    np.savez_compressed(output/'encodings.npz', train=direct['train'], test=direct['test'],
        train_all_C_draws=cache_splits['train']['direct'].numpy(),
        test_all_C_draws=cache_splits['test']['direct'].numpy(),
        train_labels=labels['train'], test_labels=labels['test'], K=knots, mean_by_scale=means)
    write_json(output/'metrics.json', result)
    write_json(output/'fixed_points.json', roots)
    write_json(output/'critical_exponents.json', result)
    write_json(output/'loss_history.json', history)
    with chinese_figures():
        plot_maps(loaded.model, direct['train'], direct['test'], knots, means, critical, roots, output)
        plot_arm_diagnostics(closure, response, history, output)
    write_json(output/'report.json', dict(status='complete', report_complete=True, additional_R_updates=result['additional_R_updates'],
        arm=name, public_files=sorted(path.name for path in output.iterdir() if path.is_file()),
        Q_C_inference_performed=False, training_started=False, physical_sampling_started=False, MI_estimated=False))
    return result


def plot_comparison(reports, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    for index, (name, report) in enumerate(reports.items()):
        label = ARM_LABELS[name]
        closure = report['closure']['test']['all_five_draws']['all_start_horizon']
        axes[0].plot(range(3), [row['nrmse'] for row in closure], 'o-', label=label)
        axes[1].plot([1, 2], report['physical_response']['test']['response_relative_norm_by_scale'], 'o-', label=label)
        for root in report['roots']:
            for eigenvalue in root['eigenvalues']:
                axes[2].scatter(index, eigenvalue['magnitude'], s=45,
                    facecolors=f'C{index}' if root['geometric_support_candidate'] else 'none', edgecolors=f'C{index}')
    axes[0].set(title='测试集闭环误差', xlabel='路径：108→54、108→27、54→27', ylabel='训练目标标准差归一误差', xticks=range(3))
    axes[1].set(title='测试集物理响应误差', xlabel='变换编号', ylabel='相对范数误差', xticks=[1, 2])
    axes[2].set(title='所有已找到根的完整谱；非认证热指数', xlabel='预定实验臂', ylabel='特征值模长',
                xticks=range(len(reports)), xticklabels=[ARM_LABELS[name] for name in reports])
    for ax in axes:
        ax.grid(alpha=.2)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    fig.suptitle('冻结同一无监督 Q/C：只比较预定 R 更新终点；不按接近理论筛根')
    fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)


def validate_arm(loaded, folder, updates, plan):
    # The base architecture/config and historical epoch remain 12000 by design;
    # the filename denotes EXTRA R updates and must not be mistaken for full QCR epochs.
    metadata = loaded.payload.get('metadata', {})
    actual_saved = loaded.payload.get('extra_R_updates', metadata.get('extra_R_updates'))
    if loaded.payload.get('epoch') != plan.get('source_epoch', 12000) or actual_saved != updates:
        raise ValueError('R arm numbered checkpoint does not reach predeclared additional updates')
    summary = json.loads((folder/'summary.json').read_text())
    actual = summary.get('actual_extra_R_updates', summary.get('actual_updates', summary.get('actual_epochs')))
    if actual != updates or any(summary.get(key) for key in
        ('stopped_by_wall_limit', 'stopped_by_training_limit', 'stopped_by_train_plateau')):
        raise ValueError('R arm summary is incomplete or early stopped')
    for key in ('source_checkpoint_sha256', 'frozen_QC_sha256', 'initial_R_sha256'):
        if key in metadata and key in plan and metadata[key] != plan[key]:
            raise ValueError('R arm provenance mismatch: '+key)
    if 'model' in loaded.payload and plan.get('frozen_QC_sha256'):
        if state_sha256(loaded.payload['model']) != plan['frozen_QC_sha256']:
            raise ValueError('Frozen Q/C tensor fingerprint differs from source plan')


def run_report(run, output, *, updates=6000):
    run, output = Path(run), Path(output)
    if updates < 1 or output.exists():
        raise ValueError('Require positive predeclared updates and a fresh report directory')
    plan_path = run/'training'/'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan.get('updates') != updates:
        raise ValueError('Requested report updates differ from predeclared training plan')
    source = Path(plan['source_run'])
    source_report = Path(plan.get('source_report', source.parent/'report'))
    cache_path = Path(plan.get('cache_path', run/'cache'/'response_cache.pt'))
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if plan.get('response_cache_sha256') and sha256_file(cache_path) != plan['response_cache_sha256']:
        raise ValueError('Saved response cache fingerprint differs from TRAIN plan')
    splits = {split: cached_split(cache, split) for split in ('train', 'test')}
    if splits['train']['direct'].shape[1:] != splits['test']['direct'].shape[1:]:
        raise ValueError('TRAIN/TEST saved dimensions differ')
    labels = {split: evaluation_labels(source_report, cache, split, len(values['direct'])) for split, values in splits.items()}
    loaded = {'source': load_run(source, plan.get('source_epoch', 12000))}
    expected_sha = plan.get('source_checkpoint_sha256')
    if expected_sha and sha256_file(loaded['source'].checkpoint) != expected_sha:
        raise ValueError('Original source numbered checkpoint fingerprint mismatch')
    for arm in ('control', 'response'):
        folder = run/'training'/arm
        loaded[arm] = load_run(folder, updates)
        validate_arm(loaded[arm], folder, updates, plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'report.json', dict(status='running', report_complete=False,
        Q_C_inference_performed=False, training_started=False))
    reports = {name: arm_report(name, value, splits, labels, plan, output/name, updates=updates)
               for name, value in loaded.items()}
    write_json(output/'comparison.json', dict(schema_version=1, reports=reports, source_plan=plan,
        source_plan_sha256=sha256_file(plan_path), cache_sha256=sha256_file(cache_path),
        caution='此消融只训练 R；标签仅评价。物理响应匹配改善不自动等于渐近热指数已认证。'))
    with chinese_figures():
        plot_comparison(reports, output)
    write_json(output/'report.json', dict(status='complete', report_complete=True, updates=updates,
        arms=list(reports), Q_C_inference_performed=False, training_started=False,
        physical_sampling_started=False, MI_estimated=False,
        public_files=sorted(str(path.relative_to(output)) for path in output.rglob('*') if path.is_file())))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=6000)
    args = parser.parse_args(argv)
    run_report(args.run, args.output, updates=args.updates)


if __name__ == '__main__':
    main()
