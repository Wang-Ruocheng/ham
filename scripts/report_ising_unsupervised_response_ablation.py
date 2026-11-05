"""旧无监督 Q/C 冻结后的 K-free R 割线诊断；K 标签仅在本报告评价。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.runner import file_fingerprint
from scripts.report_ising_zh import chinese_figures
from scripts.report_ising_scalar_exponents import scalar_coupling_means
from scripts.report_ising_response_ablation import (
    ARMS, COLORS, analyze_arm, json_value, load_encodings,
    shared_representation_statistics, validate_fixed_final,
)


LABELS = dict(control='仅闭合对照', response='增加无标签割线损失')


def validate_unsupervised_final(plan, source, loaded, summaries, root):
    """先检查完全无监督的来源及新训练契约，再允许报告读取评价标签。"""
    metadata = source.payload.get('metadata', {})
    if (plan['source_epoch'] != 12000 or metadata.get('labels_used_in_training') is not False or
            metadata.get('q_supervision') not in (None, {})):
        raise ValueError('Requires the actual 12000-update source trained entirely without label supervision')
    for arm in ARMS:
        meta = loaded[arm].payload.get('metadata', {})
        if (meta.get('labels_used_in_training') is not False or
                meta.get('K_label_access') is not False or
                meta.get('q_supervision') not in (None, {})):
            raise ValueError('The new R-only arms must explicitly forbid training K-label access')
    validate_fixed_final(plan, source, loaded, summaries, root)


def label_free_response_metrics(models, saved_direct, plan):
    """训练口径只接收 direct；TEST 共用 TRAIN 的尺度、距离阈值及归一量。"""
    from experiments.ising_unsupervised_response_ablation import prepare_secant_data, metrics
    options = dict(neighbors=plan.get('neighbors', 8),
                   min_distance=plan.get('min_distance', .05),
                   max_distance=plan.get('max_distance', .25))
    train = prepare_secant_data(torch.from_numpy(saved_direct['train']), **options)
    sigma = torch.as_tensor(plan['train_sigma'], dtype=torch.float64)
    normalization = torch.as_tensor(plan['normalization'], dtype=torch.float64)
    expected = torch.stack([target.square().mean() for target in train.target_secants]).clamp_min(1e-12)
    if (not torch.allclose(train.train_sigma, sigma, rtol=1e-12, atol=1e-14) or
            not torch.allclose(normalization, expected, rtol=1e-12, atol=1e-14)):
        raise ValueError('Pair geometry and normalizers must come only from frozen TRAIN encodings')
    test = prepare_secant_data(torch.from_numpy(saved_direct['test']), train_sigma=sigma, **options)
    prepared = dict(train=train, test=test)
    result = {arm: {split: json_value(metrics(model, data, normalization,
                        alpha=0. if arm == 'control' else plan['alpha']))
                    for split, data in prepared.items()}
              for arm, model in models.items()}
    return result, {split: json_value(data.diagnostics) for split, data in prepared.items()}


def microscopic_K_evaluation(encodings):
    """直接数值对照，不拟合坐标，不把 K 回归误差用于训练或选择。"""
    return {split: dict(raw_g0_minus_K_mse=float(np.mean((encodings[split][:, 0, 0] -
                           encodings[split+'_labels'].ravel())**2)),
                training_use=False,
                caution='原始无监督 g 坐标未对齐或拟合到 K；此 MSE 仅为评价对照，不是坐标不变指标。')
            for split in ('train', 'test')}


def draw_maps(analyses, models, encodings, initial, output):
    knots, means = scalar_coupling_means(encodings['train'], encodings['test'],
                                       encodings['train_labels'], encodings['test_labels'])
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    for arm in ARMS:
        info = analyses[arm]
        g, mapped = np.asarray(info['flow']['g']), np.asarray(info['flow']['R'])
        axes[0].plot(g, mapped, color=COLORS[arm], label=LABELS[arm])
        axes[1].plot(g, mapped-g, color=COLORS[arm], label=LABELS[arm])
        for root in info['roots']:
            point = root['g'][0]
            axes[0].scatter(point, point, marker='*', s=95, c=COLORS[arm], edgecolors='black', zorder=6)
            axes[1].scatter(point, 0., marker='*', s=95, c=COLORS[arm], edgecolors='black', zorder=6)
        critical = np.asarray(info['critical']['g_by_scale'])
        with torch.no_grad():
            r = models[arm].advance_theory(torch.from_numpy(critical[:, None])).numpy()[:, 0]
        for panel, values in ((0, r), (1, r-critical)):
            axes[panel].scatter(critical, values, marker='D', s=45, c='magenta',
                edgecolors=COLORS[arm], zorder=7,
                label='Kc 前向编码（仅评价；非不动点）' if arm == 'control' else None)
    g = np.asarray(analyses['control']['flow']['g'])
    with torch.no_grad():
        start = initial.advance_theory(torch.from_numpy(g[:, None])).numpy()[:, 0]
    axes[0].plot(g, start, color='gray', ls=':', label='共同初始 R')
    axes[1].plot(g, start-g, color='gray', ls=':', label='共同初始 R')
    axes[0].plot(g, g, 'k--', lw=.8, label='恒等映射')
    axes[1].axhline(0., color='black', lw=.8, ls='--')
    for ax in axes[:2]:
        ax.axvspan(*analyses['control']['training_input_range'], color='green', alpha=.07,
                   label='R 的训练输入范围')
    axes[0].set(title='固定末点：完整 R(g)', xlabel='g', ylabel='R(g)')
    axes[1].set(title='固定末点：位移与全部已找到的根', xlabel='g', ylabel='R(g)-g')
    for scale in range(3):
        axes[2].plot(knots, means[:, scale], label=f'k={scale}')
    reference = analyses['control']['critical']
    axes[2].axvline(reference['Kc'], color='magenta', ls=':', label='已知微观 Kc（仅评价）')
    axes[2].scatter([reference['Kc']]*3, reference['g_by_scale'], marker='D', c='magenta', s=45)
    for scale, value in enumerate(reference['g_by_scale']):
        axes[2].annotate(f'k={scale}', (reference['Kc'], value),
            xytext=((12, 24), (12, -24), (34, 0))[scale], textcoords='offset points', color='magenta',
            arrowprops=dict(arrowstyle='-', color='magenta', lw=.6))
    axes[2].set(title='原无监督编码完全不变；K 仅评价', xlabel='微观 K（仅评价）', ylabel='g')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f'冻结旧无监督 Q、C：各增加 {analyses["control"]["extra_R_updates"]} 次 R 更新；训练完全不读 K 标签')
    fig.supxlabel('这是保存轨迹的仅 R 诊断，不是完整 QCR 从头重训。紫色菱形非不动点；末尺度无下一尺度目标；数值根搜索有界。', fontsize=9)
    fig.savefig(output / 'R_maps_comparison.png', dpi=170)
    plt.close(fig)


def draw_losses(histories, output):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), layout='constrained')
    for row, split in enumerate(('train', 'test')):
        split_name = '训练集' if split == 'train' else '测试集（仅评价）'
        for column, (key, title) in enumerate((('closure', '加权单步与多步闭合'),
                ('response_mse', '无标签割线误差'),
                ('response_normalized', '按训练目标归一的割线损失'), ('total', '实际目标总损失'))):
            for arm in ARMS:
                history = histories[arm]
                axes[row, column].plot([point['extra_R_updates'] for point in history],
                    [point[split+'_'+key] for point in history], color=COLORS[arm], label=LABELS[arm])
            axes[row, column].set(title=split_name+'：'+title, xlabel='额外 R 更新次数', ylabel='损失')
            axes[row, column].set_yscale('symlog', linthresh=1e-10)
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('固定相同末点预算：无标签局部割线一致性；测试结果不参与训练或选择')
    fig.savefig(output / 'loss_curves.png', dpi=150)
    plt.close(fig)


def draw_test_metrics(analyses, measured, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    panels = ((axes[0, 0], '闭合绝对均方根误差', lambda arm: analyses[arm]['closure']['test']['one_step']['rmse']),
              (axes[1, 0], '按训练目标标准差归一的闭合误差', lambda arm: analyses[arm]['closure']['test']['one_step']['nrmse']),
              (axes[0, 1], '无标签割线均方误差', lambda arm: measured[arm]['test']['response_mse_by_scale']),
              (axes[1, 1], '按训练目标归一的割线误差', lambda arm: measured[arm]['test']['response_normalized_by_scale']))
    for ax, title, getter in panels:
        for index, arm in enumerate(ARMS):
            values = [np.nan if value is None else value for value in getter(arm)]
            ax.bar(np.arange(2)+(index-.5)*.34, values, width=.34, color=COLORS[arm], label=LABELS[arm])
        ax.set(title='测试集：'+title, ylabel='误差', xticks=[0, 1], xticklabels=['k=0→1', 'k=1→2'])
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('测试仅评价：邻居、距离尺度与归一量均按训练规则固定；完全不使用 K 标签')
    fig.supxlabel('割线一致性不是独立热响应或对 K 的导数；比较以同一份保存 C 抽样编码为条件。', fontsize=9)
    fig.savefig(output / 'label_free_test_metrics.png', dpi=160)
    plt.close(fig)


def draw_posthoc_K_windows(analyses, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for scale, ax in enumerate(axes):
        target_drawn = False
        for arm in ARMS:
            x, predicted, observed = [], [], []
            for window in analyses[arm]['local_response']['critical_windows']:
                if window['status'] != 'complete':
                    continue
                point = window['transitions'][scale]
                x.append(window['half_width']); predicted.append(point['predicted_response'])
                observed.append(point['empirical_response'])
                ax.vlines(window['half_width'], *point['predicted_response_bootstrap_95'], color=COLORS[arm], alpha=.5)
            ax.plot(x, predicted, 'o-', color=COLORS[arm], label=LABELS[arm])
            if not target_drawn:
                ax.plot(x, observed, 'k--', label='原编码的数据响应')
                target_drawn = True
        ax.set(title=f'尺度 k={scale}→{scale+1}：Kc 邻域事后评价', xlabel='K 窗口半宽',
               ylabel='局部响应比', xticks=[.02, .03, .04])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.supxlabel('K 标签只用于本图评价，未进入割线训练。区间以单模型、保存 C 抽样为条件，不是热乘子置信区间。', fontsize=9)
    fig.savefig(output / 'posthoc_K_window_response.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.bootstraps < 1:
        raise ValueError('Positive bootstrap count required')
    torch.set_num_threads(1)
    plan = json.loads((args.run / 'plan.json').read_text())
    source = load_run(Path(plan['source_checkpoint']))
    loaded = {arm: load_run(args.run / arm, 'checkpoint_last.pt') for arm in ARMS}
    summaries = {arm: json.loads((args.run / arm / 'summary.json').read_text()) for arm in ARMS}
    validate_unsupervised_final(plan, source, loaded, summaries, args.run)
    # Only the completed-run report now opens labels, solely for evaluation.
    source_report = Path(plan['source_report'])
    if file_fingerprint(source_report / 'encodings.npz') != plan['encodings_sha256']:
        raise ValueError('Frozen saved encodings changed')
    encodings = load_encodings(source_report)
    models = {arm: item.model.cpu().double().eval() for arm, item in loaded.items()}
    measured, pair_diagnostics = label_free_response_metrics(
        models, {split: encodings[split] for split in ('train', 'test')}, plan)
    values = np.concatenate((encodings['train'], encodings['test']))
    pad = max(.2, .2*float(np.ptp(values)))
    bounds = [float(values.min()-pad), float(values.max()+pad)]
    histories = {arm: json.loads((args.run / arm / 'history_progress.json').read_text()) for arm in ARMS}
    if any(history[-1]['extra_R_updates'] != plan['updates'] for history in histories.values()):
        raise ValueError('Both histories must end at the identical fixed R-only budget')
    args.output.mkdir(parents=True, exist_ok=False)
    with chinese_figures():
        analyses = {arm: analyze_arm(loaded[arm], encodings, bounds, args.output / arm,
                        plan['updates'], bootstraps=args.bootstraps) for arm in ARMS}
        draw_maps(analyses, models, encodings, source.model.cpu().double().eval(), args.output)
        draw_losses(histories, args.output)
        draw_test_metrics(analyses, measured, args.output)
        draw_posthoc_K_windows(analyses, args.output)
    if file_fingerprint(source_report / 'encodings.npz') != plan['encodings_sha256']:
        raise RuntimeError('Frozen source encodings changed while reporting')
    result = dict(status='complete', source_QCR_epoch=12000, extra_R_updates_per_arm=plan['updates'],
        alpha=plan['alpha'], source_checkpoint=str(source.checkpoint),
        source_checkpoint_sha256=plan['source_checkpoint_sha256'], encodings_sha256=plan['encodings_sha256'],
        source_labels_used_in_training=False, labels_used_in_new_training=False,
        K_label_access_in_training=False, labels_used_for_report_evaluation=True,
        Q_C_inference=False, physical_sampling=False, MI_calculation=False,
        shared_initial_Q_C_R_verified=True, frozen_Q_C_verified=True, full_framework_retrained=False,
        scope='冻结旧完全无监督 Q/C 的保存轨迹，仅重新拟合 R；不等于从头新训完整无监督 QCR 框架。',
        conditioning='两组共用一次保存的 C 抽样编码；统计区间不代表 C 重抽样、独立训练种子或完整框架不确定性。',
        selection='固定相同额外 R 更新预算末点；不按 TEST、Kc、理论值、ν 或根选择。',
        label_free_response='只在当前 g 尺度邻域取割线，比较 R 差与保存的下一尺度差；TEST 使用 TRAIN 标准差及目标归一量；不是对 K 的导数或热响应认证。',
        posthoc_K_windows='K 标签只在完成训练后的报告评价使用；不进入邻居、归一化、损失、预算或模型选择。',
        microscopic_K_evaluation=microscopic_K_evaluation(encodings),
        shared_representation_statistics=shared_representation_statistics(encodings),
        train_sigma=plan['train_sigma'], normalization=plan['normalization'],
        pair_diagnostics=pair_diagnostics, common_flow_bounds=bounds,
        arms={arm: dict(roots=analyses[arm]['roots'], closure=analyses[arm]['closure'],
            label_free_metrics=measured[arm], critical=analyses[arm]['critical'],
            posthoc_K_windows=analyses[arm]['local_response']['critical_windows'],
            checkpoint=str(loaded[arm].checkpoint), checkpoint_sha256=file_fingerprint(loaded[arm].checkpoint))
            for arm in ARMS})
    (args.output / 'comparison.json').write_text(json.dumps(json_value(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()), extra_R_updates=plan['updates'],
                         root_counts={arm: len(analyses[arm]['roots']) for arm in ARMS}), ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
