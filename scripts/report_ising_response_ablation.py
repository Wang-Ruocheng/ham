"""固定末点的 R 局部响应消融报告；只读取保存编码，不运行 Q、C 或 MI。"""
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
from recurrent_rg.evaluation import closure_metrics, scalar_flow
from recurrent_rg.runner import file_fingerprint
from scripts.report_ising_zh import chinese_figures
from scripts.report_ising_scalar_exponents import scalar_coupling_means, write_exponent_report
from scripts.report_k_supervised_continuation import response_from_encodings


ARMS = ('control', 'response')
LABELS = dict(control='仅闭合对照', response='增加响应损失')
COLORS = dict(control='#df861f', response='#2574ad')


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def load_encodings(report):
    path = report / 'encodings.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key], dtype=np.float64).copy()
                  for key in ('train', 'test', 'train_labels', 'test_labels')}
    scalar_coupling_means(arrays['train'], arrays['test'], arrays['train_labels'], arrays['test_labels'])
    if arrays['train'].shape[1:] != (3, 1):
        raise ValueError('Expected saved scalar b2 depth2 encodings, including all three scales')
    train_k = np.unique(arrays['train_labels'])
    test_k = np.unique(arrays['test_labels'])
    if np.intersect1d(train_k, test_k).size:
        raise ValueError('TRAIN and TEST K knots must remain disjoint')
    return arrays


def frozen_state_matches(source, candidate):
    if set(source) != set(candidate):
        return False
    return all(torch.equal(source[key], candidate[key]) for key in source
               if not key.startswith('theory_rg.'))


def validate_fixed_final(plan, source, loaded, summaries, root):
    """固定预算和相同源状态；拒绝把额外 R 更新伪称完整 QCR 续训。"""
    if source.payload.get('epoch') != plan['source_epoch']:
        raise ValueError('Source epoch differs from the preregistered plan')
    if file_fingerprint(source.checkpoint) != plan['source_checkpoint_sha256']:
        raise ValueError('Source checkpoint fingerprint changed')
    if (source.model_config.dimension != 2 or source.model_config.theory_dim != 1 or
            source.model_config.block_factor != 2 or source.model_config.categorical_states):
        raise ValueError('Requires the original binary scalar b2 architecture')
    source_state = source.payload['model']
    for arm in ARMS:
        run, summary = loaded[arm], summaries[arm]
        meta = run.payload.get('metadata', {})
        if (summary['actual_extra_R_updates'] != plan['updates'] or
                run.payload.get('epoch') != plan['source_epoch'] or
                meta.get('extra_R_updates') != plan['updates'] or
                meta.get('R_only_ablation') is not True or meta.get('exact_resume') is not False):
            raise ValueError('Each arm must be its fixed-budget R-only final endpoint')
        if run.model_config != source.model_config or not frozen_state_matches(source_state, run.payload['model']):
            raise ValueError('Q/C or the checkpoint architecture changed')
        initial = torch.load(root / arm / 'training_step_0.pt', map_location='cpu', weights_only=True)
        if (set(initial['model']) != set(source_state) or
                any(not torch.equal(initial['model'][key], source_state[key]) for key in source_state)):
            raise ValueError('Both arms must start from the identical complete source Q/C/R state')
        if meta.get('arm') != arm or meta.get('alpha') != (0. if arm == 'control' else plan['alpha']):
            raise ValueError('Control and response objective weights differ from the plan')
        if (meta.get('source_checkpoint_sha256') != plan['source_checkpoint_sha256'] or
                meta.get('encodings_sha256') != plan['encodings_sha256']):
            raise ValueError('Both arms must use the identical source checkpoint and saved encodings')


def predictions(model, direct):
    """R-only predictions against frozen saved targets; no encoder calls."""
    direct = np.asarray(direct, dtype=np.float64)
    x = torch.from_numpy(direct)
    with torch.no_grad():
        one = model.advance_theory(x[:, :-1]).cpu().numpy()
        chain = [x[:, 0]]
        for _ in range(direct.shape[1] - 1):
            chain.append(model.advance_theory(chain[-1]))
    return dict(direct=direct, one_step=one,
                rollout=torch.stack(chain, 1).cpu().numpy())


def training_response_metrics(models, encodings, plan):
    """沿用训练 D7 定义；TEST 仅评价，始终使用 TRAIN 的尺度归一量。"""
    from experiments.ising_response_loss_ablation import prepare_response_data, metrics
    prepared = {split: prepare_response_data(
        torch.from_numpy(encodings[split]), torch.from_numpy(encodings[split + '_labels']), neighbors=7)
        for split in ('train', 'test')}
    normalization = torch.as_tensor(plan['normalization'], dtype=torch.float64)
    expected = prepared['train'].target_derivative.square().mean(0).clamp_min(1e-12)
    if not torch.allclose(normalization, expected, rtol=1e-12, atol=1e-14):
        raise ValueError('Response normalizers must be derived exclusively from the saved TRAIN targets')
    result = {}
    for arm, model in models.items():
        result[arm] = {split: json_value(metrics(model, data, normalization,
                            alpha=0. if arm == 'control' else plan['alpha']))
                       for split, data in prepared.items()}
    return result, {split: json_value(data.diagnostics) for split, data in prepared.items()}


def shared_representation_statistics(encodings):
    result = {}
    for split in ('train', 'test'):
        values = np.asarray(encodings[split], dtype=np.float64)
        variances = values[:, :, 0].var(0)
        result[split] = dict(ensembles=len(values), mean=values[:, :, 0].mean(0).tolist(),
            std=np.sqrt(variances).tolist(), covariance_eigenvalues=[[float(value)] for value in variances],
            effective_rank=[1 if value > 0 else 0 for value in variances],
            convention='固定保存编码的系综总体方差；标量 g 的非零协方差有效秩为 1，零方差为 0。')
    return result


def analyze_arm(loaded, encodings, bounds, output, extra_updates, *, bootstraps=1000):
    output.mkdir(parents=True, exist_ok=False)
    model = loaded.model.cpu().double().eval()
    flow = scalar_flow(model, *bounds)
    fixed = dict(roots=flow['roots'], search_box=[[bounds[0]], [bounds[1]]],
        search_note=flow['search_note'], convention='完整离散 R 的雅可比；不对位移 R(g)-g 求指数')
    metrics = dict(fixed_points=fixed, closure={})
    for split in ('train', 'test'):
        metrics['closure'][split] = closure_metrics(
            predictions(model, encodings[split]), reference_direct=encodings['train'])
    (output / 'fixed_points.json').write_text(json.dumps(fixed, indent=2, allow_nan=False) + '\n')
    exponents = write_exponent_report(loaded, metrics, fixed, encodings, output)
    inputs = encodings['train'][:, :-1, 0].ravel()
    means = np.asarray(exponents['coupling_means']['g_by_scale'])
    for root in exponents['roots']:
        point = root['g'][0]
        root['nearest_training_input_distance'] = float(np.min(abs(inputs-point)))
        root['distance_to_mean_encoding_curve_by_scale'] = [
            float(max(means[:, k].min()-point, point-means[:, k].max(), 0.)) for k in range(3)]
    exponents.update(source_QCR_epoch=loaded.payload['epoch'], extra_R_updates=extra_updates)
    exponents['selection']['epoch_selection'] = '预先固定的相同额外 R 更新预算末点；不按 ν 选择检查点或根。'
    exponents['epoch_clock_note'] = 'checkpoint_epoch 是源 QCR 更新数；extra_R_updates 单独计数，二者不合并。'
    (output / 'critical_exponents.json').write_text(json.dumps(exponents, indent=2, allow_nan=False) + '\n')
    response = response_from_encodings(model, encodings, loaded.payload['epoch'], bootstraps=bootstraps)
    response['extra_R_updates'] = extra_updates
    response['scope'] = '此三窗口分析使用保存的 TRAIN+TEST 标签作事后诊断；训练响应项只使用 TRAIN。'
    (output / 'local_response.json').write_text(json.dumps(response, indent=2, allow_nan=False) + '\n')
    return dict(roots=exponents['roots'], flow=dict(g=flow['grid'].tolist(), R=flow['mapped'].tolist()),
        training_input_range=exponents['training_input_range'], critical=exponents['critical_reference'],
        closure=metrics['closure'], local_response=response,
        source_QCR_epoch=loaded.payload['epoch'], extra_R_updates=extra_updates)


def draw_maps(analyses, models, encodings, source, output):
    knots, means = scalar_coupling_means(encodings['train'], encodings['test'],
                                       encodings['train_labels'], encodings['test_labels'])
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    low, high = analyses['control']['training_input_range']
    for arm in ARMS:
        result, color = analyses[arm], COLORS[arm]
        x, y = np.asarray(result['flow']['g']), np.asarray(result['flow']['R'])
        axes[0].plot(x, y, color=color, label=LABELS[arm])
        axes[1].plot(x, y-x, color=color, label=LABELS[arm])
        for root in result['roots']:
            point = root['g'][0]
            axes[0].scatter(point, point, marker='*', c=color, edgecolors='black', s=90, zorder=5)
            axes[1].scatter(point, 0., marker='*', c=color, edgecolors='black', s=90, zorder=5)
        critical_g = np.asarray(result['critical']['g_by_scale'])
        with torch.no_grad():
            critical_r = models[arm].advance_theory(torch.from_numpy(critical_g[:, None])).numpy()[:, 0]
        for panel, ordinate in ((0, critical_r), (1, critical_r-critical_g)):
            axes[panel].scatter(critical_g, ordinate, marker='D', c='magenta',
                edgecolors=color, s=46, zorder=7,
                label='Kc 前向编码（非不动点）' if arm == 'control' else None)
    x = np.asarray(analyses['control']['flow']['g'])
    with torch.no_grad():
        original = source.advance_theory(torch.from_numpy(x[:, None])).numpy()[:, 0]
    axes[0].plot(x, original, color='gray', ls=':', lw=1., label='共同初始 R')
    axes[1].plot(x, original-x, color='gray', ls=':', lw=1., label='共同初始 R')
    axes[0].plot(x, x, 'k--', lw=.8, label='恒等映射')
    axes[1].axhline(0., color='black', ls='--', lw=.8)
    for ax in axes[:2]:
        ax.axvspan(low, high, color='green', alpha=.07, label='R 的训练输入范围')
    axes[0].set(title='固定末点：完整 R(g)', xlabel='g', ylabel='R(g)')
    axes[1].set(title='固定末点：离散位移与全部已找到的根', xlabel='g', ylabel='R(g)-g')
    for k in range(means.shape[1]):
        axes[2].plot(knots, means[:, k], label=f'k={k}')
    reference = analyses['control']['critical']
    axes[2].axvline(reference['Kc'], color='magenta', ls=':', label='已知微观输入 Kc')
    axes[2].scatter([reference['Kc']]*3, reference['g_by_scale'], marker='D', c='magenta', s=45, zorder=6)
    for k, value in enumerate(reference['g_by_scale']):
        axes[2].annotate(f'k={k}', (reference['Kc'], value),
            xytext=((12, 24), (12, -24), (34, 0))[k], textcoords='offset points', color='magenta',
            arrowprops=dict(arrowstyle='-', color='magenta', lw=.6))
    axes[2].set(title='原保存编码：两组共用且完全不变', xlabel='初始 K=J/T，B=0', ylabel='g')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle(f'固定 Q、C：两组各增加 {analyses["control"]["extra_R_updates"]} 次 R 更新；不按 ν 选择')
    fig.supxlabel('紫色菱形是 Kc 输入的前向编码，不是所选不动点；末尺度 k=2 无下一尺度目标。根搜索有界且可能遗漏切触根。', fontsize=9)
    fig.savefig(output / 'R_maps_comparison.png', dpi=170)
    plt.close(fig)


def draw_losses(histories, output):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), layout='constrained')
    for row, split in enumerate(('train', 'test')):
        split_name = '训练集' if split == 'train' else '测试集（仅评价）'
        for column, (key, title) in enumerate((('closure', '加权单步与多步闭合'),
                ('response_mse', '原量纲局部响应均方误差'),
                ('response_normalized', '按训练目标归一的响应损失'), ('total', '实际优化目标总损失'))):
            ax = axes[row, column]
            for arm in ARMS:
                history = histories[arm]
                ax.plot([point['extra_R_updates'] for point in history],
                        [point[split+'_'+key] for point in history],
                        color=COLORS[arm], label=LABELS[arm])
            ax.set(title=split_name+'：'+title, xlabel='额外 R 更新次数', ylabel='损失')
            ax.set_yscale('symlog', linthresh=1e-10)
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('固定末点预算：闭合与响应分别列出；测试损失不参与训练或末点选择')
    fig.savefig(output / 'loss_curves.png', dpi=150)
    plt.close(fig)


def draw_response_comparison(analyses, metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for scale, ax in enumerate(axes):
        target_drawn = False
        for arm in ARMS:
            widths, predicted, observed = [], [], []
            for window in analyses[arm]['local_response']['critical_windows']:
                if window['status'] != 'complete':
                    continue
                point = window['transitions'][scale]
                widths.append(window['half_width']); predicted.append(point['predicted_response'])
                observed.append(point['empirical_response'])
                interval = point['predicted_response_bootstrap_95']
                ax.vlines(window['half_width'], *interval, color=COLORS[arm], alpha=.5)
            ax.plot(widths, predicted, 'o-', color=COLORS[arm], label=LABELS[arm])
            if not target_drawn:
                ax.plot(widths, observed, 'k--', label='保存编码的数据响应')
                target_drawn = True
        ax.set(title=f'尺度 k={scale}→{scale+1}：Kc 邻域事后诊断',
               xlabel='K 窗口半宽', ylabel='局部响应比', xticks=[.02, .03, .04])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.supxlabel('使用保存的全部 K 标签评价；竖线为单模型、单次保存 C 抽样条件下的系综自助抽样区间，非热乘子置信区间。', fontsize=9)
    fig.savefig(output / 'window_response_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    categories = [('train', 'boundary', '训练边界'), ('train', 'interior', '训练内部'),
                  ('test', 'boundary', '测试边界'), ('test', 'interior', '测试内部')]
    for ax, suffix, title in zip(axes, ('mse', 'normalized'), ('原量纲响应误差', '按训练目标归一的响应误差')):
        for index, arm in enumerate(ARMS):
            values = [metrics[arm][split]['response_'+scope+'_'+suffix] for split, scope, _ in categories]
            ax.bar(np.arange(4) + (index-.5)*.34, values, width=.34,
                   color=COLORS[arm], label=LABELS[arm])
        ax.set(title=title, ylabel='均方误差', xticks=np.arange(4),
               xticklabels=[item[2] for item in categories])
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
    fig.supxlabel('边界与内部按训练时相同的七点局部三次微分定义；测试仅评价，归一量始终来自训练目标。', fontsize=9)
    fig.savefig(output / 'boundary_interior_response.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='包含 plan.json 和 control/response 的核心实验目录')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.bootstraps < 1:
        raise ValueError('Positive bootstrap count required')
    torch.set_num_threads(1)
    plan = json.loads((args.run / 'plan.json').read_text())
    source_report = Path(plan['source_report'])
    encoded_sha = file_fingerprint(source_report / 'encodings.npz')
    if encoded_sha != plan['encodings_sha256']:
        raise ValueError('Saved source encodings changed')
    encodings = load_encodings(source_report)
    source = load_run(Path(plan['source_checkpoint']))
    loaded = {arm: load_run(args.run / arm, 'checkpoint_last.pt') for arm in ARMS}
    summaries = {arm: json.loads((args.run / arm / 'summary.json').read_text()) for arm in ARMS}
    validate_fixed_final(plan, source, loaded, summaries, args.run)
    models = {arm: value.model.cpu().double().eval() for arm, value in loaded.items()}
    measured, preprocessing = training_response_metrics(models, encodings, plan)
    all_values = np.concatenate((encodings['train'], encodings['test']))
    pad = max(.2, .2 * float(np.ptp(all_values)))
    bounds = [float(all_values.min()-pad), float(all_values.max()+pad)]
    histories = {arm: json.loads((args.run / arm / 'history_progress.json').read_text()) for arm in ARMS}
    if any(history[-1]['extra_R_updates'] != plan['updates'] for history in histories.values()):
        raise ValueError('Loss histories must include the identical fixed final update')
    args.output.mkdir(parents=True, exist_ok=False)
    with chinese_figures():
        analyses = {arm: analyze_arm(loaded[arm], encodings, bounds, args.output / arm,
                        plan['updates'], bootstraps=args.bootstraps) for arm in ARMS}
        draw_maps(analyses, models, encodings, source.model.cpu().double().eval(), args.output)
        draw_losses(histories, args.output)
        draw_response_comparison(analyses, measured, args.output)
    if file_fingerprint(source_report / 'encodings.npz') != encoded_sha:
        raise RuntimeError('Source encodings changed during reporting; preserve incomplete outputs')
    comparison = dict(status='complete', source_QCR_epoch=plan['source_epoch'],
        extra_R_updates_per_arm=plan['updates'], alpha=plan['alpha'],
        source_checkpoint=str(source.checkpoint), source_checkpoint_sha256=plan['source_checkpoint_sha256'],
        encodings_reference=str((source_report / 'encodings.npz').resolve()), encodings_sha256=encoded_sha,
        shared_initial_Q_C_R_verified=True, frozen_Q_C_verified=True,
        Q_C_inference=False, physical_sampling=False, MI_calculation=False,
        selection='固定相同更新预算末点；不按 TEST、理论值、ν 或根位置筛选。',
        conditioning='全部比较以同一源模型保存的单次 C 抽样编码为条件；不重新推断 Q/C，不代表独立训练种子或 C 重抽样不确定性。',
        response_definition='训练为各 TRAIN K 等权的七点局部三次微分，逐尺度按 TRAIN 目标导数均方归一；TEST 仅评价。三窗口曲线另作事后路径诊断，不认证热乘子。',
        common_flow_bounds=bounds, preprocessing=preprocessing,
        shared_representation_statistics=shared_representation_statistics(encodings),
        arms={arm: dict(roots=analyses[arm]['roots'], closure=analyses[arm]['closure'],
            response_metrics=measured[arm], critical=analyses[arm]['critical'],
            critical_windows=analyses[arm]['local_response']['critical_windows'],
            checkpoint=str(loaded[arm].checkpoint), checkpoint_sha256=file_fingerprint(loaded[arm].checkpoint))
            for arm in ARMS})
    (args.output / 'comparison.json').write_text(json.dumps(json_value(comparison), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output.resolve()), extra_R_updates=plan['updates'],
                          root_counts={arm: len(analyses[arm]['roots']) for arm in ARMS}), ensure_ascii=False))
    return comparison


if __name__ == '__main__':
    main()
