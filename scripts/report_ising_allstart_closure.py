"""单种子 b=2、四尺度闭合消融的冻结 CPU 报告。

只读取已完成的两个 12000 步终点及既有物理缓存。五次固定 C 抽样不是
五条独立物理链；不重估 MI、不训练、不补物理采样、不搜索五维不动点。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from scripts import report_ising_highdim as original
from scripts.report_ising_b0_multiseed import (
    TrainingMatchedBinaryCollector, encoding_array_metadata, encoding_metadata,
    file_stat, grouped_path_metrics, path_errors, sampling_quality,
)
from scripts.report_ising_highdim_fast import (
    AnalysisBudgetHold, Deadline, Progress, _assert_frozen, atomic_json,
    atomic_npz, single_thread_control,
)
from scripts.report_ising_zh import chinese_figures


C_SEEDS = (12345, 0, 1, 2, 3)
ARMS = ('control', 'anchor')
ARM_LABELS = dict(control='原闭合目标', anchor='同终点全起点锚定')
SIZES = (32, 16, 8, 4)
CHECKPOINT = 12000


def fit_train_pca(direct):
    """只用 TRAIN 全尺度编码，方向不随 TEST 或理论临界点调整。"""
    direct = np.asarray(direct, dtype=np.float64)
    if direct.ndim != 3 or direct.shape[1:] != (4, 5) or not np.isfinite(direct).all():
        raise ValueError('需要 [TRAIN物理链,4尺度,5坐标] 的有限编码')
    values = direct.reshape(-1, direct.shape[-1])
    center = values.mean(axis=0)
    _, singular, basis = np.linalg.svd(values - center, full_matrices=False)
    # A deterministic orientation, not an orientation chosen with K labels.
    for vector in basis:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    variance = singular ** 2 / max(1, len(values) - 1)
    total = variance.sum()
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    return dict(center=center, components=basis, variance=variance,
                variance_ratio=ratio, fitted_to='本臂 TRAIN、固定基准 C 抽样、全部四个尺度；不使用 K 标签')


def project(points, pca):
    return (np.asarray(points) - pca['center']) @ pca['components'].T


def module_fingerprint(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def module_comparison(first, second):
    a, b = first.state_dict(), second.state_dict()
    if a.keys() != b.keys() or any(a[key].shape != b[key].shape for key in a):
        raise ValueError('对照臂模块结构不同')
    differences = [float((a[key] - b[key]).abs().max()) for key in a if a[key].numel()]
    maximum = max(differences, default=0.)
    return dict(exact_equal=all(torch.equal(a[key], b[key]) for key in a),
                max_abs_difference=maximum, control_sha256=module_fingerprint(first),
                anchor_sha256=module_fingerprint(second))


def comparable_configs(first, second):
    """闭合目标以外不得静默改变数据、模型、梯度、种子或训练长度。"""
    a, b = copy.deepcopy(first), copy.deepcopy(second)
    objectives = (a['loss'].pop('closure_objective', 'one_step_and_rollout'),
                  b['loss'].pop('closure_objective', 'one_step_and_rollout'))
    if objectives != ('one_step_and_rollout', 'all_start_anchor') or a != b:
        raise ValueError('两臂必须只改变 closure_objective，分别为旧目标与 all_start_anchor')
    return objectives


def validate_run(loaded, cfg, manifest, summary, arm):
    original.validate_completed_run(cfg, summary, loaded.payload, CHECKPOINT)
    if (cfg.schema_version != 5 or cfg.model.dimension != 2 or cfg.model.theory_dim != 5
            or cfg.model.block_factor != 2 or cfg.training.max_rg_steps != 3
            or cfg.evaluation.max_rg_steps != 3 or cfg.evaluation.samples != 128
            or cfg.training.sample_batch != 128 or cfg.training.seed != 91
            or cfg.training.gradient_mode != 'qc_r_separate'
            or set(cfg.training.train_modules) != {'Q', 'C', 'R'} or cfg.model.fixed_kernel
            or cfg.model.kernel_initialization != 'random'
            or cfg.model.kernel_sampling != 'bernoulli'
            or cfg.model.kernel_layout != 'nonoverlapping'
            or cfg.model.categorical_states or cfg.data.spin_flip_augment
            or tuple(cfg.data.coupling_names) != ('K',)
            or cfg.loss.weights.commutator != 5 or cfg.loss.rollout_weight != 5):
        raise ValueError('报告固定 schema5、seed91、d_g=5、b2、32→16→8→4、新随机无监督 QCR')
    expected = 'one_step_and_rollout' if arm == 'control' else 'all_start_anchor'
    if cfg.loss.closure_objective != expected:
        raise ValueError('闭合目标与臂名称不匹配')
    weights = cfg.loss.closure_scale_weights
    if weights is not None and (len(weights) != 3 or len(set(weights)) != 1 or weights[0] <= 0):
        raise ValueError('报告要求三个终点均匀加权')
    metadata = loaded.payload.get('metadata', {})
    for record in (manifest, metadata):
        if (record.get('labels_used_in_training') is not False
                or record.get('q_supervision') not in (None, {})
                or record.get('q_future_physics') is not None
                or record.get('source') is not None or record.get('resumed_from') is not None
                or record.get('frozen_C') or record.get('data_driven_initialization')):
            raise ValueError('不允许监督、续训、冻结 C 或数据驱动初始化混入本次消融')
    if (manifest.get('rsmi_weight') != 2 or not manifest.get('independent_c')
            or manifest.get('estimator') != 'histogram' or manifest.get('environment_mode') != 'full_shell'
            or manifest.get('buffer') != 1 or not cfg.model.affine_reconstruction
            or cfg.loss.weights.separation != 10 or cfg.loss.weights.anchor != 2
            or cfg.loss.weights.coarse_consistency != 5):
        raise ValueError('要求默认经验全壳层 MI、独立梯度以及共享仿射 Q 表示目标')
    if (not manifest.get('data_sha256')
            or original.json_value(manifest.get('config')) != original.json_value(cfg.to_dict())):
        raise ValueError('缺少数据 SHA 或 manifest 配置不一致')


def history_for(loaded, run):
    history = loaded.payload.get('history')
    if not history:
        history = json.loads((Path(run) / 'history_progress.json').read_text())
    if not history or history[-1].get('epoch') != CHECKPOINT:
        raise ValueError('冻结历史必须到达实际 12000 步')
    return history


def _coupling_means(values, couplings):
    couplings = np.asarray(couplings).reshape(-1)
    knots = np.unique(couplings)
    return knots, np.stack([np.mean(values[couplings == k], axis=0) for k in knots])


def plot_results(output, result, collected, pcas, couplings, histories, deadline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with chinese_figures():
        deadline.check()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        for arm, ax in zip(ARMS, axes):
            ratio = pcas[arm]['variance_ratio']
            ax.bar(np.arange(1, 6), ratio * 100)
            ax.plot(np.arange(1, 6), np.cumsum(ratio) * 100, 'o-', color='C1', label='累计占比')
            ax.set(title=ARM_LABELS[arm], xlabel='主成分编号', ylabel='方差占比（%）', xticks=range(1, 6))
            ax.legend(); ax.grid(alpha=.2)
        fig.suptitle('仅用训练集四尺度编码确定主成分；两臂分别拟合')
        fig.savefig(output / 'principal_components.png', dpi=160); plt.close(fig)

        deadline.check()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
        for column, arm in enumerate(ARMS):
            pca = pcas[arm]
            direct = collected[arm]['test'][0]['direct']
            mapped = collected[arm]['test_R_of_g'][0]
            x = project(direct[:, :-1], pca)[..., 0]
            y = project(mapped, pca)[..., 0]
            observed = project(direct[:, 1:], pca)[..., 0]
            ax = axes[0, column]
            for scale in range(3):
                ax.scatter(x[:, scale], y[:, scale], s=13, alpha=.5, color=f'C{scale}', label=f'R：k={scale}')
                ax.scatter(x[:, scale], observed[:, scale], s=17, marker='x', alpha=.35,
                           color=f'C{scale}', label=f'直接粗粒化：k={scale}→{scale+1}')
            limits = (min(x.min(), y.min(), observed.min()), max(x.max(), y.max(), observed.max()))
            ax.plot(limits, limits, '--', color='black', lw=.8, label='恒等映射')
            ax.set(title=ARM_LABELS[arm] + '：数据点上的 R(g)',
                   xlabel='输入的第一主成分投影', ylabel='输出的第一主成分投影')
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
            ax = axes[1, column]
            # Every arrow is a projection of an actual full-dimensional R(g)-g.
            # Vertical offsets separate the three scales; they are not coordinates.
            ids = np.linspace(0, len(x)-1, min(40, len(x))).astype(int)
            for scale in range(3):
                base = np.full(len(ids), float(scale))
                ax.quiver(x[ids, scale], base + .12, y[ids, scale] - x[ids, scale],
                          np.zeros(len(ids)), angles='xy', scale_units='xy', scale=1,
                          color=f'C{scale}', width=.003, label='R(g)-g' if scale == 0 else None)
                ax.quiver(x[ids, scale], base - .12, observed[ids, scale] - x[ids, scale],
                          np.zeros(len(ids)), angles='xy', scale_units='xy', scale=1,
                          color='grey', alpha=.5, width=.002,
                          label='直接粗粒化位移' if scale == 0 else None)
            ax.set(title='真实位移的第一主成分投影（箭头未归一化）',
                   xlabel='第一主成分投影', ylabel='起点尺度 k（上下错位仅为显示）',
                   yticks=range(3), ylim=(-.5, 2.5))
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle('仅绘数据支持点；投影零位移不等于五维不动点，不作临界指数认证')
        fig.savefig(output / 'R_flow_pc1.png', dpi=160); plt.close(fig)

        deadline.check()
        fig, axes = plt.subplots(2, 5, figsize=(21, 8), layout='constrained')
        for row, arm in enumerate(ARMS):
            direct = np.mean([draw['direct'] for draw in collected[arm]['test']], axis=0)
            knots, mean = _coupling_means(direct, couplings['test'])
            for component, ax in enumerate(axes[row]):
                for scale, size in enumerate(SIZES):
                    ax.plot(knots, mean[:, scale, component], label=f'k={scale}，尺寸{size}')
                ax.set(title=f'{ARM_LABELS[arm]}：g{component+1}', xlabel='生成与事后评价用 K', ylabel=f'g{component+1}')
                ax.grid(alpha=.2)
                if component == 0: ax.legend(fontsize=7)
        fig.suptitle('测试集 K→g；先在物理链内平均五次 C 抽样，再按相同 K 平均')
        fig.savefig(output / 'coupling_to_g.png', dpi=150); plt.close(fig)

        deadline.check()
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
        fields = (('commutator', '共同诊断：单步闭合（未加权）'),
                  ('rollout', '共同诊断：微观起点多步闭合（未加权）'),
                  ('closure_total', '各自实际加权 R 目标（目标不同）'),
                  ('separation', 'Q 重构与表示目标'),
                  ('kernel_noncollapse', 'C 逐系综方差下限'),
                  ('rsmi_bits', '既有经验直方图 MI（比特）'))
        for ax, (field, title) in zip(axes.flat, fields):
            for arm in ARMS:
                for split, style in (('train', '-'), ('test', '--')):
                    rows = []
                    for row in histories[arm]:
                        value = row.get(f'{split}_{field}')
                        if field == 'closure_total' and value is None:
                            comm, rollout = row.get(f'{split}_commutator'), row.get(f'{split}_rollout')
                            if comm is not None and rollout is not None: value = 5 * comm + 5 * rollout
                        if value is not None: rows.append((row['epoch'], value))
                    if rows:
                        ax.plot(*np.asarray(rows).T, style, label=ARM_LABELS[arm] + ('：训练集' if split == 'train' else '：测试集'))
            ax.set(title=title, xlabel='训练更新次数', ylabel='损失或已保存诊断')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.suptitle('损失曲线仅读取训练历史；报告阶段不重估 MI')
        fig.savefig(output / 'loss_curves.png', dpi=160); plt.close(fig)

        deadline.check()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
        for arm in ARMS:
            paths = result['arms'][arm]['closure']['test']['paths']
            axes[0].plot(range(6), [p['R']['nrmse'] for p in paths], 'o-', label=ARM_LABELS[arm])
            axes[1].plot(range(6), [p['R_to_identity_rmse_ratio'] for p in paths], 'o-', label=ARM_LABELS[arm])
        labels = [f'{row["start"]}→{row["end"]}' for row in paths]
        for ax in axes:
            ax.set_xticks(range(6), labels); ax.set_xlabel('完整尺度路径'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        axes[0].set(title='六条路径的测试集归一误差', ylabel='本臂原训练目标标准差归一均方根误差')
        axes[1].set(title='与配对恒等基准比较', ylabel='R 的均方根误差／恒等映射均方根误差')
        axes[1].axhline(1., color='black', ls='--', lw=.8)
        fig.suptitle('五次 C 抽样先在物理链内平均平方误差；1→3 为新增受约束路径')
        fig.savefig(output / 'closure_comparison.png', dpi=160); plt.close(fig)


def run_report(args, deadline, progress):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖已有正式或部分报告')
    output.mkdir(parents=True, exist_ok=False)
    loaded, cfgs, manifests, histories, states, collected, pcas = {}, {}, {}, {}, {}, {}, {}
    arrays, data_stats = {}, {}
    try:
        for arm, run in (('control', args.run_control), ('anchor', args.run_anchor)):
            progress('读取冻结终点：' + ARM_LABELS[arm], 0, 1)
            loaded[arm] = load_run(run, CHECKPOINT)
            cfgs[arm] = RunConfig.from_dict(loaded[arm].config)
            cfgs[arm].validate()
            manifests[arm] = json.loads((Path(run) / 'manifest.json').read_text())
            summary = json.loads((Path(run) / 'summary.json').read_text())
            validate_run(loaded[arm], cfgs[arm], manifests[arm], summary, arm)
            states[arm] = (original.sha256(loaded[arm].checkpoint), original.model_state_sha256(loaded[arm].model))
            _assert_frozen(loaded[arm], *states[arm])
            histories[arm] = history_for(loaded[arm], run)
        comparable_configs(*(cfgs[arm].to_dict() for arm in ARMS))
        if len({manifests[arm]['data_sha256'] for arm in ARMS}) != 1:
            raise ValueError('两臂数据指纹不同')
        initial = [manifests[arm].get('training_start_module_sha256') for arm in ARMS]
        if not initial[0] or initial[0] != initial[1]:
            raise ValueError('两臂必须具有同一实际训练起点 Q/C/R 指纹')
        cfg = cfgs['control']
        data_path = Path(cfg.data.cache)
        data_stats[str(data_path)] = file_stat(data_path)
        sidecar_path = data_path.with_suffix('.json')
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        train, test = original.report_datasets(cfg)
        datasets = dict(train=train, test=test)
        couplings = {split: dataset.couplings.detach().cpu().numpy().reshape(-1)
                     for split, dataset in datasets.items()}
        if any(tuple(dataset.configurations.shape[1:]) != (128, 32, 32, 1) for dataset in datasets.values()):
            raise ValueError('本次报告只接受原生 L32、每物理链128构型；不裁剪或重标记粗格点 K')
        for arm in ARMS:
            collector = TrainingMatchedBinaryCollector(loaded[arm].model, samples=128, deadline=deadline,
                progress=lambda stage, n, total, extra, arm=arm: progress(ARM_LABELS[arm] + '：' + stage, n, total, extra))
            collected[arm] = {}
            for split, dataset in datasets.items():
                collected[arm][split], collected[arm][split + '_R_of_g'] = [], []
                for draw, seed in enumerate(C_SEEDS):
                    deadline.check()
                    progress(ARM_LABELS[arm] + '：冻结编码', draw, len(C_SEEDS), dict(split=split, C_seed=seed))
                    values = collector.collect(dataset.configurations, samples=128, max_rg_steps=3,
                        spin_flip_augment=False, chunk_size=16, sampling_seed=seed)
                    direct = values['direct']
                    mapped = original.mapped(loaded[arm].model, direct[:, :-1].reshape(-1, 5)).reshape(-1, 3, 5)
                    collected[arm][split].append(values)
                    collected[arm][split + '_R_of_g'].append(mapped)
                arrays[arm + '__' + split + '__direct'] = np.stack([v['direct'] for v in collected[arm][split]])
                arrays[arm + '__' + split + '__R_of_g'] = np.stack(collected[arm][split + '_R_of_g'])
            pcas[arm] = fit_train_pca(collected[arm]['train'][0]['direct'])
            for key in ('center', 'components', 'variance', 'variance_ratio'):
                arrays[arm + '__pca_' + key] = pcas[arm][key]
            _assert_frozen(loaded[arm], *states[arm])
        qc = dict(Q=module_comparison(loaded['control'].model.theory_encoder, loaded['anchor'].model.theory_encoder),
                  C=module_comparison(loaded['control'].model.rg_kernel, loaded['anchor'].model.rg_kernel),
                  R=module_comparison(loaded['control'].model.theory_rg, loaded['anchor'].model.theory_rg))
        result = dict(schema_version=1, experiment='ising2d_B0_allstart_closure_b2_L32_dg5_seed91',
            status='complete', report_complete=True, training_seed=91, actual_updates_per_arm=CHECKPOINT,
            lattice_sizes=list(SIZES), block_factor=2, theory_dim=5, labels_used_in_training=False,
            data_sha256=manifests['control']['data_sha256'],
            training_start_module_sha256=initial[0], terminal_module_comparison=qc,
            original_latent_coordinates_identical=qc['Q']['exact_equal'] and qc['C']['exact_equal'],
            C_seeds=list(C_SEEDS), physical_chain_counts={k: len(v.configurations) for k, v in datasets.items()},
            physical_sampling_quality=sampling_quality(sidecar), thermal_nu=None,
            fixed_point_search=dict(performed=False, reason='本轮只比较闭合目标；不将一维投影过零解释为完整五维不动点。'),
            loss_definition=dict(control='(5/3) sum_t e[t-1,t] + (5/3) sum_t e[0,t]',
                anchor='(10/3) sum_t (1/t) sum_(s<t) e[s,t]',
                e_definition='R^(t-s)(g[s]) 与 g[t] 的逐物理系综平均平方误差；坐标归约依照保存配置。',
                coordinate_reduction=cfg.loss.coordinate_reduction,
                evaluation_RMSE_definition='报告的 RMSE 始终先对物理链及坐标取均方再开方；与训练配置的坐标求和目标明确分开。',
                t=[1, 2, 3], target_total_weight=[10/3]*3, prediction_prediction_term=False,
                newly_constrained_path=[1, 3]),
            arms={}, limitations=[
                '一个优化种子、一个物理缓存；不评估训练种子或物理采样种子不确定性。',
                '末端4×4格点有显著有限尺寸可能；本轮不认证普适热指数或临界点。',
                '主成分只对 TRAIN 拟合，丢弃方向可能仍包含流；投影相交或零位移不等于五维不动点。',
                '均方根误差和主成分比例依赖潜变量坐标；两臂分别报告原 TRAIN 目标标准差归一误差。',
                '五次条件 C 抽样不是五条独立物理链；既有侧车诊断不构成平衡认证。',
                '图上 R(g) 在实际编码点逐点计算；未把 R(mean g) 当成 mean R(g)，不绘离流形规则网格。'],
            analysis=dict(max_seconds=deadline.limit, CPU_threads=1, physical_sampling_started=False,
                training_started=False, MI_estimated=False, raw_physical_data_rehashed=False), **encoding_metadata())
        for arm in ARMS:
            closure = {}
            reference = collected[arm]['train'][0]['direct']
            for split in datasets:
                errors = [path_errors(loaded[arm].model, values) for values in collected[arm][split]]
                closure[split] = grouped_path_metrics(errors, reference)
                arrays[arm + '__' + split + '__R_squared_error'] = np.stack([v['R'] for v in errors])
                arrays[arm + '__' + split + '__identity_squared_error'] = np.stack([v['identity'] for v in errors])
            result['arms'][arm] = dict(label=ARM_LABELS[arm], run=str(getattr(args, 'run_' + arm)),
                closure=closure, newly_constrained_test_path_1_to_3=next(
                    row for row in closure['test']['paths'] if (row['start'], row['end']) == (1, 3)),
                PCA=pcas[arm], freeze_verification=_assert_frozen(loaded[arm], *states[arm]))
        paths = np.asarray(path_errors(loaded['control'].model, collected['control']['test'][0])['paths'])
        arrays.update(paths=paths, C_seeds=np.asarray(C_SEEDS), lattice_sizes=np.asarray(SIZES),
                      train_K=couplings['train'], test_K=couplings['test'], **encoding_array_metadata())
        atomic_npz(output / 'collected_arrays.npz', **arrays)
        atomic_json(output / 'history.json', histories)
        progress('生成中文比较图', 0, 1)
        plot_results(output, result, collected, pcas, couplings, histories, deadline)
        for arm in ARMS:
            result['arms'][arm]['freeze_verification'] = _assert_frozen(loaded[arm], *states[arm])
        if data_stats != {path: file_stat(path) for path in data_stats}:
            raise ValueError('物理缓存 stat 在报告过程中发生变化')
        result['immutable_data_stat'] = data_stats
        result['analysis']['elapsed_seconds'] = deadline.elapsed
        deadline.check()
        atomic_json(output / 'comparison.json', result)
        public = sorted(p.name for p in output.iterdir() if p.is_file() and p.suffix in ('.json', '.npz', '.png'))
        public = sorted(set(public + ['report.json', 'status.json']))
        completion = dict(status='complete', report_complete=True, actual_updates_per_arm=CHECKPOINT,
            public_files=public, thermal_nu=None, analysis=result['analysis'],
            checkpoint_frozen=True, model_gradients_none=True, **encoding_metadata())
        atomic_json(output / 'report.json', completion)
        atomic_json(output / 'status.json', completion)
        progress('正式报告完成', 1, 1)
        return result
    except Exception as error:
        held = isinstance(error, AnalysisBudgetHold)
        if arrays:
            atomic_npz(output / 'partial_arrays.npz', **arrays)
        freeze = {arm: _assert_frozen(value, *states[arm]) for arm, value in loaded.items() if arm in states}
        record = dict(status='analysis_budget_hold' if held else 'failed', report_complete=False,
            requires_user_action=True, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=deadline.elapsed, max_seconds=deadline.limit, freeze_verification=freeze,
            training_started=False, physical_sampling_started=False, MI_estimated=False,
            thermal_nu=None, latest_stage=progress.latest)
        atomic_json(output / 'report.json', record); atomic_json(output / 'status.json', record)
        if held:
            return record
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-control', '--control-run', dest='run_control', type=Path, required=True)
    parser.add_argument('--run-anchor', '--anchor-run', dest='run_anchor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=1200.)
    parser.add_argument('--progress-path', type=Path)
    args = parser.parse_args(argv)
    if args.threads != 1:
        parser.error('冻结报告固定 CPU 单线程')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('拒绝覆盖已有正式或部分报告')
    args.progress_path = args.progress_path or args.output.with_name(args.output.name + '.progress.json')
    if args.progress_path.exists() or args.progress_path == args.output or args.output in args.progress_path.parents:
        parser.error('进度路径必须为输出目录之外的新文件')
    torch.set_num_threads(1)
    deadline = Deadline(args.max_seconds)
    with single_thread_control():
        return run_report(args, deadline, Progress(args.progress_path, deadline))


if __name__ == '__main__':
    result = main()
    if result.get('status') == 'analysis_budget_hold':
        raise SystemExit(75)
