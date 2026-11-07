"""JSON/NPZ-only Chinese report for the continuous ensemble-VAE experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_dual_ising_current import plot_flow
from scripts.report_dual_ising_q_vae import polynomial_features
from scripts.report_phi4_hopping_qcr import _setup_plot
from scripts.run_dual_ising_current import write_json


def parameter_readout(train, test, ytrain, ytest):
    center = train.mean(0); scale = np.maximum(train.std(0), 1e-8)
    a = polynomial_features((train-center)/scale); b = polynomial_features((test-center)/scale)
    penalty = np.eye(a.shape[1])*1e-3; penalty[0, 0] = 0
    prediction = b@np.linalg.solve(a.T@a+penalty, a.T@ytrain)
    error = ((prediction-ytest)**2).sum(0); variance = ((ytest-ytest.mean(0))**2).sum(0)
    return [float(1-e/v) if v > 0 else None for e, v in zip(error, variance)]


def scale_diagnostics(a, reconstruction, design=None):
    labels, g = a['labels'], a['direct_g']
    train = a['splits'] == 'train'; test = ~train
    cells, inverse = np.unique(labels, axis=0, return_inverse=True)
    means = np.stack([g[inverse == i].mean(0) for i in range(len(cells))])
    counts = [int((inverse == i).sum()) for i in range(len(cells))]
    if min(counts) < 2: raise ValueError('At least two physical chains per cell required')
    design_cells = cells
    parameter_names = ['r', 'u']
    if design is not None:
        lookup = {tuple(x['parameters']): x['design_coordinates'] for x in design['cells']}
        design_cells = np.array([lookup[tuple(x)] for x in cells])
        parameter_names = ['r-r_0(u)', 'u']
    r, u = np.unique(design_cells[:, 0]), np.unique(design_cells[:, 1])
    if len(cells) != len(r)*len(u): raise ValueError('Full post-hoc design grid required')
    order = np.lexsort((design_cells[:, 1], design_cells[:, 0]))
    if len(np.unique(design_cells, axis=0)) != len(cells): raise ValueError('Duplicate design coordinates')
    rows = []
    for k in range(3):
        grid = means[order, k].reshape(len(r), len(u), 2)
        variance = grid.var((0, 1)); denom = np.maximum(variance, 1e-30)
        effects = np.column_stack([grid.mean(1).var(0)/denom, grid.mean(0).var(0)/denom])
        eig = np.linalg.eigvalsh(np.cov(means[:, k].T, bias=True))[::-1]
        residuals = np.stack([g[inverse == i, k].var(0, ddof=1)/counts[i] for i in range(len(cells))])
        row = dict(scale=k, sites=[36, 12, 4][k], parameter_main_effects=effects.tolist(),
            main_effect_parameter_names=parameter_names,
            interaction_fraction=np.maximum(0, 1-effects.sum(1)).tolist(),
            PCA_variance_fractions=(eig/max(eig.sum(), 1e-30)).tolist(),
            estimated_cell_mean_noise_by_coordinate=residuals.mean(0).tolist(),
            heldout_Spearman_g_parameters=[[float(spearmanr(g[test, k, j], labels[test, i]).statistic)
                if g[test, k, j].std() > 0 and labels[test, i].std() > 0 else None for i in range(2)] for j in range(2)],
            heldout_cubic_parameter_R2=parameter_readout(g[train, k], g[test, k], labels[train], labels[test]),
            reconstruction={})
        pred, target = reconstruction['decoded'][:, :, k], reconstruction['targets'][:, :, k]
        center = target[:, train].mean((0, 1))
        for name, selector in [('train', train), ('test', test)]:
            mse = float(((pred[:, selector]-target[:, selector])**2).sum(-1).mean())
            baseline = float(((center-target[:, selector])**2).sum(-1).mean())
            row['reconstruction'][name] = dict(sum_bin_mse=mse, constant_TRAIN_mean_mse=baseline,
                improvement=1-mse/baseline if baseline else None)
        rows.append(row)
    return rows, cells, means, counts


def report(root):
    root = Path(root); out = root/'report'; out.mkdir(exist_ok=False)
    summary = json.loads((root/'summary.json').read_text())
    if summary['status'] != 'complete': raise ValueError('Completed evaluation required')
    a = dict(np.load(root/'evaluation_arrays.npz', allow_pickle=False))
    reconstruction = dict(np.load(root/'reconstruction_arrays.npz', allow_pickle=False))
    roots = json.loads((root/'roots.json').read_text())
    history = json.loads((root/'history_progress.json').read_text())
    plt = _setup_plot(None)
    style = plot_flow(a, roots['roots'], out, plt, parameter_names=('r', 'u'), model_title=r'二维 $\phi^4$',
                     parameter_ranges=[(float(a['labels'][:, j].min()), float(a['labels'][:, j].max())) for j in range(2)])
    config = json.loads((root/'config.json').read_text())
    design_path = Path(config['data']['root'])/'sampling_design.json'
    design = json.loads(design_path.read_text()) if design_path.exists() else None
    diagnostics, cells, means, counts = scale_diagnostics(a, reconstruction, design)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for j in range(2):
        for k in range(3):
            sc = axes[j, k].scatter(*cells.T, c=means[:, k, j], s=90, marker='s', cmap='viridis')
            axes[j, k].set(xlabel='$r$', ylabel='$u$', title=f'k={k}，$g_{j+1}$')
            fig.colorbar(sc, ax=axes[j, k], label=f'$g_{j+1}$')
    fig.suptitle(f'参数到潜变量：每个参数点 {min(counts)}–{max(counts)} 条物理链均值；不填补未采样区域')
    fig.savefig(out/'coupling_to_g.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for k, ax in enumerate(axes):
        v = np.array(diagnostics[k]['parameter_main_effects'])
        im = ax.imshow(v, vmin=0, vmax=1, cmap='Blues')
        for j in range(2):
            for i in range(2): ax.text(i, j, f'{v[j, i]:.3f}', ha='center', va='center')
        ax.set(title=f'k={k}', xticks=[0, 1], xticklabels=[f'${x}$' for x in diagnostics[k]['main_effect_parameter_names']], yticks=[0, 1], yticklabels=['$g_1$', '$g_2$'])
    fig.colorbar(im, ax=axes, label='参数主效应 / 编码总方差')
    fig.suptitle('事后设计坐标主效应；带状扫描时沿带方向同时改变 r/u' if design else '事后参数分离诊断；潜坐标未经旋转或监督校正')
    fig.savefig(out/'parameter_main_effects.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for k in range(3): ax.bar(np.arange(2)+(k-1)*.23, diagnostics[k]['PCA_variance_fractions'], width=.23, label=f'k={k}')
    ax.set(xticks=[0, 1], xticklabels=['第一主成分', '第二主成分'], ylabel='编码方差占比', title='各尺度的主成分分布')
    ax.legend(); fig.savefig(out/'pca_variance.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    groups = [('Q 表示目标', [('q_total', '交叉熵＋KL'), ('vae_KL_level_0', '微观后验 KL')]),
        ('C 连续粗粒化目标', [('c_information', '信息代理'), ('c_variance', '方差下限'), ('c_uniform', '均匀幅度保持')]),
        ('R 闭合诊断', [('closure_all_start', '全起点锚定'), ('closure_one_step', '单步'), ('closure_rollout', '微观起点多步')])]
    for ax, (title, terms) in zip(axes, groups):
        for split, ls, label in [('train', '-', '训练'), ('test', '--', '留出')]:
            for key, name in terms:
                ax.plot([h['step'] for h in history], [h[split][key] for h in history], ls, label=name+'（'+label+'）')
        ax.set(title=title, xlabel='联合训练更新次数', ylabel='目标／诊断值'); ax.set_yscale('symlog', linthresh=1e-5)
        ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.savefig(out/'loss_curves.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    rows = summary['terminal']['history']
    ax.plot([r['extra_R_updates'] for r in rows], [r['train_loss'] for r in rows])
    ax.set(xlabel='额外 R 更新次数', ylabel='固定训练缓存闭环损失', yscale='log',
           title='冻结 Q/C 后拟合 R：'+('训练平台停止' if summary['terminal']['train_plateau_reached'] else '达到更新上限'))
    ax.grid(alpha=.15); fig.savefig(out/'terminal_R_loss.png', dpi=170); plt.close(fig)
    summary.update(scales=diagnostics, root_count=len(roots['roots']), flow=roots['search'], plot_style=style,
        parameter_chains=counts, limitations=[
            '连续 C 使用高斯核、平滑信息代理和幅度保持误差，不等于离散经验 MI／保色 NLL。',
            '625 个软局域概率是有限描述，不是完整连续场概率密度或充分统计量。',
            '四次子集/C 重复不是独立物理链；原采样警告保留，不证明平衡。',
            '主成分方差和参数主效应不证明可识别性；无监督坐标不必逐轴对应 r/u。',
            '数值不动点、满秩和低闭环误差均不认证真实相边界、Wilson–Fisher 点或临界指数。',
            '有限尺寸、有限局域基、单训练种子；粗尺度标签仍为原始 r/u。'])
    write_json(out/'report.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True)
    report(p.parse_args().run)
