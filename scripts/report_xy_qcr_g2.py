"""Array-only Chinese diagnostics for temperature-only, two-coordinate XY QCR."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import spearmanr
from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_loss, _plot_all_start_closure


def variance_ratios(v):
    return v/v.sum() if v.sum() > 0 else np.zeros_like(v)


def correlation(x, y):
    # Collapse is a reportable outcome, not a JSON failure or a fabricated zero correlation.
    return float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else None


def representation_metrics(a):
    result = {}
    for split in ('train', 'test'):
        mask = a['splits'] == split
        error = a['reconstructed_features'][mask]-a['target_features'][mask]
        result[split] = dict(feature_rmse_by_scale=np.sqrt(np.mean(error**2, axis=(0, 2))).tolist(),
            joint_probability_squared_l2_by_scale=np.mean(np.sum(error**2, axis=-1), axis=0).tolist())
    return result


def geometry_statistics(a):
    g = a['direct_g']; train = a['splits'] == 'train'; cells = a['cell_ids']
    inputs = g[train, :2].reshape(-1, 2)
    mean = inputs.mean(0)
    _, s, components = np.linalg.svd(inputs-mean, full_matrices=False)
    var = s*s/len(inputs)
    result = dict(pca_fit='TRAIN inputs k=0,1; no labels in fitted axes', mean=mean.tolist(),
        components=components.tolist(), pooled_input_explained_variance_ratio=variance_ratios(var).tolist(),
        condition_mean_pca=[], physical_dimension_certified=False,
        caveat='A curved one-parameter manifold may have two nonzero PCs; four chains only')
    for k in range(3):
        means, mean_noise = [], []
        for cell in np.unique(cells[train]):
            z = g[train & (cells == cell), k]
            means.append(z.mean(0))
            mean_noise.append(np.cov(z, rowvar=False, ddof=1)/len(z))
        means = np.asarray(means); centered = means-means.mean(0)
        _, ss, vv = np.linalg.svd(centered, full_matrices=False)
        variances = ss**2/len(means)
        noise = np.mean(mean_noise, axis=0)
        result['condition_mean_pca'].append(dict(scale=k,
            between_temperature_mean_variance=variances.tolist(),
            explained_variance_ratio=variance_ratios(variances).tolist(),
            mean_sampling_noise_projected=np.diag(vv@noise@vv.T).tolist(),
            note='Observed variance and estimated chain-mean noise; not a rank certification'))
    return result


def plot_flow(a, roots, out, plt):
    """Use saved vectors only: equal arrow lengths, linear magnitude in gray."""
    out = Path(out)
    g = a['direct_g']; temp = a['labels'][:, 0]; temperatures = np.unique(temp)
    train = a['splits'] == 'train'
    grid = a['flow_g']; delta = a['flow_displacement']; bounds = a['flow_bounds']
    norm = np.linalg.norm(delta, axis=1)
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list('flow_gray', ['#eeeeee', '#333333'])
    magnitude_max = float(norm.max())
    shade = Normalize(vmin=0., vmax=magnitude_max if magnitude_max > 0 else 1.)
    display = delta/np.maximum(norm[:, None], 1e-15)*.65*min(np.diff(bounds, axis=1)[:, 0])/32
    for k in range(3):
        fig, ax = plt.subplots(figsize=(7.8, 7.0), layout='compressed')
        arrows = ax.quiver(grid[:, 0], grid[:, 1], display[:, 0], display[:, 1], norm,
                  angles='xy', scale_units='xy', scale=1, cmap=cmap, norm=shade, width=.0025, zorder=1)
        for split, marker in [('train', 'o'), ('test', '^')]:
            m = a['splits'] == split
            sc = ax.scatter(g[m, k, 0], g[m, k, 1], c=temp[m], vmin=temperatures.min(),
                vmax=temperatures.max(), cmap='viridis', s=35, marker=marker,
                edgecolors='white', linewidths=.45, zorder=3,
                label=('训练' if split=='train' else '留出')+'温度的独立物理链编码')
        # Hull is only a coverage outline; it does not turn sparse data into full physical support.
        try:
            hull = ConvexHull(g[train, k]); points = g[train, k][hull.vertices]
            ax.plot(*np.vstack([points, points[0]]).T, ls='--', lw=1, c='#555555', label='该尺度训练点凸包（非物理支撑认证）')
            eq = hull.equations
        except QhullError:
            eq = None
        for i, rr in enumerate(roots['roots'], 1):
            p = np.asarray(rr['g']); inside = eq is not None and np.all(eq[:, :2]@p+eq[:, 2] <= 1e-10)
            ax.scatter(*p, marker='*', s=110, color='#c23d45', zorder=5)
            ax.annotate(f'数值根 {i}'+('（外推）' if not inside else ''), p,
                        xytext=(4, 7), textcoords='offset points', fontsize=8, color='#a32f37')
        fig.colorbar(sc, ax=ax, label='生成数据时的原始温度 T/J')
        fig.colorbar(arrows, ax=ax, orientation='horizontal', fraction=.05, pad=.09,
                     label=r'流动幅度 $\|R(g)-g\|_2$（线性灰度：越小越浅）')
        ax.set(xlabel='$g_1$', ylabel='$g_2$', aspect='equal',
               title=f'二维 XY：k={k} 编码与 R(g)−g 方向'+('\n此尺度仅作目标，流为额外外推诊断' if k==2 else ''))
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=.12)
        ax.set_title(ax.get_title()+'\n箭头等长，越浅表示流动越弱；不动点为数值候选', fontsize=12)
        fig.savefig(out/f'flow_k{k}.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    return dict(quantity='Euclidean norm of R(g)-g, not a Jacobian or loss gradient',
                mapping='linear grayscale', vmin=0., vmax=magnitude_max,
                arrow_length='normalized', shared_across_scales=True)


def render(root):
    root = Path(root); out = root/'report'; out.mkdir(exist_ok=False)
    with np.load(root/'evaluation_arrays.npz', allow_pickle=False) as z:
        a = {k:z[k].copy() for k in z.files}
    if a['direct_g'].shape != (60, 3, 2): raise ValueError('Expected 60 chains, three scales, dg=2')
    if any(not np.isfinite(v).all() for v in a.values() if v.dtype.kind in 'fc'):
        raise ValueError('Nonfinite public evaluation arrays')
    summary = json.loads((root/'summary.json').read_text())
    roots = json.loads((root/'roots.json').read_text())
    history = json.loads((root/'history.json').read_text())
    metadata = json.loads((root/'data/metadata.json').read_text())
    reference = json.loads((root/'reference_scalar.json').read_text())
    plt = _setup_plot(None)
    g = a['direct_g']; temp = a['labels'][:, 0]; temperatures = np.unique(temp)
    train = a['splits'] == 'train'
    colors = ['#277da8', '#e49a30', '#20856f']
    geo = geometry_statistics(a)
    pc = (g-np.asarray(geo['mean'])) @ np.asarray(geo['components']).T
    correlations = {split:[[correlation(temp[a['splits']==split], g[a['splits']==split, k, j])
                   for j in range(2)] for k in range(3)] for split in ('train', 'test')}
    # Direct coordinate plots, not a fitted physical inverse of the latent plane.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for j, ax in enumerate(axes):
        for k in range(3):
            means = [g[temp==t, k, j].mean() for t in temperatures]
            se = [g[temp==t, k, j].std(ddof=1)/np.sqrt(np.sum(temp==t)) for t in temperatures]
            ax.errorbar(temperatures, means, yerr=se, fmt='o-', ms=4, color=colors[k], label=f'k={k}')
        ax.axvline(.89294, ls=':', color='gray', label='文献温度参考，非拟合')
        ax.set(xlabel='原始温度 T/J', ylabel=f'潜变量 $g_{j+1}$', title=f'第 {j+1} 个坐标：四条物理链均值与标准误')
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.savefig(out/'temperature_g.png', dpi=170); plt.close(fig)

    flow_style = plot_flow(a, roots, out, plt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    ratios = np.asarray([v['explained_variance_ratio'] for v in geo['condition_mean_pca']])
    for j in range(2):
        axes[0].bar(np.arange(3)+(j-.5)*.3, ratios[:, j], width=.3, label=f'主成分 {j+1}')
    axes[0].set(xticks=np.arange(3), xticklabels=['k=0', 'k=1', 'k=2'], ylim=(0, 1.05),
                ylabel='方差占比', title='各尺度训练温度均值的主成分占比')
    axes[0].legend(); axes[0].grid(alpha=.15, axis='y')
    for k in range(3):
        means = [pc[temp==t, k, 1].mean() for t in temperatures]
        se = [pc[temp==t, k, 1].std(ddof=1)/2 for t in temperatures]
        axes[1].errorbar(temperatures, means, yerr=se, fmt='o-', ms=4, color=colors[k], label=f'k={k}')
    axes[1].set(xlabel='原始温度 T/J', ylabel='第二主成分坐标（共同训练输入轴）', title='第二方向的温度与尺度变化；误差为链均值标准误')
    axes[1].legend(); axes[1].grid(alpha=.15)
    fig.suptitle('第二主成分可能来自曲率或采样噪声，不等于第二个独立物理耦合', fontsize=11)
    fig.savefig(out/'latent_geometry.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout='constrained')
    drift_summary = []
    for k in range(2):
        scale = np.sqrt(np.var(g[train, k], axis=0).sum())
        drift = np.linalg.norm(a['one_step_g'][:, k]-g[:, k], axis=1)/max(scale, 1e-15)
        means = [drift[temp==t].mean() for t in temperatures]
        se = [drift[temp==t].std(ddof=1)/2 for t in temperatures]
        ax.errorbar(temperatures, means, yerr=se, fmt='o-', label=f'输入 k={k}', color=colors[k])
        drift_summary.append(dict(input_scale=k, temperature=temperatures.tolist(), mean=means, standard_error=se))
    ax.set(xlabel='原始温度 T/J', ylabel='位移模长 / 该尺度训练编码总标准差', title='真实数据点的流动幅度（四条链均值）')
    ax.legend(); ax.grid(alpha=.15); fig.savefig(out/'flow_magnitude.png', dpi=170); plt.close(fig)
    _plot_loss(history, out, plt, uniform_label='均匀角度保持')
    _plot_all_start_closure(summary['closure'], out, plt)
    result = dict(status='complete', actual_steps=summary['actual_steps'], theory_dim=2,
        closure=summary['closure'], representation=representation_metrics(a), geometry=geo, flow_style=flow_style,
        temperature_g_spearman=correlations, normalized_flow_magnitude=drift_summary, roots=roots,
        information_diagnostic=summary['information_diagnostic'], sampling_warnings=metadata['warnings'],
        scalar_reference=reference, C_final_exactly_matches_scalar=summary['C_final_exactly_matches_scalar'],
        BKT_certified=False, fixed_line_certified=False, physical_dimension_certified=False,
        critical_temperature_estimate=None, thermal_nu=None,
        limitations=['Single physical size, one training seed and only two b3 transitions.',
            'A temperature curve can bend in 2D; PCA rank does not certify two physical scaling fields.',
            'Only four physical chains per temperature; no equilibration or two-direction identifiability certification.',
            'Coordinates differ from dg=1; raw latent RMSE cannot be directly compared as a physical error.',
            'Arrow lengths are normalized; linear grayscale shows ||R(g)-g|| with a shared scale. This is not a Jacobian or loss gradient.',
            'No temperature/stiffness/vortex supervision; numerical roots are not a BKT certification.'])
    (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result
