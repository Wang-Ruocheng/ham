"""Offline g_j(r,u) plots of the completed four-arm Q trial, plus source Q.

No model loading, inference, training, interpolation, or edits to formal results.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

MODELS = ('source', 'warm_base', 'warm_local', 'fresh_base', 'fresh_local')
NAMES = {'source': '原联合训练 Q（6000步）', 'warm_base': '热启动600步／原目标',
         'warm_local': '热启动600步／局部项', 'fresh_base': '随机初始化600步／原目标',
         'fresh_local': '随机初始化600步／局部项'}
SIDES = (36, 12, 4)


def correlation(x, y):
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return {'pearson': None, 'spearman': None}
    return {'pearson': float(np.corrcoef(x, y)[0, 1]),
            'spearman': float(spearmanr(x, y).statistic)}


def group_statistics(g, labels, cells):
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    if any(len(ix) < 2 or not np.all(labels[ix, :2] == labels[ix[0], :2]) for ix in groups):
        raise ValueError('Repeated independent chains and constant cell labels required')
    means = np.array([g[ix].mean(0) for ix in groups])
    sem = np.array([g[ix].std(0, ddof=1)/np.sqrt(len(ix)) for ix in groups])
    params = np.array([labels[ix[0], :2] for ix in groups])
    r, u = np.unique(params[:, 0]), np.unique(params[:, 1])
    if len(groups) != len(r)*len(u) or len(set(map(tuple, params))) != len(groups):
        raise ValueError('Complete rectangular parameter grid required; no interpolation')
    grid = np.empty((len(u), len(r), *means.shape[1:]))
    for i, (rv, uv) in enumerate(params):
        grid[np.flatnonzero(u == uv)[0], np.flatnonzero(r == rv)[0]] = means[i]
    stats = []
    for k in range(g.shape[1]):
        row = {'k': k, 'g1_g2': correlation(means[:, k, 0], means[:, k, 1]), 'components': []}
        for d in range(2):
            component = {'component': d+1, 'r': correlation(means[:, k, d], params[:, 0]),
                         'u': correlation(means[:, k, d], params[:, 1]),
                         'range': [float(means[:, k, d].min()), float(means[:, k, d].max())],
                         'fixed_u': [], 'fixed_r': []}
            for uv in u:
                ix = params[:, 1] == uv
                component['fixed_u'].append({'u': float(uv), **correlation(means[ix, k, d], params[ix, 0])})
            for rv in r:
                ix = params[:, 0] == rv
                component['fixed_r'].append({'r': float(rv), **correlation(means[ix, k, d], params[ix, 1])})
            row['components'].append(component)
        stats.append(row)
    return dict(means=means, sem=sem, params=params, r=r, u=u, grid=grid, statistics=stats)


def edges(x):
    middle = (x[1:]+x[:-1])/2
    return np.r_[x[0]-(x[1]-x[0])/2, middle, x[-1]+(x[-1]-x[-2])/2]


def conditional_figure(plt, grouped, name, scales, output):
    params, means, sem = (grouped[key] for key in ('params', 'means', 'sem'))
    r, u = grouped['r'], grouped['u']
    # Four columns: two output components, varying r; then varying u.
    fig, axes = plt.subplots(len(scales), 4, figsize=(16.5, 3.5*len(scales)+1.2),
                             squeeze=False, layout='constrained')
    colors_u = plt.get_cmap('viridis')(np.linspace(.07, .93, len(u)))
    colors_r = plt.get_cmap('plasma')(np.linspace(.05, .93, len(r)))
    for row, k in enumerate(scales):
        for col in range(4):
            d = col % 2
            variable, fixed = (0, 1) if col < 2 else (1, 0)
            axis_name, fixed_name = ('r', 'u') if col < 2 else ('u', 'r')
            fixed_values, colors = (u, colors_u) if col < 2 else (r, colors_r)
            ax = axes[row, col]
            for value, color in zip(fixed_values, colors):
                ids = np.flatnonzero(params[:, fixed] == value)
                ids = ids[np.argsort(params[ids, variable])]
                ax.errorbar(params[ids, variable], means[ids, k, d], yerr=sem[ids, k, d],
                            fmt='-o', ms=3.3, lw=1.15, elinewidth=.65, capsize=1.5,
                            color=color, label=f'{fixed_name}={value:.2f}')
            ax.set(xlabel=f'扫描参数 ${axis_name}$', ylabel=f'$g_{d+1}$',
                   title=f'k={k}，L={SIDES[k]}：固定 {fixed_name}，改变 {axis_name}')
            ax.grid(alpha=.18)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    hu, lu = axes[0, 0].get_legend_handles_labels()
    hr, lr = axes[0, 2].get_legend_handles_labels()
    fig.legend(hu+hr, lu+lr, loc='outside lower center', ncol=7, fontsize=8,
               title='左两列按 u 分组；右两列按 r 分组', title_fontsize=9)
    fig.suptitle(NAMES[name]+'：参数依赖曲线\n点为4条物理链均值，误差棒为链间标准误；直线仅连接已有采样点', fontsize=13)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Fresh output directory required; preserve previous plots')
    job = json.loads((root/'job.json').read_text())
    if not job['workflow_complete'] or job['returncode'] != 0:
        raise ValueError('Completed formal experiment required')
    npz = root/'evaluation_arrays.npz'
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    if sha != job['public_files']['evaluation_arrays.npz']:
        raise ValueError('Public array SHA mismatch')
    with np.load(npz, allow_pickle=False) as z:
        labels, cells = z['labels'].copy(), z['cell_ids'].copy()
        g = {name: z[name+'_g'].copy() for name in MODELS}
    if not all(x.shape == (180, 3, 2) and np.isfinite(x).all() for x in g.values()):
        raise ValueError('Expected five finite 180-chain, three-scale, two-component arrays')
    grouped = {name: group_statistics(x, labels, cells) for name, x in g.items()}
    output.mkdir(parents=True)
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt = _setup_plot(None)
    for name, data in grouped.items():
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), layout='constrained')
        for k in range(3):
            for d in range(2):
                ax = axes[k, d]
                mesh = ax.pcolormesh(edges(data['r']), edges(data['u']), data['grid'][:, :, k, d],
                                     shading='flat', cmap='viridis', rasterized=True)
                ax.scatter(*data['params'].T, color='black', s=7, alpha=.45)
                ax.set(xlabel='扫描参数 $r$', ylabel='扫描参数 $u$',
                       title=f'k={k}，L={SIDES[k]}：$g_{d+1}(r,u)$')
                fig.colorbar(mesh, ax=ax, label=f'链均值 $g_{d+1}$')
        fig.suptitle(NAMES[name]+'：二维参数到两个潜变量的映射\n每格为已有参数点的4链均值；黑点为采样位置；各子图色标独立，无插值', fontsize=13)
        fig.savefig(output/(name+'_heatmaps.png'), dpi=160)
        plt.close(fig)
        conditional_figure(plt, data, name, range(3), output/(name+'_curves_all_scales.png'))
        conditional_figure(plt, data, name, (0,), output/(name+'_curves_k0.png'))
    fig, axes = plt.subplots(1, 5, figsize=(17, 5.8), layout='constrained')
    for ax, name in zip(axes, MODELS):
        matrix = np.array([[c[p]['pearson'] for p in ('r', 'u')]
                           for row in grouped[name]['statistics'] for c in row['components']])
        mesh = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        for i in range(6):
            for j in range(2):
                value = matrix[i, j]
                ax.text(j, i, f'{value:+.3f}', ha='center', va='center', color='white' if abs(value)>.65 else 'black')
        ax.set(xticks=[0, 1], xticklabels=['参数 r', '参数 u'], yticks=range(6),
               yticklabels=[f'k={k}，$g_{d+1}$' for k in range(3) for d in range(2)], title=NAMES[name])
    fig.colorbar(mesh, ax=axes, label='皮尔逊相关系数')
    fig.suptitle('45个条件均值的边际线性相关；尺度分开计算，不把链重复当独立参数点\n相关系数接近零不排除非单调依赖；不是独立测试泛化指标')
    fig.savefig(output/'pearson_comparison.png', dpi=160)
    plt.close(fig)
    arrays = {'parameters_ru': grouped['source']['params']}
    for name, data in grouped.items():
        arrays[name+'_means'] = data['means']
        arrays[name+'_chain_sem'] = data['sem']
    np.savez_compressed(output/'parameter_means.npz', **arrays)
    report = dict(input=str(npz), input_sha256=sha, models={name: data['statistics'] for name, data in grouped.items()},
                  inference_or_training=False, parameter_labels_used_only_for_evaluation=True,
                  definitions={'g': 'Average of Q on the first and second 32-configuration halves of the fixed64 cache, then mean over4 physical chains',
                               'curves': 'Conditional curves: fix one physical parameter and vary the other; no fitted/interpolated function',
                               'correlation': 'Pearson and Spearman across45 equally weighted condition means, separately at each scale',
                               'errorbar': 'sample standard deviation across4 chain values divided by sqrt(4); not an equilibrium, C-draw, or training-seed uncertainty certificate'},
                  qualifications=['源 Q/C 已固定；四条链不认证平衡，保留历史采样警告。',
                                  '展示整个采样网格，包含 TRAIN、内部验证、TEST；不作为独立测试误差。',
                                  'g 分量依赖潜坐标选取；相关性不能直接认证有效维度或分离的物理耦合方向。'])
    (output/'correlations.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if hashlib.sha256(npz.read_bytes()).hexdigest() != sha:
        raise RuntimeError('Input changed during plotting')
    print(json.dumps({'output': str(output), 'png_count': len(list(output.glob('*.png'))),
                      'input_sha256': sha, 'source_statistics': report['models']['source']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.root, args.output)
