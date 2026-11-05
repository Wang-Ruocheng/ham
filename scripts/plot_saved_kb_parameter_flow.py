"""Replot saved two-dimensional KB flow, coloring real data by input K or B.

Reads public JSON/NPZ only. No model loading, interpolation, training or root
search. Original reports remain immutable; outputs go in a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cell_means(arrays):
    """Average independent chains only within the same physical parameter cell."""
    g, labels = arrays['direct_g'], arrays['labels']
    ids, splits = arrays['cell_ids'], arrays['splits']
    if (g.ndim != 3 or g.shape[-1] != 2 or labels.shape != (len(g), 2)
            or ids.shape != (len(g),) or splits.shape != (len(g),)
            or not np.isfinite(g).all() or not np.isfinite(labels).all()):
        raise ValueError('Expected finite two-dimensional encodings and K/B labels')
    means, parameters, counts = [], [], []
    for cell in np.unique(ids):
        mask = ids == cell
        if len(np.unique(labels[mask], axis=0)) != 1 or len(np.unique(splits[mask])) != 1:
            raise ValueError('Cannot aggregate across parameters or data splits')
        means.append(g[mask].mean(axis=0))
        parameters.append(labels[mask][0])
        counts.append(int(mask.sum()))
    return np.array(means), np.array(parameters), counts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError('Keep display outputs outside the original report directory')
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing figures')
    job = json.loads((source/'job.json').read_text())
    if job['status'] != 'complete' or not job['workflow_complete'] or job['returncode'] != 0:
        raise ValueError('Only completed saved results are supported')
    inputs = ('evaluation_arrays.npz', 'roots.json')
    pins = {name: sha256(source/name) for name in inputs}
    if any(pins[name] != job['public_files'][name] for name in inputs):
        raise ValueError('Public source SHA mismatch')
    with np.load(source/'evaluation_arrays.npz', allow_pickle=False) as z:
        a = {name: z[name] for name in z.files}
    roots = json.loads((source/'roots.json').read_text())['roots']
    means, labels, counts = cell_means(a)
    grid, drift, bounds = a['flow_g'], a['flow_displacement'], a['flow_bounds']
    if (grid.ndim != 2 or grid.shape[1] != 2 or drift.shape != grid.shape
            or bounds.shape != (2, 2) or not np.all(bounds[:, 1] > bounds[:, 0])
            or not all(np.isfinite(v).all() for v in (grid, drift, bounds))
            or not np.allclose(a['flow_R']-grid, drift, rtol=1e-12, atol=1e-12)):
        raise ValueError('Invalid saved R(g)-g field')
    spacing = min(np.min(np.diff(np.unique(grid[:, d]))) for d in range(2))
    arrows = drift / np.maximum(np.linalg.norm(drift, axis=1, keepdims=True), 1e-15) * .68*spacing

    from scripts.report_phi4_hopping_qcr import _setup_plot
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D
    plt = _setup_plot(None)
    output.mkdir(parents=True, exist_ok=False)
    norms = [Normalize(labels[:, 0].min(), labels[:, 0].max()),
             TwoSlopeNorm(vmin=labels[:, 1].min(), vcenter=0, vmax=labels[:, 1].max())]
    names = {'attracting': '吸引点', 'repelling': '排斥点',
             'saddle': '鞍点', 'neutral_or_unresolved': '近中性或未定'}
    figures = []
    for k in range(means.shape[1]):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.6), layout='constrained')
        for d, ax in enumerate(axes):
            ax.quiver(*grid.T, *arrows.T, angles='xy', scale_units='xy', scale=1,
                      color='#7c8792', alpha=.66, width=.0023, zorder=1)
            sc = ax.scatter(*means[:, k].T, c=labels[:, d], s=23,
                            cmap='viridis' if d == 0 else 'coolwarm', norm=norms[d],
                            edgecolors='#30343b', linewidths=.28, zorder=3)
            bar = fig.colorbar(sc, ax=ax, fraction=.046, pad=.025, shrink=.82)
            bar.set_label('原始最近邻耦合 K' if d == 0 else '原始外场 B')
            bar.set_ticks([.20, .32, .44, .56, .68] if d == 0 else [-.01, -.005, 0, .005, .01])
            for i, root in enumerate(roots, 1):
                x, y = root['g']
                ax.scatter(x, y, marker='*', s=180, facecolors='white',
                           edgecolors='black', linewidths=1.2, zorder=5)
                outside = not root['within_train_hull']
                ax.annotate(f"{names[root['classification']]}{i}" + ('（外推）' if outside else ''),
                            (x, y), xytext=(-8, 12), textcoords='offset points',
                            ha='right', fontsize=9, zorder=6,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.5))
            ax.set(xlabel='$g_1$', ylabel='$g_2$', aspect='equal',
                   xlim=bounds[0], ylim=bounds[1],
                   title='按真实 K 着色' if d == 0 else '按真实 B 着色')
            ax.grid(alpha=.12, lw=.5)
            handles = [Line2D([], [], marker='o', ls='none', color='#555555',
                              markersize=5, label='采样参数点的平均编码')]
            if roots:
                handles.append(Line2D([], [], marker='*', ls='none', color='black',
                                      markerfacecolor='white', markersize=10, label='已保存的数值不动点'))
            ax.legend(handles=handles, fontsize=8, loc='lower left', framealpha=.93)
        fig.suptitle(f'二维 Ising 有监督实验：$g_1$–$g_2$ 流场，k={k}\n'
                     '箭头为 R(g)−g 的方向（等长）；散点颜色为生成数据时的实际参数', fontsize=13)
        note = (f'每点为同一参数下两条物理链的平均编码，共{len(means)}点；训练与留出均展示，无插值填色。'
                if set(counts) == {2} else f'共{len(means)}个参数点，各点仅平均同参数物理链；无插值填色。')
        note += '\n数据覆盖外的箭头为外推；星号不代表已认证物理临界点。'
        if k > 0:
            note += ' 颜色是原始 K/B，不是粗粒化后的有效耦合。'
        fig.supxlabel(note, fontsize=8.5)
        name = f'R_flow_by_KB_k{k}.png'
        fig.savefig(output/name, dpi=190)
        plt.close(fig)
        figures.append(name)
    if any(sha256(source/name) != pins[name] for name in inputs):
        raise RuntimeError('Source files changed during plotting')
    receipt = dict(source=str(source), source_sha256=pins, source_unchanged=True,
                   coordinate_array='direct_g (mean of two subset encodings)',
                   point_rule='physical-cell mean, never across labels or splits',
                   cells=len(means), chains_per_cell=counts,
                   color_labels='original K and B; not inferred effective coarse couplings',
                   arrow_rule='saved R(g)-g normalized to equal length',
                   color_normalization=dict(K='linear', B='linear, centered at zero'),
                   data_interpolated=False, model_loaded=False, roots_recomputed=False,
                   saved_roots=roots, figures={name: sha256(output/name) for name in figures})
    (output/'plot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output), ensure_ascii=False, indent=2))
