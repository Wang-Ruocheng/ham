"""只读已完成公共报告，绘制多 seed 的主成分占比、映射与真实位移。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plot_ising_primary_components import (
    _style_font, prepare_data, sha, write_plots,
)


SCALE_COLORS = ('#2474b5', '#e78b22', '#26966b')
SEED_COLORS = ('#2867ad', '#e69535', '#329873', '#9c6cba', '#cf6064')


def panel_data(data):
    if data['selected'].tolist() != [0]:
        raise ValueError('本次一维主成分总览只适用于按阈值仅选择 PC1 的报告')
    x, y = data['x'][..., 0], data['y'][..., 0]
    if x.shape != y.shape or x.ndim != 2 or not np.all(np.isfinite(y - x)):
        raise ValueError('主成分投影数组不一致')
    roots = []
    for root in data['roots']:
        position = float((np.asarray(root['g']) - data['center']) @ data['components'][0])
        roots.append(dict(root_id=root['root_id'], position=position,
                          stability=root['stability'],
                          physical_support_certified=root['physical_support_certified']))
    return dict(seed=int(data['metrics']['training_seed']), x=x, y=y, delta=y-x,
                fractions=data['fractions'], roots=roots,
                lattice_sizes=data['metrics']['lattice_sizes'])


def arrow_segments(panel, scale, count=9):
    """抽取实际保存点；箭头终点严格为 x+Δ，不平滑、不归一化箭长。"""
    n = len(panel['x'])
    indices = np.unique(np.rint(np.linspace(0, n-1, min(count, n))).astype(int))
    start = panel['x'][indices, scale]
    delta = panel['delta'][indices, scale]
    return indices, start, delta, start + delta


def scale_label(panel, scale):
    if scale == len(panel['lattice_sizes']) - 1:
        return f'k={scale}：{panel["lattice_sizes"][scale]}（输入尺度外推）'
    return f'k={scale}：{panel["lattice_sizes"][scale]}（已训练输入尺度）'


def _limits(panel):
    values = np.concatenate((panel['x'].ravel(), panel['y'].ravel(),
                             np.asarray([r['position'] for r in panel['roots']])))
    low, high = float(values.min()), float(values.max())
    pad = max((high-low)*.07, 1e-6)
    return low-pad, high+pad


def _format(axis):
    axis.grid(alpha=.17)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=10)
    axis.spines[['top', 'right']].set_visible(False)


def _root_marks(axis, panel, *, displacement=False):
    low, high = _limits(panel)
    width = high-low
    for index, root in enumerate(panel['roots']):
        p = root['position']
        axis.scatter([p], [0 if displacement else p], marker='D', s=42,
                     facecolors='none', edgecolors='#707880', linewidths=1.2, zorder=6)
        kind = {'stable': '吸引', 'saddle': '鞍点'}.get(root['stability'], '数值根')
        near_right, near_left = p > high-.27*width, p < low+.22*width
        horizontal, alignment = (-5, 'right') if near_right else (5, 'left')
        vertical = 10 if index % 2 == 0 else -17
        if not displacement and (near_right or near_left):
            vertical = -17 if near_right else 10
        axis.annotate(f'{root["root_id"]}（{kind}）', (p, 0 if displacement else p),
                      xytext=(horizontal, vertical), ha=alignment,
                      textcoords='offset points', fontsize=9, color='#545d67',
                      annotation_clip=True)


def plot_variance(panels, destination, font):
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.2, 6.3),
                                    gridspec_kw={'width_ratios': [1, 1.15]})
    pc1 = [100*p['fractions'][0] for p in panels]
    bars = left.bar(np.arange(len(panels)), pc1, color=SEED_COLORS[:len(panels)], width=.64)
    for bar, value in zip(bars, pc1):
        left.text(bar.get_x()+bar.get_width()/2, value+1.5, f'{value:.4f}%',
                  ha='center', va='bottom', fontsize=11)
    left.set(xticks=np.arange(len(panels)), xticklabels=[str(p['seed']) for p in panels],
             ylim=(0, 110), xlabel='训练种子', ylabel='解释方差占比（%）',
             title='第一主成分：五组均超过 99%')
    width = .145
    positives = []
    for i, panel in enumerate(panels):
        values = 100*panel['fractions'][1:]
        positive = values > 0
        positives.extend(values[positive])
        positions = np.arange(len(values))+(i-(len(panels)-1)/2)*width
        right.bar(positions[positive], values[positive], width=width*.93,
                  color=SEED_COLORS[i % len(SEED_COLORS)], label=f'种子{panel["seed"]}')
    right.set(xticks=np.arange(len(panels[0]['fractions'])-1),
              xticklabels=[f'第{i}成分' for i in range(2, len(panels[0]['fractions'])+1)],
              xlabel='其余主成分', ylabel='解释方差占比（%；对数轴）',
              title='其余成分放大显示，未舍弃弱方向')
    right.set_yscale('log')
    right.set_ylim(min(positives)*.3, max(positives)*5)
    right.legend(fontsize=10, ncol=2, loc='upper right')
    for axis in (left, right):
        _format(axis)
    fig.suptitle('五个训练种子的主成分方差占比', fontsize=17, y=.97)
    fig.text(.5, .035, 'PCA 仅拟合训练集的 R 输入尺度 k=0、1；方差集中不等于物理耦合维数认证。',
             ha='center', fontsize=11, color='#536170')
    fig.subplots_adjust(left=.07, right=.985, bottom=.16, top=.84, wspace=.28)
    _style_font(fig, font)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def plot_maps(panels, destination, font):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 11.2))
    for panel, axis in zip(panels, axes.flat):
        for scale in range(panel['x'].shape[1]):
            axis.plot(panel['x'][:, scale], panel['y'][:, scale],
                      '.--' if scale == 2 else '.-', color=SCALE_COLORS[scale],
                      lw=1.6, ms=3, label=scale_label(panel, scale))
        low, high = _limits(panel)
        axis.plot([low, high], [low, high], '--', color='#6c747d', lw=1.1, label='恒等映射')
        _root_marks(axis, panel)
        axis.set(xlim=(low, high), ylim=(low, high), xlabel='输入第一主成分',
                 ylabel='R 输出的第一主成分',
                 title=f'种子{panel["seed"]} · 第1成分 {100*panel["fractions"][0]:.4f}%')
        axis.set_aspect('equal', adjustable='box')
        _format(axis)
    note = axes.flat[-1]
    note.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    note.legend(handles, labels, loc='upper left', frameon=False, fontsize=11)
    note.text(.03, .55,
              '空心菱形：已保存的全5维数值根投影。\n吸引／鞍点类型来自完整雅可比矩阵。\n所有根均未认证物理支持。\n\n每个种子的坐标、PCA 符号独立；\n不跨种子叠加或平均坐标。',
              transform=note.transAxes, va='top', fontsize=11, linespacing=1.65, color='#536170')
    fig.suptitle('五个训练种子：各自第一主成分上的 R 映射', fontsize=17, y=.98)
    fig.text(.5, .034, '输出为 R(条件平均 g)，不是平均 R(g)；这是完整5维映射的投影，不是独立一维自治模型。',
             ha='center', fontsize=11, color='#536170')
    fig.subplots_adjust(left=.065, right=.98, top=.90, bottom=.10, wspace=.30, hspace=.34)
    _style_font(fig, font)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def plot_flows(panels, destination, font):
    import matplotlib.pyplot as plt
    cols = 3 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels)/cols))
    fig = plt.figure(figsize=(15, 11.6) if len(panels)>1 else (8.8, 7.4))
    outer = fig.add_gridspec(rows, cols, left=.065, right=.98, top=.90, bottom=.12,
                            wspace=.32, hspace=.38)
    for position, panel in enumerate(panels):
        cell = outer[position//cols, position % cols].subgridspec(2, 1, height_ratios=[3, 1], hspace=.09)
        axis = fig.add_subplot(cell[0])
        arrows = fig.add_subplot(cell[1], sharex=axis)
        for scale in range(panel['x'].shape[1]):
            axis.plot(panel['x'][:, scale], panel['delta'][:, scale],
                      '.--' if scale == 2 else '.-', color=SCALE_COLORS[scale],
                      lw=1.6, ms=3, label=scale_label(panel, scale))
            _, start, delta, _ = arrow_segments(panel, scale)
            arrows.quiver(start, np.full(len(start), scale), delta, np.zeros(len(start)),
                          angles='xy', scale_units='xy', scale=1, color=SCALE_COLORS[scale],
                          width=.005, headwidth=3.7, headlength=4.7, headaxislength=4.2,
                          minlength=0)
        axis.axhline(0, color='#6c747d', ls='--', lw=1)
        _root_marks(axis, panel, displacement=True)
        axis.set(ylabel='第一主成分的预测位移',
                 title=f'种子{panel["seed"]} · 第1成分 {100*panel["fractions"][0]:.4f}%')
        axis.tick_params(labelbottom=False)
        arrows.set(xlim=_limits(panel), ylim=(2.55, -.55), yticks=[0, 1, 2],
                   yticklabels=['k=0', 'k=1', 'k=2'], xlabel='输入第一主成分')
        _format(axis)
        _format(arrows)
        arrows.grid(False)
        arrows.spines['left'].set_visible(False)
    if len(panels)>1:
        note = fig.add_subplot(outer[-1, -1])
        note.axis('off')
        handles, labels = fig.axes[0].get_legend_handles_labels()
        note.legend(handles, labels, loc='upper left', frameon=False, fontsize=11)
        note.text(.03, .56,
                  '上图：ΔPC1=P[R(平均g)−平均g]。\n下方三行：实际长度、方向的位移箭头，\n没有归一化或人为放大。\n\n菱形仅定位全维数值根的投影，\n不代表所画条件曲线在此过零。',
                  transform=note.transAxes, va='top', fontsize=11, linespacing=1.65, color='#536170')
    fig.suptitle('各自第一主成分上的真实投影流' if len(panels)>1
                 else f'种子{panels[0]["seed"]}：第一主成分上的真实投影流', fontsize=17, y=.98)
    fig.text(.5, .045, '只使用保存的 R 输出；不重新推理、拟合或求根。不同种子的坐标独立，曲线投影过零不是新不动点。',
             ha='center', fontsize=11, color='#536170')
    _style_font(fig, font)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def write_multiseed_plots(reports_root, output, *, seeds=(91, 92, 93, 94, 95), threshold=.99):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_ising_zh import select_chinese_font
    reports_root, output = Path(reports_root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('使用新目录，保留原正式报告和旧展示图')
    if len(set(seeds)) != len(seeds) or not 1 <= len(seeds) <= 5:
        raise ValueError('需1至5个不重复的训练 seed')
    data = [prepare_data(reports_root/f'seed{s}/report', threshold=threshold) for s in seeds]
    panels = [panel_data(item) for item in data]
    if [p['seed'] for p in panels] != list(seeds):
        raise ValueError('路径与正式训练 seed 不一致')
    output.mkdir(parents=True)
    font = select_chinese_font()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
    for seed, panel in zip(seeds, panels):
        write_plots(reports_root/f'seed{seed}/report', output/f'seed{seed}', threshold=threshold)
        plot_flows([panel], output/f'seed{seed}/主要成分上的真实投影流.png', font)
    plot_variance(panels, output/'五种子主成分方差占比.png', font)
    plot_maps(panels, output/'五种子主要成分R映射.png', font)
    plot_flows(panels, output/'五种子主要成分真实投影流.png', font)
    receipt = dict(
        training_seeds=list(seeds), variance_threshold=threshold,
        source_root=str(reports_root.resolve()), model_loaded=False, inference_performed=False,
        PCA_refitted=False, roots_recomputed=False, formal_report_overwritten=False,
        coordinate_alignment='none; each seed retains its original TRAIN PCA basis and sign',
        arrow_scaling='exact x→x+delta; no normalization or magnification; selected existing points',
        mapping='x=P(mean_g−center); y=P(R_of_mean_g−center); delta=y−x',
        roots='saved full-coordinate numerical roots only; projected positions are not new roots',
        seeds=[dict(seed=p['seed'], explained_variance_fraction=p['fractions'].tolist(),
                    selected_components_zero_based=[0], roots=p['roots'],
                    source_sha256={**item['source_sha256'],
                                   'report.json': sha(reports_root/f'seed{p["seed"]}/report/report.json')})
               for p, item in zip(panels, data)])
    for seed, item in zip(seeds, data):
        for name, fingerprint in item['source_sha256'].items():
            if sha(reports_root/f'seed{seed}/report'/name) != fingerprint:
                raise RuntimeError('绘图期间源文件发生变化')
    (output/'绘图回执.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    receipt = write_multiseed_plots(args.reports_root, args.output)
    print(json.dumps({'training_seeds': receipt['training_seeds'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
