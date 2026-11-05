"""从公共编码数组重绘 g(K)，不加载模型、不修改历史报告。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.report_ising_zh import chinese_figures


KC = .4406867935
SIZES = (108, 54, 27)
COLORS = ('#0072B2', '#D55E00', '#009E73')
PURPLE = '#9C27B0'
PANELS = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


def summarize(path, dimension):
    with np.load(path, allow_pickle=False) as arrays:
        train, test = arrays['train'], arrays['test']
        train_k, test_k = arrays['train_labels'], arrays['test_labels']
    if train.shape != (516, 3, dimension) or test.shape != (252, 3, dimension):
        raise ValueError('公共编码的样本数、尺度数或维数不符')
    if set(train_k).intersection(test_k):
        raise ValueError('原训练/测试整 K 划分发生重叠')
    values = np.concatenate((train, test))
    labels = np.concatenate((train_k, test_k))
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError('编码或标签存在非有限值')
    knots = np.unique(labels)
    groups = [values[labels == k] for k in knots]
    if len(knots) != 64 or any(len(group) != 12 for group in groups):
        raise ValueError('预期每个 K 恰有12个独立系综，共64个 K')
    critical = np.flatnonzero(abs(knots-KC) < 1e-9)
    if len(critical) != 1:
        raise ValueError('必须有唯一已保存的 Kc 数据点，不用插值伪造')
    return dict(dimension=dimension, K=knots,
        mean=np.stack([g.mean(0) for g in groups]),
        sd=np.stack([g.std(0, ddof=1) for g in groups]),
        count=np.array([len(g) for g in groups]),
        critical_index=int(critical[0]), train_K=np.unique(train_k),
        test_K=np.unique(test_k), source=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def draw(rows, destination, *, zoom=False):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    fig.subplots_adjust(left=.065, right=.985, bottom=.13, top=.81,
                        wspace=.27, hspace=.43)
    low, high = (KC-.035, KC+.035) if zoom else (.2, .68)
    for ax, (dimension, coordinate) in zip(axes.flat, PANELS):
        row = rows[dimension]
        x, means, deviations = row['K'], row['mean'], row['sd']
        ic = row['critical_index']
        for k, (size, color) in enumerate(zip(SIZES, COLORS)):
            y, sd = means[:, k, coordinate], deviations[:, k, coordinate]
            ax.fill_between(x, y-sd, y+sd, color=color, alpha=.13, lw=0)
            ax.plot(x, y, color=color, lw=1.8, marker='o',
                    markersize=3.6 if zoom else 2.2)
            ax.scatter(x[ic], y[ic], marker='D', s=58, c=PURPLE,
                       edgecolors='white', linewidths=.7, zorder=6)
        ax.axvline(KC, color=PURPLE, ls=':', lw=1.25, alpha=.85)
        ax.set(xlim=(low, high), xlabel='初始耦合 K',
               ylabel=f'编码坐标 g{coordinate+1}',
               title=f'潜变量维数 = {dimension}：坐标 g{coordinate+1}')
        if zoom:
            inside = (x >= low) & (x <= high)
            envelope = np.concatenate(((means[inside, :, coordinate]-deviations[inside, :, coordinate]).ravel(),
                                       (means[inside, :, coordinate]+deviations[inside, :, coordinate]).ravel()))
            pad = max(float(np.ptp(envelope))*.09, 1e-6)
            ax.set_ylim(float(envelope.min()-pad), float(envelope.max()+pad))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([0], [0], color=color, lw=2,
        label=f'k={k}，尺寸 {size}×{size}')
        for k, (size, color) in enumerate(zip(SIZES, COLORS))]
    handles.append(Line2D([0], [0], color=PURPLE, lw=0, marker='D',
        markersize=7, label='原始 Kc 输入的编码（非不动点）'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .895),
               ncol=4, frameon=False, fontsize=11)
    title = '近临界区域放大' if zoom else '完整耦合范围'
    fig.suptitle(f'完全无监督二维伊辛模型：g 随初始 K 的变化｜{title}',
                 fontsize=18, y=.978)
    fig.text(.5, .93, '三组模型均训练12000次；2×2粗粒化；108→54→27；种子91；零场',
             ha='center', fontsize=12)
    fig.text(.065, .067,
        '实线及圆点：每个 K 的12个独立系综均值；阴影：±1个系综标准差（不是置信区间）。', fontsize=11)
    fig.text(.065, .031,
        '横轴始终是原始微观 K，包含训练集与测试集；不同模型坐标未对齐，面板间 g 数值不可直接等同。', fontsize=10)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=175, facecolor='white')
        with destination.open('xb') as stream:
            stream.write(buffer.getvalue())
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output_dir/name for name in
        ('g_vs_K.png', 'g_vs_K_critical_zoom.png', 'g_vs_K_data.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('拒绝覆盖已有重绘输出')
    rows = {d:summarize(args.root/f'dim{d}/report/encodings.npz', d) for d in (1, 2, 3)}
    if not all(np.array_equal(rows[1]['K'], rows[d]['K']) for d in (2, 3)):
        raise ValueError('三模型的原始 K 网格不一致')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with chinese_figures():
        draw(rows, paths[0])
        draw(rows, paths[1], zoom=True)
    payload = dict(Kc_reference=KC, physical_sampling=False, model_inference=False,
        smoothing_or_fitting=False, independent_ensembles_per_K=12,
        band='12独立系综样本标准差；非置信区间',
        aggregation='按原始K分别求均值，不池化物理构型；仅重绘已保存的Q编码。',
        arms=list(rows.values()))
    with paths[2].open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  default=lambda value:value.tolist() if isinstance(value, np.ndarray) else value,
                  allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', outputs=[str(p.resolve()) for p in paths]),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
