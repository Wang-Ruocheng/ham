"""从既有公共 JSON/NPZ 整理两张主图；不导入模型、不推理、不重新求根。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_components(fractions, threshold=.99):
    values = np.asarray(fractions, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.all(np.isfinite(values))
            or np.any(values < 0) or not 0 < threshold <= 1
            or np.any(np.diff(values) > 1e-12) or values.sum() > 1 + 1e-10):
        raise ValueError('需要按方差降序排列的非负有限占比和有效累计阈值')
    cumulative = np.cumsum(values)
    if cumulative[-1] + 1e-12 < threshold:
        raise ValueError('保存的主成分不足以达到阈值；不能假称覆盖全部主要方向')
    return np.arange(min(int(np.searchsorted(cumulative + 1e-12, threshold)) + 1, len(values)))


def prepare_data(report, *, threshold=.99):
    report = Path(report)
    metrics = json.loads((report / 'metrics.json').read_text())
    completion = json.loads((report / 'report.json').read_text())
    if completion.get('status') != 'complete' or completion.get('report_complete') is not True:
        raise ValueError('仅使用已完成的正式报告')
    required = ['metrics.json', 'fixed_points.json', 'trajectories.npz']
    if not set(required) <= set(completion['public_files']):
        raise ValueError('绘图输入必须在正式报告公共白名单中')
    with np.load(report / 'trajectories.npz', allow_pickle=False) as archive:
        values = {key: archive[key] for key in ('mean_by_scale', 'R_of_mean', 'pca_center',
                  'pca_components', 'encoding_estimator', 'projected_mean_by_scale', 'projected_true_displacement')}
    center, components = values['pca_center'], values['pca_components']
    means, mapped = values['mean_by_scale'], values['R_of_mean']
    if (means.shape != mapped.shape or means.ndim != 3 or means.shape[-1] != len(center)
            or values['encoding_estimator'].item() != metrics['encoding_estimator']
            or not np.allclose(center, metrics['pca']['center'], atol=1e-14, rtol=0)
            or not np.allclose(components, metrics['pca']['components'], atol=1e-14, rtol=0)
            or not np.all(np.isfinite(means)) or not np.all(np.isfinite(mapped))):
        raise ValueError('正式编码/PCA/映射数据不一致')
    x, y = (means-center) @ components.T, (mapped-center) @ components.T
    if (not np.allclose(x, values['projected_mean_by_scale'], atol=1e-13, rtol=0)
            or not np.allclose(y-x, values['projected_true_displacement'], atol=1e-13, rtol=0)):
        raise ValueError('与正式报告保存的输入投影或真实位移不一致')
    fractions = np.asarray(metrics['pca']['explained_variance_fraction'])
    selected = select_components(fractions, threshold)
    roots = json.loads((report / 'fixed_points.json').read_text())['roots']
    return dict(metrics=metrics, fractions=fractions, selected=selected, x=x, y=y,
                roots=roots, center=center, components=components,
                source_sha256={name: sha(report / name) for name in required})


def _style_font(figure, font_path):
    from matplotlib.text import Text
    for artist in figure.findobj(Text):
        properties = artist.get_fontproperties().copy()
        properties.set_file(str(font_path))
        artist.set_fontproperties(properties)


def write_plots(report, output, *, threshold=.99):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_ising_zh import select_chinese_font
    data = prepare_data(report, threshold=threshold)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('展示图使用新目录，不覆盖正式报告或已存在的图')
    output.mkdir(parents=True)
    font = select_chinese_font()
    metrics, fractions, selected = data['metrics'], data['fractions'], data['selected']
    seed, dimension = metrics['training_seed'], metrics['theory_dim']
    percent = 100 * fractions
    plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    colors = ['#2867ad' if index in selected else '#aab6c4' for index in range(len(fractions))]
    bars = axis.bar(np.arange(len(fractions)), percent, color=colors, width=.61)
    for index, (bar, value) in enumerate(zip(bars, percent)):
        label = (f'{value:.4f}%' if value >= .01 else f'{value:.6f}%'
                 if value >= .00001 else f'{value:.3g}%')
        axis.text(bar.get_x()+bar.get_width()/2, max(value, .5)+1.3, label,
                  ha='center', va='bottom', fontsize=12, color='#253b53')
    axis.set(xticks=np.arange(len(fractions)),
        xticklabels=[f'第{index+1}主成分' for index in range(len(fractions))],
        ylabel='解释方差占比（%）', ylim=(0, 112),
        title=f'训练种子{seed}：各主成分的方差占比（潜变量维数{dimension}）')
    axis.set_yticks(np.arange(0, 101, 20))
    axis.grid(axis='y', alpha=.18)
    axis.set_axisbelow(True)
    axis.text(.98, .83,
        f'主要成分：前{len(selected)}个\n累计占比：{percent[selected].sum():.4f}%\n选择阈值：{100*threshold:g}%',
        ha='right', va='top', transform=axis.transAxes, fontsize=12,
        bbox=dict(boxstyle='round,pad=.65', facecolor='#f3f6fa', edgecolor='#d5dfe9'))
    figure.text(.5, .035, '仅由训练输入尺度拟合；方差占比不等于独立物理耦合方向数。',
                ha='center', fontsize=11, color='#536170')
    figure.subplots_adjust(left=.09, right=.97, bottom=.16, top=.87)
    _style_font(figure, font)
    figure.savefig(output / '主成分方差占比.png', dpi=180)
    plt.close(figure)

    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected)/columns))
    figure, axes = plt.subplots(rows, columns, figsize=(8.3*columns, 7.5*rows), squeeze=False)
    palette = ['#2474b5', '#e78b22', '#2b9b58']
    for panel, axis in enumerate(axes.flat):
        if panel >= len(selected):
            axis.set_visible(False)
            continue
        coordinate = int(selected[panel])
        x, y = data['x'][..., coordinate], data['y'][..., coordinate]
        for scale in range(x.shape[1]):
            label = f'k={scale}（{metrics["lattice_sizes"][scale]}）'
            if scale == x.shape[1]-1:
                label = f'k={scale}（{metrics["lattice_sizes"][scale]}；输入尺度外推）'
            axis.plot(x[:, scale], y[:, scale], '.-' if scale < x.shape[1]-1 else '.--',
                color=palette[scale % len(palette)], lw=2, ms=4, label=label)
        root_positions = [(root, float((np.asarray(root['g'])-data['center']) @ data['components'][coordinate]))
                          for root in data['roots']]
        low = min(float(x.min()), float(y.min()), *(point for _, point in root_positions))
        high = max(float(x.max()), float(y.max()), *(point for _, point in root_positions))
        width = high-low
        axis.plot([low-.025*width, high+.025*width], [low-.025*width, high+.025*width],
                  '--', color='#747c85', lw=1.3, label='恒等映射')
        for root_index, (root, point) in enumerate(root_positions):
            axis.scatter(point, point, marker='*', s=190, c='#f9c74f', edgecolors='#333333', zorder=5,
                         label='全坐标数值根的投影' if root_index == 0 else None)
            kind = {'stable': '数值吸引点', 'saddle': '数值鞍点'}.get(root['stability'], '数值根')
            axis.annotate(f'{root["root_id"]}：{kind}', (point, point),
                          xytext=(8, 11), textcoords='offset points', fontsize=11)
        axis.set(xlabel=f'输入第{coordinate+1}主成分', ylabel=f'输出第{coordinate+1}主成分',
            title=f'第{coordinate+1}主成分上的 R(g)（方差占比{percent[coordinate]:.4f}%）',
            xlim=(low-.055*width, high+.12*width), ylim=(low-.055*width, high+.12*width))
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.18)
        axis.legend(loc='upper left', fontsize=10, framealpha=.94)
    figure.suptitle(f'训练种子{seed}：主要成分上的完整{dimension}维映射投影', fontsize=15)
    figure.text(.5, .055, '按各条件平均编码作图：输出为 R(平均 g)，不是平均 R(g)。', ha='center', fontsize=10.5)
    figure.text(.5, .024, '投影不是独立一维自治映射；投影交点不是新根；已标数值根均未认证物理支持。',
                ha='center', fontsize=10.5, color='#536170')
    figure.subplots_adjust(left=.105, right=.965, bottom=.17, top=.86, wspace=.25, hspace=.4)
    _style_font(figure, font)
    figure.savefig(output / '主要成分上的R映射.png', dpi=180)
    plt.close(figure)
    receipt = dict(source_report=str(Path(report).resolve()), training_seed=seed,
        source_sha256=data['source_sha256'], model_loaded=False, inference_performed=False,
        roots_recomputed=False, PCA_refitted=False, formal_report_overwritten=False,
        encoding_estimator=metrics['encoding_estimator'],
        explained_variance_fraction=fractions.tolist(), selected_component_indices_zero_based=selected.tolist(),
        selection_rule='minimum number of leading TRAIN-fitted PCs reaching cumulative explained variance threshold',
        cumulative_variance_threshold=threshold, selected_cumulative_fraction=float(fractions[selected].sum()),
        map_definition='x=P(mean_g-center); y=P(R_of_mean_g-center); not P(mean_R_g-center)',
        out_of_training_R_input_scale=2,
        roots='All saved full-coordinate numerical roots projected, without new root search; none physically certified',
        files=['主成分方差占比.png', '主要成分上的R映射.png'])
    (output / '绘图回执.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variance-threshold', type=float, default=.99)
    args = parser.parse_args()
    print(json.dumps(write_plots(args.report, args.output, threshold=args.variance_threshold), ensure_ascii=False))


if __name__ == '__main__':
    main()
