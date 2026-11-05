"""Plot Kc scale markers and closure gaps from saved public JSON only.

No model, checkpoint, sampling, or training modules are imported. All R values
and root locations must already exist in the input. Existing output is refused.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_zh import chinese_figures


PURPLE = '#a50bb6'
STABILITY = {'stable': '稳定', 'unstable': '不稳定',
             'nonhyperbolic': '非双曲', 'saddle': '鞍点', 'unknown': '未知'}


def finite_array(value, name, *, ndim=1):
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected a nonempty finite {ndim}D array')
    return result


def validate_saved_data(data):
    """Check saved scalar coordinates, without fitting or recalculating R."""
    if not isinstance(data, dict):
        raise ValueError('Input must be a JSON object')
    if not isinstance(data['actual_epochs'], int) or data['actual_epochs'] < 1:
        raise ValueError('actual_epochs must be a positive integer')
    kc = float(data['Kc'])
    if not np.isfinite(kc):
        raise ValueError('Kc must be finite')
    for key in ('flow', 'zoom_flow'):
        x = finite_array(data[key]['g'], key + '.g')
        y = finite_array(data[key]['R'], key + '.R')
        if x.shape != y.shape or len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError(f'{key}: matching, increasing g and R arrays required')
    support = finite_array(data['training_input_range'], 'training_input_range')
    if support.shape != (2,) or support[0] > support[1]:
        raise ValueError('Invalid training input range')
    critical = data['critical']
    g = finite_array(critical['g_by_scale'], 'critical.g_by_scale')
    n = len(g)
    if n < 2 or len(data['lattice_sizes']) != n:
        raise ValueError('At least two scales and matching lattice sizes required')
    for key in ('R_of_mean_by_scale', 'mean_R_by_scale', 'jensen_gap_by_scale'):
        if finite_array(critical[key], 'critical.' + key).shape != (n,):
            raise ValueError('Critical arrays must have one value per saved scale')
    for root in data['roots']:
        if finite_array(root['g'], 'root.g').shape != (1,):
            raise ValueError('Only saved scalar roots are supported')
    transitions = critical['transitions']
    if (len(transitions) != n - 1 or
            sorted((row['input_scale'], row['output_scale']) for row in transitions)
            != [(k, k + 1) for k in range(n - 1)]):
        raise ValueError('Require each observed adjacent-scale transition exactly once; no last-scale target')
    for row in transitions:
        a, b = row['input_scale'], row['output_scale']
        saved = finite_array([row[key] for key in (
            'input_g', 'next_g', 'R_of_mean', 'closure_residual', 'rg_drift',
            'mean_R', 'paired_mean_closure_residual', 'jensen_gap')], 'transition')
        expected = [g[a], g[b], critical['R_of_mean_by_scale'][a],
                    row['R_of_mean'] - row['next_g'], row['R_of_mean'] - row['input_g'],
                    critical['mean_R_by_scale'][a], row['mean_R'] - row['next_g'],
                    row['mean_R'] - row['R_of_mean']]
        if not np.allclose(saved, expected, rtol=1e-6, atol=1e-9):
            raise ValueError('Saved transition fields disagree; refusing a misleading closure plot')
    knots = finite_array(data['coupling']['K'], 'coupling.K')
    means = finite_array(data['coupling']['g_by_scale'], 'coupling.g_by_scale', ndim=2)
    if means.shape != (len(knots), n) or not np.all(np.diff(knots) > 0):
        raise ValueError('Coupling means must match increasing K knots and saved scales')
    if not knots[0] <= kc <= knots[-1]:
        raise ValueError('Saved K knots must bracket Kc')
    return data


def draw_saved_markers(data):
    """Return a figure made exclusively from supplied numeric arrays."""
    validate_saved_data(data)
    critical = data['critical']
    g = np.asarray(critical['g_by_scale'])
    predicted = np.asarray(critical['R_of_mean_by_scale'])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.4))
    fig.subplots_adjust(left=.055, right=.985, top=.82, bottom=.29, wspace=.29)
    for panel, flow_name in enumerate(('flow', 'zoom_flow')):
        ax = axes[panel]
        flow = data[flow_name]
        x = np.asarray(flow['g'])
        ax.plot(x, flow['R'], color='#2878b5', lw=1.8, label='完整 R(g)')
        ax.plot(x, x, 'k--', lw=1.1, label='恒等映射')
        ax.axvspan(*data['training_input_range'], color='forestgreen', alpha=.08,
                   label='R 的训练输入范围')
        for index, root in enumerate(data['roots']):
            point = root['g'][0]
            stability = STABILITY.get(root.get('stability'), '未知')
            color = '#2a8b47' if root.get('stability') == 'stable' else '#e09228'
            ax.scatter(point, point, marker='*', s=150, c=color, edgecolors='black',
                       zorder=5, label=f'F{index + 1}：{stability}' if panel == 0 else None)
        ax.scatter(g, predicted, marker='D', s=65, c=PURPLE, edgecolors='white',
                   linewidths=.6, zorder=8, label='Kc 输入编码（非不动点）')
        for k, (xk, yk) in enumerate(zip(g, predicted)):
            # Opposite offsets keep nearby scales readable without arrowheads.
            dx, dy = ((7, 10), (7, -17), (-31, 11))[k % 3]
            ax.annotate(f'k={k}', (xk, yk), xytext=(dx, dy),
                        textcoords='offset points', color=PURPLE, fontsize=10,
                        zorder=9, clip_on=True)
        ax.set(xlabel='输入 g', ylabel='R(g)', xlim=(x[0], x[-1]))
        ax.grid(alpha=.2)
    axes[0].set_title('全域离散映射：保留全部已保存不动点', fontsize=12)
    axes[0].legend(fontsize=8, loc='best')
    zoom = axes[1]
    zoom.set_title('Kc 编码邻域：逐尺度预测与下一尺度目标', fontsize=12)
    zoom_values = [np.asarray(data['zoom_flow']['R']),
                   np.asarray(data['zoom_flow']['g']), g, predicted]
    for index, row in enumerate(critical['transitions']):
        x, target, estimate = row['input_g'], row['next_g'], row['R_of_mean']
        zoom.plot([x, x], [target, estimate], color=PURPLE, lw=1.8, alpha=.9,
                  zorder=6, label='预测减下一尺度目标' if index == 0 else None)
        zoom.scatter(x, target, marker='o', s=85, facecolors='none', edgecolors=PURPLE,
                     linewidths=1.5, zorder=7, label='下一尺度编码目标' if index == 0 else None)
        zoom_values.append(np.array([target]))
    # Saved zoom arrays supply the x-range. Include every observed next-scale
    # target vertically, even when its closure gap is larger than map drift.
    all_y = np.concatenate(zoom_values)
    pad = max(float(np.ptp(all_y)) * .09, 1e-7)
    zoom.set_ylim(float(all_y.min() - pad), float(all_y.max() + pad))
    zoom.legend(fontsize=8, loc='best')
    knots = data['coupling']['K']
    means = np.asarray(data['coupling']['g_by_scale'])
    for k, size in enumerate(data['lattice_sizes']):
        axes[2].plot(knots, means[:, k], label=f'k={k}，尺寸 {size}×{size}')
    axes[2].axvline(data['Kc'], color=PURPLE, ls=':', lw=1.3, label='已知输入 Kc')
    axes[2].scatter([data['Kc']] * len(g), g, marker='D', s=65, c=PURPLE,
                    edgecolors='white', linewidths=.6, zorder=8)
    label_offsets = ((12, 24), (12, -24), (34, 0))
    for k, value in enumerate(g):
        axes[2].annotate(f'k={k}', (data['Kc'], value),
                         xytext=label_offsets[k % len(label_offsets)],
                         textcoords='offset points', color=PURPLE, fontsize=10,
                         arrowprops=dict(arrowstyle='-', color=PURPLE, lw=.65, alpha=.7))
    axes[2].set(xlabel='初始 K=J/T，B=0', ylabel='g', title='已保存的耦合到 g 映射')
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=8, loc='best')
    fig.suptitle(f'{data["actual_epochs"]} 次更新：紫色菱形是各尺度 Kc 输入编码，并非不动点',
                 fontsize=15, y=.97)
    fig.text(.5, .895, '左、中图的紫色菱形为（当前尺度平均 g，R 作用于该平均值）；紫色空心圆为（当前尺度平均 g，下一尺度平均 g）。',
             ha='center', fontsize=10)
    fig.text(.055, .20, '竖线仅显示预测与下一尺度目标的差，不表示流向；输入来自相同的微观 Kc，并不保证各尺度位于 R 不动点。', fontsize=10)
    for index, row in enumerate(sorted(critical['transitions'], key=lambda item: item['input_scale'])):
        fig.text(.055, .155 - .042 * index,
                 f'k={row["input_scale"]}→{row["output_scale"]}：预测减目标={row["closure_residual"]:+.6g}；'
                 f'系综配对后的平均差={row["paired_mean_closure_residual"]:+.6g}；'
                 f'平均次序差={row["jensen_gap"]:+.6g}。', fontsize=10)
    fig.text(.055, .035, f'末尺度 k={len(g) - 1} 没有下一尺度编码；仅显示其 R(g)，不计算该尺度的闭合误差。所有数值读取自保存的公共数据。', fontsize=10)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.output.suffix.lower() != '.png':
        raise ValueError('Output must use a .png extension')
    data = json.loads(args.input.read_text())
    figure = draw_saved_markers(data)
    try:
        with chinese_figures():
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=180)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against a file appearing mid-render.
        with args.output.open('xb') as stream:
            stream.write(buffer.getvalue())
    finally:
        plt.close(figure)
    print(json.dumps(dict(output=str(args.output.resolve()),
                          plotted_scales=len(data['critical']['g_by_scale']),
                          roots=len(data['roots']), transitions=len(data['critical']['transitions'])),
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
