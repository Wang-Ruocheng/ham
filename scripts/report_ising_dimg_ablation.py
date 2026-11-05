"""只读的无监督 Ising dimg=1/2/3 统一报告；K 与 Kc 仅用于评价。

多维有界多起点搜索并非完备求根。保留找到的每个根和完整 DR 的全部
特征谱，不按最大特征值或理论邻近性选择热模态。B0 的 K 曲线仅约束
能量路径，不认证横向支持；物理响应另由独立 MCRG 诊断评价。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from scripts.report_ising_zh import chinese_figures

KC = .4406867935
STABILITY_ZH = dict(stable='稳定', unstable='不稳定', saddle='鞍点', nonhyperbolic='非双曲')


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def mapped(model, points):
    with torch.no_grad():
        result = model.advance_theory(torch.as_tensor(np.asarray(points), dtype=torch.float64))
    return result.detach().cpu().numpy()


def full_r_jacobian(model, point, *, h=1e-5):
    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1 or not 1 <= len(point) <= 3 or not np.isfinite(point).all() or h <= 0:
        raise ValueError('Expected a finite dim 1/2/3 point and positive FD step')
    tensor = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(model.advance_theory, tensor).detach().cpu().numpy()
    finite = np.column_stack([(mapped(model, point + np.eye(len(point))[j]*h) -
                               mapped(model, point - np.eye(len(point))[j]*h))/(2*h)
                              for j in range(len(point))])
    if not np.isfinite(jacobian).all() or not np.isfinite(finite).all():
        raise ValueError('Nonfinite complete R derivative')
    residual = float(np.linalg.norm(mapped(model, point)-point))
    return dict(jacobian=jacobian.tolist(), finite_difference_jacobian=finite.tolist(),
                finite_difference_h=h, derivative_max_abs_error=float(np.max(abs(jacobian-finite))),
                derivative_verified=bool(np.allclose(jacobian, finite, rtol=1e-5, atol=1e-7)),
                residual_norm=residual, root_verified=bool(residual <= 1e-8))


def curve_projections(knots, curve, point, *, tolerance=1e-10):
    """所有精确逆分支与最近折线段；多维最近投影不是精确的逆 K。"""
    knots, curve, point = [np.asarray(v, dtype=np.float64) for v in (knots, curve, point)]
    if (knots.ndim != 1 or len(knots) < 2 or curve.shape != (len(knots), len(point)) or
            not np.all(np.diff(knots) > 0)):
        raise ValueError('Expected increasing K and one coordinate vector at each knot')
    rows, inverse, intervals = [], [], []
    for index, (start, delta) in enumerate(zip(curve[:-1], np.diff(curve, axis=0))):
        norm2 = float(delta @ delta)
        fraction = float(np.clip((point-start) @ delta / norm2, 0., 1.)) if norm2 else 0.
        projection = start + fraction*delta
        distance = float(np.linalg.norm(projection-point))
        coupling = float(knots[index] + fraction*(knots[index+1]-knots[index]))
        rows.append(dict(segment=index, K=coupling, distance=distance, projected_g=projection.tolist(),
                         tangent=(delta/math.sqrt(norm2) if norm2 else np.zeros_like(point)).tolist()))
        if distance <= tolerance:
            if not norm2:
                intervals.append([float(knots[index]), float(knots[index+1])])
            elif not any(abs(coupling-old) <= tolerance for old in inverse):
                inverse.append(coupling)
    rows.sort(key=lambda row: (row['distance'], row['segment']))
    return dict(exact_inverse_K_branches=sorted(inverse), exact_inverse_K_intervals=intervals,
                nearest_distance=rows[0]['distance'],
                nearest_segments=[row for row in rows if abs(row['distance']-rows[0]['distance']) <= tolerance],
                closest_segments=rows[:5],
                caution='折线最近投影须连同距离报告，不是唯一物理逆 K；保留全部精确分支与平坦区间。')


def representation_statistics(direct):
    direct = np.asarray(direct, dtype=np.float64)
    rows = []
    for scale in range(direct.shape[1]):
        values = direct[:, scale]
        covariance = np.atleast_2d(np.cov(values.T, bias=True))
        spectrum = np.linalg.eigvalsh(covariance).clip(min=0.)
        total = float(spectrum.sum())
        probabilities = spectrum[spectrum > 0]/total if total > 0 else np.array([])
        rows.append(dict(scale=scale, coordinate_std=values.std(0).tolist(),
            pooled_coordinate_std=float(np.sqrt(values.var(0).mean())),
            covariance=covariance.tolist(), covariance_eigenvalues=spectrum.tolist(),
            participation_rank=float(total**2/(spectrum@spectrum)) if total else None,
            entropy_effective_rank=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if total else None,
            numerical_rank=int(np.linalg.matrix_rank(covariance)),
            caution='协方差有效秩不等于物理相关算符数。'))
    return rows


def training_support(point, train):
    from scipy.spatial import cKDTree, Delaunay, QhullError
    inputs = np.asarray(train)[:, :-1].reshape(-1, len(point))
    point = np.asarray(point)
    sigma = inputs.std(0)
    scale = np.where(sigma > 1e-12, sigma, 1.)
    whitened = inputs/scale
    tree = cKDTree(whitened)
    nearest = float(tree.query(point/scale)[0])
    nn = tree.query(whitened, k=min(2, len(whitened)))[0]
    radius = float(np.quantile(nn[:, -1], .95))*2 if nn.ndim == 2 else 0.
    inside_box = bool(np.all(point >= inputs.min(0)) and np.all(point <= inputs.max(0)))
    hull = inside_box if len(point) == 1 else None
    if len(point) > 1 and np.linalg.matrix_rank(whitened-whitened.mean(0)) == len(point):
        try:
            hull = bool(Delaunay(whitened).find_simplex(point/scale) >= 0)
        except QhullError:
            pass
    return dict(nearest_training_input_distance=float(np.linalg.norm(inputs-point, axis=1).min()),
        nearest_training_input_distance_in_coordinate_sd=nearest,
        train_nearest_neighbor_95_radius_in_coordinate_sd=radius,
        inside_training_coordinate_box=inside_box, inside_training_convex_hull=hull,
        geometric_support_candidate=bool(inside_box and nearest <= max(radius, 1e-10)),
        physical_support_certified=False,
        caution='盒内、凸包内、近曲线及近邻距离仅为几何诊断，不认证热方向或横向导数的数据支持。')


def eigenvalue_records(jacobian, projections, *, block_factor, verified, support_candidate):
    values, vectors = np.linalg.eig(np.asarray(jacobian))
    inverse = np.linalg.inv(vectors) if np.linalg.matrix_rank(vectors) == len(values) else None
    rows = []
    for index, value in enumerate(values):
        vector = vectors[:, index]
        alignments, amplitudes = [], []
        for projection in projections:
            tangents = [np.asarray(segment['tangent']) for segment in projection['nearest_segments']]
            alignments.append([float(abs(np.vdot(vector, tangent))/np.linalg.norm(vector)) for tangent in tangents])
            amplitudes.append([float(abs(inverse[index] @ tangent)) for tangent in tangents] if inverse is not None else None)
        relevant = bool(abs(value.imag) <= 1e-8 and value.real > 1+1e-6)
        rows.append(dict(index=index, real=float(value.real), imag=float(value.imag), magnitude=float(abs(value)),
            eigenvector_real=vector.real.tolist(), eigenvector_imag=vector.imag.tolist(),
            K_curve_tangent_alignment_by_scale=alignments,
            oblique_K_tangent_coefficient_by_scale=amplitudes,
            positive_real_relevant=relevant,
            conditional_inverse_y=(float(math.log(block_factor)/math.log(value.real))
                if relevant and verified and support_candidate else None),
            thermal_nu=None, thermal_identity_certified=False,
            physical_sector='未识别；仅报告能量路径对齐，不按最大值或理论接近性选择',
            caution='有条件逆 y 仅为算术换算；单一 B0 K 曲线不能认证完整横向谱，未据此认定热指数。'))
    return rows


def find_roots(model, train, knots, means, *, block_factor=2, seed_points=7):
    from scipy.optimize import root, brentq, minimize_scalar
    dim = train.shape[-1]
    envelope = train.reshape(-1, dim)
    span = np.maximum(np.ptp(envelope, axis=0), .1)
    lower, upper = envelope.min(0)-.25*span, envelope.max(0)+.25*span
    seeds = list(itertools.product(*[np.linspace(a, b, seed_points) for a, b in zip(lower, upper)]))
    # TRAIN geometry only determines the search; evaluation K values never select seeds.
    seeds += [point.tolist() for point in train[::max(1, len(train)//32)].reshape(-1, dim)]
    residual = lambda z: mapped(model, z)-z
    def jac(z):
        tensor = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(model.advance_theory, tensor).detach().numpy()-np.eye(dim)
    candidates, attempts = [], []
    if dim == 1:
        grid = np.linspace(lower[0], upper[0], 1501)
        f = residual(grid[:, None])[:, 0]
        if not np.all(abs(f) < 1e-12):
            for index in np.flatnonzero(f[:-1]*f[1:] < 0):
                candidates.append(np.array([brentq(lambda x: residual(np.array([x]))[0], grid[index], grid[index+1])]))
            candidates += [np.array([value]) for value in grid[f == 0]]
            for index in np.flatnonzero((abs(f[1:-1]) < abs(f[:-2])) & (abs(f[1:-1]) < abs(f[2:]))):
                solution = minimize_scalar(lambda x: residual(np.array([x]))[0]**2,
                    bounds=(grid[index], grid[index+2]), method='bounded', options=dict(xatol=1e-14))
                candidates.append(np.array([solution.x]))
        else:
            return dict(roots=[], search_box=[lower.tolist(), upper.tolist()], seed_count=len(seeds),
                        search_complete=False, note='映射在网格上数值恒等；无法列出孤立根。')
    for guess in seeds:
        solution = root(residual, guess, jac=jac)
        candidate = solution.x
        if np.isfinite(candidate).all():
            candidates.append(candidate)
        attempts.append(bool(solution.success))
    accepted = []
    dedup = max(float(np.linalg.norm(span))*1e-6, 1e-8)
    for point in candidates:
        if (np.any(point < lower) or np.any(point > upper) or
                np.linalg.norm(residual(point)) > 1e-8 or any(np.linalg.norm(point-old) < dedup for old in accepted)):
            continue
        accepted.append(point)
    accepted.sort(key=lambda point: tuple(point))
    records = []
    for index, point in enumerate(accepted):
        derivative = full_r_jacobian(model, point)
        support = training_support(point, train)
        projections = [curve_projections(knots, means[:, scale], point) for scale in range(means.shape[1])]
        eig = eigenvalue_records(derivative['jacobian'], projections, block_factor=block_factor,
            verified=derivative['derivative_verified'] and derivative['root_verified'],
            support_candidate=support['geometric_support_candidate'])
        magnitudes = np.asarray([item['magnitude'] for item in eig])
        stability = ('nonhyperbolic' if np.any(abs(magnitudes-1) <= 1e-6) else
            'stable' if np.all(magnitudes < 1) else 'unstable' if np.all(magnitudes > 1) else 'saddle')
        records.append(dict(root_id=f'F{index+1}', g=point.tolist(), **derivative, **support,
            eigenvalues=eig, stability=stability, curve_projections_by_scale=projections,
            selected_thermal_eigenvalue=None, thermal_nu=None))
    return dict(roots=records, search_box=[lower.tolist(), upper.tolist()], seed_count=len(seeds),
        solver_success_count=sum(attempts), root_dedup_tolerance=dedup, search_complete=False,
        note='TRAIN 包络外扩 25%；确定性网格与 TRAIN 种子，多起点有界验收；可能遗漏根，不能声称已穷尽。')


def closure_details(model, trajectories, train):
    direct = trajectories['direct']
    result = closure_metrics(trajectories, reference_direct=train)
    std = train.std(axis=0)
    for key, target, prediction, sigma in (
            ('one_step', direct[:, 1:], trajectories['one_step'], std[1:]),
            ('rollout', direct, trajectories['rollout'], std)):
        error = np.sqrt(np.mean((prediction-target)**2, axis=0))
        result[key]['coordinate_rmse'] = error.tolist()
        result[key]['coordinate_nrmse'] = [[float(x/y) if y > 1e-12 else None for x, y in zip(a, b)]
                                          for a, b in zip(error, sigma)]
    horizons = []
    for start in range(direct.shape[1]-1):
        prediction = direct[:, start]
        for end in range(start+1, direct.shape[1]):
            prediction = mapped(model, prediction)
            rmse = float(np.sqrt(np.mean((prediction-direct[:, end])**2)))
            scale = float(np.sqrt(train[:, end].var(0).mean()))
            horizons.append(dict(start=start, end=end, rmse=rmse, nrmse=rmse/scale if scale > 1e-12 else None))
    result['all_start_horizon'] = horizons
    result['normalization'] = 'TRAIN 目标尺度的均方坐标标准差；逐坐标另列，不混淆不同模型坐标。'
    return result


def coupling_statistics(train, test, train_labels, test_labels):
    direct = np.concatenate((train, test))
    labels = np.concatenate((np.asarray(train_labels).reshape(-1), np.asarray(test_labels).reshape(-1)))
    if len(direct) != len(labels) or not np.isfinite(labels).all():
        raise ValueError('Need finite evaluation K labels for each ensemble')
    knots = np.unique(labels)
    if len(knots) < 2:
        raise ValueError('Need at least two evaluation K knots')
    if not knots[0] <= KC <= knots[-1]:
        raise ValueError('Evaluation K knots must bracket Kc; do not silently extrapolate its encoding')
    means = np.stack([direct[labels == k].mean(0) for k in knots])
    critical = np.stack([[np.interp(KC, knots, means[:, scale, coordinate])
        for coordinate in range(direct.shape[-1])] for scale in range(direct.shape[1])])
    return knots, means, critical


def energy_path_diagnostics(model, knots, means, *, half_width=.03):
    """K 路径切向是能量方向线索，不是对热本征模的认证。"""
    mask = abs(knots-KC) <= half_width
    if mask.sum() < 4:
        return dict(status='insufficient_K_knots', half_width=half_width, thermal_nu=None)
    t = knots[mask]-KC
    tangent = np.stack([np.polyfit(t, means[mask, scale], 3)[-2] for scale in range(means.shape[1])])
    predicted_mean = mapped(model, means[mask])
    predicted_tangent = np.stack([np.polyfit(t, predicted_mean[:, scale], 3)[-2]
                                 for scale in range(means.shape[1]-1)])
    rows = []
    for scale in range(means.shape[1]-1):
        v, target, predicted = tangent[scale], tangent[scale+1], predicted_tangent[scale]
        denominator = float(v@v)
        scalar = float(v@target/denominator) if denominator > 1e-20 else None
        residual = float(np.linalg.norm(target-scalar*v)/np.linalg.norm(target)) if scalar is not None and np.linalg.norm(target) else None
        rows.append(dict(start_scale=scale, input_K_tangent=v.tolist(), observed_next_K_tangent=target.tolist(),
            R_of_mean_K_tangent=predicted.tolist(), best_scalar_response=scalar,
            scalar_direction_residual=residual,
            R_tangent_relative_error=float(np.linalg.norm(predicted-target)/np.linalg.norm(target)) if np.linalg.norm(target) else None,
            thermal_nu=None))
    return dict(status='complete', half_width=half_width, polynomial_degree=3, transitions=rows,
        caution='仅保存编码的 K 路径三次拟合；不是独立能量重加权响应，不认证热本征倍率。')


def plot_maps(model, train, test, knots, means, critical, roots, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dim = train.shape[-1]
    fig, axes = plt.subplots(1, dim, figsize=(6*dim, 5), squeeze=False, layout='constrained')
    for coordinate, ax in enumerate(axes.flat):
        if dim == 1:
            low, high = np.asarray(roots['search_box'])[:, 0]
            grid = np.linspace(low, high, 801)[:, None]
            ax.plot(grid[:, 0], mapped(model, grid)[:, 0], label='完整 R(g)')
            ax.plot(grid[:, 0], grid[:, 0], 'k--', lw=.8, label='恒等映射')
        else:
            for scale in range(means.shape[1]):
                ax.plot(means[:, scale, coordinate], mapped(model, means[:, scale])[:, coordinate],
                        '.-', ms=3, label=f'k={scale}，沿 K 曲线')
        for scale, point in enumerate(critical):
            ax.scatter(point[coordinate], mapped(model, point)[coordinate], c='magenta', marker='D', s=40,
                       label='Kc 编码（非不动点）' if scale == 0 else None)
            ax.annotate(f'k={scale}', (point[coordinate], mapped(model, point)[coordinate]),
                        xytext=(4, 7+10*scale), textcoords='offset points', fontsize=7)
        for root in roots['roots']:
            ax.scatter(root['g'][coordinate], root['g'][coordinate], marker='*', s=90, c='gold', edgecolors='black')
            ax.annotate(root['root_id'], (root['g'][coordinate], root['g'][coordinate]), xytext=(4, 4), textcoords='offset points')
        ax.set(xlabel=f'输入 g{coordinate+1}', ylabel=f'输出 R(g){coordinate+1}',
               title='完整离散映射' if dim == 1 else '完整向量映射的坐标投影；非标量映射')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f'无监督 QCR，潜变量维数={dim}；标签仅用于评价')
    fig.savefig(out/'R_vs_g.png', dpi=160); plt.close(fig)
    if dim == 1:
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        grid = np.linspace(*np.asarray(roots['search_box'])[:, 0], 801)[:, None]
        delta = mapped(model, grid)-grid
        ax.plot(grid[:, 0], delta[:, 0], label='位移 R(g)-g'); ax.axhline(0., c='black', ls='--', lw=.8)
        for scale, point in enumerate(critical):
            ax.scatter(point[0], (mapped(model, point)-point)[0], c='magenta', marker='D', label='Kc 编码（非不动点）' if scale == 0 else None)
        for root in roots['roots']:
            ax.scatter(root['g'][0], 0., marker='*', s=90, c='gold', edgecolors='black')
        ax.set(xlabel='g', ylabel='R(g)-g', title='完整离散位移与已找到的根'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    else:
        pairs = list(itertools.combinations(range(dim), 2))
        fig, axes = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 5), squeeze=False, layout='constrained')
        for ax, (a, b) in zip(axes.flat, pairs):
            for scale in range(means.shape[1]):
                curve = means[:, scale]
                displacement = mapped(model, curve)-curve
                ax.plot(curve[:, a], curve[:, b], label=f'k={scale}')
                every = max(1, len(curve)//24)
                ax.quiver(curve[::every, a], curve[::every, b], displacement[::every, a], displacement[::every, b],
                          angles='xy', scale_units='xy', scale=1., color=f'C{scale}', width=.004)
                ax.scatter(critical[scale, a], critical[scale, b], marker='D', c='magenta', s=40,
                           label='Kc 编码（非不动点）' if scale == 0 else None)
            for root in roots['roots']:
                ax.scatter(root['g'][a], root['g'][b], marker='*', s=90, c='gold', edgecolors='black')
                ax.annotate(root['root_id'], (root['g'][a], root['g'][b]), xytext=(4, 4), textcoords='offset points')
            ax.set(xlabel=f'g{a+1}', ylabel=f'g{b+1}', title='沿数据曲线的真实位移投影 R(g)-g')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('箭头为完整 R 的位移，不把横向最大特征值自动解释为热倍率')
    fig.savefig(out/'R_displacement.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, dim, figsize=(6*dim, 5), squeeze=False, layout='constrained')
    for coordinate, ax in enumerate(axes.flat):
        for scale in range(means.shape[1]):
            ax.plot(knots, means[:, scale, coordinate], label=f'k={scale}')
        ax.axvline(KC, c='magenta', ls=':', label='Kc（仅评价）')
        ax.scatter([KC]*len(critical), critical[:, coordinate], marker='D', c='magenta')
        ax.set(xlabel='微观 K（仅评价）', ylabel=f'g{coordinate+1}', title='各尺度 K → g 映射')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'coupling_by_scale.png', dpi=160); plt.close(fig)


def plot_losses(history, out):
    import matplotlib.pyplot as plt
    keys = [('anchor', '微观子集一致性'), ('coarse_consistency', '粗尺度子集一致性'),
            ('separation', 'Q 表示目标'), ('commutator', '单步闭合'), ('rollout', '多步闭合'),
            ('kernel_noncollapse', 'C 方差下限'), ('kernel_sign', '颜色保持'),
            ('total', '总损失'), ('qcr_total', 'QCR 总损失')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for ax, split, label in zip(axes[0], ('train', 'test'), ('训练集', '测试集（仅评价）')):
        for key, title in keys:
            available = [row for row in history if split+'_'+key in row]
            if available:
                ax.plot([row.get('epoch', row.get('step')) for row in available],
                        [row[split+'_'+key] for row in available], label=title)
        ax.set(xlabel='实际训练更新次数', ylabel='未加权记录值', title=label, yscale='symlog')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    for split, style, label in (('train', '-', '训练集'), ('test', '--', '测试集（仅评价）')):
        available = [row for row in history if split+'_rsmi_bits' in row]
        if available:
            axes[1, 0].plot([row['epoch'] for row in available], [row[split+'_rsmi_bits'] for row in available],
                            style, label=label)
        reconstruction_keys = sorted({key for row in history for key in row
            if key.startswith(split+'_reconstruction_mse_level_')})
        for key in reconstruction_keys:
            scale = int(key.rsplit('_', 1)[-1])
            available = [row for row in history if key in row]
            axes[1, 1].plot([row['epoch'] for row in available], [row[key] for row in available],
                style, color=f'C{scale}', label=label+f'：k={scale}')
    axes[1, 0].set(xlabel='实际训练更新次数', ylabel='经验 MI（比特）', title='既有全壳层经验互信息记录；未重算')
    axes[1, 1].set(xlabel='实际训练更新次数', ylabel='概率重构均方误差（按箱求和）',
                  title='共享仿射 Q 的各尺度重构误差', yscale='log')
    for ax in axes[1]:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle('原始分项记录；不按指数或测试损失选择检查点')
    fig.savefig(out/'loss_comparison.png', dpi=160); plt.close(fig)


def plot_representation(statistics, closure, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    for scale, row in enumerate(statistics):
        axes[0].plot(np.arange(len(row['coordinate_std']))+1, row['coordinate_std'], 'o-', label=f'k={scale}')
        axes[1].plot(np.arange(len(row['covariance_eigenvalues']))+1, row['covariance_eigenvalues'], 'o-', label=f'k={scale}')
    for name, label in (('one_step', '单步闭合'), ('rollout', '微观起点多步闭合')):
        for split, style, title in (('train', '-', '训练集'), ('test', '--', '测试集')):
            values = closure[split][name]['nrmse']
            axes[2].plot(np.arange(len(values))+(1 if name == 'one_step' else 0),
                         [np.nan if v is None else v for v in values], 'o'+style, label=title+'：'+label)
    axes[0].set(title='训练集各坐标标准差', xlabel='坐标编号', ylabel='标准差')
    axes[1].set(title='训练集协方差谱', xlabel='升序特征值编号', ylabel='协方差特征值')
    axes[2].set(title='按训练目标标准差归一的闭合误差', xlabel='目标尺度 k', ylabel='归一化均方根误差')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(out/'representation_diagnostics.png', dpi=160); plt.close(fig)


def write_report(loaded, train, test, train_labels, test_labels, history, output,
                 *, repeats=None, data_sha256=None, lattice_sizes=None):
    output = Path(output)
    dim = train['direct'].shape[-1]
    if dim not in (1, 2, 3) or train['direct'].shape[1:] != test['direct'].shape[1:]:
        raise ValueError('Matching dim 1/2/3 trajectories required')
    knots, means, critical = coupling_statistics(train['direct'], test['direct'], train_labels, test_labels)
    block = loaded.model_config.block_factor
    roots = find_roots(loaded.model, train['direct'], knots, means, block_factor=block)
    closure = {name: closure_details(loaded.model, values, train['direct'])
               for name, values in (('train', train), ('test', test))}
    statistics = representation_statistics(train['direct'])
    all_direct = np.concatenate((train['direct'], test['direct']))
    all_labels = np.concatenate((np.asarray(train_labels).ravel(), np.asarray(test_labels).ravel()))
    mean_R = np.stack([mapped(loaded.model, all_direct[all_labels == K]).mean(0) for K in knots])
    report = dict(schema_version=1, run=str(loaded.checkpoint.parent), checkpoint=str(loaded.checkpoint),
        checkpoint_epoch=loaded.payload.get('epoch'), theory_dim=dim, block_factor=block,
        lattice_sizes=lattice_sizes, data_sha256=data_sha256, labels_used_in_training=False,
        labels_used_in_report=True, evaluation_seed=12345, sampling_repeats=repeats,
        final_logged_metrics=history[-1] if history else None,
        logged_loss_caution='训练驱动原始记录；MI 与各尺度概率重构均为既有日志，不在报告中重估 MI。',
        closure=closure, train_representation=statistics, fixed_points=roots, roots=roots['roots'],
        energy_path=energy_path_diagnostics(loaded.model, knots, means),
        critical_reference=dict(Kc=KC, thermal_lambda=block, nu=1., evaluation_only=True,
            encoded_g_by_scale=critical.tolist(), R_of_encoded_mean=mapped(loaded.model, critical).tolist(),
            R_drift=(mapped(loaded.model, critical)-critical).tolist(),
            direct_scale_drift=np.diff(critical, axis=0).tolist(),
            caution='紫色菱形是 Kc 的各尺度前向编码，不是求得的不动点。'),
        jensen=dict(K=knots.tolist(), mean_R=mean_R.tolist(), R_of_mean=mapped(loaded.model, means).tolist(),
                    difference=(mean_R-mapped(loaded.model, means)).tolist()),
        selection=dict(checkpoint_rule='预定实际 12000 终点；不按指数接近理论挑选',
            selected_thermal_eigenvalue=None, theory_used_for_selection=False,
            all_found_roots_retained=True, all_eigenvalues_retained=True, exhaustive_search_claim=False),
        limitations=['单训练 seed；有限格点；两次粗粒化；仅 B0，不识别独立磁指数。',
            '不同 dimg 的坐标、标准差、参数数目与 RNG 消耗不同；归一误差不是相同坐标绝对误差。',
            '重复 C 闭合只条件于一个训练模型，不是特征值或指数置信区间。',
            'K 曲线切向对齐不是物理热模态认证；独立物理响应诊断须另读其报告。'])
    np.savez_compressed(output/'encodings.npz', train=train['direct'], test=test['direct'],
        train_rollout=train['rollout'], test_rollout=test['rollout'],
        train_labels=np.asarray(train_labels), test_labels=np.asarray(test_labels))
    np.savez_compressed(output/'trajectories.npz', **test, training_inputs=train['direct'][:, :-1],
                        couplings=np.asarray(test_labels), K=knots, mean_by_scale=means)
    write_json(output/'metrics.json', report)
    write_json(output/'critical_exponents.json', report)
    write_json(output/'fixed_points.json', roots)
    write_json(output/'loss_history.json', history)
    with chinese_figures():
        plot_maps(loaded.model, train['direct'], test['direct'], knots, means, critical, roots, output)
        plot_losses(history, output)
        plot_representation(statistics, closure, output)
    write_json(output/'report.json', dict(status='complete', report_complete=True,
        checkpoint_epoch=report['checkpoint_epoch'], theory_dim=dim,
        public_files=sorted(path.name for path in output.iterdir() if path.is_file()),
        training_started=False, physical_sampling_started=False, MI_estimated=False))
    return report


def compare_reports(specifications, output):
    """只读预定终点，不比较跨维度无意义的根编号或坐标。"""
    reports = {}
    responses = {}
    for specification in specifications:
        name, path = specification.split('=', 1)
        path = Path(path)
        report_path = path if (path/'metrics.json').is_file() else path/'report'
        completion = json.loads((report_path/'report.json').read_text())
        if completion.get('status') != 'complete' or not completion.get('report_complete'):
            raise ValueError('Per-arm report is incomplete')
        report = json.loads((report_path/'metrics.json').read_text())
        if report['checkpoint_epoch'] != 12000:
            raise ValueError('All comparison arms must reach actual 12000')
        reports[name] = report
        response_path = report_path.parent/'response'/'comparison.json'
        responses[name] = json.loads(response_path.read_text()) if response_path.is_file() else None
    if sorted(report['theory_dim'] for report in reports.values()) != [1, 2, 3]:
        raise ValueError('Comparison needs exactly dimg 1/2/3')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison.json', dict(reports=reports, physical_response=responses,
        caution='不跨实验匹配根编号；不把最大横向特征值或接近理论者自动选为热模态。'))
    import matplotlib.pyplot as plt
    with chinese_figures():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for name, report in reports.items():
            label = f'潜变量维数={report["theory_dim"]}'
            axes[0].plot(report['closure']['test']['one_step']['nrmse'], 'o-', label=label)
            axes[1].plot([row['participation_rank'] for row in report['train_representation']], 'o-', label=label)
            for root in report['roots']:
                for eigenvalue in root['eigenvalues']:
                    axes[2].scatter(report['theory_dim'], eigenvalue['magnitude'], marker='o',
                        facecolors='C0' if root['geometric_support_candidate'] else 'none', edgecolors='C0')
        axes[0].set(title='测试集单步归一闭合误差', xlabel='变换编号', ylabel='归一化均方根误差')
        axes[1].set(title='训练集协方差参与秩', xlabel='尺度 k', ylabel='有效秩')
        axes[2].set(title='全部已找到根的完整谱；非热指数', xlabel='潜变量维数', ylabel='特征值模长', xticks=[1, 2, 3])
        for ax in axes:
            ax.grid(alpha=.2)
        for ax in axes[:2]:
            ax.legend(fontsize=8)
        fig.suptitle('固定 12000 步：升维消融；空心表示几何支持候选之外，实心亦不认证物理支持')
        fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)
    write_json(output/'report.json', dict(status='complete', report_complete=True,
        arms=list(reports), physical_response_available={key: value is not None for key, value in responses.items()}))
    return reports


def validate_completed_run(cfg, summary, payload, checkpoint):
    """正式节点必须实际完整12000；已有文件名不等于训练已经完成。"""
    if (checkpoint != 12000 or payload.get('epoch') != checkpoint or cfg.training.epochs != checkpoint or
            summary.get('actual_epochs') != checkpoint or summary.get('requested_epochs') != checkpoint or
            json_value(summary.get('config')) != json_value(cfg.to_dict()) or summary.get('stopped_by_wall_limit') or
            summary.get('stopped_by_training_limit') or summary.get('stopped_by_train_plateau')):
        raise ValueError('Formal dimg comparison requires actual complete 12000, matching config, and no early stop')
    history = summary.get('history', [])
    if not history or history[-1].get('epoch') != checkpoint:
        raise ValueError('Completed summary history must reach actual 12000')
    saved_history = payload.get('history')
    if saved_history and saved_history[-1].get('epoch') != checkpoint:
        raise ValueError('Saved numbered checkpoint history must reach actual 12000')


def report_datasets(cfg):
    """mmap 既有数据，只检查元数据，不再完整扫描物理缓存。"""
    if cfg.data.train_pool_replicas != 1 or cfg.data.test_pool_replicas != 1:
        raise ValueError('The declared dimg ablation must not pool independent ensembles')
    payload = torch.load(str(Path(cfg.data.cache)), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(payload, (tuple, list)) and len(payload) == 4:
        tx, ty, vx, vy = payload
        pairs = ((tx, ty), (vx, vy))
    elif isinstance(payload, dict) and payload.get('schema_version') == 1:
        pairs = [(payload[split]['configurations'], payload[split]['couplings']) for split in ('train', 'test')]
    else:
        raise ValueError('Existing supported data cache required; no sampling fallback')
    datasets = []
    for configurations, couplings in pairs:
        if (configurations.ndim != 5 or configurations.shape[-1] != 1 or
                configurations.shape[2] != configurations.shape[3] or
                cfg.evaluation.samples > configurations.shape[1] or
                cfg.training.sample_batch > configurations.shape[1] or couplings is None or
                couplings.reshape(len(configurations), -1).shape != (len(configurations), 1)):
            raise ValueError('Cache metadata does not match single-layer K-only evaluation geometry')
        if configurations.shape[2] % cfg.model.block_factor**cfg.evaluation.max_rg_steps:
            raise ValueError('Saved lattice is not divisible at every requested scale')
        if not np.isfinite(couplings.detach().cpu().numpy()).all():
            raise ValueError('Nonfinite K metadata')
        datasets.append(SimpleNamespace(configurations=configurations, couplings=couplings))
    if datasets[0].configurations.shape[1:] != datasets[1].configurations.shape[1:]:
        raise ValueError('Train/test physical geometries differ')
    keys = [set(dataset.couplings.detach().cpu().numpy().ravel()) for dataset in datasets]
    if keys[0] & keys[1]:
        raise ValueError('Complete K cells must remain disjoint across train/test')
    return tuple(datasets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=12000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--sampling-repeats', type=int, default=5)
    parser.add_argument('--compare', nargs='+')
    args = parser.parse_args(argv)
    if args.compare:
        if args.run:
            parser.error('--compare and --run are mutually exclusive')
        return compare_reports(args.compare, args.output)
    if args.run is None or args.threads < 1 or args.sampling_repeats < 0:
        parser.error('--run, positive threads and nonnegative repeats required')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Refuse to overwrite or rerun an existing report output')
    torch.set_num_threads(args.threads)
    loaded = load_run(args.run, args.checkpoint)
    cfg = RunConfig.from_dict(loaded.config)
    cfg.validate()
    if (cfg.model.dimension != 2 or cfg.model.theory_dim not in (1, 2, 3) or cfg.model.categorical_states or
            cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',)):
        raise ValueError('Requires documented single-layer K-only Ising dimg 1/2/3')
    if loaded.payload.get('epoch') != args.checkpoint:
        raise ValueError('Numbered checkpoint epoch mismatch')
    summary = json.loads((args.run/'summary.json').read_text())
    validate_completed_run(cfg, summary, loaded.payload, args.checkpoint)
    metadata = loaded.payload.get('metadata', {})
    if metadata.get('labels_used_in_training') is not False or metadata.get('q_supervision') not in (None, {}):
        raise ValueError('Requires explicitly label-free training provenance')
    sidecar = json.loads(Path(cfg.data.cache).with_suffix('.json').read_text())
    manifest = json.loads((args.run/'manifest.json').read_text())
    if (sidecar.get('B') != 0. or sidecar.get('coupling_names') != ['K'] or
            not manifest.get('data_sha256') or sidecar.get('data_sha256') != manifest['data_sha256']):
        raise ValueError('B0 K-only sidecar and training data fingerprints must match')
    train_data, test_data = report_datasets(cfg)
    options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
                   spin_flip_augment=cfg.data.spin_flip_augment, sampling_seed=12345)
    train = collect_trajectories(loaded.model, train_data.configurations, **options)
    test = collect_trajectories(loaded.model, test_data.configurations, **options)
    repeats = []
    for seed in range(args.sampling_repeats):
        draw = collect_trajectories(loaded.model, test_data.configurations, **dict(options, sampling_seed=seed))
        repeats.append(closure_details(loaded.model, draw, train['direct']))
    history = loaded.payload.get('history')
    if history is None:
        history = json.loads((args.run/'history_progress.json').read_text())
    history = [row for row in history if row.get('epoch', 0) <= args.checkpoint]
    args.output.mkdir(parents=True, exist_ok=False)
    return write_report(loaded, train, test, train_data.couplings.numpy(), test_data.couplings.numpy(),
        history, args.output, repeats=dict(seeds=list(range(args.sampling_repeats)), closure=repeats,
            caution='同一模型的随机 C 诊断，不是独立训练 seed 或 λ/ν 置信区间。'),
        data_sha256=manifest['data_sha256'], lattice_sizes=[train_data.configurations.shape[2]//cfg.model.block_factor**k
            for k in range(train['direct'].shape[1])])


if __name__ == '__main__':
    main()
