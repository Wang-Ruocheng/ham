"""独立初始尺寸、相同末尺度诊断：只读取保存数组，不加载模型或构型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_mcrg import BASES, RCONDS, bootstrap_response, covariance_pair, solve_response
from scripts.report_ising_zh import chinese_figures


ARMS = ('dim1_control', 'dim1_natural', 'dim3_control', 'dim3_natural')
LABELS = {'dim1_control': '一维原表示', 'dim1_natural': '一维自然耦合',
          'dim3_control': '三维原表示', 'dim3_natural': '三维自然耦合'}
KC = .44068679350977147
CAUTION = ('独立初始尺寸、共同末尺度只检验有限尺度一致性；均值兼容不证明全分布相同，'
           '差异不能单独归因于有限尺寸或 Q 错误。仅单个训练种子、零磁场和有限算符基；不认证热模。')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.complexfloating, complex)):
        return {'real': float(value.real), 'imag': float(value.imag)}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def bootstrap_indices(replicas, repeats, seed):
    if replicas < 1 or repeats < 1:
        raise ValueError('Need positive physical replica and bootstrap counts')
    return np.random.default_rng(seed).integers(replicas, size=(repeats, replicas))


def mean_interval(values, indices):
    """values 首轴是物理系综；C 抽样必须在调用前平均。"""
    return np.percentile(np.asarray(values)[indices].mean(1), [2.5, 50, 97.5], axis=0)


def independent_difference(left, right, left_indices, right_indices):
    """不同初始尺寸独立重采；相同编号 replica 绝不伪配对。"""
    left, right = np.asarray(left), np.asarray(right)
    if len(left_indices) != len(right_indices):
        raise ValueError('Independent bootstrap panels must have the same repeat count')
    draws = left[left_indices].mean(1) - right[right_indices].mean(1)
    dimension = max(1, int(np.prod(left.shape[1:])))
    return dict(mean=left.mean(0) - right.mean(0),
                bootstrap_percentiles=np.percentile(draws, [2.5, 50, 97.5], axis=0),
                bonferroni_coordinate_family_percentiles=np.percentile(draws,
                    [2.5 / dimension, 50, 100 - 2.5 / dimension], axis=0),
                coordinate_family_dimension=dimension,
                bootstrap_method='两独立尺寸分别重采物理系综；先平均每系综的 C 抽样，不按编号配对。',
                interval_caution='普通区间为点态95%；Bonferroni仅覆盖本行坐标族，不覆盖所有K/尺度/模型的联合比较；仅有限重采精度。')


def train_norm(metrics, lattice, *, smoke=False):
    """按实际格点大小映射固定 TRAIN 标准差，绝不按输入文件的 scale 编号。"""
    sizes = list(metrics['lattice_sizes'])
    if lattice in sizes:
        scale = sizes.index(lattice)
    elif smoke:
        scale = int(np.argmin(np.abs(np.asarray(sizes) - lattice)))
    else:
        raise ValueError(f'No fixed TRAIN normalization for lattice {lattice}')
    row = next(r for r in metrics['train_representation'] if r['scale'] == scale)
    pooled = float(row['pooled_coordinate_std'])
    coordinate = np.asarray(row['coordinate_std'], dtype=float)
    if pooled <= 0 or not np.isfinite(pooled) or np.any(coordinate <= 0) or not np.isfinite(coordinate).all():
        raise ValueError('Need positive frozen TRAIN coordinate deviations')
    return dict(lattice=lattice, source_lattice=sizes[scale], source_scale=scale,
                pooled_coordinate_std=pooled, coordinate_std=coordinate,
                smoke_nearest_substitute=bool(smoke and lattice not in sizes))


def load_collection(folder):
    folder = Path(folder)
    status = read_json(folder / 'status.json')
    if status.get('status') != 'complete' or status.get('workflow_complete') is not True:
        raise ValueError(f'Incomplete collection: {folder}')
    with np.load(folder / 'operator_statistics.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    g = arrays['g']
    if g.ndim != 5 or g.shape[-1] not in (1, 3) or g.shape[3] not in (2, 3):
        raise ValueError('Need g[K,physical replica,C draw,scale,d] with depth one or two')
    nk, nr, nc, levels, dim = g.shape
    if arrays['lattice_sizes'].shape != (levels,) or arrays['K'].shape != (nk,):
        raise ValueError('Lattice and K axes do not match g')
    if arrays['operators'].shape[:4] != (nk, nr, nc, levels) or arrays['operators'].shape[-1] != 7:
        raise ValueError('Need seven saved per-configuration physical operators')
    expected = {'full_DR': g.shape[:-1] + (dim, dim),
                'r_fd_jacobian': g.shape[:-1] + (dim, dim),
                'r_of_g': g.shape, 'r_of_mean_g': (nk, levels, dim),
                'one_step_pred': (nk, nr, nc, levels - 1, dim),
                'all_start_horizon_pred': (nk, nr, nc, levels, levels, dim),
                'q_response_in': (nk, nr, nc, levels - 1, dim, 7),
                'q_response_out': (nk, nr, nc, levels - 1, dim, 7),
                'q_response_predicted': (nk, nr, nc, levels - 1, dim, 7),
                'score_std': (nk, nr, nc, levels, 7)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Unexpected shape or nonfinite values: {key}, expected {shape}')
    if not np.isfinite(g).all() or np.any(np.diff(arrays['lattice_sizes']) >= 0):
        raise ValueError('Need finite g and decreasing physical lattice sizes')
    return arrays, read_json(folder / 'diagnostic.json')


def closure_analysis(arrays, metrics, *, smoke=False):
    g, pred = arrays['g'], arrays['all_start_horizon_pred']
    sizes = arrays['lattice_sizes'].astype(int)
    rows = []
    for ik, coupling in enumerate(arrays['K']):
        for start in range(len(sizes) - 1):
            for end in range(start + 1, len(sizes)):
                norm = train_norm(metrics, int(sizes[end]), smoke=smoke)
                delta = pred[ik, :, :, start, end] - g[ik, :, :, end]
                coordinate_rmse = np.sqrt(np.mean(delta ** 2, axis=(0, 1)))
                rmse = np.sqrt(np.mean(delta ** 2))
                row = dict(K=float(coupling), K_index=ik, start=start, end=end,
                           path=f'{sizes[start]}→{sizes[end]}',
                           updates_of_R=end - start, rmse=rmse,
                           nrmse=rmse / norm['pooled_coordinate_std'],
                           coordinate_rmse=coordinate_rmse,
                           coordinate_nrmse=coordinate_rmse / norm['coordinate_std'],
                           fixed_train_normalization=norm,
                           per_physical_replica_rmse=np.sqrt(np.mean(delta ** 2, axis=(1, 2))))
                if end == start + 1:
                    mean_next = g[ik, :, :, end].mean((0, 1))
                    mean_R = arrays['r_of_g'][ik, :, :, start].mean((0, 1))
                    R_mean = arrays['r_of_mean_g'][ik, start]
                    row.update(mean_R_of_g_minus_mean_next=mean_R - mean_next,
                               R_of_mean_g_minus_mean_next=R_mean - mean_next,
                               jensen_mean_R_minus_R_of_mean=mean_R - R_mean,
                               mean_next_g=mean_next,
                               distinction='R(mean g)、mean R(g) 与真实下一尺度编码分别列出；后两者不可互换。')
                rows.append(row)
    return rows


def response_analysis(arrays, indices):
    vin, target, predicted = [arrays[k] for k in ('q_response_in', 'q_response_out', 'q_response_predicted')]
    sigma = arrays['score_std'][..., :-1, :6]
    divisor = np.where(sigma > 1e-12, sigma, np.inf)[..., None, :]
    x, y, p = [a[..., :6] / divisor for a in (vin, target, predicted)]
    best = arrays['best_J_even'] @ x
    rows = []
    for ik, coupling in enumerate(arrays['K']):
        for step in range(vin.shape[3]):
            sl = (ik, slice(None), slice(None), step)
            rr, rb, yy = p[sl] - y[sl], best[sl] - y[sl], y[sl]
            # Frobenius norm over C draws, coordinates and six score directions.
            denominator = np.sqrt(np.sum(yy ** 2, axis=(1, 2, 3)))
            relative = lambda e: np.divide(np.sqrt(np.sum(e ** 2, axis=(1, 2, 3))), denominator,
                out=np.full_like(denominator, np.nan), where=denominator > 1e-15)
            rrel, brel = relative(rr), relative(rb)
            per_draw_denominator = np.linalg.norm(yy, axis=(-2, -1))
            relative_per_draw = lambda e: np.divide(np.linalg.norm(e, axis=(-2, -1)),
                per_draw_denominator, out=np.full_like(per_draw_denominator, np.nan),
                where=per_draw_denominator > 1e-15)
            replica_mean_draw_R = relative_per_draw(rr).mean(1)
            replica_mean_draw_best = relative_per_draw(rb).mean(1)
            row = dict(K_index=ik, K=float(coupling), transition=step,
                path=f'{arrays["lattice_sizes"][step]}→{arrays["lattice_sizes"][step + 1]}',
                full_DR_relative_frobenius=float(np.linalg.norm(rr) / np.linalg.norm(yy)) if np.linalg.norm(yy) > 1e-15 else None,
                best_local_matrix_relative_frobenius=float(np.linalg.norm(rb) / np.linalg.norm(yy)) if np.linalg.norm(yy) > 1e-15 else None,
                full_DR_relative_per_physical_replica=rrel,
                best_local_matrix_relative_per_physical_replica=brel,
                full_DR_mean_replica_relative_percentiles=mean_interval(rrel, indices),
                best_matrix_mean_replica_relative_percentiles=mean_interval(brel, indices),
                existing_R_relative_error_mean=float(replica_mean_draw_R.mean()),
                best_J_relative_error_mean=float(replica_mean_draw_best.mean()),
                existing_R_relative_error_per_replica=replica_mean_draw_R,
                best_J_relative_error_per_replica=replica_mean_draw_best,
                existing_R_relative_error_percentiles=mean_interval(replica_mean_draw_R, indices),
                best_J_relative_error_percentiles=mean_interval(replica_mean_draw_best, indices),
                C_draw_aggregation='existing_R/best_J 字段与阶段二一致：逐 C 抽样相对范数后先平均每物理系综；full_DR_relative_frobenius 另列整个面板范数比，不能混用。',
                signed_full_DR_residual_mean=rr.mean((0, 1)),
                signed_best_matrix_residual_mean=rb.mean((0, 1)),
                score_zero_fraction=float(np.mean(sigma[sl] <= 1e-12)),
                thermal_nu=None,
                definition='六偶方向按同尺度每系综 score 标准差归一；相对 Frobenius 范数，不是 MSE；最佳矩阵在每系综每 C 抽样独立拟合。')
            for key in ('response_input_rank_even', 'response_input_rank_all',
                        'response_input_singular_values_even', 'response_input_singular_values_all',
                        'response_input_effective_rank_even', 'response_input_effective_rank_all',
                        'response_input_condition_even', 'response_input_condition_all'):
                if key in arrays:
                    row[key] = arrays[key][sl]
            ranks = np.asarray(row.get('response_input_rank_even', vin.shape[-2]))
            if not np.isfinite(rrel).any() or np.max(ranks) == 0:
                row['status'] = 'unidentifiable'
            elif np.any(ranks < vin.shape[-2]):
                row['status'] = 'rank_deficient_minimum_norm_extension_not_physical_spectrum'
            else:
                row['status'] = 'numeric_response_identified_not_thermal_certification'
            rows.append(row)
    fd = {}
    for side, expected in [('in', vin), ('out', target)]:
        actual = arrays['q_fd_response_' + side]
        expected = np.expand_dims(expected, axis=-3)
        fd[side] = dict(max_absolute_error=float(np.max(abs(actual - expected))),
            relative_frobenius_error=float(np.linalg.norm(actual - expected) /
                max(np.linalg.norm(np.broadcast_to(expected, actual.shape)), 1e-300)))
    rdiff = arrays['full_DR'] - arrays['r_fd_jacobian']
    fd['full_DR_h_1e_5'] = dict(max_absolute_error=float(np.max(abs(rdiff))),
        relative_frobenius_error=float(np.linalg.norm(rdiff) / max(np.linalg.norm(arrays['full_DR']), 1e-300)))
    return rows, fd


def spectra_analysis(arrays, indices):
    A = arrays['cross_covariance'].mean(2)
    B = arrays['covariance'].mean(2)[:, :, 1:]
    families = [('isotropic', A, B, [(str(n), list(range(n))) for n in BASES]),
                ('mixed', A, B, [('6_even_plus_M', list(range(7)))])]
    if 'orientation_operators' in arrays:
        oa, ob = covariance_pair(arrays['orientation_operators'])
        families.append(('orientation', oa.mean(2), ob.mean(2),
                         [('axes_2', [0, 1]), ('oriented_7', list(range(7)))]))
    rows = []
    for family, fa, fb, selections in families:
        for ik, coupling in enumerate(arrays['K']):
            for step in range(A.shape[2]):
                for basis, selected in selections:
                    aa = fa[ik, :, step][:, selected][:, :, selected]
                    bb = fb[ik, :, step][:, selected][:, :, selected]
                    for rcond in RCONDS:
                        row = dict(family=family, basis=basis, K_index=ik, K=float(coupling),
                                   transition=step, rcond=rcond, operator_indices=selected, thermal_nu=None)
                        try:
                            row.update(status='identified_numeric_response',
                                **solve_response(aa.mean(0), bb.mean(0), rcond, 6 if family == 'mixed' else None))
                            row['bootstrap'] = bootstrap_response(aa, bb, indices, rcond,
                                6 if family == 'mixed' else None)
                            if row['retained_rank'] < len(selected):
                                row['status'] = 'rank_truncated_not_full_physical_identification'
                        except (ValueError, np.linalg.LinAlgError) as error:
                            row.update(status='unidentifiable', error=f'{type(error).__name__}: {error}',
                                       eigenvalues=[], bootstrap=None)
                        rows.append(row)
    return rows


def shared_scale_analysis(smaller, larger, metrics, small_indices, large_indices, *, smoke=False):
    rows = []
    shared = sorted(set(smaller['lattice_sizes']) & set(larger['lattice_sizes']), reverse=True)
    if not np.array_equal(smaller['K'], larger['K']):
        raise ValueError('Independent-size collections must have identical evaluation K values')
    for size in shared:
        a = list(smaller['lattice_sizes']).index(size)
        b = list(larger['lattice_sizes']).index(size)
        norm = train_norm(metrics, int(size), smoke=smoke)
        for ik, coupling in enumerate(smaller['K']):
            sg = smaller['g'][ik, :, :, a].mean(1)
            lg = larger['g'][ik, :, :, b].mean(1)
            # Both systems now have exactly the same number of spatial sites.
            sop = smaller['operators'][ik, :, :, a, :, :6].mean((1, 2)) / size ** 2
            lop = larger['operators'][ik, :, :, b, :, :6].mean((1, 2)) / size ** 2
            gd = independent_difference(lg, sg, large_indices, small_indices)
            od = independent_difference(lop, sop, large_indices, small_indices)
            rows.append(dict(lattice=int(size), K_index=ik, K=float(coupling),
                smaller_block_depth=a, larger_block_depth=b,
                g_larger_minus_smaller=gd,
                g_difference_l2=float(np.linalg.norm(gd['mean'])),
                g_difference_coordinate_nrmse=float(np.sqrt(np.mean((gd['mean'] / norm['coordinate_std']) ** 2))),
                fixed_train_normalization=norm,
                physical_even_operator_density_larger_minus_smaller=od,
                physical_density_difference_l2=float(np.linalg.norm(od['mean'])),
                thermal_nu=None, caution=CAUTION))
    return rows


def repeat_sampling_analysis(new, old, new_indices, old_indices):
    """同初始尺寸独立新旧采样，模型和旧数组完全只读。"""
    rows = []
    if not np.array_equal(new['K'], old['K']) or new['g'].shape[-2:] != old['g'].shape[-2:]:
        return dict(status='not_comparable', reason='旧公开数组的 K 或尺度/维度不一致。')
    for ik, coupling in enumerate(new['K']):
        for step, size in enumerate(new['lattice_sizes']):
            ng = new['g'][ik, :, :, step].mean(1)
            og = old['g'][ik, :, :, step].mean(1)
            no = new['operators'][ik, :, :, step, :, :6].mean((1, 2)) / size ** 2
            oo = old['operators'][ik, :, :, step, :, :6].mean((1, 2)) / size ** 2
            rows.append(dict(K=float(coupling), lattice=int(size),
                g_new_minus_original=independent_difference(ng, og, new_indices, old_indices),
                physical_even_density_new_minus_original=independent_difference(no, oo, new_indices, old_indices)))
    return dict(status='complete', rows=rows,
        caution='同尺寸独立采样复验，不是新训练种子复现；不读取原始物理缓存。')


def root_distances(arrays, metrics):
    """仅描述已保存根到新采集点的距离；绝不重搜或认证支持。"""
    rows = []
    means = arrays['g'].mean((1, 2))
    for root in metrics.get('fixed_points', {}).get('roots', []):
        point = np.asarray(root['g'], dtype=float)
        for scale, lattice in enumerate(arrays['lattice_sizes']):
            distances = np.linalg.norm(means[:, scale] - point, axis=-1)
            nearest = int(np.argmin(distances))
            rows.append(dict(root_id=root.get('root_id'), lattice=int(lattice),
                nearest_sampled_K=float(arrays['K'][nearest]),
                nearest_K_mean_g_distance=float(distances[nearest]),
                per_K_mean_g_distance=distances,
                exact_inverse_K_recomputed=False, thermal_nu=None,
                caution='仅对三个采集 K 点均值测距，不是折线投影或精确逆 K，也不认证数据支持。'))
    return rows


def draw(summary, collections, output):
    kindex = int(np.argmin(abs(np.asarray(summary['K']) - KC)))
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), layout='constrained')
    for i, arm in enumerate(ARMS):
        entry = summary['arms'][arm]
        shared = [r for r in entry['shared_scales'] if r['K_index'] == kindex]
        x = np.arange(len(shared))
        axes[i, 0].bar(x - .18, [r['g_difference_l2'] for r in shared], .36, label='编码均值差范数')
        axes[i, 0].bar(x + .18, [r['physical_density_difference_l2'] for r in shared], .36, label='六偶算符密度差范数')
        axes[i, 0].set(xticks=x, xticklabels=[str(r['lattice']) for r in shared], xlabel='共同格点边长', ylabel='不同单位分别解释', title=LABELS[arm] + '：同末尺度差异')
        closure_paths = list(dict.fromkeys(row['path'] for size in reversed(summary['initial_sizes'])
            for row in entry['sizes'][str(size)]['closure'] if row['K_index'] == kindex))
        response_paths = list(dict.fromkeys(row['path'] for size in reversed(summary['initial_sizes'])
            for row in entry['sizes'][str(size)]['response'] if row['K_index'] == kindex))
        for size, marker in zip(summary['initial_sizes'], ('o', 's')):
            data = entry['sizes'][str(size)]
            rows = [r for r in data['closure'] if r['K_index'] == kindex]
            axes[i, 1].plot([closure_paths.index(r['path']) for r in rows],
                            [r['nrmse'] for r in rows], marker=marker, label=f'初始边长 {size}')
            rr = [r for r in data['response'] if r['K_index'] == kindex]
            positions = [response_paths.index(r['path']) for r in rr]
            axes[i, 2].plot(positions, [r['full_DR_relative_frobenius'] for r in rr], marker=marker, label=f'{size}：神经雅可比')
            axes[i, 2].plot(positions, [r['best_local_matrix_relative_frobenius'] for r in rr], marker=marker, ls=':', label=f'{size}：局部最佳矩阵')
        axes[i, 1].set(title='闭环误差：固定原训练集标尺', xlabel='实际闭环路径', ylabel='归一均方根误差',
                       xticks=np.arange(len(closure_paths)), xticklabels=closure_paths)
        axes[i, 2].set(title='六方向物理响应残差', xlabel='实际变换路径', ylabel='相对弗罗贝尼乌斯范数',
                       xticks=np.arange(len(response_paths)), xticklabels=response_paths)
        for ax in axes[i]:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('第三阶段：独立尺寸与相同末尺度诊断（临界参考 K；不认证热指数）')
    fig.savefig(output / 'comparison.png', dpi=150); plt.close(fig)

    for kind in ('coupling_by_scale', 'R_vs_g', 'R_displacement'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        for arm, ax in zip(ARMS, axes.flat):
            arrow_bounds = []
            for size_index, (size, marker) in enumerate(zip(summary['initial_sizes'], ('o', 's'))):
                a = collections[arm][size]
                g, rg = a['g'].mean((1, 2)), a['r_of_g'].mean((1, 2))
                dim = g.shape[-1]
                for scale, lattice in enumerate(a['lattice_sizes']):
                    if kind == 'coupling_by_scale':
                        for j in range(dim):
                            ax.plot(a['K'], g[:, scale, j], marker=marker, label=f'{size}起点→{lattice}，g{j + 1}')
                    elif kind == 'R_vs_g':
                        sampled_g = a['g'][..., scale, :].reshape(-1, dim)
                        sampled_rg = a['r_of_g'][..., scale, :].reshape(-1, dim)
                        for j in range(dim):
                            ax.scatter(sampled_g[:, j], sampled_rg[:, j], marker=marker, s=8, alpha=.35,
                                       label=f'{size}起点→{lattice}，分量{j + 1}')
                    elif dim == 1:
                        sampled_g = a['g'][..., scale, 0].ravel()
                        sampled_rg = a['r_of_g'][..., scale, 0].ravel()
                        ax.scatter(sampled_g, sampled_rg - sampled_g, marker=marker, s=8, alpha=.35,
                                   label=f'{size}起点→{lattice}')
                    else:
                        sampled_g = a['g'][..., scale, :].reshape(-1, dim)
                        sampled_rg = a['r_of_g'][..., scale, :].reshape(-1, dim)
                        delta = sampled_rg - sampled_g
                        ax.quiver(sampled_g[:, 0], sampled_g[:, 1], delta[:, 0], delta[:, 1],
                                  angles='xy', scale_units='xy', scale=1, alpha=.3,
                                  color=f'C{size_index * 3 + scale}', width=.003,
                                  label=f'{size}起点→{lattice}')
                        arrow_bounds.extend((sampled_g[:, :2], sampled_rg[:, :2]))
            if kind == 'coupling_by_scale':
                ax.axvline(KC, color='purple', ls=':', label='理论临界 K（仅参考）')
                ax.set(xlabel='耦合 K', ylabel='编码 g 各分量')
            elif kind == 'R_vs_g':
                lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], 'k:', label='恒等映射')
                ax.set(xlabel='实际采集的 g 分量', ylabel='对应 R(g) 分量')
            elif collections[arm][summary['initial_sizes'][0]]['g'].shape[-1] == 1:
                ax.axhline(0, color='black', lw=.7); ax.set(xlabel='g', ylabel='真实位移 R(g)−g')
            else:
                ax.set(xlabel='g1', ylabel='g2：箭头为真实位移的前两分量')
                points = np.concatenate(arrow_bounds)
                lower, upper = points.min(0), points.max(0)
                padding = np.maximum(upper - lower, .01) * .08
                ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
                ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
                ax.text(.02, .02, '三维图为二维投影；第三分量保存在数值报告中。', transform=ax.transAxes, fontsize=7)
            ax.set_title(LABELS[arm]); ax.grid(alpha=.2); ax.legend(fontsize=6, ncol=2)
        fig.suptitle({'coupling_by_scale': '独立尺寸的耦合到编码映射（三点直连，无平滑）',
                      'R_vs_g': '实际采集点上的神经 R 映射（非额外场扫描）',
                      'R_displacement': '实际采集点上的真实位移（无根重搜、无热模筛选）'}[kind])
        fig.savefig(output / (kind + '.png'), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for arm, ax in zip(ARMS, axes.flat):
        for size, color in zip(summary['initial_sizes'], ('tab:blue', 'tab:orange')):
            rows = summary['arms'][arm]['sizes'][str(size)]['spectra']
            for row in rows:
                if row['family'] != 'isotropic' or row['K_index'] != kindex or row['rcond'] != 1e-6:
                    continue
                for real, imag in row['eigenvalues']:
                    ax.scatter(int(row['basis']) + row['transition'] * .08, real, color=color,
                               marker='o' if abs(imag) < 1e-8 else 'x', alpha=.6,
                               label=f'{size} 起点' if row['basis'] == '1' and row['transition'] == 0 else None)
        ax.axhline(2, color='purple', ls=':', label='理论热倍率（仅参考）')
        ax.set(title=LABELS[arm], xlabel='偶算符基数量', ylabel='全部特征值的实部（复根用叉号）', xticks=BASES)
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('独立物理响应谱：图示固定阈值；全部复谱、基、阈值与重采结果保留在数值报告')
    fig.savefig(output / 'mcrg_basis.png', dpi=150); plt.close(fig)


def report(root, phase2_root, output, bootstraps=1000, smoke=False):
    root, phase2_root, output = map(Path, (root, phase2_root, output))
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite historical report: {output}')
    if bootstraps < 1:
        raise ValueError('Need at least one grouped bootstrap')
    sizes = (8, 16) if smoke else (54, 108)
    collections, collection_metadata, summary = {}, {}, dict(schema_version=1, status='running', root=str(root),
        phase2_root=str(phase2_root), initial_sizes=list(sizes), arms={},
        new_training_updates=0, no_model_loading=True, no_extra_inference=True, no_MI=True,
        thermal_nu=None, thermal_mode_certified=False, smoke=bool(smoke),
        bootstrap_repeats=bootstraps,
        bootstrap='物理系综为分组单位，先平均 C；独立尺寸和新旧采样独立重采，同一尺寸各模型共用重采索引。',
        caution=CAUTION)
    # Validate every input before creating an output, so incomplete collections never appear complete.
    for arm in ARMS:
        collections[arm] = {}
        collection_metadata[arm] = {}
        for size in sizes:
            arrays, diagnostic = load_collection(root / 'arms' / arm / f'L{size}' / 'collection')
            collections[arm][size] = arrays
            collection_metadata[arm][size] = diagnostic
    first = collections[ARMS[0]][sizes[0]]
    summary['K'] = first['K'].tolist()
    indices = {size: bootstrap_indices(collections[ARMS[0]][size]['g'].shape[1], bootstraps,
                                       2026101600 + size) for size in sizes}
    for arm in ARMS:
        metrics = read_json(phase2_root / arm / 'report' / 'metrics.json')
        entry = dict(sizes={}, phase2_reference_metrics=metrics,
                     reference_caution='原12000步的全部根、完整 DR 谱、精确逆 K 分支与投影原样保留；不重新搜索，不跨臂匹配根。')
        for size in sizes:
            arrays = collections[arm][size]
            if not np.array_equal(first['K'], arrays['K']) or arrays['g'].shape[1] != len(indices[size][0]):
                raise ValueError('Every arm must share the same K/physical-replica panels')
            response, fd = response_analysis(arrays, indices[size])
            eig = np.linalg.eigvals(arrays['full_DR'])
            entry['sizes'][str(size)] = dict(lattice_sizes=arrays['lattice_sizes'],
                collection_provenance=collection_metadata[arm][size],
                closure=closure_analysis(arrays, metrics, smoke=smoke), response=response,
                finite_difference_validation=fd, spectra=spectra_analysis(arrays, indices[size]),
                saved_full_DR=arrays['full_DR'], saved_full_DR_all_eigenvalues=eig,
                saved_g=arrays['g'], saved_R_of_g=arrays['r_of_g'],
                saved_R_displacement=arrays['r_of_g'] - arrays['g'],
                chain_seeds=arrays.get('chain_seeds'),
                original_root_distances_to_new_points=root_distances(arrays, metrics), thermal_nu=None)
        entry['shared_scales'] = shared_scale_analysis(collections[arm][sizes[0]],
            collections[arm][sizes[1]], metrics, indices[sizes[0]], indices[sizes[1]], smoke=smoke)
        old_file = phase2_root / arm / 'response' / 'operator_statistics.npz'
        if not smoke and old_file.exists():
            with np.load(old_file, allow_pickle=False) as old_saved:
                old = {key: old_saved[key] for key in ('K', 'g', 'operators')}
            old_indices = bootstrap_indices(old['g'].shape[1], bootstraps, 2026101699)
            entry['independent_same_initial_size_replication'] = repeat_sampling_analysis(
                collections[arm][sizes[1]], old, indices[sizes[1]], old_indices)
        else:
            entry['independent_same_initial_size_replication'] = dict(status='unavailable',
                reason='执行检查使用缩小尺寸，或缺少旧公开数组；不打开原物理数据。')
        summary['arms'][arm] = entry
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='running', workflow_complete=False))
    try:
        with chinese_figures():
            draw(summary, collections, output)
        copied = []
        for arm in ARMS:
            source = phase2_root / arm / 'report' / 'loss_comparison.png'
            if source.exists():
                name = arm + '_original_training_loss_reference.png'
                shutil.copyfile(source, output / name); copied.append(name)
        summary.update(status='complete', report_complete=True, workflow_complete=True,
                       original_training_loss_reference_files=copied,
                       loss_note='仅复制原训练曲线作为参考；第三阶段没有训练、优化或新损失曲线。')
        write_json(output / 'comparison.json', summary)
        write_json(output / 'report.json', dict(status='complete', report_complete=True,
            workflow_complete=True, new_training_updates=0, thermal_nu=None, smoke=bool(smoke),
            files=sorted(p.name for p in output.iterdir() if p.name != 'status.json') + ['report.json'],
            caution=CAUTION))
        write_json(output / 'status.json', dict(status='complete', workflow_complete=True,
                                               report_complete=True, smoke=bool(smoke)))
        return clean(summary)
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', workflow_complete=False,
                                                error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--phase2-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true')
    return report(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
