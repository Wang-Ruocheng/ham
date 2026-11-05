"""冻结二维 Ising K/B 高维表示；有界 CPU、完整坐标求根及几何诊断。

只读已有物理缓存、checkpoint 和损失日志，不训练、不生成物理样本、
不重估 MI。PCA 只在 TRAIN 的 R 输入尺度拟合；KB 标签仅用于评价。
局部二维压缩与法向泄漏都是几何诊断，不认证物理维数或自治流形。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from scripts.dual_ising_grid_diagnostics import grid_encodings
from scripts.report_ising_dimg_ablation import mapped, plot_losses, validate_completed_run
from scripts.report_ising_highdim import (
    architecture_diagnostics, fit_training_pca, full_r_jacobian,
    model_state_sha256, project_displacement, project_points,
    representation_statistics, select_training_seeds, sha256, stability_counts,
    training_inputs, training_support,
)
from scripts.report_ising_highdim_fast import (
    AnalysisBudgetHold, Deadline, Progress, atomic_json, atomic_npz, single_thread_control,
)
from scripts.report_ising_zh import chinese_figures

MAX_SEEDS, MAX_NFEV, MAX_ROOTS = 64, 200, 16
ROOT_TOLERANCE = 1e-8
LOCAL_NEIGHBORS = 24


def _check(deadline):
    if deadline is not None:
        deadline.check()


def _progress(progress, stage, completed, total, extra=None):
    if progress is not None:
        progress(stage, completed, total, extra)


class ProgressChunks:
    """Preserve the standard collector's one RNG context and exact chunk order."""
    def __init__(self, configurations, *, deadline=None, progress=None, stage='编码'):
        self.configurations, self.deadline = configurations, deadline
        self.progress, self.stage = progress, stage

    def split(self, chunk_size):
        total, completed = len(self.configurations), 0
        _progress(self.progress, self.stage, 0, total)
        for chunk in self.configurations.split(chunk_size):
            _check(self.deadline)
            # Bounded validation, never a full-cache float64 copy.
            if not bool(((chunk == 1) | (chunk == -1)).all()):
                raise ValueError('物理缓存必须是有限的二值自旋')
            yield chunk
            completed += len(chunk)
            _progress(self.progress, self.stage, completed, total)
            _check(self.deadline)


def collect_bounded(model, configurations, *, deadline=None, progress=None, stage='编码', **options):
    """Standard collect_trajectories with per-chunk progress/deadline checks."""
    _check(deadline)
    data = ProgressChunks(configurations, deadline=deadline, progress=progress, stage=stage)
    result = collect_trajectories(model, data, **options)
    _check(deadline)
    return result


def report_datasets(cfg):
    """Read mmap cache metadata; no fallback to physical sampling or replica pooling."""
    if cfg.data.train_pool_replicas != 1 or cfg.data.test_pool_replicas != 1:
        raise ValueError('报告保留每个独立系综，不能合并物理副本')
    payload = torch.load(str(Path(cfg.data.cache)), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(payload, (tuple, list)) and len(payload) == 4:
        tx, ty, vx, vy = payload
        pairs = ((tx, ty), (vx, vy))
    elif isinstance(payload, dict) and payload.get('schema_version') == 1:
        pairs = [(payload[split]['configurations'], payload[split]['couplings']) for split in ('train', 'test')]
    else:
        raise ValueError('需要已有的受支持缓存；不会重新采样')
    datasets, cells = [], []
    for configurations, couplings in pairs:
        if (configurations.ndim != 5 or configurations.shape[-1] != 1 or
                configurations.shape[2] != configurations.shape[3] or
                cfg.evaluation.samples > configurations.shape[1] or
                cfg.training.sample_batch > configurations.shape[1] or couplings is None or
                tuple(couplings.shape) != (len(configurations), 2)):
            raise ValueError('缓存必须为单层二维构型与两列 K/B 标签')
        if configurations.shape[2] % cfg.model.block_factor**cfg.evaluation.max_rg_steps:
            raise ValueError('所有请求尺度必须可整除粗粒化块尺寸')
        labels = couplings.detach().cpu().numpy()
        if not np.isfinite(labels).all():
            raise ValueError('K/B 标签必须有限')
        cells.append(set(map(tuple, labels.tolist())))
        datasets.append(SimpleNamespace(configurations=configurations, couplings=couplings))
    if datasets[0].configurations.shape[1:] != datasets[1].configurations.shape[1:]:
        raise ValueError('TRAIN/TEST 物理几何不一致')
    if cells[0] & cells[1]:
        raise ValueError('TRAIN/TEST 必须按完整 K/B 参数格点分离')
    return tuple(datasets)


def grid_statistics(direct, labels):
    labels, direct = np.asarray(labels), np.asarray(direct)
    if not np.isfinite(direct).all() or not np.isfinite(labels).all():
        raise ValueError('编码和标签必须有限')
    axes, means = grid_encodings(labels, direct)
    counts = np.zeros(means.shape[:2], dtype=int)
    within = np.zeros((*means.shape[2:-1], means.shape[-1], means.shape[-1]))
    degrees = 0
    for i, K in enumerate(axes[0]):
        for j, B in enumerate(axes[1]):
            group = direct[(labels[:, 0] == K) & (labels[:, 1] == B)]
            counts[i, j] = len(group)
            residual = group-means[i, j]
            within += np.einsum('nsd,nse->sde', residual, residual)
            degrees += max(len(group)-1, 0)
    within = within/degrees if degrees else np.zeros_like(within)
    return axes, means, dict(replicas_per_cell=counts.tolist(),
        within_cell_covariance_by_scale=within.tolist(),
        within_cell_covariance_degrees_of_freedom=degrees,
        within_cell_covariance_eigenvalues_by_scale=np.linalg.eigvalsh(within).clip(min=0).tolist(),
        caution='同参数物理副本差异含有限采样误差及随机 C；不能仅据此分离物理波动与 C 噪声。')


def local_geometry(model, train, *, neighbors=LOCAL_NEIGHBORS, deadline=None, progress=None):
    """TRAIN-only per-scale local PCA; T2 is a diagnostic, not a rank claim."""
    from scipy.spatial import cKDTree
    train = np.asarray(train, dtype=float)
    rows = []
    for scale in range(train.shape[1]):
        points = train[:, scale]
        count = min(neighbors, len(points))
        tree = cKDTree(points)
        indices = tree.query(points, k=count)[1]
        indices = np.asarray(indices).reshape(len(points), count)
        spectrum, tail_fraction, compressed, leakage = [], [], [], []
        for index, local_indices in enumerate(indices):
            _check(deadline)
            local = points[local_indices]
            _, singular, vectors = np.linalg.svd(local-local.mean(0), full_matrices=False)
            variance = singular**2/max(count-1, 1)
            padded = np.zeros(train.shape[-1]); padded[:len(variance)] = variance
            spectrum.append(padded)
            total = float(variance.sum())
            tail_fraction.append(float(variance[2:].sum()/total) if total else None)
            if scale < train.shape[1]-1:
                # No numerical-rank-to-physical-dimension inference is made.
                tangent = vectors[:min(2, len(vectors))].T
                x = torch.tensor(points[index], dtype=torch.float64, requires_grad=True)
                jacobian = torch.autograd.functional.jacobian(model.advance_theory, x).detach().numpy()
                image = jacobian@tangent
                projected = tangent.T@image
                normal = image-tangent@projected
                compressed.append(projected.tolist())
                norm = float(np.linalg.norm(image))
                leakage.append(float(np.linalg.norm(normal)/norm) if norm else None)
        rows.append(dict(scale=scale, neighbors=count,
            covariance_eigenvalues_descending=np.asarray(spectrum).tolist(),
            variance_fraction_outside_T2=tail_fraction,
            T2_compressed_DR=compressed, T2_normal_leakage_fraction=leakage,
            fit_source='同尺度 TRAIN 编码的24近邻；不使用标签或 TEST',
            caution='近邻谱含曲率、密度变化、有限构型噪声及随机 C；T2 为固定二维几何压缩，不认证流形维数、物理方向或自治映射。'))
        _progress(progress, '局部几何与二维压缩', scale+1, train.shape[1])
    return rows


def kb_tangent_residuals(model, axes, means):
    """Full and tangent/normal residuals use the target scale's KB tangent."""
    derivatives = np.stack(np.gradient(means, *axes, axis=(0, 1), edge_order=1), axis=-1)
    left, singular, _ = np.linalg.svd(derivatives, full_matrices=False)
    # A declared relative threshold avoids treating machine precision as physics.
    mask = singular > np.maximum(singular[..., :1]*1e-3, 1e-10)
    tangent = left*mask[..., None, :]
    prediction = mapped(model, means)
    displacement = prediction-means
    parallel = np.einsum('...di,...d->...i', tangent, displacement)
    parallel = np.einsum('...di,...i->...d', tangent, parallel)
    error = prediction[:, :, :-1]-means[:, :, 1:]
    target_tangent = tangent[:, :, 1:]
    error_parallel = np.einsum('...di,...d->...i', target_tangent, error)
    error_parallel = np.einsum('...di,...i->...d', target_tangent, error_parallel)
    return dict(KB_derivative_singular_values=singular,
        KB_diagnostic_tangent_rank=mask.sum(-1), R_of_mean=prediction,
        displacement=displacement, displacement_tangent=parallel,
        displacement_normal=displacement-parallel,
        closure_residual=error, closure_residual_tangent=error_parallel,
        closure_residual_normal=error-error_parallel)


def nearest_grid_points(point, axes, means, *, count=5):
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    rows = []
    for scale in range(means.shape[2]):
        distance = np.linalg.norm(means[:, :, scale].reshape(-1, len(point))-point, axis=-1)
        ids = np.argsort(distance, kind='stable')[:count]
        rows.append(dict(scale=scale, cells=[dict(K=float(cells[i, 0]), B=float(cells[i, 1]),
            distance=float(distance[i])) for i in ids]))
    return dict(nearest_grid_points_by_scale=rows,
        caution='仅列最近离散参数格点及完整坐标距离；不建立唯一 K/B 逆映射。')


def find_roots(model, train, axes, means, *, deadline=None, progress=None,
               partial_path=None, max_seeds=MAX_SEEDS, max_nfev=MAX_NFEV, max_roots=MAX_ROOTS):
    """At most 64 TRAIN seeds x 200 evaluations; never a Cartesian latent grid."""
    from scipy.optimize import least_squares
    if not 1 <= max_nfev <= MAX_NFEV or not 1 <= max_roots <= MAX_ROOTS:
        raise ValueError('求根预算不能超过200次评价、16个详细根')
    inputs = training_inputs(train)
    seeds, indices = select_training_seeds(train, max_seeds=max_seeds)
    dim = inputs.shape[-1]
    span = np.maximum(np.ptp(inputs, axis=0), .1)
    lower, upper = inputs.min(0)-.25*span, inputs.max(0)+.25*span
    dedup = max(float(np.linalg.norm(span))*1e-6, 1e-8)
    attempts, accepted, records = [], [], []

    def save_partial():
        if partial_path is not None:
            atomic_json(partial_path, dict(status='partial', report_complete=False,
                attempts=attempts, all_accepted_root_coordinates=accepted, roots=records,
                seed_flat_training_input_indices=indices, search_box=[lower, upper],
                search_complete=False, exhaustive_search_claim=False))

    def residual(point):
        _check(deadline)
        return mapped(model, point)-point

    def jacobian(point):
        _check(deadline)
        x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(model.advance_theory, x).detach().numpy()-np.eye(dim)

    try:
        save_partial()
        for ordinal, (index, guess) in enumerate(zip(indices, seeds)):
            _check(deadline)
            _progress(progress, '完整坐标有界求根', ordinal, len(seeds))
            solution = least_squares(residual, guess, jac=jacobian, bounds=(lower, upper),
                max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            point = np.asarray(solution.x)
            norm = float(np.linalg.norm(residual(point)))
            inside = bool(np.isfinite(point).all() and np.all(point >= lower) and np.all(point <= upper))
            verified = inside and math.isfinite(norm) and norm <= ROOT_TOLERANCE
            unique = verified and not any(np.linalg.norm(point-old) <= dedup for old in accepted)
            if unique:
                accepted.append(point)
            attempts.append(dict(training_flat_input_index=int(index), solver_success=bool(solution.success),
                nfev=int(solution.nfev), residual_norm=norm, inside_search_box=inside,
                full_coordinate_root_verified=verified, new_unique_root=unique))
            save_partial()
        _progress(progress, '完整坐标有界求根', len(seeds), len(seeds))
        for index, point in enumerate(accepted[:max_roots]):
            _check(deadline)
            derivative = full_r_jacobian(model, point)
            eigenvalues, eigenvectors = np.linalg.eig(np.asarray(derivative['jacobian']))
            eigen = [dict(real=float(v.real), imag=float(v.imag), magnitude=float(abs(v)),
                eigenvector_real=eigenvectors[:, i].real.tolist(),
                eigenvector_imag=eigenvectors[:, i].imag.tolist()) for i, v in enumerate(eigenvalues)]
            support = training_support(point, train)
            support['support_caution'] = support.pop('caution')
            nearest = nearest_grid_points(point, axes, means)
            nearest['grid_projection_caution'] = nearest.pop('caution')
            records.append(dict(root_id=f'F{index+1}', g=point.tolist(), **derivative,
                **stability_counts(derivative['jacobian']), **support, **nearest,
                eigenvalues=eigen, isolated_root_certified=False,
                caution='完整坐标残差验收；可能非孤立，几何近邻不认证物理不动点。'))
            save_partial()
            _progress(progress, '完整导数与全部实复谱', index+1, min(len(accepted), max_roots))
    except AnalysisBudgetHold:
        save_partial()
        raise
    return dict(roots=records, all_accepted_root_coordinates=[p.tolist() for p in accepted],
        attempts=attempts, search_box=[lower.tolist(), upper.tolist()],
        seed_flat_training_input_indices=indices.tolist(), seed_count=len(seeds),
        max_seed_count=max_seeds, max_nfev_per_seed=max_nfev, max_analyzed_roots=max_roots,
        acceptance_residual_l2=ROOT_TOLERANCE, root_dedup_tolerance=dedup,
        accepted_unique_root_count=len(accepted), retained_root_count=len(records),
        root_analysis_truncated=len(accepted)>max_roots, search_complete=False,
        finite_budget_search_finished=True, exhaustive_search_claim=False,
        bounds_source='TRAIN direct[:, :-1] only; 25% coordinate-span expansion; span floor 0.1',
        caution='有限预算非穷尽搜索；未找到根不证明无根；全部验收坐标保留，最多16个根保留完整导数和全部谱。')


def _save_figure(fig, path, *, deadline=None):
    import matplotlib.pyplot as plt
    _check(deadline)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    _check(deadline)


def plot_geometry(model, axes, means, pca, geometry, local, roots, out, *, deadline=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps, dimension = means.shape[2:]
    projected = project_points(means, pca)
    drift = project_displacement(geometry['displacement'], pca)
    # All components are retained in numeric files; show all original coordinates.
    for values, filename, title, count in (
            (projected, 'coupling_by_scale.png', '训练主成分', min(3, projected.shape[-1])),
            (means, 'coupling_all_coordinates.png', '原始坐标 g', dimension)):
        fig, axs = plt.subplots(count, steps, figsize=(4.1*steps, 2.6*count), squeeze=False, layout='constrained')
        for j in range(count):
            for scale in range(steps):
                ax = axs[j, scale]
                z = ax.pcolormesh(*axes, values[:, :, scale, j].T, shading='nearest')
                ax.set(xlabel='微观 K（仅评价）', ylabel='微观 B（仅评价）', title=f'{title}{j+1}；k={scale}')
                fig.colorbar(z, ax=ax)
        _save_figure(fig, out/filename, deadline=deadline)

    fig = plt.figure(figsize=(6*steps, 10), layout='constrained')
    for scale in range(steps):
        points, vectors = projected[:, :, scale].reshape(-1, projected.shape[-1]), drift[:, :, scale].reshape(-1, projected.shape[-1])
        cells = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
        every = max(1, len(points)//100)
        ax = fig.add_subplot(2, steps, scale+1)
        color = ax.scatter(*points[:, :2].T, c=cells[:, 1], cmap='coolwarm', s=8)
        for j in range(len(axes[1])):
            ax.plot(*projected[:, j, scale, :2].T, c='.6', lw=.4)
        ax.quiver(*points[::every, :2].T, *vectors[::every, :2].T,
                  angles='xy', scale_units='xy', scale=1.)
        ax.set(xlabel='训练主成分1', ylabel='训练主成分2', title=f'k={scale}；完整 R(g)-g 的二维投影')
        ax3 = fig.add_subplot(2, steps, steps+scale+1, projection='3d')
        ax3.scatter(*points[:, :3].T, c=cells[:, 1], cmap='coolwarm', s=5)
        ax3.quiver(*points[::every, :3].T, *vectors[::every, :3].T, normalize=False)
        ax3.set(xlabel='训练主成分1', ylabel='训练主成分2', zlabel='训练主成分3', title='三维投影；箭头未归一化')
        visible = np.concatenate((points, points[::every]+vectors[::every]))
        for root in roots['roots']:
            p = project_points(root['g'], pca)
            ax.scatter(*p[:2], marker='*', c='gold', edgecolors='black', s=75)
            ax3.scatter(*p[:3], marker='*', c='gold', edgecolors='black', s=75)
            visible = np.concatenate((visible, p[None]))
        low, high = visible.min(0), visible.max(0)
        margin = np.maximum((high-low)*.07, 1e-8)
        for coordinate, setter in enumerate((ax.set_xlim, ax.set_ylim)):
            setter(low[coordinate]-margin[coordinate], high[coordinate]+margin[coordinate])
        for coordinate, setter in enumerate((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim)):
            setter(low[coordinate]-margin[coordinate], high[coordinate]+margin[coordinate])
        fig.colorbar(color, ax=ax, label='微观 B（仅评价）', fraction=.04)
    fig.suptitle('真实完整位移投影；投影交叉或投影零位移不代表完整坐标不动点')
    _save_figure(fig, out/'R_displacement.png', deadline=deadline)

    rprojected = project_points(geometry['R_of_mean'], pca)
    fig, axs = plt.subplots(1, steps, figsize=(5*steps, 4), squeeze=False, layout='constrained')
    for scale, ax in enumerate(axs.flat):
        ax.scatter(projected[:, :, scale, 0], rprojected[:, :, scale, 0], s=8)
        low = min(projected[:, :, scale, 0].min(), rprojected[:, :, scale, 0].min())
        high = max(projected[:, :, scale, 0].max(), rprojected[:, :, scale, 0].max())
        ax.plot([low, high], [low, high], 'k--')
        ax.set(xlabel='输入主成分1', ylabel='R(g) 的主成分1', title=f'k={scale}；完整映射的坐标投影')
    _save_figure(fig, out/'R_vs_g.png', deadline=deadline)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    variance = np.asarray(pca['singular_values'])**2
    fraction = variance/variance.sum() if variance.sum() else variance
    axs[0].plot(np.arange(len(fraction))+1, fraction, 'o-')
    axs[0].set(xlabel='训练主成分编号', ylabel='方差比例', title='全局训练输入协方差谱')
    for row in local:
        spec = np.asarray(row['covariance_eigenvalues_descending'])
        axs[1].plot(np.arange(dimension)+1, np.median(spec, axis=0), 'o-', label=f'k={row["scale"]}')
    axs[1].set(xlabel='局部谱降序编号', ylabel='近邻协方差特征值中位数', title='24近邻几何谱；含曲率与噪声')
    for name, label in (('closure_residual', '完整残差'), ('closure_residual_tangent', '切向残差'), ('closure_residual_normal', '法向残差')):
        value = np.linalg.norm(geometry[name], axis=-1)
        axs[2].plot(np.arange(1, steps), np.sqrt(np.mean(value**2, axis=(0, 1))), 'o-', label=label)
    axs[2].set(xlabel='目标尺度 k', ylabel='完整坐标残差范数的均方根', title='R(平均 g) 到下一尺度平均 g')
    for ax in axs:
        ax.grid(alpha=.2)
        if ax.lines and ax is not axs[0]:
            ax.legend(fontsize=8)
    _save_figure(fig, out/'representation_diagnostics.png', deadline=deadline)


def freeze_receipt(loaded, checkpoint_before, state_before, data_path, data_before):
    checkpoint_after, state_after, data_after = sha256(loaded.checkpoint), model_state_sha256(loaded.model), sha256(data_path)
    gradients_none = all(p.grad is None for p in loaded.model.parameters())
    if checkpoint_after != checkpoint_before or state_after != state_before or data_after != data_before or not gradients_none:
        raise ValueError('冻结 checkpoint/model/data 指纹或 gradNone 改变')
    return dict(checkpoint_sha256_before=checkpoint_before, checkpoint_sha256_after=checkpoint_after,
        model_state_sha256_before=state_before, model_state_sha256_after=state_after,
        data_sha256_before=data_before, data_sha256_after=data_after,
        checkpoint_unchanged=True, model_state_unchanged=True, data_unchanged=True, model_gradients_none=True)


def write_report(loaded, train, test, train_labels, test_labels, history, output, *,
                 repeats=None, data_sha256=None, lattice_sizes=None, deadline=None, progress=None,
                 partial_root_path=None, root_options=None):
    """Pure saved-array report entry point, also usable with tiny synthetic models."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    before = model_state_sha256(loaded.model)
    if any(p.grad is not None for p in loaded.model.parameters()):
        raise ValueError('冻结报告要求模型所有 grad 为 None')
    direct_train, direct_test = np.asarray(train['direct']), np.asarray(test['direct'])
    training_inputs(direct_train)
    if direct_train.shape[1:] != direct_test.shape[1:] or direct_train.shape[-1] < 3:
        raise ValueError('需要匹配的 TRAIN/TEST 高维轨迹，d>=3')
    atomic_json(out/'status.json', dict(status='running', report_complete=False))
    _check(deadline)
    labels = np.concatenate((train_labels, test_labels))
    direct = np.concatenate((direct_train, direct_test))
    axes, means, grid_stats = grid_statistics(direct, labels)
    pca = fit_training_pca(direct_train, components=direct_train.shape[-1])
    _progress(progress, 'KB网格与TRAIN主成分', 1, 1)
    geometry = kb_tangent_residuals(loaded.model, axes, means)
    _, geometry['mean_R'] = grid_encodings(labels, mapped(loaded.model, direct))
    geometry['jensen_difference'] = geometry['mean_R']-geometry['R_of_mean']
    local = local_geometry(loaded.model, direct_train, deadline=deadline, progress=progress)
    roots = find_roots(loaded.model, direct_train, axes, means, deadline=deadline,
        progress=progress, partial_path=partial_root_path, **(root_options or {}))
    closure = {name: closure_metrics(values, reference_direct=direct_train)
               for name, values in (('train', train), ('test', test))}
    target_norm_sd = np.sqrt(direct_train[:, 1:].var(axis=0).sum(axis=-1))
    residual_summary = {}
    for name in ('closure_residual', 'closure_residual_tangent', 'closure_residual_normal'):
        value = geometry[name]
        rms_norm = np.sqrt(np.mean(np.sum(value**2, axis=-1), axis=(0, 1)))
        residual_summary[name] = dict(rms_full_coordinate_norm=rms_norm.tolist(),
            normalized_by_TRAIN_target_norm_sd=[float(error/spread) if spread > 1e-12 else None
                                               for error, spread in zip(rms_norm, target_norm_sd)])
    report = dict(schema_version=1, run=str(loaded.checkpoint.parent), checkpoint=str(loaded.checkpoint),
        checkpoint_epoch=loaded.payload.get('epoch'), theory_dim=direct.shape[-1],
        block_factor=loaded.model_config.block_factor, lattice_sizes=lattice_sizes,
        data_sha256=data_sha256, labels_used_in_training=False, labels_used_in_report=True,
        coupling_names=['K', 'B'], grid_axes=[a.tolist() for a in axes],
        train_ensemble_count=len(direct_train), test_ensemble_count=len(direct_test),
        evaluation_seed=12345, sampling_repeats=repeats, grid_replica_statistics=grid_stats,
        closure=closure, train_representation=representation_statistics(direct_train),
        pca=dict(center=pca['center'], components=pca['components'], singular_values=pca['singular_values'],
            explained_variance_fraction=pca['explained_variance_fraction'],
            fit_source='TRAIN direct[:, :-1] only; no labels or TEST'),
        local_geometry=local, architecture=architecture_diagnostics(loaded.model, theory_dim=direct.shape[-1]),
        tangent_diagnostics=dict(relative_singular_threshold=1e-3, absolute_singular_threshold=1e-10,
            displacement_frame='输入尺度的KB网格切平面', closure_frame='目标尺度的KB网格切平面',
            mean_grid_closure_residuals=residual_summary,
            caution='标签仅评价；有限差分、阈值与T2压缩仅为几何诊断，不认证物理维数、相关方向或自治流形。'),
        fixed_points=roots, final_logged_metrics=history[-1] if history else None,
        limitations=['单训练seed、有限格点、两个RG步骤。',
            '协方差有效秩、近邻谱、二维压缩不是物理算符数；局部小特征值不能单独认定为噪声。',
            'KB均值用全部评价格点，PCA/局部近邻/求根种子只用TRAIN。',
            '不同dimg坐标与随机数消耗不同；比较归一化误差与几何谱，不直接比较原坐标值。',
            '随机C重复条件于同一模型及同一物理构型，不是独立物理副本或独立训练seed。',
            'R(平均g)与平均R(g)一般不相同；均值闭合残差与逐系综闭合分别报告。'])
    np.savez_compressed(out/'encodings.npz', **{f'{name}_{key}': value for name, values in
        (('train', train), ('test', test)) for key, value in values.items()},
        train_labels=train_labels, test_labels=test_labels)
    np.savez_compressed(out/'grid_encodings.npz', K=axes[0], B=axes[1], mean_by_scale=means,
        pca_center=pca['center'], pca_components=pca['components'],
        projected_mean_by_scale=project_points(means, pca),
        projected_true_displacement=project_displacement(geometry['displacement'], pca), **geometry)
    atomic_json(out/'fixed_points.json', roots)
    atomic_json(out/'loss_history.json', history)
    with chinese_figures():
        _progress(progress, '中文图形', 0, 1)
        plot_geometry(loaded.model, axes, means, pca, geometry, local, roots, out, deadline=deadline)
        _check(deadline)
        plot_losses(history, out)
        _check(deadline)
        _progress(progress, '中文图形', 1, 1)
    if model_state_sha256(loaded.model) != before or any(p.grad is not None for p in loaded.model.parameters()):
        raise ValueError('报告改变冻结模型或参数梯度')
    atomic_json(out/'metrics.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', default=12000, type=int)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--sampling-repeats', default=5, type=int)
    parser.add_argument('--max-seconds', default=840., type=float)
    parser.add_argument('--progress-path', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.threads != 1 or args.sampling_repeats != 5 or args.checkpoint != 12000:
        parser.error('正式报告固定CPU1线程、完整12000终点和5次额外TEST随机C重复')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('拒绝覆盖已有报告或部分报告')
    if args.progress_path == args.output or args.output in args.progress_path.parents:
        parser.error('进度文件必须位于报告目录外')
    torch.set_num_threads(1)
    deadline = Deadline(args.max_seconds)
    progress = Progress(args.progress_path, deadline)
    with single_thread_control() as thread_control:
        result = _run(args, deadline, progress, thread_control)
    if result.get('status') == 'analysis_budget_hold':
        raise SystemExit(75)
    return result


def _run(args, deadline, progress, thread_control):
    completed, repeat_closure, partial_files = {}, [], []
    prefix = args.progress_path.with_suffix('')
    collection_path = Path(str(prefix)+'.collection.npz')
    root_path = Path(str(prefix)+'.roots.partial.json')
    partial_files.extend((collection_path, root_path))
    loaded = None
    checkpoint_before = state_before = data_before = data_path = None
    try:
        progress('校验冻结终点与数据', 0, 1)
        checkpoint_before = sha256(args.run/f'training_step_{args.checkpoint}.pt')
        _check(deadline)
        loaded = load_run(args.run, args.checkpoint)
        cfg = RunConfig.from_dict(loaded.config)
        cfg.validate()
        if (cfg.model.dimension != 2 or cfg.model.theory_dim not in (5, 10) or cfg.model.categorical_states or
                tuple(cfg.data.coupling_names) != ('K', 'B') or cfg.data.spin_flip_augment or cfg.data.spin_flip_symmetric or
                cfg.model.block_factor != 3 or cfg.model.kernel_layout != 'nonoverlapping' or
                cfg.evaluation.max_rg_steps != 2 or cfg.evaluation.samples != 64 or cfg.training.seed != 91):
            raise ValueError('需要seed91、dimg5/10、64构型、非重叠b3、两步的单层KB终点')
        summary = json.loads((args.run/'summary.json').read_text())
        validate_completed_run(cfg, summary, loaded.payload, args.checkpoint)
        metadata = loaded.payload.get('metadata', {})
        if metadata.get('labels_used_in_training') is not False or metadata.get('q_supervision') not in (None, {}):
            raise ValueError('要求明确的无标签训练来源')
        manifest = json.loads((args.run/'manifest.json').read_text())
        data_path = Path(cfg.data.cache)
        data_before = sha256(data_path)
        if not manifest.get('data_sha256') or manifest['data_sha256'] != data_before:
            raise ValueError('缓存SHA256与训练manifest不一致')
        state_before = model_state_sha256(loaded.model)
        if any(p.grad is not None for p in loaded.model.parameters()):
            raise ValueError('加载冻结模型的grad必须为None')
        _check(deadline)
        train_data, test_data = report_datasets(cfg)
        if (len(train_data.configurations), len(test_data.configurations)) != (474, 240):
            raise ValueError('必须保留完整474 TRAIN和240 TEST系综')
        if tuple(train_data.configurations.shape[2:4]) != (36, 36):
            raise ValueError('必须保留36→12→4物理尺度')
        train_labels, test_labels = train_data.couplings.numpy(), test_data.couplings.numpy()
        labels = np.concatenate((train_labels, test_labels))
        unique, counts = np.unique(labels, axis=0, return_counts=True)
        if (unique.shape != (357, 2) or not np.all(counts == 2) or
                not np.allclose(np.unique(labels[:, 0]), np.linspace(.20, .68, 17), rtol=0, atol=1e-12) or
                not np.allclose(np.unique(labels[:, 1]), np.linspace(-.01, .01, 21), rtol=0, atol=1e-12)):
            raise ValueError('需要完整17×21 K/B网格，每格点两个物理副本')
        progress('校验冻结终点与数据', 1, 1)
        options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
            spin_flip_augment=cfg.data.spin_flip_augment, chunk_size=16)
        passes = [('train', train_data.configurations, 12345), ('test', test_data.configurations, 12345)]+[
            (f'repeat_{seed}', test_data.configurations, seed) for seed in range(5)]
        for name, configurations, seed in passes:
            completed[name] = collect_bounded(loaded.model, configurations, deadline=deadline,
                progress=progress, stage='编码 '+name, sampling_seed=seed, **options)
            if name.startswith('repeat_'):
                repeat_closure.append(closure_metrics(completed[name], reference_direct=completed['train']['direct']))
            atomic_npz(collection_path, **{f'{part}__{key}': value for part, draw in completed.items()
                for key, value in draw.items()}, train_labels=train_labels, test_labels=test_labels)
            _check(deadline)
        history = loaded.payload.get('history')
        if history is None:
            history = json.loads((args.run/'history_progress.json').read_text())
        history = [row for row in history if row.get('epoch', 0) <= args.checkpoint]
        c_draws = np.stack([completed[f'repeat_{seed}']['direct'] for seed in range(5)])
        c_variance = c_draws.var(axis=0, ddof=1).mean(axis=0)
        repeats = dict(seeds=list(range(5)), closure=repeat_closure,
            coordinate_C_variance_by_scale=c_variance.tolist(),
            caution='固定物理构型及同一模型的随机C重复，不是独立物理副本或训练seed。')
        report = write_report(loaded, completed['train'], completed['test'], train_labels, test_labels,
            history, args.output, repeats=repeats, data_sha256=data_before, lattice_sizes=[36, 12, 4],
            deadline=deadline, progress=progress, partial_root_path=root_path)
        _check(deadline)
        freeze = freeze_receipt(loaded, checkpoint_before, state_before, data_path, data_before)
        report['freeze_verification'] = freeze
        report['execution'] = dict(cpu_threads=1, chunk_size=16, max_seconds=deadline.limit,
            elapsed_seconds=deadline.elapsed, thread_control=thread_control, stage_seconds=progress.timings)
        atomic_json(args.output/'metrics.json', report)
        _check(deadline)
        status = dict(status='complete', report_complete=True, checkpoint_epoch=args.checkpoint,
            theory_dim=cfg.model.theory_dim, training_started=False, physical_sampling_started=False,
            MI_estimated=False, public_files=sorted({p.name for p in args.output.iterdir()}|{'status.json', 'report.json'}),
            **freeze)
        atomic_json(args.output/'status.json', status)
        atomic_json(args.output/'report.json', status)
        atomic_json(progress.path, status)
        return report
    except Exception as error:
        hold = isinstance(error, AnalysisBudgetHold)
        freeze = None
        if loaded is not None and all(v is not None for v in (checkpoint_before, state_before, data_path, data_before)):
            try:
                freeze = freeze_receipt(loaded, checkpoint_before, state_before, data_path, data_before)
            except Exception as freeze_error:
                error = RuntimeError(f'{error}; 冻结校验失败: {freeze_error}')
                hold = False
        args.output.mkdir(parents=True, exist_ok=True)
        record = dict(status='analysis_budget_hold' if hold else 'failed', report_complete=False,
            error=f'{type(error).__name__}: {error}', max_seconds=deadline.limit,
            elapsed_seconds=deadline.elapsed, latest_stage=progress.latest,
            completed_encoding_passes=list(completed),
            partial_files=[str(p) for p in partial_files if p.is_file()],
            active_partial_pass_resumable=False, freeze_verification=freeze,
            training_started=False, physical_sampling_started=False, MI_estimated=False,
            caution='预算不足不等于没有根或报告完成；已完成pass保留，活动pass没有精确RNG收据，不可当作完整数据。')
        atomic_json(args.output/'status.json', record)
        atomic_json(args.output/'report.json', record)
        atomic_json(progress.path, record)
        if not hold:
            raise
        return record


if __name__ == '__main__':
    main()
