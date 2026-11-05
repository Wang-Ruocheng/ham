"""二维 Ising 高维 QCR 的冻结终点报告，不训练、不采样物理构型、不重估 MI。

支持任意正潜变量维数；求根只使用最多 64 个 TRAIN 输入种子，单次
有界最小二乘最多 200 次函数评价，最多分析 16 个完整坐标验收根。
这是有限预算的非穷尽搜索。图中的主成分投影不参与求根或稳定性判断。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories
from scripts.report_ising_dimg_ablation import (
    KC, STABILITY_ZH, closure_details, coupling_statistics, curve_projections,
    eigenvalue_records, energy_path_diagnostics, json_value, mapped, plot_losses,
    plot_representation, report_datasets, representation_statistics,
    validate_completed_run, write_json,
)
from scripts.report_ising_zh import chinese_figures

MAX_SEEDS = 64
MAX_NFEV = 200
MAX_ROOTS = 16
ROOT_TOLERANCE = 1e-8
FD_STEP = 1e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _matrix_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = (float(singular[0])*max(matrix.shape)*np.finfo(np.float64).eps
                 if len(singular) else 0.)
    return dict(shape=list(matrix.shape), singular_values=singular.tolist(),
                numerical_rank=int(np.count_nonzero(singular > threshold)),
                rank_threshold=threshold)


def _full_jacobian(model, point):
    tensor = torch.tensor(np.asarray(point), dtype=torch.float64, requires_grad=True)
    return torch.autograd.functional.jacobian(
        model.advance_theory, tensor).detach().cpu().numpy()


def full_r_jacobian(model, point, *, h=FD_STEP):
    """完整离散 R 的导数，包含残差连接中的恒等项；逐坐标中心差分。"""
    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1 or not len(point) or not np.isfinite(point).all() or h <= 0:
        raise ValueError('Expected a finite positive-dimensional point and positive FD step')
    jacobian = _full_jacobian(model, point)
    directions = np.eye(len(point))*h
    # A batch of 2*d points is linear, not exponential, in latent dimension.
    finite = ((mapped(model, point+directions)-mapped(model, point-directions))/(2*h)).T
    if (jacobian.shape != (len(point), len(point)) or
            not np.isfinite(jacobian).all() or not np.isfinite(finite).all()):
        raise ValueError('Nonfinite or malformed complete R derivative')
    residual = float(np.linalg.norm(mapped(model, point)-point))
    displacement = jacobian-np.eye(len(point))
    return dict(jacobian=jacobian.tolist(), finite_difference_jacobian=finite.tolist(),
        finite_difference_h=h, derivative_max_abs_error=float(np.max(abs(jacobian-finite))),
        derivative_verified=bool(np.allclose(jacobian, finite, rtol=1e-5, atol=1e-7)),
        residual_norm=residual, root_verified=bool(residual <= ROOT_TOLERANCE),
        displacement_jacobian_rank=int(np.linalg.matrix_rank(displacement)),
        displacement_jacobian_nullity=int(len(point)-np.linalg.matrix_rank(displacement)),
        derivative_definition='完整 DR；非 D(R-g)，恒等项保留。')


def stability_counts(jacobian, *, tolerance=1e-6):
    """全谱按模长分类；负实或复特征值也不能被丢弃。"""
    values = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
    magnitude = abs(values)
    expanding = int(np.count_nonzero(magnitude > 1+tolerance))
    contracting = int(np.count_nonzero(magnitude < 1-tolerance))
    neutral = len(values)-expanding-contracting
    kind = ('nonhyperbolic' if neutral else 'unstable' if expanding == len(values)
            else 'stable' if contracting == len(values) else 'saddle')
    return dict(stability=kind, expanding_direction_count=expanding,
        contracting_direction_count=contracting, neutral_direction_count=neutral,
        positive_real_expanding_count=int(np.count_nonzero(
            (abs(values.imag) <= 1e-8) & (values.real > 1+tolerance))),
        negative_real_count=int(np.count_nonzero((abs(values.imag) <= 1e-8) & (values.real < 0))),
        nonreal_count=int(np.count_nonzero(abs(values.imag) > 1e-8)),
        stability_tolerance=tolerance,
        stability_caution='高维临界点可为鞍点而非全方向排斥子；扩张方向尚未识别为物理热方向。')


def training_inputs(train):
    train = np.asarray(train, dtype=np.float64)
    if (train.ndim != 3 or train.shape[1] < 2 or min(train.shape) < 1 or
            not np.isfinite(train).all()):
        raise ValueError('Expected finite TRAIN [ensemble, scale>=2, dimension] encodings')
    return train[:, :-1].reshape(-1, train.shape[-1])


def select_training_seeds(train, *, max_seeds=MAX_SEEDS):
    """标签不可传入此 API；只按 TRAIN 输入的固定文件次序等距取样。"""
    if not 1 <= max_seeds <= MAX_SEEDS:
        raise ValueError('Need 1..64 deterministic TRAIN seeds')
    inputs = training_inputs(train)
    indices = np.unique(np.linspace(0, len(inputs)-1, min(max_seeds, len(inputs)), dtype=int))
    return inputs[indices].copy(), indices


def training_support(point, train):
    """近邻与局部 PCA 支持诊断；高维不构建 Delaunay 或凸包。"""
    from scipy.spatial import cKDTree
    inputs = training_inputs(train)
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (inputs.shape[-1],) or not np.isfinite(point).all():
        raise ValueError('Point dimension must match TRAIN inputs')
    sigma = inputs.std(0)
    scale = np.where(sigma > 1e-12, sigma, 1.)
    standardized = inputs/scale
    tree = cKDTree(standardized)
    nearest = float(tree.query(point/scale)[0])
    pair_distances = tree.query(standardized, k=min(2, len(inputs)))[0]
    radius = (2*float(np.quantile(pair_distances[:, -1], .95))
              if np.ndim(pair_distances) == 2 else 0.)
    count = min(len(inputs), max(16, 2*len(point)+1))
    _, indices = tree.query(point/scale, k=count)
    local = standardized[np.atleast_1d(indices)]
    center = local.mean(0)
    _, singular, basis = np.linalg.svd(local-center, full_matrices=False)
    threshold = (float(singular[0])*max(local.shape)*np.finfo(np.float64).eps
                 if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > threshold))
    offset = point/scale-center
    residual = offset-(offset@basis[:rank].T)@basis[:rank]
    inside_box = bool(np.all(point >= inputs.min(0)) and np.all(point <= inputs.max(0)))
    return dict(nearest_training_input_distance=float(np.linalg.norm(inputs-point, axis=1).min()),
        nearest_training_input_distance_in_coordinate_sd=nearest,
        train_nearest_neighbor_95_radius_in_coordinate_sd=radius,
        inside_training_coordinate_box=inside_box, inside_training_convex_hull=None,
        convex_hull_computed=False,
        local_pca=dict(input_scales=list(range(np.asarray(train).shape[1]-1)),
            neighbor_count=count, numerical_rank=rank, ambient_dimension=len(point),
            singular_values=singular.tolist(), rank_threshold=threshold,
            orthogonal_distance_in_coordinate_sd=float(np.linalg.norm(residual)),
            center_in_coordinate_sd=center.tolist()),
        geometric_support_candidate=bool(inside_box and nearest <= max(radius, 1e-10)),
        physical_support_certified=False,
        caution='只使用 TRAIN 的 R 输入尺度；近邻、局部主成分及盒内判断不认证物理横向支持。')


def architecture_diagnostics(model, *, theory_dim, hidden=None):
    """公开秩/容量，不修改、归一化或训练任何已有层。"""
    result = dict(theory_dim=int(theory_dim), configured_hidden=hidden)
    encoder = getattr(model, 'theory_encoder', None)
    decoder = getattr(encoder, 'distribution_decoder', None)
    if isinstance(decoder, torch.nn.Linear):
        weight = decoder.weight.detach().cpu().numpy()
        result['affine_decoder'] = dict(**_matrix_diagnostics(weight),
            column_norms=np.linalg.norm(weight, axis=0).tolist(),
            latent_nullity=int(theory_dim-np.linalg.matrix_rank(weight)),
            normalized_probability_target_degrees_of_freedom_upper_bound=int(weight.shape[0]-1),
            rank_forced=False,
            caution='重构目标的维数与解码器秩不是物理相关方向数；不强制满秩。')
    else:
        result['affine_decoder'] = None
    head = getattr(encoder, 'head', None)
    head_layers = [module for module in head.modules() if isinstance(module, torch.nn.Linear)] if head is not None else []
    if head_layers:
        result['Q_final_affine'] = dict(**_matrix_diagnostics(head_layers[-1].weight.detach().cpu().numpy()),
            hidden_width=head_layers[-1].in_features,
            pooled_local_feature_count=head_layers[0].in_features,
            local_differential_rank_upper_bound=min(theory_dim, head_layers[0].in_features,
                *(layer.out_features for layer in head_layers)),
            caution='局部微分秩受池化局部特征数限制；非线性曲面的全局协方差秩可更高，不代表同数目的独立切向方向。最后仿射层另限制输出仿射子空间。')
    network = getattr(model, 'theory_rg', None)
    layers = [module for module in network.modules() if isinstance(module, torch.nn.Linear)] if network is not None else []
    if layers and layers[-1].out_features == theory_dim:
        last = layers[-1]
        weight = last.weight.detach().cpu().numpy()
        bias = (last.bias.detach().cpu().numpy() if last.bias is not None else np.zeros(theory_dim))
        left, singular, _ = np.linalg.svd(weight, full_matrices=False)
        diag = _matrix_diagnostics(weight)
        rank = diag['numerical_rank']
        outside = bias-left[:, :rank]@(left[:, :rank].T@bias)
        lower = float(np.linalg.norm(outside))
        result['R_residual_output'] = dict(**diag,
            hidden_width=last.in_features,
            structural_displacement_rank_upper_bound=min(theory_dim, last.in_features),
            structural_unit_eigenvalue_count_lower_bound=max(0, theory_dim-last.in_features),
            bias_outside_output_column_space=outside.tolist(),
            displacement_norm_lower_bound=lower,
            root_tolerance=ROOT_TOLERANCE,
            output_affine_span_allows_root_at_tolerance=bool(lower <= ROOT_TOLERANCE),
            lower_bound_is_numerical=True,
            caution='R=g+W h+b；列空间外的 b 给位移范数下界。若维数超过隐藏宽度，完整 DR 必有单位特征值；若下界超阈值则本结构不能达到验收根。')
    else:
        result['R_residual_output'] = None
    return result


def find_roots(model, train, knots, means, *, block_factor=2,
               max_seeds=MAX_SEEDS, max_nfev=MAX_NFEV, max_roots=MAX_ROOTS):
    """固定多起点有界验收；不使用 K、Kc、TEST 或投影选择种子。"""
    from scipy.optimize import least_squares
    if not 1 <= max_nfev <= MAX_NFEV or not 1 <= max_roots <= MAX_ROOTS:
        raise ValueError('Solver cap must be <=200 evaluations and <=16 analyzed roots')
    inputs = training_inputs(train)
    dim = inputs.shape[-1]
    seeds, indices = select_training_seeds(train, max_seeds=max_seeds)
    span = np.maximum(np.ptp(inputs, axis=0), .1)
    lower, upper = inputs.min(0)-.25*span, inputs.max(0)+.25*span
    def residual(point):
        return mapped(model, point)-point
    def jacobian(point):
        return _full_jacobian(model, point)-np.eye(dim)
    attempts, accepted = [], []
    dedup = max(float(np.linalg.norm(span))*1e-6, 1e-8)
    for index, guess in zip(indices, seeds):
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
            status=int(solution.status), nfev=int(solution.nfev),
            njev=int(solution.njev) if solution.njev is not None else None,
            residual_norm=norm, inside_search_box=inside, full_coordinate_root_verified=verified,
            new_unique_root=bool(unique)))
    # Keep deterministic solver encounter order, never rank by lambda, K or residual.
    retained = accepted[:max_roots]
    records = []
    for index, point in enumerate(retained):
        derivative = full_r_jacobian(model, point)
        support = training_support(point, train)
        projections = [curve_projections(knots, means[:, scale], point) for scale in range(means.shape[1])]
        spectra = eigenvalue_records(derivative['jacobian'], projections, block_factor=block_factor,
            verified=derivative['derivative_verified'] and derivative['root_verified'],
            support_candidate=support['geometric_support_candidate'])
        records.append(dict(root_id=f'F{index+1}', g=point.tolist(), **derivative, **support,
            **stability_counts(derivative['jacobian']), eigenvalues=spectra,
            curve_projections_by_scale=projections, selected_thermal_eigenvalue=None, thermal_nu=None,
            isolated_root_certified=False,
            root_qualification='全坐标残差验收点；可能属于非孤立根集；投影交点不作根证据。'))
    return dict(roots=records, search_box=[lower.tolist(), upper.tolist()],
        bounds_source='TRAIN direct[:, :-1] only; 25% coordinate-span expansion; span floor 0.1',
        seed_count=len(seeds), seed_flat_training_input_indices=indices.tolist(),
        seed_rule='最多64个TRAIN输入按固定文件次序确定性等距抽样；不用标签、TEST或理论',
        max_seed_count=max_seeds, max_nfev_per_seed=max_nfev, max_analyzed_roots=max_roots,
        acceptance_residual_l2=ROOT_TOLERANCE, root_dedup_tolerance=dedup,
        solver_success_count=sum(row['solver_success'] for row in attempts), attempts=attempts,
        accepted_unique_root_count=len(accepted), retained_root_count=len(retained),
        all_accepted_root_coordinates=[point.tolist() for point in accepted],
        all_accepted_coordinates_retained=True,
        root_analysis_truncated=bool(len(accepted) > max_roots),
        truncated_unique_root_count=max(0, len(accepted)-max_roots),
        search_complete=False, exhaustive_search_claim=False,
        thermal_nu=None,
        note='非穷尽有界多起点搜索；没有发现根不证明不存在根。超过16个验收点时按种子遇到次序截断，不能声称无其他根。')


def fit_training_pca(train, *, components=6):
    inputs = training_inputs(train)
    center = inputs.mean(0)
    _, singular, vectors = np.linalg.svd(inputs-center, full_matrices=False)
    vectors = vectors[:min(components, len(vectors))].copy()
    for vector in vectors:
        pivot = int(np.argmax(abs(vector)))
        if vector[pivot] < 0:
            vector *= -1
    denominator = float(singular@singular)
    return dict(center=center, components=vectors, singular_values=singular,
        explained_variance_fraction=(singular[:len(vectors)]**2/denominator
            if denominator else np.zeros(len(vectors))),
        training_input_scales=np.arange(np.asarray(train).shape[1]-1))


def project_points(points, pca):
    return (np.asarray(points)-pca['center'])@pca['components'].T


def project_displacement(displacement, pca):
    # No centering for vectors: this is projection of the true full R(g)-g.
    return np.asarray(displacement)@pca['components'].T


def _mark_roots(ax, roots, pca, *, coordinates=(0, 1)):
    for root in roots['roots']:
        point = project_points(root['g'], pca)
        xy = tuple(point[j] for j in coordinates)
        ax.scatter(*xy, marker='*', s=90, c='gold', edgecolors='black')
        if len(xy) == 2:
            ax.annotate(root['root_id'], xy, xytext=(4, 4), textcoords='offset points', fontsize=8)


def plot_maps(model, train, test, knots, means, critical, roots, pca, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dimension = train.shape[-1]
    count = min(6, len(pca['components']))
    projected = project_points(means, pca)
    Rmean = mapped(model, means)
    Rprojected = project_points(Rmean, pca)
    drift = project_displacement(Rmean-means, pca)
    critical_projected = project_points(critical, pca)
    rows = math.ceil(count/3)
    fig, axes = plt.subplots(rows, min(3, count), figsize=(6*min(3, count), 4.8*rows),
                             squeeze=False, layout='constrained')
    for coordinate, ax in enumerate(axes.flat):
        if coordinate >= count:
            ax.set_visible(False); continue
        for scale in range(means.shape[1]):
            ax.plot(projected[:, scale, coordinate], Rprojected[:, scale, coordinate], '.-', ms=3, label=f'k={scale}')
        low = min(float(projected[..., coordinate].min()), float(Rprojected[..., coordinate].min()))
        high = max(float(projected[..., coordinate].max()), float(Rprojected[..., coordinate].max()))
        ax.plot([low, high], [low, high], 'k--', lw=.8, label='恒等映射投影')
        for root in roots['roots']:
            value = project_points(root['g'], pca)[coordinate]
            ax.scatter(value, value, marker='*', s=80, c='gold', edgecolors='black')
        ax.set(xlabel=f'输入主成分{coordinate+1}', ylabel=f'输出主成分{coordinate+1}',
               title='完整 R(g) 的投影；不是标量自治映射')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f'潜变量维数={dimension}；主成分仅由训练输入拟合；投影交点不是根')
    fig.savefig(out/'R_vs_g.png', dpi=150); plt.close(fig)

    if dimension == 1:
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        grid = np.linspace(*np.asarray(roots['search_box'])[:, 0], 801)[:, None]
        ax.plot(grid[:, 0], (mapped(model, grid)-grid)[:, 0], label='完整位移 R(g)-g')
        ax.axhline(0., c='black', ls='--', lw=.8)
        for root in roots['roots']:
            ax.scatter(root['g'][0], 0., marker='*', s=80, c='gold', edgecolors='black')
        ax.set(xlabel='g', ylabel='R(g)-g', title='完整标量位移；根搜索仍为固定多起点非穷尽搜索')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    else:
        fig = plt.figure(figsize=(14 if count >= 3 else 8, 6), layout='constrained')
        ax = fig.add_subplot(121 if count >= 3 else 111)
        for scale in range(means.shape[1]):
            points, vectors = projected[:, scale], drift[:, scale]
            ax.plot(points[:, 0], points[:, 1], '.-', ms=3, label=f'k={scale}')
            every = max(1, len(points)//24)
            ax.quiver(points[::every, 0], points[::every, 1], vectors[::every, 0], vectors[::every, 1],
                      angles='xy', scale_units='xy', scale=1., color=f'C{scale}', width=.004)
            ax.scatter(*critical_projected[scale, :2], marker='D', c='magenta', s=35,
                       label='Kc 编码（非不动点）' if scale == 0 else None)
        _mark_roots(ax, roots, pca)
        ax.set(xlabel='训练主成分1', ylabel='训练主成分2', title='真实位移的二维投影；箭头未归一化')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
        if count >= 3:
            ax3 = fig.add_subplot(122, projection='3d')
            for scale in range(means.shape[1]):
                points, vectors = projected[:, scale], drift[:, scale]
                every = max(1, len(points)//24)
                ax3.plot(*points[:, :3].T, '.-', ms=3, label=f'k={scale}')
                ax3.quiver(*points[::every, :3].T, *vectors[::every, :3].T, color=f'C{scale}', normalize=False)
            _mark_roots(ax3, roots, pca, coordinates=(0, 1, 2))
            ax3.set(xlabel='训练主成分1', ylabel='训练主成分2', zlabel='训练主成分3', title='真实位移的三维投影')
        fig.suptitle('投影箭头为完整 R(g)-g；投影重合或交叉不代表完整坐标不动点')
    fig.savefig(out/'R_displacement.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(rows, min(3, count), figsize=(6*min(3, count), 4.5*rows),
                             squeeze=False, layout='constrained')
    for coordinate, ax in enumerate(axes.flat):
        if coordinate >= count:
            ax.set_visible(False); continue
        for scale in range(means.shape[1]):
            ax.plot(knots, projected[:, scale, coordinate], '.-', ms=3, label=f'k={scale}')
        ax.axvline(KC, c='magenta', ls=':', label='Kc（仅评价）')
        ax.set(xlabel='微观 K（仅评价）', ylabel=f'训练主成分{coordinate+1}', title='各尺度 K → g 的主成分投影')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('主成分坐标只用于显示；全部原始坐标另存数值文件')
    fig.savefig(out/'coupling_by_scale.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, means.shape[1], figsize=(5*means.shape[1], 7), squeeze=False, layout='constrained')
    for scale, ax in enumerate(axes.flat):
        image = ax.imshow(means[:, scale].T, aspect='auto', origin='lower',
            extent=(float(knots[0]), float(knots[-1]), .5, dimension+.5), interpolation='nearest')
        ax.axvline(KC, c='magenta', ls=':', label='Kc（仅评价）')
        ax.set(xlabel='微观 K（仅评价）', ylabel='原始坐标编号', title=f'k={scale}；全部 {dimension} 个坐标')
        fig.colorbar(image, ax=ax, label='系综平均 g')
        ax.legend(fontsize=7)
    fig.suptitle('完整 K → g 坐标热图；颜色为原坐标值，未按热指数筛选')
    fig.savefig(out/'coupling_all_coordinates.png', dpi=150); plt.close(fig)


def plot_spectrum(roots, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    angle = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(angle), np.sin(angle), 'k--', lw=.8, label='单位圆')
    for root in roots['roots']:
        values = root['eigenvalues']
        ax.scatter([value['real'] for value in values], [value['imag'] for value in values], s=25,
                   label=f"{root['root_id']}：{STABILITY_ZH[root['stability']]}")
    ax.set(xlabel='完整 R 导数的特征值实部', ylabel='完整 R 导数的特征值虚部',
           title='全部已分析根的完整谱；保留负值与复值，未选择热模态', aspect='equal')
    ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(out/'fixed_point_spectrum.png', dpi=150); plt.close(fig)


def write_report(loaded, train, test, train_labels, test_labels, history, output,
                 *, repeats=None, data_sha256=None, lattice_sizes=None,
                 expected_checkpoint_sha256=None):
    """独占新目录；输出完整坐标/谱及冻结前后指纹，失败不自动重跑。"""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refuse to overwrite or rerun an existing report output')
    source = Path(loaded.checkpoint)
    checkpoint_before = expected_checkpoint_sha256 or (sha256(source) if source.is_file() else None)
    state_before = model_state_sha256(loaded.model)
    if any(parameter.grad is not None for parameter in loaded.model.parameters()):
        raise ValueError('Frozen report requires model gradients None before analysis')
    direct_train, direct_test = np.asarray(train['direct']), np.asarray(test['direct'])
    training_inputs(direct_train)
    if direct_train.shape[1:] != direct_test.shape[1:] or not np.isfinite(direct_test).all():
        raise ValueError('Matching finite TRAIN/TEST trajectory dimensions required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', dict(status='running', training_started=False,
        physical_sampling_started=False, MI_estimated=False))
    try:
        dimension = direct_train.shape[-1]
        knots, means, critical = coupling_statistics(direct_train, direct_test, train_labels, test_labels)
        block = loaded.model_config.block_factor
        architecture = architecture_diagnostics(loaded.model, theory_dim=dimension,
            hidden=getattr(loaded.model_config, 'hidden', None))
        roots = find_roots(loaded.model, direct_train, knots, means, block_factor=block)
        closure = {name: closure_details(loaded.model, values, direct_train)
                   for name, values in (('train', train), ('test', test))}
        statistics = representation_statistics(direct_train)
        pca = fit_training_pca(direct_train)
        all_direct = np.concatenate((direct_train, direct_test))
        all_labels = np.concatenate((np.asarray(train_labels).ravel(), np.asarray(test_labels).ravel()))
        mean_R = np.stack([mapped(loaded.model, all_direct[all_labels == coupling]).mean(0) for coupling in knots])
        R_of_mean = mapped(loaded.model, means)
        report = dict(schema_version=1, run=str(source.parent), checkpoint=str(source),
            checkpoint_epoch=loaded.payload.get('epoch'), theory_dim=dimension, block_factor=block,
            lattice_sizes=lattice_sizes, data_sha256=data_sha256, labels_used_in_training=False,
            labels_used_in_report=True, evaluation_seed=12345, sampling_repeats=repeats,
            final_logged_metrics=history[-1] if history else None,
            logged_loss_caution='仅读取既有 MI 与重构日志，不在报告中重估 MI。',
            closure=closure, train_representation=statistics, architecture=architecture,
            fixed_points=roots, roots=roots['roots'], thermal_nu=None,
            energy_path=energy_path_diagnostics(loaded.model, knots, means),
            pca=dict(center=pca['center'].tolist(), components=pca['components'].tolist(),
                singular_values=pca['singular_values'].tolist(),
                explained_variance_fraction=pca['explained_variance_fraction'].tolist(),
                fit_source='TRAIN direct[:, :-1] only; no labels or TEST',
                changes_model_or_root_coordinates=False),
            critical_reference=dict(Kc=KC, thermal_lambda=block, nu=1., evaluation_only=True,
                encoded_g_by_scale=critical.tolist(), R_of_encoded_mean=mapped(loaded.model, critical).tolist(),
                R_drift=(mapped(loaded.model, critical)-critical).tolist(),
                direct_scale_drift=np.diff(critical, axis=0).tolist(),
                caution='Kc 编码不是不动点；理论值不用来选种子、根或特征值。'),
            jensen=dict(K=knots.tolist(), mean_R=mean_R.tolist(), R_of_mean=R_of_mean.tolist(),
                        difference=(mean_R-R_of_mean).tolist()),
            selection=dict(checkpoint_rule='预定实际 12000 终点，不按指数、根或测试损失筛选',
                selected_thermal_eigenvalue=None, theory_used_for_selection=False,
                all_found_roots_retained=not roots['root_analysis_truncated'],
                all_found_root_coordinates_retained=True,
                root_detail_limit=MAX_ROOTS,
                all_eigenvalues_of_retained_roots_retained=True, exhaustive_search_claim=False),
            limitations=['有限 TRAIN 输入曲线不认证高维横向导数；近邻与局部主成分仅为几何诊断。',
                '非穷尽求根；最多64种子，每种子200次函数评价，最多分析16个根。',
                '高维临界候选可为鞍点；不要求所有方向排斥，不把最大倍率自动认定为热方向。',
                '单训练 seed、有限格点及两次粗粒化；B0 数据不能确定独立磁指数。',
                '不同潜变量维数的坐标、参数量和随机数消耗不同；不混比原始坐标误差。',
                '仿射解码器/隐藏层存在秩上界；升高输出维数不等于得到同数目的物理方向。',
                '随机 C 重复只条件于当前模型，不是独立训练 seed 或特征值置信区间。'])
        np.savez_compressed(output/'encodings.npz', train=direct_train, test=direct_test,
            train_rollout=train['rollout'], test_rollout=test['rollout'],
            train_one_step=train['one_step'], test_one_step=test['one_step'],
            train_labels=np.asarray(train_labels), test_labels=np.asarray(test_labels))
        np.savez_compressed(output/'trajectories.npz', **test, training_inputs=direct_train[:, :-1],
            K=knots, mean_by_scale=means, mean_R=mean_R, R_of_mean=R_of_mean,
            displacement_of_mean=R_of_mean-means, critical_encodings=critical,
            pca_center=pca['center'], pca_components=pca['components'],
            projected_mean_by_scale=project_points(means, pca),
            projected_true_displacement=project_displacement(R_of_mean-means, pca))
        count = len(roots['roots'])
        np.savez_compressed(output/'fixed_point_arrays.npz',
            all_accepted_root_coordinates=np.asarray(roots['all_accepted_root_coordinates']).reshape(
                roots['accepted_unique_root_count'], dimension),
            roots=np.asarray([row['g'] for row in roots['roots']]).reshape(count, dimension),
            full_DR=np.asarray([row['jacobian'] for row in roots['roots']]).reshape(count, dimension, dimension),
            full_DR_centered_FD=np.asarray([row['finite_difference_jacobian'] for row in roots['roots']]).reshape(count, dimension, dimension),
            eigenvalues=np.asarray([[complex(item['real'], item['imag']) for item in row['eigenvalues']]
                for row in roots['roots']], dtype=np.complex128).reshape(count, dimension),
            eigenvectors=np.asarray([[np.asarray(item['eigenvector_real'])+1j*np.asarray(item['eigenvector_imag'])
                for item in row['eigenvalues']] for row in roots['roots']], dtype=np.complex128).reshape(count, dimension, dimension))
        with chinese_figures():
            plot_maps(loaded.model, direct_train, direct_test, knots, means, critical, roots, pca, output)
            plot_losses(history, output)
            plot_representation(statistics, closure, output)
            plot_spectrum(roots, output)
        state_after = model_state_sha256(loaded.model)
        checkpoint_after = sha256(source) if checkpoint_before is not None else None
        gradients_none = all(parameter.grad is None for parameter in loaded.model.parameters())
        if state_after != state_before or checkpoint_after != checkpoint_before or not gradients_none:
            raise ValueError('Frozen source checkpoint/model fingerprint or gradNone changed during report')
        report['freeze_verification'] = dict(checkpoint_sha256_before=checkpoint_before,
            checkpoint_sha256_after=checkpoint_after, checkpoint_unchanged=checkpoint_before == checkpoint_after,
            model_state_sha256_before=state_before, model_state_sha256_after=state_after,
            model_state_unchanged=True, model_gradients_none=gradients_none)
        for filename, value in (('metrics.json', report), ('critical_exponents.json', report),
                                ('fixed_points.json', roots), ('loss_history.json', history)):
            write_json(output/filename, value)
        public = sorted(path.name for path in output.iterdir() if path.is_file())
        public = sorted(set(public+['report.json']))
        completion = dict(status='complete', report_complete=True,
            checkpoint_epoch=report['checkpoint_epoch'], theory_dim=dimension,
            public_files=public, training_started=False, physical_sampling_started=False, MI_estimated=False,
            **report['freeze_verification'])
        write_json(output/'status.json', completion)
        write_json(output/'report.json', completion)
        return report
    except Exception as error:
        write_json(output/'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            report_complete=False, training_started=False, physical_sampling_started=False, MI_estimated=False))
        raise


def compare_reports(specifications, output):
    """维数集合由已冻结计划决定；允许先比较完成的1/5/10或含100。"""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refuse to overwrite a comparison')
    reports = {}
    for specification in specifications:
        name, folder = specification.split('=', 1)
        if not name or name in reports:
            raise ValueError('Comparison arm names must be nonempty and unique')
        folder = Path(folder)
        if not (folder/'metrics.json').is_file():
            folder = folder/'report'
        completion = json.loads((folder/'report.json').read_text())
        value = json.loads((folder/'metrics.json').read_text())
        if completion.get('status') != 'complete' or completion.get('report_complete') is not True:
            raise ValueError('Every arm report must be complete')
        if value.get('checkpoint_epoch') != 12000:
            raise ValueError('Every comparison arm must reach actual 12000')
        reports[name] = value
    dimensions = [row['theory_dim'] for row in reports.values()]
    if not reports or len(set(dimensions)) != len(dimensions):
        raise ValueError('Need nonempty arms with distinct latent dimensions')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison.json', dict(reports=reports, thermal_nu=None,
        caution='不同维数的坐标不能直接混比；保留全部已分析根和全谱，不按最大倍率或理论邻近选择热方向。'))
    import matplotlib.pyplot as plt
    with chinese_figures():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for name, report in reports.items():
            label = f'潜变量维数={report["theory_dim"]}'
            values = report['closure']['test']['one_step']['nrmse']
            axes[0].plot(np.arange(len(values))+1, values, 'o-', label=label)
            axes[1].plot([row['participation_rank'] for row in report['train_representation']], 'o-', label=label)
            for root in report['roots']:
                axes[2].scatter([report['theory_dim']]*len(root['eigenvalues']),
                    [value['magnitude'] for value in root['eigenvalues']], s=18,
                    facecolors='C0' if root['geometric_support_candidate'] else 'none', edgecolors='C0')
        axes[0].set(title='测试集单步归一闭合误差', xlabel='目标尺度 k', ylabel='归一化均方根误差')
        axes[1].set(title='训练集协方差参与秩', xlabel='尺度 k', ylabel='有效秩')
        axes[2].set(title='全部已分析根的完整谱；非热指数', xlabel='潜变量维数', ylabel='特征值模长')
        axes[2].axhline(1., c='black', ls='--', lw=.8)
        for ax in axes:
            ax.grid(alpha=.2)
        for ax in axes[:2]:
            ax.legend(fontsize=8)
        fig.suptitle('预定 12000 步高维消融；根搜索非穷尽，投影交点不作根证据')
        fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    write_json(output/'report.json', dict(status='complete', report_complete=True, arms=list(reports),
        public_files=['comparison.json', 'comparison.png', 'report.json']))
    return reports


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
    if args.run is None or args.threads < 1 or not 0 <= args.sampling_repeats <= 5:
        parser.error('--run, positive threads and 0..5 fixed C repeats required')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Refuse to overwrite or rerun an existing report output')
    torch.set_num_threads(args.threads)
    checkpoint_path = args.run/f'training_step_{args.checkpoint}.pt'
    checkpoint_before = sha256(checkpoint_path)
    loaded = load_run(args.run, args.checkpoint)
    cfg = RunConfig.from_dict(loaded.config)
    cfg.validate()
    if (cfg.model.dimension != 2 or cfg.model.theory_dim < 1 or cfg.model.categorical_states or
            cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',)):
        raise ValueError('Requires documented single-layer K-only two-dimensional Ising')
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
    state_before = model_state_sha256(loaded.model)
    if any(parameter.grad is not None for parameter in loaded.model.parameters()):
        raise ValueError('Frozen loaded model must have None gradients')
    train_data, test_data = report_datasets(cfg)
    options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
        spin_flip_augment=cfg.data.spin_flip_augment, sampling_seed=12345)
    train = collect_trajectories(loaded.model, train_data.configurations, **options)
    test = collect_trajectories(loaded.model, test_data.configurations, **options)
    repeats = []
    for seed in range(args.sampling_repeats):
        draw = collect_trajectories(loaded.model, test_data.configurations, **dict(options, sampling_seed=seed))
        repeats.append(closure_details(loaded.model, draw, train['direct']))
    if state_before != model_state_sha256(loaded.model) or sha256(checkpoint_path) != checkpoint_before:
        raise ValueError('Frozen model or checkpoint changed during fixed C collection')
    history = loaded.payload.get('history')
    if history is None:
        history = json.loads((args.run/'history_progress.json').read_text())
    history = [row for row in history if row.get('epoch', 0) <= args.checkpoint]
    return write_report(loaded, train, test, train_data.couplings.numpy(), test_data.couplings.numpy(),
        history, args.output,
        repeats=dict(seeds=list(range(args.sampling_repeats)), closure=repeats,
            caution='固定模型的随机 C 重复，不是独立训练 seed、物理链或谱置信区间。'),
        data_sha256=manifest['data_sha256'], expected_checkpoint_sha256=checkpoint_before,
        lattice_sizes=[train_data.configurations.shape[2]//cfg.model.block_factor**scale
            for scale in range(train['direct'].shape[1])])


if __name__ == '__main__':
    main()
