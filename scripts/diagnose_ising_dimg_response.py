"""固定无监督 dimg=1/2/3 终点的物理算符与多维表示响应诊断。

只读保存的物理构型与训练终点；独立 C 抽样不生成新的物理系综。
K 仅用于事后选择诊断分布，不优化模型，不重估 MI，不读取理论指数作选择。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from scripts.diagnose_ising_mcrg import (
    EXPECTED_DATA_SHA256, KC, OPERATOR_DEFINITIONS, OPERATOR_NAMES, ORIENTATION_NAMES,
    c_draw, c_probability_table, covariance, d4_probability_errors, full_r_jacobian,
    local_feature_means, mmap_dataset, model_sha256, operator_values,
    orientation_operator_values, select_points, sha256_file, weighted_q_response,
    write_json,
)


RESPONSE_RCOND = 1e-6


def canonical_config(config):
    """JSON tuple/list differences are serialization, not a recipe difference."""
    return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))


def validate_source(source_run, data, loaded, summary):
    """Verify existing small receipts without hashing the physical data cache.

    New arms need not have the historical job.json layout or historical d=1 SHA.
    The caller records and can additionally pin each new numbered checkpoint SHA.
    """
    source_run, data = Path(source_run), Path(data)
    metadata = loaded.payload.get('metadata', {})
    manifest = json.loads((source_run / 'manifest.json').read_text())
    cfg = loaded.config
    if (canonical_config(manifest.get('config')) != canonical_config(metadata.get('config'))
            or canonical_config(cfg) != canonical_config(metadata.get('config'))
            or canonical_config(summary.get('config')) != canonical_config(cfg)):
        raise ValueError('Manifest/checkpoint configuration provenance mismatch')
    if (metadata.get('data_sha256') != EXPECTED_DATA_SHA256
            or manifest.get('data_sha256') != EXPECTED_DATA_SHA256):
        raise ValueError('Existing physical-data fingerprint receipt mismatch')
    if Path(cfg['data']['cache']).resolve() != data.resolve():
        raise ValueError('Explicit physical cache differs from source configuration')
    for receipt in (metadata, manifest):
        if receipt.get('labels_used_in_training') is not False or receipt.get('q_supervision') is not None:
            raise ValueError('Require fully unsupervised manifest and checkpoint')
    model_cfg, training = cfg['model'], cfg['training']
    if (cfg.get('schema_version') != 5 or model_cfg.get('dimension') != 2
            or model_cfg.get('theory_dim') not in (1, 2, 3)
            or model_cfg.get('block_factor') != 2 or model_cfg.get('width') != 3
            or model_cfg.get('kernel_layout') != 'nonoverlapping'
            or model_cfg.get('kernel_initialization') != 'random'
            or model_cfg.get('fixed_kernel') is not False
            or model_cfg.get('kernel_sampling') != 'bernoulli'
            or training.get('max_rg_steps') != 2
            or cfg.get('evaluation', {}).get('max_rg_steps') != 2
            or list(training.get('train_modules', ())) != ['Q', 'C', 'R']
            or training.get('gradient_mode') != 'qc_r_separate'):
        raise ValueError('Require schema5 dimg1/2/3 fully trained random probabilistic b2 depth-two QCR')
    if (loaded.payload.get('epoch') != 12000 or summary.get('actual_epochs') != 12000
            or summary.get('requested_epochs') != 12000 or training.get('epochs') != 12000):
        raise ValueError('Require completed actual 12000-update numbered checkpoint')
    if (summary.get('stopped_by_wall_limit') is True or summary.get('stopped_by_training_limit') is True
            or summary.get('stopped_by_train_plateau') is True):
        raise ValueError('Wall-limited arm is not a completed 12000-update endpoint')
    return manifest


def best_local_jacobian(vin, vout, score_std, *, rcond=RESPONSE_RCOND):
    """Fit Y=J X on independently unit-score-normalized response directions.

    Each baseline ensemble/draw has its OWN J. No different g points are pooled.
    A rank-deficient X identifies J only on span(X); minimum-norm zero extension
    outside that span must not be interpreted as physical eigenvalues.
    """
    vin, vout, std = map(lambda v: np.asarray(v, dtype=float), (vin, vout, score_std))
    if (vin.ndim != 2 or vin.shape != vout.shape or std.shape != (vin.shape[1],)
            or not np.isfinite(vin).all() or not np.isfinite(vout).all()
            or not np.isfinite(std).all() or np.any(std < 0) or not 0 < rcond < 1):
        raise ValueError('Need finite paired [d,operators] responses and nonnegative score standard deviations')
    inverse_std = np.divide(1., std, out=np.zeros_like(std), where=std > 1e-12)
    x, y = vin * inverse_std, vout * inverse_std
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    retained = singular > rcond * singular[0] if singular.size and singular[0] else np.zeros_like(singular, bool)
    jacobian = y @ np.linalg.pinv(x, rcond=rcond)
    projector = u[:, retained] @ u[:, retained].T
    residual = jacobian @ x - y
    norm_y = np.linalg.norm(y)
    power = singular ** 2
    probabilities = power / power.sum() if power.sum() else np.zeros_like(power)
    positive = probabilities > 0
    effective_rank = np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))) if positive.any() else 0.
    return dict(jacobian=jacobian, projector=projector, input_rank=int(retained.sum()),
        input_singular_values=singular, input_effective_rank=float(effective_rank),
        retained_condition_number=float(singular[0] / singular[retained][-1]) if retained.any() else None,
        residual=residual, target=y, relative_error=float(np.linalg.norm(residual) / max(norm_y, 1e-15)),
        unidentifiable_output_fraction=float(np.linalg.norm(y @ (np.eye(x.shape[1]) - np.linalg.pinv(x, rcond=rcond) @ x))
                                             / max(norm_y, 1e-15)))


def full_r_finite_difference(model, g, h=1e-5):
    """Complete DR including its identity part, columns are input coordinates."""
    columns = []
    with torch.no_grad():
        for coordinate in range(g.numel()):
            perturbation = torch.zeros_like(g)
            perturbation[coordinate] = h
            columns.append((model.advance_theory(g + perturbation)
                            - model.advance_theory(g - perturbation)) / (2 * h))
    return torch.stack(columns, dim=-1)


def energy_mode_diagnostics(matrix, response):
    """Retain ALL modes; overlaps with energy response do not certify a thermal mode.

    Right overlaps and modal excitation weights are descriptive in this latent
    coordinate metric; nonnormal/ill-conditioned eigenvectors can be misleading.
    No mode is selected by its eigenvalue or closeness to a theoretical number.
    """
    matrix, response = np.asarray(matrix, float), np.asarray(response, float)
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(-np.abs(values), kind='stable')
    values, vectors = values[order], vectors[:, order]
    norm = np.linalg.norm(response)
    if norm <= 1e-15:
        overlap, weights = np.zeros(len(values)), np.zeros(len(values))
    else:
        overlap = abs(vectors.conj().T @ response) ** 2 / (np.sum(abs(vectors) ** 2, axis=0) * norm ** 2)
        excitation = np.linalg.pinv(vectors, rcond=RESPONSE_RCOND) @ response
        contributions = abs(excitation) ** 2 * np.sum(abs(vectors) ** 2, axis=0)
        weights = contributions / contributions.sum() if contributions.sum() else contributions
    return dict(eigenvalues=np.column_stack((values.real, values.imag)).tolist(),
        energy_right_overlap=overlap.tolist(), energy_modal_excitation_weights=weights.tolist(),
        eigenvector_condition_number=float(np.linalg.cond(vectors)), energy_response_norm=float(norm),
        caution='保留所有谱值；能量方向重叠仅为当前坐标度量中的描述，不能认证热模；离响应张成空间的谱未被物理扰动识别。')


def add_physical_energy_projections(spectra, arrays):
    """Annotate every retained physical mode, never select the one nearest two.

    The coupling perturbation for the microscopic nearest-neighbor energy is
    projected into the SAME retained covariance coordinates used by MCRG. For
    the directional basis it excites x and y nearest-neighbor terms together.
    Energy overlaps are descriptive and basis/truncation dependent, not an
    independent proof of thermal-mode identity.
    """
    from scripts.report_ising_mcrg import covariance_pair
    oriented_a, oriented_b = covariance_pair(arrays['orientation_operators'])
    base_a, base_b = arrays['cross_covariance'], arrays['covariance'][..., 1:, :, :]
    for row in spectra:
        ik, step = row['K_index'], row['transition']
        selection = row['operator_indices']
        aa, bb = ((oriented_a, oriented_b) if row['family'] == 'orientation' else (base_a, base_b))
        a = aa[ik, :, :, step].mean((0, 1))[np.ix_(selection, selection)]
        b = bb[ik, :, :, step].mean((0, 1))[np.ix_(selection, selection)]
        scale = 1 / np.sqrt(np.diag(b))
        bw = scale[:, None] * b * scale[None, :]
        aw = scale[:, None] * a * scale[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh((bw + bw.T) / 2)
        keep = eigenvalues > row['rcond'] * eigenvalues[-1]
        q = eigenvectors[:, keep]
        reduced = (q.T @ aw @ q) / eigenvalues[keep, None]
        values, modes = np.linalg.eig(reduced)
        order = np.argsort(-abs(values), kind='stable')
        values, modes = values[order], modes[:, order]
        perturbation = np.array([1. if physical_index == 0 or
            (row['family'] == 'orientation' and physical_index == 1) else 0. for physical_index in selection])
        energy = perturbation / scale  # same similarity coordinates as Tw
        projected = q.T @ energy
        if np.linalg.norm(projected) > 1e-15:
            overlap = abs(modes.conj().T @ projected) ** 2 / (np.sum(abs(modes) ** 2, axis=0) * np.linalg.norm(projected) ** 2)
            coefficients = np.linalg.pinv(modes, rcond=RESPONSE_RCOND) @ projected
            contributions = abs(coefficients) ** 2 * np.sum(abs(modes) ** 2, axis=0)
            weights = contributions / contributions.sum() if contributions.sum() else contributions
        else:
            overlap, weights = np.zeros(len(values)), np.zeros(len(values))
        row['energy_perturbation_retained_fraction'] = float(np.linalg.norm(projected) ** 2 / max(np.linalg.norm(energy) ** 2, 1e-30))
        row['energy_right_overlap'] = overlap.tolist()
        row['energy_modal_excitation_weights'] = weights.tolist()
        row['energy_projection_eigenvalues'] = np.column_stack((values.real, values.imag)).tolist()
        row['energy_projection_caution'] = '最近邻能量耦合扰动在共同预条件保留空间中的投影；各模式全保留，无理论接近筛选；重叠依赖算符基、截断及坐标度量，不单独认证热模。'
    return spectra


def response_analysis(arrays, bootstrap_indices):
    """General d analysis, retaining the physical-ensemble clustering."""
    vin, vout, prediction = (arrays[key] for key in ('q_response_in', 'q_response_out', 'q_response_predicted'))
    std = arrays['score_std'][..., :-1, :]
    inverse_std = np.divide(1., std, out=np.zeros_like(std), where=std > 1e-12)
    x, y, pred = (value * inverse_std[..., None, :] for value in (vin, vout, prediction))
    rows = []
    for ik, k_value in enumerate(arrays['K']):
        for step in range(2):
            sl = (ik, slice(None), slice(None), step)
            matrices = arrays['full_DR'][sl]
            target = y[sl][..., :6]
            err_r = pred[sl][..., :6] - target
            err_best = arrays['best_response_residual_even'][sl]
            denom = np.maximum(np.linalg.norm(target, axis=(-2, -1)), 1e-15)
            relative_r = np.linalg.norm(err_r, axis=(-2, -1)) / denom
            relative_best = np.linalg.norm(err_best, axis=(-2, -1)) / denom
            # Scores are unit-standardized; direction stays unchanged.
            energy = x[sl][..., 0]
            energy_out = y[sl][..., 0]
            norm_squared = np.sum(energy ** 2, axis=-1)
            r_energy = np.einsum('...ij,...j->...i', matrices, energy)
            projection = np.divide(np.sum(energy * r_energy, axis=-1), norm_squared,
                                   out=np.zeros_like(norm_squared), where=norm_squared > 1e-30)
            off = np.linalg.norm(r_energy - projection[..., None] * energy, axis=-1) / np.maximum(np.linalg.norm(r_energy, axis=-1), 1e-15)
            gain = np.linalg.norm(energy_out, axis=-1) / np.maximum(np.linalg.norm(energy, axis=-1), 1e-15)
            rank = arrays['response_input_rank_even'][sl]
            ranks, counts = np.unique(rank, return_counts=True)
            cluster_r, cluster_best = relative_r.mean(1), relative_best.mean(1)
            # These intervals condition on this model and C draws; only 12
            # original ensembles are the independent resampling unit.
            interval = lambda z: np.percentile(z[bootstrap_indices].mean(1), [2.5, 50, 97.5], axis=0).tolist()
            modes = [energy_mode_diagnostics(matrix, vector)
                     for matrix, vector in zip(matrices.reshape(-1, matrices.shape[-1], matrices.shape[-1]),
                                               energy.reshape(-1, energy.shape[-1]))]
            rows.append(dict(K_index=ik, K=float(k_value), transition=step,
                mean_g=arrays['g'][sl].mean((0, 1)).tolist(), full_DR_mean=matrices.mean((0, 1)).tolist(),
                best_J_even_mean=arrays['best_J_even'][sl].mean((0, 1)).tolist(),
                input_response_rank_counts={str(int(r)):int(n) for r, n in zip(ranks, counts)},
                input_response_singular_values_mean=arrays['response_input_singular_values_even'][sl].mean((0, 1)).tolist(),
                input_response_effective_rank_mean=float(arrays['response_input_effective_rank_even'][sl].mean()),
                existing_R_relative_error_mean=float(cluster_r.mean()), best_J_relative_error_mean=float(cluster_best.mean()),
                existing_R_relative_error_per_replica=cluster_r.tolist(), best_J_relative_error_per_replica=cluster_best.tolist(),
                existing_R_relative_error_percentiles=interval(cluster_r), best_J_relative_error_percentiles=interval(cluster_best),
                signed_R_residual_mean=err_r.mean((0, 1)).tolist(), signed_best_residual_mean=err_best.mean((0, 1)).tolist(),
                signed_R_residual_percentiles=interval(err_r.mean(1)), signed_best_residual_percentiles=interval(err_best.mean(1)),
                R_energy_projection_mean=float(projection.mean()), R_energy_off_direction_fraction_mean=float(off.mean()),
                observed_energy_gain_mean=float(gain.mean()), full_DR_modes_per_replica_draw=modes,
                best_J_per_replica_draw=arrays['best_J_even'][sl].tolist(),
                caution='每系综每C抽样单独拟合最佳矩阵；响应秩不足时只识别输入响应子空间，未识别的离流形DR谱不能解释为热指数。最佳矩阵残差小不证明物理闭合。能量投影和模态重叠均不是认证热倍率。'))
    validations = {}
    for direction in ('in', 'out'):
        fd = arrays['q_fd_response_' + direction]
        target = arrays['q_response_' + direction][..., None, :, :]
        delta = fd - target
        validations[direction] = dict(max_absolute_error=float(abs(delta).max()),
            relative_frobenius_error=float(np.linalg.norm(delta) / max(np.linalg.norm(np.broadcast_to(target, fd.shape)), 1e-300)))
    delta = arrays['r_fd_jacobian'] - arrays['full_DR']
    validations['full_DR'] = dict(h=1e-5, identity_included=True, max_absolute_error=float(abs(delta).max()),
        relative_frobenius_error=float(np.linalg.norm(delta) / max(np.linalg.norm(arrays['full_DR']), 1e-300)))
    return rows, validations


def draw_report(summary, output):
    """All visible plot prose is Chinese; variables preserve their notation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_ising_mcrg import BASES, RCONDS
    from scripts.report_ising_zh import chinese_figures
    output = Path(output)
    ik = int(np.argmin(abs(np.asarray(summary['K']) - KC)))
    with chinese_figures():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for step, ax in enumerate(axes):
            for threshold, marker in zip(RCONDS, ('o', 's', '^')):
                for row in summary['spectra']:
                    if row['family'] != 'isotropic' or row['K_index'] != ik or row['transition'] != step or row['rcond'] != threshold:
                        continue
                    ci = row['bootstrap']['leading_positive_real_percentiles']
                    value = row['leading_positive_real']
                    if value is None:
                        continue
                    xpos = int(row['basis']) + (np.log10(threshold) + 6) * .04
                    ax.scatter(xpos, value, marker=marker, label=f'截断阈值 {threshold:g}' if row['basis'] == '1' else None)
                    if ci is not None:
                        ax.plot([xpos, xpos], [ci[0], ci[2]], color='gray', alpha=.4)
            ax.axhline(2, color='purple', ls=':', label='理论热倍率（仅作参照）')
            ax.set(title=('108→54' if step == 0 else '54→27')+'：有限尺度物理响应谱',
                   xlabel='各向同性偶算符数量', ylabel='偶投影最大正实谱值')
            ax.set_xticks(BASES); ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(f"潜变量维数={summary['theory_dim']}：固定无监督 C，未按接近理论筛选谱")
        fig.supxlabel('区间以独立系综为重采单位；不含有限尺寸、算符截断和训练种子系统误差。', fontsize=9)
        fig.savefig(output / 'mcrg_basis.png', dpi=170); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for step, ax in enumerate(axes):
            row = next(r for r in summary['q_response'] if r['K_index'] == ik and r['transition'] == step)
            labels = ['已训练 R', '每系综最佳局部矩阵']
            values = [row['existing_R_relative_error_mean'], row['best_J_relative_error_mean']]
            cis = [row['existing_R_relative_error_percentiles'], row['best_J_relative_error_percentiles']]
            ax.bar(range(2), 100 * np.asarray(values), color=['tab:red', 'tab:blue'], alpha=.7)
            for i, ci in enumerate(cis):
                ax.plot([i, i], 100 * np.asarray([ci[0], ci[2]]), color='black')
            ax.set_xticks(range(2), labels)
            ax.set(title=('108→54' if step == 0 else '54→27')+'：六种偶扰动的编码响应闭合',
                   ylabel='相对响应残差（%）')
            ax.grid(axis='y', alpha=.2)
        fig.suptitle(f"潜变量维数={summary['theory_dim']}：矩阵拟合不混合不同基准系综")
        fig.supxlabel('坐标范数随模型改变；跨维度误差不能当作同一物理坐标绝对误差。', fontsize=9)
        fig.savefig(output / 'vector_response.png', dpi=170); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for step, ax in enumerate(axes):
            row = next(r for r in summary['q_response'] if r['K_index'] == ik and r['transition'] == step)
            singular = row['input_response_singular_values_mean']
            ax.semilogy(np.arange(1, len(singular) + 1), np.maximum(singular, 1e-18), 'o-')
            ax.set_xticks(range(1, len(singular) + 1))
            ax.set(title=('108→54' if step == 0 else '54→27')+'：可辨识响应方向',
                   xlabel='奇异值序号', ylabel='单位波动算符扰动的输入响应奇异值')
            ax.text(.03, .05, '数值秩计数：'+str(row['input_response_rank_counts'])+
                    f"\n平均有效秩：{row['input_response_effective_rank_mean']:.3g}", transform=ax.transAxes, fontsize=9)
            ax.grid(alpha=.2)
        fig.suptitle('额外坐标不等于额外物理方向；小奇异值及离响应空间谱需谨慎解释')
        fig.savefig(output / 'response_conditioning.png', dpi=170); plt.close(fig)


def diagnose(source_run, data, output, *, k_count=3, c_repeats=5, seed=12345,
             expected_checkpoint_sha256=None, smoke=False, bootstraps=1000):
    source_run, data, output = map(lambda p: Path(p).resolve(), (source_run, data, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing overwrite: {output}')
    if source_run == output or source_run in output.parents or output in source_run.parents:
        raise ValueError('Diagnostic output must be separate from source run')
    if type(c_repeats) is not int or c_repeats < 1 or type(bootstraps) is not int or bootstraps < 100:
        raise ValueError('Require positive C repeat count and at least 100 bootstrap draws')
    torch.set_num_threads(1)
    checkpoint = source_run / 'training_step_12000.pt'
    fingerprint = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is not None and fingerprint != expected_checkpoint_sha256:
        raise ValueError('Numbered source checkpoint fingerprint mismatch')
    loaded = load_run(source_run, 12000)
    metadata = loaded.payload.get('metadata', {})
    summary = json.loads((source_run / 'summary.json').read_text())
    manifest = validate_source(source_run, data, loaded, summary)
    model = loaded.model.double().eval()
    dimension = int(loaded.config['model']['theory_dim'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_hash = model_sha256(model)
    dataset = mmap_dataset(data)
    label_values = np.concatenate([labels.detach().cpu().numpy().reshape(-1) for _, labels, _ in dataset])
    k_values = select_points(label_values, k_count)
    selected = []
    for value in k_values:
        rows = [(configurations, i, split) for configurations, labels, split in dataset
                for i in np.flatnonzero(labels.detach().cpu().numpy().reshape(-1) == value)]
        if len(rows) != 12:
            raise ValueError('Require exactly 12 independent physical ensembles at each selected K')
        selected.append(rows[:1] if smoke else rows)
    patterns, probabilities = c_probability_table(model)
    probability_symmetry_error = probabilities + probabilities.flip(0) - 1
    probability_d4_error = d4_probability_errors(patterns, probabilities)
    nrep = 1 if smoke else 12
    shape = (len(k_values), nrep, c_repeats)
    arrays = dict(
        operators=np.empty(shape + (3, 128, 7)), means=np.empty(shape + (3, 7)),
        orientation_operators=np.empty(shape + (3, 128, 7)),
        covariance=np.empty(shape + (3, 7, 7)), cross_covariance=np.empty(shape + (2, 7, 7)),
        g=np.empty(shape + (3, dimension)), full_DR=np.empty(shape + (3, dimension, dimension)),
        r_fd_jacobian=np.empty(shape + (3, dimension, dimension)),
        q_response_micro=np.empty(shape + (3, dimension, 7)),
        q_response_in=np.empty(shape + (2, dimension, 7)), q_response_out=np.empty(shape + (2, dimension, 7)),
        q_response_predicted=np.empty(shape + (2, dimension, 7)), score_std=np.empty(shape + (3, 7)),
        q_fd_response_in=np.empty(shape + (2, 2, dimension, 7)),
        q_fd_response_out=np.empty(shape + (2, 2, dimension, 7)),
        symmetry_sample_mismatch=np.empty(shape + (3,)), symmetry_magnetization_sum=np.empty(shape + (3,)),
    )
    for name, count in (('even', 6), ('all', 7)):
        arrays.update({f'best_J_{name}':np.empty(shape + (2, dimension, dimension)),
            f'response_input_rank_{name}':np.empty(shape + (2,), dtype=int),
            f'response_input_singular_values_{name}':np.empty(shape + (2, dimension)),
            f'response_input_effective_rank_{name}':np.empty(shape + (2,)),
            f'response_input_projector_{name}':np.empty(shape + (2, dimension, dimension)),
            f'best_response_residual_{name}':np.empty(shape + (2, dimension, count))})
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(output / 'status.json', dict(status='running', source_run=str(source_run),
        K=k_values.tolist(), theory_dim=dimension, c_repeats=c_repeats, completed_ensembles=0))
    try:
        for ik, rows in enumerate(selected):
            for replica, (source, source_index, split) in enumerate(rows):
                x0 = source[source_index].detach().to(torch.float64).clone()
                if tuple(x0.shape) != (128, 108, 108, 1):
                    raise ValueError('Selected source ensemble must have [128,108,108,1] shape')
                x0 = x0[..., 0]
                op0, oriented0 = operator_values(x0), orientation_operator_values(x0)
                feature0 = local_feature_means(model, x0)
                for draw in range(c_repeats):
                    generator = torch.Generator(device='cpu').manual_seed(seed + ik * 1_000_000 + replica * 10_000 + draw)
                    levels, flipped = [x0], [-x0]
                    for step in range(2):
                        uniform = torch.rand((128, levels[-1].shape[-2] // 2, levels[-1].shape[-1] // 2),
                                             dtype=torch.float64, generator=generator)
                        levels.append(c_draw(levels[-1], probabilities, uniform))
                        flipped.append(c_draw(flipped[-1], probabilities, 1 - uniform))
                    ops = [op0] + [operator_values(level) for level in levels[1:]]
                    oriented = [oriented0] + [orientation_operator_values(level) for level in levels[1:]]
                    features = [feature0] + [local_feature_means(model, level) for level in levels[1:]]
                    ix = (ik, replica, draw)
                    arrays['operators'][ix] = torch.stack(ops).numpy()
                    arrays['orientation_operators'][ix] = torch.stack(oriented).numpy()
                    arrays['means'][ix] = torch.stack([op.mean(0) for op in ops]).numpy()
                    arrays['covariance'][ix] = torch.stack([covariance(op) for op in ops]).numpy()
                    arrays['cross_covariance'][ix] = torch.stack([covariance(ops[k + 1], ops[k]) for k in range(2)]).numpy()
                    for scale in range(3):
                        g, response_micro, _, _ = weighted_q_response(model, features[scale], op0)
                        arrays['g'][ix + (scale,)] = g.numpy()
                        arrays['full_DR'][ix + (scale,)] = full_r_jacobian(model, g).numpy()
                        arrays['r_fd_jacobian'][ix + (scale,)] = full_r_finite_difference(model, g).numpy()
                        arrays['q_response_micro'][ix + (scale,)] = response_micro.numpy()
                        arrays['score_std'][ix + (scale,)] = ops[scale].std(0, correction=0).numpy()
                        arrays['symmetry_sample_mismatch'][ix + (scale,)] = float((levels[scale] != -flipped[scale]).double().mean())
                        arrays['symmetry_magnetization_sum'][ix + (scale,)] = float((levels[scale] + flipped[scale]).mean())
                    for step in range(2):
                        _, vin, fin, _ = weighted_q_response(model, features[step], ops[step])
                        _, vout, fout, _ = weighted_q_response(model, features[step + 1], ops[step])
                        arrays['q_response_in'][ix + (step,)] = vin.numpy()
                        arrays['q_response_out'][ix + (step,)] = vout.numpy()
                        arrays['q_response_predicted'][ix + (step,)] = arrays['full_DR'][ix + (step,)] @ vin.numpy()
                        arrays['q_fd_response_in'][ix + (step,)] = fin.numpy()
                        arrays['q_fd_response_out'][ix + (step,)] = fout.numpy()
                        for name, count in (('even', 6), ('all', 7)):
                            fit = best_local_jacobian(vin[:, :count], vout[:, :count], arrays['score_std'][ix + (step,)][:count])
                            arrays[f'best_J_{name}'][ix + (step,)] = fit['jacobian']
                            arrays[f'response_input_rank_{name}'][ix + (step,)] = fit['input_rank']
                            arrays[f'response_input_singular_values_{name}'][ix + (step,)] = fit['input_singular_values']
                            arrays[f'response_input_effective_rank_{name}'][ix + (step,)] = fit['input_effective_rank']
                            arrays[f'response_input_projector_{name}'][ix + (step,)] = fit['projector']
                            arrays[f'best_response_residual_{name}'][ix + (step,)] = fit['residual']
                completed = ik * nrep + replica + 1
                write_json(output / 'status.json', dict(status='running', source_run=str(source_run),
                    K=k_values.tolist(), theory_dim=dimension, c_repeats=c_repeats, completed_ensembles=completed,
                    elapsed_seconds=time.monotonic() - start))
                print(json.dumps(dict(completed_ensembles=completed, total_ensembles=len(k_values) * nrep,
                    K=float(k_values[ik]), replica=replica, elapsed_seconds=time.monotonic() - start)), flush=True)
        after_hash = model_sha256(model)
        if before_hash != after_hash or any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError('Read-only model state or parameter gradients changed')
        arrays.update(K=k_values, operator_names=np.array(OPERATOR_NAMES),
            c_draw_seeds=np.array([[[seed + ik * 1_000_000 + replica * 10_000 + draw
                for draw in range(c_repeats)] for replica in range(nrep)] for ik in range(len(k_values))]),
            source_indices=np.array([[row[1] for row in rows] for rows in selected]),
            replica_index=np.array([[row[1] for row in rows] for rows in selected]),
            split_index=np.array([[0 if row[2] == 'train' else 1 for row in rows] for rows in selected]),
            source_splits=np.array([[row[2] for row in rows] for rows in selected]),
            c_patterns=patterns.numpy(), c_p_plus=probabilities.numpy(),
            c_spin_flip_probability_error=probability_symmetry_error.numpy(), c_d4_probability_error=probability_d4_error.numpy())
        np.savez_compressed(output / 'operator_statistics.npz', **arrays)
        from scripts.report_ising_mcrg import build_spectra, clean
        indices = np.random.default_rng(20261015).integers(0, nrep, size=(bootstraps, nrep))
        q_response, derivative_validation = response_analysis(arrays, indices)
        diagnostic = dict(status='complete', source_run=str(source_run), source_checkpoint=str(checkpoint),
            source_epoch=12000, theory_dim=dimension, source_checkpoint_sha256=fingerprint,
            source_manifest_sha256=sha256_file(source_run / 'manifest.json'), source_data=str(data),
            source_data_file_bytes=data.stat().st_size, source_data_sha256_from_checkpoint=metadata.get('data_sha256'),
            source_data_sha256_from_manifest=manifest.get('data_sha256'), source_data_rehashed=False,
            source_data_loading='torch.load mmap=True; selected existing ensembles only, no full data audit',
            source_config=canonical_config(loaded.config), source_labels_used_in_training=False,
            labels_used_here='post-hoc Kc and adjacent evaluation-cell selection only',
            model_sha256_before=before_hash, model_sha256_after=after_hash,
            no_training=True, no_MI=True, no_new_physical_sampling=True, no_QC_parameter_changes=True,
            device='cpu', threads=1, smoke=smoke, independent_replicas=nrep, c_repeats=c_repeats, seed=seed,
            K=k_values.tolist(), critical_reference_K=KC, scales=[108, 54, 27],
            operator_names=OPERATOR_NAMES, operator_definitions=OPERATOR_DEFINITIONS,
            orientation_operator_names=ORIENTATION_NAMES,
            operator_normalization='extensive sums; no separate fine/coarse area normalization',
            covariance_definition='unbiased within each 128-configuration ensemble (N-1); no pooling physical replicas',
            cross_covariance_definition='A=Cov(S_coarse,S_fine), B=Cov(S_coarse,S_coarse), solve B T=A',
            q_response_definition='head Jacobian times population covariance of per-configuration local features with extensive physical score',
            q_response_transition_definition='same current-scale perturbation of paired configurations on input/output',
            q_response_micro_definition='same microscopic source perturbation through all three saved-C scales',
            q_fd_definition='fixed C draws, exp(+-epsilon centered score), epsilon=t/population_std(score)', q_fd_t_values=[.001, .003],
            best_J_definition='per ensemble/draw/transition, fit J on six even or all seven score-standardized response directions; minimum norm outside identified span is not physical',
            best_J_rcond=RESPONSE_RCOND, local_response_summary=q_response,
            derivative_validation=derivative_validation,
            c_repeats_interpretation='conditional C draws on the same physical configurations, not independent physical replicas',
            symmetry_definition='paired spin inversion with antithetic 1-u, diagnostic only, not augmented or pooled into statistics',
            symmetry_probability_max_abs_error=float(probability_symmetry_error.abs().max()),
            d4_probability_max_abs_error=float(probability_d4_error.abs().max()),
            d4_probability_rms_error=float(probability_d4_error.square().mean().sqrt()),
            array_axes='K, independent replica, C draw, then scale/transition, optional FD epsilon, latent coordinates, physical operators',
            arrays={key:list(value.shape) for key, value in arrays.items()},
            limitation='One trained seed, finite 108/54/27, B0, 12 replicas x128 configurations. Numeric response rank depends on coordinate metric and noise; rank alone does not certify physical dimension. Off-manifold DR modes and energy overlaps are not certified thermal exponents.',
            elapsed_seconds=time.monotonic() - start)
        spectra = add_physical_energy_projections(build_spectra(arrays, indices), arrays)
        comparison = clean(dict(status='complete', source_diagnostic=diagnostic, theory_dim=dimension,
            K=arrays['K'], Kc_reference=KC, independent_replicas=nrep, c_repeats=c_repeats, bootstraps=bootstraps,
            spectra=spectra, q_response=q_response, derivative_validation=derivative_validation,
            thermal_identification='No selection by proximity to theory. Full spectra retained. Physical energy perturbation projections are descriptive; latent off-manifold eigenmodes are unverified. No independent magnetic exponent from B0.',
            protocol='固定各臂实际12000无监督模型；无训练、无MI、无新物理样本；五次C抽样；系综成组重采。'))
        write_json(output / 'diagnostic.json', clean(diagnostic))
        write_json(output / 'comparison.json', comparison)
        draw_report(comparison, output)
        write_json(output / 'status.json', dict(status='complete', workflow_complete=True,
            completed_ensembles=len(k_values) * nrep, theory_dim=dimension,
            elapsed_seconds=time.monotonic() - start))
        return diagnostic
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', workflow_complete=False,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - start))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k-count', type=int, default=3)
    parser.add_argument('--c-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true', help='One replica per evaluation cell; not formal evidence')
    args = parser.parse_args(argv)
    diagnose(**vars(args))


if __name__ == '__main__':
    main()
