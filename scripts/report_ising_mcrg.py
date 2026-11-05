"""固定无监督 C 的有限尺度算符响应与标量表示诊断；不加载模型或物理数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_zh import chinese_figures


EVEN_NAMES = ('最近邻', '对角邻居', '单位方格', '轴向距离二', '矩形四角', '轴向距离三')
RCONDS = (1e-8, 1e-6, 1e-4)
BASES = (1, 3, 5, 6)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def solve_response(A, B, rcond=1e-6, odd_index=None):
    """Solve B T=A with the SAME operator rescaling on both sides.

    D=diag(B)^(-1/2), Bw=D B D, Aw=D A D. The full-rank result
    Tw=D^(-1) T D is a similarity transform, not separate fine/coarse
    standardization. Rank-reduced spectra contain only retained modes.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if (A.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]
            or not np.isfinite(A).all() or not np.isfinite(B).all()
            or not 0 < rcond < 1):
        raise ValueError('Need finite square A/B and 0<rcond<1')
    B = .5 * (B + B.T)
    if np.any(np.diag(B) <= 0):
        raise ValueError('Every retained operator must have positive coarse variance')
    scale = 1 / np.sqrt(np.diag(B))
    Bw = scale[:, None] * B * scale[None, :]
    Aw = scale[:, None] * A * scale[None, :]
    values, vectors = np.linalg.eigh(Bw)
    keep = values > rcond * values[-1]
    if not keep.any():
        raise ValueError('No covariance modes retained')
    Q = vectors[:, keep]
    reduced = (Q.T @ Aw @ Q) / values[keep, None]
    eigenvalues, eigenvectors = np.linalg.eig(reduced)
    order = np.argsort(-np.abs(eigenvalues), kind='stable')
    eigenvalues, eigenvectors = eigenvalues[order], Q @ eigenvectors[:, order]
    T = Q @ reduced @ Q.T
    positive = [float(z.real) for z in eigenvalues
                if abs(z.imag) <= 1e-8 * max(1., abs(z.real)) and z.real > 0]
    lead = max(positive) if positive else None
    result = dict(eigenvalues=np.column_stack((eigenvalues.real, eigenvalues.imag)).tolist(),
        leading_positive_real=lead, retained_rank=int(keep.sum()), dimension=len(B),
        condition_number=float(values[-1] / values[0]) if values[0] > 0 else None,
        retained_condition_number=float(values[-1] / values[keep][0]),
        covariance_eigenvalues=values.tolist(), operator_rescaling=scale.tolist(),
        residual=float(np.linalg.norm(Bw @ T - Aw) / max(np.linalg.norm(Aw), 1e-300)),
        whitened_response_matrix=T.tolist(),
        caution='最大正实根按数值大小定义，未按接近理论筛选；非认证热倍率。小代数残差不证明物理闭合。')
    if odd_index is not None:
        if not 0 <= odd_index < len(B):
            raise ValueError('Invalid odd operator index')
        result['odd_coordinate_fraction'] = (
            np.abs(eigenvectors[odd_index]) ** 2 /
            np.sum(np.abs(eigenvectors) ** 2, axis=0)).tolist()
        result['odd_fraction_caution'] = '在共同预条件坐标中的描述性重叠，不是热模认证或坐标不变量。'
    return result


def bootstrap_response(A_by_rep, B_by_rep, indices, rcond=1e-6, odd_index=None):
    vals, ranks, complex_dominant, failed, no_positive = [], [], 0, 0, 0
    for ids in indices:
        try:
            solved = solve_response(A_by_rep[ids].mean(0), B_by_rep[ids].mean(0), rcond, odd_index)
            ranks.append(solved['retained_rank'])
            spectrum = np.asarray(solved['eigenvalues'])
            if abs(spectrum[0, 1]) > 1e-8 * max(1., abs(spectrum[0, 0])):
                complex_dominant += 1
            if solved['leading_positive_real'] is not None:
                vals.append(solved['leading_positive_real'])
            else:
                no_positive += 1
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
    return dict(leading_positive_real_percentiles=np.percentile(vals, [2.5, 50, 97.5]).tolist() if vals else None,
        bootstrap_successes=len(vals), bootstrap_failures=failed,
        no_positive_real_draws=no_positive,
        retained_rank_counts={str(rank): ranks.count(rank) for rank in sorted(set(ranks))},
        complex_dominant_fraction=complex_dominant / len(indices),
        caution='先平均同一系综的C抽样协方差，再成组重采独立系综；每次取最大正实谱值，可能切换模式，非认证热模区间；不含训练、有限尺寸或算符截断系统误差。')


def cluster_interval(values, indices):
    return np.percentile(np.asarray(values)[indices].mean(1), [2.5, 50, 97.5], axis=0)


def covariance_pair(operators):
    centered = operators - operators.mean(-2, keepdims=True)
    n = operators.shape[-2]
    B = np.einsum('...sni,...snj->...sij', centered, centered) / (n - 1)
    A = np.einsum('...sni,...snj->...sij', centered[..., 1:, :, :],
                  centered[..., :-1, :, :]) / (n - 1)
    return A, B[..., 1:, :, :]


def build_spectra(arrays, indices):
    A = arrays['cross_covariance'].mean(2)  # K, independent replica, transition, op, op
    B = arrays['covariance'].mean(2)[:, :, 1:]
    families = [('isotropic', A, B, [(str(n), list(range(n))) for n in BASES])]
    families.append(('mixed', A, B, [('6_even_plus_M', list(range(7)))]))
    if 'orientation_operators' in arrays:
        oa, ob = covariance_pair(arrays['orientation_operators'])
        families.append(('orientation', oa.mean(2), ob.mean(2),
                         [('axes_2', [0, 1]), ('oriented_7', list(range(7)))]))
    rows = []
    for family, fa, fb, selections in families:
        for ik in range(len(arrays['K'])):
            for transition in range(2):
                for basis, selected in selections:
                    aa = fa[ik, :, transition][:, selected][:, :, selected]
                    bb = fb[ik, :, transition][:, selected][:, :, selected]
                    for rcond in RCONDS:
                        odd = 6 if family == 'mixed' else None
                        solved = solve_response(aa.mean(0), bb.mean(0), rcond, odd)
                        bs = bootstrap_response(aa, bb, indices, rcond, odd)
                        repeat_values = []
                        if family in ('isotropic', 'mixed') and rcond == 1e-6:
                            for repeat in range(arrays['operators'].shape[2]):
                                ca = arrays['cross_covariance'][ik, :, repeat, transition]
                                cb = arrays['covariance'][ik, :, repeat, transition + 1]
                                ca = ca[:, selected][:, :, selected].mean(0)
                                cb = cb[:, selected][:, :, selected].mean(0)
                                repeat_values.append(solve_response(ca, cb, rcond, odd)['leading_positive_real'])
                        rows.append(dict(family=family, K_index=ik, K=float(arrays['K'][ik]),
                            transition=transition, basis=basis, operator_indices=selected,
                            rcond=rcond, C_repeat_leading_positive_real=repeat_values,
                            **solved, bootstrap=bs))
    return rows


def response_analysis(arrays, indices):
    vin, vout = arrays['q_response_in'][..., 0, :], arrays['q_response_out'][..., 0, :]
    pred = arrays['q_response_predicted'][..., 0, :]
    sigma = arrays['score_std'][:, :, :, :-1]
    valid = sigma > 1e-12
    normalize = np.where(valid, sigma, np.inf)
    x, y, p = vin / normalize, vout / normalize, pred / normalize
    # Fit a separate scalar Jacobian at EACH ensemble/draw/scale. Never force
    # different latent g values to share one linearization.
    x_even, y_even = x[..., :6], y[..., :6]
    denom = np.sum(x_even ** 2, axis=-1)
    best = np.divide(np.sum(x_even * y_even, axis=-1), denom,
                     out=np.zeros_like(denom), where=denom > 1e-24)
    residual_R = p[..., :6] - y_even
    residual_best = best[..., None] * x_even - y_even
    target_norm = np.maximum(np.linalg.norm(y_even, axis=-1), 1e-15)
    rows = []
    for ik in range(len(arrays['K'])):
        for transition in range(2):
            # C draws are conditional repeats, not independent physical ensembles.
            rr = residual_R[ik, :, :, transition].mean(1)
            rb = residual_best[ik, :, :, transition].mean(1)
            r_rel = (np.linalg.norm(residual_R[ik, :, :, transition], axis=-1) /
                     target_norm[ik, :, :, transition]).mean(1)
            b_rel = (np.linalg.norm(residual_best[ik, :, :, transition], axis=-1) /
                     target_norm[ik, :, :, transition]).mean(1)
            sl = (ik, slice(None), slice(None), transition)
            ratio = np.divide(vout[sl], vin[sl], out=np.full_like(vout[sl], np.nan),
                              where=np.abs(vin[sl]) > 1e-12)
            rows.append(dict(K_index=ik, transition=transition,
                mean_g=float(arrays['g'][ik, :, :, transition, 0].mean()),
                mean_DR=float(arrays['full_DR'][ik, :, :, transition, 0, 0].mean()),
                mean_best_local_scalar=float(best[sl].mean()),
                existing_R_relative_response_error=float(r_rel.mean()),
                best_local_scalar_relative_response_error=float(b_rel.mean()),
                existing_R_error_percentiles=cluster_interval(r_rel, indices).tolist(),
                best_scalar_error_percentiles=cluster_interval(b_rel, indices).tolist(),
                signed_R_residual_mean=rr.mean(0).tolist(),
                signed_R_residual_percentiles=cluster_interval(rr, indices).tolist(),
                signed_best_residual_mean=rb.mean(0).tolist(),
                signed_best_residual_percentiles=cluster_interval(rb, indices).tolist(),
                physical_response_in_mean=vin[sl].mean((0, 1)).tolist(),
                physical_response_out_mean=vout[sl].mean((0, 1)).tolist(),
                per_draw_per_replica_response_ratios=clean(ratio),
                best_scalar_per_draw_per_replica=best[sl].tolist(),
                C_repeat_signed_R_sd=float(np.std(residual_R[sl], axis=1, ddof=1).mean()),
                caution='最佳标量为每个基准系综/抽样单独拟合；正残差范数不构成缺维证明。只有两次有限尺度变换。'))
    fd = {}
    for name, target in [('in', vin), ('out', vout)]:
        key = 'q_fd_response_' + name
        if key in arrays:
            actual = arrays[key][..., 0, :]
            expected = target[..., None, :]
            difference = np.abs(actual - expected)
            fd[name] = dict(max_absolute_error=float(difference.max()),
                relative_frobenius_error=float(np.linalg.norm(difference) /
                     max(np.linalg.norm(np.broadcast_to(expected, actual.shape)), 1e-300)))
    return rows, fd


def draw(summary, output):
    ik = int(np.argmin(np.abs(np.asarray(summary['K']) - summary['Kc_reference'])))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    for transition, ax in enumerate(axes):
        for rcond, marker in zip(RCONDS, ('o', 's', '^')):
            rows = [r for r in summary['spectra'] if r['family'] == 'isotropic' and
                    r['K_index'] == ik and r['transition'] == transition and r['rcond'] == rcond]
            for row in rows:
                ci = row['bootstrap']['leading_positive_real_percentiles']
                value = row['leading_positive_real']
                if ci is not None and value is not None:
                    x = int(row['basis']) + (np.log10(rcond) + 6) * .04
                    ax.plot([x, x], [ci[0], ci[2]], color='gray', alpha=.35)
                    ax.scatter(x, value, marker=marker, s=28,
                               label=f'阈值 {rcond:g}' if row['basis'] == '1' else None)
        ax.axhline(2., color='purple', ls=':', label='理论热倍率（仅参照）')
        rrow = next(r for r in summary['q_response'] if r['K_index'] == ik and r['transition'] == transition)
        ax.axhline(rrow['mean_DR'], color='tab:red', ls='--', label='原标量 R 的平均斜率')
        ax.set(title=('108→54' if transition == 0 else '54→27')+'：有限尺度响应谱',
               xlabel='各向同性偶算符数量', ylabel='偶投影最大正实特征值')
        ax.set_xticks(BASES)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('固定原无监督 C：未按接近理论筛选模式或检查点')
    fig.supxlabel('误差线为独立系综成组重采区间；不含有限尺寸、算符截断和训练种子误差。', fontsize=9)
    fig.savefig(output / 'mcrg_basis.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
    x = np.arange(6)
    for transition, ax in enumerate(axes):
        row = next(r for r in summary['q_response'] if r['K_index'] == ik and r['transition'] == transition)
        for key, color, label, offset in [('R', 'tab:red', '原 R 斜率', -.08),
                                         ('best', 'tab:blue', '每系综最佳标量斜率', .08)]:
            mean = np.asarray(row['signed_'+key+'_residual_mean'])
            ci = np.asarray(row['signed_'+key+'_residual_percentiles'])
            ax.scatter(x + offset, mean, c=color, label=label, s=25)
            ax.vlines(x + offset, ci[0], ci[2], color=color, alpha=.65)
        ax.axhline(0., color='black', lw=.8)
        ax.set_xticks(x, EVEN_NAMES, rotation=35, ha='right')
        ax.set(title=('108→54' if transition == 0 else '54→27')+'：物理扰动响应闭合',
               ylabel='单位波动扰动下的编码响应残差', xlabel='独立偶算符扰动')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('更多维度是否有用：先检验任何局部标量映射能否解释多方向响应')
    fig.supxlabel('符号残差区间用于描述稳定方向偏差；不能用范数非零或单一窗口认证维度不足。', fontsize=9)
    fig.savefig(output / 'scalar_response.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstraps', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.bootstraps < 100:
        raise ValueError('At least 100 cluster bootstrap draws required')
    if args.output.exists():
        raise FileExistsError('Refusing overwrite or implicit rerun of a completed report')
    status = json.loads((args.run / 'status.json').read_text())
    if status.get('status') != 'complete':
        raise ValueError('Collection must complete before reporting')
    diagnostic = json.loads((args.run / 'diagnostic.json').read_text())
    if (diagnostic.get('source_epoch') != 12000 or
            diagnostic.get('source_labels_used_in_training') is not False or
            diagnostic.get('model_sha256_before') != diagnostic.get('model_sha256_after') or
            not diagnostic.get('model_sha256_before') or not diagnostic.get('no_training') or
            not diagnostic.get('no_MI') or not diagnostic.get('no_new_physical_sampling')):
        raise ValueError('Read-only fully unsupervised source provenance must be verified')
    with np.load(args.run / 'operator_statistics.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    nrep = arrays['operators'].shape[1]
    indices = np.random.default_rng(20261015).integers(0, nrep, size=(args.bootstraps, nrep))
    spectra = build_spectra(arrays, indices)
    q_response, fd = response_analysis(arrays, indices)
    summary = clean(dict(status='complete', K=arrays['K'], Kc_reference=.44068679350977147,
        source_diagnostic=diagnostic, bootstraps=args.bootstraps, independent_replicas=nrep,
        spectra=spectra, q_response=q_response, derivative_validation=fd,
        protocol='原12000无监督Q/C/R固定；无训练、无MI、无新物理采样；K仅选择后验诊断分布。',
        interpretation='有限尺度MCRG及经验物理响应，非认证热指数；不以接近理论筛选谱。额外维度未训练。'))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with chinese_figures():
        draw(summary, args.output)
    print(json.dumps(dict(status='complete', output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
