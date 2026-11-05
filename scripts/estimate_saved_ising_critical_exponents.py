"""从公共保存数组重算有限尺度物理耦合谱；不加载或运行模型。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_mcrg import clean, solve_response
from scripts.report_ising_zh import chinese_figures


SIZES = (108, 54, 27)
BASES = (1, 3, 5, 6)
RCONDS = (1e-8, 1e-6, 1e-4)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(result):
    pairs = np.asarray(result['eigenvalues'])
    return pairs[:, 0] + 1j * pairs[:, 1]


def inverse_y(value, block_factor):
    """只对正实相关模作算术换算；此函数不识别热方向。"""
    value = complex(value)
    if abs(value.imag) > 1e-8 * max(1., abs(value.real)) or value.real <= 1:
        return None
    return float(np.log(block_factor) / np.log(value.real))


def unique_even_candidate(result, block_factor):
    values = spectrum(result)
    candidates = [i for i, value in enumerate(values)
                  if inverse_y(value, block_factor) is not None]
    index = candidates[0] if len(candidates) == 1 else None
    return dict(positive_real_relevant_count=len(candidates), mode_index=index,
        multiplier=float(values[index].real) if index is not None else None,
        conditional_inverse_y=inverse_y(values[index], block_factor) if index is not None else None,
        thermal_nu_certified=False,
        caution='只记录有限偶算符基中唯一正实相关候选，不按接近理论或最大谱值选热模。')


def clustered_bootstrap(A, B, indices, block_factor):
    """输入已先平均同一物理系综的五次 C 抽样。"""
    multipliers, inverse_values, counts, ranks = [], [], [], []
    failures = 0
    for selection in indices:
        try:
            solved = solve_response(A[selection].mean(0), B[selection].mean(0))
            candidate = unique_even_candidate(solved, block_factor)
            counts.append(candidate['positive_real_relevant_count'])
            ranks.append(solved['retained_rank'])
            if candidate['multiplier'] is not None:
                multipliers.append(candidate['multiplier'])
                inverse_values.append(candidate['conditional_inverse_y'])
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    return dict(draws=len(indices), unique_candidate_draws=len(multipliers), failures=failures,
        positive_relevant_count_histogram={str(k): counts.count(k) for k in set(counts)},
        retained_rank_histogram={str(k): ranks.count(k) for k in set(ranks)},
        multiplier_percentiles=np.percentile(multipliers, [2.5, 50, 97.5]) if multipliers else None,
        conditional_inverse_y_percentiles=np.percentile(inverse_values, [2.5, 50, 97.5]) if inverse_values else None,
        caution='12个独立物理系综成组重采；同一抽样配对所有尺度。条件于单训练模型及固定 C 抽样，不含有限尺寸、算符截断、临界位置或训练种子系统误差。')


def direct_covariance(operators, first, last):
    """各128构型系综内部计算，绝不池化12系综或五次抽样。"""
    centered = operators - operators.mean(-2, keepdims=True)
    return np.einsum('...ni,...nj->...ij', centered[..., last, :, :],
                     centered[..., first, :, :]) / (operators.shape[-2] - 1)


def physical_estimates(root, *, bootstraps=1000):
    path = root / 'dim1/response/operator_statistics.npz'
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    if arrays['operators'].shape != (3, 12, 5, 3, 128, 7):
        raise ValueError('预期三个K、12系综、五次固定C抽样、三个尺度、128构型、七算符')
    critical_K = .5 * np.log(1 + np.sqrt(2))
    critical_index = int(np.argmin(abs(arrays['K'] - critical_K)))
    if abs(arrays['K'][critical_index] - critical_K) > 1e-12:
        raise ValueError('缺少已有Kc评估点')
    source_report = json.loads((root / 'dim1/response/comparison.json').read_text())
    # All seven original physical operators stay available, but the thermal
    # candidate screen below applies ONLY to the six even operators.
    a_direct = direct_covariance(arrays['operators'], 0, 2)
    rng = np.random.default_rng(20261016)
    indices = rng.integers(0, 12, (bootstraps, 12))
    rows, verification = [], []
    for ik, K in enumerate(arrays['K']):
        for first, last in ((0, 1), (1, 2), (0, 2)):
            A = (arrays['cross_covariance'][ik, :, :, first]
                 if last == first + 1 else a_direct[ik]).mean(1)
            B = arrays['covariance'][ik, :, :, last].mean(1)
            factor = 2 ** (last - first)
            for n in BASES:
                aa, bb = A[:, :n, :n], B[:, :n, :n]
                for cutoff in RCONDS:
                    solved = solve_response(aa.mean(0), bb.mean(0), cutoff)
                    candidate = unique_even_candidate(solved, factor)
                    if last == first + 1:
                        old = next(row for row in source_report['spectra']
                            if row['family'] == 'isotropic' and row['K_index'] == ik
                            and row['transition'] == first and row['basis'] == str(n)
                            and row['rcond'] == cutoff)
                        remaining = list(spectrum(old))
                        errors = []
                        for value in spectrum(solved):
                            match = int(np.argmin(abs(np.asarray(remaining) - value)))
                            errors.append(abs(value - remaining.pop(match)))
                        error = float(max(errors))
                        verification.append(error)
                        if error > 1e-8:
                            raise ValueError('重算谱与已有正式算符报告不一致')
                    else:
                        old = None
                    row = dict(K=float(K), K_index=ik, first_scale=first, last_scale=last,
                        sizes=[SIZES[first], SIZES[last]], total_block_factor=factor,
                        even_basis_size=n, rcond=cutoff, **solved,
                        arithmetic_inverse_y_all_modes=[inverse_y(value, factor) for value in spectrum(solved)],
                        unique_even_candidate=candidate)
                    if old is not None:
                        row['energy_right_overlap'] = old['energy_right_overlap']
                        row['energy_modal_excitation_weights'] = old['energy_modal_excitation_weights']
                    if ik == critical_index and n == 6 and cutoff == 1e-6:
                        row['conditional_ensemble_bootstrap'] = clustered_bootstrap(aa, bb, indices, factor)
                    rows.append(row)
    equality = []
    for dimension in (2, 3):
        other_path = root / f'dim{dimension}/response/operator_statistics.npz'
        with np.load(other_path, allow_pickle=False) as other:
            equality.append(dict(dimg=dimension, source=str(other_path.resolve()),
                source_sha256=sha256(other_path),
                physical_operators_bitwise_equal=bool(np.array_equal(arrays['operators'], other['operators'])),
                covariance_bitwise_equal=bool(np.array_equal(arrays['covariance'], other['covariance'])),
                cross_covariance_bitwise_equal=bool(np.array_equal(arrays['cross_covariance'], other['cross_covariance']))))
    return dict(source=str(path.resolve()), source_sha256=sha256(path),
        critical_K=critical_K, critical_index=critical_index, rows=rows,
        same_physical_statistics_across_dimensions=equality,
        original_formal_spectrum_max_abs_recomputation_error=max(verification),
        all_original_families_and_spectra=source_report['spectra'],
        definition='A=Cov(S_out,S_in), B=Cov(S_out,S_out), B T=A；广延算符，不额外除格点面积。',
        two_step_definition='108→27直接用Cov(S_27,S_108)重新求解，不是平均两次指数，也不是把两个最大谱值相乘。')


def load_curves(root):
    path = root / 'dim1/report/encodings.npz'
    with np.load(path, allow_pickle=False) as z:
        labels = np.r_[z['train_labels'], z['test_labels']]
        values = np.r_[z['train'], z['test']][..., 0]
    knots = np.unique(labels)
    grouped = np.stack([values[labels == key] for key in knots])
    if grouped.shape != (64, 12, 3):
        raise ValueError('保存编码网格不符合64K×12系综×3尺度')
    return knots, grouped, path


def plot_summary(payload, output):
    physical = payload['physical_mcrg']
    ik = physical['critical_index']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9))
    colors = ('#0072B2', '#D55E00', '#009E73')
    for color, pair in zip(colors, ((0, 1), (1, 2), (0, 2))):
        rows = [row for row in physical['rows'] if row['K_index'] == ik
                and (row['first_scale'], row['last_scale']) == pair and row['rcond'] == 1e-6]
        label = f'{SIZES[pair[0]]}→{SIZES[pair[1]]}'
        axes[0].plot([row['even_basis_size'] for row in rows],
            [row['unique_even_candidate']['conditional_inverse_y'] for row in rows],
            '-o', c=color, label=label)
        row = next(row for row in rows if row['even_basis_size'] == 6)
        low, _, high = row['conditional_ensemble_bootstrap']['conditional_inverse_y_percentiles']
        axes[0].vlines(6, low, high, colors=color, alpha=.7)
    for color, step in zip(colors[:2], (0, 1)):
        rows = [row for row in payload['curve_matching']['records']
                if row['transition'] == step and row['anchored']]
        ys = np.array([row['nominal_nu'] for row in rows])
        bounds = np.array([row['nominal_nu_conditional_percentiles'] for row in rows])
        axes[1].errorbar([row['half_width'] for row in rows], ys,
            yerr=np.maximum(np.array([ys-bounds[:, 0], bounds[:, 2]-ys]), 0),
            fmt='-o', c=color, capsize=3, label=f'{SIZES[step]}→{SIZES[step+1]}')
    for ax in axes:
        ax.axhline(1, c='#555555', ls='--', label='理论参照 ν=1')
        ax.axhline(payload['neural_R']['conditional_inverse_y'], c='#9C27B0', ls=':',
                   label='神经 R 根的原有换算值')
        ax.grid(alpha=.18)
        ax.set_ylabel('有条件的指数换算值 ν')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9)
    axes[0].set(xlabel='保留的偶算符数', title='物理耦合响应：算符截断检查', xticks=BASES)
    axes[1].set(xlabel='以已知 Kc 为中心的窗口半宽', title='编码曲线匹配：有限窗口有效缩放')
    fig.suptitle('保存数据的数值估计：物理耦合谱不等于神经 R 谱', fontsize=16)
    fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.25, wspace=.26)
    fig.text(.07, .14, '完全无监督训练结果；已知 Kc 仅用于事后选点评价。曲线匹配允许固定的尺度基线差。', fontsize=10)
    fig.text(.07, .09, '误差线仅为12个物理系综的条件自助重采样；不覆盖有限尺寸、训练种子、算符截断或拟合窗口偏差。', fontsize=10)
    fig.text(.07, .04, '仅重算保存数组；没有训练、模型推理、新物理采样或互信息估计。不能据此宣称 R 已学到 ν=1。', fontsize=10)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format='png', dpi=180, facecolor='white')
        with output.open('xb') as stream:
            stream.write(buffer.getvalue())
    finally:
        plt.close(fig)


def main():
    from scripts.estimate_saved_curve_collapse import estimate_curve_collapse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('拒绝覆盖已有报告')
    if args.bootstraps < 100:
        raise ValueError('至少100次条件自助重采样')
    physical = physical_estimates(args.root, bootstraps=args.bootstraps)
    knots, groups, curve_path = load_curves(args.root)
    curve = estimate_curve_collapse(knots, groups, critical_K=physical['critical_K'],
                                   bootstraps=args.bootstraps, seed=20261016)
    roots_path = args.root / 'dim1/report/critical_exponents.json'
    report = json.loads(roots_path.read_text())
    roots = report['roots']
    root = next(root for root in roots if root['root_id'] == 'F2')
    multiplier = root['jacobian'][0][0]
    payload = dict(status='complete', model_loading=False, model_inference=False,
        training=False, physical_sampling=False, MI_estimation=False, remote_access=False,
        checkpoint_access=False, bootstrap_seed=20261016,
        critical_K_used_as_evaluation_reference=True, physical_mcrg=physical,
        curve_matching=curve, curve_source_sha256=sha256(curve_path),
        neural_R=dict(source=str(roots_path.resolve()), source_sha256=sha256(roots_path),
            all_original_roots=roots, multiplier=multiplier, conditional_inverse_y=inverse_y(multiplier, 2),
            thermal_nu_certified=False, source_training_labels_used=False),
        primary_method_reference='https://arxiv.org/html/1707.08683',
        limitations=['单训练种子、有限108/54/27、两次粗粒化、B0。',
            '物理算符基为有限截断；C有方向各向异性，完整方向基与混合磁化基谱均保留。',
            '三种表示维数的物理算符统计相同，不能当作三次独立物理验证。',
            '曲线横向匹配不是固定点导数，使用已知Kc及保存编码插值；不能认证热指数。',
            '不存在根据这些数据单独确定的无偏热力学极限指数或总误差界。'])
    args.output.mkdir(parents=True, exist_ok=False)
    with chinese_figures():
        plot_summary(payload, args.output / 'numerical_exponents.png')
    with (args.output / 'numerical_exponents.json').open('x') as stream:
        json.dump(clean(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', output=str(args.output.resolve()),
        physical=[dict(sizes=row['sizes'], candidate=row['unique_even_candidate'],
            bootstrap=row['conditional_ensemble_bootstrap']) for row in physical['rows']
            if 'conditional_ensemble_bootstrap' in row]), ensure_ascii=False, default=clean))


if __name__ == '__main__':
    main()
