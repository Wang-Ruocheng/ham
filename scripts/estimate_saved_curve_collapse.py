"""从保存编码估计有限窗口曲线横向缩放；不加载或推理任何模型。

这里的 ``ln(2)/ln(a)`` 只是曲线缩放假设下的算术值，并非已认证热指数。
"""
from __future__ import annotations

import numpy as np


SEARCH_INTERVAL = (0.5, 3.0)


def _global_piecewise_fit(source_x, source_y, target_x, target_y, offsets):
    """在每个折线插值断点区间精确最小化一元二次最小二乘。

    横向坐标已相对临界参考点平移；首维为条件 bootstrap 样本。
    """
    source_y = np.atleast_2d(source_y)
    target_y = np.atleast_2d(target_y)
    offsets = np.asarray(offsets).reshape(-1, 1)
    nonzero = np.abs(target_x) > 1e-14
    ratios = (source_x[:, None] / target_x[nonzero][None, :]).ravel()
    lower, upper = SEARCH_INTERVAL
    boundaries = np.unique(np.r_[lower, ratios[(ratios > lower) & (ratios < upper)], upper])
    best_cost = np.full(len(source_y), np.inf)
    best_a = np.full(len(source_y), np.nan)
    best_curvature = np.zeros(len(source_y))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        query = 0.5 * (left + right) * target_x
        if query.min() < source_x[0] or query.max() > source_x[-1]:
            continue
        index = np.searchsorted(source_x, query, side='right') - 1
        index = np.clip(index, 0, len(source_x) - 2)
        slope = ((source_y[:, index + 1] - source_y[:, index])
                 / (source_x[index + 1] - source_x[index]))
        coefficient = slope * target_x
        constant = source_y[:, index] - slope * source_x[index] + offsets - target_y
        curvature = np.sum(coefficient ** 2, axis=1)
        numerator = -np.sum(coefficient * constant, axis=1)
        multiplier = np.divide(numerator, curvature, out=np.full_like(numerator, left),
                               where=curvature > 1e-30)
        multiplier = np.clip(multiplier, left, right)
        cost = np.mean((coefficient * multiplier[:, None] + constant) ** 2, axis=1)
        better = cost < best_cost
        best_a[better] = multiplier[better]
        best_cost[better] = cost[better]
        best_curvature[better] = curvature[better]
    # A constant local source curve does not identify horizontal scaling.
    best_a[best_curvature <= 1e-30] = np.nan
    return best_a, np.sqrt(best_cost)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _quantiles(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return [float(v) for v in np.percentile(valid, [2.5, 50, 97.5])] if len(valid) else None


def _nominal_nu(multiplier):
    multiplier = np.asarray(multiplier)
    result = np.full(multiplier.shape, np.nan)
    valid = np.isfinite(multiplier) & (multiplier > 1)
    result[valid] = np.log(2) / np.log(multiplier[valid])
    return result


def estimate_curve_collapse(knots, grouped_encodings, *, critical_K,
                            widths=(.02, .03, .04, .06), bootstraps=1000,
                            seed=20261016):
    """估计相邻保存尺度的有限窗口横向缩放，返回纯 JSON 兼容结果。

    输入形状为 ``(K点数, 独立物理系综数, 保存尺度数)``。每个 K 内成组重采样
    系综，同一组索引用于所有尺度；不同 K 独立。既不重采样物理构型，也不训练。
    """
    knots = np.asarray(knots, dtype=float)
    groups = np.asarray(grouped_encodings, dtype=float)
    widths = tuple(float(w) for w in widths)
    if (knots.ndim != 1 or groups.ndim != 3 or groups.shape[0] != len(knots)
            or len(knots) < 3 or groups.shape[1] < 2 or groups.shape[2] < 2):
        raise ValueError('需要递增 K 网格及形状为 K×独立系综×尺度的标量编码')
    if not np.isfinite(groups).all() or not np.isfinite(knots).all() or not np.isfinite(critical_K):
        raise ValueError('输入必须有限')
    if not np.all(np.diff(knots) > 0) or not widths or any(w <= 0 for w in widths):
        raise ValueError('K 网格必须严格递增，窗口半宽必须为正')
    if isinstance(bootstraps, bool) or int(bootstraps) != bootstraps or bootstraps < 1:
        raise ValueError('bootstrap 次数必须为正整数')
    critical_index = int(np.argmin(abs(knots - critical_K)))
    if abs(knots[critical_index] - critical_K) > 1e-12:
        raise ValueError('固定偏移模式需要保存的精确临界参考 K 点')
    means = groups.mean(axis=1)
    rng = np.random.default_rng(seed)
    selection = rng.integers(0, groups.shape[1], (int(bootstraps), len(knots), groups.shape[1]))
    samples = groups[np.arange(len(knots))[None, :, None], selection].mean(axis=2)
    centered = knots - critical_K
    records = []
    for width in widths:
        mask = abs(centered) <= width
        indices = np.flatnonzero(mask)
        if mask.sum() < 3:
            raise ValueError('每个窗口至少需要三个保存 K 点')
        for step in range(groups.shape[2] - 1):
            for anchored in (False, True):
                offset = means[critical_index, step + 1] - means[critical_index, step] if anchored else 0.
                boot_offset = (samples[:, critical_index, step + 1] - samples[:, critical_index, step]
                               if anchored else np.zeros(int(bootstraps)))
                a, residual = _global_piecewise_fit(centered, means[:, step], centered[mask],
                                                     means[mask, step + 1], offset)
                boot_a, _ = _global_piecewise_fit(centered, samples[:, :, step], centered[mask],
                                                   samples[:, mask, step + 1], boot_offset)
                nu = _nominal_nu(a)
                boot_nu = _nominal_nu(boot_a)
                boundary_hit = (np.isclose(boot_a, SEARCH_INTERVAL[0], rtol=0, atol=1e-10)
                                | np.isclose(boot_a, SEARCH_INTERVAL[1], rtol=0, atol=1e-10))
                records.append(dict(
                    half_width=width, transition=step, K_count=int(mask.sum()),
                    K_indices=indices.tolist(), K_knots=knots[mask].tolist(), anchored=anchored,
                    offset_mode='固定为各尺度临界参考 K 编码之差' if anchored else '不允许纵向平移',
                    offset=float(offset), best_horizontal_multiplier=_number(a[0]),
                    best_rmse=_number(residual[0]), nominal_nu=_number(nu[0]),
                    multiplier_conditional_percentiles=_quantiles(boot_a),
                    nominal_nu_conditional_percentiles=_quantiles(boot_nu),
                    bootstrap_identified_count=int(np.isfinite(boot_a).sum()),
                    bootstrap_multiplier_above_one_count=int(np.isfinite(boot_nu).sum()),
                    bootstrap_boundary_hit_count=int(boundary_hit.sum()),
                    thermal_nu_certified=False,
                ))
    return dict(
        method='保存曲线折线插值的全局分段二次最小二乘',
        equation='g_{k+1}(K) ≈ g_k(K参考+a(K−K参考))+c_k',
        critical_reference_K=float(critical_K), critical_index=critical_index,
        search_interval=list(SEARCH_INTERVAL), extrapolation=False,
        bootstraps=int(bootstraps), seed=int(seed),
        independent_ensembles_per_K=int(groups.shape[1]),
        quantile_levels=[2.5, 50., 97.5],
        bootstrap='各 K 内按独立物理系综成组重采样，所有尺度配对；不同 K 独立',
        records=records,
        limitations=[
            '已使用已知临界 K 作曲线比较中心，不是独立估计临界 K。',
            '横向缩放倍率描述有限窗口曲线形状，不是 R 不动点的局部导数。',
            'ln2/ln(a) 只对 a>1 作条件算术换算，不认证物理热方向或热指数。',
            '条件 bootstrap 不包含训练 seed、有限尺寸、算符截断、离散网格、拟合窗口和模型偏差。',
            '不同窗口不独立；窗口敏感性不能当作统计置信区间。',
            '曲线使用折线插值，无新增物理采样、模型加载、Q/C/R 推理或训练。',
        ],
    )
