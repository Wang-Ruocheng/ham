"""冻结 Q 的两项候选目标诊断；仅消费已保存的公开统计。

本脚本不导入 torch、不加载模型、不打开 PT、不使用耦合/临界指数标签。
辅助线性读出和探针只是统计诊断，绝不更新 Q/C/R。
"""
from __future__ import annotations

import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np


VERSION = "q_loss_candidate_validation_v1"
RIDGES = (1e-6, 1e-3, .1, 1., 10.)
BANDWIDTHS = (.25, .5, 1., 2.)
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 2026101702
SCALES = (108, 54, 27)
MODEL_LABELS = ("无监督", "K监督")
OPERATOR_LABELS = ("最近邻", "对角", "小方格", "距离2", "长方形", "距离3")
TARGET_LABELS = ("距离4关联", "距离8关联", "最低轴向结构因子", "第二轴向结构因子", "最低对角结构因子")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_mean(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    numerator = np.where(mask, values, 0.).sum(axis=axis)
    denominator = mask.sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=np.float64), where=denominator > 0)


def percentile_interval(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, (2.5, 97.5), axis=0)


def stratified_bootstrap_weights(groups, rng, repeats=BOOTSTRAPS):
    """每个固定 group 内重采物理链；不重采 C 或 half。"""
    groups = np.asarray(groups)
    n = len(groups)
    result = np.zeros((repeats, n), dtype=np.float64)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        picks = rng.integers(0, len(members), size=(repeats, len(members)))
        counts = np.zeros((repeats, len(members)), dtype=np.float64)
        np.add.at(counts, (np.arange(repeats)[:, None], picks), 1.)
        result[:, members] = counts / n
    return result


def weighted_bootstrap_means(values, weights):
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    valid = np.isfinite(flat)
    numerator = weighted_product(weights, np.where(valid, flat, 0.))
    denominator = weighted_product(weights, valid.astype(np.float64))
    output = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return output.reshape((len(weights),) + values.shape[1:])


def weighted_product(weights, values):
    # 不走平台 BLAS，避免某些 Accelerate/numpy 组合残留浮点标志造成伪 overflow 警告。
    return np.einsum("bi,ij->bj", weights, values, optimize=False)


def mean_summary(values, weights):
    values = np.asarray(values, dtype=np.float64)
    samples = weighted_bootstrap_means(values, weights)
    interval = percentile_interval(samples)
    return dict(mean=finite_mean(values, axis=0), ci95=interval, valid_physical_replicates=np.isfinite(values).sum(axis=0))


def paired_summary(values, weights):
    """values: model, physical replicate, ...；差始终为监督减无监督。"""
    return dict(
        models={name: mean_summary(values[m], weights) for m, name in enumerate(MODEL_LABELS)},
        supervised_minus_unsupervised=mean_summary(values[1] - values[0], weights),
    )


def rank_one_residual(target):
    singular = np.linalg.svd(target, compute_uv=False)
    squared_norm = np.square(target).sum(axis=(-2, -1))
    residual_sq = np.maximum(squared_norm - np.square(singular[..., 0]), 0.)
    return residual_sq, singular


def fit_shared_response_readout(q_response, target, train):
    """仅 TRAIN；共同一个 16x1 W，跨全部尺度/重复/六方向。"""
    if q_response.shape[-2] != 1:
        raise ValueError("本预注册比较限历史一维 g，不能悄悄改变模型容量")
    ws = []
    prediction = []
    for model in range(len(q_response)):
        x = q_response[model, train, ..., 0, :].reshape(-1, target.shape[-1])
        y = target[train].reshape(-1, target.shape[-2], target.shape[-1])
        denominator = np.square(x).sum()
        if denominator <= 1e-30:
            w = np.zeros(target.shape[-2], dtype=np.float64)
        else:
            w = np.einsum("pmo,po->m", y, x) / denominator
        ws.append(w[:, None])
        prediction.append(w[None, None, None, :, None] * q_response[model, ..., 0, :][..., None, :])
    return np.stack(ws), np.stack(prediction)


def response_metrics(prediction, target, lower_sq, threshold, direction_threshold):
    norm = np.linalg.norm(target, axis=(-2, -1))
    difference_sq = np.square(prediction - target[None]).sum(axis=(-2, -1))
    relative = np.divide(np.sqrt(difference_sq), norm[None], out=np.full_like(difference_sq, np.nan), where=norm[None] > threshold)
    lower_relative = np.divide(np.sqrt(lower_sq), norm, out=np.full_like(lower_sq, np.nan), where=norm > threshold)
    # 在同一公共度量中 native 误差不可能低于最佳 rank1；数值舍入可出现极小负值。
    excess_relative = relative - lower_relative[None]
    excess_mse = (difference_sq - lower_sq[None]) / (target.shape[-2] * target.shape[-1])
    if np.nanmin(excess_mse) < -1e-11:
        raise ValueError("一维 decoder 的残差显著低于最佳 rank1 下限；检查数组轴/容量")
    directional_norm = np.linalg.norm(target, axis=-2)
    directional_sq = np.square(prediction - target[None]).sum(axis=-2)
    directional_relative = np.divide(np.sqrt(directional_sq), directional_norm[None], out=np.full_like(directional_sq, np.nan), where=directional_norm[None] > direction_threshold)
    return dict(
        relative_frobenius=finite_mean(relative, axis=2),
        raw_mse=finite_mean(difference_sq / (target.shape[-2] * target.shape[-1]), axis=2),
        relative_excess_over_rank1=finite_mean(excess_relative, axis=2),
        mse_excess_over_rank1=finite_mean(excess_mse, axis=2),
        direction_relative=finite_mean(directional_relative, axis=2),
        direction_raw_mse=finite_mean(directional_sq / target.shape[-2], axis=2),
    )


def fit_train_reliable_projection(target, train, groups, rng, repeats=BOOTSTRAPS):
    """补充设计：仅 TRAIN 公共目标决定固定右投影，不读取 TEST 数值。"""
    target_per_rep = np.asarray(target)[train].mean(axis=1)
    train_groups = np.asarray(groups)[train]
    unique = np.unique(train_groups)
    members = [np.flatnonzero(train_groups == group) for group in unique]
    scale_count, motif_count, operator_count = target_per_rep.shape[1:]
    stack = np.stack([target_per_rep[ids].mean(axis=0) for ids in members], axis=1)
    stack = stack.reshape(scale_count, len(unique) * motif_count, operator_count) / np.sqrt(len(unique))
    _, spectrum, right = np.linalg.svd(stack, full_matrices=False)
    noise = np.empty((repeats, scale_count), dtype=np.float64)
    for begin in range(0, repeats, 50):
        batch = min(50, repeats - begin)
        samples = []
        for ids in members:
            picks = rng.integers(0, len(ids), size=(batch, len(ids)))
            samples.append(target_per_rep[ids][picks].mean(axis=1))
        sampled_stack = np.stack(samples, axis=2).reshape(batch, scale_count, len(unique) * motif_count, operator_count) / np.sqrt(len(unique))
        noise[begin:begin + batch] = np.linalg.svd(sampled_stack - stack[None], compute_uv=False)[..., 0]
    noise95 = np.quantile(noise, .95, axis=0)
    ratio = np.divide(spectrum, spectrum[:, :1], out=np.zeros_like(spectrum), where=spectrum[:, :1] > 1e-30)
    reliable = (spectrum > 2 * noise95[:, None]) & (ratio >= .01)
    projection = np.einsum("sik,si,sij->skj", right, reliable.astype(np.float64), right)
    return dict(
        definition="每尺度 TRAIN 先平均 C、每 group 平均 12rep，再竖向堆叠 A_g/sqrt(G)；只从该矩阵的共同右基选择方向",
        threshold_definition="sigma > 2 * group内物理rep bootstrap噪声谱范数95%分位，且 sigma/sigma1 >= .01",
        split_used_for_selection="TRAIN only", fixed_groups=len(unique), bootstrap_repeats=repeats,
        singular_values=spectrum, right_basis_rows=right,
        bootstrap_noise_spectral_norm_p95=noise95, absolute_threshold=2 * noise95,
        relative_threshold=.01, reliable_singular_values_mask=reliable,
        reliable_rank=reliable.sum(axis=-1), projection=projection,
        projection_idempotence_max_abs=float(np.abs(np.einsum("sij,sjk->sik", projection, projection) - projection).max()),
    )


def analyze_train_projection(target, predictions, train, test, groups, weights, rng):
    fitted = fit_train_reliable_projection(target, train, groups, rng)
    projection = fitted["projection"]
    verifiable = fitted["reliable_rank"] > 0
    projected_target = np.einsum("ndsio,sop->ndsip", target, projection)
    norm = np.linalg.norm(projected_target, axis=(-2, -1))
    threshold = max(1e-14, 1e-10 * float(np.median(np.linalg.norm(target[train], axis=(-2, -1)))))
    directional_norm = np.linalg.norm(projected_target, axis=-2)
    direction_threshold = np.maximum(1e-14, 1e-10 * np.median(directional_norm[train], axis=(0, 1, 2)))
    floor, singular = rank_one_residual(projected_target)
    lower_relative = np.divide(np.sqrt(floor), norm, out=np.full_like(floor, np.nan), where=norm > threshold)
    metric_sets = {}
    saved = dict(response_train_reliable_projection=projection,
                 response_train_reliable_projected_target=projected_target,
                 response_train_reliable_projected_target_singular_values=singular)
    for name, prediction in predictions.items():
        projected = np.einsum("mndsio,sop->mndsip", prediction, projection)
        metrics = response_metrics(projected, projected_target, floor, threshold, direction_threshold)
        for metric_name, values in metrics.items():
            # P=0 的尺度没有可验证方向；连 raw MSE=0 也不解释为完美。
            values[:, :, ~verifiable, ...] = np.nan
            saved[f"response_train_reliable_{name}_{metric_name}"] = values
        metric_sets[name] = metrics
    floor_for_reporting = floor.copy()
    floor_for_reporting[:, :, ~verifiable] = np.nan
    saved["response_train_reliable_rank1_residual_sq"] = floor_for_reporting
    split_records = {}
    for split_name, mask in (("TRAIN", train), ("TEST", test)):
        w = weights[split_name]
        record = dict(
            verifiable_scale_mask=verifiable,
            no_verifiable_directions_by_scale=~verifiable,
            near_zero_panels_by_scale=(norm[mask] <= threshold).sum(axis=(0, 1)),
            rank1_floor=dict(relative_frobenius=mean_summary(finite_mean(lower_relative[mask], axis=1), w),
                             raw_mse=mean_summary(finite_mean(floor_for_reporting[mask] / np.prod(target.shape[-2:]), axis=1), w)),
            methods={},
        )
        for name, metrics in metric_sets.items():
            record["methods"][name] = {}
            for metric_name, values in metrics.items():
                subset = values[:, mask]
                record["methods"][name][metric_name] = dict(by_scale=paired_summary(subset, w),
                    all_scales_equal_weight=paired_summary(finite_mean(subset, axis=2), w))
        split_records[split_name] = record
    return dict(
        preregistered_supplement="由主代理消息预先明确的 TRAIN-only 可靠方向补充；保留全六方向原指标，不按 TEST 或结果改阈值",
        fitted_train_metric=fitted, splits=split_records,
        metric="target 与 prediction 同时右乘每尺度固定 TRAIN 投影 P；TEST 不重新选方向",
        auxiliary="辅助读出沿用全六方向 TRAIN 拟合的共享 W，再投影；不额外重拟合或查看 TEST",
        null_direction_policy="P=0 的尺度全部误差为 null/无可验证方向，不把投影后全零称为完美重构",
    ), saved


def analyze_response(arrays, train, test, groups, weights, rng):
    target = arrays["response_target"]
    prediction = arrays["response_prediction"]
    q_response = arrays["q_response"]
    norm = np.linalg.norm(target, axis=(-2, -1))
    # 仅 TRAIN 定参考数量级；近零 panel 不以 0/0 当成完美重构。
    threshold = max(1e-14, 1e-10 * float(np.median(norm[train])))
    direction_norm = np.linalg.norm(target, axis=-2)
    direction_threshold = np.maximum(1e-14, 1e-10 * np.median(direction_norm[train], axis=(0, 1, 2)))
    lower_sq, singular = rank_one_residual(target)
    lower_relative = np.divide(np.sqrt(lower_sq), norm, out=np.full_like(lower_sq, np.nan), where=norm > threshold)
    readout, auxiliary = fit_shared_response_readout(q_response, target, train)
    methods = {"native_decoder": response_metrics(prediction, target, lower_sq, threshold, direction_threshold),
               "train_shared_linear_readout": response_metrics(auxiliary, target, lower_sq, threshold, direction_threshold)}
    panels = {}
    output_arrays = dict(response_rank1_residual_sq=lower_sq, response_target_singular_values=singular,
                         response_shared_linear_W=readout, response_auxiliary_prediction=auxiliary,
                         response_near_zero_panel=norm <= threshold,
                         response_direction_near_zero=direction_norm <= direction_threshold)
    reliable_projection, projection_arrays = analyze_train_projection(target,
        {"native_decoder": prediction, "train_shared_linear_readout": auxiliary}, train, test, groups, weights, rng)
    output_arrays.update(projection_arrays)
    half_difference = arrays["response_target_halves"][..., 0, :, :] - arrays["response_target_halves"][..., 1, :, :]
    half_relative = np.divide(np.linalg.norm(half_difference, axis=(-2, -1)), norm, out=np.full_like(norm, np.nan), where=norm > threshold)
    half_mse = np.square(half_difference).mean(axis=(-2, -1))
    for split_name, mask in (("TRAIN", train), ("TEST", test)):
        w = weights[split_name]
        record = dict(
            independent_physical_replicates=int(mask.sum()), fixed_groups=int(len(np.unique(groups[mask]))),
            near_zero_panels_by_scale=(norm[mask] <= threshold).sum(axis=(0, 1)),
            near_zero_directions_by_scale=(direction_norm[mask] <= direction_threshold).sum(axis=(0, 1)),
            rank1_floor=dict(relative_frobenius=mean_summary(finite_mean(lower_relative[mask], axis=1), w),
                             raw_mse=mean_summary((lower_sq[mask] / np.prod(target.shape[-2:])).mean(axis=1), w)),
            half_target_difference=dict(relative_frobenius=mean_summary(finite_mean(half_relative[mask], axis=1), w),
                                        raw_mse=mean_summary(half_mse[mask].mean(axis=1), w)),
            methods={},
        )
        for method_name, metric in methods.items():
            record["methods"][method_name] = {}
            for metric_name, values in metric.items():
                sub = values[:, mask]
                record["methods"][method_name][metric_name] = dict(
                    by_scale=paired_summary(sub, w),
                    all_scales_equal_weight=paired_summary(finite_mean(sub, axis=2), w),
                )
                output_arrays[f"response_{method_name}_{metric_name}"] = values
        panels[split_name] = record
    # 对各固定 group 的公共 target 做物理链 bootstrap 噪声参考。
    # 这是已保存协方差估计的组均值可辨识谱，不是 Q 维数，也不是热模认证。
    target_per_rep = target.mean(axis=1)
    support = []
    for split_name, mask in (("TRAIN", train), ("TEST", test)):
        for group in np.unique(groups[mask]):
            ids = np.flatnonzero(mask & (groups == group))
            picks = rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))
            group_mean = target_per_rep[ids].mean(axis=0)
            boot_means = target_per_rep[ids][picks].mean(axis=1)
            noise_spectrum = np.linalg.svd(boot_means - group_mean[None], compute_uv=False)[..., 0]
            noise95 = np.quantile(noise_spectrum, .95, axis=0)
            spectrum = np.linalg.svd(group_mean, compute_uv=False)
            ratio = np.divide(spectrum, spectrum[..., :1], out=np.zeros_like(spectrum), where=spectrum[..., :1] > threshold)
            reliable = (spectrum > 2 * noise95[:, None]) & (ratio >= .01)
            covariance_mean = arrays["score_cov"][ids].mean(axis=(0, 1))
            support.append(dict(split=split_name, group_index=int(group), physical_replicates=len(ids),
                                singular_values=spectrum, bootstrap_noise_spectral_norm_p95=noise95,
                                reliable_singular_values_mask=reliable, reliable_rank=reliable.sum(axis=-1),
                                score_covariance_spectrum=np.linalg.eigvalsh(covariance_mean)[:, ::-1],
                                score_std_mean=arrays["score_std"][ids].mean(axis=(0, 1))))
    return dict(
        definition="每 panel 的 16x6 目标/预测先算相对 Frobenius 范数，再平均 C；物理链为 bootstrap 单位",
        metric="当前尺度各算符 score 用本物理系综标准差归一；相同目标/度量，不是 MSE",
        raw_mse_definition="相同 score 标准化单位下，对全部 16x6 项平方误差取均值",
        rank1_floor_definition="公共 16x6 target 每 panel 最佳 rank1 SVD 的不可约残差；只给容量下限，不要求 native 的六方向残差全零",
        auxiliary_definition="仅 TRAIN 全部尺度共同最小二乘 W[16,1]，无截距；不是重新训练 Q，不替代原生 decoder",
        auxiliary_W=readout, near_zero_absolute_threshold=threshold, near_zero_direction_threshold=direction_threshold,
        splits=panels, train_only_reliable_projection=reliable_projection, public_target_group_spectra=support,
        half_noise_caution="两个 half 的差是有限子样本差异参考，不直接当成完整 target 的无偏噪声方差或校正下限",
    ), output_arrays


def make_basis(x, normalizer, kind, centers=None, bandwidth=None):
    z = (np.asarray(x) - normalizer[0]) / normalizer[1]
    if kind == "linear":
        return np.column_stack((np.ones(len(z)), z))
    return np.column_stack((np.ones(len(z)), np.exp(-.5 * np.square((z[:, None] - centers[None, :]) / bandwidth))))


def fit_probe(x, y, kind, ridge, bandwidth=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)
    xnormalizer = (float(x.mean()), max(float(x.std(ddof=0)), 1e-12))
    ymean, ystd = y.mean(axis=0), np.maximum(y.std(axis=0, ddof=0), 1e-12)
    centers = np.quantile((x - xnormalizer[0]) / xnormalizer[1], np.linspace(0, 1, 16)) if kind == "rbf" else None
    basis = make_basis(x, xnormalizer, kind, centers, bandwidth)
    penalty = np.eye(basis.shape[1]) * ridge
    penalty[0, 0] = 0.
    coefficient = np.linalg.solve(weighted_product(basis.T, basis) / len(x) + penalty,
                                  weighted_product(basis.T, (y - ymean) / ystd) / len(x))
    return dict(kind=kind, ridge=ridge, bandwidth=bandwidth, x_normalizer=xnormalizer,
                target_mean=ymean, target_std=ystd, centers=centers, coefficient=coefficient)


def predict_probe(model, x):
    basis = make_basis(x, model["x_normalizer"], model["kind"], model["centers"], model["bandwidth"])
    return weighted_product(basis, model["coefficient"]) * model["target_std"] + model["target_mean"]


def choose_probe(x, y, groups, kind):
    candidates = [(ridge, None) for ridge in RIDGES] if kind == "linear" else [(ridge, width) for width in BANDWIDTHS for ridge in RIDGES]
    records = []
    for ridge, bandwidth in candidates:
        squared_sum, count = 0., 0
        fold_losses = []
        for fold in range(5):
            validation = groups % 5 == fold
            fitting = ~validation
            if not validation.any() or not fitting.any():
                raise ValueError("整组五折缺少拟合或验证样本")
            model = fit_probe(x[fitting], y[fitting], kind, ridge, bandwidth)
            normalized_sq = np.square((predict_probe(model, x[validation]) - y[validation]) / model["target_std"])
            squared_sum += float(normalized_sq.sum())
            count += normalized_sq.size
            fold_losses.append(float(normalized_sq.mean()))
        records.append(dict(ridge=ridge, bandwidth=bandwidth, validation_nrmse_sq=squared_sum / count, fold_validation_nrmse_sq=fold_losses))
    # 固定候选顺序使精确并列时结果确定；从不查看 TEST。
    best = min(range(len(records)), key=lambda i: records[i]["validation_nrmse_sq"])
    chosen = records[best]
    return fit_probe(x, y, kind, chosen["ridge"], chosen["bandwidth"]), dict(selected_index=best, candidates=records)


def probe_metric_samples(error_sq, target, target_std, weights):
    """每物理链已先平均两个交叉 half；weights 不把 half 当独立样本。"""
    mse = weighted_product(weights, error_sq)
    ymean = weighted_product(weights, target.mean(axis=1))
    ysecond = weighted_product(weights, np.square(target).mean(axis=1))
    variance = np.maximum(ysecond - np.square(ymean), 0.)
    rmse = np.sqrt(np.maximum(mse, 0.))
    nrmse = rmse / target_std
    r2 = 1 - np.divide(mse, variance, out=np.full_like(mse, np.nan), where=variance > 1e-24)
    return dict(rmse=rmse, nrmse=nrmse, r2=r2,
                all_targets_raw_rmse=np.sqrt(np.maximum(mse.mean(axis=-1), 0.)),
                all_targets_nrmse=np.sqrt(np.maximum((mse / np.square(target_std)).mean(axis=-1), 0.)))


def probe_summary(prediction, truth, train, test, weights):
    # model, physical replicate, scale, cross-half, target
    reference_std = np.maximum(truth[train].std(axis=(0, 2), ddof=0), 1e-12)
    train_mean = truth[train].mean(axis=(0, 2))
    baseline_prediction = np.broadcast_to(train_mean[None, :, None, :], truth.shape)
    error_sq = np.square(prediction - truth[None]).mean(axis=3)
    baseline_error_sq = np.square(baseline_prediction - truth).mean(axis=2)
    summary = {}
    for split_name, mask in (("TRAIN", train), ("TEST", test)):
        split_models = {}
        boot_models = []
        point_models = []
        for model in range(2):
            # 明确把 half 移到 scale 前；直接 reshape 会使合法形状掩盖轴语义错误。
            target_flat = truth[mask].transpose(0, 2, 1, 3).reshape(mask.sum(), 2, -1)
            boot = probe_metric_samples(error_sq[model, mask].reshape(mask.sum(), -1), target_flat, reference_std.reshape(-1), weights[split_name])
            point = probe_metric_samples(error_sq[model, mask].reshape(mask.sum(), -1), target_flat, reference_std.reshape(-1), np.full((1, mask.sum()), 1 / mask.sum()))
            # 每尺度/逐 target，而非混合三尺度后丢掉细节。
            records = {}
            for metric in ("rmse", "nrmse", "r2"):
                values = boot[metric].reshape(len(weights[split_name]), len(SCALES), len(TARGET_LABELS))
                records[metric] = dict(value=point[metric][0].reshape(len(SCALES), len(TARGET_LABELS)), ci95=percentile_interval(values))
            for metric in ("all_targets_raw_rmse", "all_targets_nrmse"):
                per_scale = np.sqrt(np.maximum(weighted_product(weights[split_name], error_sq[model, mask].reshape(mask.sum(), -1)).reshape(len(weights[split_name]), len(SCALES), -1).mean(axis=-1), 0.))
                point_scale = np.sqrt(error_sq[model, mask].mean(axis=(0, 2)))
                if metric == "all_targets_nrmse":
                    per_scale = np.sqrt(np.maximum(weighted_product(weights[split_name], (error_sq[model, mask] / np.square(reference_std)).reshape(mask.sum(), -1)).reshape(len(weights[split_name]), len(SCALES), -1).mean(axis=-1), 0.))
                    point_scale = np.sqrt((error_sq[model, mask] / np.square(reference_std)).mean(axis=(0, 2)))
                records[metric] = dict(value=point_scale, ci95=percentile_interval(per_scale))
                boot[metric + "_by_scale"] = per_scale
                point[metric + "_by_scale"] = point_scale[None]
            records["all_scales_all_targets_nrmse"] = dict(value=point["all_targets_nrmse"][0], ci95=percentile_interval(boot["all_targets_nrmse"]))
            split_models[MODEL_LABELS[model]] = records
            boot_models.append(boot)
            point_models.append(point)
        differences = {}
        for metric in ("rmse", "nrmse", "r2"):
            differences[metric] = dict(value=(point_models[1][metric] - point_models[0][metric])[0].reshape(len(SCALES), -1),
                                       ci95=percentile_interval((boot_models[1][metric] - boot_models[0][metric]).reshape(len(weights[split_name]), len(SCALES), -1)))
        for metric in ("all_targets_raw_rmse", "all_targets_nrmse"):
            differences[metric] = dict(value=(point_models[1][metric + "_by_scale"] - point_models[0][metric + "_by_scale"])[0],
                                       ci95=percentile_interval(boot_models[1][metric + "_by_scale"] - boot_models[0][metric + "_by_scale"]))
        differences["all_scales_all_targets_nrmse"] = dict(value=(point_models[1]["all_targets_nrmse"] - point_models[0]["all_targets_nrmse"])[0],
                                                          ci95=percentile_interval(boot_models[1]["all_targets_nrmse"] - boot_models[0]["all_targets_nrmse"]))
        baseline = {}
        target_flat = truth[mask].transpose(0, 2, 1, 3).reshape(mask.sum(), 2, -1)
        base_boot = probe_metric_samples(baseline_error_sq[mask].reshape(mask.sum(), -1), target_flat, reference_std.reshape(-1), weights[split_name])
        base_point = probe_metric_samples(baseline_error_sq[mask].reshape(mask.sum(), -1), target_flat, reference_std.reshape(-1), np.full((1, mask.sum()), 1 / mask.sum()))
        for metric in ("rmse", "nrmse", "r2"):
            baseline[metric] = dict(value=base_point[metric][0].reshape(len(SCALES), -1), ci95=percentile_interval(base_boot[metric].reshape(len(weights[split_name]), len(SCALES), -1)))
        summary[split_name] = dict(models=split_models, supervised_minus_unsupervised=differences, train_mean_baseline=baseline)
    return dict(splits=summary, common_train_target_std=reference_std, common_train_target_mean=train_mean), error_sq


def analyze_probes(arrays, train, test, groups, weights):
    # C 首先平均；同一条物理链的 A->B、B->A 保留成对相关。
    inputs = arrays["g"].mean(axis=2)
    truth = arrays["observables"].mean(axis=1)[:, :, ::-1, :].copy()
    cv_group = np.repeat(groups[train], 2)
    methods, saved = {}, dict(probe_cross_half_truth=truth, probe_inputs_mean_C=inputs)
    for kind in ("linear", "rbf"):
        prediction = np.empty((2,) + truth.shape, dtype=np.float64)
        fitted = []
        for model in range(2):
            model_fits = []
            for scale in range(len(SCALES)):
                x = inputs[model, :, scale, :].reshape(-1)
                y = truth[:, scale, :, :].reshape(-1, len(TARGET_LABELS))
                row_train = np.repeat(train, 2)
                fit, cv = choose_probe(x[row_train], y[row_train], cv_group, kind)
                prediction[model, :, scale] = predict_probe(fit, x).reshape(len(train), 2, -1)
                model_fits.append(dict(scale=SCALES[scale], fit=fit, cross_validation=cv))
            fitted.append(dict(model=MODEL_LABELS[model], scales=model_fits))
        record, error_sq = probe_summary(prediction, truth, train, test, weights)
        record["fitted_models"] = fitted
        methods[kind] = record
        saved[f"probe_{kind}_prediction"] = prediction
        saved[f"probe_{kind}_squared_error_mean_half"] = error_sq
    return dict(
        definition="先平均固定 C 重复；每物理链 A 的 g 预测 B 的观测量，同时 B->A；评估先平均两 half 的平方误差",
        auxiliary_only="只拟合辅助探针，不更新原 Q/C/R；K、Kc、nu 及理论倍率不是目标、输入或选参依据",
        cross_validation="仅 TRAIN，group_index%5 留整组；每折均值/标准差/16分位中心仅用拟合部分；按五目标标准化平方误差选参",
        capacity="线性 ridge 与 16 中心高斯 RBF ridge；两模型相同容量/候选网格，允许独立 TRAIN 选参",
        ridge_definition="平均平方误差 + lambda*斜率/基系数平方和；截距不惩罚，目标逐维 TRAIN 标准化",
        normalization="两模型共用每尺度每目标的 TRAIN 标准差；R2 的分母为该评分 split 的目标方差",
        raw_rmse_caution="五观测量定义/幅度不同，混合 raw RMSE 仅附录；主要合并指标用共同 TRAIN 标准差的 NRMSE",
        methods=methods,
    ), saved


def validate_inputs(arrays, protocol, receipt):
    n = len(arrays["split"])
    d = arrays["response_target"].shape[1]
    expected = dict(
        g=(2, n, d, 3, 2), observables=(n, d, 3, 2, 5), motif_means=(n, d, 3, 2, 16),
        response_target=(n, d, 3, 16, 6), response_target_halves=(n, d, 3, 2, 16, 6),
        response_prediction=(2, n, d, 3, 16, 6), q_response=(2, n, d, 3, 1, 6),
        score_cov=(n, d, 3, 6, 6), score_std=(n, d, 3, 6),
    )
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} 有 NaN/Inf")
    if list(arrays["model_names"]) not in (["unsupervised", "supervised"], ["U", "S"]):
        raise ValueError("模型顺序必须为无监督、监督")
    if d != protocol["draws"] or protocol["endpoint"] != 9000 or tuple(protocol["scales"]) != SCALES:
        raise ValueError("冻结协议的 C 重复/源步数/尺度不符")
    if receipt.get("labels_accessed") is not False or receipt.get("MI_opened") is not False:
        raise ValueError("采集回执缺少无标签/不打开 MI 确认")
    for model in receipt["models"]:
        if model["model_sha256_before"] != model["model_sha256_after"] or not model["all_parameter_grad_none"]:
            raise ValueError("冻结模型状态/梯度确认不符")
    for split in (0, 1):
        indices = np.flatnonzero(arrays["split"] == split)
        for group in np.unique(arrays["group_index"][indices]):
            selected = indices[arrays["group_index"][indices] == group]
            if len(selected) != 12 or sorted(arrays["replica_index"][selected]) != list(range(12)):
                raise ValueError("必须在每组 12 条物理链内 bootstrap，不能把 C/half 当物理链")


def evidence_verdict(difference):
    lower, upper = np.asarray(difference["ci95"])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "没有可验证的有效方向/有限指标，不能以全零投影宣称完美"
    if upper < 0:
        return "在该固定模型/数据/度量下，监督误差较低的配对区间不跨零"
    if lower > 0:
        return "在该固定模型/数据/度量下，监督误差反而较高的配对区间不跨零"
    return "配对区间跨零，不能确认监督误差较低"


def chinese_font(text):
    from matplotlib import font_manager, ft2font
    needed = {ord(c) for c in text if "\u4e00" <= c <= "\u9fff"}
    preferred = ("Arial Unicode MS", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "PingFang SC")
    entries = sorted(font_manager.fontManager.ttflist, key=lambda x: (next((i for i, name in enumerate(preferred) if name.lower() in x.name.lower()), len(preferred)), x.name))
    for entry in entries:
        try:
            if needed.issubset(ft2font.FT2Font(entry.fname).get_charmap()):
                return entry.fname
        except (OSError, RuntimeError):
            pass
    raise RuntimeError("没有覆盖所需中文字形的字体；保存统计，不以缺字图冒充中文图")


def make_figure(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    titles = ("TRAIN可靠方向：原生响应", "全部六方向：原生响应", "TRAIN可靠方向：最佳线性读出", "线性探针：长程留出预测", "RBF 探针：长程留出预测", "RBF 探针：各观测量留出误差")
    all_text = "".join(titles + MODEL_LABELS + TARGET_LABELS) + "冻结步无监督与监督两项候选损失诊断输入尺度相对范数均方标准差归一误差最佳秩一下限低者更好物理链分层配对重采条件于固定模型重复与已拟合探针不含训练种子和拟合不确定性非热指数认证该尺度无可靠方向"
    font_path = chinese_font(all_text)
    font = FontProperties(fname=font_path)
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    colors = ("#2563A6", "#D15A22")
    x = np.arange(3)
    response = report["response"]["splits"]["TEST"]
    reliable_response = report["response"]["train_only_reliable_projection"]["splits"]["TEST"]
    for axis, panel, method in ((axes[0, 0], reliable_response, "native_decoder"), (axes[0, 1], response, "native_decoder"), (axes[0, 2], reliable_response, "train_shared_linear_readout")):
        for model, label in enumerate(MODEL_LABELS):
            values = panel["methods"][method]["relative_frobenius"]["by_scale"]["models"][label]
            point, interval = np.asarray(values["mean"]), np.asarray(values["ci95"])
            axis.errorbar(x + (model - .5) * .12, point, yerr=np.maximum(np.stack((point - interval[0], interval[1] - point)), 0), marker="o", capsize=3, label=label, color=colors[model])
        floor = panel["rank1_floor"]["relative_frobenius"]
        axis.plot(x, floor["mean"], "k--", label="最佳秩一下限")
        if "no_verifiable_directions_by_scale" in panel:
            for scale, invalid in enumerate(panel["no_verifiable_directions_by_scale"]):
                if invalid:
                    axis.text(scale, .05, "无可靠方向", ha="center", transform=axis.get_xaxis_transform(), fontsize=9)
        axis.set_ylabel("相对范数误差（低者更好）")
    for axis, kind in ((axes[1, 0], "linear"), (axes[1, 1], "rbf")):
        record = report["longrange_probe"]["methods"][kind]["splits"]["TEST"]
        for model, label in enumerate(MODEL_LABELS):
            values = record["models"][label]["all_targets_nrmse"]
            point, interval = np.asarray(values["value"]), np.asarray(values["ci95"])
            axis.errorbar(x + (model - .5) * .12, point, yerr=np.maximum(np.stack((point - interval[0], interval[1] - point)), 0), marker="o", capsize=3, label=label, color=colors[model])
        axis.set_ylabel("TRAIN 标准差归一误差")
    record = report["longrange_probe"]["methods"]["rbf"]["splits"]["TEST"]
    for model, label in enumerate(MODEL_LABELS):
        # 此图仅展示已报告逐尺度 NRMSE 的算术平均，不冒充另一个 bootstrap CI。
        values = np.asarray(record["models"][label]["nrmse"]["value"]).mean(axis=0)
        axes[1, 2].bar(np.arange(5) + (model - .5) * .34, values, width=.32, label=label, color=colors[model])
    axes[1, 2].set_xticks(np.arange(5), TARGET_LABELS, rotation=18, ha="right")
    axes[1, 2].set_ylabel("各尺度 NRMSE 的均值（低者更好）")
    for i, axis in enumerate(axes.flat):
        axis.set_title(titles[i])
        axis.grid(alpha=.2, axis="y")
        if i != 5:
            axis.set_xticks(x, [str(s) for s in SCALES])
            axis.set_xlabel("输入尺度")
        axis.legend(fontsize=9)
    fig.suptitle("冻结 9000 步 Q：无监督与 K监督的两项候选损失诊断", fontsize=17)
    fig.text(.5, .015, "误差低者更好；区间为物理链分层配对重采。条件于固定模型、C重复和已拟合探针；不含训练种子/拟合不确定性，非热指数认证。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return dict(font=font.get_name(), font_path=font_path, chinese_glyph_check=True)


def main():
    parser = argparse.ArgumentParser(description="冻结历史 Q 两项候选目标的无标签统计验证")
    parser.add_argument("--input", required=True, type=Path, help="含 collected.npz、protocol.json、receipt.json 的目录")
    parser.add_argument("--output", required=True, type=Path, help="新建分析目录；不会覆盖已有目录")
    args = parser.parse_args()
    start = time.process_time()
    protocol = json.loads((args.input / "protocol.json").read_text())
    receipt = json.loads((args.input / "receipt.json").read_text())
    source_hash = sha256(args.input / "collected.npz")
    if receipt.get("collected_sha256") != source_hash:
        raise ValueError("公开统计文件 SHA 与采集回执不符")
    with np.load(args.input / "collected.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_inputs(arrays, protocol, receipt)
    args.output.mkdir(parents=True, exist_ok=False)
    train, test = arrays["split"] == 0, arrays["split"] == 1
    groups = arrays["group_index"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = {"TRAIN": stratified_bootstrap_weights(groups[train], rng), "TEST": stratified_bootstrap_weights(groups[test], rng)}
    response, saved_response = analyze_response(arrays, train, test, groups, weights, rng)
    probes, saved_probes = analyze_probes(arrays, train, test, groups, weights)
    response_difference = response["splits"]["TEST"]["methods"]["native_decoder"]["relative_frobenius"]["all_scales_equal_weight"]["supervised_minus_unsupervised"]
    reliable_response_difference = response["train_only_reliable_projection"]["splits"]["TEST"]["methods"]["native_decoder"]["relative_frobenius"]["all_scales_equal_weight"]["supervised_minus_unsupervised"]
    linear_difference = probes["methods"]["linear"]["splits"]["TEST"]["supervised_minus_unsupervised"]["all_scales_all_targets_nrmse"]
    rbf_difference = probes["methods"]["rbf"]["splits"]["TEST"]["supervised_minus_unsupervised"]["all_scales_all_targets_nrmse"]
    report = dict(
        diagnostic="冻结历史监督/无监督 Q 的候选损失验证", version=VERSION,
        source_actual_steps=[9000, 9000], model_order=MODEL_LABELS, scales=SCALES,
        observable_order=TARGET_LABELS, response_operator_order=OPERATOR_LABELS,
        qcr_updates=0, physical_sampling=0, MI_estimates=0, coupling_labels_accessed=False,
        input_public_statistics_sha256=source_hash, protocol=protocol, collection_receipt=receipt,
        statistical_unit="物理系综/链；先平均固定 C 重复和同链两半误差，固定 group 内 12rep 分层配对重采",
        bootstrap=dict(repeats=BOOTSTRAPS, seed=BOOTSTRAP_SEED, interval="逐项百分位95%；未校正多重比较", conditions="固定模型、数据域、C随机数与拟合后探针；不包含训练seed、探针重拟合或物理链内完整估计噪声"),
        response=response, longrange_probe=probes,
        verdict=dict(native_response_all_six=evidence_verdict(response_difference), native_response_train_reliable=evidence_verdict(reliable_response_difference),
                     linear_probe=evidence_verdict(linear_difference), rbf_probe=evidence_verdict(rbf_difference)),
        limitations=[
            "只验证这两个匹配9000步历史模型、原最近邻 Ising family、既有有限尺寸与 C；不是普适因果结论。",
            "监督组较好也不证明增加对应 loss 必能改进无监督 Q；监督组不较好则直接削弱这两项候选的当前动机。",
            "原生 decoder 误差包含 decoder 校准与一维容量限制，不可单独归因 Q 丢信息；TRAIN 最佳线性读出仅作拆分参考。",
            "长程线性可读性不是信息充分性；16中心 RBF 也不是所有可行解码器，差不证明信息绝对不存在。",
            "原family上的可读性/响应重构不验证 off-family 闭包、物理热方向、固定点或临界指数。",
            "公共目标可靠谱为物理链 bootstrap 的组均值谱参考；不强造秩，不强制一维g重构全部六个独立方向。",
            "各尺度已有限样本/链相关性，两个half与5C不作为独立物理链；区间不包含所有不确定性。",
        ], thermal_nu=None,
    )
    output_arrays = dict(**saved_response, **saved_probes, split=arrays["split"], group_index=groups,
                         replica_index=arrays["replica_index"], bootstrap_weights_train=weights["TRAIN"], bootstrap_weights_test=weights["TEST"])
    np.savez_compressed(args.output / "analysis_arrays.npz", **output_arrays)
    save_json(args.output / "analysis.json", report)
    try:
        report["figure_qa"] = make_figure(report, args.output / "comparison.png")
    except Exception as error:
        report["figure_qa"] = dict(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(args.output / "analysis.json", report)
        raise
    report["cpu_seconds"] = time.process_time() - start
    save_json(args.output / "analysis.json", report)
    save_json(args.output / "status.json", dict(status="complete", workflow_complete=True, qcr_updates=0,
                                               public_files={name: sha256(args.output / name) for name in ("analysis.json", "analysis_arrays.npz", "comparison.png")}))
    print(json.dumps(jsonable(dict(status="complete", verdict=report["verdict"], cpu_seconds=report["cpu_seconds"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
