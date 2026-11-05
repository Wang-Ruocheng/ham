"""历史冻结 g 的微观 Ising 条件 NLL 诊断；绝不导入模型或 PT。

主分析用预先固定的独立物理链配对，次分析用已有构型半组交叉评分。
仅读公开 g 数组和五类整数计数；不读取 K 标签，不更新 Q/C/R。
"""
from __future__ import annotations

import os

for _thread_option in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_option] = "1"

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "ising_micro_conditional_nll_check_v1"
EXPECTED_COLLECTED_SHA256 = "ca791cd4b78d32d5830490f0f1c111b0eaa7852a8d4203250ed70d2f751f47dd"
T_VALUES = np.array([-4., -2., 0., 2., 4.])
IDENTITY_KEYS = ("split", "ensemble_index", "group_index", "replica_index")
MODEL_NAMES = ("unsupervised", "supervised")
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 2026101713
GRAD_TOLERANCE = 1e-10


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            raise ValueError("诊断输出包含非有限数值，不允许静默转换")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def normalized_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape[-1] != len(T_VALUES) or not np.isfinite(counts).all():
        raise ValueError("五类计数形状或有限性错误")
    if (counts < 0).any() or (counts.sum(axis=-1) <= 0).any():
        raise ValueError("计数必须非负且每行具有正总数")
    return counts / counts.sum(axis=-1, keepdims=True)


def conditional_nll(kappa: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """单位 nats/site；计数末轴依次为 t=-4,-2,0,2,4。"""
    kappa = np.asarray(kappa, dtype=np.float64)
    if not np.isfinite(kappa).all():
        raise ValueError("kappa 中出现非有限数值")
    probabilities = normalized_counts(counts)
    return np.sum(probabilities * np.logaddexp(0., -2. * kappa[..., None] * T_VALUES), axis=-1)


def objective_gradient_hessian(
    theta: np.ndarray, design: np.ndarray, counts: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """逐行等权的凸 NLL、精确梯度和 Hessian；不含正则项。"""
    theta = np.asarray(theta, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or theta.shape != (design.shape[1],):
        raise ValueError("设计矩阵或参数维度错误")
    if counts.shape != (len(design), 5):
        raise ValueError("计数与设计矩阵不匹配")
    probabilities = normalized_counts(counts)
    kappa = np.einsum("ni,i->n", design, theta, optimize=False)
    logits = -2. * kappa[:, None] * T_VALUES
    softplus = np.logaddexp(0., logits)
    sigmoid = np.exp(-np.logaddexp(0., -logits))
    # 此表达式避免 sigmoid 先舍入成 1 后丢失曲率。
    sigmoid_variance = np.exp(-softplus - np.logaddexp(0., -logits))
    derivative = np.sum(probabilities * (-2. * T_VALUES) * sigmoid, axis=-1)
    curvature = np.sum(probabilities * (4. * T_VALUES ** 2) * sigmoid_variance, axis=-1)
    gradient = np.einsum("ni,n->i", design, derivative, optimize=False) / len(design)
    hessian = np.einsum("ni,n,nj->ij", design, curvature, design, optimize=False) / len(design)
    objective = float(np.mean(np.sum(probabilities * softplus, axis=-1)))
    if not (np.isfinite(objective) and np.isfinite(gradient).all() and np.isfinite(hessian).all()):
        raise FloatingPointError("凸优化目标/梯度/Hessian 非有限")
    return objective, gradient, hessian


def newton_fit(design: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """确定性的 Newton + Armijo 回溯；不加 ridge、不改变阈值重试。

    正负 t 均有计数的 TRAIN 行所张成的设计空间满秩，足以保证有限唯一
    最优点；允许其余行只有单侧计数。常数拟合只要求全 TRAIN 合计双侧。
    对无法通过这些充分条件的数据明确拒绝，绝不把无限参数当作收敛。
    """
    design = np.asarray(design, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if design.ndim != 2 or len(design) == 0 or not np.isfinite(design).all():
        raise ValueError("设计矩阵为空或包含非有限值")
    if counts.shape != (len(design), 5):
        raise ValueError("计数与设计矩阵不匹配")
    normalized_counts(counts)
    negative_counts = counts[:, :2].sum(axis=1)
    positive_counts = counts[:, 3:].sum(axis=1)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("设计矩阵列秩不足；不使用伪逆或正则化制造可识别性")
    mixed_rows = (negative_counts > 0) & (positive_counts > 0)
    mixed_design_rank = int(np.linalg.matrix_rank(design[mixed_rows])) if mixed_rows.any() else 0
    if design.shape[1] == 1 and np.array_equal(design, np.ones_like(design)):
        if negative_counts.sum() <= 0 or positive_counts.sum() <= 0:
            raise ValueError("常数拟合 TRAIN 合计未包含双侧计数，有限最优点不存在")
        finite_optimum_certificate = "constant_design_and_pooled_training_counts_have_both_signs_of_nonzero_t"
    elif mixed_design_rank == design.shape[1]:
        finite_optimum_certificate = "mixed_sign_training_rows_span_the_full_design_space"
    else:
        raise ValueError("双侧 TRAIN 行设计秩不足：未通过有限最优点充分条件，不宣称已证明完全分离")

    theta = np.zeros(design.shape[1], dtype=np.float64)
    line_search_reductions = 0
    for iteration in range(101):
        objective, gradient, hessian = objective_gradient_hessian(theta, design, counts)
        eigenvalues = np.linalg.eigvalsh(hessian)
        if eigenvalues[0] <= 0 or eigenvalues[-1] / eigenvalues[0] > 1e14:
            raise RuntimeError("Hessian 非正定或数值病态；无正则化回退")
        if np.max(np.abs(gradient)) <= GRAD_TOLERANCE:
            return theta, {
                "converged": True,
                "iterations": iteration,
                "objective_nats_per_site": objective,
                "gradient": gradient,
                "gradient_inf_norm": np.max(np.abs(gradient)),
                "gradient_tolerance": GRAD_TOLERANCE,
                "hessian": hessian,
                "hessian_eigenvalues": eigenvalues,
                "hessian_condition_number": eigenvalues[-1] / eigenvalues[0],
                "newton_decrement_squared": float(gradient @ np.linalg.solve(hessian, gradient)),
                "line_search_reductions": line_search_reductions,
                "finite_unique_optimum_certificate": finite_optimum_certificate,
                "training_rows": len(design),
                "mixed_sign_training_rows": int(mixed_rows.sum()),
                "mixed_sign_design_rank": mixed_design_rank,
                "minimum_negative_t_count_per_training_row": float(negative_counts.min()),
                "minimum_positive_t_count_per_training_row": float(positive_counts.min()),
                "regularization": 0.,
            }
        if iteration == 100:
            break
        direction = np.linalg.solve(hessian, gradient)
        directional_derivative = float(gradient @ direction)
        accepted = False
        for reduction in range(61):
            step = 2. ** (-reduction)
            candidate = theta - step * direction
            next_objective, _, _ = objective_gradient_hessian(candidate, design, counts)
            # At very small gradients the decrease can round to zero; Armijo
            # remains the only acceptance rule (no test-set-based fallback).
            if next_objective <= objective - 1e-4 * step * directional_derivative:
                theta = candidate
                line_search_reductions += reduction
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Newton 回溯未找到下降步；不返回未收敛拟合器")
    raise RuntimeError("Newton 超过 100 步且梯度未收敛")


def fit_affine_train_only(x: np.ndarray, counts: np.ndarray, train: np.ndarray) -> dict[str, Any]:
    """输入每个统计单元两个方向；仅 TRAIN 决定中心、尺度和拟合参数。"""
    x = np.asarray(x, dtype=np.float64)
    counts = np.asarray(counts)
    train = np.asarray(train, dtype=bool)
    if x.shape != counts.shape[:-1] or train.shape != (len(x),) or not train.any():
        raise ValueError("TRAIN 掩码/输入/计数形状错误")
    train_x = x[train].reshape(-1)
    mean = float(train_x.mean())
    std = float(train_x.std(ddof=0))
    if not np.isfinite(std) or std <= 0.:
        raise ValueError("TRAIN g 为常数，二参数仿射读出不可识别")
    standardized = (train_x - mean) / std
    design = np.column_stack((standardized, np.ones_like(standardized)))
    theta, audit = newton_fit(design, counts[train].reshape(-1, 5))
    a = float(theta[0] / std)
    b = float(theta[1] - a * mean)
    return {
        "a": a, "b": b, "train_g_mean": mean, "train_g_std_ddof0": std,
        "standardized_slope": float(theta[0]), "standardized_intercept": float(theta[1]),
        "audit": audit,
    }


def apply_affine(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    # 使用拟合时的 TRAIN 标准化，减轻 a*g+b 在极端 gauge 下相消。
    return fit["standardized_slope"] * ((x - fit["train_g_mean"]) / fit["train_g_std_ddof0"]) + fit["standardized_intercept"]


def fit_constant_train_only(counts: np.ndarray, train: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(counts)[np.asarray(train, dtype=bool)].reshape(-1, 5)
    if not len(selected):
        raise ValueError("TRAIN 为空")
    theta, audit = newton_fit(np.ones((len(selected), 1)), selected)
    return {"b": float(theta[0]), "audit": audit}


def build_independent_pairs(split: np.ndarray, groups: np.ndarray, replicas: np.ndarray) -> np.ndarray:
    """同 split/condition 的固定 (0,1),(2,3),...,(10,11)，每链只使用一次。"""
    split, groups, replicas = map(np.asarray, (split, groups, replicas))
    if not (split.ndim == 1 and split.shape == groups.shape == replicas.shape):
        raise ValueError("物理链身份维度不匹配")
    result = []
    for split_id in np.unique(split):
        for group in np.unique(groups[split == split_id]):
            rows = np.flatnonzero((split == split_id) & (groups == group))
            if len(rows) != 12 or not np.array_equal(np.sort(replicas[rows]), np.arange(12)):
                raise ValueError("每个 split/condition 必须正好有 replica 0..11 各一条")
            ordered = rows[np.argsort(replicas[rows])]
            result.extend(ordered.reshape(6, 2).tolist())
    pairs = np.asarray(result, dtype=np.int64)
    if len(np.unique(pairs)) != len(split) or len(pairs) * 2 != len(split):
        raise ValueError("链配对不互斥或有遗漏")
    if not (np.array_equal(split[pairs[:, 0]], split[pairs[:, 1]]) and
            np.array_equal(groups[pairs[:, 0]], groups[pairs[:, 1]])):
        raise ValueError("链配对错误地跨 split 或 condition")
    return pairs


def stratified_bootstrap_indices(groups: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    """各固定 condition 内重采统计单元，不重采 condition 本身。"""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or repeats <= 0:
        raise ValueError("bootstrap 组或重复数错误")
    rng = np.random.default_rng(seed)
    pieces = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        picks = rng.integers(0, len(members), size=(repeats, len(members)))
        pieces.append(members[picks])
    return np.concatenate(pieces, axis=1)


def paired_summary(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError("配对比较必须为两个模型的有限单元级损失")
    model_means = values[:, indices].mean(axis=-1)
    delta = values[1] - values[0]
    delta_bootstrap = delta[indices].mean(axis=-1)
    return {
        "models": {
            name: {"mean_nats_per_site": float(values[m].mean()),
                   "bootstrap_percentile_ci95": np.quantile(model_means[m], [.025, .975])}
            for m, name in enumerate(MODEL_NAMES)
        },
        "supervised_minus_unsupervised": {
            "mean_nats_per_site": float(delta.mean()),
            "bootstrap_percentile_ci95": np.quantile(delta_bootstrap, [.025, .975]),
            "negative_favors": "supervised",
        },
        "statistical_units": values.shape[1],
    }


def analyze_protocol(
    name: str, x: np.ndarray, target_counts: np.ndarray,
    split: np.ndarray, groups: np.ndarray, source_rows: np.ndarray,
    target_rows: np.ndarray, bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """x 为 model,unit,direction；每个单元的两个方向等权平均。"""
    train = split == 0
    if x.shape != (2, len(split), 2) or target_counts.shape != (len(split), 2, 5):
        raise ValueError("评分协议数组维度错误")
    fits = [fit_affine_train_only(x[m], target_counts, train) for m in range(2)]
    constant_fit = fit_constant_train_only(target_counts, train)
    kappas = {
        "raw_fixed_kappa_equals_g": x.copy(),
        "train_only_affine": np.stack([apply_affine(x[m], fits[m]) for m in range(2)]),
        "train_only_constant": np.full_like(x, constant_fit["b"]),
    }
    losses = {method: conditional_nll(kappa, target_counts).mean(axis=-1) for method, kappa in kappas.items()}
    result: dict[str, Any] = {
        "name": name,
        "score_unit": "nats_per_site",
        "direction_aggregation": "equal_mean_of_two_directions_within_unit",
        "affine_fits": dict(zip(MODEL_NAMES, fits)),
        "constant_fit": constant_fit,
        "bootstrap": {
            "repeats": BOOTSTRAPS,
            "seed_train": bootstrap_seed,
            "seed_test": bootstrap_seed + 1,
            "scheme": "stratified_within_each_fixed_condition_and_paired_across_models_and_methods",
            "interval": "percentile_95_percent_conditional_on_frozen_fitted_readouts",
            "excludes": ["readout_fitting_uncertainty", "QCR_training_seed_uncertainty", "condition_sampling_uncertainty", "finite_size_or_model_truncation_bias"],
        },
        "methods": {},
    }
    arrays = {"unit_split": split, "unit_group_index": groups,
              "source_ensemble_row": source_rows, "target_ensemble_row": target_rows}
    bootstrap = {}
    for split_id, label in ((0, "train"), (1, "test")):
        mask = split == split_id
        bootstrap[label] = stratified_bootstrap_indices(groups[mask], BOOTSTRAPS, bootstrap_seed + split_id)
        arrays[f"bootstrap_indices_{label}"] = bootstrap[label]
    for method, values in losses.items():
        arrays[f"{method}_kappa"] = kappas[method]
        arrays[f"{method}_unit_nll"] = values
        result["methods"][method] = {}
        for split_id, label in ((0, "train"), (1, "test")):
            mask = split == split_id
            summary = paired_summary(values[:, mask], bootstrap[label])
            summary["fixed_conditions"] = len(np.unique(groups[mask]))
            summary["statistical_units_per_condition"] = [int(np.sum(groups[mask] == group)) for group in np.unique(groups[mask])]
            result["methods"][method][label] = summary
    return result, arrays


def validate_histograms(histograms: dict[str, Any], collected: dict[str, np.ndarray]) -> np.ndarray:
    for key in IDENTITY_KEYS:
        if key not in histograms or not np.array_equal(np.asarray(histograms[key]), collected[key]):
            raise ValueError(f"histogram 与 collected 的 {key} 不完全一致")
    counts = np.asarray(histograms.get("counts"))
    if counts.shape != (768, 2, 5) or counts.dtype.kind not in "iu":
        raise ValueError("counts 必须为 (768,2,5) 精确整数数组")
    if (counts < 0).any() or not np.all(counts.sum(axis=-1) == 64 * 108 ** 2):
        raise ValueError("每个半组必须恰好包含 64*108² 个自旋位置")
    if "bins" in histograms and not np.array_equal(np.asarray(histograms["bins"]), T_VALUES):
        raise ValueError("五类 t bins 不一致")
    return counts.astype(np.int64, copy=False)


def validate_histogram_receipt(histograms: dict[str, Any]) -> None:
    if histograms.get("status") != "complete":
        raise ValueError("histogram 收集未成功 complete")
    if histograms.get("workflow_complete") is not True:
        raise ValueError("histogram 收集 workflow_complete 未确认")
    if histograms.get("source_metadata_unchanged") is not True:
        raise ValueError("histogram 收集前后 source_metadata_unchanged 未确认")


def load_inputs(collected_path: Path, histogram_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    collected_sha = sha256(collected_path)
    if collected_sha != EXPECTED_COLLECTED_SHA256:
        raise ValueError("历史公开 collected.npz SHA256 与固定指纹不匹配")
    with np.load(collected_path, allow_pickle=False) as stored:
        needed = ("g", "g_full", "model_names") + IDENTITY_KEYS
        data = {key: stored[key] for key in needed}
    if tuple(data["model_names"].tolist()) != MODEL_NAMES:
        raise ValueError("模型顺序不一致")
    if data["g"].shape != (2, 768, 5, 3, 2) or data["g_full"].shape != (2, 768, 5, 3):
        raise ValueError("历史 g 数组形状不一致")
    for key in IDENTITY_KEYS:
        if data[key].shape != (768,) or data[key].dtype.kind not in "iu":
            raise ValueError(f"历史身份数组 {key} 不正确")
    if not np.array_equal(np.unique(data["split"], return_counts=True)[0], np.array([0, 1])) or not np.array_equal(np.unique(data["split"], return_counts=True)[1], np.array([516, 252])):
        raise ValueError("TRAIN/TEST 分割不匹配 516/252")
    for key in ("g", "g_full"):
        # 微观尺度独立于 C 抽样，严格逐位相等而非近似相等。
        micro = data[key][:, :, :, 0]
        if not np.isfinite(micro).all() or not np.array_equal(micro, np.repeat(micro[:, :, :1], 5, axis=2)):
            raise ValueError(f"{key} 微观五个 C draw 不是完全相同的有限值")
    histogram = json.loads(histogram_path.read_text())
    validate_histogram_receipt(histogram)
    counts = validate_histograms(histogram, data)
    audit = {
        "collected_path": str(collected_path.resolve()), "collected_sha256": collected_sha,
        "histograms_path": str(histogram_path.resolve()), "histograms_sha256": sha256(histogram_path),
        "identity_arrays_exact_match": True, "microscopic_C_draws_exactly_equal": True,
        "histogram_status": histogram["status"], "histogram_workflow_complete": True,
        "histogram_source_metadata_unchanged": True,
        "microscopic_C_draw_used": 0, "histogram_sites_per_half": 64 * 108 ** 2,
        "model_names": MODEL_NAMES,
    }
    return data, counts, audit


def run(collected_path: Path, histogram_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录: {output}")
    data, counts, input_audit = load_inputs(collected_path, histogram_path)
    pairs = build_independent_pairs(data["split"], data["group_index"], data["replica_index"])
    pair_split = data["split"][pairs[:, 0]]
    pair_groups = data["group_index"][pairs[:, 0]]
    primary, primary_arrays = analyze_protocol(
        "independent_chain_pairs", data["g_full"][:, pairs, 0, 0],
        counts.sum(axis=1)[pairs[:, ::-1]], pair_split, pair_groups,
        pairs, pairs[:, ::-1], BOOTSTRAP_SEED,
    )
    primary["statistical_unit"] = "disjoint_pair_of_independently_generated_physical_chains"
    primary["pair_rule"] = [[i, i + 1] for i in range(0, 12, 2)]
    primary["input_and_target_configurations_per_direction"] = 128
    rows = np.repeat(np.arange(768)[:, None], 2, axis=1)
    secondary, secondary_arrays = analyze_protocol(
        "secondary_cross_half", data["g"][:, :, 0, 0, :], counts[:, ::-1],
        data["split"], data["group_index"], rows, rows, BOOTSTRAP_SEED + 2,
    )
    secondary["statistical_unit"] = "one_physical_chain_with_two_correlated_configuration_halves"
    secondary["halves_independent"] = False
    secondary["input_and_target_configurations_per_direction"] = 64
    secondary_arrays["source_half"] = np.tile([0, 1], (768, 1))
    secondary_arrays["target_half"] = np.tile([1, 0], (768, 1))
    arrays = {f"{prefix}__{key}": value for prefix, protocol_arrays in (
        ("independent_chain_pairs", primary_arrays), ("secondary_cross_half", secondary_arrays),
    ) for key, value in protocol_arrays.items()}
    for key in IDENTITY_KEYS:
        arrays[key] = data[key]
    result = {
        "version": VERSION, "status": "complete", "training_loss_implemented": False,
        "QCR_parameter_updates": 0, "new_model_inference": False,
        "uses_K_labels": False, "input_audit": input_audit,
        "interpretation": {
            "raw": "Exact proposed fixed readout; sensitive to the coordinate gauge of g.",
            "affine": "Equal two-parameter TRAIN-only calibration, a separate information diagnostic, not the original fixed-readout loss.",
            "constant": "TRAIN-only one-parameter baseline with no g information.",
            "scope": "Known nearest-neighbor zero-field Ising conditional family, not a model-agnostic physical loss.",
        },
        "independent_chain_pairs": primary, "secondary_cross_half": secondary,
    }
    # 所有输入、优化和统计验证通过后才创建全新输出目录。
    output.mkdir(parents=True, exist_ok=False)
    array_path = output / "analysis_arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["analysis_arrays_sha256"] = sha256(array_path)
    result["analyzer_sha256"] = sha256(Path(__file__))
    (output / "analysis.json").write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--histograms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.collected, args.histograms, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "QCR_parameter_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
