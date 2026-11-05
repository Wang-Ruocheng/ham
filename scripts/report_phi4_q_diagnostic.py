"""固定 C 的连续场 Q 隔离诊断：仅读已保存 JSON/NPZ，不调用模型。

python -m scripts.report_phi4_q_diagnostic --training TRAINING --output NEW_DIR

Changing Q changes its coordinates. This report deliberately has no R-flow,
root-search, sampling, checkpoint, mutual-information, or inference code path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _json, _numeric, _plotting, sha256


INPUT_FILES = ("summary.json", "history_progress.json", "evaluation_arrays.npz", "status.json")
ARMS = ("source", "A", "B", "C", "D", "E")
BASELINES = ("constant", "PCA1", "PCA3")
LABELS = {
    "source": "原训练 Q（零新增更新）", "A": "A：源 Q 热启动",
    "B": "B：原随机初始化", "C": "C：训练均值偏置初始化",
    "D": "D：线性编码器", "E": "E：去除子集优化项",
    "constant": "训练均值常数", "PCA1": "训练集仿射主成分一维参考",
    "PCA3": "训练集仿射主成分三维参考",
}
COLORS = {"source": "#222222", "A": "#527ac5", "B": "#c5687f", "C": "#309374",
          "D": "#9e64bf", "E": "#de993e", "constant": "#777777", "PCA1": "#2ca5a9", "PCA3": "#829253"}
LIMITATIONS = [
    "这是固定已训练 C 的 Q-only 隔离诊断，不是完整 QCR 重训；正式源实验、模型和报告保持不变。",
    "原采样的低 ESS 和热冷差异警告原样保留，数据未认证平衡；重构改善不能反过来认证热化。",
    "训练使用固定 C 的两个有限随机粗粒化缓存，不等于原实验中持续变化的 C，也不是新增独立物理系综。",
    "所有更新仅用 TRAIN 的无标签局部统计；r、u、h 只用于事后绘图，未用于监督或选模。",
    "五臂同一固定训练种子和预定终点，无 TEST 早停；A 是源 Q 权重加全新 Adam 的热启动诊断，非原训练续训。",
    "C 的 TRAIN 均值偏置不引入 PCA 初始化；D 与 E 分别以 C 为起点改变编码结构或去掉子集优化项。",
    "E 同时按原完整 Q 目标评价；仅降低重构误差不意味着保留子集一致性或原目标整体改善。",
    "主成分参考只在 TRAIN 拟合、仅针对重构，不包含子集一致性目标，也不等于物理耦合维度或理论最优 RG。",
    "不同 Q 的潜变量属于不同坐标，不拼接、比较其数值大小或套用旧 R；本报告不提供新的 R(g)、不动点或临界指数。",
    "诊断只涉及保存的软局部统计；不保证完整连续场分布、高阶、长程信息、C 设计或物理闭合正确。",
]


def _metric_value(value, *names):
    for name in names:
        if name in value:
            return value[name]
    raise ValueError(f"Missing metric: {names[0]}")


def _metric_vectors(metric, scales):
    values = {}
    aliases = {
        "reconstruction_rmse_by_scale": ("reconstruction_rmse_by_scale", "rmse_by_scale"),
        "reconstruction_nrmse_by_scale": ("reconstruction_nrmse_by_scale", "nrmse_by_scale"),
        "reconstruction_l2_by_scale": ("reconstruction_l2_by_scale", "reconstruction_l2"),
        "subset_l2_by_scale": ("subset_l2_by_scale", "subset_by_scale"),
    }
    for key, names in aliases.items():
        data = _numeric(_metric_value(metric, *names), key)
        if data.shape != (scales,) or np.any(data < 0):
            raise ValueError(f"{key} must be a nonnegative vector over scales")
        values[key] = data
    expected = (10 * values["reconstruction_l2_by_scale"].mean()
                + 20 * values["subset_l2_by_scale"][0]
                + 5 * values["subset_l2_by_scale"][1:].mean())
    full = float(_metric_value(metric, "q_full_objective"))
    if not np.isfinite(full) or not np.isclose(full, expected, atol=1e-10, rtol=1e-7):
        raise ValueError("Saved full Q objective disagrees with 10/20/5 components")
    values["q_full_objective"] = full
    return values


def _reconstruction(prediction, target, denominator):
    error = prediction - target
    rmse = np.sqrt(np.mean(error * error, axis=(0, 2)))
    return dict(reconstruction_rmse_by_scale=rmse.tolist(),
                reconstruction_l2_by_scale=np.mean(np.sum(error * error, axis=-1), axis=0).tolist(),
                reconstruction_nrmse_by_scale=(rmse / denominator).tolist())


def variance_decomposition(direct, cell_ids):
    """Finite-sample total = weighted between-condition + within-condition."""
    center = direct.mean(axis=0)
    between = np.zeros_like(center)
    within = np.zeros_like(center)
    for cell in np.unique(cell_ids):
        values = direct[cell_ids == cell]
        weight = len(values) / len(direct)
        between += weight * (values.mean(axis=0) - center) ** 2
        within += weight * np.var(values, axis=0, ddof=0)
    total = np.var(direct, axis=0, ddof=0)
    if not np.allclose(total, between + within, atol=1e-12, rtol=1e-8):
        raise ValueError("Within/between variance decomposition is inconsistent")
    fraction = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    return dict(total_variance_by_scale=total[:, 0].tolist(),
                between_condition_variance_by_scale=between[:, 0].tolist(),
                within_condition_variance_by_scale=within[:, 0].tolist(),
                between_fraction_by_scale=[float(v) if t > 0 else None
                    for v, t in zip(fraction[:, 0], total[:, 0])],
                estimator="按实际物理链数加权、ddof=0 的描述性方差分解；非独立误差条或物理维度认证")


def load_diagnostic(training):
    """Only read the four public artifacts; reject an incomplete endpoint."""
    root = Path(training).resolve()
    fingerprints = {name: sha256(root / name) for name in INPUT_FILES}
    summary, history, status = (_json(root / name) for name in ("summary.json", "history_progress.json", "status.json"))
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action", False) or summary.get("status") != "complete"):
        raise ValueError("Q-only training must have a successfully completed endpoint")
    for field, filename in (("summary_sha256", "summary.json"),
                            ("history_progress_sha256", "history_progress.json"),
                            ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(field) is not None and status[field] != fingerprints[filename]:
            raise ValueError(f"Saved source {field} mismatch")
    if summary.get("labels_used_in_training", False):
        raise ValueError("Q-only diagnosis must remain unsupervised")
    if summary.get("source_unchanged") is False:
        raise ValueError("Frozen source changed during Q-only diagnosis")
    steps = summary.get("actual_steps")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("A fixed positive Q-only training endpoint is required")
    rows = history.get("history", [])
    if not isinstance(rows, list) or not rows or rows[0].get("step") != 0 or rows[-1].get("step") != steps:
        raise ValueError("History must contain initialization and the fixed endpoint")
    history_steps = [row.get("step") for row in rows]
    if any(not isinstance(step, (int, float)) or not np.isfinite(step) for step in history_steps):
        raise ValueError("Nonfinite history step")
    if any(b <= a for a, b in zip(history_steps, history_steps[1:])):
        raise ValueError("History steps must be strictly increasing")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    dimensions = None
    for split in ("train", "test"):
        target = _numeric(arrays[f"{split}_target"], f"{split}_target")
        if target.ndim != 3 or len(target) == 0 or target.shape[1] < 2 or target.shape[2] < 1:
            raise ValueError("Feature targets must be [physical ensemble, scale, feature]")
        if dimensions is not None and target.shape[1:] != dimensions:
            raise ValueError("TRAIN/TEST target feature geometry disagrees")
        dimensions = target.shape[1:]
        arrays[f"{split}_target"] = target
        labels = _numeric(arrays[f"eval_{split}_labels"], f"eval_{split}_labels")
        if labels.shape != (len(target), 3):
            raise ValueError("Evaluation labels must use r, u, h columns")
        ids = np.asarray(arrays[f"eval_{split}_cell_ids"])
        if ids.shape != (len(target),) or ids.dtype.kind not in "iUS":
            raise ValueError("Cell IDs must identify every physical ensemble")
        for cell in np.unique(ids):
            if not np.all(labels[ids == cell] == labels[ids == cell][0]):
                raise ValueError("Cell IDs and physical evaluation labels disagree")
        arrays[f"eval_{split}_labels"] = labels
    if set(arrays["eval_train_cell_ids"].tolist()) & set(arrays["eval_test_cell_ids"].tolist()):
        raise ValueError("TRAIN/TEST cells overlap")
    if (set(map(tuple, arrays["eval_train_labels"].tolist()))
            & set(map(tuple, arrays["eval_test_labels"].tolist()))):
        raise ValueError("TRAIN/TEST evaluation conditions overlap")
    scales, features = dimensions
    denominator = np.sqrt(np.mean(np.var(arrays["train_target"], axis=0, ddof=0), axis=-1))
    if np.any(denominator <= 0):
        raise ValueError("Original TRAIN statistics must have nonzero per-scale variance")
    saved = _numeric(arrays["train_std_by_scale"], "train_std_by_scale")
    if saved.shape != (scales,) or not np.allclose(saved, denominator, atol=1e-12, rtol=1e-8):
        raise ValueError("NRMSE denominator is not the original fixed TRAIN population std")
    checked = {}
    for arm in ARMS:
        record = summary.get("arms", {}).get(arm)
        if not isinstance(record, dict):
            raise ValueError(f"Missing Q-only arm {arm}")
        checked[arm] = {}
        for split in ("train", "test"):
            target = arrays[f"{split}_target"]
            direct = _numeric(arrays[f"{arm}_{split}_direct"], f"{arm}_{split}_direct")
            if direct.shape != (*target.shape[:2], 1):
                raise ValueError("Each Q-only arm must preserve scalar direct coordinates")
            arrays[f"{arm}_{split}_direct"] = direct
            prediction = _numeric(arrays[f"{arm}_{split}_reconstruction"], "reconstruction")
            if prediction.shape != target.shape:
                raise ValueError("Saved reconstruction and target shapes disagree")
            arrays[f"{arm}_{split}_reconstruction"] = prediction
            result = _reconstruction(prediction, target, denominator)
            metric = _metric_vectors(record["metrics"][split], scales)
            for key in result:
                if not np.allclose(metric[key], result[key], atol=1e-12, rtol=1e-7):
                    raise ValueError(f"Saved {arm}/{split} {key} disagrees with public arrays")
            result.update(subset_l2_by_scale=metric["subset_l2_by_scale"].tolist(),
                          q_full_objective=metric["q_full_objective"])
            result["latent_variance_decomposition"] = variance_decomposition(direct,
                arrays[f"eval_{split}_cell_ids"])
            optimized = record["metrics"][split].get("optimization_objective")
            if optimized is not None:
                expected = 10 * np.mean(result["reconstruction_l2_by_scale"]) if arm == "E" else result["q_full_objective"]
                if not np.isfinite(optimized) or not np.isclose(optimized, expected, atol=1e-10, rtol=1e-7):
                    raise ValueError("Optimization objective must distinguish E from original full Q objective")
                result["optimization_objective"] = float(optimized)
            oracle_key = f"{arm}_{split}_oracle_reconstruction"
            if oracle_key in arrays:
                oracle = _numeric(arrays[oracle_key], oracle_key)
                if oracle.shape != target.shape:
                    raise ValueError("Saved affine-decoder oracle shape mismatch")
                result["decoder_oracle"] = _reconstruction(oracle, target, denominator)
            pooled_key = f"{arm}_{split}_direct_full"
            if pooled_key in arrays:
                pooled = _numeric(arrays[pooled_key], pooled_key)
                if pooled.shape != direct.shape:
                    raise ValueError("Pooled-feature and mean-half Q coordinates disagree in shape")
                difference = pooled - direct
                result["pooled_vs_mean_half_Q_rms_by_scale"] = np.sqrt(
                    np.mean(difference * difference, axis=(0, 2))).tolist()
            checked[arm][split] = result
    baselines = {}
    for baseline in BASELINES:
        if not all(f"{baseline}_{split}_reconstruction" in arrays for split in ("train", "test")):
            continue
        baselines[baseline] = {}
        for split in ("train", "test"):
            target = arrays[f"{split}_target"]
            prediction = _numeric(arrays[f"{baseline}_{split}_reconstruction"], "baseline reconstruction")
            if prediction.shape != target.shape:
                raise ValueError("Baseline and feature targets disagree")
            baselines[baseline][split] = _reconstruction(prediction, target, denominator)
    if "constant" not in baselines or "PCA1" not in baselines:
        raise ValueError("Matched TRAIN constant and PCA1 references must be saved")
    if fingerprints != {name: sha256(root / name) for name in INPUT_FILES}:
        raise ValueError("Public source changed while reading")
    return dict(root=root, summary=summary, history=history, arrays=arrays,
                fingerprints=fingerprints, metrics=checked, baselines=baselines,
                scales=scales, features=features, train_std_by_scale=denominator.tolist())


def plot_reconstruction(evidence, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), constrained_layout=True)
    x = np.arange(evidence["scales"])
    combined = {**evidence["metrics"], **evidence["baselines"]}
    for name, metrics in combined.items():
        for ax, key in zip(axes, ("reconstruction_rmse_by_scale", "reconstruction_nrmse_by_scale")):
            ax.plot(x, metrics["test"][key], "o--" if name in BASELINES else "o-",
                    color=COLORS[name], label=LABELS[name], linewidth=1.5, markersize=4)
    for ax, ylabel in zip(axes, ("测试集绝对均方根误差", "按原训练统计标准差归一化的误差")):
        ax.set(xlabel="尺度 k", ylabel=ylabel, xticks=x)
        ax.grid(alpha=.22)
    axes[0].set_title("相同局部统计坐标下的重构")
    axes[1].set_title("所有臂共用原训练集、分尺度分母")
    axes[1].legend(fontsize=8)
    fig.suptitle("固定 C 的 Q 隔离诊断：主成分参考仅评价重构，不用于初始化")
    fig.savefig(output / "Q_reconstruction_comparison.png", dpi=170)
    plt.close(fig)


def plot_losses(evidence, output, plt):
    rows = evidence["history"]["history"]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    panels = [("train", "full", "训练集原完整 Q 目标"),
              ("test", "full", "测试集原完整 Q 目标"),
              ("test", "optimized", "测试集各臂优化目标（E 不同）"),
              ("test", "reconstruction", "测试集加权重构：10 倍"),
              ("test", "micro", "测试集微观子集项：20 倍"),
              ("test", "coarse", "测试集粗尺度子集项：5 倍")]
    for ax, (split, component, title) in zip(axes.flat, panels):
        for name in ARMS[1:]:
            pairs = []
            for row in rows:
                metric = row.get("arms", {}).get(name, {}).get(split)
                if not isinstance(metric, dict):
                    continue
                if "metrics" in metric:
                    metric = metric["metrics"]
                vectors = _metric_vectors(metric, evidence["scales"])
                if component == "full":
                    value = vectors["q_full_objective"]
                elif component == "optimized":
                    value = metric.get("optimization_objective", vectors["q_full_objective"])
                elif component == "reconstruction":
                    value = 10 * vectors["reconstruction_l2_by_scale"].mean()
                elif component == "micro":
                    value = 20 * vectors["subset_l2_by_scale"][0]
                else:
                    value = 5 * vectors["subset_l2_by_scale"][1:].mean()
                pairs.append((row["step"], float(value)))
            if pairs:
                ax.plot(*zip(*pairs), color=COLORS[name], label=LABELS[name], linewidth=1.5)
        ax.set(xlabel="Q 独立更新次数", ylabel="保存的评价目标", title=title)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.grid(alpha=.22)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("固定终点评价，无测试集早停：统一完整目标与实际优化目标分开展示")
    fig.savefig(output / "Q_losses.png", dpi=170)
    plt.close(fig)


def plot_coupling(evidence, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.5), constrained_layout=True)
    a = evidence["arrays"]
    scale_colors = ("#315ba0", "#dc8240", "#418a64", "#9968b8")
    for ax, arm in zip(axes.flat, ARMS):
        for split, split_label, marker in (("train", "训练", "o"), ("test", "测试", "^")):
            r = a[f"eval_{split}_labels"][:, 0]
            ids = a[f"eval_{split}_cell_ids"]
            direct = a[f"{arm}_{split}_direct"][:, :, 0]
            for scale in range(evidence["scales"]):
                color = scale_colors[scale % len(scale_colors)]
                ax.scatter(r, direct[:, scale], marker=marker, color=color, s=12,
                           alpha=.26, linewidths=0)
                means = sorted((float(r[ids == cell][0]), float(direct[ids == cell, scale].mean()))
                               for cell in np.unique(ids))
                ax.plot(*zip(*means), marker=marker, linestyle="-" if split == "train" else "--",
                        markersize=3, linewidth=1.1, color=color, label=f"{split_label}，k={scale}")
        ax.set(xlabel="生成参数 r（仅评价）", ylabel="该 Q 自身坐标 g", title=LABELS[arm])
        ax.grid(alpha=.22)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("不同面板为不同 Q 坐标：只观察耦合到表示的形状，不套用旧 R 或寻找新不动点")
    fig.savefig(output / "r_to_g.png", dpi=170)
    plt.close(fig)


def plot_subset(evidence, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    x = np.arange(evidence["scales"])
    for name, metrics in evidence["metrics"].items():
        for ax, split, title in zip(axes, ("train", "test"), ("训练集", "测试集")):
            ax.plot(x, metrics[split]["subset_l2_by_scale"], "o-", color=COLORS[name], label=LABELS[name])
            ax.set(xlabel="尺度 k", ylabel="两半系综编码差的平方范数（未加权）", title=title, xticks=x)
            ax.set_yscale("symlog", linthresh=1e-7)
            ax.grid(alpha=.22)
    axes[1].legend(fontsize=8)
    fig.suptitle("子集稳定性：E 没有优化该项，但仍以同一原始指标评价")
    fig.savefig(output / "subset_consistency.png", dpi=170)
    plt.close(fig)


def write_report(training, output, *, font=None):
    evidence = load_diagnostic(training)
    output = Path(output).resolve()
    if output == evidence["root"] or evidence["root"].is_relative_to(output):
        raise ValueError("Output must not contain or replace the training source")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_reconstruction(evidence, output, plt)
    plot_losses(evidence, output, plt)
    plot_coupling(evidence, output, plt)
    plot_subset(evidence, output, plt)
    metrics = dict(schema_version=1, experiment="phi4_fixed_C_Q_only_diagnostic",
        actual_steps=evidence["summary"]["actual_steps"], theory_dim=1,
        arms=evidence["metrics"], baselines=evidence["baselines"],
        diagnostic={name: evidence["summary"]["arms"][name].get("diagnostic", {}) for name in ARMS},
        pooled_TRAIN_PC_variance_fraction=evidence["summary"].get("pooled_TRAIN_PC_variance_fraction"),
        source_quality=evidence["summary"].get("source_quality"),
        train_std_by_scale=evidence["train_std_by_scale"],
        normalization="原 TRAIN 每尺度、逐特征人口方差的均方根；ddof=0；所有臂分母固定",
        q_original_objective="10 mean_scale reconstruction_L2 + 20 subset_L2[k=0] + 5 mean_coarse subset_L2",
        pca_reference_is_training_initialization=False, new_R_flow_available=False,
        labels_used_in_training=False, labels_used_in_report=True, thermal_nu=None,
        physical_critical_point_certified=False, equilibrium_certified=False,
        limitations=LIMITATIONS)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Forward precisely the public statistical arrays; never physical configurations.
    np.savez_compressed(output / "evaluation_arrays.npz", **evidence["arrays"])
    after = {name: sha256(evidence["root"] / name) for name in INPUT_FILES}
    if after != evidence["fingerprints"]:
        raise ValueError("Public sources changed during reporting")
    outputs = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    receipt = dict(schema_version=1, status="complete", source_training=str(evidence["root"]),
        sources_sha256_before=evidence["fingerprints"], sources_sha256_after=after,
        public_sources_unchanged=True, model_loaded=False, checkpoint_loaded=False,
        physical_data_loaded=False, GPU_used=False, inference_performed=False,
        physical_sampling_performed=False, MI_computed=False, root_search_performed=False,
        new_R_flow_available=False, thermal_nu=None, equilibrium_certified=False,
        outputs_sha256=outputs, public_files=sorted([*outputs, "report.json", "status.json"]),
        limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    receipt, _ = write_report(args.training, args.output, font=args.font)
    print(json.dumps(dict(status=receipt["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
