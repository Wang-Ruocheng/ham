"""Offline, model-free Chinese report for the explicitly requested R-only ablation.

Inputs are public JSON/NPZ files produced by the bounded CPU runner.  No model,
optimizer, physical configuration, checkpoint, root search, or exponent fit is
loaded or evaluated here.  A fresh ``report`` directory is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_FILES = ("plan.json", "history.json", "metrics.json", "summary.json", "evaluation_arrays.npz")
MODELS = ("source", "A_equal_target", "B_macro_source")
ARMS = MODELS[1:]
DISPLAY = {"source": "源 R（QCR 3000 步）", "A_equal_target": "A：等终点 R-only", "B_macro_source": "B：宏观起点加权 R-only"}
COLORS = {"source": "#777777", "A_equal_target": "#0072B2", "B_macro_source": "#D55E00"}
EXPECTED_PAIRS = np.asarray([(s, t) for t in range(1, 4) for s in range(t)], dtype=np.int64)
LIMITATIONS = [
    "这是用户明确请求的冻结 Q/C、仅更新 R 的起点加权消融，不改变当前默认 QCR 配方。",
    "A 对终点等权并在各终点内平均起点；B 对起点按 1:2:3 加权并在各起点内平均后续终点；两者保留全部六个预测对。",
    "源模型来自已完成的 QCR 3000 步；A/B 各自 fresh Adam、额外 R 更新，不能称为完整 QCR 6000 步。",
    "固定缓存 direct_g 是每条链两半各 64 个构型的 Q 编码之平均，不是对全部 128 个构型合并后再编码；这与旧训练随机抽取 64 个构型的目标实现不完全相同，不是严格原训练续训。",
    "κ、λ、h 只用于生成和事后评价，不参与 R 训练。原始 TRAIN 目标坐标标准差在所有比较中固定。",
    "每条件四条物理链；误差带为链间样本标准差而非置信区间，链内保存构型并非独立样本。原有 75 项采样质量警告仍保留。",
    "单物理数据与单优化器种子；低闭环误差、图中与恒等线的交点均不认证物理不动点、相变耦合或热临界指数。",
    "图中显示完整 R(g)，箭头为未经长度归一化的 R(g)−g；训练输入包络外为外推，包络内也不保证处处有数据支撑。",
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Nonfinite or nonnumeric public array: {name}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Incorrect public array shape: {name}; expected {shape}, got {value.shape}")
    return value.astype(np.float64, copy=False)


def load_public_run(output):
    """Validate the saved, immutable public contract without importing torch."""
    root = Path(output).resolve()
    pins = {name: _sha(root / name) for name in INPUT_FILES}
    plan, history, metrics, summary = [json.loads((root / name).read_text()) for name in INPUT_FILES[:4]]
    if not all(isinstance(value, dict) for value in (plan, metrics, summary)):
        raise ValueError("Plan, metrics, and summary must be objects")
    if summary.get("status") != "complete":
        raise ValueError("The R-only evaluation must be complete before reporting")
    for key in ("thermal_nu", "critical_kappa"):
        if summary.get(key) is not None:
            raise ValueError("The ablation must not claim critical parameters")
    if isinstance(history, dict):
        history = history.get("rows", history.get("history"))
    if not isinstance(history, list) or not history:
        raise ValueError("Missing R-only training history")
    steps = []
    for row in history:
        if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 0:
            raise ValueError("History step must be a nonnegative integer")
        steps.append(row["step"])
        for arm in ARMS:
            for key in ("objective", "equal_target_objective", "macro_source_objective"):
                value = row.get("arms", {}).get(arm, {}).get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError(f"Missing or invalid history value: {arm}/{key}")
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError("History steps must increase strictly")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["direct_g"] = _numeric(arrays["direct_g"], "direct_g", (60, 4, 1))
    if "direct_full_g" in arrays:
        arrays["direct_full_g"] = _numeric(arrays["direct_full_g"], "direct_full_g", (60, 4, 1))
    arrays["labels"] = _numeric(arrays["labels"], "labels", (60, 3))
    labels = arrays["labels"]
    if ((labels[:, 0] <= 0).any() or (labels[:, 1] <= 0).any()
            or not np.all(labels[:, 1] == labels[0, 1]) or not np.all(labels[:, 2] == 0)):
        raise ValueError("Expected positive κ, fixed λ>0, h=0 labels for evaluation only")
    splits = arrays["splits"]
    if splits.shape != (60,) or splits.dtype.kind not in "US" or set(splits.astype(str).tolist()) != {"train", "test"}:
        raise ValueError("Both lowercase train/test split vectors must be saved")
    arrays["splits"] = splits.astype(str)
    cells = arrays["cell_ids"]
    if cells.shape != (60,) or cells.dtype.kind not in "iuUS" or len(np.unique(cells)) != 15:
        raise ValueError("Expected 15 physical conditions with four chain encodings each")
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() != 4 or len(np.unique(splits[mask])) != 1 or not np.all(labels[mask] == labels[mask][0]):
            raise ValueError("A condition must retain four physical chains, one label and one split")
    train = arrays["splits"] == "train"
    if set(map(tuple, labels[train])) & set(map(tuple, labels[~train])):
        raise ValueError("Physical labels overlap TRAIN/TEST conditions")
    pairs = arrays["all_start_pairs"]
    if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, EXPECTED_PAIRS):
        raise ValueError("All-start pairs must have complete target-major order")
    std = _numeric(arrays["train_target_std"], "train_target_std")
    if std.shape == (3,):
        std = np.r_[np.nan, std]
    elif std.shape != (4,):
        raise ValueError("Expected three target or four scale TRAIN standard deviations")
    if (std[1:] <= 0).any():
        raise ValueError("TRAIN target standard deviations must be strictly positive")
    arrays["train_target_std"] = std
    if "train_target_mean" in arrays:
        arrays["train_target_mean"] = _numeric(arrays["train_target_mean"], "train_target_mean", (4,))
    arrays["flow_grid"] = _numeric(arrays["flow_grid"], "flow_grid")
    if arrays["flow_grid"].ndim == 1:
        arrays["flow_grid"] = arrays["flow_grid"][:, None]
    grid = arrays["flow_grid"]
    if grid.shape[1:] != (1,) or len(grid) < 17 or not (np.diff(grid[:, 0]) > 0).all():
        raise ValueError("Expected an ordered scalar flow grid")
    for model in MODELS:
        for suffix, shape in (("one_step", (60, 3, 1)), ("all_start", (60, 6, 1)),
                              ("rollout", (60, 4, 1)), ("flow_next", grid.shape)):
            key = f"{model}_{suffix}"
            arrays[key] = _numeric(arrays[key], key, shape)
        if not np.allclose(arrays[f"{model}_rollout"][:, 0], arrays["direct_g"][:, 0], atol=1e-12, rtol=1e-10):
            raise ValueError("Each rollout must start at the fixed microscopic Q encoding")
        for index, (s, t) in enumerate(pairs):
            prediction = arrays[f"{model}_all_start"][:, index]
            if t == s + 1 and not np.allclose(prediction, arrays[f"{model}_one_step"][:, s], atol=1e-12, rtol=1e-10):
                raise ValueError("Adjacent all-start and one-step predictions differ")
            if s == 0 and not np.allclose(prediction, arrays[f"{model}_rollout"][:, t], atol=1e-12, rtol=1e-10):
                raise ValueError("Microscopic all-start and rollout predictions differ")
    if pins != {name: _sha(root / name) for name in INPUT_FILES}:
        raise ValueError("Public input changed during loading")
    return plan, history, metrics, summary, arrays, pins


def comparison_metrics(arrays):
    """Same-coordinate metrics using the saved original TRAIN target scales."""
    direct = arrays["direct_g"]
    pairs = arrays["all_start_pairs"]
    stds = arrays["train_target_std"]
    means = arrays.get("train_target_mean", direct[arrays["splits"] == "train"].mean(axis=0)[:, 0])
    result = {"models": {}, "train_target_std_by_target": stds[1:].tolist(),
              "train_target_mean_by_target": means[1:].tolist(), "all_start_pairs": pairs.tolist(),
              "normalization": "fixed original TRAIN target std; ddof=0"}
    for model in MODELS:
        result["models"][model] = {}
        pred = arrays[f"{model}_all_start"]
        for split in ("train", "test"):
            mask = arrays["splits"] == split
            values = []
            pair_mse = []
            for i, (s, t) in enumerate(pairs):
                mse = float(np.mean((pred[mask, i] - direct[mask, t]) ** 2))
                baseline = float(np.sqrt(np.mean((direct[mask, s] - direct[mask, t]) ** 2)))
                constant = float(np.sqrt(np.mean((means[t] - direct[mask, t]) ** 2)))
                rmse = float(np.sqrt(mse))
                pair_mse.append(mse)
                values.append({"start": int(s), "target": int(t), "rmse": rmse,
                               "nrmse": rmse / float(stds[t]), "identity_rmse": baseline,
                               "identity_nrmse": baseline / float(stds[t]), "train_target_std": float(stds[t]),
                               "constant_train_mean_rmse": constant,
                               "constant_train_mean_nrmse": constant / float(stds[t])})
            pair_mse = np.asarray(pair_mse)
            equal = 10. / 3 * sum(float(pair_mse[pairs[:, 1] == t].mean()) for t in range(1, 4))
            macro = 10. / 6 * sum((s + 1) * float(pair_mse[pairs[:, 0] == s].mean()) for s in range(3))
            result["models"][model][split] = {"physical_chains": int(mask.sum()), "pairs": values,
                "equal_target_objective": equal, "macro_source_objective": macro}
    return result


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    choices = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
               "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    font = next((Path(item) for item in choices if Path(item).is_file()), None)
    if font is None:
        raise FileNotFoundError("An explicit Chinese font is required")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 160})
    return plt


def _save(fig, path, plt):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _flow(arrays, path, plt):
    grid = arrays["flow_grid"][:, 0]
    training = arrays["direct_g"][arrays["splits"] == "train", :3, 0].ravel()
    low, high = float(training.min()), float(training.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    left, right = axes
    left.plot(grid, grid, "k--", lw=1, label="恒等映射")
    for model in MODELS:
        left.plot(grid, arrays[f"{model}_flow_next"][:, 0], color=COLORS[model], lw=1.8, label=DISPLAY[model])
    for lo, hi in ((float(grid.min()), low), (high, float(grid.max()))):
        if hi > lo:
            left.axvspan(lo, hi, color="#dddddd", alpha=.6)
    left.plot(training, np.zeros_like(training), "|", color="#222222", alpha=.3,
              transform=left.get_xaxis_transform(), label="训练输入位置")
    left.set(xlabel="输入 g", ylabel="输出 R(g)", title="完整离散映射；灰区为训练输入包络外")
    left.legend(fontsize=8)
    left.grid(alpha=.2)
    indices = np.unique([int(np.argmin(abs(grid - item))) for item in np.linspace(low, high, 17)])
    starts = grid[indices]
    all_x = [starts]
    for row, model in enumerate(MODELS):
        ends = arrays[f"{model}_flow_next"][indices, 0]
        all_x.append(ends)
        right.quiver(starts, np.full_like(starts, row), ends - starts, np.zeros_like(starts),
                     angles="xy", scale_units="xy", scale=1, color=COLORS[model], width=.004,
                     headwidth=4, headlength=5, headaxislength=4.5, minlength=0)
        right.scatter(starts, np.full_like(starts, row), s=8, color=COLORS[model], alpha=.7)
    right.plot(training, np.full_like(training, -.55), "|", color="black", alpha=.25)
    minimum, maximum = min(float(item.min()) for item in all_x), max(float(item.max()) for item in all_x)
    pad = max((maximum - minimum) * .06, 1e-6)
    right.set(xlim=(minimum - pad, maximum + pad), ylim=(-.8, 2.5), xlabel="坐标 g（箭头终点为 R(g)）",
              yticks=range(3), yticklabels=[DISPLAY[name] for name in MODELS], title="实际位移 R(g)−g：未归一化、未裁切")
    right.grid(axis="x", alpha=.2)
    fig.suptitle("冻结同一 Q/C 的 R-only 对照；不搜索不动点或临界指数")
    _save(fig, path, plt)


def _loss(history, computed, path, plt):
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    tiny = np.finfo(float).tiny
    for arm in ARMS:
        own = [max(float(row["arms"][arm]["objective"]), tiny) for row in history]
        common = [max(float(row["arms"][arm]["equal_target_objective"]), tiny) for row in history]
        axes[0].plot(steps, own, color=COLORS[arm], label=DISPLAY[arm])
        axes[1].plot(steps, common, color=COLORS[arm], label=DISPLAY[arm])
    source = computed["models"]["source"]["train"]["equal_target_objective"]
    axes[1].axhline(max(source, tiny), color=COLORS["source"], ls="--", label="源 R 的固定缓存误差")
    for ax in axes:
        ax.set(xlabel="额外 R 更新次数", ylabel="固定训练缓存上的损失", yscale="log")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_title("各自优化目标（A/B 权重定义不同）")
    axes[1].set_title("共同评价：原等终点全起点闭环目标")
    fig.suptitle("Q/C 零更新；仅 R fresh Adam；零损失绘图保护不改变原始数值")
    _save(fig, path, plt)


def _condition_mean_sd(values, arrays):
    groups = []
    for cell in np.unique(arrays["cell_ids"]):
        mask = arrays["cell_ids"] == cell
        groups.append((float(arrays["labels"][mask, 0][0]), str(arrays["splits"][mask][0]),
                       values[mask].mean(axis=0), values[mask].std(axis=0, ddof=1), mask))
    return sorted(groups, key=lambda item: item[0])


def _coupling(arrays, path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes.flat[0]
    direct = arrays["direct_g"][:, :, 0]
    groups = _condition_mean_sd(direct, arrays)
    kappas = np.asarray([item[0] for item in groups])
    means = np.asarray([item[2] for item in groups])
    stds = np.asarray([item[3] for item in groups])
    scale_colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
    for k in range(4):
        ax.errorbar(kappas, means[:, k], yerr=stds[:, k], marker="o", color=scale_colors[k],
                    ms=3, lw=1, capsize=2, label=f"k={k}，边长 {128 // 2 ** k}")
        for kappa, split, _, _, mask in groups:
            ax.scatter(np.full(mask.sum(), kappa), direct[mask, k], color=scale_colors[k],
                       marker="o" if split == "train" else "x", s=12, alpha=.45)
    ax.set(title="固定 Q 的直接编码（四条链均值±链间标准差）", xlabel="初始耦合 κ", ylabel=r"直接编码 $g_k$")
    ax.legend(fontsize=8)
    for k, ax in enumerate(list(axes.flat)[1:]):
        target = direct[:, k + 1]
        true_groups = _condition_mean_sd(target, arrays)
        ax.errorbar(kappas, [row[2] for row in true_groups], yerr=[row[3] for row in true_groups],
                    color="black", lw=1.6, marker="o", ms=3, capsize=2, label="直接目标 Q 编码")
        for model in MODELS:
            predicted = arrays[f"{model}_one_step"][:, k, 0]
            model_groups = _condition_mean_sd(predicted, arrays)
            ax.plot(kappas, [row[2] for row in model_groups], color=COLORS[model], lw=1.3, label=DISPLAY[model])
            for kappa, split, _, _, mask in model_groups:
                ax.scatter(np.full(mask.sum(), kappa), predicted[mask], color=COLORS[model],
                           marker="o" if split == "train" else "x", s=12, alpha=.45)
        ax.set(title=f"单步 {128 // 2 ** k}→{128 // 2 ** (k + 1)}：" + r"$R(g_k)$ 对照 $g_{k+1}$",
               xlabel="初始耦合 κ", ylabel="目标尺度坐标 g")
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("全部 15 个既有条件；点线仅连接观察值；圆点为训练链、叉号为留出链；不插值 κc")
    _save(fig, path, plt)


def _errors(computed, path, plt):
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    positions = np.arange(6)
    width = .15
    for j, model in enumerate(MODELS):
        values = computed["models"][model]["test"]["pairs"]
        ax.bar(positions + (j - 2) * width, [row["nrmse"] for row in values], width,
               color=COLORS[model], label=DISPLAY[model])
    baseline = computed["models"]["source"]["test"]["pairs"]
    ax.bar(positions + width, [row["identity_nrmse"] for row in baseline], width,
           color="#BBBBBB", hatch="//", label="恒等预测")
    ax.bar(positions + 2 * width, [row["constant_train_mean_nrmse"] for row in baseline], width,
           color="white", edgecolor="#444444", hatch="..", label="固定训练目标均值预测")
    ax.set(xticks=positions, xticklabels=[f"k={s}→{t}" for s, t in EXPECTED_PAIRS],
           xlabel="全起点预测对", ylabel="留出归一化均方根误差",
           title="同坐标留出误差：固定原 TRAIN 目标标准差；越低越好")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    _save(fig, path, plt)


def render_report(output):
    """Write a fresh public report and return sorted relative JSON/PNG paths."""
    root = Path(output).resolve()
    plan, history, metrics, summary, arrays, pins = load_public_run(root)
    destination = root / "report"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite report: {destination}")
    computed = comparison_metrics(arrays)
    plt = _plot_setup()
    destination.mkdir()
    plots = ("R_flow_comparison.png", "loss_curves.png", "coupling_by_scale.png", "error_comparison.png")
    _flow(arrays, destination / plots[0], plt)
    _loss(history, computed, destination / plots[1], plt)
    _coupling(arrays, destination / plots[2], plt)
    _errors(computed, destination / plots[3], plt)
    if pins != {name: _sha(root / name) for name in INPUT_FILES}:
        raise ValueError("Public input changed while rendering")
    result = {"status": "complete", "report_kind": "phi4_frozen_QC_macro_source_R_only_ablation_v1",
              "source_sha256": pins, "labels_used_in_training": False,
              "fixed_point_search_performed": False, "critical_kappa": None, "thermal_nu": None,
              "source_qcr_step": 3000, "additional_R_updates": history[-1]["step"],
              "coordinate_comparison": "same frozen Q/C and paired physical-chain cache",
              "macroscopic_source_weights": [1 / 6, 2 / 6, 3 / 6],
              "per_pair_weight_equal_target": [1 / 3, 1 / 6, 1 / 6, 1 / 9, 1 / 9, 1 / 9],
              "per_pair_weight_macro_source": [1 / 18, 1 / 18, 1 / 6, 1 / 18, 1 / 6, 1 / 2],
              "history_metric_semantics": "full fixed TRAIN-cache evaluation after recorded R update",
              "comparison": computed, "limitations": LIMITATIONS, "plots": list(plots)}
    (destination / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return sorted([f"report/{name}" for name in plots] + ["report/comparison.json"])


def main():
    parser = argparse.ArgumentParser(description="冻结 Q/C 的宏观加权 R-only 离线中文报告")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"public_files": render_report(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
