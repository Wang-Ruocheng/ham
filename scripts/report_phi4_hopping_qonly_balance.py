"""Model-free Chinese report for the explicit frozen-C/R Q-only ablation.

Only already-saved JSON/NPZ statistics are read.  This module does not load a
checkpoint, evaluate a neural network, sample fields, select an endpoint, or
search for a fixed point.  A new ``report`` directory is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_FILES = ("plan.json", "history.json", "metrics.json", "summary.json", "evaluation_arrays.npz")
MODELS = ("source", "A_absolute", "B_relative")
ARMS = MODELS[1:]
DISPLAY = {"source": "源 Q（QCR 3000 步）", "A_absolute": "A：原绝对重构", "B_relative": "B：逐尺度相对重构"}
COLORS = {"source": "#777777", "A_absolute": "#0072B2", "B_relative": "#D55E00"}
LIMITATIONS = [
    "本实验为用户明确请求的 Q-only 诊断：仅更新 Q 与其共享仿射解码器，C/R 完全冻结；不修改当前默认 QCR 配方。",
    "两臂从同一源 Q 出发，fresh Adam、相同配对批次；源 QCR 3000 步与本次额外 Q 更新分开报告，不能称为完整 QCR 3300 步。",
    "仅使用既有物理数据与同一固定 C 轨迹；训练批次每条物理链抽取 32 个构型，不是原全 QCR 随机 64 构型训练的精确续训。",
    "B 仅对全部 625 个统计分量的每尺度总重构误差作标量加权；不逐分量白化、不逐尺度重定义 g、不使用耦合标签训练。",
    "相对权重只由 TRAIN 统计及初始化误差确定后冻结，并保留噪声下限及最大权重保护；四个相关的 32 构型块只给出启发式噪声下限，不能认证采样误差或热化。",
    "所有绝对 RMSE 与 NRMSE 使用同一目标 p，NRMSE 固定除以原 TRAIN 每尺度目标标准差；不是按各臂重新归一化。",
    "κ、λ、h 只供生成和事后描述；κ 与 g 的 Spearman 相关不用于选模型、调权或挑终点。g 更分散或更单调本身不是完整物理正确性的证据。",
    "每个条件四条物理链；图中误差带为链间样本标准差而非置信区间。链内构型不保证独立，原有 75 项采样质量警告继续保留。",
    "各臂 Q 训练后可改变潜变量坐标；不能将冻结的旧 R 与新 Q 拼接后宣称得到新的 RG 流。R 图只展示原 Q 坐标下的原映射。",
    "单物理数据与单训练种子、固定 C 单次轨迹的短程诊断；不推断不动点、相变耦合、热临界指数或宏观极限。",
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
        raise ValueError(f"Nonfinite or nonnumeric saved array: {name}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Incorrect saved shape for {name}: {value.shape}, expected {shape}")
    return value.astype(np.float64, copy=False)


def load_public_run(output):
    """Load and validate the complete, fixed saved-statistics contract."""
    root = Path(output).resolve()
    pins = {name: _sha(root / name) for name in INPUT_FILES}
    plan, history, metrics, summary = [json.loads((root / name).read_text()) for name in INPUT_FILES[:4]]
    if not all(isinstance(item, dict) for item in (plan, metrics, summary)):
        raise ValueError("Plan, metrics, and summary must be objects")
    if summary.get("status") != "complete":
        raise ValueError("Q-only evaluation must finish before reporting")
    if any(summary.get(key) is not None for key in ("critical_kappa", "thermal_nu")):
        raise ValueError("No new critical parameters are allowed")
    if isinstance(history, dict):
        history = history.get("rows", history.get("history"))
    if not isinstance(history, list) or not history:
        raise ValueError("Missing TRAIN history")
    steps = []
    for row in history:
        if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 0:
            raise ValueError("History requires nonnegative integer steps")
        steps.append(row["step"])
        for arm in ARMS:
            item = row.get("arms", {}).get(arm, {})
            for key in ("q_total", "reconstruction_raw", "reconstruction_relative"):
                value = item.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError(f"Missing or invalid TRAIN history: {arm}/{key}")
            _numeric(item.get("reconstruction_by_scale"), f"history/{arm}/reconstruction_by_scale", (4,))
    if steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError("History must start at zero and increase strictly")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["target_features"] = _numeric(arrays["target_features"], "target_features", (60, 4, 625))
    arrays["labels"] = _numeric(arrays["labels"], "labels", (60, 3))
    labels = arrays["labels"]
    if (labels[:, 0] <= 0).any() or not np.all(labels[:, 1] == 1) or not np.all(labels[:, 2] == 0):
        raise ValueError("Expected positive κ, fixed λ=1 and h=0")
    splits = arrays["splits"]
    if splits.shape != (60,) or splits.dtype.kind not in "US":
        raise ValueError("Invalid split vector")
    arrays["splits"] = splits.astype(str)
    if set(arrays["splits"].tolist()) != {"train", "test"}:
        raise ValueError("Both lowercase train/test splits are required")
    train = arrays["splits"] == "train"
    if train.sum() != 40:
        raise ValueError("Expected 40 TRAIN and 20 TEST chains")
    cells = arrays["cell_ids"]
    if cells.shape != (60,) or cells.dtype.kind not in "iuUS" or len(np.unique(cells)) != 15:
        raise ValueError("Expected 15 saved physical conditions")
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() != 4 or len(np.unique(arrays["splits"][mask])) != 1 or not np.all(labels[mask] == labels[mask][0]):
            raise ValueError("Each condition must retain four physical chains, one split, and one label")
    if set(map(tuple, labels[train])) & set(map(tuple, labels[~train])):
        raise ValueError("TRAIN/TEST physical conditions overlap")
    if len(set(map(tuple, labels))) != 15:
        raise ValueError("The 15 conditions must have distinct physical labels")
    mean = _numeric(arrays["train_target_mean"], "train_target_mean", (4, 625))
    std = _numeric(arrays["train_target_std"], "train_target_std", (4,))
    if (std < 0).any() or not np.allclose(mean, arrays["target_features"][train].mean(axis=0), rtol=1e-8, atol=1e-11):
        raise ValueError("Target normalization must be fixed from original TRAIN only")
    expected_std = np.sqrt(arrays["target_features"][train].var(axis=0, ddof=0).mean(axis=1))
    if not np.allclose(std, expected_std, rtol=1e-8, atol=1e-11):
        raise ValueError("Saved TRAIN std must use all 625 target components, ddof=0")
    arrays["train_target_mean"], arrays["train_target_std"] = mean, std
    for key in ("metric_denominators", "relative_weights"):
        arrays[key] = _numeric(arrays[key], key, (4,))
        if (arrays[key] <= 0).any():
            raise ValueError(f"Positive frozen scalar scale weights required: {key}")
    for model in MODELS:
        for suffix in ("direct_g", "direct_full_g", "g_first", "g_second"):
            key = f"{model}_{suffix}"
            arrays[key] = _numeric(arrays[key], key, (60, 4, 1))
        key = f"{model}_decoded"
        arrays[key] = _numeric(arrays[key], key, (60, 4, 625))
        if not np.allclose(arrays[f"{model}_direct_g"], .5 * (arrays[f"{model}_g_first"] + arrays[f"{model}_g_second"]), rtol=1e-9, atol=1e-11):
            raise ValueError("Direct g must be the mean of the two half-ensemble encodings")
    for key in ("source_flow_grid", "source_flow_next"):
        arrays[key] = _numeric(arrays[key], key)
        if arrays[key].ndim == 1:
            arrays[key] = arrays[key][:, None]
    grid = arrays["source_flow_grid"]
    if grid.ndim != 2 or grid.shape[1] != 1 or len(grid) < 17 or not (np.diff(grid[:, 0]) > 0).all():
        raise ValueError("Original scalar R grid must be strictly ordered")
    if arrays["source_flow_next"].shape != grid.shape:
        raise ValueError("Original R reference shape mismatch")
    if pins != {name: _sha(root / name) for name in INPUT_FILES}:
        raise ValueError("Inputs changed during loading")
    return plan, history, metrics, summary, arrays, pins


def _rank(values):
    """Average ranks including ties, avoiding a scipy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = .5 * (start + end - 1)
        start = end
    return result


def _spearman(x, y):
    x, y = _rank(x), _rank(y)
    x, y = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if den <= np.finfo(float).eps else float(np.dot(x, y) / den)


def _ratio(numerator, denominator):
    """Undefined normalization is reported as null, never a fabricated zero."""
    return None if denominator <= np.finfo(float).tiny else float(numerator / denominator)


def _groups(values, arrays):
    result = []
    for cell in np.unique(arrays["cell_ids"]):
        mask = arrays["cell_ids"] == cell
        result.append({"cell_id": str(cell), "kappa": float(arrays["labels"][mask, 0][0]),
                       "split": str(arrays["splits"][mask][0]), "mean": values[mask].mean(axis=0),
                       "chain_sd": values[mask].std(axis=0, ddof=1)})
    return sorted(result, key=lambda item: item["kappa"])


def _first_pc(values):
    """One leading component from the small sample Gram matrix, once per scale."""
    centered = values - values.mean(axis=0, keepdims=True)
    energy = float(np.square(centered).sum())
    if energy <= np.finfo(float).tiny:
        return None, None
    eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
    first = centered.T @ vectors[:, -1]
    norm = float(np.linalg.norm(first))
    return (None, None) if norm <= np.finfo(float).tiny else (first / norm, max(0., float(eigenvalues[-1])) / energy)


def comparison_metrics(arrays):
    """Compute label-free reconstruction plus explicitly post-hoc descriptions."""
    target = arrays["target_features"]
    train = arrays["splits"] == "train"
    std = arrays["train_target_std"]
    mean = arrays["train_target_mean"]
    weights = arrays["relative_weights"]
    all_pc = _first_pc(target[train].reshape(-1, 625))
    scale_pc = [_first_pc(target[train, k]) for k in range(4)]
    result = {"models": {}, "constant_train_mean": {}, "train_target_std": std.tolist(),
              "metric_denominators": arrays["metric_denominators"].tolist(), "relative_weights": weights.tolist(),
              "normalization": "fixed original TRAIN sqrt(mean component variance), ddof=0",
              "reconstruction_objective_semantics": "mean over physical chains and scales of SUM625 squared errors, matching original _square_l2; RMSE instead averages the 625 components",
              "geometry": {"global_train_pc1_fraction": all_pc[1],
                           "per_scale_train_pc1_fraction": [item[1] for item in scale_pc]},
              "posthoc_spearman_semantics": "condition mean g versus κ; descriptive only, never used for training or endpoint selection"}
    for split in ("train", "test"):
        mask = arrays["splits"] == split
        baseline_mse = np.square(target[mask] - mean).mean(axis=(0, 2))
        result["constant_train_mean"][split] = [{"scale": k, "rmse": float(np.sqrt(baseline_mse[k])),
            "nrmse": _ratio(np.sqrt(baseline_mse[k]), std[k])} for k in range(4)]
    for model in MODELS:
        g = arrays[f"{model}_direct_g"][:, :, 0]
        decoded = arrays[f"{model}_decoded"]
        first = arrays[f"{model}_g_first"][:, :, 0]
        second = arrays[f"{model}_g_second"][:, :, 0]
        full = arrays[f"{model}_direct_full_g"][:, :, 0]
        groups = _groups(g, arrays)
        output = {"condition_statistics": [{**{key: value for key, value in row.items() if key not in ("mean", "chain_sd")},
                                            "mean_by_scale": row["mean"].tolist(), "chain_sd_by_scale": row["chain_sd"].tolist()} for row in groups]}
        for split in ("train", "test"):
            mask = arrays["splits"] == split
            mse = np.square(decoded[mask] - target[mask]).mean(axis=(0, 2))
            selected_groups = [row for row in groups if row["split"] == split]
            kappas = [row["kappa"] for row in selected_groups]
            output[split] = {"physical_chains": int(mask.sum()), "reconstruction_raw": float(625 * mse.mean()),
                "reconstruction_relative": float(625 * np.mean(weights * mse)),
                "reconstruction_component_mse": float(mse.mean()),
                "by_scale": [{"scale": k, "lattice_size": 128 // 2 ** k,
                    "rmse": float(np.sqrt(mse[k])), "nrmse": _ratio(np.sqrt(mse[k]), std[k]),
                    "g_std": float(g[mask, k].std(ddof=0)),
                    "between_condition_mean_std": float(np.std([row["mean"][k] for row in selected_groups], ddof=0)),
                    "half_pair_rmse": float(np.sqrt(np.mean(np.square(first[mask, k] - second[mask, k])))),
                    "half_mean_vs_full_rmse": float(np.sqrt(np.mean(np.square(g[mask, k] - full[mask, k])))),
                    "posthoc_kappa_spearman": _spearman(kappas, [row["mean"][k] for row in selected_groups])}
                    for k in range(4)]}
        x = g.reshape(-1)
        y = decoded.reshape(-1, 625)
        xc = x - x.mean()
        variance = float(np.dot(xc, xc))
        identifiable = bool(np.ptp(x) > 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(x)))))
        if not identifiable:
            direction = np.zeros(625)
            fitted = np.broadcast_to(y.mean(axis=0), y.shape)
        else:
            direction = xc @ (y - y.mean(axis=0)) / variance
            fitted = y.mean(axis=0) + xc[:, None] * direction
        norm = float(np.linalg.norm(direction))
        cosine = lambda pc: None if norm <= np.finfo(float).tiny or pc is None else float(abs(np.dot(direction / norm, pc)))
        output["decoder_geometry"] = {"direction_identifiable_from_saved_g": identifiable,
            "direction_norm": norm if identifiable else None,
            "affine_fit_max_abs_residual": float(np.max(np.abs(fitted - y))),
            "abs_cos_global_train_pc1": cosine(all_pc[0]),
            "abs_cos_per_scale_train_pc1": [cosine(item[0]) for item in scale_pc],
            "warning": None if identifiable else "All saved g values are constant; decoder direction cannot be recovered from these arrays",
            "method": "recover the shared affine direction from saved decoded/g arrays; PC1 from TRAIN target statistics only"}
        result["models"][model] = output
    result["normalization_warning"] = None if np.all(std > 0) else "A TRAIN scale has zero target variance; its NRMSE is undefined and saved as null"
    return result


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/PingFang.ttc",
                  "/System/Library/Fonts/STHeiti Medium.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    font = next((Path(item) for item in candidates if Path(item).is_file()), None)
    if font is None:
        raise FileNotFoundError("An explicit Chinese font is required")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 160})
    return plt


def _save(fig, path, plt):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _q_maps(arrays, computed, path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        for model in MODELS:
            groups = _groups(arrays[f"{model}_direct_g"][:, :, 0], arrays)
            kappas = np.asarray([row["kappa"] for row in groups])
            means = np.asarray([row["mean"][k] for row in groups])
            errors = np.asarray([row["chain_sd"][k] for row in groups])
            std = computed["models"][model]["train"]["by_scale"][k]["g_std"]
            ax.plot(kappas, means, color=COLORS[model], lw=1.4, alpha=.85, label=f"{DISPLAY[model]}；训练标准差 {std:.3g}")
            ax.fill_between(kappas, means - errors, means + errors, color=COLORS[model], alpha=.1)
            for split, marker in (("train", "o"), ("test", "x")):
                mask = np.asarray([row["split"] == split for row in groups])
                ax.scatter(kappas[mask], means[mask], marker=marker, color=COLORS[model], s=25, zorder=3)
        ax.set(xlabel="初始耦合 κ（仅事后评价）", ylabel="直接 Q 编码 g", title=f"k={k}，边长 {128 // 2 ** k}；各子图纵轴独立")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("冻结同一 C 的 Q-only 对照：四条链均值±链间标准差（非置信区间）\n圆点：训练条件；叉号：留出条件；连线仅辅助观察，不估计相变位置")
    _save(fig, path, plt)


def _reconstruction(computed, path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    positions, width = np.arange(4), .105
    for ax, key, title in zip(axes, ("rmse", "nrmse"), ("全部 625 个统计分量的绝对 RMSE", "固定原训练目标标准差的 NRMSE")):
        for index, model in enumerate(MODELS):
            for split_index, split in enumerate(("train", "test")):
                offset = (2 * index + split_index - 3.5) * width
                values = np.asarray([row[key] for row in computed["models"][model][split]["by_scale"]], dtype=float)
                ax.bar(positions + offset, values, width, color=COLORS[model], alpha=.85 if split_index == 0 else .5,
                       hatch=None if split_index == 0 else "//", label=f"{DISPLAY[model]}：{'训练' if split == 'train' else '留出'}")
        for index, split in enumerate(("train", "test")):
            ax.bar(positions + (index + 2.5) * width, np.asarray([row[key] for row in computed["constant_train_mean"][split]], dtype=float), width,
                   facecolor="white", edgecolor="#333333", hatch=".." if index == 0 else "xx",
                   label=f"训练均值常数基准：{'训练' if split == 'train' else '留出'}")
        ax.set(xticks=positions, xticklabels=[f"k={k}\n边长 {128 // 2 ** k}" for k in range(4)],
               ylabel=key.upper(), title=title)
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("重构评价使用共同目标 p；未按各臂重定义坐标或归一化；越低越好")
    _save(fig, path, plt)


def _loss(history, path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    steps = [row["step"] for row in history]
    fields = (("q_total", "各臂自身 Q 总目标（定义不同，不直接排名）"),
              ("reconstruction_raw", "共同口径：原绝对重构误差"),
              ("reconstruction_relative", "共同口径：冻结相对权重重构误差"))
    for ax, (key, title) in zip(list(axes.flat)[:3], fields):
        for arm in ARMS:
            values = [max(float(row["arms"][arm][key]), np.finfo(float).tiny) for row in history]
            ax.plot(steps, values, color=COLORS[arm], label=DISPLAY[arm])
        ax.set(title=title, xlabel="额外 Q 更新次数", ylabel="固定训练评价缓存上的损失", yscale="log")
        ax.legend(fontsize=8)
    ax = axes.flat[3]
    scale_colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
    for k in range(4):
        for arm, style in zip(ARMS, ("--", "-")):
            values = [max(float(row["arms"][arm]["reconstruction_by_scale"][k]), np.finfo(float).tiny) for row in history]
            ax.plot(steps, values, color=scale_colors[k], ls=style, label=f"k={k}，{'A' if arm == ARMS[0] else 'B'}")
    ax.set(title="各尺度原绝对重构误差：虚线 A，实线 B", xlabel="额外 Q 更新次数", ylabel="尺度重构误差", yscale="log")
    ax.legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("C/R 零更新；历史仅训练集；终点固定，不用留出误差选步数")
    _save(fig, path, plt)


def _frozen_r(arrays, path, plt):
    grid, next_g = arrays["source_flow_grid"][:, 0], arrays["source_flow_next"][:, 0]
    original_inputs = arrays["source_direct_g"][arrays["splits"] == "train", :3, 0].ravel()
    low, high = float(original_inputs.min()), float(original_inputs.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    left, right = axes
    left.plot(grid, grid, "k--", lw=1, label="恒等映射")
    left.plot(grid, next_g, color=COLORS["source"], lw=1.8, label="原 Q 坐标下的原 R")
    for lo, hi in ((float(grid.min()), low), (high, float(grid.max()))):
        if hi > lo:
            left.axvspan(lo, hi, color="#dddddd", alpha=.6)
    left.plot(original_inputs, np.zeros_like(original_inputs), "|", color="black", alpha=.3,
              transform=left.get_xaxis_transform(), label="仅原 Q 的训练输入位置")
    left.set(xlabel="原 Q 坐标 g", ylabel="原 R(g)", title="原映射参考；灰区为原训练输入包络外")
    left.legend(fontsize=8)
    indices = np.unique([int(np.argmin(abs(grid - value))) for value in np.linspace(low, high, 19)])
    starts, ends = grid[indices], next_g[indices]
    right.quiver(starts, np.zeros_like(starts), ends - starts, np.zeros_like(starts), angles="xy", scale_units="xy", scale=1,
                 color=COLORS["source"], width=.004, headwidth=4, headlength=5, headaxislength=4.5, minlength=0)
    right.scatter(starts, np.zeros_like(starts), color=COLORS["source"], s=10)
    right.plot(original_inputs, np.full_like(original_inputs, -.18), "|", color="black", alpha=.3)
    lo, hi = min(float(starts.min()), float(ends.min())), max(float(starts.max()), float(ends.max()))
    pad = max((hi - lo) * .06, 1e-6)
    right.set(xlim=(lo - pad, hi + pad), ylim=(-.3, .3), yticks=[], xlabel="原 Q 坐标（箭头终点为原 R(g)）",
              title="实际位移 R(g)−g；不归一化箭头长度")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("R 完全冻结，未针对新 Q 适配，无新 RG 结论\n仅复制源模型保存的原 Q／R 参考，不将旧 R 与新 Q 拼接，不搜索不动点")
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
    plots = ("Q_g_kappa_by_scale.png", "reconstruction_comparison.png", "loss_curves.png", "R_frozen_reference.png")
    _q_maps(arrays, computed, destination / plots[0], plt)
    _reconstruction(computed, destination / plots[1], plt)
    _loss(history, destination / plots[2], plt)
    _frozen_r(arrays, destination / plots[3], plt)
    if pins != {name: _sha(root / name) for name in INPUT_FILES}:
        raise ValueError("Public inputs changed while reporting")
    result = {"status": "complete", "report_kind": "phi4_frozen_CR_Q_only_scale_balance_v1", "source_sha256": pins,
              "source_qcr_step": 3000, "additional_Q_updates_per_arm": history[-1]["step"],
              "C_updates": 0, "R_updates": 0, "labels_used_in_training": False,
              "fixed_point_search_performed": False, "critical_kappa": None, "thermal_nu": None,
              "old_R_valid_for_new_Q": False, "comparison": computed, "limitations": LIMITATIONS, "plots": list(plots)}
    (destination / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return sorted([f"report/{name}" for name in plots] + ["report/comparison.json"])


def main():
    parser = argparse.ArgumentParser(description="冻结 C/R 的 Q-only 尺度平衡离线报告")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"public_files": render_report(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
