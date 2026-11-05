"""只绘制已保存的冻结 Q/C、R-only 诊断结果，不加载模型或搜索根。

The caller supplies public arrays and endpoint metrics.  Coupling labels are
used only for the final diagnostic scatter plots, never to fit a map here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _plotting


ARMS = ("source", "raw96", "scaled96", "scaled256", "scale_specific96")
AUTONOMOUS = ARMS[:4]
PATHS = ("0_to_1", "1_to_2", "0_to_2")
PATH_LABELS = ("36→12", "12→4", "36→4")
SCALE_COLORS = ("#2f6ca4", "#d98632", "#8a5ba6")
ARM_COLORS = ("#7d7d7d", "#2878b5", "#29a078", "#c265a1", "#df8f27")


def _finite_array(arrays, name, shape=None):
    value = np.asarray(arrays[name])
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite numeric array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Unexpected {name} shape: {value.shape}, expected {shape}")
    return value.astype(np.float64, copy=False)


def _validate(arrays, metrics, labels):
    for split in ("train", "test"):
        direct = _finite_array(arrays, f"{split}_direct")
        if direct.ndim != 3 or direct.shape[1:] != (3, 2) or not len(direct):
            raise ValueError(f"{split}_direct must be nonempty [ensemble, 3, 2]")
        _finite_array(arrays, f"{split}_labels", (len(direct), 3))
        ids = np.asarray(arrays[f"{split}_cell_ids"])
        if ids.shape != (len(direct),):
            raise ValueError("Physical condition IDs must match ensemble rows")
        for arm in ARMS:
            _finite_array(arrays, f"{arm}_{split}_one_step", (len(direct), 2, 2))
            rollout = _finite_array(arrays, f"{arm}_{split}_rollout", (len(direct), 3, 2))
            if not np.allclose(rollout[:, 0], direct[:, 0], atol=1e-12, rtol=1e-10):
                raise ValueError("Rollout must start at the fixed microscopic encoding")
    if set(np.asarray(arrays["train_cell_ids"]).tolist()) & set(
        np.asarray(arrays["test_cell_ids"]).tolist()
    ):
        raise ValueError("TRAIN and TEST condition IDs must be disjoint")
    grid = _finite_array(arrays, "flow_g")
    if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid):
        raise ValueError("Flow grid must be nonempty [point, 2]")
    mask = np.asarray(arrays["flow_in_train_hull"])
    if mask.shape != (len(grid),):
        raise ValueError("TRAIN hull mask must match the flow grid")
    if mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1]).all():
        raise ValueError("TRAIN hull mask must be Boolean")
    for arm in AUTONOMOUS:
        _finite_array(arrays, f"{arm}_flow_R", grid.shape)
    for scale in (0, 1):
        _finite_array(arrays, f"scale_specific96_flow_R{scale}", grid.shape)
    for arm in ARMS:
        if arm not in labels or not isinstance(labels[arm], str):
            raise ValueError(f"Missing display label for {arm}")
        found = set()
        for row in metrics[arm]:
            key = (row["split"], row["path"])
            if key in found:
                raise ValueError(f"Duplicate metric {arm}: {key}")
            found.add(key)
            for field in ("rmse", "nrmse", "relative_displacement_error", "identity_rmse"):
                value = row[field]
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {arm} {key} {field}: {value}")
        if found != {(split, path) for split in ("train", "test") for path in PATHS}:
            raise ValueError(f"Incomplete endpoint metrics for {arm}")


def _sparse_flow_indices(grid, mask):
    """Sparse lattice selection without changing the magnitude of any arrow."""
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    if len(xs) * len(ys) == len(grid):
        xstep, ystep = max(1, int(np.ceil(len(xs) / 12))), max(1, int(np.ceil(len(ys) / 12)))
        sparse = np.isin(grid[:, 0], xs[::xstep]) & np.isin(grid[:, 1], ys[::ystep])
        selected = np.flatnonzero(mask & sparse)
    else:
        selected = np.flatnonzero(mask)
        selected = selected[::max(1, int(np.ceil(len(selected) / 140)))]
    if not len(selected) and mask.any():
        selected = np.flatnonzero(mask)[::max(1, int(np.ceil(mask.sum() / 140)))]
    return selected


def _save(fig, output, name, plt):
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _limits(values):
    lower, upper = np.min(values, axis=0), np.max(values, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    return lower - 0.06 * span, upper + 0.06 * span


def _plot_flow(output, arrays, labels, plt):
    grid = np.asarray(arrays["flow_g"], dtype=np.float64)
    mask = np.asarray(arrays["flow_in_train_hull"], dtype=bool)
    indices = _sparse_flow_indices(grid, mask)
    train, test = [np.asarray(arrays[f"{split}_direct"]) for split in ("train", "test")]
    panels = [(labels[arm], f"{arm}_flow_R") for arm in AUTONOMOUS]
    panels += [("尺度条件化对照：R₀（36→12，非自治）", "scale_specific96_flow_R0"),
               ("尺度条件化对照：R₁（12→4，非自治）", "scale_specific96_flow_R1")]
    # All six panels share raw coordinates and limits. Include displayed arrow
    # endpoints so large residuals are not silently clipped or rescaled away.
    extent = [train.reshape(-1, 2), test.reshape(-1, 2), grid[mask]]
    extent += [np.asarray(arrays[key])[indices] for _, key in panels]
    lower, upper = _limits(np.concatenate(extent))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for panel, (ax, (title, key)) in enumerate(zip(axes.flat, panels)):
        for scale, color in enumerate(SCALE_COLORS):
            ax.scatter(train[:, scale, 0], train[:, scale, 1], s=14, color=color, alpha=.36,
                       linewidths=0, label=f"训练点 k={scale}" if panel == 0 else None, zorder=2)
            ax.scatter(test[:, scale, 0], test[:, scale, 1], s=12, color=color, alpha=.55,
                       marker="x", linewidths=.65, zorder=3)
        mapped = np.asarray(arrays[key])
        displacement = mapped[indices] - grid[indices]
        ax.quiver(grid[indices, 0], grid[indices, 1], displacement[:, 0], displacement[:, 1],
                  angles="xy", scale_units="xy", scale=1, color="#243641", alpha=.8,
                  width=.0034, headwidth=3.7, headlength=4.8, zorder=4)
        ax.set(title=title, xlabel="原始坐标 g₁", ylabel="原始坐标 g₂",
               xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    axes.flat[0].legend(loc="best", fontsize=8, framealpha=.8)
    fig.suptitle("冻结 Q/C 后的离散流场：箭头严格为 R(g)−g", fontsize=15, y=.99)
    fig.text(.5, .025, "前四格为自治映射；后两格为不同尺度映射，不能合称一个自治 RG 流。\n"
             "圆点为训练系综、叉号为留出系综；仅在训练输入凸包内绘制稀疏箭头，不归一化箭长；未搜索或认证不动点。",
             ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(top=.92, bottom=.13, hspace=.31, wspace=.22)
    _save(fig, output, "R_vs_g.png", plt)


def _plot_loss(output, history, labels, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = (("loss", "R-only 闭合总目标"), ("e01", "一步平方误差：36→12"),
              ("e12", "一步平方误差：12→4"), ("e02", "两步平方误差：36→4"))
    for ax, (field, title) in zip(axes.flat, fields):
        plotted = False
        for arm, color in zip(ARMS, ARM_COLORS):
            rows = [row for row in history if arm in row.get("arms", {})
                    and field in row["arms"][arm]]
            if not rows:
                continue
            steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
            values = np.asarray([row["arms"][arm][field] for row in rows], dtype=np.float64)
            if not np.isfinite(steps).all() or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Nonfinite or negative {arm} loss history")
            if len(steps) > 1 and not np.all(np.diff(steps) > 0):
                raise ValueError(f"Non-increasing {arm} history steps")
            # Exact zeros are valid but not representable on a log axis; omit
            # those markers instead of silently assigning a positive value.
            ax.plot(steps, np.ma.masked_less_equal(values, 0), color=color,
                    label=labels[arm], linewidth=1.7)
            plotted = True
        ax.set(title=title, xlabel="额外 R 更新步数", ylabel="均方误差／加权目标", yscale="log")
        ax.grid(alpha=.2, which="both")
        if not plotted:
            ax.text(.5, .5, "未提供逐步记录", transform=ax.transAxes, ha="center")
    handles, legends = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("固定编码与固定目标上的 R-only 优化过程", fontsize=14)
    fig.tight_layout(rect=(0, .1, 1, .95))
    _save(fig, output, "loss.png", plt)


def _plot_metrics(output, metrics, labels, plt):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    columns = (("rmse", "绝对 RMSE"), ("nrmse", "原始 TRAIN 目标标准差 NRMSE"),
               ("relative_displacement_error", "误差／真实尺度位移 RMS"))
    positions = np.arange(3)
    width = .15
    for row_index, split in enumerate(("train", "test")):
        for col_index, (metric, title) in enumerate(columns):
            ax = axes[row_index, col_index]
            for arm_index, (arm, color) in enumerate(zip(ARMS, ARM_COLORS)):
                by_path = {row["path"]: row for row in metrics[arm] if row["split"] == split}
                values = [by_path[path][metric] for path in PATHS]
                ax.bar(positions + (arm_index - 2) * width, values, width,
                       color=color, label=labels[arm] if row_index == col_index == 0 else None)
            if metric == "rmse":
                source = {row["path"]: row for row in metrics["source"] if row["split"] == split}
                identity = [source[path]["identity_rmse"] for path in PATHS]
                ax.scatter(positions, identity, marker="_", s=250, color="#222222", linewidths=2,
                           label="恒等映射" if row_index == 0 else None, zorder=5)
            elif metric == "relative_displacement_error":
                ax.axhline(1, linestyle="--", linewidth=1, color="#444444")
            ax.set_xticks(positions, PATH_LABELS)
            ax.set(title=("训练集：" if split == "train" else "留出集：") + title,
                   xlabel="尺度转移", ylabel=title)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("同一冻结 Q/C、同一原始 g 坐标下的端点预测误差", fontsize=14)
    fig.tight_layout(rect=(0, .1, 1, .95))
    _save(fig, output, "metrics.png", plt)


def _plot_couplings(output, arrays, plt):
    train, test = [np.asarray(arrays[f"{split}_direct"]) for split in ("train", "test")]
    physical = {split: np.asarray(arrays[f"{split}_labels"]) for split in ("train", "test")}
    lower, upper = _limits(np.concatenate([train.reshape(-1, 2), test.reshape(-1, 2)]))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for parameter, name in enumerate(("r", "u")):
        values = np.concatenate([physical[split][:, parameter] for split in ("train", "test")])
        vmin, vmax = float(values.min()), float(values.max())
        for scale in range(3):
            ax = axes[parameter, scale]
            image = ax.scatter(train[:, scale, 0], train[:, scale, 1], c=physical["train"][:, parameter],
                               cmap="viridis" if name == "r" else "plasma", vmin=vmin, vmax=vmax,
                               s=23, alpha=.75, edgecolors="none", label="训练条件")
            ax.scatter(test[:, scale, 0], test[:, scale, 1], c=physical["test"][:, parameter],
                       cmap="viridis" if name == "r" else "plasma", vmin=vmin, vmax=vmax,
                       s=30, marker="x", linewidths=1.05, label="留出条件")
            ax.set(title=f"k={scale}，L={36 // (3 ** scale)}：按 {name} 着色",
                   xlabel="原始坐标 g₁", ylabel="原始坐标 g₂",
                   xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.15)
        fig.colorbar(image, ax=list(axes[parameter, :]), label=f"物理参数 {name}（仅作事后评价）", shrink=.85)
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle("冻结的参数–表示关系：不重新编码、不另作 PCA\n每点为一条物理链的固定编码；R-only 诊断不会改变这些点", fontsize=14)
    _save(fig, output, "coupling_by_scale.png", plt)


def write_plots(output: Path, arrays: dict, metrics: dict, history: list, labels: dict):
    """Write four Chinese diagnostic figures from finite, saved public arrays.

    ``flow_R`` arrays must contain the complete map, not its residual branch.
    This function subtracts ``flow_g`` exactly once and never searches for zeros.
    """
    _validate(arrays, metrics, labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ("R_vs_g.png", "loss.png", "metrics.png", "coupling_by_scale.png")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing diagnostic figures")
    plt = _plotting(None)
    _plot_flow(output, arrays, labels, plt)
    _plot_loss(output, history, labels, plt)
    _plot_metrics(output, metrics, labels, plt)
    _plot_couplings(output, arrays, plt)
    return [str(output / name) for name in names]
