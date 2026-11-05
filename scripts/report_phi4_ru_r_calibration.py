"""从公开数值数组生成冻结 Q/C 的 R 校准报告；不加载模型或搜索根。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _plotting
from scripts.report_phi4_ru_frozen_r import (
    _finite_array, _limits, _save, _sparse_flow_indices,
)


GROUPS = ("source", "raw96", "scaled96", "scaled256")
NAMES = ("原 R", "原坐标 R-96", "内部归一化 R-96", "内部归一化 R-256")
COLORS = ("#7d7d7d", "#2878b5", "#29a078", "#c265a1")
PATHS = ("0_to_1", "1_to_2", "0_to_2")
PATH_LABELS = ("36→12", "12→4", "36→4")
SCALE_COLORS = ("#2f6ca4", "#d98632", "#8a5ba6")
SEEDS = (91, 92, 93)


def _keys(group):
    return ("source_seed0",) if group == "source" else tuple(
        f"{group}_seed{seed}" for seed in SEEDS
    )


def _metric_rows(metrics, key, split):
    rows = [row for row in metrics[key] if row["split"] == split]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate metric path for {key}/{split}")
    return {row["path"]: row for row in rows}


def _validate(arrays, metrics, labels):
    repeats = None
    for split in ("train", "test"):
        direct = _finite_array(arrays, f"{split}_direct")
        if direct.ndim != 4 or direct.shape[2:] != (3, 2) or min(direct.shape[:2]) < 1:
            raise ValueError(f"{split}_direct must have shape [repeat, ensemble, 3, 2]")
        if direct.shape[0] < 2:
            raise ValueError("Multiple evaluation C repeats are required")
        if repeats is None:
            repeats = direct.shape[0]
        elif direct.shape[0] != repeats:
            raise ValueError("TRAIN/TEST repeat counts differ")
        _finite_array(arrays, f"{split}_labels", (direct.shape[1], 3))
        ids = np.asarray(arrays[f"{split}_cell_ids"])
        if ids.shape != (direct.shape[1],):
            raise ValueError("Condition IDs must match physical ensemble rows")
        for group in GROUPS:
            for key in _keys(group):
                if key not in labels or not isinstance(labels[key], str):
                    raise ValueError(f"Missing label for {key}")
                _finite_array(arrays, f"eval_{key}_{split}_one_step", direct.shape[:2] + (2, 2))
                rollout = _finite_array(arrays, f"eval_{key}_{split}_rollout", direct.shape)
                if not np.allclose(rollout[:, :, 0], direct[:, :, 0], atol=1e-12, rtol=1e-10):
                    raise ValueError("Rollout must start at paired microscopic encodings")
                rows = _metric_rows(metrics, key, split)
                if set(rows) != set(PATHS):
                    raise ValueError(f"Incomplete metrics for {key}/{split}")
                for row in rows.values():
                    for field in ("rmse", "nrmse", "relative_displacement_error", "identity_rmse"):
                        value = row[field]
                        if value is None or not np.isfinite(value) or value < 0:
                            raise ValueError(f"Invalid {key}/{split}/{field}")
    if set(np.asarray(arrays["train_cell_ids"]).tolist()) & set(
        np.asarray(arrays["test_cell_ids"]).tolist()
    ):
        raise ValueError("TRAIN and TEST conditions overlap")
    grid = _finite_array(arrays, "flow_g")
    if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid):
        raise ValueError("flow_g must have shape [point,2]")
    for name in ("flow_in_train_hull", "flow_near_train_support"):
        if name in arrays:
            mask = np.asarray(arrays[name])
            if mask.shape != (len(grid),) or mask.dtype.kind not in "biu" or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid {name} mask")
    if "flow_in_train_hull" not in arrays:
        raise ValueError("Missing TRAIN hull mask")
    for group in GROUPS:
        for key in _keys(group):
            _finite_array(arrays, f"{key}_flow_R", grid.shape)


def _flow(output, arrays, plt):
    grid = np.asarray(arrays["flow_g"])
    mask = np.asarray(arrays["flow_in_train_hull"], dtype=bool).copy()
    if "flow_near_train_support" in arrays:
        mask &= np.asarray(arrays["flow_near_train_support"], dtype=bool)
    indices = _sparse_flow_indices(grid, mask)
    direct = {split: np.asarray(arrays[f"{split}_direct"]).mean(axis=0)
              for split in ("train", "test")}
    extent = [value.reshape(-1, 2) for value in direct.values()]
    extent += [grid[indices]]
    extent += [np.asarray(arrays[f"{key}_flow_R"])[indices]
               for group in GROUPS for key in _keys(group)]
    lower, upper = _limits(np.concatenate(extent))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for panel, (ax, group, name) in enumerate(zip(axes.flat, GROUPS, NAMES)):
        for scale, color in enumerate(SCALE_COLORS):
            ax.scatter(direct["train"][:, scale, 0], direct["train"][:, scale, 1],
                       s=15, color=color, alpha=.38, linewidths=0,
                       label=f"训练点 k={scale}" if panel == 0 else None)
            ax.scatter(direct["test"][:, scale, 0], direct["test"][:, scale, 1],
                       s=13, color=color, alpha=.6, marker="x", linewidths=.6)
        # The dark arrows are one actual map (seed 91), never an averaged map.
        for seed_index, key in reversed(list(enumerate(_keys(group)))):
            delta = np.asarray(arrays[f"{key}_flow_R"])[indices] - grid[indices]
            primary = seed_index == 0
            ax.quiver(grid[indices, 0], grid[indices, 1], delta[:, 0], delta[:, 1],
                      angles="xy", scale_units="xy", scale=1,
                      color="#243641" if primary else COLORS[GROUPS.index(group)],
                      alpha=.85 if primary else .2, width=.0032 if primary else .0022,
                      headwidth=3.7, headlength=4.8)
        title = name + ("（源模型）" if group == "source" else "（深色：种子91）")
        ax.set(title=title, xlabel="原始坐标 g₁", ylabel="原始坐标 g₂",
               xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.16)
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("冻结 Q/C 后的共享 R 校准：箭头严格为 R(g)−g", fontsize=15)
    fig.text(.5, .025, "圆点为训练条件、叉号为留出条件的 C 重复均值；浅色箭头为优化种子92、93。\n"
             "在实际训练输入点绘制箭头，使用原坐标和真实长度；未搜索或认证不动点。", ha="center", fontsize=10)
    fig.subplots_adjust(top=.92, bottom=.12, hspace=.25, wspace=.2)
    _save(fig, output, "R_vs_g.png", plt)


def _loss(output, history, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = (("loss", "R 校准闭合总目标"), ("e01", "一步平方误差：36→12"),
              ("e12", "一步平方误差：12→4"), ("e02", "两步平方误差：36→4"))
    for ax, (field, title) in zip(axes.flat, fields):
        for group, name, color in zip(GROUPS[1:], NAMES[1:], COLORS[1:]):
            for seed_index, key in enumerate(_keys(group)):
                rows = [row for row in history if field in row.get("arms", {}).get(key, {})]
                if not rows:
                    continue
                steps = np.asarray([row["step"] for row in rows])
                values = np.asarray([row["arms"][key][field] for row in rows], dtype=float)
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(steps) <= 0):
                    raise ValueError(f"Invalid loss history: {key}/{field}")
                ax.plot(steps, np.ma.masked_less_equal(values, 0), color=color,
                        alpha=.95 if seed_index == 0 else .32,
                        linewidth=1.6 if seed_index == 0 else .9,
                        label=name if seed_index == 0 else None)
        ax.set(title=title, xlabel="额外 R 更新步数", ylabel="平方误差／加权目标", yscale="log")
        ax.grid(alpha=.18, which="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("多 C 重复与构型子集下的 R 校准过程\n深色为种子91；浅色为种子92、93；均报告固定终点", fontsize=14)
    fig.tight_layout(rect=(0, .08, 1, .91))
    _save(fig, output, "loss.png", plt)


def _metrics(output, metrics, plt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    fields = (("rmse", "绝对 RMSE"), ("nrmse", "固定 TRAIN 目标标准差 NRMSE"),
              ("relative_displacement_error", "误差／真实尺度位移 RMS"))
    positions, width = np.arange(3), .19
    for row_index, split in enumerate(("train", "test")):
        for ax, (field, title) in zip(axes[row_index], fields):
            for index, (group, name, color) in enumerate(zip(GROUPS, NAMES, COLORS)):
                values = np.asarray([[ _metric_rows(metrics, key, split)[path][field]
                                       for path in PATHS] for key in _keys(group)])
                x = positions + (index - 1.5) * width
                mean = values.mean(axis=0)
                ax.bar(x, mean, width, color=color, label=name)
                ax.errorbar(x, mean, yerr=np.maximum(0., np.stack((mean-values.min(0), values.max(0)-mean))),
                            fmt="none", color="#252525", linewidth=.8, capsize=2)
                if len(values) > 1:
                    for seed_index in range(len(values)):
                        ax.scatter(x + (seed_index - 1)*.025, values[seed_index], s=10,
                                   color="#303030", alpha=.7, zorder=4)
            if field == "rmse":
                rows = _metric_rows(metrics, "source_seed0", split)
                ax.scatter(positions, [rows[p]["identity_rmse"] for p in PATHS],
                           color="black", marker="_", s=190, linewidths=2, zorder=5)
            elif field == "relative_displacement_error":
                ax.axhline(1, linestyle="--", color="#333333", linewidth=.8)
            ax.set_xticks(positions, PATH_LABELS)
            ax.set(title=("训练条件：" if split == "train" else "留出条件：") + title,
                   xlabel="尺度转移", ylabel=title)
            ax.grid(axis="y", alpha=.16)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("独立 C 重复上的端点预测误差\n柱为三个优化种子均值，细线为最小–最大值，散点为单种子；非置信区间", fontsize=14)
    fig.tight_layout(rect=(0, .08, 1, .91))
    _save(fig, output, "metrics.png", plt)


def _couplings(output, arrays, plt):
    from matplotlib.cm import ScalarMappable

    direct = {split: np.asarray(arrays[f"{split}_direct"]) for split in ("train", "test")}
    labels = {split: np.asarray(arrays[f"{split}_labels"]) for split in ("train", "test")}
    lower, upper = _limits(np.concatenate([value.reshape(-1, 2) for value in direct.values()]))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), sharex=True, sharey=True,
                             constrained_layout=True)
    for parameter, variable in enumerate(("r", "u")):
        values = np.concatenate([value[:, parameter] for value in labels.values()])
        for scale in range(3):
            ax = axes[parameter, scale]
            for split in ("train", "test"):
                points = direct[split][:, :, scale]
                colors = np.broadcast_to(labels[split][None, :, parameter], points.shape[:2])
                image = ax.scatter(points[..., 0].ravel(), points[..., 1].ravel(),
                                   c=colors.ravel(), s=9, alpha=.16,
                                   cmap="viridis" if parameter == 0 else "plasma",
                                   vmin=values.min(), vmax=values.max(), linewidths=0)
                mean = points.mean(axis=0)
                ax.scatter(mean[:, 0], mean[:, 1], c=labels[split][:, parameter],
                           s=23, alpha=.9, marker="o" if split == "train" else "x",
                           cmap="viridis" if parameter == 0 else "plasma",
                           vmin=values.min(), vmax=values.max(), linewidths=.7,
                           label="训练条件均值" if split == "train" else "留出条件均值")
            ax.set(title=f"k={scale}，L={36 // (3 ** scale)}：按 {variable} 着色",
                   xlabel="原始坐标 g₁", ylabel="原始坐标 g₂",
                   xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.15)
        fig.colorbar(ScalarMappable(norm=image.norm, cmap=image.cmap), ax=list(axes[parameter]),
                     label=f"参数 {variable}（仅事后评价）", shrink=.85)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("冻结 Q/C 的参数–表示关系及 C 重复波动\n浅点为各次 C 重复，实点／叉号为同一物理链的重复均值；C 重复不作为独立物理链", fontsize=14)
    _save(fig, output, "coupling_by_scale.png", plt)


def _prediction(arrays, key, split, path):
    if path == "0_to_2":
        return np.asarray(arrays[f"eval_{key}_{split}_rollout"])[:, :, 2]
    return np.asarray(arrays[f"eval_{key}_{split}_one_step"])[:, :, 0 if path == "0_to_1" else 1]


def _stability(output, arrays, metrics, plt):
    """Describe error variability, without treating repeats as independent chains."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for group, name, color in zip(GROUPS, NAMES, COLORS):
        c_spread, seed_spread = [], []
        for path in PATHS:
            target_scale = 1 if path == "0_to_1" else 2
            target = np.asarray(arrays["test_direct"])[:, :, target_scale]
            errors = np.stack([_prediction(arrays, key, "test", path) - target for key in _keys(group)])
            # The denominator is common to all arms and comes from TRAIN only.
            first = _metric_rows(metrics, _keys(group)[0], "test")[path]
            denominator = first.get("train_target_std")
            if denominator is None:
                denominator = first["rmse"] / first["nrmse"] if first["nrmse"] else 0.
            if not np.isscalar(denominator) or not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("A positive scalar TRAIN target std is required for stability")
            c_spread.append(np.sqrt(np.mean((errors - errors.mean(axis=1, keepdims=True))**2)) / denominator)
            seed_spread.append(np.sqrt(np.mean((errors - errors.mean(axis=0, keepdims=True))**2)) / denominator)
        for ax, values in zip(axes, (c_spread, seed_spread)):
            ax.plot(range(3), values, "o-", color=color, label=name, linewidth=1.6)
    for ax, title in zip(axes, ("C 重复引起的误差波动", "优化种子引起的误差波动")):
        ax.set_xticks(range(3), PATH_LABELS)
        ax.set(title=title, xlabel="尺度转移", ylabel="误差波动 RMS／固定 TRAIN 目标标准差")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle("留出条件上的稳定性描述：固定 Q/C、固定物理链\n左图先对每个种子的 C 重复误差去均值；右图先对种子误差去均值；均为描述量，非统计置信区间", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .86))
    _save(fig, output, "stability.png", plt)


def write_plots(output: Path, arrays: dict, metrics: dict, history: list, labels: dict):
    """Write Chinese figures from saved arrays only; arrows are actual raw-g maps."""
    _validate(arrays, metrics, labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ("R_vs_g.png", "loss.png", "metrics.png", "coupling_by_scale.png", "stability.png")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing calibration figures")
    plt = _plotting(None)
    _flow(output, arrays, plt)
    _loss(output, history, plt)
    _metrics(output, metrics, plt)
    _couplings(output, arrays, plt)
    _stability(output, arrays, metrics, plt)
    return [str(output / name) for name in names]
