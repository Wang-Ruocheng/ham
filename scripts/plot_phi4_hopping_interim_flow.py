"""只读中期公共 JSON 绘制中文流场；不加载权重、构型或运行模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _array(payload, key, shape=None):
    value = np.asarray(payload[key], dtype=float)
    if (not np.isfinite(value).all()
            or (shape is not None and value.shape != shape)):
        raise ValueError(f"Invalid shape or nonfinite values: {key}")
    return value


def load_capture(source):
    """Validate saved public numbers without estimating roots or evaluating R."""
    payload = json.loads(Path(source).read_text())
    direct = _array(payload, "direct")
    if direct.ndim != 2 or direct.shape[0] < 2 or direct.shape[1] != 4:
        raise ValueError("Require one scalar g at four scales for each condition")
    count = direct.shape[0]
    _array(payload, "one_step", (count, 3))
    labels = _array(payload, "labels", (count, 3))
    if np.any(np.diff(labels[:, 0]) <= 0):
        raise ValueError("Kappa conditions must be strictly increasing")
    if not np.allclose(labels[:, 1:], labels[0, 1:], rtol=0, atol=0):
        raise ValueError("A one-parameter kappa scan at fixed lambda and h is required")
    if (len(payload["splits"]) != count or len(payload["cell_ids"]) != count
            or payload["lattice_sizes"] != [128, 64, 32, 16]):
        raise ValueError("Condition metadata or explicit b=2 scale trajectory differs")
    if type(payload["checkpoint_step"]) is not int or payload["checkpoint_step"] < 1:
        raise ValueError("Require a positive saved checkpoint step")
    if payload.get("thermal_nu") is not None:
        raise ValueError("This interim diagnostic does not certify a thermal exponent")
    grid = _array(payload, "flow_g")
    if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0):
        raise ValueError("Flow grid must be one-dimensional and strictly increasing")
    full_r = _array(payload, "flow_R", grid.shape)
    displacement = _array(payload, "flow_displacement", grid.shape)
    if not np.allclose(full_r - grid, displacement, rtol=1e-10, atol=1e-12):
        raise ValueError("Arrows must use the full map displacement R(g)-g")
    ranges = _array(payload, "per_input_scale_sampled_ranges", (3, 2))
    combined = _array(payload, "sampled_input_range", (2,))
    actual = np.stack((direct[:, :3].min(axis=0), direct[:, :3].max(axis=0)), axis=1)
    if (np.any(ranges[:, 0] > ranges[:, 1]) or combined[0] >= combined[1]
            or not np.allclose(ranges, actual, rtol=1e-10, atol=1e-12)
            or not np.allclose(combined, [actual[:, 0].min(), actual[:, 1].max()],
                               rtol=1e-10, atol=1e-12)):
        raise ValueError("Reported sampled input ranges disagree with direct g")
    for zero in payload["interpolated_zeros"]:
        if (zero["classification"] not in {"attracting", "repelling", "neutral"}
                or not all(np.isfinite(float(zero[key])) for key in ("g", "full_DR", "residual"))
                or not isinstance(zero["inside_sampled_input_range"], bool)):
            raise ValueError("Invalid saved interpolated-zero metadata")
        inside = bool(combined[0] <= zero["g"] <= combined[1])
        if inside != zero["inside_sampled_input_range"]:
            raise ValueError("Zero's sampled-range classification is inconsistent")
    return payload


def plot(source, output, font=None):
    """Render a fresh PNG from a captured checkpoint diagnostic only."""
    from scripts.report_phi4_hopping_qcr import _setup_plot

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Preserve every historical diagnostic image")
    payload = load_capture(source)
    direct = np.asarray(payload["direct"])
    prediction = np.asarray(payload["one_step"])
    labels = np.asarray(payload["labels"])
    grid = np.asarray(payload["flow_g"])
    full_r = np.asarray(payload["flow_R"])
    drift = np.asarray(payload["flow_displacement"])
    ranges = np.asarray(payload["per_input_scale_sampled_ranges"])
    lo, hi = payload["sampled_input_range"]
    colors = ["#2b7bba", "#dd8b26", "#258b75", "#9559ab"]
    sizes = payload["lattice_sizes"]
    plt = _setup_plot(font)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    map_ax, drift_ax, coupling_ax = axes

    map_ax.axvspan(lo, hi, color="#dddddd", alpha=.24, zorder=0)
    map_ax.plot(grid, grid, "--", color="#999999", lw=1.2, label="恒等映射")
    map_ax.plot(grid, full_r, color="#222222", lw=2, label="完整映射 R(g)")
    for k, color in enumerate(colors[:3]):
        map_ax.scatter(direct[:, k], prediction[:, k], s=20, color=color, zorder=3,
                       label=f"k={k}：预测 " + r"$R(g_k)$")
        map_ax.scatter(direct[:, k], direct[:, k + 1], s=35, marker="o", linewidth=.9,
                       facecolors="none", edgecolors=color, zorder=4,
                       label=f"k={k}：直接 " + r"$g_{k+1}$")
    map_ax.set(title="完整映射与一步闭合", xlabel="输入 g", ylabel="输出 g")
    map_ax.legend(fontsize=8, ncol=2, loc="best")

    drift_ax.axhline(0, color="#999999", lw=1, ls="--")
    drift_ax.axvspan(lo, hi, color="#dddddd", alpha=.24, zorder=0)
    drift_ax.plot(grid, drift, color="#222222", lw=2, label="位移 R(g)−g")
    span = max(float(np.ptp(drift)), float(np.max(np.abs(drift))), 1e-6)
    lower = min(float(np.min(drift)), 0.)
    upper = max(float(np.max(drift)), 0.)
    arrow_y = lower - .20 * span
    indices = np.unique(np.linspace(0, len(grid) - 1, 5).astype(int))
    # scale=1 in xy units means the endpoint is exactly x + R(x) - x.
    drift_ax.quiver(grid[indices], np.full(len(indices), arrow_y), drift[indices],
                    np.zeros(len(indices)), angles="xy", scale_units="xy", scale=1,
                    color="#555555", width=.004, headwidth=3.5, headlength=4.5,
                    minlength=0, label="真实位移箭头（不放大）")
    for k, color in enumerate(colors[:3]):
        y = lower - (.37 + .13 * k) * span
        drift_ax.plot(ranges[k], [y, y], lw=3, color=color, solid_capstyle="butt",
                      label=f"k={k} 预览范围")
    drift_ax.set_ylim(lower - .83 * span, upper + .14 * span)
    drift_ax.set(title="位移、插值候选零点与预览范围", xlabel="输入 g", ylabel="位移 R(g)−g")
    root_style = {"attracting": ("v", "#3077b5", "吸引候选"),
                  "repelling": ("^", "#bd4556", "排斥候选"),
                  "neutral": ("D", "#777777", "中性候选")}
    seen = set()
    for zero in payload["interpolated_zeros"]:
        marker, color, title = root_style[zero["classification"]]
        inside = zero["inside_sampled_input_range"]
        label = title + ("（包络内）" if inside else "（包络外）")
        for ax, y in ((map_ax, zero["g"]), (drift_ax, 0.)):
            ax.scatter([zero["g"]], [y], s=72, marker=marker,
                       facecolors=color if inside else "white", edgecolors=color,
                       linewidths=1.4, zorder=7,
                       label=label if ax is drift_ax and label not in seen else None)
        seen.add(label)
    if not payload["interpolated_zeros"]:
        drift_ax.set_title("位移与预览范围：所查区间未见插值零点")
    drift_ax.legend(fontsize=8, loc="best")

    for k, color in enumerate(colors):
        coupling_ax.plot(labels[:, 0], direct[:, k], "o-", color=color, lw=1.5,
                         ms=4, label=f"k={k}，边长 {sizes[k]}")
    coupling_ax.set(title="耦合参数到表示的尺度映射", xlabel="耦合参数 κ", ylabel=r"直接表示 $g_k$")
    coupling_ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.18)
    map_ax.set_xlim(grid[0], grid[-1])
    # Keep actual arrow endpoints visible; the curve is still evaluated only on grid.
    arrow_endpoints = grid[indices] + drift[indices]
    arrow_lo = min(float(grid[0]), float(arrow_endpoints.min()))
    arrow_hi = max(float(grid[-1]), float(arrow_endpoints.max()))
    padding = .02 * (arrow_hi - arrow_lo)
    drift_ax.set_xlim(arrow_lo - padding, arrow_hi + padding)
    fig.suptitle(f"连续标量场中期诊断：第 {payload['checkpoint_step']} 步权重", fontsize=16, y=.98)
    fig.text(.5, .035,
             "每个 κ 仅一条物理链的少量构型、一次 C 随机轨迹；连线仅作引导，不代表独立系综平均。\n"
             "灰带与短线仅为预览输入 g 的最小—最大包络，不代表连续物理支撑；候选零点不认证临界点，包络外为外推；未估计热临界指数。",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .115, 1, .93), w_pad=2.)
    # Exclusive creation prevents accidental overwrites even if another process wins a race.
    with destination.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=175, facecolor="white")
    plt.close(fig)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    print(plot(args.source, args.output, font=args.font))


if __name__ == "__main__":
    main()
