"""仅绘制已保存JSON历史；不加载模型、物理构型或执行远端计算。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def plot(history, output, *, expected_sha256, font=None):
    from scripts.report_phi4_hopping_qcr import _setup_plot

    source, destination = Path(history), Path(output)
    if destination.exists():
        raise FileExistsError("Preserve every historical preview image")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("History snapshot SHA256 differs from the remote receipt")
    rows = json.loads(raw)
    keys = ("q_total", "c_total", "r_total", "closure_all_start",
            "closure_target_1", "closure_target_2", "closure_target_3")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Require nonempty logged training history")
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    values = {key: np.asarray([row[key] for row in rows], dtype=float) for key in keys}
    if (steps[0] < 1 or np.any(np.diff(steps) <= 0)
            or any(not np.isfinite(value).all() for value in values.values())):
        raise ValueError("Nonfinite history or nonincreasing recorded steps")
    for row in rows:
        expected_q = (10 * row["q_reconstruction"] + 20 * row["q_subset_micro"]
                      + 5 * row["q_subset_coarse"])
        expected_c = -2 * row["c_information"] + 2 * row["c_variance"] + row["c_uniform"]
        expected_closure = sum(row[f"closure_target_{t}"] for t in (1, 2, 3)) / 3
        if not all(np.isclose(actual, expected, rtol=1e-10, atol=1e-12) for actual, expected in (
                (row["q_total"], expected_q), (row["c_total"], expected_c),
                (row["r_total"], 10 * row["closure_all_start"]),
                (row["closure_all_start"], expected_closure))):
            raise ValueError("History objective weights differ from the frozen recipe")
    if any(np.any(values[key] <= 0) for key in keys if key != "c_total"):
        raise ValueError("Positive values required for logarithmic loss axes")

    plt = _setup_plot(font)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    specifications = (
        (axes[0, 0], "q_total", "Q：表示目标", "#2471a3", True),
        (axes[0, 1], "c_total", "C：粗粒化目标", "#d97922", False),
        (axes[1, 0], "r_total", "R：闭环目标", "#7956a6", True),
    )
    for ax, key, title, color, logarithmic in specifications:
        ax.plot(steps, values[key], "o-", color=color, linewidth=1.9, markersize=5)
        if logarithmic:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="gray", linestyle="--", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("训练更新步数")
        ax.set_ylabel("加权目标值" + ("（对数纵轴）" if logarithmic else "（线性纵轴）"))
        ax.text(.97, .92, f"末值：{values[key][-1]:.4g}", transform=ax.transAxes,
                ha="right", va="top", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))

    ax = axes[1, 1]
    for t, length, color in ((1, 64, "#198c74"), (2, 32, "#d97922"), (3, 16, "#ba3c59")):
        ax.plot(steps, values[f"closure_target_{t}"], "o-", color=color,
                markersize=4, label=f"目标尺度{t}（边长{length}）")
    ax.plot(steps, values["closure_all_start"], "s--", color="#333333", markersize=4,
            label="三个目标尺度的均值")
    ax.set_yscale("log")
    ax.set(title="闭环误差的目标尺度分解", xlabel="训练更新步数",
           ylabel="未加权平方误差（对数纵轴）")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(steps)
        ax.grid(alpha=.2, which="major")
        ax.set_xlim(steps[0] - .035 * max(steps[-1], 1), steps[-1] * 1.04)
    fig.suptitle(f"连续标量场中期训练损失：记录至第{steps[-1]}／3000步", fontsize=15, y=.98)
    fig.text(.5, .025,
             "每点为当步更新前的随机训练微批次与随机粗粒化轨迹；连线仅作视觉引导，非固定验证集曲线。\n"
             "C目标含负的信息代理项，允许负值；各闭环目标尺度内先平均起点，再均匀平均三个尺度。",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95), h_pad=2., w_pad=2.)
    fig.savefig(destination, dpi=175, facecolor="white")
    plt.close(fig)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    print(plot(args.history, args.output, expected_sha256=args.expected_sha256, font=args.font))


if __name__ == "__main__":
    main()
