"""Plot a public, aggregate-only final-C convergence diagnostic.

This script never imports Torch, reads a checkpoint, or reads configurations.
It accepts descriptive statistics computed separately on the original host.
The expected row layout is ``results: [{kappa, method, scale, metrics}]``.
Only the six plotted metrics are parsed as finite numbers or objects with
``mean``; other diagnostic structures (for example, distance decompositions)
are left untouched and ignored by the plotting layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COLORS = {36: "#333333", 12: "#0072B2", 4: "#D55E00"}
METRICS = (
    ("q", r"空间同号程度：$q=|m|/\mathrm{RMS}(\phi)$", "越接近 1，单张场越均匀同号"),
    ("majority_fraction", "多数符号的格点占比", "越接近 1，单张场越接近全正或全负"),
    ("aligned_relative_rms_distance", "跨物理链的相对逐点均方根距离", "按原始场整体符号对齐；越小越相似"),
    ("aligned_absolute_rms_distance_over_s", r"跨物理链的绝对逐点距离 / $s$", "同样先对齐；保留幅度差异，越小越相似"),
)
PLOT_METRIC_NAMES = frozenset(name for name, _, _ in METRICS) | {
    "unaligned_relative_rms_distance", "unaligned_absolute_rms_distance_over_s",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value, context: str) -> float:
    if isinstance(value, dict):
        if "mean" not in value:
            raise ValueError(f"Expected a mean aggregate in {context}")
        value = value["mean"]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite aggregate in {context}")
    return result


def canonical_rows(document: dict) -> list[dict]:
    rows = document.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty public results list")
    result = []
    seen = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Every result row must be an object")
        method = raw["method"]
        if method not in {"original", "block_mean", "learned_sample", "learned_mean_chain"}:
            raise ValueError(f"Unknown diagnostic method: {method}")
        kappa = _finite_number(raw["kappa"], "kappa")
        scale = int(raw["scale"])
        if scale not in COLORS:
            raise ValueError("This diagnostic plots only the trained 36 -> 12 -> 4 chain")
        key = (kappa, method, scale)
        if key in seen:
            raise ValueError(f"Duplicate aggregate: {key}")
        seen.add(key)
        metrics = {name: _finite_number(value, f"{key}/{name}")
                   for name, value in raw["metrics"].items()
                   if name in PLOT_METRIC_NAMES}
        result.append(dict(kappa=kappa, method=method, scale=scale, metrics=metrics))
    return result


def configure_fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        raise FileNotFoundError("A Chinese font is required; refusing an unreadable plot")
    font_manager.fontManager.addfont(str(selected))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(selected)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10.5,
                         "savefig.facecolor": "white"})
    return plt


def series(rows, method: str, scale: int, metric: str):
    selected = sorted((row for row in rows if row["method"] == method
                       and row["scale"] == scale), key=lambda row: row["kappa"])
    if not selected:
        raise ValueError(f"Missing series: {method}/{scale}")
    x = np.asarray([row["kappa"] for row in selected])
    y = np.asarray([row["metrics"][metric] for row in selected])
    return x, y


def draw_series(ax, rows, metric: str, reference_kappa: float):
    from matplotlib.lines import Line2D
    specifications = [("original", 36, "-", "o")]
    specifications += [("learned_sample", size, "-", "o") for size in (12, 4)]
    specifications += [("block_mean", size, "--", "s") for size in (12, 4)]
    for method, size, style, marker in specifications:
        x, y = series(rows, method, size, metric)
        ax.plot(x, y, color=COLORS[size], linestyle=style, marker=marker,
                markersize=4.0, linewidth=1.8 if style == "-" else 1.5,
                alpha=.95 if method != "block_mean" else .72)
    ax.axvline(reference_kappa, color="#777777", linewidth=.9, linestyle=(0, (2, 3)), zorder=0)
    ax.set_xlabel(r"物理耦合 $\kappa$")
    ax.grid(alpha=.20)
    color_handles = [Line2D([0], [0], color=COLORS[size], label=f"尺寸 {size}乘{size}", linewidth=2)
                     for size in (36, 12, 4)]
    style_handles = [Line2D([0], [0], color="black", linestyle="-", marker="o", label="原始场／实际随机 C"),
                     Line2D([0], [0], color="black", linestyle="--", marker="s", label="普通块平均")]
    return color_handles + style_handles


def render_main(plt, rows, target: Path, reference_kappa: float):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    for index, (ax, (metric, ylabel, title)) in enumerate(zip(axes.flat, METRICS)):
        handles = draw_series(ax, rows, metric, reference_kappa)
        ax.set_title(f"（{'甲乙丙丁'[index]}）{title}", fontsize=12)
        ax.set_ylabel(ylabel)
        if metric == "q":
            ax.set_ylim(0, 1.03)
        elif metric == "majority_fraction":
            ax.set_ylim(.48, 1.03)
        else:
            ax.set_ylim(bottom=0)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=3, frameon=False, fontsize=10.5)
    subtitle = "实际随机 C 与普通三乘三块平均"
    fig.suptitle("最终 C 对多组随机构型的粗粒化：是否更同号、是否更相似？\n" + subtitle,
                 fontsize=16, y=.992)
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.835, wspace=.28, hspace=.35)
    fig.text(.5, .087, "每个条件 4 条物理链 × 16 个预先固定保存时刻；随机 C 每张场使用 2 组独立固定噪声。", ha="center", fontsize=10.5)
    fig.text(.5, .059, "两组 C 噪声不是两倍独立物理样本；曲线为描述性均值，不显示伪独立置信区间。", ha="center", fontsize=10.5)
    fig.text(.5, .031, "下排仅为比较而按原始场均值对齐整体正负；不是 C 的实际操作。竖线为临界参考值，不是本次估计。", ha="center", fontsize=10.0)
    fig.savefig(target, dpi=175)
    plt.close(fig)


def render_unaligned(plt, rows, target: Path, reference_kappa: float):
    metrics = (("unaligned_relative_rms_distance", "未对齐整体正负：相对逐点均方根距离"),
               ("unaligned_absolute_rms_distance_over_s", r"未对齐整体正负：绝对逐点距离 / $s$"))
    if not all(all(metric in row["metrics"] for row in rows) for metric, _ in metrics):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    for ax, (metric, title) in zip(axes, metrics):
        handles = draw_series(ax, rows, metric, reference_kappa)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("距离（越小越相似）")
        ax.set_ylim(bottom=0)
    fig.suptitle("辅助检查：保留零场系统的整体正／负扇区", fontsize=15)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.18, top=.72, wspace=.27)
    fig.text(.5, .065, "整体为正与整体为负属于两个对称扇区；未对齐距离不趋零，本身不证明粗粒化失败。", ha="center", fontsize=10.5)
    fig.savefig(target, dpi=175)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    rows = canonical_rows(document)
    # Validate the four mandatory aggregates before producing any output.
    for method, sizes in (("original", (36,)), ("learned_sample", (12, 4)),
                          ("block_mean", (12, 4))):
        for size in sizes:
            for metric, _, _ in METRICS:
                series(rows, method, size, metric)
    targets = ("C_ensemble_convergence.png", "C_ensemble_unaligned_distance.png", "plot_receipt.json")
    if any((args.output / name).exists() for name in targets):
        raise FileExistsError("Refusing to overwrite existing diagnostic figures")
    args.output.mkdir(parents=True, exist_ok=True)
    plt = configure_fonts()
    reference_kappa = _finite_number(document.get("metadata", {}).get("reference_kappa", .680601), "reference_kappa")
    main_path = args.output / "C_ensemble_convergence.png"
    render_main(plt, rows, main_path, reference_kappa)
    paths = [main_path]
    raw_path = args.output / "C_ensemble_unaligned_distance.png"
    if render_unaligned(plt, rows, raw_path, reference_kappa):
        paths.append(raw_path)
    receipt = dict(input_path=str(args.input.resolve()), input_sha256=sha256(args.input),
                   checkpoint_or_model_loaded=False, raw_fields_loaded=False,
                   inference_or_training=False, labels_language="Chinese",
                   metric_definitions=document.get("metric_definitions", {}),
                   metadata=document.get("metadata", {}),
                   public_files=[dict(path=path.name, bytes=path.stat().st_size,
                                      sha256=sha256(path)) for path in paths])
    receipt_path = args.output / "plot_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), public_files=receipt["public_files"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
