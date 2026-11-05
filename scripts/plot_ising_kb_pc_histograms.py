"""用正式报告的固定 PCA 绘制 KB 编码分布；仅读取公共 JSON/NPZ。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plot_ising_kb_pc12 import _font_setup, load_report, sha256


NOTES = [
    "PCA 中心和基底沿用正式报告，仅由 TRAIN 的 k=0、1 编码拟合；没有重新拟合或白化。",
    "每条记录是一个物理系综的直接 Q 编码，不是单个自旋构型，也不是 R 的预测。",
    "直方图展示 TRAIN 与 TEST 的完整 KB 扫描；每个参数格点有两条物理链，等权计数。",
    "同一主成分的三个尺度共用分箱；每条密度直方图的面积分别归一为 1。",
    "不同主成分横轴独立缩放，不能以图形看起来同样宽判断其方差相同；原始单位不变。",
    "k=2 是直接粗粒化的最终观测尺度，不是额外 R 输入或模型外推。",
    "方差谱由 TRAIN 输入 k=0、1 定义，不是全尺度直方图重新估算的方差解释率。",
    "两模型有各自的 PCA 坐标；PC 排序与符号可能不同，不能按相同编号直接对齐物理方向。",
    "分布峰形受 KB 网格取样、非线性编码与有限系综影响，不能单凭双峰判定相变或不动点。",
    "全局方差集中于两轴不等于有效物理维数已认证，也不等于二维自治 RG 流。",
]


def principal_scores(report):
    """Project saved direct ensemble encodings without labels, refitting or rescaling."""
    direct = np.concatenate((report.encoded["train_direct"], report.encoded["test_direct"]), axis=0)
    center, components = report.grid["pca_center"], report.grid["pca_components"]
    return (direct - center) @ components.T


def distribution_summary(scores, variance_fractions, bins=32):
    scores = np.asarray(scores, dtype=np.float64)
    fractions = np.asarray(variance_fractions, dtype=np.float64)
    if (scores.ndim != 3 or scores.shape[0] == 0 or scores.shape[1] != 3
            or scores.shape[2] == 0 or not np.isfinite(scores).all()
            or fractions.shape != (scores.shape[2],) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.)
            or not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2):
        raise ValueError("Invalid scores, variance fractions or histogram bin count")
    coordinates = []
    for pc, fraction in enumerate(fractions):
        values = scores[:, :, pc]
        edges = np.histogram_bin_edges(values.ravel(), bins=bins)
        rows = []
        for scale in range(3):
            sample = values[:, scale]
            counts, _ = np.histogram(sample, bins=edges)
            density = counts / (sample.size * np.diff(edges))
            rows.append(dict(scale=scale, n=sample.size, mean=float(sample.mean()),
                std=float(sample.std(ddof=0)), minimum=float(sample.min()), maximum=float(sample.max()),
                histogram_counts=counts.tolist(), histogram_density=density.tolist()))
        coordinates.append(dict(principal_component=pc + 1,
            training_input_variance_fraction=float(fraction), bin_edges=edges.tolist(), by_scale=rows))
    return dict(coordinates=coordinates, records_per_scale=scores.shape[0],
        dimensions=scores.shape[2], scales=[0, 1, 2], bins=bins)


def _plot_histograms(reports, summaries, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    colors = ("#246a9f", "#e77c30", "#2e9464")
    styles = ("-", "--", "-.")
    fig, axs = plt.subplots(2, 5, figsize=(18, 8.1))
    fig.subplots_adjust(left=.055, right=.985, bottom=.17, top=.79, hspace=.6, wspace=.4)
    for row, (report, summary) in enumerate(zip(reports, summaries)):
        for pc, ax in enumerate(axs[row]):
            coordinate = summary["coordinates"][pc]
            edges = np.asarray(coordinate["bin_edges"])
            for scale, values in enumerate(coordinate["by_scale"]):
                density = np.asarray(values["histogram_density"])
                ax.stairs(density, edges, fill=True, color=colors[scale], alpha=.07)
                ax.stairs(density, edges, color=colors[scale], lw=1.5, linestyle=styles[scale])
            ax.axvline(0, lw=.65, color=".4", alpha=.45)
            ax.set_title(f"{report.dimension}维 · 主成分{pc + 1}\n"
                f"训练输入方差 {coordinate['training_input_variance_fraction'] * 100:.4g}%", fontsize=11)
            ax.set_xlabel(f"主成分{pc + 1}坐标（原始单位）", fontsize=9)
            ax.set_ylabel("概率密度", fontsize=9)
            ax.grid(axis="y", alpha=.15)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            for axis in (ax.xaxis, ax.yaxis):
                formatter = ScalarFormatter(useOffset=False, useMathText=True)
                formatter.set_powerlimits((-2, 3))
                axis.set_major_formatter(formatter)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
    fig.suptitle("KB 扫描：前五个主成分的编码分布", y=.98, fontsize=19)
    counts = "、".join(f"{report.dimension}维：{summary['records_per_scale']} 个系综"
                        for report, summary in zip(reports, summaries))
    fig.text(.5, .92, f"固定同一组 PCA 轴，三个尺度叠加；每个尺度 {counts}，含训练集与测试集",
             ha="center", fontsize=11)
    handles = [Line2D([0], [0], color=color, lw=2, linestyle=style,
        label=f"尺度 k={scale}，L={36 // 3 ** scale}")
        for scale, (color, style) in enumerate(zip(colors, styles))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895), ncol=3, frameon=False)
    fig.text(.5, .04, "各小图横轴独立缩放：后三个主成分已放大，不能据图形宽度直接比较方差。\n"
        "同一主成分的三条曲线共用分箱、各自面积为1；展示的是直接 Q 编码，不是额外 RG 预测。",
        ha="center", fontsize=11)
    fig.savefig(output / "PC_histograms.png", dpi=180)
    plt.close(fig)


def _plot_variance(reports, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=.09, right=.98, bottom=.22, top=.80, wspace=.32)
    for ax, report in zip(axs, reports):
        fractions = np.asarray(report.metrics["pca"]["explained_variance_fraction"])
        x = np.arange(1, len(fractions) + 1)
        positive = fractions > 0
        ax.bar(x[positive], 100 * fractions[positive], color=["#246a9f" if p <= 2 else "#b1bdc7"
                                                           for p in x[positive]])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("主成分编号")
        ax.set_ylabel("训练输入方差比例（%，对数轴）")
        ax.set_title(f"{report.dimension}维模型：前两轴合计 {fractions[:2].sum() * 100:.3f}%")
        ticks = 10. ** np.arange(-4, 3)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in ticks]))
        ax.set_ylim(max(float(fractions[positive].min() * 100) * .15, 1e-12), 200)
        for pc, fraction in zip(x, fractions):
            if fraction > 0:
                ax.text(pc, fraction * 130, f"{fraction * 100:.3g}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("各主成分的方差贡献：对照直方图的真实宽度", fontsize=16)
    fig.text(.5, .05, "仅 TRAIN 的 k=0、1 输入尺度定义 PCA；没有把 k=2 或测试集用于重新选轴。\n"
        "方差贡献小不等于物理作用小；不能仅凭此谱认证有效物理维数或忽略相应 RG 方向。",
        ha="center", fontsize=10)
    fig.savefig(output / "PC_variance_spectrum.png", dpi=180)
    plt.close(fig)


def write_plots(source, output, bins=32):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite an existing plot directory")
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the frozen source reports")
    reports = [load_report(source, dimension) for dimension in (5, 10)]
    scores = [principal_scores(report) for report in reports]
    summaries = [distribution_summary(values, report.metrics["pca"]["explained_variance_fraction"], bins)
                 for report, values in zip(reports, scores)]
    source_hashes = {key: value for report in reports for key, value in report.source_hashes.items()}
    font = _font_setup()
    output.mkdir(parents=True, exist_ok=False)
    _plot_histograms(reports, summaries, output)
    _plot_variance(reports, output)
    arrays = {}
    for report, values in zip(reports, scores):
        prefix = f"dim{report.dimension}_"
        arrays.update({prefix + key: value for key, value in dict(
            principal_scores=values, pca_center=report.grid["pca_center"],
            pca_components=report.grid["pca_components"],
            variance_fractions=np.asarray(report.metrics["pca"]["explained_variance_fraction"]),
            split_is_train=np.arange(len(values)) < len(report.encoded["train_direct"])).items()})
    np.savez_compressed(output / "PC_histogram_arrays.npz", **arrays)
    if any(sha256(source / key) != value for key, value in source_hashes.items()):
        raise ValueError("Source reports changed during plotting")
    result = dict(status="complete", source=str(source), source_sha256=source_hashes,
        script_sha256=sha256(Path(__file__)), font_path=str(font),
        models=[dict(dimension=report.dimension, **summary) for report, summary in zip(reports, summaries)],
        model_inference=False, physical_sampling=False, training_updates=0, pca_refit=False,
        records="每个物理系综在各尺度的直接 Q 编码；TRAIN 后接 TEST；未平均两条物理链",
        notes=NOTES, output_sha256={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output / "PC_histogram_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=32)
    args = parser.parse_args(argv)
    result = write_plots(args.source, args.output, args.bins)
    print(json.dumps(dict(status=result["status"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
