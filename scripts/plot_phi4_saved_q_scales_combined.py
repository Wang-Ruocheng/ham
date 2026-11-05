"""合并已有 Q 尺度轨迹；小数 k 仅作线性绘图连线，不是新增评价。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plot_phi4_saved_q_scales import sha256
from scripts.report_phi4_hopping_qcr import _setup_plot


MEASURED_K = np.asarray([0., 1., 2.])


def visual_interpolation(values, points=201):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Require the three measured finite scale values")
    if type(points) is not int or points < 3 or points % 2 != 1:
        raise ValueError("Use an odd grid including all three measured anchors")
    grid = np.linspace(0., 2., points)
    return grid, np.interp(grid, MEASURED_K, values)


def render(source, output, *, points=201, font=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_sha = sha256(source)
    document = json.loads(source.read_text())
    if document.get("no_model_inference") is not True or document.get("actual_steps") != 3000:
        raise ValueError("Expected the frozen endpoint public summary")
    rows = sorted(document["conditions"], key=lambda row: row["kappa"])
    if len(rows) != 13 or not np.allclose([r["kappa"] for r in rows], .50 + .03*np.arange(13)):
        raise ValueError("Retain all thirteen measured κ conditions")
    for row in rows:
        visual_interpolation(row["mean"], points)
        if np.asarray(row["pooled_g_by_chain"]).shape != (4, 3):
            raise ValueError("Expected four physical-chain values at three scales")
    run = Path(document["source_run"])
    pins = document["source_sha256"]
    if pins != {name: sha256(run / name) for name in pins}:
        raise ValueError("Original public endpoint no longer matches the summary")
    plt = _setup_plot(font)
    from matplotlib import colors
    from matplotlib.lines import Line2D
    cmap = colors.ListedColormap(plt.cm.viridis(np.linspace(.06, .91, 13)))
    kappas = np.asarray([r["kappa"] for r in rows])
    edges = np.r_[kappas[0]-.015, (kappas[:-1]+kappas[1:])/2, kappas[-1]+.015]
    norm = colors.BoundaryNorm(edges, cmap.N)
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(11.3, 7.3))
    for row in rows:
        color = cmap(norm(row["kappa"]))
        x, y = visual_interpolation(row["mean"], points)
        ax.plot(x, y, color=color, lw=2, alpha=.94, zorder=2)
        ax.errorbar(MEASURED_K, row["mean"], yerr=row["chain_sd_ddof1"],
            color=color, marker="o", ls="none", ms=5, capsize=3, lw=1.2, zorder=3)
    ax.set(xlim=(-.06, 2.06), ylim=(.58, .78),
        xlabel="粗粒化次数 k（圆点为实测整数尺度）", ylabel="直接编码 g",
        title="全部 13 组 κ：同一个 Q 的跨尺度轨迹")
    ax.set_xticks(np.arange(0., 2.01, .25))
    ax.set_xticklabels([f"{x:g}" for x in np.arange(0., 2.01, .25)])
    ax.grid(alpha=.19)
    ax.legend(handles=[Line2D([], [], color=".25", marker="o", ls="none", label="实测链均值 ± 链间标准差"),
        Line2D([], [], color=".25", lw=2, label="线性连线，仅辅助观察")], loc="upper left", fontsize=10)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(scalar, ax=ax, ticks=kappas, fraction=.045, pad=.045)
    bar.set_ticklabels([f"{value:.2f}" for value in kappas])
    bar.set_label("物理耦合 κ（现有实际采样条件）")
    fig.suptitle("高、低 κ 轨迹合并：36 → 12 → 4", fontsize=16, y=.98)
    fig.subplots_adjust(left=.10, right=.93, top=.87, bottom=.22)
    fig.text(.5, .125, "实测尺度仍只有 k=0、1、2；每条线采用 201 个绘图点连接，不代表存在分数次粗粒化的模型或测量。", ha="center", fontsize=9.5)
    fig.text(.5, .085, "每个 κ：4 条物理链 × 每链 128 张保存构型；误差棒不是置信区间，且未单独分离 C 随机性。", ha="center", fontsize=9.5)
    fig.text(.5, .045, "仅重绘既有终点评价；无新增 κ 采样、模型推理或训练，原采样警告保留。", ha="center", fontsize=9.5)
    figure = output / "Q_all_kappa_scale_trajectories.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    if sha256(source) != source_sha or pins != {name: sha256(run / name) for name in pins}:
        raise RuntimeError("Source public data changed during plotting")
    receipt = dict(source_summary=str(source), source_summary_sha256=source_sha,
        original_public_source_sha256=pins, plotting_script_sha256=sha256(__file__),
        figure_sha256=sha256(figure), measured_k=MEASURED_K.tolist(),
        measured_kappas=kappas.tolist(), visual_points_per_line=points,
        interpolation="piecewise linear visual guide only; fractional k is not evaluated",
        no_new_sampling=True, no_model_inference=True, no_training=True)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    result = render(args.source, args.output, font=args.font)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
