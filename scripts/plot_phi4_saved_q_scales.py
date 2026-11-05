"""只读正式公共评价数组，检查最终 Q 的跨尺度表示；不调用模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_hopping_qcr import load_public_run, _setup_plot


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_statistics(arrays):
    full = np.asarray(arrays["direct_full_g"], dtype=float)
    direct = np.asarray(arrays["direct_g"], dtype=float)
    if full.shape != direct.shape or full.ndim != 3 or full.shape[1:] != (3, 1):
        raise ValueError("Require matching scalar full/half encodings at three scales")
    if not np.isfinite(full).all():
        raise ValueError("Nonfinite pooled Q output")
    cells = np.asarray(arrays["cell_ids"])
    labels = np.asarray(arrays["labels"])
    splits = np.asarray(arrays["splits"])
    rows = []
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() != 4:
            raise ValueError("This saved run must retain all four physical chains per condition")
        if not np.all(labels[mask] == labels[mask][0]) or len(set(splits[mask])) != 1:
            raise ValueError("Condition metadata is inconsistent")
        values = full[mask, :, 0]
        rows.append(dict(cell=int(cell), kappa=float(labels[mask][0, 0]),
            split=str(splits[mask][0]), physical_chains=4,
            pooled_g_by_chain=values.tolist(), mean=values.mean(0).tolist(),
            chain_sd_ddof1=values.std(0, ddof=1).tolist(),
            mean_empirical_delta=np.diff(values, axis=1).mean(0).tolist(),
            half_encoder_mean=direct[mask, :, 0].mean(0).tolist()))
    rows.sort(key=lambda row: row["kappa"])
    if len(rows) != 13 or not np.allclose([r["kappa"] for r in rows], np.arange(13) * .03 + .50):
        raise ValueError("Require all thirteen original conditions, not selected favorable cases")
    return rows


def descriptive_summary(rows, arrays):
    means = np.asarray([r["mean"] for r in rows])
    low = [r for r in rows if r["kappa"] < .67]
    high = [r for r in rows if r["kappa"] > .70]
    high_means = np.asarray([r["mean"] for r in high])
    low_values = np.concatenate([r["pooled_g_by_chain"] for r in low])
    high_values = np.concatenate([r["pooled_g_by_chain"] for r in high])
    first = np.asarray(arrays["direct_g"])[..., 0]
    one = np.asarray(arrays["one_step_g"])[..., 0]
    return dict(
        means_strictly_increasing_with_kappa_by_scale=(np.diff(means, axis=0) > 0).all(0).tolist(),
        high_group_mean_range_by_scale=np.ptp(high_means, axis=0).tolist(),
        low_high_observed_chain_gap_by_scale=(high_values.min(0) - low_values.max(0)).tolist(),
        boundary_sample_mean_gap_065_to_071=(np.asarray(high[0]["mean"]) - np.asarray(low[-1]["mean"])).tolist(),
        max_absolute_pooled_vs_two_half_difference=float(np.max(np.abs(arrays["direct_full_g"] - arrays["direct_g"]))),
        formal_two_half_one_step_rmse_by_transition=np.sqrt(np.mean((one-first[:, 1:])**2, axis=0)).tolist(),
        low_group_definition="κ=0.50..0.65; descriptive group, no learned phase labels",
        high_group_definition="κ=0.71..0.86; descriptive group, no learned phase labels",
    )


def draw_q(plt, rows, output):
    fig = plt.figure(figsize=(14.4, 8.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1))
    ax = fig.add_subplot(grid[:, 0])
    kappas = np.asarray([r["kappa"] for r in rows])
    means = np.asarray([r["mean"] for r in rows])
    sd = np.asarray([r["chain_sd_ddof1"] for r in rows])
    colors = ["#3b3b3b", "#0072B2", "#D55E00"]
    for k, color in enumerate(colors):
        ax.errorbar(kappas, means[:, k], yerr=sd[:, k], color=color, lw=1.8,
            capsize=3, label=f"k={k}，尺寸 {36 // 3**k}×{36 // 3**k}")
        for row in rows:
            values = np.asarray(row["pooled_g_by_chain"])[:, k]
            ax.scatter(np.full(4, row["kappa"]), values, s=15, color=color,
                marker="o" if row["split"] == "train" else "x", alpha=.62, zorder=3)
    ax.axvline(.68, color=".6", ls=":", lw=1)
    ax.text(.683, .585, "κ=0.68 样本", rotation=90, color=".45", fontsize=9, va="bottom")
    ax.set(title="同一个 Q：物理耦合与直接编码", xlabel=r"物理耦合 $\kappa$", ylabel=r"系综表示 $g_k=Q(C^kX)$")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(.489, .872)
    axes = [ax]
    for position, selected, title, cmap in (
        (grid[0, 1], rows[:6], "低 κ 六组：尺度轨迹", plt.cm.Blues),
        (grid[1, 1], rows[7:], "高 κ 六组：尺度轨迹", plt.cm.Oranges),
    ):
        sub = fig.add_subplot(position)
        for index, row in enumerate(selected):
            color = cmap(.40 + .55 * index / 5)
            values = np.asarray(row["pooled_g_by_chain"])
            sub.plot([0, 1, 2], values.T, color=color, alpha=.14, lw=.8)
            sub.errorbar([0, 1, 2], row["mean"], yerr=row["chain_sd_ddof1"],
                color=color, capsize=3, marker="o", lw=1.8, ms=4,
                label=f"κ={row['kappa']:.2f}")
        sub.set(title=title, xlabel="粗粒化次数 k", ylabel="直接编码 g", xticks=[0, 1, 2])
        sub.legend(ncol=3, fontsize=8, loc="best")
        axes.append(sub)
    for sub in axes:
        sub.set_ylim(.58, .785)
        sub.grid(alpha=.18)
    fig.suptitle("最终 Q 在两侧多个 κ 上的跨尺度表示：36 → 12 → 4", fontsize=16, y=.98)
    fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.91, wspace=.23, hspace=.34)
    fig.text(.5, .083, "每个 κ：4 条物理链，每条链用全部 128 张已保存构型组成系综；曲线为链均值，误差棒为链间标准差。", ha="center", fontsize=10)
    fig.text(.5, .052, "来自正式终点评价的一次固定 C 随机序列；链间离散混合物理采样与 C 随机性，不是独立 C 重复或置信区间。", ha="center", fontsize=9.5)
    fig.text(.5, .022, "左图实心圆为训练条件、叉号为留出条件；κ=0.68 仅作样本位置标记。未重新训练、采样或调用模型。", ha="center", fontsize=9.5)
    fig.savefig(output / "Q_g_kappa_and_scales.png", dpi=170)
    plt.close(fig)


def draw_r(plt, arrays, output):
    first = arrays["direct_g"][..., 0]
    one = arrays["one_step_g"][..., 0]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    grid = arrays["flow_grid"][:, 0]
    mapped = arrays["flow_next"][:, 0]
    axes[0].plot(grid, mapped, color="black", label="保存的 R(g)")
    axes[0].plot(grid, grid, "--", color=".6", label="恒等映射")
    for k, color in enumerate(("#0072B2", "#D55E00")):
        axes[0].scatter(first[:, k], first[:, k+1], color=color, s=20,
            label=f"直接编码：k={k} → {k+1}", alpha=.8)
        axes[1].scatter(first[:, k], np.full(len(first), k), color=color, s=15)
        axes[1].quiver(first[:, k], np.full(len(first), k), one[:, k]-first[:, k],
            np.zeros(len(first)), angles="xy", scale_units="xy", scale=1,
            color=color, width=.004, label=f"k={k} 输入")
    axes[0].set(title="可区分相区，不等于共享 R 已闭合", xlabel="输入 g", ylabel="下一尺度 g")
    axes[1].set(title="数据点上的预测位移 R(g)−g", xlabel="输入及预测 g", ylabel="输入尺度 k", yticks=[0, 1], ylim=(-.4, 1.4))
    axes[1].set_xlim(min(grid.min(), one.min())-.005, max(grid.max(), one.max())+.005)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("同一终点的 Q 实测下一尺度与 R 预测：仅复用保存数组", fontsize=14)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.20, wspace=.25)
    fig.text(.5, .09, "本图使用正式闭环口径：两个 64 张子系综分别编码后取均值；不是把 R 预测当成 Q 的直接测量。", ha="center", fontsize=10)
    fig.text(.5, .035, "不搜索新不动点、不拟合临界点或指数；图中误差不单独归因于 Q、C 或 R。", ha="center", fontsize=10)
    fig.savefig(output / "Q_direct_vs_saved_R.png", dpi=170)
    plt.close(fig)


def render(run, output, font=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, summary, _, arrays, pins = load_public_run(run)
    status = json.loads((run / "status.json").read_text())
    if (status.get("status") != "complete" or status.get("returncode") != 0
            or status.get("evaluation_arrays_sha256") != pins["evaluation_arrays.npz"]):
        raise ValueError("Public endpoint hash/status mismatch")
    for item in status["public_files"]:
        if item["path"] in pins and item["sha256"] != pins[item["path"]]:
            raise ValueError("Public input differs from endpoint whitelist")
    rows = grouped_statistics(arrays)
    result = dict(schema_version=1, source_run=str(run), actual_steps=summary["actual_steps"],
        final_module_sha256=summary["final_module_sha256"], source_sha256=pins,
        plotting_script_sha256=sha256(Path(__file__)),
        no_model_inference=True, no_sampling=True, no_training=True,
        Q_plot_encoding="Q of all 128 saved fields pooled within each physical chain",
        R_comparison_encoding=manifest["evaluation_encoding"],
        c_evaluation_seed=manifest["evaluation_seed"], independent_c_repeats=1,
        original_data_warning_count=summary["data_warning_count"], equilibrium_certified=False,
        error_bars="four-chain descriptive standard deviation, ddof=1; not confidence intervals",
        conditions=rows, metrics=descriptive_summary(rows, arrays),
        critical_kappa=None, thermal_nu=None, fixed_point_search_performed=False)
    plt = _setup_plot(font)
    output.mkdir(parents=True, exist_ok=False)
    draw_q(plt, rows, output)
    draw_r(plt, arrays, output)
    if pins != {name: sha256(run / name) for name in pins}:
        raise RuntimeError("Public input changed while plotting")
    result["plot_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.png"))}
    (output / "Q_scale_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    result = render(args.run, args.output, args.font)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
