"""Saved-array-only finite-size physical crossover and latent trajectory check.

The resampling unit is an independent physical chain. Four subset/C repeats
are averaged within each chain before any cell mean or paired resampling.
No model is loaded, evaluated, trained, or searched for roots by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


MOMENTS = ("magnetization", "absolute_magnetization", "magnetization_squared", "magnetization_fourth")
METRICS = ("absolute_magnetization", "binder", "susceptibility_connected", "susceptibility_absolute_subtraction")
LABELS = (r"绝对磁化 $\langle |M|\rangle$", r"宾德累积量 $U_4$",
          r"零场连通量 $\chi=L^2(\langle M^2\rangle-\langle M\rangle^2)$",
          r"绝对磁化扣除量 $L^2(\langle M^2\rangle-\langle |M|\rangle^2)$")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def physical_metrics(moment):
    m1, ma, m2, m4 = np.moveaxis(np.asarray(moment), -1, 0)
    return np.stack((ma, 1 - m4 / (3 * m2 ** 2), 36 ** 2 * (m2 - m1 ** 2),
                     36 ** 2 * (m2 - ma ** 2)), axis=-1)


def read_inputs(source, cache, calibration):
    physics = json.loads((source / "report/physics_summary.json").read_text())
    sidecars = sorted(physics["sampling_sidecars"], key=lambda x: x["cell_index"])
    if len(sidecars) != 45 or [x["cell_index"] for x in sidecars] != list(range(45)):
        raise ValueError("Expected all 45 parameter conditions")
    moments, labels = [], []
    for cell in sidecars:
        chains = sorted(cell["chains"], key=lambda x: x["chain_index"])
        if [x["chain_index"] for x in chains] != list(range(4)) or cell["size"] != 36:
            raise ValueError("The physical resampling unit must be four L=36 chains")
        if any(any(x["diagnostics"][name]["samples"] != 128 for name in MOMENTS) for x in chains):
            raise ValueError("Unequal chain lengths are unsupported")
        moments.append([[x["diagnostics"][name]["mean"] for name in MOMENTS] for x in chains])
        labels.append([cell["r"], cell["u"]])
    moments, labels = np.asarray(moments), np.asarray(labels)
    with np.load(cache / "cache.npz", allow_pickle=False) as z:
        direct = np.empty((4, 180, 3, 2))
        assigned = np.zeros(180, dtype=bool)
        for split in ("train", "test"):
            index = z[f"{split}_global_chain_indices"]
            cid = z[f"{split}_cell_ids"]
            if not np.array_equal(index // 4, cid) or assigned[index].any():
                raise ValueError("Chain identity or split overlap mismatch")
            if not np.allclose(z[f"{split}_labels"][:, :2], labels[cid], rtol=0, atol=1e-13):
                raise ValueError("Coupling labels are not aligned with physical chain identity")
            direct[:, index] = z[f"eval_{split}_direct"]
            assigned[index] = True
        if not assigned.all():
            raise ValueError("Missing physical chains")
    with np.load(calibration / "evaluation_arrays.npz", allow_pickle=False) as z, np.load(cache / "cache.npz", allow_pickle=False) as c:
        for split in ("train", "test"):
            if not np.array_equal(z[f"{split}_direct"], c[f"eval_{split}_direct"]):
                raise ValueError("Calibration evaluation and frozen cache disagree")
    with np.load(source / "run/evaluation_arrays.npz", allow_pickle=False) as z:
        source_cell_means = z["grid_mean_direct"].copy()
        if not np.array_equal(z["grid_cell_ids"], np.arange(45)) or not np.allclose(z["grid_labels"][:, :2], labels):
            raise ValueError("Source evaluation parameter order disagrees")
    with np.load(source / "report/analysis_arrays.npz", allow_pickle=False) as z:
        if not np.array_equal(z["g_mean_by_scale"], source_cell_means):
            raise ValueError("Source public report and evaluation disagree")
        report_metric = np.stack((z["absolute_magnetization"], z["binder"], z["susceptibility_absolute_subtraction_per_site"]), axis=-1)
        calculated = physical_metrics(moments.mean(1))[:, [0, 1, 3]]
        if not np.allclose(calculated, report_metric, rtol=1e-10, atol=1e-11):
            raise ValueError("Per-chain moments do not reproduce the original physical report")
    # C/subset repeats are measurements on the same chains, not 720 chains.
    chain_direct = direct.mean(0).reshape(45, 4, 3, 2)
    return physics, sidecars, labels, moments, direct, chain_direct, source_cell_means


def make_summary(physics, labels, moments, direct, chain_direct, bootstrap=4000):
    rng = np.random.default_rng(2026102903)
    draws = rng.integers(0, 4, size=(bootstrap, 45, 4))
    cell = np.arange(45)[None, :, None]
    boot_moments = moments[cell, draws].mean(2)
    boot_metrics = physical_metrics(boot_moments)
    boot_g = chain_direct[cell, draws].mean(2)
    metrics, g = physical_metrics(moments.mean(1)), chain_direct.mean(1)
    rows, crossover = [], []
    for i in range(45):
        rows.append(dict(cell_index=i, r=float(labels[i, 0]), u=float(labels[i, 1]),
            physical_metrics={name: dict(mean=float(metrics[i, j]), interval95=np.quantile(boot_metrics[:, i, j], [.025, .975]).tolist()) for j, name in enumerate(METRICS)},
            g_mean_by_scale=g[i].tolist(),
            g_interval95_by_scale=np.quantile(boot_g[:, i], [.025, .975], axis=0).tolist()))
    for u in np.unique(labels[:, 1]):
        ids = np.flatnonzero(labels[:, 1] == u)
        ids = ids[np.argsort(labels[ids, 0])]
        rs = labels[ids, 0]
        j = int(np.argmin(np.diff(metrics[ids, 1]) / np.diff(rs)))
        left, right = ids[j:j + 2]
        difference = g[right] - g[left]
        distance = np.linalg.norm(difference, axis=-1)
        boot_distance = np.linalg.norm(boot_g[:, right] - boot_g[:, left], axis=-1)
        # Endpoints share the same bootstrap draw at all scales; cells are
        # independently resampled. No interpolation or critical-point fit.
        ratios = boot_distance[:, 1:] / boot_distance[:, :-1]
        boot_slope = np.diff(boot_metrics[:, ids, 1], axis=1) / np.diff(rs)
        selected_bins = np.argmin(boot_slope, axis=1)
        peak_abs = int(np.argmax(metrics[ids, 3]))
        crossover.append(dict(u=float(u), descriptive_selection="largest adjacent Binder decline on the sampled r grid; evaluation only",
            cell_ids=[int(left), int(right)], sampled_r_interval=rs[j:j + 2].tolist(),
            binder_at_endpoints=metrics[[left, right], 1].tolist(),
            absolute_magnetization_at_endpoints=metrics[[left, right], 0].tolist(),
            interval_bootstrap_selection_fraction=float(np.mean(selected_bins == j)),
            alternative_interval_selection_counts=np.bincount(selected_bins, minlength=8).tolist(),
            absolute_subtraction_grid_peak_r=float(rs[peak_abs]),
            latent_endpoint_distance_by_scale=distance.tolist(),
            latent_endpoint_distance_interval95_by_scale=np.quantile(boot_distance, [.025, .975], axis=0).tolist(),
            latent_distance_ratio_12_over_36_and_4_over_12=(distance[1:] / distance[:-1]).tolist(),
            latent_distance_ratio_interval95=np.quantile(ratios, [.025, .975], axis=0).tolist(),
            per_scale_physical_chain_mean_trajectories=g[[left, right]].tolist()))
    correlations = []
    for k in range(3):
        correlations.append(dict(scale=k, size=[36, 12, 4][k],
            spearman_absM_vs_g=[float(spearmanr(metrics[:, 0], g[:, k, d]).statistic) for d in range(2)],
            spearman_binder_vs_g=[float(spearmanr(metrics[:, 1], g[:, k, d]).statistic) for d in range(2)],
            qualification="45 condition means; u and r both vary, descriptive coordinate correlation only"))
    summary = dict(status="complete", workflow_complete=True, returncode=0, error=None,
        labels_used_for_evaluation_only=True, model_loaded=False, inference_performed=False,
        training_performed=False, new_physical_sampling=False, source_size=36,
        independent_physical_chains=180, chains_per_condition=4, evaluation_C_subset_repeats=4,
        independent_chains_added=0, repeated_measurement_count=720,
        uncertainty=dict(method="percentile bootstrap of the four independent physical-chain moment vectors within each condition",
            repetitions=bootstrap, seed=2026102903, confidence=.95,
            same_chain_resample_for_all_moments_and_all_latent_scales=True,
            C_subset_repeats_averaged_before_resampling=True,
            cells_resampled_independently=True, raw_saved_fields_resampled=False,
            qualification="Only four independent chains per condition, with hot/cold starts and finite autocorrelation. These conditional, descriptive intervals do not certify equilibrium or cover systematic bias; no iid claim for 128 fields or C repeats."),
        physical_rows=rows, crossover_intervals=crossover, latent_correlations=correlations,
        sampling_warnings_preserved=int(physics["sampling_diagnostics"]["declared_warning_count"]),
        equilibrium_certified=False, physical_critical_point_certified=False,
        critical_r=None, thermal_nu=None,
        susceptibility_conventions=dict(connected="L^2*(<M^2>-<M>^2), M is magnetization per site; h=0 finite-volume Z2-sector fluctuations can make this large throughout ordered side",
            absolute_subtraction="L^2*(<M^2>-<|M|>^2), useful descriptive crossover fluctuation; not the h=0 linear response"),
        limitations=["单微观尺寸L=36，仅粗粒化到12和4；不拟合无限体积临界线或临界指数。", "描述性区间由已观察到的宾德变化选出；同数据上的区间分离不能当独立验证。", "g的端点距离变化不是热本征值，不能据此计算临界指数或认证排斥边界。", "物理统计来自每链128个已保存场；g为64场子集的4次独立编码平均，二者共享物理链但测量方式不同。", "仅有微观物理矩，不能把相同物理量当作粗尺度12、4的直接测量。", "没有对流场外推区推断边界、插值耦合标签或搜索不动点。"])
    arrays = dict(labels_ru=labels, chain_moments=moments, physical_metrics=metrics,
        physical_metric_interval95=np.quantile(boot_metrics, [.025, .975], axis=0),
        g_repeated=direct, g_C_averaged_per_chain=chain_direct, g_condition_mean=g,
        g_condition_interval95=np.quantile(boot_g, [.025, .975], axis=0))
    return summary, arrays


def plotting(arrays, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Heiti TC", "STHeiti", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        raise RuntimeError("A Chinese font is required")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    labels, metrics = arrays["labels_ru"], arrays["physical_metrics"]
    ci, g, chains = arrays["physical_metric_interval95"], arrays["g_condition_mean"], arrays["g_C_averaged_per_chain"]
    colors = ("#4477aa", "#ee7733", "#228833", "#cc6677", "#8855aa")
    us = np.unique(labels[:, 1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, ax in enumerate(axes.flat):
        for u, color in zip(us, colors):
            ids = np.flatnonzero(labels[:, 1] == u)
            # Percentile intervals can be asymmetric about the observed value;
            # draw their endpoints directly rather than clipping error lengths.
            ax.vlines(labels[ids, 0], ci[0, ids, j], ci[1, ids, j], color=color, alpha=.65, lw=1)
            ax.scatter(labels[ids, 0], metrics[ids, j], color=color, s=25, label=f"u={u:g}", zorder=3)
        ax.set(xlabel="裸质量参数 r", ylabel=LABELS[j])
        ax.spines[["top", "right"]].set_visible(False)
        if j == 2:
            ax.set_yscale("log")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
    fig.suptitle("微观物理统计：45个条件、每条件4条独立链")
    fig.text(.5, .012, "散点为链均值汇总；竖线为物理链重采样95%描述性区间；单尺寸不认证临界线", ha="center")
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output / "physical_observables.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharex=True, sharey=True)
    norm = matplotlib.colors.Normalize(vmin=float(metrics[:, 1].min()), vmax=float(metrics[:, 1].max()))
    for k, ax in enumerate(axes):
        pts = chains[:, :, k].reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], c=np.repeat(metrics[:, 1], 4), cmap="viridis", norm=norm,
                   s=11, alpha=.55, edgecolors="none")
        ax.set(xlabel=r"$g_1$", title=f"k={k}，尺寸 {[36,12,4][k]}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$g_2$")
    fig.suptitle("物理交叉区域在各尺度表示中的位置")
    fig.subplots_adjust(left=.07, right=.88, bottom=.15, top=.84, wspace=.17)
    bar_ax = fig.add_axes((.91, .22, .013, .56))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=bar_ax,
                 label=r"微观宾德累积量 $U_4$")
    fig.text(.5, .025, "每点是一条物理链的4次编码均值；颜色使用同一微观条件的物理统计", ha="center")
    fig.savefig(output / "physical_regions_in_g.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.3), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        ax.scatter(chains[..., k, 0].ravel(), chains[..., k, 1].ravel(), c="#c4cbd0", s=8, alpha=.45, edgecolors="none")
        for row, color in zip(summary["crossover_intervals"], colors):
            for marker, cid in zip(("o", "s"), row["cell_ids"]):
                p, q = g[cid, k], g[cid, k + 1]
                ax.scatter(*p, marker=marker, s=34, c=color, zorder=4)
                ax.quiver(p[0], p[1], q[0] - p[0], q[1] - p[1], angles="xy", scale_units="xy", scale=1,
                          color=color, width=.006, headwidth=4, headlength=5, zorder=3)
        ax.set(xlabel=r"$g_1$", title=f"实际配对位移：{[36,12][k]}→{[12,4][k]}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$g_2$")
    handles = [matplotlib.lines.Line2D([], [], color=c, marker="o", lw=0, label=f"u={u:g}") for u,c in zip(us, colors)]
    axes[1].legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("宾德下降最明显区间两端的实际粗粒化轨迹")
    fig.text(.5, .025, "圆点：区间低r端；方点：高r端；箭头为 g(k+1)−g(k)，端点为各条件均值", ha="center")
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output / "crossover_paired_trajectories.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run(source, cache, calibration, output):
    started, cpu_started = time.monotonic(), time.process_time()
    source, cache, calibration, output = map(lambda x: Path(x).resolve(), (source, cache, calibration, output))
    for root in (source, cache, calibration):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be separate from immutable inputs")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    paths = [source / "report/physics_summary.json", source / "report/analysis_arrays.npz",
        source / "run/evaluation_arrays.npz", cache / "cache.npz", cache / "manifest.json",
        calibration / "evaluation_arrays.npz", Path(__file__).resolve()]
    before = {str(p): digest(p) for p in paths}
    physics, sidecars, labels, moments, direct, chain_direct, source_g = read_inputs(source, cache, calibration)
    summary, arrays = make_summary(physics, labels, moments, direct, chain_direct)
    arrays["source_128_field_cell_g_mean"] = source_g
    output.mkdir(parents=True, exist_ok=False)
    plotting(arrays, summary, output)
    write(output / "summary.json", summary)
    np.savez_compressed(output / "analysis_arrays.npz", **arrays)
    after = {str(p): digest(p) for p in paths}
    if before != after:
        raise RuntimeError("Frozen input changed")
    receipt = dict(status="complete", workflow_complete=True, returncode=0, error=None,
        sources_sha256_before=before, sources_sha256_after=after, immutable_sources_unchanged=True,
        model_loaded=False, inference_performed=False, new_sampling=False, training_performed=False,
        device="cpu", gpu_seconds=0, wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        outputs_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write(output / "receipt.json", receipt)
    print(json.dumps({"output":str(output), "status":"complete", "crossover_intervals":summary["crossover_intervals"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.source, args.cache, args.calibration, args.output)
