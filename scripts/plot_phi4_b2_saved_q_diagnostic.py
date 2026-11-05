"""仅用已保存的公共评价数组检查 b=2 的 Q；不调用模型、采样或优化。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_hopping_qcr import load_public_run, _setup_plot


KAPPAS = np.asarray([.32, .50, .53, .56, .59, .62, .65, .68, .71, .74,
                     .77, .80, .83, .86, 1.04])
SIZES = [128, 64, 32, 16]
DATA_SHA = "3d2023b401b15e20a6dca6c6b602959c237a030ff0e2f8f89676a7ca9ed30fa5"
COLORS = ["#333333", "#0072B2", "#D55E00", "#009E73"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_average_ties(values):
    """One-based midranks, including exact ties; no scipy dependency."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranks require one finite vector")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end + 1) / 2
        start = end
    return result


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation requires matching finite vectors")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else None


def grouped_statistics(arrays):
    full = np.asarray(arrays["direct_full_g"], dtype=float)
    direct = np.asarray(arrays["direct_g"], dtype=float)
    if full.shape != direct.shape or full.shape != (60, 4, 1) or not np.isfinite(full).all():
        raise ValueError("Require sixty physical chains, four scales, one coordinate")
    labels = np.asarray(arrays["labels"], dtype=float)
    cells, splits = np.asarray(arrays["cell_ids"]), np.asarray(arrays["splits"])
    if labels.shape != (60, 3) or cells.shape != (60,) or splits.shape != (60,):
        raise ValueError("Physical chain metadata shape mismatch")
    rows = []
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() != 4 or len(set(splits[mask])) != 1 or not np.all(labels[mask] == labels[mask][0]):
            raise ValueError("Each condition must retain four chains with one split and label")
        if splits[mask][0] not in ("train", "test"):
            raise ValueError("Unknown split")
        values, halves = full[mask, :, 0], direct[mask, :, 0]
        rows.append(dict(cell=int(cell), kappa=float(labels[mask][0, 0]),
            split=str(splits[mask][0]), physical_chains=4,
            pooled_g_by_chain=values.tolist(),
            mean=values.mean(0).tolist(), chain_sd_ddof1=values.std(0, ddof=1).tolist(),
            chain_min=values.min(0).tolist(), chain_max=values.max(0).tolist(),
            half_encoder_mean=halves.mean(0).tolist(),
            half_encoder_g_by_chain=halves.tolist()))
    rows.sort(key=lambda row: row["kappa"])
    if len(rows) != 15 or not np.allclose([r["kappa"] for r in rows], KAPPAS, atol=1e-12, rtol=0):
        raise ValueError("All fifteen predeclared κ conditions are required")
    return rows


def descriptive_summary(rows, arrays):
    means = np.asarray([row["mean"] for row in rows])
    chain_values = np.asarray([row["pooled_g_by_chain"] for row in rows])
    low = np.concatenate([r["pooled_g_by_chain"] for r in rows if .50 - 1e-12 <= r["kappa"] <= .65 + 1e-12])
    high = np.concatenate([r["pooled_g_by_chain"] for r in rows if .71 - 1e-12 <= r["kappa"] <= .86 + 1e-12])
    by_scale = []
    for k in range(4):
        differences = np.diff(means[:, k])
        by_scale.append(dict(k=k, lattice_side=SIZES[k],
            pearson_condition_means=correlation(KAPPAS, means[:, k]),
            spearman_condition_means=correlation(rank_average_ties(KAPPAS), rank_average_ties(means[:, k])),
            pearson_physical_chains=correlation(np.repeat(KAPPAS, 4), chain_values[:, :, k].ravel()),
            spearman_physical_chains=correlation(rank_average_ties(np.repeat(KAPPAS, 4)), rank_average_ties(chain_values[:, :, k].ravel())),
            strictly_increasing_condition_means=bool((differences > 0).all()),
            strictly_decreasing_condition_means=bool((differences < 0).all()),
            positive_adjacent_differences=int((differences > 0).sum()),
            negative_adjacent_differences=int((differences < 0).sum()),
            condition_mean_span=float(np.ptp(means[:, k])),
            low_group_observed_chain_range=[float(low[:, k].min()), float(low[:, k].max())],
            high_group_observed_chain_range=[float(high[:, k].min()), float(high[:, k].max())],
            signed_high_min_minus_low_max=float(high[:, k].min() - low[:, k].max()),
            signed_low_min_minus_high_max=float(low[:, k].min() - high[:, k].max())))
    delta = np.asarray(arrays["direct_full_g"]) - np.asarray(arrays["direct_g"])
    first = np.asarray(arrays["direct_g"])[..., 0]
    one = np.asarray(arrays["one_step_g"])[..., 0]
    return dict(by_scale=by_scale,
        pooled_minus_half_mean_by_scale=delta.mean(0)[:, 0].tolist(),
        pooled_vs_half_rmse_by_scale=np.sqrt(np.mean(delta ** 2, axis=0))[:, 0].tolist(),
        pooled_vs_half_max_absolute_difference_by_scale=np.max(np.abs(delta), axis=0)[:, 0].tolist(),
        formal_half_encoder_one_step_rmse_by_transition=np.sqrt(np.mean((one - first[:, 1:]) ** 2, axis=0)).tolist(),
        low_group_definition="κ=0.50…0.65；仅事后描述分组，不作训练标签",
        high_group_definition="κ=0.71…0.86；仅事后描述分组，不作训练标签",
        observed_gap_note="正值表示本次观测链区间分离；负值不自动证明重叠，须同时考虑坐标方向；无置信区间或相变认证")


def reconstruction_summary(arrays):
    target = np.asarray(arrays["target_features"], dtype=float)
    recon = np.asarray(arrays["reconstructed_features"], dtype=float)
    splits = np.asarray(arrays["splits"])
    if target.shape != (60, 4, 625) or recon.shape != target.shape:
        raise ValueError("Expected sixty-chain, four-scale joint-probability arrays")
    if not np.isfinite(target).all() or not np.isfinite(recon).all() or not (splits == "train").any():
        raise ValueError("Invalid reconstruction arrays")
    # Descriptive constant predictor only; no new Q/decoder fit is performed.
    baseline = target[splits == "train"].mean(0)
    learned_by_chain = np.sqrt(np.mean((recon - target) ** 2, axis=-1))
    baseline_by_chain = np.sqrt(np.mean((baseline[None] - target) ** 2, axis=-1))
    result = dict(metric="625 个联合概率分量上的 RMSE；仅既有解码器，无新拟合",
        baseline="每尺度固定为该尺度 TRAIN 物理链目标均值；不使用 TEST 估计均值",
        learned_rmse_by_physical_chain_and_scale=learned_by_chain.tolist(),
        constant_train_mean_rmse_by_physical_chain_and_scale=baseline_by_chain.tolist(), splits={})
    for split in ("train", "test"):
        mask = splits == split
        if not mask.any():
            raise ValueError("Both splits are required")
        learned = np.sqrt(np.mean((recon[mask] - target[mask]) ** 2, axis=(0, 2)))
        constant = np.sqrt(np.mean((baseline[None] - target[mask]) ** 2, axis=(0, 2)))
        result["splits"][split] = dict(physical_chains=int(mask.sum()),
            learned_rmse_by_scale=learned.tolist(), constant_train_mean_rmse_by_scale=constant.tolist(),
            learned_over_constant_rmse_by_scale=[float(a / b) if b else None for a, b in zip(learned, constant)])
    return result


def decoder_geometry_summary(arrays):
    """Describe the saved affine direction and TRAIN target geometry, without a model.

    Recover b,d exactly from two cached decoder outputs, not by fitting a new
    decoder. TRAIN SVD is a descriptive reference: its minimum reconstruction
    error is a TRAIN affine-subspace bound, never a TEST error bound.
    """
    g = np.asarray(arrays["direct_g"], dtype=float)[..., 0]
    target = np.asarray(arrays["target_features"], dtype=float)
    recon = np.asarray(arrays["reconstructed_features"], dtype=float)
    train = np.asarray(arrays["splits"]) == "train"
    flat_g, flat_recon = g.ravel(), recon.reshape(-1, 625)
    lo, hi = int(np.argmin(flat_g)), int(np.argmax(flat_g))
    if flat_g[hi] == flat_g[lo]:
        raise ValueError("Cannot identify an affine direction from constant cached g")
    direction = (flat_recon[hi] - flat_recon[lo]) / (flat_g[hi] - flat_g[lo])
    bias = flat_recon[lo] - flat_g[lo] * direction
    norm2 = float(np.dot(direction, direction))
    if (not np.isclose(norm2, 1., atol=1e-9, rtol=1e-9)
            or not np.allclose(recon, g[..., None] * direction + bias, atol=1e-10, rtol=1e-8)):
        raise ValueError("Cache is not one shared unit-norm affine decoder")
    error = target - recon
    parallel_shift = np.einsum("esk,k->es", error, direction) / norm2
    optimal_g = g + parallel_shift
    tr = target[train].reshape(-1, 625)
    _, singular, vectors = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
    total = float(np.sum(singular**2))
    if not total > 0:
        raise ValueError("Require nonconstant TRAIN targets for geometry diagnostic")
    lower = float(np.sum(singular[1:]**2) / len(tr))
    actual = float(np.mean(np.sum(error[train]**2, axis=-1)))
    within = target[train] - target[train].mean(0, keepdims=True)
    variance_by_scale = np.sum(within**2, axis=(0, 2))
    per_scale = []
    for k in range(4):
        centered = within[:, k]
        _, sk, vk = np.linalg.svd(centered, full_matrices=False)
        per_scale.append(dict(k=k,
            TRAIN_target_PC1_variance_fraction=float(sk[0]**2 / np.sum(sk**2)),
            saved_decoder_abs_cosine_to_scale_PC1=float(abs(np.dot(direction, vk[0])) / np.sqrt(norm2)),
            share_of_summed_within_scale_TRAIN_target_variance=float(variance_by_scale[k] / variance_by_scale.sum())))
    splits = {}
    for split, mask in (("train", train), ("test", ~train)):
        squared = np.sum(error[mask]**2, axis=-1)
        perpendicular = np.maximum(squared - parallel_shift[mask]**2 * norm2, 0.)
        splits[split] = dict(
            residual_perpendicular_to_decoder_fraction_by_scale=[
                float(a / b) if b > 0 else None
                for a, b in zip(perpendicular.mean(0), squared.mean(0))],
            g_minus_fixed_decoder_optimum_rmse_by_scale=np.sqrt(np.mean(parallel_shift[mask]**2, axis=0)).tolist())
    return dict(
        diagnostic_only=True, no_model_updates=True, no_model_inference=True,
        saved_decoder_norm=float(np.sqrt(norm2)),
        saved_decoder_recovery="exact difference of two cached affine outputs; all rows verified",
        TRAIN_global_target_PC1_variance_fraction=float(singular[0]**2 / total),
        saved_decoder_abs_cosine_to_global_TRAIN_PC1=float(abs(np.dot(direction, vectors[0])) / np.sqrt(norm2)),
        TRAIN_actual_reconstruction_squared_L2=actual,
        TRAIN_optimal_1D_affine_subspace_squared_L2=lower,
        TRAIN_actual_over_affine_subspace_minimum=float(actual / lower) if lower else None,
        per_scale=per_scale, splits=splits,
        fixed_decoder_optimum_g_by_physical_chain_and_scale=optimal_g.tolist(),
        qualifications=["TRAIN SVD is a descriptive affine-subspace reference, not a new Q or a physical RG coordinate.",
                        "The TRAIN minimum ignores encoder constraints and is not a TEST lower bound.",
                        "Target variance refers to the saved finite 625-bin local descriptor, not all physical information."])


def _load_endpoint(run, early):
    run = Path(run).resolve()
    manifest, summary, _, arrays, pins = load_public_run(run)
    status_path = run / "status.json"
    status_sha = sha256(status_path)
    status = json.loads(status_path.read_text())
    whitelist = {item["path"]: item for item in status.get("public_files", [])}
    if len(whitelist) != len(status.get("public_files", [])):
        raise ValueError("Duplicate endpoint whitelist entries")
    if (status.get("status") != "complete" or status.get("returncode") != 0
            or status.get("workflow_complete") is not True or status.get("error")
            or status.get("requires_user_action") is not False):
        raise ValueError("Complete, successful endpoint required")
    for name, digest in pins.items():
        if name not in whitelist or whitelist[name].get("sha256") != digest or whitelist[name].get("bytes") != (run / name).stat().st_size:
            raise ValueError("Public input differs from endpoint whitelist")
    if status.get("evaluation_arrays_sha256") != pins["evaluation_arrays.npz"] or status.get("summary_sha256") != pins["summary.json"]:
        raise ValueError("Endpoint public hash mismatch")
    if (manifest.get("lattice_sizes") != SIZES or manifest.get("model", {}).get("theory_dim") != 1
            or manifest.get("block_size") != 2 or manifest.get("stride") != 2
            or manifest.get("data_manifest_sha256") != DATA_SHA
            or summary.get("data_manifest_sha256") != DATA_SHA
            or summary.get("data_warning_count") != 75):
        raise ValueError("Require the frozen b2 L128 scalar experiment and its warnings")
    if early:
        if summary.get("actual_steps") != 140 or summary.get("evaluated_checkpoint_step") != 120:
            raise ValueError("Expected actual 140 updates, selected checkpoint 120")
        title = "早停选中 120 步（实际更新 140 步）"
    else:
        if summary.get("actual_steps") != 3000 or summary.get("early_stopping") is not None:
            raise ValueError("Expected the original 3000-step endpoint")
        title = "正式终点 3000 步"
    rows = grouped_statistics(arrays)
    pins["status.json"] = status_sha
    return dict(run=run, title=title, manifest=manifest, summary=summary,
                arrays=arrays, pins=pins, conditions=rows)


def draw_g_by_scale(plt, runs, output):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for rindex, run in enumerate(runs):
        rows = run["conditions"]
        means, sd = np.asarray([r["mean"] for r in rows]), np.asarray([r["chain_sd_ddof1"] for r in rows])
        for k, ax in enumerate(axes[rindex]):
            ax.errorbar(KAPPAS, means[:, k], yerr=sd[:, k], color=COLORS[k], lw=1.5, capsize=2)
            for row in rows:
                values = np.asarray(row["pooled_g_by_chain"])[:, k]
                ax.scatter(np.full(4, row["kappa"]), values, s=14, color=COLORS[k],
                           marker="o" if row["split"] == "train" else "x", alpha=.65)
            ax.set(title=f"{run['title']}\nk={k}，{SIZES[k]}×{SIZES[k]}",
                   xlabel="物理耦合 κ（仅事后评价）", ylabel="直接编码 g")
            ax.grid(alpha=.18)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Q 对不同物理条件的识别：各尺度分别放大观察", fontsize=16)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.text(.5, .072, "每个点为一条物理链的 128 张构型整体编码；实心圆为训练条件，叉号为留出条件；曲线为四链均值±链间标准差。", ha="center", fontsize=10)
    fig.text(.5, .044, "各面板纵轴独立，跨模型／尺度不比较绝对幅度；线段仅连接实测 κ，不拟合或认证临界点。", ha="center", fontsize=10)
    fig.text(.5, .016, "仅复用正式缓存；一次固定 C 随机序列，无新增推理；原有 75 项采样质量警告保留。误差棒不是置信区间。", ha="center", fontsize=9.5)
    fig.savefig(output / "Q_g_kappa_by_scale.png", dpi=170)
    plt.close(fig)


def draw_trajectories(plt, runs, output):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6))
    norm, cmap = Normalize(KAPPAS.min(), KAPPAS.max()), plt.cm.viridis
    for ax, run in zip(axes, runs):
        for row in run["conditions"]:
            values = np.asarray(row["pooled_g_by_chain"])
            color = cmap(norm(row["kappa"]))
            ax.plot(np.arange(4), values.T, color=color, lw=.6, alpha=.15)
            ax.errorbar(np.arange(4), row["mean"], yerr=row["chain_sd_ddof1"],
                        color=color, marker="o" if row["split"] == "train" else "x", lw=1.5, capsize=2, ms=4)
        ax.set(title=run["title"], xlabel="粗粒化次数 k（仅实测点）", ylabel="直接编码 g",
               xticks=np.arange(4), xticklabels=["0\n128×128", "1\n64×64", "2\n32×32", "3\n16×16"])
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.subplots_adjust(left=.075, right=.88, bottom=.23, top=.86, wspace=.25)
    cax = fig.add_axes([.9, .25, .018, .57])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label("物理耦合 κ（评价标签）")
    fig.suptitle("同一 Q 在 128→64→32→16 上的表示轨迹", fontsize=16)
    fig.text(.5, .115, "每个模型内四尺度使用同一纵轴；两模型纵轴独立。细线为物理链，粗线为四链均值±链间标准差。", ha="center", fontsize=10)
    fig.text(.5, .075, "圆点为训练条件，叉号为留出条件；全部十五组 κ 同时保留。线段不代表连续尺度或 RG 轨迹插值。", ha="center", fontsize=10)
    fig.text(.5, .033, "整体系综编码；仅一次固定 C 随机序列。链间标准差混合物理采样和 C 随机性，不是置信区间。", ha="center", fontsize=9.5)
    fig.savefig(output / "Q_scale_trajectories.png", dpi=170)
    plt.close(fig)


def draw_direct_vs_r(plt, runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3))
    for ax, run in zip(axes, runs):
        arrays = run["arrays"]
        grid, mapped = arrays["flow_grid"][:, 0], arrays["flow_next"][:, 0]
        ax.plot(grid, mapped, color="black", lw=1.8, label="保存的 R(g)")
        ax.plot(grid, grid, color=".55", ls="--", label="恒等映射")
        direct, splits = arrays["direct_g"][:, :, 0], np.asarray(arrays["splits"])
        for k in range(3):
            for split, marker, text in (("train", "o", "训练"), ("test", "x", "留出")):
                mask = splits == split
                ax.scatter(direct[mask, k], direct[mask, k + 1], color=COLORS[k + 1],
                           s=23, marker=marker, alpha=.7, label=f"实测 {k}→{k+1}：{text}")
        ax.set(title=run["title"], xlabel="当前尺度直接编码 g", ylabel="下一尺度直接编码／R(g)")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    fig.suptitle("Q 的实测一步关系与已保存的 R(g) 对照", fontsize=16)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.86, wspace=.25)
    fig.text(.5, .115, "散点使用正式闭环口径：两半系综分别经 Q 编码后取均值，不是 128 张构型整体编码。每条物理链含三组一步关系。", ha="center", fontsize=10)
    fig.text(.5, .074, "黑线仅复用正式评价保存的 R(g)；散点为 Q 实测下一尺度，不是 R 预测。未加载或调用 Q、C、R。", ha="center", fontsize=10)
    fig.text(.5, .032, "低闭环误差不等于物理不动点正确；本图不搜索不动点、不拟合临界 κ、不计算临界指数。", ha="center", fontsize=10)
    fig.savefig(output / "Q_direct_vs_R.png", dpi=170)
    plt.close(fig)


def render(early_run, final_run, output, font=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("New diagnostic directory required; refusing to overwrite")
    runs = [_load_endpoint(early_run, True), _load_endpoint(final_run, False)]
    if runs[0]["manifest"]["data_shards"] != runs[1]["manifest"]["data_shards"]:
        raise ValueError("Endpoint data shard pins differ")
    for key in ("labels", "cell_ids", "splits"):
        if not np.array_equal(runs[0]["arrays"][key], runs[1]["arrays"][key]):
            raise ValueError("Physical chain metadata differs between endpoints")
    result = dict(schema_version=1, no_model_inference=True, no_sampling=True, no_training=True,
        saved_decoder_unchanged=True, no_optimizer_updates=True,
        TRAIN_SVD_is_descriptive_reference_only=True, lattice_sizes=SIZES, data_manifest_sha256=DATA_SHA,
        plotted_q_encoding="Q of all 128 saved configurations in each physical chain",
        formal_r_encoding="mean of separately encoded first and second half ensembles",
        uncertainty="four-chain descriptive sample standard deviation, ddof=1; not confidence intervals",
        independent_c_repeats=1, original_data_warning_count=75, equilibrium_certified=False,
        critical_kappa=None, thermal_nu=None, fixed_point_search_performed=False,
        script_sha256=sha256(Path(__file__)), runs=[])
    for run in runs:
        result["runs"].append(dict(source_run=str(run["run"]), title=run["title"],
            actual_steps=run["summary"]["actual_steps"],
            evaluated_checkpoint_step=run["summary"].get("evaluated_checkpoint_step", run["summary"]["actual_steps"]),
            final_module_sha256=run["summary"]["final_module_sha256"],
            source_public_sha256=run["pins"], frozen_source_sha256=run["manifest"]["source_sha256"],
            c_evaluation_seed=run["manifest"]["evaluation_seed"], conditions=run["conditions"],
            metrics=descriptive_summary(run["conditions"], run["arrays"]),
            reconstruction=reconstruction_summary(run["arrays"]),
            decoder_geometry=decoder_geometry_summary(run["arrays"])))
    plt = _setup_plot(font)
    output.mkdir(parents=True, exist_ok=False)
    draw_g_by_scale(plt, runs, output)
    draw_trajectories(plt, runs, output)
    draw_direct_vs_r(plt, runs, output)
    for run in runs:
        if run["pins"] != {name: sha256(run["run"] / name) for name in run["pins"]}:
            raise RuntimeError("Public source changed while plotting")
    result["plot_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.png"))}
    (output / "Q_diagnostic_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-run", required=True)
    parser.add_argument("--final-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    result = render(args.early_run, args.final_run, args.output, args.font)
    print(json.dumps({"output": str(Path(args.output).resolve()), "plots": result["plot_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
