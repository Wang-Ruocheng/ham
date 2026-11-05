"""Compare completed scalar phi4 QCR endpoints using public JSON/NPZ only.

python -m scripts.report_phi4_qcr_mean_init_comparison \
  --old-run OLD --new-run NEW --output NEW_REPORT \
  [--sampling-quality PUBLIC_QUALITY_JSON]

No model/checkpoint/configuration loading, inference, sampling, MI evaluation,
new root search or critical-exponent estimation occurs here. Latent coordinates
are deliberately not identified across the independently trained endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_gaussian_scalar_qcr import (
    _optional_feature_arrays, prediction_metrics, rank_correlations,
    reconstruction_metrics,
)
from scripts.report_phi4_qcr import (
    INPUT_FILES, _json, _plotting, history_rows, load_public_run, sha256,
)


RUN_LABELS = {"old": "原随机偏置", "new": "训练均值偏置"}
LIMITATIONS = [
    "只比较同一物理数据与留出划分的两个 seed91、3000 次更新端点；不是多种子鲁棒性检验。",
    "两次联合训练的 g 坐标不同，不直接比较原始 g 的误差、数值根位置或导数大小来判断物理好坏。",
    "潜坐标闭合分别使用各次运行的原 TRAIN 同目标尺度总体标准差归一，并与各自恒等及 TRAIN 均值基线比较。",
    "共同的 105 维原生软局部统计用于评价重构和预测；逐尺度核对目标数组，相同目标才进行逐链配对。",
    "C 的梯度与 Q/R 独立，本次训练随机数保持一致；若粗尺度目标也数值相同，可比较其统计预测误差，但仍不能直接认证物理相变。",
    "仅展示正式端点已保存的数值根与 R(g)-g 网格，不搜索新根，不据此认证相变或热方向。",
    "原采样 ESS、热冷初态和相关链警告全部保留；改善 Q 初始化不修复采样误差，也不认证平衡。",
    "有限基、有限尺寸、固定 u 的单参数族和一维 g 不保证 RG 闭合；critical_r 与 thermal_nu 始终为 null。",
]


def _same(old, new, key):
    if key not in old or key not in new or old[key] != new[key]:
        raise ValueError(f"Paired runs disagree in {key}")


def _model_source(manifest):
    values = [digest for path, digest in manifest.get("source_sha256", {}).items()
              if Path(path).as_posix().endswith("recurrent_rg/phi4_qcr.py")]
    if len(values) != 1:
        raise ValueError("Each run must pin exactly one continuous QCR model source")
    return values[0]


def validate_pair(old, new):
    """Check the one-initialization-change comparison, without tensor/model I/O."""
    a, b = old.manifest, new.manifest
    if old.root == new.root:
        raise ValueError("Old and new runs must be distinct")
    for run in (old, new):
        if (run.manifest.get("schema_version") != 5
                or run.summary.get("actual_steps") != 3000
                or run.manifest.get("options", {}).get("steps") != 3000
                or run.manifest.get("options", {}).get("seed") != 91):
            raise ValueError("Comparison requires completed seed91, 3000-update endpoints")
    for key in ("recipe", "data_manifest_sha256", "data_shards", "theory_dim", "hidden",
                "lattice_sizes", "block_size", "stride", "dtype", "scale", "scale_definition",
                "learning_rate", "learning_rate_policy", "optimizer", "closure_objective",
                "closure_target_scales", "closure_implementation", "closure_scale_weights",
                "closure_branch_normalization", "gradient_routing", "labels_used_in_training",
                "model_input_members", "evaluation_label_columns", "evaluation_seed",
                "evaluation_encoding", "evaluation_sample_count", "training_sample_count",
                "training_encoding", "data_warning_count", "sampler_schema", "sample_semantics"):
        _same(a, b, key)
    for key in ("steps", "system_batch", "sample_batch", "seed", "threads", "log_every",
                "checkpoint_every", "information_pairs"):
        _same(a["options"], b["options"], key)
    if (a["theory_dim"] != 1 or a["dtype"] != "float64" or a["lattice_sizes"] != [36, 12, 4]
            or a["labels_used_in_training"] is not False):
        raise ValueError("Expected unsupervised float64 scalar 36→12→4 geometry")
    if _model_source(a) != _model_source(b):
        raise ValueError("Q/C/R architecture and loss model source must be unchanged")
    if a.get("q_initialization", "random_legacy") != "random_legacy":
        raise ValueError("Old run must use the original random-bias initialization")
    if b.get("q_initialization") != "train_mean_bias":
        raise ValueError("New run must use TRAIN mean-bias initialization")
    constructor = b.get("random_construction_state_sha256")
    if not constructor or constructor != a.get("initial_state_sha256"):
        raise ValueError("New random constructor must match old complete initial Q/C/R state")
    if old.summary.get("initial_state_sha256") != constructor:
        raise ValueError("Old initial-state summary disagrees with its manifest")
    receipt = b.get("q_initialization_receipt", {})
    if (new.summary.get("q_initialization_receipt") != receipt
            or receipt.get("mode") != "train_mean_bias"
            or receipt.get("constructor_state_sha256") != constructor
            or b.get("training_start_state_sha256") != receipt.get("training_start_state_sha256")
            or b.get("initial_state_sha256") != receipt.get("training_start_state_sha256")
            or new.summary.get("initial_state_sha256") != b.get("initial_state_sha256")):
        raise ValueError("Mean-bias initialization state receipts disagree")
    before, after = receipt.get("constructor_module_sha256", {}), receipt.get("training_start_module_sha256", {})
    if (set(before) != {"Q", "C", "R"} or set(after) != set(before)
            or before["C"] != after["C"] or before["R"] != after["R"]
            or receipt.get("constructor_protected_sha256") != receipt.get("training_start_protected_sha256")
            or not receipt.get("constructor_protected_sha256")
            or b.get("random_construction_module_sha256") != before
            or b.get("training_start_module_sha256") != after):
        raise ValueError("Initialization changed protected C/R/encoder/decoder-direction parameters")
    if (any(receipt.get(key) is not False for key in
            ("labels_used", "test_data_used", "pca_initialization", "warm_start", "kernel_selection"))
            or receipt.get("optimizer_updates") != dict(Q=0, C=0, R=0)
            or receipt.get("global_cpu_rng_unchanged") is not True
            or receipt.get("global_cuda_rng_unchanged") is False
            or receipt.get("all_parameter_gradients_none") is not True
            or receipt.get("C_trainable") is not True
            or receipt.get("training_batch_rng_created_after_initialization") is not True
            or receipt.get("training_field_rng_created_after_initialization") is not True
            or receipt.get("generator_seed") != 9091
            or receipt.get("scales") != [0, 1, 2]
            or receipt.get("scale_weighting") != "uniform"
            or receipt.get("physical_ensemble_weighting") != "uniform"):
        raise ValueError("Initialization must be label-free, TRAIN-only, zero-update and RNG-isolated")
    n_train = len(new.arrays["train_direct"])
    samples = b["evaluation_sample_count"]
    if (receipt.get("physical_ensemble_count") != n_train
            or receipt.get("fields_per_ensemble") != samples
            or receipt.get("total_fields") != n_train * samples
            or len(receipt.get("mean", [])) != 105
            or not np.isfinite(receipt["mean"]).all()
            or receipt.get("bias_after_sha256") != receipt.get("mean_sha256")
            or not receipt.get("mean_sha256")):
        raise ValueError("TRAIN mean-bias calibration counts or feature geometry disagree")
    for split in ("train", "test"):
        for key in ("labels", "cell_ids"):
            if not np.array_equal(old.arrays[f"{split}_{key}"], new.arrays[f"{split}_{key}"]):
                raise ValueError("Physical chain order, labels or TRAIN/TEST cells are not paired")
    return dict(data_manifest_sha256=a["data_manifest_sha256"], shared_random_constructor_sha256=constructor,
                model_source_sha256=_model_source(a), data_and_chain_order_paired=True,
                initial_C_R_encoder_decoder_direction_unchanged=True,
                sole_intended_change="TRAIN mean affine-decoder bias initialization",
                training_endpoint=3000, seed=91, initialization_receipt=receipt)


def sampling_warnings(path, manifests):
    """Count unique original chain/condition warnings, not warning-string records."""
    result = dict(recorded_data_warning_strings=manifests[0]["data_warning_count"],
                  equilibrium_certified=False, original_warnings_retained=True,
                  note="警告字符串条数不是独立异常物理链数；未读取物理构型或重算 ESS。")
    if path is None:
        result.update(public_quality_summary_loaded=False, unique_low_ess_chains=None,
                      unique_hot_cold_warning_conditions=None)
        return result, None
    path = Path(path).resolve()
    if path.suffix != ".json":
        raise ValueError("Sampling warning source must be public JSON, not physical data")
    digest, quality = sha256(path), _json(path)
    if (quality.get("data_manifest_sha256") is not None
            and quality["data_manifest_sha256"] != manifests[0].get("data_manifest_sha256")):
        raise ValueError("Sampling quality belongs to different physical data")
    count = quality.get("warning_count", quality.get("source_warning_count"))
    if count is not None and count != manifests[0]["data_warning_count"]:
        raise ValueError("Public quality warning count disagrees with paired data")
    if "conditions" in quality and "issues" not in quality:
        conditions = quality["conditions"]
        low = {(row["cell_index"], chain["chain_index"]) for row in conditions
               for chain in row.get("chain_warnings", [])
               if any("ESS below" in warning for warning in chain.get("warnings", []))}
        hotcold = {row["cell_index"] for row in conditions if row.get("hot_cold_warning_observables")}
        total = sum(len(row.get("chain_warnings", [])) for row in conditions)
    else:
        issues = quality.get("issues")
        if not isinstance(issues, list):
            raise ValueError("Public sampling quality JSON has no original warning records")
        low = {(row["cell_index"], row["chain_index"]) for row in issues
               if row.get("kind") == "existing_ess_below32_or_unavailable"}
        hotcold = {row["cell_index"] for row in issues if row.get("kind") == "existing_hot_cold_three_se_warning"}
        total = sum(len(row.get("chains", [])) for row in quality.get("conditions", [])) or None
    result.update(public_quality_summary_loaded=True, unique_low_ess_chains=len(low),
                  unique_low_ess_chain_ids=[list(value) for value in sorted(low)],
                  unique_hot_cold_warning_conditions=len(hotcold), total_physical_chains=total,
                  original_public_quality=quality)
    if sha256(path) != digest:
        raise ValueError("Public sampling warning JSON changed during reading")
    return result, (path, digest)


def _metrics(run):
    available = _optional_feature_arrays(run.arrays)
    if not all(available.values()) or run.arrays["train_feature_target"].shape[-1] != 105:
        raise ValueError("All saved 105-dimensional native targets/reconstructions/predictions are required")
    rows = history_rows(run.history)
    return dict(actual_steps=run.summary["actual_steps"], optimizer_steps=run.summary["optimizer_steps"],
                q_reconstruction=reconstruction_metrics(run.arrays, available),
                latent_closure=prediction_metrics(run.arrays),
                native_statistic_prediction=prediction_metrics(run.arrays, features=True, available=available),
                r_information=rank_correlations(run.arrays),
                saved_numerical_roots=run.summary.get("numerical_fixed_points", run.summary.get("roots", [])),
                saved_flow_range=[float(run.arrays["flow_g"].min()), float(run.arrays["flow_g"].max())],
                saved_displacement_range=[float(run.arrays["flow_displacement"].min()), float(run.arrays["flow_displacement"].max())],
                history_first=rows[0], history_last=rows[-1],
                initial_state_sha256=run.summary["initial_state_sha256"],
                final_state_sha256=run.summary.get("final_state_sha256"), timing=run.summary.get("timing"),
                critical_r=None, thermal_nu=None, physical_critical_point_certified=False)


def _paired_native(old, new):
    records = []
    for split in ("train", "test"):
        a, b = old.arrays[f"{split}_feature_target"], new.arrays[f"{split}_feature_target"]
        if a.shape != b.shape:
            raise ValueError("Native feature target dimensions disagree")
        for scale in range(a.shape[1]):
            difference = b[:, scale] - a[:, scale]
            same = bool(np.allclose(a[:, scale], b[:, scale], atol=1e-14, rtol=0))
            if scale == 0 and not same:
                raise ValueError("Microscopic feature targets must match for the paired physical data")
            records.append(dict(split=split, scale=scale, target_max_abs_difference=float(np.max(np.abs(difference))),
                                targets_numerically_equal=same, targets_exactly_equal=bool(np.array_equal(a[:, scale], b[:, scale])),
                                common_saved_statistic_target=same,
                                coarse_target_depends_on_learned_C=scale > 0))
    paired = {}
    for split in ("train", "test"):
        paired[split] = []
        for scale in range(3):
            record = next(row for row in records if row["split"] == split and row["scale"] == scale)
            if not record["targets_numerically_equal"]:
                continue
            target = old.arrays[f"{split}_feature_target"][:, scale]
            old_error = old.arrays[f"{split}_feature_reconstruction"][:, scale] - target
            new_error = new.arrays[f"{split}_feature_reconstruction"][:, scale] - target
            per_chain = np.mean(new_error**2, axis=-1) - np.mean(old_error**2, axis=-1)
            paired[split].append(dict(scale=scale, new_minus_old_per_physical_chain_mse=per_chain.tolist(),
                mean_new_minus_old_mse=float(per_chain.mean()),
                fraction_physical_chains_with_lower_new_mse=float(np.mean(per_chain < 0)),
                bootstrap_performed=False))
    return dict(target_comparability_by_scale=records, reconstruction_pairing_on_matching_targets=paired,
                all_saved_native_targets_equal=all(row["targets_numerically_equal"] for row in records))


def plot_reconstruction(metrics, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for name, color, offset in (("old", "#a16a98", -.18), ("new", "#237b9b", .18)):
        rows = [row for row in metrics[name]["q_reconstruction"]["rows"] if row["split"] == "test"]
        for ax, key in zip(axes, ("absolute_rmse", "fixed_train_nrmse")):
            ax.bar(np.arange(3) + offset, [row["reconstruction"][key] for row in rows], .34,
                   label=RUN_LABELS[name], color=color)
            ax.plot(np.arange(3) + offset, [row["same_scale_train_mean_constant"][key] for row in rows],
                    "x", color=color, markersize=8, label=f"{RUN_LABELS[name]}：各自训练均值基线")
    for ax in axes:
        ax.set_xticks(range(3), ["微观 k=0", "粗尺度 k=1", "粗尺度 k=2"])
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set(ylabel="原生统计绝对 RMSE", title="留出条件 Q 重构")
    axes[1].set(ylabel="各自原 TRAIN 目标标准差 NRMSE", title="按各自目标离散度归一")
    fig.suptitle("Q 初始化修正：原生统计重构误差，配对有效性由目标数组逐尺度核验")
    fig.savefig(output / "Q_reconstruction_comparison.png", dpi=180)
    plt.close(fig)


def plot_flow(runs, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    kinds = {"attracting": "数值吸引", "repelling": "数值排斥", "neutral": "数值中性"}
    for (name, run), ax in zip(runs.items(), axes):
        a = run.arrays
        g, drift = a["flow_g"], a["flow_displacement"]
        ax.plot(g, drift, color="#234f9c", label="保存的 R(g)−g")
        ax.axhline(0, linestyle="--", color=".45", linewidth=1)
        indices = np.unique(np.linspace(0, len(g) - 1, min(18, len(g))).astype(int))
        ax.quiver(g[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
                  angles="xy", scale_units="xy", scale=1, width=.004, color="#e47e30")
        support = a["train_direct"][:, :-1, 0].ravel()
        ax.axvspan(support.min(), support.max(), alpha=.13, color="#6caed6", label="各自训练输入范围")
        roots = run.summary.get("numerical_fixed_points", run.summary.get("roots", []))
        for index, root in enumerate(roots):
            ax.scatter(root["g"], 0, marker="*", s=110, edgecolors="#ac2663", facecolors="white", zorder=5,
                       label="正式保存数值根（未认证物理）" if index == 0 else None)
            label = kinds.get(root.get("classification"), "保存数值根")
            ax.annotate(label, (root["g"], 0), xytext=(4, 8), textcoords="offset points", fontsize=8)
        if not roots:
            ax.text(.03, .95, "正式结果未保存数值根", transform=ax.transAxes, va="top", fontsize=9)
        ax.set(xlabel="各自的 g 坐标", ylabel="位移 R(g)−g", title=RUN_LABELS[name])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("两次联合训练使用独立坐标轴：仅展示已保存流场和数值根，不估计相变或指数")
    fig.savefig(output / "R_vs_g_comparison.png", dpi=180)
    plt.close(fig)


def plot_coupling(runs, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    colors = ["#237b9b", "#d08231", "#9467a8"]
    for (name, run), ax in zip(runs.items(), axes):
        a = run.arrays
        for scale, color in enumerate(colors):
            for split, marker, style, label in (("train", "o", "-", "训练"), ("test", "s", "--", "留出")):
                ids, r, values = a[f"{split}_cell_ids"], a[f"{split}_labels"][:, 0], a[f"{split}_direct"][:, scale, 0]
                ax.scatter(r, values, color=color, marker=marker, s=12, alpha=.22)
                means = sorted((float(r[ids == cell][0]), float(values[ids == cell].mean())) for cell in np.unique(ids))
                ax.plot([value[0] for value in means], [value[1] for value in means],
                        marker=marker, linestyle=style, markersize=4, color=color, label=f"k={scale}，{label}条件均值")
        ax.set(xlabel="生成参数 r（仅评价）", ylabel="各自直接编码 g", title=RUN_LABELS[name])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("耦合到表示：三尺度共同展示；两次运行的 g 不能直接视为同一物理坐标")
    # Distinct from R_vs_g on case-insensitive macOS filesystems.
    fig.savefig(output / "coupling_by_scale_comparison.png", dpi=180)
    plt.close(fig)


def plot_losses(runs, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(13.1, 8), constrained_layout=True)
    groups = [("Q 表示损失", [("q_loss", None, "总损失"), ("reconstruction", "q_reconstruction", "统计重构"),
                            ("subset_micro", "q_subset_micro", "微观子集"), ("subset_coarse", "q_subset_coarse", "粗尺度子集")]),
              ("C 粗粒化损失", [("c_loss", None, "总损失"), ("information_proxy", "c_information", "平滑信息代理"),
                             ("variance_floor", "c_variance", "方差下限"), ("uniform_amplitude", "c_uniform", "均匀块保持")]),
              ("R 闭合损失", [("r_loss", None, "总损失"), ("closure_one_step", "r_one_step", "单步"),
                           ("closure_rollout", "r_rollout", "微观起点多步")])]
    for row_index, (name, run) in enumerate(runs.items()):
        rows = history_rows(run.history)
        for ax, (title, entries) in zip(axes[row_index], groups):
            for key, alias, label in entries:
                points = [(row["step"], row.get(key, row.get(alias))) for row in rows
                          if isinstance(row.get(key, row.get(alias)), (int, float)) and np.isfinite(row.get(key, row.get(alias)))]
                if points:
                    ax.plot(*zip(*points), label=label)
            ax.set(xlabel="训练更新次数", ylabel="已保存损失值", title=f"{RUN_LABELS[name]}：{title}")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(alpha=.2)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
    fig.suptitle("3000 次更新完整曲线：损失定义未改；仅读取历史记录，不重算 MI")
    fig.savefig(output / "loss_curves_comparison.png", dpi=180)
    plt.close(fig)


def write_report(old_run, new_run, output, *, sampling_quality=None, font=None):
    runs = dict(old=load_public_run(old_run), new=load_public_run(new_run))
    pair = validate_pair(runs["old"], runs["new"])
    metrics = {name: _metrics(run) for name, run in runs.items()}
    native_pairing = _paired_native(runs["old"], runs["new"])
    warnings, quality_source = sampling_warnings(sampling_quality, [run.manifest for run in runs.values()])
    output = Path(output).resolve()
    for run in runs.values():
        if output == run.root or run.root.is_relative_to(output):
            raise ValueError("Output must not contain or replace either source run")
    if output.exists():
        raise FileExistsError(f"Comparison directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_reconstruction(metrics, output, plt)
    plot_flow(runs, output, plt)
    plot_coupling(runs, output, plt)
    plot_losses(runs, output, plt)
    comparison = dict(schema_version=1, status="complete", experiment="phi4_joint_QCR_mean_bias_initialization_comparison",
        pair_validation=pair, runs=metrics, native_target_pairing=native_pairing, sampling_warnings=warnings,
        latent_coordinates_identified_across_runs=False, latent_raw_error_cross_run_ranking_performed=False,
        normalization="Each run's immutable same-scale TRAIN target population std; ddof=0",
        critical_r=None, thermal_nu=None, equilibrium_certified=False, physical_critical_point_certified=False,
        root_search_performed=False, limitations=LIMITATIONS)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Forward only the original public arrays, namespaced; never model payloads.
    arrays = {f"{name}_{key}": value for name, run in runs.items() for key, value in run.arrays.items()}
    np.savez_compressed(output / "comparison_arrays.npz", **arrays)
    after = {name: {file: sha256(run.root / file) for file in INPUT_FILES} for name, run in runs.items()}
    if any(after[name] != run.fingerprints for name, run in runs.items()):
        raise ValueError("Public run sources changed during comparison generation")
    if quality_source and sha256(quality_source[0]) != quality_source[1]:
        raise ValueError("Public sampling warnings changed during comparison generation")
    digests = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = dict(schema_version=1, status="complete", sources={name: str(run.root) for name, run in runs.items()},
        sources_sha256_before={name: run.fingerprints for name, run in runs.items()}, sources_sha256_after=after,
        sampling_quality_source=(dict(path=str(quality_source[0]), sha256=quality_source[1]) if quality_source else None),
        public_sources_unchanged=True, model_loaded=False, physical_data_loaded=False, GPU_used=False,
        inference_performed=False, root_search_performed=False, MI_reestimated=False,
        outputs_sha256=digests, public_files=sorted([*digests, "report.json", "status.json"]),
        critical_r=None, thermal_nu=None, equilibrium_certified=False, limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    status = dict(status="complete", workflow_complete=True, returncode=0, error=None, requires_user_action=False,
                  comparison_sha256=sha256(output / "comparison.json"), report_sha256=sha256(output / "report.json"),
                  critical_r=None, thermal_nu=None, equilibrium_certified=False)
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--new-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampling-quality", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.old_run, args.new_run, args.output, sampling_quality=args.sampling_quality, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
