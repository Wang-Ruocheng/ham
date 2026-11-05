"""Public-array-only report for the three separately controlled phi4 checks.

No checkpoints, physical configurations or model code are loaded.  In
particular, saved flow samples are plotted without a new root search.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scripts.report_gaussian_scalar_qcr import (
    _optional_feature_arrays, prediction_metrics, rank_correlations,
    reconstruction_metrics,
)
from scripts.report_phi4_b2_closure_comparison import (
    INPUT_FILES, SIZES, _digest, _finite_json, _json, _modules, _plotting,
    _root_support, _validate_arrays, load_public_comparison, sha256,
)


RECIPE = "phi4_b2_single_factor_validation_v1"
STAGE_ARMS = {"joint": ("cubic_decoder", "odd_C"), "r_tail": ("r_tail",)}
NAMES = ("baseline", "cubic_decoder", "odd_C", "r_tail")
LABELS = {"baseline": "原共享仿射基准", "cubic_decoder": "仅改三阶解码器",
          "odd_C": "仅加 C 奇对称", "r_tail": "仅追加 R 收尾"}
STEPS = 3000
LIMITATIONS = [
    "明确授权的二倍粗粒化三项单因素检验，不改变默认配方；所有源实验与物理数据保持不变。",
    "三阶解码与奇对称 C 分别从原实验共同初始化重新联合训练 3000 步；R 收尾从原 all-start 端点追加 R 更新 3000 步，Q/C 零更新。",
    "原三阶段 Q/C/R 各 3000 次与 R 追加 3000 次分开报告；R 收尾不是 6000 次全 QCR 联合训练。",
    "三阶解码与原基准的 C 输出原生统计目标相同，可以比较同一 105 维目标；其 g 坐标不同，禁止跨坐标直接排名 g 误差。",
    "奇对称 C 会改变粗粒化后的目标统计；其原生统计误差仅对本臂 C 输出成立，不是同物理目标的因果优劣排名。",
    "每臂报告绝对 RMSE 与本臂 TRAIN 同目标尺度总体标准差 NRMSE，并列源尺度恒等预测及 TRAIN 同尺度均值常数基线。",
    "全体误差使用保存的固定端点、物理链及条件留出；没有重新采样、模型推理、互信息估计、根搜索或临界点插值。",
    "C 诊断来自既有固定评价中被动观察的场输出；原 C 参照取三阶解码臂，与 R 收尾臂逐元素一致，不是额外运行原基准。",
    "源数据 35 条质量警告照实保留，有限 ESS 与热冷检查不认证平衡；不因结果自动补样、换种子或改变训练目标。",
    "固定 u、有限尺寸 32→16→8→4、单训练种子和有限链；局域统计预测不保证完整分布或长程物理正确。",
    "不由保存流场的过零认证物理不动点；不选热模、不估计临界指数，critical_r 与 thermal_nu 始终为空。",
]


@dataclass
class PublicStage:
    root: Path
    stage: str
    manifest: dict
    summary: dict
    status: dict
    history: dict
    endpoints: dict
    fingerprints: dict


@dataclass
class Validation:
    source: object
    stages: dict
    endpoints: dict


def _source_training(path):
    path = Path(path).resolve()
    return path / "training" if (path / "training" / "manifest.json").is_file() else path


def _fingerprints(root):
    return {name: sha256(root / name) for name in INPUT_FILES}


def _history_check(history, arms):
    rows = history.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError("History must have saved progress rows")
    previous = -1
    for row in rows:
        step = row.get("step") if isinstance(row, dict) else None
        if type(step) is not int or not previous < step <= STEPS:
            raise ValueError("History steps must increase through the requested endpoint")
        previous = step
        if set(row.get("arms", {})) != set(arms):
            raise ValueError("History arm identity differs from its stage")
        for name in arms:
            metrics = row["arms"][name]
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("History requires nonempty per-arm metrics")
    if previous != STEPS or history.get("latest_step") != STEPS:
        raise ValueError("History endpoint does not match 3000 updates")


def load_stage(path, stage, source):
    """Strictly verify the five public files, never open a model checkpoint."""
    root = Path(path).resolve()
    fingerprints = _fingerprints(root)
    manifest, summary, history, status = [_json(root / name) for name in INPUT_FILES[:4]]
    for name, value in (("manifest", manifest), ("summary", summary), ("history", history), ("status", status)):
        _finite_json(value, name)
    arms = STAGE_ARMS[stage]
    for value in (manifest, summary, status):
        if value.get("recipe") != RECIPE or value.get("stage") != stage:
            raise ValueError("Public recipe or stage mismatch")
    if (manifest.get("schema_version") != 5 or manifest.get("options", {}).get("steps") != STEPS
            or manifest.get("lattice_sizes") != list(SIZES) or manifest.get("theory_dim") != 1
            or manifest.get("block_size") != 2 or manifest.get("stride") != 2
            or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training", False)):
        raise ValueError("Expected unsupervised b=2 four-scale 3000-step validation")
    weights = manifest.get("closure_scale_weights")
    if (manifest.get("closure_objective") != "all_start_anchor" or manifest.get("learning_rate") != .001
            or manifest.get("closure_weights") != dict(commutator=5, rollout_weight=5, total_all_start=10)
            or weights is not None and (not isinstance(weights, list) or len(weights) != 3
                or any(type(value) not in (int, float) or value <= 0 for value in weights)
                or any(value != weights[0] for value in weights))):
        raise ValueError("Single factors must preserve all-start, equal scales and learning rate")
    if (summary.get("status") != "complete" or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)):
        raise ValueError("Only completed successful stages may be reported")
    for key, filename in (("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
                          ("history_progress_sha256", "history_progress.json"),
                          ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != fingerprints[filename]:
            raise ValueError(f"Public SHA256 receipt mismatch: {key}")
    if (manifest.get("data_manifest_sha256") != source.manifest["data_manifest_sha256"]
            or summary.get("data_manifest_sha256") != source.manifest["data_manifest_sha256"]):
        raise ValueError("Physical data manifest fingerprint differs from source")
    source_pins = manifest.get("source_public_sha256")
    if not isinstance(source_pins, dict) or any(source_pins.get(key) != digest for key, digest in source.fingerprints.items()):
        raise ValueError("Five source public SHA256 fingerprints must match")
    for key, digest in source_pins.items():
        _digest(digest, f"source_public_sha256.{key}")
    _digest(manifest.get("source_checkpoint_sha256"), "source_checkpoint_sha256")
    if summary.get("source_public_sha256") != source_pins:
        raise ValueError("Summary source public fingerprints disagree with manifest")
    schedule = source.manifest["paired_schedule_sha256"]
    if manifest.get("paired_schedule_sha256") != schedule or summary.get("paired_schedule_sha256") != schedule:
        raise ValueError("Physical minibatch schedule fingerprint differs from source")
    if summary.get("source_inputs_unchanged") is not True or summary.get("gradient_routing_verified") is not True:
        raise ValueError("Source immutability and independent gradients need explicit receipts")
    if set(manifest.get("arms", {})) != set(arms) or set(summary.get("arms", {})) != set(arms):
        raise ValueError("Stage arm set differs from planned single factors")
    source_initial = source.summary["initial_module_sha256"]
    source_final = source.summary["arms"]["all_start"]["final_module_sha256"]
    for name in arms:
        planned, receipt = manifest["arms"][name], summary["arms"][name]
        if planned.get("variant") != name:
            raise ValueError("Arm variant is not the planned single factor")
        initial = _modules(planned.get("initial_module_sha256"), f"{name} initial")
        common = _modules(planned.get("initial_common_module_sha256"), f"{name} common initial")
        final = _modules(receipt.get("final_module_sha256"), f"{name} final")
        if (_modules(receipt.get("initial_module_sha256"), f"{name} summary initial") != initial
                or _modules(receipt.get("initial_common_module_sha256"), f"{name} summary common initial") != common):
            raise ValueError("Initial source parameter receipts differ between manifest and summary")
        expected_initial = source_final if stage == "r_tail" else source_initial
        if common != expected_initial:
            raise ValueError("Common initialization fingerprints differ from the designated source")
        if stage == "r_tail" and initial != source_final:
            raise ValueError("R tail must start at the source endpoint")
        if stage == "joint" and any(initial[key] != source_initial[key] for key in ("C", "R")):
            raise ValueError("Joint arms must preserve the original C/R initialization")
        if name == "odd_C" and initial != source_initial:
            raise ValueError("Odd C must preserve the original Q/C/R parameter initialization")
        if receipt.get("actual_steps") != STEPS or receipt.get("source_QCR_steps") != STEPS:
            raise ValueError("Source and additional update counts must be explicit and separate")
        if receipt.get("closure_objective") != "all_start_anchor":
            raise ValueError("Arm closure objective must match all-start")
        for module in ("Q", "C", "R"):
            expected = 0 if stage == "r_tail" and module in ("Q", "C") else STEPS
            count = receipt.get("optimizer_steps", {}).get(module, {})
            if (count.get("min") != expected or count.get("max") != expected
                    or type(count.get("parameters")) is not int or count["parameters"] <= 0
                    or receipt.get(f"{module}_updates") != expected):
                raise ValueError(f"{name}.{module} optimizer counts do not match its authorization")
        if name == "cubic_decoder" and final["C"] != source_final["C"]:
            raise ValueError("Cubic decoder C fingerprint must match source")
        if name == "r_tail" and any(final[key] != source_final[key] for key in ("Q", "C")):
            raise ValueError("R tail changed frozen source Q/C fingerprints")
        if receipt.get("numerical_fixed_points"):
            raise ValueError("New stages must not search for numerical roots")
        for key in ("critical_r", "thermal_nu"):
            if receipt.get(key) is not None:
                raise ValueError("New arms cannot certify critical r or thermal exponent")
    for key in ("critical_r", "thermal_nu"):
        if summary.get(key) is not None:
            raise ValueError("Validation cannot certify critical r or thermal exponent")
    if summary.get("equilibrium_certified", False):
        raise ValueError("Validation cannot certify equilibrium")
    if summary.get("actual_steps") != STEPS or summary.get("actual_steps_by_arm") != {name: STEPS for name in arms}:
        raise ValueError("Summary update endpoint differs from requested stage steps")
    if stage == "joint" and summary.get("cubic_C_target_source_equal") is not True:
        raise ValueError("Cubic C/native-target pairing requires an explicit receipt")
    if stage == "r_tail" and summary.get("r_tail_QC_source_equal") is not True:
        raise ValueError("R-only frozen Q/C requires an explicit receipt")
    _history_check(history, arms)
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    endpoints = {}
    for name in arms:
        local = {key[len(name) + 1:]: value for key, value in arrays.items() if key.startswith(name + "_")}
        if not local:
            raise ValueError(f"Missing endpoint public arrays: {name}")
        _validate_arrays(local, name)
        for split in ("train", "test"):
            expected_shape = local[f"{split}_direct"].shape[:2]
            for suffix in ("field_mean", "field_rms", "field_mirror_l1"):
                key = f"{split}_{suffix}"
                if key not in local:
                    raise ValueError(f"Missing saved C diagnostic array: {name}.{key}")
                value = np.asarray(local[key])
                if value.shape != expected_shape or value.dtype.kind not in "fi" or not np.isfinite(value).all():
                    raise ValueError("C diagnostics must be finite [physical chain, 4] arrays")
                local[key] = value.astype(np.float64, copy=False)
            if (np.any(local[f"{split}_field_rms"] < np.abs(local[f"{split}_field_mean"]) - 1e-12)
                    or np.any(local[f"{split}_field_mirror_l1"] < -1e-12)
                    or np.any(local[f"{split}_field_mirror_l1"] > 2 + 1e-12)):
                raise ValueError("C RMS/mean or normalized mirror-L1 bounds are inconsistent")
        endpoints[name] = local
    if fingerprints != _fingerprints(root):
        raise ValueError("Stage public sources changed while reading")
    return PublicStage(root, stage, manifest, summary, status, history, endpoints, fingerprints)


def load_validation(source, joint, r_tail):
    baseline = load_public_comparison(_source_training(source))
    if baseline.summary.get("actual_steps") != STEPS:
        raise ValueError("Validation source must be the actual 3000-step endpoint")
    stages = {stage: load_stage(path, stage, baseline) for stage, path in (("joint", joint), ("r_tail", r_tail))}
    endpoints = {"baseline": baseline.endpoints["all_start"]}
    for stage in stages.values():
        endpoints.update(stage.endpoints)
    original = endpoints["baseline"]
    common_keys = [f"{split}_{suffix}" for split in ("train", "test") for suffix in ("labels", "cell_ids")]
    common_keys += ["grid_labels"]
    for name in NAMES[1:]:
        for key in common_keys:
            if not np.array_equal(original[key], endpoints[name][key]):
                raise ValueError(f"{name}: physical-chain order, labels or split differs from source ({key})")
    for name, suffixes in (("cubic_decoder", ("feature_target",)),
                           ("r_tail", ("direct", "direct_full", "feature_target", "feature_reconstruction"))):
        keys = [f"{split}_{suffix}" for split in ("train", "test") for suffix in suffixes]
        if name == "r_tail":
            keys.append("grid_mean_direct")
        for key in keys:
            if not np.array_equal(original[key], endpoints[name][key]):
                raise ValueError(f"{name}: same-target/source-coordinate pairing broken ({key})")
    for split in ("train", "test"):
        for suffix in ("field_mean", "field_rms", "field_mirror_l1"):
            key = f"{split}_{suffix}"
            if not np.array_equal(endpoints["cubic_decoder"][key], endpoints["r_tail"][key]):
                raise ValueError(f"Same C fixed-evaluation field diagnostic differs between cubic and R tail ({key})")
            if not np.array_equal(endpoints["cubic_decoder"][key][:, 0], endpoints["odd_C"][key][:, 0]):
                raise ValueError(f"Microscopic physical field diagnostic changed between arms ({key})")
    return Validation(baseline, stages, endpoints)


def C_diagnostics(validation):
    """Summarize saved fields, preserving chains and distinguishing signed bias."""
    rows, paired = [], []
    for split in ("train", "test"):
        for scale, size in enumerate(SIZES):
            for name in ("cubic_decoder", "odd_C", "r_tail"):
                local = validation.endpoints[name]
                mean = local[f"{split}_field_mean"][:, scale]
                rms = local[f"{split}_field_rms"][:, scale]
                mirror = local[f"{split}_field_mirror_l1"][:, scale]
                def summarize(values):
                    return dict(mean=float(values.mean()), population_std=float(values.std(ddof=0)),
                        minimum=float(values.min()), maximum=float(values.max()))
                rows.append(dict(arm=name, split=split, scale=scale, lattice_size=size,
                    physical_chains=len(mean), signed_field_mean=summarize(mean),
                    absolute_chain_field_mean=summarize(np.abs(mean)), field_rms=summarize(rms),
                    field_mirror_l1=summarize(mirror)))
            for metric, suffix, absolute in (("absolute_chain_field_mean", "field_mean", True),
                ("field_rms", "field_rms", False), ("field_mirror_l1", "field_mirror_l1", False)):
                reference = validation.endpoints["cubic_decoder"][f"{split}_{suffix}"][:, scale]
                odd = validation.endpoints["odd_C"][f"{split}_{suffix}"][:, scale]
                if absolute:
                    reference, odd = np.abs(reference), np.abs(odd)
                change = odd - reference
                denominator = float(reference.mean())
                paired.append(dict(metric=metric, split=split, scale=scale, lattice_size=size,
                    original_C_replay_mean=denominator, odd_C_mean=float(odd.mean()),
                    odd_minus_reference_mean=float(change.mean()),
                    relative_mean_change=float(odd.mean()) / denominator - 1 if denominator else None,
                    per_physical_chain_odd_minus_reference=change.tolist()))
    return dict(rows=rows, paired_changes=paired,
        original_C_reference="三阶解码臂的既有固定评估；其 C 端点与原基准完全相同，三项场统计与 R 收尾逐元素相同。",
        original_C_replay_from_arm="cubic_decoder", original_C_and_R_tail_exactly_equal=True,
        additional_baseline_inference_performed=False, additional_conditional_draws=0,
        observation="在原既有 evaluate 的 C 输出处以只读 hook 观察，不增加模型调用或 RNG 消耗。",
        mirror_definition="每点五中心 [-2,-1,0,1,2]×原 TRAIN 场 RMS、宽度 0.5×RMS 的 Gaussian-softmax 概率；链内所有构型和格点平均后 sum_i|p_i-p_(4-i)|，不除以 2。",
        mirror_bounds=[0, 2], error_bars="物理链之间的总体标准差，非置信区间；有限链不能认证平衡或完整 Z2 对称分布。",
        raw_chain_arrays_location="joint/evaluation_arrays.npz 与 r_tail/evaluation_arrays.npz 的 train/test_field_*",
        complete_distribution_symmetry_certified=False, equilibrium_certified=False)


def _metrics(arrays):
    available = _optional_feature_arrays(arrays)
    reconstruction = reconstruction_metrics(arrays, available)
    reconstruction["estimator"] = "保存的本臂解码重构；不假定三阶解码为仿射解码。"
    return dict(q_reconstruction=reconstruction, latent_closure=prediction_metrics(arrays),
        native_statistic_prediction=prediction_metrics(arrays, features=True, available=available),
        r_information=rank_correlations(arrays))


def _paired_rows(validation, metrics, name, kind):
    """Compare only outcomes whose coordinates/targets are identified exactly."""
    if kind == "latent_closure" and name != "r_tail":
        raise ValueError("Cross-coordinate g error ranking is forbidden")
    if kind == "native_statistic_prediction" and name == "odd_C":
        raise ValueError("Odd C changes native targets: same-physical-target ranking is forbidden")
    rows = []
    before = metrics["baseline"][kind]["rows"]
    after = metrics[name][kind]["rows"]
    for a, b in zip(before, after):
        identity = {key: a[key] for key in ("split", "branch", "source_scale", "target_scale")}
        if any(a[key] != b[key] for key in identity):
            raise ValueError("Prediction metric row identity differs")
        if a["model"]["train_target_population_std"] != b["model"]["train_target_population_std"]:
            raise ValueError("Same-target comparison needs the same original TRAIN denominator")
        old, new = a["model"]["absolute_rmse"], b["model"]["absolute_rmse"]
        rows.append(dict(**identity, baseline_absolute_rmse=old, variant_absolute_rmse=new,
            difference=new - old, relative_change=new / old - 1 if old else None,
            baseline_fixed_train_nrmse=a["model"]["fixed_train_nrmse"],
            variant_fixed_train_nrmse=b["model"]["fixed_train_nrmse"],
            shared_train_target_population_std=a["model"]["train_target_population_std"]))
    return rows


def comparisons(validation, metrics):
    reconstruction_rows = []
    for a, b in zip(metrics["baseline"]["q_reconstruction"]["rows"], metrics["cubic_decoder"]["q_reconstruction"]["rows"]):
        if any(a[key] != b[key] for key in ("split", "scale")):
            raise ValueError("Reconstruction row identity differs")
        old, new = a["reconstruction"], b["reconstruction"]
        if old["train_target_population_std"] != new["train_target_population_std"]:
            raise ValueError("Same-target reconstruction requires a common TRAIN denominator")
        reconstruction_rows.append(dict(split=a["split"], scale=a["scale"],
            baseline_absolute_rmse=old["absolute_rmse"], variant_absolute_rmse=new["absolute_rmse"],
            difference=new["absolute_rmse"] - old["absolute_rmse"],
            relative_change=new["absolute_rmse"] / old["absolute_rmse"] - 1 if old["absolute_rmse"] else None,
            baseline_fixed_train_nrmse=old["fixed_train_nrmse"], variant_fixed_train_nrmse=new["fixed_train_nrmse"],
            shared_train_target_population_std=old["train_target_population_std"]))
    return dict(
        cubic_decoder=dict(native_targets_exactly_equal=True, g_coordinates_identified=False,
            latent_cross_arm_ranking_performed=False, same_native_target_comparison_valid=True,
            native_reconstruction_rows=reconstruction_rows,
            native_prediction_rows=_paired_rows(validation, metrics, "cubic_decoder", "native_statistic_prediction")),
        odd_C=dict(native_targets_exactly_equal=all(np.array_equal(validation.endpoints["baseline"][f"{s}_feature_target"],
            validation.endpoints["odd_C"][f"{s}_feature_target"]) for s in ("train", "test")),
            g_coordinates_identified=False, latent_cross_arm_ranking_performed=False,
            same_native_target_comparison_valid=False, native_cross_arm_ranking_performed=False,
            note="目标由本臂 C 改变；只列各自目标的误差，不做相同物理任务的因果优劣排名。"),
        r_tail=dict(native_targets_exactly_equal=True, g_coordinates_identified=True,
            Q_C_fingerprints_unchanged=True, additional_Q_updates=0, additional_C_updates=0,
            additional_R_updates=STEPS, source_QCR_steps=STEPS,
            latent_rows=_paired_rows(validation, metrics, "r_tail", "latent_closure"),
            native_prediction_rows=_paired_rows(validation, metrics, "r_tail", "native_statistic_prediction")),
        bootstrap_performed=False, physical_critical_point_certified=False)


def plot_flow(validation, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for ax, name in zip(axes.ravel(), NAMES):
        a = validation.endpoints[name]
        inputs = a["train_direct"][:, :-1, 0].ravel()
        ax.axvspan(inputs.min(), inputs.max(), alpha=.14, color="#438aaf", label="本臂训练输入范围")
        ax.plot(a["flow_g"], a["flow_displacement"], color="#26798c", label="保存的位移")
        indices = np.unique(np.linspace(0, len(a["flow_g"]) - 1, 15).astype(int))
        ax.quiver(a["flow_g"][indices], np.zeros(len(indices)), a["flow_displacement"][indices], np.zeros(len(indices)),
            angles="xy", scale_units="xy", scale=1, color="#c77735", width=.003, label="真实长度箭头")
        if name == "baseline":
            for root in validation.source.summary["arms"]["all_start"].get("numerical_fixed_points", []):
                ax.scatter(root["g"], 0, marker="*", color="#a93b42", s=70, label="源端点已保存数值根")
        ax.axhline(0, color=".5", linestyle="--", linewidth=1)
        ax.set(title=LABELS[name], xlabel="本臂输入 g", ylabel="位移 R(g)−g")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("分别展示各自坐标中的 R(g) 流向；新阶段不搜索或插值不动点")
    fig.savefig(output / "R_vs_g.png", dpi=170)
    plt.close(fig)


def plot_coupling(validation, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for ax, name in zip(axes.ravel(), NAMES):
        a = validation.endpoints[name]
        for k, color in enumerate(("#247b9a", "#ce8235", "#9864a7", "#488057")):
            for split, marker, style, label in (("train", "o", "-", "训练"), ("test", "s", "--", "留出")):
                ids = a[f"{split}_cell_ids"]
                r, g = a[f"{split}_labels"][:, 0], a[f"{split}_direct"][:, k, 0]
                means = sorted((float(r[ids == cell][0]), float(g[ids == cell].mean())) for cell in np.unique(ids))
                ax.scatter(r, g, color=color, marker=marker, alpha=.2, s=10)
                ax.plot(*zip(*means), color=color, marker=marker, linestyle=style, markersize=3,
                    label=f"k={k}，L={SIZES[k]}，{label}")
        ax.set(title=LABELS[name], xlabel="生成参数 r（仅事后评价）", ylabel="本臂直接表示 g")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("四尺度 r→g 映射；不同表示不强行对齐坐标")
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def plot_losses(validation, output, plt):
    fig, axes = plt.subplots(4, 3, figsize=(14, 12.5), constrained_layout=True)
    groups = (("Q 表示目标", (("q_loss", "总表示损失"), ("reconstruction", "原生统计重构"))),
        ("C 粗粒化目标", (("c_loss", "总粗粒化损失"), ("information_proxy", "全壳信息代理"), ("variance_floor", "方差下限"))),
        ("R 闭环目标", (("r_loss", "实际闭环目标"), ("closure_one_step", "单步闭合"), ("closure_rollout", "微观多步闭合"))))
    for index, name in enumerate(NAMES):
        stage = validation.source if name == "baseline" else validation.stages["r_tail" if name == "r_tail" else "joint"]
        history_name = "all_start" if name == "baseline" else name
        for ax, (title, entries) in zip(axes[index], groups):
            for key, label in entries:
                points = [(row["step"], row["arms"][history_name].get(key)) for row in stage.history["history"]]
                points = [(step, value) for step, value in points if isinstance(value, (int, float)) and not isinstance(value, bool)]
                if points:
                    ax.plot(*zip(*points), label=label)
            ax.set(title=f"{LABELS[name]}：{title}", xlabel="本阶段实际更新次数", ylabel="保存损失")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            if name == "r_tail" and title != "R 闭环目标":
                ax.text(.04, .08, "Q/C 冻结：额外更新为零", transform=ax.transAxes, fontsize=8)
    fig.suptitle("三项单因素检验的损失曲线；原联合训练与追加 R 更新分别计数")
    fig.savefig(output / "loss_comparison.png", dpi=170)
    plt.close(fig)


def plot_native(metrics, output, plt):
    fig, axes = plt.subplots(4, 3, figsize=(19, 12), constrained_layout=True)
    for index, name in enumerate(NAMES):
        rows = [row for row in metrics[name]["native_statistic_prediction"]["rows"] if row["split"] == "test"]
        x = np.arange(len(rows))
        for ax, key, ylabel in ((axes[index, 0], "absolute_rmse", "绝对 RMSE"),
                                (axes[index, 1], "fixed_train_nrmse", "本臂 TRAIN 标准差 NRMSE")):
            for offset, kind, label in ((-.24, "model", "模型预测"), (0, "identity_source_statistic", "源尺度恒等预测"),
                                         (.24, "same_scale_train_mean_constant", "TRAIN 同尺度常数")):
                values = [row[kind][key] for row in rows]
                ax.bar(x + offset, [np.nan if value is None else value for value in values], .24, label=label)
            ax.set_xticks(x, [("单步 " if row["branch"] == "one_step" else "微观多步 ")
                + f"{SIZES[row['source_scale']]}→{SIZES[row['target_scale']]}" for row in rows], rotation=12, fontsize=8)
            ax.set(title=f"{LABELS[name]}：本臂 105 维原生统计", ylabel=ylabel)
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=7)
        ax = axes[index, 2]
        reconstruction = [row for row in metrics[name]["q_reconstruction"]["rows"] if row["split"] == "test"]
        positions = np.arange(len(reconstruction))
        for offset, kind, label in ((-.17, "reconstruction", "Q 原生统计重构"),
                                    (.17, "same_scale_train_mean_constant", "TRAIN 同尺度常数")):
            values = [row[kind]["fixed_train_nrmse"] for row in reconstruction]
            ax.bar(positions + offset, [np.nan if value is None else value for value in values], .34, label=label)
        ax.set_xticks(positions, [f"k={row['scale']}，L={SIZES[row['scale']]}" for row in reconstruction], fontsize=8)
        ax.set(title=f"{LABELS[name]}：原生统计重构", ylabel="本臂 TRAIN 标准差 NRMSE")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("留出原生统计预测；奇对称 C 改变目标，不能据跨行误差直接判物理优劣")
    fig.savefig(output / "native_error.png", dpi=170)
    plt.close(fig)


def plot_C_diagnostics(diagnostics, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2), constrained_layout=True)
    metrics = (("absolute_chain_field_mean", "链均值绝对值平均", "场均值幅度"),
        ("field_rms", "场 RMS", "场 RMS"), ("field_mirror_l1", "五软箱镜像差", "镜像 L1（不除以二）"))
    for row_index, split in enumerate(("train", "test")):
        for ax, (metric, title, ylabel) in zip(axes[row_index], metrics):
            for name, label, color in (("cubic_decoder", "原 C 固定评估重放参照", "#367f9c"),
                                       ("odd_C", "仅加 C 奇对称", "#c37932")):
                rows = [row for row in diagnostics["rows"] if row["arm"] == name and row["split"] == split]
                rows.sort(key=lambda row: row["scale"])
                ax.errorbar(np.arange(4), [row[metric]["mean"] for row in rows],
                    yerr=[row[metric]["population_std"] for row in rows], marker="o", capsize=3,
                    color=color, label=label)
            ax.set_xticks(np.arange(4), [f"k={scale}\nL={size}" for scale, size in enumerate(SIZES)])
            ax.set(title=f"{'训练' if split == 'train' else '留出'}物理链：{title}", ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("C 对称性与幅度诊断：仅观察既有评价输出\n误差条为物理链间标准差，不是置信区间；原 C 参照不额外运行模型")
    fig.savefig(output / "C_symmetry.png", dpi=170)
    plt.close(fig)


def write_report(source, joint, r_tail, output, *, font=None):
    validation = load_validation(source, joint, r_tail)
    output = Path(output).resolve()
    inputs = [validation.source, *validation.stages.values()]
    for public in inputs:
        if output == public.root or public.root.is_relative_to(output):
            raise ValueError("Report must not replace or contain any source run")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    metrics = {name: _metrics(validation.endpoints[name]) for name in NAMES}
    c_diagnostics = C_diagnostics(validation)
    saved_roots = _root_support(validation.source.summary["arms"]["all_start"].get("numerical_fixed_points", []),
        validation.endpoints["baseline"])
    counts = {"baseline": dict(source_QCR_steps=STEPS, additional_Q_updates=0, additional_C_updates=0, additional_R_updates=0)}
    for stage in validation.stages.values():
        for name in stage.endpoints:
            counts[name] = stage.summary["arms"][name]
    comparison = dict(schema_version=1, recipe=RECIPE, status="complete", lattice_sizes=list(SIZES), theory_dim=1,
        endpoints=metrics, controlled_comparisons=comparisons(validation, metrics), training_receipts=counts,
        C_diagnostics=c_diagnostics,
        source_saved_numerical_roots_with_support=saved_roots, new_numerical_root_search_performed=False,
        data_manifest_sha256=validation.source.manifest["data_manifest_sha256"],
        data_warnings=validation.source.summary.get("data_warnings", validation.source.manifest.get("data_warnings", [])),
        data_warning_count=validation.source.summary.get("data_warning_count", validation.source.manifest.get("data_warning_count")),
        sampler_qualification=validation.source.summary.get("sampler_qualification", validation.source.manifest.get("sampler_qualification")),
        labels_used_in_training=False, cross_coordinate_latent_ranking_performed=False,
        odd_C_same_physical_target_ranking_performed=False,
        normalization="逐臂固定 TRAIN 同目标尺度总体标准差，ddof=0；不使用 TEST 拟合归一或基线。",
        critical_r=None, thermal_nu=None, equilibrium_certified=False,
        physical_critical_point_certified=False, limitations=LIMITATIONS)
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_flow(validation, output, plt)
    plot_coupling(validation, output, plt)
    plot_losses(validation, output, plt)
    plot_native(metrics, output, plt)
    plot_C_diagnostics(c_diagnostics, output, plt)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    source_receipts = {}
    for name, public in zip(("source", "joint", "r_tail"), inputs):
        after = _fingerprints(public.root)
        if after != public.fingerprints:
            raise ValueError("Public sources changed during report generation")
        source_receipts[name] = dict(root=str(public.root), before_sha256=public.fingerprints, after_sha256=after)
    digests = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = dict(schema_version=1, recipe=RECIPE, status="complete", source_receipts=source_receipts,
        public_sources_unchanged=True, model_loaded=False, physical_data_loaded=False, cache_loaded=False,
        GPU_used=False, inference_performed=False, root_search_performed=False, MI_reestimated=False,
        outputs_sha256=digests, public_files=sorted([*digests, "report.json", "status.json"]),
        critical_r=None, thermal_nu=None, equilibrium_certified=False, limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    status = dict(recipe=RECIPE, status="complete", workflow_complete=True, returncode=0, error=None,
        requires_user_action=False, comparison_sha256=sha256(output / "comparison.json"),
        report_sha256=sha256(output / "report.json"), critical_r=None, thermal_nu=None, equilibrium_certified=False)
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--r-tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.source, args.joint, args.r_tail, args.output, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
