"""Read-only public-array report for the explicit b=2 closure-objective ablation.

python -m scripts.report_phi4_b2_closure_comparison --training RUN --output NEW_DIR

Only the five named public JSON/NPZ files are read. This custom recipe is not
passed through a legacy phi4 loader. No checkpoint, configuration archive,
model inference, sampling, MI evaluation, or new root search is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from scripts.report_gaussian_scalar_qcr import (
    _optional_feature_arrays, prediction_metrics, rank_correlations,
    reconstruction_metrics,
)
from scripts.report_phi4_qcr import INPUT_FILES, _json, _numeric, _plotting, sha256


RECIPE = "phi4_b2_closure_objective_comparison_v1"
NAMES = ("legacy", "all_start")
MODES = dict(legacy="one_step_and_rollout", all_start="all_start_anchor")
LABELS = dict(legacy="早期：一步＋微观起点多步", all_start="全起点同终点锚定")
SIZES = (32, 16, 8, 4)
LIMITATIONS = [
    "用户明确授权的 b=2、32→16→8→4、一维表示闭环目标消融；不改变默认 b=3 或默认训练配方。",
    "两臂从相同随机 Q/C/R 开始，使用同一无标签数据与配对批次计划；不加载旧实验端点作为初始化。",
    "Q 仅接受表示目标，C 仅接受粗粒化目标，R 仅接受闭环目标；没有对 Q 的闭合反馈。",
    "两臂 Q/C 的端点指纹、直接编码、重构与原生统计目标逐元素相同，因此可在共同坐标下比较 R。",
    "所有误差并列报告绝对 RMSE 和固定 TRAIN 同目标尺度总体标准差归一误差，另列恒等及 TRAIN 常数基线。",
    "固定 u、单种子、有限物理链与末尺度 4×4 均有限；局域统计预测不等于完整分布或长程物理正确。",
    "源数据原 ESS、热冷初态与质量警告保留；本报告不认证平衡、不自动补样或修改数据。",
    "仅读取端点已经保存的全部数值根；输入范围内也不等于临界流形上，不选择热模或认证物理临界点。",
    "不输入耦合标签训练、不定位临界 r、不估计临界指数；critical_r 与 thermal_nu 始终为 null。",
]


@dataclass
class PublicComparison:
    root: Path
    manifest: dict
    summary: dict
    status: dict
    history: dict
    endpoints: dict
    fingerprints: dict


def _finite_json(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{name}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} contains a nonfinite value")


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _modules(value, name):
    if not isinstance(value, dict) or set(value) != {"Q", "C", "R"}:
        raise ValueError(f"{name} requires Q/C/R module SHA256 receipts")
    return {key: _digest(digest, f"{name}.{key}") for key, digest in value.items()}


def _history(history, steps):
    rows = history.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError("History must be a nonempty list")
    previous = -1
    for row in rows:
        step = row.get("step") if isinstance(row, dict) else None
        if (not isinstance(step, int) or isinstance(step, bool)
                or step <= previous or step < 0 or step > steps):
            raise ValueError("History steps must increase to the actual endpoint")
        previous = step
        for arm in NAMES:
            metrics = row.get("arms", {}).get(arm)
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Both arms need metrics at every paired history step")
        for key in ("q_loss", "c_loss"):
            values = [row["arms"][arm].get(key) for arm in NAMES]
            if values[0] is None or values[0] != values[1]:
                raise ValueError(f"Paired history {key} differs between identical Q/C arms")
    if previous != steps or history.get("latest_step") != steps:
        raise ValueError("History does not reach the requested update endpoint")


def _validate_arrays(arrays, name):
    for split in ("train", "test"):
        direct = _numeric(arrays[f"{split}_direct"], f"{name}.{split}_direct")
        if direct.ndim != 3 or direct.shape[1:] != (4, 1) or not len(direct):
            raise ValueError("Direct g must be nonempty [physical chain, 4, 1]")
        arrays[f"{split}_direct"] = direct
        for suffix, shape in (("direct_full", direct.shape), ("one_step", (len(direct), 3, 1)),
                              ("rollout", direct.shape)):
            value = _numeric(arrays[f"{split}_{suffix}"], f"{name}.{split}_{suffix}")
            if value.shape != shape:
                raise ValueError(f"{suffix} shape disagrees with four-scale direct g")
            arrays[f"{split}_{suffix}"] = value
        if not np.array_equal(arrays[f"{split}_rollout"][:, 0], direct[:, 0]):
            raise ValueError("Microscopic rollout must start at the direct microscopic encoding")
        labels = _numeric(arrays[f"{split}_labels"], f"{name}.{split}_labels")
        ids = np.asarray(arrays[f"{split}_cell_ids"])
        if labels.shape != (len(direct), 3):
            raise ValueError("Evaluation labels must be [physical chain, r/u/h]")
        if ids.shape != (len(direct),) or ids.dtype.kind not in "UiS":
            raise ValueError("Cell IDs must be an integer/string vector")
        for cell in np.unique(ids):
            if not np.all(labels[ids == cell] == labels[ids == cell][0]):
                raise ValueError("A parameter cell cannot have inconsistent coupling labels")
        arrays[f"{split}_labels"] = labels
    if set(arrays["train_cell_ids"].tolist()) & set(arrays["test_cell_ids"].tolist()):
        raise ValueError("TRAIN/TEST cell IDs overlap")
    if set(map(tuple, arrays["train_labels"].tolist())) & set(map(tuple, arrays["test_labels"].tolist())):
        raise ValueError("TRAIN/TEST physical parameter conditions overlap")
    labels = np.concatenate([arrays[f"{split}_labels"] for split in ("train", "test")])
    if not np.all(labels[:, 1:] == labels[0, 1:]):
        raise ValueError("Scalar-r comparison requires fixed u and h")
    available = _optional_feature_arrays(arrays)
    if not all(available.values()) or arrays["train_feature_target"].shape[-1] != 105:
        raise ValueError("All 105-dimensional native targets and predictions are required")
    for key in ("flow_g", "flow_R", "flow_displacement"):
        value = _numeric(arrays[key], f"{name}.{key}")
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value[:, 0]
        if value.ndim != 1 or len(value) < 2:
            raise ValueError("Saved flow must be a vector with at least two grid points")
        arrays[key] = value
    if (not (arrays["flow_g"].shape == arrays["flow_R"].shape == arrays["flow_displacement"].shape)
            or not np.all(np.diff(arrays["flow_g"]) > 0)
            or not np.allclose(arrays["flow_R"] - arrays["flow_g"], arrays["flow_displacement"], atol=1e-12, rtol=1e-10)):
        raise ValueError("Saved flow must be ordered and displacement must equal R(g)-g")
    means = _numeric(arrays["grid_mean_direct"], f"{name}.grid_mean_direct")
    grid_labels = _numeric(arrays["grid_labels"], f"{name}.grid_labels")
    if means.ndim != 3 or means.shape[1:] != (4, 1) or grid_labels.shape != (len(means), 3):
        raise ValueError("Condition grid must contain four-scale means and r/u/h labels")
    if len(set(map(tuple, grid_labels.tolist()))) != len(grid_labels):
        raise ValueError("Condition grid labels must be unique")
    direct = np.concatenate([arrays[f"{split}_direct"] for split in ("train", "test")])
    if set(map(tuple, labels.tolist())) != set(map(tuple, grid_labels.tolist())):
        raise ValueError("Condition grid labels disagree with evaluation parameter conditions")
    for index, label in enumerate(grid_labels):
        observed = direct[np.all(labels == label, axis=1)].mean(axis=0)
        if not np.allclose(means[index], observed, atol=1e-12, rtol=1e-10):
            raise ValueError("Condition means disagree with saved per-chain direct encodings")
    arrays["grid_mean_direct"], arrays["grid_labels"] = means, grid_labels
    jensen_key = "grid_jensen_difference" if "grid_jensen_difference" in arrays else "grid_jensen"
    for key in ("grid_R_of_mean", "grid_mean_R", "grid_displacement", jensen_key):
        value = _numeric(arrays[key], f"{name}.{key}")
        if value.shape != (len(means), 3, 1):
            raise ValueError("Condition-grid R statistics must have three input scales")
        arrays[key] = value
    if not np.allclose(arrays["grid_displacement"], arrays["grid_R_of_mean"] - means[:, :-1], atol=1e-12, rtol=1e-10):
        raise ValueError("Condition-grid displacement must equal R(mean g)-mean g")
    if not np.allclose(arrays[jensen_key], arrays["grid_mean_R"] - arrays["grid_R_of_mean"], atol=1e-12, rtol=1e-10):
        raise ValueError("Saved Jensen difference must keep mean R(g) and R(mean g) distinct")
    return available


def load_public_comparison(training):
    """Validate only this recipe's five public artifacts, never a checkpoint."""
    root = Path(training).resolve()
    fingerprints = {name: sha256(root / name) for name in INPUT_FILES}
    manifest, summary, history, status = [_json(root / name) for name in INPUT_FILES[:4]]
    for name, value in (("manifest", manifest), ("summary", summary), ("history", history), ("status", status)):
        _finite_json(value, name)
    if manifest.get("schema_version") != 5 or any(value.get("recipe") != RECIPE for value in (manifest, summary, status)):
        raise ValueError("Expected the explicit b=2 closure-objective comparison recipe, schema 5")
    if (summary.get("status") != "complete" or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)):
        raise ValueError("Only a successfully completed b=2 comparison can be reported")
    for key, filename in (("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
                          ("history_progress_sha256", "history_progress.json"),
                          ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != fingerprints[filename]:
            raise ValueError(f"Completed status {key} disagrees with public SHA256")
    steps = manifest.get("options", {}).get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Manifest must pin positive requested update steps")
    if summary.get("actual_steps") != steps or summary.get("actual_steps_by_arm") != {arm: steps for arm in NAMES}:
        raise ValueError("Summary actual update counts disagree with the requested endpoint")
    if (manifest.get("lattice_sizes") != list(SIZES) or manifest.get("theory_dim") != 1
            or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training", False)):
        raise ValueError("Expected an unsupervised dg=1, 32→16→8→4 comparison")
    if (manifest.get("block_size") != 2 or manifest.get("stride") != 2
            or manifest.get("Q_closure_feedback") is not False
            or manifest.get("warm_start") is not False
            or manifest.get("C_pretraining") is not False
            or manifest.get("kernel_selection") is not False
            or manifest.get("learning_rate") != .001):
        raise ValueError("Expected fresh b=2 Q/C/R with no Q closure feedback or kernel pretraining")
    if manifest.get("closure_weights") != dict(commutator=5, rollout_weight=5, total_all_start=10):
        raise ValueError("The closure comparison must retain the requested 5/5 outer weights")
    scale_weights = manifest.get("closure_scale_weights")
    if scale_weights is not None and (not isinstance(scale_weights, list) or len(scale_weights) != 3
            or any(not isinstance(value, (int, float)) or value <= 0 for value in scale_weights)
            or any(value != scale_weights[0] for value in scale_weights)):
        raise ValueError("Only equal target-scale closure weights are authorized")
    if manifest.get("closure_modes") != MODES or summary.get("closure_modes") != MODES:
        raise ValueError("The two arms must pin different legacy/all-start closure modes")
    for key in ("paired_schedule_sha256", "data_manifest_sha256"):
        _digest(manifest.get(key), key)
        if summary.get(key) != manifest[key]:
            raise ValueError(f"Summary {key} differs from the pinned manifest")
    if summary.get("C_Q_pairing_verified") is not True:
        raise ValueError("Identical C/Q pairing must be verified by training")
    initial = _modules(summary.get("initial_module_sha256"), "common initial")
    for key in ("critical_r", "thermal_nu"):
        if summary.get(key) is not None:
            raise ValueError("This comparison must not claim a critical r or thermal exponent")
    if summary.get("equilibrium_certified", False):
        raise ValueError("This report cannot certify sampling equilibrium")
    final = {}
    for arm in NAMES:
        receipt = summary.get("arms", {}).get(arm, {})
        if receipt.get("actual_steps") != steps or receipt.get("closure_objective") != MODES[arm]:
            raise ValueError("Arm steps or closure-objective identity disagree with the plan")
        for module in ("Q", "C", "R"):
            count = receipt.get("optimizer_steps", {}).get(module, {})
            if count.get("min") != steps or count.get("max") != steps or count.get("parameters", 0) <= 0:
                raise ValueError(f"{arm}.{module} Adam counters disagree with the endpoint")
        if _modules(receipt.get("initial_module_sha256"), f"{arm} initial") != initial:
            raise ValueError("Both arms must start from identical Q/C/R module fingerprints")
        final[arm] = _modules(receipt.get("final_module_sha256"), f"{arm} final")
        for numerical_root in receipt.get("numerical_fixed_points", []):
            if (not isinstance(numerical_root, dict) or not np.isfinite(numerical_root.get("g", np.nan))
                    or numerical_root.get("thermal_nu") is not None
                    or numerical_root.get("critical_r") is not None
                    or numerical_root.get("physical_critical_point_certified", False)):
                raise ValueError("Saved numerical roots cannot certify physical criticality")
    if any(final["legacy"][module] != final["all_start"][module] for module in ("Q", "C")):
        raise ValueError("Final Q/C fingerprints differ, so this is not an R-only comparison")
    _history(history, steps)
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    endpoints = {}
    for arm in NAMES:
        local = {key[len(arm) + 1:]: value for key, value in arrays.items() if key.startswith(arm + "_")}
        if not local:
            raise ValueError(f"Missing public arrays for endpoint {arm}")
        _validate_arrays(local, arm)
        endpoints[arm] = local
    paired_keys = [f"{split}_{suffix}" for split in ("train", "test")
                   for suffix in ("direct", "direct_full", "feature_target", "feature_reconstruction", "labels", "cell_ids")]
    paired_keys += ["grid_mean_direct", "grid_labels"]
    for key in paired_keys:
        if not np.array_equal(endpoints["legacy"][key], endpoints["all_start"][key]):
            raise ValueError(f"Paired C/Q source array {key} changed between arms")
    if fingerprints != {name: sha256(root / name) for name in INPUT_FILES}:
        raise ValueError("Public sources changed while reading")
    return PublicComparison(root, manifest, summary, status, history, endpoints, fingerprints)


def _root_support(roots, arrays):
    """Annotate all saved roots; deliberately never locate or select a new one."""
    inputs = arrays["train_direct"][:, :-1, 0].ravel()
    result = []
    for root in roots:
        g, bounds = float(root["g"]), []
        for split in ("train", "test"):
            for scale, size in enumerate(SIZES):
                values = arrays[f"{split}_direct"][:, scale, 0]
                bounds.append(dict(split=split, scale=scale, lattice_size=size,
                    minimum=float(values.min()), maximum=float(values.max()),
                    within_saved_range=bool(values.min() <= g <= values.max())))
        result.append(dict(saved_root=root, training_input_range=[float(inputs.min()), float(inputs.max())],
            within_training_input_range=bool(inputs.min() <= g <= inputs.max()), support_by_scale=bounds,
            physical_critical_point_certified=False, critical_r=None, thermal_nu=None))
    return result


def endpoint_metrics(public, arm):
    arrays = public.endpoints[arm]
    receipt = public.summary["arms"][arm]
    available = _optional_feature_arrays(arrays)
    dispersion = []
    for split in ("train", "test"):
        for scale, size in enumerate(SIZES):
            values = arrays[f"{split}_direct"][:, scale, 0]
            dispersion.append(dict(split=split, scale=scale, lattice_size=size,
                population_std=float(values.std(ddof=0)), mean=float(values.mean()),
                minimum=float(values.min()), maximum=float(values.max())))
    return dict(actual_steps=receipt["actual_steps"], closure_objective=MODES[arm],
        optimizer_steps=receipt["optimizer_steps"], q_reconstruction=reconstruction_metrics(arrays, available),
        latent_closure=prediction_metrics(arrays),
        native_statistic_prediction=prediction_metrics(arrays, features=True, available=available),
        r_information=rank_correlations(arrays), dispersion=dispersion,
        condition_mean_vs_mean_map=dict(
            grid_labels=arrays["grid_labels"].tolist(),
            R_of_mean=arrays["grid_R_of_mean"].tolist(), mean_R=arrays["grid_mean_R"].tolist(),
            jensen_difference=arrays.get("grid_jensen_difference", arrays.get("grid_jensen")).tolist(),
            note="R(mean g) 与 mean R(g) 分别保留；条件平均不是新增独立物理链。"),
        saved_numerical_roots_with_support=_root_support(receipt.get("numerical_fixed_points", []), arrays),
        critical_r=None, thermal_nu=None, physical_critical_point_certified=False)


def paired_comparison(public, metrics):
    rows = []
    for kind in ("latent_closure", "native_statistic_prediction"):
        old = metrics["legacy"][kind]["rows"]
        new = metrics["all_start"][kind]["rows"]
        for a, b in zip(old, new):
            identity = {key: a[key] for key in ("split", "branch", "source_scale", "target_scale")}
            if any(a[key] != b[key] for key in identity):
                raise ValueError("Metric row pairing changed")
            target = public.endpoints["legacy"][f"{a['split']}_" + ("direct" if kind == "latent_closure" else "feature_target")]
            suffix = a["branch"] == "one_step"
            prediction_key = "one_step" if suffix else "rollout"
            if kind != "latent_closure":
                prediction_key += "_feature_prediction"
            index = a["target_scale"] - 1 if suffix else a["target_scale"]
            errors = [public.endpoints[arm][f"{a['split']}_{prediction_key}"][:, index]
                      - target[:, a["target_scale"]] for arm in NAMES]
            difference = np.mean(errors[1] ** 2 - errors[0] ** 2, axis=-1)
            before, after = a["model"]["absolute_rmse"], b["model"]["absolute_rmse"]
            rows.append(dict(metric=kind, **identity,
                source_lattice_size=SIZES[a["source_scale"]], target_lattice_size=SIZES[a["target_scale"]],
                legacy_rmse=before, all_start_rmse=after, all_start_minus_legacy_rmse=after - before,
                relative_rmse_change=(after / before - 1) if before > 0 else None,
                legacy_nrmse=a["model"]["fixed_train_nrmse"], all_start_nrmse=b["model"]["fixed_train_nrmse"],
                fixed_common_train_target_std=a["model"]["train_target_population_std"],
                all_start_minus_legacy_per_physical_chain_mse=difference.tolist(),
                mean_all_start_minus_legacy_mse=float(difference.mean()),
                fraction_chains_with_lower_all_start_mse=float(np.mean(difference < 0))))
    return dict(rows=rows, all_native_targets_exactly_equal=True, all_direct_coordinates_exactly_equal=True,
        all_Q_reconstructions_exactly_equal=True, final_Q_C_fingerprints_exactly_equal=True,
        physical_chain_order_paired=True, raw_latent_error_cross_arm_comparison_valid=True,
        bootstrap_performed=False, cross_block_size_coordinate_comparison_performed=False)


def plot_flow(public, metrics, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), constrained_layout=True)
    colors = dict(legacy="#247b9c", all_start="#c5752a")
    inputs = public.endpoints["legacy"]["train_direct"][:, :-1, 0].ravel()
    for ax in axes:
        ax.axvspan(inputs.min(), inputs.max(), alpha=.12, color="#5d9fc5", label="共同训练输入范围")
        ax.grid(alpha=.2)
    for arm in NAMES:
        a, color = public.endpoints[arm], colors[arm]
        g, mapped, drift = a["flow_g"], a["flow_R"], a["flow_displacement"]
        axes[0].plot(g, mapped, color=color, label=LABELS[arm])
        axes[1].plot(g, drift, color=color, label=LABELS[arm])
        indices = np.unique(np.linspace(0, len(g) - 1, min(12, len(g))).astype(int))
        axes[1].quiver(g[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
            color=color, angles="xy", scale_units="xy", scale=1, width=.0035, alpha=.7)
        kinds = {"attracting": "数值吸引", "repelling": "数值排斥", "neutral": "数值中性"}
        for index, row in enumerate(metrics[arm]["saved_numerical_roots_with_support"]):
            root, x = row["saved_root"], row["saved_root"]["g"]
            axes[0].scatter(x, x, marker="*", s=90, color=color, zorder=5)
            axes[1].scatter(x, 0, marker="*", s=90, color=color, zorder=5)
            location = "范围内" if row["within_training_input_range"] else "范围外"
            axes[1].annotate(f"{kinds.get(root.get('classification'), '保存数值根')}，{location}", (x, 0),
                xytext=(4, 10 + 16 * (index + NAMES.index(arm))), textcoords="offset points", fontsize=8, color=color)
    bounds = [*axes[0].get_xlim(), *axes[1].get_xlim()]
    axes[0].plot([min(bounds), max(bounds)], [min(bounds), max(bounds)], "--", color=".5", label="恒等映射")
    axes[1].axhline(0, color=".5", linestyle="--", linewidth=1)
    axes[0].set(title="共同坐标中的重整化映射", xlabel="输入 g", ylabel="输出 R(g)")
    axes[1].set(title="真实长度的离散流向箭头", xlabel="输入 g", ylabel="位移 R(g)−g")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("二倍粗粒化闭环目标对照：仅标记端点保存的数值根，不认证临界点")
    fig.savefig(output / "R_vs_g.png", dpi=170)
    plt.close(fig)


def plot_coupling(public, output, plt):
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    arrays = public.endpoints["legacy"]
    for scale, color in enumerate(("#237b9b", "#d08231", "#9467a8", "#458254")):
        for split, marker, style, label in (("train", "o", "-", "训练"), ("test", "s", "--", "留出")):
            ids = arrays[f"{split}_cell_ids"]
            r, values = arrays[f"{split}_labels"][:, 0], arrays[f"{split}_direct"][:, scale, 0]
            means = sorted((float(r[ids == cell][0]), float(values[ids == cell].mean())) for cell in np.unique(ids))
            ax.scatter(r, values, color=color, marker=marker, s=15, alpha=.25)
            ax.plot(*zip(*means), color=color, marker=marker, linestyle=style, markersize=4,
                label=f"k={scale}，L={SIZES[scale]}，{label}条件均值")
    ax.set(title="两臂共享 Q/C：四尺度直接编码完全相同", xlabel="生成参数 r（仅事后评价）", ylabel="共同直接表示 g")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def plot_losses(public, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.1), constrained_layout=True)
    groups = [("Q 表示目标", (("q_loss", "总表示损失"), ("reconstruction", "统计重构"),
            ("subset_micro", "微观子集一致性"), ("subset_coarse", "粗尺度子集一致性"))),
        ("C 粗粒化目标", (("c_loss", "总粗粒化损失"), ("info_proxy", "全壳信息代理"),
            ("variance_floor", "方差下限"), ("uniform_amplitude", "均匀块幅度保持"))),
        ("R 闭环目标", (("r_loss", "实际加权闭环目标"), ("closure_one_step", "平均单步闭合"),
            ("closure_rollout", "微观起点多步闭合"), ("closure_all_start_anchor", "全起点锚定闭合")))]
    for index, arm in enumerate(NAMES):
        for ax, (title, entries) in zip(axes[index], groups):
            for key, label in entries:
                points = [(row["step"], row["arms"][arm].get(key)) for row in public.history["history"]]
                points = [(step, value) for step, value in points if isinstance(value, (int, float)) and not isinstance(value, bool)]
                if points:
                    ax.plot(*zip(*points), label=label)
            ax.set(title=f"{LABELS[arm]}\n{title}", xlabel="实际训练更新次数", ylabel="保存的损失值")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    fig.suptitle("配对训练损失：Q/C 目标相同，仅 R 闭环定义不同")
    fig.savefig(output / "loss_comparison.png", dpi=170)
    plt.close(fig)


def write_report(training, output, *, font=None):
    public = load_public_comparison(training)
    output = Path(output).resolve()
    if output == public.root or public.root.is_relative_to(output):
        raise ValueError("Report must not replace or contain the source run")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    metrics = {arm: endpoint_metrics(public, arm) for arm in NAMES}
    comparison = dict(schema_version=1, recipe=RECIPE, status="complete", lattice_sizes=list(SIZES), theory_dim=1,
        closure_modes=MODES, endpoints=metrics, paired_comparison=paired_comparison(public, metrics),
        paired_schedule_sha256=public.manifest["paired_schedule_sha256"],
        data_manifest_sha256=public.manifest["data_manifest_sha256"],
        C_Q_pairing_verified=True, labels_used_in_training=False,
        latent_coordinates_identified_across_arms=True, latent_raw_error_cross_arm_ranking_performed=True,
        closure_error_weights=dict(
            legacy=dict(e01=10 / 3, e12=5 / 3, e23=5 / 3, e02=5 / 3, e03=5 / 3, e13=0.),
            all_start=dict(e01=10 / 3, e12=5 / 3, e23=10 / 9, e02=5 / 3, e03=10 / 9, e13=10 / 9)),
        data_warnings=public.summary.get("data_warnings", public.manifest.get("data_warnings", [])),
        data_warning_count=public.summary.get("data_warning_count", public.manifest.get("data_warning_count")),
        sampler_qualification=public.summary.get("sampler_qualification", public.manifest.get("sampler_qualification")),
        training_timing=public.summary.get("timing", {}),
        normalization="共同原 TRAIN 同目标尺度总体标准差，ddof=0；不使用 TEST 拟合归一或基线。",
        root_search_performed=False, critical_r=None, thermal_nu=None, equilibrium_certified=False,
        physical_critical_point_certified=False, limitations=LIMITATIONS)
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_flow(public, metrics, output, plt)
    plot_coupling(public, output, plt)
    plot_losses(public, output, plt)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after = {name: sha256(public.root / name) for name in INPUT_FILES}
    if after != public.fingerprints:
        raise ValueError("Public sources changed during report generation")
    digests = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = dict(schema_version=1, recipe=RECIPE, status="complete", source=str(public.root),
        sources_sha256_before=public.fingerprints, sources_sha256_after=after, public_sources_unchanged=True,
        model_loaded=False, physical_data_loaded=False, cache_loaded=False, GPU_used=False,
        inference_performed=False, root_search_performed=False, MI_reestimated=False,
        outputs_sha256=digests, public_files=sorted([*digests, "report.json", "status.json"]),
        critical_r=None, thermal_nu=None, equilibrium_certified=False, limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    status = dict(recipe=RECIPE, status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, comparison_sha256=sha256(output / "comparison.json"),
        report_sha256=sha256(output / "report.json"), critical_r=None, thermal_nu=None, equilibrium_certified=False)
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", "--run", dest="training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.training, args.output, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
