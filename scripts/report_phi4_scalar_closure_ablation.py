"""Read-only public-array report for the frozen-C scalar closure ablation.

python -m scripts.report_phi4_scalar_closure_ablation --run RUN --output NEW_DIR

Reads only manifest/summary/status/history JSON and evaluation_arrays.npz.
It never loads models, checkpoints, physical configurations, or additional
cache files. No inference, sampling, MI evaluation, root search, phase location,
or exponent estimation occurs here. This is an explicitly requested ablation,
not a change to the default QCR recipe.
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
from scripts.report_phi4_qcr import INPUT_FILES, _json, _numeric, _plotting, sha256


NAMES = ("source", "A", "B")
LABELS = {"source": "共同源端点", "A": "对照 A：原表示目标＋共同防塌缩",
          "B": "实验 B：另加表示闭合兼容"}
LIMITATIONS = [
    "用户授权的冻结 C、一维 Q/R 消融；不改变默认 QCR 配方，也不是重新联合训练完整 QCR。",
    "A/B 从相同源 Q/R 暖启动，使用相同缓存与随机批次计划、独立新 Adam；源训练步数与额外更新分开报告。",
    "两臂共同含防塌缩约束；A 不是原训练的完全原样续训，A/B 唯一预定差异是 Q 的闭合兼容项。",
    "C 缓存抽样重复不是新的独立物理链；物理链与参数条件的留出划分保持不变。",
    "各端点 g 可重新缩放或变形，不直接以原始潜坐标误差跨臂排名；同时报告各自 TRAIN 标准差与相对源离散度。",
    "105 维原生软局部统计的目标逐元素配对；局域统计预测改善不等于完整分布或长程物理正确。",
    "闭合兼容目标与防塌缩约束可以限制坐标选择，但不能保证存在一维无记忆闭合或物理临界不动点。",
    "只转发训练端点已保存的数值根；范围内不等于流形上，数值分类不认证物理相变或热方向。",
    "固定 u、单种子、有限样本与 36→12→4 尺度均有限；全部原 ESS 与热冷初态警告保留，未认证平衡。",
    "不使用 r/u/h 监督，不定位临界 r，不选择热模，不估计临界指数；critical_r 与 thermal_nu 始终为 null。",
]


@dataclass
class PublicAblation:
    root: Path
    manifest: dict
    summary: dict
    status: dict
    history: dict
    arrays: dict
    endpoints: dict
    fingerprints: dict


def _hashes(value, name):
    if (not isinstance(value, dict) or set(value) != {"Q", "C", "R"}
            or any(not isinstance(digest, str) or not digest for digest in value.values())):
        raise ValueError(f"{name} requires nonempty Q/C/R module SHA256 receipts")
    return value


def _history(history, steps):
    rows = history.get("history", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("History must be a nonempty list")
    previous = -1
    for row in rows:
        step = row.get("step") if isinstance(row, dict) else None
        if (not isinstance(step, int) or isinstance(step, bool)
                or step <= previous or step < 0 or step > steps):
            raise ValueError("History steps must increase to the actual endpoint")
        previous = step
        for arm in ("A", "B"):
            metrics = row.get("arms", {}).get(arm)
            if not isinstance(metrics, dict):
                raise ValueError("Both arms must have metrics at every paired history step")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    raise ValueError(f"History {arm}.{key} contains a nonfinite value")
    if previous != steps or history.get("latest_step") != steps:
        raise ValueError("History does not reach the requested update endpoint")
    return rows


def _validate_endpoint_arrays(arrays, name):
    scales = None
    for split in ("train", "test"):
        direct = _numeric(arrays[f"{split}_direct"], f"{name}.{split}_direct")
        if direct.ndim != 3 or direct.shape[-1] != 1 or not len(direct) or direct.shape[1] < 2:
            raise ValueError("Direct g must be nonempty [physical chain, scale, 1]")
        if scales is not None and direct.shape[1] != scales:
            raise ValueError("TRAIN/TEST scale counts differ")
        scales = direct.shape[1]
        arrays[f"{split}_direct"] = direct
        for suffix, shape in (("direct_full", direct.shape),
                ("one_step", (len(direct), scales - 1, 1)),
                ("rollout", direct.shape)):
            value = _numeric(arrays[f"{split}_{suffix}"], f"{name}.{split}_{suffix}")
            if value.shape != shape:
                raise ValueError(f"{suffix} shape disagrees with direct g")
            arrays[f"{split}_{suffix}"] = value
        if not np.allclose(arrays[f"{split}_rollout"][:, 0], direct[:, 0], rtol=1e-10, atol=1e-12):
            raise ValueError("Rollout must start at the direct microscopic encoding")
        labels = _numeric(arrays[f"{split}_labels"], f"{name}.{split}_labels")
        ids = np.asarray(arrays[f"{split}_cell_ids"])
        if labels.shape != (len(direct), 3):
            raise ValueError("Evaluation labels must be [physical chain, r/u/h]")
        if ids.shape != (len(direct),) or ids.dtype.kind not in "UiS":
            raise ValueError("Physical cell IDs must be an integer/string vector")
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
        raise ValueError("Scalar-r report requires fixed u and h")
    available = _optional_feature_arrays(arrays)
    if not all(available.values()) or arrays["train_feature_target"].shape[-1] != 105:
        raise ValueError("All 105-dimensional native targets and predictions are required")
    for key in ("flow_g", "flow_R", "flow_displacement"):
        value = _numeric(arrays[key], f"{name}.{key}").reshape(-1)
        if len(value) < 2:
            raise ValueError("Saved flow must have at least two grid points")
        arrays[key] = value
    if not (arrays["flow_g"].shape == arrays["flow_R"].shape == arrays["flow_displacement"].shape):
        raise ValueError("Flow array shapes differ")
    if (not np.all(np.diff(arrays["flow_g"]) > 0)
            or not np.allclose(arrays["flow_R"] - arrays["flow_g"], arrays["flow_displacement"], atol=1e-12, rtol=1e-10)):
        raise ValueError("Saved flow must be ordered and displacement must equal R(g)-g")
    return available


def load_public_ablation(run):
    """Read exactly five public files; never deserialize a checkpoint/cache."""
    root = Path(run).resolve()
    fingerprints = {name: sha256(root / name) for name in INPUT_FILES}
    manifest, summary, history, status = [_json(root / name) for name in INPUT_FILES[:4]]
    if (manifest.get("schema_version") != 5 or summary.get("status") != "complete"
            or status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action", False)):
        raise ValueError("Only a successfully completed scalar closure ablation can be reported")
    for key, filename in (("summary_sha256", "summary.json"),
                          ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != fingerprints[filename]:
            raise ValueError(f"Completed status {key} disagrees with public SHA256")
    for key, filename in (("manifest_sha256", "manifest.json"),
                          ("history_progress_sha256", "history_progress.json")):
        if key in status and status[key] != fingerprints[filename]:
            raise ValueError(f"Completed status {key} disagrees with public SHA256")
    steps = manifest.get("options", {}).get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Manifest must pin positive additional update steps")
    if manifest.get("theory_dim", 1) != 1 or manifest.get("labels_used_in_training") is not False:
        raise ValueError("Expected an unsupervised one-dimensional representation ablation")
    if (not manifest.get("source_pins") or not manifest.get("paired_schedule_sha256")
            or summary.get("source_pins") != manifest["source_pins"]
            or summary.get("paired_schedule_sha256") != manifest["paired_schedule_sha256"]
            or summary.get("frozen_source_verified") is not True):
        raise ValueError("Source fingerprints and paired schedule must be pinned and verified")
    if summary.get("critical_r") is not None or summary.get("thermal_nu") is not None:
        raise ValueError("This ablation must not claim a critical r or thermal exponent")
    initial = {}
    for arm in ("A", "B"):
        receipt = summary.get("arms", {}).get(arm, {})
        if receipt.get("actual_steps") != steps or receipt.get("C_updates") != 0:
            raise ValueError("A/B actual updates must match the plan and C must remain frozen")
        for module in ("Q", "R"):
            count = receipt.get("optimizer_steps", {}).get(module, {})
            if count.get("min") != steps or count.get("max") != steps or count.get("parameters", 0) <= 0:
                raise ValueError(f"{arm}.{module} Adam counters disagree with the endpoint")
        initial[arm] = _hashes(receipt.get("initial_module_sha256"), f"{arm} initial")
        final = _hashes(receipt.get("final_module_sha256"), f"{arm} final")
        if initial[arm]["C"] != final["C"]:
            raise ValueError("Frozen C module fingerprint changed")
        for numerical_root in receipt.get("numerical_fixed_points", []):
            if (not isinstance(numerical_root, dict)
                    or not np.isfinite(numerical_root.get("g", np.nan))
                    or numerical_root.get("thermal_nu") is not None
                    or numerical_root.get("physical_critical_point_certified", False)):
                raise ValueError("Saved numerical roots cannot certify physical criticality")
    if initial["A"] != initial["B"]:
        raise ValueError("A/B must begin from identical Q/C/R source module fingerprints")
    source = summary.get("source_module_sha256", manifest.get("source_module_sha256",
        summary.get("arms", {}).get("source", {}).get("final_module_sha256")))
    if source is not None and _hashes(source, "source") != initial["A"]:
        raise ValueError("A/B warm starts disagree with the pinned source modules")
    _history(history, steps)
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    endpoints = {}
    for name in NAMES:
        endpoint = {key[len(name) + 1:]: value for key, value in arrays.items() if key.startswith(name + "_")}
        if not endpoint:
            raise ValueError(f"Missing public arrays for endpoint {name}")
        _validate_endpoint_arrays(endpoint, name)
        endpoints[name] = endpoint
    for name in ("A", "B"):
        for split in ("train", "test"):
            for suffix in ("labels", "cell_ids", "feature_target"):
                if not np.array_equal(endpoints["source"][f"{split}_{suffix}"], endpoints[name][f"{split}_{suffix}"]):
                    raise ValueError(f"Paired {split} {suffix} changed between source/A/B")
    if fingerprints != {name: sha256(root / name) for name in INPUT_FILES}:
        raise ValueError("Public sources changed while reading")
    return PublicAblation(root, manifest, summary, status, history, arrays, endpoints, fingerprints)


def _root_support(roots, arrays):
    """Annotate existing endpoint roots only; no interpolation or root search."""
    result = []
    inputs = arrays["train_direct"][:, :-1, 0].ravel()
    for root in roots:
        g = float(root["g"])
        bounds = []
        for split in ("train", "test"):
            for scale in range(arrays[f"{split}_direct"].shape[1]):
                values = arrays[f"{split}_direct"][:, scale, 0]
                bounds.append(dict(split=split, scale=scale, minimum=float(values.min()), maximum=float(values.max()),
                    within_saved_range=bool(values.min() <= g <= values.max())))
        result.append(dict(saved_root=root, training_input_range=[float(inputs.min()), float(inputs.max())],
            within_training_input_range=bool(inputs.min() <= g <= inputs.max()), support_by_scale=bounds,
            physical_critical_point_certified=False, critical_r=None, thermal_nu=None))
    return result


def _dispersion(arrays, source):
    rows = []
    for split in ("train", "test"):
        for scale in range(arrays[f"{split}_direct"].shape[1]):
            values = arrays[f"{split}_direct"][:, scale, 0]
            old = source[f"{split}_direct"][:, scale, 0]
            std, source_std = float(np.std(values, ddof=0)), float(np.std(old, ddof=0))
            rows.append(dict(split=split, scale=scale, population_std=std, source_population_std=source_std,
                std_ratio_to_source=std / source_std if source_std > 0 else None,
                mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max())))
    return dict(rows=rows, warning="保留方差不等于保留物理信息；标准差变化需与归一闭合和共同原生统计误差一起判断。")


def endpoint_metrics(public, name):
    arrays = public.endpoints[name]
    available = _optional_feature_arrays(arrays)
    receipt = public.summary.get("arms", {}).get(name, {})
    if name == "source":
        roots = public.summary.get("source_numerical_fixed_points", receipt.get("numerical_fixed_points",
            public.summary.get("source", {}).get("numerical_fixed_points", [])))
    else:
        roots = receipt.get("numerical_fixed_points", [])
    closure = prediction_metrics(arrays)
    # Supplement each coordinate's own normalization with the unchanged source
    # standard deviation, without declaring raw latent cross-arm rankings.
    for row in closure["rows"]:
        reference = public.endpoints["source"]["train_direct"][:, row["target_scale"]]
        std = float(np.sqrt(np.mean(np.var(reference, axis=0, ddof=0))))
        row["original_source_train_target_population_std"] = std
        row["source_std_scaled_rmse_diagnostic_only"] = row["model"]["absolute_rmse"] / std if std > 0 else None
    return dict(actual_additional_steps=receipt.get("actual_steps", 0),
        optimizer_steps=receipt.get("optimizer_steps", {}), C_updates=receipt.get("C_updates", 0),
        q_reconstruction=reconstruction_metrics(arrays, available), latent_closure=closure,
        native_statistic_prediction=prediction_metrics(arrays, features=True, available=available),
        r_information=rank_correlations(arrays), dispersion=_dispersion(arrays, public.endpoints["source"]),
        saved_numerical_roots_with_support=_root_support(roots, arrays),
        critical_r=None, thermal_nu=None, physical_critical_point_certified=False)


def _paired_native(public):
    rows = []
    for split in ("train", "test"):
        target = public.endpoints["source"][f"{split}_feature_target"]
        for scale in range(target.shape[1]):
            a = public.endpoints["A"][f"{split}_feature_reconstruction"][:, scale] - target[:, scale]
            b = public.endpoints["B"][f"{split}_feature_reconstruction"][:, scale] - target[:, scale]
            delta = np.mean(b * b - a * a, axis=-1)
            rows.append(dict(split=split, scale=scale, B_minus_A_per_physical_chain_mse=delta.tolist(),
                mean_B_minus_A_mse=float(delta.mean()), fraction_chains_with_lower_B_mse=float(np.mean(delta < 0))))
    return dict(all_native_targets_exactly_equal=True, physical_chain_order_paired=True,
                reconstruction_B_minus_A=rows, bootstrap_performed=False)


def plot_flow(public, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.1), constrained_layout=True)
    kinds = {"attracting": "数值吸引", "repelling": "数值排斥", "neutral": "数值中性"}
    for column, name in enumerate(NAMES):
        a, top, bottom = public.endpoints[name], axes[0, column], axes[1, column]
        g, mapped, drift = a["flow_g"], a["flow_R"], a["flow_displacement"]
        top.plot(g, mapped, label="保存的 R(g)", color="#21679a")
        top.plot(g, g, "--", color=".5", label="恒等映射")
        bottom.plot(g, drift, label="保存的 R(g)−g", color="#21679a")
        bottom.axhline(0, color=".5", linestyle="--", linewidth=1)
        indices = np.unique(np.linspace(0, len(g) - 1, min(16, len(g))).astype(int))
        bottom.quiver(g[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
            angles="xy", scale_units="xy", scale=1, width=.004, color="#d37c29")
        support = a["train_direct"][:, :-1, 0].ravel()
        roots = endpoint_metrics(public, name)["saved_numerical_roots_with_support"]
        for ax in (top, bottom):
            ax.axvspan(support.min(), support.max(), alpha=.13, color="#68aeda", label="各自训练输入范围")
            ax.grid(alpha=.2)
        for index, row in enumerate(roots):
            root = row["saved_root"]
            x = root["g"]
            top.scatter(x, x, marker="*", s=100, color="#ab2964", zorder=5)
            bottom.scatter(x, 0, marker="*", s=100, color="#ab2964", zorder=5,
                label="端点已保存数值根" if index == 0 else None)
            location = "范围内" if row["within_training_input_range"] else "范围外"
            bottom.annotate(f"{kinds.get(root.get('classification'), '保存数值根')}，{location}", (x, 0),
                xytext=(4, 8), textcoords="offset points", fontsize=8)
        if not roots:
            bottom.text(.03, .94, "端点未保存数值根", transform=bottom.transAxes, va="top", fontsize=9)
        top.set(title=LABELS[name], xlabel="各自输入坐标 g", ylabel="输出 R(g)")
        bottom.set(xlabel="各自输入坐标 g", ylabel="位移 R(g)−g")
        top.legend(fontsize=8)
        bottom.legend(fontsize=8)
    fig.suptitle("冻结粗粒化的一维闭合消融：仅展示保存流场，不搜索新根或认证临界点")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)


def plot_coupling(public, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.3), constrained_layout=True)
    for name, ax in zip(NAMES, axes):
        a = public.endpoints[name]
        for scale, color in enumerate(("#237b9b", "#d08231", "#9467a8")):
            for split, marker, style, label in (("train", "o", "-", "训练"), ("test", "s", "--", "留出")):
                ids, r, values = a[f"{split}_cell_ids"], a[f"{split}_labels"][:, 0], a[f"{split}_direct"][:, scale, 0]
                means = sorted((float(r[ids == cell][0]), float(values[ids == cell].mean())) for cell in np.unique(ids))
                ax.scatter(r, values, color=color, marker=marker, s=13, alpha=.25)
                ax.plot(*zip(*means), color=color, marker=marker, linestyle=style, markersize=4,
                    label=f"k={scale}，{label}条件均值")
        ax.set(title=LABELS[name], xlabel="生成参数 r（仅事后评价）", ylabel="各自直接表示 g")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("参数与表示的关系：只使用评价标签，不将跨臂坐标视为同一物理量")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)


def plot_losses(public, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.3), constrained_layout=True)
    groups = [("Q 表示损失", [("q_loss", "总表示目标"), ("reconstruction", "统计重构"),
                ("subset_micro", "微观子集一致性"), ("subset_coarse", "粗尺度子集一致性")]),
        ("R 闭合损失", [("r_loss", "总闭合目标"), ("closure_one_step", "单步闭合"),
                ("closure_rollout", "微观起点多步闭合")]),
        ("表示闭合与防塌缩", [("q_compatible_closure", "表示闭合兼容"),
                ("q_closure_compatibility", "表示闭合兼容"),
                ("q_closure", "表示闭合兼容"), ("q_guard", "共同防塌缩"),
                ("variance_guard", "共同防塌缩"), ("anchor_guard", "共同坐标锚定")])]
    aliases = {"reconstruction": "q_reconstruction", "subset_micro": "q_subset_micro",
        "subset_coarse": "q_subset_coarse", "closure_one_step": "r_one_step", "closure_rollout": "r_rollout"}
    for row_index, arm in enumerate(("A", "B")):
        for ax, (title, entries) in zip(axes[row_index], groups):
            for key, label in entries:
                points = [(row["step"], row["arms"][arm].get(key, row["arms"][arm].get(aliases.get(key))))
                          for row in public.history["history"]]
                points = [(step, value) for step, value in points if isinstance(value, (int, float)) and np.isfinite(value)]
                if points:
                    ax.plot(*zip(*points), label=label)
            ax.set(title=f"{LABELS[arm]}\n{title}", xlabel="额外训练更新次数", ylabel="已保存损失值")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            else:
                ax.text(.05, .8, "未保存该项曲线", transform=ax.transAxes)
    fig.suptitle("配对批次的损失曲线：仅读取训练日志；冻结 C，不计算信息项")
    fig.savefig(output / "loss_comparison.png", dpi=180)
    plt.close(fig)


def write_report(run, output, *, font=None):
    public = load_public_ablation(run)
    output = Path(output).resolve()
    if output == public.root or public.root.is_relative_to(output):
        raise ValueError("Report must not replace or contain the source run")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_flow(public, output, plt)
    plot_coupling(public, output, plt)
    plot_losses(public, output, plt)
    comparison = dict(schema_version=1, status="complete", experiment="phi4_scalar_closure_compatibility_ablation",
        source_pins=public.manifest["source_pins"], paired_schedule_sha256=public.manifest["paired_schedule_sha256"],
        endpoints={name: endpoint_metrics(public, name) for name in NAMES}, native_pairing=_paired_native(public),
        data_warnings=public.summary.get("data_warnings", public.manifest.get("data_warnings", [])),
        frozen_source_verified=True, labels_used_in_training=False,
        latent_coordinates_identified_across_arms=False, latent_raw_error_cross_arm_ranking_performed=False,
        normalization="各端点原 TRAIN 同目标尺度总体标准差，ddof=0；共同原生统计目标逐元素一致。",
        root_search_performed=False, critical_r=None, thermal_nu=None, equilibrium_certified=False,
        physical_critical_point_certified=False, limitations=LIMITATIONS)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after = {name: sha256(public.root / name) for name in INPUT_FILES}
    if after != public.fingerprints:
        raise ValueError("Public source changed during report generation")
    digests = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = dict(schema_version=1, status="complete", source=str(public.root),
        sources_sha256_before=public.fingerprints, sources_sha256_after=after, public_sources_unchanged=True,
        model_loaded=False, physical_data_loaded=False, cache_loaded=False, GPU_used=False,
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
    parser.add_argument("--run", "--training", dest="run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.run, args.output, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
