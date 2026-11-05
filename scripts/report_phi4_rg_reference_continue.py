"""只读比较 φ⁴ 同作用量基准的 3000 步端点和精确续训 6000 步端点。

Only public JSON and saved evaluation NPZ files are read. The original public
physical diagnostics are copied byte-for-byte, never resampled or recomputed.
No model/checkpoint loading, new root search, zero interpolation or exponent fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts import report_phi4_qcr as generic
from scripts import report_phi4_rg_reference as reference
from scripts.report_gaussian_scalar_qcr import (
    _optional_feature_arrays, prediction_metrics, rank_correlations,
    reconstruction_metrics,
)


SOURCE_STEPS, ADDITIONAL_STEPS, TOTAL_STEPS = 3000, 3000, 6000
COPIED_PHYSICS = ("physics_vs_r.png", "physics_summary.json", "physics_arrays.npz")
PUBLIC_FILES = tuple(sorted(("R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png",
    "comparison.json", "report.json", "status.json", *(f"source_physics/{name}" for name in COPIED_PHYSICS))))
LIMITATIONS = [
    "从原 3000 步端点恢复 Q/C/R、三套 Adam 状态及两个训练随机数流，联合追加 3000 步，总计 6000 步；不是仅 R 收尾，也不是重新初始化。",
    "数据、模型、学习率、独立梯度与等权 all-start 目标不变；原实验、数据和报告不覆盖。",
    "Q 在续训中改变，两个端点的 g 坐标没有对齐保证；流场分面使用各自坐标，不能把跨端点绝对 g 误差当成同一物理量直接排名。",
    "原生 105 维软局部特征的定义和尺度归一化不变，但 C 继续更新会改变 k≥1 的目标分布；这些误差是各端点表示条件下的诊断，不构成同物理目标的跨 C 优劣排名。",
    "每个端点均并列绝对 RMSE、该端点 TRAIN 同目标尺度总体标准差 NRMSE、源尺度恒等预测和 TRAIN 同尺度均值基线；不混用两个端点的分母。",
    "原物理统计三份公共文件逐字节复制并核对 SHA256；未重新采样、打开原始构型、估计 MI 或产生新的热化证据。",
    "仅使用已保存的 R(g)、R(g)−g 和数值根；不搜索、插值新根，不把数值根认证为物理相变，也不拟合临界指数。",
    "保留原采样的 ESS／热冷初态诊断警告；单尺寸 36→12→4、一维 g、单训练种子及有限链不保证平衡、完整 RG 闭合或普适指数恢复。",
]


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _safe_file(root, relative):
    relative = Path(relative)
    candidate = (root / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts or not candidate.is_relative_to(root)
            or candidate.suffix not in (".json", ".npz", ".png") or not candidate.is_file()):
        raise ValueError("Expected an in-root public JSON/NPZ/PNG file")
    return candidate


def _public_run(path):
    root = Path(path).resolve()
    for name in generic.INPUT_FILES:
        _safe_file(root, name)
    return generic.load_public_run(root)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_comparison(source, run):
    """Check public exact-resume receipts and the unchanged physical split."""
    parent = Path(source).resolve()
    old, new = _public_run(parent / "run"), _public_run(run)
    if old.root == new.root:
        raise ValueError("Source and continued run must be different directories")
    for endpoint, steps in ((old, SOURCE_STEPS), (new, TOTAL_STEPS)):
        m = endpoint.manifest
        if (endpoint.summary.get("actual_steps") != steps or m.get("options", {}).get("steps") != steps
                or m.get("lattice_sizes") != [36, 12, 4] or m.get("theory_dim") != 1
                or m.get("block_size") != 3 or m.get("stride") != 3
                or m.get("closure_objective") != "all_start_anchor"
                or m.get("closure_scale_weights") is not None
                or m.get("learning_rate") != .001 or m.get("labels_used_in_training") is not False):
            raise ValueError("Expected the unchanged b3 dg1 equal-all-start 3000/6000-step benchmark")
    for obj in (new.manifest, new.summary):
        if (obj.get("source_steps") != SOURCE_STEPS or obj.get("additional_steps") != ADDITIONAL_STEPS
                or obj.get("profile_only") is not False or obj.get("scientific_result") is not True
                or any(obj.get(key) is not False for key in ("q_reinitialized", "data_resampled", "loss_changed"))):
            raise ValueError("Continuation must be a scientific 3000-step joint resume without new data/loss/init")
    if new.summary.get("fresh_adam") is not False:
        raise ValueError("Continuation must restore Adam, not start fresh Adam")
    prov = new.manifest.get("source_provenance")
    if not isinstance(prov, dict) or prov != new.summary.get("source_provenance"):
        raise ValueError("Manifest and summary source provenance disagree")
    if (prov.get("source_steps") != SOURCE_STEPS
            or prov.get("final_state_sha256") != old.summary.get("final_state_sha256")
            or prov.get("final_module_sha256") != old.summary.get("final_module_sha256")
            or not _digest(prov.get("checkpoint_sha256"))
            or prov.get("data_manifest_sha256") != old.manifest.get("data_manifest_sha256")):
        raise ValueError("Resume source model/data/checkpoint provenance disagrees")
    for name in ("manifest.json", "summary.json", "history_progress.json", "status.json"):
        if prov.get("public_files_sha256", {}).get("run/" + name) != old.fingerprints[name]:
            raise ValueError("Resume source public SHA256 fingerprints disagree")
    if (new.manifest.get("initial_state_sha256") != old.summary.get("final_state_sha256")
            or new.manifest.get("training_start_state_sha256") != old.summary.get("final_state_sha256")
            or new.manifest.get("training_start_module_sha256") != old.summary.get("final_module_sha256")
            or new.summary.get("initial_state_sha256") != old.summary.get("final_state_sha256")):
        raise ValueError("Continuation must begin at the complete source endpoint")
    q_receipt = new.manifest.get("q_initialization_receipt", {})
    if (new.manifest.get("q_initialization") != "checkpoint_restore_no_calibration"
            or new.manifest.get("source_q_initialization") != old.manifest.get("q_initialization")
            or new.manifest.get("source_q_initialization_receipt") != old.manifest.get("q_initialization_receipt")
            or q_receipt.get("mode") != "checkpoint_restore_no_calibration"
            or q_receipt.get("calibration_rerun") is not False
            or q_receipt.get("source_steps") != SOURCE_STEPS
            or q_receipt.get("source_checkpoint_sha256") != prov["checkpoint_sha256"]):
        raise ValueError("Q must restore the source endpoint without initialization or recalibration")
    for key, names in (("resumed_optimizer_state_sha256", ("Q", "C", "R")),
                       ("resumed_rng_sha256", ("batch_rng", "field_rng"))):
        values = new.manifest.get(key, {})
        if set(values) != set(names) or any(not _digest(values[name]) for name in names):
            raise ValueError("Explicit restored Adam and training RNG fingerprints are required")
    # Only execution/continuation receipts may change the historical manifest.
    mutable = {"options", "initial_state_sha256", "initial_state_sha256_semantics",
        "training_start_state_sha256", "training_start_module_sha256", "optimizer", "device",
        "max_training_seconds_semantics", "initialization", "q_initialization", "q_initialization_receipt"}
    if any(new.manifest.get(key) != value for key, value in old.manifest.items() if key not in mutable):
        raise ValueError("Source model, loss, data or evaluation policy changed during continuation")
    operational = {"output", "device", "steps", "max_training_seconds", "max_cuda_bytes", "expected_gpu_uuid"}
    if any(new.manifest["options"].get(key) != value
           for key, value in old.manifest["options"].items() if key not in operational):
        raise ValueError("Source seed, minibatch or training policy changed")
    if (new.manifest.get("data_manifest_sha256") != old.manifest.get("data_manifest_sha256")
            or new.summary.get("data_manifest_sha256") != old.manifest.get("data_manifest_sha256")):
        raise ValueError("Source and continuation data fingerprints disagree")
    old_history, new_history = generic.history_rows(old.history), generic.history_rows(new.history)
    if new_history[:len(old_history)] != old_history or new_history[len(old_history)]["step"] != SOURCE_STEPS + 1:
        raise ValueError("Continued history must preserve the exact source prefix and start at 3001")
    for split in ("train", "test"):
        for suffix in ("labels", "cell_ids"):
            if not np.array_equal(old.arrays[f"{split}_{suffix}"], new.arrays[f"{split}_{suffix}"]):
                raise ValueError("Physical-chain order or held-out conditions changed")
        labels = old.arrays[f"{split}_labels"]
        if not np.all(labels[:, 1:] == np.array([1., 0.])):
            raise ValueError("Expected fixed u=1, h=0 physical parameters")
    r = np.unique(np.concatenate((old.arrays["train_labels"][:, 0], old.arrays["test_labels"][:, 0])))
    if r.shape != (13,) or not np.allclose(r, reference.R_VALUES, rtol=0, atol=1e-13):
        raise ValueError("Physical parameter grid differs from the reference benchmark")
    if not np.any(np.isclose(old.arrays["test_labels"][:, 0], reference.R_CRITICAL, rtol=0, atol=1e-13)):
        raise ValueError("Critical reference condition must remain held out")
    return parent, old, new


def _physics_sources(parent):
    root = (parent / "report").resolve()
    receipt_path = _safe_file(root, "report.json")
    before = {"report.json": generic.sha256(receipt_path)}
    receipt = generic._json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("public_sources_unchanged") is not True:
        raise ValueError("Original physics report must be successfully complete")
    for name in COPIED_PHYSICS:
        digest = generic.sha256(_safe_file(root, name))
        if (name not in receipt.get("public_files", []) or receipt.get("outputs_sha256", {}).get(name) != digest):
            raise ValueError("Original public physics file SHA256 disagrees with its report receipt")
        before[name] = digest
    summary = generic._json(_safe_file(root, "physics_summary.json"))
    if (summary.get("actual_steps") != SOURCE_STEPS or summary.get("reference") != reference.REFERENCE
            or summary.get("labels_used_in_training") is not False or summary.get("equilibrium_certified") is not False):
        raise ValueError("Original public physics identity or qualification differs")
    return root, summary, before


def endpoint_metrics(public):
    available = _optional_feature_arrays(public.arrays)
    if not all(available.values()) or public.arrays["train_feature_target"].shape[-1] != 105:
        raise ValueError("Continuation comparison requires all saved 105-dimensional native feature targets/predictions")
    inputs = public.arrays["train_direct"][:, :-1, 0]
    grid, drift = public.arrays["flow_g"], public.arrays["flow_displacement"]
    inside = (grid >= inputs.min()) & (grid <= inputs.max())
    roots = public.summary.get("numerical_fixed_points", public.summary.get("roots", []))
    return dict(actual_steps=public.summary["actual_steps"], optimizer_steps=public.summary["optimizer_steps"],
        latent_closure=prediction_metrics(public.arrays),
        native_statistic_prediction=prediction_metrics(public.arrays, features=True, available=available),
        q_reconstruction=reconstruction_metrics(public.arrays, available), r_information=rank_correlations(public.arrays),
        own_train_input_range=[float(inputs.min()), float(inputs.max())],
        saved_flow_grid_points_in_train_input_range=int(inside.sum()),
        saved_flow_displacement_range_in_train_input_range=(
            [float(drift[inside].min()), float(drift[inside].max())] if inside.any() else None),
        saved_numerical_roots=roots, saved_numerical_root_count=len(roots),
        saved_numerical_root_count_in_own_train_input_range=int(sum(inputs.min() <= root["g"] <= inputs.max() for root in roots)),
        root_search_performed=False, saved_grid_zero_interpolation_performed=False,
        critical_r=None, thermal_nu=None, physical_critical_point_certified=False)


def plot_flow(old, new, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row, public in enumerate((old, new)):
        g, mapped, drift = (public.arrays[key] for key in ("flow_g", "flow_R", "flow_displacement"))
        inputs = public.arrays["train_direct"][:, :-1, 0]
        axes[row, 0].plot(g, mapped, label="已保存 R(g)", color="#234f9c")
        axes[row, 0].plot(g, g, "--", color=".5", label="恒等映射")
        axes[row, 1].plot(g, drift, label="已保存位移 R(g)−g", color="#234f9c")
        axes[row, 1].axhline(0, linestyle="--", color=".5", linewidth=1)
        indices = np.unique(np.linspace(0, len(g)-1, min(20, len(g))).astype(int))
        axes[row, 1].quiver(g[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
            angles="xy", scale_units="xy", scale=1, color="#e47e30", width=.004)
        for col, ax in enumerate(axes[row]):
            ax.axvspan(float(inputs.min()), float(inputs.max()), alpha=.12, color="#234f9c", label="本端点训练输入范围")
            ax.set(xlabel="本端点输入 g", ylabel="R(g)" if col == 0 else "R(g)−g",
                   title=f"累计 {public.summary['actual_steps']} 步：" + ("完整离散映射" if col == 0 else "真实长度的流向箭头"))
            for index, root in enumerate(public.summary.get("numerical_fixed_points", public.summary.get("roots", []))):
                ax.scatter([root["g"]], [root["g"] if col == 0 else 0], marker="*", s=70,
                    color="#ac2663", label="已保存数值根（未认证物理）" if index == 0 else None)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("二维 φ⁴ 联合续训：上下使用各自 g 坐标，不做跨端点坐标对齐；未搜索或插值新根")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)


def plot_coupling(old, new, output, plt):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.3), constrained_layout=True)
    for row, public in enumerate((old, new)):
        for scale, ax in enumerate(axes[row]):
            for split, label, color in (("train", "训练", "#234f9c"), ("test", "留出", "#e47e30")):
                ids, labels, direct = (public.arrays[f"{split}_{suffix}"] for suffix in ("cell_ids", "labels", "direct"))
                ax.scatter(labels[:, 0], direct[:, scale, 0], s=14, alpha=.32, color=color, label=f"{label}物理链")
                cells = sorted(np.unique(ids), key=lambda cell: labels[ids == cell, 0][0])
                r = [float(labels[ids == cell, 0][0]) for cell in cells]
                ax.plot(r, [direct[ids == cell, scale, 0].mean() for cell in cells], "o-", markersize=3,
                    color=color, label=f"{label}条件均值")
                if scale:
                    rolled = public.arrays[f"{split}_rollout"]
                    ax.plot(r, [rolled[ids == cell, scale, 0].mean() for cell in cells], "--", color=color,
                        label=f"{label}微观起点 R 迭代")
            ax.axvline(reference.R_CRITICAL, color="#ac2663", linestyle=":", label="文献无限体积蒙特卡罗参考")
            ax.set(xlabel="生成参数 r（仅用于评价）", ylabel="本端点 g 坐标",
                title=f"累计 {public.summary['actual_steps']} 步，k={scale}，L={[36, 12, 4][scale]}")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.suptitle("r→g 与各自闭合预测：Q/C 继续更新，上下坐标和粗尺度目标未保证对齐")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)


def plot_losses(public, output, plt):
    groups = (("Q 表示目标（未加权）", (("reconstruction", "共享统计重构"), ("subset_micro", "微观子集一致性"), ("subset_coarse", "粗尺度子集一致性"))),
        ("C 连续场目标（未加权）", (("information_proxy", "平滑全壳信息代理"), ("variance_floor", "系综方差下限"), ("uniform_amplitude", "均匀块保持"))),
        ("R 闭合目标（未加权）", (("closure_one_step", "单步闭合"), ("closure_rollout", "微观起点多步闭合"))))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    rows = generic.history_rows(public.history)
    for ax, (title, entries) in zip(axes, groups):
        for key, label in entries:
            pairs = [(row["step"], row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
            if pairs:
                ax.plot(*zip(*pairs), label=label)
        ax.axvline(SOURCE_STEPS, linestyle="--", color="#ac2663", label="原端点／精确续训分界")
        ax.set(xlabel="累计联合更新次数", ylabel="本端点训练目标值", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("原始日志前缀逐项保持：Q/C/R 均追加 3000 步；目标随学习中的表示而变化")
    fig.savefig(output / "loss_comparison.png", dpi=180)
    plt.close(fig)


def return_files(output=None):
    if output is not None:
        root = Path(output).resolve()
        for name in PUBLIC_FILES:
            _safe_file(root, name)
    return list(PUBLIC_FILES)


def write_report(source, run, output, *, font=None):
    parent, old, new = load_comparison(source, run)
    physics_root, physics, physics_fingerprints = _physics_sources(parent)
    output = Path(output).resolve()
    for root in (parent, old.root, new.root):
        if root.is_relative_to(output) or output.is_relative_to(root):
            raise ValueError("Fresh report output must not overlap a source directory")
    if output.exists():
        raise FileExistsError("A fresh independent report directory is required")
    endpoints = {name: endpoint_metrics(public) for name, public in (("source_3000", old), ("continued_6000", new))}
    same_micro = {split: np.array_equal(old.arrays[f"{split}_feature_target"][:, 0],
        new.arrays[f"{split}_feature_target"][:, 0]) for split in ("train", "test")}
    if not all(same_micro.values()):
        raise ValueError("Unchanged microscopic physical targets must be identical at both endpoints")
    comparison = dict(schema_version=1, status="complete", source_steps=SOURCE_STEPS,
        additional_joint_QCR_steps=ADDITIONAL_STEPS, total_joint_QCR_steps=TOTAL_STEPS,
        endpoints=endpoints, reference=reference.REFERENCE, lattice_sizes=[36, 12, 4],
        normalization="each endpoint's own same-scale TRAIN target population standard deviation, ddof=0",
        latent_coordinates_aligned=False, coarse_C_targets_identical_not_assumed=True,
        native_feature_dimension=105, native_feature_definition_unchanged=True,
        microscopic_feature_targets_identical=same_micro,
        fair_across_C_physical_prediction_ranking_certified=False,
        original_sampling_diagnostics=physics.get("sampling_diagnostics"),
        original_sampling_quality_summary=physics.get("sampling_quality_summary"),
        labels_used_in_training=False, labels_used_in_report=True, equilibrium_certified=False,
        model_loaded=False, physical_data_loaded=False, inference_performed=False,
        root_search_performed=False, saved_grid_zero_interpolation_performed=False,
        critical_r=None, thermal_nu=None, limitations=LIMITATIONS)
    plt = generic._plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_physics").mkdir()
    for name in COPIED_PHYSICS:
        shutil.copyfile(_safe_file(physics_root, name), output / "source_physics" / name)
        if generic.sha256(output / "source_physics" / name) != physics_fingerprints[name]:
            raise ValueError("Copied public physics SHA256 mismatch")
    plot_flow(old, new, output, plt)
    plot_coupling(old, new, output, plt)
    plot_losses(new, output, plt)
    _write(output / "comparison.json", comparison)
    before = dict(source=old.fingerprints, continued=new.fingerprints, original_physics=physics_fingerprints)
    after = dict(source={name: generic.sha256(old.root / name) for name in old.fingerprints},
        continued={name: generic.sha256(new.root / name) for name in new.fingerprints},
        original_physics={name: generic.sha256(physics_root / name) for name in physics_fingerprints})
    if before != after:
        raise ValueError("Public source files changed during continuation reporting")
    files = return_files()
    report = dict(schema_version=1, status="complete", source_parent=str(parent), source_run=str(old.root),
        continued_run=str(new.root), sources_sha256_before=before, sources_sha256_after=after,
        public_sources_unchanged=True, original_physics_copied_without_recomputation=True,
        outputs_sha256={name: generic.sha256(output / name) for name in files if name not in ("report.json", "status.json")},
        public_files=files, return_files=files, model_loaded=False, physical_data_loaded=False,
        GPU_used=False, inference_performed=False, root_search_performed=False,
        saved_grid_zero_interpolation_performed=False, critical_r=None, thermal_nu=None, limitations=LIMITATIONS)
    _write(output / "report.json", report)
    _write(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, equilibrium_certified=False, thermal_nu=None))
    return_files(output)
    return report, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "run", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.source, args.run, args.output, font=args.font)
    print(json.dumps(dict(status="complete", output=str(args.output.resolve()), return_files=report["return_files"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
