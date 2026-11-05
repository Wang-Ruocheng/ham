"""低 k 新训练与历史等权／高 k 的三方公开报告；不加载模型或缓存。"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.report_ising_closure_scale_ablation import (
    SCALES, PATHS, INITIAL_FINGERPRINT_QUALIFICATION, closure_summary,
    encoding_metadata, initialization_metadata, load_arrays, training_start_fingerprints_match,
)
from scripts.ising_qcr_evaluation import require_matching_estimators
from scripts.report_ising_closure_scale_late import validate_reference
from scripts.report_ising_future_physics import chinese_plot_context, public_sha, read_json, save_chinese


ARMS = {
    "reference_control": "历史等权 [1,1]（未重训）",
    "reference_late": "历史高 k [0.2,1.8]（未重训）",
    "early": "本次低 k [1.8,0.2] 新训练",
}
PROFILES = {"reference_control": [1., 1.], "reference_late": [.2, 1.8], "early": [1.8, .2]}
FIGURES = ("comparison.png", "loss_comparison.png", "R_vs_g.png", "coupling_by_scale.png")


def _normalized_config(config):
    result = copy.deepcopy(config)
    result["loss"].pop("closure_scale_weights", None)
    result["data"].pop("cache", None)
    return result


def validate_weighted(manifest, summary, metrics, arm, expected_steps):
    """Check actual weighted metadata without rewriting a historical protocol."""
    cfg, profile = manifest["config"], PROFILES[arm]
    model, training = cfg["model"], cfg["training"]
    if (cfg.get("schema_version") != 5 or model.get("dimension") != 2 or model.get("theory_dim") != 1
            or model.get("block_factor") != 3 or model.get("kernel_layout") != "nonoverlapping"
            or model.get("kernel_initialization") != "random" or model.get("kernel_sampling") != "bernoulli"
            or model.get("fixed_kernel") is not False or model.get("affine_reconstruction") is not True
            or training.get("max_rg_steps") != 2 or training.get("gradient_mode") != "qc_r_separate"
            or training.get("seed") != 91):
        raise ValueError("需要随机可训练 C、独立梯度、seed91 的 schema5 标量 b3 配置")
    if (manifest.get("labels_used_in_training") is not False or summary.get("config") != cfg
            or manifest.get("source") is not None or manifest.get("data_driven_initialization") is not None):
        raise ValueError("必须无标签训练并全新初始化；配置必须一致")
    initial = manifest.get("initial_module_sha256", {})
    if (set(initial) != {"Q", "C", "R"} or not all(initial.values())
            or summary.get("contract", {}).get("initial_module_sha256") != initial):
        raise ValueError("缺少一致的 Q/C/R 初始指纹")
    contract = manifest.get("closure_scale_weighting", {})
    if (cfg["loss"].get("closure_scale_weights") != profile or contract.get("weights") != profile
            or contract.get("target_levels") != [1, 2]
            or contract.get("application") != ["one_step_commutator", "microscopic_start_rollout"]
            or contract.get("labels_used") is not False
            or summary.get("contract", {}).get("closure_scale_weighting") != contract):
        raise ValueError("单步与微观起点 rollout 的尺度权重协议不匹配")
    np.testing.assert_allclose(contract.get("normalized_weights", []), np.asarray(profile) / sum(profile))
    future = manifest.get("q_future_physics", {})
    if (future.get("weight") != 0. or future.get("labels_used", False) is not False
            or manifest.get("auxiliary_future_parameters", 0) != 0):
        raise ValueError("不得混入未来物理预测目标或参数")
    actual, history = summary.get("actual_epochs"), summary.get("history", [])
    epochs = [row.get("epoch") for row in history]
    if (type(actual) is not int or not 0 < actual <= expected_steps or not history
            or any(type(epoch) is not int for epoch in epochs) or epochs != sorted(set(epochs))
            or epochs[-1] != actual or summary.get("requested_epochs") != expected_steps
            or training.get("epochs") != expected_steps):
        raise ValueError("固定终点、实际步数与训练历史不一致")
    if any(isinstance(v, (int, float)) and not np.isfinite(v) for row in history for v in row.values()):
        raise ValueError("训练历史存在非有限值")
    if (metrics.get("status") != "complete" or metrics.get("actual_steps") != actual
            or metrics.get("scales") != list(SCALES) or not metrics.get("model_sha256_before")
            or metrics.get("model_sha256_before") != metrics.get("model_sha256_after")):
        raise ValueError("采集实际步数、尺度或模型指纹不一致")
    complete = actual == expected_steps and not any(summary.get(key, False) for key in
        ("stopped_by_wall_limit", "stopped_by_training_limit", "stopped_by_train_plateau"))
    return dict(status="complete" if complete else "incomplete", actual_steps=actual,
        requested_steps=expected_steps, trained_to_fixed_endpoint=complete, scale_contract=contract,
        config=cfg, history=history, collection_metrics=metrics, data_sha256=manifest.get("data_sha256"),
        initial_module_sha256=initial, initial_fingerprints_available=True,
        original_manifest=manifest, original_summary=summary,
        **initialization_metadata(manifest, summary))


def stratified_flow_points(arrays, split="test", max_points_per_scale=80):
    """Sample each scale independently; never alias a flattening stride with scale count."""
    if type(max_points_per_scale) is not int or max_points_per_scale < 1:
        raise ValueError("每尺度抽点数必须为正整数")
    per_draw = "direct_draws_" + split in arrays
    direct = arrays[("direct_draws_" if per_draw else "direct_") + split]
    mapped = arrays[("R_g_draws_" if per_draw else "R_g_") + split]
    g, rg, levels = [], [], []
    for level in range(len(SCALES)):
        x, y = direct[..., level, 0].reshape(-1), mapped[..., level, 0].reshape(-1)
        selected = np.linspace(0, len(x) - 1, min(len(x), max_points_per_scale), dtype=int)
        g.append(x[selected]); rg.append(y[selected]); levels.append(np.full(len(selected), level))
    return np.concatenate(g), np.concatenate(rg), np.concatenate(levels), per_draw


def flow_identity_bounds(arrays, split="test"):
    """Include every plotted saved value, including all R(mean g) markers."""
    prefixes = ["direct_", "R_g_", "R_of_mean_g_"]
    if "direct_draws_" + split in arrays: prefixes += ["direct_draws_", "R_g_draws_"]
    values = [arrays[key + split] for key in prefixes]
    return min(float(v.min()) for v in values), max(float(v.max()) for v in values)


def build_report(root, expected_steps=6000, *, execution_smoke=False):
    root, inputs, arms, panels = Path(root), {}, {}, {}
    def load(path):
        inputs[str(path.relative_to(root))] = public_sha(path)
        return read_json(path)
    optional = {key: load(root / (key + ".json")) if (root / (key + ".json")).is_file() else None
                for key in ("job", "plan", "input_reuse", "preflight_reuse", "dataset_receipt")}
    for arm in ARMS:
        run = root / ("run_early" if arm == "early" else arm)
        role = "new_training" if arm == "early" else "previously_completed_reference_not_retrained"
        names = ("manifest.json", "summary.json", "report/metrics.json", "report/encodings.npz")
        if not all((run / name).is_file() for name in names):
            arms[arm] = dict(status="incomplete", execution_role=role, reason="缺少完整公开输入")
            continue
        manifest, summary, metrics = [load(run / name) for name in names[:3]]
        info = (validate_reference(manifest, summary, metrics, expected_steps) if arm == "reference_control"
                else validate_weighted(manifest, summary, metrics, arm, expected_steps))
        info["execution_role"] = role
        info.setdefault("initial_module_sha256", manifest.get("initial_module_sha256"))
        path = run / names[3]
        arrays = load_arrays(path)
        inputs[str(path.relative_to(root))] = public_sha(path)
        if metrics.get("encodings_sha256") != inputs[str(path.relative_to(root))]:
            raise ValueError("保存数组与原采集回执指纹不符")
        info.update(encoding_metadata(metrics, arrays))
        if not isinstance(info.get("data_sha256"), str) or not info["data_sha256"]:
            raise ValueError("缺少来源数据指纹")
        arms[arm], panels[arm] = info, arrays
    # Gate all pairwise coordinates and historical fixed-denominator metrics.
    estimator = require_matching_estimators(arms[arm]["collection_metrics"] for arm in panels)
    pairs = {}
    for left, right in itertools.combinations(panels, 2):
        a, b = arms[left], arms[right]
        training_start_equal = training_start_fingerprints_match(a, b)
        if a["data_sha256"] != b["data_sha256"]:
            raise ValueError("三方物理数据指纹必须一致")
        compatible = _normalized_config(a["config"]) == _normalized_config(b["config"])
        if not compatible: raise ValueError("除尺度权重和缓存路径外配置不同")
        for split in ("train", "test"):
            for prefix in ("couplings_", "group_ids_"):
                np.testing.assert_array_equal(panels[left][prefix + split], panels[right][prefix + split])
        metrics_a, metrics_b = a["collection_metrics"], b["collection_metrics"]
        qc_equal = all(metrics_a.get(key) is not None and metrics_a.get(key) == metrics_b.get(key)
                       for key in ("Q_sha256", "C_sha256"))
        mean_equal = all(np.array_equal(panels[left]["direct_" + split], panels[right]["direct_" + split])
                         for split in ("train", "test"))
        draws_available = all("direct_draws_" + split in panels[arm] for arm in (left, right) for split in ("train", "test"))
        draws_equal = (all(np.array_equal(panels[left]["direct_draws_" + split], panels[right]["direct_draws_" + split])
                           for split in ("train", "test")) if draws_available else None)
        initials_available = bool(a.get("initial_module_sha256") and b.get("initial_module_sha256"))
        initial_equal = (a["initial_module_sha256"] == b["initial_module_sha256"]) if initials_available else None
        if {left, right} == {"reference_late", "early"} and initial_equal is not True:
            raise ValueError("低 k 新训练必须与历史高 k 使用相同全新 Q/C/R 初始指纹")
        pairs[left + "__" + right] = dict(compatible_config_except_weights_and_cache_path=compatible,
            encoding_estimator=estimator, encoding_estimators_matched=True,
            shared_QC_fingerprints=qc_equal, identical_direct_arrays=mean_equal,
            identical_direct_draw_arrays=draws_equal, initial_fingerprints_available=initials_available,
            same_initial_module_fingerprints=initial_equal,
            same_training_start_module_fingerprints=training_start_equal,
            initial_fingerprint_comparison_scope=INITIAL_FINGERPRINT_QUALIFICATION,
            saved_coordinate_values_equal=mean_equal and draws_equal is True,
            shared_coordinate_evaluation_verified=qc_equal and mean_equal and draws_equal is True)
    baseline = panels.get("reference_control", {}).get("direct_train")
    for arm, arrays in panels.items():
        arms[arm]["closure_of_C_means"] = {split: closure_summary(arrays, split, baseline) for split in ("train", "test")}
        arms[arm]["closure_per_C_draw"] = {split: closure_summary(arrays, split, baseline, per_draw=True) for split in ("train", "test")}
        arms[arm]["jensen_gap_RMSE"] = {split: float(np.sqrt(np.mean((arrays["R_g_" + split]
            - arrays["R_of_mean_g_" + split]) ** 2))) for split in ("train", "test")}
        arms[arm]["physical_chain_count"] = {split: len(arrays["direct_" + split]) for split in ("train", "test")}
        arms[arm]["C_draw_count_per_chain"] = {split: (arrays["direct_draws_" + split].shape[1]
            if "direct_draws_" + split in arrays else None) for split in ("train", "test")}
    complete = arms["early"]["status"] == "complete"
    available = complete and all(arms[arm]["status"] == "complete" for arm in ARMS)
    return dict(schema_version=1, status="complete" if complete else "incomplete", workflow_complete=complete,
        new_training_complete=complete, qualified_reference_comparison_available=available,
        formal_fresh_paired_comparison_available=False, execution_smoke=execution_smoke,
        scientific_result=not execution_smoke, requested_steps=expected_steps,
        experiment="低 k 权重 [1.8,0.2] 新训练；历史等权和高 k 不重训",
        profiles=PROFILES, normalized_profiles={k: (np.asarray(v) / sum(v)).tolist() for k, v in PROFILES.items()},
        new_training_arms=["early"], historical_reference_arms=["reference_control", "reference_late"],
        scales=list(SCALES), arms=arms, pairwise_coordinate_comparison=pairs,
        encoding_estimator=estimator, encoding_estimators_matched=True,
        dataset_receipt=optional["dataset_receipt"], input_reuse=optional["input_reuse"],
        preflight_reuse=optional["preflight_reuse"], input_sha256=inputs,
        thermal_nu=None, thermal_mode_certified=False, root_search_performed=False,
        weighted_objectives_compared_as_physical_improvement=False,
        flow_plot_sampling=dict(method="independent_stratified_per_scale", max_points_per_scale=80,
            identity_bounds="all saved plotted g/R(g)/R(mean g), not subsampled bounds"),
        interpretation=[
            "只有低 k 权重臂为本次新训练；历史等权和高 k 是已完成参考，不重训、不重新采集、不补造初始协议。",
            "三方仅改变 R 单步及微观起点 rollout 的尺度权重与缓存路径；外层权重、Q/C 目标和独立梯度保持不变，无 K 监督且未来物理目标为零。",
            "初始指纹只据实际元数据比较；同 seed 不等于历史缺失的初始协议已经认证。",
            INITIAL_FINGERPRINT_QUALIFICATION,
            "Q/C 参数指纹相同与已保存 direct 坐标相同分开报告；保存坐标相同不证明全部函数或参数相同。",
            "所有跨臂指标先验证编码估计器一致；历史全体 Q 口径不转换，也不与新版半子集均值混比。",
            "新版固定划分只保证估计器函数与训练一致，不等于训练随机划分的期望 loss。",
            "三个路径分别报告未加权绝对 RMSE、本臂 TRAIN 标准差 NRMSE、固定历史等权 TRAIN 分母 NRMSE；不同加权优化分数不能直接判优。",
            "固定等权分母不消除不同潜坐标的单位差异。108→12 为微观起点两步 rollout，36→12单步另列。",
            "逐 C 与先平均 C 的误差分开；C 重复不作为独立物理链，历史未保存逐 C rollout 时保留为空，绝不新增推理补齐。",
            "R(g)、R(平均 g)、平均 R(g) 分开；流场每尺度独立抽点，箭头为真实 R(g)-g，恒等线覆盖完整相关范围。",
            "有限扰动耦合只用于事后评价，不合并成唯一 g(K)；不加载模型、物理缓存或检查点，不搜根或估计指数。",
            "单训练种子、有限尺寸、有限样本与来源平衡/自相关警告保留；闭合误差降低不认证热模或临界指数。",
        ]), panels


def _missing(ax):
    ax.text(.5, .5, "尚无完整保存结果", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_report(result, panels, output, font=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    check_label = "合成执行检查（非科学结果）\n" if result.get("execution_smoke") else ""
    with chinese_plot_context(font) as selected:
        fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharey="row", layout="constrained")
        labels = [f"{SCALES[a]}→{SCALES[b]}" for a, b in PATHS]
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            for row, key, name in ((0, "rmse", "绝对 RMSE"), (1, "nrmse", "本臂 TRAIN 标准差 NRMSE"),
                                   (2, "fixed_uniform_nrmse", "固定等权 TRAIN 标准差 NRMSE")):
                for split, marker, name_split in (("train", "o", "训练"), ("test", "s", "测试")):
                    values = [v[key] if v[key] is not None else np.nan for v in result["arms"][arm]["closure_of_C_means"][split]]
                    axes[row, col].plot(labels, values, marker + "-", label=name_split)
                axes[row, col].set(title=title, ylabel=name); axes[row, col].legend(); axes[row, col].grid(alpha=.2)
        fig.suptitle(check_label + "三种尺度权重：未加权路径误差；先平均 C 重复")
        save_chinese(fig, output / "comparison.png", selected); plt.close(fig)

        groups = (("separation", "anchor", "coarse_consistency"), ("rsmi", "kernel_noncollapse", "kernel_sign"),
                  ("commutator", "rollout"), ("one_step_level_1", "one_step_level_2", "rollout_level_1", "rollout_level_2"))
        names = dict(separation="仿射重构", anchor="微观子集一致性", coarse_consistency="粗尺度子集一致性",
            rsmi="负经验互信息", kernel_noncollapse="方差下限", kernel_sign="颜色保持", commutator="加权单步闭合",
            rollout="加权多步闭合", one_step_level_1="单步 k=1", one_step_level_2="单步 k=2",
            rollout_level_1="多步 k=1", rollout_level_2="多步 k=2")
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), layout="constrained")
        for col, (arm, title) in enumerate(ARMS.items()):
            history = result["arms"][arm].get("history", [])
            for row, group in enumerate(groups):
                ax = axes[row, col]
                for color, key in enumerate(group):
                    for split, style, name_split in (("train", "-", "训练"), ("test", "--", "测试")):
                        rows = [v for v in history if split + "_" + key in v]
                        if rows: ax.plot([v["epoch"] for v in rows], [v[split + "_" + key] for v in rows],
                            style, color=f"C{color}", label=names[key] + "·" + name_split)
                ax.set(title=title + "：" + ("Q 分项", "C 分项", "R 加权分项", "R 未加权尺度分项")[row],
                    xlabel="实际更新次数", ylabel="原始记录分项", yscale="symlog")
                if ax.lines: ax.legend(fontsize=7); ax.grid(alpha=.2)
                else: _missing(ax)
        fig.suptitle(check_label + "历史日志不改口径；不同加权闭合总分不能直接判优；缺失记录不补造")
        save_chinese(fig, output / "loss_comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(18, 8), layout="constrained")
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            arrays = panels[arm]
            g, rg, levels, per_draw = stratified_flow_points(arrays)
            ax = axes[0, col]
            order = np.argsort(g)
            ax.plot(g[order], rg[order], lw=.6, label="保存点排序连线；非新网格推理")
            ax.scatter(g, rg, c=levels, cmap="viridis", vmin=0, vmax=2, s=8, alpha=.55,
                label="逐 C 的 R(g)" if per_draw else "平均 R(g)")
            ax.scatter(arrays["direct_test"].reshape(-1), arrays["R_of_mean_g_test"].reshape(-1),
                marker="x", s=8, alpha=.35, label="R(平均 g)")
            lower, upper = flow_identity_bounds(arrays)
            ax.plot([lower, upper], [lower, upper], "k--", lw=.8, label="恒等映射")
            ax.set(title=title, xlabel="输入 g", ylabel="输出 R(g) 或平均 R(g)")
            ax.legend(fontsize=7); ax.grid(alpha=.2)
            ax = axes[1, col]
            ax.quiver(g, levels, rg - g, np.zeros_like(g), levels, cmap="viridis", clim=(0, 2),
                angles="xy", scale_units="xy", scale=1, width=.003)
            ax.set(title="三个尺度分别抽点：真实 R(g)-g", xlabel="编码 g；箭头不归一化",
                ylabel="输入晶格边长 L", yticks=[0, 1, 2], yticklabels=list(map(str, SCALES)), ylim=(-.5, 2.5))
            ax.update_datalim(np.column_stack((rg, levels))); ax.autoscale_view(scaley=False); ax.grid(alpha=.2)
        fig.suptitle(check_label + "保存流场三方对照；不搜索不动点，不估计临界指数")
        save_chinese(fig, output / "R_vs_g.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(18, 10), layout="constrained")
        maximum = max((float(np.linalg.norm(a["couplings_" + s][:, 1:], axis=1).max())
                       for a in panels.values() for s in ("train", "test")), default=0.)
        norm, scatter = Normalize(0., max(maximum, 1e-12)), None
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            for row, scale in enumerate(SCALES):
                ax = axes[row, col]
                for split, marker, name in (("train", "o", "训练"), ("test", "^", "测试")):
                    coupling = panels[arm]["couplings_" + split]
                    scatter = ax.scatter(coupling[:, 0], panels[arm]["direct_" + split][:, row, 0],
                        c=np.linalg.norm(coupling[:, 1:], axis=1), norm=norm, cmap="viridis", marker=marker,
                        s=12, alpha=.6, label=name)
                ax.set(title=title + f"：L={scale}", xlabel="生成基准 K0（仅评价）", ylabel="编码 g")
                ax.legend(fontsize=7); ax.grid(alpha=.2)
        if scatter is not None: fig.colorbar(scatter, ax=axes.ravel().tolist(), label="六个有限扰动物理耦合的模长")
        fig.suptitle(check_label + "耦合到 g：不同有限扰动条件不连成唯一 g(K) 曲线")
        save_chinese(fig, output / "coupling_by_scale.png", selected); plt.close(fig)


def report_run(root, output=None, *, expected_steps=6000, font=None, make_plots=True, execution_smoke=False):
    root, output = Path(root), Path(output) if output else Path(root) / "report"
    if output.exists() and any(output.iterdir()): raise FileExistsError("不覆盖既有报告")
    result, panels = build_report(root, expected_steps, execution_smoke=execution_smoke)
    output.mkdir(parents=True, exist_ok=True)
    if make_plots: plot_report(result, panels, output, font)
    np.savez_compressed(output / "evaluation_arrays.npz", **{arm + "__" + key: value
        for arm, arrays in panels.items() for key, value in arrays.items()})
    result["public_files"] = ([*FIGURES] if make_plots else []) + ["report.json", "evaluation_arrays.npz", "status.json"]
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    status = {key: result[key] for key in ("status", "workflow_complete", "new_training_complete",
        "qualified_reference_comparison_available", "formal_fresh_paired_comparison_available", "execution_smoke", "scientific_result")}
    status.update(thermal_nu=None, public_files=result["public_files"], report_sha256=public_sha(output / "report.json"),
        evaluation_arrays_sha256=public_sha(output / "evaluation_arrays.npz"))
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-steps", type=int, default=6000); parser.add_argument("--font", type=Path)
    parser.add_argument("--execution-smoke", action="store_true", help="标记合成执行检查，不能作为科学结果")
    args = parser.parse_args(argv)
    if args.expected_steps < 1: parser.error("expected-steps 必须为正整数")
    result = report_run(args.root, args.output, expected_steps=args.expected_steps, font=args.font,
                        execution_smoke=args.execution_smoke)
    print(json.dumps({k: result[k] for k in ("status", "workflow_complete", "scientific_result")}, ensure_ascii=False))


if __name__ == "__main__": main()
