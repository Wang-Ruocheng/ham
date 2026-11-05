"""单臂高 k 闭合权重新训练报告；历史对照仅读原有公共 JSON/NPZ。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.report_ising_closure_scale_ablation import (
    SCALES, PATHS, INITIAL_FINGERPRINT_QUALIFICATION, closure_summary,
    encoding_metadata, initialization_metadata, load_arrays, training_start_fingerprints_match, validate_run,
)
from scripts.ising_qcr_evaluation import require_matching_estimators
from scripts.report_ising_future_physics import (
    chinese_plot_context, flow_points, public_sha, read_json, save_chinese,
)


ARMS = {"reference_control": "历史已完成等权对照（未重训）", "late": "本次高 k 加权新训练"}
FIGURES = ("comparison.png", "loss_comparison.png", "R_vs_g.png", "coupling_by_scale.png")


def _normalized_config(config):
    result = copy.deepcopy(config)
    result["loss"].pop("closure_scale_weights", None)
    result["data"].pop("cache", None)
    return result


def validate_reference(manifest, summary, metrics, expected_steps):
    """Validate the original metadata without inventing a new ablation contract."""
    cfg = manifest["config"]
    model = cfg["model"]
    future = manifest.get("q_future_physics", {})
    if (cfg.get("schema_version") != 5 or model.get("dimension") != 2
            or model.get("theory_dim") != 1 or model.get("block_factor") != 3
            or model.get("kernel_initialization") != "random" or model.get("fixed_kernel") is not False
            or manifest.get("labels_used_in_training") is not False
            or future.get("mode") != "paired_control" or future.get("weight") != 0.
            or summary.get("config") != cfg or cfg["training"].get("seed") != 91):
        raise ValueError("历史对照不是指定的新 b3 无监督基础目标")
    if (summary.get("actual_epochs") != expected_steps or summary.get("requested_epochs") != expected_steps
            or cfg["training"].get("epochs") != expected_steps
            or not summary.get("history") or summary["history"][-1].get("epoch") != expected_steps
            or any(summary.get(key, False) for key in ("stopped_by_wall_limit", "stopped_by_training_limit", "stopped_by_train_plateau"))):
        raise ValueError("历史对照必须已完成固定终点")
    if (metrics.get("status") != "complete" or metrics.get("actual_steps") != expected_steps
            or metrics.get("scales") != list(SCALES) or not metrics.get("model_sha256_before")
            or metrics.get("model_sha256_before") != metrics.get("model_sha256_after")):
        raise ValueError("历史对照采集回执不完整")
    return dict(status="complete", actual_steps=expected_steps, requested_steps=expected_steps,
        trained_to_fixed_endpoint=True, execution_role="previously_completed_reference_not_retrained",
        config=cfg, history=summary["history"], collection_metrics=metrics,
        original_manifest=manifest, original_summary=summary,
        data_sha256=manifest.get("data_sha256"), initial_fingerprints_available="initial_module_sha256" in manifest,
        **initialization_metadata(manifest, summary))


def build_report(root, expected_steps=6000):
    root, inputs, arms, panels = Path(root), {}, {}, {}
    def load(path):
        inputs[str(path.relative_to(root))] = public_sha(path)
        return read_json(path)
    optional = {name: load(root / (name + ".json")) if (root / (name + ".json")).is_file() else None
                for name in ("job", "plan", "input_reuse", "preflight_reuse", "dataset_receipt")}
    for arm in ARMS:
        run = root / ("run_late" if arm == "late" else "reference_control")
        names = ("manifest.json", "summary.json", "report/metrics.json", "report/encodings.npz")
        if not all((run / name).is_file() for name in names):
            arms[arm] = dict(status="incomplete", execution_role="new_training" if arm == "late" else "previously_completed_reference_not_retrained")
            continue
        manifest, summary, metrics = [load(run / name) for name in names[:3]]
        if arm == "late":
            info = validate_run(manifest, summary, metrics, "late", expected_steps)
            info.update(status="complete" if info["trained_to_fixed_endpoint"] else "incomplete",
                config=manifest["config"], history=summary["history"], collection_metrics=metrics,
                execution_role="new_training")
        else:
            info = validate_reference(manifest, summary, metrics, expected_steps)
        path = run / names[3]
        arrays = load_arrays(path)
        inputs[str(path.relative_to(root))] = public_sha(path)
        if metrics.get("encodings_sha256") != inputs[str(path.relative_to(root))]:
            raise ValueError("保存数组与原采集回执指纹不符")
        info.update(encoding_metadata(metrics, arrays))
        arms[arm], panels[arm] = info, arrays
    # Never compute coordinates or baseline-normalized errors across estimators.
    estimator = require_matching_estimators(arms[arm]["collection_metrics"] for arm in panels)
    comparison = dict(reference_available="reference_control" in panels,
        execution_design="one_new_late_arm_plus_previously_completed_reference",
        new_training_arms=["late"], fresh_paired_training_performed=False,
        same_fresh_initial_fingerprints_claimed=False, compatible_config_except_weights_and_cache_path=None,
        shared_QC_fingerprints=None, identical_direct_arrays=None, shared_coordinate_evaluation_verified=False,
        encoding_estimator=estimator, encoding_estimators_matched=True,
        same_training_start_module_fingerprints=None,
        initial_fingerprint_comparison_scope=INITIAL_FINGERPRINT_QUALIFICATION)
    if all(arm in panels for arm in ARMS):
        reference, late = arms["reference_control"], arms["late"]
        comparison["same_training_start_module_fingerprints"] = training_start_fingerprints_match(reference, late)
        if reference["data_sha256"] != late["data_sha256"]:
            raise ValueError("历史对照与本次训练的数据指纹不一致")
        comparison["compatible_config_except_weights_and_cache_path"] = (_normalized_config(reference["config"])
            == _normalized_config(late["config"]))
        if not comparison["compatible_config_except_weights_and_cache_path"]:
            raise ValueError("除尺度权重和缓存绝对路径外配置不同")
        for split in ("train", "test"):
            for prefix in ("couplings_", "group_ids_"):
                np.testing.assert_array_equal(panels["reference_control"][prefix + split], panels["late"][prefix + split])
        metrics = [arms[arm]["collection_metrics"] for arm in ARMS]
        comparison["shared_QC_fingerprints"] = all(metrics[0].get(key) is not None
            and metrics[0].get(key) == metrics[1].get(key) for key in ("Q_sha256", "C_sha256"))
        comparison["identical_direct_arrays"] = all(np.array_equal(panels["reference_control"]["direct_" + split],
            panels["late"]["direct_" + split]) for split in ("train", "test"))
        comparison["shared_coordinate_evaluation_verified"] = (comparison["shared_QC_fingerprints"]
            and comparison["identical_direct_arrays"])
    reference_train = panels.get("reference_control", {}).get("direct_train")
    for arm, arrays in panels.items():
        arms[arm]["closure_of_C_means"] = {split: closure_summary(arrays, split, reference_train) for split in ("train", "test")}
        arms[arm]["closure_per_C_draw"] = {split: closure_summary(arrays, split, reference_train, per_draw=True) for split in ("train", "test")}
        arms[arm]["jensen_gap_RMSE"] = {split: float(np.sqrt(np.mean((arrays["R_g_" + split]
            - arrays["R_of_mean_g_" + split]) ** 2))) for split in ("train", "test")}
    complete = arms["late"]["status"] == "complete"
    return dict(schema_version=1, status="complete" if complete else "incomplete",
        workflow_complete=complete, new_training_complete=complete,
        qualified_reference_comparison_available=complete and comparison["reference_available"],
        formal_fresh_paired_comparison_available=False, requested_steps=expected_steps,
        experiment="仅新训练高 k 权重 [0.2,1.8]；历史等权结果不重训",
        profiles=dict(late=[.2, 1.8], historical_reference="original unmodified control metadata"),
        arms=arms, scales=list(SCALES), coordinate_comparison=comparison,
        encoding_estimator=estimator, encoding_estimators_matched=True,
        dataset_receipt=optional["dataset_receipt"], input_reuse=optional["input_reuse"],
        preflight_reuse=optional["preflight_reuse"], input_sha256=inputs,
        thermal_nu=None, thermal_mode_certified=False, root_search_performed=False,
        weighted_objectives_compared_as_physical_improvement=False,
        interpretation=[
            "本次只有高 k 加权臂进行全新6000步训练；历史等权对照是已完成结果，未重训、未重新采集、未改写原始协议。",
            "历史对照缺少新初始指纹协议时不补造；早期日志一致不等于最终 Q/C 相同。",
            INITIAL_FINGERPRINT_QUALIFICATION,
            "只有最终 Q/C 指纹和保存 direct 数组均一致才确认同坐标评价；否则跨臂潜变量误差仅作描述。",
            "先验证编码估计器一致再跨臂比较；旧版全体 Q 数组不转换、不与新版半子集 Q 均值混比。",
            "新版固定半子集划分只匹配训练估计器函数，不等于训练随机划分的期望 loss。",
            "不比较不同加权优化总分以宣称物理改善；三个路径分别报告未加权绝对 RMSE、本臂 TRAIN 标准差 NRMSE及固定历史对照分母 NRMSE。",
            "固定历史分母保留 helper 的 fixed_uniform 字段名；它不是一次新训练等权臂，也不消除不同 Q 坐标的单位差异。",
            "k=2 rollout 是108→12；36→12单步另列，不能用两步误差抵消证明后期单步更好。",
            "逐 C 与先平均 C 的误差分开；历史对照未保存逐 C rollout 时其两步逐 C 误差为空，不新增推理补齐。",
            "C 重复不作为独立物理链；单训练种子、有限尺寸、有限样本及来源数据的自相关/平衡警告保留。",
            "Q/C 不接收闭合梯度；本次不使用 K 标签或未来物理预测目标。R(g) 和真实 R(g)-g 只来自保存点。",
            "不新搜根、不识别热模、不从闭合改善推出临界指数改善；thermal_nu 为空。",
        ]), panels


def _missing(ax):
    ax.text(.5, .5, "尚无完整保存结果", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_report(result, panels, output, font=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    with chinese_plot_context(font) as selected:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        labels = [f"{SCALES[a]}→{SCALES[b]}" for a, b in PATHS]
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            for row, key, name in ((0, "rmse", "绝对 RMSE"), (1, "nrmse", "本臂 TRAIN 标准差 NRMSE"),
                                   (2, "fixed_uniform_nrmse", "固定历史对照 TRAIN 标准差 NRMSE")):
                for split, marker, split_label in (("train", "o", "训练"), ("test", "s", "测试")):
                    values = [r[key] if r[key] is not None else np.nan for r in result["arms"][arm]["closure_of_C_means"][split]]
                    axes[row, col].plot(labels, values, marker+"-", label=split_label)
                axes[row, col].set(title=title, ylabel=name); axes[row, col].legend(); axes[row, col].grid(alpha=.2)
        fig.suptitle("历史等权与本次高 k 加权：未加权路径误差；先平均 C")
        save_chinese(fig, output / "comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        groups = (("separation", "anchor", "coarse_consistency"), ("rsmi", "kernel_noncollapse", "kernel_sign"), ("commutator", "rollout"))
        names = dict(separation="仿射重构", anchor="微观子集一致性", coarse_consistency="粗尺度子集一致性",
            rsmi="负经验互信息", kernel_noncollapse="方差下限", kernel_sign="颜色保持", commutator="单步闭合", rollout="多步闭合")
        for col, (arm, title) in enumerate(ARMS.items()):
            history = result["arms"][arm].get("history", [])
            for row, group in enumerate(groups):
                ax = axes[row, col]
                for color, key in enumerate(group):
                    for split, style, split_label in (("train", "-", "训练"), ("test", "--", "测试")):
                        rows = [value for value in history if split+"_"+key in value]
                        if rows:
                            ax.plot([v["epoch"] for v in rows], [v[split+"_"+key] for v in rows], style, color=f"C{color}", label=names[key]+"·"+split_label)
                ax.set(title=title+"："+("Q 分项", "C 分项", "R 优化分项（权重不同）")[row], xlabel="实际更新次数", ylabel="原始记录分项", yscale="symlog")
                if ax.lines: ax.legend(fontsize=7); ax.grid(alpha=.2)
                else: _missing(ax)
        fig.suptitle("历史日志保持原口径；不能据不同加权闭合分数直接判优")
        save_chinese(fig, output / "loss_comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            arrays = panels[arm]
            g, mapped, levels, flow_label = flow_points(arrays, max_points=240)
            order = np.argsort(g); ax = axes[0, col]
            ax.plot(g[order], mapped[order], lw=.7, label="保存点排序连线；非新网格推理")
            ax.scatter(g, mapped, c=levels, cmap="viridis", s=8, alpha=.5, label="逐 C 的 R(g)" if "direct_draws_test" in arrays else "平均 R(g)")
            ax.scatter(arrays["direct_test"].reshape(-1), arrays["R_of_mean_g_test"].reshape(-1), marker="x", s=8, alpha=.4, label="R(平均 g)")
            lower, upper = min(float(g.min()), float(mapped.min())), max(float(g.max()), float(mapped.max()))
            ax.plot([lower, upper], [lower, upper], "k--", lw=.8, label="恒等映射")
            ax.set(title=title, xlabel="输入 g", ylabel="输出 R(g) 或平均 R(g)"); ax.legend(fontsize=7); ax.grid(alpha=.2)
            ax = axes[1, col]
            ax.quiver(g, levels, mapped-g, np.zeros_like(g), levels, cmap="viridis", angles="xy", scale_units="xy", scale=1, width=.003)
            ax.set(title=flow_label, xlabel="编码 g；箭头不归一化", ylabel="输入晶格边长 L", yticks=[0, 1, 2], yticklabels=[str(v) for v in SCALES], ylim=(-.5, 2.5))
            ax.update_datalim(np.column_stack((mapped, levels))); ax.autoscale_view(scaley=False); ax.grid(alpha=.2)
        fig.suptitle("保存的 R(g) 和真实位移 R(g)-g；不搜根，不估计指数")
        save_chinese(fig, output / "R_vs_g.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        maximum = max((float(np.linalg.norm(a["couplings_"+s][:, 1:], axis=1).max()) for a in panels.values() for s in ("train", "test")), default=0.)
        norm, scatter = Normalize(0., max(maximum, 1e-12)), None
        for col, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            for row, scale in enumerate(SCALES):
                ax = axes[row, col]
                for split, marker, name in (("train", "o", "训练"), ("test", "^", "测试")):
                    coupling = panels[arm]["couplings_"+split]
                    scatter = ax.scatter(coupling[:, 0], panels[arm]["direct_"+split][:, row, 0], c=np.linalg.norm(coupling[:, 1:], axis=1), norm=norm, cmap="viridis", marker=marker, s=12, alpha=.6, label=name)
                ax.set(title=title+f"：L={scale}", xlabel="生成时基准 K0（仅评价）", ylabel="编码 g"); ax.legend(fontsize=7); ax.grid(alpha=.2)
        if scatter is not None: fig.colorbar(scatter, ax=axes.ravel().tolist(), label="六个有限扰动物理耦合的模长")
        fig.suptitle("各尺度耦合到 g；有限扰动条件不合并成唯一 g(K) 曲线")
        save_chinese(fig, output / "coupling_by_scale.png", selected); plt.close(fig)


def report_run(root, output=None, *, expected_steps=6000, font=None, make_plots=True):
    root, output = Path(root), Path(output) if output else Path(root) / "report"
    if output.exists() and any(output.iterdir()): raise FileExistsError("不覆盖既有报告")
    result, panels = build_report(root, expected_steps)
    output.mkdir(parents=True, exist_ok=True)
    if make_plots: plot_report(result, panels, output, font)
    np.savez_compressed(output / "evaluation_arrays.npz", **{arm+"__"+key: value for arm, arrays in panels.items() for key, value in arrays.items()})
    result["public_files"] = ([*FIGURES] if make_plots else []) + ["report.json", "evaluation_arrays.npz", "status.json"]
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    status = dict(status=result["status"], workflow_complete=result["workflow_complete"], new_training_complete=result["new_training_complete"],
        qualified_reference_comparison_available=result["qualified_reference_comparison_available"], formal_fresh_paired_comparison_available=False,
        thermal_nu=None, public_files=result["public_files"], report_sha256=public_sha(output / "report.json"), evaluation_arrays_sha256=public_sha(output / "evaluation_arrays.npz"))
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-steps", type=int, default=6000); parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    if args.expected_steps < 1: parser.error("expected-steps 必须为正整数")
    result = report_run(args.root, args.output, expected_steps=args.expected_steps, font=args.font)
    print(json.dumps(dict(status=result["status"], workflow_complete=result["workflow_complete"], thermal_nu=None), ensure_ascii=False))


if __name__ == "__main__": main()
