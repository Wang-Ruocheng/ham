"""仅从公开 JSON/NPZ 汇总闭合尺度权重消融；不加载模型或物理缓存。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.ising_qcr_evaluation import (
    FULL_ENSEMBLE_ESTIMATOR, HALF_SUBSET_ESTIMATOR,
    require_matching_estimators, validate_encoding_arrays,
)
from scripts.report_ising_future_physics import (
    chinese_plot_context, flow_points, public_sha, read_json, save_chinese,
)


ARMS = {"uniform": "等权闭合对照", "late": "高 k 加权闭合"}
PROFILES = {"uniform": [1., 1.], "late": [.2, 1.8]}
SCALES = (108, 36, 12)
PATHS = ((0, 1), (1, 2), (0, 2))
FIGURES = ("comparison.png", "loss_comparison.png", "R_vs_g.png", "coupling_by_scale.png")
INITIAL_FINGERPRINT_QUALIFICATION = "旧 initial 指纹记录时间可能在校准前，不证明首步参数相同；不替代实际 training_start 指纹。"


def initialization_metadata(manifest, summary):
    """Keep constructor and actual first-update provenance distinct."""
    actual = manifest.get("training_start_module_sha256")
    available = "training_start_module_sha256" in manifest
    if not available and "training_start_module_sha256" in summary.get("contract", {}):
        raise ValueError("实际训练起点指纹只出现在 summary.contract，manifest 缺失")
    if available:
        if (not isinstance(actual, dict) or set(actual) != {"Q", "C", "R"}
                or any(not isinstance(value, str) or not value for value in actual.values())
                or summary.get("contract", {}).get("training_start_module_sha256") != actual):
            raise ValueError("实际训练起点 Q/C/R 指纹与 summary.contract 缺失或不一致")
    provenance = manifest.get("module_initialization_provenance")
    return dict(actual_training_start_fingerprints_available=available,
        training_start_module_sha256=copy.deepcopy(actual) if available else None,
        training_start_module_sha256_scope=(provenance.get("training_start_scope")
            if isinstance(provenance, dict) else None),
        initial_module_sha256_scope=manifest.get("initial_module_sha256_scope"),
        random_construction_module_sha256=copy.deepcopy(manifest.get("random_construction_module_sha256")),
        pre_affine_initialization_module_sha256=copy.deepcopy(manifest.get("pre_affine_initialization_module_sha256")),
        module_initialization_provenance=copy.deepcopy(provenance),
        initial_module_sha256_qualification=INITIAL_FINGERPRINT_QUALIFICATION)


def training_start_fingerprints_match(left, right):
    """Missing historical first-update hashes remain unknown, never aliases."""
    if not all(info.get("actual_training_start_fingerprints_available", False) for info in (left, right)):
        return None
    same = left["training_start_module_sha256"] == right["training_start_module_sha256"]
    if not same:
        raise ValueError("实际训练起点 Q/C/R 指纹不同，不能作为仅改尺度权重的配对比较")
    return same


def encoding_metadata(metrics, arrays):
    """Expose the saved estimator contract without converting historical arrays."""
    estimator = validate_encoding_arrays(metrics, arrays)
    legacy = "encoding_estimator" not in metrics
    result = dict(encoding_estimator=estimator,
        encoding_protocol=copy.deepcopy(metrics.get("encoding_protocol")),
        legacy_unmarked_full_ensemble_Q=legacy,
        encoding_definition=("历史全体 Q 口径：Q(全部构型)，未改写或转换历史数组。"
            if estimator == FULL_ENSEMBLE_ESTIMATOR else
            "训练函数口径：两个不相交等长半子集的 Q 值取均值。"),
        encoding_qualification=("一次固定划分与训练随机划分的期望不等价；估计器函数一致不代表评价 loss 等于训练期望。"
            if estimator == HALF_SUBSET_ESTIMATOR else
            "全体 Q 与半子集 Q 均值在非线性 Q 下不等价；不得与新版半子集数组混比。"))
    if estimator == HALF_SUBSET_ESTIMATOR:
        result["Q_full_minus_half_RMSE"] = {split: float(np.sqrt(np.mean(
            arrays["Q_full_minus_half_draws_" + split] ** 2))) for split in ("train", "test")}
    return result


def load_arrays(path):
    """Read numeric saved evaluations only; never manufacture a missing R output."""
    if Path(path).suffix != ".npz":
        raise ValueError("报告只接受公开 NPZ 数组")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ValueError(f"公开数组必须是有限数值：{key}")
    for split in ("train", "test"):
        direct = arrays.get("direct_" + split)
        if direct is None or direct.ndim != 3 or direct.shape[1:] != (3, 1) or len(direct) < 2:
            raise ValueError("需要三个尺度的标量 direct 数组")
        count = len(direct)
        for name in ("R_g", "R_of_mean_g", "rollout"):
            if arrays.get(name + "_" + split, np.empty(0)).shape != direct.shape:
                raise ValueError("缺少已保存 R(g) 或 rollout；不能用下一尺度 Q 冒充")
        if arrays.get("couplings_" + split, np.empty(0)).shape != (count, 7):
            raise ValueError("评价耦合必须为 K0 与六个有限扰动")
        if arrays.get("group_ids_" + split, np.empty(0)).shape != (count,):
            raise ValueError("缺少物理条件分组")
        np.testing.assert_allclose(arrays["rollout_" + split][:, 0], direct[:, 0], rtol=1e-10, atol=1e-12)
        draws = [name + "_draws_" + split for name in ("direct", "R_g")]
        if any(key in arrays for key in draws):
            if not all(key in arrays for key in draws):
                raise ValueError("逐 C 的 g 和 R(g) 必须成对保存")
            shape = arrays[draws[0]].shape
            if len(shape) != 4 or shape[0] != count or shape[2:] != (3, 1) or shape[1] < 1:
                raise ValueError("逐 C 数组必须为 [物理链,C重复,尺度,坐标]")
            if arrays[draws[1]].shape != shape:
                raise ValueError("逐 C 的 g 与 R(g) 形状不一致")
            for key, mean in zip(draws, (direct, arrays["R_g_" + split])):
                np.testing.assert_allclose(arrays[key].mean(1), mean, rtol=1e-9, atol=1e-11)
        rollout_key = "rollout_draws_" + split
        if rollout_key in arrays:
            if draws[0] not in arrays or arrays[rollout_key].shape != arrays[draws[0]].shape:
                raise ValueError("逐 C 的 rollout 需要对应逐 C 编码")
            np.testing.assert_allclose(arrays[rollout_key].mean(1), arrays["rollout_" + split], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(arrays[rollout_key][:, :, 0], arrays[draws[0]][:, :, 0], rtol=1e-10, atol=1e-12)
    return arrays


def _std(values):
    # C repeats are averaged before estimating physical-chain target variation.
    return float(np.sqrt(np.mean(np.var(values, axis=0, ddof=0))))


def closure_summary(arrays, split, reference_train=None, *, per_draw=False):
    """Unweighted path errors, both own TRAIN and one fixed baseline normalization."""
    direct = arrays["direct_" + split]
    reference = reference_train
    result = []
    for source, target in PATHS:
        own_std = _std(arrays["direct_train"][:, target])
        baseline_std = _std(reference[:, target]) if reference is not None else None
        if per_draw:
            direct_key = "direct_draws_" + split
            prediction_key = ("rollout_draws_" if source == 0 and target == 2 else "R_g_draws_") + split
            if direct_key not in arrays or prediction_key not in arrays:
                mse = rmse = None
            else:
                level = target if (source, target) == (0, 2) else source
                mse = float(np.mean((arrays[prediction_key][:, :, level] - arrays[direct_key][:, :, target]) ** 2))
                rmse = float(np.sqrt(mse))
        else:
            predicted = (arrays["rollout_" + split][:, target] if (source, target) == (0, 2)
                         else arrays["R_g_" + split][:, source])
            mse = float(np.mean((predicted - direct[:, target]) ** 2))
            rmse = float(np.sqrt(mse))
        result.append(dict(source_level=source, target_level=target, source=SCALES[source], target=SCALES[target],
            prediction="R^2(g0)" if (source, target) == (0, 2) else f"R(g{source})",
            mse=mse, rmse=rmse, train_target_std=own_std,
            nrmse=rmse / own_std if rmse is not None and own_std > 1e-12 else None,
            fixed_uniform_train_target_std=baseline_std,
            fixed_uniform_nrmse=rmse / baseline_std if rmse is not None and baseline_std is not None and baseline_std > 1e-12 else None,
            available=rmse is not None))
    return result


def validate_run(manifest, summary, metrics, arm, expected_steps):
    cfg = manifest["config"]
    model, training, loss = cfg["model"], cfg["training"], cfg["loss"]
    if (cfg.get("schema_version") != 5 or model.get("dimension") != 2 or model.get("theory_dim") != 1
            or model.get("block_factor") != 3 or model.get("kernel_layout") != "nonoverlapping"
            or model.get("kernel_initialization") != "random" or model.get("kernel_sampling") != "bernoulli"
            or model.get("fixed_kernel") is not False or model.get("affine_reconstruction") is not True
            or training.get("max_rg_steps") != 2 or training.get("gradient_mode") != "qc_r_separate"
            or training.get("seed") != 91):
        raise ValueError("只报告新 schema5、随机可训练 C、独立梯度、seed91 的标量 b3 实验")
    if manifest.get("labels_used_in_training") is not False or summary.get("config") != cfg:
        raise ValueError("标签训练声明或 summary 配置不匹配")
    initial = manifest.get("initial_module_sha256", {})
    if (set(initial) != {"Q", "C", "R"}
            or summary.get("contract", {}).get("initial_module_sha256") != initial):
        raise ValueError("缺少一致的 Q/C/R 初始指纹")
    if list(loss.get("closure_scale_weights", [])) != PROFILES[arm]:
        raise ValueError("闭合尺度权重不匹配")
    contract = manifest.get("closure_scale_weighting", {})
    if (contract.get("weights") != PROFILES[arm] or contract.get("target_levels") != [1, 2]
            or contract.get("application") != ["one_step_commutator", "microscopic_start_rollout"]
            or contract.get("labels_used") is not False
            or summary.get("contract", {}).get("closure_scale_weighting") != contract):
        raise ValueError("单步与微观起点 rollout 的尺度权重协议不匹配")
    np.testing.assert_allclose(contract.get("normalized_weights", []), np.asarray(PROFILES[arm]) / sum(PROFILES[arm]))
    future = manifest.get("q_future_physics", {})
    if future.get("weight", 0.) != 0. or manifest.get("auxiliary_future_parameters", 0) != 0:
        raise ValueError("尺度权重消融不得混入新增未来物理目标")
    actual, history = summary.get("actual_epochs"), summary.get("history", [])
    epochs = [row.get("epoch") for row in history]
    if (type(actual) is not int or not 0 < actual <= expected_steps or not history
            or any(type(value) is not int for value in epochs)
            or epochs != sorted(set(epochs)) or epochs[-1] != actual
            or summary.get("requested_epochs") != expected_steps or training.get("epochs") != expected_steps):
        raise ValueError("固定终点、实际步数与训练历史不一致")
    for row in history:
        if any(isinstance(value, (int, float)) and not np.isfinite(value) for value in row.values()):
            raise ValueError("训练历史存在非有限值")
    if metrics is not None:
        if (metrics.get("status") != "complete" or metrics.get("actual_steps") != actual
                or metrics.get("scales") != list(SCALES)):
            raise ValueError("采集检查点步数或尺度不匹配")
        if (not metrics.get("model_sha256_before")
                or metrics.get("model_sha256_before") != metrics.get("model_sha256_after")):
            raise ValueError("评价前后模型指纹不一致")
    complete = actual == expected_steps and not any(summary.get(key, False) for key in
        ("stopped_by_wall_limit", "stopped_by_training_limit", "stopped_by_train_plateau"))
    return dict(actual_steps=actual, requested_steps=expected_steps, trained_to_fixed_endpoint=complete,
        scale_contract=contract, data_sha256=manifest.get("data_sha256"),
        initial_module_sha256=manifest.get("initial_module_sha256"), final_logged_metrics=history[-1],
        **initialization_metadata(manifest, summary))


def build_report(root, expected_steps=6000):
    root = Path(root)
    inputs, panels, arms = {}, {}, {}
    def load(path):
        inputs[str(path.relative_to(root))] = public_sha(path)
        return read_json(path)
    optional = {key: load(root / (key + ".json")) if (root / (key + ".json")).is_file() else None
                for key in ("job", "plan", "dataset_receipt", "input_reuse")}
    for arm in ARMS:
        run = root / ("run_" + arm)
        if not all((run / name).is_file() for name in ("manifest.json", "summary.json")):
            arms[arm] = dict(status="incomplete", trained_to_fixed_endpoint=False, reason="尚无完整 manifest/summary")
            continue
        manifest, summary = load(run / "manifest.json"), load(run / "summary.json")
        metrics_path, arrays_path = run / "report/metrics.json", run / "report/encodings.npz"
        metrics = load(metrics_path) if metrics_path.is_file() else None
        info = validate_run(manifest, summary, metrics, arm, expected_steps)
        complete = info["trained_to_fixed_endpoint"] and metrics is not None and arrays_path.is_file()
        info.update(status="complete" if complete else "incomplete", collection_metrics=metrics,
                    config=manifest["config"], history=summary["history"])
        if arrays_path.is_file():
            arrays = load_arrays(arrays_path)
            inputs[str(arrays_path.relative_to(root))] = public_sha(arrays_path)
            if metrics is not None and metrics.get("encodings_sha256") != inputs[str(arrays_path.relative_to(root))]:
                raise ValueError("采集数组指纹与回执不一致")
            if metrics is not None:
                info.update(encoding_metadata(metrics, arrays))
                panels[arm] = arrays
        arms[arm] = info
    # This gate precedes every cross-arm coordinate or fixed-baseline metric.
    estimator = require_matching_estimators(arms[arm]["collection_metrics"] for arm in panels)
    comparison = dict(shared_QC_fingerprints=False, identical_direct_arrays=False,
                      shared_coordinate_evaluation_verified=False,
                      encoding_estimator=estimator, encoding_estimators_matched=True,
                      same_training_start_module_fingerprints=None,
                      initial_fingerprint_comparison_scope=INITIAL_FINGERPRINT_QUALIFICATION)
    if all(arm in panels for arm in ARMS):
        comparison["same_training_start_module_fingerprints"] = training_start_fingerprints_match(
            arms["uniform"], arms["late"])
        for split in ("train", "test"):
            for prefix in ("couplings_", "group_ids_"):
                np.testing.assert_array_equal(panels["uniform"][prefix + split], panels["late"][prefix + split])
        if arms["uniform"]["data_sha256"] != arms["late"]["data_sha256"]:
            raise ValueError("两臂物理数据指纹必须一致")
        cfgs = [copy.deepcopy(arms[arm]["config"]) for arm in ARMS]
        for cfg in cfgs:
            cfg["loss"].pop("closure_scale_weights", None)
        if cfgs[0] != cfgs[1]:
            raise ValueError("除尺度权重外两臂配置必须相同")
        metrics = [arms[arm]["collection_metrics"] or {} for arm in ARMS]
        comparison["shared_QC_fingerprints"] = all(metrics[0].get(key) is not None and
            metrics[0].get(key) == metrics[1].get(key) for key in ("Q_sha256", "C_sha256"))
        comparison["identical_direct_arrays"] = all(np.array_equal(panels["uniform"]["direct_" + split],
            panels["late"]["direct_" + split]) for split in ("train", "test"))
        comparison["shared_coordinate_evaluation_verified"] = (comparison["shared_QC_fingerprints"]
            and comparison["identical_direct_arrays"])
        comparison["same_initial_module_fingerprints"] = (arms["uniform"]["initial_module_sha256"] is not None
            and arms["uniform"]["initial_module_sha256"] == arms["late"]["initial_module_sha256"])
        if not comparison["same_initial_module_fingerprints"]:
            raise ValueError("两臂必须使用相同的全新 Q/C/R 初始指纹")
    baseline = panels.get("uniform", {}).get("direct_train")
    for arm, arrays in panels.items():
        info = arms[arm]
        info["closure_of_C_means"] = {split: closure_summary(arrays, split, baseline) for split in ("train", "test")}
        info["closure_per_C_draw"] = {split: closure_summary(arrays, split, baseline, per_draw=True) for split in ("train", "test")}
        info["jensen_gap_RMSE"] = {split: float(np.sqrt(np.mean((arrays["R_g_" + split] -
            arrays["R_of_mean_g_" + split]) ** 2))) for split in ("train", "test")}
    complete = all(info["status"] == "complete" for info in arms.values())
    result = dict(schema_version=1, status="complete" if complete else "incomplete",
        formal_comparison_available=complete, experiment="无监督 b3 闭合尺度权重配对消融",
        requested_steps_per_arm=expected_steps, profiles=PROFILES, scales=list(SCALES), arms=arms,
        coordinate_comparison=comparison, dataset_receipt=optional["dataset_receipt"], input_reuse=optional["input_reuse"],
        encoding_estimator=estimator, encoding_estimators_matched=True,
        root_job_status=(optional["job"] or {}).get("status"), input_sha256=inputs,
        weighted_objectives_compared_as_physical_improvement=False, thermal_mode_certified=False,
        thermal_nu=None, root_search_performed=False,
        interpretation=[
            "只改变 R 的单步闭合及微观起点 rollout 的尺度权重；Q/C 不接收闭合梯度，不含 K 监督或未来物理预测目标。",
            "各尺度权重除以其总和；外层单步/rollout 权重保持不变。k=2 rollout 是108→12，不是36→12单步。",
            "优化后的加权总分不直接比较；判读使用三个路径未加权的绝对 RMSE 与 TRAIN 目标标准差 NRMSE。",
            "逐 C 误差与先平均 C 的误差分别保存；C 重复不是独立物理链，未据其估计置信区间。",
            "同时保存各臂 TRAIN 与固定等权对照 TRAIN 分母；固定分母不代表任意不同 Q 坐标下具有可比物理单位。",
            "跨臂比较先验证编码估计器一致；缺少旧版元数据的数组明确保留全体 Q 口径，不与新版半子集均值混比。",
            "新版半子集评价使用一次固定划分；只匹配训练的估计器函数，不等于训练随机划分的期望 loss。",
            "仅当最终 Q/C 指纹及已采集 direct 数组均相同才确认同坐标比较；否则跨臂原始潜变量误差只作描述。",
            INITIAL_FINGERPRINT_QUALIFICATION,
            "R(g)、R(平均 g)、平均 R(g) 分开保存；箭头为实际保存点的 R(g)-g，不以 Q(CD) 代替 R(g)。",
            "耦合标签仅用于评价；有限扰动条件不连成唯一 g(K) 曲线。",
            "不新搜不动点，不选择热模，不从闭合误差下降推断临界指数改善；thermal_nu 始终为空。",
            "单训练种子、有限样本、有限尺寸和来源数据的自相关/平衡警告保留；此消融不认证低维自治闭合。",
        ])
    return result, panels


def _missing(ax):
    ax.text(.5, .5, "尚无完整的已保存结果", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_report(result, panels, output, font=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    with chinese_plot_context(font) as selected:
        fig, axes = plt.subplots(4, 2, figsize=(14, 14), layout="constrained")
        for column, (arm, label) in enumerate(ARMS.items()):
            info, history = result["arms"][arm], result["arms"][arm].get("history", [])
            if not history:
                for ax in axes[:, column]: _missing(ax)
                continue
            groups = (("separation", "anchor", "coarse_consistency"), ("rsmi", "kernel_noncollapse", "kernel_sign"),
                      ("commutator", "rollout"), ("one_step_level_1", "one_step_level_2", "rollout_level_1", "rollout_level_2"))
            names = {"separation": "仿射重构", "anchor": "微观子集一致性", "coarse_consistency": "粗尺度子集一致性",
                     "rsmi": "负经验互信息", "kernel_noncollapse": "方差下限", "kernel_sign": "颜色保持",
                     "commutator": "加权单步闭合", "rollout": "加权多步闭合", "one_step_level_1": "单步 k=1",
                     "one_step_level_2": "单步 k=2", "rollout_level_1": "rollout k=1", "rollout_level_2": "rollout k=2"}
            for row, group in enumerate(groups):
                ax = axes[row, column]
                for color, key in enumerate(group):
                    for split, style, split_label in (("train", "-", "训练"), ("test", "--", "测试")):
                        rows = [value for value in history if split + "_" + key in value]
                        if rows:
                            ax.plot([value["epoch"] for value in rows], [value[split + "_" + key] for value in rows],
                                    style, color=f"C{color}", label=names[key] + "·" + split_label)
                ax.set(title=label + "：" + ("Q 分项", "C 分项", "R 加权分项", "R 未加权尺度分项")[row],
                       xlabel="实际更新次数", ylabel="记录的分项值", yscale="symlog")
                ax.grid(alpha=.2)
                if ax.lines: ax.legend(fontsize=7)
                else: _missing(ax)
        fig.suptitle("尺度权重 [1,1] 对比 [0.2,1.8]；不同优化总分不能直接判定物理改善")
        save_chinese(fig, output / "loss_comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        labels = [f"{SCALES[a]}→{SCALES[b]}" for a, b in PATHS]
        for column, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, column]: _missing(ax)
                continue
            info = result["arms"][arm]
            for row, key, name in ((0, "rmse", "绝对 RMSE"), (1, "nrmse", "本臂 TRAIN 标准差 NRMSE"),
                                    (2, "fixed_uniform_nrmse", "固定等权对照 TRAIN 标准差 NRMSE")):
                for split, marker, split_label in (("train", "o", "训练"), ("test", "s", "测试")):
                    values = [value[key] if value[key] is not None else np.nan for value in info["closure_of_C_means"][split]]
                    axes[row, column].plot(labels, values, marker + "-", label=split_label)
                axes[row, column].set(title=title + "：未加权路径误差", ylabel=name)
                axes[row, column].legend(); axes[row, column].grid(alpha=.2)
        fig.suptitle("先平均 C 后的闭合；逐 C 未加权误差另见报告，不以加权分数代替")
        save_chinese(fig, output / "comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
        for column, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, column]: _missing(ax)
                continue
            arrays = panels[arm]
            g, mapped, levels, flow_label = flow_points(arrays, max_points=240)
            order = np.argsort(g)
            ax = axes[0, column]
            ax.plot(g[order], mapped[order], lw=.7, alpha=.5, label="保存点排序连线；非新网格推理")
            point_label = "逐 C 的真实 R(g)" if "direct_draws_test" in arrays else "平均 R(g)，非 R(平均 g)"
            ax.scatter(g, mapped, c=levels, cmap="viridis", s=8, alpha=.6, label=point_label)
            mean_g = arrays["direct_test"].reshape(-1)
            ax.scatter(mean_g, arrays["R_of_mean_g_test"].reshape(-1), marker="x", s=9, alpha=.4, label="R(平均 g)")
            lower, upper = min(float(g.min()), float(mapped.min())), max(float(g.max()), float(mapped.max()))
            ax.plot([lower, upper], [lower, upper], "k--", lw=.8, label="恒等映射")
            ax.set(title=title + "：保存的映射点", xlabel="输入 g", ylabel="输出 R(g)"); ax.legend(fontsize=7); ax.grid(alpha=.2)
            ax = axes[1, column]
            ax.quiver(g, levels, mapped-g, np.zeros_like(g), levels, cmap="viridis", angles="xy", scale_units="xy", scale=1, width=.003)
            ax.set(title=flow_label, xlabel="编码 g；箭头为 R(g)-g，不归一化", ylabel="输入晶格边长 L",
                   yticks=[0, 1, 2], yticklabels=[str(value) for value in SCALES], ylim=(-.5, 2.5))
            ax.update_datalim(np.column_stack((mapped, levels))); ax.autoscale_view(scaley=False); ax.grid(alpha=.2)
        fig.suptitle("真实 R(g) 与位移；不搜根，不据曲线猜测热指数")
        save_chinese(fig, output / "R_vs_g.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        maximum = max((float(np.linalg.norm(arrays["couplings_" + split][:, 1:], axis=1).max())
                       for arrays in panels.values() for split in ("train", "test")), default=0.)
        norm, scatter = Normalize(0., max(maximum, 1e-12)), None
        for column, (arm, title) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, column]: _missing(ax)
                continue
            for row, scale in enumerate(SCALES):
                ax = axes[row, column]
                for split, marker, name in (("train", "o", "训练"), ("test", "^", "测试")):
                    coupling = panels[arm]["couplings_" + split]
                    scatter = ax.scatter(coupling[:, 0], panels[arm]["direct_" + split][:, row, 0],
                        c=np.linalg.norm(coupling[:, 1:], axis=1), norm=norm, cmap="viridis", marker=marker, s=12, alpha=.6, label=name)
                ax.set(title=title + f"：L={scale}", xlabel="生成时基准 K0（仅评价）", ylabel="编码 g")
                ax.legend(fontsize=8); ax.grid(alpha=.2)
        if scatter is not None:
            fig.colorbar(scatter, ax=axes.ravel().tolist(), label="六个有限扰动物理耦合的模长")
        fig.suptitle("耦合到三个尺度的编码；有限扰动条件不合并为唯一 g(K) 曲线")
        save_chinese(fig, output / "coupling_by_scale.png", selected); plt.close(fig)


def report_run(root, output=None, *, expected_steps=6000, font=None, make_plots=True):
    root, output = Path(root), Path(output) if output else Path(root) / "report"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("只写新空报告目录，不覆盖历史报告")
    result, panels = build_report(root, expected_steps)
    output.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_report(result, panels, output, font)
    np.savez_compressed(output / "evaluation_arrays.npz", **{arm + "__" + key: value
                        for arm, arrays in panels.items() for key, value in arrays.items()})
    result["public_files"] = ([*FIGURES] if make_plots else []) + ["report.json", "evaluation_arrays.npz", "status.json"]
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    status = dict(status=result["status"], workflow_complete=result["formal_comparison_available"],
        formal_comparison_available=result["formal_comparison_available"], thermal_nu=None,
        public_files=result["public_files"], report_sha256=public_sha(output / "report.json"),
        evaluation_arrays_sha256=public_sha(output / "evaluation_arrays.npz"))
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-steps", type=int, default=6000)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    if args.expected_steps < 1:
        parser.error("expected-steps 必须为正整数")
    result = report_run(args.root, args.output, expected_steps=args.expected_steps, font=args.font)
    print(json.dumps(dict(status=result["status"], formal_comparison_available=result["formal_comparison_available"], thermal_nu=None), ensure_ascii=False))


if __name__ == "__main__":
    main()
