"""仅从新 b3 实验的公开 JSON/NPZ 生成中文报告，不加载模型或物理缓存。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from scripts.report_ising_zh import select_chinese_font, _font_codepoints
from scripts.ising_qcr_evaluation import (
    FULL_ENSEMBLE_ESTIMATOR, collection_estimator, require_matching_estimators,
    validate_encoding_arrays,
)


ARMS = {"control": "基础目标对照", "future": "新增未来物理目标"}
SCALES = (108, 36, 12)
PATHS = ((0, 1), (1, 2), (0, 2))
OBSERVABLES = ("轴向距离L/6关联", "轴向距离L/3关联", "最低非零轴向结构因子",
               "第二非零轴向结构因子", "最低对角结构因子")
FIGURES = ("comparison.png", "loss_comparison.png", "R_vs_g.png", "coupling_by_scale.png")


def read_json(path):
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("报告只读取 JSON/NPZ")
    return json.loads(path.read_text())


def public_sha(path):
    path = Path(path)
    if path.suffix not in (".json", ".npz"):
        raise ValueError("不能读取或重新散列模型与物理缓存")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arrays(path, metrics=None):
    if Path(path).suffix != ".npz":
        raise ValueError("报告只接受公开 NPZ 数组")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for split in ("train", "test"):
        direct = arrays.get("direct_" + split)
        if direct is None or direct.ndim != 3 or direct.shape[1:] != (3, 1) or len(direct) < 2:
            raise ValueError("需要保存的标量 b3 三尺度 direct 数组；不能临时模型推理")
        count = len(direct)
        for name in ("R_g", "R_of_mean_g", "rollout"):
            if name + "_" + split not in arrays or arrays[name + "_" + split].shape != direct.shape:
                raise ValueError("缺少真实保存的 R 数组；不能用 Q(CD) 冒充 R(g)")
        if arrays.get("observables_" + split, np.empty(0)).shape != (count, 3, 5):
            raise ValueError("未来物理量必须为五个物理统计量、三个尺度")
        if arrays.get("couplings_" + split, np.empty(0)).shape != (count, 7):
            raise ValueError("评价耦合需为 K0 与六个有限扰动；不能当成单一 K 曲线")
        if arrays.get("group_ids_" + split, np.empty(0)).shape != (count,):
            raise ValueError("缺少物理条件分组信息")
        np.testing.assert_allclose(arrays["rollout_" + split][:, 0], direct[:, 0], rtol=1e-10, atol=1e-12)
        for name in ("direct_draws", "R_g_draws"):
            if name + "_" + split in arrays:
                value = arrays[name + "_" + split]
                if value.ndim != 4 or value.shape[0] != count or value.shape[2:] != (3, 1):
                    raise ValueError("逐 C 数组需为 [物理链,C重复,尺度,坐标]")
        draw_keys = [name + "_" + split for name in ("direct_draws", "R_g_draws")]
        if any(key in arrays for key in draw_keys):
            if not all(key in arrays for key in draw_keys) or arrays[draw_keys[0]].shape != arrays[draw_keys[1]].shape:
                raise ValueError("逐 C 的 g 与 R(g) 必须成对保存")
            for key, mean in zip(draw_keys, (direct, arrays["R_g_" + split])):
                np.testing.assert_allclose(arrays[key].mean(1), mean, rtol=1e-9, atol=1e-11)
    for key, value in arrays.items():
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ValueError(f"公开数组不是有限数值：{key}")
    if metrics is not None:
        validate_encoding_arrays(metrics, arrays)
    return arrays


def closure_summary(arrays, split):
    """平均 C 后的描述统计，不冒充逐 C 闭合损失或置信区间。"""
    direct = arrays["direct_" + split]
    prediction = arrays["R_g_" + split]
    reference = arrays["direct_train"]
    result = []
    for source, target in PATHS:
        predicted = (arrays["rollout_" + split][:, 2] if (source, target) == (0, 2)
                     else prediction[:, source])
        rmse = float(np.sqrt(np.mean((predicted - direct[:, target]) ** 2)))
        std = float(np.sqrt(np.mean(np.var(reference[:, target], axis=0))))
        result.append(dict(source=SCALES[source], target=SCALES[target], rmse=rmse,
                           train_target_std=std, nrmse=rmse / std if std > 1e-12 else None))
    return result


def validate_run(manifest, summary, metrics, arm, expected_steps):
    cfg = manifest["config"]
    model = cfg["model"]
    training = cfg["training"]
    if (cfg.get("schema_version") != 5 or model.get("block_factor") != 3
            or model.get("dimension") != 2 or model.get("theory_dim") != 1
            or model.get("kernel_initialization") != "random"
            or model.get("kernel_sampling") != "bernoulli"
            or model.get("kernel_layout") != "nonoverlapping"
            or model.get("fixed_kernel") is not False or model.get("affine_reconstruction") is not True
            or training.get("max_rg_steps") != 2):
        raise ValueError("本报告仅用于新 schema5、随机可训练 C、标量 b3 仿射 Q 实验")
    if manifest.get("labels_used_in_training") is not False:
        raise ValueError("必须显式证明耦合标签不参与训练")
    if summary.get("config") != cfg:
        raise ValueError("manifest 与 summary 配置不一致")
    future = manifest.get("q_future_physics", {})
    if (future.get("mode") != ("paired_control" if arm == "control" else "future_physics")
            or future.get("labels_used") is not False
            or future.get("paths") != [list(p) for p in PATHS]
            or tuple(future.get("observables", ())) != OBSERVABLES
            or future.get("data_sha256") != manifest.get("data_sha256")):
        raise ValueError("未来物理目标的公开协议不匹配")
    if (arm == "control" and future.get("weight") != 0) or (arm == "future" and not future.get("weight", 0) > 0):
        raise ValueError("对照/新增目标的权重不匹配")
    if "contract" in summary and summary["contract"].get("q_future_physics") != future:
        raise ValueError("manifest 与 summary 的新增目标协议不一致")
    actual = summary.get("actual_epochs")
    history = summary.get("history", [])
    epochs = [row.get("epoch") for row in history]
    if (type(actual) is not int or not history or any(type(i) is not int for i in epochs)
            or epochs != sorted(set(epochs)) or epochs[-1] != actual
            or not 0 < actual <= expected_steps):
        raise ValueError("实际训练步数与历史记录不一致")
    if (summary.get("requested_epochs") != expected_steps
            or training.get("epochs") != expected_steps):
        raise ValueError("固定请求步数不匹配，不能把短跑称正式终点")
    if metrics is not None and metrics.get("actual_steps") != actual:
        raise ValueError("采集检查点与实际训练步数不一致")
    if metrics is not None and metrics.get("scales", list(SCALES)) != list(SCALES):
        raise ValueError("采集尺度不匹配")
    for row in history:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"历史记录有非有限值：{key}")
    if arm == "future":
        for split in ("train", "test"):
            required = ("future_physics", "future_physics_nrmse", "future_physics_raw_mse", "future_physics_raw_rmse")
            if any(split + "_" + key not in history[-1] for key in required):
                raise ValueError("新增目标必须同时记录原始与固定归一化误差")
            for mse, rmse in (("future_physics", "future_physics_nrmse"), ("future_physics_raw_mse", "future_physics_raw_rmse")):
                np.testing.assert_allclose(history[-1][split + "_" + rmse] ** 2,
                                           history[-1][split + "_" + mse], rtol=1e-8, atol=1e-12)
    complete = actual == expected_steps and not any(summary.get(key, False) for key in
                ("stopped_by_wall_limit", "stopped_by_training_limit", "stopped_by_train_plateau"))
    return dict(actual_steps=actual, requested_steps=expected_steps, trained_to_fixed_endpoint=complete,
                future_contract=future, final_logged_metrics=history[-1])


def flow_points(arrays, split="test", max_points=160):
    """优先逐 C 点；无逐 C 数组时明确使用平均位移，不伪造非线性点。"""
    if type(max_points) is not int or max_points < 1:
        raise ValueError("max_points 必须为正整数")
    if "direct_draws_" + split in arrays:
        g, rg = arrays["direct_draws_" + split], arrays["R_g_draws_" + split]
        label = "逐 C 点的真实位移 R(g)-g"
    else:
        g, rg = arrays["direct_" + split], arrays["R_g_" + split]
        label = "平均 C 位移：平均 R(g)-平均 g\n不是 R(平均 g)-平均 g"
    flat_g, flat_rg = g.reshape(-1), rg.reshape(-1)
    levels = np.broadcast_to(np.arange(3).reshape((1,) * (g.ndim - 2) + (3, 1)), g.shape).reshape(-1)
    stride = max(1, int(np.ceil(len(flat_g) / max_points)))
    return flat_g[::stride], flat_rg[::stride], levels[::stride], label


def build_report(root, expected_steps=6000):
    root = Path(root)
    inputs, panels, arms = {}, {}, {}
    def load(path):
        inputs[str(path.relative_to(root))] = public_sha(path)
        return read_json(path)
    optional = {}
    for key, candidates in {
        "job": ("job.json",), "dataset_receipt": ("dataset_receipt.json", "data/dataset_receipt.json"),
        "normalizer": ("normalizer.json", "data/normalizer.json"),
        "pairs": ("pairs.json", "data/pairs.json"), "config": ("config.json",),
    }.items():
        found = [root / name for name in candidates if (root / name).is_file()]
        if len(found) > 1:
            raise ValueError(f"公开输入有歧义：{key}")
        optional[key] = load(found[0]) if found else None
    for arm in ARMS:
        run = root / ("run_" + arm)
        required = (run / "manifest.json", run / "summary.json")
        if not all(path.is_file() for path in required):
            arms[arm] = dict(status="not_complete", trained_to_fixed_endpoint=False,
                             reason="尚无完整 manifest/summary；未读取检查点")
            continue
        manifest, summary = (load(path) for path in required)
        metrics_path, arrays_path = run / "report/metrics.json", run / "report/encodings.npz"
        metrics = load(metrics_path) if metrics_path.is_file() else None
        info = validate_run(manifest, summary, metrics, arm, expected_steps)
        future = info["future_contract"]
        if optional["pairs"] is not None and optional["pairs"].get("data_sha256") != manifest.get("data_sha256"):
            raise ValueError("物理链配对与训练数据指纹不匹配")
        if optional["pairs"] is not None:
            pair_paths = [key for key in inputs if key.endswith("pairs.json")]
            if inputs[pair_paths[0]] != future.get("pairs_sha256"):
                raise ValueError("配对文件 SHA 与训练协议不匹配")
        if arm == "future":
            norm = optional["normalizer"]
            if (norm is None or norm.get("fitting_split") != "train"
                    or norm.get("data_sha256") != manifest.get("data_sha256")
                    or norm.get("pairs_sha256") != future.get("pairs_sha256")
                    or norm.get("target_mean") != future.get("target_mean")
                    or norm.get("target_std") != future.get("target_std")):
                raise ValueError("固定初始随机 C 的 TRAIN 归一化回执缺失或不匹配")
            norm_paths = [key for key in inputs if key.endswith("normalizer.json")]
            if inputs[norm_paths[0]] != future.get("normalizer_sha256"):
                raise ValueError("归一化文件 SHA 与训练协议不匹配")
        complete = info["trained_to_fixed_endpoint"] and metrics is not None and arrays_path.is_file()
        info.update(status="complete" if complete else "incomplete", collection_metrics=metrics,
                    future_metrics_available=arm == "future")
        if metrics is not None:
            estimator = collection_estimator(metrics)
            info.update(encoding_estimator=estimator,
                        encoding_protocol=metrics.get("encoding_protocol"),
                        encoding_matches_training=metrics.get("encoding_matches_training", False),
                        encoding_legacy_unmarked="encoding_estimator" not in metrics,
                        encoding_qualification=("历史全体 Q 编码；未按本次修复重采集，不宣称匹配训练半子集估计器"
                            if estimator == FULL_ENSEMBLE_ESTIMATOR
                            else "训练闭合的半子集 Q 均值；固定划分，不是训练随机划分的平均"))
        if metrics is not None and arrays_path.is_file():
            arrays = load_arrays(arrays_path, metrics)
            inputs[str(arrays_path.relative_to(root))] = public_sha(arrays_path)
            info["closure_of_C_means"] = {split: closure_summary(arrays, split) for split in ("train", "test")}
            info["jensen_gap_RMSE"] = {split: float(np.sqrt(np.mean((arrays["R_g_" + split] - arrays["R_of_mean_g_" + split]) ** 2))) for split in ("train", "test")}
            panels[arm] = arrays
        info["config"] = manifest["config"]
        info["history"] = summary["history"]
        arms[arm] = info
    comparison_estimator = None
    if all(arm in panels for arm in ARMS):
        comparison_estimator = require_matching_estimators(
            arms[arm]["collection_metrics"] for arm in ARMS)
        for split in ("train", "test"):
            for key in ("couplings_", "group_ids_"):
                np.testing.assert_array_equal(panels["control"][key + split], panels["future"][key + split])
        for key in ("data_sha256", "pairs_sha256"):
            if arms["control"]["future_contract"].get(key) != arms["future"]["future_contract"].get(key):
                raise ValueError("两臂的数据与配对协议必须一致")
    complete = all(info["status"] == "complete" for info in arms.values())
    result = dict(schema_version=2, status="complete" if complete else "incomplete",
        formal_comparison_available=complete, requested_steps_per_arm=expected_steps,
        comparison_encoding_estimator=comparison_estimator,
        legacy_encoding_comparison=comparison_estimator == FULL_ENSEMBLE_ESTIMATOR,
        experiment="新随机初始化 b3 QCR 配对对照；不是历史 b2 的 9000 步模型延续或复现",
        scales=list(SCALES), observable_names=list(OBSERVABLES), arms=arms,
        thermal_mode_certified=False, thermal_nu=None, root_search_performed=False,
        objectives_compared_as_total=False, control_future_loss=None,
        fixed_normalizer=optional["normalizer"], dataset_receipt=optional["dataset_receipt"],
        root_job_status=None if optional["job"] is None else optional["job"].get("status"),
        input_sha256=inputs,
        interpretation=[
            "新增目标只约束有限物理量的跨尺度条件均值，不证明完整分布充分性或正确临界指数。",
            "两臂 Q 坐标独立学习：潜变量和原始闭合误差只作各臂内描述，不能据跨臂原始数值大小断言物理更准确。",
            "固定初始随机 C 的 TRAIN 均值/标准差及声明的下限不随 C 更新；原始 RMSE 与归一化 RMSE 必须并列。",
            "原始未来误差将五类不同物理统计量等权混合，仅作辅助；归一化误差按路径和观测量等权。",
            "对照臂没有辅助预测头，未来预测损失为空而非零；不能用新增臂自身预测头证明相对对照的预测改进。",
            "图中闭合指标先平均 C；它不同于训练逐随机视图损失。归一化分母为本臂 TRAIN 相应目标尺度标准差。",
            "每臂编码估计器显式记录；历史未标记采集保持全体 Q 口径，禁止与训练半子集 Q 均值跨臂混比。",
            "R(平均 g) 与平均 R(g) 分别保存和展示；没有逐 C 数组时只称平均位移。",
            "有限链的自相关警告原样继承；独立链种子不等于链内构型独立，也不能证明已平衡。",
            "耦合标签仅用于评价图；不同有限扰动不能合并成唯一 g(K) 曲线。",
            "单训练种子、有限尺寸、有限偶观测基与零磁场限制保留；本报告不检验或选择热模。",
        ])
    return result, panels


@contextmanager
def chinese_plot_context(font=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    selected = select_chinese_font(font)
    with plt.rc_context({"font.family": font_manager.FontProperties(fname=str(selected)).get_name(), "axes.unicode_minus": False}), warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        yield selected


def save_chinese(fig, path, font):
    from matplotlib.text import Text
    codepoints = _font_codepoints(str(font))
    for artist in fig.findobj(match=Text):
        text = artist.get_text()
        missing = {c for c in text if "\u3400" <= c <= "\u9fff" and ord(c) not in codepoints}
        if missing:
            raise ValueError("中文字体缺字：" + "".join(sorted(missing)))
        properties = artist.get_fontproperties().copy()
        properties.set_file(str(font))
        artist.set_fontproperties(properties)
    fig.savefig(path, dpi=160)


def _missing(ax, text="尚无已保存的完整采集结果"):
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_report(result, panels, output, font=None):
    import matplotlib.pyplot as plt
    with chinese_plot_context(font) as selected:
        fig, axes = plt.subplots(4, 2, figsize=(14, 15), layout="constrained")
        for col, (arm, label) in enumerate(ARMS.items()):
            info = result["arms"][arm]
            history = info.get("history", [])
            if not history:
                for ax in axes[:, col]: _missing(ax)
                continue
            cfg = info["config"]["loss"]
            weights = cfg["weights"]
            groups = [(("separation", weights["separation"], "仿射概率重构"), ("anchor", weights["anchor"], "微观子集一致性"), ("coarse_consistency", weights["coarse_consistency"], "粗尺度子集一致性")),
                      (("rsmi", 2., "经验互信息贡献"), ("kernel_noncollapse", 2., "方差下限"), ("kernel_sign", 1., "均匀块颜色保持")),
                      (("commutator", weights["commutator"], "单步闭合"), ("rollout", cfg["rollout_weight"], "多步闭合"))]
            for row, group in enumerate(groups):
                ax = axes[row, col]
                for color, (key, weight, name) in enumerate(group):
                    # rsmi is recorded as -I; its weighted contribution is -2I.
                    for split, style, split_label in (("train", "-", "训练"), ("test", "--", "测试")):
                        rows = [r for r in history if split + "_" + key in r]
                        if rows:
                            ax.plot([r["epoch"] for r in rows], [weight * r[split + "_" + key] for r in rows], style, color=f"C{color}", label=name + "·" + split_label)
                ax.set(title=label + "：" + ("Q 基础表示", "C 目标分项", "R 闭合分项")[row], xlabel="实际更新次数", ylabel="分项加权值", yscale="symlog")
                ax.grid(alpha=.2)
                if ax.lines: ax.legend(fontsize=8)
            ax = axes[3, col]
            if arm == "control":
                _missing(ax, "对照无未来预测头；该误差不适用\n不将缺失值记为零")
            else:
                right = ax.twinx()
                for split, style, split_label in (("train", "-", "训练"), ("test", "--", "测试")):
                    for target_ax, key, color, name in ((ax, "future_physics_nrmse", "C0", "固定归一化"), (right, "future_physics_raw_rmse", "C3", "原始物理量")):
                        rows = [r for r in history if split + "_" + key in r]
                        target_ax.plot([r["epoch"] for r in rows], [r[split + "_" + key] for r in rows], style, color=color, label=name + "·" + split_label)
                ax.set(title="未来物理预测：两种单位分开", xlabel="实际更新次数", ylabel="固定归一化 RMSE")
                right.set_ylabel("原始物理量 RMSE")
                handles, labels = ax.get_legend_handles_labels(); h2, l2 = right.get_legend_handles_labels()
                ax.legend(handles + h2, labels + l2, fontsize=8)
        fig.suptitle("基础分项与新增预测误差分别展示；不比较不同目标的总损失")
        save_chinese(fig, output / "loss_comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
        path_labels = [f"{SCALES[a]}→{SCALES[b]}" for a, b in PATHS]
        for col, (arm, label) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            arrays, info = panels[arm], result["arms"][arm]
            for row, key, name in ((0, "rmse", "绝对 RMSE"), (1, "nrmse", "本臂 TRAIN 标准差归一化 RMSE")):
                for split, marker, split_label in (("train", "o", "训练"), ("test", "s", "测试")):
                    vals = [r[key] if r[key] is not None else np.nan for r in info["closure_of_C_means"][split]]
                    axes[row, col].plot(path_labels, vals, marker + "-", label=split_label)
                axes[row, col].set(title=label + "：平均 C 后闭合", ylabel=name)
                axes[row, col].legend(); axes[row, col].grid(alpha=.2)
            ax = axes[2, col]
            # Equal-weight condition means are descriptive; never bootstrap C draws as chains.
            groups = arrays["group_ids_test"]
            for index, group in enumerate(np.unique(groups)):
                ids = groups == group
                ax.plot(SCALES, arrays["direct_test"][ids, :, 0].mean(0), "-", color=f"C{index % 10}", alpha=.65, label="直接 Q" if index == 0 else None)
                ax.plot(SCALES, arrays["rollout_test"][ids, :, 0].mean(0), "--", color=f"C{index % 10}", alpha=.65, label="微观起点 R 迭代" if index == 0 else None)
            ax.set(title=label + "：测试物理条件均值", xlabel="晶格边长 L", ylabel="本臂编码 g", xticks=SCALES)
            ax.invert_xaxis(); ax.legend(); ax.grid(alpha=.2)
        fig.suptitle("新 b3：108→36→12；各臂潜坐标不同，不据原始数值跨臂判优")
        save_chinese(fig, output / "comparison.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        for col, (arm, label) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            arrays = panels[arm]
            g, rg, levels, flow_label = flow_points(arrays)
            ax = axes[0, col]
            ax.scatter(g, rg, c=levels, cmap="viridis", s=10, alpha=.5, label="逐 C 的 R(g)" if "direct_draws_test" in arrays else "平均 R(g)")
            mean_g = arrays["direct_test"].reshape(-1)
            ax.scatter(mean_g, arrays["R_of_mean_g_test"].reshape(-1), s=9, marker="x", color="C3", alpha=.45, label="R(平均 g)")
            lower, upper = min(float(g.min()), float(rg.min())), max(float(g.max()), float(rg.max()))
            ax.plot([lower, upper], [lower, upper], "k--", lw=.8, label="恒等映射")
            ax.set(title=label + "：保存的真实映射", xlabel="输入 g（或平均 g）", ylabel="输出 R(g)")
            ax.legend(fontsize=8); ax.grid(alpha=.2)
            ax = axes[1, col]
            ax.quiver(g, levels, rg-g, np.zeros_like(g), levels, cmap="viridis", angles="xy", scale_units="xy", scale=1, width=.003)
            ax.set(title=flow_label, xlabel="输入编码 g；箭头不归一化", ylabel="输入尺度", yticks=[0, 1, 2], yticklabels=[str(v) for v in SCALES], ylim=(-.5, 2.5))
            ax.update_datalim(np.column_stack((rg, levels))); ax.autoscale_view(scaley=False); ax.grid(alpha=.2)
        fig.suptitle("仅绘制已采集点，不生成网格推理、不搜根、不估计热指数")
        save_chinese(fig, output / "R_vs_g.png", selected); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
        from matplotlib.colors import Normalize
        maximum_delta = max((float(np.linalg.norm(arrays["couplings_" + split][:, 1:], axis=1).max())
                             for arrays in panels.values() for split in ("train", "test")), default=0.)
        delta_norm = Normalize(vmin=0., vmax=max(maximum_delta, 1e-12))
        last_scatter = None
        for col, (arm, label) in enumerate(ARMS.items()):
            if arm not in panels:
                for ax in axes[:, col]: _missing(ax)
                continue
            arrays = panels[arm]
            for row, scale in enumerate(SCALES):
                ax = axes[row, col]
                for split, marker, split_label in (("train", "o", "训练"), ("test", "^", "测试")):
                    coupling = arrays["couplings_" + split]
                    last_scatter = ax.scatter(coupling[:, 0], arrays["direct_" + split][:, row, 0], c=np.linalg.norm(coupling[:, 1:], axis=1), cmap="viridis", norm=delta_norm, marker=marker, s=12, alpha=.55, label=split_label)
                ax.set(title=label + f"：L={scale}", xlabel="生成时基准 K0（仅评价）", ylabel="本臂编码 g")
                ax.legend(fontsize=8); ax.grid(alpha=.2)
        if last_scatter is not None:
            fig.colorbar(last_scatter, ax=axes.ravel().tolist(), label="六个有限物理耦合扰动的模长")
        fig.suptitle("耦合到各尺度 g；颜色为有限扰动模长，各条件不连成唯一 g(K) 曲线")
        save_chinese(fig, output / "coupling_by_scale.png", selected); plt.close(fig)


def report_run(root, output=None, *, expected_steps=6000, font=None, make_plots=True):
    root = Path(root)
    output = Path(output) if output else root / "report"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("只写新空报告目录，不覆盖历史报告")
    result, panels = build_report(root, expected_steps)
    output.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_report(result, panels, output, font)
    result["public_files"] = ([*FIGURES] if make_plots else []) + ["report.json"]
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
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
