"""Chinese numeric-only reports for explicitly data-aligned baseline adapters.

No model/checkpoint/physical cache is imported or loaded, and no root search,
PCA refit, training, sampling or inference occurs.  All R outputs and any roots
must already be supplied by the caller.  Different C define different targets;
these plots do not turn within-method latent errors into a fair global ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARM_FILES = ("arm_report.json", "history.json", "主成分方差占比.png",
    "主成分R映射.png", "主成分位移流.png", "耦合到主成分.png",
    "损失曲线.png", "物理统计误差.png")
COMPARISON_FILES = ("comparison_report.json", "方法比较.png")
CAUTION = "不同粗粒化核定义不同目标；潜坐标误差不能直接横比胜负。"
MAP_DEFINITION = "先对相同评价条件的物理链和条件C重复平均已保存的(g,R(g))；输出为平均R(g)，不是R(平均g)。"
PALETTE = ("#2867ad", "#df8b27", "#2b9867", "#a052a0", "#8a6547")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("报告输入必须为有限的数字和标准JSON类型")


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限实数数组，禁止对象或模型输入")
    return array


def _validate_arrays(arrays):
    required = ("g", "r_one", "rollout", "labels", "split", "p16",
                "decoded_direct", "decoded_rollout", "pca_fraction")
    if not isinstance(arrays, dict) or any(key not in arrays for key in required):
        raise ValueError("缺少已保存的编码、实际R输出、物理统计或分组数组")
    result = {key: _numeric(arrays[key], key) for key in required}
    g = result["g"]
    if g.ndim != 4 or g.shape[2:] != (3, 5) or min(g.shape) < 1:
        raise ValueError("g必须为[物理链,C重复,3尺度,5坐标]")
    n, draws = g.shape[:2]
    if (result["r_one"].shape != (n, draws, 2, 5)
            or result["rollout"].shape != g.shape
            or result["labels"].shape != (n,) or result["split"].shape != (n,)
            or not np.isin(result["split"], (0, 1)).all()):
        raise ValueError("已保存R输出、参数标签或训练测试划分形状不一致")
    for key in ("p16", "decoded_direct", "decoded_rollout"):
        if result[key].shape != (n, draws, 3, 16):
            raise ValueError("物理统计与读出必须为[物理链,C重复,3尺度,16图案]")
    p = result["p16"]
    if p.min() < -1e-12 or p.max() > 1 + 1e-12 or not np.allclose(p.sum(-1), 1., atol=1e-10, rtol=1e-10):
        raise ValueError("观测图案必须是归一化概率；不修剪或归一化原始解码输出")
    fraction = result["pca_fraction"]
    if (fraction.shape != (5,) or np.any(fraction < 0)
            or fraction.sum() > 1 + 1e-9 or np.any(np.diff(fraction) > 1e-10)):
        raise ValueError("必须提供按降序排列的五个训练主成分方差占比")
    motif_fraction_keys = ("input_motif_pca_selected_fraction", "retained_input_variance_fraction")
    if any(key in arrays for key in motif_fraction_keys):
        if not all(key in arrays for key in motif_fraction_keys):
            raise ValueError("原图案方差占比与总保留率必须成对提供，避免混淆分母")
        selected = _numeric(arrays[motif_fraction_keys[0]], motif_fraction_keys[0])
        retained = _numeric(arrays[motif_fraction_keys[1]], motif_fraction_keys[1])
        if (selected.shape != (5,) or retained.ndim != 0 or np.any(selected < 0)
                or not 0 <= float(retained) <= 1 + 1e-9
                or np.any(np.diff(selected) > 1e-10)
                or not np.isclose(selected.sum(), float(retained), atol=1e-10, rtol=1e-9)):
            raise ValueError("原512图案的前五方差占比与总保留率不一致")
        expected = selected / float(retained) if float(retained) > 0 else np.zeros(5)
        if not np.allclose(fraction, expected, atol=1e-10, rtol=1e-9):
            raise ValueError("五维g内部占比必须按保留方差归一化；零塌缩保持全零")
        result[motif_fraction_keys[0]], result[motif_fraction_keys[1]] = selected, retained
    if not np.allclose(result["rollout"][:, :, 0], g[:, :, 0], atol=1e-12, rtol=1e-10):
        raise ValueError("已保存自主迭代必须从实际微观编码开始")
    for key in ("r_of_C_mean", "mean_R"):
        if key in arrays:
            value = _numeric(arrays[key], key)
            if value.shape != (n, 2, 5):
                raise ValueError("逐物理链的条件C均值诊断必须为[物理链,2转移,5坐标]")
            result[key] = value
    if "mean_R" in result and not np.allclose(result["mean_R"], result["r_one"].mean(axis=1),
                                               atol=1e-11, rtol=1e-9):
        raise ValueError("mean_R必须等于每条物理链已保存R(g)的C均值")
    return result


def _prepare_output(output_dir, names):
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("报告输出目录不得为符号链接")
    if output.exists() and not output.is_dir():
        raise ValueError("报告输出必须是目录")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() or (output / name).is_symlink() for name in names):
        raise FileExistsError("不覆盖已有报告；请选择新的报告输出目录")
    return output


def _save_json(path, value):
    serialized = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(serialized)


def _plot_context():
    # Existing font discovery only.  It never downloads/installs fonts, and its
    # save helper checks all actual Chinese glyphs rather than translating keys.
    from scripts.report_ising_future_physics import chinese_plot_context
    return chinese_plot_context()


def _save_figure(figure, path, font):
    from matplotlib.text import Text
    from scripts.report_ising_zh import _font_codepoints
    codepoints = _font_codepoints(str(font))
    for artist in figure.findobj(match=Text):
        missing = {c for c in artist.get_text() if "\u3400" <= c <= "\u9fff" and ord(c) not in codepoints}
        if missing:
            raise ValueError("中文字体缺字：" + "".join(sorted(missing)))
        properties = artist.get_fontproperties().copy()
        properties.set_file(str(font))
        artist.set_fontproperties(properties)
    with Path(path).open("xb") as stream:
        figure.savefig(stream, format="png", dpi=145)


def _grouped(arrays, mask):
    labels = arrays["labels"][mask]
    unique = np.unique(labels)
    grouped = {}
    for key in ("g", "r_one", "rollout"):
        values = arrays[key][mask]
        grouped[key] = np.stack([values[labels == label].mean(axis=(0, 1)) for label in unique])
    for key in ("r_of_C_mean", "mean_R"):
        if key in arrays:
            values = arrays[key][mask]
            grouped[key] = np.stack([values[labels == label].mean(axis=0) for label in unique])
    return unique, grouped


def _root_positions(fixed_points):
    if fixed_points is None:
        return []
    if not isinstance(fixed_points, list):
        raise ValueError("不动点只能是调用者已保存的列表；不在报告中求根")
    result = []
    for index, root in enumerate(fixed_points):
        if not isinstance(root, dict):
            raise ValueError("已保存不动点必须是字典")
        if "pc1" in root:
            point = _numeric(root["pc1"], "不动点主成分投影")
            if point.ndim:
                raise ValueError("已保存主成分一根投影必须是标量")
            position = float(point)
        elif "g" in root:
            point = _numeric(root["g"], "不动点坐标")
            if point.shape != (5,):
                raise ValueError("根坐标必须使用与当前图相同的五维坐标，或显式提供pc1")
            position = float(point[0])
        else:
            raise ValueError("不动点缺少已保存坐标；禁止推测投影位置")
        result.append((index + 1, position))
    return result


def _empty(axis, message):
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _variance_note(data):
    if "retained_input_variance_fraction" not in data:
        return "仅显示已保存的训练占比；方差集中不等于物理耦合维数。"
    retained = float(data["retained_input_variance_fraction"])
    if retained == 0:
        return "保留的五维 g 方差为零，占比保持全零；对原512图案总方差保留0%。\n方差集中不等于物理耦合维数。"
    return (f"按保留的五维 g 总方差归一化；对原512图案总方差保留{100 * retained:.5g}%。\n"
            "方差集中不等于物理耦合维数；两类占比的分母不同。")


def _loss_value(row, split):
    keys = (split + "_loss", split + "_total", split + "_closure", split + "_r_loss")
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
            return float(value)
    nested = row.get(split)
    if isinstance(nested, dict):
        for key in ("loss", "total", "closure", "r_loss"):
            value = nested.get(key)
            if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                return float(value)
    if split == "train":
        for key in ("loss", "total", "closure", "r_loss"):
            value = row.get(key)
            if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                return float(value)
    return None


def _history_points(history, getter):
    points = []
    for index, row in enumerate(history):
        value = getter(row)
        if value is None:
            continue
        step = row.get("step", row.get("epoch", row.get("update", index)))
        if not isinstance(step, (int, float, np.number)) or not np.isfinite(step):
            raise ValueError("损失更新次数必须为有限数字")
        points.append((step, value))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _native_loss(row):
    if row.get("stage") != "C":
        return None
    for key in ("native_loss", "loss"):
        value = row.get(key)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
            return float(value)
    return None


def _C_mi(row):
    value = row.get("mi_estimate") if row.get("stage") == "C" else None
    return (float(value) if isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_)) else None)


def _R_component(row, key):
    if row.get("stage", "R") != "R":
        return None
    value = row.get(key)
    return (float(value) if isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_)) else None)


def write_arm_report(output_dir, arrays: dict, metrics: dict, history: list[dict],
                     fixed_points: list[dict] | None = None) -> list[str]:
    """Write six Chinese figures and JSON receipts, preserving supplied metrics.

    ``g`` already uses the supplied shared TRAIN principal-component coordinates;
    this function does not fit or rotate a PCA.  Per-condition averages use
    actual paired g/R arrays, so their nonlinear map is mean R(g), never R(mean
    g).  With no saved roots there are no root markers.  K labels only group and
    color the post-training evaluation; they never fit a map in this function.
    """
    data = _validate_arrays(arrays)
    if not isinstance(metrics, dict) or not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("metrics必须为字典，history必须为数字记录列表")
    # Validate JSON inputs before creating any output file; do not mutate them.
    saved_metrics, saved_history = _jsonable(metrics), _jsonable(history)
    roots = _root_positions(fixed_points)
    output = _prepare_output(output_dir, ARM_FILES)
    selected_split = 1 if np.any(data["split"] == 1) else 0
    split_name = "测试集" if selected_split else "训练集（无测试数据）"
    mask = data["split"] == selected_split
    labels, grouped = _grouped(data, mask)
    with _plot_context() as font:
        import matplotlib.pyplot as plt
        with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
            fig, axis = plt.subplots(figsize=(9, 4.8))
            fractions = data["pca_fraction"]
            bars = axis.bar(np.arange(5), 100 * fractions, color=PALETTE)
            for bar, value in zip(bars, fractions):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{100 * value:.4g}%", ha="center", va="bottom")
            axis.set(title="训练统计的主成分方差分布", ylabel="解释方差占比（%）",
                xticks=np.arange(5), xticklabels=[f"第{i+1}主成分" for i in range(5)],
                ylim=(0, max(10., 110 * float(fractions.max()))))
            axis.grid(axis="y", alpha=.2)
            fig.text(.5, .015, _variance_note(data), ha="center", fontsize=8.5)
            fig.tight_layout(rect=(0, .10 if "retained_input_variance_fraction" in data else .05, 1, 1))
            _save_figure(fig, output / ARM_FILES[2], font); plt.close(fig)

            fig, axis = plt.subplots(figsize=(8, 6))
            for k in range(2):
                x, y = grouped["g"][:, k, 0], grouped["r_one"][:, k, 0]
                axis.scatter(data["g"][mask, :, k, 0], data["r_one"][mask, :, k, 0],
                    s=7, alpha=.12, color=PALETTE[k])
                axis.plot(x, y, ".-", color=PALETTE[k], label=f"{108//3**k}→{108//3**(k+1)}：平均 R(g)")
                if "r_of_C_mean" in grouped:
                    axis.plot(x, grouped["r_of_C_mean"][:, k, 0], "--", color=PALETTE[k],
                        label=f"{108//3**k}→{108//3**(k+1)}：逐链先均 C 后 R，再均链")
            for index, point in roots:
                axis.scatter(point, point, marker="D", s=45, facecolor="none", edgecolor="#7b3387",
                    label="已保存全坐标数值根的投影" if index == 1 else None)
            low, high = min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1])
            axis.plot([low, high], [low, high], "--", color="#777777", label="恒等映射")
            axis.set(xlabel="输入主成分一的条件均值", ylabel="输出主成分一的条件均值",
                title=f"{split_name}：主成分一上的已保存 R 映射")
            axis.grid(alpha=.2); axis.legend(fontsize=8)
            map_note = ("实线为平均 R(g)；虚线为逐链 C 均值经 R 后再均物理链，均不是 R(同条件全链均值)。"
                        if "r_of_C_mean" in grouped else
                        "输出为平均 R(g)，不是 R(平均 g)；曲线由相同条件的已保存点连接。")
            fig.text(.5, .055, map_note, ha="center", fontsize=8.5)
            fig.text(.5, .02, "投影交点不是新不动点；未对模型推理、重新拟合或求根。", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .09, 1, 1))
            _save_figure(fig, output / ARM_FILES[3], font); plt.close(fig)

            fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [2, 1]})
            for k in range(2):
                x = grouped["g"][:, k, 0]
                displacement = grouped["r_one"][:, k, 0] - x
                axes[0].plot(x, displacement, ".-", color=PALETTE[k], label=f"{108//3**k}→{108//3**(k+1)}")
                axes[1].quiver(x, np.full_like(x, k), displacement, np.zeros_like(x),
                    angles="xy", scale_units="xy", scale=1, color=PALETTE[k], width=.004)
            axes[0].axhline(0., ls="--", color="#777777")
            for _, point in roots:
                axes[0].scatter(point, 0., marker="D", s=40, facecolor="none", edgecolor="#7b3387")
            axes[0].set(title=f"{split_name}：真实投影位移", ylabel="平均 [R(g)−g] 的主成分一")
            axes[0].legend(); axes[0].grid(alpha=.2)
            axes[1].set(xlabel="输入主成分一的条件均值", ylabel="尺度路径", ylim=(-.5, 1.5),
                yticks=[0, 1], yticklabels=["108→36", "36→12"])
            x_all = np.r_[grouped["g"][:, :2, 0].ravel(), grouped["r_one"][:, :, 0].ravel()]
            span = max(float(np.ptp(x_all)), 1e-8)
            axes[1].set_xlim(float(x_all.min()) - .05 * span, float(x_all.max()) + .05 * span)
            fig.text(.5, .02, "箭头长度为实际位移，未归一化；纵向分层仅为避免遮挡。", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            _save_figure(fig, output / ARM_FILES[4], font); plt.close(fig)

            fig, axis = plt.subplots(figsize=(9, 5))
            for split, line, name in ((0, "-", "训练"), (1, "--", "测试")):
                here = data["split"] == split
                if not here.any():
                    continue
                k_values, group = _grouped(data, here)
                for k in range(3):
                    axis.plot(k_values, group["g"][:, k, 0], "." + line,
                        color=PALETTE[k], label=f"{name}：k={k}，L={108//3**k}")
            axis.set(xlabel="初始耦合 K（仅用于评价）", ylabel="主成分一条件均值",
                title="耦合条件与各尺度表示的对应关系")
            axis.grid(alpha=.2); axis.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            _save_figure(fig, output / ARM_FILES[5], font); plt.close(fig)

            fig, loss_axes = plt.subplots(2, 1, figsize=(10, 8))
            native_axis, axis = loss_axes
            native = _history_points(history, _native_loss)
            if len(native):
                native_axis.plot(native[:, 0], native[:, 1], color=PALETTE[0], label="原生粗粒化目标")
                native_axis.set(title="C：已保存的原生训练目标", xlabel="C 实际更新次数",
                    ylabel="原生目标值", yscale="symlog")
                native_axis.grid(alpha=.2)
                native_axis.legend(loc="upper left", fontsize=9)
            else:
                _empty(native_axis, "没有已保存的 C 原生目标历史；不将缺失值记为零")
            mi = _history_points(history, _C_mi)
            if len(mi):
                mi_axis = native_axis.twinx()
                mi_axis.plot(mi[:, 0], mi[:, 1], "--", color=PALETTE[3], label="互信息估计")
                mi_axis.set_ylabel("互信息估计（原生单位）")
                mi_axis.legend(loc="upper right", fontsize=9)
            for split, style, label in (("train", "-", "训练"), ("test", "--", "测试")):
                points = _history_points(history, lambda row, split=split: (
                    _loss_value(row, split) if row.get("stage", "R") == "R" else None))
                if len(points):
                    axis.plot(points[:, 0], points[:, 1], style, label=label)
            for key, label, color in (("one_step", "一步闭合", PALETTE[2]),
                                      ("rollout", "多步闭合", PALETTE[3])):
                points = _history_points(history, lambda row, key=key: _R_component(row, key))
                if len(points):
                    axis.plot(points[:, 0], points[:, 1], ":", color=color, label=label)
            if axis.lines:
                axis.set(title="R：共同流读出的已保存闭合历史", xlabel="R 实际更新次数", ylabel="已记录损失",
                    yscale="symlog")
                axis.grid(alpha=.2); axis.legend()
            else:
                _empty(axis, "没有可用的已保存损失历史；缺失不记为零")
            fig.text(.5, .02, "C 原生目标与 R 共同读出分开；不同方法的原生目标不作数值横比，互信息单位保持原记录。", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            _save_figure(fig, output / ARM_FILES[6], font); plt.close(fig)

            fig, axis = plt.subplots(figsize=(9, 5))
            p = data["p16"][mask]
            for key, label, color in (("decoded_direct", "直接编码读出", PALETTE[0]),
                                      ("decoded_rollout", "自主迭代读出", PALETTE[1])):
                # Each physical chain receives equal weight; draws remain
                # conditional repeats and are averaged after squaring errors.
                per_chain_scale_mse = ((data[key][mask] - p) ** 2).mean(axis=(1, 3))
                error = np.sqrt(per_chain_scale_mse.mean(axis=0))
                axis.plot(np.arange(3), error, "o-", color=color, label=label)
            axis.set(xticks=np.arange(3), xticklabels=["108", "36", "12"],
                xlabel="实际格点边长", ylabel="局域图案概率的均方根误差",
                title=f"{split_name}：物理统计读出误差（原始概率单位）")
            axis.grid(alpha=.2); axis.legend()
            fig.text(.5, .055, "先平均每条物理链内的 C 平方误差；未裁剪或重新归一化解码输出。", ha="center", fontsize=9)
            fig.text(.5, .02, CAUTION, ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .09, 1, 1))
            _save_figure(fig, output / ARM_FILES[7], font); plt.close(fig)

    _save_json(output / ARM_FILES[1], saved_history)
    _save_json(output / ARM_FILES[0], dict(schema_version=1, status="complete", metrics=saved_metrics,
        files=list(ARM_FILES), plotting_split="test" if selected_split else "train",
        array_shapes={key: list(value.shape) for key, value in data.items()},
        model_loaded=False, inference_performed=False, PCA_refitted=False, roots_recomputed=False,
        root_marker_count=len(roots), fixed_points=_jsonable(fixed_points),
        map_definition=MAP_DEFINITION, displacement="条件平均的实际R(g)-g；箭头不归一化",
        conditional_mean_R_diagnostic_available="r_of_C_mean" in data,
        conditional_mean_R_definition=("每条物理链先平均其C重复的g，再使用已保存R结果，最后按同K物理链平均；不是R作用于同K全链总均值"
            if "r_of_C_mean" in data else None),
        variance_fraction_definition=_variance_note(data),
        projection_caution="主成分一投影不是独立一维自治映射；根仅为已保存全坐标数值根的投影",
        comparison_caution=CAUTION, metrics_recomputed_or_overwritten=False,
        local_plot_error="图内逐尺度误差只从传入公开数组绘制，不覆盖主指标",
        labels_used_in_training=False, labels_used_for="事后评价分组和耦合到表示图"))
    return list(ARM_FILES)


def _display_name(key, number):
    lower = key.lower()
    if "qcr_c_pca" in lower:
        return "冻结已学习 C＋主成分读出"
    if "nmcrg" in lower or "neural_mcrg" in lower:
        return "神经蒙卡重整化适配"
    if "rsmi" in lower:
        return "实空间互信息适配"
    if "decimation" in lower:
        return "固定抽取＋主成分读出"
    if "majority" in lower:
        return "多数规则＋主成分读出"
    if "reference" in lower or key == "参照":
        return "当前框架参照"
    if any("\u3400" <= c <= "\u9fff" for c in key) and not any(c.isascii() and c.isalpha() for c in key):
        return key
    return f"方法{number}"


def _last_scalar(value):
    if value is None:
        return None
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or not values.size or not np.isfinite(values).all():
        return None
    return float(values.reshape(-1)[-1])


def _comparison_value(metrics, family, key):
    split = "test" if isinstance(metrics.get("test"), dict) else "train"
    section = metrics.get(split, {}).get(family, {})
    return _last_scalar(section.get(key)) if isinstance(section, dict) else None


def write_comparison_report(output_dir, summaries: dict, reference: dict | None = None) -> list[str]:
    """Plot saved endpoint indicators with explicit cross-C limitations.

    Expected metrics include test/train.latent.rollout_nrmse and
    test/train.motif16.rollout_rmse.  A list-valued indicator uses its LAST saved
    target, never the best target; missing indicators remain absent.  ``reference``
    is an optional metrics dictionary, not a model or a new inference request.
    """
    if not isinstance(summaries, dict) or not summaries or any(not isinstance(value, dict) for value in summaries.values()):
        raise ValueError("需要至少一个方法的已保存数值汇总")
    if reference is not None and not isinstance(reference, dict):
        raise ValueError("参照必须为已保存指标字典，不得传入模型")
    saved_summaries, saved_reference = _jsonable(summaries), _jsonable(reference)
    rows = [(str(key), value) for key, value in summaries.items()]
    if reference is not None:
        rows.append(("reference", reference))
    names = [_display_name(key, index + 1) for index, (key, _) in enumerate(rows)]
    output = _prepare_output(output_dir, COMPARISON_FILES)
    with _plot_context() as font:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(max(12., 2.3 * len(rows)), 6.5))
        for axis, family, key, title, ylabel in (
            (axes[0], "latent", "rollout_nrmse", "各自坐标内的两步预测", "本方法训练标准差归一化误差"),
            (axes[1], "motif16", "rollout_rmse", "各自粗粒化目标的局域统计预测", "图案概率均方根误差")):
            values = [_comparison_value(metrics, family, key) for _, metrics in rows]
            for index, value in enumerate(values):
                if value is None:
                    axis.text(index, 0., "缺失", ha="center", va="bottom", color="#777777")
                else:
                    axis.bar(index, value, color=PALETTE[index % len(PALETTE)], width=.65)
                    axis.annotate(f"{value:.4g}", (index, value), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
            axis.set(title=title, ylabel=ylabel, xticks=np.arange(len(rows)), xticklabels=names)
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
        fig.suptitle("单种子同数据适配比较：指标不等于统一物理排名")
        fig.text(.5, .075, CAUTION, ha="center", fontsize=10)
        fig.text(.5, .04, "即使共享 C，坐标和读出训练协议仍需对齐；与历史框架参照的比较不是严格同坐标排名。", ha="center", fontsize=9)
        fig.text(.5, .01, "缺少测试指标时显示训练指标；缺失值不记为零。", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .95))
        _save_figure(fig, output / COMPARISON_FILES[1], font); plt.close(fig)
    _save_json(output / COMPARISON_FILES[0], dict(schema_version=1, status="complete",
        summaries=saved_summaries, reference=saved_reference,
        display_names={key: name for (key, _), name in zip(rows, names)},
        files=list(COMPARISON_FILES), model_loaded=False, inference_performed=False,
        metrics_recomputed_or_overwritten=False, cross_method_latent_ranking_valid=False,
        comparison_caution=CAUTION, shared_C_caution="qcr_c_pca与参照即使共用C，也不自动拥有相同坐标或读出训练协议",
        scalar_selection="列表型指标固定取最后一个已保存目标；不挑选最优目标",
        missing_values_are_zero=False))
    return list(COMPARISON_FILES)
