"""从新探针的公开 JSON/NPZ 生成少链、冻结模型大尺寸泛化报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_ising_future_physics import (
    chinese_plot_context, public_sha, read_json, save_chinese,
)


SIZES = (1080, 108)
COUPLINGS = (.42, .44, .46)
FIGURES = ("configurations.png", "latent_prediction.png", "motif_prediction.png",
           "errors.png", "R_vs_g.png", "coupling_by_scale.png")
ENCODING_ESTIMATOR = "mean_of_two_disjoint_Q_halves_v1"


def _root_file(root, relative, suffix=None):
    """Only read files inside this new output, never follow a receipt outside it."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("公开输入必须是新输出根内的相对路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or (suffix and path.suffix != suffix):
        raise ValueError("公开输入不得越过新输出根或读取非 JSON/NPZ")
    if not path.is_file():
        raise ValueError(f"缺少公开输入：{relative}")
    return path


def _load_npz(path):
    if Path(path).suffix != ".npz":
        raise ValueError("报告只读取公开 NPZ")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    for key, value in result.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"数组必须为有限数值：{key}")
    return result


def validate_arrays(arrays):
    """Validate labels and row grouping without estimating anything from a model."""
    for key in ("row_size0", "row_K0", "row_replica"):
        if key not in arrays or arrays[key].shape != (12,):
            raise ValueError("需要两尺寸、三耦合、每条件两条物理链的 12 行标签")
    sizes, couplings, replicas = (arrays[k] for k in ("row_size0", "row_K0", "row_replica"))
    if sizes.dtype.kind not in "iu" or replicas.dtype.kind not in "iu":
        raise ValueError("尺寸和物理链编号必须为整数")
    expected = {(size, coupling, replica) for size in SIZES for coupling in COUPLINGS for replica in (0, 1)}
    found = {(int(size), float(k), int(rep)) for size, k, rep in zip(sizes, couplings, replicas)}
    if found != expected or len(found) != len(sizes):
        raise ValueError("尺寸、耦合和物理链组合不匹配或重复")
    direct = arrays.get("direct_g")
    if direct is None or direct.ndim != 4 or direct.shape[:3] != (12, 5, 4) or direct.shape[3] != 1:
        raise ValueError("冻结标量 b3 编码必须为 [12物理链,5次C,4尺度,1坐标]")
    for key in ("R_rollout_g", "R_one_step_g"):
        if arrays.get(key, np.empty(0)).shape != direct.shape:
            raise ValueError("缺少真实保存的 R 预测数组")
    for key in ("p_observed", "p_decoded_direct", "p_decoded_rollout"):
        if arrays.get(key, np.empty(0)).shape != (12, 5, 4, 16):
            raise ValueError("局部 motif 数组必须为 [物理链,C重复,尺度,16概率]")
    if "long_range_observed" in arrays and arrays["long_range_observed"].shape != (12, 5, 4, 5):
        raise ValueError("长程实测诊断必须为 [物理链,C重复,尺度,5观测]")
    if arrays.get("scales", np.empty(0)).shape != (12, 4):
        raise ValueError("缺少逐行格点尺寸")
    for row, size in enumerate(sizes):
        np.testing.assert_array_equal(arrays["scales"][row], [int(size) // 3 ** k for k in range(4)])
    np.testing.assert_allclose(arrays["R_rollout_g"][:, :, 0], direct[:, :, 0], rtol=1e-10, atol=1e-12)
    observed = arrays["p_observed"]
    if observed.min() < -1e-12 or observed.max() > 1 + 1e-12:
        raise ValueError("实际 motif 频率不是有效概率")
    np.testing.assert_allclose(observed.sum(-1), 1., rtol=1e-9, atol=1e-10)
    # Do not clip or normalize the raw affine decoder: negative values are diagnostics.
    if "Q_full_minus_half" in arrays:
        if arrays["Q_full_minus_half"].shape != direct.shape or arrays.get("full_direct_g", np.empty(0)).shape != direct.shape:
            raise ValueError("全体编码诊断 shape 不一致")
        np.testing.assert_allclose(arrays["Q_full_minus_half"], arrays["full_direct_g"] - direct)
    for key, expected_value in (("error_total", arrays["p_decoded_rollout"] - observed),
                                ("error_R", arrays["p_decoded_rollout"] - arrays["p_decoded_direct"]),
                                ("error_repr", arrays["p_decoded_direct"] - observed)):
        if key in arrays:
            np.testing.assert_allclose(arrays[key], expected_value, rtol=1e-9, atol=1e-11)


def error_decomposition(observed, decoded_direct, decoded_rollout, *, average_C=True):
    """C first, then chain; MSE_total = MSE_readout + MSE_R + cross, not RMSE addition."""
    values = (observed, decoded_direct, decoded_rollout)
    if average_C:
        values = tuple(value.mean(1) for value in values)
    observed, decoded_direct, decoded_rollout = values
    readout, transport = decoded_direct - observed, decoded_rollout - decoded_direct
    total = readout + transport
    chain_axes = tuple(range(1, total.ndim))
    per_chain = dict(total_mse=np.mean(total ** 2, axis=chain_axes),
                     readout_mse=np.mean(readout ** 2, axis=chain_axes),
                     R_transport_mse=np.mean(transport ** 2, axis=chain_axes),
                     cross_term=2 * np.mean(readout * transport, axis=chain_axes))
    result = {key: float(value.mean()) for key, value in per_chain.items()}
    for name in ("total", "readout", "R_transport"):
        result[name + "_rmse"] = float(np.sqrt(result[name + "_mse"]))
    result["per_physical_chain"] = {key: value.tolist() for key, value in per_chain.items()}
    np.testing.assert_allclose(result["total_mse"], result["readout_mse"] + result["R_transport_mse"] + result["cross_term"], rtol=1e-9, atol=1e-12)
    return result


def _raw_decoder_summary(value):
    sums = value.sum(-1) - 1.
    return dict(minimum=float(value.min()), maximum=float(value.max()),
                negative_fraction=float(np.mean(value < 0)), above_one_fraction=float(np.mean(value > 1)),
                sum_minus_one_mean=float(sums.mean()), sum_minus_one_max_abs=float(np.abs(sums).max()),
                clipped=False, renormalized=False)


def summarize(arrays):
    results = {}
    for size in SIZES:
        conditions = []
        for coupling in COUPLINGS:
            ids = (arrays["row_size0"] == size) & (arrays["row_K0"] == coupling)
            rows = np.flatnonzero(ids)
            chain_order = np.argsort(arrays["row_replica"][ids])
            rows = rows[chain_order]
            scales = []
            for k in range(4):
                direct, predicted = arrays["direct_g"][rows, :, k], arrays["R_rollout_g"][rows, :, k]
                delta = predicted.mean(1) - direct.mean(1)
                latent = dict(rmse=float(np.sqrt(np.mean(delta ** 2))),
                    per_physical_chain_rmse=np.sqrt(np.mean(delta ** 2, axis=-1)).tolist(),
                    per_C_draw_rmse=float(np.sqrt(np.mean((predicted - direct) ** 2))),
                    direct_C_mean_per_chain=direct.mean(1).tolist(),
                    rollout_C_mean_per_chain=predicted.mean(1).tolist())
                values = [arrays[key][rows, :, k] for key in ("p_observed", "p_decoded_direct", "p_decoded_rollout")]
                scales.append(dict(k=k, L=size // 3 ** k, trained_depth=k <= 2, latent=latent,
                    motif_of_C_means=error_decomposition(*values),
                    motif_per_C_draw=error_decomposition(*values, average_C=False),
                    raw_affine_decoder=dict(direct=_raw_decoder_summary(values[1]), rollout=_raw_decoder_summary(values[2]))))
                if "long_range_observed" in arrays:
                    observed = arrays["long_range_observed"][rows, :, k].mean(1)
                    scales[-1]["long_range_C_observed_only"] = dict(
                        per_physical_chain=observed.tolist(), two_chain_mean=observed.mean(0).tolist(),
                        R_prediction_available=False, interpretation="仅从实际 C 构型测量，不是 R 的预测结果")
            conditions.append(dict(K0=coupling, role="近临界评价点；非精确临界耦合" if coupling == .44 else "零扰动物理评价点",
                physical_chain_count=len(rows), C_draw_count=5, row_indices=rows.tolist(), scales=scales))
        results[str(size)] = dict(initial_size=size, scales=[size // 3 ** k for k in range(4)], conditions=conditions)
    return results


def build_report(root):
    root = Path(root)
    inputs = {}
    def load_json(relative):
        path = _root_file(root, relative, ".json")
        inputs[str(relative)] = public_sha(path)
        return read_json(path)
    status = load_json("collection/status.json")
    receipt = load_json("collection/receipt.json")
    if status.get("status") != "complete" or receipt.get("status") != "complete":
        raise ValueError("采集尚未完整结束，不生成完成报告")
    if receipt.get("training_updates") != 0 or receipt.get("mutual_information_evaluations") != 0:
        raise ValueError("本探针必须零训练、零互信息计算")
    if receipt.get("all_parameters_frozen") is not True or receipt.get("all_grad_none") is not True:
        raise ValueError("缺少参数冻结和 grad 全空证明")
    for key in ("model_sha256", "module_sha256"):
        before, after = receipt.get(key + "_before"), receipt.get(key + "_after")
        if not before or before != after:
            raise ValueError("冻结模型或 Q/C/R 指纹发生变化")
        if key == "module_sha256" and (set(before) != {"Q", "C", "R"} or not all(before.values())):
            raise ValueError("缺少完整 Q/C/R 模块指纹")
    plan = load_json("plan.json") if (root / "plan.json").is_file() else None
    if plan and plan.get("source", {}).get("module_sha256") != receipt["module_sha256_before"]:
        raise ValueError("Q/C/R 模块指纹与本次冻结计划不符")
    if receipt.get("encoding_estimator") != ENCODING_ESTIMATOR:
        raise ValueError("必须使用训练一致的半子集编码估计器")
    protocol = receipt.get("encoding_protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("必须保存固定完整编码划分协议")
    partition = protocol.get("sample_partition")
    if (type(protocol.get("version")) is not int or protocol.get("version") != 1
            or protocol.get("estimator") != ENCODING_ESTIMATOR
            or type(protocol.get("samples")) is not int or protocol.get("samples") != 64
            or type(protocol.get("half_samples")) is not int or protocol.get("half_samples") != 32
            or not isinstance(partition, list) or any(type(i) is not int for i in partition)
            or sorted(partition) != list(range(64))):
        raise ValueError("必须保存 64 构型、两半各32的固定完整编码划分协议")
    path = _root_file(root, "collection/arrays.npz", ".npz")
    arrays = _load_npz(path)
    inputs["collection/arrays.npz"] = public_sha(path)
    if receipt.get("arrays_sha256") != inputs["collection/arrays.npz"]:
        raise ValueError("公开评价数组指纹不符")
    validate_arrays(arrays)
    snapshot_path = _root_file(root, "collection/snapshots.npz", ".npz")
    inputs["collection/snapshots.npz"] = public_sha(snapshot_path)
    if receipt.get("snapshots_sha256") != inputs["collection/snapshots.npz"]:
        raise ValueError("构型快照指纹不符")
    # Load only the predetermined snapshots, never select by the apparent result.
    snapshots = {}
    with np.load(snapshot_path, allow_pickle=False) as archive:
        for row in np.flatnonzero(arrays["row_replica"] == 0):
            for k, L in enumerate(arrays["scales"][row]):
                key = f"row_{row:03d}_scale_{k}"
                if key not in archive.files:
                    raise ValueError("缺少预先固定的构型快照")
                value = archive[key]
                if value.ndim != 3 or value.shape[0] < 1 or value.shape[1:] != (int(L), int(L)) or not np.isin(value, (-1, 1)).all():
                    raise ValueError("构型快照必须为实际二值自旋格点")
                snapshots[key] = value[0]
    sampling = {}
    for row in receipt.get("dataset_rows", []):
        relative = row.get("sidecar_path")
        if relative and relative not in sampling:
            sampling[relative] = load_json(relative)
    if not sampling:
        raise ValueError("缺少新采样公开 sidecar，不能隐藏平衡或相关性警告")
    source_summary = None
    if (root / "source_record/summary.json").is_file():
        source_summary = load_json("source_record/summary.json")
    source_records = {}
    for name in ("manifest", "job", "plan"):
        if (root / f"source_record/{name}.json").is_file():
            source_records[name] = load_json(f"source_record/{name}.json")
    result = dict(schema_version=1, status="complete", workflow_complete=True,
        experiment="少链冻结 b3 模型的大尺寸与深度外推探针", scientific_scope="描述性探索，非统计认证",
        collection_receipt=receipt, sampling_sidecars=sampling, input_sha256=inputs,
        encoding_estimator=ENCODING_ESTIMATOR, training_updates=0, mutual_information_evaluations=0,
        baseline_constant_fitted=False, decoder_refitted=False, root_search_performed=False,
        thermal_nu=None, confidence_intervals_computed=False, bootstrap_performed=False,
        aggregation="每物理链先平均 5 次 C，再算链级误差；每条件两条链等权；不同条件不冒充独立重复",
        snapshot_selection=dict(replica=0, C_draw=0, sample=0, selection="预先固定，不择优", generated_by="实际 C 链；不是 R 生成"),
        sizes=summarize(arrays), source_training_reference=source_summary,
        source_record=source_records, probe_plan=plan,
        original_source_workflow_status=(source_records.get("job") or {}).get("status"),
        interpretation=[
            "只有冻结模型评价，没有训练、重拟合解码器或互信息计算；同一 Q/C/R 指纹前后核对。",
            "1080→360→120 是较大尺寸上的已训练深度 k≤2；120→40 对应未训练的第三步，尺寸与深度泛化分开解读。",
            "108→36→12→4 的 k=3 末端只有 16 个自旋；1080→360→120→40 的末端有限尺寸效应不同，不能把差异都归因 R。",
            "构型图是实际 C 粗粒化得到，不是 R 生成；视觉相似不能单独证明 R 的物理预测正确。",
            "每条件只有两条物理链，五次 C 不是十条独立物理链；不计算置信区间、bootstrap 或显著性。",
            "K=.44 只是近临界评价点，不等于精确 Kc；三个零扰动点不代表全耦合域或所有物理系统。",
            "局部 16 类 motif 的共享仿射读出只检验局部统计，不能代替长程关联或完整构型分布。",
            "若保存长程观测，它们仅来自实际 C 构型，未用固定物理读出经过 R 预测，不作为 R 长程泛化成绩。",
            "总误差=读出误差+R 传输误差；MSE 还包含交叉项，三个 RMSE 不能相加，可能因相消而出现小总误差。",
            "仿射解码器原始负值、超一概率和归一化偏差原样保存，不裁剪、不重新归一化掩盖读出问题。",
            "原源模型正式工作流若记录采集失败，其历史状态保持不变；本次探针完成不是修复或覆盖原正式工作流。",
            "未用测试条件拟合常数基线；误差仅作绝对描述，不宣称优于未经检验的基线。",
            "固定半子集划分与训练编码函数相同，不等于对训练随机划分取期望；全体编码只作口径差诊断。",
            "保留原始采样 sidecar 和其有限链/自相关/预热警告；有限样本通过检查不证明平衡或构型独立。",
            "单训练 seed、固定 Q/C、有限链、有限尺寸与表示截断均限制结论，不据本探针搜根或认证临界指数。",
        ])
    return result, arrays, snapshots


def _shade(ax):
    ax.axvspan(2.5, 3.45, color="C3", alpha=.08)
    ax.set_xticks(range(4), ["0", "1", "2", "3（未训练）"])


def plot_report(result, arrays, snapshots, output, font=None):
    import matplotlib.pyplot as plt
    with chinese_plot_context(font) as selected:
        fig, axes = plt.subplots(6, 4, figsize=(15, 20), layout="constrained")
        for group, (size, coupling) in enumerate((size, k) for size in SIZES for k in COUPLINGS):
            row = int(np.flatnonzero((arrays["row_size0"] == size) & (arrays["row_K0"] == coupling) & (arrays["row_replica"] == 0))[0])
            for k in range(4):
                ax = axes[group, k]
                ax.imshow(snapshots[f"row_{row:03d}_scale_{k}"], cmap="gray", vmin=-1, vmax=1, interpolation="nearest", resample=False)
                ax.set(title=f"起始 L={size}，K={coupling:.2f}\nk={k}，L={arrays['scales'][row,k]}" + ("（未训练深度）" if k == 3 else ""))
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("预选物理链0、C重复0、构型0：实际 C 粗粒化，非 R 生成；格点不平滑")
        save_chinese(fig, output / "configurations.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        for i, size in enumerate(SIZES):
            for j, coupling in enumerate(COUPLINGS):
                ax = axes[i, j]
                ids = (arrays["row_size0"] == size) & (arrays["row_K0"] == coupling)
                for key, marker, name, color, offset in (("direct_g", "o", "实际 C 后直接 Q", "C0", -.035), ("R_rollout_g", "x", "微观起点 R 迭代", "C3", .035)):
                    values = arrays[key][ids].mean(1)[..., 0]
                    ax.plot(range(4), values.mean(0), marker + "-", color=color, label=name)
                    for chain, vals in enumerate(values):
                        ax.scatter(np.arange(4) + offset, vals, marker=marker, color=color, alpha=.5, s=22)
                _shade(ax)
                ax.set(title=f"起始 L={size}，K={coupling:.2f}", xlabel="迭代深度 k", ylabel="编码 g（两物理链散点，无置信区间）")
                ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle("先平均每链 5 次 C；实线为两链描述均值，第三步为未训练深度")
        save_chinese(fig, output / "latent_prediction.png", selected); plt.close(fig)

        motif_values = [arrays[key] for key in ("p_observed", "p_decoded_direct", "p_decoded_rollout")]
        low, high = min(float(v.min()) for v in motif_values), max(float(v.max()) for v in motif_values)
        fig, axes = plt.subplots(6, 3, figsize=(17, 16), layout="constrained")
        image = None
        for i, size in enumerate(SIZES):
            for k in (1, 2, 3):
                for col, (key, label) in enumerate((("p_observed", "实际局部频率 p"), ("p_decoded_direct", "直接编码的固定读出 D(g)"), ("p_decoded_rollout", "R 迭代的固定读出 D(R^k(g0))"))):
                    ax = axes[i * 3 + k - 1, col]
                    values = np.stack([arrays[key][(arrays["row_size0"] == size) & (arrays["row_K0"] == coupling), :, k].mean((0, 1)) for coupling in COUPLINGS])
                    image = ax.imshow(values, interpolation="nearest", aspect="auto", vmin=low, vmax=high, cmap="viridis")
                    ax.set(title=f"起始 L={size}，k={k}" + ("（未训练）" if k == 3 else "") + "\n" + label,
                           xlabel="2×2 局部图案类别", ylabel="生成耦合 K（仅评价）", xticks=[0, 3, 7, 11, 15], yticks=range(3), yticklabels=[f"{v:.2f}" for v in COUPLINGS])
        fig.colorbar(image, ax=axes.ravel().tolist(), label="原始读出值：允许负值，不裁剪或归一化")
        fig.suptitle("相同固定共享解码器；各条件两链先平均 C 再等权，不拟合测试标签")
        save_chinese(fig, output / "motif_prediction.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
        for i, size in enumerate(SIZES):
            for j, info in enumerate(result["sizes"][str(size)]["conditions"]):
                ax = axes[i, j]
                for key, label, color in (("total_rmse", "总物理预测", "C0"), ("readout_rmse", "直接编码读出", "C1"), ("R_transport_rmse", "R 传输分量", "C3")):
                    ax.plot(range(4), [row["motif_of_C_means"][key] for row in info["scales"]], "o-", label=label, color=color)
                _shade(ax)
                ax.set(title=f"起始 L={size}，K={info['K0']:.2f}", xlabel="迭代深度 k", ylabel="局部 motif 绝对 RMSE")
                ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle("三个 RMSE 不能相加；MSE 交叉项与逐链值完整保存在报告")
        save_chinese(fig, output / "errors.png", selected); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for i, size in enumerate(SIZES):
            ids = arrays["row_size0"] == size
            g, rg = arrays["direct_g"][ids, ..., 0], arrays["R_one_step_g"][ids, ..., 0]
            levels = np.broadcast_to(np.arange(4), g.shape)
            axes[i, 0].scatter(g.ravel(), rg.ravel(), c=levels.ravel(), cmap="viridis", s=15, alpha=.6)
            lo, hi = min(g.min(), rg.min()), max(g.max(), rg.max())
            axes[i, 0].plot([lo, hi], [lo, hi], "k--", label="恒等映射")
            axes[i, 0].set(title=f"起始 L={size}：逐 C 的真实 R(g)", xlabel="输入 g", ylabel="输出 R(g)")
            axes[i, 0].legend(fontsize=8)
            axes[i, 1].quiver(g.ravel(), levels.ravel(), (rg-g).ravel(), np.zeros(g.size), levels.ravel(), cmap="viridis", angles="xy", scale_units="xy", scale=1, width=.003)
            axes[i, 1].update_datalim(np.column_stack((rg.ravel(), levels.ravel()))); axes[i, 1].autoscale_view(scaley=False)
            axes[i, 1].set(title=f"起始 L={size}：箭头 R(g)-g（不归一化）", xlabel="输入 g", ylabel="输入深度 k", yticks=range(4), ylim=(-.5, 3.5))
            for ax in axes[i]: ax.grid(alpha=.2)
        fig.suptitle("只绘制已保存点，不搜根；k=3 的 R(g) 没有相应下一尺度实测验证")
        save_chinese(fig, output / "R_vs_g.png", selected); plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(17, 5), layout="constrained")
        for k, ax in enumerate(axes):
            for size, color in zip(SIZES, ("C0", "C3")):
                mean = []
                for coupling in COUPLINGS:
                    ids = (arrays["row_size0"] == size) & (arrays["row_K0"] == coupling)
                    values = arrays["direct_g"][ids, :, k, 0].mean(1)
                    mean.append(values.mean()); ax.scatter(np.full(len(values), coupling), values, color=color, alpha=.5)
                ax.plot(COUPLINGS, mean, "o-", color=color, label=f"起始 L={size}，当前 L={size // 3**k}")
            ax.set(title=f"k={k}" + ("（未训练）" if k == 3 else ""), xlabel="生成时 K（仅评价）", ylabel="直接编码 g")
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle("三零扰动点的耦合—编码描述；连线仅引导视线，不插值相变点")
        save_chinese(fig, output / "coupling_by_scale.png", selected); plt.close(fig)

        summary = result.get("source_training_reference")
        if summary and summary.get("history"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
            term_labels = dict(separation="仿射重构", anchor="微观子集一致性", coarse_consistency="粗尺度子集一致性",
                               rsmi="负经验互信息", kernel_noncollapse="方差下限", kernel_sign="均匀块颜色保持",
                               commutator="单步闭合", rollout="多步闭合")
            for ax, terms, title in zip(axes, (("separation", "anchor", "coarse_consistency"), ("rsmi", "kernel_noncollapse", "kernel_sign"), ("commutator", "rollout")), ("Q 表示分项", "C 原目标分项", "R 闭合分项")):
                for term in terms:
                    for split, style, label in (("train", "-", "训练"), ("test", "--", "测试")):
                        rows = [row for row in summary["history"] if split + "_" + term in row]
                        if rows: ax.plot([row["epoch"] for row in rows], [row[split + "_" + term] for row in rows], style, label=term_labels[term] + "·" + label)
                ax.set(title=title, xlabel="源模型实际训练步数", ylabel="历史记录值（未重新训练）", yscale="symlog")
                ax.grid(alpha=.2)
                if ax.lines: ax.legend(fontsize=7)
            fig.suptitle("源模型历史 loss 参考；本次冻结探针零参数更新，不产生新训练曲线")
            save_chinese(fig, output / "loss_reference.png", selected); plt.close(fig)


def report_run(root, output=None, *, make_plots=True, font=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "report"
    if output.exists():
        raise FileExistsError("只创建新的报告目录，不覆盖已有报告")
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("报告必须保存在本次新输出根内")
    result, arrays, snapshots = build_report(root)
    output.mkdir(parents=True, exist_ok=False)
    if make_plots:
        plot_report(result, arrays, snapshots, output, font)
    figures = list(FIGURES) if make_plots else []
    if make_plots and (result.get("source_training_reference") or {}).get("history"):
        figures.append("loss_reference.png")
    result["public_files"] = figures + ["report.json", "status.json"]
    result["arrays_reference"] = "collection/arrays.npz"
    result["snapshots_reference"] = "collection/snapshots.npz"
    result["large_snapshots_copied"] = False
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    status = dict(status="complete", workflow_complete=True, report_sha256=public_sha(output / "report.json"),
                  training_updates=0, public_files=result["public_files"])
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report_run(args.root, args.output, make_plots=not args.no_plots, font=args.font)


if __name__ == "__main__":
    main()
