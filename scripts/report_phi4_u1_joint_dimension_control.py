"""Stored-array report for fresh trainable-C u=1 QCR, dg=1 versus dg=2.

No model, checkpoint, physical field or new root search is used.  The two
unchanged training CLIs do not promise paired Q initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.report_phi4_scalar_r_validation import sha256, write_json


FILES = ("manifest.json", "summary.json", "status.json", "history_progress.json", "evaluation_arrays.npz")
PATHS = (("0_to_1", 0, 1, "one_step_feature_prediction", 0),
         ("1_to_2", 1, 2, "one_step_feature_prediction", 1),
         ("0_to_2", 0, 2, "rollout_feature_prediction", 2))
LABELS = ("36→12", "12→4", "36→4")
CPU_BUDGET = 300.
SIZE_BUDGET = 128*1024**2


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_run(root, dim, pins):
    root = Path(root).resolve()
    for name in FILES:
        pins[str(root/name)] = sha256(root/name)
    manifest, summary, status, history = [json.loads((root/name).read_text()) for name in FILES[:4]]
    options = manifest.get("options", {})
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action") is not False
            or status.get("summary_sha256") != pins[str(root/"summary.json")]
            or status.get("evaluation_arrays_sha256") != pins[str(root/"evaluation_arrays.npz")]
            or status.get("actual_steps") != 3000 or summary.get("actual_steps") != 3000
            or manifest.get("theory_dim") != dim or summary.get("theory_dim") != dim
            or manifest.get("lattice_sizes") != [36, 12, 4] or manifest.get("block_size") != 3
            or manifest.get("stride") != 3 or options.get("steps") != 3000 or options.get("seed") != 91
            or manifest.get("q_initialization") != "train_mean_bias"
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training") is not False
            or summary.get("frozen_evaluation_verified") is not True):
        raise ValueError("需要成功完成的3000步、seed91、b3、无标签完整QCR端点")
    for name in ("Q", "C", "R"):
        clocks = summary.get("optimizer_steps", {}).get(name, {})
        if clocks.get("min") != 3000 or clocks.get("max") != 3000 or clocks.get("parameters", 0) < 1:
            raise ValueError("所有Q/C/R Adam参数时钟必须为3000")
    receipt = manifest.get("q_initialization_receipt", {})
    initial = manifest.get("training_start_module_sha256", {}).get("C")
    constructor = manifest.get("random_construction_module_sha256", {}).get("C")
    final = summary.get("final_module_sha256", {}).get("C")
    if (not all(_digest(value) for value in (initial, constructor, final))
            or receipt.get("constructor_module_sha256", {}).get("C") != constructor
            or receipt.get("training_start_module_sha256", {}).get("C") != initial
            or initial != constructor):
        raise ValueError("随机C初始化与训练开始指纹回执不一致")
    sources = manifest.get("source_sha256", {})
    if not sources or any(not str(path).endswith(".py") or not _digest(digest) for path, digest in sources.items()):
        raise ValueError("需要冻结训练源代码指纹")
    for path, digest in sources.items():
        if sha256(path) != digest:
            raise ValueError("训练源代码指纹改变")
        if path in pins and pins[path] != digest:
            raise ValueError("训练源代码指纹冲突")
        pins[path] = digest
    rows = history.get("history", [])
    if not rows or rows[-1].get("step") != 3000 or history.get("latest_step") != 3000:
        raise ValueError("完整训练历史必须到达3000步")
    with np.load(root/"evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for split, count in (("train", 32), ("test", 20)):
        for name, shape in (("direct", (count, 3, dim)), ("feature_target", (count, 3, 105)),
                ("feature_reconstruction", (count, 3, 105)), ("one_step_feature_prediction", (count, 2, 105)),
                ("rollout_feature_prediction", (count, 3, 105)), ("labels", (count, 3)), ("cell_ids", (count,))):
            value = arrays[f"{split}_{name}"]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"公开评价数组形状或数值无效：{split}/{name}")
        if not np.allclose(arrays[f"{split}_labels"][:, 1], 1., rtol=0, atol=1e-12):
            raise ValueError("需要固定u=1参数族")
    if set(arrays["train_cell_ids"]) & set(arrays["test_cell_ids"]):
        raise ValueError("TRAIN与TEST参数条件重叠")
    g, mapped, drift = [arrays[name] for name in ("flow_g", "flow_R", "flow_displacement")]
    if (g.ndim != 2 or g.shape[1] != dim or g.shape != mapped.shape or g.shape != drift.shape
            or any(not np.isfinite(value).all() for value in (g, mapped, drift))
            or not np.allclose(mapped-g, drift, rtol=1e-10, atol=1e-12)):
        raise ValueError("缓存流场与R(g)−g不一致")
    return dict(manifest=manifest, summary=summary, history=history, arrays=arrays,
                C_constructor=constructor, C_initial=initial, C_final=final)


def target_normalizers(target):
    target = np.asarray(target, dtype=np.float64)
    mean = target.mean(0)
    scales = np.sqrt(np.mean((target-mean)**2, axis=(0, 2)))
    if target.ndim != 3 or target.shape[1:] != (3, 105) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("TRAIN共同特征目标必须有非零变化")
    return scales, mean


def error_metric(residual, scale):
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 105 or not np.isfinite(value).all():
        raise ValueError("需要物理链×105特征的有限残差")
    rmse = float(np.sqrt(np.mean(value**2)))
    return dict(rmse=rmse, nrmse=rmse/float(scale), train_target_mean_baseline_rmse=float(scale),
        coordinate_rmse=np.sqrt(np.mean(value**2, axis=0)).tolist(), physical_chains=len(value),
        evaluation_encodings_per_physical_chain=1)


def feature_metrics(arrays):
    scales, mean = target_normalizers(arrays["train_feature_target"])
    result = {}
    for split in ("train", "test"):
        target = arrays[f"{split}_feature_target"]
        result[split] = []
        for path, start, end, kind, index in PATHS:
            residuals = dict(prediction=arrays[f"{split}_{kind}"][:, index]-target[:, end],
                target_Q_reconstruction=arrays[f"{split}_feature_reconstruction"][:, end]-target[:, end],
                identity_feature=target[:, start]-target[:, end],
                identity_decoded=arrays[f"{split}_feature_reconstruction"][:, start]-target[:, end],
                TRAIN_mean=mean[end]-target[:, end])
            result[split].append(dict(path=path, target_scale=end, **{
                name: error_metric(value, scales[end]) for name, value in residuals.items()}))
    return result, scales.tolist()


def task_comparison(runs):
    one, two = runs[1], runs[2]
    m1, m2 = one["manifest"], two["manifest"]
    if (m1.get("data_manifest_sha256") != m2.get("data_manifest_sha256")
            or m1.get("data_shards") != m2.get("data_shards")
            or one["summary"].get("data_manifest_sha256") != m1.get("data_manifest_sha256")
            or two["summary"].get("data_manifest_sha256") != m2.get("data_manifest_sha256")
            or m1.get("evaluation_seed") != m2.get("evaluation_seed")):
        raise ValueError("两个维度必须使用同一物理数据、分片及评价随机种子")
    targets = {}
    for split in ("train", "test"):
        for name in ("labels", "cell_ids"):
            if not np.array_equal(one["arrays"][f"{split}_{name}"], two["arrays"][f"{split}_{name}"]):
                raise ValueError("两个维度的物理链、参数及划分必须严格相同")
        a, b = (run["arrays"][f"{split}_feature_target"] for run in (one, two))
        targets[split] = dict(exactly_equal=bool(np.array_equal(a, b)),
            max_abs_difference=float(np.max(np.abs(a-b))),
            max_abs_difference_by_scale=np.max(np.abs(a-b), axis=(0, 2)).tolist(),
            within_absolute_1e12=bool(np.allclose(a, b, rtol=0, atol=1e-12)))
    constructors = one["C_constructor"] == two["C_constructor"]
    initial = one["C_initial"] == two["C_initial"]
    final = one["C_final"] == two["C_final"]
    identical = constructors and initial and final and all(row["exactly_equal"] for row in targets.values())
    return dict(C_constructor_SHA_equal=constructors, C_training_start_SHA_equal=initial,
        C_final_SHA_equal=final, feature_targets=targets, identical_coarse_graining_tasks=bool(identical),
        directly_comparable_common_target_errors=bool(identical),
        qualification=("C初末指纹与逐链特征目标完全一致，可比较同一特征目标的预测误差；Q初始化不严格配对。" if identical else
            "两组粗粒化任务未证实完全相同；分别报告各自误差，不能把差异单独归因于dg。"),
        Q_initialization_strictly_paired=False, same_QCR_seed=91, independent_QCR_seeds=1)


def draw(output, runs, metrics, comparison):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    colors = {1: "#3e7e9d", 2: "#b86b55"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), constrained_layout=True)
    for row, split in enumerate(("train", "test")):
        for index, ax in enumerate(axes[row]):
            for dim, x in ((1, 0.), (2, 1.)):
                value = metrics[str(dim)][split][index]
                ax.scatter(x, value["prediction"]["rmse"], color=colors[dim], s=45,
                           label="R预测后解码" if dim == 1 else None)
                for kind, marker, color, label in (("target_Q_reconstruction", "_", "#6c996c", "目标Q重构"),
                    ("identity_feature", "x", ".45", "特征恒等"), ("TRAIN_mean", "+", "#9678ac", "TRAIN均值")):
                    ax.scatter(x, value[kind]["rmse"], color=color, marker=marker, s=75, label=label if dim == 1 else None)
            ax.set(xticks=[0, 1], xticklabels=["$d_g=1$", "$d_g=2$"],
                title=f"{'TRAIN' if split == 'train' else '原已查看TEST'}：{LABELS[index]}", ylabel="105维特征RMSE")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("随机可训练C的完整QCR：共同目标一致" if comparison["identical_coarse_graining_tasks"] else
                 "随机可训练C的完整QCR：两组目标未严格相同，误差按各自任务解释")
    fig.savefig(output/"feature_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    a = runs[1]["arrays"]
    train = a["train_direct"][:, :2].reshape(-1)
    low, high = float(train.min()), float(train.max())
    pad = .1*max(high-low, 1e-6)
    keep = (a["flow_g"][:, 0] >= low-pad) & (a["flow_g"][:, 0] <= high+pad)
    g, mapped, drift = [a[name][keep, 0] for name in ("flow_g", "flow_R", "flow_displacement")]
    axes[0].plot(g, mapped, color=colors[1], label="缓存R(g)")
    axes[0].plot(g, g, ":", color=".45", label="恒等映射")
    axes[1].plot(g, drift, color=colors[1])
    axes[1].axhline(0, color=".5", lw=.8)
    selected = np.arange(0, len(g), max(1, len(g)//20))
    axes[1].quiver(g[selected], np.zeros(len(selected)), np.zeros(len(selected)), drift[selected],
        angles="xy", scale_units="xy", scale=1, color=colors[1], width=.003)
    for root in runs[1]["summary"].get("numerical_fixed_points", []):
        if low-pad <= root["g"] <= high+pad:
            axes[0].scatter(root["g"], root["g"], color=".2", s=30, marker="x")
    a = runs[2]["arrays"]
    axes[2].quiver(a["flow_g"][:, 0], a["flow_g"][:, 1], a["flow_displacement"][:, 0],
        a["flow_displacement"][:, 1], angles="xy", scale_units="xy", scale=1, width=.0025, color=colors[2], alpha=.65)
    for k, color in ((0, "#659873"), (1, "#8a74a1")):
        for ax in axes[:2]:
            values = runs[1]["arrays"]["train_direct"][:, k, 0]
            ax.scatter(values, np.full_like(values, .025+.03*k), transform=ax.get_xaxis_transform(),
                s=8, color=color, alpha=.5, label=f"TRAIN输入k={k}")
        values = a["train_direct"][:, k]
        axes[2].scatter(values[:, 0], values[:, 1], s=10, color=color, alpha=.5, label=f"TRAIN输入k={k}")
    for root in runs[2]["summary"].get("numerical_fixed_points", []):
        axes[2].scatter(*root["g"], marker="x", color=".2", s=35)
    axes[0].set(title="$d_g=1$：缓存映射", xlabel="潜变量g", ylabel="R(g)", xlim=[low-pad, high+pad])
    axes[1].set(title="$d_g=1$：真实位移", xlabel="潜变量g", ylabel="R(g)−g", xlim=[low-pad, high+pad])
    axes[2].set(title="$d_g=2$：真实位移箭头", xlabel="$g_1$", ylabel="$g_2$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    axes[2].legend(fontsize=7, frameon=False)
    fig.suptitle("仅转绘缓存：叉号为原运行数值根；一维显示裁至TRAIN区间，原搜索区间含TEST")
    fig.savefig(output/"flow.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9), constrained_layout=True)
    for row, (dim, coordinate) in enumerate(((1, 0), (2, 0), (2, 1))):
        a = runs[dim]["arrays"]
        for k, ax in enumerate(axes[row]):
            for split, marker, color, label in (("train", "o", colors[1], "TRAIN"), ("test", "^", colors[2], "原已查看TEST")):
                ids, labels, direct = [a[f"{split}_{name}"] for name in ("cell_ids", "labels", "direct")]
                points = np.array(sorted((float(labels[ids == cell, 0][0]), float(direct[ids == cell, k, coordinate].mean()))
                                         for cell in np.unique(ids)))
                ax.plot(points[:, 0], points[:, 1], marker=marker, color=color, lw=.8, ms=4, label=label)
            ax.set(title=f"$d_g={dim}$，坐标{coordinate+1}，k={k}", xlabel="裸参数r（仅事后评价）", ylabel="潜变量g")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("固定u=1：每条件四条物理链的直接编码均值；不同维度的g坐标不等同")
    fig.savefig(output/"coupling_by_scale.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, kind, label in zip(axes, ("q_loss", "c_loss", "r_loss"), ("Q表示目标", "C粗粒化目标", "R闭合目标")):
        for dim in (1, 2):
            rows = runs[dim]["history"]["history"]
            y = [row[kind] if kind == "c_loss" else max(row[kind], 1e-18) for row in rows]
            ax.plot([row["step"] for row in rows], y, color=colors[dim], label=f"$d_g={dim}$")
        ax.set(title=label, xlabel="完整QCR的Adam更新数", ylabel="原训练目标")
        if kind != "c_loss":
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("每组Q/C/R均为3000步；C目标含信息奖励，可为负；潜变量R目标不跨维度直接判优")
    fig.savefig(output/"loss.png", dpi=160)
    plt.close(fig)


def run(root, output):
    started = time.process_time()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or output in root.parents or any(child in output.parents for child in (root/"dg1", root/"dg2")):
        raise FileExistsError("联合对照报告必须写入独立新目录")
    pins = {}
    runs = {dim: load_run(root/f"dg{dim}", dim, pins) for dim in (1, 2)}
    plan_path = root/"plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        pins[str(plan_path)] = sha256(plan_path)
        if plan.get("data_manifest_sha256") not in (None, runs[1]["manifest"]["data_manifest_sha256"]):
            raise ValueError("父计划物理数据指纹不一致")
    from scripts import report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation
    for module in (report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation):
        pins[str(Path(module.__file__).resolve())] = sha256(module.__file__)
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    comparison = task_comparison(runs)
    metrics, normalizers = {}, {}
    for dim in (1, 2):
        metrics[str(dim)], normalizers[str(dim)] = feature_metrics(runs[dim]["arrays"])
    output.mkdir(parents=True, exist_ok=False)
    draw(output, runs, metrics, comparison)
    if time.process_time()-started > CPU_BUDGET or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > SIZE_BUDGET:
        raise RuntimeError("联合报告CPU300秒或输出128MiB预算已用尽")
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise RuntimeError("报告期间不可变公开输入或训练源代码改变")
    result = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        fixed_u=1., physical_dimension=2, theory_dims=[1, 2], QCR_seed=91, actual_QCR_updates=3000,
        fresh_random_trainable_C=True, Q_initialization_strictly_paired=False,
        task_comparison=comparison, feature_metrics=metrics, target_TRAIN_normalizers=normalizers,
        normalization="各尺度TRAIN105维目标相对该尺度TRAIN均值向量的逐坐标RMSE；若C任务不同则各组单独解释",
        C_fingerprints={str(dim): dict(constructor=runs[dim]["C_constructor"], training_start=runs[dim]["C_initial"],
            final=runs[dim]["C_final"]) for dim in (1, 2)},
        numerical_fixed_points={str(dim): runs[dim]["summary"].get("numerical_fixed_points", []) for dim in (1, 2)},
        original_flow_analysis={str(dim): runs[dim]["summary"].get("flow_analysis") for dim in (1, 2)},
        original_flow_bounds_scope={"1": "原scalar_flow由TRAIN和TEST输入共同确定范围；本报告仅裁剪显示TRAIN输入范围，不改变原根搜索证据",
                                    "2": "原bounded_flow仅使用TRAIN输入尺度0/1确定边界、凸包及搜索起点"},
        input_sha256=pins, stored_arrays_only=True, Q_C_R_inference_performed=False,
        checkpoint_loaded=False, physical_fields_loaded=False, new_root_search=False,
        new_physical_samples=0, new_training_updates=0, CPU_threads=1, GPU_seconds=0,
        cpu_seconds=time.process_time()-started, cpu_budget_seconds=CPU_BUDGET,
        output_budget_bytes=SIZE_BUDGET, thermal_nu=None, critical_r=None, physical_fixed_point_certified=False,
        limitations=["两组是同一seed91的完整QCR控制，不是三个独立QCR种子；先前三个Q种子属于冻结C阶段。",
            "未改变原训练CLI；Q的输出层与decoder随机初始化不严格配对，不把差异视为仅增加一个坐标的严格因果效应。",
            "共同物理链和评价随机种子不保证C轨迹数值相同，需同时检查C初末SHA和逐链特征目标差异。",
            "每条链只有一次全128场评价编码，不是四次C重复；同条件四条链也不是四个参数条件。",
            "旧TEST已查看，结果属于探索性评价，不按TEST或根数挑模型。",
            "一维原搜索范围含TEST，二维原搜索范围仅TRAIN；二者数值根数量不作为维度优劣指标。",
            "本报告完整转录已有数值根，不重搜根、不挑热模、不认证物理临界点或临界指数。",
            "原物理采样质量警告继续保留，未进行新采样或平衡认证。"])
    write_json(output/"analysis.json", result)
    write_json(output/"status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, analysis_sha256=sha256(output/"analysis.json"), cpu_seconds=time.process_time()-started))
    write_json(output/"public_files.json", {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description="固定u=1的新完整QCR维度对照公共数组报告")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps(dict(status=result["status"], cpu_seconds=result["cpu_seconds"])))


if __name__ == "__main__":
    main()
