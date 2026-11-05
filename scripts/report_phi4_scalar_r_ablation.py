"""Public-array report for the four-arm, condition-fold scalar R diagnostic.

Only frozen R parameters are evaluated.  The separate-transition arm is a
diagnostic of closure and is never interpreted as a single autonomous RG map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.report_phi4_scalar_r_validation import (
    FrozenScalarMap, find_roots, sha256, state_fingerprint, verified_public,
    write_json,
)


SEEDS = (91, 92, 93)
FOLDS = (0, 1, 2, 3)
ARMS = {
    "A_shared96": dict(kind="shared", hidden=96, heads=1),
    "B_separate96": dict(kind="separate", hidden=96, heads=2),
    "C_shared136": dict(kind="shared", hidden=136, heads=1),
    "D_shared16": dict(kind="shared", hidden=16, heads=1),
}
ARM_LABELS = {"A_shared96": "A：共享 R，宽度96", "B_separate96": "B：分尺度 R，宽度96",
              "C_shared136": "C：共享 R，宽度136", "D_shared16": "D：共享 R，宽度16"}
COLORS = ("#3e7b99", "#b66853", "#7d6ba4")
PATHS = (("0_to_1", 1, "one_step", 0), ("1_to_2", 2, "one_step", 1),
         ("0_to_2", 2, "rollout", 2))
PATH_LABELS = ("36→12", "12→4", "36→4")
CPU_BUDGET_SECONDS = 300.
OUTPUT_BUDGET_BYTES = 128 * 1024**2


class FrozenClosureMaps:
    """One shared or two transition-specific immutable public NumPy heads."""

    def __init__(self, state, expected_sha256, *, heads, hidden):
        self.state = {name: np.array(value, copy=True) for name, value in state.items()}
        if state_fingerprint(self.state) != expected_sha256:
            raise ValueError("四臂 R 参数整体指纹不一致")
        self.expected_sha256 = expected_sha256
        self.heads = []
        for index in range(heads):
            prefix = f"heads.{index}."
            head_state = {name[len(prefix):]: value for name, value in self.state.items()
                          if name.startswith(prefix)}
            head = FrozenScalarMap(head_state, state_fingerprint(head_state))
            if head.state["net.0.weight"].shape[0] != hidden:
                raise ValueError("R 隐藏宽度与四臂预定设置不符")
            self.heads.append(head)
        expected_names = {f"heads.{index}.{name}" for index, head in enumerate(self.heads)
                          for name in head.state}
        if set(self.state) != expected_names:
            raise ValueError("R 头部数目或参数名称不完整")
        for value in self.state.values():
            value.setflags(write=False)

    @classmethod
    def restore(cls, archive, key, expected_sha256, spec):
        prefix = key + "."
        state = {name[len(prefix):]: archive[name].copy() for name in archive.files
                 if name.startswith(prefix)}
        return cls(state, expected_sha256, heads=spec["heads"], hidden=spec["hidden"])

    def predict(self, direct):
        direct = np.asarray(direct, dtype=np.float64)
        if direct.ndim != 4 or direct.shape[-2:] != (3, 1) or not np.isfinite(direct).all():
            raise ValueError("需要有限的 [编码重复,物理链,3,1] 直接编码")
        first, _ = self.heads[0].forward_and_derivative(direct[..., 0, :].reshape(-1, 1))
        second_head = self.heads[-1]
        second, _ = second_head.forward_and_derivative(direct[..., 1, :].reshape(-1, 1))
        composition, _ = second_head.forward_and_derivative(first)
        shape = direct.shape[:-2] + (1,)
        first, second, composition = (value.reshape(shape) for value in (first, second, composition))
        return dict(one_step=np.stack((first, second), -2),
                    rollout=np.stack((direct[..., 0, :], first, composition), -2))

    def assert_unchanged(self):
        if (state_fingerprint(self.state) != self.expected_sha256
                or any(value.flags.writeable for value in self.state.values())):
            raise RuntimeError("报告期间 R 参数发生变化")
        for head in self.heads:
            head.assert_unchanged()


def model_key(fold, arm, seed, endpoint):
    return f"fold{fold}_{arm}_seed{seed}_{endpoint}"


def check_predictions(model, arrays, key):
    difference = 0.
    for split in ("train", "test"):
        predicted = model.predict(arrays[f"{split}_direct"])
        for kind in ("one_step", "rollout"):
            actual, expected = predicted[kind], arrays[f"eval_{key}_{split}_{kind}"]
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise ValueError(f"NumPy R 转移／复合预测与缓存不一致：{key}/{split}/{kind}")
            difference = max(difference, float(np.max(np.abs(actual - expected))))
    return difference


def metric_rows(direct, prediction, target_stds):
    """Same original-coordinate errors as training, with repeat-aware scores."""
    rows, closure = [], 0.
    for (path, target, kind, index), weight in zip(PATHS, (5., 2.5, 2.5)):
        residual = np.asarray(prediction[kind])[..., index, :] - direct[..., target, :]
        rmse = float(np.sqrt(np.mean(residual**2)))
        chain_rmse = float(np.sqrt(np.mean(residual.mean(0)**2)))
        std = float(target_stds[target])
        rows.append(dict(path=path, rmse=rmse, nrmse=rmse/std,
            chain_mean_rmse=chain_rmse, chain_mean_nrmse=chain_rmse/std,
            physical_chains=int(direct.shape[1]), encoding_repeats=int(direct.shape[0]),
            train_target_std=std))
        closure += weight * chain_rmse**2
    return rows, closure


def checked_metrics(arrays, summary, splits, target_stds):
    recomputed, closures, difference = {}, {}, 0.
    for key in summary["final_R_sha256"]:
        fold = int(key.split("_", 1)[0][4:])
        recomputed[key], closures[key] = {}, {}
        for name in ("train", "fit", "validation", "test"):
            source = "test" if name == "test" else "train"
            mask = (np.asarray(splits[fold][f"{name}_mask"], dtype=bool)
                    if name in ("fit", "validation") else slice(None))
            direct = arrays[f"{source}_direct"][:, mask]
            prediction = {kind: arrays[f"eval_{key}_{source}_{kind}"][:, mask]
                          for kind in ("one_step", "rollout")}
            rows, score = metric_rows(direct, prediction, target_stds)
            saved = summary["latent_metrics"][key][name]
            if len(saved) != 3 or [row["path"] for row in saved] != [row["path"] for row in rows]:
                raise ValueError("训练保存的闭合路径不完整")
            for actual, expected in zip(rows, saved):
                for field in ("rmse", "nrmse", "chain_mean_rmse", "chain_mean_nrmse"):
                    if not np.isclose(actual[field], expected[field], rtol=1e-10, atol=1e-12):
                        raise ValueError(f"公开预测与已保存指标不一致：{key}/{name}/{field}")
                    difference = max(difference, abs(actual[field]-expected[field]))
            recomputed[key][name], closures[key][name] = rows, score
    return recomputed, closures, float(difference)


def roughness(head, bounds):
    grid = np.linspace(*bounds, 1025)[:, None]
    mapped, derivative = head.forward_and_derivative(grid)
    width = float(bounds[1]-bounds[0])
    # d²(R/width)/d(g/width)² = width * d²R/dg².  This is a
    # dimensionless descriptive roughness statistic, not a regularizer.
    curvature = width * np.diff(derivative[:, 0, 0]) / np.diff(grid[:, 0])
    h = width*1e-6
    probe = grid[::64]
    left, _ = head.forward_and_derivative(probe-h)
    right, _ = head.forward_and_derivative(probe+h)
    fd = (right-left)/(2*h)
    analytic = derivative[::64, :, 0]
    if not np.allclose(fd, analytic, rtol=2e-4, atol=1e-5):
        raise ValueError("完整 DR 的中心差分核对失败")
    return dict(grid_points=len(grid), full_DR_includes_identity=True,
        abs_full_DR_p95=float(np.quantile(np.abs(derivative), .95)),
        abs_full_DR_max=float(np.max(np.abs(derivative))),
        normalized_curvature_abs_p95=float(np.quantile(np.abs(curvature), .95)),
        normalized_curvature_rms=float(np.sqrt(np.mean(curvature**2))),
        total_variation_full_DR=float(np.sum(np.abs(np.diff(derivative[:, 0, 0])))),
        full_DR_finite_difference_max_abs_difference=float(np.max(np.abs(fd-analytic))))


def diagnose_maps(models, bounds, guard):
    shape, roots, searches = {}, {}, {}
    for key, model in models.items():
        guard()
        heads = [roughness(head, bounds) for head in model.heads]
        row = dict(heads=heads, autonomous_map=len(heads) == 1)
        if len(heads) == 2:
            grid = np.linspace(*bounds, 1025)[:, None]
            first, _ = model.heads[0].forward_and_derivative(grid)
            second, _ = model.heads[1].forward_and_derivative(grid)
            distance = (first-second)/(bounds[1]-bounds[0])
            row["normalized_head_separation_rms"] = float(np.sqrt(np.mean(distance**2)))
            row["normalized_head_separation_max"] = float(np.max(np.abs(distance)))
            searches[key] = dict(performed=False,
                reason="分尺度 R0、R1 仅用于闭合诊断，不定义共享自主映射，不搜索其物理不动点")
        elif key.endswith("_selected"):
            roots[key], searches[key] = find_roots(model.heads[0], bounds, grid_points=1025)
            searches[key]["performed"] = True
        else:
            searches[key] = dict(performed=False,
                reason="Adam6000端点仅作预定收敛对照；本报告搜索共享映射的验证选定端点")
        shape[key] = row
    return shape, roots, searches


def paired_comparisons(metrics, closures):
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for endpoint in ("selected", "adam6000"):
                keys = {arm: model_key(fold, arm, seed, endpoint) for arm in ARMS}
                for first, second, question in (("B_separate96", "C_shared136", "分尺度与近似参数量匹配共享映射"),
                                                ("D_shared16", "A_shared96", "小容量与原容量共享映射")):
                    for split in ("validation", "test"):
                        a, b = metrics[keys[first]][split], metrics[keys[second]][split]
                        ratios = [x["chain_mean_rmse"]/y["chain_mean_rmse"]
                                  if y["chain_mean_rmse"] > 0 else None for x, y in zip(a, b)]
                        raw_ratios = [x["rmse"]/y["rmse"] if y["rmse"] > 0 else None for x, y in zip(a, b)]
                        denominator = closures[keys[second]][split]
                        rows.append(dict(fold=fold, seed=seed, endpoint=endpoint, split=split,
                            first=first, second=second, question=question,
                            chain_mean_rmse_ratios=ratios, raw_repeat_rmse_ratios=raw_ratios,
                            weighted_chain_mean_loss_ratio=closures[keys[first]][split]/denominator
                                if denominator > 0 else None,
                            first_better_on_all_three_chain_mean_paths=all(value is not None and value < 1 for value in ratios)))
    consistency = []
    for first, second in (("B_separate96", "C_shared136"), ("D_shared16", "A_shared96")):
        subset = [row for row in rows if row["first"] == first and row["second"] == second
                  and row["endpoint"] == "selected" and row["split"] == "validation"]
        consistency.append(dict(first=first, second=second, fold_seed_pairs=len(subset),
            all_three_paths_improved=sum(row["first_better_on_all_three_chain_mean_paths"] for row in subset),
            weighted_loss_improved=sum(row["weighted_chain_mean_loss_ratio"] is not None
                and row["weighted_chain_mean_loss_ratio"] < 1 for row in subset),
            use="描述跨折与 R 种子的配对一致性；选点验证存在乐观偏差，不作显著性或物理闭合认证"))
    return dict(pairs=rows, validation_consistency=consistency,
                automatic_Q_dimension_or_sampling_decision=False)


def _plot(output, arrays, cache_arrays, records, models, metrics, roots, bounds, splits, guard):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    grid = np.linspace(*bounds, 512)[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, name, split, index, target in ((axes[0], "36→12", "train", 0, 1),
                                           (axes[1], "12→4", "train", 1, 2)):
        direct = arrays[f"{split}_direct"]
        labels = arrays[f"{split}_labels"]
        ax.scatter(direct[..., index, 0].ravel(), direct[..., target, 0].ravel(),
            c=np.tile(labels[:, 0], direct.shape[0]), cmap="viridis", s=12, alpha=.55)
        ax.plot(grid[:, 0], grid[:, 0], ":", color=".45", lw=1)
        ax.set(title=f"实际 TRAIN 编码配对：{name}", xlabel="输入 $g$", ylabel="下一尺度直接编码 $g$")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("颜色表示事后裸参数 r；四次编码复用同一物理链，不是四倍独立物理样本")
    fig.savefig(output / "transition_pairs.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    for row, split in enumerate(("validation", "test")):
        for path_index, ax in enumerate(axes[row]):
            for endpoint, offset, color, label in (("selected", -.13, "#b66853", "验证选定端点"),
                                                    ("adam6000", .13, "#3e7b99", "Adam6000端点")):
                for index, arm in enumerate(ARMS):
                    values = np.array([metrics[model_key(fold, arm, seed, endpoint)][split][path_index]["nrmse"]
                                       for fold in FOLDS for seed in SEEDS])*100
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    x = index+offset
                    jitter = np.linspace(-.05, .05, len(values))
                    ax.scatter(x+jitter, values, s=10, color=color, alpha=.35)
                    ax.errorbar(x, median, yerr=[[median-q25], [q75-median]], fmt="o", color=color,
                                capsize=3, label=label if index == 0 else None)
            ax.set(xticks=np.arange(4), xticklabels=["A共享96", "B分尺度96", "C共享136", "D共享16"],
                title=f"{'选点验证条件' if split == 'validation' else '原已查看 TEST'}：{PATH_LABELS[path_index]}",
                ylabel="原 TRAIN 标准差归一化误差（%）")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("四折 × 三个 R 种子：散点为配对结果，圆点和线段为中位数及四分位范围；不是置信区间")
    fig.savefig(output / "closure_comparison.png", dpi=170)
    plt.close(fig)
    guard()

    for displacement in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True, sharey=True, constrained_layout=True)
        for ax, arm in zip(axes.flat, ARMS):
            ax.plot(grid[:, 0], np.zeros(len(grid)) if displacement else grid[:, 0], ":", color=".4", lw=1,
                    label="零位移" if displacement else "恒等映射")
            for seed, color in zip(SEEDS, COLORS):
                key = model_key(FOLDS[0], arm, seed, "selected")
                for index, head in enumerate(models[key].heads):
                    mapped, _ = head.forward_and_derivative(grid)
                    y = mapped[:, 0]-grid[:, 0] if displacement else mapped[:, 0]
                    label = f"R种子{seed}" + (f"，{'36→12' if index == 0 else '12→4'}" if len(models[key].heads) == 2 else "")
                    ax.plot(grid[:, 0], y, "-" if index == 0 else "--", color=color, lw=1.1, label=label)
                    if displacement:
                        # Arrows show the actual unnormalized R(g)-g values.
                        g = grid[::28, 0]
                        delta = y[::28]
                        ax.quiver(g, np.zeros_like(g), np.zeros_like(g), delta, angles="xy",
                                  scale_units="xy", scale=1, color=color, alpha=.45, width=.003)
                if not displacement:
                    for root in roots.get(key, []):
                        ax.scatter(root["g"], root["g"], color=color, s=25,
                            marker="o" if root["classification"] == "attracting" else "^")
            for k, color in ((0, "#558c67"), (1, "#9477a8")):
                points = arrays["train_direct"][..., k, 0].ravel()
                ax.scatter(points, np.full_like(points, .025+.025*k), s=3, color=color, alpha=.25,
                           transform=ax.get_xaxis_transform())
            ax.set(title=ARM_LABELS[arm], xlabel="潜变量 $g$", xlim=bounds,
                   ylabel="$R(g)-g$" if displacement else "$R(g)$")
            ax.legend(fontsize=6.8, frameon=False, ncol=2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"代表性第{FOLDS[0]+1}折：验证选定端点；B 的两条转移仅作闭合诊断" +
                     ("；箭头为真实位移" if displacement else "；共享映射圆为吸引点、三角为其他数值根"))
        fig.savefig(output / ("R_displacement.png" if displacement else "R_vs_g.png"), dpi=170)
        plt.close(fig)
        guard()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7), constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS):
        for seed, color in zip(SEEDS, COLORS):
            base = model_key(FOLDS[0], arm, seed, "selected").removesuffix("_selected")
            for stage, linestyle, offset in (("adam", "-", 0), ("lbfgs", "--", 6000)):
                rows = records[base]["history"][stage]
                x = [offset+row["step"] for row in rows]
                fit = [max(row["fit_loss"], 1e-18) for row in rows]
                val = [max(row["validation"]["chain_mean_loss"], 1e-18) for row in rows]
                ax.plot(x, fit, linestyle, color=color, lw=.9, alpha=.45)
                ax.plot(x, val, linestyle, color=color, lw=1.4,
                        label=f"R种子{seed}验证" if stage == "adam" else None)
        ax.axvline(6000, color=".7", lw=.6)
        ax.set(title=ARM_LABELS[arm], xlabel="Adam更新数＋L-BFGS累计迭代数",
               ylabel="原 g 坐标闭合目标", yscale="log")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("第1折收敛曲线：浅色为拟合，深色为选点验证；虚线为 L-BFGS")
    fig.savefig(output / "loss.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for k, ax in enumerate(axes):
        for split, label, color, marker in (("train", "原 TRAIN 条件", "#3e7b99", "o"),
                                           ("test", "原已查看 TEST 条件", "#b66853", "^") ):
            direct = arrays[f"{split}_direct"][..., k, 0].mean(0)
            ids, labels = arrays[f"{split}_cell_ids"], arrays[f"{split}_labels"]
            points = np.array(sorted((float(labels[ids == cell, 0][0]), float(direct[ids == cell].mean()))
                                     for cell in np.unique(ids)))
            ax.plot(points[:, 0], points[:, 1], marker=marker, color=color, lw=.8, ms=4, label=label)
        ax.set(title=f"直接编码尺度 k={k}", xlabel="裸参数 r（仅事后评价）", ylabel="潜变量 $g$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("固定 u=1 的旧 Q/C：先平均编码重复，再平均同参数物理链；所有四臂共用这些编码")
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def run(training, cache, output):
    started = time.process_time()
    training, cache, output = [Path(value).resolve() for value in (training, cache, output)]
    if output.exists() or any(output == source or source in output.parents or output in source.parents
                              for source in (training, cache)):
        raise FileExistsError("四臂报告必须写入独立新目录")
    pins = {}
    verified_public(training, ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz", "R_parameters.npz"), pins)
    verified_public(cache, ("cache.npz", "manifest.json"), pins)
    from scripts import report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation
    for module in (report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation):
        path = Path(module.__file__).resolve()
        pins[str(path)] = sha256(path)
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    manifest = json.loads((training / "manifest.json").read_text())
    summary = json.loads((training / "summary.json").read_text())
    records = json.loads((training / "history.json").read_text())
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    declared_arms = manifest.get("arms", manifest.get("arm_specs"))
    if declared_arms != ARMS or manifest.get("cache_manifest") != cache_manifest:
        raise ValueError("四臂设置或冻结缓存契约不一致")
    if summary.get("all_runs_selected_before_TEST_and_labels_loaded",
                   summary.get("all_endpoints_selected_before_TEST_and_labels_loaded",
                               summary.get("all_seeds_selected_before_TEST_and_labels_loaded"))) is not True:
        raise ValueError("必须在所有端点选定后读取 TEST 与耦合标签")
    splits = manifest["splits"]
    if len(splits) != len(FOLDS) or [split["fold"] for split in splits] != list(FOLDS):
        raise ValueError("需要四个预定的条件级折")
    keys = {model_key(fold, arm, seed, endpoint) for fold in FOLDS for arm in ARMS
            for seed in SEEDS for endpoint in ("selected", "adam6000")}
    if set(summary["final_R_sha256"]) != keys:
        raise ValueError("四臂、四折、三个 R 种子的两个端点不完整")
    with np.load(training / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(cache / "cache.npz", allow_pickle=False) as archive:
        cache_arrays = {name: archive[name].copy() for name in ("eval_train_direct", "eval_test_direct")}
    for split in ("train", "test"):
        direct = arrays[f"{split}_direct"]
        if direct.dtype != np.float64 or direct.ndim != 4 or direct.shape[0] != 4 or direct.shape[-2:] != (3, 1):
            raise ValueError("直接编码形状或 float64 精度不符")
        if not np.array_equal(direct, cache_arrays[f"eval_{split}_direct"]):
            raise ValueError("训练评价编码与不可变缓存不一致")
        labels, ids = arrays[f"{split}_labels"], arrays[f"{split}_cell_ids"]
        if (labels.shape != (direct.shape[1], 3) or ids.shape != (direct.shape[1],)
                or not np.isfinite(labels).all() or not np.allclose(labels[:, 1], 1., rtol=0, atol=1e-12)):
            raise ValueError("需要固定 u=1 的物理链条件标签")
    val_coverage = np.zeros(arrays["train_direct"].shape[1], dtype=int)
    for split in splits:
        fit_mask, val_mask = (np.asarray(split[f"{name}_mask"], dtype=bool) for name in ("fit", "validation"))
        if (fit_mask.shape != val_coverage.shape or val_mask.shape != val_coverage.shape
                or np.any(fit_mask & val_mask) or not np.all(fit_mask | val_mask)):
            raise ValueError("条件折的拟合与验证掩码必须互斥且完整")
        ids = arrays["train_cell_ids"]
        if set(ids[fit_mask]) & set(ids[val_mask]):
            raise ValueError("同一参数条件跨越 R 拟合和验证")
        val_coverage += val_mask
    if not np.all(val_coverage == 1):
        raise ValueError("每条 TRAIN 物理链必须恰好在一折验证")
    observed = np.asarray(cache_manifest.get("train_input_bounds"), dtype=np.float64)
    stds = np.asarray(manifest["target_stds"], dtype=np.float64)
    if (observed.shape != (2,) or not np.isfinite(observed).all() or observed[1] <= observed[0]
            or cache_manifest.get("flow_padding_fraction") != .1 or stds.shape != (3,)
            or not np.isfinite(stds).all() or np.any(stds <= 0)
            or not np.array_equal(stds, cache_manifest["target_stds"])):
        raise ValueError("需要不可变原 TRAIN 区间、10%边界和目标标准差")
    padding = .1*max(float(np.ptp(observed)), 1e-6)
    bounds = [float(observed[0]-padding), float(observed[1]+padding)]

    def guard():
        if time.process_time()-started > CPU_BUDGET_SECONDS:
            raise RuntimeError("四臂报告 CPU 预算300秒已用尽")
        if output.exists() and sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_BUDGET_BYTES:
            raise RuntimeError("四臂报告输出预算128MiB已用尽")

    models, checks = {}, {}
    with np.load(training / "R_parameters.npz", allow_pickle=False) as archive:
        for fold in FOLDS:
            for arm, spec in ARMS.items():
                for seed in SEEDS:
                    for endpoint in ("selected", "adam6000"):
                        guard()
                        key = model_key(fold, arm, seed, endpoint)
                        model = FrozenClosureMaps.restore(archive, key, summary["final_R_sha256"][key], spec)
                        checks[key] = check_predictions(model, arrays, key)
                        models[key] = model
    metrics, closures, metric_difference = checked_metrics(arrays, summary, splits, stds)
    shape, roots, searches = diagnose_maps(models, bounds, guard)
    for rows in roots.values():
        for row in rows:
            row["inside_original_TRAIN_input_interval"] = bool(observed[0] <= row["g"] <= observed[1])
    comparisons = paired_comparisons(metrics, closures)
    output.mkdir(parents=True, exist_ok=False)
    _plot(output, arrays, cache_arrays, records, models, metrics, roots, bounds, splits, guard)
    for model in models.values():
        model.assert_unchanged()
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise RuntimeError("报告期间不可变输入发生变化")
    guard()
    result = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        physical_dimension=2, fixed_u=1., theory_dim=1, R_seeds=list(SEEDS), folds=list(FOLDS), arms=ARMS,
        original_TRAIN_input_interval=observed.tolist(), search_interval=bounds,
        bounds_use_labels_or_TEST=False, representative_plot_fold=FOLDS[0],
        cached_prediction_max_abs_difference=checks, metric_recomputation_max_abs_difference=metric_difference,
        final_R_sha256=summary["final_R_sha256"], latent_metrics=metrics, weighted_chain_mean_loss=closures,
        paired_comparisons=comparisons, roughness=shape, numerical_fixed_points=roots, searches=searches,
        input_sha256=pins, frozen_parameters_unchanged=True, Q_or_C_loaded=False, checkpoint_loaded=False,
        new_training_updates=0, new_physical_samples=0, CPU_threads=1, GPU_seconds=0,
        cpu_seconds=time.process_time()-started, cpu_budget_seconds=CPU_BUDGET_SECONDS,
        output_budget_bytes=OUTPUT_BUDGET_BYTES, critical_r=None, thermal_nu=None,
        physical_fixed_point_certified=False, smoothness_regularization_added=False,
        limitations=["四折验证条件用于各端点选点，验证分数存在乐观偏差，不是无偏交叉验证估计。",
            "原 Q/C 见过全部八个 TRAIN 条件；条件折只相对 R 拟合独立。",
            "原 TEST 已被历史实验查看，仅作为探索性留出评价，不用于跨种子选模型。",
            "三个 R 种子共用一个 Q/C 与旧物理数据，不是三个独立 QCR 实验。",
            "四次 C／子集编码重复不等于四倍独立物理链，折与种子结果也不是独立物理重复。",
            "B 具有两种分尺度映射，仅作状态闭合诊断，不构成共享自主 RG 流或物理不动点。",
            "曲线崎岖与完整 DR 统计依赖有限有界网格，不作为新增训练目标。",
            "共享映射的有限网格数值根搜索不保证穷尽，不认证热方向、临界点或临界指数。",
            "原采样质量警告保留；这些 R-only 结果不能单独区分有限采样与 Q/C 表示误差。"])
    write_json(output / "analysis.json", result)
    write_json(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, analysis_sha256=sha256(output / "analysis.json"),
        cpu_seconds=time.process_time()-started))
    guard()
    write_json(output / "public_files.json", {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description="固定u=1标量表示：四臂四折冻结R诊断报告")
    parser.add_argument("--training", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = run(arguments.training, arguments.cache, arguments.output)
    print(json.dumps(dict(status=result["status"], cpu_seconds=result["cpu_seconds"])))


if __name__ == "__main__":
    main()
