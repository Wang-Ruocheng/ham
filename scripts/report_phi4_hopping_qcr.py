"""新连续 QCR 的离线中文报告；只读公共 JSON/NPZ，不加载模型或构型。

完整 2×2 软联合统计是有限分箱描述，不是连续分布的完备表示。
本报告不搜索不动点，不插值临界耦合，不计算临界指数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_ising_zh import select_chinese_font


RECIPE = "phi4_hopping_raw_patch_joint625_v1"
INPUT_FILES = ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz")
LIMITATIONS = [
    "连续场 Gaussian C、全壳核平滑信息代理和均匀幅度保持不是二值经验直方图 MI／保色 NLL 的逐项等价物。",
    "625 维目标是五个软分箱的完整四点联合统计，不保证保留连续场的所有局域或长程信息。",
    "原始局域块 Q、仿射重构与等目标尺度闭环已对齐；这不保证有限维表示在粗粒化下闭合。",
    "微观 TRAIN RMS 在所有尺度和条件共享；κ、λ、h 不参与训练。",
    "有限独立物理链内的保存构型仍相关；低训练损失不证明热化、相变或普适指数。",
    "图中箭头为 R(g)−g；数据覆盖区域之外为外推，不据此认证物理不动点。",
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Non-finite or nonnumeric public array: {name}")
    return value.astype(np.float64, copy=False)


def load_public_run(run):
    """Fail closed on a mixed recipe, physical labels, split, or endpoint."""
    root = Path(run).resolve()
    pins = {name: _sha(root / name) for name in INPUT_FILES}
    manifest, summary, history = [json.loads((root / name).read_text())
                                  for name in INPUT_FILES[:3]]
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("Manifest and summary must be JSON objects")
    if manifest.get("recipe") != RECIPE or summary.get("recipe") != RECIPE:
        raise ValueError("New raw-patch hopping recipe required; historical runs are incompatible")
    if manifest.get("schema_version") != 5:
        raise ValueError("Expected schema_version=5")
    if (manifest.get("evaluation_label_columns") != ["kappa", "quartic_lambda", "h"]
            or manifest.get("field_representation") != "phi"
            or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training") is not False):
        raise ValueError("Hopping parameter semantics must be explicit and evaluation-only")
    steps = manifest.get("options", {}).get("steps")
    early = summary.get("early_stopping")
    actual_steps = summary.get("actual_steps")
    if early is not None:
        policy = manifest.get("early_stopping", {})
        selected = summary.get("evaluated_checkpoint_step")
        if (manifest.get("options", {}).get("early_stopping") != "train_closure_100_200_patience20_v1"
                or policy.get("policy") != "train_closure_100_200_patience20_v1"
                or steps != 200 or type(actual_steps) is not int or not 100 <= actual_steps <= 200
                or type(selected) is not int or not 100 <= selected <= actual_steps
                or early.get("actual_steps") != actual_steps or early.get("best_step") != selected
                or early.get("evaluated_checkpoint_step") != selected
                or early.get("test_data_used") is not False or early.get("labels_used") is not False):
            raise ValueError("Inconsistent explicit TRAIN-only early-stop endpoint")
        from recurrent_rg.phi4_early_stopping import EarlyStopping
        replay = EarlyStopping()
        for observation in early.get("observations", []):
            replay.observe(observation["step"], observation["value"])
        if (not replay.stopped or replay.best_step != selected
                or replay.receipt() != {key: early.get(key) for key in replay.receipt()}
                or early["observations"][-1]["step"] != actual_steps):
            raise ValueError("TRAIN stopping receipt does not replay to the saved endpoint")
        for name in ("Q", "C", "R"):
            clock = summary.get("selected_optimizer_steps", {}).get(name, {})
            if clock.get("min") != selected or clock.get("max") != selected:
                raise ValueError("Selected checkpoint Adam counters differ")
    elif manifest.get("options", {}).get("early_stopping") is not None:
        raise ValueError("Missing explicit early-stopping receipt")
    if (type(steps) is not int or steps < 1 or summary.get("status") != "complete"
            or (early is None and actual_steps != steps) or summary.get("thermal_nu") is not None):
        raise ValueError("Training endpoint is incomplete or claims a thermal exponent")
    for name in ("Q", "C", "R"):
        clock = summary.get("optimizer_steps", {}).get(name, {})
        if clock.get("min") != actual_steps or clock.get("max") != actual_steps or clock.get("parameters", 0) < 1:
            raise ValueError(f"{name} Adam counters do not reach the requested endpoint")
    if isinstance(history, dict):
        history = history.get("rows", history.get("history"))
    if (not isinstance(history, list) or not history
            or any(not isinstance(row, dict) for row in history)
            or history[-1].get("step") != actual_steps):
        raise ValueError("Loss history must reach the training endpoint")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    direct = _numeric(arrays["direct_g"], "direct_g")
    if direct.ndim != 3 or direct.shape[1] not in (3, 4) or min(direct.shape) < 1:
        raise ValueError("Expected [physical ensembles, three or four scales, latent coordinates]")
    count, scales, dim = direct.shape
    depth = scales - 1
    if scales == 4:
        if (manifest.get("block_size") != 2 or manifest.get("stride") != 2
                or manifest.get("coarse_graining_depth") != depth
                or manifest.get("model", {}).get("depth") != depth
                or manifest.get("options", {}).get("block_size") != 2
                or manifest.get("options", {}).get("depth") != depth):
            raise ValueError("Four-scale reports require explicit b2/depth3 geometry")
        sizes = manifest.get("lattice_sizes")
        if (not isinstance(sizes, list) or len(sizes) != scales
                or any(type(side) is not int or side < 1 for side in sizes)
                or any(sizes[i] != 2 * sizes[i + 1] for i in range(depth))
                or summary.get("lattice_sizes") != sizes):
            raise ValueError("Inconsistent b2 lattice size hierarchy")
    for name, shape in (("one_step_g", (count, depth, dim)), ("rollout_g", direct.shape),
                        ("labels", (count, 3)), ("target_features", (count, scales, 625)),
                        ("reconstructed_features", (count, scales, 625))):
        arrays[name] = _numeric(arrays[name], name)
        if arrays[name].shape != shape:
            raise ValueError(f"Incorrect public array shape: {name}")
    arrays["direct_g"] = direct
    if not np.allclose(arrays["rollout_g"][:, 0], direct[:, 0], atol=1e-12, rtol=1e-10):
        raise ValueError("Microscopic rollout must begin at direct g0")
    if "direct_full_g" in arrays:
        arrays["direct_full_g"] = _numeric(arrays["direct_full_g"], "direct_full_g")
        if arrays["direct_full_g"].shape != direct.shape:
            raise ValueError("Incorrect pooled direct encoding shape")
    predictions, pairs = all_start_predictions(arrays)
    if scales == 4 and "all_start_g" not in arrays:
        raise ValueError("Four-scale evaluation must save every all-start prediction")
    # Historical S=2 reports did not save a complete all-start cache.  Their
    # independently stored one-step/rollout arrays remain supported, without
    # retroactively imposing a new cross-cache consistency requirement.
    if "all_start_g" in arrays:
        for index, (source, target_scale) in enumerate(pairs):
            if target_scale == source + 1 and not np.allclose(predictions[:, index],
                    arrays["one_step_g"][:, source], atol=1e-12, rtol=1e-10):
                raise ValueError("All-start adjacent prediction disagrees with one-step cache")
            if source == 0 and not np.allclose(predictions[:, index],
                    arrays["rollout_g"][:, target_scale], atol=1e-12, rtol=1e-10):
                raise ValueError("All-start microscopic prediction disagrees with rollout cache")
    target = arrays["target_features"]
    if (target < -1e-14).any() or not np.allclose(target.sum(-1), 1., atol=1e-10, rtol=1e-10):
        raise ValueError("Soft joint targets must be normalized nonnegative probabilities")
    splits = arrays["splits"]
    ids = arrays["cell_ids"]
    if splits.shape != (count,) or ids.shape != (count,) or ids.dtype.kind not in "iuUS":
        raise ValueError("Physical ensemble split and condition IDs must be vectors")
    if set(splits.tolist()) != {"train", "test"}:
        raise ValueError("Both TRAIN and TEST must be retained")
    labels = arrays["labels"]
    if (labels[:, 0] <= 0.).any() or labels[0, 1] <= 0. or not np.all(labels[:, 1:] == labels[0, 1:]) or labels[0, 2] != 0.:
        raise ValueError("Require positive κ, fixed λ>0 and h=0")
    for cell in np.unique(ids):
        mask = ids == cell
        if len(set(splits[mask].tolist())) != 1 or not np.all(labels[mask] == labels[mask][0]):
            raise ValueError("Whole physical conditions must share one split and label")
    train = splits == "train"
    if set(map(tuple, labels[train])) & set(map(tuple, labels[~train])):
        raise ValueError("TRAIN/TEST physical conditions overlap")
    if dim == 1:
        for name in ("flow_grid", "flow_next"):
            arrays[name] = _numeric(arrays[name], name)
            if arrays[name].ndim == 1:
                arrays[name] = arrays[name][:, None]
            if arrays[name].ndim != 2 or arrays[name].shape[1] != 1 or len(arrays[name]) < 2:
                raise ValueError("Scalar flow must use saved vectors [grid points,1]")
        if arrays["flow_grid"].shape != arrays["flow_next"].shape or not (np.diff(arrays["flow_grid"][:, 0]) > 0).all():
            raise ValueError("Scalar flow grid must be ordered and match mapped values")
    if pins != {name: _sha(root / name) for name in INPUT_FILES}:
        raise ValueError("Public input changed during report loading")
    return manifest, summary, history, arrays, pins


def all_start_predictions(arrays):
    """Read saved source/target pairs; recover only the complete historical S=2 case."""
    direct = arrays["direct_g"]
    count, scales, dim = direct.shape
    expected = np.asarray([(s, t) for t in range(1, scales) for s in range(t)], dtype=np.int64)
    present = ("all_start_g" in arrays, "all_start_pairs" in arrays)
    if any(present):
        if not all(present):
            raise ValueError("All-start predictions and pair metadata must appear together")
        pairs = np.asarray(arrays["all_start_pairs"])
        if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, expected):
            raise ValueError("All-start pairs must cover every s<t in target-major order")
        values = _numeric(arrays["all_start_g"], "all_start_g")
        if values.shape != (count, len(expected), dim):
            raise ValueError("Incorrect all-start prediction shape")
        return values, expected
    if scales != 3:
        raise ValueError("Missing all-start predictions for the multi-scale evaluation")
    return np.stack((arrays["one_step_g"][:, 0], arrays["rollout_g"][:, 2],
                     arrays["one_step_g"][:, 1]), axis=1), expected


def closure_metrics(arrays):
    """Same-coordinate absolute errors and original TRAIN target-std normalization."""
    direct = arrays["direct_g"]
    train = direct[arrays["splits"] == "train"]
    depth = direct.shape[1] - 1
    stds = np.sqrt(np.mean(np.var(train, axis=0, ddof=0), axis=-1))
    predictions, pairs = all_start_predictions(arrays)
    result = {}
    for split in ("train", "test"):
        mask = arrays["splits"] == split
        result[split] = {}
        def metric(prediction, source, target):
            error = prediction[mask] - direct[mask, target]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            std = float(stds[target])
            return dict(source_scale=int(source), target_scale=int(target), rmse=rmse,
                train_target_std=std, nrmse=rmse / std if std else None)
        for name, values in (("one_step", arrays["one_step_g"]),
                             ("rollout", arrays["rollout_g"][:, 1:])):
            result[split][name] = [metric(values[:, t - 1], t - 1 if name == "one_step" else 0, t)
                for t in range(1, depth + 1)]
        result[split]["all_start"] = [metric(predictions[:, i], s, t)
            for i, (s, t) in enumerate(pairs)]
        result[split]["identity_all_start"] = [metric(direct[:, s], s, t) for s, t in pairs]
        per_target = []
        for target in range(1, depth + 1):
            selected = np.flatnonzero(pairs[:, 1] == target)
            error = predictions[mask][:, selected] - direct[mask, target, None]
            per_target.append(float(np.mean(np.sum(error ** 2, axis=-1))))
        result[split]["all_start_target_mse_coordinate_sum"] = per_target
        result[split]["weighted_all_start_objective"] = 10. * float(np.mean(per_target))
    return result


def _setup_plot(font):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    path = select_chinese_font(font)
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(path)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    return plt


def _plot_loss(rows, output, plt, *, uniform_label="均匀幅度保持"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    groups = [(("q_loss", "q_total", "Q"), ("reconstruction", "q_reconstruction", "联合统计重构"),
               ("subset_micro", "q_subset_micro", "微观子集一致性"), ("subset_coarse", "q_subset_coarse", "粗尺度子集一致性")),
              (("c_loss", "c_total", "C"), ("information_proxy", "c_information", "信息代理"),
               ("variance_floor", "c_variance", "方差下限"), ("uniform_amplitude", "c_uniform", uniform_label)),
              (("r_loss", "r_total", "R"), ("closure_one_step", "closure_one_step", "单步闭合"),
               ("closure_rollout", "closure_rollout", "微观起点多步闭合"))]
    for ax, group, title in zip(axes, groups, ("表示目标 Q", "粗粒化目标 C", "闭环目标 R")):
        plotted = False
        for key, alternate, label in group:
            points = [(row["step"], row.get(key, row.get(alternate))) for row in rows]
            points = [(step, value) for step, value in points if value is not None and np.isfinite(value)]
            if points:
                ax.plot(*np.asarray(points).T, label=label)
                plotted = True
        ax.set(title=title, xlabel="更新次数", ylabel="损失或诊断量")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(fontsize=8)
    fig.savefig(output / "loss_curves.png", dpi=150)
    plt.close(fig)


def _plot_flow(a, output, plt):
    direct, mapped = a["direct_g"], a["one_step_g"]
    depth = direct.shape[1] - 1
    train = a["splits"] == "train"
    if direct.shape[-1] == 1:
        grid, nxt = a["flow_grid"][:, 0], a["flow_next"][:, 0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        axes[0].plot(grid, nxt, label="共享映射 R(g)")
        axes[0].plot(grid, grid, "--", color=".5", label="恒等映射")
        lo, hi = direct[train, :-1, 0].min(), direct[train, :-1, 0].max()
        axes[0].axvspan(lo, hi, color="#70a277", alpha=.15, label="训练输入范围")
        axes[0].set(xlabel="g", ylabel="R(g)", title="标量重整化映射；未搜索不动点")
        for scale in range(depth):
            for split, color, marker, text in ((train, "#1c6095", "o", "训练集"), (~train, "#ba5039", "x", "测试集")):
                x = direct[split, scale, 0]
                drift = mapped[split, scale, 0] - x
                axes[1].scatter(x, np.full_like(x, scale), s=13, c=color, marker=marker,
                                label=text if scale == 0 else None)
                axes[1].quiver(x, np.full_like(x, scale), drift, np.zeros_like(drift),
                               angles="xy", scale_units="xy", scale=1, width=.004, color=color)
        axes[1].set(xlabel="g", ylabel="输入尺度 k", yticks=list(range(depth)), title="数据覆盖点及真实位移 R(g)−g")
        axes[1].set_xlim(min(grid.min(), mapped[..., 0].min()), max(grid.max(), mapped[..., 0].max()))
        axes[1].set_ylim(-.4, depth - .6)
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fraction = [1.]
    else:
        fitted = direct[train].reshape(-1, direct.shape[-1])
        center = fitted.mean(0)
        _, singular, basis = np.linalg.svd(fitted - center, full_matrices=False)
        axes_basis = basis[:2]
        origin = (direct[:, :-1] - center) @ axes_basis.T
        delta = (mapped - direct[:, :-1]) @ axes_basis.T
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for mask, color, text in ((train, "#1c6095", "训练集"), (~train, "#ba5039", "测试集")):
            p, d = origin[mask].reshape(-1, 2), delta[mask].reshape(-1, 2)
            ax.scatter(p[:, 0], p[:, 1], c=color, s=15, label=text)
            ax.quiver(p[:, 0], p[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, color=color, width=.003)
        ax.set(xlabel="训练主成分 1", ylabel="训练主成分 2", title="数据点流场投影；箭头为 R(g)−g")
        ax.legend()
        fraction = (singular ** 2 / np.sum(singular ** 2)).tolist() if np.any(singular) else [0.] * len(singular)
    fig.savefig(output / "R_vs_g.png", dpi=150)
    plt.close(fig)
    return fraction


def _plot_couplings(a, output, plt):
    dim = a["direct_g"].shape[-1]
    fig, axes = plt.subplots(1, min(dim, 3), figsize=(5 * min(dim, 3), 4.3), squeeze=False, constrained_layout=True)
    for coord, ax in enumerate(axes[0]):
        for scale in range(a["direct_g"].shape[1]):
            for split, marker, text in (("train", "o", "训练集"), ("test", "x", "测试集")):
                mask = a["splits"] == split
                ax.scatter(a["labels"][mask, 0], a["direct_g"][mask, scale, coord],
                           s=18, marker=marker, color=f"C{scale}", label=f"k={scale}，{text}")
        ax.set(xlabel="耦合 κ", ylabel=f"g$_{{{coord + 1}}}$", title="耦合与直接编码；仅事后评价")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output / "coupling_by_scale.png", dpi=150)
    plt.close(fig)


def _plot_all_start_closure(metrics, output, plt):
    """Every saved s<t pair, with a same-coordinate identity baseline."""
    pairs = metrics["train"]["all_start"]
    labels = [f"{row['source_scale']}→{row['target_scale']}" for row in pairs]
    locations = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    groups = (("train", "all_start", "训练集 R", "#1c6095"),
              ("test", "all_start", "测试集 R", "#ba5039"),
              ("train", "identity_all_start", "训练集恒等映射", "#82aecb"),
              ("test", "identity_all_start", "测试集恒等映射", "#dfa392"))
    width = .19
    for ax, key, title, ylabel in zip(axes, ("rmse", "nrmse"),
            ("全部起点到终点的闭合误差", "按原训练终点尺度标准差归一化"),
            ("绝对均方根误差", "归一化均方根误差")):
        for offset, (split, field, label, color) in enumerate(groups):
            values = [row[key] if row[key] is not None else np.nan
                      for row in metrics[split][field]]
            ax.bar(locations + (offset - 1.5) * width, values, width,
                   color=color, label=label)
        ax.set(xticks=locations, xticklabels=labels, xlabel="起点尺度 s → 终点尺度 t",
               ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "all_start_closure.png", dpi=150)
    plt.close(fig)


def render_report(run, output=None, *, font=None):
    """Render saved endpoint arrays; never mutate or infer from a checkpoint."""
    manifest, summary, history, arrays, pins = load_public_run(run)
    output = Path(output) if output is not None else Path(run) / "report"
    if output.resolve() == Path(run).resolve():
        raise ValueError("Report must not overwrite training source artifacts")
    plt = _setup_plot(font)  # Verify an existing font before creating output.
    output.mkdir(parents=True, exist_ok=False)
    try:
        _plot_loss(history, output, plt)
        fractions = _plot_flow(arrays, output, plt)
        _plot_couplings(arrays, output, plt)
        closures = closure_metrics(arrays)
        _plot_all_start_closure(closures, output, plt)
        metrics = {}
        for split in ("train", "test"):
            mask = arrays["splits"] == split
            delta = arrays["one_step_g"][mask] - arrays["direct_g"][mask, 1:]
            recons = arrays["reconstructed_features"][mask] - arrays["target_features"][mask]
            metrics[split] = {"ensembles": int(mask.sum()),
                "one_step_coordinate_rmse_by_scale": np.sqrt(np.mean(delta ** 2, axis=(0, 2))).tolist(),
                "joint_probability_rmse_by_scale": np.sqrt(np.mean(recons ** 2, axis=(0, 2))).tolist(),
                "closure": closures[split]}
        scales = arrays["direct_g"].shape[1]
        _, pairs = all_start_predictions(arrays)
        report = {"schema_version": 5, "recipe": RECIPE, "status": "complete",
            "actual_steps": summary["actual_steps"], "labels_used_in_training": False,
            "thermal_nu": None, "critical_kappa": None, "fixed_point_search_performed": False,
            "source_sha256": pins, "field_representation": "phi",
            "evaluation_label_columns": ["kappa", "quartic_lambda", "h"],
            "scales": scales, "depth": scales - 1,
            "lattice_sizes": manifest.get("lattice_sizes"),
            "all_start_pairs": pairs.tolist(),
            "all_start_predictions_source": ("saved_complete_cache" if "all_start_g" in arrays
                else "historical_two_transition_one_step_and_rollout_fallback"),
            "closure_definition": "L_R=(10/S)Σ_{t=1..S}(1/t)Σ_{s=0..t-1}||R^(t-s)(g_s)-g_t||²；固定直接编码终点；坐标平方和",
            "closure_normalization": "每个终点 t 使用原TRAIN直接编码 g_t 的总体标准差；不按TEST重新归一化",
            "train_only_projection_variance_fraction": fractions, "metrics": metrics,
            "limitations": LIMITATIONS, "plots": ["loss_curves.png", "R_vs_g.png", "coupling_by_scale.png", "all_start_closure.png"]}
        if summary.get("early_stopping") is not None:
            report.update(evaluated_checkpoint_step=summary["evaluated_checkpoint_step"],
                          early_stopping=summary["early_stopping"])
        if pins != {name: _sha(Path(run) / name) for name in INPUT_FILES}:
            raise ValueError("Public source changed during report generation")
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        plt.close("all")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.run, args.output, font=args.font), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
