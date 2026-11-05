"""Public JSON/NPZ report for the explicit fixed-C phi4 Q-dimension ablation.

The encoder is never run.  Frozen residual maps are restored in NumPy and
saved latent predictions are checked before their affine feature decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.report_phi4_scalar_r_validation import sha256, state_fingerprint, verified_public, write_json


DIMS = (1, 2)
SEEDS = (91, 92, 93)
ENDPOINTS = ("selected", "adam6000")
PATHS = (("0_to_1", 0, 1, "one_step", 0), ("1_to_2", 1, 2, "one_step", 1),
         ("0_to_2", 0, 2, "rollout", 2))
PATH_LABELS = ("36→12", "12→4", "36→4")
COLORS = ("#3d7d9c", "#b66b55", "#7c69a2")
CPU_BUDGET_SECONDS = 300.
OUTPUT_BUDGET_BYTES = 128*1024**2


def extract_state(archive, key):
    prefix = key+"."
    return {name[len(prefix):]: archive[name].copy() for name in archive.files if name.startswith(prefix)}


def validate_state(state, shapes, digest):
    if (set(state) != set(shapes) or any(value.dtype != np.float64 or value.shape != shapes[name]
            or not np.isfinite(value).all() for name, value in state.items())):
        raise ValueError("公开参数形状、精度或数值无效")
    if state_fingerprint(state) != digest:
        raise ValueError("公开参数指纹不一致")
    for value in state.values():
        value.setflags(write=False)


def linear(value, weight):
    return np.einsum("ni,ji->nj", value, weight, optimize=False)


class FrozenMap:
    """Original-coordinate normalized residual map, with analytic full DR."""

    def __init__(self, state, digest, *, dim, hidden):
        self.state = {name: np.array(value, copy=True) for name, value in state.items()}
        shapes = {"center": (dim,), "scale": (dim,), "net.0.weight": (hidden, dim),
            "net.0.bias": (hidden,), "net.2.weight": (hidden, hidden), "net.2.bias": (hidden,),
            "net.4.weight": (dim, hidden), "net.4.bias": (dim,)}
        validate_state(self.state, shapes, digest)
        if np.any(self.state["scale"] <= 0):
            raise ValueError("R 的固定输入尺度必须为正")
        self.digest, self.dim = digest, dim

    def forward_and_derivative(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.dim or not np.isfinite(points).all():
            raise ValueError("R 输入维数或有限性不符")
        s = self.state
        z = (points-s["center"])/s["scale"]
        h1 = np.tanh(linear(z, s["net.0.weight"])+s["net.0.bias"])
        h2 = np.tanh(linear(h1, s["net.2.weight"])+s["net.2.bias"])
        result = points+s["scale"]*(linear(h2, s["net.4.weight"])+s["net.4.bias"])
        d1 = (1-h1**2)[..., None]*s["net.0.weight"][None]/s["scale"][None, None]
        d2 = (1-h2**2)[..., None]*np.einsum("nhi,jh->nji", d1, s["net.2.weight"], optimize=False)
        full_dr = np.eye(self.dim)[None]+s["scale"][None, :, None]*np.einsum(
            "nhi,jh->nji", d2, s["net.4.weight"], optimize=False)
        if not np.isfinite(result).all() or not np.isfinite(full_dr).all():
            raise FloatingPointError("冻结 R 或完整 DR 非有限")
        return result, full_dr

    def predict(self, direct):
        direct = np.asarray(direct, dtype=np.float64)
        if direct.ndim != 4 or direct.shape[-2:] != (3, self.dim) or not np.isfinite(direct).all():
            raise ValueError("直接编码维数或有限性不符")
        shape = direct.shape[:-2]+(self.dim,)
        first = self.forward_and_derivative(direct[..., 0, :].reshape(-1, self.dim))[0]
        second = self.forward_and_derivative(direct[..., 1, :].reshape(-1, self.dim))[0]
        rollout = self.forward_and_derivative(first)[0]
        first, second, rollout = (value.reshape(shape) for value in (first, second, rollout))
        return dict(one_step=np.stack((first, second), -2),
                    rollout=np.stack((direct[..., 0, :], first, rollout), -2))

    def assert_unchanged(self):
        if state_fingerprint(self.state) != self.digest or any(value.flags.writeable for value in self.state.values()):
            raise RuntimeError("冻结 R 参数发生变化")


class FrozenDecoder:
    """Pin the entire Q state; use its affine decoder only, never its encoder."""

    def __init__(self, state, digest, *, dim, hidden):
        self.state = {name: np.array(value, copy=True) for name, value in state.items()}
        shapes = {"net.0.weight": (hidden, 105), "net.0.bias": (hidden,),
            "net.2.weight": (12, hidden), "net.2.bias": (12,),
            "net.4.weight": (hidden, 12), "net.4.bias": (hidden,),
            "net.6.weight": (dim, hidden), "net.6.bias": (dim,),
            "decoder.weight": (105, dim), "decoder.bias": (105,)}
        validate_state(self.state, shapes, digest)
        self.digest, self.dim = digest, dim
        norm = np.linalg.norm(self.state["decoder.weight"], axis=0)
        if np.any(norm <= 0) or not np.allclose(norm, 1., rtol=1e-10, atol=1e-12):
            raise ValueError("Q 仿射 decoder 的各列应为单位范数；不要求列间满秩")
        self.weight = self.state["decoder.weight"]/np.maximum(norm, 1e-15)
        self.weight.setflags(write=False)

    def decode(self, points):
        value = np.asarray(points, dtype=np.float64)
        if value.shape[-1] != self.dim or not np.isfinite(value).all():
            raise ValueError("仿射解码输入无效")
        shape = value.shape[:-1]+(105,)
        return (linear(value.reshape(-1, self.dim), self.weight)+self.state["decoder.bias"]).reshape(shape)

    def assert_unchanged(self):
        if (state_fingerprint(self.state) != self.digest or any(value.flags.writeable for value in self.state.values())
                or self.weight.flags.writeable):
            raise RuntimeError("冻结 Q 参数或仿射 decoder 发生变化")


def feature_normalizers(train_target):
    value = np.asarray(train_target, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] != 4 or value.shape[-2:] != (3, 105) or not np.isfinite(value).all():
        raise ValueError("需要四次编码重复的 TRAIN 三尺度105维特征")
    mean = value.mean((0, 1))
    scales = np.sqrt(np.mean((value-mean)**2, axis=(0, 1, 3)))
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("每个目标尺度的共同特征基线误差必须为正")
    return scales, mean


def residual_metrics(value, normalizer):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 4 or value.shape[-1] != 105 or not np.isfinite(value).all():
        raise ValueError("需要 [四次重复,物理链,105维特征] 残差")
    rmse = float(np.sqrt(np.mean(value**2)))
    chain_rmse = float(np.sqrt(np.mean(value.mean(0)**2)))
    return dict(raw_repeat_rmse=rmse, chain_mean_rmse=chain_rmse,
        raw_repeat_nrmse=rmse/normalizer, chain_mean_nrmse=chain_rmse/normalizer,
        repeat_rmse=np.sqrt(np.mean(value**2, axis=(1, 2))).tolist(),
        coordinate_rmse=np.sqrt(np.mean(value**2, axis=(0, 1))).tolist(),
        normalizer=float(normalizer), physical_chains=int(value.shape[1]), encoding_repeats=4,
        feature_dim=105, directly_comparable_across_dg=True,
        repeats_are_not_independent_physical_chains=True)


def common_feature_metrics(decoder, direct, predicted, target, normalizers):
    rows = []
    for path, start, end, kind, index in PATHS:
        reconstructed = decoder.decode(direct[..., end, :])
        values = dict(prediction=decoder.decode(predicted[kind][..., index, :])-target[..., end, :],
            target_Q_reconstruction=reconstructed-target[..., end, :],
            identity_decoded=decoder.decode(direct[..., start, :])-target[..., end, :],
            identity_feature=target[..., start, :]-target[..., end, :])
        rows.append(dict(path=path, target_scale=end, **{
            name: residual_metrics(value, normalizers[end]) for name, value in values.items()}))
    return rows


def input_bounds(direct):
    direct = np.asarray(direct, dtype=np.float64)
    value = direct[..., :2, :].reshape(-1, direct.shape[-1])
    low, high = value.min(0), value.max(0)
    width = high-low
    if not np.isfinite(value).all() or np.any(width <= 0):
        raise ValueError("TRAIN 输入坐标区间退化或非有限")
    pad = .1*np.maximum(width, 1e-6)
    return np.stack((low-pad, high+pad), -1).tolist(), np.stack((low, high), -1).tolist()


def verify_model_predictions(model, decoder, arrays, key, dim):
    maximum = 0.
    for name in ("train", "test"):
        direct = arrays[f"dg{dim}_{name}_direct"]
        for kind, actual in model.predict(direct).items():
            expected = arrays[f"eval_{key}_{name}_{kind}"]
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise ValueError("NumPy R 一步／复合预测与缓存不一致")
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
        reconstructed = decoder.decode(direct)
        saved = arrays[f"dg{dim}_{name}_Q_reconstruction"]
        if reconstructed.shape != saved.shape or not np.allclose(reconstructed, saved, rtol=1e-10, atol=1e-12):
            raise ValueError("仿射 Q 重构与缓存不一致")
        maximum = max(maximum, float(np.max(np.abs(reconstructed-saved))))
    return maximum


def dimension_comparison(metrics):
    pairs = []
    for seed in SEEDS:
        for endpoint in ENDPOINTS:
            for split in ("validation", "test"):
                one, two = (metrics[f"dg{dim}_seed{seed}_{endpoint}"][split] for dim in DIMS)
                def ratios(field):
                    return [b["prediction"][field]/a["prediction"][field]
                            if a["prediction"][field] > 0 else None for a, b in zip(one, two)]
                raw, chain = ratios("raw_repeat_rmse"), ratios("chain_mean_rmse")
                scores = [sum(weight*row["prediction"]["chain_mean_rmse"]**2
                              for weight, row in zip((5., 2.5, 2.5), values)) for values in (one, two)]
                pairs.append(dict(R_seed=seed, endpoint=endpoint, split=split,
                    dg2_over_dg1_raw_repeat_rmse=raw, dg2_over_dg1_chain_mean_rmse=chain,
                    weighted_chain_mean_feature_loss_ratio=scores[1]/scores[0] if scores[0] > 0 else None,
                    all_three_raw_paths_improved=all(value is not None and value < 1 for value in raw),
                    all_three_chain_mean_paths_improved=all(value is not None and value < 1 for value in chain)))
    selected_validation = [row for row in pairs if row["endpoint"] == "selected" and row["split"] == "validation"]
    validation_ratios = [row["weighted_chain_mean_feature_loss_ratio"] for row in selected_validation]
    finite_ratios = [value for value in validation_ratios if value is not None and np.isfinite(value)]
    median_ratio = float(np.median(finite_ratios)) if len(finite_ratios) == len(SEEDS) else None
    improved = sum(value < 1 for value in finite_ratios)
    return dict(pairs=pairs, selected_validation_R_seed_pairs=len(selected_validation),
        all_three_raw_paths_improved_count=sum(row["all_three_raw_paths_improved"] for row in selected_validation),
        all_three_chain_mean_paths_improved_count=sum(row["all_three_chain_mean_paths_improved"] for row in selected_validation),
        Q_seed_replication_gate=dict(metric="5*e01+2.5*e12+2.5*e02 of chain-mean common-feature residual MSE",
            ratio="dg2/dg1; same physical chains, C draws and original TRAIN target space",
            median_ratio_threshold=.8, improved_R_seed_count_threshold=2,
            observed_median_ratio=median_ratio, observed_improved_R_seeds=improved,
            passed=bool(median_ratio is not None and median_ratio <= .8 and improved >= 2),
            engineering_gate_only=True, physical_dimension_certified=False,
            TEST_or_roots_used=False, new_Q_seed_runs_launched_by_report=False),
        decision_uses_TEST_or_roots=False, automatic_next_Q_seed_selection=False,
        interpretation="同一Q种子下的描述性配对诊断；R验证参与端点选择，不是独立显著性检验")


def geometry_diagnostics(decoder, train_direct):
    """Descriptive rank/variance evidence; never force rank or select axes."""
    gram = decoder.weight.T@decoder.weight
    singular_values = np.linalg.svd(decoder.weight, compute_uv=False)
    values = np.asarray(train_direct, dtype=np.float64)[..., :2, :].reshape(-1, decoder.dim)
    centered = values-values.mean(0)
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False)/len(values)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    variance = np.maximum(eigenvalues, 0.)
    total = float(variance.sum())
    fractions = variance/total if total > 0 else np.zeros_like(variance)
    return dict(normalized_decoder_column_Gram=gram.tolist(),
        normalized_decoder_singular_values=singular_values.tolist(),
        normalized_decoder_condition_number=float(singular_values.max()/singular_values.min())
            if singular_values.min() > 0 else None,
        TRAIN_input_covariance=covariance.tolist(), TRAIN_input_covariance_ddof=0,
        TRAIN_input_covariance_eigenvalues=eigenvalues.tolist(),
        TRAIN_input_variance_explained_fractions=fractions.tolist(),
        TRAIN_input_variance_participation_ratio=float(total**2/np.sum(variance**2))
            if np.any(variance > 0) else None,
        input_scales=[0, 1], full_rank_forced=False,
        interpretation="网络原坐标下的描述性方差／decoder方向；单独不认证有效物理维度")


def plot_results(output, arrays, models, metrics, summary, bounds, guard):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    for normalized in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), constrained_layout=True)
        field = "raw_repeat_nrmse" if normalized else "raw_repeat_rmse"
        factor = 100. if normalized else 1.
        for row_index, split in enumerate(("validation", "test")):
            for path_index, ax in enumerate(axes[row_index]):
                for dim, x in ((1, 0.), (2, 1.)):
                    for endpoint, offset, color, marker, label in (("selected", -.12, "#b66b55", "o", "验证选定端点"),
                            ("adam6000", .12, "#3d7d9c", "x", "Adam6000端点")):
                        values = [metrics[f"dg{dim}_seed{seed}_{endpoint}"][split][path_index]["prediction"][field]*factor
                                  for seed in SEEDS]
                        ax.scatter(x+offset+np.linspace(-.025, .025, len(values)), values, color=color,
                                   marker=marker, s=32, label=label if dim == 1 else None)
                    baseline = metrics[f"dg{dim}_seed{SEEDS[0]}_selected"][split][path_index]
                    for name, color, style, label in (("target_Q_reconstruction", "#6c9a70", "-", "目标尺度Q重构"),
                            ("identity_decoded", "#9a79ac", "--", "潜变量恒等后解码"),
                            ("identity_feature", ".45", ":", "特征恒等基线")):
                        value = baseline[name][field]*factor
                        ax.plot([x-.25, x+.25], [value, value], style, color=color, lw=1.2,
                                label=label if dim == 1 else None)
                ax.set(xticks=[0, 1], xticklabels=["$d_g=1$", "$d_g=2$"],
                    title=f"{'选点验证条件' if split == 'validation' else '原已查看 TEST'}：{PATH_LABELS[path_index]}",
                    ylabel="共同特征 NRMSE（%）" if normalized else "共同105维特征 RMSE")
                ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(fontsize=7.2, frameon=False)
        fig.suptitle("冻结同一个C：比较共同特征空间；散点为三个R种子，不按TEST挑选种子")
        fig.savefig(output/("feature_nrmse.png" if normalized else "feature_comparison.png"), dpi=170)
        plt.close(fig)
        guard()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    scalar_grid = np.linspace(*bounds["1"][0], 512)[:, None]
    direct = arrays["dg1_train_direct"]
    for ax, seed, color in zip(axes, SEEDS, COLORS):
        mapped = models[f"dg1_seed{seed}_selected"].forward_and_derivative(scalar_grid)[0]
        ax.plot(scalar_grid[:, 0], scalar_grid[:, 0], ":", color=".4", label="恒等映射")
        ax.plot(scalar_grid[:, 0], mapped[:, 0], color=color, label="验证选定R")
        for k, shade in ((0, "#5e9067"), (1, "#9478aa")):
            values = direct[..., k, 0].ravel()
            ax.scatter(values, np.full_like(values, .025+.03*k), s=5, alpha=.3, color=shade,
                       transform=ax.get_xaxis_transform(), label=f"TRAIN输入 k={k}")
        ax.set(title=f"$d_g=1$；R种子{seed}", xlabel="潜变量 $g$", ylabel="$R(g)$", xlim=bounds["1"][0])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("固定u=1的新一维Q：区间仅由TRAIN输入尺度0/1确定；本诊断不搜索数值根")
    fig.savefig(output/"R_vs_g.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True, constrained_layout=True)
    vectors = [np.linspace(low, high, 18) for low, high in bounds["2"]]
    xx, yy = np.meshgrid(*vectors, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    for ax, seed in zip(axes, SEEDS):
        mapped = models[f"dg2_seed{seed}_selected"].forward_and_derivative(grid)[0]
        drift = mapped-grid
        ax.quiver(grid[:, 0], grid[:, 1], drift[:, 0], drift[:, 1], angles="xy",
                  scale_units="xy", scale=1, width=.003, color="#3d7d9c", alpha=.8)
        for k, color in ((0, "#659b75"), (1, "#b87c66")):
            direct = arrays["dg2_train_direct"][..., k, :].reshape(-1, 2)
            ax.scatter(direct[:, 0], direct[:, 1], s=7, color=color, alpha=.35, label=f"TRAIN输入 k={k}")
        ax.set(title=f"$d_g=2$；R种子{seed}", xlabel="潜变量 $g_1$", ylabel="潜变量 $g_2$",
               xlim=bounds["2"][0], ylim=bounds["2"][1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("箭头为原坐标真实位移 R(g)−g；网格来自TRAIN输入包围盒；未进行二维不动点搜索")
    fig.savefig(output/"R_displacement.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), constrained_layout=True)
    for row_index, (dim, coordinate) in enumerate(((1, 0), (2, 0), (2, 1))):
        for k, ax in enumerate(axes[row_index]):
            for split, label, color, marker in (("train", "原TRAIN", "#3d7d9c", "o"),
                                               ("test", "原已查看TEST", "#b66b55", "^")):
                direct = arrays[f"dg{dim}_{split}_direct"][..., k, coordinate].mean(0)
                ids, labels = arrays[f"{split}_cell_ids"], arrays[f"{split}_labels"]
                points = np.array(sorted((float(labels[ids == cell, 0][0]), float(direct[ids == cell].mean()))
                                         for cell in np.unique(ids)))
                ax.plot(points[:, 0], points[:, 1], marker=marker, color=color, lw=.8, ms=4, label=label)
            ax.set(title=f"$d_g={dim}$，坐标{coordinate+1}，尺度k={k}", xlabel="裸参数 r（仅事后评价）",
                   ylabel="$g$" if dim == 1 else f"$g_{coordinate+1}$")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("同一固定C与配对Q初始化：先平均编码重复，再平均同参数物理链；坐标轴不跨维度等同")
    fig.savefig(output/"coupling_by_scale.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, dim in zip(axes, DIMS):
        history = summary["Q_records"][str(dim)]["history"]
        x = [row["step"] for row in history]
        ax.plot(x, [max(row["loss"], 1e-18) for row in history], label="加权Q总目标", color=".25")
        for name, weight, label, color in (("reconstruction", 10, "重构×10", COLORS[0]),
                ("subset_micro", 20, "微观子集一致性×20", COLORS[1]),
                ("subset_coarse", 5, "粗尺度子集一致性×5", COLORS[2])):
            ax.plot(x, [max(weight*row["terms"][name], 1e-18) for row in history], color=color, label=label)
        ax.set(title=f"配对Q种子{summary['q_seed']}；$d_g={dim}$", xlabel="Q的Adam更新数",
               ylabel="Q表示目标", yscale="log")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("冻结C、有限特征缓存的显式维度消融：Q仅训练3000步，R另行训练")
    fig.savefig(output/"Q_loss.png", dpi=170)
    plt.close(fig)
    guard()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)
    for row_index, dim in enumerate(DIMS):
        for ax, seed in zip(axes[row_index], SEEDS):
            history = summary["R_records"][f"dg{dim}_seed{seed}"]["history"]
            for stage, style, offset in (("adam", "-", 0), ("lbfgs", "--", 6000)):
                rows = history[stage]
                x = [offset+row["step"] for row in rows]
                ax.plot(x, [max(row["fit_loss"], 1e-18) for row in rows], style, color=COLORS[0],
                        label="拟合" if stage == "adam" else None)
                ax.plot(x, [max(row["validation"]["chain_mean_loss"], 1e-18) for row in rows], style,
                        color=COLORS[1], label="选点验证" if stage == "adam" else None)
            ax.axvline(6000, lw=.6, color=".7")
            ax.set(title=f"$d_g={dim}$；R种子{seed}", xlabel="Adam更新数＋L-BFGS累计迭代数",
                   ylabel="各自g坐标闭合目标", yscale="log")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("R收敛检查：虚线为L-BFGS；不同维度的潜变量loss不直接比较")
    fig.savefig(output/"R_loss.png", dpi=170)
    plt.close(fig)


def run(training, features, output):
    started = time.process_time()
    training, features, output = [Path(value).resolve() for value in (training, features, output)]
    if output.exists() or any(output == source or source in output.parents or output in source.parents
                              for source in (training, features)):
        raise FileExistsError("维度对照报告必须写入独立新目录")
    pins = {}
    verified_public(training, ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz",
                              "Q_parameters.npz", "R_parameters.npz"), pins)
    verified_public(features, ("features.npz", "manifest.json", "receipt.json"), pins)
    from scripts import report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation
    for module in (report_phi4_qcr, report_ising_zh, report_phi4_scalar_r_validation):
        path = Path(module.__file__).resolve()
        pins[str(path)] = sha256(path)
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    manifest = json.loads((training/"manifest.json").read_text())
    summary = json.loads((training/"summary.json").read_text())
    history = json.loads((training/"history.json").read_text())
    feature_manifest = json.loads((features/"manifest.json").read_text())
    receipt = json.loads((features/"receipt.json").read_text())
    if (manifest.get("theory_dims") != list(DIMS) or manifest.get("q_steps") != 3000
            or manifest.get("q_hidden") != 96 or manifest.get("r_hidden") != 96
            or manifest.get("feature_manifest") != feature_manifest
            or manifest.get("explicit_fixed_C_dimension_ablation") is not True
            or manifest.get("exact_online_QCR_reproduction") is not False
            or manifest.get("labels_loaded_before_all_selections") is not False
            or manifest.get("TEST_loaded_before_all_selections") is not False
            or summary.get("all_Q_fits_and_R_selections_before_TEST_and_labels") is not True
            or summary.get("Q_frozen_during_R") is not True or summary.get("C_updates") != 0
            or summary.get("common_feature_targets_identical") is not True
            or feature_manifest.get("lattice_sizes") != [36, 12, 4]
            or feature_manifest.get("feature_dim") != 105
            or receipt.get("source_unchanged") is not True
            or receipt.get("features_sha256") != sha256(features/"features.npz")
            or history != dict(Q=summary.get("Q_records"), R=summary.get("R_records"))):
        raise ValueError("冻结C维度消融、输入来源或端点选定顺序不符")
    keys = {f"dg{dim}_seed{seed}_{endpoint}" for dim in DIMS for seed in SEEDS for endpoint in ENDPOINTS}
    if set(summary["final_R_sha256"]) != keys or set(summary["final_Q_sha256"]) != {"1", "2"}:
        raise ValueError("两个Q维度及三个R种子的端点不完整")
    with np.load(training/"evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(features/"features.npz", allow_pickle=False) as archive:
        for split, chains in (("train", 32), ("test", 20)):
            target = arrays[f"{split}_feature_target"]
            if (target.shape != (4, chains, 3, 105) or target.dtype != np.float64
                    or not np.isfinite(target).all() or not np.array_equal(target, archive[f"eval_{split}_feature_target"])):
                raise ValueError("共同105维特征目标与不可变缓存不一致")
            for name in ("cell_ids", "labels"):
                if not np.array_equal(arrays[f"{split}_{name}"], archive[f"{split}_{name}"]):
                    raise ValueError("物理链条件顺序与冻结特征缓存不同")
            labels = arrays[f"{split}_labels"]
            if labels.shape != (chains, 3) or not np.isfinite(labels).all() or not np.allclose(labels[:, 1], 1., atol=1e-12, rtol=0):
                raise ValueError("报告只适用固定u=1单参数族")
            for dim in DIMS:
                direct = arrays[f"dg{dim}_{split}_direct"]
                if direct.shape != (4, chains, 3, dim) or direct.dtype != np.float64 or not np.isfinite(direct).all():
                    raise ValueError("编码重复、物理链数或表示维度不符")
    train_ids, test_ids = arrays["train_cell_ids"], arrays["test_cell_ids"]
    if set(train_ids) & set(test_ids):
        raise ValueError("TRAIN与TEST参数条件重叠")
    split = manifest["split"]
    masks = {name: np.asarray(split[f"{name}_mask"], dtype=bool) for name in ("fit", "validation")}
    if (any(value.shape != (32,) for value in masks.values()) or np.any(masks["fit"] & masks["validation"])
            or not np.all(masks["fit"] | masks["validation"])
            or set(train_ids[masks["fit"]]) & set(train_ids[masks["validation"]])):
        raise ValueError("R拟合和验证必须按完整参数条件划分")
    normalizers, feature_mean = feature_normalizers(arrays["train_feature_target"])
    summary["q_seed"] = manifest["options"]["q_seed"]

    def guard():
        if time.process_time()-started > CPU_BUDGET_SECONDS:
            raise RuntimeError("维度对照报告CPU预算300秒已用尽")
        if output.exists() and sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_BUDGET_BYTES:
            raise RuntimeError("维度对照报告输出预算128MiB已用尽")

    decoders, models, checks = {}, {}, {}
    with np.load(training/"Q_parameters.npz", allow_pickle=False) as archive:
        for dim in DIMS:
            decoders[dim] = FrozenDecoder(extract_state(archive, f"dg{dim}"), summary["final_Q_sha256"][str(dim)],
                                          dim=dim, hidden=manifest["q_hidden"])
    with np.load(training/"R_parameters.npz", allow_pickle=False) as archive:
        for key in sorted(keys):
            guard()
            dim = int(key[2])
            models[key] = FrozenMap(extract_state(archive, key), summary["final_R_sha256"][key],
                                   dim=dim, hidden=manifest["r_hidden"])
            checks[key] = verify_model_predictions(models[key], decoders[dim], arrays, key, dim)
    metrics, maximum_metric_difference = {}, 0.
    for key in sorted(keys):
        dim = int(key[2])
        metrics[key] = {}
        for name in ("train", "fit", "validation", "test"):
            source = "test" if name == "test" else "train"
            mask = masks[name] if name in masks else slice(None)
            direct = arrays[f"dg{dim}_{source}_direct"][:, mask]
            prediction = {kind: arrays[f"eval_{key}_{source}_{kind}"][:, mask] for kind in ("one_step", "rollout")}
            rows = common_feature_metrics(decoders[dim], direct, prediction,
                arrays[f"{source}_feature_target"][:, mask], normalizers)
            saved = summary["feature_metrics"][key][name]
            if len(saved) != 3 or [row["path"] for row in saved] != [row["path"] for row in rows]:
                raise ValueError("共同特征预测路径不完整")
            for actual, expected in zip(rows, saved):
                for kind in ("prediction", "target_Q_reconstruction", "identity_decoded", "identity_feature"):
                    for field in ("raw_repeat_rmse", "chain_mean_rmse", "repeat_rmse", "coordinate_rmse"):
                        a, b = np.asarray(actual[kind][field]), np.asarray(expected[kind][field])
                        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                            raise ValueError(f"公开预测与共同特征指标不一致：{key}/{name}/{kind}/{field}")
                        maximum_metric_difference = max(maximum_metric_difference, float(np.max(np.abs(a-b))))
            metrics[key][name] = rows
    bounds, observed = {}, {}
    for dim in DIMS:
        bounds[str(dim)], observed[str(dim)] = input_bounds(arrays[f"dg{dim}_train_direct"])
    comparison = dimension_comparison(metrics)
    geometry = {str(dim): geometry_diagnostics(decoders[dim], arrays[f"dg{dim}_train_direct"]) for dim in DIMS}
    output.mkdir(parents=True, exist_ok=False)
    plot_results(output, arrays, models, metrics, summary, bounds, guard)
    for item in (*models.values(), *decoders.values()):
        item.assert_unchanged()
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise RuntimeError("维度对照报告期间不可变公开输入变化")
    guard()
    result = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        physical_dimension=2, fixed_u=1., theory_dims=list(DIMS), Q_seed=manifest["options"]["q_seed"],
        R_seeds=list(SEEDS), Q_steps=3000, R_Adam_steps=6000,
        explicit_fixed_C_dimension_ablation=True, exact_fresh_joint_QCR_reproduction=False,
        source_C_sha256=manifest["source_C_sha256"], final_Q_sha256=summary["final_Q_sha256"],
        final_R_sha256=summary["final_R_sha256"], common_feature_dim=105,
        feature_metrics=metrics, dimension_comparison=comparison, geometry_diagnostics=geometry,
        feature_target_normalizers=normalizers.tolist(), feature_TRAIN_means=feature_mean.tolist(),
        feature_normalizer_definition="各尺度固定TRAIN目标特征相对该尺度TRAIN均值向量的逐坐标RMSE；先在重复和物理链上中心化，再对105坐标平均",
        normalization_uses_TEST=False, normalization_identical_across_dimensions=True,
        latent_metrics_comparable_across_dimensions=False,
        original_TRAIN_input_bounds=observed, plot_bounds=bounds,
        plot_bounds_use_TEST_or_labels=False, root_search_performed=False,
        root_search_reason="本轮以共同特征预测和闭合诊断为主，未进行一维或二维不动点搜索，也不按根数选模型",
        cached_prediction_max_abs_difference=checks, feature_metric_recomputation_max_abs_difference=maximum_metric_difference,
        input_sha256=pins, all_frozen_parameter_fingerprints_unchanged=True,
        Q_encoder_invoked=False, C_invoked=False, physical_fields_loaded=False,
        affine_decoder_only_for_cached_predictions=True, checkpoint_loaded=False,
        new_training_updates=0, new_physical_samples=0, CPU_threads=1, GPU_seconds=0,
        cpu_seconds=time.process_time()-started, cpu_budget_seconds=CPU_BUDGET_SECONDS,
        output_budget_bytes=OUTPUT_BUDGET_BYTES, critical_r=None, thermal_nu=None,
        physical_fixed_point_certified=False,
        limitations=["显式冻结C、有限特征缓存的Q维度消融，不是随机可训练C的完整默认QCR复现实验。",
            "Q训练3000步与后续R训练6000步分别记账，不能称为QCR训练9000步。",
            "Q见过全部八个TRAIN条件；两个R验证条件仅相对R拟合独立，且验证参与端点选择。",
            "本轮只有一个配对Q初始化种子；三个R种子是优化重复，不是三个独立QCR实验。",
            "不同维度的潜变量坐标误差与loss不直接比较；主要比较相同105维目标的仿射解码误差。",
            "四次编码重复复用物理链与C，不能当作新增独立物理样本；保留全部原采样警告。",
            "TEST已被历史实验查看，仅作探索性留出评价；不用于挑选维度、种子、训练端点或绘图边界。",
            "二维g的坐标是网络原始坐标；方形网格覆盖TRAIN包围盒内无样本区域，箭头不代表这些位置已有物理支撑。",
            "本轮未搜索数值不动点，不认证相边界、物理临界点、热方向或临界指数。"])
    write_json(output/"analysis.json", result)
    write_json(output/"status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, analysis_sha256=sha256(output/"analysis.json"),
        cpu_seconds=time.process_time()-started))
    guard()
    write_json(output/"public_files.json", {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description="固定C的phi4一维／二维Q表示对照报告")
    parser.add_argument("--training", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.training, args.features, args.output)
    print(json.dumps(dict(status=result["status"], cpu_seconds=result["cpu_seconds"])))


if __name__ == "__main__":
    main()
