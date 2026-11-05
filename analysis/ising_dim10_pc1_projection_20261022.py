"""Replot saved ten-dimensional arrays only; never load a model or checkpoint.

The scalar intersections below interpolate already saved input/output pairs.
They are not extra evaluations of R, full-coordinate roots, or an autonomous
one-dimensional physical model. The frozen formal report is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.report_ising_zh import chinese_figures


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing analysis")
    names = ("trajectories.npz", "metrics.json", "fixed_points.json")
    hashes = {name: digest(source / name) for name in names}
    metrics = json.loads((source / "metrics.json").read_text())
    roots = json.loads((source / "fixed_points.json").read_text())
    assert metrics["theory_dim"] == 10 and metrics["checkpoint_epoch"] == 12000
    with np.load(source / "trajectories.npz", allow_pickle=False) as saved:
        g = saved["mean_by_scale"]
        rg = saved["R_of_mean"]
        mean_r = saved["mean_R"]
        center = saved["pca_center"]
        vector = saved["pca_components"][0]
        coupling = saved["K"]
        saved_projected = saved["projected_mean_by_scale"][..., 0]
        saved_delta = saved["projected_true_displacement"][..., 0]
    assert g.shape == rg.shape == mean_r.shape == (64, 3, 10)
    assert np.isfinite(g).all() and np.isfinite(rg).all()
    assert np.allclose(vector @ vector, 1.0, rtol=0, atol=1e-12)
    x = (g - center) @ vector
    y = (rg - center) @ vector
    y_mean_r = (mean_r - center) @ vector
    displacement = rg - g
    delta = displacement @ vector
    perpendicular = displacement - delta[..., None] * vector
    hidden_norm = np.linalg.norm(perpendicular, axis=-1)
    assert np.allclose(x, saved_projected, rtol=0, atol=1e-12)
    assert np.allclose(delta, y - x, rtol=0, atol=1e-12)
    assert np.allclose(delta, saved_delta, rtol=0, atol=1e-12)
    assert np.allclose(perpendicular @ vector, 0.0, rtol=0, atol=1e-12)
    assert np.allclose(np.linalg.norm(displacement, axis=-1) ** 2,
                       delta ** 2 + hidden_norm ** 2, rtol=1e-12, atol=1e-15)

    intersections = []
    for scale in range(3):
        for left in np.flatnonzero(delta[:-1, scale] * delta[1:, scale] < 0):
            right = left + 1
            fraction = -delta[left, scale] / (delta[right, scale] - delta[left, scale])
            blend = lambda values: (1 - fraction) * values[left, scale] + fraction * values[right, scale]
            residual = blend(displacement)
            projected_x = float(blend(x))
            hidden = residual - (residual @ vector) * vector
            dx = x[right, scale] - x[left, scale]
            assert abs(residual @ vector) < 1e-12
            intersections.append(dict(
                scale=scale, left_saved_index=int(left), right_saved_index=int(right),
                interpolation_fraction=float(fraction), projected_x=projected_x,
                evaluation_only_K=float((1 - fraction) * coupling[left] + fraction * coupling[right]),
                sampled_path_secant_slope=float((y[right, scale] - y[left, scale]) / dx),
                interpolated_output_pair_pc1_displacement=float(residual @ vector),
                interpolated_output_pair_full_displacement_norm=float(np.linalg.norm(residual)),
                interpolated_output_pair_pc1_orthogonal_norm=float(np.linalg.norm(hidden)),
                actual_R_at_interpolated_input_evaluated=False,
                physical_fixed_point_certified=False))
    assert len(intersections) == 3
    assert {r["scale"] for r in intersections} == {0, 1, 2}
    output.mkdir(parents=True, exist_ok=False)
    colors = ["#2474a8", "#e07b17", "#289b42"]
    projected_root = [(np.asarray(r["g"]) - center) @ vector for r in roots["roots"]]

    with chinese_figures():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), layout="constrained")
        for ax in axes:
            for scale, color in enumerate(colors):
                ax.plot(x[:, scale], y[:, scale], ".-", ms=3, lw=1.8,
                        color=color, label=f"尺度 k={scale}")
                ax.plot(x[:, scale], y_mean_r[:, scale], ":", lw=1.0,
                        color=color, alpha=.8)
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([low - .02, high + .06], [low - .02, high + .06],
                    "--", color=".35", lw=1.1, label="恒等映射")
            for row in intersections:
                value = row["projected_x"]
                ax.scatter(value, value, s=65, marker="o", facecolors="none",
                           edgecolors=colors[row["scale"]], zorder=6)
            ax.set(xlabel="输入第一主成分", ylabel="输出第一主成分")
            ax.grid(alpha=.25)
        for index, value in enumerate(projected_root):
            axes[0].scatter(value, value, marker="*", s=130, c="gold", edgecolors="black",
                            zorder=7, label="十维数值根的投影（支撑外）" if index == 0 else None)
        axes[0].set_title("完整十维映射沿第一主成分的投影")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], c=".3", ls=":", label="点线：先映射再平均"))
        labels.append("点线：先映射再平均")
        axes[0].legend(handles, labels, fontsize=9, loc="upper left")
        middle = float(np.mean([r["projected_x"] for r in intersections]))
        axes[1].set(xlim=(middle - .045, middle + .045),
                    ylim=(middle - .065, middle + .065), title="投影交点放大；圆圈不是完整不动点")
        axes[1].annotate("样本路径割线斜率约 1.62 至 1.64", xy=(middle, middle),
                         xytext=(middle - .040, middle + .047), fontsize=10,
                         arrowprops=dict(arrowstyle="->", color=".3"))
        fig.suptitle("十维终点：训练输入拟合的第一主成分解释 99.778% 方差\n"
                     "实线：平均编码输入的完整映射；点线：编码逐样本映射后平均", fontsize=13)
        fig.savefig(output / "R_pc1_projection.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
        for scale, color in enumerate(colors):
            axes[0].plot(x[:, scale], delta[:, scale], ".-", ms=3, lw=1.6,
                         color=color, label=f"尺度 k={scale}")
            axes[1].plot(x[:, scale], hidden_norm[:, scale], ".-", ms=3, lw=1.6,
                         color=color, label=f"尺度 k={scale}")
        for row in intersections:
            col = colors[row["scale"]]
            axes[0].scatter(row["projected_x"], 0., s=55, facecolors="none", edgecolors=col, zorder=5)
            axes[1].scatter(row["projected_x"], row["interpolated_output_pair_pc1_orthogonal_norm"],
                            s=55, facecolors="none", edgecolors=col, zorder=5)
        axes[0].axhline(0, color=".3", ls="--", lw=1)
        axes[0].set(title="第一主成分上的位移", ylabel="输出减输入的第一主成分位移")
        axes[1].set(title="第一主成分投影隐藏的位移", ylabel="完整位移在第一主成分正交补中的范数")
        for ax in axes:
            ax.set_xlabel("输入第一主成分")
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
        fig.suptitle("圆圈：已保存相邻输入输出对的线性插值；没有新增模型推理\n"
                     "第一主成分正交补位移不等于相对弯曲流形的法向误差", fontsize=12)
        fig.savefig(output / "R_pc1_residuals.png", dpi=170)
        plt.close(fig)

    assert hashes == {name: digest(source / name) for name in names}
    result = dict(
        status="complete", source=str(source), source_sha256=hashes,
        source_unchanged=True, checkpoint_loaded=False, model_inference_performed=False,
        training_started=False, full_coordinate_root_search_performed=False,
        pc1_fit="Frozen TRAIN inputs at scales 0 and 1 only; unchanged from formal report",
        x_definition="v1 dot (mean_g - TRAIN_center)",
        y_definition="v1 dot (R(mean_g) - TRAIN_center), using saved arrays",
        dotted_definition="v1 dot (mean_R - TRAIN_center), using saved arrays",
        pc1_explained_variance=float(metrics["pca"]["explained_variance_fraction"][0]),
        intersections=intersections,
        geometric_cautions=[
            "A projected crossing is not a full-coordinate fixed point.",
            "Secant slopes along saved curves are not eigenvalues of an autonomous reduced model.",
            "Linear interpolation does not evaluate R at the interpolated input.",
            "The PC1-orthogonal complement is not the normal space of a curved physical manifold.",
            "Scale 2 mapping is a saved R diagnostic, not an extra physical coarse-graining validation.",
            "No uncertainty interval is inferred from this deterministic replot."],
        checks_passed=["PC1 unit norm", "saved projection reproduced", "centering consistent",
                       "orthogonal residual", "Pythagorean identity", "three scalar crossings",
                       "crossing scalar residual zero", "source SHA unchanged"],
        public_files=["R_pc1_projection.png", "R_pc1_residuals.png", "projection_diagnostics.json"])
    (output / "projection_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
