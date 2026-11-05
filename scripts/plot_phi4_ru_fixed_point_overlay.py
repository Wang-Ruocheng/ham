"""Mark bounded numerical fixed points on the previous frozen-R arrow plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
import torch

from recurrent_rg.phi4_ru_flow import classify_eigenvalues, full_map_and_jacobian
from scripts import plot_phi4_ru_cached_directions as cached
from scripts import plot_phi4_ru_full_grid_directions as previous
from scripts.report_phi4_qcr import _plotting


STYLE = {
    "attracting": ("吸引点", "o", "#257744"),
    "repelling": ("排斥点", "^", "#ba3333"),
    "saddle": ("鞍点", "X", "#9b4baf"),
    "neutral_or_unresolved": ("近中性／未定", "D", "#ad7909"),
}


def find_roots(model, bounds, clouds=(), *, nx=11, ny=15, max_nfev=128):
    """Bounded multistart search, not a proof of exhaustive root enumeration.

    Solve the unnormalized vector R(g)-g (in fixed plotting-frame units),
    never the unit-length display arrows. DR includes the residual identity.
    Only input derivatives are taken; all model parameters stay frozen.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    previous.make_grid(bounds, nx, ny)  # Validate the same rectangular frame.
    center, half = bounds.mean(1), np.diff(bounds, axis=1).ravel() / 2
    normalized_grid = np.stack(np.meshgrid(np.linspace(-.999, .999, nx),
        np.linspace(-.999, .999, ny)), axis=-1).reshape(-1, 2)
    starts = list(normalized_grid)
    for cloud in clouds:
        value = (np.asarray(cloud).mean(0) - center) / half
        if np.all(np.abs(value) < 1):
            starts.append(value)
    starts = np.unique(np.asarray(starts), axis=0)
    roots, attempts = [], []
    residual_tolerance, deduplication_tolerance = 1e-9, 1e-6

    def function(z):
        point = center + half * z
        return (previous.mapped(model, point[None])[0] - point) / half

    def jacobian(z):
        _, dr = full_map_and_jacobian(model, (center + half * z)[None], "cpu")
        return (dr[0] - np.eye(2)) * half[None, :] / half[:, None]

    for start in starts:
        result = least_squares(function, start, jac=jacobian,
            bounds=(-np.ones(2), np.ones(2)), max_nfev=max_nfev,
            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        point = center + half * result.x
        scaled = function(result.x)
        accepted = bool(np.isfinite(scaled).all()
                        and np.max(np.abs(scaled)) <= residual_tolerance)
        attempts.append(dict(start_g=(center + half * start).tolist(),
            optimizer_success=bool(result.success), optimizer_status=int(result.status),
            nfev=int(result.nfev), njev=int(result.njev or 0),
            scaled_residual_inf=float(np.max(np.abs(scaled))), accepted=accepted))
        if not accepted:
            continue
        match = next((root for root in roots
            if np.linalg.norm((point - np.asarray(root["g"])) / half)
            <= deduplication_tolerance), None)
        if match is not None:
            match["converged_starts"] += 1
            continue
        _, dr = full_map_and_jacobian(model, point[None], "cpu")
        # An independent central difference audit in scaled frame units.
        h = 1e-5
        fd = np.column_stack([(previous.mapped(model, (point + half[axis] * h
            * np.eye(2)[axis])[None])[0] - previous.mapped(model,
            (point - half[axis] * h * np.eye(2)[axis])[None])[0]) / (2 * h * half[axis])
            for axis in range(2)])
        if not np.allclose(fd, dr[0], rtol=1e-6, atol=1e-7):
            raise RuntimeError("不动点完整DR的中心差分核对失败")
        eigenvalues = np.linalg.eigvals(dr[0])
        classification, contractions, expansions = classify_eigenvalues(eigenvalues)
        cloud_points = np.concatenate(clouds) if len(clouds) else np.empty((0, 2))
        singular = np.linalg.svd(jacobian(result.x), compute_uv=False)
        roots.append(dict(g=point.tolist(), residual=float(np.linalg.norm(scaled * half)),
            scaled_residual_inf=float(np.max(np.abs(scaled))), full_DR=dr[0].tolist(),
            DR_finite_difference_max_abs_error=float(np.max(np.abs(fd - dr[0]))),
            eigenvalues=[dict(real=float(v.real), imag=float(v.imag), modulus=float(abs(v)))
                         for v in eigenvalues], classification=classification,
            contracting_directions=contractions, expanding_directions=expansions,
            full_DR_includes_identity=True, eigenvalue_modulus_tolerance=1e-5,
            scaled_root_jacobian_singular_values=singular.tolist(),
            locally_nonsingular=bool(singular[-1] > 1e-8),
            inside_evaluation_local_envelope=bool(previous.coverage_mask(point[None], clouds)[0]),
            nearest_evaluation_sample_distance=(float(np.linalg.norm(cloud_points - point, axis=1).min())
                                                if len(cloud_points) else None),
            distance_to_frame_boundary_scaled=float(np.min(1 - np.abs(result.x))),
            converged_starts=1, physical_critical_point_certified=False, thermal_nu=None))
    roots.sort(key=lambda item: tuple(item["g"]))
    for number, root in enumerate(roots, 1):
        root["id"] = number
    return roots, dict(bounds=bounds.tolist(), grid_starts=[nx, ny],
        extra_starts="本次评价条件局部点云的均值", actual_starts=len(starts),
        max_nfev_per_start=max_nfev, scaled_residual_inf_tolerance=residual_tolerance,
        normalized_deduplication_tolerance=deduplication_tolerance,
        attempts=attempts, exhaustive_root_count_certified=False,
        search_only_inside_previous_plot_frame=True,
        stability_rule="离散R映射的完整DR特征值模；不是DR-I的连续流稳定性")


def draw(bounds, grid, raw, length, clouds, roots, seed, path):
    plt = _plotting(None)
    from matplotlib.patches import Patch, Polygon
    from matplotlib.lines import Line2D
    covered = previous.coverage_mask(grid, clouds)
    delta, keep = cached.directions(raw, length)
    fig, ax = plt.subplots(figsize=(7.3, 8.9))
    for cloud in clouds:
        try:
            hull = ConvexHull(cloud)
            ax.add_patch(Polygon(cloud[hull.vertices], closed=True, facecolor="#a4abb2",
                edgecolor="none", alpha=.32, zorder=1))
        except QhullError:
            pass
        ax.scatter(cloud[:, 0], cloud[:, 1], s=5, color="#8e959d", alpha=.30,
                   edgecolors="none", zorder=2)
    for mask, opacity in ((keep & ~covered, .58), (keep & covered, .95)):
        ax.quiver(grid[mask, 0], grid[mask, 1], delta[mask, 0], delta[mask, 1],
            angles="xy", scale_units="xy", scale=1, pivot="mid", color="#26323d",
            alpha=opacity, width=.0021, headwidth=4.1, headlength=5., zorder=3)
    handles = [Patch(facecolor="#a4abb2", alpha=.5, label="本次评价数据覆盖")]
    for classification in STYLE:
        entries = [root for root in roots if root["classification"] == classification]
        if not entries:
            continue
        name, marker, color = STYLE[classification]
        handles.append(Line2D([], [], marker=marker, linestyle="none", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.5, markersize=9, label=name))
        for root in entries:
            point = root["g"]
            ax.scatter(*point, marker=marker, s=130, linewidths=1.8, edgecolors=color,
                facecolors=color if root["inside_evaluation_local_envelope"] else "white", zorder=6)
            ax.annotate(f"{name}{root['id']}", point, xytext=(9, 8), textcoords="offset points",
                color=color, fontsize=10, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
    if not roots:
        ax.text(.03, .97, "当前画框内未检出满足残差阈值的不动点", transform=ax.transAxes,
                va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel=r"潜变量 $g_1$", ylabel=r"潜变量 $g_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"全区域流向与数值不动点：种子{seed}", y=.98)
    fig.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)),
               bbox_to_anchor=(.53, .946), frameon=False)
    fig.text(.5, .079, r"箭头：$R(g)-g$ 的等长方向；标记满足 $R(g)=g$。", ha="center", fontsize=10)
    fig.text(.5, .054, "空心标记：位于本次评价局部包络外；不等于训练分布外。", ha="center", fontsize=9)
    fig.text(.5, .029, "类型由完整DR的特征值模判定；有限多起点搜索不保证找全。", ha="center", fontsize=9)
    fig.text(.5, .005, "这些是冻结网络的数值不动点，不是已认证的物理临界点。",
             ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, .106, 1, .92))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(root, validation, previous_plot, output):
    started = time.process_time()
    root, validation, previous_plot, output = map(lambda p: Path(p).resolve(),
        (root, validation, previous_plot, output))
    if output.exists():
        raise FileExistsError("使用新目录，不覆盖上一幅图或正式报告")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    direct, labels, predictions, groups, summary, pins = cached.read_cached(root)
    clouds, _, _, _ = cached.geometry(direct, labels, predictions, groups)
    old = json.loads(previous_plot.read_text())
    if old.get("status") != "complete" or old.get("returncode") != 0:
        raise ValueError("需要上一幅已完成全区域方向图")
    for path, expected in old["input_sha256"].items():
        if cached.sha(path) != expected:
            raise ValueError("上一幅图的输入或实现已变化")
        pins[path] = expected
    for name, expected in old["output_sha256"].items():
        path = previous_plot.parent / name
        if cached.sha(path) != expected:
            raise ValueError("上一幅图的输出指纹已变化")
        pins[str(path)] = expected
    inventory = json.loads((validation / "public_files.json").read_text())
    for name in ("summary.json", "R_parameters.npz"):
        if cached.sha(validation / name) != inventory.get(name):
            raise ValueError("冻结R公开参数指纹不一致")
        pins[str(validation / name)] = inventory[name]
    expected_maps = json.loads((validation / "summary.json").read_text())["final_R_sha256"]
    import recurrent_rg.phi4_r_calibration as map_module
    import recurrent_rg.phi4_ru_flow as flow_module
    import scripts.report_phi4_qcr as plotting_module
    for path in (previous_plot, validation / "public_files.json", Path(__file__).resolve(),
        Path(cached.__file__).resolve(), Path(previous.__file__).resolve(),
        Path(map_module.__file__).resolve(), Path(flow_module.__file__).resolve(),
        Path(plotting_module.__file__).resolve()):
        pins[str(path)] = cached.sha(path)
    bounds = np.asarray(old["bounds"], dtype=float)
    grid = np.asarray(old["grid_points"], dtype=float)
    length = float(old["arrow_length_data_units"])
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    for seed in cached.SEEDS:
        key = f"selected_seed{seed}"
        if (expected_maps[key] != summary["fixed_R_sha256"][key]
                or expected_maps[key] != old["models"][str(seed)]["fixed_R_sha256"]):
            raise ValueError("冻结R与上一幅图／正式报告端点不一致")
        model = previous.restore_model(validation / "R_parameters.npz", key, expected_maps[key])
        check = previous.mapped(model, direct[:, :, :2].reshape(-1, 2))
        target = predictions[seed].reshape(-1, 2)
        if not np.allclose(check, target, rtol=1e-10, atol=1e-12):
            raise RuntimeError("冻结R与保存的评价预测不一致")
        raw = previous.mapped(model, grid) - grid
        if not np.allclose(raw, old["models"][str(seed)]["grid_raw_delta"], rtol=1e-10, atol=1e-12):
            raise RuntimeError("网格流向与上一幅图不一致")
        roots, search = find_roots(model, bounds, clouds)
        draw(bounds, grid, raw, length, clouds, roots, seed,
             output / f"full_grid_fixed_points_seed{seed}.png")
        if previous.fingerprint(model) != expected_maps[key] or any(
                p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("求根或绘图期间冻结模型参数／梯度状态发生变化")
        models[str(seed)] = dict(model_key=key, fixed_R_sha256=expected_maps[key], roots=roots,
            search=search, model_unchanged=True, gradients_none=True,
            cached_prediction_max_abs_difference=float(np.max(np.abs(check - target))))
        print(json.dumps(dict(seed=seed, roots=roots), ensure_ascii=False), flush=True)
    if any(cached.sha(path) != value for path, value in pins.items()):
        raise RuntimeError("不可变输入在求根或绘图期间发生变化")
    receipt = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        bounds=bounds.tolist(), previous_plot_preserved=True, grid_directions_unchanged=True,
        input_sha256=pins, models=models, root_search=True, physical_critical_point_certified=False,
        coverage_means="本次独立评价局部点云包络，不代表完整训练支持域",
        training_updates=0, new_physical_samples=0, Q_or_C_loaded=False,
        cpu_seconds=time.process_time() - started,
        output_sha256={p.name: cached.sha(p) for p in output.glob("*.png")})
    (output / "plot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--previous-plot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.run, args.validation, args.previous_plot, args.output)
    print(json.dumps(dict(status=receipt["status"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
