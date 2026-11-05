"""Chinese phase portrait of one fixed TRAIN-selected calibrated R endpoint.

This is a new post-hoc analysis. It never changes training artifacts, uses no
labels for model/root selection, and distinguishes a bounded numerical saddle
from a physically supported phase boundary. Arrows are the discrete R(g)-g;
continuous streamlines are deliberately not used as RG trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import torch

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_r_calibration import FrozenResidualMap
from recurrent_rg.phi4_ru_flow import bounded_flow
from scripts.report_phi4_qcr import _plotting


def restore(root, key):
    with np.load(root / "R_parameters.npz", allow_pickle=False) as archive:
        state = {name[len(key) + 1:]: torch.tensor(archive[name])
                 for name in archive.files if name.startswith(key + ".")}
    hidden = int(state["net.0.weight"].shape[0])
    model = FrozenResidualMap(2, hidden, center=state["center"], scale=state["scale"])
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def support(points):
    center, scale = points.mean(0), points.std(0, ddof=0)
    normalized = (points - center) / scale
    hull = ConvexHull(normalized)
    tree = cKDTree(normalized)
    # This describes proximity to the fixed repeated TRAIN input cloud. It is
    # not a confidence region or an equilibrium/identifiability criterion.
    radii = tree.query(normalized, k=9)[0][:, -1]
    radius = float(np.quantile(radii, .95))

    def inspect(query):
        z = (np.atleast_2d(query) - center) / scale
        inside = np.all(z @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)
        distance = tree.query(z)[0]
        return inside, distance, distance <= radius

    return inspect, dict(center=center.tolist(), scale=scale.tolist(),
        near_radius_standardized=radius, near_radius_rule="95th percentile of TRAIN 8th-neighbor distances",
        hull_area_raw=float(hull.volume * np.prod(scale)),
        proximity_is_not_physical_certification=True)


@torch.no_grad()
def mapped(model, grid):
    values = [model.advance(torch.tensor(grid[k:k + 512], dtype=torch.float64)).numpy()
              for k in range(0, len(grid), 512)]
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite frozen map")
    return result


def draw_panel(ax, model, train, test, inspect, bounds, roots, *, title, local=False):
    # Colormap describes the residual norm, not a phase label or basin label.
    x, y = np.linspace(*bounds[0], 121), np.linspace(*bounds[1], 181)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    values = mapped(model, grid)
    delta = values - grid
    speed = np.linalg.norm(delta, axis=1)
    inside, distance, near = inspect(grid)
    image = ax.pcolormesh(xx, yy, np.log10(np.maximum(speed.reshape(xx.shape), 1e-8)),
                          shading="auto", cmap="viridis", vmin=-4, vmax=-1, rasterized=True, zorder=0)
    outside = (~inside).reshape(xx.shape)
    sparse = (inside & ~near).reshape(xx.shape)
    ax.contourf(xx, yy, outside, levels=[.5, 1.5], colors=["#dddddd"], alpha=.48, zorder=1)
    if outside.any() and (~outside).any():
        ax.contour(xx, yy, inside.reshape(xx.shape), levels=[.5], colors=["#f1a340"], linewidths=1.2, zorder=4)
    if sparse.any():
        ax.contourf(xx, yy, sparse, levels=[.5, 1.5], colors=["white"], alpha=.45, zorder=2)
    # Arrows use raw data units and are not normalized. The background is
    # only a model prediction, with support qualification explicitly overlaid.
    qx, qy = np.linspace(*bounds[0], 11), np.linspace(*bounds[1], 17)
    qgrid = np.stack(np.meshgrid(qx, qy), axis=-1).reshape(-1, 2)
    endpoints = mapped(model, qgrid)
    qdelta = endpoints - qgrid
    visible = np.all((endpoints >= bounds[:, 0]) & (endpoints <= bounds[:, 1]), axis=1)
    qinside, _, qnear = inspect(qgrid)
    for mask, alpha in ((qinside & qnear, .85), (qinside & ~qnear, .46), (~qinside, .20)):
        mask &= visible
        ax.quiver(qgrid[mask, 0], qgrid[mask, 1], qdelta[mask, 0], qdelta[mask, 1],
            angles="xy", scale_units="xy", scale=1, color="#172a35", alpha=alpha,
            width=.003, headwidth=3.5, headlength=4.5, zorder=3)
    # Saved data show the mean over stochastic repeats, preserving physical
    # chain identity. Labels are never used here to choose points or bounds.
    for k, color in enumerate(("#57a0db", "#f2ac50", "#b179c1")):
        ax.scatter(train[:, k, 0], train[:, k, 1], c=color, s=14 if not local else 24,
            edgecolors="white", linewidths=.25, alpha=.75, zorder=5, label=f"训练点 k={k}")
        ax.scatter(test[:, k, 0], test[:, k, 1], c=color, s=17 if not local else 28,
            marker="x", linewidths=.85, alpha=.8, zorder=5)
    for i, root in enumerate(roots):
        point = np.asarray(root["g"])
        if not (bounds[0, 0] <= point[0] <= bounds[0, 1] and bounds[1, 0] <= point[1] <= bounds[1, 1]):
            continue
        ax.scatter(*point, marker="*", s=190, facecolors="none", edgecolors="#ef3340", linewidths=1.8, zorder=8)
        root_label = {"saddle": "数值鞍点", "attracting": "数值吸引点",
                      "repelling": "数值排斥点"}.get(root["classification"], "数值不动点")
        root_label += "（外推）" if not root["within_fit_train_hull"] else "（候选）"
        ax.annotate(root_label,
            point, xytext=(10, -23 if local else 10), textcoords="offset points",
            color="#b51025", fontsize=10, zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
        if local and root["classification"] == "saddle":
            values_e, vectors = np.linalg.eig(np.asarray(root["full_DR"]))
            stable = np.flatnonzero(np.abs(values_e) < 1)
            if len(stable) == 1 and np.max(np.abs(vectors[:, stable[0]].imag)) < 1e-8:
                direction = vectors[:, stable[0]].real
                direction /= np.linalg.norm(direction)
                radius = .22 * min(np.diff(bounds, axis=1).ravel())
                segment = point + np.array([[-radius], [radius]]) * direction
                ax.plot(segment[:, 0], segment[:, 1], "--", color="#ce2434", linewidth=1.7,
                    label="局部稳定方向（非已验证相边界）", zorder=7)
    ax.set(title=title, xlabel="原始坐标 g₁", ylabel="原始坐标 g₂",
           xlim=bounds[0], ylim=bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.12)
    return image, dict(points=len(grid), inside_hull=int(inside.sum()),
        near_train=int(near.sum()), speed_min=float(speed.min()), speed_max=float(speed.max()))


def run(run_root, cache_root, output):
    started = time.process_time()
    run_root, cache_root, output = map(lambda p: Path(p).resolve(), (run_root, cache_root, output))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    frozen = {}
    for root in (run_root, cache_root):
        inventory = json.loads((root / "public_files.json").read_text())
        for name, expected in inventory.items():
            path = root / name
            if sha256_file(path) != expected:
                raise ValueError(f"Source public SHA mismatch: {path}")
            frozen[str(path)] = expected
        frozen[str(root / "public_files.json")] = sha256_file(root / "public_files.json")
    history = json.loads((run_root / "history_progress.json").read_text())["history"]
    final = history[-1]
    if final["step"] != 6000:
        raise ValueError("Only fixed6000 endpoints can be selected")
    best = min(final["arms"], key=lambda key: final["arms"][key]["loss"])
    best96 = min((key for key in final["arms"] if key.startswith("scaled96_")),
                 key=lambda key: final["arms"][key]["loss"])
    summary = json.loads((run_root / "summary.json").read_text())
    with np.load(cache_root / "cache.npz", allow_pickle=False) as archive:
        fit = archive["fit_direct"].copy()
        train, test = archive["eval_train_direct"].copy(), archive["eval_test_direct"].copy()
    points = fit[..., :2, :].reshape(-1, 2)
    inspect, support_receipt = support(points)
    arrays = dict(train_direct=fit.reshape(-1, 3, 2), test_direct=test.reshape(-1, 3, 2))
    models, analyses, model_sha = {}, {}, {}
    for key in dict.fromkeys((best, best96)):
        model = restore(run_root, key)
        model_sha[key] = state_fingerprint(model)
        if model_sha[key] != summary["final_R_sha256"][key]:
            raise ValueError("Restored R state differs from the formal endpoint")
        # The fixed49-start bounded search reports all accepted distinct roots;
        # starting inside the hull does not force a converged root inside it.
        roots, receipt = bounded_flow(model, {k: v.copy() for k, v in arrays.items()}, torch.device("cpu"))
        for root in roots:
            inside, distance, near = inspect(np.asarray(root["g"])[None])
            root.update(within_fit_train_hull=bool(inside[0]),
                nearest_train_distance_standardized=float(distance[0]), near_fit_train=bool(near[0]))
        models[key], analyses[key] = model, dict(roots=roots, bounded_search=receipt)
    plt = _plotting(None)
    best_roots = analyses[best]["roots"]
    whole_bounds = np.asarray(analyses[best]["bounded_search"]["bounds"])
    if best_roots:
        center = np.asarray(best_roots[0]["g"])
    else:
        with torch.no_grad():
            residual = mapped(models[best], points) - points
        center = points[np.argmin(np.linalg.norm(residual, axis=1))]
    halfwidth = np.array([.028, .036])
    zoom = np.column_stack((center - halfwidth, center + halfwidth))
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), constrained_layout=True)
    panel_receipts = []
    for ax, bounds, title, local in ((axes[0], whole_bounds, "整体流场与训练覆盖", False),
                                     (axes[1], zoom, "候选分界附近放大", True)):
        image, receipt = draw_panel(ax, models[best], train.mean(0), test.mean(0), inspect,
            bounds, best_roots, title=title, local=local)
        panel_receipts.append(receipt)
        fig.colorbar(image, ax=ax, label="log₁₀ ‖R(g)−g‖", shrink=.64, pad=.025, extend="both")
    axes[0].legend(loc="upper left", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    if labels:
        axes[1].legend(handles[-1:], labels[-1:], loc="upper right", fontsize=8)
    fig.suptitle("最低 TRAIN 闭合目标的 R 流场：归一化256，种子93，固定6000步\n"
        "箭头为真实 R(g)−g；灰色为训练凸包外，浅白色为距训练点较远区域\n"
        "仅显示端点也在画框内的箭头；圆点／叉号为训练／留出链重复均值；虚线为局部稳定方向", fontsize=13)
    fig.savefig(output / "best_flow.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # A matched diagnostic of the best96 endpoint reveals whether the displayed
    # candidate structure depends on the particular fitted network.
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key in zip(axes, (best, best96)):
        image, _ = draw_panel(ax, models[key], train.mean(0), test.mean(0), inspect, zoom,
            analyses[key]["roots"], title="归一化256，种子93" if key == best else "归一化96，种子93", local=True)
        fig.colorbar(image, ax=ax, label="log₁₀ ‖R(g)−g‖", shrink=.64, pad=.02, extend="both")
    fig.suptitle("同一局部区域的模型敏感性对照\n真实 R(g)−g；灰色为训练凸包外，浅白色为稀疏区；虚线为局部稳定方向", fontsize=13)
    fig.savefig(output / "flow_comparison.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for key, model in models.items():
        if state_fingerprint(model) != model_sha[key] or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen R changed during analysis")
    if any(sha256_file(path) != digest for path, digest in frozen.items()):
        raise RuntimeError("An immutable training/cache artifact changed")
    result = dict(status="complete", workflow_complete=True, returncode=0,
        selected_model=best, selection="minimum TRAIN all-start closure objective among fixed6000 endpoints",
        selected_training_loss=final["arms"][best]["loss"],
        comparison_model=best96, comparison_training_loss=final["arms"][best96]["loss"],
        selection_used_TEST=False, selection_used_coupling_labels=False,
        new_training=False, Q_loaded=False, C_loaded=False, GPU_used=False,
        model_state_sha256=model_sha, immutable_input_sha256=frozen,
        source_sha256={str(Path(__file__).resolve()): sha256_file(__file__)},
        support=support_receipt, selected_model_panels=panel_receipts, models=analyses,
        thermal_nu=None, physical_phase_boundary_certified=False,
        qualification="Post-hoc finite49-start bounded numerical search; absence of roots is not exhaustive; extrapolated saddle/local eigendirection does not establish a physical repelling phase boundary",
        cpu_seconds=time.process_time() - started)
    write_json(output / "analysis.json", result)
    write_json(output / "public_files.json", {p.name: sha256_file(p) for p in output.iterdir()
        if p.is_file() and p.name != "public_files.json"})
    print(json.dumps({"selected_model": best, "output": str(output),
                      "roots": {k: v["roots"] for k, v in analyses.items()}}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.cache, args.output)


if __name__ == "__main__":
    main()
