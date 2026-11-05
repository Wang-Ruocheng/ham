"""Add geometric coverage outlines using saved public flow arrays only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(run, output):
    run = Path(run).resolve()
    output = Path(output).resolve()
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"]
    names = ("evaluation_arrays.npz", "roots.json")
    pins = {name: sha(run / name) for name in names}
    for name, digest in pins.items():
        assert digest == status["public_files"][name], name
    roots = json.loads((run / "roots.json").read_text())["roots"]
    with np.load(run / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ("direct_g", "splits", "flow_g", "flow_displacement", "flow_bounds")}
    g = arrays["direct_g"]
    assert g.shape == (180, 3, 2) and np.isfinite(g).all()
    train = arrays["splits"] == "train"
    grid, drift, bounds = (arrays[key] for key in
                           ("flow_g", "flow_displacement", "flow_bounds"))
    assert grid.shape == drift.shape == (1089, 2)
    assert np.isfinite(grid).all() and np.isfinite(drift).all()

    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next(name for name in ("Arial Unicode MS", "Heiti TC", "Noto Sans CJK SC")
                if name in installed)
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.8, 8.5), layout="constrained")
    length = np.linalg.norm(drift, axis=1, keepdims=True)
    display = drift / np.maximum(length, 1e-15) * .65 * np.min(np.ptp(grid, axis=0)) / 32
    ax.quiver(grid[:, 0], grid[:, 1], display[:, 0], display[:, 1], angles="xy",
              scale_units="xy", scale=1, width=.0018, color="#6d7885", alpha=.65,
              zorder=1)
    ax.add_patch(Rectangle(bounds[:, 0], *np.diff(bounds, axis=1)[:, 0],
                           fill=False, color="#8e949b", linewidth=1, linestyle=":", zorder=2))

    colors = ("#0072B2", "#E69F00", "#009E73")
    handles, coverage = [], []
    for k, color in enumerate(colors):
        points = g[:, k]
        hull = ConvexHull(points)
        polygon = points[np.r_[hull.vertices, hull.vertices[0]]]
        line, = ax.plot(*polygon.T, color=color, linewidth=2.1, zorder=4)
        line.set_path_effects([patheffects.Stroke(linewidth=3.8, foreground="white"),
                              patheffects.Normal()])
        for mask, marker in ((train, "o"), (~train, "x")):
            ax.scatter(*points[mask].T, s=15 if marker == "o" else 22,
                       marker=marker, color=color, alpha=.78, zorder=5)
        handles.append(Line2D([], [], color=color, lw=2.1,
                              label=f"k={k}，L={36 // 3**k}：数据外包络"))
        coverage.append({"scale": k, "includes_train_and_test": True,
                         "points": len(points), "convex_hull_vertices": polygon[:-1].tolist()})

    inputs = g[train, :2].reshape(-1, 2)
    input_hull = ConvexHull(inputs)
    input_polygon = inputs[np.r_[input_hull.vertices, input_hull.vertices[0]]]
    ax.plot(*input_polygon.T, color="#333333", lw=1.5, linestyle="--", zorder=6)
    handles.extend([
        Line2D([], [], color="#333333", lw=1.5, ls="--", label="R 训练输入外包络（k=0,1）"),
        Line2D([], [], color="#8e949b", lw=1, ls=":", label="已保存箭头的网格范围"),
        Line2D([], [], color="#555555", marker="o", linestyle="none", ms=5, label="训练数据"),
        Line2D([], [], color="#555555", marker="x", linestyle="none", ms=6, label="留出数据"),
    ])
    classifications = {"saddle": "鞍点", "attracting": "吸引点", "repelling": "排斥点",
                       "neutral_or_unresolved": "近中性或未定"}
    for index, root in enumerate(roots, 1):
        point = np.asarray(root["g"])
        inside = bool(Delaunay(inputs).find_simplex(point, tol=1e-10) >= 0)
        assert inside == root["within_train_hull"]
        ax.scatter(*point, marker="D", s=95, facecolor="black" if inside else "white",
                   edgecolor="black", linewidth=1.6, zorder=9)
        ax.annotate(f"{classifications[root['classification']]} {index}"
                    + ("" if inside else "（训练输入外推）"), xy=point,
                    xytext=(-5, -33), textcoords="offset points", ha="left", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "black", "lw": .8},
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 2}, zorder=10)
    extents = np.vstack([g.reshape(-1, 2), bounds.T,
                         np.array([root["g"] for root in roots]).reshape(-1, 2)])
    padding = .045 * np.ptp(extents, axis=0)
    ax.set(xlim=(extents[:, 0].min() - padding[0], extents[:, 0].max() + padding[0]),
           ylim=(extents[:, 1].min() - padding[1], extents[:, 1].max() + padding[1]),
           xlabel="$g_1$", ylabel="$g_2$", aspect="equal",
           title="共享 R 流场与数据覆盖框线\n箭头为 R(g)−g 的方向（等长显示）")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.grid(alpha=.13)
    fig.supxlabel("彩色框线为各尺度训练与留出点的凸包，不是相边界或置信区间。\n"
                  "包络内部也可能缺少数据；扩展显示完整数据，矩形点线外未新增流场计算。",
                  fontsize=10)
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "R_flow_coverage.png", dpi=190)
    plt.close(fig)
    for name, digest in pins.items():
        assert sha(run / name) == digest
    receipt = {"source": str(run), "source_sha256": pins, "new_model_inference": False,
               "new_root_search": False, "coverage_method": "per-scale convex hull, train plus test",
               "coverage": coverage, "train_input_hull_vertices": input_polygon[:-1].tolist(),
               "original_flow_bounds": bounds.tolist(), "original_report_preserved": True,
               "figure_sha256": sha(output / "R_flow_coverage.png")}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figure": str(output / "R_flow_coverage.png"),
                      "source_verified": True, "new_model_inference": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.output)
