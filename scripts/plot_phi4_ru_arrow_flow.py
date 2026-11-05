"""Minimal arrow-and-scatter portrait of the saved best phi4 R endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from scripts.plot_phi4_ru_best_flow import mapped, restore
from scripts.report_phi4_qcr import _plotting


def panel(model, direct, bounds, path, *, zoom=False):
    plt = _plotting(None)
    fig, ax = plt.subplots(figsize=(7.3, 9.2) if not zoom else (8.3, 8.3))
    nx, ny = (15, 25) if not zoom else (17, 21)
    xs = np.linspace(*bounds[0], nx + 2)[1:-1]
    ys = np.linspace(*bounds[1], ny + 2)[1:-1]
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    delta = mapped(model, grid) - grid
    norms = np.linalg.norm(delta, axis=1)
    spacing = min(xs[1] - xs[0], ys[1] - ys[0])
    # One positive constant scales every vector in this panel. Components are
    # still the exact R(g)-g; neither direction nor relative magnitude changes.
    scale = float(np.quantile(norms, .90) / (.78 * spacing))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("A finite positive common arrow scale is required")
    arrows = ax.quiver(grid[:, 0], grid[:, 1], delta[:, 0], delta[:, 1],
        angles="xy", scale_units="xy", scale=scale, pivot="mid",
        color="#36434c", alpha=.85, width=.0026, headwidth=3.8, headlength=4.8,
        headaxislength=4.2, zorder=2)
    reference = float(10 ** np.round(np.log10(np.quantile(norms, .90))))
    ax.quiverkey(arrows, .97, .035, reference, f"位移 {reference:g}",
        labelpos="W", coordinates="axes", fontproperties={"size": 9})
    for k, color in enumerate(("#4387bb", "#df9c47", "#9d7cb5")):
        value = direct[..., k, :].reshape(-1, 2)
        label = f"k={k}" + ("（目标）" if k == 2 else "（输入）")
        ax.scatter(value[:, 0], value[:, 1], s=9, c=color, alpha=.40,
            edgecolors="none", label=label, zorder=3)
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="g₁", ylabel="g₂",
           title="局部流场" if zoom else "重整化流场：归一化256，种子93")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False, markerscale=1.8,
              handletextpad=.35, borderpad=.2)
    fig.text(.5, .025, "箭头：R(g)−g，统一比例缩放；散点：各尺度重复编码", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"grid_points": len(grid), "quiver_scale": scale,
            "scale_units": "xy", "individual_vector_normalization": False,
            "reference_displacement": reference, "bounds": bounds.tolist()}


def run(root, analysis, output):
    root, analysis, output = map(lambda p: Path(p).resolve(), (root, analysis, output))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    input_paths = [root / "evaluation_arrays.npz", root / "R_parameters.npz",
                   root / "summary.json", analysis / "analysis.json"]
    fingerprints = {str(p): sha256_file(p) for p in input_paths}
    saved = json.loads((analysis / "analysis.json").read_text())
    key = saved["selected_model"]
    model = restore(root, key)
    initial_sha = state_fingerprint(model)
    expected = json.loads((root / "summary.json").read_text())["final_R_sha256"][key]
    if initial_sha != expected:
        raise ValueError("Saved model fingerprint mismatch")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        # Repeat/chain axes are retained; every shown point is a saved encoding,
        # not a hull fill, interpolation, mean trajectory or synthetic sample.
        direct = np.concatenate((archive["train_direct"], archive["test_direct"]), axis=1)
    bounds = np.asarray(saved["models"][key]["bounded_search"]["bounds"])
    roots = saved["models"][key]["roots"]
    center = np.asarray(roots[0]["g"]) if roots else direct[..., :2, :].mean((0, 1, 2))
    zoom = np.column_stack((center - [.028, .036], center + [.028, .036]))
    receipt = dict(model=key, primary=panel(model, direct, bounds, output / "arrow_flow.png"),
        zoom=panel(model, direct, zoom, output / "arrow_flow_zoom.png", zoom=True),
        plots_contain="arrows and saved data scatter only; no inferred boundary overlays",
        raw_vector_components="R(g)-g", plotting_uses_no_root_search=True,
        new_training=False, inputs_sha256=fingerprints, model_sha256=initial_sha)
    if state_fingerprint(model) != initial_sha or any(sha256_file(p) != v for p, v in fingerprints.items()):
        raise RuntimeError("Immutable model or inputs changed")
    write_json(output / "plot_receipt.json", receipt)
    print(json.dumps({"output": str(output), "model": key}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.analysis, args.output)


if __name__ == "__main__":
    main()
