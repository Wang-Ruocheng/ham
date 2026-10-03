#!/usr/bin/env python3
"""Interpolate the microscopic TFIM ensemble coordinate against h/J."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.run / "summary.json").read_text())["config"]
    if config["theory_dim"] != 1:
        raise ValueError("Expected scalar theory coordinate")
    model = RecurrentRGNetwork1D(levels=config["levels"], hidden=config["hidden"],
                                theory_dim=1, learnable_kernel=not config.get("fixed_kernel", False),
                                kernel_layout=config.get("kernel_layout", "overlapping")).double()
    model.load_state_dict(torch.load(args.run / "qcr_model.pt", weights_only=True), strict=False)
    model.eval()
    train_x, train_h, test_x, test_h = torch.load(args.cache, weights_only=True)
    x = torch.cat((train_x, test_x))
    h = torch.cat((train_h, test_h)).numpy()
    samples = x.shape[1]
    values = []
    with torch.no_grad():
        for chunk in x.split(8):
            values.extend(model.theory_encoder(balanced_views(chunk, samples)).squeeze(-1).tolist())
    values = np.asarray(values)
    knots = np.unique(h)
    means = np.asarray([values[h == v].mean() for v in knots])
    stds = np.asarray([values[h == v].std(ddof=1) for v in knots])
    grid = np.linspace(knots[0], knots[-1], 1000)
    curve = np.interp(grid, knots, means)
    g1 = float(np.interp(1., knots, means))
    g0 = float(means[0] - knots[0] * (means[1] - means[0]) / (knots[1] - knots[0]))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(h, values, s=10, alpha=.22, color="gray", label="Individual replicas")
    ax.errorbar(knots, means, yerr=stds, fmt="o", markersize=3, linewidth=.8,
                capsize=2, color="tab:blue", label="Replica mean +/- 1 SD (3 chains)")
    ax.plot(grid, curve, color="tab:blue", linewidth=1.5, label="Piecewise-linear interpolation")
    ax.plot([0, knots[0]], [g0, means[0]], "--", color="tab:blue", linewidth=1.5,
            label="Short extrapolation to h/J=0")
    ax.axvline(1., color="tab:red", linestyle="--", linewidth=1)
    ax.scatter([1.], [g1], color="tab:red", s=40, zorder=5)
    ax.annotate(f"h/J=1: g={g1:.4f}", xy=(1., g1), xytext=(1.12, g1-.35),
                arrowprops={"arrowstyle": "->", "color": "tab:red"}, color="tab:red")
    ax.set(xlabel=r"Transverse-field ratio $h/J$", ylabel=r"Microscopic coordinate $g_0=Q(\mathcal{D}_0)$",
           title="TFIM: learned coordinate calibration (sample_batch=128)", xlim=(0, 2.02))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.run / "h_over_J_vs_g.png", dpi=200)
    plt.close(fig)
    np.savetxt(args.run / "h_over_J_vs_g.csv", np.column_stack((knots, means, stds)),
               delimiter=",", header="h_over_J,g_mean,g_replica_std", comments="")
    (args.run / "coordinate_calibration.json").write_text(json.dumps({
        "definition": "Q applied once to all saved microscopic configurations and their spin flips",
        "samples_per_replica": samples, "replicas_per_coupling": 3,
        "interpolation": "piecewise linear through replica means",
        "g_at_h_over_J_0_extrapolated": g0, "g_at_h_over_J_1_interpolated": g1,
        "error_bars": "sample SD across replicas; not a confidence interval",
    }, indent=2) + "\n")
    print(json.dumps({"g_at_0_extrapolated": g0, "g_at_1": g1,
                      "g_at_2": float(means[-1]), "figure": str(args.run / "h_over_J_vs_g.png")}))


if __name__ == "__main__":
    main()
