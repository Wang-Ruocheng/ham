#!/usr/bin/env python3
"""Plot a learned scalar theory map R(g) against the identity line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.ising import DTYPE
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--g-min", type=float)
    parser.add_argument("--g-max", type=float)
    args = parser.parse_args()

    summary_path = args.model_path.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    config = summary["config"]
    theory_dim = int(config["theory_dim"])
    if theory_dim != 1:
        raise ValueError("R(g) versus g is defined here only for theory_dim=1")

    constructor = RecurrentRGNetwork1D if args.dimension == 1 else RecurrentRGNetwork2D
    model = constructor(
        levels=int(config["levels"]),
        theory_dim=theory_dim,
        hidden=int(config["hidden"]),
        kernel_layout=config.get("kernel_layout", "overlapping"),
        **({"learnable_kernel": not config.get("fixed_kernel", False)} if args.dimension == 1 else {}),
    ).to(dtype=DTYPE)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    learned_r = summary["learned_R"]
    support = learned_r["trained_support"]
    if len(support) != 2 or not all(isinstance(value, (int, float)) for value in support):
        raise ValueError("summary does not contain a scalar trained-support interval")
    span = max(float(support[1] - support[0]), 1e-6)
    g_min = args.g_min if args.g_min is not None else support[0] - 0.1 * span
    g_max = args.g_max if args.g_max is not None else support[1] + 0.1 * span
    if not np.isfinite([g_min, g_max]).all() or g_min >= g_max:
        raise ValueError("plot limits must be finite with g_min < g_max")
    g_values = np.linspace(g_min, g_max, 600)
    with torch.no_grad():
        r_values = model.advance_theory(
            torch.tensor(g_values[:, None], dtype=DTYPE)
        ).squeeze(-1).numpy()

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(g_values, r_values, linewidth=2, label=r"learned $R(g)$")
    axes[0].plot(g_values, g_values, "k--", linewidth=1, label=r"$R(g)=g$")
    axes[0].axvspan(*support, color="tab:blue", alpha=0.1, label="trained support")
    axes[0].set(xlabel=r"$g$", ylabel=r"$R(g)$", title="Learned theory RG map")
    axes[0].legend(frameon=False)

    axes[1].plot(g_values, r_values - g_values, linewidth=2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].axvspan(*support, color="tab:blue", alpha=0.1)
    axes[1].set(xlabel=r"$g$", ylabel=r"$R(g)-g$", title="Fixed-point residual")

    for root in learned_r.get("fixed_points_in_support", []):
        color = "tab:green" if root["stability"] == "stable" else "tab:red"
        axes[0].scatter(root["g"], root["g"], color=color, zorder=3)
        axes[1].scatter(root["g"], 0.0, color=color, zorder=3)

    for axis in axes:
        axis.set_xlim(g_min, g_max)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output = args.output or args.model_path.parent / "R_vs_g.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
