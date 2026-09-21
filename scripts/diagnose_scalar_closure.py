#!/usr/bin/env python3
"""Cross-scale scalar-state sufficiency checks; no training or coupling labels in inference."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._utils import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


NAMES = ["g", "next_g", "R_g", "flow_residual", "G1", "G2", "G4", "G8", "G4_over_G1", "G8_over_G1"]


def observables(level):
    """Balanced-spin two-point functions in CURRENT lattice units, not physical distance."""
    spin = level.squeeze(-1)
    return torch.stack([(spin * spin.roll(-r, dims=-1)).mean(dim=(1, 2))
                        for r in (1, 2, 4, 8)], dim=-1)


def match_pairs(mean, h, tolerance):
    # One nearest neighbour in the other scale per source coupling; no physical
    # observable/flow outcome is used to select the pairs.
    pairs = []
    for k, l in combinations(range(mean.shape[1]), 2):
        for i in range(len(h)):
            candidates = np.where((np.abs(h - h[i]) >= .05)
                                  & (mean[:, l, 4] > .05) & (mean[:, l, 4] < .95))[0]
            if not (.05 < mean[i, k, 4] < .95) or not len(candidates):
                continue
            j = candidates[np.argmin(np.abs(mean[candidates, l, 0] - mean[i, k, 0]))]
            if abs(mean[j, l, 0] - mean[i, k, 0]) <= tolerance:
                pairs.append((i, k, int(j), l))
    return pairs


def describe_pairs(mean, pairs):
    if not pairs:
        return {"count": 0}
    delta = np.array([mean[i, k] - mean[j, l] for i, k, j, l in pairs])
    return {"count": len(pairs), **{
        name: {"median_abs_difference": float(np.median(np.abs(delta[:, n]))),
               "p90_abs_difference": float(np.quantile(np.abs(delta[:, n]), .9))}
        for n, name in enumerate(NAMES)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads((args.run / "summary.json").read_text())["config"]
    if cfg["theory_dim"] != 1:
        raise ValueError("Scalar coordinate required")
    model = RecurrentRGNetwork1D(levels=cfg["levels"], hidden=cfg["hidden"], theory_dim=1,
                                learnable_kernel=not cfg.get("fixed_kernel", False),
                                kernel_layout=cfg.get("kernel_layout", "overlapping")).double().eval()
    inc = model.load_state_dict(torch.load(args.run / "qcr_model.pt", weights_only=True), strict=False)
    if inc.unexpected_keys or any(not key.startswith("action_network.") for key in inc.missing_keys):
        raise ValueError(str(inc))
    train_x, train_h, test_x, test_h = torch.load(args.cache, weights_only=True)
    x = torch.cat((train_x, test_x))
    h = torch.cat((train_h, test_h)).numpy()
    rows = []
    with torch.no_grad():
        for chunk in x.split(4):
            levels = model.coarse_grain(balanced_views(chunk, x.shape[1]))
            if min(level.shape[2] for level in levels[:-1]) < 32:
                raise ValueError("Need L>=32 for source levels with r=8<=L/4")
            gs = [model.theory_encoder(level).squeeze(-1) for level in levels]
            entries = []
            for k, level in enumerate(levels[:-1]):
                cor = observables(level)
                pred = model.advance_theory(gs[k][:, None]).squeeze(-1)
                entries.append(torch.cat((torch.stack((gs[k], gs[k+1], pred, gs[k+1]-pred), -1),
                                          cor, cor[:, 2:] / cor[:, :1].clamp_min(1e-8)), -1))
            rows.append(torch.stack(entries, 1))
    rows = torch.cat(rows).numpy()
    knots = np.unique(h)
    replicas = np.array([rows[h == value] for value in knots])  # coupling, replica, level, feature
    if replicas.shape[1] != 3:
        raise ValueError("This diagnostic expects three independent replicas per coupling")
    mean, sd = replicas.mean(1), replicas.std(1, ddof=1)
    sweeps = {str(tol): describe_pairs(mean, match_pairs(mean, knots, tol)) for tol in (.01, .02, .05)}
    pairs = match_pairs(mean, knots, .02)
    checks = []
    for held in range(3):
        fit = replicas[:, [j for j in range(3) if j != held]].mean(1)
        selected = match_pairs(fit, knots, .02)
        # Report all selected pairs, and separately ones that remain close in held-out g.
        validated = [p for p in selected if abs(replicas[p[0], held, p[1], 0]
                                               - replicas[p[2], held, p[3], 0]) <= .05]
        checks.append({"held_out_replica": held, "selection": describe_pairs(fit, selected),
                       "held_out_all_selected": describe_pairs(replicas[:, held], selected),
                       "held_out_g_gap_le_0p05": describe_pairs(replicas[:, held], validated)})
    details = []
    for i, k, j, l in pairs:
        spread = np.sqrt(sd[i, k]**2 + sd[j, l]**2)
        delta = mean[i, k] - mean[j, l]
        details.append({"a": {"h_over_J": float(knots[i]), "level": k, "sites": cfg["sites"]//2**k,
                              **{n: float(v) for n, v in zip(NAMES, mean[i, k])}},
                        "b": {"h_over_J": float(knots[j]), "level": l, "sites": cfg["sites"]//2**l,
                              **{n: float(v) for n, v in zip(NAMES, mean[j, l])}},
                        "delta": dict(zip(NAMES, delta.tolist())),
                        "combined_replica_sd": dict(zip(NAMES, spread.tolist())),
                        "replica_a": replicas[i, :, k].tolist(), "replica_b": replicas[j, :, l].tolist()})
    # Descriptive examples only: selected after seeing outcomes, not hypothesis tests.
    examples = sorted(details, key=lambda p: abs(p["delta"]["G4_over_G1"]), reverse=True)[:5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for k in range(mean.shape[1]):
        mask = (mean[:, k, 4] > .05) & (mean[:, k, 4] < .95)
        for ax, col in zip(axes.flat, (1, 3, 8, 9)):
            ax.errorbar(mean[mask, k, 0], mean[mask, k, col], xerr=sd[mask, k, 0],
                        yerr=sd[mask, k, col], fmt="o", ms=3, lw=.7, alpha=.75,
                        label=f"k={k}, L={cfg['sites']//2**k}")
    grid = torch.linspace(float(mean[..., 0].min()), float(mean[..., 0].max()), 500).double()[:, None]
    with torch.no_grad():
        axes[0, 0].plot(grid[:, 0], model.advance_theory(grid)[:, 0], "k--", lw=1, label="learned R(g)")
    axes[0, 0].set(ylabel="Actual next g", title="Do different scales share the same next-step map?")
    axes[0, 1].axhline(0, color="black", ls="--", lw=1)
    axes[0, 1].set(ylabel="Actual next g - R(g)", title="Scale-dependent prediction residual")
    axes[1, 0].set(ylabel="G(4) / G(1)", title="Correlation shape beyond Q's local window")
    axes[1, 1].set(ylabel="G(8) / G(1)", title="Longer-distance correlation shape")
    for ax in axes.flat:
        ax.set_xlabel("Encoded scalar g")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.text(.5, .01, "All train/test couplings; 3 replicas; bars = replica SD, not CI; 0.05<G(1)<0.95; distances in coarse lattice units", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(args.run / "scalar_sufficiency.png", dpi=200)
    plt.close(fig)
    report = {"run": str(args.run), "feature_order": NAMES,
              "protocol": "128 saved samples + global flips, Q evaluated on full ensemble; compare source k=0,1,2 with next k+1",
              "matching": "Cross-scale nearest g, |delta h/J|>=0.05, 0.05<G1<0.95 in both, tolerance absolute g; pairs share ensembles and are not independent",
              "correlations": "G(r)=mean(s_i s_{i+r}); balanced mean spin is zero; ratios remove a common field-amplitude normalization, not finite-size/temperature effects",
              "cautions": ["Three replicas only; SD is not a confidence interval", "Different levels have different L and effective thermal scales", "Nonclosure of current Q does not prove impossibility of every scalar coordinate", "Examples ranked on observed differences; descriptive, not significance tests"],
              "replica_holdout_scope": "Held out from pair selection only, not from network training; training data and weights are unchanged",
              "tolerance_sweep": sweeps, "leave_one_replica_out": checks, "matched_pairs": details,
              "largest_correlation_difference_examples": examples}
    (args.run / "scalar_sufficiency.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = []
    for i in range(len(x)):
        for k in range(rows.shape[1]):
            flat.append([h[i], i >= len(train_x), k, cfg["sites"]//2**k, *rows[i, k]])
    np.savetxt(args.run / "scalar_sufficiency_replicas.csv", flat, delimiter=",",
               header=",".join(["h_over_J", "is_test", "level", "sites", *NAMES]), comments="")
    print(json.dumps({"tolerance_sweep": sweeps,
                      "held_out_validation": [c["held_out_g_gap_le_0p05"] for c in checks],
                      "examples": examples[:2]}, indent=2))


if __name__ == "__main__":
    main()
