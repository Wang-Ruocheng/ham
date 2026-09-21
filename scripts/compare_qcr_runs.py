#!/usr/bin/env python3
"""Compare scalar 1D QCR runs on a shared cached ensemble dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._utils import balanced_views
from experiments.ensemble_theory_encoder_ising_1d import correlation_loss, upper_triangle
from experiments.joint_qcr_ising import two_spin_distribution
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def anchor_components(model, x, samples):
    first, second, patterns = [], [], []
    with torch.no_grad():
        for chunk in x.split(8):
            v = balanced_views(chunk, samples)
            first.append(model.theory_encoder(v[:, :samples]))
            second.append(model.theory_encoder(v[:, samples:]))
            patterns.append(two_spin_distribution(v))
    a, b, p = torch.cat(first), torch.cat(second), torch.cat(patterns)
    g = (a + b) / 2
    centered = g - g.mean(0)
    cov = centered.T @ centered / len(g)
    terms = {
        "consistency_times10": 10 * (a - b).square().mean(),
        "variance": (g.std(0, unbiased=False) - 1).square().mean(),
        "covariance": (cov - torch.diag(torch.diag(cov))).square().mean(),
        "geometry": correlation_loss(upper_triangle(torch.cdist(g, g)),
                                     upper_triangle(torch.cdist(p, p))),
    }
    return {"total": float(sum(terms.values())), **{k: float(v) for k, v in terms.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train_x, _, test_x, _ = torch.load(args.cache, weights_only=True)
    report = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        cfg, history = summary["config"], summary["history"]
        if cfg["theory_dim"] != 1:
            raise ValueError("Comparison requires scalar g")
        model = RecurrentRGNetwork1D(levels=cfg["levels"], hidden=cfg["hidden"],
                                    theory_dim=1, learnable_kernel=not cfg.get("fixed_kernel", False),
                                    kernel_layout=cfg.get("kernel_layout", "overlapping")).double()
        model.load_state_dict(torch.load(run / "qcr_model.pt", weights_only=True), strict=False)
        model.eval()
        label = f"{cfg.get('kernel_layout', 'overlapping')}, batch={cfg['sample_batch']}"
        epochs = [h["epoch"] for h in history]
        line, = axes[0].plot(epochs, [h["test_total"] for h in history], label=label + " test")
        axes[0].plot(epochs, [h["train_total"] for h in history], "--", color=line.get_color(),
                     label=label + " train", alpha=.65)
        g = torch.linspace(-.5, 4, 1000, dtype=torch.float64)[:, None]
        with torch.no_grad():
            r = model.advance_theory(g).squeeze().numpy()
        axes[1].plot(g.numpy().ravel(), r, label=label)
        axes[2].plot(g.numpy().ravel(), r - g.numpy().ravel(), label=label)
        diagnostics = {}
        for split, x in (("train", train_x), ("test", test_x)):
            indices = torch.linspace(0, len(x)-1, min(cfg["eval_systems"], len(x))).long()
            diagnostics[split] = {
                "original_eval": anchor_components(model, x[indices], cfg["eval_samples"]),
                "all_systems_128": anchor_components(model, x, 128),
            }
        report.append({"run": str(run), "config": cfg, "final": history[-1],
                       "anchor_components": diagnostics,
                       "test_metrics": summary["test_metrics"], "learned_R": summary["learned_R"]})
    axes[0].set(xlabel="Epoch", ylabel="Total objective", yscale="log", title="Loss curves")
    axes[1].plot([-.5, 4], [-.5, 4], "k--", linewidth=1)
    axes[1].set(xlabel="g", ylabel="R(g)", title="Learned maps (unaligned coordinates)", xlim=(-.5, 4))
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="g", ylabel="R(g) - g", title="Fixed-point residual", xlim=(-.5, 4))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
