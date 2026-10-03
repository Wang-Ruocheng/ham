#!/usr/bin/env python3
"""Plot scalar coordinates after actual TFIM coarse-graining at each scale."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def plot_closure(model, values, h, train_count, run):
    """Roll out each replica separately before taking means or interpolating."""
    with torch.no_grad():
        direct = torch.from_numpy(values)
        predicted = [direct[:, :1]]
        for _ in range(values.shape[1] - 1):
            predicted.append(model.advance_theory(predicted[-1]))
        rollout = torch.cat(predicted, dim=1).numpy()
        one_step = model.advance_theory(direct[:, :-1, None]).squeeze(-1).numpy()
    knots = np.unique(h)
    means = np.array([values[h == v].mean(axis=0) for v in knots])
    pred_means = np.array([rollout[h == v].mean(axis=0) for v in knots])
    at_one = np.array([np.interp(1., knots, means[:, k]) for k in range(values.shape[1])])
    pred_one = np.array([np.interp(1., knots, pred_means[:, k]) for k in range(values.shape[1])])
    metrics = {}
    for name, subset in (("train", slice(None, train_count)), ("test", slice(train_count, None))):
        rmse = np.sqrt(np.mean((rollout[subset] - values[subset])**2, axis=0))
        scale = values[subset].std(axis=0)
        metrics[name] = {
            "rollout_rmse": rmse.tolist(),
            "rollout_rmse_over_direct_std": (rmse / scale).tolist(),
            "one_step_rmse": np.sqrt(np.mean((one_step[subset] - values[subset, 1:])**2, axis=0)).tolist(),
        }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax, critical, errors, aggregate = axes.flat
    for k in range(values.shape[1]):
        ax.plot(knots, means[:, k], color=f"C{k}", label=rf"$k={k}$")
        ax.plot(knots, pred_means[:, k], color=f"C{k}", ls="--")
        if k:
            errors.plot(knots, pred_means[:, k] - means[:, k], color=f"C{k}", label=rf"$k={k}$")
    ax.set(title="Solid: actual Q(C^k D); dashed: R rollout", xlabel=r"Initial $h/J$", ylabel="g")
    ax.axvline(1., color="gray", ls=":")
    ax.legend()
    steps = np.arange(values.shape[1])
    critical.plot(steps, at_one, "o-", label=r"Actual $Q(C^k\mathcal{D})$")
    critical.plot(steps, pred_one, "s--", label=r"Predicted $R^k(Q(\mathcal{D}))$")
    for k, (actual, pred) in enumerate(zip(at_one, pred_one)):
        actual_offset, pred_offset = (-17, 10) if pred > actual else (10, -17)
        critical.annotate(f"{actual:.3f}", (k, actual), xytext=(0, actual_offset), textcoords="offset points", ha="center", color="C0")
        if k:
            critical.annotate(f"{pred:.3f}", (k, pred), xytext=(0, pred_offset), textcoords="offset points", ha="center", color="C1")
    critical.set(title=r"Initial $h/J=1$ (linear interpolation)", xlabel="RG step k", ylabel="g", xticks=steps)
    critical.margins(x=.12, y=.18)
    critical.legend()
    errors.axhline(0, color="black", lw=.8)
    errors.axvline(1., color="gray", ls=":")
    errors.set(title="Signed difference of replica means", xlabel=r"Initial $h/J$", ylabel="R rollout - actual encoding")
    errors.legend()
    for name, marker in (("train", "o"), ("test", "s")):
        aggregate.plot(steps, metrics[name]["rollout_rmse"], marker=marker, label=name.capitalize())
    aggregate.set(title="Rollout RMSE across individual ensembles", xlabel="RG step k", ylabel="RMSE", xticks=steps)
    aggregate.legend()
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.text(.5, .01, "Latest sample_batch=128 checkpoint; evaluation: all 128 configurations + spin flips per replica; no retraining", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    output = run / "qcr_closure_comparison.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    report = {"evaluation": "Full 128 saved configurations plus global flips per replica; Q applied once per level; rollout starts independently from each replica's g0",
              "interpolation": "At h/J=1, independently interpolate direct and rollout replica-mean curves; not an actual sampled h/J=1 ensemble",
              "direct_at_1": at_one.tolist(), "rollout_at_1": pred_one.tolist(),
              "metrics": metrics, "figure": str(output.resolve())}
    (run / "qcr_closure_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [h, (np.arange(len(h)) >= train_count).astype(int)]
    names = ["h_over_J", "is_test"]
    for k in range(values.shape[1]):
        columns.extend([values[:, k], rollout[:, k]])
        names.extend([f"direct_g{k}", f"rollout_g{k}"])
    np.savetxt(run / "qcr_closure_replicas.csv", np.column_stack(columns), delimiter=",",
               header=",".join(names), comments="")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional model or training-state checkpoint")
    parser.add_argument("--output-dir", type=Path, help="Separate directory for intermediate checkpoint plots")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads((args.run / "summary.json").read_text())["config"]
    if cfg["theory_dim"] != 1:
        raise ValueError("Expected a scalar coordinate")
    model = RecurrentRGNetwork1D(levels=cfg["levels"], hidden=cfg["hidden"],
                                theory_dim=1, learnable_kernel=not cfg["fixed_kernel"],
                                kernel_layout=cfg.get("kernel_layout", "overlapping")).double()
    checkpoint = args.checkpoint or args.run / 'qcr_model.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    step = state.get('epoch')
    incompatible = model.load_state_dict(state.get('model', state), strict=False)
    # QCR checkpoints omit the unused action module; all Q/C/R weights are required.
    if incompatible.unexpected_keys or any(not key.startswith("action_network.")
                                          for key in incompatible.missing_keys):
        raise ValueError(f"Incompatible QCR checkpoint: {incompatible}")
    model.eval()
    if args.output_dir:
        args.run = args.output_dir
        args.run.mkdir(parents=True, exist_ok=True)
    train_x, train_h, test_x, test_h = torch.load(args.cache, weights_only=True)
    x = torch.cat((train_x, test_x))
    h = torch.cat((train_h, test_h)).numpy()
    with torch.no_grad():
        values = torch.cat([
            torch.stack([model.theory_encoder(level).squeeze(-1)
                         for level in model.coarse_grain(balanced_views(chunk, x.shape[1]))], dim=1)
            for chunk in x.split(4)
        ]).numpy()
    plot_closure(model, values, h, len(train_x), args.run)
    knots = np.unique(h)
    means = np.array([values[h == v].mean(axis=0) for v in knots])
    stds = np.array([values[h == v].std(axis=0, ddof=1) for v in knots])
    at_one = np.array([np.interp(1., knots, means[:, k]) for k in range(cfg["levels"])])

    def residual(g):
        with torch.no_grad():
            return model.theory_rg(torch.tensor([[g]], dtype=torch.float64)).item()

    grid = np.linspace(-1.2, 5., 2001)
    residuals = [residual(g) for g in grid]
    roots = [brentq(residual, a, b) for a, b, fa, fb in
             zip(grid[:-1], grid[1:], residuals[:-1], residuals[1:]) if fa * fb < 0]
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    for k in range(cfg["levels"]):
        color = f"C{k}"
        ax.plot(knots, means[:, k], color=color, label=rf"$g_{k}$: {cfg['sites'] // 2**k} sites")
        ax.fill_between(knots, means[:, k] - stds[:, k], means[:, k] + stds[:, k],
                        color=color, alpha=.16)
        ax.scatter([1.], [at_one[k]], color=color, s=23, zorder=4)
    ax.axvline(1., color="gray", ls=":", label=r"$h/J=1$")
    for root in roots:
        if values.min() - .2 <= root <= values.max() + .2:
            ax.axhline(root, color="black", ls="--", lw=1, label=rf"$R$ fixed point: {root:.3f}")
    ax.set(xlabel=r"Initial $h/J$", ylabel=r"$g_k = Q(C^k\mathcal{D})$",
           title="Actual coarse-graining: coordinate vs. initial field", xlim=(0, 2.02))
    ax.legend(fontsize=9)
    detail.plot(range(cfg["levels"]), at_one, "o-", color="C4")
    for k, val in enumerate(at_one):
        detail.annotate(f"{val:.4f}", (k, val), xytext=(0, 10),
                        textcoords="offset points", ha="center")
    for root in roots:
        if values.min() - .2 <= root <= values.max() + .2:
            detail.axhline(root, color="black", ls="--", lw=1, label=rf"$R$ fixed point: {root:.3f}")
    detail.set(xlabel="RG step k", ylabel=r"$g_k(h/J=1)$",
               title=r"Trajectory at $h/J=1$ (interpolated)", xticks=range(cfg["levels"]))
    detail.margins(x=.15, y=.2)
    detail.legend(fontsize=9)
    for axis in (ax, detail):
        axis.grid(alpha=.2)
    fig.text(.5, .01, "128 saved configurations + spin flips / replica; 3 replicas; bands: +/-1 replica SD", ha="center", fontsize=9)
    if step is not None:
        fig.suptitle(f'TFIM dim(g)=1 — checkpoint at {step} updates')
    fig.tight_layout(rect=(0, .04, 1, .93 if step is not None else 1))
    output = args.run / "g_by_scale_vs_h_over_J.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    cols = [knots]
    names = ["h_over_J"]
    for k in range(cfg["levels"]):
        cols.extend([means[:, k], stds[:, k]])
        names.extend([f"g{k}_mean", f"g{k}_replica_sd"])
    np.savetxt(args.run / "g_by_scale_vs_h_over_J.csv", np.column_stack(cols),
               delimiter=",", header=",".join(names), comments="")
    report = {"definition": "Q(C^k D); spin flips added only at microscopic level; no R rollout",
              "run": str(args.run), "cache": str(args.cache),
              "checkpoint": str(checkpoint), "training_step": step,
              "g_at_h_over_J_1_linear_interpolation": at_one.tolist(),
              "g_at_smallest_h_over_J": means[0].tolist(),
              "g_at_largest_h_over_J": means[-1].tolist(),
              "R_roots_search_interval": [-1.2, 5.], "R_roots": roots,
              "figure": str(output.resolve())}
    (args.run / "g_by_scale_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
