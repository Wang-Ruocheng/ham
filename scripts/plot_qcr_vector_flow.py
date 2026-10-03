#!/usr/bin/env python3
"""Plot a two-coordinate discrete RG map and distinguish sampled states from extrapolation."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.optimize import root
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D
from recurrent_rg.checkpoints import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run / "summary.json").read_text())
    cfg = summary["config"]
    if cfg["theory_dim"] != 2:
        raise ValueError("This figure requires dim(g)=2")
    is_2d = cfg.get("dimension", 1) == 2
    is_quantum = 'time_slices' in cfg
    coupling_name = 'h/J' if is_quantum or not is_2d else 'K'
    model = load_run(args.run, overrides={'dimension':2 if is_2d else 1}).model
    train_x, train_h, test_x, test_h = torch.load(args.cache, weights_only=True)
    x = torch.cat((train_x, test_x))
    h = torch.cat((train_h, test_h)).numpy()
    direct, rollout = [], []
    with torch.no_grad():
        for chunk in x.split(4):
            trajectory = model(balanced_views(chunk, x.shape[1]))
            direct.append(trajectory.direct_theory)
            rollout.append(trajectory.rollout_theory)
        direct = torch.cat(direct).numpy()
        rollout = torch.cat(rollout).numpy()
        predicted = model.advance_theory(torch.from_numpy(direct[:, :-1])).numpy()
    metrics = {}
    for split, subset in (("train", slice(None, len(train_x))), ("test", slice(len(train_x), None))):
        d, r = direct[subset], rollout[subset]
        rms = np.sqrt(np.mean((r-d)**2, axis=(0, 2)))
        scale = np.sqrt(d.var(axis=0).mean(axis=-1))
        metrics[split] = {
            "rollout_rmse_per_coordinate": rms.tolist(),
            "rollout_normalized_rmse": (rms/scale).tolist(),
            "one_step_rmse_per_coordinate": np.sqrt(np.mean((predicted[subset]-d[:, 1:])**2, axis=(0, 2))).tolist(),
            "coordinate_covariance_eigenvalues": [np.linalg.eigvalsh(np.cov(d[:, k].T, bias=True)).tolist() for k in range(cfg["levels"])],
        }
    points = direct[:, :-1].reshape(-1, 2)  # last level is NOT an observed R input
    envelope = direct.reshape(-1, 2)
    span = np.maximum(np.ptp(envelope, axis=0), .1)
    lower, upper = envelope.min(0)-.18*span, envelope.max(0)+.18*span
    # Equal-coordinate metric, with a documented, purely descriptive support radius.
    radius = .12 * float(np.sqrt(points.var(0).sum()))
    tree = cKDTree(points)

    def residual(z):
        with torch.no_grad():
            return model.theory_rg(torch.as_tensor(z, dtype=torch.float64)).numpy()

    def jacobian_residual(z):
        t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(model.theory_rg, t).detach().numpy()

    found = []
    seeds = np.array(list(itertools.product(np.linspace(lower[0], upper[0], 9), np.linspace(lower[1], upper[1], 9))))
    for guess in seeds:
        solution = root(residual, guess, jac=jacobian_residual)
        z = solution.x
        if (not np.all(np.isfinite(z)) or np.linalg.norm(residual(z)) > 1e-8
                or np.any(z < lower) or np.any(z > upper)
                or any(np.linalg.norm(z-np.array(f["g"])) < 1e-4 for f in found)):
            continue
        eig = np.linalg.eigvals(np.eye(2)+jacobian_residual(z))
        modulus = np.abs(eig)
        stability = ("stable" if np.all(modulus < 1-1e-4) else "unstable" if np.all(modulus > 1+1e-4)
                     else "saddle" if np.min(modulus) < 1-1e-4 and np.max(modulus) > 1+1e-4 else "near-marginal")
        distance = float(tree.query(z)[0])
        found.append({"g": z.tolist(), "R_jacobian_eigenvalues": [{"real": float(v.real), "imag": float(v.imag)} for v in eig],
                      "stability": stability, "residual_norm": float(np.linalg.norm(residual(z))),
                      "distance_to_sampled_R_input": distance, "near_sampled_input": distance <= radius})

    gx, gy = np.meshgrid(np.linspace(lower[0], upper[0], 27), np.linspace(lower[1], upper[1], 27))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    delta = residual(grid)
    speed = np.linalg.norm(delta, axis=1)
    direction = delta/np.maximum(speed[:, None], 1e-12)
    near = tree.query(grid)[0] <= radius
    arrow_length = .030 * float(min(upper-lower))
    u, v = (direction * arrow_length).T
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    ax, paths = axes
    ax.quiver(grid[~near, 0], grid[~near, 1], u[~near], v[~near], color=".83", angles="xy",
              scale_units="xy", scale=1, width=.003, pivot="mid")
    norm = Normalize(vmin=0, vmax=max(float(speed[near].max()) if near.any() else float(speed.max()), 1e-8))
    colored = ax.quiver(grid[near, 0], grid[near, 1], u[near], v[near], speed[near], cmap="viridis", norm=norm,
                        angles="xy", scale_units="xy", scale=1, width=.004, pivot="mid")
    fig.colorbar(colored, ax=ax, shrink=.75, label=r"One-step displacement $\|R(g)-g\|$")
    for k in range(cfg["levels"]):
        ax.scatter(direct[:, k, 0], direct[:, k, 1], s=13, alpha=.45, marker="o" if k<cfg["levels"]-1 else "x",
                   color=f"C{k}", label=f"Actual k={k}" + (" (targets only)" if k==cfg["levels"]-1 else ""))
    for i, f in enumerate(found):
        z = f["g"]
        ax.scatter(*z, marker="*", s=145, facecolors="black" if f["near_sampled_input"] else "white", edgecolors="black", zorder=6)
        right_half = z[0] > (lower[0]+upper[0])/2
        ax.annotate(f"F{i+1}: {f['stability']}", z, xytext=(-6 if right_half else 6, 7),
                    ha="right" if right_half else "left", textcoords="offset points", fontsize=8)
    ax.set(title=r"Discrete RG field: $\Delta g=R(g)-g$")
    ax.legend(fontsize=8, loc="best")
    knots = np.unique(h)
    means = np.array([direct[h==value].mean(0) for value in knots])
    selected_fields = []
    critical = float(cfg.get('reference_critical_coupling', .5*np.log(1+np.sqrt(2)) if is_2d else 1.))
    if is_quantum and is_2d:
        if 'reference_critical_coupling' not in cfg:
            raise ValueError('2D TFIM requires its own critical-coupling reference')
        requested_fields=(2.,2.8,critical,3.4,4.5)
    else:
        requested_fields = (.25, .38, critical, .5, .62) if is_2d else (.5, .8, 1., 1.2, 1.7)
    for i, requested in enumerate(requested_fields):
        j = int(np.argmin(abs(knots-requested)))
        selected_fields.append(float(knots[j]))
        actual = means[j]
        path = [actual[0]]
        for _ in range(cfg["levels"]-1):
            path.append(path[-1]+residual(path[-1]))
        path = np.array(path)
        paths.plot(actual[:, 0], actual[:, 1], "o-", color=f"C{i}", label=f"initial {coupling_name}={knots[j]:.3f}", ms=4)
        paths.plot(path[:, 0], path[:, 1], "x--", color=f"C{i}", ms=5, alpha=.8)
        paths.annotate("k=0", actual[0], xytext=(3, 4), textcoords="offset points", fontsize=7)
    paths.set(title="Solid: Q after C; dashed: R rollout from mean g0")
    paths.legend(fontsize=8)
    for axis in axes:
        axis.set(xlabel=r"Coordinate $g^{(1)}$", ylabel=r"Coordinate $g^{(2)}$", xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.15)
    fig.text(.5, .025, "Arrow lengths normalized; grey arrows = far from sampled inputs; stars are numerical fixed points, not established physical fixed points.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.run / "R_vector_field.png", dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for k in range(cfg['levels']):
        for d in range(2):
            axes[d].plot(knots,means[:,k,d],label=f'k={k}')
        axes[2].plot(means[:,k,0],means[:,k,1],'.-',label=f'k={k}')
        point=[float(np.interp(critical,knots,means[:,k,d])) for d in range(2)]
        axes[2].scatter(*point,marker='x',color='black')
    for d in range(2):
        axes[d].axvline(critical,color='black',ls=':',label='reference')
        axes[d].set(xlabel=coupling_name,ylabel=f'g{d+1}')
    axes[2].set(xlabel='g1',ylabel='g2',title='Actual Q curve; crosses = reference coupling')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Coupling labels used only for evaluation; curves are Q(C^k D), not R rollout')
    fig.tight_layout();fig.savefig(args.run/'coupling_vs_g.png',dpi=180);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    history = summary["history"]
    axes[0].plot([v["epoch"] for v in history], [v["train_total"] for v in history], label="dim g=2 train")
    axes[0].plot([v["epoch"] for v in history], [v["test_total"] for v in history], label="dim g=2 test")
    if args.baseline:
        old = json.loads((args.baseline/"summary.json").read_text())["history"]
        axes[0].plot([v["epoch"] for v in old], [v["test_total"] for v in old], "--", label="dim g=1 test (reference)", alpha=.6)
    axes[0].set(xlabel="Optimizer step", ylabel="Total objective", yscale="log",
                title="Loss (vector objective differs from scalar)" if args.baseline else "Training and held-out loss")
    for split in metrics:
        axes[1].plot(range(cfg["levels"]), metrics[split]["rollout_normalized_rmse"], "o-", label=split)
    axes[1].set(xlabel="RG step k", ylabel="RMSE / pooled coordinate SD", title="Full-ensemble rollout error", xticks=range(cfg["levels"]))
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run / "vector_training_diagnostics.png", dpi=200)
    plt.close(fig)
    patterns = torch.tensor(list(itertools.product([-1., 1.], repeat=4 if is_2d else 2)), dtype=torch.float64)
    with torch.no_grad():
        if cfg.get('fixed_kernel', False) and not is_2d:
            outputs = model.rg_kernel(patterns.reshape(4,1,2,1)).flatten().tolist()
        else:
            outputs = model.rg_kernel.field_head(model.rg_kernel.local_encoder(patterns)).squeeze().tolist()
    line_slope, line_intercept = np.polyfit(envelope[:, 0], envelope[:, 1], 1)
    report = {"config": cfg, "parameter_count": sum(p.numel() for p in model.parameters()),
              "optimized_parameter_count": sum(p.numel() for module in (model.rg_kernel,model.theory_encoder,model.theory_rg) for p in module.parameters()),
              "evaluation": f"All {x.shape[1]} saved samples plus global flips per replica, Q once per entire ensemble; all train/test systems",
              "metrics": metrics, "fixed_points_in_plot_box": found,
              "pooled_coordinate_diagnostic": {
                  "covariance_eigenvalues": np.linalg.eigvalsh(np.cov(envelope.T, bias=True)).tolist(),
                  "g2_vs_g1_slope": float(line_slope), "g2_vs_g1_intercept": float(line_intercept),
                  "line_residual_rmse": float(np.sqrt(np.mean((envelope[:, 1]-line_slope*envelope[:, 0]-line_intercept)**2))),
                  "caution": "Nearly collinear samples do not constrain transverse derivatives, even at roots close to sampled states"},
              "fixed_point_search": "81 initial guesses, residual tolerance 1e-8; not exhaustive; discrete R Jacobian used",
              "plot_box": [lower.tolist(),upper.tolist()], "near_sample_radius": radius,
              "near_sample_definition": f"Euclidean distance to observed k=0,...,{cfg['levels']-2} inputs; heuristic, not a confidence region",
              "kernel_truth_table": [{"input": p, "output": o} for p,o in zip(patterns.tolist(),outputs)],
              "trajectory_fields": selected_fields,
              "critical_coupling": float(critical),
              "g_at_critical_coupling_by_level": [[float(np.interp(critical, knots, means[:, k, d])) for d in range(2)] for k in range(cfg["levels"])]}
    if not is_2d:
        report["g_at_h_over_J_1_interpolated_by_level"] = report["g_at_critical_coupling_by_level"]
    np.savez_compressed(args.run / "vector_flow_data.npz", direct=direct, rollout=rollout, **({"K": h} if is_2d and not is_quantum else {"h_over_J": h}),
                        grid=grid, delta_g=delta, near_sample=near, is_test=np.arange(len(x))>=len(train_x))
    (args.run / "vector_flow_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
