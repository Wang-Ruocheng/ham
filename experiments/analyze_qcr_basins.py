#!/usr/bin/env python3
"""Analyze fixed points and attraction basins of a trained scalar QCR flow."""

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

from experiments.joint_qcr_ising_1d import direct_and_rollout
from recurrent_rg.ising import DTYPE, sample_periodic_ensembles
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def flow_numpy(model: RecurrentRGNetwork1D, values: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        return model.advance_theory(tensor).squeeze(-1).numpy()


def bisect_fixed_point(
    model: RecurrentRGNetwork1D, left: float, right: float, steps: int = 80
) -> float:
    for _ in range(steps):
        middle = 0.5 * (left + right)
        left_value = float(flow_numpy(model, np.array([left]))[0] - left)
        middle_value = float(flow_numpy(model, np.array([middle]))[0] - middle)
        if left_value * middle_value <= 0.0:
            right = middle
        else:
            left = middle
    return 0.5 * (left + right)


def find_fixed_points(
    model: RecurrentRGNetwork1D, lower: float, upper: float, points: int = 4001
) -> list[float]:
    grid = np.linspace(lower, upper, points)
    beta = flow_numpy(model, grid) - grid
    brackets = np.flatnonzero(beta[:-1] * beta[1:] <= 0.0)
    roots = [bisect_fixed_point(model, float(grid[index]), float(grid[index + 1])) for index in brackets]
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1e-6:
            unique.append(root)
    return unique


def flow_derivative(model: RecurrentRGNetwork1D, value: float) -> float:
    point = torch.tensor([[value]], dtype=DTYPE, requires_grad=True)
    mapped = model.advance_theory(point)
    derivative = torch.autograd.grad(mapped, point)[0]
    return float(derivative)


def iterate_flow(
    model: RecurrentRGNetwork1D, initial: np.ndarray, steps: int
) -> np.ndarray:
    trajectory = [np.asarray(initial, dtype=float)]
    for _ in range(steps):
        trajectory.append(flow_numpy(model, trajectory[-1]))
    return np.stack(trajectory, axis=0)


def convergence_steps(
    trajectory: np.ndarray, fixed_points: list[float], tolerance: float
) -> np.ndarray:
    if not fixed_points:
        return np.full(trajectory.shape[1], np.nan)
    distance = np.min(
        np.abs(trajectory[:, :, None] - np.asarray(fixed_points)[None, None, :]), axis=2
    )
    reached = distance < tolerance
    first = np.argmax(reached, axis=0).astype(float)
    first[~reached.any(axis=0)] = np.nan
    return first


def plot_basins(
    output: Path,
    model: RecurrentRGNetwork1D,
    support: tuple[float, float],
    exploration: tuple[float, float],
    fixed_points: list[float],
    actual_initial: np.ndarray,
    actual_couplings: np.ndarray,
) -> None:
    grid = np.linspace(*exploration, 1200)
    mapped = flow_numpy(model, grid)
    beta = mapped - grid
    grid_trajectory = iterate_flow(model, grid, 300)
    times = convergence_steps(grid_trajectory, fixed_points, tolerance=1e-3)
    actual_trajectory = iterate_flow(model, actual_initial, 30)

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    axes[0].plot(grid, mapped, label=r"$R(g)$")
    axes[0].plot(grid, grid, color="black", linestyle="--", linewidth=1, label=r"$g$")
    axes[0].axvspan(*support, color="tab:blue", alpha=0.1, label="R training support")
    for root in fixed_points:
        axes[0].scatter(root, root, color="tab:red", zorder=3)
    axes[0].set(xlabel=r"$g$", ylabel=r"$R(g)$", title="Learned RG map")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].plot(grid, beta)
    axes[1].fill_between(grid, 0.0, beta, where=beta > 0.0, alpha=0.2, color="tab:red")
    axes[1].fill_between(grid, 0.0, beta, where=beta < 0.0, alpha=0.2, color="tab:blue")
    axes[1].axvspan(*support, color="tab:blue", alpha=0.1)
    axes[1].set(xlabel=r"$g$", ylabel=r"$R(g)-g$", title="Flow direction")

    selection = np.linspace(0, actual_initial.size - 1, min(7, actual_initial.size)).astype(int)
    colors = plt.cm.viridis(
        (actual_couplings[selection] - actual_couplings.min())
        / max(float(np.ptp(actual_couplings)), 1e-12)
    )
    for index, color in zip(selection, colors):
        axes[2].plot(
            np.arange(actual_trajectory.shape[0]),
            actual_trajectory[:, index],
            color=color,
            label=fr"$J={actual_couplings[index]:.2f}$",
        )
    for root in fixed_points:
        axes[2].axhline(root, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="RG iteration", ylabel=r"$g_k$", title="Held-out ensemble trajectories")
    axes[2].legend(frameon=False, ncol=2, fontsize=8)

    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        axes[1].text(
            0.03,
            0.05,
            f"grid converged: {finite_times.size / times.size:.1%}\n"
            f"median steps: {np.median(finite_times):.0f}",
            transform=axes[1].transAxes,
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/joint_qcr_ising_1d_adaptive_sep/qcr_model.pt"),
    )
    parser.add_argument("--sites", type=int, default=128)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=192)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/qcr_basin_analysis")
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = RecurrentRGNetwork1D(levels=args.levels, theory_dim=1).to(dtype=DTYPE)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=False
    )
    model.eval()

    couplings = np.linspace(0.12, 1.20, 49)
    configurations, labels = sample_periodic_ensembles(
        couplings, args.replicas, args.ensemble_size, args.sites, args.seed
    )
    _, direct, _, _ = direct_and_rollout(model, configurations, args.eval_samples)
    flow_inputs = direct[:, :-1, 0].reshape(-1)
    support = (float(flow_inputs.min()), float(flow_inputs.max()))
    span = support[1] - support[0]
    exploration = (support[0] - 0.75 * span, support[1] + 0.75 * span)
    fixed_points = find_fixed_points(model, *exploration)
    fixed_point_data = [
        {
            "g": root,
            "derivative": flow_derivative(model, root),
            "stable": abs(flow_derivative(model, root)) < 1.0,
            "inside_training_support": support[0] <= root <= support[1],
        }
        for root in fixed_points
    ]

    initial_by_coupling = direct[:, 0, 0].reshape(len(couplings), args.replicas).mean(axis=1)
    grid = np.linspace(*exploration, 1001)
    grid_trajectory = iterate_flow(model, grid, 300)
    times = convergence_steps(grid_trajectory, fixed_points, tolerance=1e-3)
    support_mask = (grid >= support[0]) & (grid <= support[1])
    support_times = times[support_mask]
    finite_support = support_times[np.isfinite(support_times)]

    summary = {
        "checkpoint": str(args.checkpoint),
        "training_supported_R_input_interval": list(support),
        "displayed_exploration_interval": list(exploration),
        "fixed_points": fixed_point_data,
        "supported_grid_convergence_fraction": float(
            finite_support.size / max(support_times.size, 1)
        ),
        "supported_grid_median_convergence_steps": (
            float(np.median(finite_support)) if finite_support.size else None
        ),
        "interpretation_warning": (
            "Only fixed points and trajectories inside the learned R-input interval are "
            "data-supported; behavior outside it is neural-network extrapolation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_basins(
        args.output_dir / "rg_basins.png",
        model,
        support,
        exploration,
        fixed_points,
        initial_by_coupling,
        couplings,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
