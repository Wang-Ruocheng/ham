#!/usr/bin/env python3
"""Learn Q, C, and R from unlabeled 2D Ising ensembles.

The coupling is used only after training.  Q is first anchored by ensemble
consistency and local-distribution geometry; Q, C, and R are then optimized
jointly with the same adaptive multilevel anti-collapse objective as the 1D
experiment.  Neither RSMI nor an action model is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._utils import QCRWeights
from experiments.ensemble_theory_encoder_ising_1d import (
    correlation_loss,
    upper_triangle,
)
from recurrent_rg.ising import DTYPE
from recurrent_rg.ising2d import bond_sum_2d, sample_wolff_ensembles
from recurrent_rg.metrics import rank_correlation, mean_pairwise_square
from recurrent_rg.model_nd import RecurrentRGNetwork2D


ISING_2D_CRITICAL_COUPLING = 0.5 * np.log(1.0 + np.sqrt(2.0))


def plaquette_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """Empirical probabilities of the 16 oriented 2x2 spin plaquettes."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = (
        8 * bits
        + 4 * torch.roll(bits, -1, dims=2)
        + 2 * torch.roll(bits, -1, dims=3)
        + torch.roll(torch.roll(bits, -1, dims=2), -1, dims=3)
    )
    return F.one_hot(pattern, num_classes=16).to(configurations.dtype).mean(dim=(1, 2, 3))


def encoder_objective_2d(
    encoder: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Subset consistency, noncollapse, and local-distribution geometry."""
    first_g = encoder(first)
    second_g = encoder(second)
    consistency = F.mse_loss(first_g, second_g)
    joined_g = 0.5 * (first_g + second_g)
    std_per_dim = joined_g.std(dim=0, unbiased=False)
    variance = (std_per_dim - 1.0).square().mean()
    centered = joined_g - joined_g.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(joined_g.shape[0], 1)
    covariance = (covariance - torch.diag(torch.diag(covariance))).square().mean()
    distributions = 0.5 * (
        plaquette_distribution(first) + plaquette_distribution(second)
    )
    geometry = correlation_loss(
        upper_triangle(torch.cdist(joined_g, joined_g)),
        upper_triangle(torch.cdist(distributions, distributions)),
    )
    total = 10.0 * consistency + variance + covariance + geometry
    return total, {
        "consistency": consistency,
        "variance": variance,
        "covariance": covariance,
        "geometry": geometry,
    }


def qcr_objective(
    model: RecurrentRGNetwork2D,
    microscopic: torch.Tensor,
    *,
    weights: QCRWeights = QCRWeights(anchor=2.0),
    separation_fraction: float = 0.05,
    separation_decay: float = 0.5,
    noise_multiplier: float = 2.0,
    snr_off: float = 2.0,
    snr_on: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    samples = microscopic.shape[1]
    midpoint = samples // 2
    if midpoint < 1 or samples % 2:
        raise ValueError("QCR batches need an even number of samples")

    levels = model.coarse_grain(microscopic)
    first = torch.stack(
        [model.theory_encoder(level[:, :midpoint]) for level in levels], dim=1
    )
    second = torch.stack(
        [model.theory_encoder(level[:, midpoint:]) for level in levels], dim=1
    )
    direct = 0.5 * (first + second)
    anchor, _ = encoder_objective_2d(
        model.theory_encoder, microscopic[:, :midpoint], microscopic[:, midpoint:]
    )
    coarse_consistency = F.mse_loss(first[:, 1:], second[:, 1:])

    pairwise = torch.stack([mean_pairwise_square(direct[:, k]) for k in range(model.levels)])
    split_noise = torch.stack(
        [F.mse_loss(first[:, k], second[:, k]) for k in range(model.levels)]
    )
    spread = direct.std(dim=0, unbiased=False).square().mean(dim=-1).sqrt()
    snr = spread / split_noise.sqrt().clamp_min(1e-12)
    signal = F.relu(pairwise - noise_multiplier * split_noise)
    separation_terms = []
    gates = []
    for level in range(1, model.levels):
        gate = ((snr[level - 1].detach() - snr_off) / (snr_on - snr_off)).clamp(0.0, 1.0)
        target = separation_fraction * signal[level - 1].detach()
        penalty = F.relu(target - signal[level])
        gates.append(gate)
        separation_terms.append(separation_decay ** (level - 1) * gate * penalty)
    separation = torch.stack(separation_terms).sum()
    predicted = torch.stack(
        [model.advance_theory(direct[:, k]) for k in range(model.levels - 1)], dim=1
    )
    commutator = F.mse_loss(predicted, direct[:, 1:])
    total = (
        weights.anchor * anchor
        + weights.coarse_consistency * coarse_consistency
        + weights.separation * separation
        + weights.commutator * commutator
    )
    terms = {
        "anchor": anchor,
        "coarse_consistency": coarse_consistency,
        "separation": separation,
        "commutator": commutator,
    }
    for level in range(model.levels):
        terms[f"snr_level_{level}"] = snr[level]
        terms[f"pairwise_level_{level}"] = pairwise[level]
    for level, gate in enumerate(gates, start=1):
        terms[f"separation_gate_level_{level}"] = gate
    return total, terms


def evaluate(
    model: RecurrentRGNetwork2D,
    configurations: torch.Tensor,
    samples: int,
    objective: str,
) -> dict[str, float]:
    model.eval()
    subset = balanced_views(configurations, samples)
    with torch.no_grad():
        if objective == "q":
            total, terms = encoder_objective_2d(
                model.theory_encoder, subset[:, :samples], subset[:, samples:]
            )
        else:
            total, terms = qcr_objective(model, subset)
    return {"total": float(total)} | {key: float(value) for key, value in terms.items()}


def balanced_views(configurations: torch.Tensor, samples: int) -> torch.Tensor:
    """Build two exactly spin-flip-balanced ensemble views of ``samples`` each."""
    if samples % 2:
        raise ValueError("sample count must be even for exact spin-flip pairing")
    half = samples // 2
    if 2 * half > configurations.shape[1]:
        raise ValueError("not enough configurations for two balanced views")
    first = configurations[:, :half]
    second = configurations[:, half : 2 * half]
    return torch.cat((first, -first, second, -second), dim=1)


def train_stage(
    model: RecurrentRGNetwork2D,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    *,
    objective: str,
    epochs: int,
    system_batch: int,
    sample_batch: int,
    eval_samples: int,
    eval_systems: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    parameters = (
        model.theory_encoder.parameters() if objective == "q" else (
            parameter
            for module in (model.theory_encoder, model.rg_kernel, model.theory_rg)
            for parameter in module.parameters()
        )
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    record_every = max(1, epochs // 50)
    train_indices = torch.linspace(
        0, train_x.shape[0] - 1, min(eval_systems, train_x.shape[0])
    ).long()
    test_indices = torch.linspace(
        0, test_x.shape[0] - 1, min(eval_systems, test_x.shape[0])
    ).long()
    train_eval = train_x[train_indices]
    test_eval = test_x[test_indices]
    for epoch in range(epochs):
        model.train()
        systems = torch.randperm(train_x.shape[0], generator=generator)[:system_batch]
        if sample_batch % 2:
            raise ValueError("sample_batch must be even for exact spin-flip pairing")
        half = sample_batch // 2
        samples = torch.randperm(train_x.shape[1], generator=generator)[: 2 * half]
        selected = train_x[systems][:, samples]
        first_base, second_base = selected[:, :half], selected[:, half:]
        batch = torch.cat((first_base, -first_base, second_base, -second_base), dim=1)
        optimizer.zero_grad(set_to_none=True)
        if objective == "q":
            loss, _ = encoder_objective_2d(
                model.theory_encoder, batch[:, :sample_batch], batch[:, sample_batch:]
            )
        else:
            loss, _ = qcr_objective(model, batch)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            train_metrics = evaluate(model, train_eval, eval_samples, objective)
            test_metrics = evaluate(model, test_eval, eval_samples, objective)
            history.append(
                {"epoch": epoch + 1}
                | {f"train_{key}": value for key, value in train_metrics.items()}
                | {f"test_{key}": value for key, value in test_metrics.items()}
            )
    return history


def trajectory_data(
    model: RecurrentRGNetwork2D, configurations: torch.Tensor, samples: int
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    subset = balanced_views(configurations, samples)
    with torch.no_grad():
        levels = model.coarse_grain(subset)
        direct = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
        rollout = [direct[:, 0]]
        for _ in range(model.levels - 1):
            rollout.append(model.advance_theory(rollout[-1]))
        first = torch.stack(
            [model.theory_encoder(level[:, :samples]) for level in levels], dim=1
        )
        second = torch.stack(
            [model.theory_encoder(level[:, samples:]) for level in levels], dim=1
        )
    return levels, direct.numpy(), torch.stack(rollout, dim=1).numpy(), (first - second).numpy()


def find_fixed_points(
    model: RecurrentRGNetwork2D,
    lower: float | list[float],
    upper: float | list[float],
    points: int = 4001,
) -> list[dict[str, float | str | list[float]]]:
    """Find fixed points: 1D bisection or 2D flow-iteration clustering."""
    theory_dim = model.theory_dim
    if theory_dim == 1:
        return _find_fixed_points_1d(model, float(lower), float(upper), points)
    if theory_dim == 2:
        return _find_fixed_points_2d(model, lower, upper)
    return []


def _find_fixed_points_1d(
    model: RecurrentRGNetwork2D, lower: float, upper: float, points: int = 4001
) -> list[dict[str, float | str]]:
    grid = torch.linspace(lower, upper, points, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        delta = (model.advance_theory(grid) - grid).squeeze(-1).numpy()
    x = grid.squeeze(-1).numpy()
    brackets = np.where(delta[:-1] * delta[1:] <= 0)[0]
    roots = []
    for index in brackets:
        left, right = float(x[index]), float(x[index + 1])
        for _ in range(50):
            midpoint = 0.5 * (left + right)
            with torch.no_grad():
                f_left = float(model.advance_theory(torch.tensor([[left]], dtype=DTYPE))[0, 0] - left)
                f_mid = float(model.advance_theory(torch.tensor([[midpoint]], dtype=DTYPE))[0, 0] - midpoint)
            if f_left * f_mid <= 0:
                right = midpoint
            else:
                left = midpoint
        root = 0.5 * (left + right)
        value = torch.tensor([[root]], dtype=DTYPE, requires_grad=True)
        derivative = torch.autograd.grad(model.advance_theory(value).sum(), value)[0].item()
        if not roots or abs(root - float(roots[-1]["g"])) > 1e-5:
            roots.append({"g": root, "derivative": derivative, "stability": "stable" if abs(derivative) < 1.0 else "unstable"})
    return roots


def _find_fixed_points_2d(
    model: RecurrentRGNetwork2D,
    lower: list[float],
    upper: list[float],
    grid_points: int = 30,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    cluster_radius: float = 0.05,
) -> list[dict[str, float | str | list[float]]]:
    """Find 2D fixed points by iterating R from a grid and clustering converged points."""
    g1 = torch.linspace(lower[0], upper[0], grid_points, dtype=DTYPE)
    g2 = torch.linspace(lower[1], upper[1], grid_points, dtype=DTYPE)
    grid = torch.stack(torch.meshgrid(g1, g2, indexing="ij"), dim=-1).reshape(-1, 2)

    converged = []
    with torch.no_grad():
        for start in grid:
            point = start.clone()
            for _ in range(max_iter):
                next_point = model.advance_theory(point.unsqueeze(0)).squeeze(0)
                if (next_point - point).norm() < tolerance:
                    break
                point = next_point
            if (next_point - point).norm() < tolerance:
                converged.append(point.numpy())

    if not converged:
        return []

    converged = np.stack(converged)
    clusters = []
    for pt in converged:
        assigned = False
        for cluster in clusters:
            if np.linalg.norm(pt - np.mean(cluster, axis=0)) < cluster_radius:
                cluster.append(pt)
                assigned = True
                break
        if not assigned:
            clusters.append([pt])

    roots = []
    for cluster in clusters:
        center = np.mean(cluster, axis=0)
        g_tensor = torch.tensor(center, dtype=DTYPE).unsqueeze(0).requires_grad_(True)
        jac = torch.autograd.functional.jacobian(
            lambda x: model.advance_theory(x), g_tensor
        ).squeeze(0).squeeze(1)  # -> (theory_dim, theory_dim)
        eigenvalues = torch.linalg.eigvals(jac)
        max_abs_eig = float(eigenvalues.abs().max())
        roots.append({
            "g": center.tolist(),
            "max_eigenvalue_magnitude": max_abs_eig,
            "stability": "stable" if max_abs_eig < 1.0 else "unstable",
        })
    return roots


def physical_metrics(levels: list[torch.Tensor]) -> dict[str, list[float]]:
    bond, magnetization = [], []
    for level in levels:
        spins = level.squeeze(-1)
        area = spins.shape[-1] * spins.shape[-2]
        bond.append((bond_sum_2d(spins) / (2 * area)).mean(dim=1).numpy())
        magnetization.append(spins.mean(dim=(-2, -1)).abs().mean(dim=1).numpy())
    return {
        "nearest_neighbor_correlation_by_level": np.stack(bond, axis=1).tolist(),
        "absolute_magnetization_by_level": np.stack(magnetization, axis=1).tolist(),
    }


def kernel_metrics(model: RecurrentRGNetwork2D) -> dict[str, float | str]:
    rng = torch.Generator().manual_seed(1234)
    patterns = torch.where(
        torch.rand((8192, 9), generator=rng, dtype=DTYPE) > 0.5,
        torch.ones((), dtype=DTYPE),
        -torch.ones((), dtype=DTYPE),
    )
    with torch.no_grad():
        output = model.rg_kernel.field_head(model.rg_kernel.local_encoder(patterns)).squeeze(-1)
        flipped = model.rg_kernel.field_head(model.rg_kernel.local_encoder(-patterns)).squeeze(-1)
    majority = torch.sign(patterns.sum(dim=1))
    center = patterns[:, 4]
    return {
        "z2_equivariance_fraction": float((output == -flipped).to(DTYPE).mean()),
        "majority_agreement_up_to_sign": float(
            torch.maximum(
                (output == majority).to(DTYPE).mean(),
                (output == -majority).to(DTYPE).mean(),
            )
        ),
        "center_decimation_agreement_up_to_sign": float(
            torch.maximum((output == center).to(DTYPE).mean(), (output == -center).to(DTYPE).mean())
        ),
        "positive_output_fraction": float((output > 0).to(DTYPE).mean()),
    }


def summarize_representation(
    direct: np.ndarray, rollout: np.ndarray, split: np.ndarray, couplings: np.ndarray
) -> dict[str, object]:
    """Representation metrics compatible with scalar or multi-dimensional theory."""
    theory_dim = direct.shape[-1]
    metric_fn = _scalar_metrics if theory_dim == 1 else _multidimensional_metrics
    return metric_fn(direct, rollout, split, couplings, theory_dim)


def _scalar_metrics(
    direct: np.ndarray, rollout: np.ndarray, split: np.ndarray,
    couplings: np.ndarray, _theory_dim: int,
) -> dict[str, object]:
    scalar = direct[..., 0]
    spread = scalar.std(axis=0)
    noise = np.sqrt(np.mean(split[..., 0] ** 2, axis=0))
    rollout_rmse = np.sqrt(np.mean((rollout[..., 0] - scalar) ** 2, axis=0))
    return {
        "theory_dim": 1,
        "absolute_spearman_g_j_by_level": [
            abs(rank_correlation(scalar[:, k], couplings)) for k in range(scalar.shape[1])
        ],
        "between_system_std_by_level": spread.tolist(),
        "split_half_rmse_by_level": noise.tolist(),
        "separation_snr_by_level": (spread / np.maximum(noise, 1e-12)).tolist(),
        "rollout_direct_rmse_by_level": rollout_rmse.tolist(),
        "normalized_rollout_direct_rmse": float(
            np.sqrt(np.mean((rollout[:, 1:, 0] - scalar[:, 1:]) ** 2))
            / max(float(np.std(scalar[:, 1:])), 1e-12)
        ),
    }


def _multidimensional_metrics(
    direct: np.ndarray, rollout: np.ndarray, split: np.ndarray,
    couplings: np.ndarray, theory_dim: int,
) -> dict[str, object]:
    levels = direct.shape[1]
    # Per-level Frobenius-norm based metrics
    spread = np.array([np.linalg.norm(direct[:, k] - direct[:, k].mean(axis=0), "fro")
                       / np.sqrt(direct.shape[0]) for k in range(levels)])
    noise = np.array([np.sqrt(np.mean(np.linalg.norm(split[:, k], axis=1) ** 2))
                      for k in range(levels)])
    rollout_diff = rollout[:, 1:] - direct[:, 1:]
    rollout_rmse = np.array([np.sqrt(np.mean(np.linalg.norm(rollout_diff[:, k], axis=1) ** 2))
                              for k in range(rollout_diff.shape[1])])
    # Per-dimension Spearman
    spearman_per_dim = []
    for d in range(theory_dim):
        spearman_per_dim.append([
            abs(rank_correlation(direct[:, k, d], couplings)) for k in range(levels)
        ])
    return {
        "theory_dim": theory_dim,
        "frobenius_spread_by_level": spread.tolist(),
        "frobenius_split_noise_by_level": noise.tolist(),
        "frobenius_snr_by_level": (spread / np.maximum(noise, 1e-12)).tolist(),
        "frobenius_rollout_direct_rmse_by_level": rollout_rmse.tolist(),
        "absolute_spearman_g_j_by_dim_and_level": spearman_per_dim,
        "mean_absolute_spearman_g_j_by_level": [
            float(np.mean([spearman_per_dim[d][k] for d in range(theory_dim)]))
            for k in range(levels)
        ],
    }


def plot_results(
    output: Path,
    q_history: list[dict[str, float]],
    qcr_history: list[dict[str, float]],
    couplings: np.ndarray,
    direct: np.ndarray,
    rollout: np.ndarray,
    model: RecurrentRGNetwork2D,
    roots: list[dict[str, float | str | list[float]]],
) -> None:
    if model.theory_dim == 2:
        _plot_results_2d(output, q_history, qcr_history, couplings, direct, rollout, model, roots)
    else:
        _plot_results_1d(output, q_history, qcr_history, couplings, direct, rollout, model, roots)


def _plot_results_1d(
    output: Path,
    q_history: list[dict[str, float]],
    qcr_history: list[dict[str, float]],
    couplings: np.ndarray,
    direct: np.ndarray,
    rollout: np.ndarray,
    model: RecurrentRGNetwork2D,
    roots: list[dict[str, float | str]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
    for history, axis, title in (
        (q_history, axes[0, 0], "Q pretraining"),
        (
            qcr_history,
            axes[0, 1],
            "Joint Q+C+R",
        ),
    ):
        axis.plot([x["epoch"] for x in history], [x["train_total"] for x in history], label="train")
        axis.plot([x["epoch"] for x in history], [x["test_total"] for x in history], label="test")
        axis.set(xlabel="epoch", ylabel="objective", title=title)
        axis.legend(frameon=False)

    for level in range(direct.shape[1]):
        axes[0, 2].scatter(couplings, direct[:, level, 0], s=16, alpha=0.65, label=f"k={level}")
    axes[0, 2].axvline(ISING_2D_CRITICAL_COUPLING, color="black", ls="--", lw=1, label=r"$J_c$")
    axes[0, 2].set(xlabel="J (evaluation only)", ylabel="Q(C^k D_0)", title="Direct RG trajectories")
    axes[0, 2].legend(frameon=False, ncol=2)

    target = direct[:, 1:, 0].reshape(-1)
    prediction = rollout[:, 1:, 0].reshape(-1)
    limits = [min(target.min(), prediction.min()), max(target.max(), prediction.max())]
    axes[1, 0].scatter(target, prediction, s=14, alpha=0.6)
    axes[1, 0].plot(limits, limits, "k--", lw=1)
    axes[1, 0].set(
        xlabel=r"direct $Q(C^k D_0)$",
        ylabel=r"rollout $R^k(Q(D_0))$",
        title="Commuting diagram",
    )

    lower, upper = float(direct.min()), float(direct.max())
    padding = 0.15 * max(upper - lower, 1e-3)
    grid = torch.linspace(lower - padding, upper + padding, 600, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        mapped = model.advance_theory(grid).squeeze(-1).numpy()
    x = grid.squeeze(-1).numpy()
    axes[1, 1].plot(x, mapped - x, label=r"$R(g)-g$")
    axes[1, 1].axhline(0, color="black", lw=1)
    for root in roots:
        axes[1, 1].scatter(root["g"], 0, marker="o" if root["stability"] == "stable" else "x", s=70)
    axes[1, 1].axvspan(lower, upper, color="tab:blue", alpha=0.08, label="trained support")
    axes[1, 1].set(xlabel="g", ylabel=r"$R(g)-g$", title="Learned scalar flow")
    axes[1, 1].legend(frameon=False)

    for system in np.argsort(couplings):
        color = "tab:blue" if couplings[system] < ISING_2D_CRITICAL_COUPLING else "tab:red"
        axes[1, 2].plot(range(direct.shape[1]), direct[system, :, 0], color=color, alpha=0.45)
    axes[1, 2].set(xlabel="RG level k", ylabel="direct g", title="Blue: disordered; red: ordered")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_results_2d(
    output: Path,
    q_history: list[dict[str, float]],
    qcr_history: list[dict[str, float]],
    couplings: np.ndarray,
    direct: np.ndarray,
    rollout: np.ndarray,
    model: RecurrentRGNetwork2D,
    roots: list[dict[str, float | str | list[float]]],
) -> None:
    """2D theory coordinate visualization: scatter, R flow field, commutator, trajectories."""
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)
    levels = direct.shape[1]
    cmap = plt.cm.viridis
    j_norm = (couplings - couplings.min()) / (couplings.max() - couplings.min() + 1e-12)

    # --- Row 0, Col 0: Q pretraining loss ---
    ax = axes[0, 0]
    ax.plot([x["epoch"] for x in q_history], [x["train_total"] for x in q_history], label="train")
    ax.plot([x["epoch"] for x in q_history], [x["test_total"] for x in q_history], label="test")
    ax.set(xlabel="epoch", ylabel="objective", title="Q pretraining")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # --- Row 0, Col 1: QCR joint loss ---
    ax = axes[0, 1]
    title = "Joint Q+C+R"
    ax.plot([x["epoch"] for x in qcr_history], [x["train_total"] for x in qcr_history], label="train")
    ax.plot([x["epoch"] for x in qcr_history], [x["test_total"] for x in qcr_history], label="test")
    ax.set(xlabel="epoch", ylabel="objective", title=title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # --- Row 0, Col 2: 2D scatter of g at all levels, colored by J ---
    ax = axes[0, 2]
    for k in range(levels):
        ax.scatter(direct[:, k, 0], direct[:, k, 1], c=j_norm, cmap=cmap,
                   s=18, alpha=0.55, edgecolors="none")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(couplings.min(), couplings.max()))
    cbar = figure.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("J (evaluation only)")
    ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$",
           title="Direct RG trajectory\n(all levels, colored by J)")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)

    # --- Row 1, Col 0: R flow field quiver + streamlines ---
    ax = axes[1, 0]
    g1_all = direct[:, :, 0].ravel()
    g2_all = direct[:, :, 1].ravel()
    margin = 0.12
    g1_range = (g1_all.min() - margin * (g1_all.max() - g1_all.min()),
                g1_all.max() + margin * (g1_all.max() - g1_all.min()))
    g2_range = (g2_all.min() - margin * (g2_all.max() - g2_all.min()),
                g2_all.max() + margin * (g2_all.max() - g2_all.min()))
    grid_n = 28
    g1_grid = np.linspace(g1_range[0], g1_range[1], grid_n)
    g2_grid = np.linspace(g2_range[0], g2_range[1], grid_n)
    G1, G2 = np.meshgrid(g1_grid, g2_grid)
    flat = np.column_stack((G1.ravel(), G2.ravel()))
    with torch.no_grad():
        flow = model.advance_theory(torch.tensor(flat, dtype=DTYPE)).numpy()
    dG1 = (flow[:, 0] - flat[:, 0]).reshape(grid_n, grid_n)
    dG2 = (flow[:, 1] - flat[:, 1]).reshape(grid_n, grid_n)
    speed = np.sqrt(dG1**2 + dG2**2)
    ax.quiver(G1, G2, dG1, dG2, speed, cmap=plt.cm.coolwarm, alpha=0.8,
              scale=1.0, scale_units="xy", angles="xy", width=0.003)
    ax.streamplot(g1_grid, g2_grid, dG1, dG2, color=speed, cmap=plt.cm.coolwarm,
                  linewidth=0.8, density=1.2, arrowsize=0.6)
    for root in roots:
        g_vals = root["g"]
        marker = "o" if root["stability"] == "stable" else "x"
        ax.scatter(g_vals[0], g_vals[1], marker=marker, s=140,
                   facecolors="white" if root["stability"] == "stable" else "none",
                   edgecolors="black", linewidths=2, zorder=10)
    ax.scatter(g1_all[::5], g2_all[::5], c="black", s=5, alpha=0.25, zorder=5)
    ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$",
           title="Learned R flow field\n(arrows=R(g)-g, streamlines, o=stable, x=unstable)")
    ax.set_xlim(g1_range)
    ax.set_ylim(g2_range)

    # --- Row 1, Col 1: Commuting diagram (2D) ---
    ax = axes[1, 1]
    target = direct[:, 1:, :].reshape(-1, 2)
    prediction = rollout[:, 1:, :].reshape(-1, 2)
    ax.scatter(target[:, 0], target[:, 1], c="tab:blue", s=12, alpha=0.5,
               label="direct Q(C^k D_0)")
    ax.scatter(prediction[:, 0], prediction[:, 1], c="tab:orange", s=12, alpha=0.5,
               label="rollout R^k(Q(D_0))")
    step = max(1, len(target) // 200)
    for t, p in zip(target[::step], prediction[::step]):
        ax.plot([t[0], p[0]], [t[1], p[1]], "k-", lw=0.3, alpha=0.3)
    ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$",
           title="Commuting diagram\n(blue=direct, orange=rollout, lines connect pairs)")
    ax.legend(frameon=False, markerscale=2)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.2)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.2)

    # --- Row 1, Col 2: RG trajectories g_0 -> ... -> g_{levels-1} ---
    ax = axes[1, 2]
    sorted_idx = np.argsort(couplings)
    traj_step = max(1, len(sorted_idx) // 60)
    for idx in sorted_idx[::traj_step]:
        g_traj = direct[idx, :, :]
        color = cmap(j_norm[idx])
        ax.plot(g_traj[:, 0], g_traj[:, 1], "o-", color=color, markersize=4,
                linewidth=0.8, alpha=0.7)
        if levels >= 2:
            ax.annotate("", xy=(g_traj[-1, 0], g_traj[-1, 1]),
                        xytext=(g_traj[-2, 0], g_traj[-2, 1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.6))
    for root in roots:
        g_vals = root["g"]
        ax.scatter(g_vals[0], g_vals[1], marker="*", s=200,
                   facecolors="gold", edgecolors="black", linewidths=1.5, zorder=10)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(couplings.min(), couplings.max()))
    cbar2 = figure.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02)
    cbar2.set_label("J")
    ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$",
           title="RG trajectories\n(g_0->g_1->g_2->g_3, *=fixed point)")

    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--theory-dim", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=384)
    parser.add_argument("--q-epochs", type=int, default=250)
    parser.add_argument("--qcr-epochs", type=int, default=500)
    parser.add_argument("--system-batch", type=int, default=32)
    parser.add_argument("--sample-batch", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--eval-systems", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/joint_qcr_ising_2d")
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    coupling_grid = np.linspace(0.20, 0.68, 37)
    train_mask = np.arange(coupling_grid.size) % 3 != 1
    train_x, train_j = sample_wolff_ensembles(
        coupling_grid[train_mask], args.replicas, args.ensemble_size, args.size, args.seed
    )
    test_x, test_j = sample_wolff_ensembles(
        coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.size, args.seed + 1
    )
    print(f"sampled train={tuple(train_x.shape)} test={tuple(test_x.shape)}", flush=True)
    model = RecurrentRGNetwork2D(
        levels=args.levels, theory_dim=args.theory_dim
    ).to(dtype=DTYPE)
    q_history = train_stage(
        model, train_x, test_x, objective="q", epochs=args.q_epochs,
        system_batch=min(args.system_batch, train_x.shape[0]), sample_batch=args.sample_batch,
        eval_samples=args.eval_samples, eval_systems=args.eval_systems,
        learning_rate=2 * args.learning_rate, seed=args.seed + 2,
    )
    print(f"finished Q pretraining: test objective={q_history[-1]['test_total']:.6g}", flush=True)
    qcr_history = train_stage(
        model, train_x, test_x, objective="qcr", epochs=args.qcr_epochs,
        system_batch=min(args.system_batch, train_x.shape[0]), sample_batch=args.sample_batch,
        eval_samples=args.eval_samples, eval_systems=args.eval_systems,
        learning_rate=args.learning_rate, seed=args.seed + 3,
    )
    print(f"finished joint QCR: test objective={qcr_history[-1]['test_total']:.6g}", flush=True)

    train_levels, train_direct, train_rollout, train_split = trajectory_data(
        model, train_x, args.eval_samples
    )
    test_levels, test_direct, test_rollout, test_split = trajectory_data(
        model, test_x, args.eval_samples
    )
    if args.theory_dim == 1:
        support = [
            float(min(train_direct.min(), test_direct.min())),
            float(max(train_direct.max(), test_direct.max())),
        ]
        roots = find_fixed_points(model, support[0], support[1])
    else:
        support = [
            [float(min(train_direct[..., 0].min(), test_direct[..., 0].min())),
             float(min(train_direct[..., 1].min(), test_direct[..., 1].min()))],
            [float(max(train_direct[..., 0].max(), test_direct[..., 0].max())),
             float(max(train_direct[..., 1].max(), test_direct[..., 1].max()))],
        ]
        roots = find_fixed_points(model, support[0], support[1])
    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "training_contract": {
            "optimized_modules": ["Q", "C", "R"],
            "rsmi_used": False,
            "action_used": False,
            "couplings_used_in_loss": False,
            "couplings_used_for_evaluation_only": True,
            "critical_coupling_used_for_plot_only": float(ISING_2D_CRITICAL_COUPLING),
            "Q_anchor": "split consistency + noncollapse + 2x2 distribution geometry",
            "adaptive_multilevel_separation": True,
            "exact_global_spin_flip_pairing": True,
        },
        "data": {"train_shape": list(train_x.shape), "test_shape": list(test_x.shape)},
        "q_final_train": {k[6:]: v for k, v in q_history[-1].items() if k.startswith("train_")},
        "q_final_test": {k[5:]: v for k, v in q_history[-1].items() if k.startswith("test_")},
        "qcr_final_train": {k[6:]: v for k, v in qcr_history[-1].items() if k.startswith("train_")},
        "qcr_final_test": {k[5:]: v for k, v in qcr_history[-1].items() if k.startswith("test_")},
        "train_metrics": summarize_representation(
            train_direct, train_rollout, train_split, train_j.numpy()
        ),
        "test_metrics": summarize_representation(
            test_direct, test_rollout, test_split, test_j.numpy()
        ),
        "kernel": kernel_metrics(model),
        "learned_R": {"trained_support": support, "fixed_points_in_support": roots},
        "test_physical_observables": physical_metrics(test_levels),
        "q_history": q_history,
        "qcr_history": qcr_history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            name: value
            for name, value in model.state_dict().items()
            if name.startswith(("theory_encoder.", "rg_kernel.", "theory_rg."))
        },
        args.output_dir / "qcr_model.pt",
    )
    plot_results(
        args.output_dir / "qcr_results.png", q_history, qcr_history,
        test_j.numpy(), test_direct, test_rollout, model, roots,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "test_metrics": summary["test_metrics"],
                "kernel": summary["kernel"],
                "learned_R": summary["learned_R"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
