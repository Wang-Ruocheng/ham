#!/usr/bin/env python3
"""Learn a one-dimensional theory coordinate from Ising ensembles only.

This experiment deliberately uses only ``g = Q(D)`` from the recurrent-RG
network.  Couplings and physical observables are excluded from optimization and
are used solely for post-training evaluation.
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

from recurrent_rg.ising import DTYPE, sample_periodic_ensembles
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def local_pattern_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """Return empirical probabilities of the eight periodic three-spin patterns."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = 4 * bits + 2 * torch.roll(bits, -1, dims=2) + torch.roll(bits, -2, dims=2)
    return F.one_hot(pattern, num_classes=8).to(configurations.dtype).mean(dim=(1, 2))


def upper_triangle(values: torch.Tensor) -> torch.Tensor:
    indices = torch.triu_indices(values.shape[0], values.shape[1], offset=1)
    return values[indices[0], indices[1]]


def correlation_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    return 1.0 - (left * right).mean() / denominator.clamp_min(1e-12)


def encoder_objective(
    encoder: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Subset invariance, noncollapse, and distribution-geometry preservation."""
    first_g = encoder(first)
    second_g = encoder(second)
    consistency = F.mse_loss(first_g, second_g)

    joined_g = 0.5 * (first_g + second_g)
    std = joined_g.std(dim=0, unbiased=False)
    variance = F.relu(1.0 - std).square().mean()

    centered = joined_g - joined_g.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(joined_g.shape[0], 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diagonal.square().mean()

    patterns = 0.5 * (
        local_pattern_distribution(first) + local_pattern_distribution(second)
    )
    latent_distance = upper_triangle(torch.cdist(joined_g, joined_g))
    distribution_distance = upper_triangle(torch.cdist(patterns, patterns))
    geometry = correlation_loss(latent_distance, distribution_distance)

    total = 10.0 * consistency + variance + covariance_loss + geometry
    return total, {
        "consistency": consistency,
        "variance": variance,
        "covariance": covariance_loss,
        "geometry": geometry,
    }


def fixed_views(configurations: torch.Tensor, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    if 2 * samples > configurations.shape[1]:
        raise ValueError("evaluation requires at least twice eval_samples configurations")
    return configurations[:, :samples], configurations[:, samples : 2 * samples]


def evaluate_loss(
    encoder: torch.nn.Module,
    configurations: torch.Tensor,
    samples: int,
) -> dict[str, float]:
    encoder.eval()
    with torch.no_grad():
        total, terms = encoder_objective(encoder, *fixed_views(configurations, samples))
    return {"total": float(total)} | {name: float(value) for name, value in terms.items()}


def polynomial_design(values: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.stack([values**power for power in range(degree + 1)], axis=1)


def fit_probe(train_g: np.ndarray, train_y: np.ndarray, test_g: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(polynomial_design(train_g), train_y, rcond=None)
    return polynomial_design(test_g) @ coefficients


def physical_observables(configurations: torch.Tensor) -> np.ndarray:
    spins = configurations.squeeze(-1)
    values = []
    for distance in (1, 2, 4, 8):
        correlation = (spins * torch.roll(spins, -distance, dims=2)).mean(dim=(1, 2))
        values.append(correlation.numpy())
    return np.stack(values, axis=1)


def latent_values(encoder: torch.nn.Module, configurations: torch.Tensor) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        return encoder(configurations).squeeze(-1).numpy()


def grouped_statistics(values: np.ndarray, couplings: np.ndarray) -> tuple[float, float]:
    unique = np.unique(couplings)
    groups = [values[couplings == coupling] for coupling in unique]
    within = float(np.mean([np.std(group) for group in groups]))
    between = float(np.std([np.mean(group) for group in groups]))
    return within, between


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.mean((prediction - target) ** 2)
    variance = np.mean((target - target.mean(axis=0, keepdims=True)) ** 2)
    return {
        "rmse": float(np.sqrt(residual)),
        "normalized_rmse": float(np.sqrt(residual / max(variance, 1e-12))),
        "r2": float(1.0 - residual / max(variance, 1e-12)),
    }


def train(
    encoder: torch.nn.Module,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    *,
    epochs: int,
    system_batch: int,
    sample_batch: int,
    eval_samples: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    record_every = max(1, epochs // 50)

    for epoch in range(epochs):
        encoder.train()
        systems = torch.randperm(train_x.shape[0], generator=generator)[:system_batch]
        samples = torch.randperm(train_x.shape[1], generator=generator)[: 2 * sample_batch]
        batch = train_x[systems][:, samples]
        first, second = batch[:, :sample_batch], batch[:, sample_batch:]

        optimizer.zero_grad(set_to_none=True)
        loss, _ = encoder_objective(encoder, first, second)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            train_metrics = evaluate_loss(encoder, train_x, eval_samples)
            test_metrics = evaluate_loss(encoder, test_x, eval_samples)
            history.append(
                {"epoch": epoch + 1}
                | {f"train_{key}": value for key, value in train_metrics.items()}
                | {f"test_{key}": value for key, value in test_metrics.items()}
            )
    return history


def plot_results(
    output: Path,
    history: list[dict[str, float]],
    train_j: np.ndarray,
    train_g: np.ndarray,
    test_j: np.ndarray,
    test_g: np.ndarray,
    test_observables: np.ndarray,
    predicted_observables: np.ndarray,
) -> None:
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    axes[0].plot(epochs, [entry["train_total"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["test_total"] for entry in history], label="test")
    axes[0].set(xlabel="epoch", ylabel="encoder objective", title="Q-only training")
    axes[0].legend(frameon=False)

    axes[1].scatter(train_j, train_g, s=13, alpha=0.55, label="train J")
    axes[1].scatter(test_j, test_g, s=20, alpha=0.8, label="held-out J")
    axes[1].set(xlabel="J (evaluation only)", ylabel="learned g", title="Ensemble coordinate")
    axes[1].legend(frameon=False)

    labels = ("r=1", "r=2", "r=4", "r=8")
    for index, label in enumerate(labels):
        axes[2].scatter(
            test_observables[:, index], predicted_observables[:, index], s=13, alpha=0.6,
            label=label,
        )
    limits = [
        min(test_observables.min(), predicted_observables.min()),
        max(test_observables.max(), predicted_observables.max()),
    ]
    axes[2].plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axes[2].set(
        xlabel="measured correlation",
        ylabel="cubic probe from g",
        title="Held-out physical sufficiency",
    )
    axes[2].legend(frameon=False, ncol=2)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--system-batch", type=int, default=48)
    parser.add_argument("--sample-batch", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/ensemble_encoder_ising_1d")
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    coupling_grid = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(coupling_grid.size) % 3 != 1
    train_x, train_j_tensor = sample_periodic_ensembles(
        coupling_grid[train_mask], args.replicas, args.ensemble_size, args.sites, args.seed
    )
    test_x, test_j_tensor = sample_periodic_ensembles(
        coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.sites, args.seed + 1
    )

    # Instantiate the existing network unchanged and optimize only its Q module.
    model = RecurrentRGNetwork1D(levels=2, theory_dim=1).to(dtype=DTYPE)
    encoder = model.theory_encoder
    initial_test_loss = evaluate_loss(encoder, test_x, args.eval_samples)
    initial_train_g = latent_values(encoder, train_x)
    initial_test_g = latent_values(encoder, test_x)
    initial_orientation = (
        1.0 if rank_correlation(initial_train_g, train_j_tensor.numpy()) >= 0.0 else -1.0
    )
    initial_train_g *= initial_orientation
    initial_test_g *= initial_orientation
    initial_within, initial_between = grouped_statistics(
        initial_test_g, test_j_tensor.numpy()
    )
    train_observables = physical_observables(train_x)
    test_observables = physical_observables(test_x)
    initial_predicted_j = fit_probe(
        initial_train_g, train_j_tensor.numpy(), initial_test_g
    )
    initial_predicted_observables = np.column_stack(
        [
            fit_probe(initial_train_g, train_observables[:, index], initial_test_g)
            for index in range(train_observables.shape[1])
        ]
    )
    history = train(
        encoder,
        train_x,
        test_x,
        epochs=args.epochs,
        system_batch=min(args.system_batch, train_x.shape[0]),
        sample_batch=args.sample_batch,
        eval_samples=args.eval_samples,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
    )

    train_g = latent_values(encoder, train_x)
    test_g = latent_values(encoder, test_x)
    train_j = train_j_tensor.numpy()
    test_j = test_j_tensor.numpy()
    orientation = 1.0 if rank_correlation(train_g, train_j) >= 0.0 else -1.0
    train_g *= orientation
    test_g *= orientation

    predicted_j = fit_probe(train_g, train_j, test_g)
    predicted_observables = np.column_stack(
        [
            fit_probe(train_g, train_observables[:, index], test_g)
            for index in range(train_observables.shape[1])
        ]
    )
    within, between = grouped_statistics(test_g, test_j)

    first_test, second_test = fixed_views(test_x, args.eval_samples)
    first_g = latent_values(encoder, first_test) * orientation
    second_g = latent_values(encoder, second_test) * orientation
    split_half_rmse = float(np.sqrt(np.mean((first_g - second_g) ** 2)))

    metrics = {
        "initial_test_objective": initial_test_loss,
        "initial_representation": {
            "absolute_spearman_g_j_test": abs(
                rank_correlation(initial_test_g, test_j)
            ),
            "test_within_between_ratio": initial_within
            / max(initial_between, 1e-12),
            "heldout_j_cubic_probe": regression_metrics(test_j, initial_predicted_j),
            "heldout_correlations_cubic_probe": regression_metrics(
                test_observables, initial_predicted_observables
            ),
        },
        "final_train_objective": {key[6:]: value for key, value in history[-1].items() if key.startswith("train_")},
        "final_test_objective": {key[5:]: value for key, value in history[-1].items() if key.startswith("test_")},
        "absolute_spearman_g_j_train": abs(rank_correlation(train_g, train_j)),
        "absolute_spearman_g_j_test": abs(rank_correlation(test_g, test_j)),
        "test_split_half_rmse": split_half_rmse,
        "test_within_j_std": within,
        "test_between_j_std": between,
        "test_within_between_ratio": within / max(between, 1e-12),
        "heldout_j_cubic_probe": regression_metrics(test_j, predicted_j),
        "heldout_correlations_cubic_probe": regression_metrics(
            test_observables, predicted_observables
        ),
    }
    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "training_contract": {
            "module_optimized": "existing EnsembleTheoryEncoder Q only",
            "couplings_used_in_loss": False,
            "physical_observables_used_in_loss": False,
            "positive_views": "two disjoint subsets of each ensemble",
            "distribution_geometry": "three-spin empirical pattern probabilities",
            "latent_dimension": 1,
        },
        "metrics": metrics,
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(encoder.state_dict(), args.output_dir / "theory_encoder.pt")
    plot_results(
        args.output_dir / "encoder_results.png",
        history,
        train_j,
        train_g,
        test_j,
        test_g,
        test_observables,
        predicted_observables,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
