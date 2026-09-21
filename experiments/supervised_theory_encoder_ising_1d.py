#!/usr/bin/env python3
"""Supervised sanity check: can Q(D) alone learn the coupling J from ensembles?

Unlike the unsupervised encoder experiment, here we train the existing
``EnsembleTheoryEncoder`` Q directly against the known coupling labels J,
using only ensemble data D as input.  This tests the representational
capacity of the Q architecture and probes sample-complexity limits.

Because the 1D Ising model is exactly solvable, all ensemble information
about J is contained in the nearest-neighbor bond distribution
``P(s_i s_{i+1})`` with ``P(same) = sigmoid(2J)``.  A sufficiently
expressive Q should therefore recover J with error that decays as
``1/sqrt(n_samples)`` once the network can resolve the bond statistics.

This script does NOT modify any physics module; only the existing
``model.theory_encoder`` submodule is optimized.
"""

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

from recurrent_rg.ising import DTYPE, sample_periodic_ensembles
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-couplings", type=int, default=40)
    parser.add_argument("--test-couplings", type=int, default=41)
    parser.add_argument("--coupling-min", type=float, default=0.01)
    parser.add_argument("--coupling-max", type=float, default=2.0)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--samples-per-system", type=int, default=256)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--sample-sweep",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024],
        help="ensemble sizes for the post-training sample-complexity sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/supervised_theory_encoder"),
    )
    return parser.parse_args()


def predict(encoder: torch.nn.Module, configurations: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return encoder(configurations).squeeze(-1).numpy()


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = predictions - targets
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "max_abs": float(np.abs(residual).max()),
    }



def plot_results(
    path: Path,
    history: list[dict[str, float]],
    train_j: np.ndarray,
    train_pred: np.ndarray,
    test_j: np.ndarray,
    test_pred: np.ndarray,
    sweep_sizes: list[int],
    sweep_rmse: list[float],
    sweep_mae: list[float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.loglog([h["epoch"] for h in history], [h["train_mse"] for h in history],
              label="train", lw=1.5)
    ax.loglog([h["epoch"] for h in history], [h["test_mse"] for h in history],
              label="test", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Q(D) -> J supervised training")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.scatter(train_j, train_pred, s=14, alpha=0.6, label="train")
    ax.scatter(test_j, test_pred, s=14, alpha=0.7, label="test")
    lims = [min(train_j.min(), test_j.min()) * 0.95,
            max(train_j.max(), test_j.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="perfect")
    ax.set_xlabel("true J")
    ax.set_ylabel("Q(D) prediction")
    ax.set_title("Learned coupling vs truth")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.loglog(sweep_sizes, sweep_rmse, "o-", label="RMSE", lw=1.5)
    ax.loglog(sweep_sizes, sweep_mae, "s--", label="MAE", lw=1.5)
    ref = sweep_rmse[0] * np.sqrt(sweep_sizes[0] / np.asarray(sweep_sizes, dtype=float))
    ax.loglog(sweep_sizes, ref, "k:", lw=1, label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("samples per ensemble")
    ax.set_ylabel("J prediction error")
    ax.set_title("Sample-complexity sweep (test systems)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_couplings = np.linspace(
        args.coupling_min, args.coupling_max, args.train_couplings
    )
    test_couplings = np.linspace(
        args.coupling_min, args.coupling_max, args.test_couplings
    )

    train_x, train_j = sample_periodic_ensembles(
        train_couplings, args.replicas, args.samples_per_system, args.sites, args.seed
    )
    test_x, test_j = sample_periodic_ensembles(
        test_couplings, args.replicas, args.samples_per_system, args.sites, args.seed + 1
    )

    # Instantiate the existing architecture unchanged; optimize only Q.
    model = RecurrentRGNetwork1D(
        levels=2, hidden=args.hidden, theory_dim=1
    ).to(dtype=DTYPE)
    encoder = model.theory_encoder

    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, float]] = []
    log_every = max(1, args.epochs // 100)
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        optimizer.zero_grad()
        prediction = encoder(train_x).squeeze(-1)
        loss = torch.mean((prediction - train_j) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == args.epochs:
            encoder.eval()
            with torch.no_grad():
                test_mse = torch.mean(
                    (encoder(test_x).squeeze(-1) - test_j) ** 2
                ).item()
            history.append(
                {"epoch": float(epoch), "train_mse": loss.item(), "test_mse": test_mse}
            )
            if len(history) == 1 or len(history) % 10 == 0:
                print(
                    f"epoch {epoch:5d}  train MSE {loss.item():.4e}  "
                    f"test MSE {test_mse:.4e}"
                )

    encoder.eval()
    train_pred = predict(encoder, train_x)
    test_pred = predict(encoder, test_x)
    train_j_np = train_j.numpy()
    test_j_np = test_j.numpy()

    metrics = {
        "train": regression_metrics(train_j_np, train_pred),
        "test": regression_metrics(test_j_np, test_pred),
    }

    # Sample-complexity sweep on the held-out systems.
    sweep_sizes: list[int] = []
    sweep_rmse: list[float] = []
    sweep_mae: list[float] = []
    for n in args.sample_sweep:
        if n > test_x.shape[1]:
            continue
        preds = predict(encoder, test_x[:, :n])
        stats = regression_metrics(test_j_np, preds)
        sweep_sizes.append(int(n))
        sweep_rmse.append(stats["rmse"])
        sweep_mae.append(stats["mae"])
    metrics["sample_sweep"] = {
        "sizes": sweep_sizes,
        "rmse": sweep_rmse,
        "mae": sweep_mae,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.output_dir / "theory_encoder_supervised.pt")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": vars(args) | {"output_dir": str(args.output_dir)},
                "training_contract": {
                    "module_optimized": "existing EnsembleTheoryEncoder Q only",
                    "loss": "MSE(Q(D), J) — supervised",
                    "physics_modules_touched": False,
                },
                "metrics": metrics,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    plot_results(
        args.output_dir / "supervised_Q_results.png",
        history,
        train_j_np,
        train_pred,
        test_j_np,
        test_pred,
        sweep_sizes,
        sweep_rmse,
        sweep_mae,
    )

    print("\n=== Final metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
