#!/usr/bin/env python3
"""Aggregate recurrent-RG 1D/2D multi-seed results and plot diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/recurrent_rg")
RUNS = {
    "1D Ising": ROOT / "multiseed_1d",
    "2D Ising": ROOT / "multiseed_2d",
}


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def load_runs(directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("seed_*/summary.json"))
    ]


def action_errors(name: str, run: dict[str, object]) -> tuple[float, float]:
    metrics = run["metrics"]
    # 2D results nest action metrics one level deeper than 1D results.
    if isinstance(metrics, dict) and "action" in metrics:
        metrics = metrics["action"]
    return (
        float(metrics["train_microscopic_action_relative_rmse"]),
        float(metrics["test_microscopic_action_relative_rmse"]),
    )


def main() -> None:
    loaded = {name: load_runs(directory) for name, directory in RUNS.items()}
    if any(len(runs) != 3 for runs in loaded.values()):
        raise RuntimeError("expected exactly three completed seeds for each model")

    aggregate: dict[str, object] = {
        "action_relative_rmse_definition": (
            "sqrt(sum((A_pred-centered - A_exact-centered)^2) / "
            "sum((A_exact-centered)^2)); centering is per physical ensemble"
        ),
        "loss_curve_subset": "8 evenly spaced systems x first 16 samples",
        "models": {},
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    error_figure, error_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for axis, error_axis, (name, runs) in zip(axes, error_axes, loaded.items()):
        epochs = np.asarray([entry["epoch"] for entry in runs[0]["history"]])
        train_curves = np.asarray(
            [[entry["train_total"] for entry in run["history"]] for run in runs]
        )
        test_curves = np.asarray(
            [[entry["test_total"] for entry in run["history"]] for run in runs]
        )
        for curve in train_curves:
            axis.plot(epochs, curve, color="tab:blue", alpha=0.22, linewidth=1)
        for curve in test_curves:
            axis.plot(epochs, curve, color="tab:orange", alpha=0.22, linewidth=1)
        for curves, color, label in (
            (train_curves, "tab:blue", "train mean"),
            (test_curves, "tab:orange", "test mean"),
        ):
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1)
            axis.plot(epochs, mean, color=color, linewidth=2, label=label)
            axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14)
        axis.set(title=name, xlabel="epoch", ylabel="joint objective")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

        seeds = [int(run["config"]["seed"]) for run in runs]
        errors = np.asarray([action_errors(name, run) for run in runs])
        positions = np.arange(len(seeds))
        error_axis.bar(positions - 0.18, errors[:, 0], 0.36, label="train")
        error_axis.bar(positions + 0.18, errors[:, 1], 0.36, label="test")
        error_axis.set(
            title=name,
            xlabel="seed",
            ylabel="centered action relative RMSE",
            xticks=positions,
            xticklabels=seeds,
        )
        error_axis.grid(axis="y", alpha=0.25)
        error_axis.legend(frameon=False)

        per_seed = []
        for run, (train_error, test_error) in zip(runs, errors):
            history = run["history"]
            per_seed.append(
                {
                    "seed": int(run["config"]["seed"]),
                    "initial_train_loss": float(history[0]["train_total"]),
                    "final_train_loss": float(history[-1]["train_total"]),
                    "final_test_loss": float(history[-1]["test_total"]),
                    "minimum_test_loss": float(
                        min(entry["test_total"] for entry in history)
                    ),
                    "train_action_relative_rmse": float(train_error),
                    "test_action_relative_rmse": float(test_error),
                }
            )
        aggregate["models"][name] = {
            "seeds": per_seed,
            "final_train_loss": mean_std([item["final_train_loss"] for item in per_seed]),
            "final_test_loss": mean_std([item["final_test_loss"] for item in per_seed]),
            "train_action_relative_rmse": mean_std(
                [item["train_action_relative_rmse"] for item in per_seed]
            ),
            "test_action_relative_rmse": mean_std(
                [item["test_action_relative_rmse"] for item in per_seed]
            ),
        }

    ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ROOT / "multiseed_loss_curves.png", dpi=180)
    error_figure.savefig(ROOT / "multiseed_action_relative_error.png", dpi=180)
    (ROOT / "multiseed_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
