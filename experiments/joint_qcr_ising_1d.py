#!/usr/bin/env python3
"""Jointly learn Q, C, and R without RSMI or an action model.

The microscopic ensemble geometry anchors Q.  C is then optimized indirectly:
the first coarse level must remain statistically nondegenerate, while the same
R cell must make the configuration-space and theory-space RG paths commute.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._utils import QCRWeights, balanced_views
from experiments.ensemble_theory_encoder_ising_1d import (
    encoder_objective,
    fit_probe,
    regression_metrics,
)
from recurrent_rg.ising import DTYPE, exact_decimation_flow, sample_periodic_ensembles
from recurrent_rg.metrics import rank_correlation, mean_pairwise_square
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def qcr_objective(
    model: RecurrentRGNetwork1D,
    microscopic: torch.Tensor,
    *,
    separation_fraction: float = 0.05,
    separation_decay: float = 0.5,
    noise_multiplier: float = 2.0,
    snr_off: float = 2.0,
    snr_on: float = 3.0,
    weights: QCRWeights = QCRWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flow closure with adaptive multilevel anti-collapse and no RSMI term."""
    samples = microscopic.shape[1]
    midpoint = samples // 2
    if midpoint < 1 or samples % 2:
        raise ValueError("QCR batches need an even sample count of at least two")

    levels = model.coarse_grain(microscopic)
    first_theory = torch.stack(
        [model.theory_encoder(level[:, :midpoint]) for level in levels], dim=1
    )
    second_theory = torch.stack(
        [model.theory_encoder(level[:, midpoint:]) for level in levels], dim=1
    )
    direct = 0.5 * (first_theory + second_theory)

    anchor, _ = encoder_objective(
        model.theory_encoder,
        microscopic[:, :midpoint],
        microscopic[:, midpoint:],
    )
    coarse_consistency = F.mse_loss(
        first_theory[:, 1:], second_theory[:, 1:]
    )

    pairwise = torch.stack(
        [mean_pairwise_square(direct[:, level]) for level in range(model.levels)]
    )
    split_noise = torch.stack(
        [
            F.mse_loss(first_theory[:, level], second_theory[:, level])
            for level in range(model.levels)
        ]
    )
    spread = direct.std(dim=0, unbiased=False).square().mean(dim=-1).sqrt()
    snr = spread / split_noise.sqrt().clamp_min(1e-12)
    signal = F.relu(pairwise - noise_multiplier * split_noise)
    separation_terms = []
    separation_gates = []
    for level in range(1, model.levels):
        gate = ((snr[level - 1].detach() - snr_off) / (snr_on - snr_off)).clamp(0.0, 1.0)
        target = separation_fraction * signal[level - 1].detach()
        penalty = F.relu(target - signal[level])
        separation_gates.append(gate)
        separation_terms.append(separation_decay ** (level - 1) * gate * penalty)
    separation = torch.stack(separation_terms).sum()

    predicted = torch.stack(
        [model.advance_theory(direct[:, level]) for level in range(model.levels - 1)],
        dim=1,
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
        "microscopic_pairwise": pairwise[0],
    }
    for level in range(1, model.levels):
        terms[f"separation_level_{level}"] = separation_terms[level - 1]
        terms[f"separation_gate_level_{level}"] = separation_gates[level - 1]
        terms[f"pairwise_level_{level}"] = pairwise[level]
        terms[f"split_noise_level_{level}"] = split_noise[level]
        terms[f"resolvable_signal_level_{level}"] = signal[level]
        terms[f"snr_level_{level}"] = snr[level]
    terms["resolvable_signal_level_0"] = signal[0]
    terms["snr_level_0"] = snr[0]
    return total, terms


def evaluate_objective(
    model: RecurrentRGNetwork1D,
    configurations: torch.Tensor,
    samples: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        total, terms = qcr_objective(model, balanced_views(configurations, samples))
    return {"total": float(total)} | {name: float(value) for name, value in terms.items()}


def train(
    model: RecurrentRGNetwork1D,
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
    parameters = itertools.chain(
        model.theory_encoder.parameters(),
        model.rg_kernel.parameters(),
        model.theory_rg.parameters(),
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    record_every = max(1, epochs // 50)
    history: list[dict[str, float]] = []

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
        loss, _ = qcr_objective(model, batch)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            train_metrics = evaluate_objective(model, train_x, eval_samples)
            test_metrics = evaluate_objective(model, test_x, eval_samples)
            history.append(
                {"epoch": epoch + 1}
                | {f"train_{key}": value for key, value in train_metrics.items()}
                | {f"test_{key}": value for key, value in test_metrics.items()}
            )
    return history


def direct_and_rollout(
    model: RecurrentRGNetwork1D,
    configurations: torch.Tensor,
    samples: int,
) -> tuple[list[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        subset = balanced_views(configurations, samples)
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


def representation_metrics(
    direct: np.ndarray,
    rollout: np.ndarray,
    split_difference: np.ndarray,
    couplings: np.ndarray,
) -> dict[str, object]:
    exact = exact_decimation_flow(torch.tensor(couplings, dtype=DTYPE), direct.shape[1]).numpy()
    direct_scalar = direct[..., 0]
    rollout_scalar = rollout[..., 0]
    split_scalar = split_difference[..., 0]
    spread = np.std(direct_scalar, axis=0)
    split_rmse = np.sqrt(np.mean(split_scalar**2, axis=0))
    commutator_rmse = np.sqrt(np.mean((rollout_scalar - direct_scalar) ** 2, axis=0))
    return {
        "absolute_spearman_g_exact_j_by_level": [
            abs(rank_correlation(direct_scalar[:, level], exact[:, level]))
            for level in range(direct.shape[1])
        ],
        "between_system_std_by_level": spread.tolist(),
        "split_half_rmse_by_level": split_rmse.tolist(),
        "separation_snr_by_level": (spread / np.maximum(split_rmse, 1e-12)).tolist(),
        "rollout_direct_rmse_by_level": commutator_rmse.tolist(),
        "normalized_rollout_direct_rmse": float(
            np.sqrt(np.mean((rollout_scalar[:, 1:] - direct_scalar[:, 1:]) ** 2))
            / max(float(np.std(direct_scalar[:, 1:])), 1e-12)
        ),
    }


def nearest_neighbor_correlations(levels: list[torch.Tensor]) -> np.ndarray:
    values = []
    for configurations in levels:
        spins = configurations.squeeze(-1)
        values.append(
            (spins * torch.roll(spins, -1, dims=2)).mean(dim=(1, 2)).numpy()
        )
    return np.stack(values, axis=1)


def probe_metrics(
    train_direct: np.ndarray,
    test_direct: np.ndarray,
    train_levels: list[torch.Tensor],
    test_levels: list[torch.Tensor],
) -> list[dict[str, float]]:
    train_observables = nearest_neighbor_correlations(train_levels)
    test_observables = nearest_neighbor_correlations(test_levels)
    results = []
    for level in range(train_direct.shape[1]):
        prediction = fit_probe(
            train_direct[:, level, 0],
            train_observables[:, level],
            test_direct[:, level, 0],
        )
        results.append(regression_metrics(test_observables[:, level], prediction))
    return results


def kernel_truth_table(model: RecurrentRGNetwork1D) -> dict[str, object]:
    patterns = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=3)), dtype=DTYPE
    )
    model.eval()
    with torch.no_grad():
        output = model.rg_kernel.field_head(
            model.rg_kernel.local_encoder(patterns)
        ).squeeze(-1)
        output = torch.where(output >= 0, 1.0, -1.0)
    majority = torch.sign(patterns.sum(dim=1))
    center = patterns[:, 1]

    def sign_invariant_accuracy(reference: torch.Tensor) -> float:
        accuracy = (output == reference).to(DTYPE).mean()
        flipped = (-output == reference).to(DTYPE).mean()
        return float(torch.maximum(accuracy, flipped))

    return {
        "patterns_left_center_right": patterns.int().tolist(),
        "kernel_output": output.int().tolist(),
        "z2_equivariance_fraction": float((output == -output.flip(0)).to(DTYPE).mean()),
        "majority_accuracy_up_to_global_sign": sign_invariant_accuracy(majority),
        "center_decimation_accuracy_up_to_global_sign": sign_invariant_accuracy(center),
    }


def plot_results(
    output: Path,
    history: list[dict[str, float]],
    test_j: np.ndarray,
    test_direct: np.ndarray,
    test_rollout: np.ndarray,
    metrics: dict[str, object],
) -> None:
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    axes[0, 0].plot(epochs, [entry["train_total"] for entry in history], label="train")
    axes[0, 0].plot(epochs, [entry["test_total"] for entry in history], label="test")
    axes[0, 0].set(xlabel="epoch", ylabel="objective", title="Q+C+R, no RSMI")
    axes[0, 0].legend(frameon=False)

    exact = exact_decimation_flow(torch.tensor(test_j, dtype=DTYPE), test_direct.shape[1]).numpy()
    for level in range(test_direct.shape[1]):
        axes[0, 1].scatter(
            exact[:, level], test_direct[:, level, 0], s=13, alpha=0.55, label=f"level {level}"
        )
    axes[0, 1].set(
        xlabel="exact decimation J (evaluation only)",
        ylabel="direct Q(D_k)",
        title="Learned theory coordinates",
    )
    axes[0, 1].legend(frameon=False)

    direct = test_direct[:, 1:, 0].reshape(-1)
    rollout = test_rollout[:, 1:, 0].reshape(-1)
    limits = [min(direct.min(), rollout.min()), max(direct.max(), rollout.max())]
    axes[1, 0].scatter(direct, rollout, s=12, alpha=0.55)
    axes[1, 0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        xlabel="Q(C^k D_0)", ylabel="R^k(Q(D_0))", title="Commuting flow on held-out J"
    )

    levels = np.arange(test_direct.shape[1])
    spread = np.asarray(metrics["between_system_std_by_level"])
    noise = np.asarray(metrics["split_half_rmse_by_level"])
    axes[1, 1].plot(levels, spread, marker="o", label="between-system std")
    axes[1, 1].plot(levels, noise, marker="o", label="split-half RMSE")
    axes[1, 1].set(xlabel="RG level", ylabel="g scale", title="Separation may decay")
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=128)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--system-batch", type=int, default=32)
    parser.add_argument("--sample-batch", type=int, default=24)
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--q-checkpoint",
        type=Path,
        default=Path("results/ensemble_encoder_ising_1d/theory_encoder.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/joint_qcr_ising_1d_adaptive_sep"),
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

    model = RecurrentRGNetwork1D(levels=args.levels, theory_dim=1).to(dtype=DTYPE)
    if not args.q_checkpoint.exists():
        raise FileNotFoundError(
            f"missing Q-only checkpoint: {args.q_checkpoint}; run the Q-only experiment first"
        )
    model.theory_encoder.load_state_dict(
        torch.load(args.q_checkpoint, map_location="cpu", weights_only=True)
    )
    initial_test = evaluate_objective(model, test_x, args.eval_samples)
    history = train(
        model,
        train_x,
        test_x,
        epochs=args.epochs,
        system_batch=min(args.system_batch, train_x.shape[0]),
        sample_batch=args.sample_batch,
        eval_samples=args.eval_samples,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
    )

    train_levels, train_direct, train_rollout, train_split = direct_and_rollout(
        model, train_x, args.eval_samples
    )
    test_levels, test_direct, test_rollout, test_split = direct_and_rollout(
        model, test_x, args.eval_samples
    )
    train_metrics = representation_metrics(
        train_direct, train_rollout, train_split, train_j_tensor.numpy()
    )
    test_metrics = representation_metrics(
        test_direct, test_rollout, test_split, test_j_tensor.numpy()
    )
    test_metrics["nearest_neighbor_correlation_probe_by_level"] = probe_metrics(
        train_direct, test_direct, train_levels, test_levels
    )
    test_metrics["kernel"] = kernel_truth_table(model)

    summary = {
        "config": vars(args)
        | {
            "q_checkpoint": str(args.q_checkpoint),
            "output_dir": str(args.output_dir),
        },
        "training_contract": {
            "optimized_modules": ["Q", "C", "R"],
            "rsmi_used": False,
            "action_used": False,
            "couplings_used_in_loss": False,
            "Q_initialization": str(args.q_checkpoint),
            "one_step_anti_collapse_only": False,
            "anti_collapse_levels": "all coarse levels",
            "anti_collapse_weight_by_transition": [
                0.5**level for level in range(args.levels - 1)
            ],
            "anti_collapse_gate": "source-level SNR piecewise linear",
            "snr_off": 2.0,
            "snr_on": 3.0,
            "gate_and_source_target_stop_gradient": True,
            "exact_global_spin_flip_pairing": True,
        },
        "loss_weights": QCRWeights().__dict__,
        "initial_test_objective": initial_test,
        "final_train_objective": {
            key[6:]: value for key, value in history[-1].items() if key.startswith("train_")
        },
        "final_test_objective": {
            key[5:]: value for key, value in history[-1].items() if key.startswith("test_")
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    retained = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(("theory_encoder.", "rg_kernel.", "theory_rg."))
    }
    torch.save(retained, args.output_dir / "qcr_model.pt")
    plot_results(
        args.output_dir / "qcr_results.png",
        history,
        test_j_tensor.numpy(),
        test_direct,
        test_rollout,
        test_metrics,
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
