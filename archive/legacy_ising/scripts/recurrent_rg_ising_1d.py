#!/usr/bin/env python3
"""Minimal recurrent-RG experiment for the periodic 1D Ising model.

The experiment deliberately separates two kinds of latent variables:

* a configuration-level coarse spin ``h`` produced by a learned stochastic
  block kernel P(h | s_left, s_right), and
* a theory-level state ``z_k`` shared by the whole ensemble at RG level k.

First, the two-spin kernel is learned by maximizing exact real-space mutual
information I(H:E).  We then apply the same kernel twice to exact finite-chain
Boltzmann distributions.  A small shared recurrent cell evolves z_k, while an
action decoder predicts the nearest-neighbour effective action

    A_k[s] = -J_k sum_i s_i s_{i+1}.

All probability distributions are obtained by exact enumeration, so the smoke
test contains no Monte Carlo or mutual-information-estimator error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


DTYPE = torch.float64
VISIBLE_STATES = torch.tensor(
    [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
    dtype=DTYPE,
)


def enumerate_spins(length: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=length)), dtype=DTYPE
    )


def bond_sum(states: torch.Tensor) -> torch.Tensor:
    return (states * torch.roll(states, -1, dims=1)).sum(dim=1)


def ising_distribution(states: torch.Tensor, coupling: float) -> torch.Tensor:
    return torch.softmax(float(coupling) * bond_sum(states), dim=0)


def enumerate_visible_environment(
    length: int, coupling: float, buffer: int
) -> torch.Tensor:
    """Exact P(V,E), with V=(s0,s1) and a buffer on each side."""
    states = enumerate_spins(length)
    probability = ising_distribution(states, coupling)
    visible = states[:, :2]
    environment_indices = list(range(2 + buffer, length - buffer))
    environment = states[:, environment_indices]
    visible_index = ((visible > 0).long() * torch.tensor([1, 2])).sum(dim=1)
    powers = 2 ** torch.arange(len(environment_indices), dtype=torch.long)
    environment_index = ((environment > 0).long() * powers).sum(dim=1)
    joint = torch.zeros(4, 2 ** len(environment_indices), dtype=DTYPE)
    joint.index_put_((visible_index, environment_index), probability, accumulate=True)
    return joint


def kernel_weights(angle: torch.Tensor, strength: float) -> torch.Tensor:
    return strength * torch.stack((torch.cos(angle), torch.sin(angle)))


def mutual_information_bits(weights: torch.Tensor, joint_ve: torch.Tensor) -> torch.Tensor:
    p_plus_given_v = torch.sigmoid(2.0 * (VISIBLE_STATES @ weights))
    p_h_given_v = torch.stack((1.0 - p_plus_given_v, p_plus_given_v), dim=1)
    joint_he = torch.einsum("ve,vh->he", joint_ve, p_h_given_v)
    p_h = joint_he.sum(dim=1, keepdim=True)
    p_e = joint_he.sum(dim=0, keepdim=True)
    eps = torch.finfo(DTYPE).tiny
    mi = joint_he * (
        torch.log(joint_he + eps) - torch.log(p_h + eps) - torch.log(p_e + eps)
    )
    return mi.sum() / math.log(2.0)


@dataclass
class KernelResult:
    seed: int
    angle: float
    weights: list[float]
    mean_mi_bits: float


def learn_kernel(
    joints: list[torch.Tensor], strength: float, seeds: int, steps: int, lr: float
) -> KernelResult:
    results: list[KernelResult] = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        angle = nn.Parameter(2.0 * math.pi * torch.rand((), dtype=DTYPE))
        optimizer = torch.optim.Adam([angle], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            weights = kernel_weights(angle, strength)
            loss = -torch.stack(
                [mutual_information_bits(weights, joint) for joint in joints]
            ).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            weights = kernel_weights(angle, strength)
            mean_mi = torch.stack(
                [mutual_information_bits(weights, joint) for joint in joints]
            ).mean()
        results.append(
            KernelResult(
                seed=seed,
                angle=float(angle.detach() % (2.0 * math.pi)),
                weights=[float(value) for value in weights],
                mean_mi_bits=float(mean_mi),
            )
        )
    return max(results, key=lambda result: result.mean_mi_bits)


def block_transition(
    fine_states: torch.Tensor, coarse_states: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return T[x,h]=P(h|x) for independent two-site block kernels."""
    if fine_states.shape[1] != 2 * coarse_states.shape[1]:
        raise ValueError("coarse chain must have exactly half as many sites")
    pairs = fine_states.reshape(fine_states.shape[0], -1, 2)
    fields = torch.einsum("nbc,c->nb", pairs, weights)
    signed_fields = fields[:, None, :] * coarse_states[None, :, :]
    log_probability = -torch.nn.functional.softplus(-2.0 * signed_fields).sum(dim=2)
    transition = torch.exp(log_probability)
    return transition / transition.sum(dim=1, keepdim=True)


def apply_transition(probability: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    coarse = probability @ transition
    return coarse / coarse.sum()


def nearest_neighbour_projection(
    probability: torch.Tensor, states: torch.Tensor, iterations: int = 80
) -> float:
    """Information projection of a distribution onto P_J(s) propto exp(J B(s))."""
    target_bond = float((probability * bond_sum(states)).sum())
    low, high = -4.0, 4.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        model = ising_distribution(states, middle)
        model_bond = float((model * bond_sum(states)).sum())
        if model_bond < target_bond:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def kl_bits_per_site(
    target: torch.Tensor, states: torch.Tensor, coupling: float
) -> float:
    model = ising_distribution(states, coupling)
    eps = torch.finfo(DTYPE).tiny
    kl_bits = (target * (torch.log(target + eps) - torch.log(model + eps))).sum()
    return float(kl_bits / (math.log(2.0) * states.shape[1]))


class RecurrentTheoryRG(nn.Module):
    """A shared theory-space RG cell and a scalar action decoder."""

    def __init__(self, latent_dim: int = 8, width: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(), nn.Linear(width, latent_dim)
        )
        self.cell = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, 1)
        )

    def forward(self, coupling: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(coupling[:, None])
        decoded, latents = [], []
        for level in range(levels):
            decoded.append(self.decoder(z).squeeze(-1))
            latents.append(z)
            if level + 1 < levels:
                z = z + self.cell(z)
        return torch.stack(decoded, dim=1), torch.stack(latents, dim=1)


def make_targets(
    coupling_values: np.ndarray,
    state_levels: list[torch.Tensor],
    transitions: list[torch.Tensor],
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    projected = []
    all_probabilities: list[list[torch.Tensor]] = []
    for coupling in coupling_values:
        probabilities = [ising_distribution(state_levels[0], float(coupling))]
        for transition in transitions:
            probabilities.append(apply_transition(probabilities[-1], transition))
        projected.append(
            [
                nearest_neighbour_projection(probability, states)
                for probability, states in zip(probabilities, state_levels)
            ]
        )
        all_probabilities.append(probabilities)
    return torch.tensor(projected, dtype=DTYPE), all_probabilities


def exact_decimation_flow(coupling: np.ndarray, levels: int) -> np.ndarray:
    values = [np.asarray(coupling, dtype=float)]
    for _ in range(levels - 1):
        values.append(0.5 * np.log(np.cosh(2.0 * values[-1])))
    return np.stack(values, axis=1)


def train_theory_model(
    train_k: np.ndarray,
    train_targets: torch.Tensor,
    test_k: np.ndarray,
    test_targets: torch.Tensor,
    latent_dim: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[RecurrentTheoryRG, dict[str, float], dict[str, list[float]]]:
    torch.manual_seed(seed)
    model = RecurrentTheoryRG(latent_dim=latent_dim).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_input = torch.tensor(train_k, dtype=DTYPE)
    test_input = torch.tensor(test_k, dtype=DTYPE)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_mse": [],
        "test_mse": [],
    }
    record_every = max(1, steps // 250)
    for step in range(steps):
        optimizer.zero_grad()
        prediction, _ = model(train_input, train_targets.shape[1])
        loss = torch.mean((prediction - train_targets) ** 2)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % record_every == 0 or step + 1 == steps:
            with torch.no_grad():
                train_prediction, _ = model(train_input, train_targets.shape[1])
                test_prediction, _ = model(test_input, test_targets.shape[1])
                history["epoch"].append(step + 1)
                history["train_mse"].append(
                    float(torch.mean((train_prediction - train_targets) ** 2))
                )
                history["test_mse"].append(
                    float(torch.mean((test_prediction - test_targets) ** 2))
                )
    with torch.no_grad():
        train_prediction, _ = model(train_input, train_targets.shape[1])
        test_prediction, _ = model(test_input, test_targets.shape[1])
    metrics = {
        "train_coupling_rmse": float(torch.mean((train_prediction - train_targets) ** 2).sqrt()),
        "test_coupling_rmse": float(torch.mean((test_prediction - test_targets) ** 2).sqrt()),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return model, metrics, history


def make_figure(
    output: Path,
    history: dict[str, list[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(
        history["epoch"], history["train_mse"], color="#264653", lw=2,
        label="train",
    )
    ax.semilogy(
        history["epoch"], history["test_mse"], color="#E76F51", lw=2,
        label="test",
    )
    ax.set(
        xlabel="training epoch",
        ylabel=r"$\mathrm{MSE}(\hat J_k,J_k)$",
        title="Recurrent theory-state training",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-length", type=int, default=12)
    parser.add_argument("--buffer", type=int, default=1)
    parser.add_argument("--kernel-strength", type=float, default=12.0)
    parser.add_argument("--kernel-seeds", type=int, default=12)
    parser.add_argument("--kernel-steps", type=int, default=800)
    parser.add_argument("--theory-steps", type=int, default=5000)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/recurrent_rg_ising_1d")
    )
    args = parser.parse_args()
    if args.chain_length % 4:
        raise ValueError("chain-length must be divisible by four")

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel_couplings = np.linspace(0.25, 1.15, 7)
    joints = [
        enumerate_visible_environment(args.chain_length, float(k), args.buffer)
        for k in kernel_couplings
    ]
    kernel = learn_kernel(
        joints, args.kernel_strength, args.kernel_seeds, args.kernel_steps, 0.04
    )
    weights = torch.tensor(kernel.weights, dtype=DTYPE)

    lengths = [args.chain_length, args.chain_length // 2, args.chain_length // 4]
    state_levels = [enumerate_spins(length) for length in lengths]
    transitions = [
        block_transition(state_levels[0], state_levels[1], weights),
        block_transition(state_levels[1], state_levels[2], weights),
    ]

    all_k = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(len(all_k)) % 3 != 1
    train_k, test_k = all_k[train_mask], all_k[~train_mask]
    train_targets, train_probabilities = make_targets(train_k, state_levels, transitions)
    test_targets, test_probabilities = make_targets(test_k, state_levels, transitions)
    model, metrics, history = train_theory_model(
        train_k,
        train_targets,
        test_k,
        test_targets,
        args.latent_dim,
        args.theory_steps,
        1e-3,
        args.seed,
    )
    with torch.no_grad():
        test_predictions, test_latents = model(
            torch.tensor(test_k, dtype=DTYPE), len(state_levels)
        )

    closure_kl = []
    prediction_kl = []
    for sample, probabilities in enumerate(test_probabilities):
        closure_row, prediction_row = [], []
        for level, (probability, states) in enumerate(zip(probabilities, state_levels)):
            closure_row.append(
                kl_bits_per_site(probability, states, float(test_targets[sample, level]))
            )
            prediction_row.append(
                kl_bits_per_site(probability, states, float(test_predictions[sample, level]))
            )
        closure_kl.append(closure_row)
        prediction_kl.append(prediction_row)

    analytic_test = torch.tensor(
        exact_decimation_flow(test_k, len(state_levels)), dtype=DTYPE
    )
    summary = {
        "setup": {
            "model": "periodic ferromagnetic 1D Ising",
            "chain_lengths": lengths,
            "rg_levels": len(lengths),
            "kernel": "P(h|a,b) proportional to exp[h(lambda_1 a + lambda_2 b)]",
            "kernel_objective": "mean exact I(H:E) over seven couplings",
            "action_family": "A_k[s] = -J_k sum_i s_i s_(i+1)",
            "theory_latent_dim": args.latent_dim,
            "train_couplings": len(train_k),
            "test_couplings": len(test_k),
        },
        "learned_kernel": asdict(kernel),
        "kernel_decimation_dominance_ratio": float(
            np.max(np.abs(kernel.weights))
            / max(np.min(np.abs(kernel.weights)), 1e-12)
        ),
        "theory_model": metrics,
        "action_closure_kl_bits_per_site_mean_by_level": np.mean(closure_kl, axis=0).tolist(),
        "decoded_action_kl_bits_per_site_mean_by_level": np.mean(prediction_kl, axis=0).tolist(),
        "test_rmse_vs_analytic_decimation": float(
            torch.mean((test_predictions - analytic_test) ** 2).sqrt()
        ),
        "loss_definition": "mean squared error of decoded J_k over samples and all three RG levels",
        "training_history": history,
        "interpretation": {
            "configuration_latent": "coarse spin h, different for each configuration",
            "theory_latent": "z_k, shared across all configurations at fixed K and RG level",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {"state_dict": model.state_dict(), "kernel_weights": kernel.weights, "setup": summary["setup"]},
        args.output_dir / "model.pt",
    )
    make_figure(
        args.output_dir / "train_test_loss.png",
        history,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
