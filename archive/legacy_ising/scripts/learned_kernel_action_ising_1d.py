#!/usr/bin/env python3
"""Learned configuration RG, theory state, and neural action for 1D Ising.

The three deliberately separate objects are:
  C_theta: X_k -> X_(k+1), a shared trainable stride-2 convolution;
  Q_psi:   ensemble X_k -> scalar theory state g_k;
  E_phi:   (query X_k, g_k) -> action A_k[X], a local neural energy.

The RG convolution is trained first with exact real-space mutual information
and then frozen.  This prevents the RG and action networks from hiding errors
inside one another in the first smoke test.  The other networks are supervised
with exact 1D Ising couplings and actions, but neither is provided as an input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .ensemble_theory_rg_ising_1d import (
        DTYPE,
        bond_sum,
        exact_rg_flow,
        sample_ensembles,
    )
    from .rsmi_ising_1d import enumerate_joint, mutual_information_bits
except ImportError:  # Direct execution: python experiments/learned_kernel_action_ising_1d.py
    from ensemble_theory_rg_ising_1d import (
        DTYPE,
        bond_sum,
        exact_rg_flow,
        sample_ensembles,
    )
    from rsmi_ising_1d import enumerate_joint, mutual_information_bits


class LearnedBlockRG(nn.Module):
    """Z2-equivariant two-site convolutional RG kernel with fixed sharpness."""

    def __init__(self, strength: float = 12.0):
        super().__init__()
        self.angle = nn.Parameter(torch.rand((), dtype=DTYPE) * 2.0 * math.pi)
        self.strength = float(strength)

    def weights(self) -> torch.Tensor:
        return self.strength * torch.stack((torch.cos(self.angle), torch.sin(self.angle)))

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        """[S,B,N,1] -> [S,B,N/2,1] using the learned convolution."""
        systems, batches, sites, _ = configurations.shape
        signal = configurations.reshape(systems * batches, sites, 1).transpose(1, 2)
        signal = signal.to(dtype=self.angle.dtype)
        weight = self.weights().reshape(1, 1, 2)
        logits = F.conv1d(signal, weight, stride=2)
        coarse = torch.where(logits >= 0, 1.0, -1.0)
        return coarse.transpose(1, 2).reshape(systems, batches, sites // 2, 1)


def train_rg_kernel(
    kernel: LearnedBlockRG,
    couplings: np.ndarray,
    chain_length: int,
    buffer: int,
    epochs: int,
) -> dict[str, float | list[float]]:
    joints = [enumerate_joint(chain_length, float(k), buffer) for k in couplings]
    optimizer = torch.optim.Adam(kernel.parameters(), lr=0.04)
    for _ in range(epochs):
        optimizer.zero_grad()
        weights = kernel.weights()
        loss = -torch.stack(
            [mutual_information_bits(weights, joint) for joint in joints]
        ).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = kernel.weights()
        mean_mi = torch.stack(
            [mutual_information_bits(weights, joint) for joint in joints]
        ).mean()
    absolute = weights.abs()
    return {
        "weights": [float(value) for value in weights],
        "mean_mutual_information_bits": float(mean_mi),
        "dominance_ratio": float(absolute.max() / absolute.min().clamp_min(1e-12)),
        "selected_block_site": int(torch.argmax(absolute)) + 1,
    }


def pattern_table(window: int) -> torch.Tensor:
    indices = torch.arange(2**window, dtype=torch.long)
    bits = (indices[:, None] >> torch.arange(window)) & 1
    return (2 * bits - 1).to(DTYPE)


def pattern_counts(configurations: torch.Tensor, window: int) -> torch.Tensor:
    """Periodic local-pattern counts: [S,B,N,1] -> [S,B,2**window]."""
    spins = configurations.squeeze(-1)
    index = torch.zeros_like(spins, dtype=torch.long)
    for offset in range(window):
        bit = (torch.roll(spins, -offset, dims=2) > 0).long()
        index = index + (2**offset) * bit
    return torch.stack(
        [(index == value).to(DTYPE).sum(dim=2) for value in range(2**window)],
        dim=2,
    )


class TheoryEncoder(nn.Module):
    """DeepSets-style local-pattern MLP: [S,B,P] -> one shared g per system."""

    def __init__(self, window: int, hidden: int = 16, features: int = 8):
        super().__init__()
        self.register_buffer("patterns", pattern_table(window))
        self.local_mlp = nn.Sequential(
            nn.Linear(window, hidden), nn.Tanh(), nn.Linear(hidden, features), nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        local_features = self.local_mlp(self.patterns)       # [P,F]
        frequencies = counts.sum(dim=1) / counts.sum(dim=(1, 2), keepdim=False)[:, None]
        ensemble_features = frequencies @ local_features    # [S,F]
        return self.head(ensemble_features).squeeze(-1)      # [S]


class NeuralAction(nn.Module):
    """Conditioned local energy density summed over space: ([S,B,P],[S]) -> [S,B]."""

    def __init__(self, window: int, hidden: int = 16):
        super().__init__()
        self.register_buffer("patterns", pattern_table(window))
        self.local_energy = nn.Sequential(
            nn.Linear(window + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, counts: torch.Tensor, theory: torch.Tensor) -> torch.Tensor:
        systems, _, patterns = counts.shape
        local_pattern = self.patterns[None].expand(systems, patterns, -1)
        conditioning = theory[:, None, None].expand(systems, patterns, 1)
        density = self.local_energy(torch.cat((local_pattern, conditioning), dim=2)).squeeze(-1)
        action = torch.einsum("sbp,sp->sb", counts, density)
        # The additive action constant is unidentifiable; fix zero ensemble mean.
        return action - action.mean(dim=1, keepdim=True)


class LearnedRGActionModel(nn.Module):
    def __init__(self, window: int = 5, hidden: int = 16):
        super().__init__()
        self.theory_encoder = TheoryEncoder(window, hidden=hidden)
        self.rg_cell = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.action_network = NeuralAction(window, hidden=hidden)

    def advance(self, theory: torch.Tensor) -> torch.Tensor:
        return theory + self.rg_cell(theory[:, None]).squeeze(-1)

    def forward(
        self, count_levels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        direct = torch.stack(
            [self.theory_encoder(counts) for counts in count_levels], dim=1
        )
        rollout = [direct[:, 0]]
        for _ in range(1, len(count_levels)):
            rollout.append(self.advance(rollout[-1]))
        rollout_tensor = torch.stack(rollout, dim=1)
        actions = [
            self.action_network(counts, direct[:, level])
            for level, counts in enumerate(count_levels)
        ]
        return direct, rollout_tensor, actions


def make_levels(kernel: LearnedBlockRG, x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    result = [x]
    with torch.no_grad():
        for _ in range(levels - 1):
            result.append(kernel(result[-1]))
    return result


def centered_true_actions(
    configurations: list[torch.Tensor], target_theory: torch.Tensor
) -> list[torch.Tensor]:
    actions = []
    for level, x in enumerate(configurations):
        raw = -target_theory[:, level, None] * bond_sum(x.squeeze(-1))
        actions.append(raw - raw.mean(dim=1, keepdim=True))
    return actions


def loss_terms(
    model: LearnedRGActionModel,
    counts: list[torch.Tensor],
    target_theory: torch.Tensor,
    target_actions: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    direct, rollout, actions = model(counts)
    direct_mse = torch.mean((direct - target_theory) ** 2)
    rollout_mse = torch.mean((rollout - target_theory) ** 2)
    action_mse = torch.stack(
        [
            torch.mean(((prediction - target) / counts[level].sum(dim=2)) ** 2)
            for level, (prediction, target) in enumerate(zip(actions, target_actions))
        ]
    ).mean()
    total = direct_mse + rollout_mse + action_mse
    return total, {
        "direct_mse": direct_mse,
        "rollout_mse": rollout_mse,
        "action_mse_per_site": action_mse,
    }


def train_model(
    model: LearnedRGActionModel,
    train_counts: list[torch.Tensor],
    train_theory: torch.Tensor,
    train_actions: list[torch.Tensor],
    test_counts: list[torch.Tensor],
    test_theory: torch.Tensor,
    test_actions: list[torch.Tensor],
    epochs: int,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"epoch": [], "train_loss": [], "test_loss": []}
    record_every = max(1, epochs // 250)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = loss_terms(model, train_counts, train_theory, train_actions)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                train_loss, _ = loss_terms(model, train_counts, train_theory, train_actions)
                test_loss, _ = loss_terms(model, test_counts, test_theory, test_actions)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(train_loss))
            history["test_loss"].append(float(test_loss))
    return history


def plot_loss(path: Path, history: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(history["epoch"], history["train_loss"], lw=2, label="train")
    ax.semilogy(history["epoch"], history["test_loss"], lw=2, label="test")
    ax.set(xlabel="training epoch", ylabel="total loss",
           title="Learned RG kernel, theory state, and neural action")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=512)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--kernel-epochs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/learned_kernel_action_ising_1d")
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kernel = LearnedBlockRG().to(dtype=DTYPE)
    kernel_metrics = train_rg_kernel(
        kernel, np.linspace(0.25, 1.15, 7), 12, 1, args.kernel_epochs
    )
    kernel.requires_grad_(False)

    coupling_grid = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(len(coupling_grid)) % 3 != 1
    train_k, test_k = coupling_grid[train_mask], coupling_grid[~train_mask]
    train_x, train_k_repeated = sample_ensembles(
        train_k, args.replicas, args.ensemble_size, args.length, args.seed
    )
    test_x, test_k_repeated = sample_ensembles(
        test_k, args.replicas, args.ensemble_size, args.length, args.seed + 1
    )
    train_levels = make_levels(kernel, train_x, args.levels)
    test_levels = make_levels(kernel, test_x, args.levels)
    train_counts = [pattern_counts(x, args.window) for x in train_levels]
    test_counts = [pattern_counts(x, args.window) for x in test_levels]
    train_theory = exact_rg_flow(train_k_repeated, args.levels)
    test_theory = exact_rg_flow(test_k_repeated, args.levels)
    train_actions = centered_true_actions(train_levels, train_theory)
    test_actions = centered_true_actions(test_levels, test_theory)

    model = LearnedRGActionModel(window=args.window).to(dtype=DTYPE)
    history = train_model(
        model, train_counts, train_theory, train_actions,
        test_counts, test_theory, test_actions, args.epochs
    )
    with torch.no_grad():
        test_total, test_terms = loss_terms(
            model, test_counts, test_theory, test_actions
        )
        direct, rollout, _ = model(test_counts)

    summary = {
        "setup": {
            "configuration_shapes": [list(x.shape) for x in train_levels],
            "local_window": args.window,
            "model_inputs_coupling_or_action": False,
            "rg_kernel_training": "exact RSMI, then frozen",
            "action_training": "centered exact action labels for this smoke test",
        },
        "rg_kernel": kernel_metrics,
        "model": {
            "parameters_excluding_rg_kernel": int(sum(p.numel() for p in model.parameters())),
            "trainable_rg_kernel_parameters": 1,
            "theory_encoder": f"pattern MLP {args.window}->16->8, mean, MLP 8->16->1",
            "theory_rg": "shared residual MLP 1->16->1",
            "action_network": f"conditioned local MLP {args.window + 1}->16->1, spatial sum",
        },
        "metrics": {
            "test_total_loss": float(test_total),
            "test_direct_theory_rmse": float(torch.mean((direct - test_theory) ** 2).sqrt()),
            "test_rollout_theory_rmse": float(torch.mean((rollout - test_theory) ** 2).sqrt()),
            "test_action_rmse_per_site": float(test_terms["action_mse_per_site"].sqrt()),
        },
        "loss": "MSE(g_direct,J) + MSE(g_rollout,J) + centered action MSE/site",
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {"model": model.state_dict(), "rg_kernel": kernel.state_dict(), "setup": summary["setup"]},
        args.output_dir / "model.pt",
    )
    plot_loss(args.output_dir / "train_test_loss.png", history)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
