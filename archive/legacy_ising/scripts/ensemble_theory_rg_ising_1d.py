#!/usr/bin/env python3
"""Infer and evolve a 1D Ising effective action from configuration ensembles.

Input tensors have shape [systems, configurations, sites, 1].  A shared local
MLP followed by spatial and ensemble averaging returns one scalar theory state
g per system.  For the nearest-neighbour 1D Ising model, g is the dimensionless
effective coupling J in A[s] = -J sum_i s_i s_(i+1).  A second shared scalar MLP
advances g by one RG step.  The configuration RG is fixed to decimation in this
minimal experiment, as selected by the preceding exact RSMI experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


DTYPE = torch.float64


def bond_sum(states: torch.Tensor) -> torch.Tensor:
    return (states * torch.roll(states, -1, dims=-1)).sum(dim=-1)


def exact_rg_flow(coupling: torch.Tensor, levels: int) -> torch.Tensor:
    values = [coupling]
    for _ in range(levels - 1):
        values.append(0.5 * torch.log(torch.cosh(2.0 * values[-1])))
    return torch.stack(values, dim=-1)


def sample_ensembles(
    couplings: np.ndarray,
    replicas: int,
    ensemble_size: int,
    length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exact periodic-chain samples using bond variables with product +1."""
    generator = torch.Generator().manual_seed(seed)
    ensembles, labels = [], []
    for coupling in couplings:
        same_probability = float(torch.sigmoid(torch.tensor(2.0 * coupling)))
        for _ in range(replicas):
            accepted = []
            accepted_count = 0
            while accepted_count < ensemble_size:
                proposal_count = 2 * (ensemble_size - accepted_count) + 16
                bonds = torch.where(
                    torch.rand(proposal_count, length, generator=generator) < same_probability,
                    torch.ones((), dtype=DTYPE),
                    -torch.ones((), dtype=DTYPE),
                )
                bonds = bonds[bonds.prod(dim=1) > 0]
                accepted.append(bonds)
                accepted_count += bonds.shape[0]
            bonds = torch.cat(accepted, dim=0)[:ensemble_size]
            first_spin = torch.where(
                torch.rand(ensemble_size, 1, generator=generator) < 0.5,
                torch.ones((), dtype=DTYPE),
                -torch.ones((), dtype=DTYPE),
            )
            spins = torch.cat(
                (first_spin, first_spin * torch.cumprod(bonds[:, :-1], dim=1)), dim=1
            )
            ensembles.append(spins)
            labels.append(float(coupling))
    return torch.stack(ensembles).unsqueeze(-1), torch.tensor(labels, dtype=DTYPE)


def decimate(configurations: torch.Tensor) -> torch.Tensor:
    """Keep the second spin in every adjacent two-site block."""
    if configurations.shape[-2] % 2:
        raise ValueError("decimation requires an even number of sites")
    return configurations[..., 1::2, :]


def make_config_levels(configurations: torch.Tensor, levels: int) -> list[torch.Tensor]:
    result = [configurations]
    for _ in range(levels - 1):
        result.append(decimate(result[-1]))
    return result


def pair_frequencies(configurations: torch.Tensor) -> torch.Tensor:
    """Losslessly summarize all two-site inputs seen by the local MLP: [S,B,N,1] -> [S,4]."""
    spins = configurations.squeeze(-1)
    neighbour = torch.roll(spins, -1, dims=2)
    pair_index = (spins > 0).long() + 2 * (neighbour > 0).long()
    return torch.stack(
        [(pair_index == index).to(DTYPE).mean(dim=(1, 2)) for index in range(4)],
        dim=1,
    )


class EnsembleTheoryRG(nn.Module):
    """Local DeepSets encoder plus a shared one-dimensional RG flow MLP."""

    def __init__(self, hidden: int = 16, features: int = 8):
        super().__init__()
        self.register_buffer(
            "pair_states",
            torch.tensor(
                [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
                dtype=DTYPE,
            ),
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, features),
            nn.Tanh(),
        )
        self.theory_head = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.rg_cell = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def infer_theory(self, configurations: torch.Tensor) -> torch.Tensor:
        """[S,B,N,1] -> [S], one theory state shared by each ensemble."""
        return self.infer_from_pair_frequency(pair_frequencies(configurations))

    def infer_from_pair_frequency(self, pair_frequency: torch.Tensor) -> torch.Tensor:
        """[S,4] -> [S]; algebraically equal to averaging the local MLP over X."""
        local_features = self.local_encoder(self.pair_states)  # [4,F]
        ensemble_features = pair_frequency @ local_features    # [S,F]
        return self.theory_head(ensemble_features).squeeze(-1)

    def advance_theory(self, theory: torch.Tensor) -> torch.Tensor:
        """[S] -> [S], with the same residual cell at every RG level."""
        return theory + self.rg_cell(theory[:, None]).squeeze(-1)

    def forward(
        self, pair_frequency_levels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direct = torch.stack(
            [self.infer_from_pair_frequency(frequency) for frequency in pair_frequency_levels],
            dim=1,
        )
        rollout = [direct[:, 0]]
        for _ in range(1, len(pair_frequency_levels)):
            rollout.append(self.advance_theory(rollout[-1]))
        return direct, torch.stack(rollout, dim=1)

    @staticmethod
    def action(configurations: torch.Tensor, theory: torch.Tensor) -> torch.Tensor:
        """Return A[s]=-g sum_i s_i s_(i+1), with shape [S,B]."""
        spins = configurations.squeeze(-1)
        return -theory[:, None] * bond_sum(spins)


def objective(
    model: EnsembleTheoryRG,
    pair_frequency_levels: list[torch.Tensor],
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direct, rollout = model(pair_frequency_levels)
    direct_mse = torch.mean((direct - target) ** 2)
    rollout_mse = torch.mean((rollout - target) ** 2)
    return direct_mse + rollout_mse, direct_mse, rollout_mse


def train(
    model: EnsembleTheoryRG,
    train_pair_frequencies: list[torch.Tensor],
    train_target: torch.Tensor,
    test_pair_frequencies: list[torch.Tensor],
    test_target: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"epoch": [], "train_loss": [], "test_loss": []}
    record_every = max(1, epochs // 250)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = objective(model, train_pair_frequencies, train_target)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                train_loss, _, _ = objective(model, train_pair_frequencies, train_target)
                test_loss, _, _ = objective(model, test_pair_frequencies, test_target)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(train_loss))
            history["test_loss"].append(float(test_loss))
    return history


def action_rmse_per_site(
    model: EnsembleTheoryRG,
    levels: list[torch.Tensor],
    inferred: torch.Tensor,
    target: torch.Tensor,
) -> list[float]:
    result = []
    for level, configurations in enumerate(levels):
        predicted_action = model.action(configurations, inferred[:, level])
        target_action = model.action(configurations, target[:, level])
        sites = configurations.shape[-2]
        result.append(float(torch.mean(((predicted_action - target_action) / sites) ** 2).sqrt()))
    return result


def plot_loss(output: Path, history: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(history["epoch"], history["train_loss"], lw=2, label="train")
    ax.semilogy(history["epoch"], history["test_loss"], lw=2, label="test")
    ax.set(
        xlabel="training epoch",
        ylabel="total theory loss",
        title="Infer and evolve the effective coupling from Ising ensembles",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=512)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/ensemble_theory_rg_ising_1d")
    )
    args = parser.parse_args()
    if args.length % (2 ** (args.levels - 1)):
        raise ValueError("length must be divisible by 2**(levels-1)")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coupling_grid = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(len(coupling_grid)) % 3 != 1
    train_k, test_k = coupling_grid[train_mask], coupling_grid[~train_mask]
    train_x, train_k_repeated = sample_ensembles(
        train_k, args.replicas, args.ensemble_size, args.length, args.seed
    )
    test_x, test_k_repeated = sample_ensembles(
        test_k, args.replicas, args.ensemble_size, args.length, args.seed + 1
    )
    train_levels = make_config_levels(train_x, args.levels)
    test_levels = make_config_levels(test_x, args.levels)
    train_pair_frequencies = [pair_frequencies(x) for x in train_levels]
    test_pair_frequencies = [pair_frequencies(x) for x in test_levels]
    train_target = exact_rg_flow(train_k_repeated, args.levels)
    test_target = exact_rg_flow(test_k_repeated, args.levels)

    model = EnsembleTheoryRG().to(dtype=DTYPE)
    history = train(
        model,
        train_pair_frequencies,
        train_target,
        test_pair_frequencies,
        test_target,
        args.epochs,
        1e-3,
    )
    with torch.no_grad():
        direct, rollout = model(test_pair_frequencies)
        test_loss, direct_mse, rollout_mse = objective(
            model, test_pair_frequencies, test_target
        )

    summary = {
        "setup": {
            "input_shape_train": list(train_x.shape),
            "configuration_level_shapes_train": [list(x.shape) for x in train_levels],
            "pair_frequency_shapes_train": [list(x.shape) for x in train_pair_frequencies],
            "number_of_physical_systems_train": len(train_k),
            "ensemble_replicas_per_system": args.replicas,
            "configurations_per_ensemble": args.ensemble_size,
            "model_input_contains_coupling": False,
            "configuration_rg": "fixed decimation selected by the preceding RSMI test",
            "theory_state_dimension": 1,
        },
        "model": {
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "local_encoder": "MLP 2->16->8, then mean over sites and configurations",
            "theory_head": "MLP 8->16->1",
            "shared_rg_cell": "residual MLP 1->16->1",
        },
        "metrics": {
            "test_total_loss": float(test_loss),
            "test_direct_theory_rmse": float(torch.sqrt(direct_mse)),
            "test_rollout_theory_rmse": float(torch.sqrt(rollout_mse)),
            "test_direct_action_rmse_per_site_by_level": action_rmse_per_site(
                model, test_levels, direct, test_target
            ),
            "test_rollout_action_rmse_per_site_by_level": action_rmse_per_site(
                model, test_levels, rollout, test_target
            ),
        },
        "loss_definition": "MSE(direct g_k, J_k) + MSE(recurrent rollout g_k, J_k)",
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "summary": summary["setup"]},
               args.output_dir / "model.pt")
    plot_loss(args.output_dir / "train_test_loss.png", history)
    compact = {key: value for key, value in summary.items() if key != "history"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
