#!/usr/bin/env python3
"""End-to-end learned RG kernel, theory state, and action for 1D Ising.

Unlike learned_kernel_action_ising_1d.py, the stride-2 RG convolution receives
no RSMI pretraining and is never frozen.  It is optimized jointly through the
theory-state and supervised-action losses.  This isolates whether end-to-end
optimization itself can identify a useful coarse-graining map.
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
    from .ensemble_theory_rg_ising_1d import DTYPE, bond_sum, exact_rg_flow, sample_ensembles
except ImportError:
    from ensemble_theory_rg_ising_1d import DTYPE, bond_sum, exact_rg_flow, sample_ensembles


def periodic_windows(configurations: torch.Tensor, window: int) -> torch.Tensor:
    spins = configurations.squeeze(-1)
    return torch.stack(
        [torch.roll(spins, -offset, dims=2) for offset in range(window)], dim=-1
    )


class TrainableRGConvolution(nn.Module):
    """Shared, bias-free, Z2-equivariant stride-2 convolution."""

    def __init__(self, strength: float = 2.0):
        super().__init__()
        self.angle = nn.Parameter(torch.rand((), dtype=DTYPE) * 2.0 * math.pi)
        self.strength = float(strength)

    def weights(self) -> torch.Tensor:
        return self.strength * torch.stack((torch.cos(self.angle), torch.sin(self.angle)))

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        systems, samples, sites, _ = configurations.shape
        signal = configurations.reshape(systems * samples, sites, 1).transpose(1, 2)
        logits = F.conv1d(signal, self.weights().reshape(1, 1, 2), stride=2)
        soft = torch.tanh(logits)
        hard = torch.where(logits >= 0, 1.0, -1.0)
        # Hard Ising spins in the forward pass; tanh derivative in backpropagation.
        coarse = hard.detach() + soft - soft.detach()
        return coarse.transpose(1, 2).reshape(systems, samples, sites // 2, 1)


class TheoryEncoder(nn.Module):
    def __init__(self, window: int = 5, hidden: int = 16, features: int = 8):
        super().__init__()
        self.window = window
        self.local_mlp = nn.Sequential(
            nn.Linear(window, hidden), nn.Tanh(), nn.Linear(hidden, features), nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        local = self.local_mlp(periodic_windows(configurations, self.window))
        return self.head(local.mean(dim=(1, 2))).squeeze(-1)


class NeuralAction(nn.Module):
    def __init__(self, window: int = 5, hidden: int = 16):
        super().__init__()
        self.window = window
        self.local_energy = nn.Sequential(
            nn.Linear(window + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, configurations: torch.Tensor, theory: torch.Tensor) -> torch.Tensor:
        windows = periodic_windows(configurations, self.window)
        systems, samples, sites, _ = windows.shape
        conditioning = theory[:, None, None, None].expand(systems, samples, sites, 1)
        density = self.local_energy(torch.cat((windows, conditioning), dim=-1)).squeeze(-1)
        # Keep the raw action.  Supervised regression fixes its arbitrary additive
        # constant by centering inside the loss, while pseudolikelihood cancels
        # that constant in A(X)-A(X_flipped) automatically.
        return density.sum(dim=2)


class EndToEndRGAction(nn.Module):
    def __init__(self, levels: int = 3, window: int = 5, hidden: int = 16):
        super().__init__()
        self.levels = levels
        self.rg_kernel = TrainableRGConvolution()
        self.theory_encoder = TheoryEncoder(window, hidden)
        self.theory_rg = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.action_network = NeuralAction(window, hidden)

    def configuration_levels(self, microscopic: torch.Tensor) -> list[torch.Tensor]:
        result = [microscopic]
        for _ in range(1, self.levels):
            result.append(self.rg_kernel(result[-1]))
        return result

    def advance_theory(self, theory: torch.Tensor) -> torch.Tensor:
        return theory + self.theory_rg(theory[:, None]).squeeze(-1)

    def forward(
        self, microscopic: torch.Tensor, compute_actions: bool = True
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        levels = self.configuration_levels(microscopic)
        direct = torch.stack([self.theory_encoder(x) for x in levels], dim=1)
        rollout = [direct[:, 0]]
        for _ in range(1, self.levels):
            rollout.append(self.advance_theory(rollout[-1]))
        rollout_tensor = torch.stack(rollout, dim=1)
        actions = (
            [self.action_network(x, direct[:, level]) for level, x in enumerate(levels)]
            if compute_actions else []
        )
        return levels, direct, rollout_tensor, actions


def supervised_targets(
    levels: list[torch.Tensor], microscopic_coupling: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    theory = exact_rg_flow(microscopic_coupling, len(levels))
    actions = []
    for level, configurations in enumerate(levels):
        # Labels are constants for backpropagation, even though the current hard
        # RG configurations are regenerated whenever the kernel changes.
        spins = configurations.detach().squeeze(-1)
        raw = -theory[:, level, None] * bond_sum(spins)
        actions.append(raw - raw.mean(dim=1, keepdim=True))
    return theory, actions


def pseudolikelihood_loss(
    model: EndToEndRGAction,
    configurations: torch.Tensor,
    theory: torch.Tensor,
    flips: int,
) -> torch.Tensor:
    """Negative log conditional probability from single-spin flips.

    Translation equivariance makes all sites statistically equivalent, so a
    small, evenly spaced subset gives an inexpensive unbiased-in-symmetry
    estimate without constructing all N flipped copies.
    """
    systems, samples, sites, _ = configurations.shape
    count = min(flips, sites)
    indices = torch.div(
        torch.arange(count, device=configurations.device) * sites,
        count,
        rounding_mode="floor",
    )
    window = model.action_network.window
    offsets = torch.arange(window, device=configurations.device)
    starts = (indices[:, None] - offsets[None, :]) % sites

    # Flipping site i affects only windows starting at i, i-1, ..., i-w+1.
    # Recompute those w local terms rather than the full N-site action for every
    # candidate flip: O(count * w) instead of O(count * N).
    all_windows = periodic_windows(configurations, window)
    original_windows = all_windows[:, :, starts, :]
    flipped_windows = original_windows.clone()
    for offset in range(window):
        flipped_windows[:, :, :, offset, offset] *= -1.0

    conditioning = theory[:, None, None, None, None].expand(
        systems, samples, count, window, 1
    )
    original_density = model.action_network.local_energy(
        torch.cat((original_windows, conditioning), dim=-1)
    ).squeeze(-1)
    flipped_density = model.action_network.local_energy(
        torch.cat((flipped_windows, conditioning), dim=-1)
    ).squeeze(-1)
    original_minus_flipped = (original_density - flipped_density).sum(dim=-1)
    # -log p(s_i | all other spins) = softplus(A(X)-A(X with s_i flipped)).
    return F.softplus(original_minus_flipped).mean()


def objective(
    model: EndToEndRGAction,
    microscopic: torch.Tensor,
    coupling: torch.Tensor,
    action_loss: str = "supervised",
    pl_flips: int = 4,
    compute_action_metric: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    need_full_actions = action_loss == "supervised" or compute_action_metric
    levels, direct, rollout, predicted_actions = model(
        microscopic, compute_actions=need_full_actions
    )
    target_theory = exact_rg_flow(coupling, len(levels))
    direct_mse = torch.mean((direct - target_theory) ** 2)
    # k=0 is already supervised by direct_mse.  The rollout term measures only
    # whether repeated applications of the learned theory RG map predict future
    # scales from Q(X_0).
    rollout_mse = torch.mean((rollout[:, 1:] - target_theory[:, 1:]) ** 2)

    # Commuting-diagram consistency:
    #       X_k --C--> X_{k+1}
    #        |            |
    #        Q            Q
    #        |            |
    #       g_k --F--> g_{k+1}
    # This directly aligns configuration-space coarse graining with the learned
    # theory-space RG step, instead of relying on their common labels alone.
    one_step = torch.stack(
        [model.advance_theory(direct[:, level]) for level in range(model.levels - 1)],
        dim=1,
    )
    commutator_mse = torch.mean((one_step - direct[:, 1:]) ** 2)
    action_mse = None
    if need_full_actions:
        _, target_actions = supervised_targets(levels, coupling)
        action_mse = torch.stack(
            [
                torch.mean((
                    (prediction - prediction.mean(dim=1, keepdim=True) - target)
                    / levels[level].shape[2]
                ) ** 2)
                for level, (prediction, target) in enumerate(
                    zip(predicted_actions, target_actions)
                )
            ]
        ).mean()
    action_pl = torch.stack([
        pseudolikelihood_loss(model, x, direct[:, level], pl_flips)
        for level, x in enumerate(levels)
    ]).mean()
    optimized_action_loss = action_mse if action_loss == "supervised" else action_pl
    assert optimized_action_loss is not None
    total = direct_mse + rollout_mse + commutator_mse + optimized_action_loss
    terms = {
        "direct_mse": direct_mse,
        "rollout_mse": rollout_mse,
        "commutator_mse": commutator_mse,
        "action_pseudolikelihood": action_pl,
    }
    if action_mse is not None:
        terms["action_mse_per_site"] = action_mse
    return total, terms


def evaluate(
    model: EndToEndRGAction,
    x: torch.Tensor,
    coupling: torch.Tensor,
    action_loss: str = "supervised",
    pl_flips: int = 4,
    sample_chunk: int = 128,
) -> tuple[float, dict[str, float]]:
    # Evaluate multiple configuration subsets because g is an ensemble statistic.
    totals = []
    pieces: dict[str, list[float]] = {
        "direct_mse": [], "rollout_mse": [], "commutator_mse": [],
        "action_pseudolikelihood": [],
        "action_mse_per_site": []
    }
    with torch.no_grad():
        for start in range(0, x.shape[1], sample_chunk):
            total, terms = objective(
                model, x[:, start:start + sample_chunk], coupling,
                action_loss=action_loss, pl_flips=pl_flips,
                compute_action_metric=True,
            )
            totals.append(float(total))
            for name, value in terms.items():
                pieces[name].append(float(value))
    return float(np.mean(totals)), {
        name: float(np.mean(values)) for name, values in pieces.items()
    }


def train(
    model: EndToEndRGAction,
    train_x: torch.Tensor,
    train_k: torch.Tensor,
    test_x: torch.Tensor,
    test_k: torch.Tensor,
    epochs: int,
    system_batch: int,
    configuration_batch: int,
    seed: int,
    action_loss: str,
    pl_flips: int,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    history = {
        "epoch": [], "train_loss": [], "test_loss": [],
        "kernel_weight_1": [], "kernel_weight_2": [],
    }
    # Evaluation is substantially more expensive with pseudolikelihood because
    # it includes counterfactual spin flips.  Sixty points are ample for a
    # convergence curve and keep validation from dominating optimization time.
    record_every = max(1, epochs // 60)
    for epoch in range(epochs):
        system_index = torch.randperm(train_x.shape[0], generator=generator)[:system_batch]
        configuration_index = torch.randperm(
            train_x.shape[1], generator=generator
        )[:configuration_batch]
        batch_x = train_x[system_index][:, configuration_index]
        batch_k = train_k[system_index]
        optimizer.zero_grad()
        loss, _ = objective(
            model, batch_x, batch_k, action_loss=action_loss, pl_flips=pl_flips
        )
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            # Use fixed subsets so the curves measure optimization, not resampling noise.
            train_loss, _ = evaluate(
                model, train_x[:32, :128], train_k[:32],
                action_loss=action_loss, pl_flips=pl_flips, sample_chunk=128,
            )
            test_loss, _ = evaluate(
                model, test_x[:, :128], test_k,
                action_loss=action_loss, pl_flips=pl_flips, sample_chunk=128,
            )
            weights = model.rg_kernel.weights().detach()
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["kernel_weight_1"].append(float(weights[0]))
            history["kernel_weight_2"].append(float(weights[1]))
    return history


def plot_loss(path: Path, history: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(history["epoch"], history["train_loss"], lw=2, label="train")
    ax.semilogy(history["epoch"], history["test_loss"], lw=2, label="test")
    ax.set(
        xlabel="training epoch", ylabel="total loss",
        title="End-to-end RG kernel, theory state, and action",
    )
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
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--system-batch", type=int, default=16)
    parser.add_argument("--configuration-batch", type=int, default=64)
    parser.add_argument(
        "--action-loss", choices=("supervised", "pseudolikelihood"),
        default="supervised",
    )
    parser.add_argument("--pl-flips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/end_to_end_rg_action_ising_1d")
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coupling_grid = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(len(coupling_grid)) % 3 != 1
    train_values, test_values = coupling_grid[train_mask], coupling_grid[~train_mask]
    train_x, train_k = sample_ensembles(
        train_values, args.replicas, args.ensemble_size, args.length, args.seed
    )
    test_x, test_k = sample_ensembles(
        test_values, args.replicas, args.ensemble_size, args.length, args.seed + 1
    )
    train_x, test_x = train_x.to(DTYPE), test_x.to(DTYPE)

    model = EndToEndRGAction(args.levels, args.window).to(dtype=DTYPE)
    initial_weights = [float(value) for value in model.rg_kernel.weights().detach()]
    history = train(
        model, train_x, train_k, test_x, test_k, args.epochs,
        args.system_batch, args.configuration_batch, args.seed + 2,
        args.action_loss, args.pl_flips,
    )
    test_loss, test_terms = evaluate(
        model, test_x, test_k, action_loss=args.action_loss, pl_flips=args.pl_flips
    )
    final_weights_tensor = model.rg_kernel.weights().detach()
    absolute = final_weights_tensor.abs()
    final_weights = [float(value) for value in final_weights_tensor]

    with torch.no_grad():
        levels, direct, rollout, _ = model(test_x[:, :128])
        target = exact_rg_flow(test_k, args.levels)
    summary = {
        "setup": {
            "input_shape_train": list(train_x.shape),
            "configuration_shapes_test_subset": [list(level.shape) for level in levels],
            "model_inputs_coupling_or_action": False,
            "rg_kernel_pretraining": False,
            "rg_kernel_frozen": False,
            "action_labels_available_for_evaluation": True,
            "action_loss": args.action_loss,
            "action_labels_used_for_training": args.action_loss == "supervised",
            "pseudolikelihood_flips_per_configuration": args.pl_flips,
            "local_window": args.window,
        },
        "model": {
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "rg_kernel": "bias-free Conv1d 1->1, kernel=2, stride=2, shared across levels",
            "theory_encoder": f"local MLP {args.window}->16->8, ensemble mean, MLP 8->16->1",
            "theory_rg": "shared residual MLP 1->16->1",
            "action_network": f"conditioned local MLP {args.window + 1}->16->1, spatial sum",
        },
        "rg_kernel": {
            "initial_weights": initial_weights,
            "final_weights": final_weights,
            "dominance_ratio": float(absolute.max() / absolute.min().clamp_min(1e-12)),
            "selected_block_site": int(torch.argmax(absolute)) + 1,
        },
        "metrics": {
            "test_total_loss": test_loss,
            "test_direct_theory_rmse": float(torch.mean((direct - target) ** 2).sqrt()),
            "test_rollout_theory_rmse": float(torch.mean((rollout - target) ** 2).sqrt()),
            "test_action_rmse_per_site": math.sqrt(test_terms["action_mse_per_site"]),
            "test_action_pseudolikelihood": test_terms["action_pseudolikelihood"],
        },
        "loss": {
            "theory": (
                "MSE(g_direct,J) + MSE(g_rollout[:,1:],J[:,1:]) + "
                "MSE(F(g_direct[:,:-1]),g_direct[:,1:])"
            ),
            "action_optimized": (
                "centered supervised action MSE/site"
                if args.action_loss == "supervised"
                else "single-spin conditional negative log pseudolikelihood"
            ),
            "supervised_action_used_only_as_test_metric_in_pl_mode": True,
        },
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "setup": summary["setup"]},
               args.output_dir / "model.pt")
    plot_loss(args.output_dir / "train_test_loss.png", history)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
