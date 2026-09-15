"""Dimension-independent configuration-only recurrent-RG objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adapters import LocalConditional
from .model_nd import RecurrentRGNetwork, RGTrajectory


@dataclass(frozen=True)
class UnsupervisedLossWeights:
    ensemble: float = 5.0
    gauge: float = 1.0
    rollout: float = 1.0
    commutator: float = 1.0
    action: float = 1.0


def ensemble_consistency(model: RecurrentRGNetwork, trajectory: RGTrajectory) -> torch.Tensor:
    samples = trajectory.configurations[0].shape[1]
    midpoint = samples // 2
    if midpoint < 1:
        raise ValueError("ensemble consistency needs at least two samples")
    losses = []
    for level in trajectory.configurations:
        first = model.theory_encoder(level[:, :midpoint])
        second = model.theory_encoder(level[:, midpoint:])
        losses.append(torch.mean((first - second) ** 2))
    return torch.stack(losses).mean()


def theory_gauge_loss(theory: torch.Tensor) -> torch.Tensor:
    """Center and whiten an arbitrary-dimensional theory coordinate."""
    centered = theory - theory.mean(dim=0, keepdim=True)
    mean_loss = theory.mean(dim=0).square().mean()
    covariance = centered.T @ centered / max(theory.shape[0], 1)
    identity = torch.eye(theory.shape[1], dtype=theory.dtype, device=theory.device)
    return mean_loss + (covariance - identity).square().mean()


def unsupervised_objective(
    model: RecurrentRGNetwork,
    microscopic: torch.Tensor,
    local_conditional: LocalConditional,
    *,
    buffer: int = 1,
    weights: UnsupervisedLossWeights = UnsupervisedLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trajectory = model(microscopic)
    ensemble = ensemble_consistency(model, trajectory)
    gauge = theory_gauge_loss(trajectory.direct_theory[:, 0])
    one_step = torch.stack(
        [
            model.advance_theory(trajectory.direct_theory[:, level])
            for level in range(model.levels - 1)
        ],
        dim=1,
    )
    commutator = torch.mean((one_step - trajectory.direct_theory[:, 1:]) ** 2)
    rollout = torch.mean(
        (trajectory.rollout_theory[:, 1:] - trajectory.direct_theory[:, 1:]) ** 2
    )
    action = torch.stack(
        [
            local_conditional(
                model.action_network, level, trajectory.direct_theory[:, level_index]
            )
            for level_index, level in enumerate(trajectory.configurations)
        ]
    ).mean()
    total = (
        weights.ensemble * ensemble
        + weights.gauge * gauge
        + weights.rollout * rollout
        + weights.commutator * commutator
        + weights.action * action
    )
    return total, {
        "ensemble_mse": ensemble,
        "gauge_loss": gauge,
        "rollout_mse": rollout,
        "commutator_mse": commutator,
        "action_local_conditional": action,
    }
