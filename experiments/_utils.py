"""Shared utilities for experiment scripts -- keep experiment-level helpers here."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QCRWeights:
    """Loss weights for the QCR (Q-C-R) joint training objective.

    The ``anchor`` default differs between 1D (1.0) and 2D (2.0); scripts should
    pass the appropriate value explicitly.
    """
    anchor: float = 1.0
    coarse_consistency: float = 5.0
    separation: float = 2.0
    commutator: float = 5.0


def fixed_eval_subset(
    configurations: torch.Tensor, systems: int, samples: int
) -> torch.Tensor:
    """Deterministic grid of systems × samples for loss-curve evaluation."""
    count = min(systems, configurations.shape[0])
    indices = torch.linspace(0, configurations.shape[0] - 1, count).long()
    return configurations[indices, : min(samples, configurations.shape[1])]


def balanced_views(configurations: torch.Tensor, samples: int) -> torch.Tensor:
    """Build two exactly spin-flip-balanced ensemble views of ``samples`` each.

    Returns ``2 * samples`` configurations where the first half are the two
    views (each of size ``samples``).  Every base configuration is paired with
    its global spin flip, preventing the encoder from exploiting spontaneous
    magnetisation to distinguish theories.
    """
    if samples % 2:
        raise ValueError("sample count must be even for exact spin-flip pairing")
    half = samples // 2
    if 2 * half > configurations.shape[1]:
        raise ValueError("not enough configurations for two balanced views")
    first = configurations[:, :half]
    second = configurations[:, half : 2 * half]
    return torch.cat((first, -first, second, -second), dim=1)