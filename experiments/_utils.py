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