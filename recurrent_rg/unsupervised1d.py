"""1D binary adapters around the dimension-independent objective."""

from __future__ import annotations

import torch

from .adapters import BinaryFlipConditional
from .model_nd import RecurrentRGNetwork1D
from .unsupervised_nd import (
    UnsupervisedLossWeights,
    unsupervised_objective,
)


def pseudolikelihood_1d(
    model: RecurrentRGNetwork1D,
    configurations: torch.Tensor,
    theory: torch.Tensor,
) -> torch.Tensor:
    if theory.ndim == 1:
        theory = theory[:, None]
    return BinaryFlipConditional()(model.action_network, configurations, theory)


def unsupervised_objective_1d(
    model: RecurrentRGNetwork1D,
    microscopic: torch.Tensor,
    *,
    buffer: int = 1,
    weights: UnsupervisedLossWeights = UnsupervisedLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return unsupervised_objective(
        model,
        microscopic,
        BinaryFlipConditional(),
        buffer=buffer,
        weights=weights,
    )
