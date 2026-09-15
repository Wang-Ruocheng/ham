"""Metrics shared by physical-system adapters and experiments."""

from __future__ import annotations

import numpy as np
import torch


def centered_action_relative_rmse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Relative RMSE after removing the unobservable per-system action offset."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must both have shape [systems,samples]")
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction - target).square().sum()
    denominator = target.square().sum().clamp_min(torch.finfo(target.dtype).tiny)
    return torch.sqrt(numerator / denominator)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation for arrays without ties."""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def mean_pairwise_square(values: torch.Tensor) -> torch.Tensor:
    """Mean squared pairwise L2 distance — anti-collapse term."""
    if values.shape[0] < 2:
        raise ValueError("pairwise separation needs at least two systems")
    return torch.pdist(values, p=2).square().mean()
