"""Exact data utilities for the periodic one-dimensional Ising benchmark."""

from __future__ import annotations

import numpy as np
import torch


DTYPE = torch.float64


def periodic_windows(configurations: torch.Tensor, width: int) -> torch.Tensor:
    """Return every periodic width-site window.

    Args:
        configurations: Spin ensembles with shape ``[S, B, N, 1]``.
        width: Number of consecutive sites in each window.

    Returns:
        A tensor with shape ``[S, B, N, width]``.
    """
    if configurations.ndim != 4 or configurations.shape[-1] != 1:
        raise ValueError("configurations must have shape [systems, samples, sites, 1]")
    spins = configurations.squeeze(-1)
    return torch.stack(
        [torch.roll(spins, -offset, dims=2) for offset in range(width)], dim=-1
    )


def bond_sum(spins: torch.Tensor) -> torch.Tensor:
    """Compute ``sum_i s_i s_(i+1)`` along the final, periodic site axis."""
    return (spins * torch.roll(spins, -1, dims=-1)).sum(dim=-1)


def exact_decimation_flow(coupling: torch.Tensor, levels: int) -> torch.Tensor:
    """Exact zero-field 1D Ising flow ``J' = 0.5 log(cosh(2J))``."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    values = [coupling]
    for _ in range(levels - 1):
        values.append(0.5 * torch.log(torch.cosh(2.0 * values[-1])))
    return torch.stack(values, dim=-1)


def sample_periodic_ensembles(
    couplings: np.ndarray,
    replicas: int,
    samples: int,
    sites: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exact periodic Ising ensembles via constrained bond variables.

    The returned shape is ``[len(couplings) * replicas, samples, sites, 1]``.
    Every first-axis entry is one physical system: an ensemble sharing one J.
    """
    if sites < 4 or sites % 2:
        raise ValueError("sites must be an even integer greater than or equal to 4")
    if replicas < 1 or samples < 1:
        raise ValueError("replicas and samples must be positive")

    generator = torch.Generator().manual_seed(seed)
    ensembles: list[torch.Tensor] = []
    labels: list[float] = []
    for coupling in couplings:
        same_probability = float(torch.sigmoid(torch.tensor(2.0 * coupling)))
        for _ in range(replicas):
            accepted: list[torch.Tensor] = []
            accepted_count = 0
            while accepted_count < samples:
                proposal_count = 2 * (samples - accepted_count) + 16
                bonds = torch.where(
                    torch.rand(proposal_count, sites, generator=generator)
                    < same_probability,
                    torch.ones((), dtype=DTYPE),
                    -torch.ones((), dtype=DTYPE),
                )
                bonds = bonds[bonds.prod(dim=1) > 0]
                accepted.append(bonds)
                accepted_count += bonds.shape[0]

            bonds = torch.cat(accepted)[:samples]
            first_spin = torch.where(
                torch.rand(samples, 1, generator=generator) < 0.5,
                torch.ones((), dtype=DTYPE),
                -torch.ones((), dtype=DTYPE),
            )
            spins = torch.cat(
                (first_spin, first_spin * torch.cumprod(bonds[:, :-1], dim=1)), dim=1
            )
            ensembles.append(spins)
            labels.append(float(coupling))

    return (
        torch.stack(ensembles).unsqueeze(-1),
        torch.tensor(labels, dtype=DTYPE),
    )

