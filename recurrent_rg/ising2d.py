"""Data and local-geometry utilities for the periodic two-dimensional Ising model."""

from __future__ import annotations

import math

import numpy as np
import torch

from .ising import DTYPE


def periodic_patches_2d(configurations: torch.Tensor, width: int) -> torch.Tensor:
    """Return centered periodic patches with shape [S,B,L,L,width**2]."""
    if configurations.ndim != 5 or configurations.shape[-1] != 1:
        raise ValueError("configurations must have shape [systems,samples,L,L,1]")
    if width < 1 or width % 2 == 0:
        raise ValueError("patch width must be a positive odd integer")
    spins = configurations.squeeze(-1)
    radius = width // 2
    patches = [
        torch.roll(spins, shifts=(-di, -dj), dims=(2, 3))
        for di in range(-radius, radius + 1)
        for dj in range(-radius, radius + 1)
    ]
    return torch.stack(patches, dim=-1)


def bond_sum_2d(spins: torch.Tensor) -> torch.Tensor:
    """Count every horizontal and vertical periodic nearest-neighbor bond once."""
    return (
        spins * torch.roll(spins, shifts=-1, dims=-2)
        + spins * torch.roll(spins, shifts=-1, dims=-1)
    ).sum(dim=(-2, -1))


def sample_wolff_ensembles(
    couplings: np.ndarray,
    replicas: int,
    samples: int,
    size: int,
    seed: int,
    thermalization_sweeps: int = 24,
    sweeps_between: int = 1,
    randomize_global_spin: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample zero-field ensembles using periodic Wolff cluster sweeps.

    At zero field, a global spin flip is an exact symmetry.  The cumulative
    cluster-sweep convention can otherwise return repeatedly to the same
    ordered-sector orientation when a cluster occupies almost the whole
    lattice.  Randomizing the orientation of each recorded draw removes this
    artificial sector locking without changing any even observable.
    """
    if size < 8 or size % 2:
        raise ValueError("size must be an even integer greater than or equal to 8")
    if replicas < 1 or samples < 1:
        raise ValueError("replicas and samples must be positive")
    rng = np.random.default_rng(seed)
    ensembles: list[torch.Tensor] = []
    labels: list[float] = []

    for coupling in couplings:
        add_probability = 1.0 - math.exp(-2.0 * float(coupling))
        for _ in range(replicas):
            spins = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(size, size))

            def cluster_update() -> int:
                start_i = int(rng.integers(size))
                start_j = int(rng.integers(size))
                target = spins[start_i, start_j]
                cluster = np.zeros((size, size), dtype=np.bool_)
                cluster[start_i, start_j] = True
                stack = [(start_i, start_j)]
                while stack:
                    i, j = stack.pop()
                    for ni, nj in (
                        ((i - 1) % size, j),
                        ((i + 1) % size, j),
                        (i, (j - 1) % size),
                        (i, (j + 1) % size),
                    ):
                        if (
                            not cluster[ni, nj]
                            and spins[ni, nj] == target
                            and rng.random() < add_probability
                        ):
                            cluster[ni, nj] = True
                            stack.append((ni, nj))
                spins[cluster] *= -1
                return int(cluster.sum())

            def cluster_sweep() -> None:
                changed = 0
                while changed < size * size:
                    changed += cluster_update()

            for _ in range(thermalization_sweeps):
                cluster_sweep()
            draws = []
            for _ in range(samples):
                for _ in range(sweeps_between):
                    cluster_sweep()
                draw = spins.copy()
                if randomize_global_spin and rng.random() < 0.5:
                    draw *= -1
                draws.append(torch.from_numpy(draw).to(DTYPE))
            ensembles.append(torch.stack(draws))
            labels.append(float(coupling))

    return (
        torch.stack(ensembles).unsqueeze(-1),
        torch.tensor(labels, dtype=DTYPE),
    )
