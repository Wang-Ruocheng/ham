"""Data utilities for the one-dimensional transverse-field Ising benchmark.

Model: ``H = -J sum_i sigma^z_i sigma^z_{i+1} - h sum_i sigma^x_i``, quantum
spins-1/2 on a periodic chain of ``sites`` sites. Physical configurations are
the sigma^z eigenvalues (+1/-1) sampled from a finite-temperature
Suzuki-Trotter approximation, mapped onto a 2D classical Ising model with
space x imaginary-time geometry. It approximates ground-state sampling only
when ``beta = time_slices * beta_over_slices`` is sufficiently large and the
Trotter discretization is converged.

The sampled data follow the same contract as ``recurrent_rg.ising``:
``train_x`` has shape ``[n_systems, ensemble, sites, 1]`` with binary spins and
``train_j`` stores the transverse-field ratio ``h / J``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


DTYPE = torch.float64


def exact_tfim_block_rg_flow(coupling: torch.Tensor, levels: int) -> torch.Tensor:
    """Drell-Weinstein-Jankowski two-site block RG flow for ``x = h/J``.

    The block map is
        ``J' = J/2 (1 + J / sqrt(J^2 + 4 h^2))``
        ``h' = (sqrt(J^2 + 4 h^2) - J) / 2``
    and the ratio ``x = h/J`` flows as
        ``x' = y (y - 1) / (y + 1)``  with  ``y = sqrt(1 + 4 x^2)``.
    This flow has a nontrivial unstable fixed point at ``x* ~= 1.277``
    (exact critical point of the 1D TFIM is ``x = 1``).
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")
    values = [coupling]
    for _ in range(levels - 1):
        x = values[-1]
        y = torch.sqrt(1.0 + 4.0 * x * x)
        values.append(y * (y - 1.0) / (y + 1.0))
    return torch.stack(values, dim=-1)


def wolff_add_probability(coupling: float) -> float:
    """Return the Wolff activation probability ``1 - exp(-2 K)``.

    ``K`` is the dimensionless ferromagnetic Ising coupling on the classical
    lattice. In particular, a zero-strength bond must never be activated.
    """
    if coupling < 0.0:
        raise ValueError("Wolff updates require non-negative couplings")
    return -math.expm1(-2.0 * coupling)


def _cluster_flip(
    spins: np.ndarray,
    p_space: float,
    p_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one Wolff single-cluster update on the periodic 2D lattice."""
    nt, ns = spins.shape
    seed = (int(rng.integers(nt)), int(rng.integers(ns)))
    target_spin = spins[seed]
    cluster = np.zeros((nt, ns), dtype=bool)
    cluster[seed] = True
    stack = [seed]

    while stack:
        t, s = stack.pop()
        neighbours = (
            ((t - 1) % nt, s, p_time),
            ((t + 1) % nt, s, p_time),
            (t, (s - 1) % ns, p_space),
            (t, (s + 1) % ns, p_space),
        )
        for neighbour_t, neighbour_s, probability in neighbours:
            if (
                not cluster[neighbour_t, neighbour_s]
                and spins[neighbour_t, neighbour_s] == target_spin
                and rng.random() < probability
            ):
                cluster[neighbour_t, neighbour_s] = True
                stack.append((neighbour_t, neighbour_s))

    out = spins.copy()
    out[cluster] = -out[cluster]
    return out


def sample_tfim_ensembles(
    couplings: np.ndarray,
    replicas: int,
    samples: int,
    sites: int,
    seed: int,
    time_slices: int = 32,
    beta_over_slices: float = 0.5,
    equilibration_sweeps: int = 32,
    decorrelation_sweeps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw periodic TFIM sigma^z ensembles via path-integral Monte Carlo.

    The Suzuki-Trotter mapping turns the 1D quantum chain into a 2D classical
    Ising model of shape ``(time_slices, sites)`` with anisotropic couplings

        ``K_space = beta_over_slices``
        ``K_time = 0.5 * log(coth(beta_over_slices * h / J))``

    Wolff single-cluster updates are run on the 2D lattice; each returned
    configuration is the sigma^z state on one time slice of the last equilibrated
    2D configuration. Every first-axis entry of the output is one physical
    system: an ensemble sharing one ``h / J``.

    Args:
        couplings: Transverse-field ratios ``h / J`` (J > 0 fixed to 1).
        replicas: Independent Markov chains per coupling.
        samples: Configurations kept per replica.
        sites: Spatial chain length.
        seed: Numpy RNG seed for reproducibility.
        time_slices: Trotter number M.
        beta_over_slices: beta / M; spatial coupling per slice is fixed to this.
        equilibration_sweeps: Single-cluster updates discarded before recording.
        decorrelation_sweeps: Single-cluster updates between configurations.

    Returns:
        ``(ensembles, labels)`` with ``ensembles`` of shape
        ``[len(couplings) * replicas, samples, sites, 1]`` and ``labels`` the
        repeated coupling values.
    """
    if sites < 4 or sites % 2:
        raise ValueError("sites must be an even integer greater than or equal to 4")
    if time_slices < 4:
        raise ValueError("time_slices must be at least 4")
    if replicas < 1 or samples < 1:
        raise ValueError("replicas and samples must be positive")
    if beta_over_slices <= 0.0:
        raise ValueError("beta_over_slices must be positive")
    if equilibration_sweeps < 0 or decorrelation_sweeps < 1:
        raise ValueError("equilibration_sweeps must be non-negative and decorrelation_sweeps positive")

    rng = np.random.default_rng(seed)
    ensembles: list[torch.Tensor] = []
    labels: list[float] = []

    for coupling in couplings:
        x = float(coupling)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("TFIM transverse-field ratios must be finite and non-negative")
        k_space = beta_over_slices
        p_space = wolff_add_probability(k_space)
        if x == 0.0:
            # K_time diverges at zero transverse field, so every aligned
            # imaginary-time bond belongs to the cluster.
            p_time = 1.0
        else:
            z = beta_over_slices * x
            k_time = 0.5 * math.log(1.0 / math.tanh(z))
            p_time = wolff_add_probability(k_time)

        for _ in range(replicas):
            spins = rng.choice(np.array([-1.0, 1.0]), size=(time_slices, sites))
            accepted: list[torch.Tensor] = []

            total_sweeps = equilibration_sweeps + samples * decorrelation_sweeps
            for sweep in range(total_sweeps):
                spins = _cluster_flip(spins, p_space, p_time, rng)
                if sweep >= equilibration_sweeps and (
                    sweep - equilibration_sweeps + 1
                ) % decorrelation_sweeps == 0:
                    # Record a random time slice as one quantum configuration.
                    slice_index = rng.integers(time_slices)
                    accepted.append(
                        torch.tensor(spins[slice_index].copy(), dtype=DTYPE)
                    )
                if len(accepted) >= samples:
                    break

            ensembles.append(torch.stack(accepted[:samples]))
            labels.append(x)

    return (
        torch.stack(ensembles).unsqueeze(-1),
        torch.tensor(labels, dtype=DTYPE),
    )
