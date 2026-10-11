"""Binary local distribution descriptors, separate from the QCR objective."""
from math import prod

import torch


def _binary_pattern_frequencies(pattern, bins, dtype, dimensions):
    """Count each ensemble's motifs without a site-by-bin one-hot tensor.

    The axes reduced here match the historical one-hot ``mean`` exactly.
    In particular, an empty sample/spatial reduction still yields NaNs,
    whereas an empty system axis remains an empty output. Count in integers
    first and use the mean's accumulation precision before the final cast.
    """
    if not dtype.is_floating_point:
        # Historical one_hot(...).to(dtype).mean(...) rejects integer inputs.
        raise RuntimeError('mean(): Input dtype must be a floating point dtype')
    reduced_axes = tuple(range(1, dimensions + 2))
    kept_axes = (0,) + tuple(range(dimensions + 2, pattern.ndim))
    kept_shape = tuple(pattern.shape[axis] for axis in kept_axes)
    rows = prod(kept_shape)
    observations = prod(pattern.shape[axis] for axis in reduced_axes)
    flat = pattern.permute(kept_axes + reduced_axes).reshape(rows, observations)
    offsets = torch.arange(rows, device=pattern.device)[:, None] * bins
    counts = torch.bincount((flat + offsets).flatten(), minlength=rows * bins)
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    frequencies = counts.reshape(rows, bins).to(accumulation_dtype) / observations
    return frequencies.to(dtype).reshape(*kept_shape, bins)


def categorical_plaquette_distribution(configurations):
    """Empirical probabilities of q**4 oriented plaquettes; no labels."""
    q = configurations.shape[-1]
    state = configurations.argmax(-1)
    pattern = (q**3*state + q**2*torch.roll(state,-1,2) + q*torch.roll(state,-1,3)
               + torch.roll(torch.roll(state,-1,2),-1,3))
    flat = pattern.reshape(len(pattern),-1)
    offsets = torch.arange(len(flat),device=flat.device)[:,None]*q**4
    counts = torch.bincount((flat+offsets).flatten(),minlength=len(flat)*q**4)
    return counts.reshape(len(flat),q**4).to(configurations.dtype)/flat.shape[1]

def two_spin_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """1D: empirical probabilities of 4 two-spin patterns."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = 2 * bits + torch.roll(bits, -1, dims=2)
    return _binary_pattern_frequencies(pattern, 4, configurations.dtype, 1)


def plaquette_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """2D: empirical probabilities of 16 oriented 2x2 spin plaquettes."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = (
        8 * bits
        + 4 * torch.roll(bits, -1, dims=2)
        + 2 * torch.roll(bits, -1, dims=3)
        + torch.roll(torch.roll(bits, -1, dims=2), -1, dims=3)
    )
    return _binary_pattern_frequencies(pattern, 16, configurations.dtype, 2)
