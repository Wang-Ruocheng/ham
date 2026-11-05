"""Explicit local categorical statistics, without physical observables or labels."""
from itertools import product

import torch
from torch import nn


def categorical_local_distribution(configurations, *, width=2, max_bins=65536):
    """Oriented overlapping width**d motif probabilities, averaged over D.

    Input [systems,samples,*lattice,q] is categorical one-hot. The input is
    intentionally detached: this Q representation cannot update C.
    """
    d = configurations.ndim-3
    q = configurations.shape[-1]
    if d < 1 or q < 2 or width < 1:
        raise ValueError('Positive lattice dimension/width and q>=2 required')
    bins = q**(width**d)
    if bins > max_bins:
        raise ValueError('Categorical motif table exceeds the explicit bin budget')
    state = configurations.detach().argmax(-1)
    code = torch.zeros_like(state)
    axes = tuple(range(2, 2+d))
    # First lattice axis varies fastest, matching the existing 2D descriptor.
    for reversed_offsets in product(range(width), repeat=d):
        offsets = reversed_offsets[::-1]
        code = q*code + torch.roll(state, tuple(-a for a in offsets), axes)
    flat = code.reshape(len(code), -1)
    row = torch.arange(len(code), device=code.device)[:, None]*bins
    count = torch.bincount((flat+row).flatten(), minlength=len(code)*bins)
    return count.reshape(len(code), bins).to(configurations.dtype)/flat.shape[1]


class LocalDistributionEncoder(nn.Module):
    """Learned linear readout of exact motifs: no near-linear pixel pooling."""
    def __init__(self, states, dimensions, theory_dim=1, width=2):
        super().__init__()
        self.states, self.dimensions, self.width = states, dimensions, width
        bins = states**(width**dimensions)
        if bins > 65536:
            raise ValueError('Explicit motif table is too large')
        self.projection = nn.Linear(bins, theory_dim)

    def from_distribution(self, distribution):
        return self.projection(distribution)

    def forward(self, configurations):
        if configurations.ndim != self.dimensions+3 or configurations.shape[-1] != self.states:
            raise ValueError('Configuration shape does not match this encoder')
        return self.from_distribution(categorical_local_distribution(configurations, width=self.width))
