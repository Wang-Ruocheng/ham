"""Retired uniform-block distinctness diagnostic; never an active C objective.

Kept only to read/compare historical distinct-loss experiments without changing
their recorded definition. Current training uses color-preservation NLL and
does not import this module.
"""
import math
import torch

from recurrent_rg.kernel_regularizers import uniform_block_probabilities


def kernel_distinct_loss(kernel, *, margin=.8):
    """Historical mean [margin-TV(P(.|a^m),P(.|b^m))]_+^2 over a<b.

    Output-permutation invariant, but not an information-preservation guarantee.
    At identical probability vectors abs retains its historical zero subgradient.
    No sampling or random initialization is performed by this diagnostic.
    """
    if not math.isfinite(margin) or not 0 < margin <= 1:
        raise ValueError('Distinct margin must be in (0,1]')
    probabilities = uniform_block_probabilities(kernel)
    pairs = torch.triu_indices(len(probabilities), len(probabilities), offset=1,
                              device=probabilities.device)
    tv = .5*(probabilities[pairs[0]]-probabilities[pairs[1]]).abs().sum(-1)
    return torch.relu(margin-tv).square().mean()
