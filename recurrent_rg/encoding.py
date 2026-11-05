"""The half-ensemble Q estimator shared by training and saved evaluation."""
from __future__ import annotations

import torch


def split_theory_views(model, levels):
    """Encode two disjoint equal halves, preserving the training gradient route.

    The caller supplies already sampled C levels. No C call, random draw or
    cached parameter-dependent table survives this call. In particular,
    averaging the returned Q values is not Q of the combined ensemble when
    the Q head is nonlinear.
    """
    levels = list(levels)
    if not levels:
        raise ValueError('Require at least one ensemble scale')
    samples = levels[0].shape[1]
    midpoint = samples // 2
    if midpoint < 1 or samples % 2 or any(level.shape[:2] != levels[0].shape[:2] for level in levels):
        raise ValueError('All scales require the same positive even ensemble size')
    encoder_levels = ([level.detach() for level in levels]
                      if getattr(model, 'detach_kernel_representation', False) else levels)
    encoder = model.theory_encoder
    if (getattr(model, 'share_binary_q_tables', False)
            and getattr(model, 'detach_kernel_representation', False)
            and getattr(encoder, 'binary_lookup', False)
            and not getattr(encoder, 'categorical_states', 0)):
        encoded = encoder.forward_many(
            [level[:, :midpoint] for level in encoder_levels]
            + [level[:, midpoint:] for level in encoder_levels],
            validated_binary_inputs=getattr(model, 'fast_validated_statistics', False))
        first = torch.stack(encoded[:len(levels)], dim=1)
        second = torch.stack(encoded[len(levels):], dim=1)
    else:
        first = torch.stack([encoder(level[:, :midpoint]) for level in encoder_levels], dim=1)
        second = torch.stack([encoder(level[:, midpoint:]) for level in encoder_levels], dim=1)
    return first, second


def half_subset_theory(model, levels):
    """The exact estimator used by the QCR closure objective."""
    first, second = split_theory_views(model, levels)
    return .5 * (first + second)
