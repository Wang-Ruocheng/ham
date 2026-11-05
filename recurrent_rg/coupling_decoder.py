"""Label-free, finite-basis Gibbs decoder for an explicit representation ablation.

The natural parameters are affine in g, not an arbitrary nonlinear action:
``H_g(s) = -sum_j (a_j + W_j g) S_j(s)``.  The seven even, directional
translation sums below are a declared finite-range prior, NOT exact RG
closure and NOT proof that a latent eigenvalue is a thermal exponent.

All observed spin tensors are detached.  Only g and decoder parameters receive
gradients.  Single-site pseudolikelihood is exact for this chosen Hamiltonian;
it does not require a partition function or construct a full flipped lattice.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


OPERATOR_NAMES = (
    'nn_x', 'nn_y', 'diagonal_plus', 'diagonal_minus',
    'axial_distance2_x', 'axial_distance2_y', 'plaquette',
)
OPERATOR_DEFINITIONS = (
    'sum_x s(x) s(x+e1)',
    'sum_x s(x) s(x+e2)',
    'sum_x s(x) s(x+e1+e2)',
    'sum_x s(x) s(x+e1-e2)',
    'sum_x s(x) s(x+2e1)',
    'sum_x s(x) s(x+2e2)',
    'sum_x s(x) s(x+e1) s(x+e2) s(x+e1+e2)',
)
BINARY_ROUNDOFF_TOLERANCE = 1e-12


def _binary_spins(spins):
    if not isinstance(spins, torch.Tensor):
        raise TypeError('Observed spins must be a torch Tensor')
    values = spins.detach()
    if values.ndim >= 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim < 2 or min(values.shape[-2:]) < 5:
        raise ValueError('Periodic spatial dimensions must both be at least five')
    # A straight-through hard+soft-soft.detach() sample can differ from +/-1
    # by a few floating-point ulps. Canonicalize only this declared roundoff,
    # never interpret a genuinely relaxed field as a binary observation.
    if (not torch.isfinite(values).all()
            or not torch.all((values.abs()-1).abs() <= BINARY_ROUNDOFF_TOLERANCE)):
        raise ValueError('Observed spins must be binary plus/minus one up to 1e-12 roundoff')
    return values.sign()


def _shift(spins, dx, dy):
    """At x return s(x+dx e1+dy e2), with periodic spatial boundaries."""
    return torch.roll(spins, shifts=(-dx, -dy), dims=(-2, -1))


def _operator_densities(spins):
    sx, sy = _shift(spins, 1, 0), _shift(spins, 0, 1)
    return torch.stack((
        spins * sx,
        spins * sy,
        spins * _shift(spins, 1, 1),
        spins * _shift(spins, 1, -1),
        spins * _shift(spins, 2, 0),
        spins * _shift(spins, 0, 2),
        spins * sx * sy * _shift(spins, 1, 1),
    ), dim=-1)


def directional_operator_values(spins):
    """Seven extensive even sums, shape ``[...,7]``; no area normalization.

    Input is ``[...,H,W]`` or ``[...,H,W,1]``.  H,W>=5 prevents periodic
    self-interaction/duplicate-neighbour degeneracies in this explicit basis.
    """
    values = _binary_spins(spins)
    return _operator_densities(values).sum(dim=(-3, -2))


def _conditional_features(spins):
    """Local h_i features: H(s_i flipped)-H(s) = 2 s_i h_i."""
    plus_x, minus_x = _shift(spins, 1, 0), _shift(spins, -1, 0)
    plus_y, minus_y = _shift(spins, 0, 1), _shift(spins, 0, -1)
    # The four plaquettes containing the conditional site.  Its own spin is
    # omitted from each triple, exactly differentiating the extensive sum.
    plaquettes = (
        plus_x * plus_y * _shift(spins, 1, 1)
        + minus_x * plus_y * _shift(spins, -1, 1)
        + plus_x * minus_y * _shift(spins, 1, -1)
        + minus_x * minus_y * _shift(spins, -1, -1)
    )
    return torch.stack((
        plus_x + minus_x,
        plus_y + minus_y,
        _shift(spins, 1, 1) + _shift(spins, -1, -1),
        _shift(spins, 1, -1) + _shift(spins, -1, 1),
        _shift(spins, 2, 0) + _shift(spins, -2, 0),
        _shift(spins, 0, 2) + _shift(spins, 0, -2),
        plaquettes,
    ), dim=-1)


class DirectionalCouplingDecoder(nn.Module):
    """Affine natural couplings shared over every configuration and scale.

    One g vector per independent ensemble is broadcast over its configurations
    and spatial sites.  The caller must supply that ensemble's cross-fit g;
    this module never reads coupling labels or infers an ensemble assignment.
    """

    def __init__(self, theory_dim, *, device=None, dtype=torch.float64):
        super().__init__()
        if type(theory_dim) is not int or theory_dim < 1:
            raise ValueError('Positive integer theory_dim required')
        self.theory_dim = theory_dim
        self.linear = nn.Linear(theory_dim, len(OPERATOR_NAMES), device=device, dtype=dtype)

    def couplings(self, theory):
        """Return seven natural parameters ``a+Wg``, shape [ensembles,7]."""
        if (not isinstance(theory, torch.Tensor) or theory.ndim != 2
                or theory.shape[-1] != self.theory_dim or not len(theory)
                or not torch.isfinite(theory).all()):
            raise ValueError('Need one finite g vector per independent ensemble')
        return self.linear(theory)

    def _observed(self, spins, theory):
        values = _binary_spins(spins)
        couplings = self.couplings(theory)
        if values.ndim < 3 or values.shape[0] != len(couplings):
            raise ValueError('The first spin axis must match the ensemble g axis')
        if values.device != couplings.device:
            raise ValueError('Observed spins and ensemble g must share a device')
        return values, couplings

    def energy(self, spins, theory):
        """Global H per saved configuration, without evaluating a normalizer."""
        values, couplings = self._observed(spins, theory)
        operators = _operator_densities(values).sum(dim=(-3, -2))
        shape = (len(couplings),) + (1,) * (operators.ndim-2) + (len(OPERATOR_NAMES),)
        return -(operators * couplings.reshape(shape)).sum(dim=-1)

    def local_field(self, spins, theory):
        """Exact conditional h, same spin shape except an optional channel axis."""
        values, couplings = self._observed(spins, theory)
        shape = (len(couplings),) + (1,) * (values.ndim-1) + (len(OPERATOR_NAMES),)
        return (_conditional_features(values) * couplings.reshape(shape)).sum(dim=-1)

    def pseudolikelihood(self, spins, theory, *, reduction='mean'):
        """Mean site/configuration negative log p(s_i | other spins,g), in nats.

        ``none`` retains one value per independent physical ensemble, NOT one
        value per spatial site or configuration. ``mean`` weights ensembles
        equally.  Spatial/configuration correlations are not claimed absent.
        """
        if reduction not in ('mean', 'none'):
            raise ValueError("Pseudolikelihood reduction must be 'mean' or 'none'")
        values = _binary_spins(spins)
        field = self.local_field(values, theory)
        loss = F.softplus(-2 * values * field).flatten(start_dim=1).mean(dim=1)
        return loss.mean() if reduction == 'mean' else loss

    def crossfit_loss(self, g_left, spins_right, g_right, spins_left, *, reduction='mean'):
        """Symmetric held-half conditional loss; no configuration generates g here."""
        left_to_right = self.pseudolikelihood(spins_right, g_left, reduction='none')
        right_to_left = self.pseudolikelihood(spins_left, g_right, reduction='none')
        if left_to_right.shape != right_to_left.shape:
            raise ValueError('Both cross-fit halves must have the same physical ensembles')
        value = .5 * (left_to_right + right_to_left)
        if reduction not in ('mean', 'none'):
            raise ValueError("Cross-fit reduction must be 'mean' or 'none'")
        return value.mean() if reduction == 'mean' else value


def crossfit_loss(decoder, g_left, spins_right, g_right, spins_left, *, reduction='mean'):
    """Functional form of :meth:`DirectionalCouplingDecoder.crossfit_loss`."""
    if not isinstance(decoder, DirectionalCouplingDecoder):
        raise TypeError('Expected the declared directional coupling decoder')
    return decoder.crossfit_loss(g_left, spins_right, g_right, spins_left, reduction=reduction)


@torch.no_grad()
def normalize_coupling_decoder_(decoder, *, epsilon=1e-12):
    """Optional column-norm training gauge; never whiten or force full rank.

    A zero column remains zero. This projection changes the decoded model if
    g is held fixed: it is a declared training convention, NOT a post-hoc
    invariance transformation or an exponent correction. Biases are unchanged.
    """
    if not isinstance(decoder, DirectionalCouplingDecoder):
        raise TypeError('Expected the declared directional coupling decoder')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Finite positive normalization epsilon required')
    weight = decoder.linear.weight
    weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(epsilon))
    return decoder
