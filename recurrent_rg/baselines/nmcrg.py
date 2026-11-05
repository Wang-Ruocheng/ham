"""Configuration-only NMCRG convolutional RBM, reimplemented from the paper.

The model is the zero-bias, +/-1 RBM of Chung and Kao, Phys. Rev. Research
3, 023230 (2021), DOI: 10.1103/PhysRevResearch.3.023230.  It learns a
normalized stochastic coarse-graining kernel by visible-marginal contrastive
divergence; a reconstruction BCE is not its training objective.

The default overlapping 9x9 filter and stride 3 are a same-data b=3 *method
adaptation*, not the paper's native 8x8, stride-2 pipeline.  This module does
not implement, or claim to reproduce, the paper's subsequent inverse-coupling
or MCRG analysis.  No upstream repository source is imported or vendored.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicConvRBM(nn.Module):
    """Zero-bias periodic convolutional RBM on square binary spin lattices.

    Fields accept ``[..., L, L]`` tensors, with L divisible by ``stride``.
    They are linear maps and consequently also accept nonbinary real tensors
    (useful for testing the exact bilinear adjoint).  Sampling and CD expect
    physical +/-1 spin states.  Inputs must already have the parameter's
    device and floating dtype; this class never moves or casts input data.

    ``free_energy`` averages over batch/prefix and hidden spatial sites, as
    in the author's training code.  Thus it is the extensive RBM free energy
    divided by the number of hidden sites, not an extensive log probability.
    """

    def __init__(
        self,
        filter_width: int = 9,
        stride: int = 3,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(filter_width, Integral)
            or isinstance(filter_width, bool)
            or not isinstance(stride, Integral)
            or isinstance(stride, bool)
            or stride <= 0
            or filter_width < stride
        ):
            raise ValueError("filter_width and stride must be integers with filter_width >= stride > 0")
        self.filter_width = int(filter_width)
        self.stride = int(stride)
        weight = torch.empty(
            (1, 1, self.filter_width, self.filter_width),
            device=device,
            dtype=dtype,
        )
        if not weight.is_floating_point():
            raise TypeError("PeriodicConvRBM requires a floating dtype")
        self.weight = nn.Parameter(weight)
        with torch.no_grad():
            self.weight.uniform_(-1e-5, 1e-5, generator=generator)

    def _check_lattice(self, lattice: torch.Tensor, *, visible: bool) -> tuple[tuple[int, ...], int]:
        if not isinstance(lattice, torch.Tensor):
            raise TypeError("lattice must be a torch.Tensor")
        if lattice.ndim < 2 or lattice.shape[-2] != lattice.shape[-1] or lattice.shape[-1] < 1:
            raise ValueError("lattice must have nonempty square trailing dimensions [..., L, L]")
        if lattice.numel() == 0:
            raise ValueError("lattice prefix dimensions must be nonempty")
        if lattice.device != self.weight.device:
            raise ValueError("input and RBM weight must already be on the same device")
        if lattice.dtype != self.weight.dtype:
            raise TypeError("input and RBM weight must already have the same floating dtype")
        size = lattice.shape[-1]
        if visible and size % self.stride:
            raise ValueError("visible lattice size must be divisible by stride")
        return tuple(lattice.shape[:-2]), size

    def hidden_fields(self, visible: torch.Tensor) -> torch.Tensor:
        """Return a_j = sum_(a,b) W_ab sigma_(stride*j+(a,b)), periodically."""
        prefix, size = self._check_lattice(visible, visible=True)
        batch = math.prod(prefix) if prefix else 1
        flat = visible.reshape(batch, 1, size, size)
        # Explicit modular indices also handle kernels wider than one lattice
        # period and avoid the zero-padding ``[:-0]`` special case entirely.
        extent = size + self.filter_width - self.stride
        periodic = torch.arange(extent, device=visible.device) % size
        padded = flat.index_select(-2, periodic).index_select(-1, periodic)
        fields = F.conv2d(padded, self.weight, stride=self.stride)
        coarse_size = size // self.stride
        return fields.reshape(*prefix, coarse_size, coarse_size)

    def visible_fields(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the exact periodic bilinear adjoint of ``hidden_fields``.

        A hidden lattice of side H maps to a visible lattice of side
        ``H * stride``.  All periodic images, including repeated wraparound
        and corner contributions, are summed rather than overwritten.
        """
        prefix, hidden_size = self._check_lattice(hidden, visible=False)
        batch = math.prod(prefix) if prefix else 1
        flat = hidden.reshape(batch, 1, hidden_size, hidden_size)
        expanded = F.conv_transpose2d(flat, self.weight, stride=self.stride)
        size = hidden_size * self.stride
        extent = expanded.shape[-1]
        # Fold rows, then columns, with a fixed addition order.  Unlike a
        # scatter-add this does not introduce CUDA atomic-sum nondeterminism.
        rows = expanded[..., :size, :]
        for start in range(size, extent, size):
            image = expanded[..., start : start + size, :]
            rows = rows + F.pad(image, (0, 0, 0, size - image.shape[-2]))
        folded = rows[..., :size]
        for start in range(size, extent, size):
            image = rows[..., start : start + size]
            folded = folded + F.pad(image, (0, size - image.shape[-1]))
        return folded.reshape(*prefix, size, size)

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """Stable negative mean log(2*cosh(a)), with no partition-function term."""
        fields = self.hidden_fields(visible)
        return -torch.logaddexp(fields, -fields).mean()

    @staticmethod
    def _draw_spins(fields: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        probabilities = torch.sigmoid(2 * fields)
        uniforms = torch.rand(
            fields.shape, dtype=fields.dtype, device=fields.device, generator=generator
        )
        return uniforms.lt(probabilities).to(fields.dtype).mul_(2).sub_(1)

    @torch.no_grad()
    def sample_hidden(
        self, visible: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample normalized P(mu|sigma), returning detached +/-1 spins."""
        return self._draw_spins(self.hidden_fields(visible), generator)

    @torch.no_grad()
    def sample_visible(
        self, hidden: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample P(sigma|mu) from the exact adjoint fields."""
        return self._draw_spins(self.visible_fields(hidden), generator)

    def cd_loss(
        self,
        visible: torch.Tensor,
        steps: int = 3,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the positive-minus-negative free-energy CD-k surrogate.

        The negative chain starts at the input and performs k successive
        visible->hidden->visible Gibbs transitions.  Samples are detached;
        gradients of both free-energy evaluations still reach ``weight``.
        The scalar is a CD gradient surrogate, not the exact KL or NLL.
        """
        if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self._check_lattice(visible, visible=True)
        with torch.no_grad():
            negative = visible.detach()
            for _ in range(int(steps)):
                hidden = self.sample_hidden(negative, generator=generator)
                negative = self.sample_visible(hidden, generator=generator)
        return self.free_energy(visible) - self.free_energy(negative.detach())

    @torch.no_grad()
    def coarse_grain(
        self, visible: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample one RG step for arbitrary prefix dimensions, without training."""
        return self.sample_hidden(visible, generator=generator)
