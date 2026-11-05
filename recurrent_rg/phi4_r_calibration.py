"""Label-free R calibration on already frozen continuous-field encodings.

The normalizer changes the network's internal units only. All closure errors
are measured in the original g coordinates, and a single autonomous map is
shared across the observed scales. This module never loads or invokes Q/C.
"""
from __future__ import annotations

import torch
from torch import nn


def _frozen_direct(direct: torch.Tensor) -> None:
    if (not isinstance(direct, torch.Tensor) or direct.ndim < 3
            or direct.shape[-2] != 3 or direct.shape[-1] < 1
            or direct.numel() == 0 or not direct.is_floating_point()
            or direct.requires_grad or not torch.isfinite(direct).all()):
        raise ValueError("Expected finite frozen g with shape [...,3,theory_dim]")


def fit_input_normalizer(train_direct) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit shared TRAIN-only population moments, pooling input scales 0 and 1.

    Leading axes can include physical chains and predeclared stochastic C
    repeats. Scale 2 is a target only and never contributes to these moments.
    Degenerate coordinates fail explicitly instead of adding a scale floor.
    """
    if isinstance(train_direct, torch.Tensor) and train_direct.requires_grad:
        raise ValueError("Normalizer requires frozen TRAIN encodings")
    direct = torch.as_tensor(train_direct, dtype=torch.float64)
    _frozen_direct(direct)
    inputs = direct[..., :2, :].reshape(-1, direct.shape[-1])
    center = inputs.mean(dim=0)
    scale = inputs.std(dim=0, unbiased=False)
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("Every TRAIN input coordinate requires a finite positive standard deviation")
    return center.detach().clone(), scale.detach().clone()


class FrozenResidualMap(nn.Module):
    """An autonomous residual map with fixed, checkpointed internal units.

    ``advance(g) = g + scale * net((g-center)/scale)``. Zero center and unit
    scale recover an ordinary residual MLP. Statistics are persistent buffers,
    not parameters, and are never recomputed by ``train()`` or ``eval()``.
    """

    def __init__(self, theory_dim: int = 2, hidden: int = 96, *,
                 center=None, scale=None):
        super().__init__()
        if (not isinstance(theory_dim, int) or isinstance(theory_dim, bool)
                or theory_dim < 1 or not isinstance(hidden, int)
                or isinstance(hidden, bool) or hidden < 1):
            raise ValueError("theory_dim and hidden must be positive integers")
        self.theory_dim, self.hidden = theory_dim, hidden
        center = torch.zeros(theory_dim, dtype=torch.float64) if center is None else torch.as_tensor(center, dtype=torch.float64)
        scale = torch.ones(theory_dim, dtype=torch.float64) if scale is None else torch.as_tensor(scale, dtype=torch.float64)
        if (center.shape != (theory_dim,) or scale.shape != (theory_dim,)
                or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise ValueError("center and positive scale must be finite theory_dim vectors")
        # Construct on CPU, then move through nn.Module.to together when needed.
        self.register_buffer("center", center.detach().cpu().clone())
        self.register_buffer("scale", scale.detach().cpu().clone())
        self.net = nn.Sequential(nn.Linear(theory_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, theory_dim)).double()

    def advance(self, g: torch.Tensor) -> torch.Tensor:
        if g.ndim < 1 or g.shape[-1] != self.theory_dim:
            raise ValueError("g must end in the configured theory_dim")
        return g + self.scale * self.net((g - self.center) / self.scale)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.advance(g)


def _require_raw(raw: FrozenResidualMap) -> None:
    if not isinstance(raw, FrozenResidualMap):
        raise TypeError("Expected FrozenResidualMap")
    if (not torch.equal(raw.center, torch.zeros_like(raw.center))
            or not torch.equal(raw.scale, torch.ones_like(raw.scale))):
        raise ValueError("Expected an original-coordinate map with zero center and unit scale")


def conditioned_copy(raw: FrozenResidualMap, center, scale) -> FrozenResidualMap:
    """Copy a raw map into new internal units, preserving physical F and DF."""
    _require_raw(raw)
    # Construction is deterministic relative to the source and does not consume
    # the caller's CPU RNG stream; every initialized value is overwritten below.
    with torch.random.fork_rng(devices=[]):
        result = FrozenResidualMap(raw.theory_dim, raw.hidden, center=center,
                                   scale=scale).to(raw.center.device)
    result.net.load_state_dict(raw.net.state_dict())
    with torch.no_grad():
        result.net[0].weight.copy_(raw.net[0].weight * result.scale[None, :])
        result.net[0].bias.copy_(raw.net[0].bias + raw.net[0].weight @ result.center)
        result.net[-1].weight.copy_(raw.net[-1].weight / result.scale[:, None])
        result.net[-1].bias.copy_(raw.net[-1].bias / result.scale)
    result.train(raw.training)
    return result


def function_preserving_widen(raw: FrozenResidualMap, new_hidden: int,
                              seed: int) -> FrozenResidualMap:
    """Embed a raw tanh MLP into a wider one without changing F or DF.

    Extra hidden units have random incoming features and zero outgoing weights.
    Their output-weight gradients can become nonzero immediately. The original
    hidden block is disconnected from extra second-layer inputs, so padding
    introduces no extra path at initialization. CPU RNG state is restored.
    """
    _require_raw(raw)
    if (not isinstance(new_hidden, int) or isinstance(new_hidden, bool)
            or new_hidden <= raw.hidden):
        raise ValueError("new_hidden must exceed the original hidden width")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        result = FrozenResidualMap(raw.theory_dim, new_hidden).to(raw.center.device)
    width = raw.hidden
    with torch.no_grad():
        result.net[0].weight[:width].copy_(raw.net[0].weight)
        result.net[0].bias[:width].copy_(raw.net[0].bias)
        result.net[2].weight[:width, :width].copy_(raw.net[2].weight)
        result.net[2].weight[:width, width:].zero_()
        result.net[2].bias[:width].copy_(raw.net[2].bias)
        result.net[-1].weight[:, :width].copy_(raw.net[-1].weight)
        result.net[-1].weight[:, width:].zero_()
        result.net[-1].bias.copy_(raw.net[-1].bias)
    result.train(raw.training)
    return result


def all_start_closure_loss(model: nn.Module, direct: torch.Tensor):
    """Equal-target all-start loss for exactly two coarse-graining steps.

    Coordinates are summed before averaging leading sample axes. The two-step
    call retains its intermediate graph; only the fixed encodings are frozen.
    This equals 5 e01 + 2.5 e12 + 2.5 e02 in original physical g coordinates.
    """
    _frozen_direct(direct)
    p01 = model.advance(direct[..., 0, :])
    p12 = model.advance(direct[..., 1, :])
    p02 = model.advance(p01)
    terms = {
        "e01": (p01 - direct[..., 1, :]).square().sum(-1).mean(),
        "e12": (p12 - direct[..., 2, :]).square().sum(-1).mean(),
        "e02": (p02 - direct[..., 2, :]).square().sum(-1).mean(),
    }
    return 5 * terms["e01"] + 2.5 * terms["e12"] + 2.5 * terms["e02"], terms


closure_loss = all_start_closure_loss
