"""Label-free prediction of independently measured future-scale physics.

The predictor and loss are observable-agnostic.  The provided observable
adapter is explicitly limited to periodic, single-channel binary 2D lattices;
it is not a universal Hamiltonian or a claim of complete RG sufficiency.
There is no K, critical-point, exponent, or theoretical-eigenvalue input.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn


DEFAULT_FUTURE_PATHS = ((0, 1), (1, 2), (0, 2))
BINARY2D_OBSERVABLE_NAMES = (
    "轴向距离L/6关联", "轴向距离L/3关联", "最低非零轴向结构因子",
    "第二非零轴向结构因子", "最低对角结构因子",
)


@torch.no_grad()
def binary2d_future_observables(level: torch.Tensor) -> torch.Tensor:
    """Return five physical ensemble means, shape [ensembles, 5].

    Accept [B, samples, H, W, 1] or [B, samples, H, W], with hard +/-1
    spins and periodic boundaries.  H=W=L must be >=12 and divisible by 6.
    Correlation separations are L/6 and L/3, so the two directions remain
    distinct at every b3 scale (108 -> 36 -> 12).  This explicit b3 adapter
    is not a reproduction of the historical fixed-r4/r8 diagnostic metric.
    S(q) uses |sum s exp(iqx)|^2 / (H*W)^2, never the zero mode.
    """
    spins = level.detach()
    if spins.ndim == 5 and spins.shape[-1] == 1:
        spins = spins[..., 0]
    if spins.ndim != 4 or min(spins.shape[:2]) < 1:
        raise ValueError("Binary 2D observables require [B,S,H,W,1] or [B,S,H,W]")
    height, width = spins.shape[-2:]
    if height != width or height < 12 or height % 6:
        raise ValueError("Binary 2D future observables require square L>=12 divisible by 6")
    if spins.is_complex():
        raise ValueError("Binary 2D future observables require real hard +/-1 spins")
    if not spins.is_floating_point():
        spins = spins.to(torch.float64)
    if spins.dtype not in (torch.float32, torch.float64):
        raise ValueError("Binary 2D future observables require float32 or float64")
    # Straight-through hard samples can differ from +/-1 by a few ulps after
    # the hard + soft - soft.detach() arithmetic.  Accept only that numerical
    # envelope, not genuinely soft spins, and canonicalize detached targets.
    tolerance = 8 * torch.finfo(spins.dtype).eps
    if (not bool(torch.isfinite(spins).all())
            or not bool(((spins.abs() - 1).abs() <= tolerance).all())):
        raise ValueError("Binary 2D future observables require finite hard +/-1 spins")
    spins = torch.where(spins >= 0, torch.ones_like(spins), -torch.ones_like(spins))

    per_configuration = []
    for distance in (height // 6, height // 3):
        correlations = spins * (
            torch.roll(spins, -distance, -2) + torch.roll(spins, -distance, -1)
        )
        per_configuration.append(correlations.mean((-2, -1)) / 2)
    real_dtype = spins.dtype
    qh = 2 * torch.pi * torch.arange(height, device=spins.device, dtype=real_dtype) / height
    qw = 2 * torch.pi * torch.arange(width, device=spins.device, dtype=real_dtype) / width
    row, column = spins.sum(-1), spins.sum(-2)
    volume_squared = (height * width) ** 2
    for harmonic in (1, 2):
        hpower = ((row * torch.cos(harmonic * qh)).sum(-1).square()
                  + (row * torch.sin(harmonic * qh)).sum(-1).square())
        wpower = ((column * torch.cos(harmonic * qw)).sum(-1).square()
                  + (column * torch.sin(harmonic * qw)).sum(-1).square())
        per_configuration.append((hpower + wpower) / (2 * volume_squared))
    diagonal = spins.new_zeros(spins.shape[:2])
    for sign in (1, -1):
        phase = qh[:, None] + sign * qw[None, :]
        diagonal = diagonal + ((spins * torch.cos(phase)).sum((-2, -1)).square()
                               + (spins * torch.sin(phase)).sum((-2, -1)).square())
    per_configuration.append(diagonal / (2 * volume_squared))
    return torch.stack(per_configuration, dim=-1).mean(dim=1)


@torch.no_grad()
def fit_future_normalizer(targets: torch.Tensor, *, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit population moments to TRAIN [ensembles, paths, observables] only.

    The caller must enforce actual split provenance.  This explicit split
    argument additionally rejects accidental validation/TEST fitting; returned
    moments are detached snapshots, not online moments of a changing C.
    """
    if split.upper() != "TRAIN":
        raise ValueError("Future-physics normalization may only be fitted on TRAIN")
    if targets.ndim != 3 or min(targets.shape) < 1 or targets.shape[0] < 2:
        raise ValueError("Normalizer targets must have shape [N>=2,paths,observables]")
    if not targets.is_floating_point() or not bool(torch.isfinite(targets).all()):
        raise ValueError("Normalizer targets must be finite floating-point tensors")
    frozen = targets.detach()
    return frozen.mean(0).clone(), frozen.std(0, unbiased=False).clone()


class FuturePhysicsPredictor(nn.Module):
    """One auxiliary tanh head per path with frozen TRAIN target moments.

    Heads predict standardized targets.  They do not share parameters with
    R, do not perform coarse graining, and do not access physical labels.
    Their state_dict includes moments and the exact path/configuration
    contract, allowing a separate auxiliary checkpoint without changing QCR.
    """

    def __init__(
        self,
        theory_dim: int,
        *,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        paths: Sequence[tuple[int, int]] = DEFAULT_FUTURE_PATHS,
        hidden: int = 32,
        std_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(theory_dim, int) or theory_dim < 1 or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("Theory dimension and hidden width must be positive integers")
        self.paths = tuple(tuple(path) for path in paths)
        if (not self.paths or len(set(self.paths)) != len(self.paths)
                or any(len(p) != 2 or any(not isinstance(i, int) for i in p)
                       or not 0 <= p[0] < p[1] for p in self.paths)):
            raise ValueError("Future paths must be unique nonnegative source<target pairs")
        if not math.isfinite(std_floor) or std_floor <= 0:
            raise ValueError("Standard deviation floor must be finite and positive")
        mean = (torch.as_tensor(target_mean) if isinstance(target_mean, torch.Tensor)
                else torch.as_tensor(target_mean, dtype=torch.float64)).detach().clone()
        std = torch.as_tensor(target_std).detach().clone().to(device=mean.device, dtype=mean.dtype)
        if (mean.ndim != 2 or mean.shape != std.shape or mean.shape[0] != len(self.paths)
                or mean.shape[1] < 1 or not mean.is_floating_point()):
            raise ValueError("Target moments must be floating [paths,observables] tensors")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std < 0).any()):
            raise ValueError("Target moments must be finite with nonnegative standard deviations")
        self.theory_dim, self.hidden = theory_dim, hidden
        self.observable_count, self.std_floor = mean.shape[1], float(std_floor)
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std.clamp_min(std_floor))
        self.register_buffer("target_std_unfloored", std)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(theory_dim, hidden), nn.Tanh(), nn.Linear(hidden, self.observable_count))
            for _ in self.paths
        ])
        self.to(device=mean.device, dtype=mean.dtype)

    def get_extra_state(self) -> dict[str, Any]:
        return dict(version=1, paths=self.paths, theory_dim=self.theory_dim,
                    hidden=self.hidden, observable_count=self.observable_count, std_floor=self.std_floor)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        expected = self.get_extra_state()
        if not isinstance(state, dict) or state != expected:
            raise ValueError("Future-physics checkpoint configuration/path contract does not match")

    def forward(self, encodings: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Predict standardized values from [B,level,d] or a level list."""
        if isinstance(encodings, torch.Tensor):
            if encodings.ndim != 3:
                raise ValueError("Encodings tensor must have shape [B,level,theory_dim]")
            encodings = list(encodings.unbind(dim=1))
        outputs = []
        batch_size = None
        for head, (source, _) in zip(self.heads, self.paths):
            if source >= len(encodings):
                raise ValueError("A source scale is absent from encodings")
            value = encodings[source]
            if value.ndim != 2 or value.shape[-1] != self.theory_dim:
                raise ValueError("Each encoding must have shape [B,theory_dim]")
            if batch_size is not None and value.shape[0] != batch_size:
                raise ValueError("Every source scale must have the same ensemble count")
            batch_size = value.shape[0]
            outputs.append(head(value))
        return torch.stack(outputs, dim=1)

    def standardize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim != 3 or targets.shape[1:] != self.target_mean.shape:
            raise ValueError("Targets must have shape [B,paths,observables]")
        return (targets.detach() - self.target_mean) / self.target_std

    def physical_predictions(self, encodings: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self(encodings) * self.target_std + self.target_mean


def _metadata_list(values: Sequence[Any] | torch.Tensor, name: str, count: int) -> list[Any]:
    result = values.detach().cpu().tolist() if isinstance(values, torch.Tensor) else list(values)
    if len(result) != count:
        raise ValueError(f"{name} must identify every physical ensemble")
    if any(isinstance(v, (list, dict, set)) or v is None for v in result):
        raise ValueError(f"{name} must contain non-null scalar identifiers")
    return result


def validate_independent_pairs(
    partner_indices: Sequence[int] | torch.Tensor,
    condition_ids: Sequence[Any] | torch.Tensor,
    chain_ids: Sequence[Any] | torch.Tensor,
    count: int,
) -> list[int]:
    """Validate same-condition, distinct-chain pairing without coupling labels.

    IDs are provenance only: numeric values never enter the loss or a head.
    Distinct IDs cannot establish independence by themselves; the data receipt
    must certify that each ID denotes an independently generated chain.
    """
    partners = _metadata_list(partner_indices, "partner_indices", count)
    conditions = _metadata_list(condition_ids, "condition_ids", count)
    chains = _metadata_list(chain_ids, "chain_ids", count)
    for source, target in enumerate(partners):
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < count:
            raise ValueError("Partner indices must be in-batch integer indices")
        if source == target or chains[source] == chains[target]:
            raise ValueError("Future-physics targets require a distinct independent physical chain")
        if conditions[source] != conditions[target]:
            raise ValueError("Each independent pair must share the same physical condition")
    return partners


def paired_future_physics_loss(
    predictor: FuturePhysicsPredictor,
    encoder: nn.Module,
    levels: Sequence[torch.Tensor],
    partner_indices: Sequence[int] | torch.Tensor,
    condition_ids: Sequence[Any] | torch.Tensor,
    chain_ids: Sequence[Any] | torch.Tensor,
    *,
    weight: float = 1.0,
    observable_fn: Callable[[torch.Tensor], torch.Tensor] = binary2d_future_observables,
    encoding_batch_chunk: int | None = None,
    validated_binary_inputs: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fresh Q graph on existing detached levels; no C/R or target gradients.

    Average standardized squared error equally across ensembles, paths and
    observable coordinates.  Return weighted loss and unweighted diagnostics.
    A zero weight takes no new forward pass or RNG draw and bypasses all
    pairing/observable work; the training driver should not create the
    optional predictor at all in its default weight-zero path.
    Optional execution chunking partitions only the independent ensemble axis,
    never a chain's sample axis.  Encoders must act independently per ensemble;
    the Q encoder's shared-table forward_many retains a single fresh Q graph.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Future-physics weight must be finite and nonnegative")
    if weight == 0:
        zero = next(predictor.parameters()).reshape(-1)[0] * 0
        return zero, {"future_physics": zero.detach(), "future_physics_nrmse": zero.detach()}
    if encoding_batch_chunk is not None and (type(encoding_batch_chunk) is not int or encoding_batch_chunk < 1):
        raise ValueError("Future encoding batch chunk must be a positive integer or None")
    if type(validated_binary_inputs) is not bool:
        raise ValueError("validated_binary_inputs must be boolean")
    if not levels or max(target for _, target in predictor.paths) >= len(levels):
        raise ValueError("A future target scale is absent from the existing trajectory")
    count = levels[0].shape[0]
    if count < 2 or any(level.shape[0] != count for level in levels):
        raise ValueError("Levels must share at least two physical ensembles")
    partners = validate_independent_pairs(partner_indices, condition_ids, chain_ids, count)
    indices = torch.tensor(partners, dtype=torch.long, device=levels[0].device)
    sources = {source for source, _ in predictor.paths}
    # No C graph is reused here, and no new stochastic coarse-graining occurs.
    if encoding_batch_chunk is None:
        # Keep the original default path, including its exact call layout.
        encodings = [encoder(level.detach()) if scale in sources else None
                     for scale, level in enumerate(levels)]
    else:
        chunks, spans = [], {}
        for source in sorted(sources):
            start = len(chunks)
            chunks.extend(levels[source].detach().split(encoding_batch_chunk, dim=0))
            spans[source] = (start, len(chunks))
        forward_many = getattr(encoder, "forward_many", None)
        encoded_chunks = (forward_many(chunks, validated_binary_inputs=validated_binary_inputs)
                          if callable(forward_many) else [encoder(chunk) for chunk in chunks])
        encodings = [None] * len(levels)
        for source, (start, stop) in spans.items():
            encodings[source] = torch.cat(encoded_chunks[start:stop], dim=0)
    predictions = predictor(encodings)
    with torch.no_grad():
        observed = {target: observable_fn(levels[target].detach())
                    for target in {target for _, target in predictor.paths}}
        targets = torch.stack([observed[target].index_select(0, indices)
                               for _, target in predictor.paths], dim=1)
        standardized = predictor.standardize_targets(targets)
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(standardized).all()):
        raise FloatingPointError("Nonfinite future-physics prediction or target")
    path_mse = (predictions - standardized).square().mean(dim=(0, 2))
    loss = path_mse.mean()
    terms = {"future_physics": loss, "future_physics_nrmse": loss.sqrt()}
    terms.update({f"future_physics_path_{source}_{target}": path_mse[index]
                  for index, (source, target) in enumerate(predictor.paths)})
    # Physical-unit errors remain interpretable as trainable C changes while
    # the initial-TRAIN normalization stays frozen.  Reporting does not add
    # another objective or retain an auxiliary autograd graph.
    with torch.no_grad():
        physical_predictions = predictions.detach() * predictor.target_std + predictor.target_mean
        raw_path_mse = (physical_predictions - targets).square().mean(dim=(0, 2))
        raw_mse = raw_path_mse.mean()
        terms.update(future_physics_raw_mse=raw_mse, future_physics_raw_rmse=raw_mse.sqrt())
        terms.update({f"future_physics_raw_path_{source}_{target}": raw_path_mse[index]
                      for index, (source, target) in enumerate(predictor.paths)})
    return weight * loss, terms
