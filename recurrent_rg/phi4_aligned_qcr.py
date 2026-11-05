"""Raw-patch continuous QCR with a joint local reconstruction target.

This is a new, explicit continuous-field recipe, not a migration of historical
checkpoints.  Q has the same learned local-patch/pooling architecture as the
binary encoder.  Its reconstruction target is the *joint* soft distribution
of four plaquette amplitudes (5**4 bins), rather than separate one-/two-point
statistics.  It is still a finite soft descriptor, not a sufficient statistic
of an arbitrary continuous probability distribution.

C deliberately retains the historical continuous Gaussian kernel, smoothed
full-shell information proxy, and uniform-amplitude penalty.  These are not an
exact empirical Shannon MI or a binary color-preservation NLL.  No physical
parameter or phase label enters this module.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .losses import all_start_anchor_closure_loss
from .phi4_qcr import (
    GaussianCoarseGrainer,
    _field,
    _positive,
    gaussian_bin_probabilities,
    sample_regions,
    smoothed_information,
)


FEATURE_DIM = 5 ** 4
PATCH_OFFSETS = tuple((i, j) for i in range(-1, 2) for j in range(-1, 2))
# This ordering agrees with the discrete plaquette_distribution bit ordering.
PLAQUETTE_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _scale_tensor(scale: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be scalar")
        result = scale.to(dtype=x.dtype, device=x.device)
        if result.device.type == "cpu":
            _positive(result.detach().item(), "scale")
        return result
    return x.new_tensor(_positive(scale, "scale"))


def _chunk_amplitudes(flat: torch.Tensor, start: int, end: int, length: int,
                      offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Gather periodic sites without creating the full site-by-patch tensor."""
    observation = torch.arange(start, end, device=flat.device)
    sample = torch.div(observation, length * length, rounding_mode="floor")
    row = torch.div(observation, length, rounding_mode="floor") % length
    column = observation % length
    index = torch.stack([
        sample * length * length + (row + i) % length * length + (column + j) % length
        for i, j in offsets
    ], dim=-1)
    return flat[:, index]


def soft_plaquette_features(x: torch.Tensor, scale: float | torch.Tensor,
                            *, chunk_size: int = 4096) -> torch.Tensor:
    """Pooled, complete four-site soft joint probabilities, shape [E,625].

    Five Gaussian soft bins use centers [-2,-1,0,1,2] and width .5 in units of
    one frozen TRAIN microscopic RMS.  They cover every finite amplitude by
    soft assignment, with no clipping or sign discretization.  Amplitudes are
    ordered (00,10,01,11), the first index varying slowest in the flat output.
    Each ensemble is pooled over samples and periodic translations only.

    The chunk limit counts site observations across all ensembles (unless E
    itself exceeds the limit).  Two 25-bin factors are contracted directly
    into a 25x25 accumulator: no [E,N,L,L,625] object is materialized.  Input
    derivatives are supported; training objectives explicitly detach targets.
    """
    _field(x)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    scale_tensor = _scale_tensor(scale, x)
    ensembles, samples, length, _, _ = x.shape
    observations = samples * length * length
    per_ensemble = max(1, chunk_size // ensembles)
    flat = x[..., 0].reshape(ensembles, observations)
    centers = torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    accumulator = x.new_zeros((ensembles, 25, 25))
    for start in range(0, observations, per_ensemble):
        amplitude = _chunk_amplitudes(
            flat, start, min(start + per_ensemble, observations), length,
            PLAQUETTE_OFFSETS)
        # Cancel the common -2*x**2 term algebraically.  Subtracting the
        # extreme-bin logit first also keeps very large finite tails stable.
        anchor = torch.where(amplitude >= 0, amplitude.new_tensor(2.), amplitude.new_tensor(-2.))
        difference = centers - anchor[..., None]
        # Multiply finite physical amplitudes by the bin difference before
        # dividing by scale.  The anchor difference is exactly zero, so even
        # finfo.max amplitudes cannot create inf*0 at the winning tail bin.
        logits = (amplitude[..., None] * (4. * difference) / scale_tensor
                  - 2. * (centers.square() - anchor[..., None].square()))
        probability = torch.softmax(logits, dim=-1)
        first = (probability[:, :, 0, :, None]
                 * probability[:, :, 1, None, :]).flatten(-2)
        second = (probability[:, :, 2, :, None]
                  * probability[:, :, 3, None, :]).flatten(-2)
        accumulator = accumulator + torch.bmm(first.transpose(1, 2), second)
    return accumulator.flatten(1) / observations


class RawPatchTheoryEncoder(nn.Module):
    """Continuous raw-patch Q with the discrete encoder's local architecture.

    Physical amplitudes are normalized only by the frozen TRAIN RMS.  Chunked
    non-reentrant activation checkpointing recomputes local patches/MLP calls
    in backward, instead of retaining every site's hidden activations.  The
    decoder is raw affine during forward; its columns are projected after an
    optimizer update, matching the discrete affine-reconstruction convention.
    """

    def __init__(self, scale: float, hidden: int = 96, theory_dim: int = 1,
                 local_features: int = 12, patch_chunk_size: int = 4096):
        super().__init__()
        hidden = _positive_integer(hidden, "hidden")
        theory_dim = _positive_integer(theory_dim, "theory_dim")
        local_features = _positive_integer(local_features, "local_features")
        self.patch_chunk_size = _positive_integer(patch_chunk_size, "patch_chunk_size")
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.local_mlp = nn.Sequential(
            nn.Linear(9, hidden), nn.Tanh(),
            nn.Linear(hidden, local_features), nn.Tanh(),
            nn.Linear(local_features, local_features), nn.Tanh(),
        ).double()
        self.head = nn.Sequential(
            nn.Linear(local_features, hidden), nn.Tanh(), nn.Linear(hidden, theory_dim)
        ).double()
        self.decoder = nn.Linear(theory_dim, FEATURE_DIM).double()
        self.normalize_decoder_()

    def _chunk_sum(self, flat: torch.Tensor, start: int, end: int,
                   length: int) -> torch.Tensor:
        patches = _chunk_amplitudes(flat, start, end, length, PATCH_OFFSETS)
        return self.local_mlp(patches / self.scale).sum(dim=1)

    def pooled_local_features(self, x: torch.Tensor) -> torch.Tensor:
        _field(x)
        ensembles, samples, length, _, _ = x.shape
        observations = samples * length * length
        flat = x[..., 0].reshape(ensembles, observations)
        per_ensemble = max(1, self.patch_chunk_size // ensembles)
        pooled = None
        for start in range(0, observations, per_ensemble):
            end = min(start + per_ensemble, observations)
            if torch.is_grad_enabled():
                value = checkpoint(self._chunk_sum, flat, start, end, length,
                                   use_reentrant=False)
            else:
                value = self._chunk_sum(flat, start, end, length)
            pooled = value if pooled is None else pooled + value
        return pooled / observations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled_local_features(x))

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return self.decoder(g)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


class _Block2GaussianCoarseGrainer(GaussianCoarseGrainer):
    """Explicit b=2 geometry ablation of the continuous Gaussian C.

    Only the first MLP layer changes from nine to four inputs.  Noise, hidden
    capacity, amplitude units, and reparameterized sampling retain the b=3
    recipe.  The historical GaussianCoarseGrainer itself is left untouched.
    """

    def __init__(self, scale: float, hidden: int, block_chunk_size: int = 65536):
        nn.Module.__init__(self)
        self.block_chunk_size = _positive_integer(block_chunk_size, "block_chunk_size")
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.register_buffer("noise_ratio", torch.tensor(.1, dtype=torch.double))
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
            nn.Tanh(), nn.Linear(hidden, 1),
        ).double()

    def mean_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim < 1 or blocks.shape[-1] != 4:
            raise ValueError("C.mean_blocks expects [...,4] block amplitudes")
        # Bound hidden activations across *all* ensemble/configuration blocks.
        # In particular, E=32,N=64,L=128 must not materialize 8M x hidden.
        # Sampling is deliberately outside this method: inherited forward()
        # draws one full-shape noise tensor after all means are reassembled.
        flat = blocks.reshape(-1, 4)
        chunks = []
        for start in range(0, len(flat), self.block_chunk_size):
            part = flat[start:start + self.block_chunk_size]
            if torch.is_grad_enabled():
                value = checkpoint(self._mean_block_chunk, part, use_reentrant=False)
            else:
                value = self._mean_block_chunk(part)
            chunks.append(value)
        return torch.cat(chunks, dim=0).reshape(*blocks.shape[:-1], 1)

    def _mean_block_chunk(self, blocks: torch.Tensor) -> torch.Tensor:
        return self.scale * self.net(blocks / self.scale)

    def mean_field(self, x: torch.Tensor) -> torch.Tensor:
        _field(x)
        ensembles, samples, length, _, _ = x.shape
        if length % 2:
            raise ValueError("L must be divisible by the coarse-graining factor 2")
        coarse = length // 2
        blocks = (x[..., 0].reshape(ensembles, samples, coarse, 2, coarse, 2)
                  .permute(0, 1, 2, 4, 3, 5)
                  .reshape(ensembles, samples, coarse, coarse, 4))
        return self.mean_blocks(blocks)


def _sample_aligned_regions(x: torch.Tensor, pairs: int,
                            generator: torch.Generator | None,
                            block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete microscopic outer shell, separated by one buffer site.

    The unchanged b=3 branch delegates to the historical 3x3 / 24-site
    implementation.  A b=2 block uses offsets (0..1,0..1); its complete 6x6
    outer perimeter has 20 sites at offsets -2..3.  Neither branch samples
    environments at the progressively coarser scales.
    """
    if block_size == 3:
        return sample_regions(x, pairs, generator)
    _field(x)
    if type(pairs) is not int or pairs < 2:
        raise ValueError("At least two region pairs are required for leave-one-out smoothing")
    ensembles, samples, length, _, _ = x.shape
    if length < 6:
        raise ValueError("The full buffered 6 x 6 shell requires L >= 6")
    configs = torch.randint(samples, (ensembles, pairs), device=x.device, generator=generator)
    row = torch.randint(length, (ensembles, pairs), device=x.device, generator=generator)
    column = torch.randint(length, (ensembles, pairs), device=x.device, generator=generator)
    ensemble = torch.arange(ensembles, device=x.device)[:, None, None]

    def gather(offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
        offset = torch.tensor(offsets, device=x.device)
        return x[ensemble, configs[..., None],
                 (row[..., None] + offset[:, 0]) % length,
                 (column[..., None] + offset[:, 1]) % length, 0]

    block = tuple((i, j) for i in range(2) for j in range(2))
    shell = tuple((i, j) for i in range(-2, 4) for j in range(-2, 4)
                  if i in (-2, 3) or j in (-2, 3))
    return gather(block), gather(shell)


def _aligned_information(mean: torch.Tensor, environment: torch.Tensor,
                         scale: torch.Tensor, noise_ratio: float,
                         block_size: int) -> torch.Tensor:
    """Same smoothed-information proxy, with full-shell geometry respected."""
    if block_size == 3:
        return smoothed_information(mean, environment, scale, noise_ratio=noise_ratio)
    if (mean.ndim != 2 or environment.ndim != 3
            or environment.shape[:2] != mean.shape or environment.shape[-1] != 20
            or mean.shape[1] < 2 or mean.shape[0] < 1):
        raise ValueError("Expected means [E,P] and full-shell environments [E,P,20], P >= 2")
    # The average squared distance remains per environment site: changing
    # shell cardinality does not silently multiply the kernel bandwidth.
    env = environment.detach() / scale
    norms = env.square().sum(-1)
    distance = (norms[..., :, None] + norms[..., None, :]
                - 2. * torch.bmm(env, env.transpose(1, 2))).clamp_min(0.) / 20.
    diagonal = torch.eye(mean.shape[1], dtype=torch.bool, device=mean.device)[None]
    weights = torch.softmax((-.5 * distance).masked_fill(diagonal, -torch.inf), dim=-1)
    probability = gaussian_bin_probabilities(mean, scale, noise_ratio)
    conditional = torch.bmm(weights, probability)
    marginal = conditional.mean(dim=1, keepdim=True)
    tiny = torch.finfo(mean.dtype).tiny
    return (conditional * (conditional.clamp_min(tiny).log()
                           - marginal.clamp_min(tiny).log())).sum(-1).mean()


class AlignedPhi4QCR(nn.Module):
    """Continuous recipe, default b=3 / two transitions, with explicit b=2.

    ``block_size=2`` is a requested geometry ablation, never an implicit new
    default.  Q still uses overlapping 3x3 patches and the full 625-component
    soft plaquette target, independent of C's block size and chain depth.
    Geometry is constructor metadata, not extra checkpoint tensors; historical
    state dicts therefore retain their exact keys and shapes for default b=3.
    """

    def __init__(self, scale: float, hidden: int = 96, theory_dim: int = 1,
                 local_features: int = 12, patch_chunk_size: int = 4096,
                 *, block_size: int = 3, steps: int = 2):
        super().__init__()
        hidden = _positive_integer(hidden, "hidden")
        theory_dim = _positive_integer(theory_dim, "theory_dim")
        local_features = _positive_integer(local_features, "local_features")
        patch_chunk_size = _positive_integer(patch_chunk_size, "patch_chunk_size")
        if type(block_size) is not int or block_size not in (2, 3):
            raise ValueError("block_size must be the explicitly supported integer 2 or 3")
        self.block_size = block_size
        self.steps = _positive_integer(steps, "steps")
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.C = (GaussianCoarseGrainer(scale, hidden) if block_size == 3
                  else _Block2GaussianCoarseGrainer(scale, hidden))
        self.Q = RawPatchTheoryEncoder(scale, hidden, theory_dim, local_features, patch_chunk_size)
        self.R = nn.Sequential(
            nn.Linear(theory_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
            nn.Tanh(), nn.Linear(hidden, theory_dim),
        ).double()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return soft_plaquette_features(x, self.scale, chunk_size=self.Q.patch_chunk_size)

    def advance(self, g: torch.Tensor) -> torch.Tensor:
        return g + self.R(g)

    def advance_theory(self, g: torch.Tensor) -> torch.Tensor:
        return self.advance(g)

    def coarse_grain(self, x: torch.Tensor,
                     generator: torch.Generator | None = None) -> list[torch.Tensor]:
        _field(x)
        divisor = self.block_size ** self.steps
        if x.shape[2] % divisor:
            if self.block_size == 3 and self.steps == 2:
                raise ValueError("Two b=3 transitions require microscopic L divisible by 9")
            raise ValueError(f"{self.steps} b={self.block_size} transitions require "
                             f"microscopic L divisible by {divisor}")
        levels = [x]
        for _ in range(self.steps):
            levels.append(self.C(levels[-1], generator=generator))
        return levels


def _square_l2(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(-1).mean()


def aligned_objectives(model: AlignedPhi4QCR, x: torch.Tensor,
                       generator: torch.Generator | None = None,
                       information_pairs: int = 128,
                       closure_targets_sink: list[torch.Tensor] | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Label-free Q/C/R objectives with strictly independent gradient graphs.

    Q receives detached physical fields, reconstructing their full soft joint
    plaquette descriptor with weights 10/20/5.  R receives detached direct
    encodings and the shared all-start objective, combined outer weight 10.
    C receives its unchanged continuous information/variance/uniform objective
    only.  A single ancestral C trajectory supplies all objectives.
    """
    _field(x)
    if x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("Two equal disjoint subsets require an even sample count >= 2")
    levels = model.coarse_grain(x, generator=generator)
    half = x.shape[1] // 2
    direct, reconstruction, subset = [], [], []
    for level in levels:
        field = level.detach()
        first, second = model.Q(field[:, :half]), model.Q(field[:, half:])
        g = .5 * (first + second)
        with torch.no_grad():
            target = model.features(field)
        direct.append(g.detach())
        reconstruction.append(_square_l2(model.Q.decode(g) - target))
        subset.append(_square_l2(first - second))
    reconstruction_loss = torch.stack(reconstruction).mean()
    coarse_subset = torch.stack(subset[1:]).mean()
    q_loss = 10. * reconstruction_loss + 20. * subset[0] + 5. * coarse_subset
    if closure_targets_sink is not None:
        closure_targets_sink.append(torch.stack(direct, dim=1).detach())
    closure_loss, closure_targets, closure_pairs = all_start_anchor_closure_loss(
        model, torch.stack(direct, dim=1), coordinate_reduction="sum")
    r_loss = 10. * closure_loss

    blocks, environment = _sample_aligned_regions(
        x.detach(), information_pairs, generator, model.block_size)
    means = model.C.mean_blocks(blocks)[..., 0]
    information = _aligned_information(
        means, environment, model.scale, float(model.C.noise_ratio), model.block_size)
    variance_terms = []
    for level in levels[:-1]:
        mean_output = model.C.mean_field(level)
        input_std = level.detach().flatten(1).std(dim=1, unbiased=False)
        output_std = mean_output.flatten(1).std(dim=1, unbiased=False)
        variance_terms.append(((input_std - output_std).clamp_min(0.) / model.scale).square().mean())
    variance_floor = torch.stack(variance_terms).mean()
    amplitudes = model.scale * torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    probes = amplitudes[:, None].expand(5, model.block_size ** 2)
    uniform_amplitude = .5 * ((model.C.mean_blocks(probes)[:, 0] - amplitudes) / model.scale).square().mean()
    c_loss = -2. * information + 2. * variance_floor + uniform_amplitude

    terms = {
        "q_total": q_loss, "c_total": c_loss, "r_total": r_loss,
        "q_reconstruction": reconstruction_loss,
        "q_subset_micro": subset[0], "q_subset_coarse": coarse_subset,
        "c_information": information, "c_variance": variance_floor,
        "c_uniform": uniform_amplitude, "closure_all_start": closure_loss,
        "closure_one_step": torch.stack([
            closure_pairs[index, index + 1] for index in range(model.steps)]).mean(),
        "closure_rollout": torch.stack([
            closure_pairs[0, target] for target in range(1, model.steps + 1)]).mean(),
    }
    for index in range(len(levels)):
        terms[f"reconstruction_level_{index}"] = reconstruction[index]
        terms[f"subset_level_{index}"] = subset[index]
    for target, value in enumerate(closure_targets, start=1):
        terms[f"closure_target_{target}"] = value
    for (start, target), value in closure_pairs.items():
        terms[f"closure_start_{start}_target_{target}"] = value
    values = torch.stack(list(terms.values())).detach().cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite aligned continuous QCR objective")
    return q_loss, c_loss, r_loss, dict(zip(terms, values))
