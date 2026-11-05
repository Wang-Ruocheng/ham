"""Explicit b=2, three-transition continuous-field closure ablation.

This independent experimental model preserves the historical continuous-field
feature recipe, not the binary-spin Shannon-MI recipe. Its full buffered shell
objective is a leave-one-out Gaussian-kernel information *proxy*, not exact
Shannon mutual information or InfoNCE. No coupling or phase label enters it.
Neither this module nor the ablation changes the standard b=3 defaults.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from recurrent_rg.phi4_qcr import (
    FEATURE_DIM, ContinuousTheoryEncoder, _field, _positive,
    gaussian_bin_probabilities, soft_local_features,
)


BLOCK_OFFSETS_B2 = tuple((i, j) for i in range(2) for j in range(2))
SHELL_OFFSETS_B2 = tuple(
    (i, j) for i in range(-2, 4) for j in range(-2, 4)
    if i in (-2, 3) or j in (-2, 3)
)
CLOSURE_MODES = ("one_step_and_rollout", "all_start_anchor")


def block_vectors_b2(x: torch.Tensor) -> torch.Tensor:
    """Nonoverlapping row-major 2 x 2 blocks with stride two."""
    _field(x)
    ensembles, samples, length, _, _ = x.shape
    if length % 2:
        raise ValueError("L must be divisible by the coarse-graining factor 2")
    coarse = length // 2
    return (x[..., 0].reshape(ensembles, samples, coarse, 2, coarse, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(ensembles, samples, coarse, coarse, 4))


class GaussianCoarseGrainerB2(nn.Module):
    """Random 4 -> hidden -> hidden -> 1 Gaussian probabilistic kernel.

    Input and output share one frozen microscopic TRAIN RMS. Output noise has
    a fixed standard deviation of ``noise_ratio * scale``. There is no kernel
    selection, pretraining, clipping, sign map, or trainable noise variance.
    """

    def __init__(self, scale: float, hidden: int = 96, noise_ratio: float = .1):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.register_buffer("noise_ratio", torch.tensor(_positive(noise_ratio, "noise_ratio"), dtype=torch.double))
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1)).double()

    @property
    def noise_std(self) -> torch.Tensor:
        return self.scale * self.noise_ratio

    def mean_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim < 1 or blocks.shape[-1] != 4:
            raise ValueError("C.mean_blocks expects [...,4] block amplitudes")
        return self.scale * self.net(blocks / self.scale)

    def mean_field(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_blocks(block_vectors_b2(x))

    def forward(self, x: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        mean = self.mean_field(x)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                            generator=generator)
        return mean + self.noise_std * noise


class Phi4B2QCR(nn.Module):
    """Fresh random C/Q/R model with L -> L/2 -> L/4 -> L/8 geometry."""

    def __init__(self, scale: float, hidden: int = 96, theory_dim: int = 1):
        super().__init__()
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        # Construction order is part of this independent experiment's seed
        # contract and must be identical in the two closure arms.
        self.C = GaussianCoarseGrainerB2(scale, hidden)
        self.Q = ContinuousTheoryEncoder(hidden, theory_dim)
        self.R = nn.Sequential(nn.Linear(theory_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, theory_dim)).double()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return soft_local_features(x, self.scale)

    def advance(self, g: torch.Tensor) -> torch.Tensor:
        return g + self.R(g)

    def coarse_grain(self, x: torch.Tensor,
                     generator: torch.Generator | None = None) -> list[torch.Tensor]:
        _field(x)
        if x.shape[2] % 8:
            raise ValueError("Three b=2 transitions require microscopic L divisible by 8")
        levels = [x]
        for _ in range(3):
            levels.append(self.C(levels[-1], generator=generator))
        return levels


def sample_regions_b2(x: torch.Tensor, pairs: int = 128,
                      generator: torch.Generator | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample V=2 x 2 and all 20 sites of its one-buffer 6 x 6 shell.

    Coordinates [-2,3] are relative to V's top-left corner. The shell is its
    complete perimeter; the intervening ring [-1,2] is the width-one buffer.
    Origins/configurations are sampled independently per ensemble with
    replacement, under periodic boundaries. No shell sites are subsampled.
    """
    _field(x)
    if not isinstance(pairs, int) or pairs < 2:
        raise ValueError("At least two region pairs are required for leave-one-out smoothing")
    ensembles, samples, length, _, _ = x.shape
    if length < 6:
        raise ValueError("The full buffered 6 x 6 shell requires L >= 6")
    configurations = torch.randint(samples, (ensembles, pairs), device=x.device, generator=generator)
    row = torch.randint(length, (ensembles, pairs), device=x.device, generator=generator)
    column = torch.randint(length, (ensembles, pairs), device=x.device, generator=generator)
    ensemble = torch.arange(ensembles, device=x.device)[:, None, None]

    def gather(offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
        offset = torch.tensor(offsets, device=x.device)
        return x[ensemble, configurations[..., None],
                 (row[..., None] + offset[:, 0]) % length,
                 (column[..., None] + offset[:, 1]) % length, 0]

    return gather(BLOCK_OFFSETS_B2), gather(SHELL_OFFSETS_B2)


def smoothed_information_b2(mean: torch.Tensor, environment: torch.Tensor,
                            scale: float | torch.Tensor, noise_ratio: float = .1,
                            bins: int = 16, bandwidth: float = 1.) -> torch.Tensor:
    """Leave-one-out full-20-site shell Gaussian information proxy in nats.

    Environment and scale are detached. Distances use the mean squared
    difference over *all* twenty environment amplitudes, with bandwidth one
    by default. A softmax with the diagonal masked out normalizes each row.
    The finite-sample kernel-smoothed KL value is not exact Shannon MI.
    Only the block means receive gradients; no sampled output noise is used.
    """
    bandwidth = _positive(bandwidth, "bandwidth")
    if mean.ndim == 3 and mean.shape[-1] == 1:
        mean = mean[..., 0]
    if (mean.ndim != 2 or environment.ndim != 3
            or environment.shape[:2] != mean.shape
            or environment.shape[-1] != 20
            or mean.shape[1] < 2 or mean.shape[0] < 1):
        raise ValueError("Expected means [E,P] and full-shell environments [E,P,20], P >= 2")
    scale_value = float(scale.detach().item()) if isinstance(scale, torch.Tensor) else float(scale)
    scale_value = _positive(scale_value, "scale")
    env = environment.detach() / scale_value
    norms = env.square().sum(-1)
    distance = (norms[..., :, None] + norms[..., None, :]
                - 2. * torch.bmm(env, env.transpose(1, 2))).clamp_min(0.) / 20.
    logits = -distance / (2. * bandwidth ** 2)
    diagonal = torch.eye(mean.shape[1], dtype=torch.bool, device=mean.device)[None]
    weights = torch.softmax(logits.masked_fill(diagonal, -torch.inf), dim=-1)
    probability = gaussian_bin_probabilities(mean, scale_value, noise_ratio, bins)
    conditional = torch.bmm(weights, probability)
    marginal = conditional.mean(dim=1, keepdim=True)
    tiny = torch.finfo(mean.dtype).tiny
    log_ratio = conditional.clamp_min(tiny).log() - marginal.clamp_min(tiny).log()
    return (conditional * log_ratio).sum(-1).mean()


def _square_l2(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(-1).mean()


def continuous_objectives_b2(model: Phi4B2QCR, x: torch.Tensor,
                             generator: torch.Generator | None = None,
                             information_pairs: int = 128,
                             closure_mode: str = "all_start_anchor"
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Three strictly independent objectives, changing only R by closure mode.

    Q receives four-scale affine reconstruction and subset consistency on
    detached coarse fields. R receives detached direct g and retains every
    recurrent R derivative. C receives its microscopic full-shell information
    proxy, three-transition mean-output variance floor, and five constant-block
    amplitude probes. Gaussian output noise cannot satisfy the variance floor.

    Legacy: 5 * mean(adjacent errors) + 5 * mean(microscopic rollout errors).
    All-start: (10/3) * sum_t mean_{s<t}(error from s to t), t=1,2,3.
    ``all_start_anchor`` and ``closure_all_start_anchor`` metrics are the full
    weight-ten objective; one-step/rollout metrics are unweighted means.
    """
    if closure_mode not in CLOSURE_MODES:
        raise ValueError(f"Unknown closure_mode {closure_mode!r}; expected one of {CLOSURE_MODES}")
    _field(x)
    if x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("Two equal disjoint subsets require an even sample count >= 2")
    levels = model.coarse_grain(x, generator=generator)
    half = x.shape[1] // 2
    direct, reconstruction, subset = [], [], []
    for level in levels:
        field = level.detach()
        first_features = model.features(field[:, :half]).detach()
        second_features = model.features(field[:, half:]).detach()
        first = model.Q.from_features(first_features)
        second = model.Q.from_features(second_features)
        g = .5 * (first + second)
        direct.append(g)
        target = .5 * (first_features + second_features)
        reconstruction.append(_square_l2(model.Q.decode(g) - target))
        subset.append(_square_l2(first - second))
    reconstruction_loss = torch.stack(reconstruction).mean()
    coarse_subset = torch.stack(subset[1:]).mean()
    q_loss = 10. * reconstruction_loss + 20. * subset[0] + 5. * coarse_subset

    detached = [g.detach() for g in direct]
    errors: dict[tuple[int, int], torch.Tensor] = {}
    for start in range(3):
        prediction = detached[start]
        for target in range(start + 1, 4):
            prediction = model.advance(prediction)
            errors[start, target] = _square_l2(prediction - detached[target])
    one_step = [errors[index, index + 1] for index in range(3)]
    rollout = [errors[0, target] for target in range(1, 4)]
    one_step_loss = torch.stack(one_step).mean()
    rollout_loss = torch.stack(rollout).mean()
    all_start_by_target = [torch.stack([errors[start, target] for start in range(target)]).mean()
                           for target in range(1, 4)]
    all_start_loss = 10. * torch.stack(all_start_by_target).mean()
    r_loss = (all_start_loss if closure_mode == "all_start_anchor"
              else 5. * one_step_loss + 5. * rollout_loss)

    blocks, environment = sample_regions_b2(x.detach(), information_pairs, generator)
    means = model.C.mean_blocks(blocks)[..., 0]
    information = smoothed_information_b2(means, environment, model.scale,
                                          noise_ratio=float(model.C.noise_ratio))
    variance_terms = []
    for level in levels[:-1]:
        mean_output = model.C.mean_field(level)
        input_std = level.detach().flatten(1).std(dim=1, unbiased=False)
        output_std = mean_output.flatten(1).std(dim=1, unbiased=False)
        variance_terms.append(((input_std - output_std).clamp_min(0.) / model.scale).square().mean())
    variance_floor = torch.stack(variance_terms).mean()
    amplitudes = model.scale * torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    probes = amplitudes[:, None].expand(5, 4)
    uniform_amplitude = .5 * ((model.C.mean_blocks(probes)[:, 0] - amplitudes) / model.scale).square().mean()
    c_loss = -2. * information + 2. * variance_floor + uniform_amplitude

    terms = {
        "q_loss": q_loss, "c_loss": c_loss, "r_loss": r_loss,
        "reconstruction": reconstruction_loss, "subset_micro": subset[0],
        "subset_coarse": coarse_subset, "information_proxy": information,
        "variance_floor": variance_floor, "uniform_amplitude": uniform_amplitude,
        "closure_one_step": one_step_loss, "closure_rollout": rollout_loss,
        "all_start_anchor": all_start_loss, "closure_all_start_anchor": all_start_loss,
    }
    for index in range(4):
        terms[f"reconstruction_level_{index}"] = reconstruction[index]
        terms[f"subset_level_{index}"] = subset[index]
    for index in range(3):
        terms[f"closure_one_step_level_{index + 1}"] = one_step[index]
        terms[f"closure_rollout_level_{index + 1}"] = rollout[index]
        terms[f"closure_all_start_target_{index + 1}"] = all_start_by_target[index]
        terms[f"variance_floor_level_{index + 1}"] = variance_terms[index]
    for (start, target), error in errors.items():
        terms[f"closure_error_{start}_{target}"] = error
    scalar_values = torch.stack(list(terms.values())).detach().cpu().tolist()
    if not all(math.isfinite(value) for value in scalar_values):
        raise FloatingPointError("Non-finite continuous b=2 QCR objective")
    return q_loss, c_loss, r_loss, dict(zip(terms, scalar_values))
