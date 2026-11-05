"""Continuous-field QCR, kept separate from the finite-alphabet recipe.

This explicitly experimental recipe uses Gaussian stochastic coarse graining,
soft local histograms, and a kernel-smoothed full-shell information *proxy*.
The proxy is not an exact empirical Shannon MI or an InfoNCE objective.  Its
finite-sample/environment-kernel bias is not an equilibrium or RG certificate.
No action parameter, temperature, coupling, or phase label enters this module.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM = 5 + 4 * 25
PAIR_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
SHELL_OFFSETS = tuple(
    (i, j) for i in range(-2, 5) for j in range(-2, 5)
    if i in (-2, 4) or j in (-2, 4)
)
BLOCK_OFFSETS = tuple((i, j) for i in range(3) for j in range(3))


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _field(x: torch.Tensor) -> None:
    if (x.ndim != 5 or x.shape[-1] != 1 or x.shape[2] != x.shape[3]
            or min(x.shape[:4]) < 1):
        raise ValueError("Expected a nonempty square field [ensembles,samples,L,L,1]")
    if not x.is_floating_point():
        raise ValueError("Continuous fields must have floating-point dtype")


def block_vectors(x: torch.Tensor) -> torch.Tensor:
    """Nonoverlapping row-major 3 x 3 blocks, with stride 3."""
    _field(x)
    e, n, length, _, _ = x.shape
    if length % 3:
        raise ValueError("L must be divisible by the coarse-graining factor 3")
    coarse = length // 3
    return (x[..., 0].reshape(e, n, coarse, 3, coarse, 3)
            .permute(0, 1, 2, 4, 3, 5).reshape(e, n, coarse, coarse, 9))


class GaussianCoarseGrainer(nn.Module):
    """Random MLP mean with fixed reparameterized Gaussian output noise.

    Both input and output amplitudes use the single frozen TRAIN microscopic
    RMS ``scale``.  There is no selected/pretrained kernel, clipping, sign map,
    or trainable noise variance.  Gaussian samples remain continuous.
    """

    def __init__(self, scale: float, hidden: int = 96, noise_ratio: float = .1):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.register_buffer("noise_ratio", torch.tensor(_positive(noise_ratio, "noise_ratio"), dtype=torch.double))
        self.net = nn.Sequential(nn.Linear(9, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1)).double()

    @property
    def noise_std(self) -> torch.Tensor:
        return self.scale * self.noise_ratio

    def mean_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim < 1 or blocks.shape[-1] != 9:
            raise ValueError("C.mean_blocks expects [...,9] block amplitudes")
        return self.scale * self.net(blocks / self.scale)

    def mean_field(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_blocks(block_vectors(x))

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        mean = self.mean_field(x)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                            generator=generator)
        return mean + self.noise_std * noise


def soft_local_features(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """105 label-free soft motif probabilities for every physical ensemble.

    Five Gaussian soft bins have centers [-2,-1,0,1,2] times the frozen scale
    and width .5 times that scale.  The univariate distribution and each of
    four oriented nearest-neighbor pair distributions have unit mass.  Their
    concatenation is divided by five.  Samples and lattice positions, but not
    different physical ensembles, are pooled.  Input gradients are preserved;
    callers implementing independent Q/C routing must detach explicitly.
    """
    _field(x)
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("scale must be scalar")
        scale_tensor = scale.to(dtype=x.dtype, device=x.device)
        # Avoid a device synchronization in this hot path. Model buffers are
        # validated once on construction; standalone CPU calls remain checked.
        if scale_tensor.device.type == "cpu":
            _positive(scale_tensor.detach().item(), "scale")
    else:
        scale_tensor = x.new_tensor(_positive(scale, "scale"))
    centers = torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    standardized = x[..., 0] / scale_tensor
    probability = torch.softmax(-.5 * ((standardized[..., None] - centers) / .5).square(), dim=-1)
    univariate = probability.mean(dim=(1, 2, 3))
    groups = [univariate / univariate.sum(-1, keepdim=True)]
    for i, j in PAIR_OFFSETS:
        neighbor = torch.roll(probability, shifts=(-i, -j), dims=(2, 3))
        # einsum avoids materializing an E x N x L x L x 5 x 5 tensor.
        pair = torch.einsum("enija,enijb->eab", probability, neighbor)
        pair = pair.flatten(1)
        groups.append(pair / pair.sum(-1, keepdim=True))
    return torch.cat(groups, dim=-1) / 5.


class ContinuousTheoryEncoder(nn.Module):
    """Shared soft-feature encoder and column-normalized affine decoder."""

    def __init__(self, hidden: int = 96, theory_dim: int = 1):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not isinstance(theory_dim, int) or theory_dim < 1:
            raise ValueError("theory_dim must be a positive integer")
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 12), nn.Tanh(),
                                 nn.Linear(12, hidden), nn.Tanh(),
                                 nn.Linear(hidden, theory_dim)).double()
        self.decoder = nn.Linear(theory_dim, FEATURE_DIM).double()
        self.normalize_decoder_()

    def from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} continuous local statistics")
        return self.net(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.from_features(features)

    def normalized_decoder_weight(self) -> torch.Tensor:
        return F.normalize(self.decoder.weight, p=2, dim=0, eps=1e-15)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.copy_(self.normalized_decoder_weight())

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return F.linear(g, self.normalized_decoder_weight(), self.decoder.bias)


class Phi4QCR(nn.Module):
    """Separate continuous-field model; it does not alter binary defaults."""

    def __init__(self, scale: float, hidden: int = 96, theory_dim: int = 1):
        super().__init__()
        self.register_buffer("scale", torch.tensor(_positive(scale, "scale"), dtype=torch.double))
        self.C = GaussianCoarseGrainer(scale, hidden)
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
        if x.shape[2] % 9:
            raise ValueError("Two b=3 transitions require microscopic L divisible by 9")
        first = self.C(x, generator=generator)
        return [x, first, self.C(first, generator=generator)]


def sample_regions(x: torch.Tensor, pairs: int = 128,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample microscopic V=3 x 3 and complete buffered 7 x 7 outer shell.

    The 24 shell sites have offsets [-2,4] on the perimeter relative to V's
    top-left origin.  The intervening buffer has width one.  Boundaries are
    periodic, and each physical ensemble independently samples its own
    configuration indices and origins, with replacement.
    """
    _field(x)
    if not isinstance(pairs, int) or pairs < 2:
        raise ValueError("At least two region pairs are required for leave-one-out smoothing")
    e, n, length, _, _ = x.shape
    if length < 7:
        raise ValueError("The full buffered 7 x 7 shell requires L >= 7")
    configs = torch.randint(n, (e, pairs), device=x.device, generator=generator)
    row = torch.randint(length, (e, pairs), device=x.device, generator=generator)
    column = torch.randint(length, (e, pairs), device=x.device, generator=generator)
    ensemble = torch.arange(e, device=x.device)[:, None, None]

    def gather(offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
        offset = torch.tensor(offsets, device=x.device)
        return x[ensemble, configs[..., None],
                 (row[..., None] + offset[:, 0]) % length,
                 (column[..., None] + offset[:, 1]) % length, 0]

    return gather(BLOCK_OFFSETS), gather(SHELL_OFFSETS)


def gaussian_bin_probabilities(mean: torch.Tensor, scale: float | torch.Tensor,
                               noise_ratio: float = .1, bins: int = 16) -> torch.Tensor:
    """Analytic fixed-noise Gaussian bin masses, including infinite tails.

    Start with ``bins+1`` equally spaced edges in [-3 scale, 3 scale] and
    replace the first and last edges by +/- infinity, preserving ``bins``
    output bins.  Finite inner edges never adapt to data or C parameters.
    """
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    scale_value = float(scale.detach().item()) if isinstance(scale, torch.Tensor) else float(scale)
    _positive(scale_value, "scale")
    noise_ratio = _positive(noise_ratio, "noise_ratio")
    edges = torch.linspace(-3., 3., bins + 1, dtype=mean.dtype, device=mean.device)[1:-1]
    z = (edges * scale_value - mean[..., None]) / (noise_ratio * scale_value)
    cdf = .5 * (1. + torch.erf(z / math.sqrt(2.)))
    zero, one = torch.zeros_like(cdf[..., :1]), torch.ones_like(cdf[..., :1])
    return torch.diff(torch.cat((zero, cdf, one), dim=-1), dim=-1)


def smoothed_information(mean: torch.Tensor, environment: torch.Tensor,
                         scale: float | torch.Tensor, noise_ratio: float = .1,
                         bins: int = 16, bandwidth: float = 1.) -> torch.Tensor:
    """Full-shell Gaussian-kernel smoothed information proxy, in nats.

    ``mean`` is [ensembles,pairs] (or [...,1]), ``environment`` is
    [ensembles,pairs,24].  Environments and scale are detached: only C's means
    receive gradients.  Leave-one-out kernel weights have zero diagonal and
    normalize each row; log-softmax stabilization avoids all-zero rows.
    The proxy averages KL(q_i || mean_i q_i) after q_i = sum_j W_ij P_j.
    It has finite-sample smoothing bias and is not an exact Shannon MI.
    """
    bandwidth = _positive(bandwidth, "bandwidth")
    if mean.ndim == 3 and mean.shape[-1] == 1:
        mean = mean[..., 0]
    if (mean.ndim != 2 or environment.ndim != 3
            or environment.shape[:2] != mean.shape or environment.shape[-1] != 24
            or mean.shape[1] < 2 or mean.shape[0] < 1):
        raise ValueError("Expected means [E,P] and full-shell environments [E,P,24], P >= 2")
    scale_value = float(scale.detach().item()) if isinstance(scale, torch.Tensor) else float(scale)
    scale_value = _positive(scale_value, "scale")
    env = environment.detach() / scale_value
    # Squared Euclidean distances are explicitly divided by all 24 sites.
    norms = env.square().sum(-1)
    distance = (norms[..., :, None] + norms[..., None, :]
                - 2. * torch.bmm(env, env.transpose(1, 2))).clamp_min(0.) / 24.
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


def continuous_objectives(model: Phi4QCR, x: torch.Tensor,
                          generator: torch.Generator | None = None,
                          information_pairs: int = 128
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Independent Q, C, R objectives with uniform target-scale weights.

    Q uses detached fields/features exclusively.  Direct g is the average of
    the two disjoint half-ensemble encodings, not Q of their pooled features.
    Affine reconstruction compares decode(direct) to pooled local statistics.
    R gets detached direct g and backpropagates through every rollout step.
    C uses only its information, mean-output variance floor, and uniform
    amplitude terms.  Sampled Gaussian noise cannot satisfy the variance floor.
    """
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
        # Local statistics are linear averages over samples, so this is
        # exactly the full-ensemble feature vector without recomputing it.
        target = .5 * (first_features + second_features)
        reconstruction.append(_square_l2(model.Q.decode(g) - target))
        subset.append(_square_l2(first - second))
    reconstruction_loss = torch.stack(reconstruction).mean()
    coarse_subset = torch.stack(subset[1:]).mean()
    q_loss = 10. * reconstruction_loss + 20. * subset[0] + 5. * coarse_subset

    one_step, rollout = [], []
    detached = [g.detach() for g in direct]
    prediction = detached[0]
    for index in range(2):
        one_step.append(_square_l2(model.advance(detached[index]) - detached[index + 1]))
        prediction = model.advance(prediction)
        rollout.append(_square_l2(prediction - detached[index + 1]))
    one_step_loss, rollout_loss = torch.stack(one_step).mean(), torch.stack(rollout).mean()
    r_loss = 5. * one_step_loss + 5. * rollout_loss

    blocks, environment = sample_regions(x.detach(), information_pairs, generator)
    means = model.C.mean_blocks(blocks)[..., 0]
    information = smoothed_information(means, environment, model.scale,
                                       noise_ratio=float(model.C.noise_ratio))
    variance_terms = []
    for level in levels[:-1]:
        mean_output = model.C.mean_field(level)
        input_std = level.detach().flatten(1).std(dim=1, unbiased=False)
        output_std = mean_output.flatten(1).std(dim=1, unbiased=False)
        variance_terms.append(((input_std - output_std).clamp_min(0.) / model.scale).square().mean())
    variance_floor = torch.stack(variance_terms).mean()
    amplitudes = model.scale * torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    probes = amplitudes[:, None].expand(5, 9)
    uniform_amplitude = .5 * ((model.C.mean_blocks(probes)[:, 0] - amplitudes) / model.scale).square().mean()
    c_loss = -2. * information + 2. * variance_floor + uniform_amplitude

    terms = {
        "q_loss": q_loss, "c_loss": c_loss, "r_loss": r_loss,
        "reconstruction": reconstruction_loss, "subset_micro": subset[0],
        "subset_coarse": coarse_subset, "information_proxy": information,
        "variance_floor": variance_floor, "uniform_amplitude": uniform_amplitude,
        "closure_one_step": one_step_loss, "closure_rollout": rollout_loss,
    }
    for index in range(3):
        terms[f"reconstruction_level_{index}"] = reconstruction[index]
        terms[f"subset_level_{index}"] = subset[index]
    for index in range(2):
        terms[f"closure_one_step_level_{index + 1}"] = one_step[index]
        terms[f"closure_rollout_level_{index + 1}"] = rollout[index]
    scalar_values = torch.stack(list(terms.values())).detach().cpu().tolist()
    if not all(math.isfinite(value) for value in scalar_values):
        raise FloatingPointError("Non-finite continuous QCR objective")
    metrics = dict(zip(terms, scalar_values))
    return q_loss, c_loss, r_loss, metrics
