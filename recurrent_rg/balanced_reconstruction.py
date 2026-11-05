"""Explicit ablation: signal-gated, TRAIN-only reconstruction metric.

This does not change the default Q objective or initialize any model weights.
The eigenvectors describe observable histograms, not latent coordinates.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class BalancedMetric:
    axes: torch.Tensor
    variances: torch.Tensor
    noise: torch.Tensor
    denominators: torch.Tensor

    def description(self):
        return dict(rank=self.axes.shape[1], variances=self.variances.tolist(),
                    subset_noise_estimate=self.noise.tolist(),
                    denominators=self.denominators.tolist())

    def state_dict(self):
        return {key: getattr(self, key).detach().clone()
                for key in ('axes', 'variances', 'noise', 'denominators')}


@torch.no_grad()
def fit_balanced_metric(first, second, *, max_directions, noise_ratio=5.,
                        max_weight_ratio=20.):
    """Fit only on TRAIN paired subsets of normalized motif probabilities.

    Noise is E[(p_a-p_b)^2]/4 projected onto each direction. This is a
    heuristic full-ensemble noise estimate; correlated samples can bias it.
    Directions without a supported signal are not forced into the loss.
    """
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
        raise ValueError('Need matching TRAIN histogram matrices')
    if max_directions < 1 or not math.isfinite(noise_ratio) or noise_ratio <= 1:
        raise ValueError('Positive rank and noise ratio above one required')
    if not math.isfinite(max_weight_ratio) or max_weight_ratio < 1:
        raise ValueError('Finite weight ratio at least one required')
    for p in (first, second):
        if not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError('Invalid histogram probabilities')
        if not torch.allclose(p.sum(-1), torch.ones(len(p), dtype=p.dtype, device=p.device)):
            raise ValueError('Histograms must be normalized')
    p = .5 * (first.detach() + second.detach())
    centered = p - p.mean(0)
    values, axes = torch.linalg.eigh(centered.T @ centered / len(p))
    values, axes = values.flip(0).clamp_min(0), axes.flip(1)
    noise = .25 * ((first - second) @ axes).square().mean(0)
    floor = values[0].clamp_min(1e-12) * 1e-8
    supported = ((values > noise_ratio * noise) & (values > floor)).nonzero().flatten()
    selected = supported[:max_directions]
    denominators = values[selected].clamp_min(values[0] / max_weight_ratio).clamp_min(1e-12)
    return BalancedMetric(axes[:, selected].detach(), values[selected].detach(),
                          noise[selected].detach(), denominators.detach())


def balanced_reconstruction_loss(decoder, theory, target, metric):
    """Dimensionless relative residual; gradients into Q/decoder, never target."""
    residual = decoder(theory) - target.detach()
    if residual.shape != target.shape or residual.shape[-1] != metric.axes.shape[0]:
        raise ValueError('Balanced reconstruction shape mismatch')
    if not metric.axes.shape[1]:
        return residual.sum() * 0
    projected = residual @ metric.axes.to(residual)
    return (projected.square() / metric.denominators.to(residual)).mean()


@torch.no_grad()
def refresh_training_metrics(model, configurations, pattern_fn, *, samples,
                             sampling_seed=9191, chunk_size=8, fitted_epoch=0):
    """Current-C metrics from ALL TRAIN ensembles, never held-out data/labels.

    Full TRAIN covariance avoids fitting the metric to just eight systems in
    a stochastic minibatch. The same noise gate/capped weights as the paired
    pilot are used. No weights are initialized; the training RNG is preserved.
    """
    if samples < 2 or samples % 2 or samples > configurations.shape[1] or chunk_size < 1:
        raise ValueError('Valid even samples and positive metric chunk size required')
    first, second = [], []
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sampling_seed)
            for chunk in configurations.split(chunk_size):
                levels = model.coarse_grain(chunk[:, :samples].double())
                half = samples // 2
                first.append(torch.stack([pattern_fn(v[:, :half]) for v in levels], 1))
                second.append(torch.stack([pattern_fn(v[:, half:]) for v in levels], 1))
        a, b = torch.cat(first), torch.cat(second)
        metrics = [fit_balanced_metric(a[:, k], b[:, k],
                    max_directions=model.theory_encoder.distribution_decoder.in_features)
                   for k in range(a.shape[1])]
        model.balanced_reconstruction_metrics = metrics
        model.balanced_metric_fitted_epoch = fitted_epoch
        return metrics
    finally:
        model.train(was_training)
