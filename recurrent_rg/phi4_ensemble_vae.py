"""Explicit continuous adaptation of the weak-KL ensemble VAE QCR.

Only finite soft local probabilities are reconstructed, not a field density.
Gaussian C and its smoothed information proxy retain the continuous recipe.
No coupling labels enter this module; this does not change standard defaults.
"""
import math
import torch
from .ensemble_vae import EnsembleVAEQ
from .phi4_aligned_qcr import (AlignedPhi4QCR, soft_plaquette_features,
    _sample_aligned_regions, _aligned_information)
from .losses import all_start_anchor_closure_loss


class ContinuousEnsembleVAEQ(EnsembleVAEQ):
    def __init__(self, field_scale, theory_dim=2, patch_chunk_size=65536):
        super().__init__(theory_dim=theory_dim, states=5)
        self.register_buffer('field_scale', torch.tensor(float(field_scale), dtype=torch.double))
        self.patch_chunk_size = patch_chunk_size

    def probabilities(self, x):
        with torch.no_grad():
            return soft_plaquette_features(x.detach(), self.field_scale,
                                           chunk_size=self.patch_chunk_size)

    def forward(self, x):
        return self.encode(self.probabilities(x))[0]


class Phi4EnsembleVAEQCR(AlignedPhi4QCR):
    def __init__(self, scale, seed=91, hidden=96, patch_chunk_size=65536):
        # Preserve the historical continuous C/R initialization and global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(scale, hidden=hidden, theory_dim=2,
                             patch_chunk_size=patch_chunk_size, block_size=3, steps=2)
            torch.manual_seed(seed)
            self.Q = ContinuousEnsembleVAEQ(scale, 2, patch_chunk_size)


@torch.no_grad()
def calibrate(model, train_configurations, systems=8):
    device = next(model.parameters()).device
    p = [model.Q.probabilities(x.to(device)) for x in train_configurations.split(systems)]
    model.Q.calibrate(torch.cat(p))
    return dict(physical_chains=len(train_configurations), fitting_split='TRAIN', scale=0,
                field_RMS=float(model.scale), std_floor=.001, labels_used=False,
                no_C_forward=True, no_pretraining=True, frozen_across_scales=True)


def encode_levels(model, levels):
    half = levels[0].shape[1] // 2
    return torch.stack([.5*(model.Q(v[:, :half])+model.Q(v[:, half:])) for v in levels], 1)


def objectives(model, x, generator=None, information_pairs=128):
    if x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError('Two disjoint equal halves required')
    levels = model.coarse_grain(x, generator=generator)
    half = x.shape[1]//2
    direct, q_losses, terms = [], [], {}
    for k, level in enumerate(levels):
        a = model.Q.probabilities(level[:, :half])
        b = model.Q.probabilities(level[:, half:])
        q, diagnostics = model.Q.cross_subset_objective(a, b, scale_index=k)
        q_losses.append(q)
        with torch.no_grad():
            direct.append(.5*(model.Q.encode(a)[0]+model.Q.encode(b)[0]))
        terms.update({f'{name}_level_{k}': value for name, value in diagnostics.items()})
    q_loss = torch.stack(q_losses).mean()
    closure, targets, pairs = all_start_anchor_closure_loss(
        model, torch.stack(direct, 1), coordinate_reduction='sum')
    r_loss = 10*closure

    # Same continuous C objective as aligned_objectives, on the same ancestral
    # trajectory; Q and R graphs have no route into these fields.
    blocks, environment = _sample_aligned_regions(x.detach(), information_pairs, generator, 3)
    information = _aligned_information(model.C.mean_blocks(blocks)[..., 0], environment,
                                      model.scale, float(model.C.noise_ratio), 3)
    variance = torch.stack([((v.detach().flatten(1).std(1, unbiased=False)
        - model.C.mean_field(v).flatten(1).std(1, unbiased=False)).clamp_min(0)
        / model.scale).square().mean() for v in levels[:-1]]).mean()
    amplitude = model.scale*torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    uniform = .5*((model.C.mean_blocks(amplitude[:, None].expand(5, 9))[:, 0]
                   - amplitude)/model.scale).square().mean()
    c_loss = -2*information+2*variance+uniform
    terms.update(q_total=q_loss, c_total=c_loss, r_total=r_loss,
        c_information=information, c_variance=variance, c_uniform=uniform,
        closure_all_start=closure,
        closure_one_step=.5*(pairs[0, 1]+pairs[1, 2]),
        closure_rollout=.5*(pairs[0, 1]+pairs[0, 2]))
    terms.update({f'closure_target_{k}': v for k, v in enumerate(targets, 1)})
    terms.update({f'closure_start_{s}_target_{t}': v for (s, t), v in pairs.items()})
    values = torch.stack(list(terms.values())).detach().cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError('Nonfinite continuous ensemble-VAE objective')
    if any(v > 1e-10 for name, v in zip(terms, values)
           if name.startswith(('decoder_probability_sum_error', 'decoder_log_probability_sum_error'))):
        raise FloatingPointError('Continuous ensemble-VAE decoder is not normalized')
    return q_loss, c_loss, r_loss, dict(zip(terms, values))


def independent_step(model, optimizers, x, step, generator=None, information_pairs=128):
    model.Q.set_objective_step(step)
    q, c, r, terms = objectives(model, x, generator, information_pairs)
    parameters = [list(getattr(model, n).parameters()) for n in ('Q', 'C', 'R')]
    # Every gradient is evaluated at the same pre-update parameter state.
    gradients = [torch.autograd.grad(loss, p) for loss, p in zip((q, c, r), parameters)]
    if any(not torch.isfinite(g).all() for group in gradients for g in group):
        raise FloatingPointError('Nonfinite independent gradient')
    for name, group, grads in zip(('Q', 'C', 'R'), parameters, gradients):
        optimizers[name].zero_grad(set_to_none=True)
        for p, g in zip(group, grads): p.grad = g
    for opt in optimizers.values(): opt.step()
    return terms
