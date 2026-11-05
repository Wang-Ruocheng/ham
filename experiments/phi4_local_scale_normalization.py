"""Explicit frozen-C, two-arm Q correction; no change to joint-QCR defaults.

Only the three scalar reconstruction weights change. They are fitted once from
TRAIN descriptors, with one global multiplier matching source reconstruction
loss. No parameter labels, PCA directions or TEST data enter the objective.
"""
from __future__ import annotations
import torch

ARMS = ('A_original', 'B_scale_normalized')
READOUTS = ('source',) + ARMS
Q_STEPS, R_STEPS = 600, 1000


@torch.no_grad()
def fit_scale_weights(train_target, train_decoded):
    if train_target.ndim != 3 or train_target.shape[1] != 3 or train_decoded.shape != train_target.shape:
        raise ValueError('Expected TRAIN [ensemble,three scales,descriptor]')
    p, d = train_target.detach(), train_decoded.detach()
    variance = ((p-p.mean(0))**2).sum(-1).mean(0)
    if not torch.isfinite(variance).all() or (variance <= 1e-12).any():
        raise ValueError('Unresolved zero descriptor variance; no automatic reweighting')
    inverse = variance.mean()/variance
    initial_raw = ((d-p)**2).sum(-1).mean(0)
    alpha = initial_raw.mean()/(initial_raw*inverse).mean().clamp_min(1e-30)
    weights = alpha*inverse
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid frozen scale weights')
    return dict(variance=variance, inverse_variance_weights=inverse,
                global_initial_loss_match=alpha, weights=weights,
                initial_raw=initial_raw, initial_weighted=initial_raw*weights)


def objective(q, levels, per_field_targets, weights):
    if len(levels) != 3 or per_field_targets.shape[1] != 3 or weights.shape != (3,):
        raise ValueError('Expected three levels and three scalar weights')
    rec, sub = [], []
    for k, x in enumerate(levels):
        half = x.shape[1]//2
        if half < 1 or 2*half != x.shape[1]:
            raise ValueError('Equal nonempty disjoint subsets required')
        first, second = q(x[:, :half].detach()), q(x[:, half:].detach())
        g = .5*(first+second)
        residual = q.decode(g)-per_field_targets[:, k].detach().mean(1)
        rec.append(residual.square().sum(-1).mean())
        sub.append((first-second).square().sum(-1).mean())
    rec, sub = torch.stack(rec), torch.stack(sub)
    weighted = weights.detach().to(rec)*rec
    total = 10*weighted.mean()+20*sub[0]+5*sub[1:].mean()
    return total, dict(reconstruction_raw=rec, reconstruction_weighted=weighted, consistency=sub)


def update(q, optimizer, levels, targets, weights):
    optimizer.zero_grad(set_to_none=True)
    loss, values = objective(q, levels, targets, weights)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite Q loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in q.parameters()):
        raise FloatingPointError('Invalid Q gradients')
    optimizer.step()
    q.normalize_decoder_()
    if any(not torch.isfinite(p).all() for p in q.parameters()):
        raise FloatingPointError('Nonfinite Q weights')
    optimizer.zero_grad(set_to_none=True)
    return dict(q_total=float(loss.detach()), **{k:v.detach().cpu().tolist() for k,v in values.items()})
