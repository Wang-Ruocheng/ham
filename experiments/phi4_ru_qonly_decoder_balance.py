"""Opt-in, paired frozen-C decoder ablation; no default recipe changes.

Training functions accept only fields, statistics and a detached TRAIN metric.
Physical couplings and evaluation splits never enter a training objective.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

ARMS = ('A_affine', 'B_nonlinear', 'C_balanced')
Q_STEPS, R_STEPS = 600, 1000
SYSTEM_BATCH, SAMPLE_BATCH, CACHE_SAMPLES = 8, 32, 64


class ResidualDecoder(nn.Module):
    """Shared affine + f(g)-f(0); exact initial affine, bounded decoder gain."""
    def __init__(self, affine):
        super().__init__()
        self.affine = copy.deepcopy(affine)
        self.net = nn.Sequential(nn.Linear(affine.in_features, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, affine.out_features, bias=False)).to(affine.weight)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.normalize_()

    def forward(self, g):
        return self.affine(g) + self.net(g) - self.net(torch.zeros_like(g))

    @torch.no_grad()
    def normalize_(self):
        self.affine.weight.div_(self.affine.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.div_((torch.linalg.matrix_norm(layer.weight, ord=2)/2).clamp_min(1.))

    def bounds(self):
        norms = [float(torch.linalg.matrix_norm(x.weight.detach(), ord=2))
                 for x in self.net if isinstance(x, nn.Linear)]
        return dict(residual_matrix_norms=norms,
                    lipschitz_upper_bound=float(torch.linalg.matrix_norm(self.affine.weight.detach(), ord=2))+math.prod(norms),
                    fixed_upper_bound=math.sqrt(self.affine.in_features)+8,
                    injectivity_certified=False)


class QArm(nn.Module):
    def __init__(self, source, nonlinear):
        super().__init__()
        self.encoder = copy.deepcopy(source)
        affine = self.encoder.decoder
        self.encoder.decoder = nn.Identity()
        self.decoder = ResidualDecoder(affine) if nonlinear else affine

    def forward(self, fields):
        return self.encoder(fields.detach())

    def decode(self, g):
        return self.decoder(g)

    @torch.no_grad()
    def normalize_(self):
        if isinstance(self.decoder, ResidualDecoder):
            self.decoder.normalize_()
        else:
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


def make_arms(source):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9103)
        a = QArm(source, False)
        b = QArm(source, True)
        c = copy.deepcopy(b)
    return dict(zip(ARMS, (a, b, c)))


@torch.no_grad()
def fit_metrics(train_target, train_groups, train_initial_decoded):
    """All resolved observable axes, not capped by dg; repeated-chain noise.

Group IDs identify repeated physical ensembles, not numeric couplings.
Noise is the within-group covariance of independent chains divided by their
count: uncertainty of the group mean. This is a finite-chain heuristic, not
an equilibration or reliable physical-rank certificate.
"""
    p = train_target.detach().cpu().double()
    decoded = train_initial_decoded.detach().cpu().double()
    groups = torch.as_tensor(train_groups)
    if p.ndim != 3 or decoded.shape != p.shape or len(groups) != len(p):
        raise ValueError('Expected matching TRAIN [chain,scale,descriptor]')
    counts = [int((groups == g).sum()) for g in groups.unique()]
    if len(counts) < 3 or min(counts) < 2:
        raise ValueError('Repeated independent TRAIN chains required')
    metrics = []
    for k in range(p.shape[1]):
        grouped = [p[groups == g, k] for g in groups.unique()]
        means = torch.stack([x.mean(0) for x in grouped])
        _, s, vh = torch.linalg.svd(means-means.mean(0), full_matrices=False)
        axes = vh.T
        variance = s.square()/len(means)
        noise = torch.stack([((x-x.mean(0))@axes).square().sum(0)/((len(x)-1)*len(x))
                             for x in grouped]).mean(0)
        reliable = (variance > 5*noise) & (variance > variance[0]*1e-8)
        axes = axes[:, reliable]
        gain = (variance[0]/variance[reliable]).clamp(1., 10.)
        residual = decoded[:, k]-p[:, k]
        raw = residual.square().sum(-1).mean()
        weighted = raw + ((residual@axes).square()*(gain-1)).sum(-1).mean()
        alpha = raw/weighted.clamp_min(1e-30)
        metrics.append(dict(axes=axes, gains=gain, alpha=alpha,
                            variances=variance, group_mean_noise=noise, reliable=reliable,
                            initial_raw=raw, initial_weighted=alpha*weighted))
    return metrics


def metric_loss(residual, metric=None):
    raw = residual.square().sum(-1)
    if metric is None:
        return raw.mean()
    axes, gains, alpha = (metric[k].to(residual) for k in ('axes', 'gains', 'alpha'))
    return (alpha*(raw+((residual@axes).square()*(gains-1)).sum(-1))).mean()


def objective(q, levels, per_field_targets, metrics=None):
    if len(levels) != 3 or per_field_targets.shape[1] != 3:
        raise ValueError('Expected 36 -> 12 -> 4')
    reconstruction, consistency, raw = [], [], []
    for k, x in enumerate(levels):
        half = x.shape[1]//2
        if half < 1 or 2*half != x.shape[1]:
            raise ValueError('Two equal disjoint nonempty subsets required')
        first, second = q(x[:, :half]), q(x[:, half:])
        g = .5*(first+second)
        residual = q.decode(g)-per_field_targets[:, k].detach().mean(1)
        raw.append(metric_loss(residual))
        reconstruction.append(metric_loss(residual, None if metrics is None else metrics[k]))
        consistency.append((first-second).square().sum(-1).mean())
    rec, sub, raw = map(torch.stack, (reconstruction, consistency, raw))
    total = 10*rec.mean()+20*sub[0]+5*sub[1:].mean()
    return total, dict(reconstruction=rec, reconstruction_raw=raw, consistency=sub)


def update(q, optimizer, levels, targets, metrics=None):
    optimizer.zero_grad(set_to_none=True)
    loss, values = objective(q, levels, targets, metrics)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite Q objective')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in q.parameters()):
        raise FloatingPointError('Missing/nonfinite Q gradient')
    optimizer.step()
    q.normalize_()
    if any(not torch.isfinite(p).all() for p in q.parameters()):
        raise FloatingPointError('Nonfinite Q weights')
    optimizer.zero_grad(set_to_none=True)
    return dict(q_total=float(loss.detach()), **{k:v.detach().cpu().tolist() for k,v in values.items()})


class RReadout(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 96), nn.Tanh(), nn.Linear(96,96),
                                 nn.Tanh(), nn.Linear(96,dim)).double()

    def advance_theory(self, g):
        return g+self.net(g)


def r_objective(r, direct):
    # Uniform target scales; all earlier starts, full recurrent R gradients.
    g = direct.detach()
    return 5*((r.advance_theory(g[:,0])-g[:,1]).square().sum(-1).mean()
              + .5*((r.advance_theory(g[:,1])-g[:,2]).square().sum(-1).mean()
                    +(r.advance_theory(r.advance_theory(g[:,0]))-g[:,2]).square().sum(-1).mean()))
