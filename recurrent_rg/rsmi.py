"""RSMI-only binary kernels: buffered environments and exact-H InfoNCE.

The critic enumerates the two possible H values. Summing over P_C(H|V)
eliminates the straight-through estimator without changing the stochastic
kernel's objective. Negatives NEVER cross ensemble boundaries.
"""
from itertools import product
import math

import torch
from torch import nn


def binary_chain_joint_histograms(x, *, flip=False):
    """Per-ensemble empirical P(V,E), V=(s0,s1), E=(s[-2],s3).

    Use all cached configurations and translations; no labels, pooling across
    ensembles, pseudocounts or critic. Overlapping translations are not
    independent observations. Optional exact global-flip data augmentation.
    """
    if x.ndim != 4 or x.shape[-1] != 1 or x.shape[2] < 6:
        raise ValueError('Need binary 1D chains of length >=6')
    if not torch.all((x==1)|(x==-1)):raise ValueError('Expected +/-1 spins')
    bits=(x[...,0]>0).long()
    v=bits+2*torch.roll(bits,-1,2)
    e=torch.roll(bits,2,2)+2*torch.roll(bits,-3,2)
    ids=(4*v+e).reshape(len(x),-1)
    offsets=16*torch.arange(len(x),device=x.device)[:,None]
    hist=torch.bincount((ids+offsets).flatten(),minlength=16*len(x)).reshape(-1,4,4).double()
    if flip:hist=.5*(hist+hist.flip((1,2)))
    return hist/hist.sum((1,2),keepdim=True)


def histogram_mutual_information(joint_ve, kernel_table):
    """Exact MI of supplied empirical distributions, in nats, differentiable C.

    Finite-cache plug-in MI is not exact population MI and can be biased.
    P(H,E)=sum_V P(V,E)P_C(H|V); sum over H rather than sampling C.
    """
    he=torch.einsum('sve,vh->she',joint_ve,kernel_table)
    ph=he.sum(-1,keepdim=True);pe=he.sum(-2,keepdim=True)
    tiny=torch.finfo(he.dtype).tiny
    return (he*(he.clamp_min(tiny).log()-ph.clamp_min(tiny).log()-pe.clamp_min(tiny).log())).sum((-1,-2))


def best_binary_pair_kernel(joint_ve):
    """Enumerate 16 rules using training histograms only; no coupling labels.

    Ties are label-gauged by ordering the two uniform-block output labels,
    then by code. This tie-break does not constrain subsequent learning.
    """
    tables=[];scores=[]
    for code in range(16):
        p=((code//(2**torch.arange(4,device=joint_ve.device)))%2).to(joint_ve.dtype)
        table=torch.stack((1-p,p),-1)
        tables.append(table);scores.append(float(histogram_mutual_information(joint_ve,table).mean()))
    best=max(scores)
    ties=[i for i,s in enumerate(scores) if abs(s-best)<1e-12]
    selected=min(ties,key=lambda i:(int(tables[i][0,1]>tables[i][-1,1]),i))
    return tables[selected],dict(selected_code=selected,tied_codes=ties,MI_nats_by_code=scores)


def initialize_kernel_from_table(kernel, table, *, steps=1200):
    """Fit only C's existing MLP to a data-selected rule, then leave it trainable."""
    states=table.shape[1];width=round(math.log(len(table),states))
    binary=kernel.input_channels==1 and states==2
    if states**width!=len(table):raise ValueError('Invalid finite-alphabet rule table length')
    codes=(torch.arange(len(table))[:,None]//(states**torch.arange(width)))%states
    patterns=((2*codes-1).to(table) if binary else torch.nn.functional.one_hot(codes,states).flatten(1).to(table))
    target=(.0001+.9998*table[:,1] if binary else .0001+(1-states*.0001)*table)
    optimizer=torch.optim.Adam(kernel.parameters(),lr=.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=kernel.field_head.logits(kernel.local_encoder(patterns))
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(2*logits.squeeze(-1),target)
              if binary else -(target*torch.log_softmax(logits,-1)).sum(-1).mean())
        loss.backward();optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        p=(kernel_probabilities(kernel,patterns[None])[0,:,1] if binary else categorical_kernel_probabilities(kernel,patterns))
    return dict(fit_steps=steps,max_probability_error=float((p-target).abs().max()),fitted_probabilities=p.tolist())


def categorical_kernel_probabilities(kernel,visible):
    return torch.softmax(kernel.field_head.logits(kernel.local_encoder(visible)),dim=-1)


def region_offsets(dimension, block=2, buffer=1):
    if dimension not in (1, 2, 3) or block < 1 or buffer < 0:
        raise ValueError('Invalid RSMI region')
    visible = list(product(range(block), repeat=dimension))
    low, high = -buffer-1, block+buffer
    environment = [p for p in product(range(low, high+1), repeat=dimension)
                   if any(v in (low, high) for v in p)]
    return torch.tensor(visible), torch.tensor(environment)


def sample_regions(x, systems, samples, generator, *, buffer=1, flip=False):
    """One random translated V/E pair per distinct configuration, per ensemble.

    x is [ensemble, sample, *lattice, 1]. The enclosing shell lies outside
    a buffer of `buffer` sites. Configurations are sampled without replacement.
    """
    dimension = x.ndim-3
    visible, environment = region_offsets(dimension, buffer=buffer)
    shape = x.shape[2:-1]
    if x.shape[-1] != 1 or any(n < 2+2*(buffer+1) for n in shape):
        raise ValueError('Lattice too small for a non-aliasing buffered environment')
    if not 2 <= samples <= x.shape[1] or not 1 <= systems <= len(x):
        raise ValueError('Unavailable RSMI batch')
    ids = torch.randperm(len(x), generator=generator)[:systems]
    sample_ids = torch.stack([torch.randperm(x.shape[1], generator=generator)[:samples]
                              for _ in ids])
    origins = torch.stack([torch.randint(n, (systems, samples), generator=generator)
                           for n in shape], dim=-1)

    def gather(offsets):
        positions = (origins[:, :, None, :] + offsets) % torch.tensor(shape)
        indices = (ids[:, None, None], sample_ids[:, :, None]) + tuple(
            positions[..., a] for a in range(dimension)) + (0,)
        return x[indices].double()

    v, e = gather(visible), gather(environment)
    if flip:
        signs = 2*torch.randint(2, (systems, samples, 1), generator=generator)-1
        v, e = v*signs, e*signs
    return v, e


def kernel_probabilities(kernel, visible):
    """Exact lookup, retaining gradients to the existing MLP C parameters."""
    parameter = next(kernel.parameters())
    if visible.device != parameter.device:
        raise ValueError('Visible blocks and C must be on the same device')
    width = visible.shape[-1]
    powers = 2**torch.arange(width, device=parameter.device)
    patterns = (2*((torch.arange(2**width, device=parameter.device)[:, None]//powers)
                   % 2)-1).to(dtype=parameter.dtype)
    logits = kernel.field_head.logits(kernel.local_encoder(patterns)).squeeze(-1)
    table = torch.sigmoid(2*logits)
    indices = ((visible > 0).long()*powers).sum(-1)
    positive = table[indices]
    return torch.stack((1-positive, positive), dim=-1)


class EnvironmentCritic(nn.Module):
    def __init__(self, environment_size, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(environment_size, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, environment):
        return self.network(environment)


def enumerated_infonce(probabilities, scores):
    """Per-ensemble InfoNCE in nats; shapes [ensemble, pairs, H=2].

    E_H log(exp f(H,E_i) / (1/N sum_j exp f(H,E_j))). Because H has
    two states, the negative denominator can be shared over anchors.
    Finite-data estimates are NOT certified population MI lower bounds.
    """
    if probabilities.shape != scores.shape or scores.ndim != 3 or scores.shape[-1] != 2:
        raise ValueError('Expected matching [ensembles,pairs,2] arrays')
    if scores.shape[1] < 2:
        raise ValueError('InfoNCE requires at least two pairs')
    log_denominator = torch.logsumexp(scores, dim=1, keepdim=True)-math.log(scores.shape[1])
    return (probabilities*(scores-log_denominator)).sum(-1).mean(-1)


def reference_probabilities(visible, rule):
    if rule == 'decimation':
        positive = (visible[..., 0]+1)/2
    elif rule == 'majority':
        # Fair random tie breaking, integrated exactly rather than sampled.
        positive = (torch.sign(visible.sum(-1))+1)/2
    elif rule == 'random':
        positive = torch.full_like(visible[..., 0], .5)
    else:
        raise ValueError(rule)
    return torch.stack((1-positive, positive), -1)
