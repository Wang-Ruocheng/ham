"""Opt-in frozen-C 2x2 Q ablation. No couplings or TEST data in training.

Local neighborhoods live in the observed descriptor space, at one scale only.
Only repeat-chain-noise-resolved directions are penalized; no forced rank two.
"""
import torch
from experiments.phi4_local_scale_normalization import objective as base_objective

ARMS = ('warm_base', 'warm_local', 'fresh_base', 'fresh_local')
Q_STEPS, R_STEPS = 600, 1000
NEIGHBORS, MAX_GAIN, NOISE_FACTOR = 6, 10., 5.


@torch.no_grad()
def fit_graph(train_p, train_groups):
    p = train_p.detach().cpu().double()
    groups = torch.as_tensor(train_groups).detach().cpu()
    if p.ndim != 3 or p.shape[1] != 3 or len(groups) != len(p) or not torch.isfinite(p).all():
        raise ValueError('Expected finite TRAIN [chains,3,descriptors]')
    unique = groups.unique(sorted=True)
    rows = [p[groups == g] for g in unique]
    if len(rows) <= NEIGHBORS or min(len(x) for x in rows) < 4:
        raise ValueError('At least seven TRAIN groups with four independent chains required')
    means = torch.stack([x.mean(0) for x in rows])
    all_axes, all_gains, all_neighbors, all_var, all_noise = [], [], [], [], []
    for k in range(3):
        distance = torch.cdist(means[:, k], means[:, k])
        distance.fill_diagonal_(float('inf'))
        neighbors = distance.argsort(dim=1)[:, :NEIGHBORS]
        axes, gains, variances, noises = [], [], [], []
        for i, near in enumerate(neighbors):
            ids = torch.cat((torch.tensor([i]), near))
            local = means[ids, k]
            _, s, vh = torch.linalg.svd(local-local.mean(0), full_matrices=False)
            v = vh[:NEIGHBORS].T
            variance = s[:NEIGHBORS].square()/len(ids)
            noise = torch.stack([((rows[j][:, k]-means[j, k])@v).square().sum(0)
                                 /((len(rows[j])-1)*len(rows[j])) for j in ids]).mean(0)
            reliable = (variance > NOISE_FACTOR*noise) & (variance > variance[0]*1e-8) & (variance > 1e-14)
            gain = (variance[0]/variance.clamp_min(1e-14)).clamp(1., MAX_GAIN)*reliable
            axes.append(v);gains.append(gain);variances.append(variance);noises.append(noise)
        all_axes.append(torch.stack(axes));all_gains.append(torch.stack(gains))
        all_neighbors.append(neighbors);all_var.append(torch.stack(variances));all_noise.append(torch.stack(noises))
    return dict(groups=unique, means=means, axes=torch.stack(all_axes), gains=torch.stack(all_gains),
                neighbors=torch.stack(all_neighbors), variance=torch.stack(all_var), noise=torch.stack(all_noise))


def local_loss(decoded_pairs, target_delta, axes, gains):
    if decoded_pairs.ndim != 3 or decoded_pairs.shape[1] != 2:
        raise ValueError('Expected [pairs,2,descriptor]')
    error = decoded_pairs[:, 0]-decoded_pairs[:, 1]-target_delta.detach().to(decoded_pairs)
    projected = torch.einsum('bd,bda->ba', error, axes.detach().to(error))
    return (projected.square()*gains.detach().to(error)).sum(-1).mean()


@torch.no_grad()
def initial_local_loss(decoded_means, graph):
    values=[]
    for k in range(3):
        n=graph['neighbors'][k];i=torch.arange(len(n))[:, None].expand_as(n).flatten();j=n.flatten()
        values.append(local_loss(torch.stack((decoded_means[i,k],decoded_means[j,k]),1),
            graph['means'][i,k]-graph['means'][j,k], graph['axes'][k,i], graph['gains'][k,i]))
    return torch.stack(values).mean()


def objective(q, levels, targets, pair_fields, target_delta, axes, gains, coefficient):
    base, values = base_objective(q, levels, targets, torch.ones(3, dtype=torch.double))
    half=pair_fields.shape[1]//2
    if half < 1 or pair_fields.shape[1] != 2*half:
        raise ValueError('Two equal nonempty pair subsets required')
    g=.5*(q(pair_fields[:, :half].detach())+q(pair_fields[:, half:].detach()))
    decoded=q.decode(g).reshape(-1,2,target_delta.shape[-1])
    local=local_loss(decoded,target_delta,axes,gains)
    total=base+10*coefficient*local
    return total,dict(**values,local=local,base_total=base)


def update(q, optimizer, *args):
    optimizer.zero_grad(set_to_none=True)
    loss, values=objective(q,*args)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Q loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in q.parameters()):
        raise FloatingPointError('Invalid Q gradients')
    optimizer.step();q.normalize_decoder_()
    if any(not torch.isfinite(p).all() for p in q.parameters()):raise FloatingPointError('Invalid Q weights')
    optimizer.zero_grad(set_to_none=True)
    return dict(q_total=float(loss.detach()), **{k:v.detach().cpu().tolist() for k,v in values.items()})


@torch.no_grad()
def validation_metrics(decoded, target, groups, graph):
    """Internal validation only. Axes/gains frozen on TRAIN; no numeric labels.

Weak validation components must exceed twice their repeat-chain standard error.
This finite-chain gate is not a physical-rank or equilibrium certificate.
"""
    groups=torch.as_tensor(groups);unique=groups.unique(sorted=True)
    p=torch.stack([target[groups==g].mean(0) for g in unique])
    d=torch.stack([decoded[groups==g].mean(0) for g in unique])
    se=[]
    total_error=0.;total_signal=0.;count=0
    for k in range(3):
        for i in range(len(unique)):
            for j in range(i+1,len(unique)):
                anchor=((graph['means'][:,k]-.5*(p[i,k]+p[j,k]))**2).sum(-1).argmin()
                axes=graph['axes'][k,anchor];gain=graph['gains'][k,anchor]
                delta=(p[i,k]-p[j,k])@axes
                noise=[]
                for row in (i,j):
                    chains=target[groups==unique[row],k]
                    noise.append(((chains-p[row,k])@axes).square().sum(0)/((len(chains)-1)*len(chains)))
                valid=(gain>0)&(delta.square()>4*(noise[0]+noise[1]))
                valid[0]=False  # local non-leading directions, never prescribed rank two
                error=((d[i,k]-d[j,k])@axes-delta).square()
                total_error+=float(error[valid].sum());total_signal+=float(delta[valid].square().sum());count+=int(valid.sum())
        se.append(float((decoded[:,k]-target[:,k]).square().mean()))
    return dict(mse=se,weak_component_count=count,weak_relative_mse=total_error/total_signal if total_signal>0 else None)


def gate(control, candidate, scale_variance):
    a=torch.tensor(control['mse']);b=torch.tensor(candidate['mse']);v=torch.tensor(scale_variance)
    resolved=min(control['weak_component_count'],candidate['weak_component_count'])>=6
    weak=(resolved and control['weak_relative_mse'] is not None and candidate['weak_relative_mse'] is not None
          and candidate['weak_relative_mse']<=.9*control['weak_relative_mse'])
    no_scale_harm=bool((b<=1.1**2*a).all())
    mean_improves=bool((b/v).mean()<=(a/v).mean())
    return dict(passed=bool(weak and no_scale_harm and mean_improves),weak_improvement_10pct=bool(weak),
                no_scale_NRMSE_worse_than_10pct=no_scale_harm,mean_relative_mse_not_worse=mean_improves,
                reliable_components_at_least_6=resolved)
