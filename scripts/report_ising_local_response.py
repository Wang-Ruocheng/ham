"""Read-only scalar response audit of saved endpoint encodings and R.

No Q/C inference, physical sampling, MI, optimization or checkpoint selection.
Bootstrap uncertainty is conditional on the trained model and saved C draws.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recurrent_rg.checkpoints import load_run
from recurrent_rg.runner import file_fingerprint


KC = float(np.log1p(np.sqrt(2)) / 2)


def predict_and_derivative(model, values):
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    y = model.advance_theory(x)
    derivative, = torch.autograd.grad(y.sum(), x)
    return (y.detach().numpy().reshape(values.shape),
            derivative.detach().numpy().reshape(values.shape))


def polynomial_response(k, means, boot, predicted, predicted_boot, center, width):
    mask = np.abs(k - center) <= width
    if mask.sum() < 5:
        return dict(center=center, half_width=width, K_count=int(mask.sum()), status='insufficient_knots')
    x = (k[mask] - center) / width
    matrix = np.stack([x**i for i in range(4)], axis=1)
    weights = np.linalg.pinv(matrix)[1] / width
    slopes = np.einsum('n,nk->k', weights, means[mask])
    samples = np.einsum('n,bnk->bk', weights, boot[:, mask])
    rp = np.einsum('n,nk->k', weights, predicted[mask])
    rb = np.einsum('n,bnk->bk', weights, predicted_boot[:, mask])
    rows = []
    for s in range(means.shape[1] - 1):
        valid = np.abs(samples[:, s]) > 1e-10
        data = samples[valid, s+1] / samples[valid, s]
        fitted = rb[valid, s] / samples[valid, s]
        rows.append(dict(input_scale=s, output_scale=s+1,
            input_dg_dK=float(slopes[s]), output_dg_dK=float(slopes[s+1]),
            empirical_response=float(slopes[s+1] / slopes[s]),
            predicted_response=float(rp[s] / slopes[s]),
            empirical_response_bootstrap_95=np.percentile(data, [2.5, 97.5]).tolist(),
            predicted_response_bootstrap_95=np.percentile(fitted, [2.5, 97.5]).tolist(),
            response_mismatch=float((rp[s] - slopes[s+1]) / slopes[s]),
            mismatch_bootstrap_95=np.percentile(fitted-data, [2.5, 97.5]).tolist(),
            input_derivative_bootstrap_95=np.percentile(samples[:, s], [2.5, 97.5]).tolist(),
            valid_bootstrap_fraction=float(valid.mean())))
    return dict(center=center, half_width=width, K_count=int(mask.sum()),
                degree=3, transitions=rows, status='complete')


def inverse_branches(k, curve, value):
    result = []
    for i in range(len(k)-1):
        a, b = curve[i:i+2]
        if a != b and min(a, b) <= value <= max(a, b):
            t = (value-a)/(b-a)
            result.append(dict(segment=i, fraction=float(t), K=float(k[i]+t*(k[i+1]-k[i]))))
    return result


def root_closure_conflicts(k, means, boot, roots):
    """Retain every inverse branch; do not extrapolate to unsupported roots."""
    records = []
    for root in roots:
        value = float(root['g'][0])
        branches = []
        for s in range(means.shape[1]-1):
            for inv in inverse_branches(k, means[:, s], value):
                i, t = inv['segment'], inv['fraction']
                y = (1-t)*means[i, s+1] + t*means[i+1, s+1]
                a, b = boot[:, i, s], boot[:, i+1, s]
                valid = (np.minimum(a,b) <= value) & (value <= np.maximum(a,b)) & (np.abs(b-a)>1e-12)
                estimates = np.full(len(boot), np.nan)
                tboot = (value-a[valid])/(b[valid]-a[valid])
                estimates[valid] = (1-tboot)*boot[valid, i, s+1]+tboot*boot[valid, i+1, s+1]
                branches.append(dict(input_scale=s, **inv, empirical_next=float(y),
                    empirical_drift=float(y-value), valid_bootstrap_fraction=float(valid.mean()),
                    bootstrap_95=np.percentile(estimates[valid], [2.5,97.5]).tolist() if valid.any() else None,
                    _samples=estimates))
        pairs = []
        for i, a in enumerate(branches):
            for b in branches[i+1:]:
                if a['input_scale'] == b['input_scale']:
                    continue
                delta = a['_samples'] - b['_samples']
                good = np.isfinite(delta)
                pairs.append(dict(scales=[a['input_scale'], b['input_scale']],
                    inverse_K=[a['K'],b['K']], next_value_difference=a['empirical_next']-b['empirical_next'],
                    bootstrap_95=np.percentile(delta[good], [2.5,97.5]).tolist() if good.any() else None,
                    valid_bootstrap_fraction=float(good.mean())))
        for row in branches:
            del row['_samples']
        records.append(dict(g=value, branches=branches, cross_scale_differences=pairs))
    return records


def audit(run, output, bootstraps=1000):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    torch.set_num_threads(1)
    summary = json.loads((run/'summary.json').read_text())
    if summary.get('actual_epochs') != 12000:
        raise ValueError('This comparison requires the actual 12000-update endpoint')
    loaded = load_run(run, 'checkpoint_last.pt')
    if loaded.payload.get('epoch') != 12000 or loaded.config['model']['theory_dim'] != 1:
        raise ValueError('Expected final scalar checkpoint')
    model = loaded.model.eval()
    source = run/'full_report/encodings.npz'
    with np.load(source, allow_pickle=False) as arrays:
        values = np.concatenate([arrays['train'], arrays['test']])[:, :, 0]
        labels = np.concatenate([arrays['train_labels'], arrays['test_labels']]).ravel()
        train_k = set(np.unique(arrays['train_labels']).tolist())
    k = np.unique(labels)
    blocks = np.stack([values[labels == v] for v in k])
    if blocks.shape[:2] != (64, 12):
        raise ValueError('Expected 64 complete K points and 12 independent ensembles per K')
    means = blocks.mean(1)
    prediction, derivative = predict_and_derivative(model, blocks[:, :, :-1])
    mean_prediction = prediction.mean(1)
    r_at_mean, dr_at_mean = predict_and_derivative(model, means[:, :-1])
    rng = np.random.default_rng(12345)
    ids = rng.integers(0, 12, (bootstraps, len(k), 12))
    boot = blocks[np.arange(len(k))[None, :, None], ids].mean(2)
    rboot = prediction[np.arange(len(k))[None, :, None], ids].mean(2)
    roots = json.loads((run/'full_report/fixed_points.json').read_text())['roots']
    windows = [polynomial_response(k, means, boot, mean_prediction, rboot, KC, h)
               for h in (.02, .03, .04)]
    root_windows = []
    for root in roots:
        for inv in inverse_branches(k, means[:, 0], float(root['g'][0])):
            for h in (.02, .03, .04):
                root_windows.append(dict(root_g=float(root['g'][0]), inverse_K=inv['K'],
                    response=polynomial_response(k, means, boot, mean_prediction, rboot, inv['K'], h)))
    secants = []
    for s in range(means.shape[1]-1):
        dx = np.diff(means[:, s]); dy = np.diff(means[:, s+1]); dp = np.diff(mean_prediction[:, s])
        noise = np.sqrt(blocks[:, :, s].var(1, ddof=1)/12)
        signal_noise = np.abs(dx)/np.sqrt(noise[:-1]**2+noise[1:]**2)
        ratios=lambda v: [float(a/b) if abs(b)>1e-12 else None for a,b in zip(v,dx)]
        secants.append(dict(scale=s, K_midpoints=((k[:-1]+k[1:])/2).tolist(),
            empirical=ratios(dy), predicted=ratios(dp),
            input_difference_SNR=signal_noise.tolist(), reliable=(signal_noise>=3).tolist()))
    closure = {}
    for split in ('train', 'test', 'all'):
        mask = np.array([v in train_k for v in k]) if split=='train' else np.array([v not in train_k for v in k])
        if split=='all': mask[:] = True
        error = prediction[mask] - blocks[mask, :, 1:]
        closure[split] = dict(rmse=np.sqrt((error**2).mean((0,1))).tolist(),
            mean_bias=error.mean((0,1)).tolist())
    report = dict(status='complete', run=str(run), actual_epochs=12000, K=k.tolist(),
        source_sha256={name:file_fingerprint(run/name) for name in
            ('checkpoint_last.pt','manifest.json','summary.json','full_report/encodings.npz')},
        bootstrap=dict(repeats=bootstraps, seed=12345, unit='independent ensemble within each fixed K',
            pairing='same resample across scales and paired R predictions',
            limitation='Conditional on one model and saved C draws; not lambda/nu confidence intervals. No interpolation or polynomial bias included.'),
        methodology='Post-hoc all-K analysis. Compare d(E[R(g_k)])/dK with d(E[g_{k+1}])/dK; retain Jensen difference from R(E[g_k]). These path responses are not certified fixed-point thermal multipliers.',
        mean_g=means.tolist(), mean_R=mean_prediction.tolist(), R_of_mean=r_at_mean.tolist(),
        mean_full_DR=derivative.mean(1).tolist(), DR_at_mean=dr_at_mean.tolist(),
        jensen_gap=(mean_prediction-r_at_mean).tolist(), closure=closure,
        critical_windows=windows, root_windows=root_windows, secants=secants,
        root_closure_conflicts=root_closure_conflicts(k, means, boot, roots),
        conflict_caution='Piecewise-linear empirical mean-curve interpolation, retaining all branches; geometric overlap does not certify physical support. Finite-size and encoding noise may cause scale dependence.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'response.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    n = means.shape[1]-1
    fig, axes = plt.subplots(2,n,figsize=(5*n,8),squeeze=False,layout='constrained')
    for s in range(n):
        axes[0,s].plot(means[:,s],means[:,s+1],'.-',label='Saved mean trajectory')
        grid = np.linspace(means[:,s].min(),means[:,s].max(),301)
        rg,_ = predict_and_derivative(model,grid)
        axes[0,s].plot(grid,rg,label='R(g)'); axes[0,s].plot(grid,grid,'k:',label='Identity')
        axes[0,s].set(xlabel='g input',ylabel='g output',title=f'Scale {s} -> {s+1}')
        z=secants[s]; mid=np.array(z['K_midpoints']); good=np.array(z['reliable']) & (abs(mid-KC)<.05)
        axes[1,s].plot(mid[good],np.array(z['empirical'],dtype=float)[good],'o-',label='Data secant response')
        axes[1,s].plot(mid[good],np.array(z['predicted'],dtype=float)[good],'s-',label='R paired response')
        axes[1,s].axvline(KC,color='k',ls=':'); axes[1,s].set(xlabel='Original K',ylabel='Response ratio',title='Adjacent K pairs; input SNR >= 3')
        for ax in axes[:,s]: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('12000 endpoint: local closure response (not thermal eigenvalue estimates)')
    fig.savefig(output/'local_response.png',dpi=160); plt.close(fig)
    print(json.dumps(dict(output=str(output),critical_windows=windows)),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.run.resolve(),args.output.resolve())
