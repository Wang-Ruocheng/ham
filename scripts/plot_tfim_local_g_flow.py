"""Distinguish drift from initial-condition sensitivity near g=-0.54."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, default=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2')
    parser.add_argument('--center', type=float)
    args = parser.parse_args()
    out = args.run
    cfg = json.loads((out/'summary.json').read_text())['config']
    model = RecurrentRGNetwork1D(levels=4, hidden=cfg['hidden'], theory_dim=1,
                                kernel_layout=cfg['kernel_layout']).double().eval()
    missing = model.load_state_dict(torch.load(out/'qcr_model.pt', weights_only=True), strict=False)
    assert not missing.unexpected_keys and all(s.startswith('action_network.') for s in missing.missing_keys)
    cache = root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt'
    tx, th, vx, vh = torch.load(cache, weights_only=True)
    x, h = torch.cat((tx, vx)), torch.cat((th, vh)).numpy()
    direct = []
    with torch.no_grad():
        for chunk in x.split(4):
            levels = model.coarse_grain(balanced_views(chunk, x.shape[1]))
            direct.append(torch.stack([model.theory_encoder(d)[:, 0] for d in levels], dim=1).numpy())
    direct = np.concatenate(direct)
    knots = np.unique(h)
    means = np.array([direct[h == value, 0].mean() for value in knots])
    center = args.center if args.center is not None else float(np.interp(1., knots, means))
    below = np.flatnonzero(direct[:, 0] < center)
    above = np.flatnonzero(direct[:, 0] >= center)
    selected = np.r_[below[np.argsort(abs(direct[below, 0]-center))[:3]],
                     above[np.argsort(abs(direct[above, 0]-center))[:3]]]
    selected = selected[np.argsort(direct[selected, 0])]
    def rollout(values, steps):
        z = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
        ys = [z[:, 0].numpy()]
        with torch.no_grad():
            for _ in range(steps):
                z = model.advance_theory(z)
                ys.append(z[:, 0].numpy())
        return np.stack(ys, axis=1)
    initials = center + np.array([-.10, -.05, -.01, 0, .01, .05, .10])
    curves = rollout(initials, 20)
    ref = curves[3]
    z0 = torch.tensor([[center]], dtype=torch.float64, requires_grad=True)
    z = z0
    derivatives = [1.]
    for k in range(20):
        z = model.advance_theory(z)
        derivatives.append(float(torch.autograd.grad(z.sum(), z0, retain_graph=True)[0].item()))
    predicted = rollout(direct[selected, 0], 3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, value in enumerate(initials):
        axes[0, 0].plot(range(21), curves[i], '.-', lw=1.4, label=f'g0={value:.2f}')
        if i != 3:
            axes[1, 0].plot(range(21), curves[i]-ref, '.-', lw=1,
                            label=f'delta0={value-center:+.2f}')
    axes[0, 0].axhline(center, color='gray', ls=':', label=f'{center:.4f} reference')
    axes[0, 0].set(title=f'Learned R rollout near g0={center:.4f}', xlabel='Iteration k', ylabel='$R^k(g_0)$')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[1, 0].axhline(0, color='gray', lw=.8)
    axes[1, 0].set(title='Subtract the MOVING reference trajectory', xlabel='Iteration k',
                  ylabel=r'$R^k(g_0)-R^k(g_{\mathrm{ref}})$')
    axes[1, 0].legend(fontsize=8, ncol=2)
    for j, idx in enumerate(selected):
        label = f'g0={direct[idx,0]:+.3f}, h/J={h[idx]:.4f}'
        axes[0, 1].plot(range(4), direct[idx], 'o-', color=f'C{j}', label=label)
        axes[0, 1].plot(range(4), predicted[j], '--', color=f'C{j}', alpha=.7)
    axes[0, 1].set(title='Actual ensembles: solid Q(C^kD), dashed R^k(g0)', xlabel='k (128 -> 64 -> 32 -> 16 sites)',
                  ylabel='$g_k$', xticks=range(4))
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].plot(range(21), derivatives, 'o-')
    axes[1, 1].axhline(1, color='gray', ls='--')
    axes[1, 1].set(title='Local amplification along reference trajectory', xlabel='Iteration k',
                  ylabel=r'$\partial R^k(g_0)/\partial g_0\ |_{g_0=g_{\mathrm{ref}}}$')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('TFIM dim(g)=1: drift and sensitivity are different questions', fontsize=15)
    fig.text(.5, .02, 'Real ensembles: three nearest g0 values on each side, selected without interpolation.\n'
             'Only k=0..3 have direct coarse-grained targets; longer R rollouts are unvalidated iterations of the fitted map.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    stem = 'local_g_minus0p54_flow' if args.center == -.54 else 'local_g_critical_flow'
    fig.savefig(out/f'{stem}.png', dpi=200)
    plt.close(fig)
    records = [{'index': int(idx), 'h_over_J': float(h[idx]), 'direct': direct[idx].tolist(),
                'R_rollout': predicted[j].tolist(), 'split': 'train' if idx < len(tx) else 'test'}
               for j, idx in enumerate(selected)]
    report = {'center': center, 'initials': initials.tolist(), 'R_rollouts': curves.tolist(),
              'reference': ref.tolist(), 'derivatives': derivatives,
              'R_center_minus_center': float(ref[1]-center), 'real_ensembles': records,
              'notes': 'No interpolation of ensembles. R-only k>3 is outside directly checked depth.'}
    (out/f'{stem}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
