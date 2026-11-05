"""Fixed-weight AT representation audit: pooled ensembles and same-g statistics."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from scripts.audit_at_scalar_closure import interpolate


def statistics(z):
    c = z.argmax(-1).numpy()[0]
    sigma = np.array([-1, -1, 1, 1])[c]
    tau = np.array([-1, 1, -1, 1])[c]
    fields = (sigma, tau, sigma*tau)
    m = [np.abs(f.mean((1, 2))).mean() for f in fields]
    corr = [np.mean([np.mean(f*np.roll(f, 1, axis)) for axis in (1, 2)]) for f in fields]
    code = c + 4*np.roll(c, -1, 1) + 16*np.roll(c, -1, 2) + 64*np.roll(np.roll(c, -1, 1), -1, 2)
    hist = np.bincount(code.ravel(), minlength=256)/code.size
    return np.array(m+corr), hist


def main():
    torch.set_num_threads(2)
    folder = Path('results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91')
    run = load_run(folder); model = run.model.eval()
    tr, te = load_ensembles(run.config['data']['cache'])
    X = torch.cat([tr.configurations, te.configurations])
    K = torch.cat([tr.couplings, te.couplings])[:, 0].numpy()
    knots = np.unique(K)
    with np.load(folder/'direct_audit/arrays.npz') as a:
        old = a['direct'][0, :, :3]
    avg = np.stack([old[K == k].mean(0) for k in knots])
    pooled, stats, histograms = [], [], []
    with torch.no_grad():
        for k in knots:
            z = X[K == k].flatten(0, 1).unsqueeze(0).double()
            levels = model.coarse_grain(z, max_rg_steps=2)
            pooled.append([float(model.theory_encoder(v).item()) for v in levels])
            rows = [statistics(v) for v in levels]
            stats.append(np.stack([v[0] for v in rows]))
            histograms.append(np.stack([v[1] for v in rows]))
    pooled, stats, histograms = map(np.asarray, (pooled, stats, histograms))
    grid = np.linspace(3, 3.5, 201)
    old_curves = np.stack([interpolate(avg, grid, k) for k in range(2)])
    curves = np.stack([interpolate(pooled, grid, k) for k in range(2)])
    probes = []
    for value in (3.1, 3.2, 3.3, 3.4):
        rows, hs = [], []
        for step in range(2):
            order = np.argsort(pooled[:, step]); x = pooled[order, step]
            rows.append(dict(k=step, microscopic_K=float(np.interp(value, x, knots[order])),
                             next_g=float(np.interp(value, x, pooled[order, step+1])),
                             observables=[float(np.interp(value, x, stats[order, step, j])) for j in range(6)]))
            hs.append(np.array([np.interp(value, x, histograms[order, step, j]) for j in range(256)]))
        probes.append(dict(g=value, states=rows, pattern_total_variation=float(abs(hs[0]-hs[1]).sum()/2)))
    # Replica spread in the *same* local histograms is a descriptive noise baseline,
    # not a p-value; it includes Monte Carlo and sector sampling differences.
    p = probes[1]; baseline = []
    with torch.no_grad():
        for state in p['states']:
            k = knots[np.argmin(abs(knots-state['microscopic_K']))]
            hs = []
            for replica in X[K == k]:
                v = model.coarse_grain(replica[None].double(), max_rg_steps=2)[state['k']]
                hs.append(statistics(v)[1])
            baseline.append(dict(k=state['k'], nearest_K=float(k),
                                 replica_pair_TV=[float(abs(hs[i]-hs[j]).sum()/2) for i in range(3) for j in range(i+1, 3)]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for k, color in enumerate(('tab:blue', 'tab:orange')):
        ax[0].plot(grid, old_curves[k]-grid, '--', color=color, label=f'k={k}: mean of 3 Q(128)')
        ax[0].plot(grid, curves[k]-grid, color=color, label=f'k={k}: Q(pooled 384)')
    ax[0].set(xlabel='same g', ylabel='direct next-step drift', title='Does pooling remove scale splitting?')
    labels = ['|m_sigma|', '|m_tau|', '|m_product|', 'nn_sigma', 'nn_tau', 'nn_product']
    for step, color in enumerate(('tab:blue', 'tab:orange')):
        ax[1].bar(np.arange(6)+(step-.5)*.36, p['states'][step]['observables'], width=.36, color=color, label=f'k={step}, same g=3.2')
    ax[1].set_xticks(np.arange(6), labels, rotation=35, ha='right')
    ax[1].set(title='Different physical statistics at the same g', ylabel='observable')
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.2)
    out = folder/'closure_audit'
    fig.savefig(out/'representation_check.png', dpi=180)
    report = dict(probes=probes, histogram_noise_baseline=baseline,
                  max_Q_pooling_change=float(abs(pooled-avg).max()),
                  observable_order=labels,
                  caveats='No new independent samples: pooling the same 3 x 128 cached configurations tests finite-ensemble head bias, not all sampling errors. Same-g states use piecewise linear interpolation. Sizes 16 and 8 differ: finite-size effects remain confounded with representation insufficiency. Physical observables are diagnostics only, not model inputs. Pairwise histogram TV baseline has only three replicas.',
                  weights_changed=False)
    (out/'representation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
