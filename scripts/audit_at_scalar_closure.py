"""Read-only scale-closure diagnosis; bootstrap independent cached replicas, not sites."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles


def interpolate(points, grid, step):
    order = np.argsort(points[:, step])
    x, y = points[order, step], points[order, step + 1]
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def main():
    torch.set_num_threads(2)
    folder = Path('results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91')
    run = load_run(folder)
    model = run.model.eval()
    train, test = load_ensembles(run.config['data']['cache'])
    with np.load(folder / 'direct_audit/arrays.npz') as a:
        K, direct = a['K'], a['direct'][0, :, :3]
    expected = torch.cat([train.couplings, test.couplings])[:, 0].numpy()
    if not np.array_equal(K, expected):
        raise ValueError('Cached diagnostic ordering differs from dataset')
    knots = np.unique(K)
    replicas = np.stack([direct[K == k] for k in knots])
    means = replicas.mean(1)
    # Audit a shared-input interval around the previously discussed g=3.2.
    grid = np.linspace(3.0, 3.5, 201)
    curves = np.stack([interpolate(means, grid, k) for k in range(2)])
    rng = np.random.default_rng(91)
    boot = []
    for _ in range(2000):
        picks = rng.integers(0, replicas.shape[1], size=replicas.shape[:2])
        average = np.take_along_axis(replicas, picks[..., None], axis=1).mean(1)
        boot.append([interpolate(average, grid, k) for k in range(2)])
    boot = np.asarray(boot)
    ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    delta = curves[0] - curves[1]
    delta_ci = np.nanpercentile(boot[:, 0] - boot[:, 1], [2.5, 97.5], axis=0)
    with torch.no_grad():
        predicted = model.advance_theory(torch.from_numpy(grid[:, None])).numpy()[:, 0]
        r_data = model.advance_theory(torch.from_numpy(direct[:, :2, None])).numpy()[..., 0]
    probes = []
    for value in (3.1, 3.2, 3.3, 3.4):
        i = np.argmin(abs(grid - value))
        probes.append(dict(g=float(grid[i]), targets=curves[:, i].tolist(),
                           difference=float(delta[i]), difference_ci95=delta_ci[:, i].tolist(),
                           R=float(predicted[i]), R_errors=(predicted[i] - curves[:, i]).tolist(),
                           equal_weight_best_possible_rmse=float(abs(delta[i]) / 2)))
    # Non-parametric diagnostic only: scale-specific interpolation calibrated
    # on training couplings and tested on wholly held-out coupling groups.
    ntrain = len(train.configurations)
    trK = K[:ntrain]
    train_means = np.stack([direct[:ntrain][trK == k].mean(0) for k in np.unique(trK)])
    diagnostics = {}
    for region, mask in [('all', np.ones(len(K), dtype=bool)),
                         ('transition_window', (K >= .17) & (K <= .245))]:
        rows = []
        for k in range(2):
            interp = interpolate(train_means, direct[:, k], k)
            selected = mask & (np.arange(len(K)) >= ntrain) & np.isfinite(interp)
            target = direct[selected, k + 1]
            rows.append(dict(n=int(selected.sum()),
                             R_rmse=float(np.sqrt(np.mean((r_data[selected, k] - target)**2))),
                             scale_specific_interpolation_rmse=float(np.sqrt(np.mean((interp[selected] - target)**2)))))
        diagnostics[region] = rows
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for k, color in enumerate(('tab:blue', 'tab:orange')):
        axes[0].scatter(direct[:, k], direct[:, k+1], s=9, alpha=.3, color=color)
        axes[0].plot(means[:, k], means[:, k+1], color=color, label=f'direct k={k} -> {k+1}')
        axes[1].plot(grid, curves[k] - grid, color=color, label=f'direct k={k} -> {k+1}')
        axes[1].fill_between(grid, ci[0, k]-grid, ci[1, k]-grid, color=color, alpha=.2)
    wide = np.linspace(direct[:, :2].min(), direct[:, :2].max(), 500)
    with torch.no_grad():
        wr = model.advance_theory(torch.from_numpy(wide[:, None])).numpy()[:, 0]
    axes[0].plot(wide, wr, 'k--', label='trained R')
    axes[0].plot(wide, wide, ':', color='.6', label='identity')
    axes[0].set(xlabel='g_k', ylabel='g_(k+1)', title='All cached ensembles; fixed Q/C')
    axes[1].plot(grid, predicted-grid, 'k--', label='trained R(g)-g')
    axes[1].set(xlabel='same input g', ylabel='next-step drift', title='Shared-input zoom; 95% replica bootstrap')
    axes[2].plot(grid, delta, color='tab:purple')
    axes[2].fill_between(grid, *delta_ci, color='tab:purple', alpha=.2)
    axes[2].axhline(0, color='k', lw=.7)
    axes[2].set(xlabel='same input g', ylabel='f_0(g)-f_1(g)', title='Scale discrepancy (not R fitting error)')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    out = folder / 'closure_audit'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'scale_closure.png', dpi=180)
    report = dict(probes=probes, heldout=diagnostics,
                  bootstrap='2000 paired replica resamples within each K, 3 replicas per K; no resampling of spatial sites. Pointwise intervals, not simultaneous bands. Limited replication, finite-L bias and interpolation bias not covered.',
                  interpretation='Scale-conditioned interpolation is a diagnostic, not a trained replacement RG. Positive same-input discrepancy prevents exact deterministic scalar closure of these estimated mean curves.',
                  weights_changed=False)
    (out / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
