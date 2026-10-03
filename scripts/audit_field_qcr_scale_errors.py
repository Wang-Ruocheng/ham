"""Read-only model audit: scale-resolved errors and coupling response from cached encodings."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, default=6000)
    args = parser.parse_args()
    folder = args.run/f'step{args.step}'
    diag = json.loads((folder/'field_diagnostics.json').read_text())
    cfg = diag['config']
    a = np.load(folder/'field_flow_data.npz')
    d, r, kb, test = [a[key] for key in ('direct', 'rollout', 'KB', 'is_test')]
    torch.set_num_threads(1)
    model = RecurrentRGNetwork2D(levels=cfg['levels'], hidden=cfg['hidden'], theory_dim=2,
                                learnable_kernel=cfg.get('learnable_kernel', True)).double().eval()
    ckpt = torch.load(args.run/f'training_step_{args.step}.pt', weights_only=True)
    mismatch = model.load_state_dict(ckpt['model'], strict=False)
    assert not mismatch.unexpected_keys
    assert all(k.startswith('action_network.') for k in mismatch.missing_keys)
    with torch.no_grad():
        one = model.advance_theory(torch.from_numpy(d[:, :-1])).numpy()
        np.testing.assert_allclose(model.advance_theory(torch.from_numpy(d[:, 0])).numpy(), r[:, 1], atol=1e-12)
    report = dict(step=args.step, test_ensembles=int(test.sum()),
                  convention='RMSE per coordinate; pooled NRMSE divided by target pooled SD; PC normalization uses held-out target covariance, diagnostic only.',
                  scales=[], regions={})
    for k in range(1, d.shape[1]):
        target = d[test, k]
        eig, axes = np.linalg.eigh(np.cov(target.T, bias=True))
        sd = np.sqrt(target.var(0).mean())
        row = dict(k=k, sites=cfg['sites']//2**k, target_pooled_sd=float(sd),
                   target_pc_sd_weak_strong=np.sqrt(eig).tolist())
        for name, prediction in (('one_step', one[:, k-1]), ('rollout', r[:, k])):
            error = prediction[test]-target
            rmse = np.sqrt(np.mean(error**2))
            pc_error = np.sqrt(np.mean((error@axes)**2, axis=0))
            row[name] = dict(rmse=float(rmse), nrmse=float(rmse/sd),
                             pc_rmse_weak_strong=pc_error.tolist(),
                             pc_nrmse_weak_strong=(pc_error/np.sqrt(eig)).tolist())
        report['scales'].append(row)
    np.testing.assert_allclose([x['rollout']['nrmse'] for x in report['scales']],
                               diag['metrics']['test']['rollout_nrmse'][1:])
    kc = .5*np.log1p(np.sqrt(2))
    for name, mask in [('B_zero', np.isclose(kb[:, 1], 0)),
                       ('B_nearest_nonzero', np.isclose(abs(kb[:, 1]), .05)),
                       ('B_large', abs(kb[:, 1]) >= .1-1e-12),
                       ('near_critical_B_zero', np.isclose(kb[:, 1], 0) & (abs(kb[:, 0]-kc)<.06))]:
        mask = test & mask
        rows = []
        for k in range(1, d.shape[1]):
            target = d[mask, k]
            sd = np.sqrt(target.var(0).mean())
            rows.append(dict(k=k, target_pooled_sd=float(sd),
                one_step_rmse=float(np.sqrt(np.mean((one[mask, k-1]-target)**2))),
                rollout_rmse=float(np.sqrt(np.mean((r[mask, k]-target)**2))),
                rollout_nrmse_within_region=float(np.sqrt(np.mean((r[mask, k]-target)**2))/max(sd, 1e-15))))
        report['regions'][name] = dict(ensembles=int(mask.sum()), scales=rows)

    # Fit axes on training data only; orient PC1 to increase with microscopic magnetization.
    center = d[~test, 0].mean(0)
    eig, axes = np.linalg.eigh(np.cov(d[~test, 0].T, bias=True))
    eig, axes = eig[::-1], axes[:, ::-1]
    pc = (d[:, 0]-center)@axes
    obs = a['observables']
    if np.corrcoef(pc[~test, 0], obs[~test, 1])[0, 1]<0:
        axes[:, 0] *= -1
        pc[:, 0] *= -1
    report['microscopic_pc'] = dict(axes_columns_pc1_pc2=axes.tolist(), sd=np.sqrt(eig).tolist(),
        second_variance_fraction=float(eig[1]/eig.sum()), correlations_test={})
    for name, target in [('K', kb[:, 0]), ('B', kb[:, 1]), ('sign_B', np.sign(kb[:, 1])),
                         ('magnetization', obs[:, 1]), ('bond_per_site', obs[:, 0])]:
        report['microscopic_pc']['correlations_test'][name] = [
            dict(pearson=float(np.corrcoef(pc[test, j], target[test])[0, 1]),
                 spearman=float(spearmanr(pc[test, j], target[test]).statistic)) for j in range(2)]
    # Frozen-representation post-hoc probes only: no change to Q, C, R.
    report['label_probe_test_R2'] = {}
    z = pc/np.sqrt(eig)
    for name, features in [('PC1_quadratic', np.column_stack((np.ones(len(z)), z[:, 0], z[:, 0]**2))),
                           ('PC1_PC2_quadratic', np.column_stack((np.ones(len(z)), z, z**2, z[:, 0]*z[:, 1])))]:
        coef = np.linalg.lstsq(features[~test], kb[~test], rcond=None)[0]
        error = features[test]@coef-kb[test]
        scores = 1-(error**2).sum(0)/((kb[test]-kb[test].mean(0))**2).sum(0)
        report['label_probe_test_R2'][name] = dict(K=float(scores[0]), B=float(scores[1]))

    means, ks, bs = a['means'], a['K_grid'], a['B_grid']
    pcs = (means[:, :, 0]-center)@axes
    report['fixed_K_B_slices'] = []
    for kval in (.2, .44, .68):
        i = int(np.argmin(abs(ks-kval)))
        rows = []
        for bval in (-.5, -.05, 0, .05, .5):
            j = int(np.argmin(abs(bs-bval)))
            mask = np.isclose(kb[:, 0], ks[i]) & np.isclose(kb[:, 1], bs[j])
            rows.append(dict(B=float(bs[j]), g=means[i, j, 0].tolist(),
                             pc1_pc2=pcs[i, j].tolist(), m=float(obs[mask, 1].mean()),
                             bond_per_site=float(obs[mask, 0].mean())))
        report['fixed_K_B_slices'].append(dict(K=float(ks[i]), values=rows))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    kvals = [x['k'] for x in report['scales']]
    for name in ('one_step', 'rollout'):
        axs[0].plot(kvals, [100*x[name]['nrmse'] for x in report['scales']], 'o-', label=name)
        for j, pcname in enumerate(('weak PC', 'strong PC')):
            axs[1].plot(kvals, [100*x[name]['pc_nrmse_weak_strong'][j] for x in report['scales']],
                        'o-' if j==0 else 'x--', label=f'{name}: {pcname}')
    for name in ('B_zero', 'B_nearest_nonzero', 'B_large'):
        axs[2].plot(kvals, [x['rollout_rmse'] for x in report['regions'][name]['scales']], 'o-', label=name)
    axs[0].set(title='Held-out pooled errors', ylabel='NRMSE (%)')
    axs[1].set(title='Direction-resolved errors', ylabel='PC NRMSE (%)')
    axs[2].set(title='Error by microscopic field region', ylabel='Rollout RMSE (raw g units)')
    for ax in axs:
        ax.set(xlabel='Target scale k', xticks=kvals)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'2D Ising, seed {cfg["seed"]}, step {args.step}: no retraining')
    fig.tight_layout()
    fig.savefig(folder/'R_error_by_scale.png', dpi=190)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for j in range(2):
        for kval in (.2, .44, .68):
            i = int(np.argmin(abs(ks-kval)))
            axs[j, 0].plot(bs, pcs[i, :, j], '.-', label=f'K={ks[i]:.3f}')
        for bval in (0, .05, .5):
            i = int(np.argmin(abs(bs-bval)))
            axs[j, 1].plot(ks, pcs[:, i, j], '.-', label=f'B={bs[i]:.2f}')
        axs[j, 0].set(xlabel='B', ylabel=f'PC{j+1} (raw latent units)', title=f'PC{j+1}: varying B at fixed K')
        axs[j, 1].set(xlabel='K', ylabel=f'PC{j+1} (raw latent units)', title=f'PC{j+1}: varying K at fixed B')
        axs[j, 1].axvline(kc, c='gray', ls='--', lw=1)
        for ax in axs[j]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('Microscopic encoding: PCA fitted on train only; cell means over 3 ensembles; panel scales differ')
    fig.tight_layout()
    fig.savefig(folder/'coupling_response_PC.png', dpi=190)
    plt.close(fig)
    (folder/'scale_and_coupling_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
