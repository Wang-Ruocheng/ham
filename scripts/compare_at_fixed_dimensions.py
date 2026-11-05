"""Matched-cache scalar/vector AT closure diagnostics, no optimization."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vector-run',type=Path,default=Path('results/ashkin_teller2d_g2_middle_dense_fixed_decimation_1200_seed91'))
    root = parser.parse_args().vector_run
    out = root/'dimension_audit'; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    report = {}
    fingerprints = []
    for dim in (1, 2):
        folder = root if dim==2 else Path('results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91')
        run = load_run(folder); model = run.model.eval()
        manifest = json.loads((folder/'manifest.json').read_text())
        fingerprints.append(manifest['data_sha256'])
        tr, te = load_ensembles(run.config['data']['cache'])
        train = collect_trajectories(model, tr.configurations, samples=128, max_rg_steps=2)
        test = collect_trajectories(model, te.configurations, samples=128, max_rg_steps=2)
        d = np.concatenate([train['direct'], test['direct']])
        K = torch.cat([tr.couplings, te.couplings])[:, 0].numpy()
        knots = np.unique(K); means = np.stack([d[K == k].mean(0) for k in knots])
        trainK = tr.couplings[:, 0].numpy(); trainknots = np.unique(trainK)
        tm = np.stack([train['direct'][trainK == k].mean(0) for k in trainknots])
        # Noise scale from replica fluctuations, not from forced whitening.
        dev = d - np.stack([means[np.argmin(abs(knots-k))] for k in K])
        replica_rms = float(np.sqrt(np.mean(np.sum(dev[:, :2]**2, axis=-1))))
        eigen = np.linalg.eigvalsh(np.cov(d[:, :2].reshape(-1, dim).T, bias=True).reshape(dim, dim))
        window = (te.couplings[:, 0].numpy() >= .17) & (te.couplings[:, 0].numpy() <= .245)
        metrics = closure_metrics(test, reference_direct=train['direct'])
        local = closure_metrics({k:v[window] for k,v in test.items()}, reference_direct=train['direct'])
        # Test whether a piecewise linear decoder of the TWO training scale
        # curves can predict held-out successors without receiving scale labels.
        # Nearest segment in full latent Euclidean distance; no test calibration.
        errors = []; same_scale = []
        for conditioned in (False, True):
            preds = []
            for step in range(2):
                sources = (step,) if conditioned else (0, 1)
                a = np.concatenate([tm[:-1, k] for k in sources])
                b = np.concatenate([tm[1:, k] for k in sources])
                ya = np.concatenate([tm[:-1, k+1] for k in sources])
                yb = np.concatenate([tm[1:, k+1] for k in sources])
                x = test['direct'][:, step]
                delta = b-a
                f = np.clip(((x[:, None]-a)*delta).sum(-1)/np.maximum((delta*delta).sum(-1), 1e-20), 0, 1)
                dist = np.linalg.norm(x[:, None]-(a+f[..., None]*delta), axis=-1)
                ids = dist.argmin(1)
                pred = ya[ids]+f[np.arange(len(x)), ids, None]*(yb[ids]-ya[ids])
                preds.append(pred)
            pred = np.stack(preds, 1)
            err = np.sqrt(np.mean((pred-test['direct'][:, 1:])**2, axis=(0, 2)))
            errors.append(err.tolist())
        report[str(dim)] = dict(parameters=sum(p.numel() for p in model.parameters()),
                               heldout=metrics, transition_window=local,
                               input_covariance_eigenvalues=eigen.tolist(), replica_rms=replica_rms,
                               scale_blind_segment_decoder_rmse=errors[0],
                               scale_conditioned_segment_decoder_rmse=errors[1])
        np.savez_compressed(out/f'g{dim}_trajectories.npz', K=K, direct=d, test_rollout=test['rollout'], test_one_step=test['one_step'])
        if dim == 2:
            centered = means[:, :2].reshape(-1, 2)
            origin = centered.mean(0)
            _, _, basis = np.linalg.svd(centered-origin, full_matrices=False)
            for step, color in enumerate(('tab:blue', 'tab:orange', 'tab:green')):
                axes[0, 0].plot(*means[:, step].T, 'o-', color=color, ms=3, label=f'k={step}')
                pc = (means[:, step]-origin)@basis.T
                axes[0, 1].plot(*pc.T, 'o-', color=color, ms=3, label=f'k={step}')
                for j in range(2):
                    axes[1, j].plot(knots, means[:, step, j], color=color, label=f'k={step}')
            axes[0, 0].set(xlabel='g1', ylabel='g2', title='Direct Q curves at each scale')
            axes[0, 1].set(xlabel='PC1', ylabel='PC2 (unwhitened)', title='Rotated view: is scale separation resolved?')
            for j in range(2): axes[1, j].set(xlabel='microscopic K', ylabel=f'g{j+1}', title='Coupling map at every observed scale')
    if fingerprints[0] != fingerprints[1]: raise ValueError('Caches do not match')
    for ax in axes.ravel(): ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'scale_curves_and_K.png', dpi=180)
    h = json.loads((root/'history_progress.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for split, style in [('train', '-'), ('test', '--')]:
        epochs = [row['epoch'] for row in h]
        ax[0].plot(epochs, [row[split+'_total'] for row in h], style, label=split)
        for name, weight in [('anchor', 2), ('coarse_consistency', 5), ('separation', 10), ('commutator', 5), ('rollout', 5)]:
            ax[1].plot(epochs, [weight*row[split+'_'+name] for row in h], style, label=f'{split} {name}')
    for a in ax: a.set(xlabel='update', ylabel='loss', yscale='log'); a.legend(fontsize=7); a.grid(alpha=.2)
    ax[0].set_title('Train/test total'); ax[1].set_title('Weighted components (anchor includes internal x10)')
    fig.savefig(out/'loss.png', dpi=180)
    report['caveat'] = 'Single seed, same cache and budget, changed latent dimension/parameter shapes. Segment decoders are diagnostics calibrated only on training K; Euclidean projection and sparse sampling may be ill-conditioned on nearly overlapping curves. No claim of population closure or infinite-scale phase recovery.'
    (out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
