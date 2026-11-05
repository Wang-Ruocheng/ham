"""Evaluate frozen Q ablations on fresh probabilistic-C draws, without training.

The microscopic held-out configurations are reused. Fresh draws therefore test
stochastic coarse-graining robustness, not new physical sampling or a new seed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.distribution_encoder import LocalDistributionEncoder
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts


ARMS = (
    'original', 'graph_coordinate', 'neural_linear_reconstruction',
    'histogram_linear_reconstruction',
)


def validate(run, *, draws=8, first_seed=9200, subset_seed=9300):
    run = Path(run)
    if not (run/'metrics.json').exists():
        raise FileNotFoundError('The four-arm training report must be complete first')
    if draws < 2:
        raise ValueError('At least two independent C draws are required')
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = json.loads((run/'manifest.json').read_text())
    source = load_run(manifest['source'])
    if file_fingerprint(source.checkpoint) != manifest['source_sha256']:
        raise ValueError('Source checkpoint fingerprint mismatch')
    datafile = source.config['data']['cache']
    if file_fingerprint(datafile) != manifest['data_sha256']:
        raise ValueError('Physical data fingerprint mismatch')
    _, test = load_ensembles(datafile)
    model = source.model
    if (model.geometry.dimensions != 2 or model.theory_dim != 1
            or model.theory_encoder.categorical_states != 3):
        raise ValueError('This validation is for the scalar signed spin-1 ablation')
    if model.rg_kernel.field_head.sampling_mode != 'categorical':
        raise ValueError('Fresh-draw validation requires a stochastic categorical C')
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    qs = {}
    for name in ARMS:
        checkpoint = run/(name+'.pt')
        if name == 'histogram_linear_reconstruction':
            q = LocalDistributionEncoder(3, 2).double()
            q.load_state_dict(torch.load(checkpoint, weights_only=True)['Q'])
        else:
            loaded = load_run(checkpoint)
            q = loaded.model.theory_encoder
            for key, value in model.rg_kernel.state_dict().items():
                torch.testing.assert_close(value, loaded.model.rg_kernel.state_dict()[key],
                                           atol=0, rtol=0)
        q.eval()
        for parameter in q.parameters():
            parameter.requires_grad_(False)
        qs[name] = q

    codes = (torch.arange(3**9)[:, None]//(3**torch.arange(9))) % 3
    patterns = torch.nn.functional.one_hot(codes, 3).flatten(1).double()
    with torch.no_grad():
        # Each frozen table is reused for every ensemble, scale and draw.
        tables = {name: q.local_mlp(patterns).numpy()
                  for name, q in qs.items() if name != 'histogram_linear_reconstruction'}
    nsamples = test.configurations.shape[1]
    if nsamples != 64:
        raise ValueError('The paired source validation expects 64 samples per ensemble')
    generator = torch.Generator().manual_seed(subset_seed)
    permutation = torch.randperm(nsamples, generator=generator).numpy()
    groups = (np.arange(nsamples), permutation[:32], permutation[32:])

    def predict_level(configurations):
        counts = sample_counts(configurations, model.geometry, 3)
        histograms = np.stack([np.asarray(counts[group].mean(0)).ravel() for group in groups])
        distributions = categorical_plaquette_distribution(
            configurations.squeeze(0).unsqueeze(1)).numpy()
        descriptor = torch.from_numpy(np.stack([distributions[group].mean(0) for group in groups]))
        values = {}
        for name, q in qs.items():
            if name == 'histogram_linear_reconstruction':
                g = q.from_distribution(descriptor)
            else:
                g = q.head(torch.from_numpy(histograms@tables[name]))
            values[name] = g.numpy().ravel()
        return values

    nsystems = len(test.configurations)
    encodings = {name: np.empty((draws, nsystems, 3, 3), dtype=np.float64) for name in ARMS}
    with torch.no_grad():
        microscopic = [predict_level(x[None].double()) for x in test.configurations]
        for draw in range(draws):
            torch.manual_seed(first_seed+draw)
            for system, x in enumerate(test.configurations):
                levels = model.coarse_grain(x[None].double())
                if len(levels) != 3:
                    raise ValueError('Expected exactly three evaluation scales')
                predictions = [microscopic[system]] + [predict_level(v) for v in levels[1:]]
                for name in ARMS:
                    encodings[name][draw, system] = np.stack([p[name] for p in predictions])
            print(f'Fresh C draw {draw+1}/{draws}; elapsed {time.monotonic()-started:.1f}s', flush=True)

    labels = test.couplings.numpy().reshape(nsystems, -1)[:, 0]
    unique = np.unique(labels)
    results = {}
    for name, values in encodings.items():
        scales = []
        for level in range(3):
            g = values[:, :, level, 0]
            signed = np.array([spearmanr(labels, sample).statistic for sample in g])
            sign = 1 if np.mean(signed) >= 0 else -1
            oriented = sign*g
            left, right = oriented[:, labels < -.6], oriented[:, labels > .6]
            subset = np.sqrt(np.mean(np.diff(values[:, :, level, 1:], axis=-1)**2, axis=(1, 2)))
            sd = g.std(axis=1)
            curves = np.stack([g[:, labels == k].mean(1) for k in unique], axis=1)
            delta = np.diff(curves, axis=1)
            reverse = np.minimum(delta.clip(min=0).sum(1), (-delta).clip(min=0).sum(1))
            reverse /= np.maximum(abs(delta).sum(1), 1e-12)
            measures = dict(abs_spearman_K=abs(signed),
                ordered_branch_gap=right.min(1)-left.max(1),
                ordered_centroid_distance=right.mean(1)-left.mean(1),
                subset_relative_RMS=subset/np.maximum(sd, 1e-12),
                reverse_variation_fraction=reverse)
            summary = {key: dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))
                       for key, a in measures.items()}
            draw_rms = float(np.sqrt(np.mean(g.var(axis=0, ddof=0))))
            summary.update(draw_to_draw_RMS=draw_rms,
                draw_to_draw_relative_RMS=draw_rms/max(float(g.mean(0).std()), 1e-12))
            scales.append(summary)
        results[name] = dict(scales=scales)
    report = dict(run=str(run), epochs=json.loads((run/'metrics.json').read_text())['completed_epochs'],
        source=str(source.checkpoint), source_sha256=manifest['source_sha256'],
        data_sha256=manifest['data_sha256'], draws=draws,
        C_draw_seeds=list(range(first_seed, first_seed+draws)),
        subset_seed=subset_seed, heldout_systems=nsystems, samples_per_system=nsamples,
        subset_protocol='One common random 32/32 split, fixed across all arms and C draws',
        labels_use='Held-out diagnostics only; no fitting or selection', results=results,
        elapsed_seconds=time.monotonic()-started,
        limitations='Same held-out microscopic configurations and frozen C weights; only coarse-graining randomness is new. One Q-training seed; source R not refitted.')
    (run/'fresh_C_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(run/'fresh_C_validation_encodings.npz', test_K=labels,
                        **{name: value for name, value in encodings.items()})
    plot(run, labels, encodings)
    print(json.dumps(report, indent=2), flush=True)
    return report


def plot(run, labels, encodings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    unique = np.unique(labels)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (name, values) in zip(axes.flat, encodings.items()):
        for level in range(3):
            curves = np.stack([values[:, labels == k, level, 0].mean(1) for k in unique], axis=1)
            ax.plot(unique, curves.mean(0), '-o', ms=3, label=f'k={level}', color=f'C{level}')
            ax.fill_between(unique, curves.min(0), curves.max(0), alpha=.2, color=f'C{level}')
        ax.set(title=name.replace('_', ' '), xlabel='Held-out K (evaluation only)', ylabel='g')
        ax.legend(); ax.grid(alpha=.2)
    draws = next(iter(encodings.values())).shape[0]
    figure.suptitle(f'Frozen Q on {draws} fresh probabilistic C draws: mean and min/max bands')
    figure.savefig(run/'fresh_C_validation.png', dpi=170)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/spin1_Qunfolding_1200_seed91'))
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--first-seed', type=int, default=9200)
    parser.add_argument('--subset-seed', type=int, default=9300)
    arguments = parser.parse_args()
    validate(arguments.run, draws=arguments.draws, first_seed=arguments.first_seed,
             subset_seed=arguments.subset_seed)
