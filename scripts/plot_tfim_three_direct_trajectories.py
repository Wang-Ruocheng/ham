"""Follow the SAME microscopic ensemble through C, then apply Q at each scale."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads((args.run/'summary.json').read_text())['config']
    assert cfg['theory_dim'] in (1, 2) and cfg['levels'] == 4
    model = RecurrentRGNetwork1D(levels=4, hidden=cfg['hidden'], theory_dim=cfg['theory_dim'],
                                kernel_layout=cfg['kernel_layout']).double().eval()
    mismatch = model.load_state_dict(torch.load(args.run/'qcr_model.pt', weights_only=True), strict=False)
    assert not mismatch.unexpected_keys and all(k.startswith('action_network.') for k in mismatch.missing_keys)
    tx, th, vx, vh = torch.load(args.cache, weights_only=True)
    x, h = torch.cat((tx, vx)), torch.cat((th, vh)).numpy()
    records = []
    cached_path = args.run/'vector_flow_data.npz'
    cached = np.load(cached_path) if cached_path.exists() else None
    for requested in (.5, 1., 1.5):
        index = int(np.argmin(abs(h-requested)))
        with torch.no_grad():
            levels = model.coarse_grain(balanced_views(x[index:index+1], x.shape[1]))
            trajectory = torch.stack([model.theory_encoder(d)[0] for d in levels]).numpy()
        if cached is not None:
            np.testing.assert_allclose(trajectory, cached['direct'][index], rtol=1e-10, atol=1e-12)
        records.append({'requested_h_over_J': requested, 'actual_h_over_J': float(h[index]),
                        'concatenated_cache_index': index, 'replica_index': 0,
                        'split': 'train' if index < len(tx) else 'test',
                        'g_by_scale': trajectory.tolist(),
                        'sites_by_scale': [d.shape[2] for d in levels]})
    if cfg['theory_dim'] == 1:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, record in enumerate(records):
            g = np.array(record['g_by_scale'])[:, 0]
            ax.plot(range(4), g, 'o-', color=f'C{i}', lw=2,
                    label=f"h/J = {record['actual_h_over_J']:.6f} ({record['split']})")
            for k, value in enumerate(g):
                ax.annotate(f'{value:.3f}', (k, value), xytext=(0, 9),
                            textcoords='offset points', ha='center', color=f'C{i}')
        ax.set(xlabel='Coarse-graining step k (128 -> 64 -> 32 -> 16 sites)',
               ylabel='$g_k=Q(C^k D)$', xticks=range(4),
               title='1D TFIM, dim(g)=1: same ensemble across four scales')
        ax.set_xticklabels(['$g_0$', '$g_1$', '$g_2$', '$g_3$'])
        ax.margins(x=.12, y=.18)
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        fig.text(.5, .02, 'One saved replica per h/J; 128 configurations + global flips. No R rollout or interpolation.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(args.run/'Q_three_ensembles_g0_g3.png', dpi=200)
        plt.close(fig)
        report = {'run': str(args.run), 'data_cache': str(args.cache),
                  'definition': 'g_k=Q(C^k D), not R^k(g_0)',
                  'selection': 'First saved replica at nearest available h/J; no averaging between replicas',
                  'configurations_per_initial_ensemble': int(x.shape[1]),
                  'global_flip_augmentation': True, 'records': records}
        (args.run/'Q_three_ensembles_g0_g3.json').write_text(json.dumps(report, indent=2)+'\n')
        rows = [[r['actual_h_over_J'], k, *g] for r in records for k, g in enumerate(r['g_by_scale'])]
        np.savetxt(args.run/'Q_three_ensembles_g0_g3.csv', rows, delimiter=',', header='h_over_J,k,g', comments='')
        print(json.dumps(report, indent=2))
        return
    fig = plt.figure(figsize=(14, 8.4))
    grid = fig.add_gridspec(2, 6, height_ratios=(1, .85))
    for i, record in enumerate(records):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2])
        g = np.array(record['g_by_scale'])
        ax.plot(*g.T, color=f'C{i}', lw=1, alpha=.4)
        for k in range(3):
            ax.annotate('', g[k+1], g[k], arrowprops={'arrowstyle': '->', 'color': f'C{i}', 'lw': 1.5,
                                                   'shrinkA': 6, 'shrinkB': 6})
        for k, point in enumerate(g):
            ax.scatter(*point, color=f'C{i}', marker=('o', 's', '^', 'D')[k], s=50, zorder=4)
            ax.annotate(f'$g_{k}$', point, xytext=(7, 7), textcoords='offset points', fontsize=11)
        ax.set(title=f"h/J = {record['actual_h_over_J']:.6f} ({record['split']})",
               xlabel='$g^{(1)}$', ylabel='$g^{(2)}$')
        ax.margins(.25)
        ax.set_aspect('equal', adjustable='box')
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.grid(alpha=.2)
    for d in range(2):
        ax = fig.add_subplot(grid[1, 3*d:3*d+3])
        for i, record in enumerate(records):
            g = np.array(record['g_by_scale'])
            ax.plot(range(4), g[:, d], 'o-', color=f'C{i}', label=f"h/J={record['actual_h_over_J']:.6f}")
        ax.set(xlabel='Coarse-graining step k (128 -> 64 -> 32 -> 16 sites)', ylabel=f'$g^{{({d+1})}}_k$', xticks=range(4))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('1D TFIM, dim(g)=2: actual $g_k=Q(C^k D)$ for three initial ensembles', fontsize=15)
    fig.text(.5, .015, 'One saved replica per h/J; 128 configurations + global flips. No R rollout and no interpolation. Top panels use different axis ranges.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(args.run/'Q_three_ensembles_g0_g3.png', dpi=200)
    plt.close(fig)
    report = {'run': str(args.run), 'data_cache': str(args.cache), 'definition': 'g_k=Q(C^k D), not R^k(g_0)',
              'selection': 'First saved replica at nearest available h/J; no averaging between replicas',
              'configurations_per_initial_ensemble': int(x.shape[1]), 'global_flip_augmentation': True,
              'records': records}
    (args.run/'Q_three_ensembles_g0_g3.json').write_text(json.dumps(report, indent=2)+'\n')
    rows = [[r['actual_h_over_J'], k, *g] for r in records for k, g in enumerate(r['g_by_scale'])]
    np.savetxt(args.run/'Q_three_ensembles_g0_g3.csv', rows, delimiter=',', header='h_over_J,k,g_component_1,g_component_2', comments='')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
