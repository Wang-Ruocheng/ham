"""Plot a bounded close-up of the discrete RG displacement from saved results."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, default=6000)
    parser.add_argument('--xlim', type=float, nargs=2, default=(0., 1.))
    parser.add_argument('--ylim', type=float, nargs=2, default=(0., .5))
    args = parser.parse_args()
    if args.xlim[0] >= args.xlim[1] or args.ylim[0] >= args.ylim[1]:
        parser.error('Axis bounds must be increasing')
    torch.set_num_threads(1)
    folder = args.run/f'step{args.step}'
    report = json.loads((folder/'field_diagnostics.json').read_text())
    cfg = report['config']
    model = RecurrentRGNetwork2D(levels=cfg['levels'], hidden=cfg['hidden'],
                                theory_dim=2, learnable_kernel=cfg.get('learnable_kernel', True)).double().eval()
    checkpoint = torch.load(args.run/f'training_step_{args.step}.pt', weights_only=True)
    assert checkpoint['epoch'] == args.step
    mismatch = model.load_state_dict(checkpoint['model'], strict=False)
    assert not mismatch.unexpected_keys
    assert all(k.startswith('action_network.') for k in mismatch.missing_keys)
    arrays = np.load(folder/'field_flow_data.npz')
    inputs = arrays['direct'][:, :-1].reshape(-1, 2)
    x, y = np.meshgrid(np.linspace(*args.xlim, 41), np.linspace(*args.ylim, 21))
    points = np.column_stack((x.ravel(), y.ravel()))
    with torch.no_grad():
        g = torch.from_numpy(points)
        delta = (model.advance_theory(g)-g).numpy()
        np.testing.assert_allclose(delta, model.theory_rg(g).numpy(), atol=1e-12)
    speed = np.linalg.norm(delta, axis=1)
    length = .016*min(np.ptp(args.xlim), 2*np.ptp(args.ylim))
    arrows = delta/np.maximum(speed[:, None], 1e-14)*length
    near = cKDTree(inputs).query(points)[0] <= report['near_sample_radius']
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style = dict(angles='xy', scale_units='xy', scale=1, width=.0024,
                 headwidth=3.4, headlength=4.4, pivot='middle')
    ax.quiver(*points[~near].T, *arrows[~near].T, color='.80', **style)
    norm = Normalize(vmin=0, vmax=max(float(speed.max()), 1e-12))
    q = ax.quiver(*points[near].T, *arrows[near].T, speed[near],
                  cmap='viridis', norm=norm, **style)
    fig.colorbar(q, ax=ax, pad=.025, shrink=.8, label=r'$\|R(g)-g\|$')
    for i, root in enumerate(report['fixed_points'], 1):
        z = np.array(root['g'])
        if not (args.xlim[0] <= z[0] <= args.xlim[1] and args.ylim[0] <= z[1] <= args.ylim[1]):
            continue
        with torch.no_grad():
            assert np.linalg.norm(model.theory_rg(torch.from_numpy(z)).numpy()) < 1e-8
        ax.scatter(*z, marker='*', s=180, color='crimson', edgecolors='white', linewidths=.8, zorder=5)
        ax.annotate(f'F{i}: {root["stability"]}\n({z[0]:.4f}, {z[1]:.4f})', z,
                    xytext=(14, 14), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=.3', fc='white', ec='.8', alpha=.95))
    ax.set(xlim=args.xlim, ylim=args.ylim, xlabel=r'$g_1$', ylabel=r'$g_2$',
           title=f'2D classical Ising QCR: $R(g)-g$ near F1 | step {args.step}')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.15)
    fig.text(.5, .055, 'Arrow lengths normalized; color = displacement magnitude; gray = far from sampled inputs.',
             ha='center', fontsize=9)
    fig.text(.5, .025, 'Discrete-map stability uses eigenvalues of DR, not the apparent convergence of displacement arrows.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, 1))
    output = folder/'R_vector_field_F1_zoom.png'
    fig.savefig(output, dpi=220)
    plt.close(fig)
    np.savez_compressed(folder/'R_vector_field_F1_zoom.npz', g=points, displacement=delta,
                        near_data=near, xlim=args.xlim, ylim=args.ylim)
    print(output)


if __name__ == '__main__':
    main()
