"""Plot saved equal-time ensembles and their actual learned coarse-graining."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads((args.run / 'summary.json').read_text())['config']
    model = RecurrentRGNetwork1D(levels=cfg['levels'], hidden=cfg['hidden'],
                                theory_dim=cfg['theory_dim'],
                                kernel_layout=cfg['kernel_layout']).double().eval()
    mismatch = model.load_state_dict(torch.load(args.run / 'qcr_model.pt', weights_only=True), strict=False)
    assert not mismatch.unexpected_keys
    assert all(k.startswith('action_network.') for k in mismatch.missing_keys)
    tx, th, vx, vh = torch.load(args.cache, weights_only=True)
    x, h = torch.cat((tx, vx)), torch.cat((th, vh)).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    records = []
    arrays = {}
    for row, requested in enumerate((.5, 1., 1.5)):
        index = int(np.argmin(abs(h - requested)))
        with torch.no_grad():
            levels = model.coarse_grain(x[index:index+1].double())
        record = {'h_over_J': float(h[index]), 'cache_index': index, 'levels': []}
        for k, level in enumerate(levels):
            spins = level[0, :, :, 0].numpy()
            np.testing.assert_allclose(np.abs(spins), 1., atol=1e-12, rtol=0)
            spins = np.sign(spins)  # Remove floating-point roundoff from binary STE outputs.
            arrays[f'row{row}_level{k}'] = spins
            mean_abs_m = float(np.abs(spins.mean(axis=1)).mean())
            nn = float((spins * np.roll(spins, -1, axis=1)).mean())
            record['levels'].append({'sites': spins.shape[1], 'mean_abs_m': mean_abs_m,
                                     'nearest_neighbor_correlation': nn})
            ax = axes[row, k]
            ax.imshow(spins[:64], cmap='gray', vmin=-1, vmax=1, interpolation='nearest',
                      aspect='auto', origin='upper', extent=(0, spins.shape[1], 64, 0))
            ax.set_title(f'$C^{k}X$: {spins.shape[1]} sites\n'
                         + rf'$\langle |m|\rangle={mean_abs_m:.3f}$, $G(1)={nn:.3f}$', fontsize=11)
            ax.set_xlabel('Spatial site (coarse lattice)')
            ax.set_xticks([0, spins.shape[1] // 2, spins.shape[1]])
            if k == 0:
                ax.set_ylabel(f'h/J = {h[index]:.6f}\nSample index (not space)', fontsize=11)
            else:
                ax.set_yticklabels([])
        records.append(record)
    fig.suptitle('1D TFIM configurations: same samples before and after learned C (dim g = 1)', fontsize=15)
    fig.text(.5, .025,
             'Black: -1; white: +1. First 64 saved samples, original order; no sorting or spin alignment.\n'
             'Statistics: all 128 original samples. C flips the overall sign at each step. Panels have different spatial resolutions.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .955))
    fig.savefig(args.run / 'X_three_ensembles_by_scale.png', dpi=200)
    plt.close(fig)
    np.savez_compressed(args.run / 'X_three_ensembles_by_scale.npz', **arrays)
    report = {'run': str(args.run), 'cache': str(args.cache), 'display_samples': 64,
              'statistics_samples': 128, 'augmentation': False, 'records': records}
    (args.run / 'X_three_ensembles_by_scale.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
