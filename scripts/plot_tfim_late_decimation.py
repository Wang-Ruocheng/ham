"""Continue existing exact L=1024 samples; do not regenerate or sort them."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plot_tfim_exact_long_chain import bond_matrix


def main():
    out = Path(__file__).resolve().parents[1] / 'results/tfim_exact_L1024_reference'
    data = np.load(out / 'all_data.npz')
    samples, xs = data['samples'], data['x']
    scales = [64, 128, 256, 512]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    records = []
    for row, x in enumerate(xs):
        matrix = bond_matrix(1024, x)
        entries = []
        for col, b in enumerate(scales):
            spins = samples[row, :, ::b]
            sign, logdet = np.linalg.slogdet(matrix[:b, :b])
            g = float(sign * np.exp(logdet))
            probability = (1 + g) / 2
            observed = float((spins == np.roll(spins, -1, axis=1)).mean())
            ax = axes[row, col]
            ax.imshow(spins, cmap='gray', vmin=-1, vmax=1, interpolation='nearest',
                      aspect='auto', extent=(0, spins.shape[1], len(spins), 0))
            ax.set_title(f'k={int(np.log2(b))}, {spins.shape[1]} spins\n'
                         + f'Exact adjacent same-color probability: {probability:.1%}', fontsize=9)
            ax.set_xlabel('Retained spatial site')
            ax.set_xticks([0, spins.shape[1]])
            if col == 0:
                ax.set_ylabel(f'h/J={x:.2f}\nSample index (not space)')
            else:
                ax.set_yticklabels([])
            entries.append({'b': b, 'sites': spins.shape[1], 'G': g,
                            'exact_adjacent_same_probability': probability,
                            'sample_adjacent_same_fraction': observed})
        records.append({'x': float(x), 'scales': entries})
    fig.suptitle('Three more decimations of the SAME exact L=1024 samples', fontsize=15)
    fig.text(.5, .02, '64 samples per row, unchanged order. No majority vote or sign alignment.\n'
             'Probability refers to a pair of adjacent retained spins, NOT to the entire row being one color.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(out / 'X_late_decimation.png', dpi=200)
    plt.close(fig)
    (out / 'late_decimation.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
