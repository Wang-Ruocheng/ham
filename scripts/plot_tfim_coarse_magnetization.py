"""Track magnetization of the same exact ensembles under spin decimation."""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/tfim_phase_transition_fss'
    out = root / 'results/tfim_exact_L1024_reference'
    ratios = [.95, 1., 1.05]
    rng = np.random.default_rng(91)
    records, magnetizations = [], []
    for x in ratios:
        spins = np.load(source / f'samples_L1024_x{x:.6f}.npz')['samples']
        group, ms = [], []
        ids = rng.integers(0, len(spins), size=(2000, len(spins)))
        for k in range(10):
            m = spins[:, ::2**k].mean(axis=1)
            ms.append(m)
            boot = np.abs(m)[ids].mean(axis=1)
            group.append({'k': k, 'remaining_sites': 1024//2**k,
                          'mean_abs_m': float(np.abs(m).mean()),
                          'SE': float(boot.std(ddof=1)),
                          'ci95': np.quantile(boot, [.025, .975]).tolist()})
        records.append({'x': x, 'scales': group})
        magnetizations.append(ms)
    random_baseline = []
    for k in range(10):
        n = 1024//2**k
        # For even n unbiased independent spins: E|m| = binom(n,n/2)/2**n.
        random_baseline.append(math.exp(math.lgamma(n+1)-2*math.lgamma(n/2+1)-n*math.log(2)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, rec in enumerate(records):
        ax.errorbar(range(10), [r['mean_abs_m'] for r in rec['scales']],
                    yerr=[r['SE'] for r in rec['scales']], fmt='o-', capsize=3,
                    label=f"h/J={rec['x']:.2f}", color=f'C{i}')
    ax.plot(range(10), random_baseline, 'k--', label='Independent unbiased spins (exact finite-n baseline)')
    ax.set(xlabel='Decimation step k (remaining number of spins below)', ylabel=r'$\langle |m_k|\rangle$',
           title='Same ground-state ensembles under repeated spin selection',
           xticks=range(10), ylim=(0, .85))
    ax.set_xticklabels([f'{k}\n{1024//2**k}' for k in range(10)])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.text(.5, .015, '256 independent initial samples per h/J. Error bars: 1 bootstrap SE. Not independently re-equilibrated shorter chains.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / 'coarse_mean_abs_m.png', dpi=200)
    plt.close(fig)
    steps = [0, 2, 4, 6]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    # Common bins are integer multiples of the smallest panel's magnetization spacing.
    bins = np.linspace(-1.0625, 1.0625, 18)
    for i, x in enumerate(ratios):
        for col, k in enumerate(steps):
            ax = axes[i, col]
            ax.hist(magnetizations[i][k], bins=bins, weights=np.full(256, 1/256),
                    color=f'C{i}', edgecolor='white', alpha=.8)
            ax.axvline(0, lw=.7, color='gray')
            mean = records[i]['scales'][k]['mean_abs_m']
            ax.set_title(f'k={k}, n={1024//2**k}\n'+rf'$\langle|m_k|\rangle={mean:.3f}$', fontsize=10)
            ax.set_ylim(0, .55)
            if col == 0:
                ax.set_ylabel(f'h/J={x:.2f}\nProbability per bin')
            if i == 2:
                ax.set_xlabel('$m_k$')
    fig.suptitle('Magnetization distributions of the SAME samples after decimation', fontsize=14)
    fig.text(.5, .02, 'Same bin width and vertical scale in every panel. No sign alignment or sorting. k=6 retains only 16 spins.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(out / 'coarse_magnetization_distributions.png', dpi=200)
    plt.close(fig)
    report = {'source': str(source), 'samples_per_x': 256,
              'operation': 's_k[i]=s_0[2^k i]; not a complete quantum RG',
              'relation_to_learned_C': 'C=-s0 adds only (-1)^k globally; |m| is identical.',
              'bootstrap_resamples': 2000, 'random_baseline': random_baseline, 'records': records}
    (out / 'coarse_magnetization.json').write_text(json.dumps(report, indent=2)+'\n')
    for rec in records:
        print(rec['x'], [(r['remaining_sites'], round(r['mean_abs_m'], 4)) for r in rec['scales']])


if __name__ == '__main__':
    main()
