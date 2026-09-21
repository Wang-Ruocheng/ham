"""Independent exact Born sampling for finite-size TFIM phase diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from plot_tfim_exact_long_chain import bond_matrix, sample_spins_blocked, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tfim_phase_transition_fss'
FIELDS = [.90, .94, .97, .985, .992, .996, .998, 1., 1.002, 1.004, 1.008, 1.015, 1.03, 1.06, 1.10]
COUNTS = {128: 2048, 256: 1024, 512: 512, 1024: 256}


def cell_path(n, x):
    return OUT / f'samples_L{n}_x{x:.6f}.npz'


def generate_cell(n, x, count):
    path = cell_path(n, x)
    if path.exists():
        old = np.load(path)
        if old['samples'].shape == (count, n):
            return {'L': n, 'x': x, 'samples': count, 'cached': True}
    start = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([91, n, round(x * 1_000_000)]))
    b = bond_matrix(n, x)
    samples = np.stack([sample_spins_blocked(b, rng.random(n)) for _ in range(count)])
    np.savez_compressed(path, samples=samples, x=x, L=n,
                        seed_components=np.array([91, n, round(x * 1_000_000)]))
    return {'L': n, 'x': x, 'samples': count, 'seconds': time.monotonic()-start, 'cached': False}


def diagnostics(samples, seed):
    m = samples.mean(axis=1)
    m2, m4 = m*m, m**4
    mean_abs = float(np.abs(m).mean())
    binder = float(1 - m4.mean() / (3 * m2.mean()**2))
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(m), size=(1000, len(m)))
    boot_abs = np.abs(m)[ids].mean(axis=1)
    boot_u = 1 - m4[ids].mean(axis=1) / (3 * m2[ids].mean(axis=1)**2)
    return {'mean_abs_m': mean_abs, 'mean_m2': float(m2.mean()), 'binder_U': binder,
            'abs_m_ci95': np.quantile(boot_abs, [.025, .975]).tolist(),
            'binder_ci95': np.quantile(boot_u, [.025, .975]).tolist(),
            'abs_m_SE': float(boot_abs.std(ddof=1)), 'binder_SE': float(boot_u.std(ddof=1))}


def plot_histograms(preliminary=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    hist_records = []
    for i, x in enumerate([.95, 1., 1.05]):
        path = cell_path(1024, x)
        if path.exists() and not preliminary:
            samples = np.load(path)['samples']
        else:
            samples = np.load(ROOT / f'results/tfim_exact_L1024_reference/ground_state_x{x:.2f}.npz')['samples']
        m = samples.mean(axis=1)
        axes[i].hist(m, bins=np.linspace(-1, 1, 31), density=True, color=f'C{i}', alpha=.75,
                     edgecolor='white')
        axes[i].axvline(0, color='gray', lw=.8)
        axes[i].set(title=rf'$h/J={x:.2f}$, $\langle |m|\rangle={np.abs(m).mean():.3f}$',
                    xlabel=r'$m=L^{-1}\sum_i s_i$')
        axes[i].text(.03, .95, f'{len(m)} independent samples', transform=axes[i].transAxes,
                     va='top', fontsize=9)
        hist_records.append({'x': x, 'samples': len(m), 'mean_abs_m': float(np.abs(m).mean())})
    axes[0].set_ylabel('Probability density')
    fig.suptitle('TFIM magnetization distributions: L=1024, exact T=0 Born sampling')
    fig.tight_layout(rect=(0, 0, 1, .94))
    name = 'magnetization_histograms_initial64.png' if preliminary else 'magnetization_histograms.png'
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    return hist_records


def plot_scan():
    records = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, (n, count) in enumerate(COUNTS.items()):
        group = []
        for j, x in enumerate(FIELDS):
            samples = np.load(cell_path(n, x))['samples']
            result = {'L': n, 'x': x, 'samples': len(samples), **diagnostics(samples, 3000 + i*100+j)}
            records.append(result)
            group.append(result)
        for ax, key, error in [(axes[0], 'mean_abs_m', 'abs_m_SE'),
                               (axes[1], 'binder_U', 'binder_SE'),
                               (axes[2], 'binder_U', 'binder_SE')]:
            ax.errorbar(FIELDS, [r[key] for r in group], yerr=[r[error] for r in group],
                        fmt='o-', ms=3, lw=1, elinewidth=.8, capsize=2, color=f'C{i}',
                        label=f'L={n}, N={count}')
    for ax in axes:
        ax.axvline(1, color='gray', ls='--', lw=1, label='Exact critical x=1' if ax is axes[0] else None)
        ax.set_xlabel('Transverse field ratio h/J')
        ax.grid(alpha=.18)
    axes[0].set(title='Order parameter vs field', ylabel=r'$\langle |m|\rangle$')
    axes[0].legend(fontsize=8)
    axes[1].set(title='Binder cumulant vs field', ylabel=r'$U_L=1-\langle m^4\rangle/(3\langle m^2\rangle^2)$')
    axes[2].set(title='Binder crossing region (zoom)', xlim=(.984, 1.016), ylim=(.15, .7))
    fig.suptitle('Independent periodic TFIM ground states at each L and h/J (no learned model)', fontsize=14)
    fig.text(.5, .02, 'Error bars: 1 bootstrap standard error (1000 resamples). Lines only connect measured points.\n'
             'The dashed critical field is a theoretical reference, not a fitted crossing. Finite-size drift and sampling noise remain.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(OUT / 'finite_size_magnetization_and_binder.png', dpi=200)
    plt.close(fig)
    # A separate scaling diagnostic: exact exponents are specified, not fitted.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, n in enumerate(COUNTS):
        group = [r for r in records if r['L'] == n]
        ax.errorbar([(r['x']-1)*n for r in group], [r['binder_U'] for r in group],
                    yerr=[r['binder_SE'] for r in group], fmt='o-', ms=3, capsize=2, label=f'L={n}')
    ax.set(xlabel=r'$(h/J-1)L$ (using exact $\nu=1$)', ylabel='$U_L$',
           title='Binder finite-size scaling check', xlim=(-8, 8))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'binder_scaling_check.png', dpi=200)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_histograms(preliminary=True)
    if not args.plot_only:
        checks = validate()
        (OUT / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
        jobs = [(n, x, count) for n, count in COUNTS.items() for x in FIELDS]
        jobs += [(1024, .95, 256), (1024, 1.05, 256)]
        jobs.sort(key=lambda job: job[0]**3 * job[2], reverse=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_cell, *job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                print(f'[{done}/{len(jobs)}] {json.dumps(future.result())}', flush=True)
    histograms = plot_histograms()
    records = plot_scan()
    report = {'method': 'Exact finite-L T=0 Gaussian bond Born sampling, independent draws',
              'boundary': 'periodic', 'seed_base': 91, 'counts': COUNTS, 'fields': FIELDS,
              'bootstrap_resamples': 1000, 'plotted_error': 'one bootstrap standard error',
              'histograms': histograms, 'records': records,
              'caution': 'Critical field and exponents are theoretical references, not fitted. No precise critical-point estimate is claimed.'}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Finished all plots.', flush=True)


if __name__ == '__main__':
    main()
