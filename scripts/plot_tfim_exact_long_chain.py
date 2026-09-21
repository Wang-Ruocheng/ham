"""Exact even-parity TFIM ground-state z snapshots via Gaussian bond sampling.

For a periodic even chain, the commuting bond variables d_i=s_i*s_{i+1}
are disjoint Majorana bilinears. Their joint moments are principal minors
of the real bond contraction matrix B. Thus P(d)=det(I+diag(d)B)/2**L.
Sequential conditioning is a rank-one Schur update; sample L-1 bonds and
one unbiased global spin. This preserves full Born probabilities, not just
two-point correlations. Antiperiodic momenta select the even ground state.

Only diagonal spin marginals are coarse-grained; this is NOT a closed
quantum-Hamiltonian RG or an x -> x**2 recursion.
"""
import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from plot_tfim_exact_marginals import ground_state


def bond_matrix(n, x):
    assert n % 2 == 0 and x > 0
    q = (2 * np.arange(n) + 1) * np.pi / n
    symbol = (1 - x * np.exp(1j * q)) / np.sqrt(1 + x*x - 2*x*np.cos(q))
    coeff = np.fft.ifft(symbol) * np.exp(1j * np.pi * np.arange(n) / n)
    assert np.max(np.abs(coeff.imag)) < 1e-10
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    return np.ascontiguousarray(coeff.real[d % n] * np.where(d < 0, -1., 1.))


@njit(cache=True)
def sample_spins(bond, uniforms):
    n = bond.shape[0]
    b = bond.copy()
    spins = np.empty(n, dtype=np.int8)
    spins[0] = 1 if uniforms[n-1] < .5 else -1
    for p in range(n-1):
        probability = .5 * (1. + b[p, p])
        if probability < -1e-9 or probability > 1 + 1e-9:
            raise ValueError('Invalid conditional probability')
        outcome = 1 if uniforms[p] < probability else -1
        spins[p+1] = spins[p] * outcome
        denominator = b[p, p] + outcome
        for i in range(p+1, n):
            factor = b[i, p] / denominator
            for j in range(p+1, n):
                b[i, j] -= factor * b[p, j]
    return spins


@njit(cache=True)
def sample_spins_blocked(bond, uniforms, block_size=32):
    """Same conditional sampler; accumulate trailing updates into BLAS GEMM."""
    n = bond.shape[0]
    b = bond.copy()
    spins = np.empty(n, dtype=np.int8)
    spins[0] = 1 if uniforms[n-1] < .5 else -1
    for start in range(0, n-1, block_size):
        end = min(start + block_size, n-1)
        u = np.empty((n-end, end-start))
        v = np.empty((end-start, n-end))
        for p in range(start, end):
            probability = .5 * (1. + b[p, p])
            if probability < -1e-9 or probability > 1 + 1e-9:
                raise ValueError('Invalid conditional probability')
            outcome = 1 if uniforms[p] < probability else -1
            spins[p+1] = spins[p] * outcome
            denominator = b[p, p] + outcome
            for i in range(end, n):
                u[i-end, p-start] = b[i, p] / denominator
                v[p-start, i-end] = b[p, i]
            # Update the two panels needed by subsequent measurements in
            # this block; postpone the lower-right matrix until block end.
            for i in range(p+1, end):
                factor = b[i, p] / denominator
                for j in range(p+1, n):
                    b[i, j] -= factor * b[p, j]
            for i in range(end, n):
                factor = b[i, p] / denominator
                for j in range(p+1, end):
                    b[i, j] -= factor * b[p, j]
        update = u @ v
        for i in range(end, n):
            for j in range(end, n):
                b[i, j] -= update[i-end, j-end]
    return spins


def configuration_probability(bond, spins):
    b = bond.copy()
    probability = .5
    for k, outcome in enumerate(spins[:-1] * spins[1:]):
        probability *= .5 * (1 + outcome * b[k, k])
        b[k+1:, k+1:] -= np.outer(b[k+1:, k], b[k, k+1:]) / (b[k, k] + outcome)
    return probability


def validate():
    checks = []
    for x in [.3, .95, 1., 1.05, 3.]:
        spins, p, energy_checks = ground_state(8, x)
        b = bond_matrix(8, x)
        np.testing.assert_allclose(b @ b.T, np.eye(8), atol=1e-12, rtol=0)
        predicted = np.array([configuration_probability(b, s) for s in spins])
        np.testing.assert_allclose(predicted, p, atol=1e-10, rtol=0)
        for r in range(1, 5):
            np.testing.assert_allclose(np.linalg.det(b[:r, :r]),
                                       p @ (spins[:, 0] * spins[:, r]), atol=1e-10, rtol=0)
        checks.append({'x': x, 'max_Born_probability_error': float(np.max(abs(predicted-p))),
                       **energy_checks})
    # Exercise the compiled sampler against exact z-basis probabilities.
    rng = np.random.default_rng(42)
    b = bond_matrix(8, 1.)
    draws = np.stack([sample_spins(b, rng.random(8)) for _ in range(20000)])
    empirical_g = (draws[:, 0] * draws[:, 1]).mean()
    assert abs(empirical_g - b[0, 0]) < .025
    checks.append({'sampler_test_draws': len(draws), 'empirical_G1': float(empirical_g), 'exact_G1': float(b[0, 0])})
    print('Validation passed: all 256 Born probabilities for five L=8 models.', flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sites', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    checks = validate()
    if args.validate_only:
        print(json.dumps(checks, indent=2))
        return
    n = args.sites
    out = Path(__file__).resolve().parents[1] / f'results/tfim_exact_L{n}_reference'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(91)
    ratios = [.95, 1., 1.05]
    distances = np.unique(np.r_[np.arange(1, 33), np.geomspace(33, n//4, 65).astype(int)])
    distances = np.unique(np.r_[distances, [1, 2, 4, 8, 16, 32, 64]])
    samples, correlations, records = [], [], []
    for x in ratios:
        start = time.monotonic()
        bond = bond_matrix(n, x)
        gs = []
        for r in distances:
            sign, logdet = np.linalg.slogdet(bond[:r, :r])
            gs.append(sign * np.exp(logdet))
        gs = np.array(gs)
        draws = []
        for s in range(args.samples):
            draws.append(sample_spins(bond, rng.random(n)))
            if (s+1) % 8 == 0:
                print(f'x={x:.2f}: {s+1}/{args.samples} exact Born samples, {time.monotonic()-start:.1f}s', flush=True)
        draws = np.stack(draws)
        samples.append(draws)
        correlations.append(gs)
        # Translation-averaged sample statistics with independent-sample SE.
        diagnostics = []
        for r in [1, 8, 32, 64]:
            per_sample = (draws * np.roll(draws, -r, axis=1)).mean(axis=1)
            diagnostics.append({'r': r, 'exact': float(gs[np.where(distances == r)[0][0]]),
                                'sample_mean': float(per_sample.mean()),
                                'sample_SE': float(per_sample.std(ddof=1) / np.sqrt(len(draws)))})
        records.append({'x': x, 'seconds': time.monotonic()-start, 'sampling_diagnostics': diagnostics})
        np.savez_compressed(out / f'ground_state_x{x:.2f}.npz', samples=draws, distances=distances, G=gs)
    correlations = np.stack(correlations)
    selected_b = [1, 4, 16, 64]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for i, x in enumerate(ratios):
        for k, b in enumerate(selected_b):
            ax = axes[i, k]
            ax.imshow(samples[i][:, ::b], cmap='gray', vmin=-1, vmax=1, interpolation='nearest',
                      aspect='auto', extent=(0, n//b, args.samples, 0))
            idx = np.where(distances == b)[0][0]
            ax.set_title(f'b={b}, {n//b} spins\n'
                         + rf'$G_x(b)/G_1(b)={correlations[i, idx]/correlations[1, idx]:.3f}$', fontsize=11)
            ax.set_xlabel('Retained spatial site')
            ax.set_xticks([0, n//(2*b), n//b])
            if k == 0:
                ax.set_ylabel(f'h/J={x:.2f}\nSample index (not space)')
            else:
                ax.set_yticklabels([])
    fig.suptitle(f'Exact TFIM ground-state samples: L={n}, T=0, periodic chain', fontsize=16)
    fig.text(.5, .025,
             f'{args.samples} independent Born samples per h/J; same sample rows across scales. No sorting, alignment, or h/J recursion.\n'
             'Columns are exact spin-selection marginals, NOT a full quantum RG. Wider pixels at larger b do not imply greater order.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .955))
    fig.savefig(out / 'X_exact_by_scale.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for i, x in enumerate(ratios):
        axes[0].loglog(distances, correlations[i], label=f'h/J={x:.2f}')
        axes[1].semilogx(distances, correlations[i]/correlations[1], label=f'h/J={x:.2f}')
        axes[2].semilogx(distances, distances**.25 * correlations[i], label=f'h/J={x:.2f}')
    axes[0].set(title='Exact order-parameter correlations', xlabel='Separation r', ylabel=r'$G_x(r)$')
    axes[1].axhline(1, color='gray', ls=':')
    axes[1].set(title='Separation from the critical reference', xlabel='Observation scale b = r', ylabel=r'$G_x(r)/G_1(r)$')
    axes[2].set(title='Remove critical power-law decay', xlabel='Observation scale b = r', ylabel=r'$r^{1/4}G_x(r)$')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.text(.5, .015, 'Exact finite-L determinants, not estimates from displayed samples. Critical exponent 1/4 is imposed, not fitted.\n'
             'Curvature near L/4 includes finite-size effects. These observables illustrate scale-dependent departure, not an inferred h/J flow.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / 'exact_scale_departure.png', dpi=200)
    plt.close(fig)
    report = {'L': n, 'temperature': 0, 'boundary': 'periodic', 'parity': 'even', 'seed': 91,
              'samples': args.samples, 'method': 'Exact Gaussian Majorana bond measurement, then unbiased global spin',
              'scope': 'Exact z-basis Born samples to numerical precision; diagonal decimation marginals, not a complete quantum RG.',
              'validation': checks, 'records': records}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(out / 'all_data.npz', samples=np.stack(samples), x=ratios,
                        distances=distances, correlations=correlations)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
