"""Finite-chain exact ground-state marginals, NOT a closed quantum RG map.

Separate the measured spin correlations from the universal linearized RG law.
All configurations are genuine Born samples; no x -> x**2 step is used.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh


def ground_state(n, x):
    indices = np.arange(2**n, dtype=np.int64)
    spins = (2 * ((indices[:, None] >> np.arange(n)) & 1) - 1).astype(np.int8)
    diagonal = -(spins * np.roll(spins, -1, axis=1)).sum(axis=1).astype(float)
    flips = [indices ^ (1 << i) for i in range(n)]

    def matvec(v):
        out = diagonal * v
        for flip in flips:
            out -= x * v[flip]
        return out

    operator = LinearOperator((2**n, 2**n), matvec=matvec, dtype=np.float64)
    energy, vectors = eigsh(operator, k=1, which='SA', tol=1e-12,
                           v0=np.ones(2**n) / np.sqrt(2**n))
    psi = vectors[:, 0]
    residual = float(np.linalg.norm(matvec(psi) - energy[0] * psi))
    momenta = (2 * np.arange(n) + 1) * np.pi / n
    analytic_energy = -np.sqrt(1 + x*x - 2*x*np.cos(momenta)).sum()
    np.testing.assert_allclose(energy[0], analytic_energy, atol=1e-9, rtol=0)
    assert residual < 1e-9
    probability = psi**2
    probability /= probability.sum()
    np.testing.assert_allclose(probability, probability[::-1], atol=1e-10, rtol=0)
    return spins, probability, {'energy': float(energy[0]), 'analytic_energy': float(analytic_energy),
                                'eigenpair_residual': residual}


def main():
    out = Path(__file__).resolve().parents[1] / 'results/tfim_exact_marginal_reference'
    out.mkdir(parents=True, exist_ok=True)
    n = 16
    ratios = [.9, 1., 1.1]
    rng = np.random.default_rng(91)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    records, saved = [], {}
    for row, x in enumerate(ratios):
        spins, p, checks = ground_state(n, x)
        # Independent ensembles across x; within each x, columns retain
        # the exact same sampled configurations.
        chosen = rng.choice(len(p), size=64, p=p)
        samples = spins[chosen]
        correlation = [float(p @ (spins[:, 0] * spins[:, r])) for r in range(1, 9)]
        marginal_checks = []
        for k in range(4):
            b = 2**k
            coarse = spins[:, ::b]
            labels = ((coarse.astype(np.int64) + 1) // 2) @ (1 << np.arange(coarse.shape[1]))
            marginal = np.bincount(labels, weights=p, minlength=2**coarse.shape[1])
            np.testing.assert_allclose(marginal.sum(), 1., atol=1e-12)
            direct_g = float(p @ (coarse[:, 0] * coarse[:, 1]))
            np.testing.assert_allclose(direct_g, correlation[b-1], atol=1e-12)
            saved[f'marginal_row{row}_k{k}'] = marginal
            marginal_checks.append({'b': b, 'sites': coarse.shape[1], 'G_coarse_1': direct_g})
            ax = axes[row, k]
            ax.imshow(samples[:, ::b], cmap='gray', vmin=-1, vmax=1, interpolation='nearest',
                      aspect='auto', extent=(0, n // b, 64, 0))
            ax.set_title(f'b={b}, {n//b} retained spins\nG(1)={direct_g:.3f}', fontsize=11)
            ax.set_xlabel('Retained spatial site')
            ax.set_xticks([0, n // b])
            if k == 0:
                ax.set_ylabel(f'h/J = {x:.2f}\nSample index (not space)')
            else:
                ax.set_yticklabels([])
        saved[f'samples_row{row}'] = samples
        records.append({'h_over_J': x, **checks, 'G_r_1_to_8': correlation,
                        'marginal_checks': marginal_checks})
        print(json.dumps(records[-1]), flush=True)
    fig.suptitle('Exact finite-chain Born samples and exact spin marginals (TFIM, L=16, T=0)', fontsize=14)
    fig.text(.5, .02, 'Each row: one ensemble, retained at spacings 1, 2, 4, 8. Black=-1; white=+1. No sorting.\n'
             'Exact diagonal measurement distributions, NOT a full quantum RG or an effective h/J flow. Finite-size rounding is substantial.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .96))
    fig.savefig(out / 'exact_configuration_marginals.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    r = np.arange(1, 9)
    critical = np.array(records[1]['G_r_1_to_8'])
    for i, record in enumerate(records):
        g = np.array(record['G_r_1_to_8'])
        axes[0].plot(r, g, 'o-', label=f"h/J={record['h_over_J']:.2f}")
        axes[1].plot(r, g / critical, 'o-')
    axes[0].set(title='Exact finite-chain correlations', xlabel='Separation r in original lattice',
                ylabel=r'$G_x(r)=\langle \sigma^z_0\sigma^z_r\rangle$')
    axes[0].legend()
    axes[1].axhline(1, color='gray', ls=':', zorder=0)
    axes[1].set(title='Deviation from the critical reference', xlabel='Separation r (= retained spacing b)',
                ylabel=r'$G_x(r)/G_{x=1}(r)$')
    steps = np.arange(5)
    for t in [-.005, 0., .005]:
        axes[2].plot(steps, t * 2.**steps, 'o-', label=f'$t_0={t:+.3f}$')
    axes[2].set(title='Universal linearized RG (separate theory)', xlabel='RG step k, b=2 per step',
                ylabel=r'$t_k=2^k t_0$ ($\nu=1$)', xticks=steps)
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.text(.5, .02, 'Left/center: exact probability sums, no sampling error (L=16 finite-size effects remain).\n'
             'Right: critical scaling-field law, not fitted to the samples and not a global h/J recursion.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / 'exact_correlations_and_critical_instability.png', dpi=200)
    plt.close(fig)
    np.savez_compressed(out / 'exact_marginals.npz', **saved)
    report = {'L': n, 'boundary': 'periodic', 'temperature': 0, 'J': 1,
              'seed': 91, 'samples_displayed': 64, 'records': records,
              'scope': 'Exact finite-chain ground state to numerical tolerance; exact diagonal spin marginalization. Not full quantum RG.',
              'theory_panel': 'Universal linearized thermal scaling field t_k=2^k t_0; not a global lattice h/J map.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
