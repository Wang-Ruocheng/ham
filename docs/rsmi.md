# RSMI-only C, followed by frozen-C Q/R

This is an explicit alternative experiment, not a change to the default QCR
objective. It follows the RSMI information objective, using an InfoNCE estimator
in the spirit of [RSMI-NE](https://arxiv.org/abs/2103.16887). It is not a verbatim
reproduction of that paper's architecture, lattice sizes, or coupling schedule.

## Objective and implementation

For each ensemble separately, sample N pairs `(V_i,E_i)` from different saved
configurations, at random periodic translations. V is a 2-site / 2x2 block;
a one-site buffer separates it from E. E is the two exterior endpoints in 1D,
or the 20 sites of a 6x6 outer perimeter in 2D. No sites alias periodically.

The existing random-initialized MLP C defines `p(H=+1|V)=sigmoid(2*logit(V))`.
A 64-64 MLP critic maps E to scores for H=-1,+1. Optimize only

```text
L_C = -mean_ensemble mean_i sum_h p_C(h|V_i)
          [ f(h,E_i) - log(mean_j exp f(h,E_j)) ].
```

Negatives j come ONLY from the same ensemble. Neither K/B labels nor energies
enter either network or loss. This averages per-ensemble information rather
than exploiting information shared only through a mixture of temperatures.
The two H outcomes are summed exactly: there is no straight-through gradient
and no differentiable probability/logit side channel to the critic. C remains
a probabilistic binary kernel during downstream coarse-graining.

No balance, variance-floor, color-preservation, symmetry-equivariance, or Q/R
loss acts on C in stage 1. There is no covariance or weak-direction weighting.
Stage 1 trains at microscopic k=0; the same C is later reused at all scales.
Critic and C are shared across ensembles, unlike separate per-coupling fits.
The zero-field 1D data use random global-sign augmentation; nonzero-field 2D
data do not. This augments data but does not constrain C to be odd.

Stage 2 freezes C and trains Q/R with the existing subset consistency,
relative_local distribution geometry, one-step and full available rollout
losses. It uses no physical supervision but is NOT itself an RSMI objective.
The script verifies C's parameters are bitwise unchanged after stage 2.

## Controls and limitations

- Single seed 91, 2400 C/critic updates, then 2400 Q/R updates; final checkpoint,
  without selecting the best checkpoint using physical fixed points.
- C batch: 16 ensembles x 64 configuration pairs; Adam 0.001.
- Separate same-size critics for positive decimation and fair-tie majority,
  each with 2400 updates and matched batch schedule. These are estimator
  comparisons, not proof of ordering of the true MI.
- Held-out coupling cells are never optimized on; each logged evaluation uses
  8 fixed-seed random batches of up to 32 ensembles. Repeated evaluation
  batches overlap, so no independent-sample uncertainty is claimed.
- Shuffling E within ensembles is a negative control. Negative InfoNCE values
  are valid loose estimates, not negative true mutual information.
- Samples from Monte Carlo caches may be autocorrelated. Finite-data InfoNCE
  is not a certified numerical bound on population MI.
- Exact 1D transfer-matrix MI independently checks the learned kernel and
  enumerates all 16 deterministic two-spin-to-one-bit truth tables.
- Report fixed-point stability from eigenvalue MODULI of the discrete R map;
  off-data roots and arrows are extrapolations. A good RSMI score alone does
  not certify critical exponents or autonomous low-dimensional closure.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/rsmi_kernel_qr.py --config configs/rsmi_ising1d_g1_seed91.json --output results/rsmi_ising1d_g1_seed91
python experiments/rsmi_kernel_qr.py --config configs/rsmi_ising2d_g2_seed91.json --output results/rsmi_ising2d_g2_seed91
python scripts/report_rsmi_experiment.py --run results/rsmi_ising1d_g1_seed91
python scripts/report_rsmi_experiment.py --run results/rsmi_ising2d_g2_seed91
```

Output folders must not already exist. Use a new name for a new run. The
`*_blas_aborted` folders are interrupted thread-oversubscribed attempts, not
additional experimental seeds or completed results.

## Seed 91 results

| Diagnostic | 1D classical Ising | 2D classical K/B Ising |
|---|---:|---:|
| C-only test InfoNCE, bit | 0.263715 | 0.238363 |
| Decimation critic test InfoNCE, bit | 0.256450 | 0.200118 |
| Fair-tie majority critic test InfoNCE, bit | 0.253569 | 0.236942 |
| Final QR train / test loss | 0.244396 / 0.229994 | 0.817731 / 0.849567 |
| Final-scale rollout NRMSE (train target SD) | 16.27% | 7.35% |
| Elapsed, including two baseline critics and QR | 193.5 s | 228.6 s |

1D C is nearly AND: ++ maps to +, both mixed blocks and -- map to -.
2D C is nearly majority with all ties sent to +. Neither is a global sign
permutation, nor a constant-output kernel. Both preserve uniform +/- blocks
with probability greater than 0.99999, but introduce zero-field bias on mixed
blocks. At J=.01, one inspected 1D zero-field ensemble has m: 0 -> -0.493 ->
-0.869 -> -0.992 -> -1 -> -1. At K=.20,B=0, one 2D ensemble has
m: -0.00012 -> 0.271 -> 0.660.

The 1D exact finite-ring calculation (uniform average over all 64 J values)
gives learned 0.391645, decimation 0.392686, fair-tie majority 0.387469 bit.
Enumerating all 16 deterministic two-spin kernels confirms the optimal
deterministic rules are signed decimations. The learned kernel is close in MI
but not equivalent physically. The neural lower-bound ranking differs from
the exact-MI ranking; finite data/critic approximation and the shared critic
must not be interpreted as proof that AND has more true MI than decimation.

1D R has a stable root g=-3.088968 (derivative .522718) and unstable root
g=.740237 (derivative 2.102835), both outside observed training R inputs.
The apparent flow cannot be certified as the zero-field Ising RG by root count.
2D R has a saddle (.125813,1.317231), eigenvalues (1.015398,.303016), and a
repeller (1.090410,.254224), eigenvalues (1.092437,1.448592). The interpolated
microscopic critical encoding is (.495190,.039578). Its mismatch with the root
alone does not prove failure (critical microscopic states may flow along
irrelevant directions), but the magnetization bias is directly measurable.
Piecewise-linear inversion through neighboring microscopic parameter-grid
triangles gives the repeller's preimage K=0.458154, B=-0.005563. This is a
microscopic encoding correspondence, NOT an effective-coupling measurement.

No critical exponent success is claimed. This experiment tests RSMI-only
kernel learning in an unrestricted MLP family, not a symmetry-constrained
replication of the original RSMI paper.
