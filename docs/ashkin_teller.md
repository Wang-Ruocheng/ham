# Two-dimensional Ashkin–Teller pilot

Hamiltonian: H/J = -sum_<ij>[sigma_i sigma_j + tau_i tau_j
+ 2 sigma_i sigma_j tau_i tau_j]. K=J/T, zero fields, periodic square lattice.
The four categories encode (--), (-+), (+-), (++). Product-spin observables
are diagnostic only; no coupling or order-parameter labels enter training.

## Sampling and validation

Conditional Swendsen–Wang updates use p=sigma*tau. The conditional bonds are
K(1+p_i p_j) for sigma and K(2+sigma_i sigma_j) for p. Both are nonnegative.
One sweep updates both fields; draws remain MCMC samples, not guaranteed IID.
The sampler was checked against exact enumeration on a 2x2 periodic lattice.

L=16, 48 uniformly spaced K values from .08 to .65, 3 replicas per K,
128 configurations each, burn=1024 sweeps, spacing=16 sweeps. Train/test
splitting holds out entire coupling groups. Supplementary ordered/random-start
probes use burn=2048, 1024 draws, spacing=16. Data and diagnostics are in
`data_cache/ashkin_teller2d_ratio2_L16_n128_seed91`.

Finite-size diagnostics show product ordering before individual-spin ordering;
the transition neighborhoods are not thermodynamic critical-point estimates.
This uniform pilot grid has limited resolution in the narrow intermediate region.

## Training

dim_g=2, hidden=60, four-state random-initialized probabilistic 2x2 C,
scales 16 -> 8 -> 4, batch 32 ensembles x 64 samples, Adam lr=.001, seed=91.
1200 updates, checkpoints every 200. Independent Q/C/R gradients:
Q uses existing relative_local representation losses; C uses -2 I_empirical
+ 2 L_variance(per ensemble, ratio=1) + L_color; R uses one-step and rollout
losses with weight 5 each. Empirical MI may have finite-sample bias.
This is NOT full mutual-backpropagation joint QCR training.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.generate_ashkin_teller2d \
  --output data_cache/ashkin_teller2d_ratio2_L16_n128_seed91
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.rsmi_separate_qcr \
  --config configs/ashkin_teller2d_g2_random_color1_seed91.json \
  --output results/ashkin_teller2d_g2_random_mi_variance2ratio1_color1_1200_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

Training launched; log: `/tmp/ashkin_teller_train.log`. No final physical
success claim yet. Validate three phase regions and two boundaries, not a
prescribed count of latent fixed points. Report R(g)-g and the K-to-g map.

## Scalar-latent comparison

The dim_g=2 run completed 1200 updates in 595.75 seconds (driver timer).
Its bounded initial search found one stable point and one saddle; this is not
yet evidence of recovering both phase boundaries.

A fresh dim_g=1 comparison uses the identical cache and objective, seed 91,
hidden=60 and 1200 updates. Parameter count changes with latent dimension;
the same seed does not guarantee identical C initialization when module shapes
change. No checkpoint is reused.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.rsmi_separate_qcr \
  --config configs/ashkin_teller2d_g2_random_color1_seed91.json --theory-dim 1 \
  --output results/ashkin_teller2d_g1_random_mi_variance2ratio1_color1_1200_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

Scalar run launched; log: `/tmp/ashkin_teller_g1_train.log`.

## Completed diagnostic comparison

The scalar run completed in 580.09 seconds. Read-only audit:
`python -m scripts.audit_ashkin_teller`; artifacts in
`results/ashkin_teller_comparison_audit/` (audit.json, arrays, comparison.png).
This diagnostic uses all 128 configurations, rather than the initial 64-sample
report, so microscopic inverse estimates can differ slightly.

- Scalar unstable root g=1.4348768 has unique microscopic inverse K=.2147772.
  Wider [-3,6] search finds another stable root g=2.9459288 outside observed
  microscopic support; do not count this as validated physical ordering.
- Vector saddle (1.6346168,.1417728) projects onto the microscopic K curve
  at K=.2112522, residual .0094053; this is a projection, not an exact inverse.
- Vector microscopic covariance eigenvalues are 7.59e-8 and .3580. Near-linearity
  alone is not failure for a one-parameter scan, but the direct Q trajectories
  already have only one principal-direction sign change in this pilot.
- C tables differ by at most .000417 between runs. Full-block color retention
  does not enforce the AT Z2 x Z2 symmetry, particularly in tied blocks.
  Exact IID product-ordered reference p=(.5,0,0,.5) is mapped to single-spin
  means about -.2478 after one step and -.9923 after four. This is a structural
  C failure to preserve the reference partially ordered distribution, not
  a Q/R fitting artifact or a Monte Carlo fluctuation.
- Five within-ensemble shuffle probes reveal severe plug-in MI bias: near
  K=.18915 actual/shuffled MI=1.96564/1.95392 bits. At K=.11638 they are equal
  at 1.96771 bits. About 90% of observed environment categories occur once
  (mean over probes). Empirical MI must not be interpreted as population MI.
- One-step error / actual direct-motion RMS in coarse transition windows:
  scalar .379/.215; vector .598/.694. Small global loss does not ensure
  accurate local flow. These windows are diagnostics, not exact critical points.

These findings identify multiple limitations, not a causal ablation proving
one unique cause. No training loss or model was changed during the audit.

## Fixed positive decimation control (dim_g=2)

Launched a fresh 1200-update control retaining the entire four-state variable
at the top-left site of every nonoverlapping 2x2 block. This preserves both
spin-flip symmetries and the spatially IID partially ordered reference
distributions. It is a reference marginalization map, not an exact closed
two-coupling AT recursion: effective longer-range/multispin terms can appear.

Same cache, hidden width, Q/R initial parameters, seed, batches, learning rate,
representation objective and separate Q/R gradients as the trainable-C run.
C has zero parameters. MI and C auxiliaries are omitted since they cannot
optimize a fixed C; total losses are therefore not directly comparable.
Generic configuration-driven runner now supports fixed categorical C.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recurrent_rg train \
  --config configs/ashkin_teller2d_g2_fixed_decimation_seed91.json \
  --output results/ashkin_teller2d_g2_fixed_decimation_1200_seed91
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recurrent_rg plot \
  --run results/ashkin_teller2d_g2_fixed_decimation_1200_seed91 \
  --output results/ashkin_teller2d_g2_fixed_decimation_1200_seed91/report
```

Log: `/tmp/at_fixed_train.log`. Launch status only; no physical success claim.

### Frozen-Q distinguishability check after completion

`python -m scripts.audit_at_q_separation` produces `q_separation/audit.json`
and `phase_separation.png` in the fixed-decimation run. Q is never updated.
For each of k=0,1,2 separately, a training-only PCA axis and two thresholds
distinguish conservative finite-size regions K<=.165, .200<=K<=.214, K>=.25.
All 45 selected held-out ensembles classify correctly (9/3/33 per region).
The middle test region is only ONE held-out K with three replicas; these
results do not establish exact boundaries or thermodynamic phase detection.
Thresholds are scale-specific, not proof of a scale-invariant phase decoder.

Frozen-g interpolation, calibrated on training couplings only, recovers original
K on held-out K in [.15,.26] with MAE .00211/.00260/.00335 for k=0/1/2.
Thus the current Q DOES distinguish representative regions and retains
substantial coupling information. Missing RG boundaries must not be described
as failure to separate these three regions. This does not establish autonomous
RG closure. The user's conditional request for supervised Q training was not
triggered; no additional neural model was trained in this check.

## Middle-dense sampling control, dim_g=1

Fresh training against the original trainable-C scalar baseline, NOT the fixed-C
vector control. Keep 48 couplings x 3 replicas x 128 configurations, L16,
U/J=2, seed91, hidden60, 1200 updates and the same independent Q/C/R objectives.
Only the coupling grid changes: 16 values in [.08,.185], 16 in [.19,.22],
16 in [.225,.65]. The [.19,.22] neighborhood rises from 2/48 to 16/48 of
couplings. This neighborhood is informed by finite-size diagnostics, not exact
phase boundaries. Labels remain excluded from neural losses. Regenerated
samples differ, so this is not a paired-identical-configuration comparison.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.generate_ashkin_teller2d \
  --grid middle_dense --output data_cache/ashkin_teller2d_ratio2_middle_dense_L16_n128_seed91
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.rsmi_separate_qcr \
  --config configs/ashkin_teller2d_g1_middle_dense_seed91.json \
  --output results/ashkin_teller2d_g1_middle_dense_mi_variance2ratio1_color1_1200_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

Sampling log: `/tmp/at_middle_dense_sampling.log`; training log:
`/tmp/at_middle_dense_train.log`. Evaluate R(g), microscopic K-to-g and direct
coarse-graining trajectories. Improved flow would support a sampling-distribution
effect but would not isolate Q: changing data also changes C and R training.

### Middle-dense completed audit

1200 updates completed (1078.82 seconds). Read-only diagnostic:
`python -m scripts.audit_at_middle_dense`, producing `direct_audit/Q_vs_R.png`
and `audit.json` under this run. All 128 samples, all three dataset replicas,
three C-sampling seeds; these are not three training seeds.

For K=.20 direct g is 3.11952 -> 3.17679 -> 3.19129; for K=.21 it is
2.96390 -> 2.98885 -> 2.98264. R rolls the same starts upward and predicts
4.26509/3.69180 at step20 (unvalidated extrapolation). No independent middle
attractor is established by the two trained steps. k=3 reaches a 2x2 lattice
and repeats sites in Q's 3x3 patches: its downturn is not asymptotic evidence.

Mean-curve interpolation gives different direct targets for the same g:
at g=3.3, drift is about .0999 for k=0->1 and .0102 for k=1->2. This is
evidence against precise autonomous scalar closure of the present representation,
subject to finite-sample interpolation uncertainty. On the middle-region mean
curves, R error / direct-motion RMS is .493 and 1.416 for the two steps.

The C-only IID product-plus reference still develops spurious magnetization:
0 -> -.37487 -> -.81288 -> -.99389. Thus both upstream C/Q limitations and
R local errors remain. The unstable root g=2.65548 has microscopic inverse
K=.231854; this is near the second finite-size transition neighborhood, not
a verified exact critical coupling. Wider [-5,10] roots outside microscopic
support must not be interpreted as additional recovered phase boundaries.

## Middle-dense data with fixed C, dim_g=1

Fresh 1200-update control on the identical middle-dense cache; seed91, hidden60,
batch32x64, learning rate .001, separate Q/R gradients. Fixed same-site positive
decimation retains the joint (sigma,tau) category; no C parameters, MI or C losses.
No guarantee of an exact closed AT recursion or recovered phase boundaries.
Basic R(g) and coupling-to-g plots are generated after training completes.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recurrent_rg train \
  --config configs/ashkin_teller2d_g1_middle_dense_fixed_decimation_seed91.json \
  --output results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recurrent_rg plot \
  --run results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91 \
  --output results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91/report
```

Log: `/tmp/at_middle_dense_fixed_train.log`. Training launched; no result yet.

### Completed fixed-C middle-dense diagnostic

1200 updates completed. `python -m scripts.audit_at_middle_dense --run
results/ashkin_teller2d_g1_middle_dense_fixed_decimation_1200_seed91` produces
`direct_audit/` artifacts without changing weights. Direct g at K=.20 is
3.06591 -> 3.26788 -> 3.36442; at K=.21 it is 2.90500 -> 3.07530 -> 3.19364.
Both remain distinguishable from K=.15 (3.72079 at k2) and K=.28167
(2.22746 at k2). At K=.20, product absolute order remains .804/.796/.806;
single-spin absolute order .243/.251/.281 is affected by shrinking volume.
No learned-C bias is present in this deterministic control.

R-only 20-step rollouts send all four representative starts toward g=3.99554,
including the ordered probe whose direct trajectory is nearly stationary.
Same-input mean-curve interpolation gives different direct next-step drifts:
at g=3.2, .1941 at k0->1 versus .1075 at k1->2. Finite-sample evidence of
non-autonomous scalar projection; not proof that all scalar RG representations
are impossible. The unstable root interpolates to K=.36119, not either
finite-size transition neighborhood.

Important interpretation correction: plain site decimation preserves every
spatially IID categorical distribution (and mixtures of such distributions),
so three isolated attractors are NOT guaranteed by this kernel. It is a
symmetry-preserving control, not a guarantee of the desired phase-flow normal
form. k3 has only a 2x2 lattice and cannot establish asymptotic convergence.
