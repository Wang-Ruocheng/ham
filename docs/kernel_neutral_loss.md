# Uniform-reference neutral loss

The opt-in `kernel_neutral_weight` defaults to zero, so existing experiment
objectives remain unchanged. This regularizer is independent of coupling
labels, physical data marginals, spatial dimension, and the ordering of state
labels. It does impose a structural convention: the independent uniform
reference measure should not acquire a one-site color preference.

For an m-site block of n-state variables, define

```text
pbar[a] = (1/n^m) sum_{v in {0,...,n-1}^m} P_C(a | v)
L_neutral = 1/2 sum_a (pbar[a] - 1/n)^2.
```

For a binary kernel this is exactly `(mean_uniform P_C(+|V)-1/2)^2`, NOT twice
that expression. Binary input encoding is +/-1; categorical encoding is
one-hot. The loss acts on sigmoid/softmax probabilities, not sampled spins,
so its gradient is exact and it consumes no RNG. The shared recurrent kernel
is penalized once per objective evaluation, not once per lattice site or scale.

`recurrent_rg/kernel_regularizers.py` supports n-state categorical fields and
arbitrary block volumes. Enumeration costs n^m; it is chunked and explicitly
limited by `kernel_neutral_max_patterns` (default 65536). Larger problems fail
clearly instead of silently changing to a Monte Carlo approximation. Tests
cover binary chains/squares/cubes and 3-, 4-, and 5-state blocks.

## What it does not enforce

- It does not force physical ensembles at nonzero field to be color balanced.
- It does not forbid global spin flips or color permutations.
- It does not imply full equivariance under color permutations.
- Even at zero field, correlated physical inputs are NOT uniformly distributed
  over block patterns. A neutral kernel can still bias such inputs.
- It does not exclude an input-independent uniform random kernel, or ensure
  retention of long-distance information. Existing variance-floor and Q/R
  objectives are retained in the joint experiment, but are not a proof of
  information retention either.
- A small nonzero one-step reference bias may grow under repeated C. The report
  explicitly computes the exact IID marginal recursion, in addition to
  magnetization on actual cached zero-field data.

## Paired test: classical 1D / 2D Ising

Use `experiments/neutral_qcr.py` to run two weights, 0 and 10, in each problem.
All runs use seed 91, random C initialization, probabilistic C, dim(g)=1 for 1D
and dim(g)=2 for 2D, hidden=60, relative_local geometry, variance-floor weight
2 (ratio .25), no preservation, no RSMI, no isotropy/weak-direction/orientation
terms, and JOINT Q/C/R training for 2400 updates. Batch: 32 ensembles x 64 saved
configurations; Adam .001. Whole coupling cells are held out. The 1D zero-field
data retain spin-flip augmentation; nonzero-field 2D data do not.

Data: 1D L=128, J=.01..2, 128 saved configurations per ensemble; 2D L=16,
K=.20:.03:.68, B=-.01:.001:.01, 64 saved configurations per ensemble. Available
training scale chains are 128->64->32->16->8->4 and 16x16->8x8->4x4.

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/neutral_qcr.py --dimension 1 --weight 0 --output results/neutral_ising1d_g1_w0_seed91
python experiments/neutral_qcr.py --dimension 1 --weight 10 --output results/neutral_ising1d_g1_w10_seed91
python experiments/neutral_qcr.py --dimension 2 --weight 0 --output results/neutral_ising2d_g2_w0_seed91
python experiments/neutral_qcr.py --dimension 2 --weight 10 --output results/neutral_ising2d_g2_w10_seed91
python scripts/compare_neutral_qcr.py
```

The comparison verifies full configuration equality except the neutral weight
and matching data SHA256. No reference enumeration RNG disturbs paired batch
selection/C draws. Final checkpoints are used, not selected by visual phase
diagram agreement. Reports include common loss excluding neutral, raw neutral,
probability truth tables, exact IID drift, actual zero-field magnetization,
closure errors, coupling maps, and discrete fixed-point eigenvalues. R^2 denotes
composition, not squaring coordinates. Off-data flow remains extrapolation.

## Completed 1D result (seed 91)

The neutral objective decreases from 6.210e-4 to 2.256e-6, but this does not
improve the physical kernel in this trial. Exact mean I(H;E) decreases from
0.387446 to 0.097905 bit; test one-step NRMSE increases from 3.20% to 11.23%.
Final-scale rollout NRMSE is 44.84% vs 41.55%, but absolute latent RMSE also
changes (.07281 vs .10768), and target spreads differ: do not interpret the
final NRMSE alone as overall improvement. Test total loss is .122863 vs .173824.

The reference-balanced kernel can sacrifice ordered blocks and compensate on
mixed blocks: P(+|--)=.000290, P(+|+-)=.707247, P(+|-+)=.872268,
P(+|++)=.426203. The two uniform inputs are distinguished by only .425913 in
output probability, versus .998872 for the no-neutral kernel (which flips
uniform states, a permitted sign convention).

For any two-spin binary C, write its conditional output mean exactly as
`m_C(s1,s2)=a+b1*s1+b2*s2+c*s1*s2`. Neutral controls only a, not c.
For the weight-10 kernel, a=.003004 and c=-.576511. On a zero-field Ising
ensemble the mean output is `a+c*<s1*s2>`, not a: input correlations expose the
unconstrained even term. At J=2 an inspected spin-flip-balanced ensemble has
m: 0 -> -.549 -> -.502 -> -.469 -> -.413 -> -.359. This is an observed failure
of the proposed sufficient condition, not an implementation error or a reason
to prohibit all global sign flips.

Both scalar R fits show one stable and one unstable root in the bounded search.
The unstable roots are outside observed training R inputs in both cases.
Consequently root count alone does not validate the kernel or its RG physics.

## Completed 2D result (seed 91)

Both weights retain a central repeller. For weight 0 its discrete DR eigenvalues
are (1.524855, -1.072060), with microscopic-grid inverse (K,B)=(.429283,-.000517).
For weight 10 they are (1.542877, -1.067521), with inverse (.430490,.0000574).
These inverse coordinates are not effective renormalized couplings. The negative
eigenvalue reverses orientation, but its modulus exceeds one; this is a repeller,
not a saddle. R(g)-g arrows alone do not diagnose discrete stability.

Neutral decreases from .00465322 to .00000979324. Exact IID reference mean after
six C iterations decreases in magnitude from .426933 to .0197623. Both kernels
still flip uniform states and distinguish their outputs almost perfectly
(probability separation .999933 and .999869). Thus neutrality is not preservation.

Test total loss changes from .758993 to .747516, while held-out first-step NRMSE
worsens from 4.07% to 5.23% and two-step rollout NRMSE from 5.58% to 6.49%.
The common objective excluding neutral is .747418 for weight 10. The high-g
endpoint has an eigenvalue extremely near -1 in both runs (-.999771 and
-1.004454); its stable/saddle label is therefore marginal and not robust evidence
of improved physics. It is also outside the microscopic interpolation grid.

Conclusion: this single-seed, single-weight ablation supports reduced reference
bias in 2D, not universally improved RG learning. In 1D it exposes a concrete
correlation-dependent bias loophole. Keep the new regularizer opt-in (default 0).

## Figures and numerical audit

- [1D R and direct g(J)](../results/neutral_comparison_seed91/ising1d_R_and_J.png)
- [2D R flow comparison](../results/neutral_comparison_seed91/ising2d_R_comparison.png)
- [1D losses and reference drift](../results/neutral_comparison_seed91/ising1d_loss_and_bias.png)
- [2D losses and reference drift](../results/neutral_comparison_seed91/ising2d_loss_and_bias.png)
- [All numerical results](../results/neutral_comparison_seed91/comparison.json)

Implementation validation: full suite, 205 tests passing. n-state support is
covered by unit/gradient tests; these full training experiments use binary Ising.
