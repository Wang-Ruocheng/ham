# Three-state square-lattice Potts pilot

The zero-field Hamiltonian convention is H=-J sum_<ij> delta(s_i,s_j),
s_i in {0,1,2}; K=J/T. Its square-lattice critical coupling is
log(1+sqrt(3))=1.0050525387. Couplings are not supplied to the QCR loss.

Latest random-C gradient-only empirical-MI test: [protocol and results](potts_random_gradient_mi.md).
The final unstable root interpolates to K≈0.9938 (theory≈1.0051), with the
high-g attractor still outside data support. No kernel search or pretraining.
The earlier [MI-initialized test](potts_mi_independent.md) did not recover
the correct critical fixed point; its limitations are preserved there.

## Reproduce

```bash
python -m scripts.generate_potts2d --output data_cache/potts2d_q3_L16_n128_seed91
python -m recurrent_rg train --config configs/potts2d_g1_probC_sign0p1_2400_seed91.json --output results/potts2d_g1_probC_sign0p1_2400_seed91
python -m scripts.report_scalar_joint --run results/potts2d_g1_probC_sign0p1_2400_seed91
python -m scripts.report_potts_scalar --run results/potts2d_g1_probC_sign0p1_2400_seed91
```

Generation/training refuse to overwrite existing output directories. Sampling
uses numba and Swendsen-Wang: independently occupy equal-color nearest-neighbor
bonds with probability 1-exp(-K), then recolor each connected component uniformly.
512 full-lattice burn-in sweeps, eight sweeps between recorded configurations.
The cache contains 64 uniform K values in [0.4,1.6], three independently seeded
chains per value, 128 recorded configurations each, periodic L=16.
Whole K cells are held out (129 train / 63 test ensembles). No color augmentation
is used. Recorded draws are not assumed independent; sampling.json includes
lag-one bond correlations and ordered/disordered-start checks.

## Model and loss extensions

The existing Q/C/R model, Tanh widths, scale sharing, optimizer, geometry and
closure terms are retained. dim_g=1, hidden=60, Q windows 3x3, C blocks 2x2,
scales 16->8->4; joint training 2400 updates, batch 32 ensembles x 64 samples,
Adam lr=0.001, seed91. Evaluation uses 64 configurations per ensemble.

- Input fields are three-channel one-hot, not ordered numerical color labels.
- C samples categorical outputs with Gumbel-max and a biased straight-through
  softmax gradient at temperature1, including in evaluation. Draws occur after
  lookup expansion, independently for repeated blocks.
- Optional lookup (`binary_lookup` retained as the historical config name)
  uses observed categorical patterns and retains the dense network's exact
  first-order input and parameter VJPs. It does not impose color symmetry.
- Geometry descriptors are the probabilities of all 81 oriented 2x2 patterns.
- C variance floor uses trace Cov(onehot)=1-||mean(onehot)||^2, comparing
  each scale's output with 0.25 times a detached input reference. It does not
  enforce equal color occupancy.
- Uniform-color preservation is -(1/3) sum_a log P_C(a|a,a,a,a), weight0.1.
  This generalizes the Ising sign convention, without full permutation
  equivariance or constraints on mixed blocks.
- Geometry relative_local weight10, microscopic subset-consistency weight2
  (with the historical internal factor10), coarse subset-consistency5,
  C variance floor2, single-step closure5, all-scale rollout5. No covariance,
  weak-direction weighting, tangent or R-orientation penalty.

This is a finite-size, single-seed pilot. Fixed-point structure, microscopic
K interpolation, and closure should be evaluated separately. Neither correct
training loss nor a critical-looking curve certifies an exact RG or precise
critical exponent. The model is not constrained to be permutation equivariant;
color bias is a possible failure mode to inspect.

## First result

The 2400-update seed91 pilot did not recover the critical unstable fixed point.
The microscopic g(K) curve distinguishes the transition region, but direct
coarse-scale encodings contract; R has one detected stable root. See the
[run report](../results/potts2d_g1_probC_sign0p1_2400_seed91/REPORT.md) and
[map](../results/potts2d_g1_probC_sign0p1_2400_seed91/full_report/R_and_K_mapping.png).

## Preservation-weight-zero ablation

`configs/potts2d_g1_probC_sign0_2400_seed91.json` differs only in preservation
weight0. Following this comparison, new schema-version-4 configurations now
default to preservation weight0; historical explicit settings are retained.
At800 updates a three-root topology
appears with central inverse K approximately1.032, but the central root drifts
and is outside the data at2400. Coarse-scale latent spread is retained much
better than with weight0.1, without a correct final critical-point location.
See the [ablation report](../results/potts2d_g1_probC_sign0_2400_seed91/REPORT.md)
and [paired plots](../results/potts2d_preservation_comparison_seed91/R_and_K_comparison.png).

## Uniform-reference neutral ablation

The new paired configs `potts2d_g1_neutral0_2400_seed91.json` and
`potts2d_g1_neutral10_2400_seed91.json` retain the no-preservation pilot's data,
model and joint optimization settings. Only neutral weight differs (0 vs 10).
The exact 81-block reference term is
`0.5 * sum_a (mean_uniform P_C(a|V) - 1/3)^2`. It does not constrain real
ensemble color occupancies and does not train with RSMI. Both start afresh with
seed91, run2400 updates, and save every400; final and scheduled checkpoints
are reported without picking the best-looking epoch.

```bash
python -m recurrent_rg train --config configs/potts2d_g1_neutral0_2400_seed91.json --output results/potts2d_g1_neutral0_2400_seed91
python -m recurrent_rg train --config configs/potts2d_g1_neutral10_2400_seed91.json --output results/potts2d_g1_neutral10_2400_seed91
# Run these three reports for EACH output directory:
python -m scripts.report_scalar_joint --run results/potts2d_g1_neutral10_2400_seed91
python -m scripts.report_potts_scalar --run results/potts2d_g1_neutral10_2400_seed91
python -m scripts.audit_potts_checkpoints --run results/potts2d_g1_neutral10_2400_seed91
python -m scripts.compare_potts_preservation --baseline results/potts2d_g1_neutral0_2400_seed91 --ablation results/potts2d_g1_neutral10_2400_seed91 --term kernel_neutral_weight --output results/potts2d_neutral_comparison_seed91
```

### Completed neutral result

This seed91/weight10 trial did **not** recover the phase transition. The
baseline retains three roots, all outside observed R inputs (central unstable
g=.0969333, R'=1.070877, no microscopic K inverse). The neutral model has only
one detected stable root, g=1.325637, R'=.0468084. Its microscopic critical
encoding is g(Kc)=.184768. All six scheduled neutral checkpoints (400 through
2400) have one detected stable root; the final wider search [-6.121,6.986]
also finds only this root. Numerical root searches are bounded, not exhaustive.

Total test loss decreases .127915 -> .0337889, and neutral .00948573 ->
3.83827e-7. However, direct training latent SD at k=0,1,2 changes from
(.9530,1.0144,1.0331) to (.6985,.00505,.00997): microscopic Q still distinguishes
the transition, but both coarse levels are nearly independent of K.

This is information-destroying randomization, not a color permutation. The
neutral model's uniform-color input/output probability matrix is

```text
[[.33659, .33459, .32882],
 [.33041, .34576, .32383],
 [.33765, .33795, .32440]]
```

Over all81 uniformly weighted blocks, exact I(V;H) drops from .693989 to
.000220832 nats. This is a posthoc local input/output diagnostic, NOT environment
RSMI and NOT a training objective. Mean H(H|V)=1.098390 nats is almost ln(3).
The final C variance-floor term is zero: random colors have high sample variance
despite carrying almost no input information. Neutral and the existing variance
floor therefore both permit this randomized trivial solution. This observation
does not establish that every neutral weight fails; only weight10 was tested
against0, with one training seed. No default settings were changed.

- [R(g) and all-scale g(K) comparison](../results/potts2d_neutral_comparison_seed91/R_and_K_comparison.png)
- [Common training/test losses](../results/potts2d_neutral_comparison_seed91/common_loss_comparison.png)
- [All neutral checkpoints](../results/potts2d_g1_neutral10_2400_seed91/full_report/R_checkpoints.png)
- [Numerical audit](../results/potts2d_neutral_comparison_seed91/comparison.json)

## Label-free early-stopping audit

`python -m scripts.audit_potts_early_stopping` evaluates the no-neutral run's
six saved checkpoints (400:400:2400), without retraining. Rules are fixed before
new evaluation: minimum mean validation objective; online patience2 with1%
relative improvement; earliest checkpoint within one SEM of the best mean.
Independent validation and audit MCMC families each cover the21 original
held-out K cells with3 new chains/cell and64 configurations/chain (burn1024,
spacing8). All seed formulas and settings are saved in `protocol.json`.
Couplings determine sampling cells only; neither Kc nor root structure enters
checkpoint selection. The audit family is not used for selection.

Minimum validation and patience select2400 (no early stop triggered before the
budget ends). The one-SEM rule selects1600. Neither selects the approximately
critical800-step model. Fresh validation objective at800/1200/1600/2400 is
.094157/.088203/.077841/.069605; independent audit objective is
.237408/.212189/.194238/.167687. Audit two-step rollout NRMSE decreases from
6.35% at800 to3.90% at2400, despite deteriorating fixed-point placement.
Fresh microscopic K interpolation at800 is1.03590, at1200 is1.14234, and the
2400 repeller remains outside microscopic support.

Thus ordinary validation-loss early stopping does not solve this run's physical
drift. This is a retrospective single-training-seed test, not a prospective
multi-seed validation. SEM uses only three independent MCMC/kernel realizations;
the large audit SEM warns against interpreting small score differences as
statistically established. All three paired audit realizations nevertheless
decrease in total loss from800 to2400. No training defaults were changed.

- [Fresh validation/audit curves](../results/potts2d_early_stopping_seed91/validation_curves.png)
- [Checkpoint maps](../results/potts2d_early_stopping_seed91/checkpoint_flows.png)
- [Full numerical results](../results/potts2d_early_stopping_seed91/audit.json)

## Freeze Q/C at800; continue R to2400

```bash
python -m experiments.potts_frozen_qc_continue
python -m scripts.compare_potts_frozen_qc
python -m scripts.audit_potts_checkpoints --run results/potts2d_g1_freezeQC800_R2400_seed91
```

The intervention resumes the seed91 no-neutral checkpoint at800 and trains only
R for1600 further updates. It retains R's Adam moments/step, original learning
rate, batch-generator state, categorical RNG state, data and objective weights.
C remains stochastic but its weights are fixed. Q/C tensors and fresh-data
direct trajectories were verified bitwise identical to the800-step source.
An optimizer-state migration test verifies the next R update against the
unrestricted Adam optimizer. The shared training defaults are unchanged.

On the same three fresh independent audit-chain families used above:

| Run | Central repeller inverse microscopic K | Two-step rollout NRMSE |
|---|---|---|
| Joint checkpoint800 |1.035900|6.3868%|
| Joint continuation2400 |Outside data support|3.9025%|
| Frozen Q/C, R continuation2400 |1.022311|4.3531%|

The frozen run retains three roots at g=(-2.491107,-.854337,-.036064), with
R'=(.646405,1.198879,.708442): stable/unstable/stable. The low-g stable root
is outside observed R inputs; the high-g one is inside pooled observed inputs
but has no microscopic K inverse. These remain finite-size approximate flows,
not verification of exact RG or of all endpoint physics.

All saved frozen checkpoints retain the central repeller. On the ORIGINAL
cached ensembles, its inverse K at1200/1600/2000/2400 is
1.01877/1.02373/1.01486/1.01896 (distinct from the fresh audit values above).
This supports representation/coarse-graining drift during joint optimization
as an important cause, rather than R training duration alone. It does not
separate Q drift from C drift, nor establish a label-free rule for choosing800
or generalize beyond this training seed.

- [R and coupling comparison](../results/potts2d_frozenQC800_comparison_seed91/R_and_K_comparison.png)
- [Full and R-dependent loss curves](../results/potts2d_frozenQC800_comparison_seed91/loss_comparison.png)
- [Numerical comparison](../results/potts2d_frozenQC800_comparison_seed91/comparison.json)

## Cosine soft-freeze ablation

The schedule is fixed in advance: keep the exact joint prefix1..400, then
`s(t)=0.5*(1+cos(pi*clip((t-400)/800,0,1)))`. Q/C learning rate is `.001*s(t)`
and its gradient is `grad(L_QC+s(t)*L_R)`. R retains learning rate .001 and
gradient `grad(L_R)` throughout. Starting at1200, Q/C are fully frozen. This
tests softening the previous freeze, not a from-scratch QC-only warmup or an
additional ramp of R's weight. Adam moments and sampling RNG states are retained.
Gradient routing, schedule endpoints and Adam-group migration have unit tests.

```bash
python -m experiments.potts_soft_freeze
python -m scripts.compare_potts_frozen_qc --soft results/potts2d_g1_softQC400_1200_R2400_seed91 --output results/potts2d_softQC_comparison_seed91
python -m scripts.audit_potts_checkpoints --run results/potts2d_g1_softQC400_1200_R2400_seed91
```

At2400 the fresh-data central repeller inverse is K=1.092329, versus1.022311
for hard freeze and no supported inverse for full joint training. Its g=-.450086
and R'=1.096887. Two flanking stable roots remain, with derivatives .520018
and .895634; the low-g stable root is outside observed support. Soft-freeze
two-step rollout NRMSE is4.1164%, close to hard-freeze4.3531%, but phase location
is less accurate. Neither tiny NRMSE differences nor this single training seed
establish statistical superiority.

The original-cache checkpoint audit locates the repeller at K=1.03619 at800,
1.07026 at1200, none detected at1600,1.07177 at2000 and1.09111 at2400. Thus
the final three-root topology was not continuously retained. Q/C weights were
verified bitwise unchanged from the freeze onset through the end. Softening the
schedule therefore does not automatically prevent representation drift; the
representation fixed at1200 differs from the successful hard800 representation.
The experiment does not isolate Q from C or validate a universal freeze time.
No default training configuration was changed.

- [Four-way R and coupling comparison](../results/potts2d_softQC_comparison_seed91/R_and_K_comparison.png)
- [Common objective and R-dependent losses](../results/potts2d_softQC_comparison_seed91/loss_comparison.png)
- [Numerical comparison](../results/potts2d_softQC_comparison_seed91/comparison.json)
