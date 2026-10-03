# Uniform-block distinguishability: single-seed test

This is an optional experiment, not a default-loss change. Start from the same
seed91/data/configurations as the simultaneous separated-gradient QC/R runs;
train from scratch for 2400 updates. No warm start, early-stop selection,
R-only polishing, RSMI, sign-preservation or neutral term.

## Loss

For each uniform input block a^m, obtain its exact conditional output
probabilities p_a. Add mean over a<b of [delta-TV(p_a,p_b)]_+^2, delta=0.8.
The complete C term is 2*(L_variance_floor + alpha*L_distinct), alpha=1.
The new loss is assigned only to QC under the separated-gradient protocol.
Binary and categorical states use the same definition. Only q uniform blocks
are evaluated; no stochastic samples or all-pattern enumeration are needed.

The term permits output permutations, including binary sign reversal. It
assumes different uniform blocks should remain distinguishable; it does not
guarantee long-range information preservation, output equivariance, or an
autonomous closed RG coordinate. At exactly identical probability vectors,
absolute-value autodiff has zero subgradient; no tie-breaking modification was
introduced. These experiments retain random initialization.

## Reproduce

```bash
python -m experiments.separate_qc_r --source results/neutral_ising1d_g1_w0_seed91/qr --output results/ising1d_g1_separateDistinct1m0p8_2400_seed91/qr --distinct-alpha 1 --distinct-margin 0.8
python -m experiments.separate_qc_r --source results/neutral_ising2d_g2_w0_seed91/qr --output results/ising2d_g2_separateDistinct1m0p8_2400_seed91/qr --distinct-alpha 1 --distinct-margin 0.8
python -m scripts.compare_ising_frozen_qc --distinct
python -m scripts.report_rsmi_experiment --run results/ising1d_g1_separateDistinct1m0p8_2400_seed91 --flow-title 'Separate QC/R + uniform distinct, alpha=1 margin=0.8'
python -m scripts.report_rsmi_experiment --run results/ising2d_g2_separateDistinct1m0p8_2400_seed91 --flow-title 'Separate QC/R + uniform distinct, alpha=1 margin=0.8'
```

Data and model settings: see [previous comparison](ising_separate_gradients.md).
The historical report-script name does not imply RSMI training. Labels are
used only for post-training diagnostics. All 229 tests pass, including loss
accounting, RNG preservation, binary/categorical gradients and output
permutation invariance. New alpha defaults to zero, old resume configs gain
only that zero default and margin0.8.

## Results

| System / mode | First-step test NRMSE | Last-scale rollout NRMSE |
|---|---:|---:|
| 1D joint | 3.20% | 44.84% (k=5) |
| 1D separate | 21.97% | 74.33% |
| 1D separate + distinct | 18.96% | 43.80% |
| 2D joint | 4.07% | 5.58% (k=2) |
| 2D separate | 5.25% | 7.40% |
| 2D separate + distinct | 4.43% | 6.88% |

NRMSE uses each model's training direct-target standard deviation at that
scale; targets differ between models. Last-scale improvement alone does not
establish recovery of the phase portrait. All cached held-out ensembles are
used for these metrics; logged losses use the original evaluation subset.

### 1D classical Ising, dim g=1

P(+|--)=0.0001311, P(+|++)=0.9273788, separation=0.9272477 > 0.8.
The loss is zero at the final checkpoint. It prevents the previous merging
of both uniform blocks into negative output. Probability of a negative
output from ++ is still 7.26%; this term neither enforces determinism nor
spin-flip symmetry.

Finite-ring mean environment MI increases from 0.019195 to 0.325987 bits
(joint baseline 0.387446, exact decimation reference 0.392686). This is a
post-training diagnostic, not a loss term.

Only one root was found in the displayed bounded search: stable
g=-3.401197, DR=0.160765. Therefore the kernel defect improved, but the desired
repelling endpoint was not recovered in this search. No claim is made about
roots at arbitrary extrapolated coordinates or the infinite-J limit.

Final train/test totals: 0.135183/0.148558. Raw test components:
anchor=0.00928777, subset=0.00520054, C variance=0.000003590,
geometry=0.000322121, one-step=0.00752487, rollout=0.01262546,
distinct=0. The unchanged respective weights are 2,5,2,10,5,5,2.

![1D flow](../results/ising1d_g1_separateDistinct1m0p8_2400_seed91/qr/report/R_vs_g.png)
![1D coupling and MI](../results/ising1d_g1_separateDistinct1m0p8_2400_seed91/qr/report/exact_MI_and_coupling.png)

### 2D classical Ising, dim g=2

Uniform blocks still undergo near-deterministic sign reversal, not merging:
P(+|----)=0.999999689, P(+|++++)=0.000000347. Final distinct loss=0.

Central repeller g=(0.344585,0.167400), DR eigenvalues
(1.565538,-1.129272), both magnitudes >1. Microscopic piecewise-linear
inverse gives (K,B)=(0.428880,-0.000175920), versus
(0.429196,-0.000368079) without the new term. This is a coordinate diagnostic,
not a learned effective Hamiltonian. No substantial improvement in critical
K localization is established.

Other detected roots: saddle (-1.247433,-0.048195), stable
(2.623984,0.523218). Stable root is relatively far from training inputs.
Final train/test totals: 0.780747/0.767578.

![2D flow](../results/ising2d_g2_separateDistinct1m0p8_2400_seed91/qr/report/R_one_two_step.png)
![2D coupling map](../results/ising2d_g2_separateDistinct1m0p8_2400_seed91/qr/report/g_KB_maps.png)

## Loss curves

![1D loss](../results/ising_distinct_comparison_seed91/ising1d_loss.png)
![2D loss](../results/ising_distinct_comparison_seed91/ising2d_loss.png)

Conclusion: the targeted uniform-state merging defect is fixed in this seed,
with improved 1D environment information and no loss of the 2D central
repeller. It is not sufficient to recover the 1D RG portrait. Default remains
unchanged; stronger margins, altered feedback and extra training were not
tested here.
