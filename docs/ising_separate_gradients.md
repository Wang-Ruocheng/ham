# Simultaneous QC/R training with separate gradients

Seed 91, 2400 updates from scratch. Paired with the same source configurations,
data caches, model construction seed, batching, loss weights and learning rate
as the earlier joint Ising runs. Only `training.gradient_mode` changes from
`joint` to `qc_r_separate`. No final R-only polishing is included.

## Objective and implementation

With the current settings, L_R = 5 L_commutator + 5 L_rollout,
and L_QC = L_total - L_R. Q/C receive only grad(L_QC), while R receives only
grad(L_R). `torch.autograd.grad` selects the parameter sets on a common forward
sample. This is equivalent for the R update to detaching Q/C encodings at both
inputs and targets, while retaining the full recurrent R gradient chain.
Adam is parameterwise with a shared unchanged learning rate; two optimizers
are not needed. There is no joint gradient clipping or cross-parameter update.

The optional training mode is persisted in manifests and resume checkpoints.
The default remains joint. Tangent/orientation losses are explicitly rejected
in this mode until their gradient allocation is separately specified.

Checks: full suite 219 passed, including actual-model QC-gradient invariance
to changes in R, detached multi-step gradient equivalence, and exact resumed
training in both spatial dimensions.

## Commands

```bash
python -m experiments.separate_qc_r --source results/neutral_ising1d_g1_w0_seed91/qr --output results/ising1d_g1_separateQCR2400_seed91/qr
python -m experiments.separate_qc_r --source results/neutral_ising2d_g2_w0_seed91/qr --output results/ising2d_g2_separateQCR2400_seed91/qr
python -m scripts.compare_ising_frozen_qc --separate
python -m scripts.report_rsmi_experiment --run results/ising1d_g1_separateQCR2400_seed91 --flow-title 'Separate QC/R gradients, 2400 updates'
python -m scripts.report_rsmi_experiment --run results/ising2d_g2_separateQCR2400_seed91 --flow-title 'Separate QC/R gradients, 2400 updates'
```

The report script's historical RSMI filename does not indicate an RSMI training
loss. Couplings and mutual information are post-training diagnostics only.
See [the matched freeze experiment](ising_frozen_qc.md) for exact data settings.

## Results

| System | Protocol | Test total | First-step NRMSE | Last-scale rollout NRMSE |
|---|---|---:|---:|---:|
| 1D, dim g=1 | Joint2400 | 0.12286 | 3.20% | 44.84% (k=5) |
| 1D, dim g=1 | Separate2400 | 0.07248 | 21.97% | 74.33% |
| 2D, dim g=2 | Joint2400 | 0.75899 | 4.07% | 5.58% (k=2) |
| 2D, dim g=2 | Separate2400 | 0.78825 | 5.25% | 7.40% |

NRMSE is normalized by each model's training direct-target SD at that scale;
different protocols learn different representations. Metrics cover all cached
held-out ensembles; training-curve losses use the original fixed evaluation
subset. All reported outcomes use the final checkpoint, not selected epochs.

### 1D: worse despite a lower total loss

Only a stable fixed point was found in the plotted/search interval:
g=-1.618441, DR=0.670491. No unstable point was found in this bounded search;
this does not exclude roots arbitrarily far outside the data.

The learned C maps uniform -- and ++ blocks to - with probabilities 0.999695
and 0.996146 respectively. This is not a simple global sign reversal: both
ordered colors are merged. The finite-ring transfer-matrix diagnostic averaged
over sampled J gives I(H;E)=0.019195 bits, versus 0.387446 for joint2400 and
0.392686 for reference decimation. This is evidence of substantial information
loss, not merely R lagging behind its target. The current C variance-floor
penalty is nevertheless small (raw test value 0.00005244).

Weighted test components:

| Component | Joint | Separate |
|---|---:|---:|
| 2 anchor | 0.020294 | 0.019455 |
| 5 subset consistency | 0.048691 | 0.018012 |
| 2 C variance floor | 0 | 0.000105 |
| 10 geometry | 0.032897 | 0.003882 |
| 5 one-step | 0.008627 | 0.014145 |
| 5 rollout | 0.012354 | 0.016880 |

Lower QC loss is not evidence for a better RG kernel: the coarse-scale
distributions themselves change with C and may become easier to describe.

![1D R](../results/ising1d_g1_separateQCR2400_seed91/qr/report/R_vs_g.png)
![1D information and coupling](../results/ising1d_g1_separateQCR2400_seed91/qr/report/exact_MI_and_coupling.png)

### 2D: representation unfolds and central repeller remains

Microscopic covariance eigenvalues [0.201562,1.218423], ratio 0.1654;
joint2400 ratio 0.1795. Unlike freezing at update800, separate-gradient training
allows the two-dimensional representation to develop.

Central root g=(0.512471,0.188912), DR eigenvalues (1.583214,-1.149204):
both magnitudes exceed one, hence a discrete-time repeller. Microscopic
piecewise-linear inverse gives (K,B)=(0.429196,-0.00036808), approximately
2.6% below Kc in K. This is a coordinate diagnostic, not an effective-coupling
measurement or a proof of the complete phase diagram.

The other detected roots are a saddle at (-1.063756,-0.024042) and an attractor
at (2.867947,0.565514); the latter is relatively far from training inputs.
R-squared arrows show composition; stars mark roots of R only.

![2D flow](../results/ising2d_g2_separateQCR2400_seed91/qr/report/R_one_two_step.png)
![2D coupling map](../results/ising2d_g2_separateQCR2400_seed91/qr/report/g_KB_maps.png)

## Loss curves

![1D losses](../results/ising_separateQCR_comparison_seed91/ising1d_loss.png)
![2D losses](../results/ising_separateQCR_comparison_seed91/ising2d_loss.png)

This single-seed experiment does not establish separate gradients as a better
default. It removes a potential collapse feedback path but also removes a
useful dynamical constraint on C. Follow-up choices include holding C fixed to
isolate Q/R effects, or allowing selected feedback to C; neither was tested
in this experiment.
