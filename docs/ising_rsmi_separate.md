# Simultaneous QC/R separated gradients with RSMI

Single seed91, from scratch, final2400 updates. The binary 1D/2D Ising data,
QCR architecture, seed, batching and Adam learning rate match the earlier
[separated-gradient experiment](ising_separate_gradients.md). No frozen phase,
early-stop selection or final R-only polishing is added.

## Gradient and sampling contract

- QC minimizes L_geometry/subset - 2 I_NCE. C receives gradients through the
  Q representation losses as well as RSMI; Q has no direct RSMI dependence.
- R minimizes 5 L_one-step + 5 L_rollout, with no gradient feedback to QC.
- A separate environment critic minimizes -2 I_NCE. Its optimizer is separate;
  Q/C/R retain parameterwise Adam at the original learning rate0.001.
- RSMI replaces all C auxiliary penalties: variance-floor weight0, neutral0,
  distinct0, sign-preservation0. No coupling/phase labels enter training.
- Reuse the existing enumerated-binary-H InfoNCE implementation: microscopic
  scale only, factor-two visible block, one-site buffer, environment shell
  size2 in 1D /20 in 2D. Negatives never cross ensemble boundaries.
- Each update adds 16 ensembles x64 visible/environment pairs from the same
  cache. There is no new generated data, but extra cached-data sampling and
  compute. Each pair comes from a distinct configuration within that ensemble.
- Critic hidden64, 4482 extra parameters in1D /5634 in2D; QCR9711/10373.
  Critic initialization and region sampling use isolated RNG streams.
- Checkpoints save QCR and critic weights, both optimizers, and both sampling
  generators. Generic QCR resume explicitly rejects this experiment's
  checkpoints to avoid silently omitting the critic/RSMI objective.

This is simultaneous training, not the older RSMI-only C pretraining followed
by frozen-C QR. Empirical InfoNCE is not an exact population MI measurement.
Full total losses include a negative MI term and cannot be directly compared
to earlier positive objectives.

## Reproduce

```bash
python -m experiments.rsmi_separate_qcr --source results/neutral_ising1d_g1_w0_seed91/qr --output results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr
python -m experiments.rsmi_separate_qcr --source results/neutral_ising2d_g2_w0_seed91/qr --output results/ising2d_g2_rsmiSeparate_w2_2400_seed91/qr
python -m scripts.compare_ising_frozen_qc --rsmi
python -m scripts.report_rsmi_experiment --run results/ising1d_g1_rsmiSeparate_w2_2400_seed91 --flow-title 'Simultaneous separate QC/R + RSMI weight 2'
python -m scripts.report_rsmi_experiment --run results/ising2d_g2_rsmiSeparate_w2_2400_seed91 --flow-title 'Simultaneous separate QC/R + RSMI weight 2'
```

Recorded training/evaluation-loop times: 251.5s in1D,277.0s in2D (concurrent).
All231 tests pass, including explicit QC/R/critic gradient routing and
deterministic, RNG-isolated information evaluation. No default loss changed.

## Results

| System / protocol | First-step test NRMSE | Last-scale rollout NRMSE |
|---|---:|---:|
| 1D joint | 3.20% | 44.84% (k=5) |
| 1D separate | 21.97% | 74.33% |
| 1D separate + distinct | 18.96% | 43.80% |
| 1D separate + RSMI | 12.76% | 13.40% |
| 2D joint | 4.07% | 5.58% (k=2) |
| 2D separate | 5.25% | 7.40% |
| 2D separate + distinct | 4.43% | 6.88% |
| 2D separate + RSMI | 3.84% | 5.53% |

Each model's training target SD normalizes its own held-out errors. Thus
different representations imply different denominators; this is not a common
physical error metric. All held-out cached ensembles are used, while logged
losses use the original evaluation subset plus fixed-seed RSMI audit batches.

### 1D: information retained, missing repeller remains

Exact finite-ring environment MI averaged over the J grid is0.391392 bits,
versus0.392686 for decimation. Earlier separated-gradient C gave0.019195,
and distinct loss gave0.325987. Uniform probabilities:
P(+|--)=0.000000302, P(+|++)=0.999790453.

The empirical critic estimate on held-out data is0.259062 bits, versus
-0.301394 after within-ensemble shuffling. The discrepancy from the exact
transfer-matrix average is not a contradiction: estimator, sampled coupling
weights and finite data differ.

The bounded search finds only an attractor at g=-2.992211, DR=0.157841,
just outside the observed training input interval. No repeller was found in
the plotted interval. High environment MI alone does not prove the kernel is
exact decimation or the learned coordinate is a complete RG coupling.

Final held-out QCR loss excluding RSMI=0.270260, total including RSMI=-0.088877.
This result rules out uniform-state merging as the sole explanation of the
remaining portrait error; it does not yet distinguish imperfect Q closure,
moving targets, R optimization, or finite-range extrapolation.

![1D R](../results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr/report/R_vs_g.png)
![1D J mapping and exact MI](../results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr/report/exact_MI_and_coupling.png)

### 2D: central repeller preserved without sign reversal

Uniform inputs are almost deterministically color-preserving:
P(+|----)=1.831e-8, P(+|++++)=0.9999999783.
Central root g=(0.396022,0.152287), DR eigenvalues(1.569195,1.140087), both
positive and greater than1. Microscopic inverse(K,B)=(0.426567,0.000165263).
This inverse is a coordinate diagnostic, not effective RG couplings; K is
about3.2% below Kc. It is not evidence that the whole phase diagram is correct.

The other bounded-search roots are two saddles and one attractor, several
away from the observed manifold. The attractor's largest eigenvalue0.98817
and some saddle eigenvalues near1 should not be treated as robust universal
features. R-squared stars mark R roots only.

Held-out InfoNCE0.213346 bits, shuffled -0.407391; QCR loss0.756504,
total with RSMI0.460744.

![2D flow](../results/ising2d_g2_rsmiSeparate_w2_2400_seed91/qr/report/R_one_two_step.png)
![2D KB mapping](../results/ising2d_g2_rsmiSeparate_w2_2400_seed91/qr/report/g_KB_maps.png)

## Curves

![1D losses](../results/ising_rsmiSeparate_comparison_seed91/ising1d_loss.png)
![2D losses](../results/ising_rsmiSeparate_comparison_seed91/ising2d_loss.png)
![1D InfoNCE](../results/ising_rsmiSeparate_comparison_seed91/ising1d_rsmi.png)
![2D InfoNCE](../results/ising_rsmiSeparate_comparison_seed91/ising2d_rsmi.png)

## Follow-up: locating the 1D failure (no retraining)

The complete kernel table is nearly AND, not decimation:
P(+|--)=3.01975e-7, P(+|+-)=6.11076e-5,
P(+|-+)=5.72329e-5, P(+|++)=0.99979045. Mixed blocks both map negative.
High buffered MI therefore did not establish preservation of the zero-field
family. It rules out uniform-state merging, not all problematic C behavior.

Weighted held-out loss excluding RSMI:
anchor0.017344, coarse subset0.037108, geometry0.009308,
one-step0.101114, rollout0.105385, sum0.270260.
R closure accounts for76.4%. Logged rollout MSE for k1..5 is
[0.006776,0.035459,0.039827,0.014872,0.008451]; it peaks at k2/3, not k5.

`python -m scripts.audit_ising1d_rsmi_loss` performs an exact finite-ring
probability audit of the learned stochastic C (L128), without Monte Carlo.
Block transfer matrices obey M_h(k+1)=sum_ab C(h|a,b)M_a(k)M_b(k).
Three-coarse-spin probabilities then provide the exact population pooled
features for the actual Q. The nonlinear Q head is applied after pooling;
this is an infinite-ensemble audit at fixed finite chain length, not the
expectation of finite-sample Q outputs. Tests compare the recursion to full
L8 enumeration and to independent-spin AND and decimation limits.

Uniform dense J in[0.01,2] yields one-step MSE at k0..4:
[0.005463,0.064881,0.003918,0.009349,0.008867].
The k1->2 transition contributes70.2% of their sum. Sampling noise is thus
not sufficient to explain the discrepancy.

Population-coordinate collision example (interpolated on dense J grid):
at J~0.018315,k1, g~-1.286323 and next g~-2.391271;
at J~1.533008,k4, g~-1.286323 and next g~-2.012938.
The next targets differ by0.37833, so a deterministic shared scalar R cannot
fit both exactly with Q/C fixed. This establishes a closure defect of the
current representation; it does not prove no alternative scalar Q can work.

At J1 the exact mean spin sequence is approximately
[0,-0.11937,-0.31707,-0.58927,-0.85143,-0.98055]. Thus C generates bias from
zero-field input immediately. The resulting distributions cannot simply be
identified with the standard zero-field decimation J-only trajectory.

![Exact scale audit](../results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr/loss_audit/exact_scale_audit.png)

Diagnosis: the observable residual is mostly in R closure, especially1->2,
but there are incompatible cross-scale targets in the current Q/C coordinate.
The RSMI objective only constrains microscopic environment information, while
geometry loss constrains within-scale distances and R gradients cannot alter
QC in this experiment. C-induced bias and unaligned/compressed Q coordinates
therefore remain possible even with high MI and low geometry loss. More R
training may reduce optimization error but cannot remove the demonstrated
target incompatibility. No new training or fix is performed in this audit.
