# Hard-freeze transfer test: 1D and 2D classical Ising

Single seed 91. Reuse the original joint-training checkpoint at update 800,
freeze Q/C exactly, and train only R for another 1600 updates (total 2400).
R Adam moments and step, batch-generator state, global torch RNG, objective,
weights and data cache are retained. No coupling labels enter training.
No neutral, RSMI or sign-preservation loss is added.

The continuation script retains its historical Potts filename but now selects
the same system-specific statistics and spin-flip augmentation as the standard
training runner.

## Reproduce

```bash
python -m experiments.potts_frozen_qc_continue --source results/neutral_ising1d_g1_w0_seed91/qr --output results/ising1d_g1_freezeQC800_R2400_seed91/qr
python -m experiments.potts_frozen_qc_continue --source results/neutral_ising2d_g2_w0_seed91/qr --output results/ising2d_g2_freezeQC800_R2400_seed91/qr
python -m scripts.compare_ising_frozen_qc
python -c "from recurrent_rg.runner import evaluate_run; evaluate_run('results/ising2d_g2_freezeQC800_R2400_seed91/qr',checkpoint=2400,output='results/ising2d_g2_freezeQC800_R2400_seed91/qr/report')"
python -m scripts.report_rsmi_experiment --run results/ising2d_g2_freezeQC800_R2400_seed91 --checkpoint 2400 --flow-title 'QC frozen at 800; R continued to 2400'
```

Report scripts with historical RSMI names only perform diagnostics; this
experiment does not train with RSMI. Output directories must not already exist
when starting a continuation.

## Data and settings

- 1D: dim_g=1, classical Ising J in [0.01,2], 64 uniform values,
  3 ensembles/value, 128 configurations/ensemble, 129 training/63 test
  ensembles, L=128, five RG steps, spin-flip paired augmentation retained.
- 2D: dim_g=2, K=0.20:0.03:0.68, B=-0.01:0.001:0.01,
  64 configurations/ensemble, 474 training/240 test ensembles, L=16,
  two RG steps, no spin-flip augmentation.
- Hidden width 60, random-initialized probabilistic binary C, nonoverlapping
  factor-two blocking. relative_local geometry; variance_floor C penalty
  with weight 2; no weak-direction weighting. Batch 32 ensembles x 64
  configurations; Adam learning rate 0.001.

## Results

NRMSE below uses the corresponding model's training target standard deviation;
comparison between joint and frozen models therefore uses different target
representations. Update800 and frozen2400 have exactly identical direct
targets under paired stochastic sampling, verified along with Q/C tensors.

| System / protocol | k=1 rollout NRMSE | Last-scale rollout NRMSE |
|---|---:|---:|
| 1D joint800 | 4.53% | 48.51% (k=5) |
| 1D joint2400 | 3.20% | 44.84% |
| 1D frozen2400 | 2.72% | 48.63% |
| 2D joint800 | 2.89% | 5.94% (k=2) |
| 2D joint2400 | 4.07% | 5.58% |
| 2D frozen2400 | 2.31% | 5.19% |

1D preserves a stable root (g=1.301934, derivative 0.316360) and an unstable
root (g=-0.569589, derivative 1.687787). The unstable root is outside observed
training inputs: this is not validation of an infinite-coupling fixed point.
The highest-scale closure error remains substantial despite improved shallow
closure. Frozen final train/test total loss is 0.08761/0.10920.

![1D comparison](../results/ising_frozenQC800_comparison_seed91/ising1d_R_and_J.png)

2D fails to preserve a useful two-dimensional phase portrait when frozen this
early. At update800, microscopic latent covariance eigenvalues are
[0.00009948, 0.858812], ratio 0.000116. Joint2400 develops
[0.202846, 1.129756], ratio 0.1795. Freezing cannot repair that representation.
The frozen model's near-data root is a saddle at (0.468987,0.220741), with
DR eigenvalues (0.959230,-1.014019). Another repeller at
(-0.805451,0.090512) is far from data (nearest training input distance 1.039).
It is not evidence for recovering the physical critical point.
Frozen final train/test total loss is 1.59785/1.84264. The weighted test
R-dependent loss is only 0.01493; most remaining loss belongs to frozen Q/C.

![2D frozen flow](../results/ising2d_g2_freezeQC800_R2400_seed91/qr/report/R_one_two_step.png)

![1D losses](../results/ising_frozenQC800_comparison_seed91/ising1d_loss.png)
![2D losses](../results/ising_frozenQC800_comparison_seed91/ising2d_loss.png)

Conclusion: a fixed freeze time of 800 does not transfer reliably between
systems. This tests the schedule, not whether staged training can ever work
on 2D Ising. A later or representation-dependent freeze is a follow-up, not
part of these results. Numerical root search is bounded, not exhaustive;
off-data arrows are extrapolation. R-squared plots show composition, and
their stars mark roots of R only.
