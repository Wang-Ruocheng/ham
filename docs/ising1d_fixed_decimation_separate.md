# Fixed positive decimation versus learned RSMI C

1D classical Ising only, dim_g1, seed91,2400 updates from scratch. Same data,
Q/R architecture, initial Q/R tensors, Adam0.001, batches and five RG steps
as the preceding RSMI separated-gradient experiment. C is now parameter-free
subsampling x[:,:,::2], not a trainable positive initialization. Q minimizes
geometry/subset losses; R minimizes one-step/multistep closure with no feedback
to Q. No RSMI critic is trained since fixed C has no information gradient;
all C auxiliary weights remain0. Default training mode is unchanged.

```bash
python -m experiments.separate_qc_r --source results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr --output results/ising1d_g1_fixedPositiveSeparate2400_seed91/qr --fixed-decimation
python -m scripts.compare_ising_frozen_qc --fixed
python -m scripts.audit_ising1d_rsmi_loss --run results/ising1d_g1_fixedPositiveSeparate2400_seed91/qr
python -m scripts.report_rsmi_experiment --run results/ising1d_g1_fixedPositiveSeparate2400_seed91 --flow-title 'Fixed positive decimation; separate Q/R, 2400 updates'
```

## Main result

The learned flow recovers one attractor and one repeller in the numerical
search interval. Repeller g=-0.092385, derivative1.605836, outside observed
training input range. Attractor g=1.838080, derivative-0.121168, within range.
This is a structural improvement, not a precise validation of the infinite-J
endpoint. Population Q values at J0 and symmetric J-infinity are1.851091 and
-0.025396, respectively, so the learned roots do not coincide exactly with
these endpoint coordinates.

| Metric | Learned RSMI C | Fixed positive decimation |
|---|---:|---:|
| Weighted test one-step+rollout loss | 0.20650 | 0.03043 |
| Test first-step NRMSE | 12.76% | 2.63% |
| Test k1->2 teacher-forced NRMSE | 32.80% | 4.24% |
| Test k5 rollout NRMSE | 13.40% | 57.32% |
| Test k5 rollout raw RMSE | 0.08218 | 0.10253 |

NRMSE denominators are each model's per-scale training target SD; these
representations differ, so not all percentages compare a common physical
quantity. The final-scale raw error also worsens, and is not hidden by the
improved shallow closure. Final train/test total0.159595/0.150427 (no RSMI).
Weighted test components: anchor0.022782, coarse subset0.090398,
geometry0.006818, one-step0.014558, rollout0.015871. Remaining logged loss is
mainly coarse subset consistency. RSMI comparison curves exclude the MI term.

## Exact finite-ring population audit

The learned RSMI C was nearly AND. Fixed decimation eliminates the negative
magnetization drift: all means remain0 to numerical precision.
Population one-step MSE at k1->2 falls from0.064881 to0.00008268.
The previously separated transition curves nearly coincide. Among the
monotonic scale pairs checked, maximum same-g/next-g mismatch drops from
0.37833 to0.00587. This comparison uses different learned latent coordinates
and is not a coordinate-invariant physical error; the visual agreement and
zero generated bias provide complementary evidence.

This strongly supports the C-induced distribution shift as an important cause
of the earlier closure problem in this seed. It does not establish that all
RSMI kernels, all one-dimensional representations or all seeds behave this
way. Residual finite-sample/coarse-scale errors and endpoint extrapolation
remain. No further polishing or model changes are included.

![R(g)](../results/ising1d_g1_fixedPositiveSeparate2400_seed91/qr/report/R_vs_g.png)
![Exact population J mapping and closure](../results/ising1d_g1_fixedPositiveSeparate2400_seed91/qr/loss_audit/exact_scale_audit.png)
![Loss comparison](../results/ising1d_fixedPositive_comparison_seed91/ising1d_loss.png)

Validation includes actual subsampling output, identical paired Q/R
initializations, and bitwise-exact separated-QR resume with fixed C.
