# QCR joint training over all nondegenerate L128 scales

Seed91, dim(g)=1, hidden60, 9711 optimized QCR parameters, nonoverlapping two-spin
C. Same cached train/test data (16512/8064 original configurations), sample128,
system_batch32, Adam lr0.001, 2000 updates. No A, RSMI, parameter labels or new
symmetry constraint. Original runs are preserved.

Change: levels4 -> levels6, so actual lattices are 128,64,32,16,8,4 sites.
One-step loss averages five transitions; multistep loss averages horizons1..5;
data geometry averages scales0..5. Other coarse-scale terms expand automatically.
Total weights stay unchanged (closure5+5, geometry10, C balance2, split terms).
Thus each term's per-scale share changes; totals are not directly comparable
with the old shorter-depth objective. n2/n1 are excluded as discussed with user.
R-only iteration beyond k5 remains unsupervised.

## Full-test NRMSE (all 63 test ensembles, 128 samples plus flips)

| Depth / updates | k1 | k2 | k3 | k4 | k5 |
|---|---:|---:|---:|---:|---:|
| Old k<=3 / 1000 | .06523 | .04890 | .05539 | .10234 | .16836 |
| New k<=5 / 1000 | .10760 | .06409 | .06486 | .09975 | .09107 |
| Old k<=3 / 2000 | .07886 | .07394 | .08284 | .12714 | .19158 |
| New k<=5 / 2000 | .08270 | .07493 | .09146 | .16130 | .18665 |

At 1000 updates, k5 improves by ~46% relative while early-scale errors increase.
At 2000 the improvement does not persist. There is no reliable overall win.
These are predeclared checkpoints, not a test-selected optimum.

New total train/test losses: 500 .180114/.174593, 1000 .200104/.201951,
2000 .182524/.194528. Loss uses the original fixed smaller evaluation subset
(24 systems,48 original samples+flips); this differs from full-test metrics.

## Important kernel change and trajectory details

At both 1000 and 2000, learned C maps [--,-+,+-,++] -> [+,+,+,-]: only ++
produces -. It is no longer decimation and is not odd under global spin flip.
No hard symmetry constraint was added. C balance is a soft batch-marginal
penalty, not exact equivariance; its test unweighted value is .015823 at these
checkpoints. Consequently, the new trajectory differences cannot be attributed
solely to R. Alternating-scale changes in direct Q also deserve attention.
There is no global Q collapse: final full-test std by scale is
[.4498,.6278,.7679,.8522,.9182,.9035].

For h/J=.957619 and1.052381, three-replica mean k5 actual/predicted g:

- 1000: ordered-side -.45642/-.42595; disordered-side .89702/.94424.
- 2000: ordered-side -.42345/.04611; disordered-side .97335/.98231.

Autonomous20-step means at1000 are -.84759 and1.01028 (separated). At2000
both approach1.03211 again. k20 has no actual-data target; these are learned
map extrapolations and must not be advertised as verified physical RG phases.
No critical point is identified from these numbers.

## Artifacts

- `depth_comparison.png/json`: matched-scale comparison with old models.
- `training_duration_comparison.png`, `duration_comparison.json`: loss curves,
  R maps and trajectories at500/1000/2000 updates.
- `qcr_closure_comparison.png/json`, `g_by_scale_vs_h_over_J.png`: final full
  ensemble closure and scale-dependent coordinates.
- `training_step_1000.pt` and `training_step_2000.pt`: model+optimizer+RNG states.
- `qcr_model.pt`, `summary.json`: final model and configuration/history.

Reproduce with the prior command, `--levels 6 --epochs 2000 --rollout-weight 5`
and this output directory. The six-scale gradient/averaging regression test
passes; all58 tests pass, plots visually checked, git diff --check clean.
