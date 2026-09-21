# TFIM dim-g2 coordinate-sum control

Same learned-C model/data/seed/init and batch sequence as joint_qcr_tfim_1d_g2_levels6_rollout5_2000_seed91. Hidden60, optimized9893 parameters, seed91,2000 steps, L128→64→32→16→8→4.129/63 train/test ensembles,128 original configurations each, global-flip augmentation,32 systems/batch, Adam .001. No fixed C, no covariance utilization or balanced metric. Original finite-temperature Suzuki–Trotter cache retained.

Only change: --coordinate-reduction sum. Microscopic subset consistency, coarse subset consistency, one-step closure, and each multistep closure term sum squared errors over latent coordinates, then average over remaining dimensions. Weights unchanged. Geometry and kernel-balance loss unchanged. At dim2 these four terms are exactly twice their old values on the same tensors; dim1 objective unchanged. Historical default mean remains available.17 tests passed, including exact dimension scaling, unchanged geometry/kernel terms, rotation/isometric embedding invariance, and finite Q/C/R gradients.

## Results

Final new train/test objective .17464792/.16830930. Cross-evaluating both models on identical24-system/48-original-sample evaluation subsets:

| Model trained with | Mean-coordinate test objective | Sum-coordinate test objective |
|---|---|---|
| Original mean | .08617900 | .16997446 |
| New sum | .08644509 | .16830930 |

No meaningful overall gain under either common evaluation definition in this one-seed experiment. Full-test rollout NRMSE k1..5 changes from [.085765,.055617,.057555,.085532,.142304] to [.077922,.053988,.059281,.084534,.140904]. Differences modest and not uniformly favorable.

Numerical search with81 starts inside plotted box still finds only one stable root: (.34807869,-.13472340), DR eigenvalues .71120530,.50959711. Search not exhaustive; off-data arrows are extrapolation. Microscopic training weak-PC variance fraction1.58678e-6 (0.00015868%); near-collinearity remains.

Final C remains [--,-+,+-,++]→[+,-,+,-], negative second-spin decimation, same truth table as original dim2 run. The long-term drift persists: at h/J=.01 all direct Q(C^kD) means equal (-.91753549,1.26961156), but50 R steps reach (.34758892,-.13560784), near the other stable root. Steps beyond5 are untrained extrapolation, used only as a numerical diagnostic.

Conclusion: eliminating division by latent dimension is mathematically consistent, but this coefficient normalization alone does not recover the missing fixed-point structure. No claim of causal uniqueness or multi-seed significance.

## Artifacts and reproduction

R_vector_field.png; g_vs_h_over_J.png; vector_training_diagnostics.png; closure_fixed_PCs.png; comparison.json (both objective conventions); summary.json; checkpoints500/1000/1500/2000. See experiments/README.md's learned-C dim2 command, adding --coordinate-reduction sum and this output directory. Then run the vector/coupling plotters and scripts/compare_tfim_coordinate_reduction.py. Previous runs preserved.
