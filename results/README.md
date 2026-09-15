# Generated results

`recurrent_rg/` contains the principal supervised and pseudo-likelihood runs.
JSON summaries and figures are lightweight records; PyTorch checkpoints are
local generated artifacts and are ignored by Git.

`recurrent_rg/unsupervised_joint_2d_z2_constrained/` contains the earlier
seed-41 configuration-only 2D Ising run with explicit Z2 symmetry in the
architecture.  It is retained as a legacy comparison and its checkpoint is not
compatible with the current general local-MLP kernel.  The earlier diagnostic
run is kept in `recurrent_rg/unsupervised_joint_2d_unconstrained/`.

The current symmetry-unconstrained, hard-field InfoNCE runs are stored under
`recurrent_rg/multiseed_1d/` (seeds 31--33) and
`recurrent_rg/multiseed_2d/` (seeds 41--43).  Aggregate statistics and figures
are in `recurrent_rg/multiseed_summary.json`,
`recurrent_rg/multiseed_loss_curves.png`, and
`recurrent_rg/multiseed_action_relative_error.png`.  Only these six current
runs retain model checkpoints; legacy checkpoints were removed before training.

`ensemble_encoder_ising_1d/` is the configuration-only smoke test that trains
only the unchanged ensemble encoder `g = Q(D)`. Its summary separates the
random-initialization baseline from the learned representation; the coupling
and correlation observables are evaluation-only probes.

`joint_qcr_ising_1d/` jointly trains Q, C, and R from configurations after the
Q-only initialization. It deliberately contains no RSMI or action objective.

`joint_qcr_ising_1d_multilevel_sep/` repeats that ablation with anti-collapse
penalties on every RG transition, weighted by 1, 0.5, and 0.25.

`joint_qcr_ising_1d_adaptive_sep/` uses the same decaying weights but gates each
transition from the source-level separation SNR: off below 2, on above 3, and
linearly interpolated between them. The gate and source target are detached.

`qcr_basin_analysis/` contains the fixed-point and basin analysis of that
adaptive-separation QCR checkpoint. Extrapolated latent coordinates are marked
separately from the R-input interval covered during training.

`joint_qcr_ising_2d_unbalanced/` preserves the failed first QCR run. Its Wolff
draws were globally balanced but individual ordered-phase ensembles were
locked into one magnetization sector, so Q learned the sign shortcut.

`joint_qcr_ising_2d/` is the corrected label-free QCR run. The sampler removes
sector locking and every Q training view is exactly spin-flip paired. The
learned scalar R has one data-supported unstable fixed point; its posthoc
critical-coupling calibration is recorded in `critical_flow_analysis.json`.

`fixed_majority_qr_ising_2d/` is the controlled ablation with the same data and
training schedule but a fixed Z2-equivariant 2x2 majority-rule C. Only Q and R
are optimized. It localizes the critical coupling substantially better than
the learned-C run, while its fixed-point derivative remains far below the
universal thermal eigenvalue.
