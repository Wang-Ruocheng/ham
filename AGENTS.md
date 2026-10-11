# Fixed training convention

Unless the user explicitly requests an ablation, all new QCR experiments must use:

- Randomly initialized, trainable probabilistic C; no kernel selection or pretraining.
- New 2D experiments default to nonoverlapping 3x3 C with stride 3;
  1D remains a two-site block with stride 2. Use schema_version 5 for new
  configurations and choose lattice sizes divisible by 3 at every requested
  2D scale (e.g. 36 -> 12 -> 4). Preserve explicit historical block sizes
  for reproductions; do not silently reuse block-2 settings for new experiments.
- C objective: -2 * empirical histogram mutual information + 2 * per-ensemble
  variance-floor loss (ratio 1) + 1 * uniform-block color-preservation NLL.
- Buffered full-shell MI (buffer 1, microscopic scale); no InfoNCE substitution.
- Simultaneous independent gradients: Q gets representation objectives only;
  C gets its objective only; R gets one-step/multistep closure only.
  Q loss must not update C; R loss must not update Q or C.
- Q now defaults to a shared affine reconstruction of local motif probabilities
  at every scale (weight 10), plus the existing microscopic/coarse subset
  consistency terms (effective weights 20/5). Normalize decoder columns to
  unit norm; do not force full rank. No PCA initialization by default. Preserve
  explicit historical geometry choices and checkpoint architectures.
- Use `python -m recurrent_rg train --config ... --output ...` or
  `python -m experiments.rsmi_separate_qcr --config ... --output ...`.
- Do not enable `--allow-nondefault-recipe`, `--allow-legacy-training`, or
  `allow_legacy_training=True` without an explicit user request to change the recipe.

Historical manifests/checkpoints are evidence, not current defaults. Preserve them.
Reproductions must match actual data fingerprints, model, loss, gradient routing,
seed and training length; disclose any difference before claiming reproduction.
Coupling labels are for data generation and evaluation, not training supervision.
Always report R(g) (plot arrows as R(g)-g), coupling-to-g maps, and loss curves.
