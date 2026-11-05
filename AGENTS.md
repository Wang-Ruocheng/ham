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
- After joint training, all new standard pipelines must append a terminal
  R-only stage (user decision 2026-11-05): freeze final Q/C, build a fixed
  TRAIN-only cache using the same two-half-mean encoding convention, then
  continue the existing R and its Adam moments/current LR on the unchanged
  closure objective. No Q/C updates or MI optimization in this stage.
  Default: eight fixed subset/C draws per physical chain, up to 10,000 extra
  R updates, TRAIN-only plateau checks; distinguish plateau from update cap.
  Cache repeats are not independent physical chains. Preserve the joint
  checkpoint/metrics and report joint Q/C updates separately from total R
  updates. Budget the terminal stage within the authorized pipeline limits;
  do not silently skip it, increase budgets, or claim global convergence.
  `training.terminal_r_fit=true` is the fresh standard-driver default.
  Old saved configurations/checkpoints missing this field retain false;
  exact historical recipes and explicit joint-only ablations stay unchanged.
  New continuous-model adapters must use the same terminal-stage convention.
- New experiments default to `loss.closure_objective="all_start_anchor"`:
  `L_R = (10/S) sum_{t=1..S} (1/t) sum_{s=0..t-1}
  ||R^(t-s)(sg g_s) - sg g_t||^2`. Average target scales uniformly, then
  average all earlier starts within each target. R alone receives gradients;
  recurrent R calls retain gradients. No prediction-prediction penalty or
  future-observable loss is added. Keep the compatibility outer weights
  `commutator=5`, `rollout_weight=5` (combined weight 10).
- Leave `loss.closure_scale_weights` unset (`None`) or equal. Nonuniform
  weights require an explicitly requested legacy-objective ablation.
  Preserve every historical run/configuration, including weighted runs.
  Historical JSON/checkpoint loading with a missing closure mode retains
  `one_step_and_rollout`; for an exact fresh reproduction explicitly pin that
  mode in a new config. Fresh standard `--config` training with an omitted mode adopts
  the current all-start default and records the actual mode in its manifest.
  Historical `--source` and explicitly named diagnostic/ablation entry points
  keep their source policy; use a new explicit config for the current default.
- The all-start objective is Koopman-inspired multi-horizon consistency, not
  an assumption that nonlinear R is a finite-dimensional linear Koopman
  operator. The default change is for a unified definition, not a claim of
  empirically superior accuracy or certified fixed-point/exponent recovery.
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

# 用户绘图与计算资源偏好

- 今后新生成的图中，标题、坐标轴文字、图例、注释和说明全部使用中文；
  数学符号、公式和必要的变量名保留原记号。历史正式报告不因这一偏好自动覆盖或重跑。
- 二维流场默认沿用早期参数着色风格：横纵轴为两个潜变量（高维时明确标注所用投影），
  灰色箭头表示 R(g)-g 的方向，散点标记实际数据编码，颜色条表示生成数据时的真实参数。
  箭头保持等长，灰度深浅线性表示流动幅度 ||R(g)-g||：越小越浅、越大越深，
  标注灰度色标，同一流场各尺度图共用范围。此量不是 Jacobian 或 loss 的梯度。
  有两个扫描参数时并排展示同一流场、分别按两个参数着色；默认主图展示微观 k=0，
  其他尺度分图，不用尺度颜色替代参数颜色。不动点保留并标明外推性质。
  同一参数下物理链均值须明确标注；不默认为未采样区域插值填色。
  粗尺度图的颜色仍是原始参数，不冒充有效耦合；不把潜空间任意点反标为物理参数。
- 后续实验优先考虑 227 以外服务器的可用 GPU；选择时同时检查共享主机的
  CPU、内存和数据传输瓶颈，不仅看 GPU 型号或空闲显存。
  这一偏好不授权自行启动、迁移、停止或重启任何实验。
