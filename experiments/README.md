# Auxiliary experiments

公共训练、loss、数据组织、检查点和评估已迁至 `recurrent_rg/`。
通用实验使用 `python -m recurrent_rg` 和 `configs/` 中的配置，当前独立梯度 MI 配方使用 `python -m experiments.rsmi_separate_qcr`；二者训练目标不同，不可直接互换。详见
[开发指南](../docs/development.md)。以下特定实验脚本保留作为历史配方，
其旧导入路径和检查点仍兼容；无需继续复制通用训练循环。

精确经验 MI 的一维 Ising 修正实验（逐系综防塌缩、数据选核初始化、Q/C/R 独立梯度）
见 [实验记录与复现命令](../docs/ising1d_mi_repair.md)。这是显式可选实验，非全项目默认设置。
**无选核版本**：[随机概率C直接梯度训练的一维/二维Ising](../docs/ising_random_gradient_mi.md)，
使用统一的ratio=1逐系综方差下限；记录了2400步结果、失败对照和外推限制。
该配方的[一维经典Ising dim_g=2验证](../docs/ising1d_g2_random_gradient_mi.md)学得近一维表示、一个鞍点和一个吸引子，保留有序端外推及深尺度闭环误差。
[Blume-Capel Delta/J=1温度扫描](../docs/blume_capel.md)提供独立数据生成、采样器精确小系统测试与dim_g=1/2及保色实验；临界定位仍有偏差。
[Ising保色对照](../docs/ising_color_comparison.md)汇总1D dim_g=1/2和2D dim_g=2的weight=1单种子结果，定性结构保留但部分精度退化。
同方案的 [二维 Ising K/B 验证](../docs/ising2d_mi_independent.md) 包含完整环境的精确经验MI、65536核枚举和二维流场。
同方案的 [三值 Potts 测试](../docs/potts_mi_independent.md) 记录了2400步结果、初始化搜索及有限样本MI的限制；本次未恢复正确临界位置。
后续 [随机概率C、ratio=1的Potts实验](../docs/potts_random_gradient_mi.md) 不做选核或预训练，
最终临界插值K≈.9938，偏差约1.12%；保留外推吸引子等限制。

主训练与最新 loss 以根目录 [README](../README.md) 为准：
当前精选配方为随机概率 C、经验 RSMI、逐系综 ratio=1 方差下限，以及 Q/C/R 独立梯度同步更新；Q 使用 relative_local 分布几何，R 使用一步与全尺度多步闭环。坐标平方误差求和，微观子集一致性有效权重为20（配置anchor=2，内部再乘10）。保色是否启用按实验明确标注。
旧固定分离目标不再提供主 CLI 选项，历史协方差满秩、磁化特征和 Q 结构消融已移入
[archive/qcr_ablations](../archive/qcr_ablations/README.md)。下列旧实验说明是历史记录，
不是当前默认配置；重现旧向量 mean-loss 需显式传入 coordinate_reduction='mean'。
旧结果与检查点全部保留，根 README 展示1D经典Ising dim_g=1/2、2D经典Ising dim_g=2及Potts；旧TFIM等记录移至[历史README](../README.history.md)。

## Two-dimensional quantum TFIM (scalar QCR)

`joint_qcr_tfim_2d.py` samples the Pauli-spin Hamiltonian
`H=-sum_<ij> Z_i Z_j - h sum_i X_i` on a periodic 16x16 spatial lattice.
This is not the classical two-parameter K/B experiment. Its periodic 2+1D
Trotter lattice has spatial coupling dtau and imaginary-time coupling
`-log(tanh(dtau*h))/2`. Fixed-count Swendsen-Wang sweeps produce one random
equal-time slice per saved observation. The default beta=16 and dtau=.025
give 640 imaginary-time slices; Q and C see only the spatial snapshots.

```bash
python experiments/joint_qcr_tfim_2d.py \
  --output-dir results/joint_qcr_tfim2d_g1_L16_M640_seed91
python scripts/report_scalar_rg_exponents.py \
  --run results/joint_qcr_tfim2d_g1_L16_M640_seed91
```

The pipeline first checks hot/cold starts, halved dtau at fixed beta, and
doubled beta at fixed dtau at h/J=2,3.04438,4 (256 observations each).
`--pilot-only` stops after these checks. The sampler has exact small-lattice
Trotter enumeration and quantum exact-diagonalization tests.

Training uses 64 uniformly spaced h/J in [.5,5], three chains each, 128
snapshots per chain, 256 discarded sweeps and eight sweeps between saves.
Every third coupling (index modulo three equal to one) is test-only, keeping
all replicas together: 129 train and 63 test ensembles. Q/C/R, hidden60,
dim(g)=1, train jointly for 2000 updates with the data-geometry objective and
equal-horizon rollout weight5 on spatial scales16,8,4. Spin flips augment data;
no hard spin-flip symmetry, physical coupling labels, action, or RSMI are used.
Scalar reports draw R(g), g(h/J), loss and closure curves. The zero-temperature
reference h/J=3.04438 is used only in reporting, not training. Pilot agreement
does not establish continuum, thermodynamic, or ground-state convergence;
any exponent inferred from R is conditional on a faithful thermal RG coordinate.

## Uniform two-parameter data grid

The current data-generation default uses strictly uniform K = 0.20:0.015:0.68
(33 values, no critical-point substitution) and B = -0.50:0.05:0.50 (21 values).
Use `--b-step 0.1` for an 11-point B grid. Old nonuniform caches are retained;
experiment entry points take an explicit cache path and do not silently switch.

```bash
python scripts/prepare_ising2d_field_qcr.py --b-step 0.05 --workers 4 \
  --output data_cache/ising2d_field_sw_v1_uniform_v2_L32_K33_B21_rep3_n256_eq512_gap16_seed91.pt
```

Sampling per cell is unchanged: three chains, 512 warmup sweeps, 256 saved
32x32 configurations per chain, 16 sweeps between saves. The 693 physical
cells split into 462 train / 231 test cells (all replicas kept together,
symmetric +/-B split), giving 1386 / 693 ensembles and 532224 configurations.
Independent-chain seeds and output order are independent of worker count.

## Q-only supervised diagnostic

### Fixed-C QCR exact-cache execution

`joint_qcr_fixed_cache.py` runs the original field QCR experiment with fixed
positive decimation, reusing the uniform-grid per-configuration counts from
`results/supervised_Q_structure_uniform_KB_seed91/`. No supervised weights or
coupling labels are used. Histograms for each randomly selected half-ensemble
are exact, and the original Q, R, objective algebra, random minibatch sequence,
Adam settings, and evaluation subsets are preserved. It verifies direct-vs-cache
loss, all diagnostics, and Q/R gradients before training. Learned C is rejected.

```bash
python experiments/joint_qcr_fixed_cache.py \
  --cache data_cache/ising2d_field_sw_v1_uniform_v2_L32_K33_B21_rep3_n256_eq512_gap16_seed91.pt \
  --output-dir results/joint_qcr_ising2d_uniform_field_g2_h60_L32_fixedC_2000_seed91_cached \
  --no-learnable-kernel --epochs 2000 --seed 91
```

The cache executor currently targets this specific uniform-grid cache and
checks its source manifest. The original direct-computation entry point remains
available for other data. On the paired seed, all logged values through step
400 agreed to within 5.33e-15, after which the redundant slow run was stopped.

`archive/qcr_ablations/supervised_q_structure_ablation.py` compares the existing shared Q with two
equal-parameter (2390 each) variants on the uniform grid:

- `normalized`: non-affine feature BatchNorm after ensemble pooling, eps=1e-5.
  Training uses minibatch statistics; evaluation uses statistics recalibrated
  from the full training set only. No test statistics enter normalization.
- `hierarchical`: move the existing final 12-to-12 Tanh layer from local patches
  to per-configuration pooled features, then average over configurations.
  This preserves the parameter count and initialization while changing the
  order of nonlinear mapping and aggregation.

Both use the same original-label supervision, seed, Adam settings, minibatch
indices and 10000-step endpoint as the baseline. The baseline is reused from
the completed run, not retrained or selected based on test performance.

```bash
python archive/qcr_ablations/supervised_q_structure_ablation.py \
  --cache data_cache/ising2d_field_sw_v1_uniform_v2_L32_K33_B21_rep3_n256_eq512_gap16_seed91.pt \
  --baseline results/supervised_Q_ising2d_uniform_KB_multiscale_h60_seed91 \
  --output-dir results/supervised_Q_structure_uniform_KB_seed91
```

Per-configuration patch counts are cached exactly (about 2.1 GiB), keeping all
256 configurations per ensemble. Tests compare hierarchical cached evaluation
with the direct local-network computation, including parameter gradients and
sample-permutation invariance. This is an experimental ablation only; the
production QCR architecture is unchanged.

`supervised_q_ising2d_multiscale.py` is a Q-only supervised control for the
two-parameter 2D Ising experiment. It uses the unchanged 2390-parameter Q,
fixed positive decimation (32 → 16 → 8 → 4), and the existing physical-cell
train/test split. One shared Q is compared with four independent, equally sized
per-scale Q controls. All predict **original microscopic (K, B)**, not effective
coarse-scale couplings. Labels are standardized using training data only.
There is no R, geometry loss, action loss, or symmetry augmentation.

```bash
python experiments/supervised_q_ising2d_multiscale.py \
  --cache data_cache/ising2d_field_sw_v1_L32_K33_B13_rep3_n256_eq512_gap16_seed91.pt \
  --output-dir results/supervised_Q_ising2d_KB_multiscale_h60_seed91
```

Exact cached binary-patch frequencies accelerate the existing local-MLP-plus-mean
Q, with output and gradient equivalence checks. Default training is 10000 Adam
steps, with a saved 2000-step checkpoint, full train/test metrics by scale,
loss curves, and predicted-vs-true coupling plots. This parameter-recovery
diagnostic does not test whether an autonomous RG flow can be learned.

`rsmi_ising_1d.py` is a self-contained exact-enumeration check of a learned
one-bit block-spin kernel. It is intentionally separate from the main recurrent
RG training pipeline:

```bash
python experiments/rsmi_ising_1d.py
```

The default computes the small-chain joint distribution by exact enumeration.
To train the same RSMI kernel from configurations alone, use the empirical
joint distribution:

```bash
python experiments/rsmi_ising_1d.py \
  --joint-source empirical \
  --samples 100000
```

Historical intermediate Ising prototypes live under `archive/legacy_ising/`.

`ensemble_theory_encoder_ising_1d.py` isolates the existing ensemble encoder
`g = Q(D)` from the full model. It learns a scalar theory coordinate using only
split-ensemble consistency, noncollapse, and distances between empirical local
pattern distributions. Couplings and observables are used only after training:

```bash
python experiments/ensemble_theory_encoder_ising_1d.py
```
`supervised_theory_encoder_ising_1d.py` is the label-aware control experiment
for the same encoder: it trains `Q(D)` directly against the known coupling J
to test representational capacity and sample complexity. It modifies only the
existing `model.theory_encoder` submodule:

```bash
python experiments/supervised_theory_encoder_ising_1d.py
```


`joint_qcr_ising.py` is the current unified 1D/2D Ising entry point. It jointly
updates Q, the hard binary coarse-grainer C, and the shared theory-flow cell R,
with no coupling labels, RSMI, or action loss:

```bash
python experiments/joint_qcr_ising.py --dimension 1
python experiments/joint_qcr_ising.py --dimension 2
```

The same entry point reports data-supported scalar fixed points when
`--theory-dim 1` is used. `scripts/plot_R_vs_g.py` provides the retained generic
plotting utility.

`joint_qcr_tfim_1d.py` applies the same QCR objective to a transverse-field
Ising chain. Its sigma-z ensembles use a finite-temperature Suzuki--Trotter
mapping and a corrected Wolff bond probability; the filename-tagged v2 cache
is incompatible with data produced by the earlier sampler:

```bash
python experiments/joint_qcr_tfim_1d.py
```

TFIM dim-g=2 all-scale control (original data-geometry loss, no balanced metric
or covariance-utilization term; same settings as the dim-g=1 six-scale run):

```bash
python experiments/joint_qcr_tfim_1d.py --sites 128 --levels 6 --theory-dim 2 --hidden 60 --replicas 3 --ensemble-size 128 --epochs 2000 --system-batch 32 --sample-batch 128 --eval-samples 48 --eval-systems 24 --learning-rate .001 --seed 91 --threads 4 --objective-mode data_geometry --geometry-scale 4 --noise-multiplier 2 --rollout-weight 5 --binary-lookup --time-slices 32 --beta-over-slices .5 --equilibration-sweeps 2048 --decorrelation-sweeps 16 --output-dir results/joint_qcr_tfim_1d_g2_levels6_rollout5_2000_seed91
python scripts/plot_qcr_vector_flow.py --run results/joint_qcr_tfim_1d_g2_levels6_rollout5_2000_seed91 --cache data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt --baseline results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91
python scripts/report_tfim_g2_allscales.py --run results/joint_qcr_tfim_1d_g2_levels6_rollout5_2000_seed91
```

Q/C/R have 9893 optimized parameters (versus 9711 at dim-g=1 with the same
hidden width). C remains learned; data include global-spin-flip augmentation,
without enforcing hard network equivariance. Preserved balanced-loss Ising
experiments are separate controls, not the default training objective.

The older full Q--C--R--A pipeline remains available for comparisons. Couplings
are kept out of its training objective and used only for post-training evaluation:

```bash
python experiments/unsupervised_recurrent_rg_ising_1d.py
```

The corresponding fully label-free 2D Ising pipeline uses periodic Wolff
sampling and the dimension-independent recurrent-RG core with square-lattice,
binary-field, and binary-local-conditional adapters. Its RSMI loss is a neural
InfoNCE bound rather than a finite binary environment histogram:

```bash
python experiments/unsupervised_recurrent_rg_ising_2d.py
```
