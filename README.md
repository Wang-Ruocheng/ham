# Recurrent RG Network：从构型系综学习 QCR

更新：2026-10-03。输入为无耦合标签的构型系综，学习局域粗粒化 C、系综坐标 Q 和共享尺度映射 R：

$$
\mathcal D_k=C^k\mathcal D_0,\qquad g_k=Q(\mathcal D_k),\qquad R(g)=g+F(g).
$$

当前精选实验使用**随机初始化的可训练概率 C、经验 RSMI 和逐系综方差下限，Q/C/R 同步更新但独立传递梯度**。不穷举选择核，不预训练 C，不使用耦合或相标签监督。真实耦合用于采样和事后物理诊断；采样区间的选择仍包含实验设计知识。

这是有限尺度统计与潜空间动力学的研究原型，不能自动视为精确、闭合的物理 RG。

- [当前模型与训练](#模型与训练)
- [Loss 组成和目的](#loss-组成和目的)
- [精选实验及复现](#精选实验)
- [开发指南](docs/development.md)、[完整实验索引](experiments/README.md)
- [旧版 README 与历史 TFIM / Ising 记录](README.history.md)

## 模型与训练

| 模块 | 输入 → 输出 | 结构及职责 |
|---|---|---|
| C | 局域自旋块 → 一个粗粒化自旋 | 不重叠一维 2 格 / 二维 2×2 块；局域 MLP 输出 Bernoulli 或 categorical 概率 |
| Q | 同一系综的多份构型 → dim_g 维坐标 | 周期局域特征网络，跨样本与位置平均，再接输出网络；无显式总磁化输入 |
| R | g → 下一尺度 g | 残差 MLP，所有尺度共享参数，不显式输入 k |

C 在构型路径上随机采样，使用直通梯度估计；MI 与均匀块保色项直接使用输出概率。没有硬编码 Z2 或颜色置换等变性。当前不学习作用量 A。

下列精选实验均为 seed=91、hidden=60、Adam 学习率 0.001、每批 32 个系综 × 64 个构型、2400 次更新。这里的“步/epoch”指训练循环更新次数，不表示完整遍历缓存一次。

当前配方入口是 `python -m experiments.rsmi_separate_qcr`。`--independent-c` 模式下：

- Q 只优化表示目标；其粗粒化输入不向 C 反传 Q loss。
- C 只优化 MI、防塌缩及可选保色目标。
- R 只优化一步和多步闭环；目标和初始 Q 坐标均停止梯度，但 R 的递推保留梯度。

因此它既不是 Q→C→R 逐个训练，也不是所有 loss 对三个模块全反传。下面描述的是**经验证的实验配方**，不代表所有历史脚本或通用 CLI 的默认值；例如保色必须显式传入 `--color-weight 1`。

## Loss 组成和目的

令两个独立子集的编码为 \(g^a_{ik},g^b_{ik}\)，\(\bar g_{ik}=(g^a_{ik}+g^b_{ik})/2\)。坐标误差使用平方范数，即对潜维度**求和**，然后对系综与尺度平均。

当前三个独立优化目标为：

$$
\begin{aligned}
\mathcal L_Q&=20\mathcal L_{\rm split,0}+5\mathcal L_{\rm split,coarse}+10\mathcal L_{\rm geometry},\\
\mathcal L_C&=-2\widehat I(H;E)+2\mathcal L_{\rm variance}+\lambda_{\rm color}\mathcal L_{\rm color},\\
\mathcal L_R&=5\mathcal L_{\rm step}+5\mathcal L_{\rm rollout}.
\end{aligned}
$$

**权重口径：**配置中的 `anchor=2` 乘的是代码内部已放大 10 倍的微观一致性项，所以相对原始平方误差的有效系数为 **20**，不是 2。日志中分项不应再重复乘这个内部因子。三个 Ising 保色实验取 λ_color=1；下面的三态 Potts 成功实验取 0。

| 项 | 目的 | 更新模块 |
|---|---|---|
| 子集一致性 | 同一系综的不同抽样给出稳定坐标，避免编码抽样噪声 | Q |
| 分布几何 | 可区分的构型分布对应可区分的 g，不把系综编号当分离标签 | Q |
| 经验 RSMI | 粗变量保留隔着 buffer 的环境信息 | C |
| 逐系综方差下限 | 防止 C 把仍有变化的输入压为偏置或常数输出 | C |
| 可选均匀块保色 | 固定颜色约定，抑制全同块的翻转 / 循环重标号 | C |
| 一步闭环 | 拟合每个已观测尺度的 Q(CD) | R |
| 全尺度 rollout | 从微观编码递推，匹配所有可用的 Q(C^kD) | R |

### 1. 同系综子集一致性

$$
\mathcal L_{\rm split,0}=\mathbb E_i\|g^a_{i0}-g^b_{i0}\|^2,\qquad
\mathcal L_{\rm split,coarse}=\mathbb E_{i,k\ge1}\|g^a_{ik}-g^b_{ik}\|^2.
$$

它提高 Q 的抽样稳定性，但单独使用允许所有系综塌缩到同一点，必须配合分布几何。

### 2. 分布几何：`relative_local`

在每个尺度统计局域模式频率 p（一维自旋对 / 二维 plaquette）。两个子集给出：

$$
\bar p_i=(p_i^a+p_i^b)/2,\quad \eta_i=\tfrac14\|p_i^a-p_i^b\|^2,\quad
d_{ij}=\|\bar p_i-\bar p_j\|,\quad a_{ij}=2\sqrt{\eta_i+\eta_j}.
$$

要求潜距离 \(r_{ij}=\|\bar g_i-\bar g_j\|\) 尽量落入统计容许区间：

$$
\ell_{ij}=[4(d_{ij}-a_{ij})_+-r_{ij}]_+^2+
[r_{ij}-4(d_{ij}+a_{ij})]_+^2.
$$

`relative_local` 用 \(s_{ij}=(4d_{ij})^2+(4a_{ij})^2+0.05^2\) 归一化：乘 \(\min(100,s_{ij}^{-1})\)，再按可靠性 \((4d_{ij})^2/s_{ij}\) 加权。全局配对平均与每个系综 4 个可分辨近邻的局部平均各占一半，无有效局部近邻时回退全局项；各尺度等权。

统计目标停止梯度。这里的噪声容许区间不是严格置信区间，也不是完整分布距离。该项允许物理上一维的流形嵌入二维，不强迫协方差各向同性或满秩。当前配方没有额外协方差白化、弱方向加权 R loss 或局部轨迹排斥项。

### 3. RSMI：有限经验分布上的直接求和

$$
\hat p(h,e)=\sum_v\hat p(v,e)P_C(h\mid v),\qquad
\widehat I(H;E)=\sum_{h,e}\hat p(h,e)\log\frac{\hat p(h,e)}{\hat p(h)\hat p(e)}.
$$

当前 `--estimator histogram` 不训练 critic；在微观尺度逐系综计数，使用全部缓存样本及平移位置。一维 V 为 2 格、E 为两侧各一个自旋；二维 V 为 2×2、E 为完整 20 自旋外壳。V 与 E 之间留出宽度为 1 的 buffer，buffer 不进入 E。

梯度经过概率核 P_C；这不是枚举候选核后选最优核。优化使用自然对数，报告的 MI 通常换算为 bit。所谓“精确 MI”仅指经验表上的有限求和，**不代表真实分布 MI 无偏**；大环境字母表尤其容易产生有限样本上偏。旧日志中的零值 shuffled MI 占位不能当作实测零偏差基线。

### 4. C 的逐系综方差下限

令 m 为一个系综内跨构型、跨位置的平均二值自旋，sg 表示停止梯度：

$$
\mathcal L_{\rm variance}=\mathbb E_{i,k}
\left[|m_{\rm out}|-\sqrt{1-\rho(1-\operatorname{sg}(m_{\rm in})^2)}\right]_+^2.
$$

多类别用颜色频率向量 p 替代：

$$
\mathbb E_{i,k}\left[\|p_{\rm out}\|-
\sqrt{1-\rho(1-\|\operatorname{sg}(p_{\rm in})\|^2)}\right]_+^2.
$$

当前取 ρ=1，逐系综、逐尺度计算，权重 2。它惩罚方差缩减，但允许常数输入保持常数，不要求每个系综各颜色均匀。这是一项软统计假设，不是对任意物理系统都严格成立的 RG 恒等式。

### 5. 可选保色

$$
\mathcal L_{\rm color}=-\frac1q\sum_{a=1}^q\log P_C(a\mid a,a,\ldots,a).
$$

只约束均匀块，混合块仍可学习；二值情形就是全正 / 全负块的 Bernoulli NLL。它不等于硬对称性，也不保证全局不存在周期轨道。三态 Potts 的精选结果未使用此项，应与保色 Ising 对照区分。

### 6. R 的一步和全尺度闭环

$$
\mathcal L_{\rm step}=\mathbb E_{i,0\le k<n}\|R(\operatorname{sg}\bar g_{ik})-\operatorname{sg}\bar g_{i,k+1}\|^2,
$$
$$
\mathcal L_{\rm rollout}=\mathbb E_{i,1\le k\le n}\|R^k(\operatorname{sg}\bar g_{i0})-\operatorname{sg}\bar g_{ik}\|^2.
$$

覆盖缓存链的所有可用尺度，不只前两步；但不约束链长之外的无限次递推。R loss 不再通过收缩 Q/C 表示来降低自身误差。日志总量含负 MI，可能为负，不能只按其大小判断相变学习是否正确。

## 精选实验

下面均为预定最终 2400 步、单种子结果，不按临界位置挑选早停检查点。保留“效果较好”的定性结构，也明确支持范围及误差；尚不能据此宣称多 seed 鲁棒性、完整相图或临界指数都已恢复。

| 问题 | dim_g | 参数量 | 已恢复的主要结构 | 主要限制 |
|---|---:|---:|---|---|
| 1D 经典 Ising | 1 | 9,711 | 有序端不稳定、无序端稳定，流向正确 | 深尺度闭环误差大 |
| 1D 经典 Ising | 2 | 9,893 | 近一维流形，有序端鞍点、无序端吸引子 | 有序端外推，位置精度退化 |
| 2D 经典 Ising，K/B | 2 | 10,373 | 中心两方向均不稳定的排斥子 | 临界 K 插值偏低约 3.20% |
| 2D 三态 Potts，零场 | 1 | 11,873 | 中间不稳定、两侧稳定的标量结构 | 一个吸引子在训练支持外；未保色 |

复现命令依赖已有 source 目录中的配置和数据缓存；`--source` 在以下命令里**仅读配置，不恢复旧模型权重**。输出用新 `_repro` 目录以免覆盖结果。完整数据生成和报告命令见各实验记录；本地缓存、检查点和图片不保证随 Git 分发。

---

### 1D 经典 Ising：dim_g=1

零场周期链 L=128→64→32→16→8→4，64 个无量纲 J=J_phys/T∈[0.01,2]，每点 3 个系综、每系综 128 构型；129 个训练系综 / 63 个留出系综。注意：**不是 TFIM，也没有有限温度临界点**。

C 学到近似正 decimation。R 的有序端不稳定根 g≈0.01475，乘子 1.54102；无序端稳定根 g≈1.95142，乘子 0.08485。300 个诊断 J 的流向均正确，相对精确 decimation 参考的一步 RMSE / 端点跨度约 1.07%。测试首步 NRMSE=3.78%，第 5 步 rollout NRMSE=58.10%，不能称为所有尺度精确闭环。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr \
  --output results/ising1d_g1_color1_repro/qr \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

![1D Ising 标量流与物理端点](results/ising1d_g1_random_independentC_variance2ratio1_color1_2400_seed91/qr/endpoint_report/physical_validation.png)

[详细记录](docs/ising_color_comparison.md) · [各尺度 J–g](results/ising1d_g1_random_independentC_variance2ratio1_color1_2400_seed91/qr/full_report/coupling_by_scale.png) · [loss 曲线](results/ising1d_g1_random_independentC_variance2ratio1_color1_2400_seed91/qr/report/loss_comparison.png)

---

### 1D 经典 Ising：dim_g=2

使用同一份数据。Q 表示近似一维，population 编码弱主成分占比约 2.64×10⁻⁶。鞍点 g≈(0.08504,−0.20728)，DR 特征值约 (1.22713,0.33305)；吸引子 g≈(1.80140,−1.46916)，特征值约 (0.32129,−0.05039)。鞍点的不稳定方向沿物理流形，横向稳定。

300 个 J 的位移方向均符合精确参考。但有序端在训练支持外，归一化端点位置误差约 11.84%；测试首步 / 第 5 步 rollout NRMSE 为 7.65% / 80.15%。无保色版本有序端误差较小（5.40%），见[原始对照](docs/ising1d_g2_random_gradient_mi.md)。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr \
  --output results/ising1d_g2_color1_repro/qr \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

![1D Ising 二维潜空间流场](results/ising1d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/full_report/R_one_vs_two_step.png)
![1D Ising J 与二维 g 的映射](results/ising1d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/full_report/coupling_by_scale.png)

[详细记录](docs/ising_color_comparison.md)

---

### 2D 经典 Ising：dim_g=2，K/B 双参数

\(P(X)\propto\exp(K\sum_{\langle ij\rangle}s_is_j+B\sum_i s_i)\)，K=J/T、B=h/T，非二维 TFIM。格点 16×16→8×8→4×4；K=0.20:0.03:0.68，B=−0.01:0.001:0.01，每个系综 64 构型。

C 学得近似对块内三个自旋做正多数票，非预设此规则。中心排斥子 g≈(−0.17197,0.05356)，DR 特征值 (1.56649,1.12415)，两者均大于 1，**不是鞍点**。微观网格内反插值 K≈0.426565、B≈−0.000493；相比理论零场 Kc≈0.440687，K 偏低约 3.20%。

测试两尺度一步 NRMSE=3.74%、5.21%，两步 rollout=6.62%。另外两个鞍点没有微观网格内反插值，不直接赋予确定的 K/B。有限范围 R² 查根未发现额外二周期，不构成全局证明。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91/qr \
  --output results/ising2d_g2_color1_repro/qr \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

![2D Ising 流场及中心放大](results/ising2d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/report/R_flow_and_center.png)
![2D Ising K/B 与 g 的映射](results/ising2d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/report/g_KB_maps.png)

[详细记录](docs/ising_color_comparison.md) · [无保色随机核对照](docs/ising_random_gradient_mi.md)

---

### 2D 三态 Potts：dim_g=1

零场 \(P(X)\propto\exp(K\sum_{\langle ij\rangle}\delta_{s_i,s_j})\)。L=16→8→4，64 个 K∈[0.4,1.6]，每点 3 个系综、每系综 128 构型；129 个训练 / 63 个留出系综。

最终不稳定点 g≈1.583052，R′≈1.393651，唯一微观反插值 K≈0.993806；理论 Kc=log(1+√3)≈1.005053，偏低约 **1.12%**。两侧稳定根 g≈0.782294 和 3.149505，其中后者位于 R 训练输入范围之外，不能声称完整验证了两个物理吸引子。

一步 NRMSE=4.37%、5.18%，两步 rollout=7.91%。此实验**保色权重为 0**；均匀块学到 0→1→2→0 的颜色循环，Q 在已测试微观构型上对全局重标号近似不变，不等于严格对称或任意尺度无周期。当前结果使用最终 2400 步，不是选取早期较好的检查点。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast \
  --output results/potts2d_g1_random_ratio1_repro \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 0
```

source 名称含 MIinit，但本命令没有 `--mi-initialize` 或恢复权重，仍从随机概率 C 开始。

![三态 Potts 的 R(g) 与 K–g 映射](results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91/full_report/R_and_K_mapping.png)

[详细记录与限制](docs/potts_random_gradient_mi.md) · [loss 曲线](results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91/report/loss_comparison.png)

---

### 补充：八态 Potts

八态零场 Potts 的单种子 dim_g=1、保色权重 1 实验也已完成。理论一级相变 Kc=log(1+√8)≈1.342454；不稳定点反插值 K≈1.327668，偏低约 1.10%。这只支持转变位置识别，尚不能从该局部流场断言已区分一级与连续相变或验证标度律。

```bash
python -m experiments.rsmi_separate_qcr \
  --config configs/potts2d_q8_g1_random_color1_seed91.json \
  --output results/potts2d_q8_color1_repro \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
```

[数据、命令与有限样本限制](docs/potts8.md) · [相图与 K–g](results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91/full_report/R_and_K_mapping.png)

## 如何解读图像与验收结果

- 流场箭头画 R(g)−g；标量图画 R(g) 并与 y=g 比较。稳定性判断使用 **DR** 特征值的模，不靠箭头的视觉印象；R² 图使用 R²(g)−g。
- 微观耦合–g 反插值仅在采样支持内且映射可辨识时成立，不能给任意粗尺度 g 都赋予唯一微观耦合。
- 区分定性不动点结构、临界位置误差、逐尺度闭环误差；三者不能互相替代。临界指数还要求正确的尺度因子、临界固定点及相关方向，以上结果并非临界指数验收。
- 当前未把 Ashkin–Teller 的三相实验列为成功案例。固定 decimation 也不保证粗粒化后出现三个孤立吸引子。

## 开发与历史记录

公共模型、loss、配置、数据和评估位于 `recurrent_rg/`；数据生成和报告入口在 `scripts/`，专项研究驱动在 `experiments/`。通用 `python -m recurrent_rg` 与上述 MI 专项驱动不是同一个训练目标，复现时不要直接互换入口。

参见 [开发指南](docs/development.md)、[实验索引](experiments/README.md) 及 [历史 README](README.history.md)。本次文档整理不修改模型、loss 实现或全项目 CLI 默认值。
