# Recurrent RG Network：从构型系综学习 QCR

更新：2026-11-07。QCR 从多组构型系综学习局域粗粒化、统计坐标与尺度演化：

$$
\mathcal D_k=C^k\mathcal D_0,\qquad g_k=Q(\mathcal D_k),\qquad R(g)=g+F(g).
$$

**本页展示最新完成的弱 KL、VAE 式 Q 实验**：双 Ising 的独立双参数扫描、二维 Ising 的 K–B 扫描、连续标量场 $\phi^4$ 的 r–u 带状扫描，以及 Ising 单参数的无监督／监督配对对照。均包含联合 QCR 和末尾固定 Q/C 的 R-only 拟合；结果以正式报告和实际更新数为准。

无监督训练不输入哈密顿量系数、耦合标签或相标签。输入仍须保留空间几何和**系综分组**：同一组来自同一制备条件，多组覆盖不同统计状态；并非把未知条件下的所有构型混成一个分布。参数用于生成数据和事后评价，采样区域的选择也包含实验设计知识。显式监督对照单独标明。

> **最新实验配方与通用默认值有区别。**这些实验显式指定 `ensemble_vae` 或 `continuous_ensemble_vae`。代码通用入口仍默认使用共享仿射重构 Q；本次文档更新不改变默认模型。复现本页结果请使用对应的 VAE 配置，不能仅给旧配置换一个输出目录。

- [当前模型与训练](#当前模型与训练)
- [最新完成实验](#最新完成实验)
- [使用与复现](#使用与复现)
- [结果解释与适用边界](#结果解释与适用边界)
- [开发指南](docs/development.md)、[实验入口索引](experiments/README.md)
- [上一版 README 完整归档](README.history.2026-11-05.md)、[更早的历史记录](README.history.md)

## 当前模型与训练

### Q、C、R 分别学习什么

| 模块 | 输入 → 输出 | 当前实验中的职责 |
|---|---|---|
| C | 局域构型块 → 随机粗变量 | 从随机初始化开始学习概率核；不预训练或枚举选择核 |
| Q | 同一系综的局域图案分布 → $d_g$ 维坐标 | 共享的弱 KL 高斯编码器与非线性概率解码器 |
| R | $g_k$ → $g_{k+1}$ | 残差 MLP，跨尺度共享，不显式输入尺度编号 |

本页已完成实验均为二维周期格点，采用不重叠 **3×3、stride 3** 的 C，尺度链为 **36→12→4**，`schema_version=5`。C 的 3×3 块与 Q 重构的 2×2 图案是不同概念。一维标准入口仍采用两格、stride 2；历史显式 b=2 的结果保留原设置。

Q/C/R 在联合阶段同步训练，**梯度独立路由**：Q 只接受表示目标，C 只接受自身的信息与防塌缩目标，R 只接受闭环目标。Q loss 不更新 C，R loss 不更新 Q/C；递推中的每一次 R 调用保留梯度。

### 弱 KL 的 VAE 式 Q

实现见 [离散系综 VAE](recurrent_rg/ensemble_vae.py) 和 [连续场适配器](recurrent_rg/phi4_ensemble_vae.py)。这不是此前的逐构型 CNN，也不是对整幅构型像素进行生成式重构。

1. 将一条物理链的构型子集分成两个不重叠的半系综，分别统计有序 2×2 局域图案概率 $p^a,p^b$。
2. 用共享编码器得到 $q_\theta(z\mid p)=\mathcal N(\mu_\theta(p),\operatorname{diag}\sigma_\theta^2(p))$。
3. 从一半系综的后验采样 z，解码并预测另一半的图案分布；两个方向对称平均。
4. 对微观与全部粗尺度等权平均，使用同一个编码器、解码器及固定归一化。

$$
\mathcal L_Q^{\rm VAE}
=\mathbb E_{i,k,(a,b)}\left[
-\sum_m p^b_{ik,m}\,\mathbb E_{z\sim q_\theta(z\mid p^a_{ik})}
\log d_\theta(z)_m
+\beta(n)D_{\rm KL}\big(q_\theta(z\mid p^a_{ik})\,\|\,\mathcal N(0,I)\big)
\right],\qquad
\beta(n)=0.01\min(n/600,1).
$$

这里 $(a,b)$ 包括两个交叉预测方向。解码器输出归一化概率；采用稳定的 softmax/log-softmax，并检查概率和与数值有限性。交叉熵包含目标分布的熵，不要求收敛到零；C 改变目标统计时，不宜仅凭原始交叉熵的上升判断过拟合。

| 设置 | 离散 Ising / 双 Ising | 连续 $\phi^4$ |
|---|---|---|
| 输入概率维数 | 二值 Ising：$2^4=16$；双 Ising 四态联合变量：$4^4=256$ | 每格点 5 个软分箱，2×2 联合特征共 $5^4=625$ |
| 编码器 | `bins→96→96`，再接 $\mu$、$\log\sigma^2$ 两个 $d_g$ 维头 | 相同 |
| 解码器 | `d_g→64→64→bins`，非线性概率重构 | 相同 |
| 激活与方差 | LeakyReLU；$\log\sigma^2$ 限于 [−8,3] | 相同 |
| 校准 | 仅微观 TRAIN 图案统计，标准差下限 0.001 | 另固定 TRAIN 场 RMS，定义软分箱尺度 |
| R 的输入 | 两个半系综的确定性后验均值之平均 | 相同 |

具体为 $\bar g_k=\tfrac12[\mu(p_k^a)+\mu(p_k^b)]$。VAE 后验采样只用于 Q 的训练，R 不接收随机 z。校准后统计量跨尺度冻结，不使用 TEST、参数标签或 Q-only 预训练。

本页离散 VAE 配置显式设置 `q_encoder="ensemble_vae"`、`affine_reconstruction=false`，并令 `anchor=0`、`coarse_consistency=0`、`separation=1`：**交叉半系综 CE＋弱 KL 替代旧 Q 目标，不叠加仿射重构 MSE 或旧的 20/5 子集一致性项**。没有 FactorVAE 的 total-correlation 判别器。弱 KL 限制表示的尺度与复杂度，但不保证独立物理参数可辨识、坐标逐轴解耦或 RG 闭合。

未显式启用 VAE 的通用默认仍为：局域特征网络 Q＋共享仿射图案重构（权重 10）＋微观／粗尺度子集一致性（有效权重 20/5）；decoder 列归一化、不强迫满秩、不默认 PCA 初始化。历史 checkpoint 保留其原架构。

### C 的目标

离散模型采用微观尺度、buffer=1 的完整环境壳层经验直方图 MI：

$$
\mathcal L_C=-2\widehat I(H;E)+2\mathcal L_{\rm variance}+\mathcal L_{\rm color}.
$$

逐系综方差下限比例为 1；均匀块保色项为 NLL。C 随机初始化、可训练，二值输出 Bernoulli、多类别输出 categorical 概率；不使用 InfoNCE 替代。经验 MI 是有限计数表上的求和，不等于真实分布 MI 的无偏估计。四态 3×3 块有 $4^9=262144$ 种局部输入，计算成本高于二态。

连续 $\phi^4$ 使用相应的随机 Gaussian C、平滑信息代理、逐系综方差保持与均匀场幅度保持，权重仍为 −2、2、1。**连续信息代理不是离散直方图 MI，幅度保持也不是离散保色 NLL**；跨模型比较不能宣称三个 loss 的数值完全同口径。

### R 的全起点闭环与末尾拟合

新实验使用 `loss.closure_objective="all_start_anchor"`：

$$
\mathcal L_R=\frac{10}{S}\sum_{t=1}^{S}\frac1t\sum_{s=0}^{t-1}
\mathbb E_i\left\|R^{t-s}(\operatorname{sg}\bar g_{is})-\operatorname{sg}\bar g_{it}\right\|_2^2.
$$

各目标尺度等权，每个终点内各早期起点等权；潜维度平方误差求和。兼容外层权重为 `commutator=5`、`rollout_weight=5`，总权重 10；`closure_scale_weights=None`。一步／微观起点多步曲线可作为诊断记录，但不再额外叠加。没有预测之间的一致性项或 future-observable loss。

这是 Koopman 启发的多预测跨度一致性，不假定非线性 R 是有限维线性 Koopman 算子。

联合训练结束后，标准新管线默认 `training.terminal_r_fit=true`：

- 冻结最终 Q/C；每条 TRAIN 物理链固定 8 组构型子集/C 抽样，按同一半系综均值约定建立缓存。
- 继续同一个 R、Adam moments 和当前学习率，目标不变；不再优化 Q/C 或 MI。
- 默认最多追加 10,000 次 R 更新。每 500 步检查 TRAIN：至少 3000 步后，过去 2000 步的最佳损失改善不足 1%、当前损失在最佳值的 5% 内，连续三次满足才记为操作性平台。
- `update_cap` 表示达到步数上限，**不称收敛或平台**。TEST、标签、根和指数不用于选择终点。

缓存重复不增加独立物理链数。报告分开记录 Q/C 联合更新数、额外 R 更新数与总 R 更新数；联合 checkpoint 和指标保留，末尾结果在 `r_only/`，`pipeline.json` 指向最终模型。标准 `load_run(运行目录)` 读取末尾模型，显式指定历史 checkpoint 则保留该阶段语义。预算应覆盖两阶段，末尾未完成的运行不能当作完整结果。

## 最新完成实验

以下为 2026-11-05 至 11-06 完成的正式结果，均为训练 seed=91、3000 次联合更新；“步/epoch”指优化更新，不是完整遍历数据。三组二维潜空间实验均追加 10,000 次 R 更新后达到上限，Q/C 保持不变。它们尚不是新版配方的多 seed 稳健性验证。

| 实验 | 数据覆盖与独立物理链 | $d_g$ | 有限搜索中的最终数值不动点 | 正式证据 |
|---|---|---:|---|---|
| 双 Ising，独立扫描 K₁/K₂ | 14×14 参数格，K₁,K₂∈[0,1.3]；每格 2 链×64 构型；TRAIN/TEST=262/130 链 | 2 | 1 排斥、3 鞍点、3 吸引 | [报告][dual-report]、[配置][dual-config] |
| 二维 Ising，K–B 扫描 | K∈[0.20,0.68] 的 17 点，B∈[−0.01,0.01] 的 21 点；每格 2 链×64 构型；474/240 链 | 2 | 1 排斥、3 鞍点、2 吸引 | [报告][kb-report]、[配置][kb-config] |
| $\phi^4$，r–u 带状扫描 | 5 个 u×9 个 r 偏移；每格 4 链×128 构型；136/44 链 | 2 | 1 鞍点，带外推性质 | [报告][phi-report]、[配置][phi-config] |
| 二维 Ising，B=0 单独扫描 K | 同数据、同初始化的无监督／微观 K 监督配对 | 1 | 两组各找到 1 排斥、1 吸引 | [配对报告][pair-report]、[配置][pair-config] |

前三组按整个参数格划分 TRAIN/TEST。原始构型来自相关 Markov 链，不称独立同分布样本；Ising K–B 的历史有限链 ESS 警告及 $\phi^4$ 的采样警告均保留。

### 留出系综上的闭环误差

每格为 **RMSE（NRMSE）**，NRMSE 使用相应目标尺度的 TRAIN 编码标准差归一化；0→2 表示 $R^2(g_0)$。同一运行联合／末尾比较保持坐标与归一化一致。

| 最终模型 | 0→1：36→12 | 1→2：12→4 | 0→2：36→4 |
|---|---:|---:|---:|
| 双 Ising | 0.05248（5.32%） | 0.06566（5.79%） | 0.07647（6.75%） |
| Ising K–B | 0.03716（3.71%） | 0.09960（8.40%） | 0.09553（8.06%） |
| $\phi^4$ 带状扫描 | 0.06927（10.06%） | 0.12472（12.68%） | 0.16446（16.72%） |
| Ising B=0，无监督 | 0.04680（4.87%） | 0.08073（6.24%） | 0.08053（6.22%） |
| Ising B=0，监督权重 10 | 0.01698（5.31%） | 0.04866（10.52%） | 0.05011（10.84%） |

末尾拟合使三组二维实验的 TEST 0→2 NRMSE 分别从 **18.49%→6.75%、11.43%→8.06%、19.99%→16.72%**。这是固定 Q/C 后的拟合改善，不证明学到精确物理 RG。不同实验／监督臂的潜空间尺度可能不同，绝对 RMSE 不可直接排名；NRMSE 也只衡量各自坐标中的相对闭合程度，不是共同物理观测量误差。

### 双 Ising：两个参数在多尺度表示中分开

新版使用完整独立 K₁/K₂ 网格，取代旧的单参数 `K1=K2/2` 首页结果。各尺度的描述性轴分离分数约为 **0.982、0.985、0.975**；微观编码的留出三次多项式参数读出 $R^2$ 分别约 **0.954、0.953**。这些标签只用于事后诊断。

有限搜索找到 7 个根，中心排斥点约为 $g^*=(0.216,0.269)$。结果支持“两个独立控制方向在共享多尺度表示中保留，并形成多分支流结构”；当前 3 个数值吸引点并不构成完整四相全局吸引域的验证，根的数量也受搜索区域限制。

![双 Ising：微观编码、参数着色与最终流场][dual-flow]

[各尺度分离诊断][dual-separation] · [耦合量到 g][dual-coupling] · [联合 loss][dual-loss] · [VAE 分项][dual-vae-loss] · [k=1 流场][dual-flow1] · [k=2 流场][dual-flow2]

### Ising K–B：二维流结构与额外根

中心排斥点约为 $g^*=(0.0628,0.0238)$，完整 $DR$ 的两个特征值约 **3.045、1.807**。微观潜空间的留出三次多项式读出 $R^2$ 为 K≈0.998、B≈0.841；磁场方向的可辨识程度弱于 K 方向。

另外还有 3 个鞍点与 2 个吸引点。数据不包含 K=0，因而把某个额外鞍点解释为 K=B=0 仍是外推假设，不能由潜空间位置直接确认。神经映射特征值也尚未完成物理热／磁方向认证。

![二维 Ising K–B：微观编码、参数着色与最终流场][kb-flow]

[耦合量到 g][kb-coupling] · [联合 loss][kb-loss] · [VAE 分项][kb-vae-loss] · [k=1 流场][kb-flow1] · [k=2 流场][kb-flow2] · [完整根与谱][kb-roots]

### 连续 $\phi^4$：带状扫描、鞍点与两相分界对照

采样作用量为

$$
S[\phi]=\frac12\sum_{x,a}(\phi_{x+e_a}-\phi_x)^2
+\sum_x\left(\frac r2\phi_x^2+\frac u4\phi_x^4\right),\qquad h=0.
$$

本轮在 u={0.5,1.125,1.75,2.13,2.375} 上，围绕预先固定的 $r_0(u)$ 取偏移 {−0.45,−0.30,−0.18,−0.08,0,0.08,0.18,0.30,0.45}。中心来自历史 TRAIN 的有限尺寸 Binder 交叉区，**不是精确临界线或训练标签**；新样本独立生成，采用 HMC 与 embedded-Ising 簇更新。完整区域见[冻结采样设计][phi-plan]。

最终找到鞍点 $g^*\approx(-1.0197,1.2893)$，$DR$ 特征值约 **1.7369、0.4362**。微观 PCA 第二成分占比约 **12.97%**；局部响应诊断显示微观有两个可辨方向，但粗尺度第二方向的信噪比减弱，不能仅凭总体 PCA 认定全域二维闭合。

![连续 phi4：微观编码、参数着色与最终流场][phi-flow]

[耦合量到 g][phi-coupling] · [联合 loss][phi-loss] · [末尾 R loss][phi-r-loss] · [PCA 成分占比][phi-pca] · [k=1 流场][phi-flow1] · [k=2 流场][phi-flow2]

进一步对离散映射 R 的鞍点稳定流形进行追踪，并与原始构型的 Binder 两相诊断对照：以 Binder≥0.62 或 ≤0.20 标记清晰相参数格，微观两侧对应为 **31/35**，留出格为 **9/10**；k=2 对应为 **32/35**、留出 **10/10**。这是固定模型后的描述性分析，分界位置相对有限尺寸 Binder 代理仍有偏移。

![phi4：稳定流形与原始参数空间两相分界的对照][phi-boundary]

鞍点位于 k=2 编码的坐标外框内，但仍在已采样编码的凸包外；k=0、1 连坐标外框也未覆盖。**外框包含不等于数据支持。**本轮还保留 62 条采样质量警告，未认证平衡、热力学临界线或 Wilson–Fisher 固定点身份。对应[稳定流形完整分析][phi-stable-report]保留这些限定及链级 bootstrap 口径。

作为同架构的前一组矩形区域对照，[修正概率归一化后的结果][phi-rectangle-report]的 TEST 0→2 NRMSE 为 9.19%，低于本轮 16.72%；由于样本区域与坐标不同，带状采样不能表述成所有指标上的一致改进。

### 单参数 Ising：无监督与耦合监督对照

这组仍是**二维** Ising、B=0、$d_g=1$。两臂从相同初始化出发，联合阶段均为 3000 步，监督臂仅对微观 Q 增加 K 标签约束（权重 10）；C 不接受标签梯度。末尾无监督臂追加 10,000 步至上限，监督臂追加 9500 步满足 TRAIN 操作性平台，总 R 更新分别为 13,000 / 12,500。

| 最终结果 | 无监督 | 微观 K 监督 |
|---|---:|---:|
| 排斥点 $g^*$ | −0.02517 | 0.44241 |
| 由微观 g(K) 插值得到的 K | 0.43256 | 0.43659 |
| 排斥点 $R'(g^*)$ | 2.17257 | 2.50102 |

理论参考 $K_c\approx0.44068679$。监督臂的微观插值更接近参考，但相对闭环误差并未全面变好；不能据此说监督使全部 RG 性能提高。两组找到的高端吸引点均没有微观 K 插值，监督臂该点还位于 TRAIN 输入包络外。正式报告 `thermal_nu=null`：目前的导数与插值是有限模型诊断，尚未认证普适热临界指数。

![单参数 Ising：无监督与监督的最终 R 曲线][pair-flow]

[耦合量到 g][pair-coupling] · [配对 loss 曲线][pair-loss] · [正式报告][pair-report]

### 在途问题

DQCP 的 J–Q₂ 量子自旋快照管线已另行启动。最近一次核对为 **2026-11-07 09:49（北京时间）**，仍处于量子采样／验证阶段，训练尚未开始；不列为已完成结果。见[实验说明](docs/dqcp_jq_snapshot_pilot.md)。空间快照上的学习流与量子哈密顿量 RG 之间的对应仍需独立验证。

## 使用与复现

Python ≥3.10，依赖见 [pyproject.toml](pyproject.toml)：PyTorch、NumPy、SciPy、Numba、Matplotlib。开发环境可安装：

```bash
python -m pip install -e '.[dev]'
```

复现最新离散 VAE 配方时，复制对应配置，设置本机实际数据缓存路径，并使用新的输出目录。例如：

```bash
cp configs/ising2d_KB_weak_kl_vae_g2_seed91_20261106.json configs/ising2d_KB_vae_local.json
# 编辑副本中的 data.cache，指向已验证的对应构型缓存
python -m recurrent_rg train \
  --config configs/ising2d_KB_vae_local.json \
  --output results/ising2d_KB_vae_local_new
```

也可使用 `python -m experiments.rsmi_separate_qcr --config ... --output ...`。配置中已有联合与末尾 R 阶段，不需要在结束后另建一个改变目标的 R-only 实验。资源选择与预算须覆盖完整管线；公共报告下载目录不包含训练 PT，不能直接当作数据缓存。

| 任务 | 配置／入口 |
|---|---|
| 双 Ising VAE | [显式配置][dual-config]、[管线](scripts/run_dual_ising_current.py) |
| Ising K–B VAE | [显式配置][kb-config]、[正式计划][kb-plan] |
| $\phi^4$ 带状 VAE | [实际训练配置][phi-config]、[采样与训练计划][phi-plan]、[管线](scripts/run_phi4_ru_band.py) |
| Ising B=0 监督配对 | [实验配置][pair-config]、[配对报告][pair-report] |

配置中的服务器绝对路径需要适配。精确复现还须匹配原始数据指纹、源代码版本、初始化、loss、梯度路由、种子和实际长度；当前配置文件本身不能替代冻结 `plan.json` 与运行 manifest。历史 `--source`、旧 checkpoint 和缺字段配置具有兼容策略，不应通过重新解释默认值覆盖旧实验。

## 结果解释与适用边界

- **流场画的是 $R(g)-g$。**二维图用等长灰色箭头表示方向，以灰度深浅线性表示流动幅度；它不是 Jacobian 或 loss 的梯度。散点按真实生成参数着色，两个参数并排展示；粗尺度颜色仍代表原始参数。
- **数据支持逐尺度判断。**主图为 k=0，其他尺度独立展示；坐标外框、凸包、局部采样密度具有不同含义。不能把任意潜空间点反标为物理参数，也不默认填色插值未采样区域。
- **根分类使用完整 $DR=I+DF$。**离散映射按特征值模与 1 的关系区分吸引、排斥和鞍点；保留复数、负数及中性方向。有限搜索没有找到根不证明无根，找到根也不自动等于物理临界点。
- **新版结果是单 seed、有限尺寸证据。**36→12→4 的尺度闭合不等于任意大尺度泛化；链级重采样不覆盖训练 seed、尺寸截断或未热化偏差。VAE 表示分离、相边界对应和普适指数认证是不同层次的结论。
- **框架学习依赖数据与表示。**无需耦合标签不等于无需合格采样；局域 motif 统计不是任意系统的充分统计量，概率 C 与方差／保色约束也含结构假设。当前结果支持若干离散与连续模型上的无标签统计坐标和候选 RG 动力学，不构成任意黑盒物理系统的通用求解保证。

旧 CNN、仿射 Q、加权尺度、Potts、自旋玻璃、早期 Ising 与标量场实验不再混列为“最新结果”；正文、原图及路径保存在 [2026-11-05 版归档](README.history.2026-11-05.md)和各实验目录中。历史多 seed 或成功案例不自动作为新版 VAE 配方的验证。

[dual-config]: configs/dual_ising2d_weak_kl_vae_g2_seed91_20261105.json
[dual-report]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/report.json
[dual-flow]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/flow_k0.png
[dual-flow1]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/flow_k1.png
[dual-flow2]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/flow_k2.png
[dual-separation]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/q_multiscale_separation.png
[dual-coupling]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/coupling_to_g.png
[dual-loss]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/loss_curves.png
[dual-vae-loss]: results/dual_ising2d_b3_g2_weakKL_vae_seed91_vm225_20261105_v1_monitor/completed/run/report/q_vae_losses.png
[kb-config]: configs/ising2d_KB_weak_kl_vae_g2_seed91_20261106.json
[kb-plan]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/plan.json
[kb-report]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/report.json
[kb-roots]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/roots.json
[kb-flow]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k0.png
[kb-flow1]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k1.png
[kb-flow2]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k2.png
[kb-coupling]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/coupling_to_g.png
[kb-loss]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/loss_curves.png
[kb-vae-loss]: results/ising2d_KB_b3_g2_weakKL_vae_seed91_vm225_20261106_v1_monitor/completed/run/report/q_vae_losses.png
[phi-config]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/training_config.json
[phi-plan]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/plan.json
[phi-report]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/report.json
[phi-flow]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k0.png
[phi-flow1]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k1.png
[phi-flow2]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/flow_k2.png
[phi-coupling]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/coupling_to_g.png
[phi-loss]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/loss_curves.png
[phi-r-loss]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/terminal_R_loss.png
[phi-pca]: results/phi4_ru_band_b3_g2_weakKL_seed91_vm225_20261106_v1_monitor/completed/run/report/pca_variance.png
[phi-stable-report]: results/phi4_ru_band_stable_manifold_20261106_v1/comparison.json
[phi-boundary]: results/phi4_ru_band_stable_manifold_20261106_v1/physical_boundary_comparison.png
[phi-rectangle-report]: results/phi4_ru_b3_g2_weakKL_vae_softmaxfix_seed91_vm225_20261106_v2_monitor/completed/run/report/report.json
[pair-config]: configs/ising2d_B0_weak_kl_vae_g1_supervision_pair_seed91_20261106.json
[pair-report]: results/ising2d_B0_b3_g1_weakKL_supervision_pair_seed91_vm225_20261106_v1_monitor/completed/report/report.json
[pair-flow]: results/ising2d_B0_b3_g1_weakKL_supervision_pair_seed91_vm225_20261106_v1_monitor/completed/report/R_flow.png
[pair-coupling]: results/ising2d_B0_b3_g1_weakKL_supervision_pair_seed91_vm225_20261106_v1_monitor/completed/report/coupling_to_g.png
[pair-loss]: results/ising2d_B0_b3_g1_weakKL_supervision_pair_seed91_vm225_20261106_v1_monitor/completed/report/loss_curves.png
