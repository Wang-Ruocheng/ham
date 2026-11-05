# 欧氏实标量 φ⁴ 数据生成

该分支只生成物理数据，不修改或启动 QCR。默认二维、周期边界、格距及温度单位为 1：

\[
S[\phi]=\tfrac12\sum_{x,a}(\phi_{x+e_a}-\phi_x)^2
+\sum_x\left(\tfrac r2\phi_x^2+\tfrac u4\phi_x^4-h\phi_x\right),\qquad
p[\phi]\propto e^{-S[\phi]}.
\]

每个正向周期边只计一次；参数约定与 RGFlow 论文式 (25) 相同（另允许外场 h）。
支持 1、2、3 维实标量场，`u>0`；也支持 `u=0,r>0` 的可归一化高斯校准。
`u=0,r<=0` 的周期零模不被误当成正常概率分布。
这不是实时复权重路径积分，也不解决符号问题。

## 使用

默认只打印计划及工作量/存储投影，不创建输出、不运行采样：

```sh
python -m scripts.prepare_phi4
```

经批准后可显式执行，例如固定 `u=1,h=0` 扫描 r：

```sh
python -m scripts.prepare_phi4 --execute --output data/phi4_2d_L36_seed91 \
  --sizes 36 --r-values -2 -1.75 -1.5 -1.25 -1 -.75 -.5 -.25 0 .25 .5
```

默认 L=36 可用于以后默认 b=3 的 `36→12→4`；数据生成器不替用户预设 C 或 RG 流。
独立尺寸可通过 `--sizes 12 36` 指定，并按体积核算工作量；不会用已有构型插值/放大出独立尺寸。
上述 r 区间仅是可配置的探索区间，未用理论临界点筛选数据或训练；`r=0` 不是已知临界点。

## 采样与证据

- 固定参数 HMC：每轨迹独立高斯动量、leapfrog、完整 Hamiltonian Metropolis 校正。
  不裁剪场值，不删除拒绝构型，不动态调参、加样、换种子或失败重试。
- 默认每参数/尺寸 4 条独立链：2 条随机初态、正负有序初态各 1 条。
  每链预热 1024 条 HMC 轨迹，之后每 8 轨迹保存一次，共 128 个构型。
  HMC 轨迹不能称作格点 sweep；独立链也不表示链内保存构型独立。
- 随机流由主 seed、物理参数、尺寸、维数、链编号分离。没有全局 Z₂ 翻转增强，保留真实符号混合诊断。
- 保存动作密度、m、|m|、m²、m⁴、局域 φ²、零动量与最小非零动量结构因子；
  保存所有预热/生产轨迹的接受标记与 ΔH，逐链计算 lag1/IAT/ESS。
  Binder 和连通磁化率仅是有限样本诊断点估计，不自动寻找相变或报告临界指数。
- 热/冷链比较使用物理链均值而非把保存样本当独立重复。
  所有输出明确 `equilibrium_certified=false`；高接受率不等于热化或遍历。
- 轨迹、力计算、内存和目录大小在采样前按保守投影检查；不是墙时保证。

## 文件及标签边界

`plan.json`、`manifest.json`、`status.json` 记录约定、参数、工作量、来源 SHA256 和完成状态。
每个 `conditions/L*_c*/data.npz` 包含：

- `configurations`：float64，`[独立链, 保存构型, *空间, 1]`；连续值不二值化。
- `raw_observables`：`[链, 保存构型, 8]`；列名在 sidecar。
- `accepted`、`delta_h`：`[链, 预热+生产轨迹]`。

r/u/h 留在 `sidecar.json`/计划及评价元数据，NPZ 不写入耦合标签。未来训练只应读取构型，不能把诊断数组当输入。
TRAIN/TEST 按完整 `(r,u,h)` 参数单元拆分，同单元所有链及尺寸属于同一 split。
脚本拒绝任何已存在输出目录，不恢复或覆盖旧实验；失败留下已完成分片和失败状态。

**尚未接入当前二值 QCR。** 当前默认 C、离散 MI 和 Q motif 重构不直接适用连续实场。
本次未开启 legacy/nondefault recipe，也未将此数据伪装为可直接训练的 `ensembles.pt`。

参考：[RGFlow 论文](https://doi.org/10.1088/2632-2153/ae493b)；
[Neal：Hamiltonian Monte Carlo](https://arxiv.org/abs/1206.1901)。

## 2026-10-31：固定局域势的 hopping 扫描

新入口 `python -m scripts.prepare_phi4_hopping` 仅打印计划，不创建输出、不采样、不训练。
正式数据生成必须显式添加 `--execute --output 全新目录`；不会覆盖、续采或恢复历史结果。

采用与上述旧 `r/u` 生成器不同的、明确命名的参数化：

\[
S=-\kappa\sum_{\langle xy\rangle}\phi_x\phi_y
+\sum_x[\phi_x^2+\lambda(\phi_x^2-1)^2],\quad h=0.
\]

每条最近邻边只计一次，乘积项系数为 `-kappa`，没有额外因子2。
默认固定 `quartic_lambda=1`，`kappa=0.50,0.53,...,0.86` 共13个条件，二维周期L36；
每条件4条独立相关链、每链128构型，预热1024 hybrid cycles、每8 cycles保存一次。
整参数条件划分：8个TRAIN、5个TEST，所有链及尺寸随同一条件划分。
默认微观尺寸支持未来非重叠b3的 `36→12→4`，不改变当前QCR默认值。
物理内核可生成1/2/3维；专用QCR loader当前只接受单个二维尺寸、4链及偶数样本。

采用已验证的固定HMC＋lazy embedded-Ising Wolff核，内部精确变换

\[
\psi=\sqrt\kappa\phi,\qquad
r_{\rm old}=\frac{2(1-2\lambda)}\kappa-2d,\qquad
u_{\rm old}=\frac{4\lambda}{\kappa^2},\qquad
S_{\rm hop}(\phi)=S_{\rm old}(\psi)+\lambda V.
\]

leapfrog步长0.08、12步均指内部psi坐标；物理phi坐标的等价HMC质量为kappa。
热初态沿用psi的标准高斯，故phi初始宽度随kappa变化；此初始化不是生产分布或额外归一化。
簇激活概率自动成为 `1-exp(-2*kappa*abs(phi_i*phi_j))`。
保存前把构型、完整磁化轨迹、单点/二点矩和所有诊断还原为原始phi单位，动作密度补回lambda常数。
不会把条件依赖的psi幅度直接交给Q，也不会改写历史采样核。
只支持正kappa、正lambda及h=0；kappa=0的物理模型虽有定义，但此坐标变换奇异，明确拒绝。

默认工作预检为106496个hybrid cycles、1794244608次force-site evaluations；
目录投影约84MiB、数组/采样工作区投影约28MiB（不含Python和依赖库运行时基线），
另有轨迹、簇工作和目录预算门控；此内存投影不是实际RSS峰值保证。
工作量不是墙时估计；当前没有启动这组正式采样。极小CPU单元测试不构成平衡或相变结果。
原始接受率、ESS、IAT、热冷和符号混合警告均保留，绝不因警告自动补样或改变参数。

新schema为 `euclidean_phi4_hopping_hmc_embedded_wolff_v1`，公开参数名是
`(kappa,quartic_lambda,h)`；构型NPZ不含耦合标签。
`recurrent_rg.data.phi4_hopping_loader.load_dataset` 严格校验完成状态和SHA，仅解压
`configurations`，其固定幅度尺度只由TRAIN原始phi的RMS决定。
**不能直接接入旧的r/u网格loader或r轴报告；未来训练入口及报告必须显式识别kappa标签语义。**

独立数值参考：lambda=1时无限体积kappa_c=0.680601(11)，见
[Zhong等，式(1)、表I](https://arxiv.org/pdf/1807.05123)。此参考只作生成范围设计和事后评价，
不作为Q/R训练标签、根搜索目标或有限L36真值；数据范围覆盖该参考不等于实际链已正确热化。

## 连续与离散QCR的实现对齐审查

截至2026-10-31，历史模型和loss未因本次数据管线开发而修改。

| 部分 | 二维离散Ising | 当前连续phi4 | 审查结论 |
|---|---|---|---|
| C几何 | 随机可训练、非重叠3×3/stride3 | 相同 | 可对齐 |
| Q输入 | 原始3×3块→可学习局域MLP→系综平均 | 固定105维单点/两点软统计→MLP | **实质不同** |
| Q仿射重构 | 完整2×2的16种模式概率 | 5个单点软bin与4组25维成对分布 | 连续版不保留完整四点联合统计 |
| Q外权重 | 重构10、微观子集20、粗尺度子集5 | 相同 | 权重相同不代表数值量纲或任务相同 |
| C概率与梯度 | Bernoulli，straight-through梯度 | 固定噪声Gaussian，重参数梯度 | 连续适配，不是逐项等价 |
| C信息项 | buffered full-shell经验直方图MI | 同壳层的核平滑信息代理、每链128区域对 | 不同估计器 |
| C防塌缩/保色 | 逐系综二值方差floor＋均匀块NLL | 条件均值标准差floor＋均匀幅度MSE | 目标目的相近，函数不等价 |
| R | residual MLP、all_start_anchor、均匀目标尺度 | 相同 | 两转移loss及R梯度已合成验证等价 |
| 梯度路由 | Q/C/R独立 | 相同 | 合成验证无串流 |

连续Q是 `105→96→12→96→d_g`；离散Q局域网络是 `9→96→12→12` 后平均，
再接 `12→96→d_g`。二值512种3×3块枚举是输入计算加速，不是16维重构目标。
两者decoder均约束列单位范数，但连续版forward可微归一化、离散版更新后投影，
Adam轨迹不严格等同。两者无PCA，decoder bias均由TRAIN随机初始C轨迹校准；
离散版最多32个TRAIN系综，连续版遍历全部TRAIN。

连续历史设置为3000次随机批更新、8物理链×64构型、hidden96、float64、Adam0.001常数、
CPU单线程。相应历史b3 Ising配置为3000次更新、32系综×64构型、CPU两线程；
不能只因更新数相同就声称训练数据暴露量完全一致。当前通用dataclass默认值也不能代替实际运行manifest。
三个独立Adam与一个包含独立参数梯度的标准Adam，在相同参数级超参数下不是梯度路由差异；
但两套程序的初始化消耗和batch/C/MI随机流不同，同seed不构成逐参数配对。

新增 `tests/test_phi4_continuous_discrete_alignment.py` 给出一个合成反例：
两组二值系综的所有单点/两点分布相同，但2×2四自旋乘积分别为+1和−1。
当前105维特征最大差约1.4e−17，离散16维plaquette分布L1差为2。
这证明表示侧确有信息缺口，即使增加d_g也不能恢复已被输入统计丢弃的信息；
**它不是实际phi4训练失败原因的实测归因。**
如需严格控制“仅数据从离散变连续”，应另建显式命名的连续raw-patch Q及对应局域联合重构，
而不是静默修改现有历史模型或假定当前连续版本已与离散版完全对齐。

## 2026-10-31：补齐后的连续对齐配方

用户授权补齐后，新增独立配方 `phi4_hopping_raw_patch_joint625_v1`，保留上述历史
105维模型、检查点、报告和实验。实现位于 `recurrent_rg/phi4_aligned_qcr.py`，
训练入口为 `experiments/phi4_hopping_qcr.py`；不会自动迁移或续训历史模型。

新的 Q 不再以固定105统计为唯一输入：重叠原始3×3块，经 `9→96→12→12`
局域MLP后对同一物理系综的样本和平移平均，再以 `12→96→d_g` 输出潜变量。
各局域层使用tanh，head末层为线性，与离散版对应；连续幅度只除以**同一个冻结的
TRAIN微观RMS**，不按κ或尺度重新归一化，不进行符号二值化。

共享仿射解码器的目标改为2×2四点的完整软联合概率。每个场值使用5个Gaussian软bin
（中心−2、−1、0、1、2，宽度0.5，均以冻结RMS为单位），四点顺序为
`(00,10,01,11)`，联合向量625维且质量为1。它包含旧单点／两点描述符丢弃的四点依赖，
但仍是**有限软分箱的局域描述**，不是任意连续分布的充分统计，也不与离散16维loss数值等价。
解码输出是原始仿射值，不加softmax或forward可微归一化；每次Adam更新后投影列范数至1，
不强制满秩。重构为625个bin的平方误差之和，再对系综和尺度平均。

默认配置 `configs/phi4_hopping_b3_g1_aligned_seed91.json` 明确记录：

- `schema_version=5`，二维非重叠b3，`36→12→4`，`d_g=1`、hidden96、局域特征12、float64。
- 3000次更新，32条物理链×64个保存构型，fresh Adam0.001且学习率不变，CPU单线程。
- Q重构10、微观子集一致性20、粗尺度子集一致性5；直接编码为两半独立Q输出的平均，
  并非将两半合并后调用一次非线性Q。
- R使用统一 `all_start_anchor`，目标尺度均匀，兼容外权重5+5；递归保留R梯度，
  不加入future/prediction-prediction项。所有Q/C/R梯度在任何参数更新前独立算完。
- 初始化只改decoder bias：前min(32,TRAIN)系综、全部保存构型、随机初始C的3个尺度均匀平均；
  私有seed+9000，不推进训练随机流，不使用TEST、耦合标签、PCA或C预训练。
- Q和625维目标均分块计算，Q使用activation checkpoint以限制中间激活；
  `patch_chunk_size=4096`是计算分块，不是减少物理样本数。新计算成本**未沿用旧105版耗时估计**，
  正式启动前仍需资源与预算核实。

C继续采用已授权的连续Gaussian概率核、buffer1的完整壳层平滑信息代理、条件均值方差floor
及均匀幅度MSE，权重仍为−2／2／1。它们是连续适配，不是二值经验直方图MI与保色NLL的
逐项等价物；manifest明确 `exact_binary_recipe=false`。这次对齐没有解决或掩盖这一差别。
连续随机流和625维decoder也不同于离散版，不能因相同seed声称逐参数配对或严格历史复现。

只核查新训练计划（不读取构型、访问GPU、创建输出或训练）：

```sh
python -m experiments.phi4_hopping_qcr \
  --config configs/phi4_hopping_b3_g1_aligned_seed91.json
```

正式生成数据和训练仍需另行启动授权与资源核实。届时可通过统一入口：

```sh
python -m recurrent_rg train \
  --config configs/phi4_hopping_b3_g1_aligned_seed91.json \
  --output results/NEW_RUN --device cuda:2
```

统一入口仅按明确recipe路由，不开启legacy/nondefault绕过开关；已有离散配置保持原入口行为。
新专用入口必须显式 `--execute` 才执行，拒绝已存在输出、续训、warm start和旧r/u loader。
训练只读取配置构型；κ／λ／h仅附到事后评价数组，并按完整物理条件划分TRAIN/TEST。
预算触发不会减步数、迁移或自动重试，已完成步骤和检查点保留。

结束后自动从公共JSON/NPZ生成中文loss、κ→g、R(g)图，箭头始终为R(g)−g。
流网格仅由TRAIN的k=0、1输入范围加边界余量确定；Q整系综编码与两半均值编码另存。
报告不加载检查点、不重新估计MI、不搜索不动点或临界κ，`thermal_nu=null`。
本次仅实现与合成／极小CPU回归验证，**未启动新正式采样或训练，暂无新物理结果**。

### 同日正式启动与预算延续

225/GPU3 的完整批量 CUDA 预检正常结束：2次warmup＋5次计时更新，
每次更新2.62–2.75秒，峰值1.42GiB。原30分钟子流程保护线不足；保守训练投影
12669.778秒（约3.52小时），故原父流程仅完成合成预检即budget_hold，未生成正式物理数据或检查点。
这不是科学结果或执行失败，原回执及其rc=3保留。

用户随后明确授权“提高并继续”：新增配置
`configs/phi4_hopping_b3_g1_aligned_seed91_budget4h.json`，相对原配置**仅**把
`training.max_training_seconds`从1800改为14400。新增父入口
`scripts.continue_phi4_hopping_budget`使用GPU总预算4卡时、CPU总预算6核时；
不改变旧默认配置或旧父程序。它在全新输出中继续采样与训练，原有534个真实计算源码SHA全部核对，
旧预检和此前零计算队列的用量结转，另行保存新预算准入回执。
不再执行预检更新、不加载合成模型、不重采旧数据、不改样本量／seed／3000步／loss。
采样完成后依旧要求非零HMC生产接受率；其余有限链质量警告保留。
资源与投影继续在阶段间核验，不因预算不足减样本或减步数。

### 同日固定积分细化：用户授权“继续”

上述4小时预算尝试在首个κ=.50条件的第二条热初态链发生leapfrog力溢出，
完整条件数为0，QCR尚未训练。原失败根、日志、源代码和累计资源保留，不原地修复或重跑。
用户随后授权固定psi坐标HMC从`0.08×12`细化为`0.04×24`，轨迹长度仍0.96。
物理作用量、52条生产链seed、样本量、模型、loss和3000次训练更新不变；
数值proposal改变，但原kernel实现文件不变。生产力格点计算从1,794,244,608增至
3,450,470,400，因此独立计数适配器显式将该工作量保护线上限从20亿改为40亿，
其他资源保护不变；不修改历史wrapper中原有硬上限。

新入口`scripts.run_phi4_hopping_refined`、`scripts.prepare_phi4_hopping_refined`和
`recurrent_rg/data/phi4_hopping_refined_pipeline.py`部署到独立冻结快照。
GPU预算4卡时、CPU预算6核时，原失败账本累计CPU32.966527042秒、GPU50.384840496秒
只结转一次，已包含原预检和取消队列，不再执行GPU预检更新或加载旧模型。

北京时间2026-10-31 14:08启动父进程12393。固定27条CPU短链验证包括13个κ各热／冷
目的分离流，加原失败热初态同seed回归流；各128预热＋32保存周期，全部不入生产数据。
27条均有限、接受率0.84375–1.0、rc0，原失败流生产段接受率0.84375。
这仅验证短期数值稳定和接受率，不认证热化。根据实测CPU18.6172秒，完整采样CPU
保守投影为917.898秒（2倍余量），联合既有训练投影通过6核时门控。

正式新根为`results/phi4_hopping_b3_g1_aligned_seed91_hmc004_budget4h_gpu3_vm225_20261031`；
14:09已启动完整物理采样，采样子进程12481，实际NI10、NLWP1。
父程序仅在采样完成及资源／预算检查通过后自动进入指定GPU3的3000步训练。
本段为启动回执而非完成结果；无需因短链通过而声称已平衡或恢复了相变结构。
