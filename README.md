# Recurrent RG Network：无耦合标签的 QCR

从构型系综联合学习局域粗粒化 \(C\)、低维理论坐标 \(Q\) 和共享的 RG 映射 \(R\)：

$$
\mathcal D_k=C^k(\mathcal D_0),\qquad
g_k=Q(\mathcal D_k),\qquad
R(g)=g+F(g).
$$

当前主线是 **Q/C/R 同步训练**，不使用 RSMI、作用量 A、耦合标签、相标签或已知临界点训练。
模型只知道哪些构型属于同一系综；真实耦合仅用于数据生成和事后绘图。
这是学习有限尺度统计及其潜空间动力学的研究原型，不预设潜坐标就是完整物理耦合。

## 关键实验：一维 TFIM，dim g=1 与 dim g=2

同一物理问题、同一份缓存数据：周期一维横场 Ising
\(H=-J\sum_i Z_iZ_{i+1}-h\sum_i X_i\)，无纵向场。
Q/C 只读取等时自旋构型，参数 \(h/J\) 不进入 loss。

| 设置 | dim g=1 | dim g=2 |
|---|---|---|
| 原始链长与尺度 | 128→64→32→16→8→4 | 相同 |
| 隐藏层宽度 / seed | 60 / 91 | 60 / 91 |
| 优化的 Q/C/R 参数量 | 9,711 | 9,893 |
| 展示检查点 | 联合训练第 1,000 步 | 联合训练第 6,000 步 |
| R-only 后处理拟合 | 无 | 无 |

数据包含64个均匀的 \(h/J\in[0.01,2]\) 取值，每个3条采样链、每系综128构型。
按耦合划分129个训练系综、63个测试系综（同一耦合的所有 replica 在同一侧），
共24,576个原始构型。训练每步抽32个系综，每系综128个样本并作全局翻转增强。
Suzuki–Trotter 数据取 \(M=32,\Delta\tau=0.5,\beta=16\)；有限温度和离散误差仍需独立控制。

**这是两个已完成联合训练检查点的定性展示，不是等训练预算、多 seed 的维度优劣比较。**
标量采用此前两侧保持分离的第1,000步：它的第2,000步曾出现临界两侧长期汇合。
向量采用原 coordinate-sum 模型续训至6,000步；并未替换成固定 Q/C 的 R-only 对照。

### dim g=1：R(g) 与恒等线

![一维 TFIM，dim g=1，联合训练1000步](docs/figures/tfim_g1_joint_R.png)

数值不动点为 \(g=-0.87699,-0.28898,1.01028\)，对应导数
\(0.78063,1.39458,0.32543\)：两侧稳定、中间不稳定。

### dim g=2：二维潜空间流

![一维 TFIM，dim g=2，联合训练6000步](results/joint_qcr_tfim_1d_g2_coordsum_continue6000_seed91/step6000/R_vector_field.png)

箭头为离散位移 **\(R(g)-g\)**，不是位置向量 \(R(g)\)；箭头长度归一化，颜色表示原始位移。
灰色箭头位于离观测输入较远的区域。绘图区内有一个稳定点
\((0.36714,-0.27303)\) 和一个鞍点 \((-0.44587,0.80690)\)；
鞍点特征值约 \(-0.06941,1.33246\)。

将鞍点投影到微观平均 \(g_0(h/J)\) 折线，最近点对应 \(h/J\approx0.9796\)，
但仍有约0.0389的潜空间残差：不是精确坐标对应，也不是独立验证的物理临界点。
Q 编码主要落在近一维曲线上；离曲线方向的导数约束较弱。

延长训练改变了定性流形态，但向量版6,000步的总测试 loss 比2,000步高，
且 C 的离散规则发生变化。不能仅凭好看的流图宣布完全收敛或正确临界指数。

- [标量完整历史报告](results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91/REPORT.md)
- [向量续训报告、误差与检查点](results/joint_qcr_tfim_1d_g2_coordsum_continue6000_seed91/REPORT.md)
- [标量各尺度 g(h/J)](results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91/step1000/g_by_scale_vs_h_over_J.png)
- [向量各尺度 g(h/J)](results/joint_qcr_tfim_1d_g2_coordsum_continue6000_seed91/coupling_mapping_step6000.png)

## 当前模型

输入张量为 [systems, samples, *spatial_shape, channels]，各尺度共享同一组 Q/C/R 参数。

- **C**：带 bias 的局域 MLP，二值 hard forward + straight-through 梯度。
  非重叠二倍分块：1D输入相邻2格，2D输入2×2格，每块直接输出一个粗变量。
- **Q**：重叠周期窗口（1D宽3、2D为3×3）→共享局域 MLP→样本与空间平均→理论坐标头。
  是系综的点估计，不是显式概率后验；没有加入额外总体磁化、BatchNorm 或强制满秩表示。
- **R**：残差 MLP，\(g+F(g)\)，两层隐藏层、Tanh，各尺度共享，无显式 k 输入。

不对可学习 Q/C/R 施加硬 Z2 等变约束。零场 Ising/TFIM 的翻转配对属于数据增强，
不适用于非零纵场。二值 lookup 保留原输入梯度，是经测试的计算加速，不是另一种 loss。
固定正 decimation 仍作为显式物理对照保留；它不是当前两张主图的 C。

## 当前 loss：全尺度数据几何 + 单步/多步闭环

实现：\(\texttt{experiments/joint\_qcr\_ising.py}\) 中的
\(\texttt{data\_geometry\_statistics\_objective}\)。
下文 \(K=\texttt{levels}-1\)，尺度为 \(k=0,\ldots,K\)，\(i\) 表示系综。
每个系综分成两个子集 a、b，定义

$$
g^a_{i,k}=Q(C^kD_i^a),\quad
g^b_{i,k}=Q(C^kD_i^b),\quad
\bar g_{i,k}=\tfrac12(g^a_{i,k}+g^b_{i,k}).
$$

当前默认目标为

$$
\boxed{
\mathcal L_{\rm QCR}
=20\mathcal L_{\rm split,0}
+5\mathcal L_{\rm split,coarse}
+2\mathcal L_C
+10\mathcal L_{\rm data}
+5\mathcal L_{\rm step}
+5\mathcal L_{\rm rollout}
}
$$

### 1. 同系综子集一致性

$$
\mathcal L_{\rm split,0}
=\langle\|g^a_{i,0}-g^b_{i,0}\|_2^2\rangle_i,\qquad
\mathcal L_{\rm split,coarse}
=\frac1K\sum_{k=1}^K\langle\|g^a_{i,k}-g^b_{i,k}\|_2^2\rangle_i.
$$

要求同一系综的不同样本子集给出相近坐标，减少采样噪声。
坐标维度**求和**、系综和尺度取平均，不因增加 dim g 自动减小约束。
标量求和与求平均等价；旧向量结果可显式指定 coordinate-reduction=mean 复现。

### 2. C 的防塌缩

零场主实验使用每层 coarse field 的 batch 均值 \(m_k\)：

$$
\mathcal L_C=\frac1K\sum_{k=1}^K m_k^2.
$$

这里平均覆盖当前 batch 的所有系综、样本和空间位置，不是要求每个系综磁化为零。
常数输出±1的惩罚为1；梯度通过 STE 回传。这是二值平衡的软约束，
不保证信息保留，也不是 RSMI。

非零纵场的独立入口保留允许偏置的 variance-floor 版本，避免错误要求零磁化；
它不用于上述 TFIM 主图。固定 C、场类型、geometry 等通用适配能力没有删除。

### 3. 由数据决定的分布几何

在每层统计局域模式频率 \(p^a_{i,k},p^b_{i,k}\)：
1D 为相邻两自旋的4种模式，2D 为2×2 plaquette的16种模式。
定义

$$
\bar p_i=\tfrac12(p_i^a+p_i^b),\quad
v_i=\tfrac14\|p_i^a-p_i^b\|_2^2,\quad
d_{ij}=\|\bar p_i-\bar p_j\|_2,\quad
e_{ij}=2\sqrt{v_i+v_j},
$$

$$
\ell_{ij}=4(d_{ij}-e_{ij})_+,\qquad u_{ij}=4(d_{ij}+e_{ij}).
$$

$$
\mathcal L_{\rm data}
=\frac1{K+1}\sum_{k=0}^K
\left\langle
(\ell_{ij,k}-\|\bar g_{i,k}-\bar g_{j,k}\|_2)_+^2+
(\|\bar g_{i,k}-\bar g_{j,k}\|_2-u_{ij,k})_+^2
\right\rangle_{i<j}.
$$

数据目标及噪声容差停止梯度；坐标距离保持梯度。因子4是固定坐标单位，
因子2控制分半噪声容差，均不是学习到的物理常数。
容差是启发式估计，不是置信区间；局域频率不是完整构型分布。
分布无法区分时允许相同 g，不强制每个系综分开或把潜空间全部维度填满。

### 4. 单步与多步闭环

$$
\mathcal L_{\rm step}
=\frac1K\sum_{k=0}^{K-1}
\langle\|R(\bar g_{i,k})-\bar g_{i,k+1}\|_2^2\rangle_i,
$$

$$
\mathcal L_{\rm rollout}
=\frac1K\sum_{k=1}^{K}
\langle\|R^k(\bar g_{i,0})-\bar g_{i,k}\|_2^2\rangle_i.
$$

所有可用尺度等权，没有旧版高 k 衰减。多步项从 g0 自主递推，保留全部 R 调用的梯度；
闭环两侧都参与联合优化，没有冻结 Q/C 的目标或交替更新。
**尚未采用**此前讨论的“所有起点、所有跨度”、最差尺度加权或更多 R 专用更新。

L128主实验取levels=6，覆盖k=0..5；k>5仍是无数据监督的外推。
当前不把只剩2格或1格的退化层加入此实验。

### 日志字段与数学项

| 日志字段（未乘外部权重） | 数学含义 | 外部权重 |
|---|---|---:|
| anchor | \(10\mathcal L_{\rm split,0}\) | 2 |
| coarse_consistency | \(\mathcal L_{\rm split,coarse}\) | 5 |
| kernel_noncollapse | \(\mathcal L_C\) | 2 |
| separation | \(\mathcal L_{\rm data}\)，不是旧固定分离下限 | 10 |
| commutator | \(\mathcal L_{\rm step}\) | 5 |
| rollout | \(\mathcal L_{\rm rollout}\) | 5 |

保留历史日志键以便读取旧报告；不再把 legacy 锚点/固定跨度目标混写为当前 loss。

## 运行与复现

安装依赖并测试：

~~~bash
python -m pip install -e ".[dev]"
python -m pytest
~~~

当前一维 TFIM 主配置（命令为明确参数，不依赖旧默认值）：

~~~bash
python experiments/joint_qcr_tfim_1d.py \
  --sites 128 --levels 6 --hidden 60 --theory-dim 1 \
  --ensemble-size 128 --sample-batch 128 --system-batch 32 \
  --epochs 1000 --seed 91 --binary-lookup \
  --objective-mode data_geometry --coordinate-reduction sum --rollout-weight 5 \
  --output-dir results/tfim_g1_new_run

python experiments/joint_qcr_tfim_1d.py \
  --sites 128 --levels 6 --hidden 60 --theory-dim 2 \
  --ensemble-size 128 --sample-batch 128 --system-batch 32 \
  --epochs 2000 --seed 91 --binary-lookup \
  --objective-mode data_geometry --coordinate-reduction sum --rollout-weight 5 \
  --output-dir results/tfim_g2_new_run
~~~

上述命令启动新训练，不保证跨硬件逐位重现已保存权重。
向量主图来自原2000步检查点的连续续训；历史恢复入口为
[scripts/continue_tfim_g2.py](scripts/continue_tfim_g2.py)，其源/输出目录固定在脚本内，
运行前检查路径，不要无意覆盖已完成的续训目录。

标量报告输出 R(g)、R(g)-g、各尺度 g(h/J) 及 loss/闭环；
向量用 [scripts/plot_qcr_vector_flow.py](scripts/plot_qcr_vector_flow.py)。
所有训练报告必须注明检查点，画 R 的流和真实 Q(C^kD) 对耦合的映射，
不能以 R 递推冒充直接 Q 编码。

## 代码与历史边界

- recurrent_rg/：模型、几何、场适配器与采样器；历史作用量能力保留，但主 QCR 不优化 A。
- experiments/joint_qcr_ising.py：共享目标、训练、评估及优化器/RNG续训。
- experiments/joint_qcr_tfim_1d.py、joint_qcr_tfim_2d.py：量子TFIM入口。
- experiments/joint_qcr_ising2d_field.py：二维经典 K/B 问题，不是二维量子TFIM。
- [archive/qcr_ablations/](archive/qcr_ablations/README.md)：旧固定分离loss、强制协方差利用、
  额外磁化与Q结构消融；从主入口移除但保留历史可追溯性。
- scripts/：报告、续训及独立 R-only 诊断；诊断脚本不参与主 loss。
- results/：历史实验保留；本页仅以一维TFIM两种潜维度作为关键展示。

本次清理不删除数据、权重和历史报告，不新增训练。
测试 loss 低、相界附近有不动点、得到两侧吸引的流，都不是精确 RG 或正确临界指数的充分证据。
