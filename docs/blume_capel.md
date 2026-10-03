# 二维 Blume-Capel：Delta/J=1，dim_g=1

## 物理问题与数据

零场方格模型H=−J sum_<ij> s_i s_j + Delta sum_i s_i²，s_i∈{−1,0,+1}。
取J=1、Delta=1，T从.6到2.4均匀64点；这是二维参数空间的一条路径，
并不保证粗粒化后仍沿此路径或一维表示精确闭合。
文献中的该截面临界温度约T_c/J=1.398，是数值估计，不是精确解析值：
[Zierenberg et al., Scaling and universality in the phase diagram of the 2D Blume-Capel model](https://arxiv.org/abs/1612.02138)。
标签只用于留出划分、画图和物理审计，不进入模型及loss。

每温度3个独立种子的MCMC系综，L16、每系综128构型；按完整温度点划分，
129训练/63测试系综，无同温度副本泄漏。one-hot通道依次表示−1、0、+1。
未作额外符号或颜色增强，尤其不能把三种状态全排列当作物理对称性。

## 采样器

实现位于`recurrent_rg/data/blume_capel.py`。每个hybrid sweep：

1. N=L²次随机格点热浴更新，条件概率P(s|neighbors)∝exp[(J s sum_neighbors s_j−Delta s²)/T]。
2. 占据格点上等号自旋之间以1−exp(−2J/T)成键，然后各簇独立赋±号；0格点保持0。

热浴更新改变空缺数，避免只用符号簇更新的不遍历性。两步均保持目标Boltzmann分布。
默认burn=1024轮，每8轮记录一次；MCMC样本不假定完全独立。
L=2使用周期多重边，与right/down能量计数一致。

验证包含：条件概率归一化及符号对称、固定seed复现、J=0独立格点占据率，
以及L=2全部81构型精确配分函数对能量/占据率/磁矩平方的核对。
测试中的小系统枚举仅是采样器单元测试，绝非选核或训练初始化。
额外在5个温度用random/plus/zero三种初态、2048轮burn及1024样本检查混合，
这些probe不加入训练缓存。输出sampling.json含逐系综均值、分块SE及lag1相关；
分块SE是近似诊断，不是平衡或独立性的严格证明。
现有检查无明显初态依赖；低温|m|≈.992，高温≈.08。

![Sampling](../data_cache/blume_capel2d_delta1_L16_n128_seed91/sampling.png)

## 模型与训练

与成功的[随机C Potts配方](potts_random_gradient_mi.md)逐字段比较，model/loss/training/evaluation完全一致。
dim_g=1、hidden60、11873参数、随机categorical概率C、2×2不重叠块，16→8→4。
Q/C/R随机初始化，无选核、预训练、冻结、保色项或硬对称。
同步独立梯度：C接受−2经验MI+2逐系综多尺度方差下限（ratio=1）；
Q接受原relative_local几何及子集一致性；R接受一步和全尺度多步闭环（权重各5）。
Adam .001，batch32×64，seed91，固定2400更新。
MI完整20格环境、buffer=1，有限样本经验MI偏差仍是限制。

## 复现

训练入口新增与--source互斥的--config，允许不依赖历史运行目录的新物理模型。

```bash
python -m scripts.generate_blume_capel2d \
  --output data_cache/blume_capel2d_delta1_L16_n128_seed91
python -m experiments.rsmi_separate_qcr \
  --config configs/blume_capel2d_delta1_g1_random_mi_seed91.json \
  --output results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_scalar_joint \
  --run results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91
python -m scripts.report_blume_capel_scalar \
  --run results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91
python -m scripts.report_mi_training \
  --run results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91
```

输出目录须不存在，旧实验不会被覆盖。本次新增功能后全套251项测试通过。

## 首轮结果：尚未通过相变结构验收

2400更新完成，训练循环902.21秒（约15分钟，不含数据/统计预处理与后处理）。
模型、loss、训练与评估配置与Potts逐字段相同，未中途改参或挑选checkpoint。

单步R在宽查根区间内只找到稳定根g=−.927462，R′=−.963364，
没有微观温度网格内的反插值。Q在文献Tc处为g≈−.163426，明显不等于该根。
观测R输入区间[−3.728091,1.622459]。
800步的单步不稳定根曾对应T≈1.80815，也不是准确临界点。

两步映射还存在：

- 稳定二周期：−3.527191 ↔ 1.279752，DR²≈−.005139。
- 不稳定二周期：−1.408632 ↔ −.521268，DR²≈1.180326。
- 原单步稳定根仍是R²稳定根，DR²≈.928071。

二周期中的两个点不是两个独立的R不动点；不稳定周期的−.521268点可以
反插值到微观T≈1.505935，比参考Tc高约7.72%，另一点没有微观插值。
该结果有两步分流信息，但不足以宣称准确恢复相变，也不能仅因出现二周期就认定
所有RG信息都错误。查根有界，未证明穷尽。

C对均匀块近似做−1→0、0→−1、+1→+1，概率均超过.99999。
这不是Blume-Capel的正负翻转对称；零态与磁性态不等价。
它可能造成尺度间状态解释与闭环的困难，但未做消融，不将其认定为唯一失败原因。
固定Delta路径本身不保证一维RG闭合、有限尺寸、经验MI偏差与优化也均待区分。

测试一步NRMSE分别7.25%、23.01%，两步rollout NRMSE=21.91%。
最终加权QCR不含MI=.668675：微观一致性.048767、粗尺度一致性.016512、
几何.227901、C方差.001259、一步闭环.199288、多步闭环.174948。
MI=1.330417bit且很早稳定，但Q/R损失后期有明显波动，不能说整个训练已收敛。
微观Q正负翻转RMSE=.019805，原编码SD=.638715；无硬对称约束。

![R and temperature mapping](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91/full_report/R_and_T_mapping.png)
![Two-step audit](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91/full_report/R2_audit.png)
![Loss](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91/report/loss_comparison.png)

## 只读原因诊断

`python -m scripts.audit_blume_capel_cause --run results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91`
不更新模型权重；使用全部128构型、3次C采样，温度仅用于事后分组评估。

C没有抹去全部相信息：标签置换不变的归一化近邻连通同态相关，在T=1.0时
由.777增至.990，在T=1.8时由.230降至.075（k=0→2）。均匀块的−1/0交换
不是哈密顿量对称，但粗变量可重新命名，不能仅凭这一点认定核无效。
奇数尺度事后交换标签只用于诊断；一般混合块的C不与该交换对易，不能把整个C解释为纯置换。

固定最终Q/C目标，在训练温度均值的(g_k,g_{k+1})上做一维分段线性插值，
测试于留出温度均值：一步MSE从R的[.007077,.071078]变为[.000365,.000652]，
两步MSE从.063849变为.001144。测试一步输入无需外推。
这是平滑目标的可拟合性诊断，不是重新训练R，也不能证明现有R架构必然达到插值精度。
C重复采样的g标准差仅约.00016/.00028，不足以解释上述闭环误差。

共同24个系综上的2000→2400 checkpoint目标漂移RMS=.3131；
仿射对齐g_2400≈1.0991 g_2000+.2583后残差RMS仅.0166，说明主要是坐标伸缩和平移。
同样数据的一步R MSE从.00842升到.11188。独立梯度训练没有通过R约束Q坐标稳定，
R追踪非平稳目标是当前最直接的优化问题，而非已证实的C相信息塌缩。

但拟合好R也不等于恢复正确临界点：全128构型下参考Tc=1.398处直接g为
[−.1806,−1.9812,.3447]；奇数尺度标签对齐后为[−.1806,.1305,.3447]，仍有漂移。
该事实不足以单独定位C或Q的问题，也不排除有限尺寸、缺失RG坐标及无关算符修正。
下一项有辨别力的控制实验是固定Q/C、只优化R；本次诊断没有执行训练。

![Cause diagnostics](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91/cause_audit/cause_diagnostics.png)

## 原框架续训至3600步

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91 \
  --resume-checkpoint results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_2400_seed91/checkpoint_last.pt \
  --epochs 3600 \
  --output results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_continue3600_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
```

恢复Adam及随机数状态，只有总更新数改变，无冻结、调学习率或改模型；新增1200步耗时450.31秒。
该驱动没有QC收敛检测或自动冻结，每一步仍更新Q/C/R。
2400→3600的测试加权闭环loss由.374236降至.156199；
一步NRMSE由[7.25%,23.01%]降至[6.56%,16.51%]，两步rollout由21.91%降至15.49%。
中途仍明显振荡，不能视为充分收敛。

共同24个系综、64构型、采样seed444下，C全部81种块的输出概率平均变化4.91e-6，
最大变化1.70e-5；Q多尺度坐标变化RMS=.16671，近似
g_3600=.94198 g_2400+.07435，仿射对齐残差RMS=.01233。
这支持C基本稳定但Q坐标仍漂移，不能认为后期只有R在优化。

最终R仅查到不稳定根g=−.612779，R′=−1.097187，无微观温度均值曲线内插值；
参考Tc的微观g≈.04475。R²另有稳定二周期−3.468067↔1.833038，乘子.207467，
两端分别略低于/高于观测R输入范围[−3.455025,1.606327]。
两端是同一个R二周期，不是两个独立相吸引子，不能据R²三根宣称恢复正确相变。

![Continued flow](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_continue3600_seed91/full_report/R_and_T_mapping.png)
![Continued losses](../results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_continue3600_seed91/report/loss_comparison.png)

## dim g=2 对照：同数据、同配方、3600步

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/blume_capel2d_delta1_g1_random_mi_variance2ratio1_continue3600_seed91 \
  --theory-dim 2 --epochs 3600 \
  --output results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_probabilistic_chain_g2 \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91 \
  --critical-coupling 1.398
python -m scripts.audit_blume_capel_vector \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91
python -m scripts.report_mi_training \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91
```

从头随机初始化，不载入单维权重；seed91，配置仅theory_dim从1改为2。
参数12055（Q3470/C4623/R3962），对比单维11873；没有额外协方差、防维度塌缩或保色项。
数据指纹相同，运行1370.84秒（22.8分钟）。最终C的81种块概率表与单维3600步
平均差1.43e-9、最大差1.16e-8，核基本相同，仍交换均匀−1与0块。

测试一步NRMSE=[5.60%,10.51%]，两步rollout=9.71%，优于单维[6.56%,16.51%]及15.49%。
加权闭环loss=.08011（单维.15620）；几何项=.22449（单维.24173）。
各尺度弱主成分占比约[.466%,1.575%,.399%]，各自仍是细曲线；合并尺度弱占比38.9%，
说明不同尺度占用不同方向，但不是两种独立物理控制参数的证据。

有限查根框内未找到R固定点。R²找到稳定二周期
(1.33213,−.93864)↔(−1.13267,1.78854)，乘子[.47874,.04860]；
以及鞍型二周期(.95076,1.41477)↔(1.35079,1.07570)，乘子[2.30254,.28367]。
后者靠近微观曲线的点最近线段投影T/J=1.46397，距离.01297，
相对参考Tc=1.398高4.72%；这是最近投影，不是严格反插值或经过有限尺寸外推的Tc估计。
实际Tc的微观编码约(1.31497,.87152)。另一鞍周期端点靠近k=1曲线，
投影T≈1.47454、距离.04423。

低温R长递推走向稳定二周期，T≈1.514及更高温朝另一方向离开观测支持；
只训练了k≤2，后者不能认定为正确高温吸引子或已验证的无穷远吸引子。
本次支持增加表示维度有帮助，但未完全恢复相变，也未证明二维是必要且充分条件。

![Two-dimensional flows](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91/full_report/R_one_vs_two_step.png)
![Temperature map](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91/full_report/coupling_by_scale.png)
![Direct and recurrent paths](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91/full_report/critical_trajectories.png)

## 加入均匀块保色：dim g=2、color weight=1

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91 \
  --epochs 3600 \
  --output results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
python -m scripts.report_probabilistic_chain_g2 \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91 \
  --critical-coupling 1.398
python -m scripts.audit_blume_capel_vector \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91
python -m scripts.report_mi_training \
  --run results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91 \
  --baseline results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_3600_seed91
```

唯一配置变化为kernel_sign_weight=1；随机初始化、seed91、同数据、12055参数，3600步。
复用categorical_kernel_preserve_loss，对q个均匀块计算平均交叉熵，无构型枚举选核。
C独立目标为−2I+2L_variance+L_color，Q/R梯度目标不变。
驱动新增--color-weight，默认0，不改变历史默认配方。新增梯度路由测试，全套252项测试通过。
训练1383.67秒（23.1分钟）。均匀−1/0/+1块保持类别概率分别约
.9999999963/.9999999955/.9999999710，保色NLL=1.2422e-8。

测试一步NRMSE=[4.06%,4.13%]，两步rollout=5.45%，优于无保色[5.60%,10.51%]及9.71%。
加权闭环=.021844（无保色.080111），几何=.147511（无保色.224493），
经验MI=1.34194 bits（无保色1.33047附近），方差加权=.000680。

同一有限查根协议找到一步R鞍点(−.64766,.46563)，特征值[1.49219,.80223]；
稳定点(1.05865,−.93534)，特征值.78641±.02569i。
R²只找到这两个R根，无额外二周期；这是有界数值搜索，不是全局排除所有周期的证明。
鞍点对微观温度曲线最近投影T=1.50893、距离.02172，较参考1.398高7.93%，
比无保色鞍型二周期的投影1.46397更偏；不能把消除周期等同于提高临界温度精度。
高温长递推仍离开训练支持，未验证完整高温归宿。

所有尺度合并后的弱主成分方差比例由无保色38.9%降至.00808%，编码近共线。
这提示先前第二维的占用很大程度与奇偶尺度标签差异有关，不足以证明物理上必须二维闭合。

![Color-preserving flows](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91/full_report/R_one_vs_two_step.png)
![Color-preserving temperature map](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91/full_report/coupling_by_scale.png)
![Loss comparison](../results/blume_capel2d_delta1_g2_random_mi_variance2ratio1_color1_3600_seed91/report/loss_comparison.png)
