# 二维经典 Ising：数据选核 + Q/C/R 独立梯度

## 设置

迁移一维修正方案，单 seed=91、dim_g=2、2400 次更新。模型10373个参数，hidden60。
概率 C 为可训练的非重叠2×2块；Q 使用3×3局部窗口。尺度16×16→8×8→4×4。
数据沿用 `data_cache/ising2d_Kfocused_Buniform0p01_L16_n64_seed91/ensembles.pt`：
K=0.20:0.03:0.68，B=−0.010:0.001:0.010，每系综64构型，训练474、测试240个系综。
该非零场数据不做全局翻转增强，无Z2网络约束、保符号项或耦合标签监督。
Adam学习率0.001，system_batch32、sample_batch64，threads2。

与一维方案一致：

- 用训练数据的精确经验 MI 选择确定性核，拟合现有C网络1200个初始化小步；随后C全部参数仍可训练。
- C只接收 `−2 I_empirical + 2 L_variance_ensemble`，variance ratio=0.25。
- Q只接收子集一致性与relative_local分布几何损失。
- R只接收 `5 L_one_step + 5 L_rollout`，不反馈Q/C。
- 不使用弱方向加权、协方差约束、物理相标签。

MI依然只在原始尺度计算。buffer厚度1，V为4个自旋，E为外侧6×6方框的20个边界自旋；
没有把buffer纳入V/E，也没有缩减环境维数。
每个系综使用全部64个构型和256个平移位置，重叠区域并非独立样本。

## 二维精确 MI 与选核

4个二值输入共有16种局部构型，全部确定性二值输出核有65536个。
利用输出整体换标签不改变MI，只需评价32768个代表；这不限制随后可训练C的函数空间。
按照训练数据的经验MI全局排序，码64704（及其输出补码）最优。
其规则为四个格点中第2、3、4个自旋的多数票，第1个自旋被忽略。
这不是手工指定decimation或majority；是枚举排序的输出。
之后拟合概率网络，最大初始化概率误差5.32e-5。
训练最终仍接近该三自旋多数票，全部16项输出与确定性规则最大差约1.18e-8。
翻转等变误差约1.45e-9是最终测量值，不是施加的硬约束。

为避免16×2^20稠密表，用
`I(H;E)=H(H)−sum_e p(e)H(H|E=e)`，仅合并具有完全相同经验P(V|E)的环境类别。
每个系综保留独立权重与边缘分布。训练统计压缩为32621个不同条件分布，测试为18490个。
这一压缩不改变经验MI或梯度；代码测试与直接计数逐项比较。

重要限制：平均约74.2%（训练）/74.6%（测试）的已观测环境类别只出现一次。
因此精确计算经验MI不等于获得可靠的总体MI估计，存在明显上偏风险；训练与测试MI也不能直接
与旧InfoNCE数值比较。训练MI约0.640592bit，测试约0.648483bit，不据此认定物理正确。

## 最终2400步结果

训练循环耗时243.14秒（不含统计预处理、枚举初始化与后处理）。
中心根 g=(0.252052,0.243957)，DR本征值(1.601468,1.126278)，为两个方向均不稳定的排斥子，非鞍点。
微观Q(K,B)网格的分片线性反插值得到唯一候选：K≈0.427529、B≈0.0001474。
理论零场Kc≈0.440687，K相对偏差约2.99%；B接近0。
此反插值是坐标定位诊断，不是有效耦合标签，也不是精确临界指数验证。
理论临界点映射约g=(0.004583,0.161533)，与学到的根仍有可见偏移。

另外找到两个鞍点和一个稳定点，均无微观K/B网格内部的反插值候选，不赋予确定物理解读。
边界外箭头是网络外推，查根为有界数值搜索，不保证穷尽。

微观g协方差本征值为0.20161、1.22818，弱/强比约0.164，未塌缩为近一维直线。
留出系综的一步NRMSE=3.11%，两步rollout NRMSE=5.31%。

| 加权测试项 | 当前 | 原InfoNCE QC/R分梯度 |
|---|---:|---:|
| 微观子集一致性 | 0.210076 | 0.214833 |
| 粗尺度子集一致性 | 0.097467 | 0.102078 |
| 分布几何 | 0.385313 | 0.392777 |
| C防塌缩 | 0.00000085 | 0 |
| 一步闭环 | 0.017718 | 0.022103 |
| 多步闭环 | 0.022918 | 0.024713 |
| QCR合计（不含MI） | 0.733493 | 0.756504 |

闭环合计0.040637，对照0.046816。表示变化意味着不是同一个坐标系中的物理误差比较。
分布几何仍是主要损失，尚未宣称完整相图或临界指数已精确恢复。

![R flow](../results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr/report/R_flow_and_center.png)
![K B mapping](../results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr/report/g_KB_maps.png)
![Loss](../results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr/report/loss_comparison.png)

## 复现

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising2d_g2_rsmiSeparate_w2_2400_seed91/qr \
  --output results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr \
  --estimator histogram --variance-weight 2 --variance-per-ensemble \
  --mi-initialize --independent-c
python -m scripts.report_rsmi_experiment \
  --run results/ising2d_g2_MIinit_independentC_variance2_2400_seed91
python -m scripts.report_mi_training \
  --run results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr \
  --baseline results/ising2d_g2_rsmiSeparate_w2_2400_seed91/qr
```

输出目录须不存在；没有覆盖旧实验。全套242项测试通过。新统计实现位于
`recurrent_rg/empirical_mi.py`，保持一维原有路径不变，未修改全项目默认训练方案。
