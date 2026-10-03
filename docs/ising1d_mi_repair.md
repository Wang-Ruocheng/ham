# 1D Ising：MI 核与 Q/R 梯度隔离修正

## 结论与边界

最终恢复了一维经典零场 Ising 的定性结构：无序端吸引、有序端排斥，测试 J∈[0.01,2] 均朝无序端流动。
没有有限温度临界点；不稳定物理端点是 J→∞，不把外推根反插值成一个有限临界 J。
本次是单 seed 的小块验证，不构成二维或任意块大小的通用性结论。

最终方案仍有误差，尤其临界端外推：Q(J=∞) 不是 R 的精确不动点，重复 R 后会偏离。
因此结果是定性结构与训练区间流向恢复，而不是全尺度、全耦合精确 RG 或可靠临界指数。
图中的 J=4 和 ∞ 都不在训练耦合范围内；它们是审计，不是训练样本或模型选择目标。

## 三个配对实验

均为相同缓存、seed=91、dim_g=1、hidden=60、9711 个 QCR 参数、2400 次更新。
无 J 标签监督，无 Z2 网络约束，无保符号项；已有配对翻转数据增强保持不变。
MI 是每个系综经验分布的精确有限求和，仅原始尺度；buffer=1，V=(s0,s1)，E=(s−2,s3)。
MI 权重2，逐系综 variance_floor 权重2、ratio=0.25。

| 实验 | 最终核 | 测试加权 R 闭环 loss | 搜索所得根 |
|---|---|---:|---|
| 仅将方差项改为逐系综 | 近 AND | 0.271943 | 一个稳定根 |
| 加训练数据 MI 最优核初始化，QC 仍共同优化 | 近 OR | 6.001720 | 一个稳定根 |
| MI 初始化 + C 独立目标梯度 | 近正 decimation | 0.030590 | 一个不稳定根、一个稳定根 |

原 MI-only 基线的 R loss 为0.272719，最终方案下降约88.8%。
三组训练循环耗时约253、261、237秒（并发运行；不含初始化拟合和后处理）。
不隐藏前两个失败结果；对应路径见下方。

## 修正内容

1. 新增可选 `variance_floor_ensemble`，每个系综独立计算均值/方差、hinge 后平均；旧 batch 模式保留以复现实验。
2. 用训练缓存统计的经验 MI 枚举16个确定性二输入二输出核，不读取耦合标签。
   最优码为3、5、10、12并列。输出标签按两个均匀块顺序确定符号约定，再按码序打破并列，选中10。
   这是标签约定，不是在学习空间中强制保色或Z2，也不是预先指定候选赢家。
3. 对现有 C 的 MLP 拟合该表1200小步（目标概率0.0001/0.9999，额外初始化成本），Q/R 不变。
   初始化最大概率误差2.25e-6。随后全部C参数仍可训练，不硬冻结。
4. 同步更新，但隔离三个模块的目标梯度：

   - Q：原有子集一致性与分布几何目标。
   - C：`−2 I_empirical + 2 L_variance_ensemble`，不接收 Q 表示目标或 R 闭环梯度。
   - R：`5 L_one_step + 5 L_rollout`，不反馈 Q/C。

这不是原来的端到端联合训练，也不是仅恢复旧 C loss。
新方法同时改变了初始化和梯度路由；仅逐系综方差项不足以纠正AND。
最终成功组的方差项在记录点仍为0，不能把改善归因于此项。
枚举规则数为2^(2^m)，只适用于当前小块；未解决一般大块优化。

## 最终物理审计（2400步，不挑早期检查点）

最终 P(+|--,+-,-+,++)：
`[1.23e-8, 0.999999959, 2.98e-8, 0.999999987]`。
测试经验 MI=0.385032136 bit，确定性 decimation=0.385032169 bit。

使用有限环 L128 的精确分布计算 population-pooled Q（不是有限样本 Q 的期望）：

| 物理端点 | Q 的坐标 | R 根 | R' | 根坐标偏差/端点间距 |
|---|---:|---:|---:|---:|
| J→∞ | −0.019814 | −0.087121 | 1.609301 | 3.585% |
| J=0 | 1.857385 | 1.845410 | −0.122352 | 0.638% |

不稳定根处于物理坐标范围之外，不声称在训练支持内精确定位了临界端。
在 J∈[0.01,2] 的300点，R(Q(J))全部朝 Q(J=0) 方向移动；
与精确 decimation 递推 J'=log(cosh(2J))/2 经同一个Q映射后的曲线相比，
RMSE/端点间距=0.504%。这些物理公式只用于训练后验证。
Population 一步闭环各尺度 RMSE 为0.00969、0.00896、0.00967、0.01085、0.01230。

有限缓存测试闭环不是所有尺度都改善：第一步 NRMSE=2.63%，
k5 rollout NRMSE=57.62%，raw RMSE=0.10310；原 MI-only 对应14.64%、21.43%、0.13189。
不同表示的目标SD不同，不能把NRMSE跨模型直接当共同物理误差。
深尺度有限样本噪声及有序端外推仍是限制。

![Physical validation](../results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr/endpoint_report/physical_validation.png)
![Loss comparison](../results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr/report/loss_comparison.png)

## 复现

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr \
  --output results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr \
  --estimator histogram --variance-weight 2 --variance-per-ensemble \
  --mi-initialize --independent-c
python -m scripts.report_ising1d_endpoints \
  --run results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr
python -m scripts.report_rsmi_experiment \
  --run results/ising1d_g1_MIinit_independentC_variance2_2400_seed91
python -m scripts.report_exact_mi_comparison \
  --run results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr \
  --baseline results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr
```

输出目录需不存在。第一个消融去掉最后两个选项，输出到
`results/ising1d_g1_exactMI_ensembleVariance2_2400_seed91/qr`；
第二个只去掉 `--independent-c`，输出到
`results/ising1d_g1_MIinit_ensembleVariance2_2400_seed91/qr`。

240项测试通过，包括逐系综掩盖问题、停止输入方差梯度、MI规则选择、初始化不改变Q/R、三模块梯度隔离。
未把该实验配方设为全项目默认值，旧实验行为保留。
