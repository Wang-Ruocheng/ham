# 精确经验 MI + 多尺度 variance_floor

## 对照设置

沿用 `ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91` 的数据和配置，从相同 seed=91 重新初始化训练。
一维经典 Ising，dim_g=1，概率 C 随机初始化，QC/R 分梯度，2400 次更新。
MI 权重 2；新增 variance_floor 权重 2、ratio=0.25。
没有保符号、Z2 结构约束、neutral 或 distinct 项。C 可以收到 QC 与 MI/方差项梯度，但不能收到 R 闭环梯度。
训练耗时 194.27 秒；237 项测试通过。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr \
  --output results/ising1d_g1_exactMI_variance2_2400_seed91/qr \
  --estimator histogram --variance-weight 2 --variance-ratio .25
python -m scripts.report_rsmi_experiment \
  --run results/ising1d_g1_exactMI_variance2_2400_seed91
python -m scripts.report_exact_mi_comparison \
  --run results/ising1d_g1_exactMI_variance2_2400_seed91/qr \
  --baseline results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr
python -m scripts.audit_mi_variance
```

## 结果：没有改善

相对 MI-only 最终模型，state_dict 最大逐元素差仅 2.89e-8。
所有训练/测试日志记录点的 C variance_floor loss 都为零。
仍学到近乎 AND，测试经验 MI=0.3839748076 bit。
搜索区间内仅检测到稳定根 g=-2.862537、R'=0.133824，略超出训练输入范围；未恢复不稳定端点。
共同 QCR loss、闭环曲线与基线重合；最终测试 QCR loss 约 0.334623。

## 为什么未触发？

现有实现先对整个 batch（包含不同系综）求平均自旋 m，再计算 1-m²。
它不是对每个系综分别检查。
在全部缓存训练系综的最终模型诊断中，相邻尺度总体方差比分别为：

`0.9720, 0.8846, 0.7769, 0.6736, 0.5462`

全部高于阈值 0.25。测试集对应为：

`0.9706, 0.8794, 0.7676, 0.6608, 0.5268`

但若只作诊断、分别检查每个系综，第 3、4、5 次粗粒化中低于同一阈值的测试系综占比约为
20.6%、39.7%、46.0%（要求输入方差非零）。这些检查使用全部缓存、配对翻转和固定随机种子，
与实际训练 minibatch 不同，不作为新增训练结果。

结论：该组合没有有效增加约束，不能归因于 MI 战胜了一个很大的 C penalty。
当前 batch 级聚合掩盖了部分系综的退化，同时 0.25 单步阈值允许显著累积收缩。
下一步可检验逐系综计算后平均的同类 loss；尚未实现或训练，不保证修复 AND，亦需警惕惩罚真实的物理方差收缩。

![Loss](../results/ising1d_g1_exactMI_variance2_2400_seed91/qr/report/loss_comparison.png)
![R(g)](../results/ising1d_g1_exactMI_variance2_2400_seed91/qr/report/R_vs_g.png)
![J versus g](../results/ising1d_g1_exactMI_variance2_2400_seed91/qr/report/coupling_vs_g.png)
