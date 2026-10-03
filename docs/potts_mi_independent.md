# Potts：数据选核、精确经验 MI 与独立 Q/C/R 梯度

## 设置

沿用 [Ising 修正实验](ising1d_mi_repair.md) 的目标及梯度路由，测试二维零场
q=3 Potts，dim_g=1，seed91。Q/C/R 同时更新，但 C 只接受
`-2 I_empirical(H;E) + 2 variance_floor_ensemble` 的梯度，Q 只接受表示损失，
R 只接受一步及多步闭环损失。没有耦合监督、保色项或硬 S3 对称性。
Q 的子集一致性、粗尺度一致性、relative_local 几何项权重为2、5、10；
R 的一步和多步权重均为5；C 方差下限比例0.25。

使用原 Potts 缓存：K 在[0.4,1.6]均匀64点，每点3条采样链，每系综128构型；
129训练/63留出系综。L=16，两个粗粒化步骤16→8→4，C为不重叠2×2块。
hidden60，11873参数，batch32系综×64构型，Adam lr=.001，2400更新。
理论 Kc=log(1+sqrt(3))≈1.00505254，仅用于训练后诊断。

## 与二值 Ising 初始化的区别

仍直接计数完整环境壳层（20自旋，buffer=1）的经验互信息，不使用InfoNCE。
三值2×2块有81种输入，确定性核共有3^81种，无法沿用Ising的穷举。
因此使用8个随机soft-table起点，各优化600步、每50步取硬候选，按训练MI选优，
再拟合到可训练C。该初始化**没有全局最优保证**，不能称为与Ising同等强度的搜索。

事后审计发现所选候选的训练MI=1.476212 bit，低于正decimation参考的约1.5003 bit，
也低于多数票（平票取首次出现颜色）的1.548232 bit。这些参考核没有参与本次训练或选优。
测试MI分别为1.473219、约1.4981、1.546886 bit。因此不能把本次结果单独归因于MI目标。

另有明显有限样本偏差：训练/测试环境类别约82.8%/83.6%只出现一次；
实际打乱V与E配对后，所选核仍有约0.8446/0.8616 bit经验MI。
精确计算经验MI并不意味着精确估计总体MI。详见原运行目录的
`mi_initialization_audit.json`；日志中的解析独立参考0不等于这个实际置乱对照。

## 复现与断点续训

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/potts2d_g1_neutral0_2400_seed91 \
  --output results/potts2d_g1_MIinit_independentC_variance2_2400_seed91 \
  --estimator histogram --variance-weight 2 --variance-per-ensemble \
  --mi-initialize --independent-c
```

本次原运行在500步后停止，从400步检查点恢复到下列目录。
优化仅省去独立梯度模式下不使用的Q→C反向计算；权重、优化器及所有训练RNG恢复。
恢复后500步全部日志与原运行完全相同。完整测试246项通过。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/potts2d_g1_neutral0_2400_seed91 \
  --output results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast \
  --estimator histogram --variance-weight 2 --variance-per-ensemble \
  --mi-initialize --independent-c \
  --resume-checkpoint results/potts2d_g1_MIinit_independentC_variance2_2400_seed91/training_step_400.pt
python -m scripts.report_scalar_joint --run results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast
python -m scripts.report_potts_scalar --run results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast
python -m scripts.report_mi_training --run results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast --baseline results/potts2d_g1_neutral0_2400_seed91
```

续训summary中的elapsed_seconds仅计恢复后的2000更新，不包含原运行、初始化和后处理。
这是一个种子的迁移测试，不修改全项目默认设置，也不按相图挑选最优检查点。

## 最终结果：未恢复正确临界流

2400步完成。宽范围查根得到稳定—不稳定—稳定结构，但不能据此判定成功：

| g不动点 | R′ | 类型 |
|---|---:|---|
| -0.490563 | 0.756490 | 稳定，远离数据支持 |
| 0.423719 | 1.252413 | 不稳定，低于观测范围 |
| 2.339578 | 0.163842 | 稳定，略高于观测范围 |

实际R输入范围[0.511769,2.318308]，微观Q在理论Kc处为g≈1.159166，
显著不等于不稳定根0.423719。该根没有微观K网格内的反插值候选。
在观测范围内R(g)>g，左右两相没有被不稳定点正确分开。
800步时不稳定根g≈0.845583已有K≈1.52–1.57的多重插值候选，
也非正确临界位置；不能称为早期成功、仅后期过拟合。

最终加权测试QCR（不含MI）=0.081802：微观一致性0.028355、粗尺度一致性0.034707、
几何0.002969、C方差下限0、一步闭环0.007471、多步闭环0.008300。
虽然这些损失不大且训练平稳，仍未保证正确临界行为。
C未塌缩成常数，但IID均匀输入的输出概率为(0.259259,0.333333,0.407407)，
存在颜色偏置、并不满足置换等变；不能仅凭这一点证明物理失败由它导致。

现有证据优先指向需要检查的两环：选核优化没有找到更高MI候选；
大环境字母表下经验MI偏差明显。此次不能验证该方案已通用于Potts，
也不足以否定RSMI或QCR本身。

![R and K mapping](../results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast/full_report/R_and_K_mapping.png)
![Loss curves](../results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast/report/loss_comparison.png)
