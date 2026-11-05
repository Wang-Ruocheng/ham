# Ashkin–Teller：Q预测一致性与局域环境MI的分离消融

两项独立实验，均使用原middle-dense缓存、seed91、总1200步，不把两者混成一个修改。

## A：固定C、dim_g=2，给Q加单向预测目标

从原固定C基线step400恢复Q/R、Adam和随机状态，继续至1200。
EMA目标Q从step400的Q复制，系数.99；新增权重在400至600步线性升至1。
L_Qpred = mean ||R_frozen(Q(D_k)) - stopgrad(Q_EMA(D_{k+1}))||²，坐标求和。
目标只向Q传梯度；R参数在这条路径冻结，但保留对其输入的导数。
R仍由原一步/rollout目标更新，C固定正decimation，Q原几何/子集目标保留。
新增目标不使用物理观测量或耦合标签；计算使用当前batch的全部64构型。

```bash
python -m experiments.at_predictive_q \
  --source results/ashkin_teller2d_g2_middle_dense_fixed_decimation_1200_seed91 \
  --output results/ashkin_teller2d_g2_middle_dense_fixed_predictQ1_1200_seed91
python -m scripts.compare_at_fixed_dimensions \
  --vector-run results/ashkin_teller2d_g2_middle_dense_fixed_predictQ1_1200_seed91
python -m scripts.audit_at_fixed \
  --run results/ashkin_teller2d_g2_middle_dense_fixed_predictQ1_1200_seed91
```

## B：可训练C、dim_g=1，完整环境改为四组二自旋环境

与原middle-dense随机C标量基线使用同配置、随机初始化和1200步预算。
仅改变MI的环境：上/下/左/右各两格，保持buffer=1；优化四个MI的算术平均，
不是它们联合环境的MI，可能丢失环境之间的协同信息。
Q/R仍独立梯度，C方差权重2、ratio1、保色1保持不变。
每方向环境最多4²=16类；原20格环境最多4²⁰类。

```bash
python -m experiments.rsmi_separate_qcr \
  --config configs/ashkin_teller2d_g1_middle_dense_seed91.json \
  --output results/ashkin_teller2d_g1_middle_dense_localPairMI_color1_1200_seed91 \
  --estimator histogram --environment local_pairs \
  --variance-weight 2 --variance-ratio 1 --variance-per-ensemble \
  --independent-c --color-weight 1
python -m scripts.report_scalar_joint \
  --run results/ashkin_teller2d_g1_middle_dense_localPairMI_color1_1200_seed91
python -m scripts.audit_at_middle_dense \
  --run results/ashkin_teller2d_g1_middle_dense_localPairMI_color1_1200_seed91
python -m scripts.report_at_tricks
```

## 验收约定

全部留出系综128构型评估，用各模型训练目标SD归一化；同时看直接运动相对误差、
表示分离、同系综打乱MI、C作用于IID乘积有序参考的表现，以及数据支持内的不动点。
两个实验的架构/C状态不同，仅各自与匹配基线比较，不横向比较总loss。
当前单seed、短尺度L16->8->4，不能将某一失败唯一归因于目标、容量或有限尺寸。

## 最终结果

两项训练均完成1200步。新增梯度/MI测试及全套265测试通过。
下表为留出系综的NRMSE，分母为各模型训练集对应尺度目标表示的SD。

| 实验 | 一步0→1 | 一步1→2 | 两步rollout |
|---|---:|---:|---:|
| 固定C、dim2基线 | 7.37% | 2.87% | 7.29% |
| 加Q预测目标 | 6.63% | 3.22% | 6.51% |
| 可训练C、dim1完整环境MI | 6.00% | 4.93% | 8.77% |
| 改局域二格环境MI | 6.67% | 5.42% | 9.50% |

### Q预测目标：小幅改善闭环，未恢复两个边界

最终原有测试总loss由0.049953降至0.043722，新增测试预测loss为0.001596。
两步NRMSE降低约11%，但第二尺度的一步误差略增。
表示仍近似一条线：小协方差本征值占比从0.000752%升至0.002668%。
低秩本身不代表错误，关键是本次修改未恢复两个相变边界。
仍只找到一个附近的鞍点，DR本征值约(1.316,0.899)；投影到微观Q(K)曲线
给出K≈0.320（基线≈0.333），投影残差0.00731，并非精确逆映射。
另一个稳定点远离数据支持区域，不作物理解读。

### 局域MI：估计偏差明显减少，但C和相图未改善

以留出K=0.202为例，完整环境真实/打乱MI分别1.936/1.840 bit；
局域环境分别0.642/0.000834 bit。二者不是同一个MI目标，不能直接比较真实MI大小；
打乱基线说明局域估计的虚假信息显著减小。
然而数据支持附近仍只有一个排斥子，插值K≈0.22465（原≈0.23185），未恢复两个边界。
这些数字不应当作与精确理论Kc的误差：此前约0.19和0.23仅是有限尺寸参考区间。

对独立格点参考分布P(--)=P(++)=1/2，即(m_sigma,m_tau,m_product)=(0,0,1)，
局域MI核使m_sigma依次变为0、-0.305、-0.693、-0.955、-0.999；
乘积磁化仍约1。它仍制造了单自旋偏置，不能维持这个乘积有序参考。
该参考不是有限K下的实际平衡系综；超过两步也超出训练尺度，作用是诊断核的迭代偏置。

结论：当前权重和单seed预算下，这两项改动都不足以解决AT的两次相变识别。
Q预测项仅小幅改善拟合；局域MI修正估计问题但不足以约束C保留不同类型的有序信息。
不能据此认定这两类方法在所有权重/容量/尺寸下均无效。默认配置未被替换。

![Q表示与标量R对照](../results/ashkin_teller_tricks_comparison/comparison.png)

数值报告：`results/ashkin_teller_tricks_comparison/audit.json`。
