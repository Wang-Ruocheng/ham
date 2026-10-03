# Potts：随机概率 C、MI 与完整方差下限

## 设置和验收

将 [随机梯度 Ising 配方](ising_random_gradient_mi.md) 原样迁移到二维零场三态Potts。
Q/C/R从seed91随机初始化，无选核、无核表拟合预训练、无旧权重恢复。
源实验目录仅提供配置和缓存路径；manifest中的data_driven_initialization=null。
单seed，固定2400更新，按最终检查点评价，不按相变位置挑选早停点。

- C：−2×逐系综经验互信息 + 2×多尺度逐系综方差下限，ratio=1。
- Q：微观子集一致性权重2、粗尺度一致性5、relative_local几何10。
- R：一步闭环与多步闭环权重均为5。
- 三模块同步更新、各接收各自目标梯度；不向Q/C反传R损失。
- 不使用耦合标签、保色项、硬S3对称性或颜色配对增强。

三值方差使用one-hot协方差的迹v=1−||p||²。ratio=1下正则为
平均[||p_out||−||stopgrad(p_in)||]_+²，逐系综逐尺度计算。
它是软统计假设，不是指定正确核或普适RG恒等式。
新增单元测试验证它排斥偏置输出、对颜色重标号不变，并允许常数输入保持常数。

缓存保持原样：L16→8→4，K∈[.4,1.6]均匀64点，每点3条采样链，
每系综128构型，129训练/63留出系综。dim_g=1、hidden60、11873参数、
C为不重叠2×2可训练categorical概率核；batch32×64、Adam lr=.001。
MI由完整20自旋环境壳层、buffer=1的经验计数可微计算，无critic。
大环境字母表的有限样本上偏仍存在；此处“精确”仅指经验分布的有限求和。

验收不只数R的根：同时核对理论Kc=log(1+sqrt(3))≈1.00505254的Q坐标、
不稳定根的K反插值、观测支持范围、Q(C^k D)及R闭环。
理论耦合只在训练后诊断中使用。

## 复现

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast \
  --output results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_scalar_joint \
  --run results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91
python -m scripts.report_potts_scalar \
  --run results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91
python -m scripts.report_mi_training \
  --run results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91 \
  --baseline results/potts2d_g1_MIinit_independentC_variance2_2400_seed91_fast
```

与上一轮Potts对比同时改变了初始化方式和方差下限比例，不是单因素消融。
保留原实验目录，不修改全项目默认值。

## 800步中间检查（未用于选模型）

不稳定根g≈2.078525，R′≈1.123617，在观测R输入区间[1.106673,2.965516]内。
微观K映射给出唯一插值K≈0.974770，理论Kc的编码g≈1.849158。
临界位置已比旧MI选核实验接近，但局部搜索只有这一个根；
更宽范围中的其他根位于外推区，不应据此宣称两相完整恢复。
保留full_report_800，最终仍使用预定2400步。

## 最终2400步结果

训练循环848.42秒（约14.1分钟，不含统计预处理及绘图）。本次定性分流结构和
临界位置明显改善，但区间外吸引子仍不能作完整物理验证。

| 根 | g | R′ | 数据支持 |
|---|---:|---:|---|
| F1稳定 | .782294 | .299788 | 观测R输入区间内 |
| F2不稳定 | 1.583052 | 1.393651 | 观测R输入区间内 |
| F3稳定 | 3.149505 | .814645 | 区间外外推 |

观测R输入区间[.758931,2.610134]。F2在微观K–g均值曲线上有唯一插值
K≈.993806，相对理论Kc≈1.005053偏低1.119%。Q(Kc)≈1.498549，
与F2仍有偏移；不是精确定位或临界指数验证。
K–g整体递减；低K侧向高g移动，高K侧向低g吸引子移动。
高g稳定点超出训练支持，不能反插值成一个已验证的物理K。
F1虽在粗尺度R输入范围内，也没有微观K网格内反插值，不能将其直接赋予有限K标签。

800步的临界插值为.974770，2400步改善到.993806，没有重复上次逐渐漂离临界区的现象。
有限缓存诊断：两个尺度的一步NRMSE为4.37%、5.18%，两步rollout为7.91%。
测试QCR（不含MI）=.098109，分项为：微观一致性.031074、粗尺度一致性.035164、
几何.002963、C方差.0000436、一步闭环.009959、多步闭环.018904。
旧MI初始化方案的QCR为.081802，虽然更低，临界流却不正确；不能只按loss挑选物理解。
测试经验MI=1.538616 bit，高于旧方案1.473218，但有限样本偏差限制仍在。

最终均匀块0、1、2分别以超过.999999的概率映射到1、2、0，未合并三种有序颜色。
这只说明均匀块的循环重标号，不表示任意混合块都等价于一个标准核加重标号。
IID均匀输入的输出比例(.259262,.333331,.407407)，依然存在颜色偏置。
微观Q在同一批测试构型的全局颜色重标号下，最大RMSE≈.002379，
相较原编码SD≈.706365很小，但并非严格颜色不变，也未验证任意深尺度的不变性。

全套248项测试通过。单seed结果不构成初始化鲁棒性或多模型普适性证明。

![R and K mapping](../results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91/full_report/R_and_K_mapping.png)
![Loss comparison](../results/potts2d_g1_random_independentC_variance2ratio1_2400_seed91/report/loss_comparison.png)
