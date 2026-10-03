# Ising 保色对照：经典1D dim_g=1/2 与2D dim_g=2

单种子seed91，从各自成功的随机C、variance ratio=1配方仅读配置，重新随机初始化。
2400步、同数据指纹、同模型参数量，唯一配置变化kernel_sign_weight由0变为1。
Q/C/R独立梯度同步更新；C目标为−2I+2L_variance+L_color。
二值保色项为均匀全负/全正块输出原符号的平均Bernoulli NLL，直接计算，不采样。
没有正decimation初始化、选核预训练、固定C或中途挑选检查点。

## 命令

```bash
for case in ising1d_g1 ising1d_g2 ising2d_g2; do
  python -m experiments.rsmi_separate_qcr \
    --source results/${case}_random_independentC_variance2ratio1_2400_seed91/qr \
    --output results/${case}_random_independentC_variance2ratio1_color1_2400_seed91/qr \
    --estimator histogram --variance-weight 2 --variance-ratio 1 \
    --variance-per-ensemble --independent-c --color-weight 1
done
```

分别使用scripts.report_scalar_joint + scripts.report_ising1d_endpoints、
scripts.report_probabilistic_chain_g2 + scripts.report_ising1d_vector_endpoints，
以及scripts.report_rsmi_experiment（--run为qr的父目录）做报告。
scripts.report_mi_training的--baseline指向各自无保色qr目录。
三个实际训练并行完成，循环耗时分别285.85、282.56、320.77秒，不能相加作为墙钟总耗时。

## 1D经典Ising，dim_g=1

9711参数；L128，J∈[.01,2]，5次粗粒化。学到近似正decimation，
P(+|--,+-,-+,++)=(3.44e-6,9.13e-5,.9998605,.9999965)，选择了与基线不同的块内自旋。
不稳定根g=.0147504，R′=1.54102；稳定根g=1.9514202，R′=.0848478。
population-pooled Q的J∞端=.0852333，J0端=1.9360391。
有序/无序端根误差除以端点间距为3.81%/.831%（基线2.21%/1.00%）。
300个J诊断点流向无序端的方向全部正确；相对精确decimation参考的一步RMSE/端点间距=1.071%。
测试首步NRMSE=3.78%，第五步rollout=58.10%（raw RMSE=.09945）；高尺度有限系综误差仍大。
R²在[-1,3]的6001点查根只得到对应两个R根，未发现额外二周期；非全局证明。

![1D scalar](../results/ising1d_g1_random_independentC_variance2ratio1_color1_2400_seed91/qr/endpoint_report/physical_validation.png)

---

## 1D经典Ising，dim_g=2

9893参数，同一维数据。C同样学得正decimation；保符号概率均大于.9999965。
有序端鞍点g=(.0850381,−.2072784)，DR=(1.2271323,.3330495)；
无序端吸引子g=(1.8013978,−1.4691609)，DR=(.3212947,−.0503880)。
负的稳定乘子不等于二周期，R²查根没有额外周期点。
表示近似一维，population编码弱主成分占比2.64e-6，鞍点不稳定方向与端点轴余弦=.99921。
有序/无序端位置误差为11.84%/.379%（基线5.40%/1.22%），有序端在训练支持外且定位退化。
300个J的位移与精确参考内积全部为正，但参考一步RMSE/端点跨度由1.13%增至2.39%。
测试首步NRMSE=7.65%（原4.04%），第五步rollout=80.15%（原60.08%），raw RMSE=.09956。
保留定性结构，但不能称为精度无损通过。

![1D vector](../results/ising1d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/full_report/R_one_vs_two_step.png)
![1D vector coupling map](../results/ising1d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/full_report/coupling_by_scale.png)

---

## 2D经典Ising，dim_g=2，K/B扫描

10373参数；L16，K=.20:.03:.68，B=−.01:.001:.01，2次粗粒化。
C近似对输入索引0/1/2做正多数票，忽略索引3，最大规则概率误差约1.61e-5；
均匀全负/全正块保符号概率超过.99999999。
中心排斥子g=(−.1719657,.0535591)，DR=(1.5664882,1.1241524)，两个正乘子均>1。
原负多数票核中心乘子为(1.569596,−1.126980)，保色消除了该局部方向反转。
微观KB网格内插值为K=.42656498、B=−.00049268，K偏低约3.20%（原3.31%）。
另外两个鞍点不在微观网格映射内，不赋予确定物理解读。
对R²独立使用17×17网格及全部微观编码作起点，仅找到上述三个R固定点，无额外二周期；查根有界。
测试一步NRMSE=[3.74%,5.21%]（原[3.99%,5.69%]），两步rollout=6.62%（原6.30%），大体相当。

![2D flow](../results/ising2d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/report/R_flow_and_center.png)
![2D coupling maps](../results/ising2d_g2_random_independentC_variance2ratio1_color1_2400_seed91/qr/report/g_KB_maps.png)

## 结论

三个问题均保留目标定性不动点结构，保色目标满足，所查范围未发现额外二周期。
但1D dim_g=2的有序端定位和深尺度闭环有明显退化；单seed不证明普适性或初始化鲁棒性。
本次是显式weight=1对照，不静默修改全项目默认设置。
