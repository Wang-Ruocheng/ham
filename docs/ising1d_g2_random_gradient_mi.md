# 一维经典 Ising：随机概率 C，dim_g=2

沿用[随机梯度配方](ising_random_gradient_mi.md)，仅将dim_g由1改为2。
Q/C/R全部随机初始化，无选核、预训练、固定C或耦合监督；独立梯度同步更新。
C目标为−2经验MI+2逐系综方差下限，ratio=1；Q为原relative_local几何与子集一致性；
R为权重均为5的一步与全尺度多步闭环。
同seed91、2400更新、hidden60、L128、5次粗粒化、batch32×64、Adam .001。
参数量因升维由9711增至9893（Q2030、C3901、R3962），训练循环162.99秒。
这是经典零场Ising，不是TFIM；没有有限温度临界点，不要求三个不动点。

## 结果

| 根 | g | DR本征值 | 类型 |
|---|---|---|---|
| F1 | (.286467,−.371107) | 1.549397，.319319 | 鞍点 |
| F2 | (1.894788,−1.586989) | .360430，.018263 | 吸引子 |

测试微观g协方差本征值约(9.14e−7,.303829)，弱方向占总方差3.01e−6，
因此本次确实学得接近直线的一维表示，并非强行占满二维。
F1不稳定特征向量与物理端点连线的绝对余弦=.999853，支持它沿物理方向排斥；
另一方向稳定。R²查根未发现额外二周期，但查根有界、不保证穷尽。

C最终P(+|--,+-,-+,++)≈(2.05e−6,.9999182,4.08e−5,.9999982)，接近正decimation。
用有限环L128的精确分布计算population-pooled Q：
Q(J=0)=(1.876145,−1.573421)，Q(J→∞)=(.365439,−.435859)。
F2相对无序端、F1相对有序端的距离除以物理端点间距，分别为1.219%、5.400%。
有序端根在训练输入支持之外，不把它的最近网格J=2解释为临界耦合。

J∈[.01,2]的300个诊断点中，R位移与精确decimation参考位移的内积全部为正。
向量一步RMSE/端点间距=1.134%，与dim_g=1的1.023%接近；
有序端位置偏差却由2.212%增至5.400%，因此并非所有精度都保持。
未训练的J=∞重复R仍有明显漂移，不声称全尺度精确RG。

有限样本测试首步NRMSE=4.04%，第5步rollout NRMSE=60.08%，raw RMSE=.07577；
深尺度表示方差小、样本及有限尺寸误差仍明显，不能用总体分布诊断掩盖这一限制。
测试加权QCR（不含MI）=.156205；微观一致性.023021、粗尺度一致性.090833、
几何.007769、C方差1.03e−7、一步闭环.015897、多步闭环.018685。
相比dim_g=1闭环合计.035103，当前.034581；表示不同，不是严格同坐标误差比较。

![Flow](../results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr/full_report/R_one_vs_two_step.png)
![Physical audit](../results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr/endpoint_report/physical_validation.png)
![Coupling curves](../results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr/full_report/coupling_by_scale.png)
![Loss](../results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr/report/loss_comparison.png)

## 复现

新增--theory-dim只覆盖配置维度，默认不传时旧命令行为不变。
source仅读配置与数据路径，不读模型权重。248项测试通过。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr \
  --output results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr \
  --theory-dim 2 --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_probabilistic_chain_g2 \
  --run results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr
python -m scripts.report_ising1d_vector_endpoints \
  --run results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr
python -m scripts.report_mi_training \
  --run results/ising1d_g2_random_independentC_variance2ratio1_2400_seed91/qr \
  --baseline results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr
```
