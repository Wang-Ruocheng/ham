# 二维六态时钟模型：单耦合、两次 BKT 转变

模型：H/J = -sum_<ij> cos(theta_i-theta_j)，theta_i=2*pi*a_i/6，
a_i=0,...,5；唯一无量纲耦合 K=J/T。不是六态 Potts。
热力学极限中低温有序、中间准长程有序、高温无序；转变温度粗略为
T/J~.70和.90，对应按K递增的边界约1.11和1.43。它们是文献数值参考，
不是精确临界值。中间相是临界相，不能预设一个孤立吸引子或“五根”结构。

参考：[六态时钟 CTMRG 研究](https://doi.org/10.1103/PhysRevE.101.062111)、
[clock/XY embedded-cluster sampling](https://arxiv.org/abs/cond-mat/0202161)。

## 数据生成与检查

`recurrent_rg/data/clock.py` 实现保持离散角集合的反射多团簇更新。
每步选择m，投影p(a)=sin(pi*(2a-m)/q)，同号邻点以
1-exp(-2*K*p_i*p_j)连边，每簇以1/2概率执行a -> (m-a) mod q。
周期右/下边计数，L=2含平行边，与精确枚举一致。

首轮L16，64个均匀K∈[.6,2]，每点3个独立种子系综，每系综128构型；
burn1024、spacing32个多团簇sweep。整个K格点划分训练/测试，129/63系综。
五处冷热启动探针各1024样本、burn2048；主要观测量lag1绝对值均<.08。
中间探针K1.27的locking冷热均值约差2.3个合并batch SE，不视为严格独立
或充分混合证明。批均值误差与所有探针记录在sampling.json。
五项单元测试包括K=.4,1.2,1.8的2x2精确枚举、零/低温极限、可复现性。

```bash
python -m pytest -q tests/test_clock.py
python -m scripts.generate_clock2d --output data_cache/clock2d_q6_L16_n128_seed91
```

## 首轮训练

dim_g=1，hidden60，seed91，1200更新，L16->8->4，batch32x64。
随机初始化、可训练六类概率C，无选核或预训练；Q/C/R同步独立梯度。
C目标=-2*经验MI+2*逐系综方差下限(ratio1)+1*均匀块保色。
Q使用relative_local几何及子集一致性；R使用一步和所有可用尺度rollout。
不使用耦合监督、角度特征或硬Z6等变性。128构型缓存不是128独立样本保证。

```bash
python -m experiments.rsmi_separate_qcr \
  --config configs/clock2d_q6_g1_random_color1_seed91.json \
  --output results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
python -m scripts.report_scalar_joint \
  --run results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91
python -m scripts.report_clock_scalar \
  --run results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91
python -m scripts.report_mi_training \
  --run results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91
```

状态：1200步已完成，训练循环1053.29秒（17.55分钟，不含采样/统计预处理/报告）。
报告只检验clock的12个二面体对称变换，不将任意颜色置换当成物理对称。
必须实测同系综打乱V/E的MI基线；大环境经验MI可能严重上偏。
L16、两步RG仅为管线探索，不足以确认BKT转变、临界线或精确临界位置。

## 最终结果：尚未恢复两次转变

Q/C/R参数分别5029/5526/3841，总计14396。最终不含MI的训练/测试QCR
loss=.0275628/.0239403。含负MI的测试总loss=-3.44935，不作物理准确度指标。

在报告的有界范围仅找到一个不稳定根g=-1.949648，R'=1.143611，位于
观测R输入区间[-2.29220,-1.29166]内。微观K-g均值曲线唯一反插值
K=1.026146。不能把它认定为两次BKT中任一转变的准确识别；第二边界和
中间临界固定线尚未恢复。无“五孤立根”验收要求。

测试一步NRMSE为10.21%/8.86%，两步rollout为12.18%，均以训练目标SD
归一化。报告另含五个C采样种子的重复诊断，非五个独立训练种子。
六种均匀块保色概率均>.9999993；Q对12个二面体变换的微观编码最大RMSE
约.00769，不是严格对称性保证。

经验MI上偏非常明显（bit；一次同系综打乱基线）：

| K | 原配对MI | 打乱配对MI |
|---|---:|---:|
| .6222 | 2.51774 | 2.51774 |
| 1.0889 | 2.55018 | 2.54979 |
| 1.2889 | 2.55962 | 2.55128 |
| 1.4222 | 2.52603 | 2.47131 |
| 1.9556 | 2.44630 | .89481 |

约95%观测环境类别仅出现一次。高MI不能证明保留了长程信息；本实验没有
通过消融证明它是失败的唯一原因，有限尺寸和表示/动力学拟合亦未排除。

![R和K映射](../results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91/full_report/R_and_K_mapping.png)
![Loss](../results/clock2d_q6_g1_random_mi_variance2ratio1_color1_1200_seed91/report/loss_comparison.png)
