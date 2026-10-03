# 二维八态 Potts：首轮单种子实验

H=−J sum_nn delta(s_i,s_j)，s∈{0,...,7}，K=J/T。
方格晶格无限体积转变点Kc=ln(1+sqrt(8))=1.34245404645，一级相变。
参考：https://www.sciencedirect.com/science/article/abs/pii/S001046551200402X
不把连续相变的临界指数验收直接搬到该问题。

## 数据与采样审计

L16、64个K∈[.8,1.8]，以精确Kc替换最接近的网格点，3个独立副本/K，128构型/系综。
按耦合分组留出，129训练/63测试系综，训练不输入K或能量。
Swendsen-Wang采样，burn4096，spacing256，seed91+1000*K_index+replica。
另外5个K各做有序/随机初态审计，burn8192、每条2048构型。

spacing32预检数据保留在potts2d_q8_L16_n128_seed91，未用于训练：Kc处能量lag1≈.77。
训练实际使用spacing256数据，Kc处lag1≈.237/.262，仍有自相关，不宣称独立样本。
Kc两种初态能量均值差约.00534，合并批均值标准误的.573倍；其他探针最大约1.99倍。
这支持初态间统计一致性，不是严格平衡证明。L16分布具有明显有限尺寸影响，
无限体积Kc处各相权重不必相等，单尺寸能量图不能独立证明一级相变。

![Sampling audit](../data_cache/potts2d_q8_L16_n128_spacing256_seed91/sampling.png)

## 模型和目标

dim_g=1，hidden60，16078参数，随机概率C，2×2不重叠，16→8→4。
Q/C/R独立梯度同步训练，C目标−2 I_empirical+2 L_variance+1 L_color；variance ratio=1。
Q保留relative_local几何与子集一致性；R一步及全训练尺度rollout权重各5。
batch32×64，Adam .001，2400步，seed91；无选核、固定C、硬类别对称或物理标签监督。

## 八态实现

V有8^4=4096种模式，环境仍为buffer1外完整20格壳层。
经验MI计数采用稀疏条件分布，避免分配环境类别数×4096稠密表；不改变MI数学目标。
小系统测试验证与原稠密实现的MI及梯度一致，八态稀疏零信息核测试通过；全套254项测试通过。
训练/测试平均约83.3%/84.7%的环境类别只出现一次，经验MI可能严重偏高。
最终须用scripts.audit_potts_mi_bias打乱同系综V/E配对作审计，不能把高MI直接解释成保留远距信息。

## 复现

```bash
python -m scripts.generate_potts2d \
  --output data_cache/potts2d_q8_L16_n128_spacing256_seed91 \
  --q 8 --k-min .8 --k-max 1.8 --burn 4096 --spacing 256 \
  --probe-samples 2048 --probe-burn 8192 --include-critical
python -m scripts.audit_potts_sampling \
  --cache data_cache/potts2d_q8_L16_n128_spacing256_seed91
python -m experiments.rsmi_separate_qcr \
  --config configs/potts2d_q8_g1_random_color1_seed91.json \
  --output results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91 \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c --color-weight 1
# 以下命令在训练完成后执行：
python -m scripts.report_scalar_joint \
  --run results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91
python -m scripts.report_potts_scalar \
  --run results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91
python -m scripts.report_mi_training \
  --run results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91
python -m scripts.audit_potts_mi_bias \
  --run results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91
```

八态报告只抽取32个类别置换做只读一致性审计，不枚举8!个置换，不用于训练。

## 状态

2400步训练已完成，seed91，随机概率C、独立梯度、保色权重1。训练循环约55.1分钟。
不稳定点g≈−1.311968，R′≈1.113903，微观反插值K≈1.327668；
相比理论Kc≈1.342454偏低约1.10%。稳定根g≈−.43338位于训练R输入支持之外，仍需外推审慎解释。
这些结果支持转变位置识别，但不等于一级相变标度或完整相图已验证；大环境字母表的经验MI上偏也仍然存在。

![最终R与K映射](../results/potts2d_q8_g1_random_mi_variance2ratio1_color1_2400_seed91/full_report/R_and_K_mapping.png)

训练目录保留history_progress.json和每400步检查点。
