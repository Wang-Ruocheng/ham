# 随机概率 C：不做选核初始化的一维/二维 Ising

## 实验边界与统一设置

本实验明确取消 MI 核搜索、核表拟合预训练及旧权重加载。Q/C/R 均由 seed91
随机初始化；C 是可训练 Bernoulli 核。数据计数仅用于构造可微经验 MI loss，
不是穷举候选核；所有2400步均靠梯度更新，不使用耦合标签、保符号项或硬Z2网络约束。
单个种子，各问题保留ratio=.25对照和ratio=1修正，未挑选早期检查点。

沿用原数据：一维经典零场Ising，L128、J∈[.01,2]、64个J、每J三个系综、
每系综128构型，dim_g=1，5次粗粒化、9711参数；二维K/B Ising，L16、
K=.20:.03:.68、B=-.01:.001:.01、每系综64构型，dim_g=2，2次粗粒化、10373参数。
两者hidden60、C为factor2不重叠块、batch32×64、Adam lr=.001。
一维保留原有翻转配对数据增强；二维不加该增强。

同步更新、独立梯度，并非所有loss向所有模块反传：

- Q：2×微观子集一致性 + 5×粗尺度子集一致性 + 10×relative_local分布几何。
- C：−2×逐系综经验MI + 2×多尺度逐系综方差下限。
- R：5×一步闭环 + 5×覆盖全部训练尺度的多步闭环，不反馈Q/C。

MI使用微观块与buffer=1外的环境，一维E=(s−2,s3)，二维完整20自旋壳层。
有限求和对C概率完全可微，不通过采样MI梯度；C产生粗粒化构型时仍使用
Bernoulli采样及STE，方差项经过这条路径。精确经验MI不等于无偏总体MI。

## 为什么加强方差下限

原ratio=.25时，AND在均匀独立输入下输出方差为.75，仍超过要求的.25，
因而该项不会排斥AND。修正只将ratio改为1，保留权重2及其他设置：

\[
L_{\rm var}=\langle[|m_{k+1}|-\sqrt{1-\alpha(1-\operatorname{sg}(m_k)^2)}]_+^2\rangle_{D,k},
\qquad\alpha=1.
\]

即软惩罚输出的单点方差低于当前输入方差；每个系综、每一步分别计算，输入参考停梯度。
它不是指定decimation，不要求均匀/有序输入生成随机自旋，也不是硬Z2约束。
但它确实是一种统计正则化假设，不是普适RG定律，可能与某些物理系统的粗粒化竞争。
单元测试明确验证：均匀四种二自旋输入的AND受到.25原始惩罚，decimation为0，
常数输入保持常数也为0。全套247项测试通过。

## 一维最终结果

ratio=.25对照学到近AND；虽然也找到一稳一不稳两个根，无序J=0编码与稳定根的
距离却达端点间距96%，不能算正确恢复。一维无有限温度临界点，不能套用TFIM的三个根标准。

ratio=1从随机初值直接学得：
P(+|--,+-,-+,++)=(0.00000204,0.99991926,0.00004078,0.99999817)，接近正decimation。

| 端点 | Q编码 | R根 | R′ | 根偏差/物理端点间距 |
|---|---:|---:|---:|---:|
| J→∞ | 0.143743 | 0.103484，不稳定 | 1.621811 | 2.212% |
| J=0 | 1.963781 | 1.945508，稳定 | 0.124807 | 1.004% |

300个J∈[.01,2]诊断点中299个流向无序端；一个极近无序端的点因根偏移未满足符号判据。
与精确decimation递推经同一个Q映射所得参考相比，一步RMSE/端点间距=1.0232%。
总体分布通过实际学习C的五步一步闭环RMSE分别为.01863、.02001、.01973、.01634、.01611。
这是population-pooled Q审计，不是把有限样本Q的期望误称为精确值。
未训练的J=4、∞仅用于展示外推；∞处R反复迭代仍漂移，不能宣称完整精确RG。

测试加权R闭环loss由对照.090361降至.035103；总QCR不含MI约.150046。
总loss下降很少也能发生物理结构改善，不能只按总loss或根数量验收。

![1D validation](../results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr/endpoint_report/physical_validation.png)
![1D losses](../results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr/report/loss_comparison.png)

## 二维最终结果

二维最终ratio=1结果：中心根g=(0.565476,0.252414)，DR本征值
(1.569596,−1.126980)，两个模均大于1，是离散映射的排斥子，不是鞍点。
负号表示一步翻转；DR²本征值约(2.463632,1.270083)，两步流场可避免箭头造成误读。
微观Q(K,B)网格反插值得K≈0.426121、B≈−0.000376，K相对理论Kc≈0.440687偏低3.31%。
理论临界点的编码为(0.316010,0.209508)，与根仍有偏移，不宣称精确定位。
另一个鞍点和稳定点没有微观KB网格内的插值候选，不赋予确定物理解读。

C学到近似“块中三个自旋多数票后取负”，忽略第3个输入（零起始索引2），
最大规则概率误差约1.91e−5；未指定规则，也未做输出符号纠正。
微观g协方差本征值(.201952,1.226928)，弱/强比≈.165，未塌缩为直线。
测试一步NRMSE在两个输入尺度为3.99%、5.69%，两步rollout NRMSE=6.30%。
测试QCR（不含MI）=.727447，其中几何.373283、微观一致性.209241、粗尺度一致性.089056、
C方差.001694、一步闭环.024694、多步闭环.029479。

ratio=.25二维对照本身也成功：中心根(.540846,.243884)，DR=(1.574099,−1.133876)，
反插值K=.426358、B=−.000368。提高方差下限不是二维成功的必要条件，
而是本次找到同时适用于一维和二维的配方；单seed不证明初始化鲁棒性。

![2D one and two steps](../results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91/qr/report/R_one_two_step.png)
![2D mapping](../results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91/qr/report/g_KB_maps.png)
![2D losses](../results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91/qr/report/loss_comparison.png)

## 复现

下列source仅提供配置和数据路径，脚本调用build_model随机建模，并不加载source权重。
关键是**不传--mi-initialize**；最终manifest中data_driven_initialization为null。

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_MIinit_independentC_variance2_2400_seed91/qr \
  --output results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_ising1d_endpoints \
  --run results/ising1d_g1_random_independentC_variance2ratio1_2400_seed91/qr
python -m experiments.rsmi_separate_qcr \
  --source results/ising2d_g2_MIinit_independentC_variance2_2400_seed91/qr \
  --output results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91/qr \
  --estimator histogram --variance-weight 2 --variance-ratio 1 \
  --variance-per-ensemble --independent-c
python -m scripts.report_rsmi_experiment \
  --run results/ising2d_g2_random_independentC_variance2ratio1_2400_seed91 \
  --flow-title 'Random C; independent Q/C/R gradients; variance ratio 1'
```

ratio=.25对照命令仅省去--variance-ratio 1，并将输出目录中的variance2ratio1改为variance2。
不覆盖既有目录。新设置是显式实验配方，未静默修改全项目默认值。
