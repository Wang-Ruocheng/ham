# 固定 C，优先优化 Q 的跨尺度闭合性

## 目的和边界

固定正 decimation，完整保留每个被抽取格点的四态 (sigma,tau)；dim_g=2。
不将固定 C 称为精确闭合的 AT 哈密顿量重整化。
保持现有 Q/R 架构、数据和原分布几何/子集一致性项，单独增加 Q 的数据目标。
不输入 K、尺度编号、磁化或理论相标签，不使用 R 或 EMA-Q 来构造新目标。

## 新增目标

令 p(D) 为256种四态2×2局部构型的经验频率，
对所有有后继的尺度定义 f_k=[p(D_k),p(C D_k)]/sqrt(2)。
把 k=0、1 的条目放入同一个池，用现有 relative_local 几何损失，
要求 Q(D_k) 的距离拟合 f_k 的距离。与原项的区别是：跨尺度比较，且目标含下一步信息。
两个不重叠的系综子集提供频率估计和采样噪声容差，目标全部停止梯度。
几何尺度4；新权重从0线性升至10。两半的 Q 输出取均值，与原几何项约定一致。
这只是有限局部统计下的预测充分性约束，不是严格闭合保证，也不保证二维足够。

训练从固定C基线第400步恢复参数、Adam和随机状态，到1200步；400—600步渐增。
原始几何、子集一致性仍更新Q；一步/多步闭环只更新R；C无参数。
默认配置不变。训练使用原middle-dense缓存，seed91，L16→8→4，batch32×64。

```bash
python -m experiments.at_predictive_q \
  --source results/ashkin_teller2d_g2_middle_dense_fixed_decimation_1200_seed91 \
  --output results/ashkin_teller2d_g2_middle_dense_fixed_futureGeometry10_1200_seed91 \
  --objective future_geometry --weight 10
python -m scripts.audit_at_future_geometry \
  --run results/ashkin_teller2d_g2_middle_dense_fixed_futureGeometry10_1200_seed91
python -m scripts.audit_at_fixed \
  --run results/ashkin_teller2d_g2_middle_dense_fixed_futureGeometry10_1200_seed91
```

## 验收

- 留出K组的128构型，一步和两步NRMSE；不只看训练损失。
- 跨尺度近邻对里，下一步局部频率显著不同的冲突数。
  同时报子集噪声阈值和潜空间RMS幅度5%阈值，不把噪声增加当成成功。
  阈值是诊断启发式，不是统计置信区间；配对样本并不独立。
- 检查Q是否只是放大坐标或破坏同系综子集一致性。
- R(g)流场及不动点是下游检查，而不是修改损失时使用的理论标签。

状态：训练已启动；全套266项测试通过。
