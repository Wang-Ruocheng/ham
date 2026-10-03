# 1D Ising：精确经验互信息对照

## 设置与定义

单 seed=91，dim_g=1，随机初始化概率 C，2400 次更新；QC/R 同时训练但分离梯度。
沿用 InfoNCE 实验数据和 QCR 设置：L=128，5 步粗粒化，J 在 0.01–2 的 64 点，
每点 3 个 128 构型的系综；训练/测试 129/63 个系综。训练耗时 168.17 秒。
没有使用 J 标签监督，没有新增物理数据。QCR 参数 9711，无可训练 critic。

对每个系综分别统计 V=(s0,s1)、E=(s[-2],s3) 的 4×4 经验联合分布，
使用该系综全部缓存构型和位置，并保留原有全局翻转数据增强。
不把不同系综混在一起；重叠位置并非独立样本。

\[
\widehat p_C(h,e)=\sum_v\widehat p(v,e)C_\theta(h|v),\qquad
I_{\rm hist}=\sum_{h,e}\widehat p_C(h,e)
\log\frac{\widehat p_C(h,e)}{\widehat p_C(h)\widehat p(e)}.
\]

QC 优化 L_QC−2 I_hist（自然对数）；R 只优化原有一步/多步闭环项。
MI 仅在原始尺度计算。全部旧 C 辅助项关闭。
每次更新抽取 16 个训练系综的完整直方图。
这是对经验分布精确求和，并非总体互信息精确已知；仍有有限数据误差。
相对旧 InfoNCE，对照同时去除了 critic 近似，并使用了更充分的缓存统计。

## 结果

最终 C 的 P(+|--,+-,-+,++)：
`[0.000000113, 0.000053430, 0.000054560, 0.999978193]`。
仍然是近乎确定性的 AND，并未学到 decimation。

| 核 | 训练经验 MI / bit | 测试经验 MI / bit |
|---|---:|---:|
| 最终学习 C | 0.39592017 | 0.38397481 |
| 确定性 AND | 0.39594865 | 0.38400189 |
| 正 decimation | 0.39696027 | 0.38503217 |

枚举全部 16 个确定性二进制核，左右 decimation 及其整体取负并列最优。
用已知 Ising 分布的有限环转移矩阵做训练后审计，平均总体 MI：
最终 C=0.39162307、AND=0.39165102、decimation=0.39268604 bit。
此审计使用 J，不反馈训练。

最终测试加权 loss：anchor 0.01694980，子集一致性 0.03569526，
分布几何 0.00925900，一步闭环 0.11577451，多步闭环 0.15694410。
QCR 合计 0.33462267，包含 −2 MI 后总目标约 −0.19768。
旧 InfoNCE QCR 测试 loss=0.27025964，本次没有改善。

指定搜索区间 [−3.92886,3.73470] 内仅检测到稳定根 g=−2.862537，
R'=0.133824，略在训练输入范围之外；未检测到预期的不稳定端点。
网格符号变化查根可能漏掉切触根，不作全实轴结论。

结论：critic 估计误差不是未学习到 decimation 的唯一原因。
当前精确经验 MI 目标确实略偏好 decimation，但这次多目标梯度训练未到达它。
局部优化、概率输出饱和与其他 QC loss 的影响尚未由消融分离，不能据此断言 RSMI 原理失效。

## 复现

```bash
python -m experiments.rsmi_separate_qcr \
  --source results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr \
  --output results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr \
  --estimator histogram
python -m scripts.report_rsmi_experiment \
  --run results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91
python -m scripts.audit_ising1d_rsmi_loss \
  --run results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr
python -m scripts.report_exact_mi_comparison
```

输出路径必须不存在。全部测试 237 项通过，包括直方图计数、MI 值、梯度检查以及系综隔离。
结果目录中的 `report/` 保存 R(g)、耦合映射、loss 曲线和逐核 MI 枚举；
`loss_audit/` 保存精确有限环分布下的多尺度诊断。
