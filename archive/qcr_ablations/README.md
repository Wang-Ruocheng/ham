# 历史消融：不属于当前主训练

这些文件从 experiments/ 主目录移入归档；原实验结果和权重没有删除。

- `legacy_objective.py`：旧固定距离/直径下限、尺度衰减及单位方差锚点。
- `balanced_geometry.py`：强制协方差利用及自适应度量方案。
- `supervised_q_magnetization.py`：监督 Q 加入平均磁化特征。
- `supervised_q_structure_ablation.py`：监督 Q 的归一化/层次池化结构对照。
- `kernel_distinct.py`：已退役的均匀块输出可区分性公式，仅供历史结果比较；
  当前训练目标和训练 CLI 已删除该项。旧配置字段保留可读，非零权重不能用于新训练或续训。

前三类约束没有混入当前 data_geometry QCR loss。监督实验只用于诊断；不是主结果的训练流程。
从仓库根目录执行归档脚本即可；其历史硬编码数据路径仍要求原数据存在。
旧 loss 可显式导入 `legacy_qcr_objective`；当前 CLI 不再提供 legacy 模式。
