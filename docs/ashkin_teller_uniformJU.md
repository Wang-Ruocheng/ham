# AT：独立均匀 J/T、U/T，固定 C、dim_g=2

目的：与 U/J=2 的单条采样线比较，检验二维数据支持是否改善表示和流场。
这是改变训练分布的探索实验，不是同数据量的严格消融。

- J/T：0至0.6，11个等距点；U/T：0至1.2，11个等距点。
- 121个参数组合，每组3个独立MCMC副本，每副本128构型，共46464构型。
- L=16，训练尺度16→8→4；burn=1024，间隔16次双条件SW更新。
- U>=J 使用sigma和product条件更新；U<J使用sigma和tau条件更新，所有条件键非负。
- 按完整耦合点分组切分，243训练系综、120测试系综；无同参数副本泄漏。
- 固定正decimation，dimg=2，hidden60，seed91，1200步，batch32×64，lr=.001。
- 使用原relative_local分布几何和子集一致性训练Q；一步/全可用尺度rollout训练R。
- Q/R梯度分离，不启用EMA预测、未来分布几何trick、协方差或弱方向加权。
- 标签仅用于分组和报告，不作为Q输入或训练目标。
- 保留六处冷热启动采样检查；有限尺寸/临界慢化仍须从报告判断。

```bash
python -m scripts.generate_ashkin_teller2d_grid \
 --output data_cache/ashkin_teller2d_uniformJU_L16_n128_seed91
python -m recurrent_rg train \
 --config configs/ashkin_teller2d_g2_uniformJU_fixed_seed91.json \
 --output results/ashkin_teller2d_g2_uniformJU_fixed_1200_seed91
python -m recurrent_rg plot \
 --run results/ashkin_teller2d_g2_uniformJU_fixed_1200_seed91 \
 --output results/ashkin_teller2d_g2_uniformJU_fixed_1200_seed91/report
```

状态：已启动数据生成→训练→报告的顺序流水线，尚未判断恢复相图是否成功。
