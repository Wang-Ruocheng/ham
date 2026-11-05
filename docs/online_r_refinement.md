# 单管线内的 R 精细拟合

当前二维 Ising 逐构型 CNN 设置使用新配置
`configs/ising2d_B0_cnn_b3_g1_refinedR_seed91.json`：
`training.r_updates_per_batch=10`。
这是优化更新频率的调整，不是新增 loss，不是训练末尾追加一个 R-only 阶段。
旧的 `ising2d_B0_cnn_b3_g1_short_seed91.json`、历史实验和服务器冻结快照均保留。
缺少该字段的旧配置和 checkpoint 始终解释为 1；没有静默修改所有历史/其他实验的默认值。

## 每轮执行

1. 新抽取 TRAIN 物理系综和构型子集，使用当前随机 C 生成尺度链。
2. Q 编码每个尺度的两个半系综，取输出均值为本轮 `g`。
3. 与旧管线相同，在任何参数更新之前独立计算 Q、C、R 的各自梯度，三者各更新一次。
4. 固定本轮更新前的 `sg(g)`，仅 R 再更新 9 次；复用同一个 Adam 及其历史矩。
5. 丢弃本轮 `g`。下一轮使用新的物理子集、随机 C 抽样和最新 Q/C 重新编码。

额外 R 更新没有 Q/C 前向、MI 计算或采样，也不消耗随机数。
共享优化器内非 R 梯度设置为 `None`，不能设为零，否则 Adam 仍可能更新 Q/C 的矩和参数。
保留 R 的递归梯度；仅直接编码的输入与目标断开梯度。

## 不变的定义

- CNN Q、随机可训练概率 C、残差 R 网络不变；b=3，36→12→4。
- Q/C 的目标、独立梯度、每个外层批次各一次的更新频率不变。
- R 保留均匀目标尺度的 `all_start_anchor`，总系数 10。
  两步时仍为 `5 e01 + 2.5 e02 + 2.5 e12`。
- Adam 学习率仍为 .001，没有新增调度、坐标归一化、监督或额外正则。
- 支持整数 1–32 次；多次更新要求 `qc_r_separate` 和 `all_start_anchor`。
- `epoch`/`training_step_N` 仍表示外层批次数；manifest、checkpoint、summary
  单独记录实际 Q/C/R Adam 参数时钟。例如 850 轮对应 Q/C 各 850 次，R 8500 次，
  不能称为 8500 次完整 QCR 训练。
- 如显式使用学习率调度，调度时钟仍为外层批次，内层 R 更新使用本轮同一学习率。
  精确续训必须保留更新比例、所有 Adam 状态及随机数状态，改变比例会被拒绝。

## 一个命令

通用入口（按配置的 3000 个外层批次训练；本身不带 20 分钟时间门控）：

```bash
python -m recurrent_rg train \
  --config configs/ising2d_B0_cnn_b3_g1_refinedR_seed91.json \
  --output results/ising2d_cnn_refinedR_fresh_run --device cuda:0
```

原短预算入口也已支持该新配置：

```bash
python -m experiments.rsmi_separate_qcr \
  --config configs/ising2d_B0_cnn_b3_g1_refinedR_seed91.json \
  --output results/ising2d_cnn_refinedR_short_fresh_run \
  --device cuda:0 --configuration-cnn-short-trial
```

短预算入口仍要求原先的服务器、GPU UUID 映射、nice 优先级和已有 MI 缓存；
需通过资源检查后由受控启动流程运行。吞吐预检现在包含每轮全部 10 次 R 更新，
不沿用旧的单次 R 计时，不自动突破 20 分钟上限。固定墙时内外层轮数可能减少，
因此不能同时宣称与旧实验训练量相同。示例命令不是资源使用授权。

## 证据与限制

此前冻结 Q/C 的 R-only 实验将同一留出目标的闭环误差降低约 93%，
证明原 R 尚有可优化空间；但不能由此保证在线 1:10 调度能复制完全相同的结果。
Q/C 在联合训练中仍会移动，同批多次更新可能过拟合本批噪声。
内循环后的同批目标只作为优化诊断；物理结论仍须检查固定 TRAIN/留出目标、
R(g)−g、g(K) 和各尺度闭环误差。没有添加按 Kc、热指数或 TEST 挑终点的规则。
后续正式试验应在独立输出目录验证；本次代码修改不自动启动或覆盖任何实验。

## 连续场 φ⁴ 的显式短预算配置

`configs/phi4_hopping_b3_cnn_refinedR_seed91.json` 将相同的 1:10 更新调度
接入既有 hopping-κ 单参数 CNN 管线，固定 λ=1、h=0、κ∈[.32,1.04]。
保持原 60 条物理链、整参数格点的 40/20 TRAIN/TEST 划分、32×64 批次、
固定 TRAIN RMS、共享 625 维软 plaquette 仿射重构、随机 Gaussian C 和残差 R。
连续 C 的 full-shell 平滑信息代理/方差底/均匀振幅项仍是此前明确的连续适配，
不是二值直方图 MI/NLL。无参数标签监督，也没有追加训练后的 R-only 阶段。

```bash
python -m recurrent_rg train \
  --config configs/phi4_hopping_b3_cnn_refinedR_seed91.json \
  --output results/phi4_cnn_refinedR_fresh_run
```

该显式配置固定在 225 的 GPU3 UUID 映射下执行；入口会验证资源与配置，
并拒绝 resume、其他设备及 legacy 开关。旧 CNN 入口缺省仍为 1:1。
10 次可丢弃测速外层更新包含 100 次 R 更新；根据吞吐固定 200–3000 个正式
外层批次，目标 900 秒、上限 1200 秒，包含加载、测速、评价和报告。
`scripts/run_phi4_cnn_refined_trial.py` 另设独立进程时限，只能终止自己创建的
训练进程组，不触碰其他 GPU/用户进程。正式源代码在新快照中冻结。

CPU 回归验证：相同外层批次下 1:1 与 1:10 的 Q/C 权重、Adam 状态及采样 RNG
逐位相同；R 额外更新使用原始半系综编码且梯度与原公式一致。
最终报告分别校验 Q/C 与 R 的逐参数 Adam 时钟，并保留同批前/后 R 目标。
