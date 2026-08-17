"""
运行所有 HNN Baseline 模型
==========================
依次训练并评估 4 个 baseline 模型，输出对比表格。
用法: python run_baselines.py
"""
import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from pendulum_string import DiscretePendulumString
from hnn_baselines import (
    SeparableHNN, PartialHNN, SIREN_HNN, SymplecticHNN, SympNet, FNO, FNOFlow,
    GraphHNN, CHNN,
    train_single_step, train_symplectic, train_sympnet, train_fno_flow,
    generate_multistep_data,
    evaluate_and_visualize,
    _maybe_unwrap,
)


def generate_data_if_needed(args, output_dir='.', seg_length=None, seg_mass=None):
    """预生成训练数据 (仅 rank 0，在 DDP 初始化之前调用)"""
    data_path = os.path.join(output_dir, f'pendulum_string_data_N{args.n_masses}.pt')
    if seg_length is None:
        seg_length = args.length
    if seg_mass is None:
        seg_mass = args.mass

    rank = int(os.environ.get('RANK', 0))
    if rank == 0 and not os.path.exists(data_path):
        print("  生成训练数据...")
        sys = DiscretePendulumString(n_masses=args.n_masses,
                                     length=seg_length, mass=seg_mass, g=args.g)
        train_ds, val_ds, test_ds = sys.generate_dataset(
            n_trajectories=args.n_trajectories, seed=args.seed)
        torch.save((train_ds, val_ds, test_ds), data_path)
        print(f"  数据已保存: {data_path}")
    return data_path


def load_data(data_path, output_dir='.'):
    """加载已生成的训练数据 (所有 rank 调用)"""
    train_ds, val_ds, test_ds = torch.load(data_path, weights_only=False)
    return train_ds, val_ds, test_ds
def summarize_results(results, output_dir='.'):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("BASELINE 对比总结")
    print("=" * 80)
    print(f"{'模型':<25} {'测试 MSE':>14} {'参数':>10} {'训练时间':>12}")
    print("-" * 80)
    for name, mse, n_params, elapsed in results:
        print(f"{name:<25} {mse:>14.6e} {n_params:>10,} {elapsed:>10.1f}s")
    print("-" * 80)

    valid = [(n, m, p, e) for n, m, p, e in results if not np.isnan(m)]
    if valid:
        best = min(valid, key=lambda x: x[1])
        print(f"\n最佳模型: {best[0]} (MSE = {best[1]:.6e})")

    # 绘制对比柱状图
    names = [r[0] for r in results]
    mses = [r[1] if not np.isnan(r[1]) else 0 for r in results]
    n_params = [r[2] for r in results]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#607D8B', '#795548', '#FF5722']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(names, mses, color=colors)
    ax1.set_ylabel('Test MSE'); ax1.set_title('HNN Baseline: Test MSE')
    ax1.set_yscale('log'); ax1.tick_params(axis='x', rotation=15)
    for i, v in enumerate(mses):
        if v > 0:
            ax1.text(i, v * 1.1, f'{v:.3e}', ha='center', fontsize=9)

    ax2.bar(names, n_params, color=colors)
    ax2.set_ylabel('Parameters'); ax2.set_title('HNN Baseline: Model Size')
    ax2.tick_params(axis='x', rotation=15)
    for i, v in enumerate(n_params):
        ax2.text(i, v * 1.02, f'{v:,}', ha='center', fontsize=9)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(fname, dpi=150)
    print(f"对比图已保存: {fname}")
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='HNN Baseline Runner')
    parser.add_argument('--n_masses', type=int, default=10)
    parser.add_argument('--length', type=float, default=1.0,
                        help='摆链总长度 (每节长度 = total_length / n_masses)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='总质量 (每节质量 = total_mass / n_masses)')
    parser.add_argument('--g', type=float, default=9.81)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_trajectories', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_symplectic', action='store_true',
                        help='跳过多步辛训练')
    parser.add_argument('--sympnet_K', type=int, default=5,
                        help='SympNet 的层数 K (default: 5)')
    parser.add_argument('--fno_modes', type=int, default=24,
                        help='FNO 保留的 Fourier 模态数 (default: 24)')
    parser.add_argument('--fno_hidden', type=int, default=128,
                        help='FNO/FNOFlow 隐藏层维度 (default: 128)')
    parser.add_argument('--fno_dropout', type=float, default=0.1,
                        help='FNOFlow dropout 率 (default: 0.1)')
    parser.add_argument('--fno_flow_dt', type=float, default=0.05,
                        help='FNOFlow 单步时间步长 (default: 0.05)')
    parser.add_argument('--graph_hidden', type=int, default=128,
                        help='GraphHNN 隐藏层维度 (default: 128)')
    parser.add_argument('--chnn_hidden', type=int, default=256,
                        help='CHNN 隐藏层维度 (default: 256)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'separable', 'partial', 'siren', 'symplectic', 'sympnet', 'fno', 'fno_flow', 'graph', 'chnn'],
                        help='单独运行某个模型 (default: all)')
    parser.add_argument('--ddp', action='store_true',
                        help='使用 DDP 多 GPU 并行训练')
    args = parser.parse_args()

    output_dir = f'results_N{args.n_masses}'

    # 每节长度/质量 = 总长度/质量 / N
    seg_length = args.length / args.n_masses
    seg_mass = args.mass / args.n_masses

    # 提前获取 rank (在 DDP 初始化之前，仅用于数据生成)
    if args.ddp:
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # ── 数据生成 (DDP 初始化之前，避免 NCCL 超时) ──
    data_path = generate_data_if_needed(args, output_dir, seg_length, seg_mass)

    # 其他 rank 等待数据文件生成完成
    if args.ddp and rank != 0:
        while not os.path.exists(data_path):
            time.sleep(2)
        # 额外等一小段时间确保文件写入完成
        time.sleep(1)

    # ── DDP 初始化 ──
    if args.ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = local_rank
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim = 2 * args.n_masses

    if rank == 0:
        print("=" * 80)
        print(f"HNN Baseline 对比: N={args.n_masses}, dim={dim}")
        print(f"总长 L={args.length}, 总质量 M={args.mass}")
        print(f"每节 l={seg_length:.4f}, m={seg_mass:.4f}, g={args.g}")
        print(f"MLP: {dim} -> {args.hidden_dim}x{args.num_layers} -> 1")
        print(f"设备: {device}, GPUs: {world_size}, epochs: {args.epochs}")
        print(f"输出: {output_dir}/")
        print("=" * 80)

    if rank == 0:
        print("\n--- 加载数据 ---")
    train_ds, val_ds, test_ds = load_data(data_path, output_dir)

    if args.ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sys = DiscretePendulumString(n_masses=args.n_masses,
                                 length=seg_length, mass=seg_mass, g=args.g)
    ml2 = seg_mass * seg_length ** 2

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    results = []

    def _wrap(model):
        """如果启用 DDP，包装模型；否则返回原模型"""
        model = model.to(device)
        if args.ddp:
            model = DDP(model, device_ids=[device])
        return model

    # ── 1. SeparableHNN ─────────────────────────────────────
    if args.model in ('all', 'separable'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("1/9: SeparableHNN — H = T(p) + V(theta)")
            print("=" * 80)
        model1 = _wrap(SeparableHNN(N=args.n_masses, hidden_dim=args.hidden_dim,
                                     num_layers=args.num_layers))
        t0 = time.time()
        train_single_step(model1, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[Separable]',
                          compute_stats_fn=lambda m, l: _maybe_unwrap(m).compute_stats(l))
        elapsed = time.time() - t0
        mse1, p1 = evaluate_and_visualize(model1, test_loader, sys, args, device, 'SeparableHNN', output_dir=output_dir)
        if rank == 0:
            results.append(('SeparableHNN', mse1, p1, elapsed))
            torch.save(_maybe_unwrap(model1).state_dict(),
                       os.path.join(output_dir, 'separable_checkpoint.pt'))

    # ── 2. PartialHNN ───────────────────────────────────────
    if args.model in ('all', 'partial'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("2/9: PartialHNN — 已知 M(theta), 仅学 V(theta)")
            print("=" * 80)
        model2 = _wrap(PartialHNN(N=args.n_masses, ml2=ml2, hidden_dim=args.hidden_dim,
                                   num_layers=args.num_layers))
        t0 = time.time()
        train_single_step(model2, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[Partial]',
                          compute_stats_fn=lambda m, l: _maybe_unwrap(m).compute_stats(l))
        elapsed = time.time() - t0
        mse2, p2 = evaluate_and_visualize(model2, test_loader, sys, args, device, 'PartialHNN', output_dir=output_dir)
        if rank == 0:
            results.append(('PartialHNN', mse2, p2, elapsed))
            torch.save(_maybe_unwrap(model2).state_dict(),
                       os.path.join(output_dir, 'partial_checkpoint.pt'))

    # ── 3. SIREN_HNN ────────────────────────────────────────
    if args.model in ('all', 'siren'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("3/9: SIREN_HNN — sin 激活函数")
            print("=" * 80)
        model3 = _wrap(SIREN_HNN(dim=dim, hidden_dim=args.hidden_dim,
                                  num_layers=args.num_layers))
        t0 = time.time()
        train_single_step(model3, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[SIREN]',
                          compute_stats_fn=lambda m, l: _maybe_unwrap(m).compute_stats(l))
        elapsed = time.time() - t0
        mse3, p3 = evaluate_and_visualize(model3, test_loader, sys, args, device, 'SIREN_HNN', output_dir=output_dir)
        if rank == 0:
            results.append(('SIREN_HNN', mse3, p3, elapsed))
            torch.save(_maybe_unwrap(model3).state_dict(),
                       os.path.join(output_dir, 'siren_checkpoint.pt'))

    # ── 4. SymplecticHNN ────────────────────────────────────
    if args.model in ('all', 'symplectic'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("4/9: SymplecticHNN — Separable + 多步辛训练")
            print("=" * 80)
        if args.skip_symplectic:
            if rank == 0:
                print("  [跳过] --skip_symplectic")
                results.append(('SymplecticHNN', float('nan'), 0, 0))
        else:
            if rank == 0:
                print("  生成多步训练数据...")
            n_steps = 5
            dt = 0.05  # 单步积分步长，与数据生成一致
            mstep_train, mstep_val = generate_multistep_data(
                sys, n_trajectories=args.n_trajectories,
                dt=dt, n_steps=n_steps, seed=args.seed)
            if args.ddp:
                mstep_sampler = DistributedSampler(mstep_train, num_replicas=world_size, rank=rank)
                mstep_train_loader = DataLoader(mstep_train, batch_size=args.batch_size,
                                                sampler=mstep_sampler, num_workers=4, pin_memory=True)
            else:
                mstep_train_loader = DataLoader(mstep_train, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)

            model4 = _wrap(SymplecticHNN(N=args.n_masses, hidden_dim=args.hidden_dim,
                                          num_layers=args.num_layers))
            t0 = time.time()
            train_symplectic(model4, mstep_train_loader, val_loader,
                             n_steps=n_steps, dt=dt,
                             epochs=args.epochs, lr=args.lr, device=device,
                             label='[Symplectic]')
            elapsed = time.time() - t0
            mse4, p4 = evaluate_and_visualize(model4, test_loader, sys, args, device, 'SymplecticHNN', output_dir=output_dir)
            if rank == 0:
                results.append(('SymplecticHNN', mse4, p4, elapsed))
                torch.save(_maybe_unwrap(model4).state_dict(),
                           os.path.join(output_dir, 'symplectic_checkpoint.pt'))

    # ── 5. SympNet ──────────────────────────────────────────
    if args.model in ('all', 'sympnet'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("5/9: SympNet — 直接学辛映射 Φ(x) = x_{t+dt}")
            print("=" * 80)
            print("  生成 SympNet 训练数据 (单步状态对)...")
        dt_symp = 0.05
        symp_train, symp_val = generate_multistep_data(
            sys, n_trajectories=args.n_trajectories,
            dt=dt_symp, n_steps=1, seed=args.seed)
        if args.ddp:
            symp_train_sampler = DistributedSampler(symp_train, num_replicas=world_size, rank=rank)
            symp_train_loader = DataLoader(symp_train, batch_size=args.batch_size,
                                           sampler=symp_train_sampler, num_workers=4, pin_memory=True)
        else:
            symp_train_loader = DataLoader(symp_train, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
        symp_val_loader = DataLoader(symp_val, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)

        model5 = _wrap(SympNet(N=args.n_masses, K=args.sympnet_K,
                                hidden_dim=args.hidden_dim,
                                num_layers=args.num_layers))
        # dt 需要设置在底层模型上
        _maybe_unwrap(model5).dt = torch.tensor(dt_symp)
        n_params_symp = sum(p.numel() for p in model5.parameters())
        if rank == 0:
            print(f"  SympNet: K={args.sympnet_K}, params={n_params_symp:,}, dt={dt_symp}")

        t0 = time.time()
        train_sympnet(model5, symp_train_loader, symp_val_loader,
                      epochs=args.epochs, lr=args.lr, device=device,
                      label='[SympNet]')
        elapsed = time.time() - t0
        mse5, p5 = evaluate_and_visualize(model5, test_loader, sys, args, device, 'SympNet', output_dir=output_dir)
        if rank == 0:
            results.append(('SympNet', mse5, p5, elapsed))
            torch.save(_maybe_unwrap(model5).state_dict(),
                       os.path.join(output_dir, 'sympnet_checkpoint.pt'))

    # ── 6. FNO — Fourier Neural Operator ─────────────────────
    if args.model in ('all', 'fno'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("6/9: FNO — Fourier Neural Operator (1D 向量场学习)")
            print("=" * 80)

        model6 = _wrap(FNO(N=args.n_masses, modes=args.fno_modes,
                           hidden_dim=args.fno_hidden,
                           num_layers=args.num_layers))
        n_params_fno = sum(p.numel() for p in model6.parameters())
        if rank == 0:
            print(f"  FNO: N={args.n_masses}, modes={args.fno_modes}, "
                  f"hidden={args.fno_hidden}, layers={args.num_layers}, "
                  f"params={n_params_fno:,}")
            print(f"  关键优势: 参数不随 N 增长 (仅取决于 modes/hidden)")

        def _compute_fno_stats(m, l):
            r = _maybe_unwrap(m)
            r.compute_stats(l)

        t0 = time.time()
        train_single_step(model6, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[FNO]',
                          compute_stats_fn=_compute_fno_stats)
        elapsed = time.time() - t0
        if rank == 0:
            ckpt_path = os.path.join(output_dir, 'fno_checkpoint.pt')
            torch.save({
                'model_state_dict': _maybe_unwrap(model6).state_dict(),
                'config': {
                    'N': args.n_masses,
                    'modes': args.fno_modes,
                    'hidden_dim': args.fno_hidden,
                    'num_layers': args.num_layers,
                }
            }, ckpt_path)
            print(f"  FNO checkpoint saved: {ckpt_path}")
        mse6, p6 = evaluate_and_visualize(model6, test_loader, sys, args, device, 'FNO', output_dir=output_dir)
        if rank == 0:
            results.append(('FNO', mse6, p6, elapsed))

    # ── 6.5. FNOFlow — Flow Map ─────────────────────────────
    if args.model in ('all', 'fno_flow'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("7/9: FNOFlow — FNO 直接学习流映射 x_t → x_{t+dt}")
            print("=" * 80)
            print("  生成 FNOFlow 训练数据 (单步状态对)...")

        dt_flow = args.fno_flow_dt
        flow_train, flow_val = generate_multistep_data(
            sys, n_trajectories=args.n_trajectories,
            dt=dt_flow, n_steps=1, seed=args.seed)
        if args.ddp:
            flow_sampler = DistributedSampler(flow_train, num_replicas=world_size, rank=rank)
            flow_train_loader = DataLoader(flow_train, batch_size=args.batch_size,
                                           sampler=flow_sampler, num_workers=4, pin_memory=True)
        else:
            flow_train_loader = DataLoader(flow_train, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
        flow_val_loader = DataLoader(flow_val, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)

        model7 = _wrap(FNOFlow(N=args.n_masses, dt=dt_flow,
                               modes=args.fno_modes,
                               hidden_dim=args.fno_hidden,
                               num_layers=args.num_layers,
                               dropout=args.fno_dropout))
        n_params_fnoflow = sum(p.numel() for p in model7.parameters())
        if rank == 0:
            print(f"  FNOFlow: N={args.n_masses}, dt={dt_flow}, "
                  f"modes={args.fno_modes}, hidden={args.fno_hidden}, "
                  f"layers={args.num_layers}, dropout={args.fno_dropout}, "
                  f"params={n_params_fnoflow:,}")
            print(f"  关键优势: 直接预测 x{{t+dt}}，无需 RK4 积分")

        t0 = time.time()
        train_fno_flow(model7, flow_train_loader, flow_val_loader,
                       epochs=args.epochs, lr=args.lr, device=device,
                       label='[FNOFlow]')
        elapsed = time.time() - t0

        # 评估: 轨迹预测 + GIF（复用 evaluate_and_visualize，已支持 FNOFlow）
        raw_fnoflow = _maybe_unwrap(model7)
        mse7, p7 = evaluate_and_visualize(model7, flow_val_loader, sys, args, device,
                                          'FNOFlow', output_dir=output_dir)
        if rank == 0:
            ckpt_path = os.path.join(output_dir, 'fnoflow_checkpoint.pt')
            torch.save({
                'model_state_dict': raw_fnoflow.state_dict(),
                'config': {
                    'N': args.n_masses, 'dt': dt_flow,
                    'modes': args.fno_modes,
                    'hidden_dim': args.fno_hidden,
                    'num_layers': args.num_layers,
                    'dropout': args.fno_dropout,
                }
            }, ckpt_path)
            print(f"  FNOFlow checkpoint saved: {ckpt_path}")
            results.append(('FNOFlow', mse7, p7, elapsed))

    # ── 8. GraphHNN ────────────────────────────────────────
    if args.model in ('all', 'graph'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("8/9: GraphHNN — 图结构参数共享 HNN (节点+边 MLP)")
            print("=" * 80)

        model8 = _wrap(GraphHNN(N=args.n_masses, hidden_dim=args.graph_hidden,
                                 num_layers=args.num_layers))
        n_params_graph = sum(p.numel() for p in model8.parameters())
        if rank == 0:
            print(f"  GraphHNN: N={args.n_masses}, hidden={args.graph_hidden}, "
                  f"layers={args.num_layers}, params={n_params_graph:,}")
            print(f"  关键优势: 共享节点/边 MLP，参数 O(1) 每节点")

        def _compute_graph_stats(m, l):
            r = _maybe_unwrap(m)
            r.compute_stats(l)

        t0 = time.time()
        train_single_step(model8, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[GraphHNN]',
                          compute_stats_fn=_compute_graph_stats)
        elapsed = time.time() - t0
        mse8, p8 = evaluate_and_visualize(model8, test_loader, sys, args, device,
                                          'GraphHNN', output_dir=output_dir)
        if rank == 0:
            results.append(('GraphHNN', mse8, p8, elapsed))
            torch.save(_maybe_unwrap(model8).state_dict(),
                       os.path.join(output_dir, 'graph_checkpoint.pt'))

    # ── 9. CHNN — Constrained HNN ──────────────────────────
    if args.model in ('all', 'chnn'):
        if rank == 0:
            print("\n" + "=" * 80)
            print("9/9: CHNN — 笛卡尔坐标约束 HNN (Lagrange 乘子)")
            print("=" * 80)
            print("  生成笛卡尔坐标训练数据...")

        cart_data_path = os.path.join(output_dir,
                                      f'cartesian_data_N{args.n_masses}.pt')
        if rank == 0:
            if not os.path.exists(cart_data_path):
                cart_train, cart_val, cart_test = sys.generate_cartesian_dataset(
                    n_trajectories=args.n_trajectories, seed=args.seed)
                torch.save((cart_train, cart_val, cart_test), cart_data_path)
                print(f"  笛卡尔数据已保存: {cart_data_path}")
            else:
                cart_train, cart_val, cart_test = torch.load(
                    cart_data_path, weights_only=False)
                print(f"  加载已保存的笛卡尔数据: {cart_data_path}")

        if args.ddp:
            dist.barrier()
            if rank != 0:
                cart_train, cart_val, cart_test = torch.load(
                    cart_data_path, weights_only=False)
                print(f"  [rank {rank}] 加载笛卡尔数据: {cart_data_path}")

            cart_train_sampler = DistributedSampler(
                cart_train, num_replicas=world_size, rank=rank)
            cart_val_sampler = DistributedSampler(
                cart_val, num_replicas=world_size, rank=rank, shuffle=False)
            cart_train_loader = DataLoader(cart_train, batch_size=args.batch_size,
                                           sampler=cart_train_sampler,
                                           num_workers=4, pin_memory=True)
            cart_val_loader = DataLoader(cart_val, batch_size=args.batch_size,
                                         sampler=cart_val_sampler,
                                         num_workers=4, pin_memory=True)
            cart_test_loader = DataLoader(cart_test, batch_size=args.batch_size,
                                          shuffle=False)
        else:
            cart_train_loader = DataLoader(cart_train, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
            cart_val_loader = DataLoader(cart_val, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)
            cart_test_loader = DataLoader(cart_test, batch_size=args.batch_size,
                                          shuffle=False)

        model9 = _wrap(CHNN(N=args.n_masses, l=seg_length, m=seg_mass,
                             hidden_dim=args.chnn_hidden,
                             num_layers=args.num_layers))
        n_params_chnn = sum(p.numel() for p in model9.parameters())
        if rank == 0:
            print(f"  CHNN: N={args.n_masses}, dim={4*args.n_masses}, "
                  f"l={seg_length:.4f}, m={seg_mass:.4f}, "
                  f"hidden={args.chnn_hidden}, params={n_params_chnn:,}")
            print(f"  关键优势: 笛卡尔坐标 + 显式约束 Lagrange 乘子")

        def _compute_chnn_stats(m, l):
            r = _maybe_unwrap(m)
            r.compute_stats(l)

        t0 = time.time()
        train_single_step(model9, cart_train_loader, cart_val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[CHNN]',
                          compute_stats_fn=_compute_chnn_stats)
        elapsed = time.time() - t0

        # CHNN 评估（笛卡尔测试集）
        if rank == 0:
            model9.eval()
            raw_chnn = _maybe_unwrap(model9)
            test_mse = 0.0; n_test = 0
            for xb, dxb in cart_test_loader:
                xb = xb.to(device); dxb = dxb.to(device)
                xb.requires_grad_(True)
                test_mse += nn.MSELoss()(raw_chnn.time_derivative(xb), dxb).item() * xb.size(0)
                n_test += xb.size(0)
            test_mse /= n_test
            print(f"\n  [CHNN] 测试 MSE (笛卡尔): {test_mse:.6e} | "
                  f"参数: {n_params_chnn:,}")
            results.append(('CHNN', test_mse, n_params_chnn, elapsed))
            torch.save(raw_chnn.state_dict(),
                       os.path.join(output_dir, 'chnn_checkpoint.pt'))

    if rank == 0:
        summarize_results(results, output_dir)

    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()