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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pendulum_string import DiscretePendulumString
from hnn_baselines import (
    SeparableHNN, PartialHNN, SIREN_HNN, SymplecticHNN,
    train_single_step, train_symplectic,
    generate_multistep_data,
    evaluate_and_visualize
)


def load_or_generate_data(args):
    """加载或生成训练数据"""
    data_path = f'pendulum_string_data_N{args.n_masses}.pt'
    if os.path.exists(data_path):
        print(f"  加载已保存的数据: {data_path}")
        train_ds, val_ds, test_ds = torch.load(data_path, weights_only=False)
    else:
        sys = DiscretePendulumString(n_masses=args.n_masses,
                                     length=args.length, mass=args.mass, g=args.g)
        train_ds, val_ds, test_ds = sys.generate_dataset(
            n_trajectories=args.n_trajectories, seed=args.seed)
        torch.save((train_ds, val_ds, test_ds), data_path)
        print(f"  数据已保存: {data_path}")
    return train_ds, val_ds, test_ds
def summarize_results(results):
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
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

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
    plt.savefig('baseline_comparison.png', dpi=150)
    print("对比图已保存: baseline_comparison.png")
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='HNN Baseline Runner')
    parser.add_argument('--n_masses', type=int, default=10)
    parser.add_argument('--length', type=float, default=1.0)
    parser.add_argument('--mass', type=float, default=1.0)
    parser.add_argument('--g', type=float, default=9.81)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_trajectories', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_symplectic', action='store_true',
                        help='跳过多步辛训练')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'separable', 'partial', 'siren', 'symplectic'],
                        help='单独运行某个模型 (default: all)')
    args = parser.parse_args()

    dim = 2 * args.n_masses
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"HNN Baseline 对比: N={args.n_masses}, dim={dim}")
    print(f"MLP: {dim} -> {args.hidden_dim}x{args.num_layers} -> 1")
    print(f"设备: {device}, epochs: {args.epochs}")
    print("=" * 80)

    print("\n--- 加载数据 ---")
    train_ds, val_ds, test_ds = load_or_generate_data(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sys = DiscretePendulumString(n_masses=args.n_masses,
                                 length=args.length, mass=args.mass, g=args.g)
    ml2 = args.mass * args.length ** 2

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    results = []

    # ── 1. SeparableHNN ─────────────────────────────────────
    if args.model in ('all', 'separable'):
        print("\n" + "=" * 80)
        print("1/4: SeparableHNN — H = T(p) + V(theta)")
        print("=" * 80)
        model1 = SeparableHNN(N=args.n_masses, hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers)
        t0 = time.time()
        train_single_step(model1, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[Separable]',
                          compute_stats_fn=lambda m, l: m.compute_stats(l))
        elapsed = time.time() - t0
        mse1, p1 = evaluate_and_visualize(model1, test_loader, sys, args, device, 'SeparableHNN')
        results.append(('SeparableHNN', mse1, p1, elapsed))

    # ── 2. PartialHNN ───────────────────────────────────────
    if args.model in ('all', 'partial'):
        print("\n" + "=" * 80)
        print("2/4: PartialHNN — 已知 M(theta), 仅学 V(theta)")
        print("=" * 80)
        model2 = PartialHNN(N=args.n_masses, ml2=ml2, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers)
        t0 = time.time()
        train_single_step(model2, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[Partial]',
                          compute_stats_fn=lambda m, l: m.compute_stats(l))
        elapsed = time.time() - t0
        mse2, p2 = evaluate_and_visualize(model2, test_loader, sys, args, device, 'PartialHNN')
        results.append(('PartialHNN', mse2, p2, elapsed))

    # ── 3. SIREN_HNN ────────────────────────────────────────
    if args.model in ('all', 'siren'):
        print("\n" + "=" * 80)
        print("3/4: SIREN_HNN — sin 激活函数")
        print("=" * 80)
        model3 = SIREN_HNN(dim=dim, hidden_dim=args.hidden_dim,
                           num_layers=args.num_layers)
        t0 = time.time()
        train_single_step(model3, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, device=device,
                          label='[SIREN]',
                          compute_stats_fn=lambda m, l: m.compute_stats(l))
        elapsed = time.time() - t0
        mse3, p3 = evaluate_and_visualize(model3, test_loader, sys, args, device, 'SIREN_HNN')
        results.append(('SIREN_HNN', mse3, p3, elapsed))

    # ── 4. SymplecticHNN ────────────────────────────────────
    if args.model in ('all', 'symplectic'):
        print("\n" + "=" * 80)
        print("4/4: SymplecticHNN — Separable + 多步辛训练")
        print("=" * 80)
        if args.skip_symplectic:
            print("  [跳过] --skip_symplectic")
            results.append(('SymplecticHNN', float('nan'), 0, 0))
        else:
            print("  生成多步训练数据...")
            n_steps = 5
            dt = 20.0 / 300 * n_steps
            mstep_train, mstep_val = generate_multistep_data(
                sys, n_trajectories=args.n_trajectories,
                n_steps=n_steps, seed=args.seed)
            mstep_train_loader = DataLoader(mstep_train, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)

            model4 = SymplecticHNN(N=args.n_masses, hidden_dim=args.hidden_dim,
                                   num_layers=args.num_layers)
            t0 = time.time()
            train_symplectic(model4, mstep_train_loader, val_loader,
                             n_steps=n_steps, dt=dt,
                             epochs=args.epochs, lr=args.lr, device=device,
                             label='[Symplectic]')
            elapsed = time.time() - t0
            mse4, p4 = evaluate_and_visualize(model4, test_loader, sys, args, device, 'SymplecticHNN')
            results.append(('SymplecticHNN', mse4, p4, elapsed))

    summarize_results(results)


if __name__ == '__main__':
    main()