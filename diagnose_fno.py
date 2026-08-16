"""
FNO 诊断脚本: 打印输入输出对，检查数值尺度
===========================================
用法:
  只看数据统计:  python diagnose_fno.py --n_masses 20 --data_only
  加载模型诊断:  python diagnose_fno.py --ckpt results_N20/fno.pt --n_masses 20
"""

import os, argparse, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pendulum_string import DiscretePendulumString
from hnn_baselines import FNO, _maybe_unwrap, generate_multistep_data


def print_data_stats(loader, label, N):
    """打印数据集的统计信息"""
    print(f"\n{'='*60}")
    print(f"  {label} 数据统计")
    print(f"{'='*60}")

    all_x, all_dx = [], []
    for xb, dxb in loader:
        all_x.append(xb.numpy()); all_dx.append(dxb.numpy())
    X = np.concatenate(all_x, axis=0)
    dX = np.concatenate(all_dx, axis=0)

    theta, p = X[:, :N], X[:, N:]
    dtheta, dp = dX[:, :N], dX[:, N:]

    print(f"\n  输入 x = (θ, p):")
    print(f"    θ:  mean={theta.mean():.4f}, std={theta.std():.4f}, "
          f"min={theta.min():.4f}, max={theta.max():.4f}")
    print(f"    p:  mean={p.mean():.4f}, std={p.std():.4f}, "
          f"min={p.min():.4f}, max={p.max():.4f}")

    print(f"\n  时间导数 dx/dt = (dθ/dt, dp/dt):")
    print(f"    dθ/dt: mean={dtheta.mean():.4f}, std={dtheta.std():.4f}, "
          f"min={dtheta.min():.4f}, max={dtheta.max():.4f}")
    print(f"    dp/dt: mean={dp.mean():.4f}, std={dp.std():.4f}, "
          f"min={dp.min():.4f}, max={dp.max():.4f}")

    print(f"\n  逐位置 dθ/dt std (前5摆):")
    for i in range(min(5, N)):
        print(f"    摆{i}: std={dtheta[:,i].std():.4f}, "
              f"min={dtheta[:,i].min():.4f}, max={dtheta[:,i].max():.4f}")
    print(f"\n  逐位置 dp/dt std (前5摆):")
    for i in range(min(5, N)):
        print(f"    摆{i}: std={dp[:,i].std():.6f}, "
              f"min={dp[:,i].min():.6f}, max={dp[:,i].max():.6f}")

    return X, dX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_io(model, loader, N, device, output_dir='.'):
    """绘制 FNO 输入→输出函数图像"""
    raw = _maybe_unwrap(model)
    model.eval()

    # 收集所有测试数据
    all_x, all_dx = [], []
    for xb, dxb in loader:
        all_x.append(xb); all_dx.append(dxb)
    X = torch.cat(all_x, dim=0)
    dX_true = torch.cat(all_dx, dim=0)

    X_dev = X.to(device)
    X_dev.requires_grad_(True)
    with torch.no_grad():
        dX_pred = raw.time_derivative(X_dev).cpu()

    dX_true_np = dX_true.numpy()
    dX_pred_np = dX_pred.numpy()
    X_np = X.numpy()

    dtheta_true = dX_true_np[:, :N].ravel()
    dtheta_pred = dX_pred_np[:, :N].ravel()
    dp_true = dX_true_np[:, N:].ravel()
    dp_pred = dX_pred_np[:, N:].ravel()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1) 散点: dθ/dt 预测 vs 真实
    ax = axes[0, 0]
    ax.scatter(dtheta_true, dtheta_pred, alpha=0.3, s=4, color='C0')
    lim = max(abs(dtheta_true).max(), abs(dtheta_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('True dθ/dt'); ax.set_ylabel('Pred dθ/dt')
    mse_t = np.mean((dtheta_true - dtheta_pred) ** 2)
    ax.set_title(f'dθ/dt (MSE={mse_t:.4e})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # 2) 散点: dp/dt 预测 vs 真实
    ax = axes[0, 1]
    ax.scatter(dp_true, dp_pred, alpha=0.3, s=4, color='C3')
    lim = max(abs(dp_true).max(), abs(dp_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('True dp/dt'); ax.set_ylabel('Pred dp/dt')
    mse_p = np.mean((dp_true - dp_pred) ** 2)
    ax.set_title(f'dp/dt (MSE={mse_p:.4e})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # 3) 空间轮廓: 一个样本的 dθ/dt 和 dp/dt 沿摆链
    ax = axes[1, 0]
    si = 0  # 第一个样本
    idx = np.arange(N)
    ax.plot(idx, dX_true_np[si, :N], 'o-', color='C0', label='True dθ/dt', ms=5)
    ax.plot(idx, dX_pred_np[si, :N], 's--', color='C1', label='Pred dθ/dt', ms=5)
    ax.set_xlabel('Pendulum index'); ax.set_ylabel('dθ/dt')
    ax.set_title(f'Spatial profile (sample {si})')
    ax.legend(); ax.grid(alpha=0.3)

    # 4) 误差直方图
    ax = axes[1, 1]
    err_theta = dtheta_pred - dtheta_true
    err_p = dp_pred - dp_true
    ax.hist(err_theta, bins=60, alpha=0.5, label=f'dθ/dt err (std={err_theta.std():.3f})', color='C0')
    ax.hist(err_p, bins=60, alpha=0.5, label=f'dp/dt err (std={err_p.std():.5f})', color='C3')
    ax.set_xlabel('Prediction error'); ax.set_ylabel('Count')
    ax.set_title('Error distribution')
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f'FNO I/O Mapping | N={N} | {sum(p.numel() for p in model.parameters()):,} params',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(output_dir, 'fno_io_plot.png')
    plt.savefig(fname, dpi=150)
    print(f"\n  函数图像已保存: {fname}")
    plt.close()
def print_model_io(model, loader, N, device, n_samples=5):
    """打印模型的输入输出对比"""
    raw = _maybe_unwrap(model)
    model.eval()

    print(f"\n{'='*60}")
    print(f"  FNO 模型 I/O 诊断")
    print(f"{'='*60}")

    # 打印归一化参数
    print(f"\n  输入归一化 (mu, sigma) 前5摆:")
    print(f"    {'摆':>4s}  {'θ_mu':>10s}  {'θ_sigma':>10s}  "
          f"{'p_mu':>10s}  {'p_sigma':>10s}")
    for i in range(min(5, N)):
        print(f"    {i:4d}  {raw.mu[0,i].item():10.4f}  "
              f"{raw.sigma[0,i].item():10.4f}  "
              f"{raw.mu[1,i].item():10.4f}  "
              f"{raw.sigma[1,i].item():10.4f}")

    print(f"\n  输出归一化 (dx_mu, dx_sigma) 前5摆:")
    print(f"    {'摆':>4s}  {'dθ_mu':>10s}  {'dθ_sigma':>10s}  "
          f"{'dp_mu':>10s}  {'dp_sigma':>10s}")
    for i in range(min(5, N)):
        print(f"    {i:4d}  {raw.dx_mu[0,i].item():10.4f}  "
              f"{raw.dx_sigma[0,i].item():10.4f}  "
              f"{raw.dx_mu[1,i].item():10.6f}  "
              f"{raw.dx_sigma[1,i].item():10.6f}")

    # 收集样本
    xb_list, dxb_list = [], []
    for xb, dxb in loader:
        xb_list.append(xb); dxb_list.append(dxb)
    X = torch.cat(xb_list, dim=0)[:n_samples]
    dX_true = torch.cat(dxb_list, dim=0)[:n_samples]

    X_dev = X.to(device)
    X_dev.requires_grad_(True)
    with torch.no_grad():
        dX_pred = raw.time_derivative(X_dev).cpu()

    for s in range(n_samples):
        print(f"\n  ── 样本 {s} ──")
        print(f"    输入 θ[:5]: {[f'{v:.4f}' for v in X[s,:5].tolist()]}")
        print(f"    输入 p[:5]: {[f'{v:.4f}' for v in X[s,N:N+5].tolist()]}")

        print(f"    dθ/dt (前5):")
        print(f"      True: {[f'{v:.4f}' for v in dX_true[s,:5].tolist()]}")
        print(f"      Pred: {[f'{v:.4f}' for v in dX_pred[s,:5].tolist()]}")
        diffs = (dX_pred[s,:5] - dX_true[s,:5]).abs().tolist()
        print(f"      Diff: {[f'{v:.4f}' for v in diffs]}")

        print(f"    dp/dt (前5):")
        print(f"      True: {[f'{v:.6f}' for v in dX_true[s,N:N+5].tolist()]}")
        print(f"      Pred: {[f'{v:.6f}' for v in dX_pred[s,N:N+5].tolist()]}")
        diffs = (dX_pred[s,N:N+5] - dX_true[s,N:N+5]).abs().tolist()
        print(f"      Diff: {[f'{v:.6f}' for v in diffs]}")

    # 整体/分通道 MSE
    mse = nn.MSELoss()(dX_pred, dX_true)
    mse_theta = nn.MSELoss()(dX_pred[:,:N], dX_true[:,:N])
    mse_p = nn.MSELoss()(dX_pred[:,N:], dX_true[:,N:])
    print(f"\n  整体 MSE: {mse.item():.6e}")
    print(f"    dθ/dt MSE: {mse_theta.item():.6e}")
    print(f"    dp/dt MSE: {mse_p.item():.6e}")

    if mse.item() > 1e3:
        print(f"\n  ⚠️  MSE 极大 ({mse.item():.2e}), 归一化可能有问题!")

    print(f"\n  预测值尺度:")
    print(f"    dθ/dt pred: mean={dX_pred[:,:N].mean():.4f}, "
          f"std={dX_pred[:,:N].std():.4f}")
    print(f"    dp/dt pred: mean={dX_pred[:,N:].mean():.6f}, "
          f"std={dX_pred[:,N:].std():.6f}")
    print(f"    dθ/dt true: mean={dX_true[:,:N].mean():.4f}, "
          f"std={dX_true[:,:N].std():.4f}")
    print(f"    dp/dt true: mean={dX_true[:,N:].mean():.6f}, "
          f"std={dX_true[:,N:].std():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_masses', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default='',
                        help='FNO checkpoint (可选)')
    parser.add_argument('--data_only', action='store_true',
                        help='只打印数据统计')
    parser.add_argument('--n_trajectories', type=int, default=10)
    parser.add_argument('--fno_modes', type=int, default=12)
    parser.add_argument('--fno_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = args.n_masses
    print(f"设备: {device}, N={N}")

    sys_pend = DiscretePendulumString(n_masses=N)
    train_val, test_set = generate_multistep_data(
        sys_pend, n_trajectories=args.n_trajectories, seed=42)
    train_loader = DataLoader(train_val, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    print_data_stats(train_loader, "训练集", N)

    if args.data_only:
        print("\n✓ 数据统计完成 (--data_only)")
        return

    model = FNO(N=N, modes=args.fno_modes, hidden_dim=args.fno_hidden,
                num_layers=args.num_layers)
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"\n加载 checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd)
    else:
        print(f"\n⚠️ 未加载 checkpoint，使用随机初始化模型")

    model = model.to(device)
    model.compute_stats(train_loader)
    print_model_io(model, test_loader, N, device, n_samples=5)
    plot_io(model, test_loader, N, device)


if __name__ == '__main__':
    main()