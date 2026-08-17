"""
FNOFlow 诊断脚本: 检查输入输出映射，诊断模型学习质量
=====================================================
用法:
  只看数据统计:  python diagnose_fnoflow.py --n_masses 20 --data_only
  加载模型诊断:  python diagnose_fnoflow.py --ckpt results_N20/fnoflow_checkpoint.pt --n_masses 20
"""

import os, argparse, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pendulum_string import DiscretePendulumString
from hnn_baselines import FNOFlow, generate_multistep_data, _maybe_unwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_data_stats(loader, label, N):
    """打印 (x_t, x_{t+dt}) 数据集的统计信息"""
    print(f"\n{'='*60}")
    print(f"  {label} 数据统计")
    print(f"{'='*60}")

    all_x, all_y = [], []
    for xb, yb in loader:
        all_x.append(xb.numpy()); all_y.append(yb.numpy())
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)

    theta_in, p_in = X[:, :N], X[:, N:]
    theta_out, p_out = Y[:, :N], Y[:, N:]

    print(f"\n  输入 x_t = (θ_t, p_t):")
    print(f"    θ_t:  mean={theta_in.mean():.4f}, std={theta_in.std():.4f}, "
          f"min={theta_in.min():.4f}, max={theta_in.max():.4f}")
    print(f"    p_t:  mean={p_in.mean():.4f}, std={p_in.std():.4f}, "
          f"min={p_in.min():.4f}, max={p_in.max():.4f}")

    print(f"\n  输出 x_{{t+dt}} = (θ_{{t+dt}}, p_{{t+dt}}):")
    print(f"    θ_{{t+dt}}: mean={theta_out.mean():.4f}, std={theta_out.std():.4f}, "
          f"min={theta_out.min():.4f}, max={theta_out.max():.4f}")
    print(f"    p_{{t+dt}}: mean={p_out.mean():.4f}, std={p_out.std():.4f}, "
          f"min={p_out.min():.4f}, max={p_out.max():.4f}")

    # 差分: 实际位移
    dtheta = theta_out - theta_in
    dp = p_out - p_in
    print(f"\n  实际位移 Δx = x_{{t+dt}} - x_t:")
    print(f"    Δθ: mean={dtheta.mean():.6f}, std={dtheta.std():.6f}, "
          f"min={dtheta.min():.6f}, max={dtheta.max():.6f}")
    print(f"    Δp: mean={dp.mean():.6f}, std={dp.std():.6f}, "
          f"min={dp.min():.6f}, max={dp.max():.6f}")

    print(f"\n  逐位置 Δθ std (前5摆):")
    for i in range(min(5, N)):
        print(f"    摆{i}: std={dtheta[:,i].std():.6f}, "
              f"min={dtheta[:,i].min():.6f}, max={dtheta[:,i].max():.6f}")
    print(f"\n  逐位置 Δp std (前5摆):")
    for i in range(min(5, N)):
        print(f"    摆{i}: std={dp[:,i].std():.6f}, "
              f"min={dp[:,i].min():.6f}, max={dp[:,i].max():.6f}")

    # 如果输出≈输入，意味着模型只需要学 identity
    input_output_diff = np.abs(Y - X).mean()
    print(f"\n  ⚠  |x_{{t+dt}} - x_t| 均值: {input_output_diff:.6f}")
    if input_output_diff < 0.01:
        print(f"  ⚠  dt 太小! 输入输出几乎相同，模型只需学 identity")
    print(f"  (参考: 如果 dt=0.05 且 θ 变化小，说明系统慢，需要更大 dt)")

    return X, Y
def plot_io(model, loader, N, device, output_dir='.'):
    """绘制 FNOFlow 输入→输出函数图像"""
    raw = _maybe_unwrap(model)
    model.eval()

    all_x, all_y = [], []
    for xb, yb in loader:
        all_x.append(xb); all_y.append(yb)
    X = torch.cat(all_x, dim=0)
    Y_true = torch.cat(all_y, dim=0)

    X_dev = X.to(device)
    with torch.no_grad():
        Y_pred = raw.predict_next(X_dev).cpu()

    Y_true_np = Y_true.numpy()
    Y_pred_np = Y_pred.numpy()
    X_np = X.numpy()

    theta_in = X_np[:, :N].ravel()
    p_in = X_np[:, N:].ravel()
    theta_pred = Y_pred_np[:, :N].ravel()
    theta_true = Y_true_np[:, :N].ravel()
    p_pred = Y_pred_np[:, N:].ravel()
    p_true = Y_true_np[:, N:].ravel()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1) 散点: θ_{t+dt} 预测 vs 真实
    ax = axes[0, 0]
    ax.scatter(theta_true, theta_pred, alpha=0.3, s=4, color='C0')
    lim = max(abs(theta_true).max(), abs(theta_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('True theta_{t+dt}'); ax.set_ylabel('Pred theta_{t+dt}')
    mse_theta = np.mean((theta_true - theta_pred) ** 2)
    ax.set_title(f'theta scatter (MSE={mse_theta:.4e})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # 2) 散点: p_{t+dt} 预测 vs 真实
    ax = axes[0, 1]
    ax.scatter(p_true, p_pred, alpha=0.3, s=4, color='C3')
    lim = max(abs(p_true).max(), abs(p_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('True p_{t+dt}'); ax.set_ylabel('Pred p_{t+dt}')
    mse_p = np.mean((p_true - p_pred) ** 2)
    ax.set_title(f'p scatter (MSE={mse_p:.4e})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # 3) 散点: Δθ 预测 vs 真实
    ax = axes[0, 2]
    dtheta_true = theta_true - theta_in
    dtheta_pred = theta_pred - theta_in
    ax.scatter(dtheta_true, dtheta_pred, alpha=0.3, s=4, color='C2')
    lim = max(abs(dtheta_true).max(), abs(dtheta_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('True Delta_theta'); ax.set_ylabel('Pred Delta_theta')
    mse_dtheta = np.mean((dtheta_true - dtheta_pred) ** 2)
    ax.set_title(f'Delta_theta (MSE={mse_dtheta:.6f})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # 4) 空间剖面
    ax = axes[1, 0]
    si = np.random.randint(0, len(X_np))
    idx = np.arange(N)
    ax.plot(idx, Y_true_np[si, :N], 'o-', color='C0', label='True theta_{t+dt}', ms=5)
    ax.plot(idx, Y_pred_np[si, :N], 's--', color='C1', label='Pred theta_{t+dt}', ms=5)
    ax.plot(idx, X_np[si, :N], 'x:', color='gray', label='Input theta_t', ms=5)
    ax.set_xlabel('Pendulum index'); ax.set_ylabel('theta')
    ax.set_title(f'Spatial profile (sample {si})')
    ax.legend(); ax.grid(alpha=0.3)

    # 5) 误差直方图
    ax = axes[1, 1]
    err_theta = theta_pred - theta_true
    err_p = p_pred - p_true
    ax.hist(err_theta, bins=60, alpha=0.5, label=f'theta err (std={err_theta.std():.4f})', color='C0')
    ax.hist(err_p, bins=60, alpha=0.5, label=f'p err (std={err_p.std():.4f})', color='C3')
    ax.set_xlabel('Prediction error'); ax.set_ylabel('Count')
    ax.set_title('Error distribution')
    ax.legend(); ax.grid(alpha=0.3)

    # 6) 输入 vs 输出 mapping (identity check)
    ax = axes[1, 2]
    ax.scatter(theta_in, theta_true, alpha=0.2, s=2, color='gray', label='True')
    ax.scatter(theta_in, theta_pred, alpha=0.2, s=2, color='C1', label='Pred')
    ax.plot([-4, 4], [-4, 4], 'k--', lw=0.8, label='Identity')
    ax.set_xlim(theta_in.min(), theta_in.max())
    ax.set_ylim(theta_in.min(), theta_in.max())
    ax.set_xlabel('Input theta_t'); ax.set_ylabel('Output theta_{t+dt}')
    ax.set_title('Input-Output mapping')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    n_params = sum(p.numel() for p in model.parameters())
    fig.suptitle(f'FNOFlow I/O Mapping | N={N} | {n_params:,} params',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(output_dir, 'fnoflow_io_plot.png')
    plt.savefig(fname, dpi=150)
    print(f"\n  函数图像已保存: {fname}")
def print_model_io(model, loader, N, device, n_samples=5):
    """打印模型的输入输出对比"""
    raw = _maybe_unwrap(model)
    model.eval()

    print(f"\n{'='*60}")
    print(f"  FNOFlow 模型 I/O 诊断")
    print(f"{'='*60}")

    # 打印归一化参数
    print(f"\n  输入归一化 (mu, sigma) 前5摆:")
    print(f"    {'idx':>4s}  {'theta_mu':>10s}  {'theta_sigma':>10s}  "
          f"{'p_mu':>10s}  {'p_sigma':>10s}")
    for i in range(min(5, N)):
        print(f"    {i:4d}  {raw.mu[0,i].item():10.4f}  "
              f"{raw.sigma[0,i].item():10.4f}  "
              f"{raw.mu[1,i].item():10.4f}  "
              f"{raw.sigma[1,i].item():10.4f}")

    print(f"\n  输出归一化 (out_mu, out_sigma) 前5摆:")
    print(f"    {'idx':>4s}  {'theta_mu':>10s}  {'theta_sigma':>10s}  "
          f"{'p_mu':>10s}  {'p_sigma':>10s}")
    for i in range(min(5, N)):
        print(f"    {i:4d}  {raw.out_mu[0,i].item():10.4f}  "
              f"{raw.out_sigma[0,i].item():10.4f}  "
              f"{raw.out_mu[1,i].item():10.4f}  "
              f"{raw.out_sigma[1,i].item():10.4f}")
    plt.close()
# 收集样本
    xb_list, yb_list = [], []
    for xb, yb in loader:
        xb_list.append(xb); yb_list.append(yb)
    X = torch.cat(xb_list, dim=0)[:n_samples]
    Y_true = torch.cat(yb_list, dim=0)[:n_samples]

    X_dev = X.to(device)
    with torch.no_grad():
        Y_pred = raw.predict_next(X_dev).cpu()

    for s in range(n_samples):
        print(f"\n  -- 样本 {s} --")
        print(f"    输入 theta_t[:5]:  {[f'{v:.4f}' for v in X[s,:5].tolist()]}")
        print(f"    输入 p_t[:5]:      {[f'{v:.4f}' for v in X[s,N:N+5].tolist()]}")

        print(f"    theta(t+dt) (前5):")
        print(f"      True: {[f'{v:.4f}' for v in Y_true[s,:5].tolist()]}")
        print(f"      Pred: {[f'{v:.4f}' for v in Y_pred[s,:5].tolist()]}")
        diffs = (Y_pred[s,:5] - Y_true[s,:5]).abs().tolist()
        print(f"      Diff: {[f'{v:.4f}' for v in diffs]}")

        print(f"    p(t+dt) (前5):")
        print(f"      True: {[f'{v:.4f}' for v in Y_true[s,N:N+5].tolist()]}")
        print(f"      Pred: {[f'{v:.4f}' for v in Y_pred[s,N:N+5].tolist()]}")
        diffs = (Y_pred[s,N:N+5] - Y_true[s,N:N+5]).abs().tolist()
        print(f"      Diff: {[f'{v:.4f}' for v in diffs]}")

    # 整体/分通道 MSE
    mse = nn.MSELoss()(Y_pred, Y_true)
    mse_theta = nn.MSELoss()(Y_pred[:,:N], Y_true[:,:N])
    mse_p = nn.MSELoss()(Y_pred[:,N:], Y_true[:,N:])
    print(f"\n  整体 MSE: {mse:.6e}")
    print(f"  theta 通道 MSE: {mse_theta:.6e}")
    print(f"  p 通道 MSE: {mse_p:.6e}")

    # 和 identity baseline 对比
    identity_mse = nn.MSELoss()(X, Y_true)
    print(f"\n  Identity baseline MSE (x_t -> x_t): {identity_mse:.6e}")
    print(f"  Model / Identity: {mse/identity_mse:.4f}x")
    if mse >= identity_mse:
        print(f"  WARNING: 模型 MSE >= Identity MSE，模型可能没学到东西!")


def main():
    parser = argparse.ArgumentParser(description='FNOFlow 诊断工具')
    parser.add_argument('--n_masses', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint 路径')
    parser.add_argument('--data_only', action='store_true', help='仅打印数据统计')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--dt', type=float, default=0.05, help='数据步长')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    total_length = 1.0
    seg_length = total_length / args.n_masses
    total_mass = 1.0
    seg_mass = total_mass / args.n_masses
    sys = DiscretePendulumString(N=args.n_masses, l=seg_length, m=seg_mass, g=9.81)

    print(f"\n生成测试数据 (N={args.n_masses}, dt={args.dt})...")
    _, test_ds = generate_multistep_data(
        sys, n_trajectories=50, dt=args.dt, n_steps=1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    if args.data_only:
        print_data_stats(test_loader, 'FNOFlow 测试数据', args.n_masses)
        return

    if args.ckpt is None:
        print("ERROR: 需要 --ckpt 参数指定 checkpoint 路径")
        return

    print(f"\n加载 checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})

    N = config.get('N', args.n_masses)
    dt = config.get('dt', args.dt)
    modes = config.get('modes', 12)
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 3)

    print(f"  配置: N={N}, dt={dt}, modes={modes}, hidden={hidden_dim}, layers={num_layers}")

    model = FNOFlow(N=N, dt=dt, modes=modes, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

    print_data_stats(test_loader, 'FNOFlow 测试数据', N)
    print_model_io(model, test_loader, N, device, n_samples=5)
    plot_io(model, test_loader, N, device, output_dir=args.output_dir)


if __name__ == '__main__':
    main()