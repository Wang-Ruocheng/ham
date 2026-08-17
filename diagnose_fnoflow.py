"""
FNOFlow 诊断: 单组数据 I/O 曲线
==============================
用法:
  python diagnose_fnoflow.py --ckpt results_N20/fnoflow_checkpoint.pt --n_masses 20
"""

import os, argparse, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pendulum_string import DiscretePendulumString
from hnn_baselines import FNOFlow, generate_multistep_data, _maybe_unwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='FNOFlow 诊断')
    parser.add_argument('--n_masses', type=int, default=20)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    N = config.get('N', args.n_masses)
    dt = config.get('dt', 0.05)
    modes = config.get('modes', 12)
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 3)

    model = FNOFlow(N=N, dt=dt, modes=modes, hidden_dim=hidden_dim,
                    num_layers=num_layers)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  N={N}, dt={dt}, params={sum(p.numel() for p in model.parameters()):,}")

    # 检查归一化参数
    raw = _maybe_unwrap(model)
    print(f"\n  归一化参数检查:")
    print(f"    输入 mu   (theta): {raw.mu[0].mean().item():.4f} +/- {raw.mu[0].std().item():.4f}")
    print(f"    输入 sigma(theta): {raw.sigma[0].mean().item():.4f} +/- {raw.sigma[0].std().item():.4f}")
    print(f"    输入 mu   (p):     {raw.mu[1].mean().item():.4f} +/- {raw.mu[1].std().item():.4f}")
    print(f"    输入 sigma(p):     {raw.sigma[1].mean().item():.4f} +/- {raw.sigma[1].std().item():.4f}")
    print(f"    输出 mu   (theta): {raw.out_mu[0].mean().item():.4f} +/- {raw.out_mu[0].std().item():.4f}")
    print(f"    输出 sigma(theta): {raw.out_sigma[0].mean().item():.4f} +/- {raw.out_sigma[0].std().item():.4f}")
    print(f"    输出 mu   (p):     {raw.out_mu[1].mean().item():.4f} +/- {raw.out_mu[1].std().item():.4f}")
    print(f"    输出 sigma(p):     {raw.out_sigma[1].mean().item():.4f} +/- {raw.out_sigma[1].std().item():.4f}")

    # 生成测试数据，取第一个样本
    sys = DiscretePendulumString(n_masses=N, length=1.0, mass=1.0, g=9.81)
    _, test_ds = generate_multistep_data(
        sys, n_trajectories=10, dt=dt, n_steps=1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    xb, yb = next(iter(test_loader))
    x_np = xb[0].numpy()
    y_np = yb[0].numpy()

    with torch.no_grad():
        y_pred = _maybe_unwrap(model).predict_next(xb.to(device)).cpu()[0].numpy()

    # 打印统计
    print(f"\n  输入 x_t:   theta std={x_np[:N].std():.4f}, p std={x_np[N:].std():.4f}")
    print(f"  真实 x_t+dt: theta std={y_np[:N].std():.4f}, p std={y_np[N:].std():.4f}")
    print(f"  预测 x_t+dt: theta std={y_pred[:N].std():.4f}, p std={y_pred[N:].std():.4f}")
    mse = np.mean((y_pred - y_np) ** 2)
    id_mse = np.mean((x_np - y_np) ** 2)
    print(f"  MSE: {mse:.6e}  |  Identity MSE: {id_mse:.6e}  |  ratio: {mse/id_mse:.4f}")

    # 绘图
    idx = np.arange(N)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(idx, x_np[:N], 'x--', color='gray', label='Input theta_t', ms=6)
    ax1.plot(idx, y_np[:N], 'o-', color='C0', label='True theta_{t+dt}', ms=5)
    ax1.plot(idx, y_pred[:N], 's-', color='C1', label='Pred theta_{t+dt}', ms=5)
    ax1.set_ylabel('theta')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(f'FNOFlow I/O | N={N}, dt={dt} | MSE={mse:.4e}')

    ax2.plot(idx, x_np[N:], 'x--', color='gray', label='Input p_t', ms=6)
    ax2.plot(idx, y_np[N:], 'o-', color='C0', label='True p_{t+dt}', ms=5)
    ax2.plot(idx, y_pred[N:], 's-', color='C1', label='Pred p_{t+dt}', ms=5)
    ax2.set_xlabel('Node index i')
    ax2.set_ylabel('p')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(args.output_dir, 'fnoflow_io_curve.png')
    plt.savefig(fname, dpi=150)
    print(f"  已保存: {fname}")
    plt.close()


if __name__ == '__main__':
    main()