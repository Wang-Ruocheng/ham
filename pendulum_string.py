"""
离散摆绳：N 个耦合谐振子链
=============================
物理: 弦离散化为 N 个质点，每个质点受重力(摆) + 弹簧连接
H = sum[½p_i² + ½ω₀²q_i²] + ½k sum(q_{i+1} - q_i)²

状态: x = [q_1, ..., q_N, p_1, ..., p_N] ∈ ℝ^{2N}
辛矩阵: J = [[0, I], [-I, 0]]
动力学: dx/dt = J·∇H

训练:
    单卡:  python pendulum_string.py --n_masses 20
    8卡DDP: torchrun --nproc_per_node=8 pendulum_string.py --n_masses 20 --ddp
"""

import os
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt


# ============================================================
# 离散摆绳物理系统
# ============================================================
class DiscretePendulumString:
    def __init__(self, n_masses=20, omega0=1.0, spring_k=10.0):
        self.N = n_masses
        self.omega0 = omega0
        self.k = spring_k

    def hamiltonian(self, state):
        q = state[:self.N]; p = state[self.N:]
        H_pend = 0.5 * (p**2).sum() + 0.5 * self.omega0**2 * (q**2).sum()
        dq = np.diff(q, prepend=0.0, append=0.0)
        H_spring = 0.5 * self.k * (dq**2).sum()
        return H_pend + H_spring

    def dynamics(self, t, state):
        q = state[:self.N]; p = state[self.N:]
        dqdt = p
        dpdt = -self.omega0**2 * q
        dpdt[0] -= self.k * (2*q[0] - q[1])
        for i in range(1, self.N - 1):
            dpdt[i] -= self.k * (2*q[i] - q[i-1] - q[i+1])
        dpdt[-1] -= self.k * (2*q[-1] - q[-2])
        return np.concatenate([dqdt, dpdt])

    def generate_trajectory(self, state0, t_span=(0, 20), n_points=300):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, state0,
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        return sol.t, sol.y.T

    def generate_dataset(self, n_trajectories=200, t_span=(0, 20),
                         n_points=300, train_ratio=0.7, val_ratio=0.15, seed=42):
        np.random.seed(seed)
        xs_list, dxs_list = [], []
        for _ in range(n_trajectories):
            modes = np.random.randn(self.N) * np.exp(-np.arange(self.N) / 5)
            state0 = np.concatenate([modes * 0.5, np.random.randn(self.N) * 0.3])
            _, traj = self.generate_trajectory(state0, t_span, n_points)
            for i in range(len(traj)):
                dx = self.dynamics(t_span[0], traj[i])
                xs_list.append(traj[i]); dxs_list.append(dx)
        xs = np.stack(xs_list); dxs = np.stack(dxs_list)
        n_total = len(xs)
        indices = np.random.permutation(n_total)
        n_train = int(train_ratio * n_total); n_val = int(val_ratio * n_total)
        xs_t = torch.tensor(xs, dtype=torch.float32)
        dxs_t = torch.tensor(dxs, dtype=torch.float32)
        train_ds = TensorDataset(xs_t[indices[:n_train]], dxs_t[indices[:n_train]])
        val_ds = TensorDataset(xs_t[indices[n_train:n_train+n_val]],
                               dxs_t[indices[n_train:n_train+n_val]])
        test_ds = TensorDataset(xs_t[indices[n_train+n_val:]],
                                dxs_t[indices[n_train+n_val:]])
        print(f"数据: {n_total} 点 | 训练: {n_train} | 验证: {n_val} | "
              f"测试: {n_total - n_train - n_val}")
        return train_ds, val_ds, test_ds
# ============================================================
# 标准 HNN
# ============================================================
class StandardHNN(nn.Module):
    def __init__(self, dim, hidden_dim=512, num_layers=4):
        super().__init__()
        self.dim = dim; self.N = dim // 2
        layers = []
        prev = dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1, bias=False))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def time_derivative(self, x):
        with torch.enable_grad():
            z = x.clone().requires_grad_(True)
            H = self.forward(z)
            dH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        dq = dH[:, self.N:]; dp = -dH[:, :self.N]
        return torch.cat([dq, dp], dim=1)


# ============================================================
# 训练函数
# ============================================================
def train_single_gpu(model, train_loader, val_loader, epochs=2000, lr=1e-3,
                     device='cuda', label=''):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, dxb in train_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model.time_derivative(xb), dxb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, dxb in val_loader:
                xb, dxb = xb.to(device), dxb.to(device)
                val_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
def train_ddp(args, train_ds, val_ds, test_ds):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')
    device = local_rank

    model = StandardHNN(dim=2 * args.n_masses,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        for xb, dxb in train_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model.module.time_derivative(xb), dxb)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * xb.size(0)
        torch.distributed.all_reduce(train_loss)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for xb, dxb in val_loader:
                val_loss += nn.MSELoss()(model.module.time_derivative(xb.to(device)), dxb.to(device)) * xb.size(0)
        torch.distributed.all_reduce(val_loss)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss.item())

        if rank == 0 and (epoch + 1) % 200 == 0:
            print(f"  [DDP-{world_size}GPU] Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Train: {train_loss.item():.6e} | Val: {val_loss.item():.6e}")

    if rank == 0:
        model.eval()
        test_mse = 0.0; n_test = 0
        with torch.no_grad():
            for xb, dxb in test_loader:
                xb, dxb = xb.to(device), dxb.to(device)
                test_mse += nn.MSELoss()(model.module.time_derivative(xb), dxb).item() * xb.size(0)
                n_test += xb.size(0)
        test_mse /= n_test
        print(f"\n  [DDP] 测试 MSE: {test_mse:.6e}")

        # --- 轨迹预测与可视化 ---
        print("\n--- 轨迹预测 ---")
        sys = DiscretePendulumString(n_masses=args.n_masses,
                                     omega0=args.omega0, spring_k=args.spring_k)
        n_params = sum(p.numel() for p in model.module.parameters())
        t_span = (0, 40); n_points = 1500; t_eval = np.linspace(*t_span, n_points)
        modes = np.random.randn(args.n_masses) * np.exp(-np.arange(args.n_masses) / 5)
        state0 = np.concatenate([modes * 0.5, np.random.randn(args.n_masses) * 0.3])
        _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
        pred_traj = integrate_rk4(model.module, state0, t_span, n_points, device)
        H_true = np.array([sys.hamiltonian(true_traj[i]) for i in range(n_points)])
        H_pred = np.array([sys.hamiltonian(pred_traj[i]) for i in range(n_points)])
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes[0, 0].imshow(true_traj[:, :args.n_masses].T[:10],
                          aspect='auto', origin='lower', cmap='RdBu_r',
                          extent=[t_span[0], t_span[1], 0, 10])
        axes[0, 0].set_title('True q (first 10)'); axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('mass index')
        axes[0, 1].imshow(pred_traj[:, :args.n_masses].T[:10],
                          aspect='auto', origin='lower', cmap='RdBu_r',
                          extent=[t_span[0], t_span[1], 0, 10])
        axes[0, 1].set_title('HNN Predicted q (first 10)'); axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('mass index')
        axes[0, 2].imshow(np.abs(pred_traj[:, :args.n_masses] - true_traj[:, :args.n_masses]).T[:10],
                          aspect='auto', origin='lower', cmap='hot',
                          extent=[t_span[0], t_span[1], 0, 10])
        axes[0, 2].set_title('|Error| (first 10)'); axes[0, 2].set_xlabel('t')
        axes[0, 2].set_ylabel('mass index')
        mid = args.n_masses // 2
        axes[1, 0].plot(t_eval, true_traj[:, mid], 'k-', lw=2, label=f'True q_{mid}')
        axes[1, 0].plot(t_eval, pred_traj[:, mid], 'r--', lw=1.5, label=f'HNN q_{mid}')
        axes[1, 0].set_title(f'Mass {mid} trajectory'); axes[1, 0].set_xlabel('t')
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
        axes[1, 1].plot(t_eval, H_true, 'k-', lw=2, label='H_true')
        axes[1, 1].plot(t_eval, H_pred, 'r--', lw=1.5, label='H_HNN')
        axes[1, 1].set_title('Hamiltonian Conservation'); axes[1, 1].set_xlabel('t')
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
        snapshot_times = [0.0, 10.0, 20.0, 30.0, 40.0]
        snap_idx = [int(t / 40.0 * (n_points - 1)) for t in snapshot_times]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(snapshot_times)))
        for ti, (t, idx, c) in enumerate(zip(snapshot_times, snap_idx, colors)):
            mass_idx = np.arange(args.n_masses)
            axes[1, 2].plot(mass_idx, true_traj[idx, :args.n_masses],
                            '-', color=c, lw=2, label=f't={t}s (true)' if ti == 0 else None)
            axes[1, 2].plot(mass_idx, pred_traj[idx, :args.n_masses],
                            '--', color=c, lw=1.5, label=f't={t}s (HNN)' if ti == 0 else None)
        axes[1, 2].set_title('Pendulum String Snapshots')
        axes[1, 2].set_xlabel('mass index'); axes[1, 2].set_ylabel('q')
        axes[1, 2].legend(fontsize=8, ncol=2); axes[1, 2].grid(alpha=0.3)
        fig.suptitle(f'Discrete Pendulum String: N={args.n_masses}, dim={2*args.n_masses}, Params={n_params:,}', fontsize=14)
        plt.tight_layout()
        fig.savefig('pendulum_string.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(" -> pendulum_string.png")
        print(f"\n{'='*70}")
        print(f"完成 | N={args.n_masses} | dim={2*args.n_masses} | 参数={n_params:,} | MSE={test_mse:.4e}")
        print("=" * 70)

    torch.distributed.destroy_process_group()
# ============================================================
# 评估: 轨迹预测
# ============================================================
def integrate_rk4(model, state0, t_span, n_steps, device='cuda'):
    model.eval()
    dt = (t_span[1] - t_span[0]) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D)); traj[0] = state0
    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32, device=device)
        k1 = model.time_derivative(x).detach().cpu().numpy()[0]
        k2 = model.time_derivative(x + 0.5*dt*torch.tensor(k1, device=device, dtype=torch.float32)).detach().cpu().numpy()[0]
        k3 = model.time_derivative(x + 0.5*dt*torch.tensor(k2, device=device, dtype=torch.float32)).detach().cpu().numpy()[0]
        k4 = model.time_derivative(x + dt*torch.tensor(k3, device=device, dtype=torch.float32)).detach().cpu().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='离散摆绳 HNN')
    parser.add_argument('--n_masses', type=int, default=20)
    parser.add_argument('--omega0', type=float, default=1.0)
    parser.add_argument('--spring_k', type=float, default=10.0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_trajectories', type=int, default=200)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    dim = 2 * args.n_masses

    print("=" * 70)
    print(f"离散摆绳 HNN: N={args.n_masses} 质点, 状态维度={dim}")
    print(f"omega0={args.omega0}, k={args.spring_k}")
    print(f"MLP: {dim} -> {args.hidden_dim}x{args.num_layers} -> 1")
    print(f"训练: {'DDP' if args.ddp else '单卡'}, {args.epochs} epochs")
    print("=" * 70)

    print("\n--- 生成数据 ---")
    sys = DiscretePendulumString(n_masses=args.n_masses,
                                 omega0=args.omega0, spring_k=args.spring_k)
    train_ds, val_ds, test_ds = sys.generate_dataset(
        n_trajectories=args.n_trajectories, seed=args.seed)

    if args.ddp:
        print(f"\n检测到 DDP 模式，由 torchrun 管理进程")
        train_ddp(args, train_ds, val_ds, test_ds)
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = StandardHNN(dim=dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    train_losses, val_losses = train_single_gpu(
        model, train_loader, val_loader, epochs=args.epochs,
        lr=args.lr, device=device, label=f"[N={args.n_masses}]")

    model.eval()
    test_mse = 0.0; n_test = 0
    with torch.no_grad():
        for xb, dxb in test_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            test_mse += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
            n_test += xb.size(0)
    test_mse /= n_test
    print(f"\n测试 MSE: {test_mse:.6e}")

    print("\n--- 轨迹预测 ---")
    t_span = (0, 40); n_points = 1500; t_eval = np.linspace(*t_span, n_points)
    modes = np.random.randn(args.n_masses) * np.exp(-np.arange(args.n_masses) / 5)
    state0 = np.concatenate([modes * 0.5, np.random.randn(args.n_masses) * 0.3])

    _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
    pred_traj = integrate_rk4(model, state0, t_span, n_points, device)

    H_true = np.array([sys.hamiltonian(true_traj[i]) for i in range(n_points)])
    H_pred = np.array([sys.hamiltonian(pred_traj[i]) for i in range(n_points)])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    im1 = axes[0, 0].imshow(true_traj[:, :args.n_masses].T[:10],
                            aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[t_span[0], t_span[1], 0, 10])
    axes[0, 0].set_title(f'True q (first 10)'); axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('mass index'); plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(pred_traj[:, :args.n_masses].T[:10],
                            aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[t_span[0], t_span[1], 0, 10])
    axes[0, 1].set_title('HNN Predicted q (first 10)'); axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('mass index'); plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(
        np.abs(pred_traj[:, :args.n_masses] - true_traj[:, :args.n_masses]).T[:10],
        aspect='auto', origin='lower', cmap='hot',
        extent=[t_span[0], t_span[1], 0, 10])
    axes[0, 2].set_title('|Error| (first 10)'); axes[0, 2].set_xlabel('t')
    axes[0, 2].set_ylabel('mass index'); plt.colorbar(im3, ax=axes[0, 2])

    mid = args.n_masses // 2
    axes[1, 0].plot(t_eval, true_traj[:, mid], 'k-', lw=2, label=f'True q_{mid}')
    axes[1, 0].plot(t_eval, pred_traj[:, mid], 'r--', lw=1.5, label=f'HNN q_{mid}')
    axes[1, 0].set_title(f'Mass {mid} trajectory'); axes[1, 0].set_xlabel('t')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t_eval, H_true, 'k-', lw=2, label='H_true')
    axes[1, 1].plot(t_eval, H_pred, 'r--', lw=1.5, label='H_HNN')
    axes[1, 1].set_title('Hamiltonian Conservation'); axes[1, 1].set_xlabel('t')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    snapshot_times = [0.0, 10.0, 20.0, 30.0, 40.0]
    snap_idx = [int(t / 40.0 * (n_points - 1)) for t in snapshot_times]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(snapshot_times)))
    for ti, (t, idx, c) in enumerate(zip(snapshot_times, snap_idx, colors)):
        mass_idx = np.arange(args.n_masses)
        axes[1, 2].plot(mass_idx, true_traj[idx, :args.n_masses],
                        '-', color=c, lw=2, label=f't={t}s (true)' if ti == 0 else None)
        axes[1, 2].plot(mass_idx, pred_traj[idx, :args.n_masses],
                        '--', color=c, lw=1.5, label=f't={t}s (HNN)' if ti == 0 else None)
    axes[1, 2].set_title('Pendulum String Snapshots')
    axes[1, 2].set_xlabel('mass index'); axes[1, 2].set_ylabel('q')
    axes[1, 2].legend(fontsize=8, ncol=2); axes[1, 2].grid(alpha=0.3)

    fig.suptitle(f'Discrete Pendulum String: N={args.n_masses}, dim={dim}, Params={n_params:,}', fontsize=14)
    plt.tight_layout()
    fig.savefig('pendulum_string.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" -> pendulum_string.png")

    print(f"\n{'='*70}")
    print(f"完成 | N={args.n_masses} | dim={dim} | 参数={n_params:,} | MSE={test_mse:.4e}")
    print("=" * 70)


if __name__ == '__main__':
    main()