"""
重力摆链：N 节摆，一端固定，全非线性动力学
============================================
物理: N 个质点由刚性杆连接，悬挂于固定点，受重力。
      每个质点位置由该质点及之前所有摆角决定。
      非小角度近似，使用 sin/cos 精确计算。

状态: x = [θ₀, …, θ_{N-1}, p_θ₀, …, p_θ_{N-1}] ∈ ℝ^{2N}
      其中 θ_i 为第 i 节杆与竖直方向的夹角 (rad)
      p_θ_i 为共轭动量

惯性矩阵: M_{ij} = m l² (N - max(i,j)) cos(θ_i - θ_j)
势能: V = -m g l Σ_k (N - k) cos(θ_k)
哈密顿量: H = ½ p^T M^{-1} p + V

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
# 重力摆链物理系统
# ============================================================
class DiscretePendulumString:
    """N 节重力摆链（一端固定，全非线性）"""
    def __init__(self, n_masses=20, length=1.0, mass=1.0, g=9.81):
        self.N = n_masses
        self.l = length          # 每节杆长
        self.m = mass            # 每个质点质量
        self.g = g               # 重力加速度
        self.ml2 = mass * length**2  # m·l²

    # ── 惯性矩阵 M(θ) ──────────────────────────────────────
    def inertia_matrix(self, theta):
        """M_{ij} = m l² (N - max(i,j)) cos(θ_i - θ_j)"""
        N = self.N
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        k = N - np.maximum(i, j)
        return self.ml2 * k * np.cos(theta[i] - theta[j])

    def d_inertia_dtheta(self, theta, idx):
        """∂M/∂θ_i (i=idx)"""
        N = self.N
        dM = np.zeros((N, N))
        for k in range(N):
            if k == idx:
                continue
            val = -self.ml2 * (N - max(idx, k)) * np.sin(theta[idx] - theta[k])
            dM[idx, k] = val
            dM[k, idx] = val
        return dM

    # ── 势能 V(θ) ──────────────────────────────────────────
    def potential(self, theta):
        """V = -m g l Σ_k (N - k) cos(θ_k)"""
        k = np.arange(self.N)
        return -self.m * self.g * self.l * np.sum((self.N - k) * np.cos(theta))

    def d_potential_dtheta(self, theta):
        """∂V/∂θ_i = m g l (N - i) sin(θ_i)"""
        k = np.arange(self.N)
        return self.m * self.g * self.l * (self.N - k) * np.sin(theta)

    # ── 哈密顿量与动力学 ────────────────────────────────────
    def hamiltonian(self, state):
        theta = state[:self.N]; p = state[self.N:]
        M = self.inertia_matrix(theta)
        T = 0.5 * p @ np.linalg.solve(M, p)
        V = self.potential(theta)
        return T + V

    def dynamics(self, t, state):
        theta = state[:self.N]; p = state[self.N:]
        N = self.N
        M = self.inertia_matrix(theta)
        theta_dot = np.linalg.solve(M, p)       # θ̇ = M^{-1} p
        dV = self.d_potential_dtheta(theta)
        v = theta_dot
        p_dot = np.zeros(N)
        for i in range(N):
            dM = self.d_inertia_dtheta(theta, i)
            p_dot[i] = -0.5 * v @ dM @ v - dV[i]
        return np.concatenate([theta_dot, p_dot])

    # ── 质点位置（用于可视化） ──────────────────────────────
    def get_positions(self, theta):
        """返回每质点 (x, y) 坐标，固定点在 (0,0)"""
        N = self.N
        x = np.zeros(N)
        y = np.zeros(N)
        for k in range(N):
            x[k] = self.l * np.sum(np.sin(theta[:k+1]))
            y[k] = -self.l * np.sum(np.cos(theta[:k+1]))
        return x, y

    # ── 轨迹与数据集 ────────────────────────────────────────
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
            theta0 = np.random.uniform(-1.0, 1.0, self.N)
            omega0 = np.random.uniform(-1.0, 1.0, self.N)
            M0 = self.inertia_matrix(theta0)
            p0 = M0 @ omega0
            state0 = np.concatenate([theta0, p0])
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
        print(f"  数据集: {n_total} 样本 | 训练 {n_train} | 验证 {n_val} | 测试 {n_total-n_train-n_val}")
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def time_derivative(self, x):
        H = self.forward(x)
        grad = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        theta_grad, p_grad = grad.chunk(2, dim=-1)
        return torch.cat([p_grad, -theta_grad], dim=-1)


# ============================================================
# 单卡训练
# ============================================================
def train_single_gpu(model, train_loader, val_loader, epochs=2000,
                     lr=1e-3, device='cuda', label=''):
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
            xb.requires_grad_(True)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model.time_derivative(xb), dxb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, dxb in val_loader:
                xb = xb.to(device); dxb = dxb.to(device)
                val_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


# ============================================================
# DDP 训练
# ============================================================
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
            xb.requires_grad_(True)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model.module.time_derivative(xb), dxb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
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

        print("\n--- 轨迹预测 ---")
        n_params = sum(p.numel() for p in model.module.parameters())
        t_span = (0, 40); n_points = 1500
        sys = DiscretePendulumString(n_masses=args.n_masses,
                                     length=args.length, mass=args.mass, g=args.g)
        # 初始条件: 拨动顶端摆杆 0.3 rad，零初速
        theta0 = np.zeros(args.n_masses); theta0[0] = 0.3
        p0 = np.zeros(args.n_masses)
        state0 = np.concatenate([theta0, p0])
        _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
        pred_traj = integrate_rk4(model.module, state0, t_span, n_points, device)
        visualize(args, sys, true_traj, pred_traj, test_mse, n_params, t_span, n_points)
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


def visualize(args, sys, true_traj, pred_traj, test_mse, n_params, t_span=(0, 40), n_points=1500):
    """生成 2×3 可视化图：时空图 + 轨迹 + 2D 摆链快照"""
    from matplotlib.patches import Circle
    t_eval = np.linspace(*t_span, n_points)
    N = args.n_masses

    H_true = np.array([sys.hamiltonian(true_traj[i]) for i in range(n_points)])
    H_pred = np.array([sys.hamiltonian(pred_traj[i]) for i in range(n_points)])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- 第 0 行: 时空图 (θ 的前 10 个) ---
    n_show = min(10, N)
    im1 = axes[0, 0].imshow(true_traj[:, :N].T[:n_show],
                            aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[t_span[0], t_span[1], 0, n_show])
    axes[0, 0].set_title('True θ (first {})'.format(n_show)); axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('mass index'); plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(pred_traj[:, :N].T[:n_show],
                            aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[t_span[0], t_span[1], 0, n_show])
    axes[0, 1].set_title('HNN Predicted θ (first {})'.format(n_show)); axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('mass index'); plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(
        np.abs(pred_traj[:, :N] - true_traj[:, :N]).T[:n_show],
        aspect='auto', origin='lower', cmap='hot',
        extent=[t_span[0], t_span[1], 0, n_show])
    axes[0, 2].set_title('|Error| (first {})'.format(n_show)); axes[0, 2].set_xlabel('t')
    axes[0, 2].set_ylabel('mass index'); plt.colorbar(im3, ax=axes[0, 2])

    # --- 第 1 行: 轨迹 + 哈密顿量 + 2D 快照 ---
    # 某质点轨迹
    idx = 0  # 顶端质点
    axes[1, 0].plot(t_eval, true_traj[:, idx], label='True θ₀', color='C0')
    axes[1, 0].plot(t_eval, pred_traj[:, idx], '--', label='HNN θ₀', color='C1')
    axes[1, 0].set_title(f'Angle θ_{idx}'); axes[1, 0].set_xlabel('t'); axes[1, 0].set_ylabel('θ (rad)')
    axes[1, 0].legend()

    # 哈密顿量守恒
    axes[1, 1].plot(t_eval, H_true, label='H_true', color='C0')
    axes[1, 1].plot(t_eval, H_pred, '--', label='H_HNN', color='C1')
    axes[1, 1].set_title('Hamiltonian'); axes[1, 1].set_xlabel('t')
    axes[1, 1].legend()
    if H_true.max() > 1e-12:
        dH = np.abs(H_pred - H_true).mean() / np.abs(H_true).mean()
        axes[1, 1].text(0.05, 0.95, f'rel ΔH = {dH:.4f}', transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')

    # 2D 摆链快照（最后时刻）
    theta_true = true_traj[-1, :N]
    theta_pred = pred_traj[-1, :N]
    x_true, y_true = sys.get_positions(theta_true)
    x_pred, y_pred = sys.get_positions(theta_pred)

    # 绘制固定点
    axes[1, 2].plot(0, 0, 'ks', markersize=8, label='Pivot')

    # 绘制真实摆链
    x_chain = np.concatenate([[0], x_true])
    y_chain = np.concatenate([[0], y_true])
    axes[1, 2].plot(x_chain, y_chain, 'o-', color='C0', label='True', markersize=4)

    # 绘制 HNN 预测摆链
    x_chain_p = np.concatenate([[0], x_pred])
    y_chain_p = np.concatenate([[0], y_pred])
    axes[1, 2].plot(x_chain_p, y_chain_p, 's--', color='C1', label='HNN', markersize=4, linewidth=1)

    axes[1, 2].set_title('2D Snapshots (t=t_final)')
    axes[1, 2].set_xlabel('x'); axes[1, 2].set_ylabel('y')
    axes[1, 2].legend(loc='upper right')

    # 自动调整 x/y 范围
    all_x = np.concatenate([[0], x_true, x_pred])
    all_y = np.concatenate([[0], y_true, y_pred])
    margin = 0.2
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, 0.1
    axes[1, 2].set_xlim(x_min, x_max)
    axes[1, 2].set_ylim(y_min, y_max)
    axes[1, 2].invert_yaxis()  # y 向下为正
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f'Gravity Pendulum Chain | N={N} | dim={2*N} | 参数={n_params:,} | test MSE={test_mse:.4e}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pendulum_string.png', dpi=150)
    print(f"  可视化已保存: pendulum_string.png")
    plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Gravity Pendulum Chain HNN')
    parser.add_argument('--n_masses', type=int, default=20)
    parser.add_argument('--length', type=float, default=1.0, help='杆长')
    parser.add_argument('--mass', type=float, default=1.0, help='质点质量')
    parser.add_argument('--g', type=float, default=9.81, help='重力加速度')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_trajectories', type=int, default=200)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    dim = 2 * args.n_masses

    print("=" * 70)
    print(f"重力摆链 HNN: N={args.n_masses} 节, 状态维度={dim}")
    print(f"杆长 l={args.length}, 质量 m={args.mass}, g={args.g}")
    print(f"MLP: {dim} -> {args.hidden_dim}x{args.num_layers} -> 1")
    print(f"训练: {'DDP' if args.ddp else '单卡'}, {args.epochs} epochs")
    print("=" * 70)

    print("\n--- 生成数据 ---")
    sys = DiscretePendulumString(n_masses=args.n_masses,
                                 length=args.length, mass=args.mass, g=args.g)
    data_path = f'pendulum_string_data_N{args.n_masses}.pt'
    if os.path.exists(data_path):
        print(f"  加载已保存的数据: {data_path}")
        train_ds, val_ds, test_ds = torch.load(data_path)
    else:
        train_ds, val_ds, test_ds = sys.generate_dataset(
            n_trajectories=args.n_trajectories, seed=args.seed)
        torch.save((train_ds, val_ds, test_ds), data_path)
        print(f"  数据已保存: {data_path}")

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
    t_span = (0, 40); n_points = 1500
    # 初始条件: 拨动顶端摆杆 0.3 rad，零初速
    theta0 = np.zeros(args.n_masses); theta0[0] = 0.3
    p0 = np.zeros(args.n_masses)
    state0 = np.concatenate([theta0, p0])

    _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
    pred_traj = integrate_rk4(model, state0, t_span, n_points, device)

    visualize(args, sys, true_traj, pred_traj, test_mse, n_params, t_span, n_points)

    print(f"\n{'='*70}")
    print(f"完成 | N={args.n_masses} | dim={dim} | 参数={n_params:,} | MSE={test_mse:.4e}")
    print("=" * 70)


if __name__ == '__main__':
    main()
