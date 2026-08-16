"""
简单单摆 — 哈密顿神经网络 (HNN)
================================
物理: 单摆由一根刚性杆和质量点组成，受重力驱动。
      状态: x = (θ, p) ∈ ℝ²
      θ = 与竖直方向夹角, p = ml² θ̇ = 角动量

哈密顿量: H(θ, p) = p²/(2ml²) + mgl(1 - cos θ)
动力学:
    dθ/dt = ∂H/∂p = p/(ml²)
    dp/dt = -∂H/∂θ = -mgl·sin θ

HNN: 用神经网络学习 H(θ, p)，通过 autograd 计算梯度得到时间导数。

用法: python simple_pendulum_hnn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ══════════════════════════════════════════════════════════════
# 1. 单摆物理系统
# ══════════════════════════════════════════════════════════════

class SimplePendulum:
    """单摆物理系统: 质量 m，杆长 l，重力加速度 g"""
    def __init__(self, m=1.0, l=1.0, g=9.81):
        self.m = m
        self.l = l
        self.g = g
        self.ml2 = m * l * l   # 转动惯量

    def hamiltonian(self, state):
        """H(θ, p) = p²/(2ml²) + mgl(1 - cos θ)"""
        theta, p = state[..., 0], state[..., 1]
        T = 0.5 * p**2 / self.ml2
        V = self.m * self.g * self.l * (1.0 - np.cos(theta))
        return T + V

    def dynamics(self, t, state):
        """时间导数: [dθ/dt, dp/dt]"""
        theta, p = state[0], state[1]
        dtheta = p / self.ml2
        dp = -self.m * self.g * self.l * np.sin(theta)
        return np.array([dtheta, dp])

    def generate_trajectory(self, theta0, p0, t_span, n_steps):
        """从初始条件 (θ₀, p₀) 生成一条轨迹"""
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)
        sol = solve_ivp(self.dynamics, t_span, [theta0, p0],
                        t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
        return sol.t, sol.y.T  # (n_steps, 2)


# ══════════════════════════════════════════════════════════════
# 2. 数据生成
# ══════════════════════════════════════════════════════════════

def generate_dataset(pendulum, n_trajectories=50, t_span=(0, 20),
                     n_steps=2000, dt=0.01):
    """生成训练数据: (x, dx/dt) 对

    每条轨迹从随机初始条件出发，用 scipy 高精度积分。
    返回 train/val/test 三个 TensorDataset。
    """
    X_list, dX_list = [], []

    for i in range(n_trajectories):
        theta0 = np.random.uniform(-np.pi, np.pi)
        p0 = np.random.uniform(-3.0, 3.0)
        t, traj = pendulum.generate_trajectory(theta0, p0, t_span, n_steps)

        X_list.append(traj[:-1])
        dX_list.append(np.array([pendulum.dynamics(ti, xi)
                                 for ti, xi in zip(t[:-1], traj[:-1])]))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    dX = np.concatenate(dX_list, axis=0).astype(np.float32)

    # 80/10/10 划分
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    return (TensorDataset(torch.tensor(X[idx[:n_train]]),
                          torch.tensor(dX[idx[:n_train]])),
            TensorDataset(torch.tensor(X[idx[n_train:n_train+n_val]]),
                          torch.tensor(dX[idx[n_train:n_train+n_val]])),
            TensorDataset(torch.tensor(X[idx[n_train+n_val:]]),
                          torch.tensor(dX[idx[n_train+n_val:]])))
# ══════════════════════════════════════════════════════════════
# 3. 哈密顿神经网络 (HNN)
# ══════════════════════════════════════════════════════════════

class HNN(nn.Module):
    """哈密顿神经网络: 学习系统的哈密顿量 H(θ, p)

    核心思想: 用神经网络 H_net(θ, p) → ℝ 近似真实哈密顿量，
    通过 autograd 计算梯度得到时间导数:
        dθ/dt = ∂H/∂p,  dp/dt = -∂H/∂θ

    训练时用 MSE 损失匹配真实时间导数 (不需要实际积分！)
    """
    def __init__(self, hidden_dim=256, num_layers=4, activation='tanh'):
        super().__init__()
        layers = []
        prev = 2  # 输入维度: (θ, p)
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'softplus':
                layers.append(nn.Softplus())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1, bias=False))  # 输出标量 H
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """计算哈密顿量 H(x)"""
        return self.net(x).squeeze(-1)

    def time_derivative(self, x):
        """计算时间导数 [dθ/dt, dp/dt] = [∂H/∂p, -∂H/∂θ]

        x: (batch, 2) — (θ, p)，必须已 requires_grad=True
        返回: (batch, 2) — (dθ/dt, dp/dt)
        """
        H = self.forward(x)
        grad = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        # 哈密顿方程: dθ/dt = ∂H/∂p, dp/dt = -∂H/∂θ
        return torch.stack([grad[:, 1], -grad[:, 0]], dim=-1)


# ══════════════════════════════════════════════════════════════
# 4. 训练
# ══════════════════════════════════════════════════════════════

def train_hnn(model, train_loader, val_loader, epochs=500, lr=1e-3,
              device='cpu'):
    """训练 HNN: 用 MSE 匹配时间导数"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, dxb in train_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            xb.requires_grad_(True)
            optimizer.zero_grad()
            dx_pred = model.time_derivative(xb)
            loss = loss_fn(dx_pred, dxb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        for xb, dxb in val_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            xb.requires_grad_(True)
            dx_pred = model.time_derivative(xb)
            val_loss += loss_fn(dx_pred, dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.2e} | Val Loss: {val_loss:.2e}")

    return train_losses, val_losses
# ══════════════════════════════════════════════════════════════
# 5. 评估与可视化
# ══════════════════════════════════════════════════════════════

def evaluate_and_visualize(model, pendulum, test_loader, train_losses,
                            val_losses, device='cpu'):
    """评估 HNN 并可视化: 损失曲线、哈密顿量对比、相空间轨迹、能量误差"""
    model.eval()

    # ── 5a. 测试集 MSE ──────────────────────────────────────
    test_mse = 0.0; n_test = 0
    loss_fn = nn.MSELoss()
    for xb, dxb in test_loader:
        xb, dxb = xb.to(device), dxb.to(device)
        xb.requires_grad_(True)
        dx_pred = model.time_derivative(xb)
        test_mse += loss_fn(dx_pred, dxb).item() * xb.size(0)
        n_test += xb.size(0)
    test_mse /= n_test
    print(f"\n{'='*60}")
    print(f"测试集 MSE: {test_mse:.6e}")
    print(f"{'='*60}")

    # ── 5b. 可视化 ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 损失曲线
    axes[0, 0].semilogy(train_losses, label='Train', alpha=0.7)
    axes[0, 0].semilogy(val_losses, label='Val', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Loss'); axes[0, 0].legend()
    axes[0, 0].grid(True)

    # (θ, p) 网格
    theta_grid = np.linspace(-np.pi, np.pi, 100)
    p_grid = np.linspace(-3, 3, 100)
    Theta, P = np.meshgrid(theta_grid, p_grid)
    grid_input = np.stack([Theta.ravel(), P.ravel()], axis=-1)

    H_true = pendulum.hamiltonian(
        np.stack([Theta.ravel(), P.ravel()], axis=-1)).reshape(100, 100)

    grid_tensor = torch.tensor(grid_input, dtype=torch.float32, device=device)
    with torch.no_grad():
        H_pred = model.forward(grid_tensor).cpu().numpy().reshape(100, 100)

    im1 = axes[0, 1].contourf(Theta, P, H_true, levels=30, cmap='viridis')
    axes[0, 1].set_xlabel(r'$\theta$ (rad)')
    axes[0, 1].set_ylabel('$p$ (kg·m²/s)')
    axes[0, 1].set_title('True Hamiltonian $H(\\theta, p)$')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].contourf(Theta, P, H_pred, levels=30, cmap='viridis')
    axes[0, 2].set_xlabel(r'$\theta$ (rad)')
    axes[0, 2].set_ylabel('$p$ (kg·m²/s)')
    axes[0, 2].set_title('Learned $\\hat{H}(\\theta, p)$')
    plt.colorbar(im2, ax=axes[0, 2])

    error = np.abs(H_true - H_pred)
    im3 = axes[1, 0].contourf(Theta, P, error, levels=30, cmap='hot')
    axes[1, 0].set_xlabel(r'$\theta$ (rad)')
    axes[1, 0].set_ylabel('$p$ (kg·m²/s)')
    axes[1, 0].set_title('|H_true - H_learned|')
    plt.colorbar(im3, ax=axes[1, 0])

    # ── 5c. 相空间轨迹预测 ──────────────────────────────────
    theta0, p0 = -1.0, 0.5
    dt = 0.01; n_steps = int(30 / dt) + 1
    t_true, traj_true = pendulum.generate_trajectory(
        theta0, p0, (0, 30), n_steps)

    state = torch.tensor([theta0, p0], dtype=torch.float32, device=device)
    traj_pred = np.zeros((n_steps, 2))
    traj_pred[0] = [theta0, p0]

    def rk4_step(model, x, dt):
        x = x.detach().requires_grad_(True)
        k1 = model.time_derivative(x.unsqueeze(0)).squeeze(0)
        k2 = model.time_derivative((x + 0.5*dt*k1).unsqueeze(0)).squeeze(0)
        k3 = model.time_derivative((x + 0.5*dt*k2).unsqueeze(0)).squeeze(0)
        k4 = model.time_derivative((x + dt*k3).unsqueeze(0)).squeeze(0)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    for i in range(1, n_steps):
        state = rk4_step(model, state, dt)
        traj_pred[i] = state.detach().cpu().numpy()

    axes[1, 1].plot(traj_true[:, 0], traj_true[:, 1], 'b-', lw=1.0,
                    alpha=0.7, label='True')
    axes[1, 1].plot(traj_pred[:, 0], traj_pred[:, 1], 'r--', lw=1.0,
                    alpha=0.7, label='HNN')
    axes[1, 1].set_xlabel(r'$\theta$ (rad)')
    axes[1, 1].set_ylabel('$p$ (kg·m²/s)')
    axes[1, 1].set_title('Phase Portrait: True vs HNN (30s)')
    axes[1, 1].legend(); axes[1, 1].grid(True)

    # ── 5d. 哈密顿量守恒 ────────────────────────────────────
    H_true_traj = pendulum.hamiltonian(traj_pred)
    H_initial = H_true_traj[0]
    H_error = np.abs(H_true_traj - H_initial) / (np.abs(H_initial) + 1e-10)

    axes[1, 2].semilogy(t_true, H_error, 'r-', lw=1.0)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('|H(t) - H(0)| / |H(0)|')
    axes[1, 2].set_title('HNN Energy Conservation Error')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('simple_pendulum_hnn_results.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print(f"\n结果已保存到 simple_pendulum_hnn_results.png")

    return test_mse
# ══════════════════════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("单摆 — 哈密顿神经网络 (HNN)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    pendulum = SimplePendulum(m=1.0, l=1.0, g=9.81)
    print(f"物理参数: m={pendulum.m}, l={pendulum.l}, g={pendulum.g}")
    print(f"真实 H = p²/(2ml²) + mgl(1 - cos θ)")

    print("\n生成训练数据...")
    train_set, val_set, test_set = generate_dataset(
        pendulum, n_trajectories=50, t_span=(0, 20), n_steps=2000)
    print(f"  训练集: {len(train_set):,} | "
          f"验证集: {len(val_set):,} | 测试集: {len(test_set):,}")

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False)

    model = HNN(hidden_dim=256, num_layers=4, activation='tanh')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: HNN (hidden_dim=256, num_layers=4, params={n_params:,})")

    print("\n开始训练...")
    train_losses, val_losses = train_hnn(
        model, train_loader, val_loader, epochs=500, lr=1e-3, device=device)

    test_mse = evaluate_and_visualize(
        model, pendulum, test_loader, train_losses, val_losses, device=device)

    print(f"\n最终测试 MSE: {test_mse:.6e}")
    print("完成!")


if __name__ == '__main__':
    main()