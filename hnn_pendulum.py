"""
Hamiltonian Neural Network (HNN) — 学习单摆系统
================================================
基于 Greydanus, Dzamba, Yosinski (2019) "Hamiltonian Neural Networks"

核心思想：
  1. 用神经网络 H_θ(q, p) 参数化哈密顿量
  2. 从 H_θ 的梯度推导动力学：
       dq/dt =  ∂H/∂p
       dp/dt = -∂H/∂q
  3. 损失函数：L = ||dθ/dt_pred - dθ/dt_true||²
  4. 训练后，H_θ 在相空间上守恒，轨迹闭合

物理系统：理想单摆
  H(q, p) = ½p² + (1 - cos q)
  运动方程：dq/dt = p, dp/dt = -sin(q)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# 1. 哈密顿神经网络 (HNN)
# ============================================================

class HNN(nn.Module):
    """
    哈密顿神经网络

    输入 (q, p) → MLP → 输出标量 H(q, p)
    通过自动微分计算 ∂H/∂q, ∂H/∂p 得到动力学

    Args:
        input_dim: 输入维度 (单摆 = 2: q, p)
        hidden_dim: 隐藏层宽度
        num_layers: 隐藏层数
    """

    def __init__(self, input_dim=2, hidden_dim=200, num_layers=3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1, bias=False))  # 输出标量 H

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
        """
        计算哈密顿量 H(q, p)

        Args:
            x: (batch, 2) — [q, p]
        Returns:
            H: (batch, 1) — 标量哈密顿量
        """
        return self.net(x)

    def time_derivative(self, x):
        """
        从哈密顿量推导时间导数

        辛结构：
            dq/dt =  ∂H/∂p
            dp/dt = -∂H/∂q

        Args:
            x: (batch, 2) — [q, p]
        Returns:
            dx_dt: (batch, 2) — [dq/dt, dp/dt]
        """
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)  # (batch, 1)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]  # (batch, 2)

        dq_dt = dH[:, 1:2]   # ∂H/∂p
        dp_dt = -dH[:, 0:1]  # -∂H/∂q
        return torch.cat([dq_dt, dp_dt], dim=1)
# ============================================================
# 2. 单摆系统 (真实物理)
# ============================================================

class SimplePendulum:
    """
    理想单摆 (保守系统，归一化 m=1, l=1, g=1)

    哈密顿量: H(q, p) = ½p² + (1 - cos q)
    运动方程: dq/dt = p, dp/dt = -sin(q)
    """

    @staticmethod
    def hamiltonian(q, p):
        """真实哈密顿量 H(q, p) = ½p² + (1 - cos q)"""
        return 0.5 * p**2 + (1.0 - np.cos(q))

    @staticmethod
    def dynamics(t, state):
        """ODE 右端: dq/dt = p, dp/dt = -sin(q)"""
        q, p = state
        return [p, -np.sin(q)]

    @staticmethod
    def generate_trajectory(q0, p0, t_span=(0, 10), n_points=200):
        """生成单条轨迹 (q(t), p(t)) 及其导数"""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            SimplePendulum.dynamics, t_span, [q0, p0],
            t_eval=t_eval, rtol=1e-9, atol=1e-9
        )
        q, p = sol.y[0], sol.y[1]
        return sol.t, q, p, p, -np.sin(q)  # t, q, p, dq/dt, dp/dt


# ============================================================
# 3. 数据生成
# ============================================================

def generate_dataset(n_trajs=100, t_span=(0, 10), n_points=200,
                     q_range=(-np.pi, np.pi), p_range=(-2.0, 2.0)):
    """
    生成训练数据集

    从保守单摆的相空间中均匀采样初始条件，生成多条轨迹。

    Args:
        n_trajs: 轨迹数量
        t_span: 时间跨度 (t_start, t_end)
        n_points: 每条轨迹采样点数
        q_range: 初始角度范围
        p_range: 初始角速度范围

    Returns:
        X:   (n_total, 2) — [q, p]，状态
        dX:  (n_total, 2) — [dq/dt, dp/dt]，时间导数
    """
    X_list, dX_list = [], []

    for i in range(n_trajs):
        q0 = np.random.uniform(*q_range)
        p0 = np.random.uniform(*p_range)
        _, q, p, dq, dp = SimplePendulum.generate_trajectory(
            q0, p0, t_span, n_points
        )
        X_list.append(np.stack([q, p], axis=1))
        dX_list.append(np.stack([dq, dp], axis=1))

    X = np.concatenate(X_list, axis=0)
    dX = np.concatenate(dX_list, axis=0)
    return X, dX


# ============================================================
# 4. 训练
# ============================================================

def train_hnn(model, train_loader, val_loader, epochs=2000, lr=1e-3):
    """
    训练 HNN

    损失函数: MSE(dx/dt_pred, dx/dt_true)
    使用 Adam + StepLR 学习率调度

    Returns:
        train_losses, val_losses: 每个 epoch 的损失列表
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_x, batch_dx in train_loader:
            optimizer.zero_grad()
            pred = model.time_derivative(batch_x)
            loss = nn.MSELoss()(pred, batch_dx)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_dx in val_loader:
                pred = model.time_derivative(batch_x)
                val_loss += nn.MSELoss()(pred, batch_dx).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")

    return train_losses, val_losses


# ============================================================
# 5. 评估
# ============================================================

def evaluate_test_mse(model, test_loader):
    """计算测试集 MSE"""
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch_x, batch_dx in test_loader:
            pred = model.time_derivative(batch_x)
            total_mse += nn.MSELoss()(pred, batch_dx).item() * batch_x.size(0)
    return total_mse / len(test_loader.dataset)


def integrate_hnn(model, state0, t_span, n_steps=200):
    """
    用 HNN 的 time_derivative 做 RK4 积分

    Args:
        model: HNN 模型
        state0: 初始状态 [q0, p0]
        t_span: (t_start, t_end)
        n_steps: 积分步数

    Returns:
        traj: (n_steps, 2) — [q(t), p(t)]
    """
    model.eval()
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    traj = np.zeros((n_steps, 2))
    traj[0] = state0

    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32)
        k1 = model.time_derivative(x).detach().numpy()[0]
        k2 = model.time_derivative(x + 0.5*dt*torch.tensor(k1, dtype=torch.float32)).detach().numpy()[0]
        k3 = model.time_derivative(x + 0.5*dt*torch.tensor(k2, dtype=torch.float32)).detach().numpy()[0]
        k4 = model.time_derivative(x + dt*torch.tensor(k3, dtype=torch.float32)).detach().numpy()[0]
        traj[i+1] = traj[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# 6. 可视化
# ============================================================

def plot_all(model, pendulum, train_losses, val_losses, test_mse):
    """生成完整的评估图集"""

    # ---- 图 1: 损失曲线 + 轨迹预测 + 相空间 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 损失曲线
    ax = axes[0, 0]
    ax.semilogy(train_losses, 'b-', alpha=0.5, lw=1, label='Train')
    ax.semilogy(val_losses, 'r-', alpha=0.7, lw=1.5, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title(f'Training Loss (Test MSE={test_mse:.2e})')
    ax.legend(); ax.grid(alpha=0.3)

    # 轨迹预测 (3 条测试轨迹)
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 200)
    test_ics = [(1.5, 0.0), (-2.0, 1.0), (0.5, 1.5)]

    for idx, (q0, p0) in enumerate(test_ics):
        # 真实轨迹
        t_true, q_true, p_true, _, _ = SimplePendulum.generate_trajectory(
            q0, p0, t_span, 200
        )
        # HNN 预测
        traj_hnn = integrate_hnn(model, [q0, p0], t_span, 200)

        # q(t) 对比
        ax = axes[0, 1+idx]
        ax.plot(t_eval, q_true, 'k-', lw=2, label='True')
        ax.plot(t_eval, traj_hnn[:, 0], 'r--', lw=1.5, label='HNN')
        ax.set_xlabel('t'); ax.set_ylabel('q')
        ax.set_title(f'q(t) — q0={q0:.1f}, p0={p0:.1f}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 相空间图
    q_grid = np.linspace(-np.pi, np.pi, 50)
    p_grid = np.linspace(-3, 3, 50)
    Q, P = np.meshgrid(q_grid, p_grid)
    H_true = pendulum.hamiltonian(Q, P)

    Q_flat = Q.reshape(-1, 1)
    P_flat = P.reshape(-1, 1)
    with torch.no_grad():
        H_pred = model.forward(
            torch.tensor(np.hstack([Q_flat, P_flat]), dtype=torch.float32)
        ).numpy().reshape(Q.shape)

    ax = axes[1, 0]
    cs = ax.contour(Q, P, H_true, levels=15, colors='k', linewidths=1.5, alpha=0.5)
    ax.clabel(cs, fontsize=7)
    ax.set_xlabel('q'); ax.set_ylabel('p')
    ax.set_title('True H(q,p) = 1/2 p^2 + (1-cos q)')

    ax = axes[1, 1]
    cs = ax.contour(Q, P, H_pred, levels=15, colors='r', linewidths=1.5, alpha=0.7)
    ax.clabel(cs, fontsize=7)
    ax.set_xlabel('q'); ax.set_ylabel('p')
    ax.set_title('HNN Learned H_theta(q,p)')

    ax = axes[1, 2]
    H_diff = np.abs(H_pred - H_true)
    im = ax.contourf(Q, P, H_diff, levels=15, cmap='hot')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('q'); ax.set_ylabel('p')
    ax.set_title('|H_theta - H_true|')

    plt.suptitle('Hamiltonian Neural Network — Simple Pendulum', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hnn_pendulum_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> hnn_pendulum_results.png")

    # ---- 图 2: 能量守恒验证 ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (q0, p0) in enumerate(test_ics):
        traj_hnn = integrate_hnn(model, [q0, p0], (0, 20), 400)
        t_long = np.linspace(0, 20, 400)

        with torch.no_grad():
            H_t = model.forward(
                torch.tensor(traj_hnn, dtype=torch.float32)
            ).numpy().flatten()

        ax = axes2[0]
        ax.plot(t_long, traj_hnn[:, 0], lw=1.5, label=f'q0={q0:.1f}, p0={p0:.1f}')
        ax.set_xlabel('t'); ax.set_ylabel('q')
        ax.set_title('HNN Predicted Trajectories (t=20)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes2[1]
        ax.plot(t_long, H_t, lw=1.5, label=f'q0={q0:.1f}, p0={p0:.1f}')
        ax.set_xlabel('t'); ax.set_ylabel('H_theta')
        ax.set_title('Energy Conservation (H_theta ~ constant)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('hnn_pendulum_energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> hnn_pendulum_energy.png")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Hamiltonian Neural Network — 单摆系统")
    print("=" * 60)

    # ---- 生成数据 ----
    print("\n[1/4] 生成数据...")
    X, dX = generate_dataset(n_trajs=100, t_span=(0, 10), n_points=200)
    print(f"  总数据点: {X.shape[0]}")

    # 划分数据集
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dX_tensor = torch.tensor(dX, dtype=torch.float32)

    n_total = len(X_tensor)
    indices = torch.randperm(n_total)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_loader = DataLoader(
        TensorDataset(X_tensor[indices[:n_train]], dX_tensor[indices[:n_train]]),
        batch_size=512, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_tensor[indices[n_train:n_train+n_val]],
                      dX_tensor[indices[n_train:n_train+n_val]]),
        batch_size=512, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_tensor[indices[n_train+n_val:]],
                      dX_tensor[indices[n_train+n_val:]]),
        batch_size=512, shuffle=False
    )
    print(f"  训练: {n_train}, 验证: {n_val}, 测试: {n_total - n_train - n_val}")

    # ---- 创建模型 ----
    print("\n[2/4] 创建 HNN 模型...")
    model = HNN(input_dim=2, hidden_dim=200, num_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {n_params:,}")

    # ---- 训练 ----
    print("\n[3/4] 训练...")
    train_losses, val_losses = train_hnn(
        model, train_loader, val_loader, epochs=2000, lr=1e-3
    )

    test_mse = evaluate_test_mse(model, test_loader)
    print(f"\n  测试 MSE: {test_mse:.6e}")

    # ---- 可视化 ----
    print("\n[4/4] 生成可视化...")
    pendulum = SimplePendulum()
    plot_all(model, pendulum, train_losses, val_losses, test_mse)

    print("\n" + "=" * 60)
    print("完成! HNN 成功学习到单摆的哈密顿量。")
    print(f"  H(q,p) = 1/2 p^2 + (1-cos q)")
    print(f"  测试 MSE: {test_mse:.6e}")
    print("  输出: hnn_pendulum_results.png, hnn_pendulum_energy.png")
    print("=" * 60)


if __name__ == '__main__':
    main()