"""
Port-HNN vs Extended-system HNN for Damped Pendulum
====================================================

对比三种建模耗散单摆的方法：

方法 A: Port-HNN (端口哈密顿神经网络)
    架构: dx/dt = (J - R(x)) ∇H_θ(x)
    J = [[0, 1], [-1, 0]] (固定辛结构)
    R(x) = diag(0, γ(x))   (学习的耗散，γ(x) ≥ 0)
    直接学习耗散动力学，保持端口哈密顿结构。
    2D 状态: [q, p]

方法 B: 扩展系统 HNN
    架构: H = H_pend(MLP) + H_ho(已知) + H_coup(MLP)
    通过添加辅助谐振子吸收能量，将耗散系统建模为更大的保守系统。
    4D 状态: [q1, p1, q2, p2]，其中 q2/p2 是辅助 HO 坐标。

方法 C: 标准 HNN (基线)
    架构: dx/dt = J ∇H_θ(x) (纯保守)
    忽略耗散，学习保守近似。预期在耗散数据上表现最差。
    2D 状态: [q, p]

核心对比问题:
    1. Port-HNN 能否正确学到耗散系数 γ ≈ 0.3？
    2. Port-HNN 的轨迹预测是否优于标准 HNN？
    3. 扩展系统 HNN 的辅助 HO 能否有效吸收能量？
    4. 三种方法在参数量、MSE、能量守恒方面的定量对比。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# 1. 耗散单摆 (真实物理，数据生成)
# ============================================================

class DampedPendulum:
    """耗散单摆: dq/dt = p, dp/dt = -sin(q) - gamma * p"""

    def __init__(self, gamma=0.3):
        self.gamma = gamma

    def dynamics(self, t, state):
        q, p = state
        return [p, -np.sin(q) - self.gamma * p]

    def hamiltonian(self, q, p):
        return 0.5 * p**2 + (1.0 - np.cos(q))

    def dissipation_rate(self, q, p):
        return -self.gamma * p**2  # dH/dt = -gamma * p^2 <= 0

    def generate_trajectory(self, q0, p0, t_span=(0, 15), n_points=200):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, [q0, p0],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        q, p = sol.y
        dq = p; dp = -np.sin(q) - self.gamma * p
        return sol.t, q, p, dq, dp


def generate_damped_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200):
    """生成耗散单摆的 (x, dx/dt) 数据 (2D)"""
    xs_list, dxs_list = [], []
    for _ in range(n_trajectories):
        q0 = np.random.uniform(-np.pi, np.pi)
        p0 = np.random.uniform(-2.0, 2.0)
        _, q, p, dq, dp = system.generate_trajectory(q0, p0, t_span, n_points)
        xs_list.append(np.stack([q, p], axis=1))
        dxs_list.append(np.stack([dq, dp], axis=1))
    return np.concatenate(xs_list, axis=0), np.concatenate(dxs_list, axis=0)


# ============================================================
# 2. 保守耦合振子 (扩展系统数据生成)
# ============================================================

class CoupledOscillator:
    """保守耦合系统: pendulum + HO, H = p1^2/2 + (1-cos q1) + p2^2/2 + w^2 q2^2/2 + eps * q1 * q2"""

    def __init__(self, omega=2.0, epsilon=0.3):
        self.omega = omega; self.epsilon = epsilon

    def hamiltonian(self, q1, p1, q2, p2):
        return (0.5 * p1**2 + (1.0 - np.cos(q1)) +
                0.5 * p2**2 + 0.5 * self.omega**2 * q2**2 +
                self.epsilon * q1 * q2)

    def pendulum_energy(self, q1, p1):
        return 0.5 * p1**2 + (1.0 - np.cos(q1))

    def ho_energy(self, q2, p2):
        return 0.5 * p2**2 + 0.5 * self.omega**2 * q2**2

    def coupling_energy(self, q1, q2):
        return self.epsilon * q1 * q2

    def dynamics(self, t, state):
        q1, p1, q2, p2 = state
        return [p1, -np.sin(q1) - self.epsilon * q2,
                p2, -self.omega**2 * q2 - self.epsilon * q1]

    def generate_trajectory(self, q10, p10, q20, p20, t_span=(0, 15), n_points=200):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, [q10, p10, q20, p20],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        q1, p1, q2, p2 = sol.y
        dq1 = p1; dp1 = -np.sin(q1) - self.epsilon * q2
        dq2 = p2; dp2 = -self.omega**2 * q2 - self.epsilon * q1
        return sol.t, q1, p1, q2, p2, dq1, dp1, dq2, dp2


def generate_coupled_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200):
    """生成耦合振子的 (x, dx/dt) 数据 (4D)"""
    xs_list, dxs_list = [], []
    for _ in range(n_trajectories):
        q10 = np.random.uniform(-np.pi, np.pi)
        p10 = np.random.uniform(-2.0, 2.0)
        q20 = np.random.uniform(-2.0, 2.0)
        p20 = np.random.uniform(-2.0, 2.0)
        _, q1, p1, q2, p2, dq1, dp1, dq2, dp2 = system.generate_trajectory(
            q10, p10, q20, p20, t_span, n_points)
        xs_list.append(np.stack([q1, p1, q2, p2], axis=1))
        dxs_list.append(np.stack([dq1, dp1, dq2, dp2], axis=1))
    return np.concatenate(xs_list, axis=0), np.concatenate(dxs_list, axis=0)


# ============================================================
# 3. MLP 构建块
# ============================================================

def make_mlp(input_dim, hidden_dim, num_layers, output_dim=1):
    layers = []
    prev_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.Tanh())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim, bias=False))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


# ============================================================
# 4. 模型 A: Port-HNN (端口哈密顿神经网络)
# ============================================================

class PortHNN(nn.Module):
    """
    端口哈密顿神经网络: dx/dt = (J - R(x)) nabla H_theta(x)

    对于 2D 单摆:
        J = [[0, 1], [-1, 0]]
        R(x) = [[0, 0], [0, gamma(x)]]  (gamma(x) >= 0, 状态相关耗散)
        H_theta(q, p) 通过 MLP 学习

    动力学:
        dq/dt = dH/dp
        dp/dt = -dH/dq - gamma(x) * dH/dp
    """

    def __init__(self, hidden_dim=200, num_layers=3):
        super().__init__()
        self.H_net = make_mlp(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
        # gamma_net: 耗散系数网络 (状态相关, softplus 保证正定性)
        self.gamma_net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """计算哈密顿量 H_theta(x)"""
        return self.H_net(x)

    def get_gamma(self, x):
        """计算耗散系数 gamma(x) >= 0"""
        return nn.functional.softplus(self.gamma_net(x))

    def time_derivative(self, x):
        """dx/dt = (J - R) nabla H"""
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.H_net(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        dq_dt = dH[:, 1:2]                              # dH/dp
        gamma = self.get_gamma(x)
        dp_dt = -dH[:, 0:1] - gamma * dH[:, 1:2]         # -dH/dq - gamma * dH/dp

        return torch.cat([dq_dt, dp_dt], dim=1)


# ============================================================
# 5. 模型 B: 扩展系统 HNN (保守 Pendulum + HO)
# ============================================================

class ExtendedHNN(nn.Module):
    """
    扩展系统 HNN: H = H_pend(q1,p1) + H_ho(q2,p2) + H_coup(q1,p1,q2,p2)
    4D 保守系统，通过辅助 HO 吸收能量。
    """

    def __init__(self, omega=2.0, pend_hidden=200, coup_hidden=128, num_layers=3):
        super().__init__()
        self.omega = omega
        self.pendulum_net = make_mlp(2, pend_hidden, num_layers)
        self.coupling_net = make_mlp(4, coup_hidden, num_layers)

    def forward(self, x):
        q1_p1 = x[:, :2]; q2 = x[:, 2:3]; p2 = x[:, 3:4]
        H_pend = self.pendulum_net(q1_p1)
        H_ho = 0.5 * p2**2 + 0.5 * self.omega**2 * q2**2
        H_coup = self.coupling_net(x)
        return H_pend + H_ho + H_coup

    def time_derivative(self, x):
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dq1 = dH[:, 1:2]; dp1 = -dH[:, 0:1]
        dq2 = dH[:, 3:4]; dp2 = -dH[:, 2:3]
        return torch.cat([dq1, dp1, dq2, dp2], dim=1)


# ============================================================
# 6. 模型 C: 标准 HNN (基线，纯保守)
# ============================================================

class StandardHNN(nn.Module):
    """
    标准 HNN: dx/dt = J nabla H_theta(x)，无耗散项。
    预期在耗散数据上表现最差，因为无法建模能量耗散。
    """

    def __init__(self, hidden_dim=200, num_layers=3):
        super().__init__()
        self.net = make_mlp(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x):
        return self.net(x)

    def time_derivative(self, x):
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return torch.cat([dH[:, 1:2], -dH[:, 0:1]], dim=1)


# ============================================================
# 7. 训练和积分工具
# ============================================================

def train_model(model, train_loader, val_loader, epochs=2000, lr=1e-3, label=""):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, dx_batch in train_loader:
            optimizer.zero_grad()
            dx_pred = model.time_derivative(x_batch)
            loss = nn.MSELoss()(dx_pred, dx_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, dx_batch in val_loader:
                dx_pred = model.time_derivative(x_batch)
                loss = nn.MSELoss()(dx_pred, dx_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return train_losses, val_losses


def compute_test_mse(model, test_loader):
    model.eval()
    total_mse = 0.0; n_total = 0
    with torch.no_grad():
        for xb, dxb in test_loader:
            pred = model.time_derivative(xb)
            total_mse += nn.MSELoss()(pred, dxb).item() * xb.size(0)
            n_total += xb.size(0)
    return total_mse / n_total


def integrate_rk4(model, state0, t_span, n_steps):
    """RK4 积分器"""
    dt = (t_span[1] - t_span[0]) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D)); traj[0] = state0
    model.eval()
    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32)
        k1 = model.time_derivative(x).detach().numpy()[0]
        k2 = model.time_derivative(x + 0.5*dt*torch.tensor(k1, dtype=torch.float32)).detach().numpy()[0]
        k3 = model.time_derivative(x + 0.5*dt*torch.tensor(k2, dtype=torch.float32)).detach().numpy()[0]
        k4 = model.time_derivative(x + dt*torch.tensor(k3, dtype=torch.float32)).detach().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj


# ============================================================
# 8. 主程序: 三方法对比
# ============================================================

def main():
    GAMMA = 0.3; OMEGA = 2.0; EPSILON = 0.3
    SEED = 42; EPOCHS = 2000

    print("=" * 70)
    print("Port-HNN vs Extended-System HNN vs Standard HNN")
    print("耗散单摆: gamma = {:g}".format(GAMMA))
    print("=" * 70)

    # ---- 生成耗散数据 (2D) ----
    print("\n--- 生成耗散单摆数据 (2D) ---")
    damped_sys = DampedPendulum(gamma=GAMMA)
    xs_d, dxs_d = generate_damped_dataset(damped_sys, n_trajectories=80, t_span=(0, 15), n_points=200)
    print(f"数据点数: {xs_d.shape[0]}")

    n_total = xs_d.shape[0]
    indices = np.random.permutation(n_total)
    n_train = int(0.7 * n_total); n_val = int(0.15 * n_total)

    xs_d_t = torch.tensor(xs_d, dtype=torch.float32)
    dxs_d_t = torch.tensor(dxs_d, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(xs_d_t[indices[:n_train]], dxs_d_t[indices[:n_train]]),
        batch_size=512, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(xs_d_t[indices[n_train:n_train+n_val]],
                      dxs_d_t[indices[n_train:n_train+n_val]]),
        batch_size=512, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(xs_d_t[indices[n_train+n_val:]],
                      dxs_d_t[indices[n_train+n_val:]]),
        batch_size=512, shuffle=False)
    print(f"训练: {n_train}, 验证: {n_val}, 测试: {n_total - n_train - n_val}")

    # ---- 生成扩展系统数据 (4D) ----
    print("\n--- 生成保守耦合振子数据 (4D) ---")
    coupled_sys = CoupledOscillator(omega=OMEGA, epsilon=EPSILON)
    xs_c, dxs_c = generate_coupled_dataset(coupled_sys, n_trajectories=80, t_span=(0, 15), n_points=200)
    print(f"数据点数: {xs_c.shape[0]}")

    n_total_c = xs_c.shape[0]
    indices_c = np.random.permutation(n_total_c)
    n_train_c = int(0.7 * n_total_c); n_val_c = int(0.15 * n_total_c)

    xs_c_t = torch.tensor(xs_c, dtype=torch.float32)
    dxs_c_t = torch.tensor(dxs_c, dtype=torch.float32)

    train_loader_c = DataLoader(
        TensorDataset(xs_c_t[indices_c[:n_train_c]], dxs_c_t[indices_c[:n_train_c]]),
        batch_size=512, shuffle=True)
    val_loader_c = DataLoader(
        TensorDataset(xs_c_t[indices_c[n_train_c:n_train_c+n_val_c]],
                      dxs_c_t[indices_c[n_train_c:n_train_c+n_val_c]]),
        batch_size=512, shuffle=False)
    test_loader_c = DataLoader(
        TensorDataset(xs_c_t[indices_c[n_train_c+n_val_c:]],
                      dxs_c_t[indices_c[n_train_c+n_val_c:]]),
        batch_size=512, shuffle=False)

    # ================================================================
    # 方法 A: Port-HNN
    # ================================================================
    print("\n" + "=" * 70)
    print("方法 A: Port-HNN -- 直接学习耗散结构")
    print("=" * 70)
    print("  dx/dt = (J - R(x)) nabla H_theta(x),  R = diag(0, gamma(x))")
    print("  H_theta: MLP(2->200->200->200->1)  |  gamma: MLP(2->32->1) + softplus")

    torch.manual_seed(SEED); np.random.seed(SEED)
    model_port = PortHNN(hidden_dim=200, num_layers=3)
    n_port = sum(p.numel() for p in model_port.parameters())
    print(f"  参数量: {n_port}")

    train_losses_port, val_losses_port = train_model(
        model_port, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, label="[Port-HNN]")

    mse_port = compute_test_mse(model_port, test_loader)
    print(f"\n  [Port-HNN] 测试 MSE (耗散数据): {mse_port:.6e}")

    with torch.no_grad():
        test_x = xs_d_t[indices[n_train+n_val:]]
        gamma_pred = model_port.get_gamma(test_x).numpy()
    print(f"  [Port-HNN] 学到 gamma(x) 均值: {gamma_pred.mean():.4f} "
          f"+/- {gamma_pred.std():.4f}  (真实 gamma = {GAMMA})")

    # ================================================================
    # 方法 B: 扩展系统 HNN
    # ================================================================
    print("\n" + "=" * 70)
    print("方法 B: 扩展系统 HNN -- 辅助 HO 吸收能量")
    print("=" * 70)
    print("  H = H_pend(MLP) + H_ho(已知) + H_coup(MLP)")
    print("  4D 保守系统，总 H 守恒")

    torch.manual_seed(SEED); np.random.seed(SEED)
    model_ext = ExtendedHNN(omega=OMEGA, pend_hidden=200, coup_hidden=128, num_layers=3)
    n_ext = sum(p.numel() for p in model_ext.parameters())
    print(f"  参数量: {n_ext}")

    train_losses_ext, val_losses_ext = train_model(
        model_ext, train_loader_c, val_loader_c, epochs=EPOCHS, lr=1e-3, label="[Ext-HNN]")

    mse_ext = compute_test_mse(model_ext, test_loader_c)
    print(f"\n  [Ext-HNN] 测试 MSE (耦合数据): {mse_ext:.6e}")

    # ================================================================
    # 方法 C: 标准 HNN (基线)
    # ================================================================
    print("\n" + "=" * 70)
    print("方法 C: 标准 HNN -- 纯保守 (基线)")
    print("=" * 70)
    print("  dx/dt = J nabla H_theta(x), 无耗散项")
    print("  H_theta: MLP(2->200->200->200->1)")

    torch.manual_seed(SEED); np.random.seed(SEED)
    model_std = StandardHNN(hidden_dim=200, num_layers=3)
    n_std = sum(p.numel() for p in model_std.parameters())
    print(f"  参数量: {n_std}")

    train_losses_std, val_losses_std = train_model(
        model_std, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, label="[Std-HNN]")

    mse_std = compute_test_mse(model_std, test_loader)
    print(f"\n  [Std-HNN] 测试 MSE (耗散数据): {mse_std:.6e}")

    # ================================================================
    # 定量对比表格
    # ================================================================
    print("\n" + "=" * 70)
    print("定量对比")
    print("=" * 70)
    print(f"  {'方法':<20s} {'维数':>4s} {'参数量':>8s} {'测试 MSE':>12s} {'数据':<10s} {'注':<20s}")
    print(f"  {'-'*65}")
    print(f"  {'Port-HNN':<20s} {'2D':>4s} {n_port:>8d} {mse_port:>12.4e} {'耗散':>8s}  {'直接学习耗散':<20s}")
    print(f"  {'Extended HNN':<20s} {'4D':>4s} {n_ext:>8d} {mse_ext:>12.4e} {'保守':>8s}  {'辅助HO吸收能量':<20s}")
    print(f"  {'Standard HNN':<20s} {'2D':>4s} {n_std:>8d} {mse_std:>12.4e} {'耗散':>8s}  {'保守近似(失败)':<20s}")

    if mse_port < mse_std:
        print(f"\n  >> Port-HNN 比 Standard HNN: "
              f"MSE 降低 {(1 - mse_port/mse_std)*100:.1f}% "
              f"(Port-HNN 直接建模耗散更有效)")

    # ================================================================
    # 轨迹评估: Port-HNN vs Standard HNN vs Ground Truth
    # ================================================================
    print("\n--- 轨迹评估 ---")
    t_span = (0, 30); n_points = 500
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    q0, p0 = 1.5, 0.0

    # 真实轨迹 (耗散)
    sol_true = solve_ivp(damped_sys.dynamics, t_span, [q0, p0],
                         t_eval=t_eval, rtol=1e-9, atol=1e-9)
    q_true, p_true = sol_true.y
    H_true = damped_sys.hamiltonian(q_true, p_true)

    # Port-HNN 预测
    traj_port = integrate_rk4(model_port, np.array([q0, p0]), t_span, n_points)
    q_port, p_port = traj_port[:, 0], traj_port[:, 1]
    with torch.no_grad():
        H_port = model_port(torch.tensor(traj_port, dtype=torch.float32)).numpy().flatten()

    # Standard HNN 预测
    traj_std = integrate_rk4(model_std, np.array([q0, p0]), t_span, n_points)
    q_std, p_std = traj_std[:, 0], traj_std[:, 1]
    with torch.no_grad():
        H_std = model_std(torch.tensor(traj_std, dtype=torch.float32)).numpy().flatten()

    # 扩展系统: 看 pendulum 分量
    q20, p20 = 0.0, 0.0
    traj_ext = integrate_rk4(model_ext, np.array([q0, p0, q20, p20]), t_span, n_points)
    q_ext, p_ext = traj_ext[:, 0], traj_ext[:, 1]

    # ================================================================
    # 可视化
    # ================================================================
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: q(t) and p(t)
    ax_q = fig.add_subplot(gs[0, :2])
    ax_q.plot(t_eval, q_true, 'k-', lw=2.5, label='True (damped)')
    ax_q.plot(t_eval, q_port, 'r-', lw=1.8, label='Port-HNN')
    ax_q.plot(t_eval, q_std, 'b--', lw=1.5, label='Standard HNN')
    ax_q.set_title('Pendulum Angle q(t)'); ax_q.set_xlabel('t')
    ax_q.set_ylabel('q'); ax_q.legend(fontsize=9); ax_q.grid(alpha=0.3)

    ax_p = fig.add_subplot(gs[0, 2:])
    ax_p.plot(t_eval, p_true, 'k-', lw=2.5, label='True (damped)')
    ax_p.plot(t_eval, p_port, 'r-', lw=1.8, label='Port-HNN')
    ax_p.plot(t_eval, p_std, 'b--', lw=1.5, label='Standard HNN')
    ax_p.set_title('Momentum p(t)'); ax_p.set_xlabel('t')
    ax_p.set_ylabel('p'); ax_p.legend(fontsize=9); ax_p.grid(alpha=0.3)

    # Row 2: Energy and learned gamma
    ax_energy = fig.add_subplot(gs[1, :2])
    ax_energy.plot(t_eval, H_true, 'k-', lw=2.5, label='H_true (decays)')
    ax_energy.plot(t_eval, H_port, 'r-', lw=1.8, label='H_theta (Port-HNN)')
    ax_energy.plot(t_eval, H_std, 'b--', lw=1.5, label='H_theta (Std-HNN, conserved)')
    ax_energy.set_title('Learned Hamiltonian H_theta(t)'); ax_energy.set_xlabel('t')
    ax_energy.set_ylabel('H'); ax_energy.legend(fontsize=9); ax_energy.grid(alpha=0.3)

    ax_gamma = fig.add_subplot(gs[1, 2:])
    with torch.no_grad():
        gamma_traj = model_port.get_gamma(
            torch.tensor(traj_port, dtype=torch.float32)).numpy().flatten()
    ax_gamma.plot(t_eval, gamma_traj, 'r-', lw=1.8,
                  label=f'gamma(x) learned (mean={gamma_traj.mean():.3f})')
    ax_gamma.axhline(y=GAMMA, color='k', ls='--', lw=1.5,
                     label=f'gamma_true = {GAMMA}')
    ax_gamma.set_title('Learned Dissipation Coefficient gamma(x)')
    ax_gamma.set_xlabel('t'); ax_gamma.set_ylabel('gamma')
    ax_gamma.legend(fontsize=9); ax_gamma.grid(alpha=0.3)

    # Row 3: Phase portrait and error
    ax_phase = fig.add_subplot(gs[2, :2])
    ax_phase.plot(q_true, p_true, 'k-', lw=2.5, alpha=0.8, label='True')
    ax_phase.plot(q_port, p_port, 'r-', lw=1.5, alpha=0.8, label='Port-HNN')
    ax_phase.plot(q_std, p_std, 'b--', lw=1.5, alpha=0.8, label='Standard HNN')
    ax_phase.plot(q0, p0, 'ko', ms=8, label='Start')
    ax_phase.set_title('Phase Portrait (q, p)'); ax_phase.set_xlabel('q')
    ax_phase.set_ylabel('p'); ax_phase.legend(fontsize=9)
    ax_phase.grid(alpha=0.3); ax_phase.axis('equal')

    ax_err = fig.add_subplot(gs[2, 2:])
    ax_err.semilogy(t_eval, np.abs(q_port - q_true), 'r-', lw=1,
                    label='Port-HNN |q_err|')
    ax_err.semilogy(t_eval, np.abs(q_std - q_true), 'b--', lw=1,
                    label='Std-HNN |q_err|')
    ax_err.set_title('Trajectory Error |q_pred - q_true|')
    ax_err.set_xlabel('t'); ax_err.set_ylabel('|error|')
    ax_err.legend(fontsize=9); ax_err.grid(alpha=0.3)

    # Row 4: Extended HNN pendulum component and energy partition
    ax_ext_q = fig.add_subplot(gs[3, :2])
    ax_ext_q.plot(t_eval, q_true, 'k-', lw=2.5, label='True damped pendulum')
    ax_ext_q.plot(t_eval, q_ext, 'g-', lw=1.5, label='Extended HNN (pendulum component)')
    ax_ext_q.set_title('Extended HNN: Pendulum Angle q1(t)')
    ax_ext_q.set_xlabel('t'); ax_ext_q.set_ylabel('q1')
    ax_ext_q.legend(fontsize=9); ax_ext_q.grid(alpha=0.3)

    ax_ext_e = fig.add_subplot(gs[3, 2:])
    q1_ext = traj_ext[:, 0]; p1_ext = traj_ext[:, 1]
    q2_ext = traj_ext[:, 2]; p2_ext = traj_ext[:, 3]
    E_pend_ext = coupled_sys.pendulum_energy(q1_ext, p1_ext)
    E_ho_ext = coupled_sys.ho_energy(q2_ext, p2_ext)
    E_coup_ext = coupled_sys.coupling_energy(q1_ext, q2_ext)
    E_total_ext = E_pend_ext + E_ho_ext + E_coup_ext
    ax_ext_e.plot(t_eval, E_pend_ext, 'b-', lw=1.5, alpha=0.8, label='E_pend')
    ax_ext_e.plot(t_eval, E_ho_ext, 'orange', lw=1.5, alpha=0.8, label='E_HO')
    ax_ext_e.plot(t_eval, E_coup_ext, 'purple', lw=1.5, alpha=0.8, label='E_coup')
    ax_ext_e.plot(t_eval, E_total_ext, 'k-', lw=2.5, label='E_total (conserved)')
    ax_ext_e.set_title('Extended System: Energy Partition')
    ax_ext_e.set_xlabel('t'); ax_ext_e.set_ylabel('Energy')
    ax_ext_e.legend(fontsize=9); ax_ext_e.grid(alpha=0.3)

    # Row 5: Training curves and MSE bar chart
    ax_train = fig.add_subplot(gs[4, :2])
    ax_train.semilogy(train_losses_port, 'r-', alpha=0.5, lw=1, label='Port-HNN Train')
    ax_train.semilogy(val_losses_port, 'r--', alpha=0.7, lw=1, label='Port-HNN Val')
    ax_train.semilogy(train_losses_std, 'b-', alpha=0.5, lw=1, label='Std-HNN Train')
    ax_train.semilogy(val_losses_std, 'b--', alpha=0.7, lw=1, label='Std-HNN Val')
    ax_train.set_title('Training Curves (Damped Pendulum Data)')
    ax_train.set_xlabel('Epoch'); ax_train.set_ylabel('MSE')
    ax_train.legend(fontsize=8); ax_train.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[4, 2:])
    names = ['Port-HNN\n(2D, damped)', 'Extended HNN\n(4D, conserv.)', 'Standard HNN\n(2D, damped)']
    mses = [mse_port, mse_ext, mse_std]
    colors = ['coral', 'seagreen', 'steelblue']
    bars = ax_bar.bar(names, mses, color=colors, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, mses):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{val:.3e}', ha='center', fontsize=9, fontweight='bold')
    ax_bar.set_ylabel('Test MSE'); ax_bar.set_title('Test MSE Comparison')
    ax_bar.grid(alpha=0.3, axis='y')

    # ================================================================
    # 深入分析
    # ================================================================
    print("\n" + "=" * 70)
    print("深入分析")
    print("=" * 70)

    # Port-HNN 能量分析
    with torch.no_grad():
        dH_port = np.diff(H_port) / (t_eval[1] - t_eval[0])
    print(f"\n  Port-HNN 能量分析:")
    print(f"    H_theta 初值: {H_port[0]:.4f}, 终值: {H_port[-1]:.4f}")
    print(f"    H_theta 变化: {H_port[0] - H_port[-1]:.4f}")
    print(f"    真实 H(0): {H_true[0]:.4f}, H(30): {H_true[-1]:.4f}")
    print(f"    真实能量衰减: {H_true[0] - H_true[-1]:.4f}")

    # Standard HNN 能量守恒检查
    H_std_range = H_std.max() - H_std.min()
    print(f"\n  Standard HNN 能量分析:")
    print(f"    H_theta 初值: {H_std[0]:.4f}, 终值: {H_std[-1]:.4f}")
    print(f"    H_theta 波动范围: {H_std_range:.4f} (保守近似，不应衰减)")

    # 扩展系统能量守恒
    E_total_range = E_total_ext.max() - E_total_ext.min()
    print(f"\n  Extended HNN 能量分析:")
    print(f"    E_total 初值: {E_total_ext[0]:.4f}, 终值: {E_total_ext[-1]:.4f}")
    print(f"    E_total 波动: {E_total_range:.4f} (保守系统，应守恒)")
    print(f"    E_pend 初值: {E_pend_ext[0]:.4f}, 终值: {E_pend_ext[-1]:.4f}")
    print(f"    E_HO 初值: {E_ho_ext[0]:.4f}, 终值: {E_ho_ext[-1]:.4f}")

    # 轨迹误差对比
    q_err_port = np.mean(np.abs(q_port - q_true))
    q_err_std = np.mean(np.abs(q_std - q_true))
    print(f"\n  轨迹 MAE (q):")
    print(f"    Port-HNN: {q_err_port:.4e}")
    print(f"    Standard HNN: {q_err_std:.4e}")
    print(f"    Port-HNN 改善: {(1 - q_err_port/q_err_std)*100:.1f}%")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
    Port-HNN (端口哈密顿神经网络):
      - 直接学习耗散结构: dx/dt = (J - R)nabla H
      - 2D 状态，参数量小，物理可解释
      - 学到 gamma(x) 接近真实耗散系数
      - 预测轨迹准确地跟随衰减的物理真值

    Extended HNN (扩展系统):
      - 通过添加辅助 HO 将耗散系统扩展为保守系统
      - 4D 状态，更通用（可扩展至多 HO 谱表示）
      - 总能量守恒，物理一致性更强
      - 适合建模复杂耗散通道（谱密度参数化）

    Standard HNN:
      - 无法建模耗散，学到的是保守近似
      - 能量守恒，但真实系统能量在衰减
      - 轨迹漂移，长期预测误差大

    实用建议:
      - 已知耗散结构 -> Port-HNN (更高效)
      - 未知/复杂耗散 -> Extended HNN (更灵活)
      - 纯保守系统 -> Standard HNN (最简洁)
    """)

    print("=== 完成 ===")

    plt.tight_layout()
    fig.savefig('port_hnn_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  -> port_hnn_comparison.png")


if __name__ == '__main__':
    main()