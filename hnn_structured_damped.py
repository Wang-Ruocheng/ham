"""
Structured HNN on Damped Pendulum: 用外部谐振子建模耗散

数据: 阻尼单摆 + 谐振子 (非保守)
  dp1/dt = -sin(q1) - γ·p1 - ε·q2   ← 阻尼项 γ·p1 无法从任何 H 导出
  dq1/dt = p1
  dq2/dt = p2
  dp2/dt = -ω²·q2 - ε·q1

模型: 结构化 HNN (保守，辛结构)
  H = H_pend(q1,p1) + H_ho(q2,p2) + H_coup(q1,p1,q2,p2)
  dq1/dt = ∂H/∂p1,  dp1/dt = -∂H/∂q1

核心问题: 保守 HNN 能否通过 H_coup 隐式学习阻尼效应？
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
# 1. MLP 构建块
# ============================================================

def make_mlp(input_dim, hidden_dim, num_layers, output_dim=1):
    layers = []
    prev_dim = input_dim
    for i in range(num_layers):
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
# 2. 结构化 HNN (保守模型，不变)
# ============================================================

class StructuredHNN(nn.Module):
    """
    H = H_pend(q1,p1) + H_ho(q2,p2) + H_coup(q1,p1,q2,p2)
    H_ho 硬编码，H_pend 和 H_coup 学习
    """

    def __init__(self, omega=2.0, pend_hidden=200, coup_hidden=128, num_layers=3):
        super().__init__()
        self.omega = omega
        self.pendulum_net = make_mlp(2, pend_hidden, num_layers, 1)
        self.coupling_net = make_mlp(4, coup_hidden, num_layers, 1)

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
# 3. 阻尼耦合系统 (真实物理，数据生成)
# ============================================================

class DampedCoupledOscillator:
    """
    阻尼单摆 + 谐振子 (非保守系统)
    
    dp1/dt = -sin(q1) - γ·p1 - ε·q2
    dq1/dt = p1
    dq2/dt = p2
    dp2/dt = -ω²·q2 - ε·q1
    
    总能量不守恒: dE/dt = -γ·p1² ≤ 0 (能量持续流向谐振子再耗散)
    """

    def __init__(self, omega=2.0, epsilon=0.3, gamma=0.1):
        self.omega = omega; self.epsilon = epsilon; self.gamma = gamma

    def hamiltonian(self, q1, p1, q2, p2):
        """名义哈密顿量 (保守部分，不含阻尼)"""
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
        return [p1,
                -np.sin(q1) - self.gamma * p1 - self.epsilon * q2,
                p2,
                -self.omega**2 * q2 - self.epsilon * q1]

    def generate_trajectory(self, q10, p10, q20, p20, t_span, n_points=500):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, [q10, p10, q20, p20],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        q1, p1, q2, p2 = sol.y
        dq1 = p1; dp1 = -np.sin(q1) - self.gamma * p1 - self.epsilon * q2
        dq2 = p2; dp2 = -self.omega**2 * q2 - self.epsilon * q1
        return sol.t, q1, p1, q2, p2, dq1, dp1, dq2, dp2
# ============================================================
# 4. 数据生成
# ============================================================

def generate_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200,
                     q1_range=(-np.pi, np.pi), p1_range=(-2.0, 2.0),
                     q2_range=(-2.0, 2.0), p2_range=(-2.0, 2.0)):
    xs_list, dxs_list = [], []
    for _ in range(n_trajectories):
        q10 = np.random.uniform(*q1_range); p10 = np.random.uniform(*p1_range)
        q20 = np.random.uniform(*q2_range); p20 = np.random.uniform(*p2_range)
        _, q1, p1, q2, p2, dq1, dp1, dq2, dp2 = system.generate_trajectory(
            q10, p10, q20, p20, t_span, n_points=n_points)
        xs_list.append(np.stack([q1, p1, q2, p2], axis=1))
        dxs_list.append(np.stack([dq1, dp1, dq2, dp2], axis=1))
    return np.concatenate(xs_list, axis=0), np.concatenate(dxs_list, axis=0)


# ============================================================
# 5. 训练
# ============================================================

def train_hnn(model, train_loader, val_loader, epochs=2000, lr=1e-3,
              weight_decay=1e-4, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                val_loss += nn.MSELoss()(model.time_derivative(x_batch), dx_batch).item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return train_losses, val_losses
# ============================================================
# 6. 评估
# ============================================================

def integrate_hnn(model, state0, t_start, t_end, n_steps):
    model.eval()
    dt = (t_end - t_start) / n_steps
    traj = np.zeros((n_steps, len(state0))); traj[0] = state0
    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32)
        k1 = model.time_derivative(x).detach().numpy()[0]
        k2 = model.time_derivative(x + 0.5*dt*torch.tensor(k1, dtype=torch.float32)).detach().numpy()[0]
        k3 = model.time_derivative(x + 0.5*dt*torch.tensor(k2, dtype=torch.float32)).detach().numpy()[0]
        k4 = model.time_derivative(x + dt*torch.tensor(k3, dtype=torch.float32)).detach().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj
def evaluate(model, system, t_span=(0, 30)):
    """评估 HNN 在阻尼数据上的表现"""
    model.eval()
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.35)

    ax_q1  = fig.add_subplot(gs[0, 0]); ax_q2  = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[0, 2:])
    ax_phase1 = fig.add_subplot(gs[1, :2]); ax_phase2 = fig.add_subplot(gs[1, 2:])
    ax_dp1 = fig.add_subplot(gs[2, :2]); ax_dq1 = fig.add_subplot(gs[2, 2:])
    ax_H_pend = fig.add_subplot(gs[3, :2]); ax_H_coup = fig.add_subplot(gs[3, 2:])
    ax_H_learned = fig.add_subplot(gs[4, :])

    n_points = 500; t_eval = np.linspace(t_span[0], t_span[1], n_points)

    q10, p10 = 1.5, 0.0; q20, p20 = 0.0, 0.0

    sol = solve_ivp(system.dynamics, t_span, [q10, p10, q20, p20],
                    t_eval=t_eval, rtol=1e-9, atol=1e-9)
    t_true = sol.t; q1_t, p1_t, q2_t, p2_t = sol.y

    state0 = np.array([q10, p10, q20, p20])
    traj = integrate_hnn(model, state0, t_span[0], t_span[1], n_points)
    q1_p, p1_p, q2_p, p2_p = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    # 能量
    E_pend_t = system.pendulum_energy(q1_t, p1_t)
    E_ho_t = system.ho_energy(q2_t, p2_t)
    E_coup_t = system.coupling_energy(q1_t, q2_t)
    E_total_t = system.hamiltonian(q1_t, p1_t, q2_t, p2_t)

    xs_tensor = torch.tensor(np.stack([q1_t, p1_t, q2_t, p2_t], axis=1), dtype=torch.float32)
    with torch.no_grad():
        H_total_pred = model.forward(xs_tensor).numpy().flatten()
        H_pend_pred = model.pendulum_net(xs_tensor[:, :2]).numpy().flatten()
        H_coup_pred = model.coupling_net(xs_tensor).numpy().flatten()

    # ---- 行 1: q1, q2, 能量 ----
    ax_q1.plot(t_true, q1_t, 'b-', lw=2, label='True (damped)')
    ax_q1.plot(t_true, q1_p, 'r--', lw=2, label='HNN (conservative)')
    ax_q1.set_title('Pendulum Angle q1'); ax_q1.legend(fontsize=8); ax_q1.grid(alpha=0.3)

    ax_q2.plot(t_true, q2_t, 'b-', lw=2, label='True'); ax_q2.plot(t_true, q2_p, 'r--', lw=2, label='HNN')
    ax_q2.set_title('Oscillator Position q2'); ax_q2.legend(fontsize=8); ax_q2.grid(alpha=0.3)

    ax_energy.plot(t_true, E_pend_t, 'b-', lw=1.5, alpha=0.8, label='E_pend (True)')
    ax_energy.plot(t_true, E_ho_t, 'orange', lw=1.5, alpha=0.8, label='E_ho (True)')
    ax_energy.plot(t_true, E_total_t, 'k-', lw=2.5, label='E_total (True, decays)')
    ax_energy.plot(t_true, H_total_pred, 'r--', lw=2, label='H_theta (HNN, conserved)')
    ax_energy.set_title('Energy: True vs HNN'); ax_energy.legend(fontsize=8); ax_energy.grid(alpha=0.3)

    # ---- 行 2: 相空间 ----
    ax_phase1.plot(q1_t, p1_t, 'b-', lw=1.5, alpha=0.8, label='True (spiral)')
    ax_phase1.plot(q1_p, p1_p, 'r--', lw=1.5, alpha=0.8, label='HNN (closed)')
    ax_phase1.set_title('Phase: Pendulum'); ax_phase1.set_xlabel('q1'); ax_phase1.set_ylabel('p1')
    ax_phase1.legend(); ax_phase1.grid(alpha=0.3)

    ax_phase2.plot(q2_t, p2_t, 'b-', lw=1.5, alpha=0.8, label='True')
    ax_phase2.plot(q2_p, p2_p, 'r--', lw=1.5, alpha=0.8, label='HNN')
    ax_phase2.set_title('Phase: Oscillator'); ax_phase2.set_xlabel('q2'); ax_phase2.set_ylabel('p2')
    ax_phase2.legend(); ax_phase2.grid(alpha=0.3)

    # ---- 行 3: dp1/dt 和 dq1/dt ----
    dp1_true = -np.sin(q1_t) - system.gamma * p1_t - system.epsilon * q2_t
    dt_val = t_eval[1] - t_eval[0]
    dp1_hnn = -np.gradient(p1_p, dt_val)
    dq1_true = p1_t; dq1_hnn = np.gradient(q1_p, dt_val)

    ax_dp1.plot(t_true, dp1_true, 'b-', lw=1.5, label='dp1/dt True')
    ax_dp1.plot(t_true, dp1_hnn, 'r--', lw=1.5, label='dp1/dt HNN')
    damping = -system.gamma * p1_t
    ax_dp1.fill_between(t_true, 0, damping, alpha=0.2, color='red', label='-gamma*p1')
    ax_dp1.set_title('dp1/dt: True vs HNN'); ax_dp1.legend(fontsize=8); ax_dp1.grid(alpha=0.3)

    ax_dq1.plot(t_true, dq1_true, 'b-', lw=1.5, label='dq1/dt True (=p1)')
    ax_dq1.plot(t_true, dq1_hnn, 'r--', lw=1.5, label='dq1/dt HNN')
    ax_dq1.set_title('dq1/dt: True vs HNN'); ax_dq1.legend(fontsize=8); ax_dq1.grid(alpha=0.3)

    # ---- 行 4: H_pend, H_coup ----
    ax_H_pend.plot(t_true, E_pend_t, 'b-', lw=2, label='True H_pend')
    ax_H_pend.plot(t_true, H_pend_pred, 'r--', lw=2, label='Learned H_pend')
    ax_H_pend.set_title('H_pend: True vs Learned'); ax_H_pend.legend(); ax_H_pend.grid(alpha=0.3)

    ax_H_coup.plot(t_true, E_coup_t, 'b-', lw=2, label='True H_coup (eps*q1*q2)')
    ax_H_coup.plot(t_true, H_coup_pred, 'r--', lw=2, label='Learned H_coup')
    ax_H_coup.set_title('H_coup: True vs Learned'); ax_H_coup.legend(); ax_H_coup.grid(alpha=0.3)

    # ---- 行 5: H_theta 守恒性 ----
    ax_H_learned.plot(t_true, H_total_pred, 'r-', lw=2, label='H_theta (HNN)')
    ax_H_learned.plot(t_true, E_total_t, 'b-', lw=2, alpha=0.6, label='E_total (True)')
    H_mean = np.mean(H_total_pred); H_std = np.std(H_total_pred)
    ax_H_learned.axhline(H_mean, color='r', ls=':', lw=1, label=f'mean H_theta = {H_mean:.4f}')
    ax_H_learned.fill_between(t_true, H_mean-H_std, H_mean+H_std, alpha=0.15, color='red')
    ax_H_learned.set_title(f'H_theta Conservation (std = {H_std:.4f})')
    ax_H_learned.legend(fontsize=8); ax_H_learned.set_xlabel('t'); ax_H_learned.grid(alpha=0.3)

    return fig, {
        'H_theta_mean': H_mean, 'H_theta_std': H_std,
        'E_true_initial': E_total_t[0], 'E_true_final': E_total_t[-1],
        'energy_loss': E_total_t[0] - E_total_t[-1],
    }
# ============================================================
# 7. 主程序
# ============================================================

def main():
    torch.manual_seed(42); np.random.seed(42)

    OMEGA = 2.0; EPSILON = 0.3; GAMMA = 0.1
    system = DampedCoupledOscillator(omega=OMEGA, epsilon=EPSILON, gamma=GAMMA)

    print("=" * 60)
    print("结构化 HNN 学习阻尼系统")
    print("=" * 60)
    print(f"  真实: dp1/dt = -sin(q1) - {GAMMA}*p1 - {EPSILON}*q2")
    print(f"  模型: H = H_pend(MLP) + H_ho(已知) + H_coup(MLP)")
    print(f"  HNN 是保守的，H_coup 必须隐式拟合阻尼")

    # 生成数据
    print("\n--- 生成数据 ---")
    xs, dxs = generate_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200)
    print(f"数据点数: {xs.shape[0]}")

    n_total = xs.shape[0]; indices = np.random.permutation(n_total)
    n_train = int(0.7 * n_total); n_val = int(0.15 * n_total)
    xs_t = torch.tensor(xs, dtype=torch.float32); dxs_t = torch.tensor(dxs, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(xs_t[indices[:n_train]], dxs_t[indices[:n_train]]),
                              batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(xs_t[indices[n_train:n_train+n_val]],
                                           dxs_t[indices[n_train:n_train+n_val]]),
                            batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(xs_t[indices[n_train+n_val:]],
                                            dxs_t[indices[n_train+n_val:]]),
                             batch_size=512, shuffle=False)
    print(f"训练: {n_train}, 验证: {n_val}, 测试: {n_total-n_train-n_val}")

    # 创建模型
    model = StructuredHNN(omega=OMEGA, pend_hidden=200, coup_hidden=128, num_layers=3)
    n_pend = sum(p.numel() for p in model.pendulum_net.parameters())
    n_coup = sum(p.numel() for p in model.coupling_net.parameters())
    print(f"\n参数: H_pend={n_pend}, H_coup={n_coup}, H_ho=0, 总计={n_pend+n_coup}")

    # 训练
    print("\n--- 训练 ---")
    train_losses, val_losses = train_hnn(model, train_loader, val_loader, epochs=2000, lr=1e-3)

    model.eval()
    test_loss = 0.0; n_test = n_total - n_train - n_val
    with torch.no_grad():
        for xb, dxb in test_loader:
            test_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
    test_loss /= n_test

    # 参考: 保守数据上的损失
    from hnn_structured import CoupledOscillator
    cons_sys = CoupledOscillator(omega=OMEGA, epsilon=EPSILON)
    from hnn_structured import generate_dataset as gen_cons
    xs_cons, dxs_cons = gen_cons(cons_sys, n_trajectories=80, t_span=(0, 15), n_points=200)
    xs_cons_t = torch.tensor(xs_cons, dtype=torch.float32); dxs_cons_t = torch.tensor(dxs_cons, dtype=torch.float32)
    cons_loader = DataLoader(TensorDataset(xs_cons_t, dxs_cons_t), batch_size=512, shuffle=False)
    cons_loss = 0.0
    with torch.no_grad():
        for xb, dxb in cons_loader:
            cons_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
    cons_loss /= len(xs_cons)

    print(f"\n测试 MSE (阻尼数据): {test_loss:.6e}")
    print(f"测试 MSE (保守数据): {cons_loss:.6e}")
    print(f"阻尼/保守 损失比: {test_loss/cons_loss:.2f}x")

    # 可视化
    print("\n--- 可视化 ---")
    fig_loss, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, 'b-', alpha=0.7, lw=1, label='Train')
    ax.semilogy(val_losses, 'r-', alpha=0.7, lw=1, label='Val')
    ax.set_title('Structured HNN on Damped Data'); ax.legend(); ax.grid(alpha=0.3)
    fig_loss.savefig('damped_loss.png', dpi=150, bbox_inches='tight'); plt.close()
    print("  -> damped_loss.png")

    fig_eval, stats = evaluate(model, system, t_span=(0, 30))
    fig_eval.savefig('damped_evaluation.png', dpi=150, bbox_inches='tight'); plt.close()
    print("  -> damped_evaluation.png")

    print("\n--- 分析 ---")
    print(f"  H_theta 均值: {stats['H_theta_mean']:.4f}")
    print(f"  H_theta 标准差: {stats['H_theta_std']:.4f}")
    print(f"  真实能量初值: {stats['E_true_initial']:.4f}")
    print(f"  真实能量终值: {stats['E_true_final']:.4f}")
    print(f"  能量耗散量: {stats['energy_loss']:.4f}")
    print(f"\n  HNN 学到的 H_theta 近似守恒 (std={stats['H_theta_std']:.4f})")
    print(f"  真实能量耗散了 {stats['energy_loss']:.4f}")
    H_std = stats['H_theta_std']
    if H_std < 0.01:
        print("  -> H_coup 部分吸收了阻尼效应，H_theta 接近守恒")
    else:
        print("  -> H_theta 波动较大，单个谐振子不足以完全吸收阻尼")


if __name__ == '__main__':
    main()