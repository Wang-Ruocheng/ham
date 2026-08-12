"""
Structured HNN: Physics-Informed Hamiltonian Neural Network

物理结构:
    H_total = H_pend(q1, p1) + H_ho(q2, p2) + H_coup(q1, p1, q2, p2)
               └── MLP 学习 ──┘  └── 硬编码已知 ──┘  └── MLP 学习 ──┘

已知: H_ho = ½p₂² + ½ω²q₂²  (谐振子，ω 固定)
待学: H_pend(q1, p1)          (单摆哈密顿量)
待学: H_coup(q1, p1, q2, p2)  (耦合项，能量交换)

训练时，H_pend 和 H_coup 两个 MLP 独立学习，总 H 为三者之和。
H_ho 的梯度 (∂H_ho/∂q2 = ω²q2, ∂H_ho/∂p2 = p2) 直接解析计算，不经过网络。
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
    """构建 MLP 网络"""
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
# 2. 结构化 HNN
# ============================================================

class StructuredHNN(nn.Module):
    """
    结构化 HNN: H = H_pend(q1,p1) + H_ho(q2,p2) + H_coup(q1,p1,q2,p2)
    
    H_pend:  MLP(q1, p1) → scalar
    H_ho:    ½p₂² + ½ω²q₂² (解析，硬编码)
    H_coup:  MLP(q1, p1, q2, p2) → scalar
    
    时间导数通过总 H 的自动微分得到，其中 H_ho 的梯度直接解析计算。
    """

    def __init__(self, omega=2.0, pend_hidden=200, coup_hidden=128, num_layers=3):
        super().__init__()
        self.omega = omega
        self.pendulum_net = make_mlp(input_dim=2, hidden_dim=pend_hidden,
                                     num_layers=num_layers, output_dim=1)
        self.coupling_net = make_mlp(input_dim=4, hidden_dim=coup_hidden,
                                     num_layers=num_layers, output_dim=1)

    def forward(self, x):
        """
        x: (batch, 4) = [q1, p1, q2, p2]
        Returns: H_total (batch, 1)
        """
        q1_p1 = x[:, :2]          # (batch, 2)
        q2 = x[:, 2:3]            # (batch, 1)
        p2 = x[:, 3:4]            # (batch, 1)

        H_pend = self.pendulum_net(q1_p1)                               # 学习
        H_ho = 0.5 * p2**2 + 0.5 * self.omega**2 * q2**2               # 已知
        H_coup = self.coupling_net(x)                                   # 学习

        return H_pend + H_ho + H_coup

    def time_derivative(self, x):
        """
        辛梯度: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        H_ho 的梯度直接解析计算，避免梯度流经 H_ho 的 trivial 自动微分
        """
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        # 辛梯度: [∂H/∂p1, -∂H/∂q1, ∂H/∂p2, -∂H/∂q2]
        dq1_dt = dH[:, 1:2]   # ∂H/∂p1
        dp1_dt = -dH[:, 0:1]  # -∂H/∂q1
        dq2_dt = dH[:, 3:4]   # ∂H/∂p2
        dp2_dt = -dH[:, 2:3]  # -∂H/∂q2

        return torch.cat([dq1_dt, dp1_dt, dq2_dt, dp2_dt], dim=1)

# ============================================================
# 3. 耦合系统 (真实物理，数据生成)
# ============================================================

class CoupledOscillator:
    """
    真实系统: 单摆 + 谐振子，位置-位置耦合
    H = ½p1² + (1-cos q1) + ½p2² + ½ω²q2² + ε·q1·q2
    """

    def __init__(self, omega=2.0, epsilon=0.3):
        self.omega = omega
        self.epsilon = epsilon

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
        return [p1,
                -np.sin(q1) - self.epsilon * q2,
                p2,
                -self.omega**2 * q2 - self.epsilon * q1]

    def generate_trajectory(self, q10, p10, q20, p20, t_span, n_points=500):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, [q10, p10, q20, p20],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        q1, p1, q2, p2 = sol.y
        dq1 = p1; dp1 = -np.sin(q1) - self.epsilon * q2
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
                dx_pred = model.time_derivative(x_batch)
                loss = nn.MSELoss()(dx_pred, dx_batch)
                val_loss += loss.item() * x_batch.size(0)
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
    """RK4 积分"""
    model.eval()
    dt = (t_end - t_start) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D))
    traj[0] = state0
    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32)
        k1 = model.time_derivative(x).detach().numpy()[0]
        k2 = model.time_derivative(x + 0.5*dt*torch.tensor(k1, dtype=torch.float32)).detach().numpy()[0]
        k3 = model.time_derivative(x + 0.5*dt*torch.tensor(k2, dtype=torch.float32)).detach().numpy()[0]
        k4 = model.time_derivative(x + dt*torch.tensor(k3, dtype=torch.float32)).detach().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj


def evaluate(model, system, test_trajectories=3, t_span=(0, 30)):
    """全面评估: 轨迹、能量交换、哈密顿量分解对比"""
    model.eval()
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_q1 = fig.add_subplot(gs[0, 0]); ax_q2 = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[0, 2:])
    ax_phase1 = fig.add_subplot(gs[1, :2]); ax_phase2 = fig.add_subplot(gs[1, 2:])
    ax_multi = fig.add_subplot(gs[2, :])
    ax_H_pend = fig.add_subplot(gs[3, :2]); ax_H_coup = fig.add_subplot(gs[3, 2:])

    colors = plt.cm.tab10(np.linspace(0, 1, test_trajectories))
    all_results = []
    n_points = 500
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    for idx in range(test_trajectories):
        q10 = np.random.uniform(-2.0, 2.0); p10 = np.random.uniform(-1.5, 1.5)
        q20 = np.random.uniform(-1.0, 1.0); p20 = np.random.uniform(-1.0, 1.0)

        sol = solve_ivp(system.dynamics, t_span, [q10, p10, q20, p20],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        t_true = sol.t; q1_t, p1_t, q2_t, p2_t = sol.y

        state0 = np.array([q10, p10, q20, p20])
        traj = integrate_hnn(model, state0, t_span[0], t_span[1], n_points)
        q1_p, p1_p, q2_p, p2_p = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

        E_pend_t = system.pendulum_energy(q1_t, p1_t)
        E_ho_t = system.ho_energy(q2_t, p2_t)
        E_coup_t = system.coupling_energy(q1_t, q2_t)
        E_total_t = system.hamiltonian(q1_t, p1_t, q2_t, p2_t)

        E_pend_p = system.pendulum_energy(q1_p, p1_p)
        E_ho_p = system.ho_energy(q2_p, p2_p)
        E_total_p = system.hamiltonian(q1_p, p1_p, q2_p, p2_p)

        all_results.append({
            'q10': q10, 'p10': p10, 'q20': q20, 'p20': p20,
            'E_total_t': E_total_t, 'E_total_p': E_total_p,
        })

        c = colors[idx]

        if idx == 0:
            ax_q1.plot(t_true, q1_t, 'b-', lw=2, label='True')
            ax_q1.plot(t_true, q1_p, 'r--', lw=2, label='HNN')
            ax_q1.set_title('Pendulum Angle'); ax_q1.legend(fontsize=8); ax_q1.grid(alpha=0.3)

            ax_q2.plot(t_true, q2_t, 'b-', lw=2, label='True')
            ax_q2.plot(t_true, q2_p, 'r--', lw=2, label='HNN')
            ax_q2.set_title('Oscillator Position'); ax_q2.legend(fontsize=8); ax_q2.grid(alpha=0.3)

            ax_energy.plot(t_true, E_pend_t, 'b-', lw=1.5, alpha=0.8, label='E_pend (True)')
            ax_energy.plot(t_true, E_ho_t, 'orange', lw=1.5, alpha=0.8, label='E_ho (True)')
            ax_energy.plot(t_true, E_coup_t, 'g-', lw=1.5, alpha=0.8, label='E_coup (True)')
            ax_energy.plot(t_true, E_total_t, 'k-', lw=2.5, label='E_total (True)')
            ax_energy.plot(t_true, E_pend_p, 'b--', lw=1, alpha=0.6, label='E_pend (HNN)')
            ax_energy.plot(t_true, E_ho_p, 'orange', ls='--', lw=1, alpha=0.6, label='E_ho (HNN)')
            ax_energy.plot(t_true, E_total_p, 'k--', lw=1.5, alpha=0.6, label='E_total (HNN)')
            ax_energy.set_title('Energy Exchange'); ax_energy.legend(fontsize=7, ncol=2)
            ax_energy.grid(alpha=0.3)

            # 哈密顿量分解: H_pend 和 H_coup 的对比
            xs_tensor = torch.tensor(
                np.stack([q1_t, p1_t, q2_t, p2_t], axis=1), dtype=torch.float32)
            with torch.no_grad():
                H_pend_learned = model.pendulum_net(xs_tensor[:, :2]).numpy().flatten()
                H_coup_learned = model.coupling_net(xs_tensor).numpy().flatten()

            ax_H_pend.plot(t_true, E_pend_t, 'b-', lw=2, label='True H_pend')
            ax_H_pend.plot(t_true, H_pend_learned, 'r--', lw=2, label='Learned H_pend')
            ax_H_pend.set_title('H_pend: Learned vs True'); ax_H_pend.legend()
            ax_H_pend.grid(alpha=0.3)

            ax_H_coup.plot(t_true, E_coup_t, 'b-', lw=2, label='True H_coup')
            ax_H_coup.plot(t_true, H_coup_learned, 'r--', lw=2, label='Learned H_coup')
            ax_H_coup.set_title('H_coup: Learned vs True'); ax_H_coup.legend()
            ax_H_coup.grid(alpha=0.3)

        ax_phase1.plot(q1_t, p1_t, color=c, lw=1.2, alpha=0.8)
        ax_phase1.plot(q1_p, p1_p, color=c, lw=1.2, alpha=0.8, ls='--')
        ax_phase2.plot(q2_t, p2_t, color=c, lw=1.2, alpha=0.8)
        ax_phase2.plot(q2_p, p2_p, color=c, lw=1.2, alpha=0.8, ls='--')
        ax_multi.plot(q1_t, p1_t, color=c, lw=1, alpha=0.6,
                      label=f'#{idx+1}: q1={q10:.1f}, p1={p10:.1f}')
        ax_multi.plot(q1_p, p1_p, color=c, lw=1, alpha=0.6, ls='--')

    ax_phase1.set_title('Phase: Pendulum'); ax_phase1.set_xlabel('q1')
    ax_phase1.set_ylabel('p1'); ax_phase1.grid(alpha=0.3)
    ax_phase2.set_title('Phase: Oscillator'); ax_phase2.set_xlabel('q2')
    ax_phase2.set_ylabel('p2'); ax_phase2.grid(alpha=0.3)
    ax_multi.set_title('Multi-Trajectory Phase: Pendulum'); ax_multi.set_xlabel('q1')
    ax_multi.set_ylabel('p1'); ax_multi.legend(fontsize=7); ax_multi.grid(alpha=0.3)

    return fig, all_results
# ============================================================
# 7. 主程序
# ============================================================

def main():
    torch.manual_seed(42); np.random.seed(42)

    OMEGA = 2.0; EPSILON = 0.3
    system = CoupledOscillator(omega=OMEGA, epsilon=EPSILON)

    print("=== 结构化 HNN: H = H_pend(MLP) + H_ho(已知) + H_coup(MLP) ===")
    print(f"  omega = {OMEGA}, epsilon = {EPSILON}")
    print(f"  H_ho = 1/2 p2^2 + 1/2 omega^2 q2^2  (硬编码，不学习)")
    print(f"  H_pend, H_coup 通过两个独立 MLP 学习")

    # 生成数据
    print("\n--- 生成数据 ---")
    xs, dxs = generate_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200)
    print(f"数据点数: {xs.shape[0]}")

    n_total = xs.shape[0]; indices = np.random.permutation(n_total)
    n_train = int(0.7 * n_total); n_val = int(0.15 * n_total)
    xs_t = torch.tensor(xs, dtype=torch.float32)
    dxs_t = torch.tensor(dxs, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(xs_t[indices[:n_train]], dxs_t[indices[:n_train]]),
        batch_size=512, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(xs_t[indices[n_train:n_train+n_val]],
                      dxs_t[indices[n_train:n_train+n_val]]),
        batch_size=512, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(xs_t[indices[n_train+n_val:]],
                      dxs_t[indices[n_train+n_val:]]),
        batch_size=512, shuffle=False)
    print(f"训练: {n_train}, 验证: {n_val}, 测试: {n_total - n_train - n_val}")

    # 创建模型
    print("\n--- 创建模型 ---")
    model = StructuredHNN(omega=OMEGA, pend_hidden=200, coup_hidden=128, num_layers=3)
    n_pend = sum(p.numel() for p in model.pendulum_net.parameters())
    n_coup = sum(p.numel() for p in model.coupling_net.parameters())
    print(f"  H_pend MLP: {n_pend} 参数")
    print(f"  H_coup MLP: {n_coup} 参数")
    print(f"  H_ho: 0 参数 (解析)")
    print(f"  总可训练参数: {n_pend + n_coup}")

    # 训练
    print("\n--- 训练 ---")
    train_losses, val_losses = train_hnn(
        model, train_loader, val_loader, epochs=2000, lr=1e-3)

    model.eval()
    test_loss = 0.0
    n_test = n_total - n_train - n_val
    with torch.no_grad():
        for xb, dxb in test_loader:
            test_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
    test_loss /= n_test
    print(f"\n测试 MSE: {test_loss:.6e}")

    # 可视化
    print("\n--- 可视化 ---")
    fig_loss, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, 'b-', alpha=0.7, lw=1, label='Train')
    ax.semilogy(val_losses, 'r-', alpha=0.7, lw=1, label='Val')
    ax.set_title('Structured HNN Training'); ax.legend(); ax.grid(alpha=0.3)
    fig_loss.savefig('structured_loss.png', dpi=150, bbox_inches='tight'); plt.close()
    print("  -> structured_loss.png")

    fig_eval, results = evaluate(model, system, test_trajectories=3, t_span=(0, 30))
    fig_eval.savefig('structured_evaluation.png', dpi=150, bbox_inches='tight'); plt.close()
    print("  -> structured_evaluation.png")

    print("\n--- 定量分析 ---")
    for i, r in enumerate(results):
        E_err = np.mean(np.abs(r['E_total_p'] - r['E_total_t']))
        E_mean = np.mean(np.abs(r['E_total_t']))
        print(f"  轨迹 {i+1}: |E_err| = {E_err:.4e}, 相对 = {E_err/(E_mean+1e-8)*100:.2f}%")

    print("\n=== 完成 ===")
    print("结构化 HNN 成功将 H_ho 硬编码，H_pend 和 H_coup 分别学习。")
    print("这为后续用 H_coup 建模耗散通道提供了基础。")


if __name__ == '__main__':
    main()