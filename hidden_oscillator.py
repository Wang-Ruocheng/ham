"""
隐藏振子问题 (Hidden Oscillator Problem)
=========================================
两个线性谐振子耦合，观测者只能看到振子 1。

物理系统:
    H = ½p₁² + ½ω₁²q₁² + ½p₂² + ½ω₂²q₂² + ε·q₁·q₂

真实动力学 (4D 保守):
    dq₁/dt = p₁
    dp₁/dt = -ω₁²q₁ - ε·q₂
    dq₂/dt = p₂
    dp₂/dt = -ω₂²q₂ - ε·q₁

观测者视角 (2D 非马尔可夫):
    dp₁/dt = -ω₁²q₁ - ε·q₂(t)
    q₂(t) 依赖于 q₁ 的整个历史 → 记忆效应
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import TensorDataset, DataLoader


class CoupledHarmonicOscillators:
    """两个线性谐振子耦合: H = ½p₁² + ½ω₁²q₁² + ½p₂² + ½ω₂²q₂² + ε·q₁·q₂"""

    def __init__(self, omega1=1.0, omega2=np.sqrt(2), epsilon=0.5):
        self.omega1 = omega1
        self.omega2 = omega2          # 无理数比例避免共振
        self.epsilon = epsilon

    def hamiltonian(self, q1, p1, q2, p2):
        return (0.5 * p1**2 + 0.5 * self.omega1**2 * q1**2 +
                0.5 * p2**2 + 0.5 * self.omega2**2 * q2**2 +
                self.epsilon * q1 * q2)

    def pendulum_energy(self, q1, p1):
        return 0.5 * p1**2 + 0.5 * self.omega1**2 * q1**2

    def ho_energy(self, q2, p2):
        return 0.5 * p2**2 + 0.5 * self.omega2**2 * q2**2

    def coupling_energy(self, q1, q2):
        return self.epsilon * q1 * q2

    def dynamics(self, t, state):
        q1, p1, q2, p2 = state
        return [p1,
                -self.omega1**2 * q1 - self.epsilon * q2,
                p2,
                -self.omega2**2 * q2 - self.epsilon * q1]

    def generate_trajectory(self, q10, p10, q20, p20,
                            t_span=(0, 20), n_points=300):
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.dynamics, t_span, [q10, p10, q20, p20],
                        t_eval=t_eval, rtol=1e-9, atol=1e-9)
        q1, p1, q2, p2 = sol.y
        dq1 = p1
        dp1 = -self.omega1**2 * q1 - self.epsilon * q2
        dq2 = p2
        dp2 = -self.omega2**2 * q2 - self.epsilon * q1
        return sol.t, q1, p1, q2, p2, dq1, dp1, dq2, dp2


def generate_datasets(system, n_trajectories=100, t_span=(0, 20), n_points=300,
                      train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    生成训练/验证/测试数据。
    返回:
        - 4D data: (q₁,p₁,q₂,p₂) → (dq₁,dp₁,dq₂,dp₂)  用于 Extended HNN
        - 2D data: (q₁,p₁) → (dq₁,dp₁)                  用于 Port-HNN
    """
    np.random.seed(seed)
    xs_4d_list, dxs_4d_list = [], []
    xs_2d_list, dxs_2d_list = [], []

    for _ in range(n_trajectories):
        q10 = np.random.uniform(-2.0, 2.0)
        p10 = np.random.uniform(-2.0, 2.0)
        q20 = np.random.uniform(-2.0, 2.0)
        p20 = np.random.uniform(-2.0, 2.0)
        _, q1, p1, q2, p2, dq1, dp1, dq2, dp2 = system.generate_trajectory(
            q10, p10, q20, p20, t_span, n_points)

        xs_4d_list.append(np.stack([q1, p1, q2, p2], axis=1))
        dxs_4d_list.append(np.stack([dq1, dp1, dq2, dp2], axis=1))

        xs_2d_list.append(np.stack([q1, p1], axis=1))
        dxs_2d_list.append(np.stack([dq1, dp1], axis=1))

    xs_4d = np.concatenate(xs_4d_list, axis=0)
    dxs_4d = np.concatenate(dxs_4d_list, axis=0)
    xs_2d = np.concatenate(xs_2d_list, axis=0)
    dxs_2d = np.concatenate(dxs_2d_list, axis=0)

    n_total = xs_4d.shape[0]
    indices = np.random.permutation(n_total)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    def _make_loaders(xs, dxs):
        xs_t = torch.tensor(xs, dtype=torch.float32)
        dxs_t = torch.tensor(dxs, dtype=torch.float32)
        train = DataLoader(
            TensorDataset(xs_t[indices[:n_train]], dxs_t[indices[:n_train]]),
            batch_size=512, shuffle=True)
        val = DataLoader(
            TensorDataset(xs_t[indices[n_train:n_train+n_val]],
                          dxs_t[indices[n_train:n_train+n_val]]),
            batch_size=512, shuffle=False)
        test = DataLoader(
            TensorDataset(xs_t[indices[n_train+n_val:]],
                          dxs_t[indices[n_train+n_val:]]),
            batch_size=512, shuffle=False)
        return train, val, test

    loaders_4d = _make_loaders(xs_4d, dxs_4d)
    loaders_2d = _make_loaders(xs_2d, dxs_2d)

    print(f"数据: {n_total} 点 | 训练: {n_train} | 验证: {n_val} | 测试: {n_total - n_train - n_val}")
    return loaders_4d, loaders_2d


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sys = CoupledHarmonicOscillators(omega1=1.0, omega2=np.sqrt(2), epsilon=0.5)
    t, q1, p1, q2, p2, dq1, dp1, dq2, dp2 = sys.generate_trajectory(
        1.5, 0.0, 0.5, 0.0, t_span=(0, 30), n_points=500)

    E1 = sys.pendulum_energy(q1, p1)
    E2 = sys.ho_energy(q2, p2)
    Ec = sys.coupling_energy(q1, q2)
    Et = E1 + E2 + Ec

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, q1, 'b-', lw=2, label='q1 (observed)')
    axes[0, 0].plot(t, q2, 'r--', lw=1.5, label='q2 (hidden)')
    axes[0, 0].set_title('Positions'); axes[0, 0].set_xlabel('t')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, dq1, 'b-', lw=2, label='dq1/dt')
    axes[0, 1].plot(t, dp1, 'r-', lw=2, label='dp1/dt')
    axes[0, 1].set_title('Observed derivatives'); axes[0, 1].set_xlabel('t')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, E1, 'b-', lw=2, label='E1 (pendulum)')
    axes[1, 0].plot(t, E2, 'orange', lw=1.5, label='E2 (hidden HO)')
    axes[1, 0].plot(t, Ec, 'purple', lw=1.5, label='E_coup')
    axes[1, 0].plot(t, Et, 'k-', lw=2.5, label='E_total (conserved)')
    axes[1, 0].set_title('Energy Partition'); axes[1, 0].set_xlabel('t')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, np.abs(Et - Et[0]), 'k-', lw=1.5)
    axes[1, 1].set_title('|E_total(t) - E_total(0)|')
    axes[1, 1].set_xlabel('t'); axes[1, 1].set_ylabel('|dE|')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(alpha=0.3)

    fig.suptitle('Hidden Oscillator: Coupled HOs (w1=1.0, w2=sqrt(2), eps=0.5)', fontsize=14)
    plt.tight_layout()
    fig.savefig('hidden_oscillator_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("-> hidden_oscillator_demo.png")