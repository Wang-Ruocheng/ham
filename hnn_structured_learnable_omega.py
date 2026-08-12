"""
Structured HNN: 可学习 ω vs 固定 ω 对比

关键问题: 将 HO 的频率 ω 从硬编码改为可学习参数（仅多 1 个参数），
         能否比固定 ω 带来更好的性能？

架构:
    H = H_pend(MLP) + H_ho(½p² + ½ω²q²) + H_coup(MLP)
    固定 ω: ω 已知，不参与训练
    可学习 ω: ω 作为 nn.Parameter，与 MLP 一起优化

对比: 在相同数据、相同种子下，比较两种方案的测试 MSE 和学到的 ω 值
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
# 2. 两种 HNN 变体
# ============================================================

class StructuredHNN_Fixed(nn.Module):
    """ω 固定 (基线)"""
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

    def get_omega(self):
        return self.omega


class StructuredHNN_Learnable(nn.Module):
    """ω 可学习 (仅多 1 个参数)"""
    def __init__(self, omega_init=2.0, pend_hidden=200, coup_hidden=128, num_layers=3):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(omega_init))
        self.pendulum_net = make_mlp(2, pend_hidden, num_layers)
        self.coupling_net = make_mlp(4, coup_hidden, num_layers)

    def forward(self, x):
        q1_p1 = x[:, :2]; q2 = x[:, 2:3]; p2 = x[:, 3:4]
        H_pend = self.pendulum_net(q1_p1)
        omega_sq = self.omega ** 2
        H_ho = 0.5 * p2**2 + 0.5 * omega_sq * q2**2
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

    def get_omega(self):
        return self.omega.item()


# ============================================================
# 3. 耦合系统 (真实物理)
# ============================================================

class CoupledOscillator:
    def __init__(self, omega=2.0, epsilon=0.3):
        self.omega = omega; self.epsilon = epsilon

    def dynamics(self, t, state):
        q1, p1, q2, p2 = state
        return [p1, -np.sin(q1) - self.epsilon * q2,
                p2, -self.omega**2 * q2 - self.epsilon * q1]

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
              weight_decay=1e-4, verbose=True, label=""):
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
                val_loss += nn.MSELoss()(
                    model.time_derivative(x_batch), dx_batch).item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 200 == 0:
            omega_val = model.get_omega()
            print(f"{label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"ω: {omega_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return train_losses, val_losses


def evaluate_test_mse(model, test_loader):
    model.eval()
    total_loss = 0.0; n = 0
    with torch.no_grad():
        for xb, dxb in test_loader:
            total_loss += nn.MSELoss()(
                model.time_derivative(xb), dxb).item() * xb.size(0)
            n += xb.size(0)
    return total_loss / n
# ============================================================
# 6. 主程序
# ============================================================

def main():
    torch.manual_seed(42); np.random.seed(42)

    TRUE_OMEGA = 2.0; EPSILON = 0.3

    print("=" * 60)
    print("Structured HNN: 固定 ω vs 可学习 ω")
    print("=" * 60)
    print(f"  真实 ω = {TRUE_OMEGA}")
    print(f"  固定 ω: 硬编码 ω={TRUE_OMEGA}，不参与训练")
    print(f"  可学习 ω: 初始化为 ω={TRUE_OMEGA}，作为 nn.Parameter 训练")
    print(f"  参数差异: 仅多 1 个标量参数")

    print("\n--- 生成数据 ---")
    system = CoupledOscillator(omega=TRUE_OMEGA, epsilon=EPSILON)
    xs, dxs = generate_dataset(system, n_trajectories=80, t_span=(0, 15), n_points=200)
    print(f"数据点数: {len(xs)}")

    n_total = len(xs); indices = np.random.permutation(n_total)
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

    # ========================
    # 方案 A: 固定 ω
    # ========================
    print("\n" + "=" * 60)
    print("方案 A: 固定 ω = 2.0 (基线)")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_fixed = StructuredHNN_Fixed(omega=TRUE_OMEGA, pend_hidden=200,
                                      coup_hidden=128, num_layers=3)
    n_fixed = sum(p.numel() for p in model_fixed.parameters())
    print(f"可训练参数: {n_fixed}")

    train_losses_fixed, val_losses_fixed = train_hnn(
        model_fixed, train_loader, val_loader, epochs=2000, lr=1e-3,
        label="[Fixed]")

    mse_fixed = evaluate_test_mse(model_fixed, test_loader)
    print(f"\n[Fixed] 测试 MSE: {mse_fixed:.6e}")

    # ========================
    # 方案 B: 可学习 ω
    # ========================
    print("\n" + "=" * 60)
    print("方案 B: 可学习 ω (初始 = 2.0)")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_learn = StructuredHNN_Learnable(omega_init=TRUE_OMEGA, pend_hidden=200,
                                          coup_hidden=128, num_layers=3)
    n_learn = sum(p.numel() for p in model_learn.parameters())
    print(f"可训练参数: {n_learn} (比固定多 {n_learn - n_fixed})")

    train_losses_learn, val_losses_learn = train_hnn(
        model_learn, train_loader, val_loader, epochs=2000, lr=1e-3,
        label="[Learn]")

    mse_learn = evaluate_test_mse(model_learn, test_loader)
    learned_omega = model_learn.get_omega()
    print(f"\n[Learn] 测试 MSE: {mse_learn:.6e}")
    print(f"[Learn] 学到的 ω: {learned_omega:.4f} (真实: {TRUE_OMEGA})")

    # ========================
    # 对比总结
    # ========================
    print(f"\n{'='*60}")
    print(f"对比总结")
    print(f"{'='*60}")
    print(f"  方案             | 参数数 | 测试 MSE    | ω")
    print(f"  {'─'*50}")
    print(f"  固定 ω (基线)    | {n_fixed:5d} | {mse_fixed:.4e} | {TRUE_OMEGA} (固定)")
    print(f"  可学习 ω         | {n_learn:5d} | {mse_learn:.4e} | {learned_omega:.4f} (学得)")
    print(f"  {'─'*50}")
    if mse_learn < mse_fixed:
        print(f"  >> 可学习 ω 更好! MSE 降低 {(1-mse_learn/mse_fixed)*100:.1f}%")
    else:
        print(f"  >> 可学习 ω 略差, MSE 升高 {(mse_learn/mse_fixed-1)*100:.1f}%")
    print(f"  >> ω 收敛到 {learned_omega:.4f} (真实 {TRUE_OMEGA}, 误差 {abs(learned_omega-TRUE_OMEGA)/TRUE_OMEGA*100:.1f}%)")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(train_losses_fixed, 'b-', alpha=0.5, lw=1, label='Fixed ω Train')
    ax.semilogy(val_losses_fixed, 'b--', alpha=0.7, lw=1, label='Fixed ω Val')
    ax.semilogy(train_losses_learn, 'r-', alpha=0.5, lw=1, label='Learn ω Train')
    ax.semilogy(val_losses_learn, 'r--', alpha=0.7, lw=1, label='Learn ω Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Training Curves'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    bars = ax.bar(['Fixed ω', 'Learn ω'], [mse_fixed, mse_learn],
                  color=['steelblue', 'coral'], alpha=0.8)
    for bar, val in zip(bars, [mse_fixed, mse_learn]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{val:.4e}', ha='center', fontsize=10)
    ax.set_ylabel('Test MSE'); ax.set_title('Test MSE Comparison')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('omega_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  -> omega_comparison.png")


if __name__ == '__main__':
    main()