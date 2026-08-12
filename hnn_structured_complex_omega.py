"""
Structured HNN: 复数 ω — 有限维度下的单向耗散

核心想法 (Caldeira-Leggett 图像):
  将 HO 的频率 ω 设为复数 ω = ω_R + iω_I。
  实数 ω → HO 与 Pendulum 之间只有周期性 Rabi 能量交换
  复数 ω → Im(ω) 引入 dp₂/dt 的阻尼项，HO 成为耗散通道：
          Pendulum → HO → 单向耗散 (不可逆)

模型:
  方案 A: 固定 ω (实数，基线)
  方案 B: 可学习 ω (实数)
   方案 C: 可学习复数 ω = ω_R + iω_I (MLP 耦合)
   方案 D: 可学习复数 ω + 乘积分解耦合 H_coup = f(q1,p1)·g(q2,p2)

数据: 单摆 + 耗散谐振子 (γ=0.1，阻尼在 HO 上)
  dp1/dt = -sin(q1) - ε·q2           ← 保守
  dq1/dt = p1
  dq2/dt = p2
  dp2/dt = -ω²·q2 - ε·q1 - γ·p2      ← 阻尼在 HO

复数 ω 模型的动力学:
  H_ho = ½p₂² + ½(ω_R² + ω_I²)q₂²           ← 实哈密顿量
  dp₂/dt = -∂H/∂q₂ - 2ω_I·p₂                  ← Im(ω) 引入阻尼
  (dq₁/dt, dp₁/dt, dq₂/dt 保持辛结构)
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
def make_mlp_small(input_dim, hidden_dim, output_dim=1):
    """小型 MLP: 用于乘积分解的子网络"""
    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh(),
              nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
              nn.Linear(hidden_dim, output_dim, bias=False)]
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net
# ============================================================
# 2. 四种 HNN 变体
# ============================================================

class StructuredHNN_Fixed(nn.Module):
    """ω 固定 (实数，基线)"""
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


class StructuredHNN_Real(nn.Module):
    """ω 可学习 (实数)"""
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
class StructuredHNN_Complex(nn.Module):
    """
    ω = ω_R + iω_I (复数，可学习)
    
    哈密顿量 (实): H_ho = ½p₂² + ½(ω_R² + ω_I²)q₂²
    运动方程 (非辛): dp₂/dt = -∂H/∂q₂ - 2ω_I·p₂
                     其余保持辛结构
    
    ω_I > 0 → HO 有耗散通道，能量单向流出
    """
    def __init__(self, omega_real_init=2.0, omega_imag_init=0.1,
                 pend_hidden=200, coup_hidden=128, num_layers=3):
        super().__init__()
        self.omega_real = nn.Parameter(torch.tensor(omega_real_init))
        self.omega_imag = nn.Parameter(torch.tensor(omega_imag_init))
        self.pendulum_net = make_mlp(2, pend_hidden, num_layers)
        self.coupling_net = make_mlp(4, coup_hidden, num_layers)

    def forward(self, x):
        """H = H_pend + H_ho(real) + H_coup"""
        q1_p1 = x[:, :2]; q2 = x[:, 2:3]; p2 = x[:, 3:4]
        H_pend = self.pendulum_net(q1_p1)
        # |ω|² = ω_R² + ω_I², 保证 H_ho 为实数
        omega_sq = self.omega_real ** 2 + self.omega_imag ** 2
        H_ho = 0.5 * p2**2 + 0.5 * omega_sq * q2**2
        H_coup = self.coupling_net(x)
        return H_pend + H_ho + H_coup

    def time_derivative(self, x):
        """
        混合动力学:
          dq/dt =  ∂H/∂p           (辛)
          dp/dt = -∂H/∂q - 2ω_I·p  (dp₂ 含非辛阻尼项)
        """
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        dq1 = dH[:, 1:2]; dp1 = -dH[:, 0:1]
        dq2 = dH[:, 3:4]
        # dp₂ 加入 Im(ω) 阻尼项
        p2 = x[:, 3:4]
        dp2 = -dH[:, 2:3] - 2.0 * self.omega_imag * p2

        return torch.cat([dq1, dp1, dq2, dp2], dim=1)

    def get_omega(self):
        return self.omega_real.item(), self.omega_imag.item()


class StructuredHNN_Complex_Product(nn.Module):
    """
    ω = ω_R + iω_I (复数) + 乘积分解耦合
    
    H_coup = f(q1,p1) · g(q2,p2)
    
    物理直觉: 耦合是"子系统 A 的状态" × "子系统 B 的状态"，
             而非任意的 4D 联合函数。
    参数效率: 两个 2→h→1 子网络 vs 一个 4→h→h→h→1 MLP
    """
    def __init__(self, omega_real_init=2.0, omega_imag_init=0.1,
                 pend_hidden=200, prod_hidden=32, num_layers=3):
        super().__init__()
        self.omega_real = nn.Parameter(torch.tensor(omega_real_init))
        self.omega_imag = nn.Parameter(torch.tensor(omega_imag_init))
        self.pendulum_net = make_mlp(2, pend_hidden, num_layers)
        # H_coup = f(q1,p1) * g(q2,p2)
        self.coupling_f = make_mlp_small(2, prod_hidden)  # Pendulum → scalar
        self.coupling_g = make_mlp_small(2, prod_hidden)  # HO → scalar

    def forward(self, x):
        q1_p1 = x[:, :2]; q2_p2 = x[:, 2:]
        q2 = x[:, 2:3]; p2 = x[:, 3:4]
        H_pend = self.pendulum_net(q1_p1)
        omega_sq = self.omega_real ** 2 + self.omega_imag ** 2
        H_ho = 0.5 * p2**2 + 0.5 * omega_sq * q2**2
        H_coup = self.coupling_f(q1_p1) * self.coupling_g(q2_p2)
        return H_pend + H_ho + H_coup

    def time_derivative(self, x):
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dq1 = dH[:, 1:2]; dp1 = -dH[:, 0:1]
        dq2 = dH[:, 3:4]
        p2 = x[:, 3:4]
        dp2 = -dH[:, 2:3] - 2.0 * self.omega_imag * p2
        return torch.cat([dq1, dp1, dq2, dp2], dim=1)

    def get_omega(self):
        return self.omega_real.item(), self.omega_imag.item()


# ============================================================
# 3. 阻尼耦合系统 (真实物理，数据生成)
# ============================================================

class DampedCoupledOscillator:
    """
    单摆 + 耗散谐振子 (阻尼在 HO 上)
    
    物理图像 (Caldeira-Leggett):
      Pendulum 通过耦合 ε·q₁·q₂ 将能量传给 HO，
      HO 通过自身阻尼 γ·p₂ 将能量耗散掉。
      → 等效于 Pendulum 感受到一个有效的耗散环境。
    
    dp1/dt = -sin(q1) - ε·q2           ← 保守 (无显式阻尼)
    dq1/dt = p1
    dq2/dt = p2
    dp2/dt = -ω²·q2 - ε·q1 - γ·p2      ← 阻尼在 HO 上
    
    总能量: dE/dt = -γ·p2² ≤ 0
    """

    def __init__(self, omega=2.0, epsilon=0.3, gamma=0.1):
        self.omega = omega; self.epsilon = epsilon; self.gamma = gamma

    def dynamics(self, t, state):
        q1, p1, q2, p2 = state
        return [p1,
                -np.sin(q1) - self.epsilon * q2,
                p2,
                -self.omega**2 * q2 - self.epsilon * q1 - self.gamma * p2]


# ============================================================
# 4. 数据生成
# ============================================================

def generate_trajectories(system, n_trajs=200, t_span=10.0, dt=0.05):
    t_eval = np.arange(0, t_span, dt)
    n_steps = len(t_eval)
    X = np.zeros((n_trajs * n_steps, 4))
    dX = np.zeros((n_trajs * n_steps, 4))
    
    for i in range(n_trajs):
        q1_0 = np.random.uniform(-np.pi, np.pi)
        p1_0 = np.random.uniform(-1.5, 1.5)
        q2_0 = np.random.uniform(-0.5, 0.5)
        p2_0 = np.random.uniform(-0.5, 0.5)
        y0 = [q1_0, p1_0, q2_0, p2_0]
        
        sol = solve_ivp(system.dynamics, [0, t_span], y0, t_eval=t_eval,
                        method='RK45', rtol=1e-9, atol=1e-12)
        traj = sol.y.T  # (n_steps, 4)
        
        idx = slice(i * n_steps, (i + 1) * n_steps)
        X[idx] = traj
        dX[idx] = np.array([system.dynamics(0, traj[j]) for j in range(n_steps)])
    
    return X, dX


# ============================================================
# 5. 训练
# ============================================================

def train_hnn(model, train_loader, val_loader, epochs=2000, lr=1e-3, label=""):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_dx in val_loader:
                pred = model.time_derivative(batch_x)
                val_loss += nn.MSELoss()(pred, batch_dx).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    
    return train_losses, val_losses


def evaluate_test_mse(model, test_loader):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch_x, batch_dx in test_loader:
            pred = model.time_derivative(batch_x)
            total_mse += nn.MSELoss()(pred, batch_dx).item() * batch_x.size(0)
    return total_mse / len(test_loader.dataset)


def integrate_hnn(model, state0, t_start, t_end, n_steps):
    """RK4 积分"""
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


# ============================================================
# 6. 主程序
# ============================================================

def main():
    TRUE_OMEGA = 2.0
    TRUE_EPSILON = 0.3
    TRUE_GAMMA = 0.1
    
    print("=" * 60)
    print("Structured HNN: 复数 ω 实验")
    print(f"真实参数: ω={TRUE_OMEGA}, ε={TRUE_EPSILON}, γ_HO={TRUE_GAMMA} (阻尼在 HO)")
    print("物理: Pendulum (保守) → 耦合 → HO (耗散 γ·p₂)")
    print("=" * 60)
    
    # 生成阻尼数据
    print("\n生成阻尼数据...")
    system = DampedCoupledOscillator(omega=TRUE_OMEGA, epsilon=TRUE_EPSILON,
                                     gamma=TRUE_GAMMA)
    X, dX = generate_trajectories(system, n_trajs=100, t_span=5.0, dt=0.05)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dX_tensor = torch.tensor(dX, dtype=torch.float32)
    
    # 划分训练/验证/测试
    n_total = len(X_tensor)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_loader = DataLoader(
        TensorDataset(X_tensor[train_idx], dX_tensor[train_idx]),
        batch_size=512, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_tensor[val_idx], dX_tensor[val_idx]),
        batch_size=512, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(X_tensor[test_idx], dX_tensor[test_idx]),
        batch_size=512, shuffle=False)
    
    print(f"训练: {n_train}, 验证: {n_val}, 测试: {n_total - n_train - n_val}")
# ========================
    # 方案 A: 固定 ω
    # ========================
    print("\n" + "=" * 60)
    print("方案 A: 固定 ω = 2.0 (实数，基线)")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_fixed = StructuredHNN_Fixed(omega=TRUE_OMEGA, pend_hidden=200,
                                      coup_hidden=128, num_layers=3)
    n_fixed = sum(p.numel() for p in model_fixed.parameters())
    print(f"可训练参数: {n_fixed}")
    
    tl_fixed, vl_fixed = train_hnn(
        model_fixed, train_loader, val_loader, epochs=500, lr=1e-3,
        label="[Fixed]")
    mse_fixed = evaluate_test_mse(model_fixed, test_loader)
    print(f"\n[Fixed] 测试 MSE: {mse_fixed:.6e}")
    
    # ========================
    # 方案 B: 可学习 ω (实数)
    # ========================
    print("\n" + "=" * 60)
    print("方案 B: 可学习 ω (实数)")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_real = StructuredHNN_Real(omega_init=TRUE_OMEGA, pend_hidden=200,
                                    coup_hidden=128, num_layers=3)
    n_real = sum(p.numel() for p in model_real.parameters())
    print(f"可训练参数: {n_real} (比固定多 {n_real - n_fixed})")
    
    tl_real, vl_real = train_hnn(
        model_real, train_loader, val_loader, epochs=500, lr=1e-3,
        label="[Real ω]")
    mse_real = evaluate_test_mse(model_real, test_loader)
    omega_real = model_real.get_omega()
    print(f"\n[Real ω] 测试 MSE: {mse_real:.6e}")
    print(f"[Real ω] 学到的 ω: {omega_real:.4f} (真实: {TRUE_OMEGA})")
    
    # ========================
    # 方案 C: 可学习复数 ω
    # ========================
    print("\n" + "=" * 60)
    print("方案 C: 可学习复数 ω = ω_R + iω_I")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_complex = StructuredHNN_Complex(
        omega_real_init=TRUE_OMEGA, omega_imag_init=0.1,
        pend_hidden=200, coup_hidden=128, num_layers=3)
    n_complex = sum(p.numel() for p in model_complex.parameters())
    print(f"可训练参数: {n_complex} (比固定多 {n_complex - n_fixed})")
    
    tl_complex, vl_complex = train_hnn(
        model_complex, train_loader, val_loader, epochs=500, lr=1e-3,
        label="[Complex]")
    mse_complex = evaluate_test_mse(model_complex, test_loader)
    omega_R, omega_I = model_complex.get_omega()
    print(f"\n[Complex] 测试 MSE: {mse_complex:.6e}")
    print(f"[Complex] 学到的 ω = {omega_R:.4f} + i·{omega_I:.4f}")
    omega_R_mlp, omega_I_mlp = omega_R, omega_I  # rename for clarity
    
    # ========================
    # 方案 D: 复数 ω + 乘积分解耦合
    # ========================
    print("\n" + "=" * 60)
    print("方案 D: 复数 ω + 乘积分解耦合 H_coup = f(q1,p1)·g(q2,p2)")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    model_prod = StructuredHNN_Complex_Product(
        omega_real_init=TRUE_OMEGA, omega_imag_init=0.1,
        pend_hidden=200, prod_hidden=32, num_layers=3)
    n_prod = sum(p.numel() for p in model_prod.parameters())
    print(f"可训练参数: {n_prod} (vs MLP 耦合: {n_complex})")
    
    tl_prod, vl_prod = train_hnn(
        model_prod, train_loader, val_loader, epochs=500, lr=1e-3,
        label="[Complex-Prod]")
    mse_prod = evaluate_test_mse(model_prod, test_loader)
    omega_R_prod, omega_I_prod = model_prod.get_omega()
    print(f"\n[Complex-Prod] 测试 MSE: {mse_prod:.6e}")
    print(f"[Complex-Prod] 学到的 ω = {omega_R_prod:.4f} + i·{omega_I_prod:.4f}")
    print(f"  耦合参数: {sum(p.numel() for p in model_prod.coupling_f.parameters()) + sum(p.numel() for p in model_prod.coupling_g.parameters())} (乘积分解) vs {sum(p.numel() for p in model_complex.coupling_net.parameters())} (MLP)")
# ========================
    # 对比总结
    # ========================
    print(f"\n{'='*70}")
    print(f"对比总结")
    print(f"{'='*70}")
    print(f"  方案                   | 耦合参数 | 总参数 | 测试 MSE    | ω")
    print(f"  {'─'*75}")
    coup_fixed = sum(p.numel() for p in model_fixed.coupling_net.parameters())
    coup_real = sum(p.numel() for p in model_real.coupling_net.parameters())
    coup_mlp  = sum(p.numel() for p in model_complex.coupling_net.parameters())
    coup_prod = sum(p.numel() for p in model_prod.coupling_f.parameters()) + sum(p.numel() for p in model_prod.coupling_g.parameters())
    print(f"  固定 ω (MLP 耦合)      | {coup_fixed:5d}  | {n_fixed:5d} | {mse_fixed:.4e} | {TRUE_OMEGA} (固定)")
    print(f"  可学习 ω (MLP 耦合)    | {coup_real:5d}  | {n_real:5d} | {mse_real:.4e} | {omega_real:.4f}")
    print(f"  复数 ω (MLP 耦合)      | {coup_mlp:5d}  | {n_complex:5d} | {mse_complex:.4e} | {omega_R_mlp:.4f} + i·{omega_I_mlp:.4f}")
    print(f"  复数 ω (乘积分解)      | {coup_prod:5d}  | {n_prod:5d} | {mse_prod:.4e} | {omega_R_prod:.4f} + i·{omega_I_prod:.4f}")
    print(f"  {'─'*75}")
    
    best_mse = min(mse_fixed, mse_real, mse_complex, mse_prod)
    best_name = ""
    if mse_prod == best_mse:
        print(f"  >> 乘积分解耦合最优! MSE = {mse_prod:.4e}")
        print(f"  >> 耦合参数: {coup_prod} (乘积分解) vs {coup_mlp} (MLP) — 节省 {coup_mlp - coup_prod} 参数")
        print(f"  >> Im(ω) = {omega_I_prod:.4f} → HO 阻尼 = {2*omega_I_prod:.4f} (真实 γ_HO = {TRUE_GAMMA})")
        best_name = "Complex-Prod"
    elif mse_complex == best_mse:
        print(f"  >> MLP 耦合最优! MSE = {mse_complex:.4e}")
        print(f"  >> Im(ω) = {omega_I_mlp:.4f} → HO 阻尼 = {2*omega_I_mlp:.4f} (真实 γ_HO = {TRUE_GAMMA})")
        best_name = "Complex-MLP"
    elif mse_real == best_mse:
        print(f"  >> 实数可学习 ω 最优")
        best_name = "Real"
    else:
        print(f"  >> 固定 ω 最优")
        best_name = "Fixed"
    
    # ========================
    # 轨迹预测 & 相空间可视化
    # ========================
    print("\n生成轨迹预测...")
    t_span = 10.0; n_points = 300
    t_eval = np.linspace(0, t_span, n_points)
    
    # 初始条件
    q10, p10 = 1.5, 0.0
    q20, p20 = 0.2, 0.0
    state0 = np.array([q10, p10, q20, p20])
    
    # 真实轨迹
    sol = solve_ivp(system.dynamics, [0, t_span], state0,
                    t_eval=t_eval, rtol=1e-9, atol=1e-9)
    t_true = sol.t
    q1_t, p1_t, q2_t, p2_t = sol.y
    
    # 四模型预测
    traj_fixed = integrate_hnn(model_fixed, state0, 0, t_span, n_points)
    traj_real = integrate_hnn(model_real, state0, 0, t_span, n_points)
    traj_complex = integrate_hnn(model_complex, state0, 0, t_span, n_points)
    traj_prod = integrate_hnn(model_prod, state0, 0, t_span, n_points)
    
    # 绘图: 2行3列 — 相空间 + 时间序列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    C = {'True': 'black', 'Fixed': 'steelblue', 'Real': 'mediumseagreen',
         'Complex-MLP': 'coral', 'Complex-Prod': 'darkviolet'}
    S = {'True': '-', 'Fixed': '--', 'Real': '--', 'Complex-MLP': '--', 'Complex-Prod': '-.'}
    W = {'True': 2.0, 'Fixed': 1.5, 'Real': 1.5, 'Complex-MLP': 1.5, 'Complex-Prod': 1.5}
    
    # (q1, p1) 相空间
    ax = axes[0, 0]
    ax.plot(q1_t, p1_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(traj_fixed[:, 0], traj_fixed[:, 1], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(traj_real[:, 0], traj_real[:, 1], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(traj_complex[:, 0], traj_complex[:, 1], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(traj_prod[:, 0], traj_prod[:, 1], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('q1'); ax.set_ylabel('p1')
    ax.set_title('Pendulum Phase Space (q1, p1)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # (q2, p2) 相空间
    ax = axes[0, 1]
    ax.plot(q2_t, p2_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(traj_fixed[:, 2], traj_fixed[:, 3], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(traj_real[:, 2], traj_real[:, 3], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(traj_complex[:, 2], traj_complex[:, 3], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(traj_prod[:, 2], traj_prod[:, 3], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('q2'); ax.set_ylabel('p2')
    ax.set_title('HO Phase Space (q2, p2)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # q1 时间序列
    ax = axes[0, 2]
    ax.plot(t_true, q1_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(t_eval, traj_fixed[:, 0], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(t_eval, traj_real[:, 0], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(t_eval, traj_complex[:, 0], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(t_eval, traj_prod[:, 0], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('t'); ax.set_ylabel('q1')
    ax.set_title('Pendulum Angle q1(t)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # p1 时间序列
    ax = axes[1, 0]
    ax.plot(t_true, p1_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(t_eval, traj_fixed[:, 1], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(t_eval, traj_real[:, 1], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(t_eval, traj_complex[:, 1], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(t_eval, traj_prod[:, 1], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('t'); ax.set_ylabel('p1')
    ax.set_title('Pendulum Momentum p1(t)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # q2 时间序列
    ax = axes[1, 1]
    ax.plot(t_true, q2_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(t_eval, traj_fixed[:, 2], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(t_eval, traj_real[:, 2], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(t_eval, traj_complex[:, 2], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(t_eval, traj_prod[:, 2], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('t'); ax.set_ylabel('q2')
    ax.set_title('HO Position q2(t)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # p2 时间序列
    ax = axes[1, 2]
    ax.plot(t_true, p2_t, color=C['True'], ls=S['True'], lw=W['True'], label='True')
    ax.plot(t_eval, traj_fixed[:, 3], color=C['Fixed'], ls=S['Fixed'], lw=W['Fixed'], label='Fixed ω')
    ax.plot(t_eval, traj_real[:, 3], color=C['Real'], ls=S['Real'], lw=W['Real'], label='Real ω')
    ax.plot(t_eval, traj_complex[:, 3], color=C['Complex-MLP'], ls=S['Complex-MLP'], lw=W['Complex-MLP'], label='Complex (MLP)')
    ax.plot(t_eval, traj_prod[:, 3], color=C['Complex-Prod'], ls=S['Complex-Prod'], lw=W['Complex-Prod'], label='Complex (Prod)')
    ax.set_xlabel('t'); ax.set_ylabel('p2')
    ax.set_title('HO Momentum p2(t)'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    plt.suptitle(f'Trajectory Prediction: MLP vs Product Decomposition Coupling\nTrue: ω={TRUE_OMEGA}, ε={TRUE_EPSILON}, γ={TRUE_GAMMA} | '
                 f'Complex ω = {omega_R_mlp:.2f}+i·{omega_I_mlp:.3f} | Prod: ω={omega_R_prod:.2f}+i·{omega_I_prod:.3f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('complex_omega_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> complex_omega_trajectories.png")
    
    # ========================
    # 训练曲线 & 指标对比
    # ========================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # 训练曲线
    ax = axes2[0]
    ax.semilogy(tl_fixed, 'b-', alpha=0.4, lw=1, label='Fixed Train')
    ax.semilogy(vl_fixed, 'b--', alpha=0.6, lw=1, label='Fixed Val')
    ax.semilogy(tl_real, 'g-', alpha=0.4, lw=1, label='Real Train')
    ax.semilogy(vl_real, 'g--', alpha=0.6, lw=1, label='Real Val')
    ax.semilogy(tl_complex, 'r-', alpha=0.4, lw=1, label='Complex MLP Train')
    ax.semilogy(vl_complex, 'r--', alpha=0.6, lw=1, label='Complex MLP Val')
    ax.semilogy(tl_prod, 'm-', alpha=0.4, lw=1, label='Complex Prod Train')
    ax.semilogy(vl_prod, 'm--', alpha=0.6, lw=1, label='Complex Prod Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Training Curves'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    
    # 测试 MSE 柱状图
    ax = axes2[1]
    bar_labels = ['Fixed ω', 'Real ω', 'Complex\n(MLP)', 'Complex\n(Prod)']
    bar_values = [mse_fixed, mse_real, mse_complex, mse_prod]
    bar_colors = ['steelblue', 'mediumseagreen', 'coral', 'darkviolet']
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8)
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{val:.4e}', ha='center', fontsize=8)
    ax.set_ylabel('Test MSE'); ax.set_title('Test MSE Comparison')
    ax.grid(alpha=0.3, axis='y')
    
    # ω 参数对比
    ax = axes2[2]
    ax.axhline(y=TRUE_OMEGA, color='gray', ls='--', lw=1, label=f'True ω={TRUE_OMEGA}')
    omega_labels = ['Fixed', 'Real ω', 'Complex\n(MLP)', 'Complex\n(Prod)']
    omega_values = [TRUE_OMEGA, omega_real,
                    np.sqrt(omega_R_mlp**2 + omega_I_mlp**2),
                    np.sqrt(omega_R_prod**2 + omega_I_prod**2)]
    ax.barh(omega_labels, omega_values, color=bar_colors, alpha=0.8, height=0.5)
    ax.set_xlabel('|ω|'); ax.set_title('Learned |ω|'); ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('complex_omega_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  -> complex_omega_comparison.png")
    
    return model_complex, omega_R_mlp, omega_I_mlp


if __name__ == '__main__':
    main()