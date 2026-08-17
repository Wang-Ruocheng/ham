"""
HNN Baseline 模型集合
=====================
1. SeparableHNN  — 分解 H = T(p) + V(θ)，两个独立网络
2. PartialHNN   — 已知 M(θ)，只学 V(θ)
3. SIREN_HNN    — sin 激活函数替代 Tanh
4. SymplecticHNN — Separable + 多步辛积分器训练
5. SympNet      — 直接学辛映射 Φ(x) = x_{t+dt}
6. FNO          — Fourier Neural Operator 直接学向量场
7. GraphHNN     — 图结构参数共享 HNN (节点+边 MLP)
8. CHNN         — 笛卡尔坐标约束 HNN (Lagrange 乘子)

所有模型共享训练/评估基础设施。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 工具函数
# ============================================================

def compute_full_stats(loader, dim, device):
    """计算全状态 (θ, p) 的 mean/std"""
    n = 0
    mean = torch.zeros(dim, device=device)
    for xb, _ in loader:
        mean += xb.to(device).sum(dim=0)
        n += xb.size(0)
    mean /= n
    var = torch.zeros(dim, device=device)
    for xb, _ in loader:
        var += ((xb.to(device) - mean) ** 2).sum(dim=0)
    var /= n
    return mean, var.sqrt().clamp(min=1e-6)


def compute_split_stats(loader, N, device):
    """分别计算 θ 和 p 的 mean/std"""
    n = 0
    theta_mean = torch.zeros(N, device=device)
    p_mean = torch.zeros(N, device=device)
    for xb, _ in loader:
        xb = xb.to(device)
        theta_mean += xb[:, :N].sum(dim=0)
        p_mean += xb[:, N:].sum(dim=0)
        n += xb.size(0)
    theta_mean /= n; p_mean /= n

    theta_var = torch.zeros(N, device=device)
    p_var = torch.zeros(N, device=device)
    for xb, _ in loader:
        xb = xb.to(device)
        theta_var += ((xb[:, :N] - theta_mean) ** 2).sum(dim=0)
        p_var += ((xb[:, N:] - p_mean) ** 2).sum(dim=0)
    theta_var /= n; p_var /= n
    return (theta_mean, theta_var.sqrt().clamp(min=1e-6),
            p_mean, p_var.sqrt().clamp(min=1e-6))


def make_mlp(dim_in, dim_out, hidden_dim, num_layers, activation, final_bias=False):
    """构建 MLP"""
    layers = []
    prev = dim_in
    for _ in range(num_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(activation)
        prev = hidden_dim
    layers.append(nn.Linear(prev, dim_out, bias=final_bias))
    return nn.Sequential(*layers)


def _maybe_unwrap(model):
    """获取 DDP 包装的底层模型"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _is_rank0():
    """是否为主进程 (rank 0)"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def init_weights(module, gain=1.0):
    """Xavier 初始化"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# ============================================================
# 1. SeparableHNN — H = T(p) + V(θ)
# ============================================================
class SeparableHNN(nn.Module):
    """可分离 HNN: 两个独立网络分别学习动能和势能"""
    def __init__(self, N, hidden_dim=512, num_layers=4):
        super().__init__()
        self.N = N
        self.T_net = make_mlp(N, 1, hidden_dim, num_layers, nn.Tanh())
        self.V_net = make_mlp(N, 1, hidden_dim, num_layers, nn.Tanh())
        init_weights(self.T_net); init_weights(self.V_net)
        self.register_buffer('theta_mu', torch.zeros(N))
        self.register_buffer('theta_sigma', torch.ones(N))
        self.register_buffer('p_mu', torch.zeros(N))
        self.register_buffer('p_sigma', torch.ones(N))

    def compute_stats(self, loader):
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        tm, ts, pm, ps = compute_split_stats(loader, target.N, device)
        target.theta_mu = tm; target.theta_sigma = ts
        target.p_mu = pm; target.p_sigma = ps

    def time_derivative(self, x):
        theta, p = x[:, :self.N], x[:, self.N:]
        theta_n = (theta - self.theta_mu) / self.theta_sigma
        p_n = (p - self.p_mu) / self.p_sigma
        T = self.T_net(p_n)
        V = self.V_net(theta_n)
        dT_dp = torch.autograd.grad(T.sum(), p_n, create_graph=True)[0] / self.p_sigma
        dV_dtheta = torch.autograd.grad(V.sum(), theta_n, create_graph=True)[0] / self.theta_sigma
        return torch.cat([dT_dp, -dV_dtheta], dim=-1)

    def symplectic_step(self, x, dt):
        """Leapfrog 辛积分器一步"""
        theta, p = x[:, :self.N], x[:, self.N:]
        theta_n = (theta - self.theta_mu) / self.theta_sigma
        V = self.V_net(theta_n)
        dV = torch.autograd.grad(V.sum(), theta_n, create_graph=True)[0] / self.theta_sigma
        p_half = p - 0.5 * dt * dV
        p_n = (p_half - self.p_mu) / self.p_sigma
        T = self.T_net(p_n)
        dT = torch.autograd.grad(T.sum(), p_n, create_graph=True)[0] / self.p_sigma
        theta_new = theta + dt * dT
        theta_new_n = (theta_new - self.theta_mu) / self.theta_sigma
        V_new = self.V_net(theta_new_n)
        dV_new = torch.autograd.grad(V_new.sum(), theta_new_n, create_graph=True)[0] / self.theta_sigma
        p_new = p_half - 0.5 * dt * dV_new
        return torch.cat([theta_new, p_new], dim=-1)


# ============================================================
# 2. PartialHNN — 已知 M(θ)，只学 V(θ)
# ============================================================
class PartialHNN(nn.Module):
    """部分已知 HNN: T = ½pᵀM⁻¹(θ)p 解析计算，V(θ) 网络学习"""
    def __init__(self, N, ml2, hidden_dim=512, num_layers=4):
        super().__init__()
        self.N = N; self.ml2 = ml2
        self.V_net = make_mlp(N, 1, hidden_dim, num_layers, nn.Tanh())
        init_weights(self.V_net)
        self.register_buffer('theta_mu', torch.zeros(N))
        self.register_buffer('theta_sigma', torch.ones(N))
        i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
        self.register_buffer('k_mat', (N - torch.maximum(i, j)).float())

    def compute_stats(self, loader):
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        tm, ts, _, _ = compute_split_stats(loader, target.N, device)
        target.theta_mu = tm; target.theta_sigma = ts

    def time_derivative(self, x):
        theta, p = x[:, :self.N], x[:, self.N:]
        theta_n = (theta - self.theta_mu) / self.theta_sigma

        # V 和 ∂V/∂θ 通过 autograd
        V = self.V_net(theta_n)
        dV_dtheta = torch.autograd.grad(V.sum(), theta, create_graph=True)[0]

        # T 梯度解析计算 (避免 linalg.solve 的 autograd 不稳定)
        B = theta.shape[0]
        cos_diff = torch.cos(theta.unsqueeze(1) - theta.unsqueeze(2))
        M = self.ml2 * self.k_mat.unsqueeze(0) * cos_diff
        # 正则化 + sanitize 防止 NaN/Inf 导致崩溃
        eps = 1e-6
        M = M + eps * torch.eye(self.N, device=M.device).unsqueeze(0)
        M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
        M_inv = torch.linalg.pinv(M.detach())
        v = torch.bmm(M_inv, p.unsqueeze(-1)).squeeze(-1)  # v = M⁻¹p

        # ∂T/∂θ_k = ml² * v_k * Σ_j k_mat[k,j] * sin(θ_k - θ_j) * v_j
        sin_diff = torch.sin(theta.unsqueeze(1) - theta.unsqueeze(2))
        A = self.k_mat.unsqueeze(0) * sin_diff  # (B, N, N)
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)  # (B, N)
        dT_dtheta = self.ml2 * v * Av

        return torch.cat([v, -dT_dtheta - dV_dtheta], dim=-1)
# ============================================================
# 3. SIREN_HNN — sin 激活函数
# ============================================================
class Sine(nn.Module):
    def __init__(self, omega0=1.0):
        super().__init__()
        self.omega0 = omega0
    def forward(self, x):
        return torch.sin(self.omega0 * x)


class SIREN_HNN(nn.Module):
    """SIREN HNN: sin 激活替代 Tanh，适合周期性动力学"""
    def __init__(self, dim, hidden_dim=512, num_layers=4, omega0=30.0):
        super().__init__()
        self.dim = dim; self.N = dim // 2
        layers = []
        first = nn.Linear(dim, hidden_dim)
        nn.init.uniform_(first.weight, -1/dim, 1/dim)
        layers.append(first); layers.append(Sine(omega0))
        for _ in range(num_layers - 1):
            linear = nn.Linear(hidden_dim, hidden_dim)
            nn.init.uniform_(linear.weight,
                             -np.sqrt(6/hidden_dim)/omega0,
                             np.sqrt(6/hidden_dim)/omega0)
            layers.append(linear); layers.append(Sine(omega0))
        last = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.uniform_(last.weight,
                         -np.sqrt(6/hidden_dim)/omega0,
                         np.sqrt(6/hidden_dim)/omega0)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        self.register_buffer('mu', torch.zeros(dim))
        self.register_buffer('sigma', torch.ones(dim))

    def compute_stats(self, loader):
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        target.mu, target.sigma = compute_full_stats(loader, target.dim, device)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def time_derivative(self, x):
        x_norm = (x - self.mu) / self.sigma
        H = self.forward(x_norm)
        grad_norm = torch.autograd.grad(H.sum(), x_norm, create_graph=True)[0]
        grad = grad_norm / self.sigma
        theta_grad, p_grad = grad.chunk(2, dim=-1)
        return torch.cat([p_grad, -theta_grad], dim=-1)


# ============================================================
# 4. SymplecticHNN — Separable + 多步辛训练
# ============================================================
class SymplecticHNN(SeparableHNN):
    """与 SeparableHNN 相同模型，但训练用多步辛积分器"""
    pass


# ============================================================
# 5. SympNet — 直接学辛映射，绕过梯度瓶颈
# ============================================================
class SympNet(nn.Module):
    """SympNet: 直接学辛映射 Φ(x) = x_{t+Δt}
    
    论文: Jin, Zhang, Zhu (2020), "SympNets: Intrinsic structure-preserving
    symplectic networks for identifying Hamiltonian systems"
    
    核心: K 层交替的 G (梯度) 和 S (剪切) 模块，保证保辛。
    优势: Loss 只涉及一阶导数，无 HNN 的 Hessian 梯度瓶颈。
    
    架构:
        q_k = q_{k-1} + W_k · p_{k-1}          (S 模块)
        p_k = p_{k-1} - a_k · ∇V_k(q_k)        (G 模块)
    """
    def __init__(self, N, K=5, hidden_dim=128, num_layers=2):
        super().__init__()
        self.N = N
        self.K = K
        
        # S 模块: 可学习的 N×N 矩阵 W_k
        self.W = nn.ParameterList([
            nn.Parameter(torch.eye(N) * 0.01) for _ in range(K)
        ])
        
        # G 模块: 可学习的势能网络 V_k(q)
        self.V_nets = nn.ModuleList([
            make_mlp(N, 1, hidden_dim, num_layers, nn.Tanh(), final_bias=False)
            for _ in range(K)
        ])
        for vn in self.V_nets:
            init_weights(vn)
        
        # 每层的步长 a_k
        self.a = nn.Parameter(torch.ones(K) * 0.1)
        
        # 训练步长 dt
        self.register_buffer('dt', torch.tensor(0.05))
    
    def forward(self, x):
        """一次辛映射: (q, p) → (q', p')
        
        训练时 (self.training=True): create_graph=True，维持完整计算图
        推理时 (self.training=False): create_graph=False，不保留梯度图
        """
        q, p = x[:, :self.N], x[:, self.N:]
        for k in range(self.K):
            # S 模块: q ← q + W_k · p
            q = q + p @ self.W[k]
            # G 模块: p ← p - a_k · ∇V_k(q)
            q.requires_grad_(True)
            V = self.V_nets[k](q)
            dV = torch.autograd.grad(V.sum(), q, create_graph=self.training)[0]
            p = p - self.a[k] * dV
        return torch.cat([q, p], dim=-1)
    
    def time_derivative(self, x):
        """近似时间导数: dx/dt ≈ (Φ(x) - x) / dt"""
        x_next = self.forward(x)
        return (x_next - x) / self.dt
    
    def predict_trajectory(self, state0, t_span, n_points, device='cpu'):
        """预测轨迹: 用模型自身 dt 积分，然后插值到 n_points 个时间点
        
        注意: 不需要 RK4 积分器，SympNet 自身就是积分器。
        """
        self.eval()
        dt_model = self.dt.item()
        total_time = t_span[1] - t_span[0]
        n_steps = int(total_time / dt_model)
        D = len(state0)
        traj = np.zeros((n_steps + 1, D))
        traj[0] = state0
        
        # 自动检测模型所在设备
        model_device = next(self.parameters()).device
        
        for i in range(n_steps):
            x = torch.tensor(traj[i:i+1], dtype=torch.float32, device=model_device)
            x_next = self.forward(x).detach().cpu().numpy()[0]
            traj[i+1] = x_next
        
        # 插值到目标时间点（与 true trajectory 对齐）
        from scipy.interpolate import interp1d
        t_model = np.linspace(t_span[0], t_span[1], n_steps + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        traj_interp = np.zeros((n_points, D))
        for d in range(D):
            traj_interp[:, d] = interp1d(t_model, traj[:, d], kind='cubic')(t_eval)
        return traj_interp


# ============================================================
# 6. FNO — Fourier Neural Operator (1D)
# ============================================================
class SpectralConv1d(nn.Module):
    """1D Fourier 谱卷积层"""
    def __init__(self, in_dim, out_dim, modes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        scale = 1.0 / (in_dim * out_dim)
        self.weights = nn.Parameter(
            scale * torch.randn(in_dim, out_dim, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (batch, N, in_dim)
        B, N, _ = x.shape
        # FFT along spatial dim
        x_ft = torch.fft.rfft(x, dim=1)  # (B, N//2+1, in_dim)
        out_ft = torch.zeros(B, N // 2 + 1, self.out_dim,
                             dtype=torch.cfloat, device=x.device)
        # 有效 modes 数不能超过 rfft 输出大小
        M = min(self.modes, N // 2 + 1)
        out_ft[:, :M, :] = torch.einsum(
            'bmi,iom->bmo', x_ft[:, :M, :], self.weights[:, :, :M])
        # 逆 FFT
        x = torch.fft.irfft(out_ft, n=N, dim=1)  # (B, N, out_dim)
        return x


class FNO(nn.Module):
    """Fourier Neural Operator: 直接学习向量场 f(x) = dx/dt

    架构:
        Input:  (batch, 2N) — flat state vector
        Reshape: (batch, N, 2) — 2 channels (θ, p) on spatial grid
        Lift:    (batch, N, hidden_dim)
        Fourier layers × num_layers
        Project: (batch, N, 2)
        Flatten: (batch, 2N)

    关键特性:
        - 分辨率不变: 参数不随 N 增长 (仅 modes 决定)
        - 空间局部性: Fourier 层天然捕捉空间结构
        - 无 autograd 瓶颈: 直接预测向量场，不需要 Hessian
    """
    def __init__(self, N, modes=12, hidden_dim=64, num_layers=4):
        super().__init__()
        self.N = N
        self.modes = modes
        self.hidden_dim = hidden_dim

        # 升维: 2 → hidden_dim
        self.fc0 = nn.Linear(2, hidden_dim)

        # Fourier 层 + skip connection
        self.convs = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, modes)
            for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 1)  # 1x1 conv for skip
            for _ in range(num_layers)
        ])

        # 降维: hidden_dim → 2
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 2)

        # 输入归一化统计量 (2 通道 × N 空间点)
        self.register_buffer('mu', torch.zeros(2, N))
        self.register_buffer('sigma', torch.ones(2, N))
        # 输出归一化统计量 (dx/dt 的逐点 std)
        self.register_buffer('dx_mu', torch.zeros(2, N))
        self.register_buffer('dx_sigma', torch.ones(2, N))

        init_weights(self)

    def compute_stats(self, loader):
        """计算输入和输出的通道-空间归一化统计量"""
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        N = target.N
        n = 0
        # 输入统计: (N, 2) 每行是一个空间点的 (θ, p)
        mean = torch.zeros(N, 2, device=device)
        # 输出统计: (N, 2) 每行是 (dθ/dt, dp/dt)
        dx_mean = torch.zeros(N, 2, device=device)
        for xb, dxb in loader:
            xb, dxb = xb.to(device), dxb.to(device)
            batch = xb.shape[0]
            xb = xb.view(batch, N, 2)  # (batch, N, 2)
            dxb = dxb.view(batch, N, 2)
            mean += xb.sum(dim=0)  # (N, 2)
            dx_mean += dxb.sum(dim=0)
            n += batch
        mean = mean / n
        dx_mean = dx_mean / n

        var = torch.zeros(N, 2, device=device)
        dx_var = torch.zeros(N, 2, device=device)
        for xb, dxb in loader:
            xb, dxb = xb.to(device), dxb.to(device)
            batch = xb.shape[0]
            xb = xb.view(batch, N, 2)
            dxb = dxb.view(batch, N, 2)
            var += ((xb - mean.unsqueeze(0)) ** 2).sum(dim=0)
            dx_var += ((dxb - dx_mean.unsqueeze(0)) ** 2).sum(dim=0)
        var = var / n
        dx_var = dx_var / n

        target.mu = mean.T         # (2, N)
        target.sigma = var.sqrt().clamp(min=1e-6).T  # (2, N)
        target.dx_mu = dx_mean.T
        target.dx_sigma = dx_var.sqrt().clamp(min=1e-6).T

    def forward(self, x):
        """x: (batch, 2N) → 输出: (batch, 2N)"""
        B = x.shape[0]
        N = self.N

        # Reshape: (batch, 2N) → (batch, N, 2)
        x = x.view(B, N, 2)

        # 归一化
        x = (x - self.mu.T.unsqueeze(0)) / self.sigma.T.unsqueeze(0)

        # 升维
        x = self.fc0(x)  # (B, N, hidden_dim)

        # Fourier 层
        for conv, w in zip(self.convs, self.ws):
            # Fourier 路径
            x_ft = conv(x)  # (B, N, hidden_dim)
            # Skip connection (1x1 卷积)
            x_skip = w(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x_ft + x_skip
            x = torch.nn.functional.gelu(x)

        # 降维
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)  # (B, N, 2) — 归一化空间的预测

        # 输出反归一化: 从归一化空间映射回原始空间
        x = x * self.dx_sigma.T.unsqueeze(0) + self.dx_mu.T.unsqueeze(0)

        # Flatten: (B, N, 2) → (B, 2N)
        return x.reshape(B, -1)

    def time_derivative(self, x):
        """FNO 直接预测向量场，不需要 autograd"""
        return self.forward(x)

class FNOFlow(nn.Module):
    """FNO-Flow (双分支残差形式): 学习 Δx = x_{t+dt} - x_t, 预测 = x_t + Δx

    θ 和 p 各用独立的 FNO 分支，共享输入归一化。
    输出归一化使用 Δx 的统计量。
    """
    def __init__(self, N, dt=0.05, modes=12, hidden_dim=64, num_layers=4, dropout=0.0):
        super().__init__()
        self.N = N
        self.dt = dt
        self.modes = modes
        self.hidden_dim = hidden_dim

        # θ 分支: 独立的 Fourier 层
        self.theta_fc0 = nn.Linear(2, hidden_dim)
        self.theta_convs = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, modes)
            for _ in range(num_layers)
        ])
        self.theta_ws = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])
        self.theta_fc1 = nn.Linear(hidden_dim, 128)
        self.theta_fc2 = nn.Linear(128, 1)

        # p 分支: 独立的 Fourier 层
        self.p_fc0 = nn.Linear(2, hidden_dim)
        self.p_convs = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, modes)
            for _ in range(num_layers)
        ])
        self.p_ws = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])
        self.p_fc1 = nn.Linear(hidden_dim, 128)
        self.p_fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 输入归一化
        self.register_buffer('mu', torch.zeros(2, N))
        self.register_buffer('sigma', torch.ones(2, N))
        # 输出归一化 (Δx = x_{t+dt} - x_t 的统计量)
        self.register_buffer('delta_mu', torch.zeros(2, N))
        self.register_buffer('delta_sigma', torch.ones(2, N))

        init_weights(self)

    def _fno_branch(self, x, fc0, convs, ws, fc1, fc2):
        """FNO 分支: (B, N, 2) → (B, N, out_dim)"""
        h = fc0(x)
        for conv, w in zip(convs, ws):
            h_ft = conv(h)
            h_skip = w(h.permute(0, 2, 1)).permute(0, 2, 1)
            h = h_ft + h_skip
            h = torch.nn.functional.gelu(h)
            h = self.dropout(h)
        h = torch.nn.functional.gelu(fc1(h))
        h = self.dropout(h)
        h = fc2(h)
        return h

    def compute_stats(self, loader):
        """计算输入和残差 Δx 的通道-空间归一化统计量"""
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        N = target.N
        n = 0
        mean = torch.zeros(N, 2, device=device)
        delta_mean = torch.zeros(N, 2, device=device)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            batch = xb.shape[0]
            xb = xb.view(batch, N, 2)
            yb = yb.view(batch, N, 2)
            mean += xb.sum(dim=0)
            delta_mean += (yb - xb).sum(dim=0)
            n += batch
        mean = mean / n
        delta_mean = delta_mean / n

        var = torch.zeros(N, 2, device=device)
        delta_var = torch.zeros(N, 2, device=device)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            batch = xb.shape[0]
            xb = xb.view(batch, N, 2)
            yb = yb.view(batch, N, 2)
            delta = yb - xb
            var += ((xb - mean.unsqueeze(0)) ** 2).sum(dim=0)
            delta_var += ((delta - delta_mean.unsqueeze(0)) ** 2).sum(dim=0)
        var = var / n
        delta_var = delta_var / n

        target.mu = mean.T
        target.sigma = var.sqrt().clamp(min=1e-6).T
        target.delta_mu = delta_mean.T
        target.delta_sigma = delta_var.sqrt().clamp(min=1e-6).T

    def forward(self, x):
        """x: (batch, 2N) → Δx: (batch, 2N)  残差预测"""
        B = x.shape[0]
        N = self.N

        x_view = x.view(B, N, 2)
        x_norm = (x_view - self.mu.T.unsqueeze(0)) / self.sigma.T.unsqueeze(0)

        # 双分支: θ 和 p 各用独立的 FNO
        h_theta = self._fno_branch(x_norm, self.theta_fc0, self.theta_convs,
                                   self.theta_ws, self.theta_fc1, self.theta_fc2)
        h_p = self._fno_branch(x_norm, self.p_fc0, self.p_convs,
                               self.p_ws, self.p_fc1, self.p_fc2)
        h = torch.cat([h_theta, h_p], dim=-1)  # (B, N, 2)

        # 输出反归一化 → Δx
        delta = h * self.delta_sigma.T.unsqueeze(0) + self.delta_mu.T.unsqueeze(0)
        return delta.reshape(B, -1)

    def predict_next(self, x):
        """单步预测: x_t → x_t + Δx = x_{t+dt}"""
        return x + self.forward(x)

    def predict_trajectory(self, state0, t_span, n_steps, device='cuda'):
        """迭代预测轨迹 (无需 RK4)"""
        self.eval()
        dt = (t_span[1] - t_span[0]) / n_steps
        D = len(state0)
        traj = np.zeros((n_steps, D))
        traj[0] = state0

        # 计算需要多少模型步（模型 dt 可能不等于请求 dt）
        steps_per_model = max(1, int(round(dt / self.dt)))
        model_dt = dt / steps_per_model

        for i in range(n_steps - 1):
            if np.isnan(traj[i]).any():
                print(f"  [FNOFlow] NaN at step {i}/{n_steps}")
                traj[i:] = traj[i - 1]
                break
            x = torch.tensor(traj[i:i + 1], dtype=torch.float32, device=device)
            for _ in range(steps_per_model):
                with torch.no_grad():
                    x = self.predict_next(x)
            traj[i + 1] = x.cpu().numpy()[0]
        return traj

# ============================================================
# 7. GraphHNN — Hamiltonian Graph Network
# ============================================================
class GraphHNN(nn.Module):
    """Hamiltonian Graph Network: 图结构参数共享 HNN

    H = Σ_i H_node(θ_i, p_i) + Σ_{edges} H_edge(θ_i - θ_{i+1})

    所有节点共享一组 MLP，所有边共享另一组 MLP。
    参数不随 N 增长（O(1) 每节点），利用链式图结构。
    """
    def __init__(self, N, hidden_dim=128, num_layers=3):
        super().__init__()
        self.N = N
        self.node_net = make_mlp(2, 1, hidden_dim, num_layers, nn.Tanh())
        self.edge_net = make_mlp(1, 1, hidden_dim // 2, num_layers, nn.Tanh())
        init_weights(self.node_net)
        init_weights(self.edge_net)
        self.register_buffer('theta_mu', torch.zeros(N))
        self.register_buffer('theta_sigma', torch.ones(N))
        self.register_buffer('p_mu', torch.zeros(N))
        self.register_buffer('p_sigma', torch.ones(N))

    def compute_stats(self, loader):
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        tm, ts, pm, ps = compute_split_stats(loader, target.N, device)
        target.theta_mu = tm
        target.theta_sigma = ts
        target.p_mu = pm
        target.p_sigma = ps

    def hamiltonian(self, theta_n, p_n):
        """计算标量哈密顿量 H = Σ node + Σ edge"""
        # 节点能量: (θ_i, p_i) → 共享 MLP
        theta_p = torch.stack([theta_n, p_n], dim=-1)  # (B, N, 2)
        H_node = self.node_net(theta_p).squeeze(-1)    # (B, N)
        H = H_node.sum(dim=-1)                          # (B,)

        # 边能量: (θ_i - θ_{i+1}) → 共享 MLP
        theta_diff = theta_n[:, :-1] - theta_n[:, 1:]   # (B, N-1)
        H_edge = self.edge_net(theta_diff.unsqueeze(-1)).squeeze(-1)  # (B, N-1)
        H = H + H_edge.sum(dim=-1)
        return H

    def time_derivative(self, x):
        theta, p = x[:, :self.N], x[:, self.N:]
        theta_n = (theta - self.theta_mu) / self.theta_sigma
        p_n = (p - self.p_mu) / self.p_sigma

        H = self.hamiltonian(theta_n, p_n)
        dH_dtheta_n = torch.autograd.grad(H.sum(), theta_n, create_graph=True)[0]
        dH_dp_n = torch.autograd.grad(H.sum(), p_n, create_graph=True)[0]

        dH_dp = dH_dp_n / self.p_sigma
        dH_dtheta = dH_dtheta_n / self.theta_sigma

        return torch.cat([dH_dp, -dH_dtheta], dim=-1)


# ============================================================
# 8. CHNN — Constrained HNN (笛卡尔坐标)
# ============================================================
class CHNN(nn.Module):
    """Constrained Hamiltonian Neural Network

    在笛卡尔坐标 (x, y, p_x, p_y) 中学习哈密顿量，
    显式强制执行刚性杆约束，通过 Lagrange 乘子求解。

    约束: N 个距离约束 g_k(q) = 0
      g_1 = x_1² + y_1² - l² = 0
      g_k = (x_k - x_{k-1})² + (y_k - y_{k-1})² - l² = 0  (k=2..N)

    动力学:
      q̇ = M^{-1} p
      ṗ = -∂H/∂q - C^T λ
      其中 C = ∂g/∂q, λ 由约束一致性条件解出
    """
    def __init__(self, N, l, m, hidden_dim=256, num_layers=3):
        super().__init__()
        self.N = N
        self.l = l    # 每节杆长
        self.m = m    # 每个质点质量
        self.dim = 4 * N  # 2N 位置 + 2N 动量
        self.H_net = make_mlp(4 * N, 1, hidden_dim, num_layers, nn.Tanh())
        init_weights(self.H_net)
        self.register_buffer('mu', torch.zeros(4 * N))
        self.register_buffer('sigma', torch.ones(4 * N))

    def compute_stats(self, loader):
        target = _maybe_unwrap(self)
        device = next(target.parameters()).device
        target.mu, target.sigma = compute_full_stats(loader, target.dim, device)

    def constraint_jacobian(self, q):
        """约束 Jacobian C = ∂g/∂q ∈ R^{B×N×2N}

        q: (batch, 2N) = [x_1, y_1, ..., x_N, y_N]
        """
        B = q.shape[0]
        N = self.N
        C = torch.zeros(B, N, 2 * N, device=q.device)

        # g_1 = x_1² + y_1² - l²
        C[:, 0, 0] = 2 * q[:, 0]   # ∂g_1/∂x_1
        C[:, 0, 1] = 2 * q[:, 1]   # ∂g_1/∂y_1

        # g_k = (x_k - x_{k-1})² + (y_k - y_{k-1})² - l²  (k=2..N)
        for k in range(1, N):
            dx = q[:, 2 * k] - q[:, 2 * k - 2]
            dy = q[:, 2 * k + 1] - q[:, 2 * k - 1]
            C[:, k, 2 * k - 2] = -2 * dx
            C[:, k, 2 * k - 1] = -2 * dy
            C[:, k, 2 * k] = 2 * dx
            C[:, k, 2 * k + 1] = 2 * dy

        return C  # (B, N, 2N)

    def constraint_acceleration(self, q_dot):
        """计算 Ċ q̇ ∈ R^N

        q_dot: (batch, 2N) = [ẋ_1, ẏ_1, ..., ẋ_N, ẏ_N]
        """
        B = q_dot.shape[0]
        N = self.N
        vx = q_dot[:, 0::2]   # (B, N)
        vy = q_dot[:, 1::2]   # (B, N)

        Cdot_qdot = torch.zeros(B, N, device=q_dot.device)
        Cdot_qdot[:, 0] = 2 * (vx[:, 0] ** 2 + vy[:, 0] ** 2)
        for k in range(1, N):
            dvx = vx[:, k] - vx[:, k - 1]
            dvy = vy[:, k] - vy[:, k - 1]
            Cdot_qdot[:, k] = 2 * (dvx ** 2 + dvy ** 2)

        return Cdot_qdot  # (B, N)

    def time_derivative(self, x):
        """x: (batch, 4N) = [q, p]"""
        B = x.shape[0]
        N = self.N
        q = x[:, :2 * N]
        p = x[:, 2 * N:]

        x_norm = (x - self.mu) / self.sigma
        H = self.H_net(x_norm)
        dH_dx_norm = torch.autograd.grad(H.sum(), x_norm, create_graph=True)[0]
        dH_dx = dH_dx_norm / self.sigma
        dH_dq = dH_dx[:, :2 * N]   # (B, 2N)

        # q̇ = M^{-1} p = p / m
        q_dot = p / self.m

        C = self.constraint_jacobian(q)   # (B, N, 2N)
        Cdot_qdot = self.constraint_acceleration(q_dot)  # (B, N)

        # C M^{-1} C^T = (1/m) C C^T
        CCT = torch.bmm(C, C.transpose(1, 2))  # (B, N, N)
        M_inv_block = CCT / self.m

        # 正则化防止奇异
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        M_inv_block = M_inv_block + 1e-6 * eye

        # RHS = -C M^{-1} ∂H/∂q + Ċ q̇
        rhs = (-torch.bmm(C, dH_dq.unsqueeze(-1)).squeeze(-1) / self.m
               + Cdot_qdot)  # (B, N)

        # 求解 Lagrange 乘子 λ
        lamb = torch.linalg.solve(M_inv_block, rhs.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # ṗ = -∂H/∂q - C^T λ
        CT_lambda = torch.bmm(C.transpose(1, 2), lamb.unsqueeze(-1)).squeeze(-1)  # (B, 2N)
        p_dot = -dH_dq - CT_lambda

        return torch.cat([q_dot, p_dot], dim=-1)


# ============================================================
# 训练函数
# ============================================================

def train_single_step(model, train_loader, val_loader, epochs=2000,
                      lr=1e-3, device='cuda', label='',
                      compute_stats_fn=None):
    """单步训练: 直接匹配 dx/dt"""
    model = model.to(device)
    raw = _maybe_unwrap(model)
    if compute_stats_fn:
        compute_stats_fn(model, train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for xb, dxb in train_loader:
            xb, dxb = xb.to(device), dxb.to(device)
            xb.requires_grad_(True)
            optimizer.zero_grad()
            loss = nn.MSELoss()(raw.time_derivative(xb), dxb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        for xb, dxb in val_loader:
            xb = xb.to(device); dxb = dxb.to(device)
            xb.requires_grad_(True)
            val_loss += nn.MSELoss()(raw.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if _is_rank0() and (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


def train_symplectic(model, train_loader, val_loader,
                     n_steps=5, dt=0.1, epochs=2000,
                     lr=1e-3, device='cuda', label=''):
    """多步辛训练: 用辛积分器预测多步轨迹"""
    model = model.to(device)
    raw = _maybe_unwrap(model)
    raw.compute_stats(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0; n_samples = 0
        for xb, xf in train_loader:
            xb, xf = xb.to(device), xf.to(device)
            x_pred = xb.clone().requires_grad_(True)
            for _ in range(n_steps):
                x_pred = raw.symplectic_step(x_pred, dt)
            optimizer.zero_grad()
            loss = nn.MSELoss()(x_pred, xf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        train_loss /= n_samples
        train_losses.append(train_loss)

        # 验证: 用单步 MSE
        model.eval()
        val_loss = 0.0
        for xb, dxb in val_loader:
            xb = xb.to(device); dxb = dxb.to(device)
            xb.requires_grad_(True)
            val_loss += nn.MSELoss()(raw.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if _is_rank0() and (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


def train_sympnet(model, train_loader, val_loader, epochs=2000,
                  lr=1e-3, device='cuda', label=''):
    """SympNet 训练: 直接监督状态迁移 (x_t, x_{t+dt})
    
    Loss 只涉及一阶导数，无 HNN 的 Hessian 瓶颈。
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for xb, xf in train_loader:
            xb, xf = xb.to(device), xf.to(device)
            xb.requires_grad_(True)  # 需要梯度通过 autograd.grad 的 create_graph
            optimizer.zero_grad()
            x_pred = model(xb)
            loss = nn.MSELoss()(x_pred, xf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证: 用状态迁移 MSE
        model.eval()
        val_loss = 0.0
        for xb, xf in val_loader:
            xb, xf = xb.to(device), xf.to(device)
            xb.requires_grad_(True)
            x_pred = model(xb)
            val_loss += nn.MSELoss()(x_pred, xf).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if _is_rank0() and (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


def train_fno_flow(model, train_loader, val_loader, epochs=2000,
                   lr=1e-3, device='cuda', label=''):
    """FNO-Flow 训练: 残差 Δx 在归一化空间计算 loss

    关键: loss 在归一化空间计算，平衡 θ 和 p 通道的梯度（避免 p 主导）。
    """
    model = model.to(device)
    raw = _maybe_unwrap(model)
    raw.compute_stats(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []

    def _loss_norm(delta_pred, delta_true):
        """在归一化空间计算 MSE: 平衡 θ/p 通道"""
        B = delta_pred.shape[0]
        N = raw.N
        dp = delta_pred.view(B, N, 2)
        dt = delta_true.view(B, N, 2)
        dp_n = (dp - raw.delta_mu.T.unsqueeze(0)) / raw.delta_sigma.T.unsqueeze(0)
        dt_n = (dt - raw.delta_mu.T.unsqueeze(0)) / raw.delta_sigma.T.unsqueeze(0)
        return nn.MSELoss()(dp_n, dt_n)

    for epoch in range(epochs):
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for xb, xf in train_loader:
            xb, xf = xb.to(device), xf.to(device)
            optimizer.zero_grad()
            delta_pred = raw.forward(xb)
            delta_true = xf - xb
            loss = _loss_norm(delta_pred, delta_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        for xb, xf in val_loader:
            xb, xf = xb.to(device), xf.to(device)
            with torch.no_grad():
                delta_pred = raw.forward(xb)
                delta_true = xf - xb
            val_loss += _loss_norm(delta_pred, delta_true).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if _is_rank0() and (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


# ============================================================
# 生成多步训练数据
# ============================================================
def generate_multistep_data(sys, n_trajectories=200, t_span=(0, 20),
                            dt=0.05, n_steps=5, seed=42):
    """生成多步训练数据: (x_t, x_{t+n*dt}) 对，dt 为单步积分步长"""
    n_points = int((t_span[1] - t_span[0]) / dt) + 1
    np.random.seed(seed)
    xs_list, xs_future_list = [], []

    for traj_idx in range(n_trajectories):
        if traj_idx % 20 == 0:
            print(f"  生成多步轨迹 {traj_idx}/{n_trajectories}...")
        theta0 = np.random.uniform(-np.pi, np.pi, sys.N)
        omega0 = np.random.uniform(-1.0, 1.0, sys.N)
        M0 = sys.inertia_matrix(theta0)
        p0 = M0 @ omega0
        state0 = np.concatenate([theta0, p0])
        _, traj = sys.generate_trajectory(state0, t_span, n_points,
                                          rtol=1e-6, atol=1e-8)
        for i in range(len(traj) - n_steps):
            xs_list.append(traj[i])
            xs_future_list.append(traj[i + n_steps])

    xs = np.stack(xs_list); xs_future = np.stack(xs_future_list)
    n_total = len(xs)
    indices = np.random.permutation(n_total)
    n_train = int(0.7 * n_total); n_val = int(0.15 * n_total)

    xs_t = torch.tensor(xs, dtype=torch.float32)
    xf_t = torch.tensor(xs_future, dtype=torch.float32)

    train_ds = TensorDataset(xs_t[indices[:n_train]], xf_t[indices[:n_train]])
    val_ds = TensorDataset(xs_t[indices[n_train:n_train+n_val]],
                           xf_t[indices[n_train:n_train+n_val]])
    print(f"  多步数据: {n_total} 对 | 训练 {n_train} | 验证 {n_val}")
    return train_ds, val_ds


# ============================================================
# 评估: RK4 积分 + 可视化
# ============================================================
def integrate_rk4(model, state0, t_span, n_steps, device='cuda'):
    """RK4 积分预测轨迹"""
    model.eval()
    raw = _maybe_unwrap(model)
    dt = (t_span[1] - t_span[0]) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D)); traj[0] = state0
    for i in range(n_steps - 1):
        if np.isnan(traj[i]).any():
            print(f"  [RK4] NaN 在 step {i}/{n_steps}，停止积分")
            traj[i:] = traj[i-1]  # 填充最后有效值
            break
        x = torch.tensor(traj[i:i+1], dtype=torch.float32, device=device)
        x.requires_grad_(True)
        k1 = raw.time_derivative(x).detach().cpu().numpy()[0]
        k2 = raw.time_derivative(
            x + 0.5*dt*torch.tensor(k1, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        k3 = raw.time_derivative(
            x + 0.5*dt*torch.tensor(k2, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        k4 = raw.time_derivative(
            x + dt*torch.tensor(k3, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj


def predict_flow_trajectory(model, state0, t_span, n_steps, device='cuda'):
    """迭代预测轨迹 (FNOFlow / SympNet 等 flow 模型)

    与 integrate_rk4 不同，flow 模型直接预测 x_{t+dt}，无需 RK4。
    """
    model.eval()
    raw = _maybe_unwrap(model)
    dt = (t_span[1] - t_span[0]) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D)); traj[0] = state0

    # 模型步长 vs 请求步长
    model_dt = getattr(raw, 'dt', dt)
    model_dt = model_dt.item() if isinstance(model_dt, torch.Tensor) else model_dt
    steps_per_model = max(1, int(round(dt / model_dt)))

    for i in range(n_steps - 1):
        if np.isnan(traj[i]).any():
            print(f"  [Flow] NaN at step {i}/{n_steps}")
            traj[i:] = traj[i - 1]
            break
        x = torch.tensor(traj[i:i + 1], dtype=torch.float32, device=device)
        for _ in range(steps_per_model):
            with torch.no_grad():
                x = raw.predict_next(x)
        traj[i + 1] = x.cpu().numpy()[0]
    return traj


def evaluate_and_visualize(model, test_loader, sys, args, device, label,
                           t_span=(0, 40), n_points=1500, output_dir='.'):
    """评估模型: 测试 MSE + 轨迹预测 + 可视化"""
    if not _is_rank0():
        return float('nan'), 0
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    raw = _maybe_unwrap(model)
    is_flow = isinstance(raw, FNOFlow)

    test_mse = 0.0; n_test = 0
    test_mse_theta = 0.0; test_mse_p = 0.0
    N_flow = raw.N if is_flow else 0
    for xb, yb in test_loader:
        xb = xb.to(device); yb = yb.to(device)
        if is_flow:
            # FNOFlow: 比较 x_{t+dt} 预测 vs 真实
            with torch.no_grad():
                pred = raw.predict_next(xb)
                test_mse += nn.MSELoss()(pred, yb).item() * xb.size(0)
                test_mse_theta += nn.MSELoss()(pred[:, :N_flow], yb[:, :N_flow]).item() * xb.size(0)
                test_mse_p += nn.MSELoss()(pred[:, N_flow:], yb[:, N_flow:]).item() * xb.size(0)
        else:
            xb.requires_grad_(True)
            test_mse += nn.MSELoss()(raw.time_derivative(xb), yb).item() * xb.size(0)
        n_test += xb.size(0)
    test_mse /= n_test
    test_mse_theta = test_mse_theta / n_test if is_flow else 0
    test_mse_p = test_mse_p / n_test if is_flow else 0
    n_params = sum(p.numel() for p in model.parameters())
    if is_flow:
        print(f"\n  [{label}] 测试 MSE: {test_mse:.6e} | θ: {test_mse_theta:.6e} | p: {test_mse_p:.6e} | 参数: {n_params:,}")
    else:
        print(f"\n  [{label}] 测试 MSE: {test_mse:.6e} | 参数: {n_params:,}")

    # 轨迹预测
    print(f"  [{label}] 轨迹预测 (θ₀ = π/4)...")
    theta0 = np.full(args.n_masses, np.pi / 4)  # 45°
    p0 = np.zeros(args.n_masses)
    state0 = np.concatenate([theta0, p0])

    _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
    if isinstance(raw, (SympNet, FNOFlow)):
        pred_traj = raw.predict_trajectory(state0, t_span, n_points, device)
    else:
        pred_traj = integrate_rk4(model, state0, t_span, n_points, device)

    t_eval = np.linspace(*t_span, n_points)
    N = args.n_masses
    H_true = np.array([sys.hamiltonian(true_traj[i]) for i in range(n_points)])
    H_pred = np.array([sys.hamiltonian(pred_traj[i]) for i in range(n_points)])
    print(f"  H_true(0) = {H_true[0]:.2f}, H_pred(0) = {H_pred[0]:.2f}")

    # 可视化: 上排哈密顿量，下排 3 帧摆链快照
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ── 哈密顿量 ──
    ax_ham = fig.add_subplot(gs[0, :])
    ax_ham.plot(t_eval, H_true, label='H_true', color='C0', lw=2)
    ax_ham.plot(t_eval, H_pred, '--', label='H_pred', color='C1', lw=2)
    ax_ham.set_title(f'Hamiltonian [{label}]'); ax_ham.set_xlabel('t')
    ax_ham.set_ylabel('H'); ax_ham.legend(); ax_ham.grid(alpha=0.3)

    # ── 摆链快照 (t=0, t=mid, t=end) ──
    snap_indices = [0, n_points // 2, n_points - 1]
    snap_times = [t_eval[i] for i in snap_indices]

    for idx, (si, st) in enumerate(zip(snap_indices, snap_times)):
        ax = fig.add_subplot(gs[1, idx])
        theta_t = true_traj[si, :N]; theta_p = pred_traj[si, :N]
        x_t, y_t = sys.get_positions(theta_t)
        x_p, y_p = sys.get_positions(theta_p)

        ax.plot(0, 0, 'ks', markersize=8)
        x_chain_t = np.concatenate([[0], x_t])
        y_chain_t = np.concatenate([[0], y_t])
        ax.plot(x_chain_t, y_chain_t, 'o-', color='C0', label='True', markersize=5, lw=2)
        x_chain_p = np.concatenate([[0], x_p])
        y_chain_p = np.concatenate([[0], y_p])
        ax.plot(x_chain_p, y_chain_p, 's--', color='C1', label='Pred', markersize=5, lw=2)

        ax.set_title(f't = {st:.1f}s'); ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.legend(loc='upper right', fontsize=8); ax.invert_yaxis(); ax.grid(alpha=0.3)
        ax.set_aspect('equal')

        all_x = np.concatenate([[0], x_t, x_p])
        all_y = np.concatenate([[0], y_t, y_p])
        margin = 0.5
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, 0.1)

    fig.suptitle(f'{label} | N={N} | {n_params:,} params | test MSE={test_mse:.4e}',
                 fontsize=14, fontweight='bold')
    fname = os.path.join(output_dir, f'pendulum_string_{label.replace(" ", "_").lower()}.png')
    plt.savefig(fname, dpi=150)
    print(f"  可视化已保存: {fname}")
    plt.close()

    # ── GIF 动画 ──
    print(f"  [{label}] 生成 GIF 动画...")
    gif_frames = np.arange(0, n_points, max(1, n_points // 200))  # ~200 帧
    n_gif = len(gif_frames)

    # 预计算所有 xy 坐标 + 全局范围
    x_all_t, y_all_t = [], []
    x_all_p, y_all_p = [], []
    for fi in gif_frames:
        xt, yt = sys.get_positions(true_traj[fi, :N])
        xp, yp = sys.get_positions(pred_traj[fi, :N])
        x_all_t.append(xt); y_all_t.append(yt)
        x_all_p.append(xp); y_all_p.append(yp)

    all_x = np.concatenate([np.concatenate(x_all_t), np.concatenate(x_all_p), [0]])
    all_y = np.concatenate([np.concatenate(y_all_t), np.concatenate(y_all_p), [0]])
    margin = 0.5
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, 0.1)

    fig_anim, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(12, 6))
    fig_anim.suptitle(f'{label} — Predicted Swing', fontsize=14, fontweight='bold')

    def animate(i):
        ax_t.clear(); ax_p.clear()
        fi = gif_frames[i]
        t = t_eval[fi]

        ax_t.plot(0, 0, 'ks', markersize=8)
        x_c = np.concatenate([[0], x_all_t[i]])
        y_c = np.concatenate([[0], y_all_t[i]])
        ax_t.plot(x_c, y_c, 'o-', color='C0', markersize=6, lw=2.5)
        ax_t.set_title(f'True  (t = {t:.1f}s)'); ax_t.set_xlabel('x'); ax_t.set_ylabel('y')
        ax_t.set_xlim(xlim); ax_t.set_ylim(ylim)
        ax_t.invert_yaxis(); ax_t.set_aspect('equal'); ax_t.grid(alpha=0.3)

        ax_p.plot(0, 0, 'ks', markersize=8)
        x_c = np.concatenate([[0], x_all_p[i]])
        y_c = np.concatenate([[0], y_all_p[i]])
        ax_p.plot(x_c, y_c, 's-', color='C1', markersize=6, lw=2.5)
        ax_p.set_title(f'Predicted  (t = {t:.1f}s)'); ax_p.set_xlabel('x'); ax_p.set_ylabel('y')
        ax_p.set_xlim(xlim); ax_p.set_ylim(ylim)
        ax_p.invert_yaxis(); ax_p.set_aspect('equal'); ax_p.grid(alpha=0.3)

        return ax_t, ax_p

    anim = FuncAnimation(fig_anim, animate, frames=n_gif, interval=50, blit=False)
    gif_fname = os.path.join(output_dir, f'pendulum_string_{label.replace(" ", "_").lower()}.gif')
    anim.save(gif_fname, writer='pillow', fps=20, dpi=100)
    print(f"  GIF 已保存: {gif_fname}")
    plt.close(fig_anim)

    return test_mse, n_params