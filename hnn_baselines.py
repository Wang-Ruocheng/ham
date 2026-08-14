"""
HNN Baseline 模型集合
=====================
1. SeparableHNN  — 分解 H = T(p) + V(θ)，两个独立网络
2. PartialHNN   — 已知 M(θ)，只学 V(θ)
3. SIREN_HNN    — sin 激活函数替代 Tanh
4. SymplecticHNN — Separable + 多步辛积分器训练

所有模型共享训练/评估基础设施。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
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
        device = next(self.parameters()).device
        tm, ts, pm, ps = compute_split_stats(loader, self.N, device)
        self.theta_mu = tm; self.theta_sigma = ts
        self.p_mu = pm; self.p_sigma = ps

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
        device = next(self.parameters()).device
        tm, ts, _, _ = compute_split_stats(loader, self.N, device)
        self.theta_mu = tm; self.theta_sigma = ts

    def _compute_T(self, theta, p):
        B = theta.shape[0]
        cos_diff = torch.cos(theta.unsqueeze(1) - theta.unsqueeze(2))
        M = self.ml2 * self.k_mat.unsqueeze(0) * cos_diff
        M_inv_p = torch.linalg.solve(M, p.unsqueeze(-1)).squeeze(-1)
        return 0.5 * (p * M_inv_p).sum(dim=-1)

    def time_derivative(self, x):
        theta, p = x[:, :self.N], x[:, self.N:]
        theta_n = (theta - self.theta_mu) / self.theta_sigma
        V = self.V_net(theta_n)
        T = self._compute_T(theta, p)
        H = T + V
        grad = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        theta_grad, p_grad = grad[:, :self.N], grad[:, self.N:]
        return torch.cat([p_grad, -theta_grad], dim=-1)
        V_new = self.V_net(theta_new_n)
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
        device = next(self.parameters()).device
        self.mu, self.sigma = compute_full_stats(loader, self.dim, device)

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
# 训练函数
# ============================================================

def train_single_step(model, train_loader, val_loader, epochs=2000,
                      lr=1e-3, device='cuda', label='',
                      compute_stats_fn=None):
    """单步训练: 直接匹配 dx/dt"""
    model = model.to(device)
    if compute_stats_fn:
        compute_stats_fn(model, train_loader)
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
        for xb, dxb in val_loader:
            xb = xb.to(device); dxb = dxb.to(device)
            xb.requires_grad_(True)
            val_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


def train_symplectic(model, train_loader, val_loader,
                     n_steps=5, dt=0.1, epochs=2000,
                     lr=1e-3, device='cuda', label=''):
    """多步辛训练: 用辛积分器预测多步轨迹"""
    model = model.to(device)
    model.compute_stats(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0; n_samples = 0
        for xb, xf in train_loader:
            xb, xf = xb.to(device), xf.to(device)
            x_pred = xb.clone().requires_grad_(True)
            for _ in range(n_steps):
                x_pred = model.symplectic_step(x_pred, dt)
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
            val_loss += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 200 == 0:
            print(f"  {label} Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    return train_losses, val_losses


# ============================================================
# 生成多步训练数据
# ============================================================
def generate_multistep_data(sys, n_trajectories=200, t_span=(0, 20),
                            n_points=300, n_steps=5, seed=42):
    """生成多步训练数据: (x_t, x_{t+n*dt}) 对"""
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
    dt = (t_span[1] - t_span[0]) / n_steps
    D = len(state0)
    traj = np.zeros((n_steps, D)); traj[0] = state0
    for i in range(n_steps - 1):
        x = torch.tensor(traj[i:i+1], dtype=torch.float32, device=device)
        x.requires_grad_(True)
        k1 = model.time_derivative(x).detach().cpu().numpy()[0]
        k2 = model.time_derivative(
            x + 0.5*dt*torch.tensor(k1, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        k3 = model.time_derivative(
            x + 0.5*dt*torch.tensor(k2, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        k4 = model.time_derivative(
            x + dt*torch.tensor(k3, device=device, dtype=torch.float32)
        ).detach().cpu().numpy()[0]
        traj[i+1] = traj[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return traj


def evaluate_and_visualize(model, test_loader, sys, args, device, label,
                           t_span=(0, 40), n_points=1500):
    """评估模型: 测试 MSE + 轨迹预测 + 可视化"""
    model.eval()
    test_mse = 0.0; n_test = 0
    for xb, dxb in test_loader:
        xb = xb.to(device); dxb = dxb.to(device)
        xb.requires_grad_(True)
        test_mse += nn.MSELoss()(model.time_derivative(xb), dxb).item() * xb.size(0)
        n_test += xb.size(0)
    test_mse /= n_test
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] 测试 MSE: {test_mse:.6e} | 参数: {n_params:,}")

    # 轨迹预测
    print(f"  [{label}] 轨迹预测 (θ₀ = π/4)...")
    theta0 = np.full(args.n_masses, np.pi / 4)  # 45°
    p0 = np.zeros(args.n_masses)
    state0 = np.concatenate([theta0, p0])

    _, true_traj = sys.generate_trajectory(state0, t_span, n_points)
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
    fname = f'pendulum_string_{label.replace(" ", "_").lower()}.png'
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
    gif_fname = f'pendulum_string_{label.replace(" ", "_").lower()}.gif'
    anim.save(gif_fname, writer='pillow', fps=20, dpi=100)
    print(f"  GIF 已保存: {gif_fname}")
    plt.close(fig_anim)

    return test_mse, n_params