"""
测试脚本：绘制摆链从水平状态下落的真解
所有 θ_i = π/2, 零初速，自由下落
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pendulum_string import DiscretePendulumString

# ── 参数 ────────────────────────────────────────────────────
N = 10          # 节数
L = 1.0         # 杆长
M = 1.0         # 质量
G = 9.81        # 重力
t_max = 6.0     # 模拟时间
n_frames = 200  # 帧数

# ── 创建系统 ────────────────────────────────────────────────
sys = DiscretePendulumString(n_masses=N, length=L, mass=M, g=G)

# 初始条件：水平状态 (θ_i = π/2)，零初速
theta0 = np.full(N, np.pi / 2)
p0 = np.zeros(N)
state0 = np.concatenate([theta0, p0])

print(f"初始状态: θ_i = {theta0[0]:.3f} rad (水平)")
print(f"哈密顿量: H₀ = {sys.hamiltonian(state0):.4f}")

# ── 积分 ────────────────────────────────────────────────────
t, traj = sys.generate_trajectory(state0, t_span=(0, t_max), n_points=n_frames,
                                   rtol=1e-7, atol=1e-9)

# 能量守恒检查
H_vals = np.array([sys.hamiltonian(traj[i]) for i in range(len(traj))])
print(f"能量守恒: H ∈ [{H_vals.min():.6f}, {H_vals.max():.6f}], "
      f"rel std = {np.std(H_vals)/np.abs(H_vals.mean()):.2e}")

# ── 绘图 ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 时空图：θ_i(t)
ax = axes[0, 0]
im = ax.imshow(traj[:, :N].T, aspect='auto', origin='lower',
               extent=[0, t_max, 0, N-1], cmap='RdBu_r')
ax.set_xlabel('t [s]'); ax.set_ylabel('质点 i')
ax.set_title(f'θ_i(t) — 水平下落 (N={N})')
plt.colorbar(im, ax=ax, label='θ [rad]')

# 2. 角速度时空图
ax = axes[0, 1]
theta_dot = np.zeros((n_frames, N))
for i in range(n_frames):
    dstate = sys.dynamics(0, traj[i])
    theta_dot[i] = dstate[:N]
im = ax.imshow(theta_dot.T, aspect='auto', origin='lower',
               extent=[0, t_max, 0, N-1], cmap='RdBu_r')
ax.set_xlabel('t [s]'); ax.set_ylabel('质点 i')
ax.set_title('ω_i(t) = dθ_i/dt')
plt.colorbar(im, ax=ax, label='ω [rad/s]')

# 3. 能量守恒
ax = axes[0, 2]
ax.plot(t, H_vals, 'b-', linewidth=0.5)
ax.set_xlabel('t [s]'); ax.set_ylabel('H')
ax.set_title('哈密顿量守恒')
ax.grid(True, alpha=0.3)

# 4. 摆链快照 (t = 0, t_max/4, t_max/2, 3t_max/4, t_max)
snapshot_times = [0, t_max/4, t_max/2, 3*t_max/4, t_max]
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))

ax = axes[1, 0]
for t_snap, c in zip(snapshot_times, colors):
    idx = np.argmin(np.abs(t - t_snap))
    theta = traj[idx, :N]
    x, y = sys.get_positions(theta)
    x_chain = np.concatenate([[0], x])
    y_chain = np.concatenate([[0], y])
    ax.plot(x_chain, y_chain, 'o-', color=c, linewidth=2, markersize=5,
            label=f't={t_snap:.1f}s')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('摆链快照')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 5. 端点轨迹 (x_N-1, y_N-1)
ax = axes[1, 1]
x_all = np.zeros((n_frames, N)); y_all = np.zeros((n_frames, N))
for i in range(n_frames):
    x_all[i], y_all[i] = sys.get_positions(traj[i, :N])
ax.plot(x_all[:, -1], y_all[:, -1], 'b-', linewidth=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('末端质点轨迹')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

# 6. 最底端质点 θ vs ω 相图
ax = axes[1, 2]
ax.plot(traj[:, 0], theta_dot[:, 0], 'b-', linewidth=0.5)
ax.set_xlabel('θ₀ [rad]'); ax.set_ylabel('ω₀ [rad/s]')
ax.set_title('顶端质点相图')
ax.grid(True, alpha=0.3)

fig.suptitle(f'摆链水平下落 (N={N}, L={L}, g={G})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pendulum_horizontal_fall.png', dpi=150)
print("\n图像已保存: pendulum_horizontal_fall.png")

# ── 动画 ────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_xlim(-N*L - 0.5, N*L + 0.5)
ax2.set_ylim(-N*L - 0.5, 0.5)
ax2.set_aspect('equal')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('摆链下落动画')

line, = ax2.plot([], [], 'o-', color='darkblue', linewidth=2, markersize=8)
time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                     verticalalignment='top', fontsize=12)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(frame):
    theta = traj[frame, :N]
    x, y = sys.get_positions(theta)
    x_chain = np.concatenate([[0], x])
    y_chain = np.concatenate([[0], y])
    line.set_data(x_chain, y_chain)
    time_text.set_text(f't = {t[frame]:.2f}s')
    return line, time_text

ani = FuncAnimation(fig2, animate, init_func=init, frames=n_frames,
                    interval=30, blit=True)
ani.save('pendulum_horizontal_fall.gif', writer='pillow', fps=20, dpi=100)
print("动画已保存: pendulum_horizontal_fall.gif")
plt.close(fig2)