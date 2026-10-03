"""QCR trajectories and numerical diagnostics independent of plotting."""
import numpy as np
import torch
from contextlib import nullcontext
from .data.views import balanced_views, ensemble_views
from .metrics import rank_correlation

DTYPE = torch.float64

def trajectory_data(model, configurations, samples: int, *, max_rg_steps=None, spin_flip_augment=True):
    """Collect direct and rollout trajectories."""
    model.eval()
    subset = ensemble_views(configurations, samples, spin_flip_augment=spin_flip_augment)
    with torch.no_grad():
        levels = model.coarse_grain(subset, max_rg_steps=max_rg_steps)
        direct = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
        rollout = [direct[:, 0]]
        for _ in range(len(levels) - 1):
            rollout.append(model.advance_theory(rollout[-1]))
        rollout = torch.stack(rollout, dim=1)
        split = torch.stack([
            model.theory_encoder(level[:, : level.shape[1] // 2])
            for level in levels
        ], dim=1)
    return levels, direct.numpy(), rollout.numpy(), split.numpy()


def collect_trajectories(model, configurations, *, samples, max_rg_steps=None,
                         spin_flip_augment=False, chunk_size=16, sampling_seed=12345):
    """CPU diagnostic arrays; evaluation depth is independent of model.levels."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    direct, rollout = [], []
    was_training = model.training
    stochastic = getattr(getattr(model.rg_kernel, 'field_head', None), 'sampling_mode', '') in ('bernoulli','categorical')
    rng_context = torch.random.fork_rng(devices=[]) if stochastic else nullcontext()
    try:
        with rng_context:
            if stochastic:
                torch.manual_seed(sampling_seed)
            for chunk in configurations.split(chunk_size):
                _, d, r, _ = trajectory_data(model, chunk, samples, max_rg_steps=max_rg_steps,
                                              spin_flip_augment=spin_flip_augment)
                direct.append(d); rollout.append(r)
        d, r = np.concatenate(direct), np.concatenate(rollout)
        with torch.no_grad():
            one = model.advance_theory(torch.from_numpy(d[:,:-1])).numpy()
        return {'direct':d, 'rollout':r, 'one_step':one}
    finally:
        model.train(was_training)


def closure_metrics(trajectories, *, reference_direct=None):
    """Per-coordinate RMS, pooled target SD; normalization is always labeled.

    No epsilon silently turns a collapsed target into a good score. Undefined
    normalized errors are reported as null (None); raw errors remain available.
    """
    d = np.asarray(trajectories['direct'])
    ref = d if reference_direct is None else np.asarray(reference_direct)
    if ref.shape[1:] != d.shape[1:]:
        raise ValueError('Reference must have the same scales and latent dimensions')
    spread = np.sqrt(ref.var(axis=0).mean(axis=-1))
    result = {'normalization':'evaluated target SD' if reference_direct is None else 'explicit reference target SD',
              'target_sd':spread.tolist()}
    for name,pred,target,sd in (
        ('rollout',trajectories['rollout'],d,spread),
        ('one_step',trajectories['one_step'],d[:,1:],spread[1:])):
        rmse = np.sqrt(np.mean((pred-target)**2,axis=(0,2)))
        result[name] = {'rmse':rmse.tolist(),
                        'nrmse':[float(a/b) if b>1e-12 else None for a,b in zip(rmse,sd)]}
    return result


def discrete_fixed_point(model, point, *, tolerance=1e-6):
    """Stability uses the Jacobian of R, NOT of its displacement R(g)-g."""
    x = torch.as_tensor(point, dtype=DTYPE).reshape(-1).requires_grad_(True)
    jac = torch.autograd.functional.jacobian(model.advance_theory, x).detach().numpy()
    eig = np.linalg.eigvals(jac); magnitude = np.abs(eig)
    kind = ('nonhyperbolic' if np.any(np.abs(magnitude-1)<=tolerance) else
            'stable' if np.all(magnitude<1) else
            'unstable' if np.all(magnitude>1) else 'saddle')
    return {'g':x.detach().tolist(), 'stability':kind,
            'residual_norm':float(torch.linalg.vector_norm(model.advance_theory(x)-x).detach()),
            'eigenvalues':[{'real':float(v.real),'imag':float(v.imag)} for v in eig]}


def scalar_flow(model, lower, upper, *, points=1501):
    from scipy.optimize import brentq
    if not lower < upper or points < 3:
        raise ValueError('Invalid search interval/grid')
    grid = np.linspace(lower,upper,points)
    with torch.no_grad():
        mapped = model.advance_theory(torch.tensor(grid[:,None],dtype=DTYPE)).numpy()[:,0]
    def residual(z):
        with torch.no_grad():
            return model.advance_theory(torch.tensor([z],dtype=DTYPE)).item()-z
    f = mapped-grid
    if np.all(np.abs(f)<1e-12):
        return {'grid':grid,'mapped':mapped,'roots':[], 'search_range':[lower,upper],
                'search_note':'Map is numerically identity on the grid; isolated roots are not determined.'}
    found = list(grid[f==0])
    for i in np.flatnonzero(f[:-1]*f[1:]<0):
        found.append(brentq(residual,grid[i],grid[i+1]))
    roots = [discrete_fixed_point(model,[g]) for g in sorted(set(found))]
    return {'grid':grid,'mapped':mapped,'roots':roots,'search_range':[lower,upper],
            'search_note':'Sign-change/grid-zero search; tangential roots may be missed.'}


def summarize_representation(direct, rollout, split, couplings):
    """Unified metrics for 1D/2D Ising."""
    levels = direct.shape[1]
    theory_dim = direct.shape[2]

    if theory_dim == 1:
        g = direct[:, :, 0]
        spearman = [rank_correlation(g[:, k], couplings) for k in range(levels)]
        std = [g[:, k].std() for k in range(levels)]
    else:
        g_norm = np.linalg.norm(direct, axis=2)
        spearman = [rank_correlation(g_norm[:, k], couplings) for k in range(levels)]
        std = [g_norm[:, k].std() for k in range(levels)]

    rmse = np.sqrt(np.mean((direct - rollout) ** 2, axis=(0, 2)))
    return {
        "absolute_spearman_g_j_by_level": [abs(float(s)) for s in spearman],
        "between_system_std_by_level": [float(s) for s in std],
        "rollout_direct_rmse_by_level": [float(r) for r in rmse],
    }


def find_fixed_points_1d(model, support_min, support_max):
    """Find fixed points of 1D scalar flow."""
    g = torch.linspace(support_min, support_max, 1000, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        Rg = model.advance_theory(g).squeeze(-1).numpy()
    g_np = g.squeeze(-1).numpy()
    residual = Rg - g_np
    sign_changes = np.where(np.diff(np.sign(residual)))[0]
    roots = []
    for idx in sign_changes:
        g0, g1 = g_np[idx], g_np[idx + 1]
        r0, r1 = residual[idx], residual[idx + 1]
        g_star = g0 - r0 * (g1 - g0) / (r1 - r0)
        eps = 1e-4
        g_plus = torch.tensor([[g_star + eps]], dtype=DTYPE)
        g_minus = torch.tensor([[g_star - eps]], dtype=DTYPE)
        with torch.no_grad():
            R_plus = model.advance_theory(g_plus).item()
            R_minus = model.advance_theory(g_minus).item()
        derivative = (R_plus - R_minus) / (2 * eps)
        stability = "stable" if abs(derivative) < 1 else "unstable"
        roots.append({"g": float(g_star), "derivative": float(derivative), "stability": stability})
    return roots
