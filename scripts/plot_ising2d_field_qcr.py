"""Physical parameter grid, learned latent geometry, and discrete RG diagnostics."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.optimize import root
from scipy.spatial import cKDTree
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.ising2d_field import field_observables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    summary_path = args.run/'summary.json'
    if not summary_path.exists():
        if args.checkpoint is None:
            parser.error('Training is unfinished; specify a saved checkpoint')
        summary_path = args.run/'run_config.json'
    summary = json.loads(summary_path.read_text())
    if 'history' not in summary:
        summary['history'] = json.loads((args.run/'history_progress.json').read_text())
    cfg = summary['config']
    torch.set_num_threads(args.threads)
    model = RecurrentRGNetwork2D(levels=cfg['levels'], hidden=cfg['hidden'], theory_dim=2,
                               learnable_kernel=cfg.get('learnable_kernel', True)).double().eval()
    model.theory_encoder.binary_lookup = model.rg_kernel.binary_lookup = True
    state = torch.load(args.checkpoint or args.run/'qcr_model.pt', weights_only=True)
    if 'model' in state:
        summary['history'] = [h for h in summary['history'] if h['epoch'] <= state['epoch']]
        cfg = dict(cfg, checkpoint_step=state['epoch'])
        state = state['model']
    else:
        cfg = dict(cfg, checkpoint_step=cfg['epochs'])
    step_title = f"seed {cfg['seed']}, step {cfg['checkpoint_step']}"
    mismatch = model.load_state_dict(state, strict=False)
    if args.output_dir:
        args.run = args.output_dir
    args.run.mkdir(parents=True, exist_ok=True)
    assert not mismatch.unexpected_keys and all(k.startswith('action_network.') for k in mismatch.missing_keys)
    tx, ty, vx, vy = torch.load(cfg['data_cache'], weights_only=True)
    data, labels = torch.cat((tx, vx)), torch.cat((ty, vy)).numpy()
    direct, rollout, observables, split_g, coarse_moments = [], [], [], [], []
    with torch.no_grad():
        for chunk in data.split(4):
            traj = model(chunk.double())
            direct.append(traj.direct_theory.numpy())
            rollout.append(traj.rollout_theory.numpy())
            moments = []
            for level in traj.configurations:
                sample_m = level.mean(dim=(2, 3, 4))
                moments.append(torch.stack((sample_m.mean(1), sample_m.abs().mean(1),
                                            sample_m.square().mean(1)), dim=-1).numpy())
            coarse_moments.append(np.stack(moments, axis=1))
            observables.append(field_observables(chunk.squeeze(-1).numpy()).mean(1))
            midpoint = chunk.shape[1]//2
            split_g.append(np.stack([model.theory_encoder(chunk[:, :midpoint].double()).numpy(),
                                     model.theory_encoder(chunk[:, midpoint:].double()).numpy()], axis=1))
    direct, rollout, observables, split_g = map(np.concatenate, (direct, rollout, observables, split_g))
    coarse_moments = np.concatenate(coarse_moments)
    with torch.no_grad():
        predicted = model.advance_theory(torch.from_numpy(direct[:, :-1])).numpy()
    metrics = {}
    for name, subset in (('train', slice(None, len(tx))), ('test', slice(len(tx), None))):
        d, r = direct[subset], rollout[subset]
        scale = np.sqrt(d.var(axis=0).mean(-1))
        rmse = np.sqrt(np.mean((r-d)**2, axis=(0, 2)))
        identity_error = np.sqrt(np.mean(np.diff(d, axis=1)**2, axis=(0, 2)))
        one_step_error = np.sqrt(np.mean((predicted[subset]-d[:, 1:])**2, axis=(0, 2)))
        pc_errors = []
        for k in range(d.shape[1]):
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(d[:, k].T, bias=True))
            projected_rmse = np.sqrt(np.mean(((r[:, k]-d[:, k])@eigenvectors)**2, axis=0))
            pc_errors.append((projected_rmse/np.sqrt(np.maximum(eigenvalues, 1e-30))).tolist())
        metrics[name] = {'rollout_rmse': rmse.tolist(), 'rollout_nrmse': (rmse/scale).tolist(),
                         'one_step_rmse': one_step_error.tolist(),
                         'identity_map_one_step_rmse': identity_error.tolist(),
                         'one_step_error_relative_to_identity': (one_step_error/np.maximum(identity_error, 1e-12)).tolist(),
                         'covariance_eigenvalues_by_scale': [np.linalg.eigvalsh(np.cov(d[:, k].T, bias=True)).tolist() for k in range(cfg['levels'])],
                         'microscopic_split_rmse': float(np.sqrt(np.mean(np.diff(split_g[subset], axis=1)**2))),
                         'rollout_pc_nrmse_by_scale_weak_then_strong': pc_errors}
    # Disaggregate closure instead of allowing strongly magnetized cells to
    # hide weaker-field errors in one global average.
    regional = {}
    for name, mask in (('B_zero', labels[:, 1] == 0),
                       ('B_nonzero_small', (abs(labels[:, 1]) > 0) & (abs(labels[:, 1]) <= .008)),
                       ('B_large', abs(labels[:, 1]) >= .02)):
        mask &= np.arange(len(labels)) >= len(tx)
        d, r = direct[mask], rollout[mask]
        regional[name] = {'ensembles': int(mask.sum()),
                         'rollout_rmse': np.sqrt(np.mean((r-d)**2, axis=(0, 2))).tolist() if mask.any() else None}
    cells = np.unique(labels, axis=0)
    ks, bs = np.unique(labels[:, 0]), np.unique(labels[:, 1])
    uniform_b = np.allclose(np.diff(bs), np.diff(bs)[0])
    near_b = float(bs[bs > 1e-12].min())
    means = np.array([direct[np.all(labels == cell, axis=1)].mean(0) for cell in cells]).reshape(len(ks), len(bs), cfg['levels'], 2)
    obs = np.array([observables[np.all(labels == cell, axis=1)].mean(0) for cell in cells]).reshape(len(ks), len(bs), 3)
    kc = .5*np.log(1+np.sqrt(2))
    zi = int(np.argmin(abs(bs)))
    ki = int(np.argmin(abs(ks-kc)))
    critical_input_label = '$Q(D_{K_c,0})$' if abs(ks[ki]-kc)<1e-10 else f'Q at K={ks[ki]:.3f}, B=0 (near Kc)'

    # This is a map of learned coordinates and measured observables, not an
    # independently inferred thermodynamic phase boundary.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fields = (obs[..., 1], means[:, :, 0, 0], means[:, :, 0, 1])
    for ax, field, title in zip(axes, fields, ('Measured magnetization (data)', 'Learned $g^{(1)}_0$', 'Learned $g^{(2)}_0$')):
        im = ax.pcolormesh(bs, ks, field, shading='nearest', cmap='coolwarm')
        ax.plot([0, 0], [kc, ks[-1]], 'k--', lw=1, label='Reference coexistence line')
        ax.scatter([0], [kc], c='black', marker='*', s=80, label='Reference critical point')
        ax.set(xlabel='$B=\\beta h$', ylabel='$K=\\beta J$', title=title)
        if not uniform_b:
            ax.set_xscale('symlog', linthresh=.002)
        fig.colorbar(im, ax=ax, shrink=.8)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{step_title}: physical parameter plane (reference boundary, not fitted)')
    fig.tight_layout()
    fig.savefig(args.run/'physical_parameter_map.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    g0 = means[:, :, 0]
    for ax, color, title in zip(axes, (np.broadcast_to(ks[:, None], g0.shape[:2]), np.broadcast_to(bs[None], g0.shape[:2])),
                                 ('Latent microscopic grid, colored by K', 'Latent microscopic grid, colored by B')):
        for i in range(len(ks)):
            ax.plot(g0[i, :, 0], g0[i, :, 1], color='.8', lw=.5)
        for j in range(len(bs)):
            ax.plot(g0[:, j, 0], g0[:, j, 1], color='.8', lw=.5)
        norm = SymLogNorm(linthresh=.002, vmin=bs.min(), vmax=bs.max()) if title.endswith('B') and not uniform_b else None
        im = ax.scatter(g0[..., 0], g0[..., 1], c=color, cmap='coolwarm', norm=norm, s=14)
        ax.plot(g0[:, zi, 0], g0[:, zi, 1], 'k.-', lw=1, ms=4, label='B=0')
        ax.scatter(*g0[ki, zi], marker='*', c='black', s=100, label=critical_input_label)
        ax.set(xlabel='$g^{(1)}$', ylabel='$g^{(2)}$', title=title)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.suptitle(step_title)
    fig.savefig(args.run/'learned_parameter_grid.png', dpi=200)
    plt.close(fig)

    # A separate explicitly rescaled view exposes weak directions without
    # implying they have comparable raw variance or are noise-free.
    center = g0.reshape(-1, 2).mean(0)
    pe, pv = np.linalg.eigh(np.cov(g0.reshape(-1, 2).T, bias=True))
    transform = pv[:, ::-1]/np.sqrt(np.maximum(pe[::-1], 1e-30))
    white = (g0-center)@transform
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, color, title in zip(axes, (np.broadcast_to(ks[:, None], g0.shape[:2]), np.broadcast_to(bs[None], g0.shape[:2])),
                                 ('Rescaled latent grid, colored by K', 'Rescaled latent grid, colored by B')):
        for i in range(len(ks)):
            ax.plot(white[i, :, 0], white[i, :, 1], color='.8', lw=.5)
        for j in range(len(bs)):
            ax.plot(white[:, j, 0], white[:, j, 1], color='.8', lw=.5)
        im = ax.scatter(white[..., 0], white[..., 1], c=color, cmap='coolwarm', s=14)
        ax.plot(white[:, zi, 0], white[:, zi, 1], 'k.-', lw=1, ms=4, label='B=0')
        ax.scatter(*white[ki, zi], marker='*', c='black', s=100, label=critical_input_label)
        ax.set(xlabel='PC1 / cell-mean SD', ylabel='PC2 / cell-mean SD', title=title)
        ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Visualization only: the weak second direction is amplified; see replica-noise diagnostics')
    fig.tight_layout()
    fig.savefig(args.run/'whitened_latent_grid.png', dpi=200)
    plt.close(fig)

    def residual(z):
        with torch.no_grad():
            return model.theory_rg(torch.as_tensor(z, dtype=torch.float64)).numpy()

    def jac(z):
        return torch.autograd.functional.jacobian(model.theory_rg, torch.tensor(z, dtype=torch.float64, requires_grad=True)).detach().numpy()

    envelope = direct.reshape(-1, 2)
    inputs = direct[:, :-1].reshape(-1, 2)
    span = np.maximum(np.ptp(envelope, axis=0), .1)
    lower, upper = envelope.min(0)-.15*span, envelope.max(0)+.15*span
    tree = cKDTree(inputs)
    radius = .1*np.sqrt(inputs.var(0).sum())
    roots = []
    for guess in itertools.product(np.linspace(lower[0], upper[0], 11), np.linspace(lower[1], upper[1], 11)):
        solution = root(residual, guess, jac=jac)
        z = solution.x
        if (not np.isfinite(z).all() or np.linalg.norm(residual(z)) > 1e-8 or np.any(z < lower) or np.any(z > upper)
                or any(np.linalg.norm(z-np.array(f['g'])) < 1e-4 for f in roots)):
            continue
        eigenvalues = np.linalg.eigvals(np.eye(2)+jac(z))
        mod = abs(eigenvalues)
        stability = ('stable' if np.all(mod < 1-1e-4) else 'unstable' if np.all(mod > 1+1e-4)
                     else 'saddle' if mod.min() < 1-1e-4 and mod.max() > 1+1e-4 else 'near-marginal')
        nearest = np.argmin(np.linalg.norm(g0.reshape(-1, 2)-z, axis=1))
        roots.append({'g': z.tolist(), 'stability': stability,
                      'eigenvalues': [{'real': float(v.real), 'imag': float(v.imag)} for v in eigenvalues],
                      'distance_to_sampled_R_input': float(tree.query(z)[0]),
                      'nearest_microscopic_grid_cell_K_B': cells[nearest].tolist(),
                      'distance_to_nearest_microscopic_cell': float(np.linalg.norm(g0.reshape(-1, 2)[nearest]-z)),
                      'mapping_caution': 'Nearest grid point only; not an inverse calibration or physical identification'})
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 29), np.linspace(lower[1], upper[1], 29))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    delta = residual(grid)
    speed = np.linalg.norm(delta, axis=1)
    arrow = delta/np.maximum(speed[:, None], 1e-12)*.027*min(upper-lower)
    near = tree.query(grid)[0] <= radius
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    ax = axes[0]
    ax.quiver(*grid[~near].T, *arrow[~near].T, color='.85', angles='xy', scale_units='xy', scale=1, width=.003)
    q = ax.quiver(*grid[near].T, *arrow[near].T, speed[near], cmap='viridis', angles='xy', scale_units='xy', scale=1, width=.0035)
    fig.colorbar(q, ax=ax, shrink=.8, label='$||R(g)-g||$')
    for level in range(cfg['levels']):
        ax.scatter(direct[:, level, 0], direct[:, level, 1], s=4, alpha=.2, color=f'C{level}', label=f'k={level}')
    zero_line = means[:, zi, 0]
    ax.plot(*zero_line.T, color='black', lw=1.4, label='B=0, varying K (not RG time)')
    ax.scatter(*zero_line[ki], marker='D', c='gold', edgecolors='black', s=55,
               label=critical_input_label, zorder=5)
    for i, f in enumerate(roots):
        ax.scatter(*f['g'], marker='*', c='black', s=100)
        ax.annotate(f"F{i+1}: {f['stability']}", f['g'], xytext=(5, 5), textcoords='offset points', fontsize=8)
    field_probe = near_b if uniform_b else .02
    for i, (k, b) in enumerate(((.25, 0), (kc, 0), (.62, 0), (kc, -field_probe), (kc, field_probe))):
        a, c = np.argmin(abs(ks-k)), np.argmin(abs(bs-b))
        actual = means[a, c]
        path = [actual[0]]
        for _ in range(cfg['levels']-1):
            path.append(path[-1]+residual(path[-1]))
        path = np.array(path)
        axes[1].plot(*actual.T, 'o-', c=f'C{i}', label=f'K={ks[a]:.3f}, B={bs[c]:.3f}', ms=4)
        axes[1].plot(*path.T, 'x--', c=f'C{i}', ms=4)
        axes[1].annotate('k=0', actual[0], xytext=(3, 3), textcoords='offset points', fontsize=7)
    for ax in axes:
        ax.set(xlabel='$g^{(1)}$', ylabel='$g^{(2)}$', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    axes[0].set_title('Discrete R field; gray = far from sampled inputs')
    axes[1].set_title('Solid: actual Q after C; dashed: R rollout')
    fig.text(.5, .02, 'Normalized arrow lengths. Numerical roots are not established physical fixed points.', ha='center', fontsize=8)
    fig.suptitle(step_title)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(args.run/'R_vector_field.png', dpi=200)
    plt.close(fig)

    history = summary['history']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, name, weight in zip(axes.flat, ('total', 'anchor', 'coarse_consistency', 'kernel_noncollapse', 'separation', 'commutator', 'rollout'), (1, 2, 5, 2, 10, 5, cfg.get('objective_kwargs', {}).get('rollout_weight', 0))):
        if f'train_{name}' not in history[0]:
            ax.set_visible(False)
            continue
        for split in ('train', 'test'):
            ax.plot([h['epoch'] for h in history], [weight*h[f'{split}_{name}'] for h in history], label=split)
        ax.set(title=f'{name}, weight={weight}', xlabel='Optimizer step', ylabel='Weighted loss')
        if name in ('total', 'anchor'):
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.run/'loss_components.png', dpi=200)
    plt.close(fig)
    patterns = torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64)
    with torch.no_grad():
        output = model.rg_kernel(patterns.reshape(16, 1, 2, 2, 1)).reshape(16).numpy()
    eig = np.linalg.eigvalsh(np.cov(direct[:, 0].T, bias=True))
    cell_g = g0.reshape(-1, 2)
    cell_eig, cell_axes = np.linalg.eigh(np.cov(cell_g.T, bias=True))
    residuals = np.concatenate([direct[np.all(labels == cell, axis=1), 0]-cell_g[i]
                                for i, cell in enumerate(cells)])
    # Unbias pooled within-cell covariance for the actual replica count.
    replica_counts = np.array([np.all(labels == cell, axis=1).sum() for cell in cells])
    assert np.all(replica_counts == replica_counts[0]) and replica_counts[0] >= 2
    replicas = int(replica_counts[0])
    within_cov = residuals.T@residuals/(len(cells)*(replicas-1))
    pc_noise = np.diag(cell_axes.T@within_cov@cell_axes)
    pc_snr = np.sqrt(np.maximum(cell_eig-pc_noise/replicas, 0)/np.maximum(pc_noise, 1e-30))
    # Post-hoc readout only: fit measured observables on training cells, then
    # evaluate held-out cells. These coefficients never affect Q/C/R training.
    z = direct[:, 0]
    mu, sd = z[:len(tx)].mean(0), z[:len(tx)].std(0)
    z = (z-mu)/np.maximum(sd, 1e-12)
    design = np.column_stack((np.ones(len(z)), z, z[:, 0]**2, z[:, 0]*z[:, 1], z[:, 1]**2))
    coefficients = np.linalg.lstsq(design[:len(tx)], observables[:len(tx), :2], rcond=None)[0]
    readout = design[len(tx):]@coefficients
    target = observables[len(tx):, :2]
    r2 = 1-((readout-target)**2).sum(0)/((target-target.mean(0))**2).sum(0)
    # Compare with a one-coordinate probe: a tiny second PCA variance is not
    # sufficient to identify intrinsic rank, since curves also have >1 PCs.
    _, axes_train = np.linalg.eigh(np.cov(direct[:len(tx), 0].T, bias=True))
    pc1 = (direct[:, 0]-direct[:len(tx), 0].mean(0))@axes_train[:, -1]
    pc1 /= pc1[:len(tx)].std()
    pc1_probes = {}
    for degree in (2, 7):
        features = np.column_stack([pc1**i for i in range(degree+1)])
        beta = np.linalg.lstsq(features[:len(tx)], observables[:len(tx), :2], rcond=None)[0]
        prediction = features[len(tx):]@beta
        scores = 1-((prediction-target)**2).sum(0)/((target-target.mean(0))**2).sum(0)
        pc1_probes[f'degree_{degree}'] = dict(bond_per_site=float(scores[0]), magnetization=float(scores[1]))
    report = {'config': cfg, 'metrics': metrics, 'test_regions': regional, 'fixed_points': roots,
              'microscopic_covariance_eigenvalues': eig.tolist(),
              'microscopic_second_pc_fraction': float(eig[0]/eig.sum()),
              'microscopic_participation_rank': float(eig.sum()**2/(eig**2).sum()),
              'replica_diagnostic': {'between_cell_mean_pc_variances_ascending': cell_eig.tolist(),
                                     'within_cell_replica_variances_in_same_axes': pc_noise.tolist(),
                                     'signal_sd_over_replica_noise_sd': pc_snr.tolist(),
                                     'caution': 'Small PCA variance alone does not prove absence of information; compare with replica noise'},
              'posthoc_quadratic_observable_readout_test_R2': {'bond_per_site': float(r2[0]), 'magnetization': float(r2[1])},
              'posthoc_PC1_only_observable_readout_test_R2': pc1_probes,
              'readout_caution': 'Quadratic probe fitted on train cells after freezing Q; not supervision, not proof of statistical sufficiency',
              'critical_direct_trajectory': means[ki, zi].tolist(),
              'kernel_table': [{'input': p, 'output': float(o)} for p, o in zip(patterns.tolist(), output)],
              'kernel_oddness_violations': int(np.count_nonzero(~np.isclose(output, -output[::-1]))),
              'kernel_uniform_inputs': {'all_minus_output': float(output[0]), 'all_plus_output': float(output[-1])},
              'plot_box': [lower.tolist(), upper.tolist()], 'near_sample_radius': float(radius),
              'root_search': '121 initial guesses, residual norm < 1e-8, within plotted box; not exhaustive',
              'phase_map_caution': f'Reference coexistence line is overlaid, not learned. Finite {cfg["sites"]}x{cfg["sites"]} system, no thermodynamic phase-boundary extraction.'}
    np.savez_compressed(args.run/'field_flow_data.npz', direct=direct, rollout=rollout, KB=labels,
                        observables=observables, coarse_moments=coarse_moments,
                        coarse_moment_names=np.array(['mean_m', 'mean_abs_m', 'mean_m_squared']),
                        means=means, K_grid=ks, B_grid=bs,
                        is_test=np.arange(len(data)) >= len(tx))
    (args.run/'field_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
