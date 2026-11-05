"""Bounded, post-training dual-Ising evaluation and Chinese public figures.

Labels only color/locate measured encodings. They never select roots or axes.
No training, MI, data generation, or unbounded root search occurs here.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from recurrent_rg.checkpoints import load_run, isolated_rng
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.phi4_ru_flow import bounded_flow, full_map_and_jacobian
from recurrent_rg.terminal_r_fit import fingerprint
from scripts.run_dual_ising_current import write_json


@torch.no_grad()
def collect(model, train, test, *, samples=64, repeats=4, seed=91951, reconstruction=False):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    draws, target_draws, decoded_draws = [], [], []
    if reconstruction:
        from recurrent_rg.data.statistics import categorical_plaquette_distribution
    with isolated_rng(device, seed+1):
        for _ in range(repeats):
            batches, targets, decoded = [], [], []
            for dataset in (train, test):
                for offset in range(0, len(dataset.configurations), 8):
                    order = torch.randperm(dataset.configurations.shape[1], generator=generator)[:samples]
                    x = dataset.configurations[offset:offset+8][:, order].to(device=device, dtype=torch.float64)
                    levels = model.coarse_grain(x)
                    half = samples//2
                    g = torch.stack([.5*(model.theory_encoder(v[:, :half]) +
                                         model.theory_encoder(v[:, half:])) for v in levels], 1)
                    batches.append(g.cpu().numpy())
                    if reconstruction:
                        target = torch.stack([categorical_plaquette_distribution(v) for v in levels], 1)
                        pred = model.theory_encoder.distribution_decoder(g)
                        if getattr(model.theory_encoder, 'distribution_reconstruction_mode', '') != 'affine_mse':
                            pred = pred.softmax(-1)
                        targets.append(target.cpu().numpy()); decoded.append(pred.cpu().numpy())
            draws.append(np.concatenate(batches))
            if reconstruction:
                target_draws.append(np.concatenate(targets)); decoded_draws.append(np.concatenate(decoded))
    result = np.stack(draws)
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite final encodings')
    if reconstruction:
        return result, dict(targets=np.stack(target_draws), decoded=np.stack(decoded_draws))
    return result


@torch.no_grad()
def predictions(model, draws):
    x = torch.as_tensor(draws, device=next(model.parameters()).device, dtype=torch.float64)
    result = []
    for source, target in [(0, 1), (1, 2), (0, 2)]:
        g = x[:, :, source]
        for _ in range(target-source): g = model.advance_theory(g)
        result.append(g.cpu().numpy())
    return np.stack(result, 2)


def closure_metrics(draws, predictions_, ntrain):
    result = {}
    for split, selector in [('train', slice(0, ntrain)), ('test', slice(ntrain, None))]:
        entries = []
        for j, (s, t) in enumerate([(0, 1), (1, 2), (0, 2)]):
            # Fixed TRAIN target centering, pooled over coordinates; same denominator for both R endpoints.
            reference = draws[:, :ntrain, t]
            variance = np.mean((reference-reference.mean(axis=(0, 1)))**2)
            rmse = float(np.sqrt(np.mean((predictions_[:, selector, j]-draws[:, selector, t])**2)))
            entries.append(dict(source_scale=s, target_scale=t, rmse=rmse,
                                train_target_std=float(np.sqrt(variance)),
                                nrmse=rmse/float(np.sqrt(variance)) if variance > 0 else None))
        result[split] = entries
    return result


def plot_flow(a, roots, out, plt):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    grid, delta, bounds = a['flow_g'], a['flow_displacement'], a['flow_bounds']
    magnitude = np.linalg.norm(delta, axis=1)
    shade = Normalize(0, max(float(magnitude.max()), 1e-15))
    cmap = LinearSegmentedColormap.from_list('flow_gray', ['#eeeeee', '#333333'])
    # Equal visual lengths in a rectangular plotting window, retaining vector directions in data coordinates.
    width = np.diff(bounds, axis=1)[:, 0]
    display = delta/np.maximum(np.linalg.norm(delta/width, axis=1)[:, None], 1e-15)*(.65/32)
    classes = dict(attracting='吸引点', repelling='排斥点', saddle='鞍点', neutral_or_unresolved='中性待定')
    for k in range(3):
        g = a['direct_g'][:, k]
        low, high = g.min(0), g.max(0)
        try: equations = ConvexHull(g).equations
        except QhullError: equations = None
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), layout='constrained')
        for parameter, ax in enumerate(axes):
            q = ax.quiver(*grid.T, *display.T, magnitude, cmap=cmap, norm=shade,
                          angles='xy', scale_units='xy', scale=1, width=.0025, zorder=1)
            for split, marker, label in [('train', 'o', '训练参数'), ('test', '^', '留出参数')]:
                m = a['splits'] == split
                sc = ax.scatter(*g[m].T, c=a['labels'][m, parameter], cmap='viridis', vmin=0, vmax=1.3,
                                s=22, marker=marker, edgecolors='white', linewidths=.25, zorder=3, label=label)
            ax.add_patch(Rectangle(low, *(high-low), fill=False, ls='--', lw=1.2, ec='#444444',
                                   label='实际编码坐标范围（框内不保证有数据）', zorder=4))
            for number, root in enumerate(roots, 1):
                point = np.asarray(root['g'])
                inside = equations is not None and np.all(equations[:, :2]@point+equations[:, 2] <= 1e-10)
                ax.scatter(*point, marker='*', s=100, color='#bb3440', zorder=5)
                ax.annotate(f'{number} '+classes[root['classification']]+('（外推）' if not inside else ''),
                            point, xytext=(4, 5), textcoords='offset points', fontsize=8, color='#a32434')
            ax.set(xlabel='$g_1$', ylabel='$g_2$', xlim=bounds[0], ylim=bounds[1],
                   title=f'按原始 $K_{parameter+1}$ 着色')
            ax.set_box_aspect(1); ax.grid(alpha=.12); ax.legend(fontsize=7, loc='best')
            fig.colorbar(sc, ax=ax, label=f'生成数据时的原始 $K_{parameter+1}$')
        fig.colorbar(q, ax=axes, location='bottom', shrink=.7,
                     label=r'流动幅度 $\|R(g)-g\|_2$：越小越浅（不是 Jacobian 或损失梯度）')
        fig.suptitle(f'双 Ising，k={k}：R(g)−g 方向与实际数据编码\n'
                     '每点为一条物理链的四次固定 C／子集抽样平均；数值根不等于物理临界点'
                     + ('；k=2 为目标尺度，流场属额外外推' if k == 2 else ''), fontsize=11)
        fig.savefig(out/f'flow_k{k}.png', dpi=170); plt.close(fig)
    return dict(quantity='Euclidean norm of R(g)-g', grayscale='linear', range=[0, float(magnitude.max())],
                equal_display_length=True, dashed_box='per-scale bounding box of actual train+test encodings',
                box_is_not_physical_support=True)


def render_arrays(a, roots, root, out):
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt = _setup_plot(None)
    style = plot_flow(a, roots, out, plt)
    cells = np.unique(a['labels'], axis=0)
    means = np.stack([a['direct_g'][(a['labels'] == cell).all(1)].mean(0) for cell in cells])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for coordinate in range(2):
        for k in range(3):
            ax = axes[coordinate, k]
            sc = ax.scatter(*cells.T, c=means[:, k, coordinate], cmap='viridis', s=65, marker='s')
            ax.set(xlabel='$K_1$', ylabel='$K_2$', title=f'k={k}，$g_{coordinate+1}$')
            fig.colorbar(sc, ax=ax, label=f'$g_{coordinate+1}$')
    fig.suptitle('参数到潜变量：每格为两条独立物理链的均值；不填补未采样参数')
    fig.savefig(out/'coupling_to_g.png', dpi=170); plt.close(fig)
    history = json.loads((root/'history_progress.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    groups = [('Q 表示诊断', [('separation', '局域统计重构'), ('anchor', '微观子集一致性'),
                             ('coarse_consistency', '粗尺度子集一致性')]),
              ('C 粗粒化诊断', [('rsmi', '负经验互信息'), ('kernel_noncollapse', '方差下限'), ('kernel_sign', '保色负对数似然')]),
              ('R 闭环诊断', [('all_start_anchor', '全起点锚定目标'),
                              ('commutator', '单步闭合'), ('rollout', '微观起点多步')])]
    if history and 'train_vae_KL_level_0' in history[0]:
        groups[0]=('Q 弱 KL 表示目标', [('separation','各尺度交叉熵＋KL 均值')]+
                   [(f'vae_KL_level_{k}',f'k={k} 后验 KL') for k in range(3)])
    for ax, (title, terms) in zip(axes, groups):
        for split, ls in [('train', '-'), ('test', '--')]:
            for key, label in terms:
                pts = [(row['epoch'], row.get(split+'_'+key)) for row in history]
                pts = [(x, y) for x, y in pts if y is not None]
                if pts: ax.plot(*np.asarray(pts).T, ls, label=label+('（训练）' if split == 'train' else '（留出）'))
        ax.set(title=title, xlabel='联合训练更新次数', ylabel='未加权损失／诊断量')
        ax.set_yscale('symlog', linthresh=1e-5); ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.savefig(out/'loss_curves.png', dpi=170); plt.close(fig)
    return style


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--comparison-run', type=Path); args = p.parse_args()
    torch.set_num_threads(1)
    root = args.run.resolve(); out = root/'report'; out.mkdir(exist_ok=False)
    final = load_run(root); joint = load_run(root/'checkpoint_last.pt')
    if not (root/'r_only/summary.json').exists(): raise ValueError('Terminal R stage required')
    for part in ('theory_encoder', 'rg_kernel'):
        if fingerprint(getattr(final.model, part)) != fingerprint(getattr(joint.model, part)):
            raise AssertionError('Q/C changed between joint and terminal endpoints')
    model = final.model.to(args.device).eval(); joint.model.to(args.device).eval()
    train, test = load_ensembles(final.config['data']['cache'], mmap=True)
    is_vae = final.config['model'].get('q_encoder') == 'ensemble_vae'
    collected = collect(model, train, test, samples=final.config['training']['sample_batch'], reconstruction=bool(args.comparison_run) or is_vae)
    draws, reconstruction = collected if (args.comparison_run or is_vae) else (collected, None)
    ntrain = len(train.configurations)
    arrays = dict(direct_draws=draws, direct_g=draws.mean(0),
                  labels=torch.cat([train.couplings, test.couplings]).numpy(),
                  splits=np.array(['train']*ntrain+['test']*len(test.configurations)))
    arrays.update(train_direct=arrays['direct_g'][:ntrain], test_direct=arrays['direct_g'][ntrain:])
    metrics = {}
    for name, m in [('joint', joint.model), ('final', model)]:
        pred = predictions(m, draws); arrays[name+'_predictions'] = pred
        metrics[name] = closure_metrics(draws, pred, ntrain)
    adapter = SimpleNamespace(advance=model.advance_theory)
    roots, search = bounded_flow(adapter, arrays, args.device)
    if np.max(np.linalg.norm(arrays['flow_displacement'], axis=1)) <= 1e-10:
        roots = []
        search['isolated_roots_unresolved'] = 'Numerically identity on the grid'
    # The displayed window contains all measured scales; root search remains TRAIN-input-only.
    points = arrays['direct_g'].reshape(-1, 2)
    low, high = points.min(0), points.max(0)
    padding = .1*np.maximum(high-low, 1e-6)
    bounds = np.column_stack([np.minimum(low-padding, arrays['flow_bounds'][:, 0]),
                              np.maximum(high+padding, arrays['flow_bounds'][:, 1])])
    grid = np.stack(np.meshgrid(*[np.linspace(*b, 33) for b in bounds], indexing='xy'), -1).reshape(-1, 2)
    mapped, derivative = full_map_and_jacobian(adapter, grid, args.device)
    # Preserve the search grid and its support masks under their own prefix.
    for key in list(arrays):
        if key.startswith('flow_'): arrays['search_'+key] = arrays.pop(key)
    arrays.update(flow_g=grid, flow_R=mapped, flow_DR=derivative,
                  flow_displacement=mapped-grid, flow_bounds=bounds)
    np.savez_compressed(out/'evaluation_arrays.npz', **arrays)
    write_json(out/'roots.json', dict(roots=roots, search=search))
    style = render_arrays(arrays, roots, root, out)
    if is_vae:
        from scripts.report_ensemble_vae_multiscale import report_multiscale
        from scripts.report_phi4_hopping_qcr import _setup_plot
        scale_diagnostics = report_multiscale(arrays,reconstruction,roots,
            json.loads((root/'history_progress.json').read_text()),out,_setup_plot(None))
        np.savez_compressed(out/'reconstruction_arrays.npz',**reconstruction)
    terminal = json.loads((root/'r_only/summary.json').read_text())
    report = dict(status='complete', joint_Q_C_updates=final.config['training']['epochs'],
                  extra_R_updates=terminal['actual_extra_R_updates'], terminal_stop_reason=terminal['stop_reason'],
                  Q_C_unchanged=True, closure=metrics, flow=search, root_count=len(roots), plot_style=style,
                  evaluation_seed=91951, draws_per_chain=4, physical_chains=[ntrain, len(test.configurations)],
                  encoding='Same two-half mean as training; independent fixed evaluation C/subset draws',
                  labels_used_for_training=False, thermal_nu=None,
                  limitations=['框内空白不代表物理数据支撑；框外是外推。',
                               '四次 C／子集重复不是四条独立物理链。',
                               '有限格点与单训练种子；低闭环误差不认证正确相结构或临界指数。',
                               '数值根搜索有界且不完备；满数值秩不认证二维物理支撑。'])
    write_json(out/'report.json', report)
    if is_vae:
        report['multiscale_axis_separation_scores']=[row['axis_separation_score'] for row in scale_diagnostics]
        report['Q_architecture']='Shared ensemble VAE, weak KL=.01, cross-subset CE, posterior-mean coordinates'
        write_json(out/'report.json',report)
    if args.comparison_run:
        from scripts.compare_dual_ising_decoders import compare
        compare(model, draws, reconstruction, train, test, args.comparison_run, out, final.config, metrics['final'])
    print(json.dumps(dict(status='complete', roots=len(roots), closure=metrics['final']), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
