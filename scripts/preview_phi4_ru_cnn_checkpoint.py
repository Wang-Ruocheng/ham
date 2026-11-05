"""Explicitly requested CPU-only ru CNN checkpoint preview, separate from training.

Run capture on stdin from the immutable remote snapshot; only public numerical
summaries leave the host. No optimizer, sampling, checkpoint, or source writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

NONLINEAR_RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_decoder_v1'
LOCAL_RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_local_window_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def choose_checkpoint(run):
    status = read(run / 'status.json')
    step = status.get('latest_checkpoint_step', 0)
    if type(step) is not int or step < 1 or step > status['actual_steps']:
        raise ValueError('No fully saved checkpoint confirmed by training status')
    path = run / f'training_step_{step}.pt'
    return path, step, sha(path)


def restore_model(saved, manifest):
    """Strictly restore either historical affine or explicit nonlinear Q."""
    import recurrent_rg.phi4_cnn_qcr as cnn
    if (saved['manifest'] != manifest or saved['recipe'] != manifest['recipe']
            or saved['recipe'] not in ('phi4_ru_b3_cnn_g2_Rlr003_saddle_window_v1', NONLINEAR_RECIPE, LOCAL_RECIPE)
            or manifest['theory_dim'] != 2 or manifest['lattice_sizes'] != [36, 12, 4]
            or manifest['labels_used_in_training'] is not False):
        raise ValueError('Incompatible checkpoint/recipe/manifest')
    model = cnn.CNNPhi4QCR(manifest['scale'], seed=91, theory_dim=2)
    if saved['recipe'] in (NONLINEAR_RECIPE, LOCAL_RECIPE):
        from experiments import phi4_ru_cnn_nonlinear as nonlinear
        if manifest.get('decoder') != nonlinear.DECODER:
            raise ValueError('Nonlinear decoder policy differs from frozen implementation')
        nonlinear.install_decoder(model)
    model.load_state_dict(saved['model'], strict=True)
    return model


def capture(root):
    import resource
    import torch
    from experiments import phi4_hopping_qcr as base
    from experiments import phi4_qcr as physical
    import recurrent_rg.phi4_cnn_qcr as cnn
    import recurrent_rg.phi4_ru_flow as flow

    started, cpu_started = time.monotonic(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (300, 310))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    host = dict(host=os.uname().nodename, load1=os.getloadavg()[0], cpus=os.cpu_count(),
        mem_available_bytes=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024,
        io_full_avg10=float(Path('/proc/pressure/io').read_text().split('full avg10=')[1].split()[0]))
    if (host['host'] != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10
            or host['mem_available_bytes'] < 36*1024**3 or host['load1'] > 1.25*host['cpus']
            or host['io_full_avg10'] >= 20 or torch.cuda.is_initialized()):
        raise RuntimeError('Read-only CPU preview resource admission failed')
    root = Path(root).resolve()
    run = root / 'run'
    plan, plan_sha = read(root/'plan.json'), sha(root/'plan.json')
    if plan_sha != read(root/'job.json')['plan_sha256']:
        raise RuntimeError('Parent plan identity mismatch')
    code = Path(plan['code_root']).resolve()
    manifest, manifest_sha = read(run/'manifest.json'), sha(run/'manifest.json')
    modules = [base, physical, cnn, flow]
    if manifest['recipe'] in (NONLINEAR_RECIPE, LOCAL_RECIPE):
        from experiments import phi4_ru_cnn_nonlinear as nonlinear
        from experiments import phi4_ru_qonly_decoder_balance as decoder
        modules.extend((nonlinear, decoder))
    for module in modules:
        if code not in Path(module.__file__).resolve().parents:
            raise RuntimeError('Module is outside the frozen code snapshot')

    def verify():
        for name, digest in plan['source_sha256'].items():
            if sha(code/name) != digest:
                raise RuntimeError(f'Frozen source mismatch: {name}')
        for name, digest in manifest['source_sha256'].items():
            if sha(name) != digest:
                raise RuntimeError(f'Manifest pin mismatch: {name}')

    def guard():
        if time.process_time()-cpu_started > 285 or time.monotonic()-started > 420:
            raise RuntimeError('Bounded preview budget exceeded; no automatic retry')

    verify()
    checkpoint, step, checkpoint_sha = choose_checkpoint(run)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['step'] != step:
        raise ValueError('Saved checkpoint step differs from training status')
    model = restore_model(saved, manifest)
    del saved
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_sha = base.state_fingerprint(model)
    rng_before = torch.get_rng_state().clone()
    data_root = Path(manifest['data_root']).resolve()
    if sha(data_root/'manifest.json') != manifest['data_manifest_sha256']:
        raise RuntimeError('Pinned physical data manifest differs before loading')
    dataset = physical.load_dataset(data_root)
    if (dataset.manifest_sha256 != manifest['data_manifest_sha256']
            or dataset.configurations.shape != (180, 128, 36, 36, 1)
            or dataset.scale != manifest['scale']):
        raise RuntimeError('Physical dataset/normalization identity mismatch')
    generator = torch.Generator(device='cpu').manual_seed(202611030300)
    direct, pooled, mapped = [], [], []
    # Same 64-field ensemble size and two-half Q definition as training. All
    # four physical chains in every cell, but only one private stochastic C draw.
    with torch.no_grad():
        for chain in dataset.configurations:
            levels = model.coarse_grain(chain[:64].unsqueeze(0), generator=generator)
            gs = [.5*(model.Q(level[:, :32])+model.Q(level[:, 32:])) for level in levels]
            direct.append(torch.stack(gs, 1).numpy()[0])
            pooled.append(torch.stack([model.Q(level) for level in levels], 1).numpy()[0])
            mapped.append(torch.stack([model.advance(g) for g in gs[:2]], 1).numpy()[0])
            guard()
    arrays = dict(direct_g=np.asarray(direct), direct_full_g=np.asarray(pooled),
        one_step_g=np.asarray(mapped), splits=dataset.splits, labels=dataset.labels,
        cell_ids=dataset.cell_ids)
    arrays['train_direct'] = arrays['direct_g'][dataset.train_indices]
    arrays['test_direct'] = arrays['direct_g'][dataset.test_indices]
    roots, receipt = flow.bounded_flow(model, arrays, 'cpu', budget_check=guard)
    # Display the whole observed cloud (including k=2 and TEST); only the
    # display bounds change. Root starts/search remain TRAIN k=0,1 only.
    points = arrays['direct_g'].reshape(-1, 2)
    low, high = points.min(0), points.max(0)
    padding = .1*np.maximum(high-low, 1e-6)
    bounds = np.column_stack((low-padding, high+padding))
    grid = np.stack(np.meshgrid(*[np.linspace(*b, 33) for b in bounds], indexing='xy'), -1).reshape(-1, 2)
    with torch.no_grad():
        next_grid = model.advance(torch.as_tensor(grid, dtype=torch.float64)).numpy()
    arrays.update(display_g=grid, display_R=next_grid, display_displacement=next_grid-grid,
                  display_bounds=bounds)
    verify()
    if (sha(checkpoint) != checkpoint_sha or sha(root/'plan.json') != plan_sha
            or sha(run/'manifest.json') != manifest_sha
            or sha(data_root/'manifest.json') != manifest['data_manifest_sha256']
            or base.state_fingerprint(model) != model_sha
            or any(p.grad is not None for p in model.parameters())
            or not torch.equal(rng_before, torch.get_rng_state()) or torch.cuda.is_initialized()):
        raise RuntimeError('Read-only model/checkpoint/source/RNG identity guard failed')
    guard()
    for value in arrays.values():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite preview array')
    return dict(kind='explicit_user_requested_interim_CPU_preview', final_report=False,
        checkpoint_step=step, checkpoint_sha256=checkpoint_sha, model_sha256=model_sha,
        plan_sha256=plan_sha, manifest_sha256=manifest_sha, roots=roots, flow_analysis=receipt,
        arrays={k:v.tolist() for k,v in arrays.items()}, thermal_nu=None,
        samples_per_chain=64, sample_indices=list(range(64)), physical_chains=180,
        C_draws_per_chain=1, preview_seed=202611030300, data_warning_count=dataset.warning_count,
        history=[x for x in read(run/'history.json') if x['step'] <= step],
        train_probes=[x for x in read(run/'train_probes.json') if x['step'] <= step],
        captured_at_utc=datetime.now(timezone.utc).isoformat(), host_resources=host,
        training_status_at_finish=read(run/'status.json'), checkpoint_unchanged=True,
        model_unchanged=True, source_unchanged=True, cpu_rng_unchanged=True,
        new_optimizer_updates=0, new_physical_sampling=0, cuda_initialized=False,
        optimizer_state_restored=False, preview_cpu_seconds=time.process_time()-cpu_started,
        preview_wall_seconds=time.monotonic()-started,
        qualifications=['中途 checkpoint，不是最终6000步结果；不得据此选择训练终点。',
            '每链前64个已保存构型、一次私有C随机轨迹；不含独立链置信区间。',
            'CPU随机C与正式GPU评价不逐比特配对；保留44条采样警告，未认证平衡。',
            '搜索仅限TRAIN输入包围盒加10%；无根不代表全局不存在根。',
            '坐标包围盒或凸包内不等于具有充分局部数据支撑；数值根不是物理认证。'])


def plot(source, output):
    from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_loss
    data = read(source)
    a = {k:np.asarray(v) for k,v in data['arrays'].items()}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    plt = _setup_plot(None)
    grid, drift, bounds = a['display_g'], a['display_displacement'], a['display_bounds']
    length = np.linalg.norm(drift, axis=1, keepdims=True)
    arrows = drift/np.maximum(length, 1e-15)*.62*np.min(np.ptp(grid, axis=0))/32
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.quiver(grid[:, 0], grid[:, 1], arrows[:, 0], arrows[:, 1], angles='xy',
              scale_units='xy', scale=1, width=.0022, color='#52606d', alpha=.60, zorder=1)
    train = a['splits']=='train'
    for k in range(3):
        for mask, marker, name in ((train, 'o', '训练'), (~train, 'x', '留出')):
            p = a['direct_g'][mask, k]
            ax.scatter(p[:, 0], p[:, 1], s=20, marker=marker, alpha=.75,
                       color=f'C{k}', label=f'{name}覆盖 k={k}，L={36//3**k}', zorder=3)
    names = dict(attracting='吸引点', repelling='排斥点', saddle='鞍点', neutral_or_unresolved='近中性或未定')
    for i, root in enumerate(data['roots'], 1):
        x, y = root['g']
        ax.scatter(x, y, marker='D', s=90, edgecolors='black', zorder=5,
                   facecolors='black' if root['within_train_hull'] else 'white')
        ax.annotate(f"{i}：{names[root['classification']]}", (x,y), xytext=(8,10), textcoords='offset points')
    if not data['roots']:
        ax.text(.02, .97, '有限 TRAIN 支撑搜索内未检出数值不动点', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel='$g_1$', ylabel='$g_2$', aspect='equal',
           title=f"第{data['checkpoint_step']}步中途流场：共享 R(g)−g\n等长箭头仅表示方向；数据覆盖外为外推")
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=.12)
    fig.savefig(out/'R_flow.png', dpi=180); plt.close(fig)
    cells = np.unique(a['cell_ids'])
    labels = np.array([a['labels'][a['cell_ids']==c][0,:2] for c in cells])
    means = np.array([a['direct_g'][a['cell_ids']==c].mean(0) for c in cells])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for d in range(2):
        for k in range(3):
            ax = axes[d,k]
            sc = ax.scatter(labels[:,0], labels[:,1], c=means[:,k,d], s=100, marker='s',
                cmap='coolwarm', vmin=means[...,d].min(), vmax=means[...,d].max())
            ax.set(xlabel='质量参数 r', ylabel='四次耦合 u', title=f'$g_{d+1}$：k={k}')
            fig.colorbar(sc, ax=ax, label='四条物理链平均编码')
    fig.suptitle(f"第{data['checkpoint_step']}步：耦合到编码的映射（标签仅用于评价）")
    fig.savefig(out/'Q_g_ru.png', dpi=160); plt.close(fig)
    _plot_loss(data['history'], out, plt)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capturing = sub.add_parser('capture')
    capturing.add_argument('--root', required=True)
    capturing.add_argument('--output', help='Optional new, separate public diagnostic JSON')
    plotting = sub.add_parser('plot')
    plotting.add_argument('--source', required=True)
    plotting.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        output = Path(args.output).resolve() if args.output else None
        root = Path(args.root).resolve()
        if output is not None:
            if output == root or root in output.parents or output.exists():
                raise ValueError('Preview output must be new and outside the active experiment')
        result = capture(root)
        if output is None:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as handle:
                json.dump(result, handle, ensure_ascii=False, allow_nan=False)
                handle.write('\n')
            print(json.dumps(dict(output=str(output), sha256=sha(output),
                checkpoint_step=result['checkpoint_step'], roots=result['roots'],
                cpu_seconds=result['preview_cpu_seconds'], wall_seconds=result['preview_wall_seconds']),
                ensure_ascii=False, allow_nan=False))
    else:
        print(plot(args.source, args.output))
