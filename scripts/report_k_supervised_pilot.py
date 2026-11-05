"""短时 K 监督消融：同一完整缓存的 GPU 分块诊断；不训练、不估计 MI。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.views import ensemble_views
from recurrent_rg.evaluation import closure_metrics
from recurrent_rg.execution import configure_cuda_execution
from recurrent_rg.plotting.reports import plot_diagnostics
from scripts.report_ising_scalar_exponents import write_exponent_report


def collect_on_device(model, configurations, *, samples, depth, device,
                      flip=False, seed=12345, chunk_size=8):
    """保存与 CPU 报告同定义的编码；CUDA 抽样流单独标记，不冒充 CPU 抽样。"""
    if chunk_size < 1:
        raise ValueError('Positive chunk size required')
    model.eval()
    direct, rollout, one = [], [], []
    with isolated_rng(device, seed), torch.no_grad():
        for chunk in configurations.split(chunk_size):
            x = ensemble_views(chunk, samples, spin_flip_augment=flip).to(device)
            levels = model.coarse_grain(x, max_rg_steps=depth)
            d = torch.stack([model.theory_encoder(level) for level in levels], 1)
            r = [d[:, 0]]
            for _ in range(depth):
                r.append(model.advance_theory(r[-1]))
            direct.append(d.cpu().numpy())
            rollout.append(torch.stack(r, 1).cpu().numpy())
            one.append(model.advance_theory(d[:, :-1]).cpu().numpy())
    return dict(direct=np.concatenate(direct), rollout=np.concatenate(rollout),
                one_step=np.concatenate(one))


def draw_losses(history, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    for ax, split, title in zip(axes, ('train', 'test'), ('训练集', '测试集')):
        epoch = [r['epoch'] for r in history]
        curves = (
            ('加权 K 监督', lambda r: r.get(split+'_weighted_coupling_supervision', 0.)),
            ('加权 Q 重构', lambda r: 10*r[split+'_separation']),
            ('加权子集一致性', lambda r: 2*r[split+'_anchor']+5*r[split+'_coarse_consistency']),
            ('加权 R 闭合', lambda r: 5*(r[split+'_commutator']+r[split+'_rollout'])),
        )
        for label, value in curves:
            ax.plot(epoch, [value(r) for r in history], label=label)
        ax.set(xlabel='更新次数', ylabel='加权损失', title=title)
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'loss_comparison.png', dpi=160)
    plt.close(fig)


def report_one(run, checkpoint, out, train, test, device):
    out.mkdir(parents=True, exist_ok=False)
    loaded = load_run(run, checkpoint)
    model, cfg = loaded.model, loaded.config
    if (cfg['model']['block_factor'] != 2 or cfg['model']['theory_dim'] != 1
            or cfg['data']['coupling_names'] != ['K']
            or cfg['training']['max_rg_steps'] != 2):
        raise ValueError('This explicit pilot requires scalar b2 depth2 K-only data')
    model.to(device)
    execution = configure_cuda_execution(model, device, independent_c=True)
    opts = dict(samples=cfg['evaluation']['samples'], depth=2, device=device,
                flip=cfg['data']['spin_flip_augment'])
    started = time.monotonic()
    tr = collect_on_device(model, train.configurations, **opts)
    te = collect_on_device(model, test.configurations, **opts)
    model.cpu()
    configure_cuda_execution(model, 'cpu', independent_c=True)
    encodings = dict(train=tr['direct'], test=te['direct'], test_rollout=te['rollout'],
                     train_labels=train.couplings.numpy(), test_labels=test.couplings.numpy())
    np.savez_compressed(out/'encodings.npz', **encodings)
    plot_diagnostics(model, te, test.couplings.numpy(), ['K'], out,
                     training_inputs=tr['direct'][:, :-1])
    fixed = json.loads((out/'fixed_points.json').read_text())
    metrics = dict(closure=closure_metrics(te, reference_direct=tr['direct']),
                   fixed_points=fixed, actual_epochs=loaded.payload['epoch'],
                   diagnostic=dict(device=str(device), systems=[len(train.configurations), len(test.configurations)],
                       samples=opts['samples'], seed=12345, sampling_repeats=0,
                       execution=execution, seconds=time.monotonic()-started,
                       note='One saved-model CUDA C draw; not the historical CPU draw or a multi-draw uncertainty audit'),
                   supervision_mse={}, covariance_eigenvalues={}, effective_rank={})
    for split, trajectory, ds in (('train', tr, train), ('test', te, test)):
        metrics['supervision_mse'][split] = float(np.mean((trajectory['direct'][:, 0, 0]-ds.couplings.numpy().reshape(-1))**2))
        variances = trajectory['direct'][:, :, 0].var(0)
        metrics['covariance_eigenvalues'][split] = [[float(v)] for v in variances]
        metrics['effective_rank'][split] = [1 if v > 0 else 0 for v in variances]
    exponents = write_exponent_report(loaded, metrics, fixed, encodings, out)
    for root in exponents['roots']:
        point = root['g'][0]
        root['nearest_training_input_distance'] = float(np.min(abs(tr['direct'][:, :-1, 0]-point)))
        means = np.asarray(exponents['coupling_means']['g_by_scale'])
        root['distance_to_mean_encoding_curve_by_scale'] = [
            float(max(means[:, k].min()-point, point-means[:, k].max(), 0.))
            for k in range(means.shape[1])]
    (out/'critical_exponents.json').write_text(json.dumps(exponents, indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    draw_losses(loaded.payload['history'], out)
    return dict(actual_epochs=loaded.payload['epoch'], roots=exponents['roots'],
                supervision_mse=metrics['supervision_mse'], closure=metrics['closure'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--control-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    cfg = json.loads((args.run/'manifest.json').read_text())['config']
    summary = json.loads((args.run/'summary.json').read_text())
    actual = summary['actual_epochs']
    control_step = min(3000, actual//1000*1000)
    if control_step < 1000:
        raise ValueError('Too few updates for a saved matched control; preserve training and request review')
    train, test = load_ensembles(cfg['data']['cache'])
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(supervised_final=report_one(args.run, 'checkpoint_last.pt', args.output/'supervised_final',
                  train, test, args.device))
    if actual != control_step:
        result['supervised_matched'] = report_one(args.run, control_step, args.output/'supervised_matched',
                                                  train, test, args.device)
    result['unsupervised_matched'] = report_one(args.control_run, control_step,
        args.output/'unsupervised_matched', train, test, args.device)
    result.update(matched_updates=control_step, actual_supervised_updates=actual,
        note='TRAIN microscopic labels only; no theoretical multiplier/exponent target; final and matched-update endpoints are not selected by exponent')
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(matched_updates=control_step, actual_supervised_updates=actual)))


if __name__ == '__main__':
    main()
