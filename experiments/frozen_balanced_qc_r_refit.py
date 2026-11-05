"""Explicit R-only refit after the paired balanced-Q ablation.

Uses the exact saved mean-of-two-subsets Q trajectories. No MI, coupling
supervision, new R architecture, scheduler, or extra loss is introduced.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run, qcr_state_dict, checkpoint_payload
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import closure_metrics
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss
from recurrent_rg.plotting.reports import plot_diagnostics
from recurrent_rg.runner import prepared_data, file_fingerprint
from experiments.frozen_qc_r_continue import r_objective, predictions


def direct_subset_mean(model, configurations, *, seed):
    """Same half-subset averaging convention as the saved Q-only targets."""
    rows = []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for x in configurations:
            levels = model.coarse_grain(x[None].double())
            n = x.shape[0]//2
            rows.append(torch.stack([.5*(model.theory_encoder(v[:, :n])+
                                         model.theory_encoder(v[:, n:]))
                                     for v in levels], dim=1))
    return torch.cat(rows)


def terms(model, direct, cfg):
    one = latent_squared_error(model.advance_theory(direct[:, :-1]), direct[:, 1:], cfg.loss.coordinate_reduction)
    roll, _ = multistep_closure_loss(model, direct, cfg.loss.coordinate_reduction)
    return dict(one_step=float(one), multistep=float(roll),
                total=float(cfg.loss.weights.commutator*one+cfg.loss.rollout_weight*roll))


def run(source, output, updates=6000, *, max_wall_seconds=540.):
    started = time.perf_counter(); source = Path(source); out = Path(output)
    if updates < 1 or not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError('Positive updates and finite positive wall limit required')
    loaded = load_run(source); cfg = RunConfig.from_dict(loaded.config)
    arm = loaded.payload.get('metadata', {}).get('arm')
    if arm not in ('baseline', 'balanced') or cfg.model.theory_dim != 2 or cfg.loss.closure_metric != 'euclidean':
        raise ValueError('Requires a vector checkpoint from the paired Q-only ablation')
    if tuple(cfg.data.coupling_names) != ('K1', 'K2') or cfg.data.spin_flip_augment:
        raise ValueError('Requires the documented independent dual-Ising K1/K2 grid')
    torch.set_num_threads(cfg.training.threads); torch.manual_seed(cfg.training.seed)
    base = loaded.model.eval(); model = copy.deepcopy(base)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.theory_rg.parameters(): p.requires_grad_(True)
    before = {k: v.clone() for k, v in qcr_state_dict(base).items()}
    # The paired source records a Q-only ablation. The generic cache validator
    # accepts the independent-gradient driver only with Q/R present; validate
    # its data contract using the original full recipe, without training it.
    data_cfg = replace(cfg, training=replace(cfg.training, train_modules=('Q','C','R')))
    train_data, test_data = prepared_data(data_cfg)
    pilot_manifest = json.loads((source.parent/'manifest.json').read_text())
    fingerprint = file_fingerprint(cfg.data.cache)
    if fingerprint != pilot_manifest['data_sha256']:
        raise ValueError('Q-only source data fingerprint changed')
    cache = source.parent/f'{arm}_encodings.npz'
    arrays = np.load(cache)
    train, test = (torch.from_numpy(arrays[s].copy()).double() for s in ('train', 'test'))
    for split, target, dataset in [('train', train, train_data), ('test', test, test_data)]:
        if target.shape != (len(dataset.configurations), cfg.training.max_rg_steps+1, 2):
            raise ValueError('Source target geometry changed')
        np.testing.assert_array_equal(arrays[split+'_labels'], dataset.couplings.numpy())
        if not torch.isfinite(target).all(): raise ValueError('Nonfinite targets')
    verified = direct_subset_mean(base, train_data.configurations[:2, :cfg.training.sample_batch],
                                  seed=pilot_manifest['coarse_draw_seed'])
    torch.testing.assert_close(verified, train[:2], atol=1e-10, rtol=1e-10)
    cfg = replace(cfg, training=replace(cfg.training, epochs=updates, train_modules=('R',)))
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=cfg.to_dict(), training_driver='frozen_balanced_qc_r_refit',
                    source=str(source.resolve()), source_checkpoint_sha256=file_fingerprint(source),
                    target_cache=str(cache.resolve()), target_cache_sha256=file_fingerprint(cache),
                    data_sha256=fingerprint, labels_used_in_training=False,
                    source_Q_ablation=arm, optimized_modules=['R'], frozen_modules=['Q','C'],
                    R_architecture='Unchanged source residual Tanh MLP; warm-start weights; fresh Adam',
                    closure='Original weighted one-step/all-horizon multistep Euclidean closure only',
                    differences='Full-batch fixed mean-of-two-half-subset targets; no fresh C draws during fitting',
                    source_target_verification_max_error=float((verified-train[:2]).abs().max()),
                    requested_R_updates=updates, max_setup_training_seconds=max_wall_seconds,
                    selection='Final completed update, no test-based selection')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    torch.save(dict(train=train, test=test, train_couplings=train_data.couplings,
                    test_couplings=test_data.couplings), out/'fixed_targets.pt')
    opt = torch.optim.Adam(model.theory_rg.parameters(), lr=cfg.training.learning_rate)
    generator = torch.Generator().manual_seed(cfg.training.seed); history = []

    def log(step):
        with torch.no_grad():
            row = dict(epoch=step, train=terms(model, train, cfg), test=terms(model, test, cfg))
        history.append(row)
        (out/'history_progress.json').write_text(json.dumps(history, indent=2)+'\n')
        print(f'R-only {step}/{updates}: train={row["train"]["total"]:.8g}, test={row["test"]["total"]:.8g}', flush=True)

    log(0); completed = 0
    for step in range(1, updates+1):
        if time.perf_counter()-started > max_wall_seconds: break
        opt.zero_grad(set_to_none=True); loss = r_objective(model, train, cfg)
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite R objective at {step}')
        loss.backward(); opt.step(); completed = step
        if step % 500 == 0 or step == updates: log(step)
        if step % 2000 == 0:
            torch.save(checkpoint_payload(model, opt, generator, step, metadata=metadata, history=history),
                       out/f'training_step_{step}.pt')
    if history[-1]['epoch'] != completed: log(completed)
    differences = {}
    for prefix in ('theory_encoder.', 'rg_kernel.'):
        differences[prefix] = max(float((v-qcr_state_dict(model)[key]).abs().max())
                                  for key, v in before.items() if key.startswith(prefix))
        if differences[prefix] != 0: raise AssertionError('Frozen Q/C changed')
    payload = checkpoint_payload(model, opt, generator, completed, metadata=metadata, history=history)
    torch.save(payload, out/'checkpoint_last.pt')
    torch.save(qcr_state_dict(model), out/'qcr_model.pt')
    (out/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(), actual_R_updates=completed,
        R_parameters=sum(p.numel() for p in model.theory_rg.parameters()), history=history), indent=2)+'\n')
    report = dict(contract=metadata, actual_R_updates=completed, frozen_max_parameter_difference=differences,
                  R_parameters=sum(p.numel() for p in model.theory_rg.parameters()), history=history)
    for name, network in [('before', base), ('after', model)]:
        report[name] = {s: closure_metrics(predictions(network, d), reference_direct=train.numpy())
                        for s, d in [('train', train), ('test', test)]}
    full = out/'full_report'
    train_predictions, test_predictions = predictions(model, train), predictions(model, test)
    plot_diagnostics(model, test_predictions, test_data.couplings.numpy(), ['K1','K2'], full,
                     training_inputs=train[:, :-1].numpy())
    roots = json.loads((full/'fixed_points.json').read_text())
    from scripts.dual_ising_grid_diagnostics import write_grid_report, grid_encodings
    final_run = load_run(out, checkpoint='checkpoint_last.pt')
    sampling = json.loads((Path(cfg.data.cache).parent/'sampling.json').read_text())
    phase_summary = write_grid_report(final_run, train_predictions, test_predictions, train_data, test_data,
                                      full, roots, sampling, fingerprint)
    phase_summary['sampling_seed'] = pilot_manifest['coarse_draw_seed']
    phase_summary['encoding_convention'] = 'Saved mean of two half-subset Q outputs; not a new C draw'
    phase_summary['caveat'] = ('Single fixed stochastic-C draw; bounded root search is not exhaustive. '
        'Mapped bare critical lines are interpolation annotations, not fitted invariant boundaries. '
        'Off-data flow is extrapolation. Labels enter evaluation only.')
    np.savez_compressed(full/'encodings.npz', train=train.numpy(), test=test.numpy(),
                        train_rollout=train_predictions['rollout'], test_rollout=test_predictions['rollout'],
                        train_labels=train_data.couplings.numpy(), test_labels=test_data.couplings.numpy())
    # Explicitly annotate root support and nearest microscopic cells, not unique inverse couplings.
    labels = np.concatenate([train_data.couplings.numpy(), test_data.couplings.numpy()])
    direct = torch.cat([train, test]).numpy()
    axes, grid = grid_encodings(labels, direct)
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1,2)
    micro = grid[:, :, 0].reshape(-1,2)
    for root in roots['roots']:
        distance = np.linalg.norm(micro-np.array(root['g']), axis=1)
        ids = np.argsort(distance)[:3]
        root['nearest_microscopic_cells_not_unique_inverse'] = [dict(K1K2=cells[i].tolist(),
            residual_latent_distance=float(distance[i])) for i in ids]
    report['fixed_points'] = roots
    phase_summary['fixed_points'] = roots
    (full/'metrics.json').write_text(json.dumps(phase_summary, indent=2)+'\n')
    (full/'fixed_points.json').write_text(json.dumps(roots, indent=2)+'\n')
    (full/'fixed_points_with_support.json').write_text(json.dumps(roots, indent=2)+'\n')
    # One new stochastic-C draw tests transfer beyond the fitting cache, without estimating MI.
    fresh = direct_subset_mean(model, test_data.configurations[:, :cfg.training.sample_batch], seed=9001)
    report['fresh_C_test_seed9001'] = {name: closure_metrics(predictions(network, fresh),
        reference_direct=train.numpy()) for name, network in [('before',base),('after',model)]}
    comparison_plots(out, model, train, test, labels, history)
    report['elapsed_seconds'] = time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('contract','history')}, indent=2), flush=True)
    return out


def comparison_plots(out, model, train, test, labels, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for s, style in [('train','-'),('test','--')]:
        for ax, key in zip(axes, ['one_step','multistep','total']):
            ax.semilogy([h['epoch'] for h in history],[h[s][key] for h in history],style,label=s)
            ax.set(xlabel='R-only updates',ylabel=key,title=key+' closure'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'loss_comparison.png',dpi=160);plt.close(fig)
    from scripts.dual_ising_grid_diagnostics import grid_encodings
    d=torch.cat([train,test]).numpy();r=predictions(model,torch.cat([train,test]))['one_step']
    cells=np.unique(labels,axis=0)
    means=np.stack([d[np.max(abs(labels-point),axis=1)<1e-8].mean(0) for point in cells])
    predictions_mean=np.stack([r[np.max(abs(labels-point),axis=1)<1e-8].mean(0) for point in cells])
    kc=float(np.log(1+np.sqrt(2))/2)
    phase=(cells[:,0]>kc).astype(int)+2*(cells[:,1]>kc).astype(int)
    colors=np.array(['#444444','#0072b2','#d55e00','#009e73'])
    fig,axs=plt.subplots(2,2,figsize=(12,11),layout='constrained')
    for k in range(2):
        source=means[:,k]
        for j,target in enumerate([means[:,k+1],predictions_mean[:,k]]):
            ax=axs[k,j];delta=target-source
            ax.quiver(*source.T,*delta.T,color=colors[phase],angles='xy',scale_units='xy',scale=1,
                       width=.003,minlength=0,alpha=.7)
            ax.scatter(*source.T,c=colors[phase],s=9,alpha=.6)
            ax.scatter(*target.T,facecolors='none',edgecolors=colors[phase],s=14,linewidths=.5)
            ax.set(xlabel='g1',ylabel='g2',title=f'k={k}: '+('Direct Q target' if j==0 else 'Fitted R(g)-g'),aspect='equal')
            ax.grid(alpha=.15)
    fig.suptitle('Observed coupling cells; actual arrow lengths, no arrow normalization')
    fig.savefig(out/'direct_vs_fitted_flow.png',dpi=160);plt.close(fig)
