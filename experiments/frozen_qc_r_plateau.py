"""R-only continuation with original online half-subset targets and Adam.

Q/C stay bitwise frozen. Only TRAIN fixed-panel closure is used for stopping;
TEST is a diagnostic, and no mutual-information objective is constructed.
"""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import checkpoint_payload, load_run, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import closure_metrics
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss
from recurrent_rg.runner import file_fingerprint, prepared_data
from experiments.frozen_qc_r_continue import continued_r_adam, predictions, r_objective


@torch.no_grad()
def frozen_subset_direct(model, batch):
    """Exact Q target convention of the joint objective, without Q/C gradients."""
    n = batch.shape[1]
    if n < 2 or n % 2:
        raise ValueError('An even, nonempty sample batch is required')
    levels = model.coarse_grain(batch)
    return torch.stack([.5 * (model.theory_encoder(x[:, :n//2]) +
                              model.theory_encoder(x[:, n//2:]))
                        for x in levels], dim=1)


def plateau_reached(rows, window=2000, rtol=.01, confirmations=3, minimum=2000,
                    stability_rtol=.05):
    """Local plateau: little improvement, with current loss still near the best."""
    if (window < 1 or minimum < 1 or confirmations < 1 or not 0 < rtol < 1
            or not 0 < stability_rtol < 1):
        raise ValueError('Invalid plateau settings')
    if len(rows) < confirmations:
        return False
    for end in rows[-confirmations:]:
        step = end['extra_R_updates']
        if step < minimum:
            return False
        candidates = [r for r in rows if r['extra_R_updates'] <= step-window]
        if not candidates:
            return False
        start = candidates[-1]['best_train_closure']
        improvement = (start-end['best_train_closure'])/max(abs(start), 1e-12)
        current = end['train']['total']
        if (not np.isfinite(current) or improvement >= rtol or
                current > end['best_train_closure']*(1+stability_rtol)):
            return False
    return True


def closure_terms(model, direct, cfg):
    with torch.no_grad():
        one = latent_squared_error(model.advance_theory(direct[:, :-1]), direct[:, 1:],
                                   cfg.loss.coordinate_reduction)
        roll, _ = multistep_closure_loss(model, direct, cfg.loss.coordinate_reduction)
    return dict(one_step=float(one), multistep=float(roll),
                total=float(cfg.loss.weights.commutator*one + cfg.loss.rollout_weight*roll))


def fixed_panel(model, configurations, cfg, *, repeats=3, seed=91910):
    rows = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(repeats):
            indices = torch.randperm(configurations.shape[1], generator=generator)[:cfg.training.sample_batch]
            for chunk in configurations.split(cfg.training.system_batch):
                selected = chunk[:, indices].double()
                half = selected.shape[1]//2
                if cfg.data.spin_flip_augment:
                    a, b = selected[:, :half], selected[:, half:]
                    selected = torch.cat((a, -a, b, -b), dim=1)
                rows.append(frozen_subset_direct(model, selected))
    return torch.cat(rows)


def plot_loss(history, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, key in zip(axes, ('one_step', 'multistep', 'total')):
        for split, style in [('train', '-'), ('test', '--')]:
            ax.semilogy([r['extra_R_updates'] for r in history],
                        [r[split][key] for r in history], style, label=split)
        ax.set(xlabel='Additional R-only updates', ylabel=key,
               title='5 one-step + 5 multistep' if key == 'total' else key)
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'loss.png', dpi=160)
    plt.close(fig)


def run(source, output, updates=30000, *, max_wall_seconds=None):
    started = time.perf_counter()
    source, out = Path(source), Path(output)
    if updates < 1 or (max_wall_seconds is not None and
                      (not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0)):
        raise ValueError('Positive updates and finite positive optional wall limit required')
    loaded = load_run(source/'checkpoint_last.pt')
    prior_metadata = loaded.payload['metadata']
    resuming = prior_metadata.get('training_driver') == 'frozen_qc_r_plateau'
    original_cfg = RunConfig.from_dict(prior_metadata['source_config'] if resuming else loaded.config)
    original_cfg.validate()
    if original_cfg.loss.closure_metric != 'euclidean' or original_cfg.loss.tangent_weight or original_cfg.loss.orientation_weight:
        raise ValueError('This continuation preserves sources with plain Euclidean closure only')
    if not loaded.payload['metadata'].get('independent_c'):
        raise ValueError('Requires an independent-gradient source checkpoint')
    if file_fingerprint(original_cfg.data.cache) != loaded.payload['metadata']['data_sha256']:
        raise ValueError('Source data fingerprint changed')
    if (original_cfg.data.train_pool_replicas, original_cfg.data.test_pool_replicas) != (1, 1):
        raise ValueError('This audit requires the original unpooled ensembles')
    torch.set_num_threads(original_cfg.training.threads)
    model = loaded.model.eval()
    before = {k: v.clone() for k, v in qcr_state_dict(model).items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.theory_rg.parameters():
        parameter.requires_grad_(True)
    optimizer = continued_r_adam(model, loaded.payload)
    initial_adam_steps = [float(optimizer.state[p]['step']) for p in model.theory_rg.parameters()]
    train_data, test_data = prepared_data(original_cfg)
    offset = int(loaded.payload.get('extra_R_updates', 0)) if resuming else 0
    base_epoch = int(prior_metadata['source_epoch']) if resuming else int(loaded.payload['epoch'])
    if loaded.payload['epoch'] != base_epoch+offset:
        raise ValueError('R-only continuation epoch bookkeeping is inconsistent')
    cfg = replace(original_cfg, training=replace(original_cfg.training,
                  train_modules=('R',), epochs=base_epoch+offset+updates))
    cfg.validate()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=cfg.to_dict(), source_config=original_cfg.to_dict(),
        training_driver='frozen_qc_r_plateau', independent_c=True,
        source=str(source.resolve()), source_epoch=base_epoch,
        resumed_extra_R_updates=offset,
        original_joint_source=prior_metadata.get('original_joint_source', str(source.resolve())),
        source_checkpoint_sha256=file_fingerprint(loaded.checkpoint),
        data_sha256=file_fingerprint(cfg.data.cache), optimized_modules=['R'], frozen_modules=['Q','C'],
        labels_used_in_training=False, MI_evaluated=False,
        target_convention='Online fresh stochastic C; mean of Q on two half-subsets at every scale',
        optimizer='Original R Adam moments/steps and lr; original batch/torch RNG restored; no scheduler',
        objective='Unchanged source weighted one-step and all-horizon multistep Euclidean closure',
        requested_extra_R_updates=updates, max_wall_seconds=max_wall_seconds,
        stopping='TRAIN only: <1% best fixed-panel improvement over 2000 updates and current loss within 5% of best, three consecutive checks every 500',
        fixed_panel_repeats=3, fixed_panel_seed=91910, selection='Final plateau endpoint, not TEST-selected',
        source_vs_continuation='Only Q/C updates disabled and operational TRAIN plateau stopping added')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    panels = (torch.load(source/'fixed_diagnostic_targets.pt', weights_only=True) if resuming else
              {split: fixed_panel(model, ds.configurations, original_cfg)
               for split, ds in [('train', train_data), ('test', test_data)]})
    torch.save(panels, out/'fixed_diagnostic_targets.pt')
    generator = torch.Generator()
    generator.set_state(loaded.payload['batch_generator_state'])
    # Fixed-panel construction is RNG-forked. Resume the actual training stream.
    torch.set_rng_state(loaded.payload['torch_rng_state'])
    history = list(loaded.payload['history']) if resuming else []
    if resuming and (not history or history[-1]['extra_R_updates'] != offset):
        raise ValueError('Missing or inconsistent R plateau history')
    previous_elapsed = history[-1]['elapsed_seconds'] if history else 0.
    best = history[-1]['best_train_closure'] if history else float('inf')
    completed = offset
    reason = 'update_limit'

    def save(step, name):
        payload = checkpoint_payload(model, optimizer, generator, metadata['source_epoch']+step,
                                     metadata=metadata, history=history)
        payload['extra_R_updates'] = step
        torch.save(payload, out/name)

    def evaluate(step):
        nonlocal best
        row = dict(epoch=metadata['source_epoch']+step, extra_R_updates=step,
                   elapsed_seconds=previous_elapsed+time.perf_counter()-started,
                   lr=optimizer.param_groups[0]['lr'])
        for split, d in panels.items():
            row[split] = closure_terms(model, d, cfg)
            row[split+'_metrics'] = closure_metrics(predictions(model, d), reference_direct=panels['train'].numpy())
        best = min(best, row['train']['total'])
        row['best_train_closure'] = best
        history.append(row)
        (out/'history_progress.json').write_text(json.dumps(history, indent=2)+'\n')
        plot_loss(history, out)
        save(step, 'checkpoint_last.pt')
        print(f'R-only +{step}: train={row["train"]["total"]:.7f} test={row["test"]["total"]:.7f} elapsed={row["elapsed_seconds"]:.1f}s', flush=True)

    if resuming:
        (out/'history_progress.json').write_text(json.dumps(history, indent=2)+'\n')
        plot_loss(history, out)
        save(offset, 'checkpoint_last.pt')
    else:
        evaluate(0)
    for step in range(offset+1, offset+updates+1):
        ids = torch.randperm(len(train_data.configurations), generator=generator)[:cfg.training.system_batch]
        indices = torch.randperm(train_data.configurations.shape[1], generator=generator)[:cfg.training.sample_batch]
        selected = train_data.configurations[ids][:, indices].double()
        half = selected.shape[1]//2
        if cfg.data.spin_flip_augment:
            a, b = selected[:, :half], selected[:, half:]
            selected = torch.cat((a, -a, b, -b), dim=1)
        target = frozen_subset_direct(model, selected)
        optimizer.zero_grad(set_to_none=True)
        loss = r_objective(model, target, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite R loss at extra step {step}')
        loss.backward(); optimizer.step(); completed = step
        wall_limit = max_wall_seconds is not None and time.perf_counter()-started >= max_wall_seconds
        if step % 500 == 0 or step == offset+updates or wall_limit:
            evaluate(step)
            save(step, f'training_step_{metadata["source_epoch"]+step}.pt')
            if plateau_reached(history):
                reason = 'train_plateau'; break
        if wall_limit:
            reason = 'wall_limit'; break
    differences = {prefix: max((float((before[k]-v).abs().max())
                    for k, v in qcr_state_dict(model).items() if k.startswith(prefix)), default=0.)
                   for prefix in ('theory_encoder.', 'rg_kernel.')}
    if any(differences.values()):
        raise AssertionError('Frozen Q/C changed')
    summary = dict(config=cfg.to_dict(), contract=metadata, actual_epochs=metadata['source_epoch']+completed,
                   actual_extra_R_updates=completed, reason=reason, history=history,
                   elapsed_seconds=time.perf_counter()-started, frozen_max_parameter_difference=differences,
                   R_parameters=sum(p.numel() for p in model.theory_rg.parameters()),
                   initial_R_Adam_steps=initial_adam_steps,
                   final_R_Adam_steps=[float(optimizer.state[p]['step']) for p in model.theory_rg.parameters()],
                   limitation='Operational stochastic-training plateau, not proof of a global optimum. Fixed-panel half-subset metrics differ from full-ensemble formal diagnostics.')
    save(completed, 'checkpoint_last.pt')
    torch.save(qcr_state_dict(model), out/'qcr_model.pt')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('TRAINING_DONE', json.dumps({k: summary[k] for k in
          ('actual_extra_R_updates','reason','elapsed_seconds','frozen_max_parameter_difference')}, indent=2), flush=True)
    return summary
