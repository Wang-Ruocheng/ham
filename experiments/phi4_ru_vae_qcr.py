"""Fresh, label-free r/u ensemble-VAE QCR with a mandatory terminal R fit."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.phi4_qcr import load_dataset
from experiments.phi4_hopping_qcr import sha256_file, write_json, _device_receipt
from recurrent_rg.checkpoints import isolated_rng
from recurrent_rg.config import LossConfig
from recurrent_rg.phi4_ensemble_vae import (Phi4EnsembleVAEQCR, calibrate,
    encode_levels, objectives, independent_step)
from recurrent_rg.terminal_r_fit import fingerprint, collect_train_targets, fit_r
from recurrent_rg.phi4_ru_flow import bounded_flow, full_map_and_jacobian
from scripts.report_dual_ising_current import predictions, closure_metrics

RECIPE = 'phi4_ru_b3_g2_ensemble_vae_terminal_v1'


def validate(config):
    if config['recipe'] != RECIPE or config['schema_version'] != 5:
        raise ValueError('Explicit schema-5 continuous VAE recipe required')
    required = dict(theory_dim=2, hidden=96, block_size=3, depth=2,
                    q_encoder='continuous_ensemble_vae', feature_bins=625,
                    kl_weight=.01, kl_warmup_steps=600, patch_chunk_size=65536)
    if config['model'] != required:
        raise ValueError('Unexpected model policy')
    if config['loss'] != dict(closure_objective='all_start_anchor', closure_scale_weights=None,
            coordinate_reduction='sum', commutator=5, rollout_weight=5,
            c_information=-2, c_variance=2, c_uniform=1,
            q_objective='mean_over_scales(cross_half_CE+warmup_KL)'):
        raise ValueError('Unexpected loss/gradient policy')
    t = config['training']
    for key in ('epochs', 'system_batch', 'sample_batch', 'terminal_r_max_steps',
                'terminal_r_cache_repeats', 'checkpoint_every', 'log_every', 'information_pairs'):
        if type(t[key]) is not int or t[key] < 1: raise ValueError('Invalid '+key)
    if (not t['terminal_r_fit'] or t['threads'] != 1 or t['learning_rate'] != .001
            or t['sample_batch'] % 2 or t['information_pairs'] < 2):
        raise ValueError('Fixed independent-gradient and terminal-stage policy required')
    if not config.get('execution_check_only', False) and any(t[k] != v for k, v in
            dict(epochs=3000, system_batch=8, sample_batch=64, terminal_r_max_steps=10000,
                 terminal_r_cache_repeats=8, information_pairs=128, seed=91).items()):
        raise ValueError('Formal run must retain the declared complete recipe')
    return config


def clocks(optimizers):
    return {n: [int(opt.state.get(p, {}).get('step', 0)) for group in opt.param_groups
                for p in group['params']] for n, opt in optimizers.items()}


@torch.no_grad()
def collect_evaluation(model, configurations, samples, check):
    """Four fixed subset/C repeats, with the exact training two-half convention."""
    gen = torch.Generator().manual_seed(91951)
    device = next(model.parameters()).device
    draws, targets, decoded = [], [], []
    with isolated_rng(device, 91952):
        for _ in range(4):
            ds, ts, ps = [], [], []
            for start in range(0, len(configurations), 8):
                check()
                order = torch.randperm(configurations.shape[1], generator=gen)[:samples]
                levels = model.coarse_grain(configurations[start:start+8][:, order].to(device))
                g = encode_levels(model, levels)
                ds.append(g.cpu().numpy())
                ts.append(torch.stack([model.Q.probabilities(v) for v in levels], 1).cpu().numpy())
                probabilities = model.Q.decode_probabilities(g)
                if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
                        or (probabilities.sum(-1)-1).abs().max() > 1e-10):
                    raise FloatingPointError('Evaluation decoder is not normalized')
                ps.append(probabilities.cpu().numpy())
            draws.append(np.concatenate(ds)); targets.append(np.concatenate(ts)); decoded.append(np.concatenate(ps))
    return np.stack(draws), dict(targets=np.stack(targets), decoded=np.stack(decoded))


def run(config_path, output, device='cpu'):
    config = validate(json.loads(Path(config_path).read_text()))
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    write_json(root/'config.json', config)
    t = config['training']; torch.set_num_threads(1)
    started = time.monotonic()
    state = dict(status='loading', workflow_complete=False, joint_updates=0,
                 extra_R_updates=0, error=None, pid=os.getpid())
    write_json(root/'status.json', state)

    def check():
        if time.monotonic()-started > config['active_wall_limit_seconds']-120:
            raise TimeoutError('Active budget reached; preserve checkpoints, no automatic retry')

    def status(**items):
        state.update(items, elapsed_seconds=time.monotonic()-started)
        write_json(root/'status.json', state)

    try:
        data_root = Path(config['data']['root'])
        if sha256_file(data_root/'manifest.json') != config['data']['manifest_sha256']:
            raise ValueError('Pinned physical manifest mismatch')
        dataset = load_dataset(data_root)
        train = dataset.configurations[dataset.train_indices]
        test = dataset.configurations[dataset.test_indices]
        if train.shape[2:] != (36, 36, 1) or t['sample_batch'] > train.shape[1]:
            raise ValueError('Unexpected physical geometry')
        model = Phi4EnsembleVAEQCR(dataset.scale, seed=t['seed']).to(device)
        torch.manual_seed(t['seed']); np.random.seed(t['seed'])
        if str(device).startswith('cuda'):
            torch.cuda.manual_seed_all(t['seed'])
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if not visible.startswith('GPU-') or ',' in visible:
                raise ValueError('GPU must be masked to one admitted physical UUID')
            receipt = _device_receipt(torch.device(device), visible)
        else: receipt = dict(device='cpu')
        calibration = calibrate(model, train, systems=t['system_batch'])
        opts = {n: torch.optim.Adam(getattr(model, n).parameters(), lr=t['learning_rate']) for n in ('Q', 'C', 'R')}
        manifest = dict(recipe=RECIPE, schema_version=5, config=config, device=receipt,
            physical_data_sha256=dataset.manifest_sha256, shards=dataset.receipts,
            physical_shape=list(dataset.configurations.shape), train_chains=len(train), test_chains=len(test),
            sampling_warning_count=dataset.warning_count, sampling_qualification=dataset.qualification,
            calibration=calibration, initialization='fresh random Q/C/R; no kernel selection/pretraining',
            gradient_policy='simultaneous independent Q representation / C own objective / R closure',
            c_adaptation='Gaussian kernel; smoothed buffered full-shell information proxy; variance floor and uniform-amplitude MSE, not literal binary MI/NLL',
            q_adaptation='625 soft continuous plaquette probabilities; cross-half CE, weak KL .01; posterior-mean encoding; finite descriptor, not full field likelihood',
            probability_normalization='shifted logsumexp, explicit autograd; fused CUDA softmax/log-softmax bypassed',
            label_policy='r/u/h read for provenance/evaluation only, never passed to objectives or model',
            C_repeats_are_not_independent_physical_chains=True, thermal_nu=None)
        write_json(root/'manifest.json', manifest)
        batch_gen = torch.Generator().manual_seed(t['seed']+71)
        c_gen = torch.Generator(device=device).manual_seed(t['seed']+72)
        history = []; model.train()

        def save(name, joint_steps, extra_steps=0):
            path = root/name
            temp = path.with_suffix('.partial')
            torch.save(dict(schema_version=5, recipe=RECIPE, config=config, field_scale=float(model.scale),
                state_dict=model.state_dict(), optimizers={n:o.state_dict() for n,o in opts.items()},
                joint_Q_C_updates=joint_steps, total_R_updates=joint_steps+extra_steps,
                batch_rng=batch_gen.get_state(), C_rng=c_gen.get_state()), temp)
            temp.replace(path)

        save('initial.pt', 0)
        status(status='joint_training')
        for step in range(1, t['epochs']+1):
            check(); tick = time.monotonic()
            rows = torch.randint(len(train), (t['system_batch'],), generator=batch_gen)
            cols = torch.randperm(train.shape[1], generator=batch_gen)[:t['sample_batch']]
            terms = independent_step(model, opts, train[rows][:, cols].to(device), step,
                                     c_gen, t['information_pairs'])
            duration = time.monotonic()-tick
            if step == 1 or step % t['log_every'] == 0 or step == t['epochs']:
                # Fixed heldout diagnostic batches, private RNG, never used to
                # stop/select/normalize the training or determine root bounds.
                with isolated_rng(torch.device(device), 91971), torch.no_grad():
                    diagnostic = objectives(model, test[:min(len(test), t['system_batch']), :t['sample_batch']].to(device),
                                            information_pairs=t['information_pairs'])[-1]
                row = dict(step=step, train=terms, test=diagnostic,
                           update_seconds=duration, elapsed_seconds=time.monotonic()-started)
                history.append(row); write_json(root/'history_progress.json', history)
                status(joint_updates=step, current_losses=terms)
                print(json.dumps(dict(step=step, q=terms['q_total'], c=terms['c_total'], r=terms['r_total'],
                                      elapsed_seconds=time.monotonic()-started)), flush=True)
            if step % t['checkpoint_every'] == 0 or step == t['epochs']:
                save(f'training_step_{step}.pt', step)
        if any(any(c != t['epochs'] for c in group) for group in clocks(opts).values()):
            raise AssertionError('Joint optimizer clocks mismatch')
        save('joint.pt', t['epochs'])
        joint_R = copy.deepcopy(model.R)
        before_clocks = clocks(opts)
        status(status='terminal_cache')
        model.eval()
        targets, cache_receipt = collect_train_targets(model, train,
            lambda batch: encode_levels(model, model.coarse_grain(batch)),
            samples=t['sample_batch'], systems=t['system_batch'],
            repeats=t['terminal_r_cache_repeats'], budget_check=check)
        (root/'r_only').mkdir()
        torch.save(targets, root/'r_only/train_cache.pt')
        write_json(root/'r_only/cache_receipt.json', cache_receipt)

        def callback(row, rows):
            status(status='terminal_R_training', extra_R_updates=row['extra_R_updates'],
                   terminal_train_loss=row['train_loss'])
            write_json(root/'r_only/history_progress.json', rows)
            save('terminal_latest.pt', t['epochs'], row['extra_R_updates'])
            print(json.dumps(dict(stage='terminal_R', **row)), flush=True)

        terminal = fit_r(model, model.R, {'Q': model.Q, 'C': model.C}, opts['R'], targets,
            LossConfig(), max_steps=t['terminal_r_max_steps'], callback=callback, budget_check=check)
        if any(clocks(opts)[n] != before_clocks[n] for n in ('Q', 'C')):
            raise AssertionError('Frozen Q/C Adam clocks advanced')
        write_json(root/'r_only/summary.json', terminal)
        save('final.pt', t['epochs'], terminal['actual_extra_R_updates'])
        status(status='evaluation')
        before = {n:fingerprint(getattr(model, n)) for n in ('Q', 'C', 'R')}
        draws, reconstruction = collect_evaluation(model, torch.cat([train, test]), t['sample_batch'], check)
        order = np.concatenate([dataset.train_indices, dataset.test_indices])
        arrays = dict(direct_draws=draws, direct_g=draws.mean(0), labels=dataset.labels[order, :2],
                      splits=dataset.splits[order], cell_ids=dataset.cell_ids[order])
        arrays.update(train_direct=arrays['direct_g'][:len(train)], test_direct=arrays['direct_g'][len(train):])
        metrics = {}
        final_R = model.R
        for name, r in [('joint', joint_R), ('final', final_R)]:
            model.R = r
            pred = predictions(model, draws); arrays[name+'_predictions'] = pred
            metrics[name] = closure_metrics(draws, pred, len(train))
        model.R = final_R
        roots, search = bounded_flow(model, arrays, device, budget_check=check)
        if np.max(np.linalg.norm(arrays['flow_displacement'], axis=1)) <= 1e-10:
            roots = []; search['isolated_roots_unresolved'] = 'Numerically identity on grid'
        points = arrays['direct_g'].reshape(-1, 2)
        low, high = points.min(0), points.max(0); pad = .1*np.maximum(high-low, 1e-6)
        bounds = np.column_stack([np.minimum(low-pad, arrays['flow_bounds'][:, 0]),
                                  np.maximum(high+pad, arrays['flow_bounds'][:, 1])])
        grid = np.stack(np.meshgrid(*[np.linspace(*b, 33) for b in bounds], indexing='xy'), -1).reshape(-1, 2)
        mapped, derivative = full_map_and_jacobian(model, grid, device)
        for key in list(arrays):
            if key.startswith('flow_'): arrays['search_'+key] = arrays.pop(key)
        eig = np.linalg.eigvals(derivative)
        arrays.update(flow_g=grid, flow_R=mapped, flow_DR=derivative, flow_displacement=mapped-grid,
                      flow_bounds=bounds, flow_eigenvalues_real=eig.real, flow_eigenvalues_imag=eig.imag)
        if before != {n:fingerprint(getattr(model, n)) for n in ('Q', 'C', 'R')}:
            raise AssertionError('Evaluation changed model')
        np.savez_compressed(root/'evaluation_arrays.npz', **arrays)
        np.savez_compressed(root/'reconstruction_arrays.npz', **reconstruction)
        write_json(root/'roots.json', dict(roots=roots, search=search))
        write_json(root/'summary.json', dict(status='complete', joint_Q_C_updates=t['epochs'],
            total_R_updates=t['epochs']+terminal['actual_extra_R_updates'], terminal=terminal,
            closure=metrics, physical_chains=[len(train), len(test)], draws_per_chain=4,
            Q_C_unchanged=True, labels_used_for_training=False, thermal_nu=None,
            sampling_warning_count=dataset.warning_count, elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0))
        status(status='complete', workflow_complete=True)
        return 0
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc, TimeoutError) else 'failed',
               error=str(exc), requires_user_action=True, traceback=traceback.format_exc())
        raise
