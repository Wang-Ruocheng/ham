"""Build a new label-free training view and collect only new-run evaluation.

Only the explicitly named completed finite-perturbation raw data are read.
Historical checkpoints, the old 9GB dataset, and old MI caches are not opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


PATHS = ((0, 1), (1, 2), (0, 2))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def state_sha(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_config(cache, *, epochs=6000):
    return dict(schema_version=5,
        data=dict(cache=str(Path(cache).resolve()), physical_model='二维零场Ising及有限六偶耦合扰动；新物理样本的固定训练视图',
                  coupling_names=['K0', 'delta_nn', 'delta_diag', 'delta_plaquette', 'delta_dist2', 'delta_rectangle', 'delta_dist3'],
                  spin_flip_augment=False, spin_flip_symmetric=True),
        model=dict(dimension=2, theory_dim=1, hidden=96, width=3, block_factor=3,
                   kernel_layout='nonoverlapping', fixed_kernel=False, binary_lookup=True,
                   kernel_initialization='random', kernel_sampling='bernoulli', kernel_temperature=1.,
                   affine_reconstruction=True, reconstruction_hidden=0, theory_head='nonlinear'),
        loss=dict(weights=dict(anchor=2., coarse_consistency=5., separation=10., commutator=5., kernel_noncollapse=2.),
                  rollout_weight=5., kernel_noncollapse_mode='variance_floor_ensemble',
                  kernel_variance_ratio=1., kernel_sign_weight=1.),
        training=dict(epochs=epochs, max_rg_steps=2, system_batch=16, sample_batch=64,
                      learning_rate=.001, seed=91, threads=1, checkpoint_every=1000,
                      log_every=200, gradient_mode='qc_r_separate', train_modules=['Q', 'C', 'R']),
        evaluation=dict(max_rg_steps=2, samples=64, systems=32))


def prepare(source, root, *, samples=64, epochs=6000):
    source, root = Path(source).resolve(), Path(root).resolve()
    if samples != 64:
        raise ValueError('This fixed experiment uses the first64 of the existing128 configurations')
    job = json.loads((source / 'job.json').read_text())
    if job.get('status') not in ('complete', 'scientific_stop') or job.get('workflow_complete') is not True:
        raise ValueError('Source experiment must already have finished')
    data_dir = root / 'data'; data_dir.mkdir(parents=True, exist_ok=False)
    receipts, warnings, xs, ys, gs, ids = [], [], {}, {}, {}, {}
    for split, conditions, reps in [('train', range(14), range(8)), ('test', range(14, 41), range(12))]:
        xx, yy, gg, chains = [], [], [], []
        for ci in conditions:
            directory = source / 'data' / 'conditions' / f'c{ci:03d}'
            status = json.loads((directory / 'status.json').read_text())
            sidecar = json.loads((directory / 'sidecar.json').read_text())
            if status.get('status') != 'complete' or not status.get('workflow_complete') or sidecar.get('smoke'):
                raise ValueError(f'Incomplete/nonformal physical condition {ci}')
            data_path = directory / 'data.pt'
            actual = sha(data_path)
            if actual != status['data_sha256'] or actual != sidecar['data_sha256']:
                raise ValueError(f'New physical data fingerprint mismatch {ci}')
            payload = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)
            spins = payload['configurations']
            if spins.shape != (12, 128, 108, 108) or spins.dtype != torch.int8 or payload['condition_index'] != ci:
                raise ValueError('Unexpected new physical data geometry or identity')
            point = sidecar['condition']
            if point['size'] != 108 or point['role'] != ('train_domain' if split == 'train' else 'heldout_domain'):
                raise ValueError('Unexpected physical-domain assignment')
            delta = sidecar.get('delta6', sidecar['provenance']['delta6'])
            labels = [point['K0']] + list(delta)
            selected = spins[list(reps), :samples].clone().unsqueeze(-1)
            if not torch.all((selected == 1) | (selected == -1)):
                raise ValueError('Nonbinary physical configurations')
            xx.append(selected)
            yy.extend([labels] * len(reps)); gg.extend([ci] * len(reps))
            chain_ids = payload['chain_seeds'][list(reps)].tolist()
            chains.extend(chain_ids)
            row = dict(split=split, condition_index=ci, physical_replicas=list(reps),
                       configuration_indices=[0, samples - 1], data_sha256=actual,
                       sidecar_sha256=sha(directory / 'sidecar.json'), chain_ids=chain_ids)
            receipts.append(row)
            for warning in sidecar['quality']['warnings']:
                if warning.get('kind') != 'production_autocorrelation' or warning.get('replica') in reps:
                    warnings.append(dict(condition_index=ci, split=split, **warning))
            print(f'新数据固定视图 {split} c{ci:03d} {len(reps)}链×{samples}构型', flush=True)
        xs[split] = torch.cat(xx); ys[split] = torch.tensor(yy, dtype=torch.float64)
        gs[split] = torch.tensor(gg, dtype=torch.long); ids[split] = chains
    if set(ids['train']) & set(ids['test']) or len(set(ids['train'] + ids['test'])) != 436:
        raise ValueError('Physical chain identity collision')
    cache = data_dir / 'ensembles.pt'
    torch.save(dict(schema_version=1, **{s:dict(configurations=xs[s], couplings=ys[s], group_ids=gs[s])
                                       for s in ('train', 'test')}), cache)
    fingerprint = sha(cache)
    pairs = dict(schema_version=1, data_sha256=fingerprint)
    for s in ('train', 'test'):
        n = len(xs[s])
        pairs[s] = dict(condition_ids=gs[s].tolist(), chain_ids=ids[s],
                        pairs=[[i, i + 1] for i in range(0, n, 2)])
    pairs['provenance'] = dict(source=str(source), labels_used_for_pairing=False,
        rule='Same source condition, fixed disjoint adjacent independent physical replicas; no replica pooling')
    save_json(root / 'pairs.json', pairs)
    save_json(root / 'config.json', make_config(cache, epochs=epochs))
    receipt = dict(schema_version=1, source=str(source), source_job_sha256=sha(source / 'job.json'),
        data_sha256=fingerprint, pair_sha256=sha(root / 'pairs.json'),
        train_shape=list(xs['train'].shape), test_shape=list(xs['test'].shape),
        new_physical_sampling=0, old_9GB_dataset_opened=False, source_files=receipts,
        warnings=warnings, warning_count=len(warnings),
        sampling_qualification='Existing finite-chain diagnostics cover the original128 draws; this fixed first64 view does not prove equilibrium or independence.',
        source_representation_not_reused='Raw spins only. New random trainable b3 C and fresh Q/R; not a historical b2 reproduction.',
        labels='K0/delta are retained only for generation provenance and post-hoc evaluation; never predictor inputs or training targets.',
        excluded='Original TRAIN rep8..11 and all independentL54 data remain outside this new training/evaluation view.')
    save_json(root / 'dataset_receipt.json', receipt)
    return receipt


def calibrate(root, device):
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import build_model, isolated_rng
    from recurrent_rg.data.ensembles import load_ensembles
    from recurrent_rg.future_physics import binary2d_future_observables, fit_future_normalizer, BINARY2D_OBSERVABLE_NAMES
    from recurrent_rg.execution import configure_cuda_execution
    root = Path(root); cfg = RunConfig.load(root / 'config.json')
    torch.set_num_threads(1); torch.manual_seed(cfg.training.seed)
    device = torch.device(device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    model = build_model(cfg.model, 2).to(device)
    model.detach_kernel_representation = True
    configure_cuda_execution(model, device, independent_c=True)
    train, _ = load_ensembles(cfg.data.cache)
    draws, rows = 5, []
    with isolated_rng(device, 2026101719), torch.no_grad():
        for start in range(0, len(train.configurations), 8):
            batch = train.configurations[start:start + 8].to(device=device, dtype=torch.float64)
            measured = []
            for _ in range(draws):
                levels = model.coarse_grain(batch)
                future = {t:binary2d_future_observables(levels[t]) for t in (1, 2)}
                measured.append(torch.stack([future[t] for _, t in PATHS], dim=1))
            rows.append(torch.stack(measured).mean(0).cpu())
    mean, std = fit_future_normalizer(torch.cat(rows), split='TRAIN')
    result = dict(schema_version=1, data_sha256=sha(cfg.data.cache), pairs_sha256=sha(root / 'pairs.json'),
        fitting_split='train', target_mean=mean.tolist(), target_std=std.clamp_min(.01).tolist(),
        target_std_unfloored=std.tolist(), std_floor=.01, std_floored_mask=(std < .01).tolist(),
        paths=[list(p) for p in PATHS], observable_names=list(BINARY2D_OBSERVABLE_NAMES),
        fitting_ensembles=len(train.configurations), samples=64, C_draws=draws,
        calibration_seed=2026101719, initial_C_sha256=state_sha(model.rg_kernel),
        model_config=cfg.to_dict()['model'], scales=[108, 36, 12],
        normalization='Frozen TRAIN moments at initially random C, floor .01 physical units. C remains trainable; no online metric refresh, no TEST fitting.',
        per_update_C_draws=1, targets='Independent physical chain; reuse existing per-update C trajectories',
        model_pretraining_updates=0, QCR_updates=0)
    target = root / 'normalizer.json'
    if target.exists(): raise FileExistsError(target)
    save_json(target, result)
    print(json.dumps(dict(status='calibrated', initial_C_sha256=result['initial_C_sha256'], target_std=result['target_std'])), flush=True)
    return result


def collect(root, run, device):
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import load_run, isolated_rng
    from recurrent_rg.data.ensembles import load_ensembles
    from recurrent_rg.execution import configure_cuda_execution
    from scripts.ising_qcr_evaluation import collect_saved_encodings
    root, run = Path(root), Path(run)
    out = run / 'report'; out.mkdir(exist_ok=False)
    loaded = load_run(run / 'checkpoint_last.pt')
    cfg = RunConfig.from_dict(loaded.config); torch.set_num_threads(1)
    device = torch.device(device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    model = loaded.model.to(device); model.detach_kernel_representation = True
    configure_cuda_execution(model, device, independent_c=True)
    before = state_sha(model)
    train, test = load_ensembles(cfg.data.cache)
    if sha(cfg.data.cache) != loaded.payload['metadata']['data_sha256']:
        raise ValueError('New training cache changed')
    draw_count = 5
    with isolated_rng(device, 2026101723), torch.no_grad():
        arrays, encoding = collect_saved_encodings(model, [('train', train), ('test', test)], device,
                                                   draw_count=draw_count)
    if state_sha(model) != before: raise RuntimeError('Evaluation changed model parameters')
    np.savez_compressed(out / 'encodings.npz', **arrays)
    save_json(out / 'metrics.json', dict(schema_version=2, status='complete', actual_steps=loaded.payload['epoch'],
        scales=[108, 36, 12], C_draws=draw_count, samples=encoding['samples'], config=cfg.to_dict(),
        encoding_estimator=encoding['estimator'], encoding_protocol=encoding,
        encoding_matches_training=True,
        model_sha256_before=before, model_sha256_after=state_sha(model),
        C_sha256=state_sha(model.rg_kernel), Q_sha256=state_sha(model.theory_encoder),
        encodings_sha256=sha(out / 'encodings.npz'), thermal_nu=None,
        qualification='Fixed training seed and finite physical samples; no new root search or thermal-mode certification.'))
    print(json.dumps(dict(status='collected', run=str(run), actual_steps=loaded.payload['epoch'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'calibrate', 'collect'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', type=Path); parser.add_argument('--run', type=Path)
    parser.add_argument('--device', default='cuda:2'); parser.add_argument('--epochs', type=int, default=6000)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.source, args.root, epochs=args.epochs)
    elif args.stage == 'calibrate': calibrate(args.root, args.device)
    else: collect(args.root, args.run, args.device)


if __name__ == '__main__': main()
