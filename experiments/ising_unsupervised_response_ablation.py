"""严格无标签、保存轨迹上的 R-only 局部割线消融；不读取任何标签数组。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.ising_response_loss_ablation import (
    SavedResidualR, closure_terms, state_sha256, tensor_sha256, write_json,
)
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint


NORMALIZATION_FLOOR = 1e-12
NEIGHBORS = 8
MIN_DISTANCE = .05
MAX_DISTANCE = .25
ARM_NAMES = ('control', 'response')


def quantiles(value):
    result = np.quantile(np.asarray(value), [0., .1, .5, .9, 1.])
    return dict(zip(('min', 'q10', 'median', 'q90', 'max'), result.tolist()))


@dataclass
class PreparedSecants:
    direct: torch.Tensor
    train_sigma: torch.Tensor
    pairs: tuple[torch.Tensor, ...]
    target_secants: tuple[torch.Tensor, ...]
    diagnostics: list[dict]


def prepare_secant_data(direct, train_sigma=None, *, neighbors=NEIGHBORS,
                        min_distance=MIN_DISTANCE, max_distance=MAX_DISTANCE):
    """Use only saved g geometry; TEST must receive the frozen TRAIN sigma."""
    direct = torch.as_tensor(direct, dtype=torch.float64, device='cpu').detach().clone()
    if (direct.ndim != 3 or tuple(direct.shape[1:]) != (3, 1) or len(direct) < 2
            or not torch.isfinite(direct).all() or type(neighbors) is not int or neighbors < 1
            or not 0 < min_distance < max_distance):
        raise ValueError('Require finite three-scale scalar encodings and valid fixed secant geometry')
    train_sigma = (direct[:, :-1, 0].std(0, unbiased=False) if train_sigma is None
                   else torch.as_tensor(train_sigma, dtype=torch.float64, device='cpu').detach().clone())
    if tuple(train_sigma.shape) != (2,) or not torch.isfinite(train_sigma).all() or torch.any(train_sigma <= 0):
        raise ValueError('Both input scales require finite positive TRAIN population standard deviations')
    pairs, targets, diagnostics = [], [], []
    for scale in range(2):
        x, y = direct[:, scale, 0].numpy(), direct[:, scale + 1, 0].numpy()
        sigma = float(train_sigma[scale]); lower, upper = min_distance * sigma, max_distance * sigma
        selected, outgoing, candidates = set(), [], []
        for i, value in enumerate(x):
            distance = np.abs(x - value)
            admissible = np.flatnonzero((distance >= lower) & (distance <= upper))
            candidates.append(len(admissible))
            ids = admissible[np.lexsort((admissible, distance[admissible]))[:neighbors]]
            outgoing.append(len(ids))
            selected.update((min(i, int(j)), max(i, int(j))) for j in ids)
        if not selected:
            raise ValueError(f'No admissible label-free pairs at scale {scale}; no automatic rule fallback')
        edges = torch.tensor(sorted(selected), dtype=torch.long)
        left, right = edges[:, 0], edges[:, 1]
        dx = direct[right, scale, 0] - direct[left, scale, 0]
        target = (direct[right, scale + 1, 0] - direct[left, scale + 1, 0]) / dx
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite saved target secants')
        distance, contribution = dx.abs().numpy(), target.square().numpy()
        covered = torch.unique(edges)
        isolated = sorted(set(range(len(direct))) - set(covered.tolist()))
        count = max(1, int(np.ceil(len(contribution) * .1)))
        total = float(contribution.sum())
        diagnostics.append(dict(scale=scale, pair_count=len(edges), node_count=len(direct),
            covered_nodes=len(covered), coverage=float(len(covered) / len(direct)),
            isolated_nodes=isolated, isolated_node_count=len(isolated),
            zero_outgoing_nodes=np.flatnonzero(np.asarray(outgoing) == 0).tolist(),
            train_sigma=sigma, lower_distance=lower, upper_distance=upper,
            pair_distance=quantiles(distance), pair_distance_in_train_sigma=quantiles(distance / sigma),
            outgoing_neighbor_count=quantiles(outgoing), candidate_count=quantiles(candidates),
            near_lower_threshold=1.1 * lower, near_lower_pair_fraction=float(np.mean(distance <= 1.1 * lower)),
            target_secant_square=quantiles(contribution), target_secant_square_mean=float(contribution.mean()),
            largest_10pct_target_square_share=float(np.sort(contribution)[-count:].sum() / total) if total else 0.,
            diagnostic_caution='局部割线等于邻域残差一致性；保存 C 抽样噪声可被小距离放大，不是独立热响应'))
        pairs.append(edges); targets.append(target.detach())
    return PreparedSecants(direct, train_sigma.detach(), tuple(pairs), tuple(targets), diagnostics)


def response_terms(model, prepared, normalization):
    normalization = torch.as_tensor(normalization, dtype=torch.float64)
    if tuple(normalization.shape) != (2,) or not torch.isfinite(normalization).all() or torch.any(normalization <= 0):
        raise ValueError('Require two positive finite TRAIN-only secant normalizers')
    predicted = model.advance_theory(prepared.direct[:, :-1])[:, :, 0]
    mse = []
    for scale, (edges, target) in enumerate(zip(prepared.pairs, prepared.target_secants)):
        left, right = edges[:, 0], edges[:, 1]
        dx = prepared.direct[right, scale, 0] - prepared.direct[left, scale, 0]
        secant = (predicted[right, scale] - predicted[left, scale]) / dx
        mse.append((secant - target).square().mean())
    per_scale = torch.stack(mse)
    return dict(response_mse=per_scale.mean(), response_normalized=(per_scale / normalization).mean(),
                response_mse_by_scale=per_scale, response_normalized_by_scale=per_scale / normalization)


def metrics(model, prepared, normalization, alpha=1., cfg=None):
    with torch.no_grad():
        result = {**closure_terms(model, prepared.direct, cfg),
                  **response_terms(model, prepared, normalization)}
        result['total'] = result['closure'] + alpha * result['response_normalized']
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite label-free R diagnostic')
    return {key: value.tolist() for key, value in result.items()}


def strictly_unsupervised(record):
    supervision = record.get('q_supervision')
    return (record.get('labels_used_in_training') is False and
            (supervision is None or isinstance(supervision, dict) and supervision.get('weight', 0.) == 0.))


def json_equal(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def load_source(source_run, source_report=None, checkpoint=12000):
    source_run = Path(source_run).resolve()
    source_report = Path(source_report).resolve() if source_report else source_run / 'full_report'
    name = str(checkpoint)
    checkpoint_path = source_run / (f'training_step_{name}.pt' if name.isdigit() else name)
    checkpoint_path = checkpoint_path.resolve()
    last_path, manifest_path, summary_path = (source_run / name for name in
        ('checkpoint_last.pt', 'manifest.json', 'summary.json'))
    receipt_path, encodings_path = source_run.parent / 'job.json', source_report / 'encodings.npz'
    paths = (checkpoint_path, last_path, manifest_path, summary_path, receipt_path, encodings_path)
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Require a regular saved unsupervised source artifact: {path}')
    hashes = {str(path): file_fingerprint(path) for path in paths}
    manifest, summary, receipt = (json.loads(path.read_text()) for path in
                                  (manifest_path, summary_path, receipt_path))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    last = torch.load(last_path, map_location='cpu', weights_only=True)
    for record in (manifest, payload['metadata'], last['metadata']):
        if not strictly_unsupervised(record):
            raise ValueError('Source must be strictly unsupervised; supervised Q history is not allowed')
    cfg = RunConfig.from_dict(payload['metadata']['config'])
    if (payload.get('epoch') != 12000 or last.get('epoch') != 12000 or summary.get('actual_epochs') != 12000
            or cfg.model.dimension != 2 or cfg.model.theory_dim != 1 or cfg.model.block_factor != 2
            or cfg.training.max_rg_steps != 2 or cfg.loss.closure_metric != 'euclidean'
            or cfg.loss.weights.commutator != 5. or cfg.loss.rollout_weight != 5.
            or not payload['metadata'].get('independent_c')):
        raise ValueError('Require the complete 12000-update unsupervised scalar b2 depth-two source')
    for record in (manifest, summary, last['metadata']):
        if not json_equal(record['config'], payload['metadata']['config']):
            raise ValueError('Unsupervised source configurations differ')
    if payload['model'].keys() != last['model'].keys() or any(
            not torch.equal(value, last['model'][key]) for key, value in payload['model'].items()):
        raise ValueError('Numbered and final source checkpoint model tensors differ')
    if (receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True
            or receipt.get('training', {}).get('actual_epochs') != 12000
            or receipt.get('training', {}).get('manifest_sha256') != hashes[str(manifest_path)]
            or receipt.get('reports', {}).get('files', {}).get('full_report/encodings.npz') != hashes[str(encodings_path)]):
        raise ValueError('Completed unsupervised source receipt or encoding/manifest fingerprints differ')
    # Deliberately never enumerate or index any other NPZ entry, including labels.
    with np.load(encodings_path, allow_pickle=False) as arrays:
        train = prepare_secant_data(arrays['train'])
        test = prepare_secant_data(arrays['test'], train_sigma=train.train_sigma)
    if len(train.direct) != 516 or len(test.direct) != 252:
        raise ValueError('Require all 516 TRAIN and 252 TEST saved ensembles')
    raw = torch.stack([target.square().mean() for target in train.target_secants])
    normalization = raw.clamp_min(NORMALIZATION_FLOOR).detach()
    provenance = dict(source_run=str(source_run), source_report=str(source_report),
        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=hashes[str(checkpoint_path)],
        source_last_checkpoint_sha256=hashes[str(last_path)], source_checkpoint_models_equal=True,
        source_epoch=12000, source_labels_used_in_training=False,
        encodings_sha256=hashes[str(encodings_path)], source_sha256=hashes,
        source_data_sha256=payload['metadata'].get('data_sha256'),
        frozen_QC_sha256=state_sha256(payload['model']), initial_R_sha256=state_sha256(payload['model'], residual=True),
        train_fingerprint=tensor_sha256(train.direct), test_fingerprint=tensor_sha256(test.direct),
        train_sigma=train.train_sigma.tolist(), normalization=normalization.tolist(), normalization_raw=raw.tolist(),
        normalization_floor=NORMALIZATION_FLOOR, normalization_floor_active=(raw < NORMALIZATION_FLOOR).tolist(),
        pair_diagnostics=dict(train=train.diagnostics, test=test.diagnostics),
        K_label_access=False, labels_used_in_training=False, accessed_encoding_arrays=['train', 'test'])
    return payload, cfg, train, test, normalization, provenance


def run(source_run, output, *, source_report=None, checkpoint=12000, updates=6000, alpha=1.,
        learning_rate=.001, threads=1, smoke=False, dry_run=False):
    output = Path(output).resolve()
    if (type(updates) is not int or updates < 1 or threads != 1 or not math.isfinite(alpha)
            or alpha <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Require positive fixed updates, alpha and LR, with one CPU thread')
    if not dry_run and (output.exists() or output.is_symlink()):
        raise FileExistsError(f'Refusing overwrite or implicit resume: {output}')
    sources = [Path(source_run).resolve()]
    if source_report: sources.append(Path(source_report).resolve())
    for source in sources:
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('Owned output must be separate from historical source artifacts')
    payload, cfg, train, test, normalization, provenance = load_source(source_run, source_report, checkpoint)
    updates = 3 if smoke else updates
    snapshots = sorted({0, updates, *([1] if smoke else [s for s in (1000, 3000, 6000) if s <= updates])})
    plan = dict(**provenance, updates=updates, alpha=alpha, learning_rate=learning_rate, threads=threads,
        snapshots=snapshots, smoke=smoke, arms=list(ARM_NAMES), neighbors=NEIGHBORS,
        min_distance=MIN_DISTANCE, max_distance=MAX_DISTANCE,
        adam=dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False, fresh_state=True),
        protocol=dict(R_only_ablation=True, exact_resume=False, K_label_access=False,
            labels_used_in_training=False, Q_C_inference=False, physical_data_loaded=False, MI_calculation=False,
            device='cpu', precision='torch.float64', optimizer='同源R初值；两臂各自fresh Adam .001；full-batch固定同预算；无早停',
            closure='原5一步MSE+5从g0起点两horizon平均rollout MSE；不新增其他起点配对',
            response='仅TRAIN g几何：每点在[.05σx,.25σx]内取最近8点，对称去重；预测与目标局部割线差；每尺度按TRAIN目标割线平方均值归一、两尺度等权；固定α',
            test='TEST独立构造几何pairs；距离阈值和normalizer仅来自TRAIN；不用于定参、归一化、早停或选择',
            interpretation='K-free局部残差一致性割线变体，不是原d/dK损失或独立热响应；单次保存C抽样噪声可被小dx放大',
            target_difference='冻结已存Q(full128)三尺度轨迹，不运行Q/C；是新R-only拟合，不是全QCR联合新训或精确续训',
            selection='预先固定快照和预算；无理论指数、Kc、耦合分组或标签选择'))
    if dry_run: return plan
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plan.json', plan); write_json(output/'manifest.json', plan)
    torch.save(dict(train_pairs=train.pairs, test_pairs=test.pairs, train_sigma=train.train_sigma,
        normalization=normalization, train_target_secants=train.target_secants,
        test_target_secants=test.target_secants), output/'secant_preprocessing.pt')
    started = time.perf_counter()
    job = dict(status='running', pid=os.getpid(), current=None, completed_arms=[], plan=plan)
    write_json(output/'training_job.json', job)
    try:
        base = SavedResidualR(payload['model'])
        active_config = deepcopy(cfg.to_dict())
        active_config['training'].update(train_modules=['R'], epochs=updates,
            learning_rate=learning_rate, lr_schedule='constant', lr_milestones=[], lr_cosine_updates=0)
        for arm in ARM_NAMES:
            arm_dir = output/arm; arm_dir.mkdir()
            model = deepcopy(base)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            arm_alpha = 0. if arm == 'control' else alpha
            metadata = deepcopy(payload['metadata'])
            metadata.update(config=active_config, training_driver='ising_unsupervised_response_ablation',
                source_config=cfg.to_dict(), arm=arm, alpha=arm_alpha, R_only_ablation=True, exact_resume=False,
                optimized_modules=['R'], frozen_modules=['Q', 'C'], labels_used_in_training=False,
                source_labels_used_in_training=False, K_label_access=False, q_supervision=None,
                source_epoch=12000, extra_R_updates=0, source_checkpoint=provenance['source_checkpoint'],
                source_checkpoint_sha256=provenance['source_checkpoint_sha256'], encodings_sha256=provenance['encodings_sha256'],
                frozen_QC_sha256=provenance['frozen_QC_sha256'], initial_R_sha256=provenance['initial_R_sha256'],
                train_sigma=train.train_sigma.tolist(), normalization=normalization.tolist(),
                requested_extra_R_updates=updates, learning_rate=learning_rate, protocol=plan['protocol'],
                execution_device='cpu', precision='torch.float64', source_optimizer_state_restored=False,
                source_rng_restored=False, learning_rate_policy=dict(schedule='constant', clock='extra_R_updates', base_learning_rate=learning_rate),
                gradient_QC='Q/C未创建；只保留源张量，逐hash验证不变',
                gradient_R='原闭合目标 + alpha * TRAIN g几何局部割线一致性；严格不访问K标签数组')
            write_json(arm_dir/'manifest.json', metadata)
            history, optimizer_seconds = [], 0.
            arm_started = time.perf_counter()

            def save(step, name):
                state = {**payload['model'], **{f'theory_rg.{key}': value.detach().clone()
                         for key, value in model.theory_rg.state_dict().items()}}
                if state_sha256(state) != provenance['frozen_QC_sha256']:
                    raise AssertionError('Frozen source Q/C tensors changed')
                saved = {**payload, 'model': state, 'metadata': {**metadata, 'extra_R_updates': step},
                    'history': deepcopy(history), 'optimizer': optimizer.state_dict(),
                    'R_only_ablation': True, 'exact_resume': False, 'extra_R_updates': step}
                torch.save(saved, arm_dir/name)

            for step in range(updates + 1):
                if step:
                    update_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    loss = closure_terms(model, train.direct, cfg)['closure']
                    if arm_alpha:
                        loss = loss + arm_alpha * response_terms(model, train, normalization)['response_normalized']
                    if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite {arm} loss at {step}')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f'Nonfinite/missing {arm} R gradient at {step}')
                    optimizer.step()
                    optimizer_seconds += time.perf_counter() - update_started
                if step % 100 == 0 or step in snapshots:
                    row = dict(step=step, extra_R_updates=step, source_epoch=12000, alpha=arm_alpha,
                        **{f'{split}_{key}': value for split, data in [('train', train), ('test', test)]
                           for key, value in metrics(model, data, normalization, arm_alpha, cfg).items()})
                    history.append(row); write_json(arm_dir/'history_progress.json', history)
                    job['current'] = dict(arm=arm, extra_R_updates=step)
                    write_json(output/'training_job.json', job)
                    print(json.dumps(dict(arm=arm, step=step, train_total=row['train_total'], test_total=row['test_total'])), flush=True)
                if step in snapshots: save(step, f'training_step_{step}.pt')
            save(updates, 'checkpoint_last.pt')
            last = torch.load(arm_dir/'checkpoint_last.pt', map_location='cpu', weights_only=True)
            torch.save(last['model'], arm_dir/'qcr_model.pt')
            metadata['extra_R_updates'] = updates
            write_json(arm_dir/'manifest.json', metadata)
            write_json(arm_dir/'summary.json', dict(config=active_config, contract=metadata,
                source_epoch=12000, actual_epochs=12000, actual_extra_R_updates=updates,
                requested_extra_R_updates=updates, history=history, frozen_QC_sha256=provenance['frozen_QC_sha256'],
                timing=dict(optimizer_seconds=optimizer_seconds, seconds_per_update=optimizer_seconds/updates,
                            arm_seconds=time.perf_counter()-arm_started)))
            job['completed_arms'].append(arm); write_json(output/'training_job.json', job)
        for path, digest in provenance['source_sha256'].items():
            if file_fingerprint(path) != digest: raise ValueError('Historical source changed during R fitting')
        job.update(status='complete', current=None, actual_extra_R_updates=updates,
                   elapsed_seconds=time.perf_counter()-started)
        write_json(output/'training_job.json', job)
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}',
                   elapsed_seconds=time.perf_counter()-started)
        write_json(output/'training_job.json', job)
        raise
    return job


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--source-report', type=Path)
    p.add_argument('--checkpoint', default='12000')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    result = run(**vars(args))
    if args.dry_run: print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
