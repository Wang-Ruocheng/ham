"""固定保存编码上的配对 R-only 局部响应消融；不创建或调用 Q/C。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from recurrent_rg.config import RunConfig
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss
from recurrent_rg.runner import file_fingerprint


NORMALIZATION_FLOOR = 1e-12
ARM_NAMES = ('control', 'response')


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)),
                                        sort_keys=True).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state, *, residual=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith('theory_rg.') == residual:
            digest.update(name.encode()); digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


class SavedResidualR(nn.Module):
    """The source residual Tanh MLP only; Q/C are never instantiated."""
    def __init__(self, state):
        super().__init__()
        r = {name.removeprefix('theory_rg.'): value for name, value in state.items()
             if name.startswith('theory_rg.')}
        required = {f'{layer}.{kind}' for layer in (0, 2, 4) for kind in ('weight', 'bias')}
        if set(r) != required or any(v.dtype != torch.float64 or not torch.isfinite(v).all()
                                     for v in r.values()):
            raise ValueError('Require finite float64 source residual R weights')
        hidden = r['0.weight'].shape[0]
        with torch.random.fork_rng(devices=[]):
            self.theory_rg = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).double()
        self.theory_rg.load_state_dict(r, strict=True)

    def advance_theory(self, value):
        return value + self.theory_rg(value)


def local_derivative_matrix(knots, neighbors=7):
    knots = np.asarray(knots, dtype=np.float64)
    if (knots.ndim != 1 or neighbors < 4 or neighbors > len(knots)
            or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0)):
        raise ValueError('Need finite increasing K knots and at least four local neighbors')
    matrix = np.zeros((len(knots), len(knots)), dtype=np.float64)
    selected, conditions, radii = [], [], []
    for row, center in enumerate(knots):
        ids = np.lexsort((np.arange(len(knots)), np.abs(knots - center)))[:neighbors]
        radius = np.max(np.abs(knots[ids] - center))
        z = (knots[ids] - center) / radius
        design = np.stack([z**power for power in range(4)], axis=1)
        if np.linalg.matrix_rank(design) != 4:
            raise ValueError('Local cubic derivative design is singular')
        matrix[row, ids] = np.linalg.pinv(design)[1] / radius
        selected.append(ids.tolist()); conditions.append(float(np.linalg.cond(design)))
        radii.append(float(radius))
    boundary = np.arange(len(knots)) < neighbors // 2
    boundary |= np.arange(len(knots)) >= len(knots) - neighbors // 2
    return torch.from_numpy(matrix), torch.from_numpy(boundary), dict(
        neighbors=neighbors, degree=3, neighbor_indices=selected, condition_numbers=conditions,
        radii=radii, boundary_mask=boundary.tolist(),
        rule='每节点按 |K-K_i| 选最近邻；等距按原排序；缩放后无权三次伪逆的一阶系数；端点单侧窗口')


@dataclass
class GroupedEncodings:
    direct: torch.Tensor
    labels: torch.Tensor
    knots: torch.Tensor
    groups: torch.Tensor
    counts: torch.Tensor
    D: torch.Tensor
    boundary_mask: torch.Tensor
    target_derivative: torch.Tensor
    diagnostics: dict

    def mean_by_K(self, values):
        total = values.new_zeros((len(self.knots), *values.shape[1:]))
        total.index_add_(0, self.groups, values)
        return total / self.counts.reshape(-1, *([1] * (values.ndim - 1)))


def prepare_response_data(direct, labels, neighbors=7):
    direct = torch.as_tensor(direct, dtype=torch.float64, device='cpu').detach().clone()
    labels = torch.as_tensor(labels, dtype=torch.float64, device='cpu').detach().clone()
    if (direct.ndim != 3 or tuple(direct.shape[1:]) != (3, 1)
            or labels.ndim not in (1, 2) or labels.numel() != len(direct)
            or not torch.isfinite(direct).all() or not torch.isfinite(labels).all()):
        raise ValueError('Require finite scalar three-scale encodings and one K label per ensemble')
    labels = labels.reshape(-1)
    knots, groups, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    if not torch.all(counts == 12):
        raise ValueError('Each whole K knot must contain exactly 12 saved independent ensembles')
    D, boundary, diagnostics = local_derivative_matrix(knots.numpy(), neighbors)
    data = GroupedEncodings(direct, labels, knots, groups, counts, D, boundary,
                            torch.empty(0), diagnostics)
    data.target_derivative = D @ data.mean_by_K(direct[:, 1:, 0])
    return data


def closure_terms(model, direct, cfg=None):
    reduction = cfg.loss.coordinate_reduction if isinstance(cfg, RunConfig) else 'sum'
    one = latent_squared_error(model.advance_theory(direct[:, :-1]), direct[:, 1:], reduction)
    rollout, _ = multistep_closure_loss(model, direct, reduction)
    return dict(one_step_mse=one, rollout_mse=rollout, closure=5 * one + 5 * rollout)


def response_terms(model, prepared, normalization):
    normalization = torch.as_tensor(normalization, dtype=torch.float64)
    if tuple(normalization.shape) != (2,) or not torch.isfinite(normalization).all() or torch.any(normalization <= 0):
        raise ValueError('Two finite positive TRAIN-only derivative normalizers required')
    predicted = model.advance_theory(prepared.direct[:, :-1])[:, :, 0]
    derivative = prepared.D @ prepared.mean_by_K(predicted)
    squared = (derivative - prepared.target_derivative).square()
    result = dict(response_mse=squared.mean(), response_normalized=(squared / normalization).mean(),
        response_mse_by_scale=squared.mean(0), response_normalized_by_scale=(squared / normalization).mean(0))
    for name, mask in [('boundary', prepared.boundary_mask), ('interior', ~prepared.boundary_mask)]:
        result[f'response_{name}_mse'] = squared[mask].mean()
        result[f'response_{name}_normalized'] = (squared[mask] / normalization).mean()
    return result


def metrics(model, prepared, normalization, alpha=1., cfg=None):
    with torch.no_grad():
        terms = {**closure_terms(model, prepared.direct, cfg),
                 **response_terms(model, prepared, normalization)}
        terms['total'] = terms['closure'] + alpha * terms['response_normalized']
    result = {key: value.tolist() for key, value in terms.items()}
    if any(not torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError('Nonfinite R-only diagnostic')
    return result


def load_source(source_run, source_report, checkpoint, neighbors=7):
    source_run, source_report = Path(source_run).resolve(), Path(source_report).resolve()
    checkpoint_path = (source_run / f'training_step_{checkpoint}.pt' if isinstance(checkpoint, int)
                       else source_run / Path(checkpoint)).resolve()
    encodings = source_report / 'encodings.npz'
    receipt_path, response_path = source_report.parent / 'checkpoint_summary.json', source_report.parent / 'local_response.json'
    for path in (checkpoint_path, encodings, receipt_path, response_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Require a regular saved source artifact: {path}')
    hashes = {str(path): file_fingerprint(path) for path in (checkpoint_path, encodings, receipt_path, response_path)}
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = RunConfig.from_dict(payload['metadata']['config'])
    if (cfg.model.dimension != 2 or cfg.model.theory_dim != 1 or cfg.model.block_factor != 2
            or cfg.training.max_rg_steps != 2 or tuple(cfg.data.coupling_names) != ('K',)
            or cfg.loss.closure_metric != 'euclidean' or cfg.loss.weights.commutator != 5.
            or cfg.loss.rollout_weight != 5. or not payload['metadata'].get('independent_c')):
        raise ValueError('Require the scalar b2 depth-two source with original Euclidean closure')
    receipt, response = json.loads(receipt_path.read_text()), json.loads(response_path.read_text())
    if (receipt.get('actual_epochs') != payload['epoch']
            or receipt.get('checkpoint_sha256') != hashes[str(checkpoint_path)]
            or response.get('source_sha256') != dict(checkpoint=hashes[str(checkpoint_path)],
                                                     encodings=hashes[str(encodings)])):
        raise ValueError('Saved encodings/report provenance does not match the source checkpoint')
    with np.load(encodings, allow_pickle=False) as arrays:
        train = prepare_response_data(arrays['train'], arrays['train_labels'], neighbors)
        test = prepare_response_data(arrays['test'], arrays['test_labels'], neighbors)
    if set(train.knots.tolist()) & set(test.knots.tolist()):
        raise ValueError('TRAIN/TEST must contain disjoint whole K knots')
    raw = train.target_derivative.square().mean(0)
    normalization = raw.clamp_min(NORMALIZATION_FLOOR).detach()
    provenance = dict(source_run=str(source_run), source_report=str(source_report),
        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=hashes[str(checkpoint_path)],
        source_epoch=payload['epoch'], encodings_sha256=hashes[str(encodings)], source_sha256=hashes,
        source_data_sha256=payload['metadata'].get('data_sha256'),
        frozen_QC_sha256=state_sha256(payload['model']), initial_R_sha256=state_sha256(payload['model'], residual=True),
        train_fingerprint=dict(encodings=tensor_sha256(train.direct), K=tensor_sha256(train.labels)),
        test_fingerprint=dict(encodings=tensor_sha256(test.direct), K=tensor_sha256(test.labels)),
        normalization=normalization.tolist(), normalization_raw=raw.tolist(),
        normalization_floor=NORMALIZATION_FLOOR, normalization_floor_active=(raw < NORMALIZATION_FLOOR).tolist(),
        derivative_diagnostics=dict(train=train.diagnostics, test=test.diagnostics))
    return payload, cfg, train, test, normalization, provenance


def run(source_run, output, *, source_report, checkpoint=9000, updates=6000, alpha=1.,
        learning_rate=.001, threads=1, neighbors=7, smoke=False, dry_run=False):
    output = Path(output).resolve()
    if (type(updates) is not int or updates < 1 or threads != 1 or not math.isfinite(alpha)
            or alpha <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0 or neighbors < 7):
        raise ValueError('Need positive fixed updates/alpha/LR, one CPU thread and at least seven neighbors')
    if not dry_run and (output.exists() or output.is_symlink()):
        raise FileExistsError(f'Refusing overwrite or implicit resume: {output}')
    for source in (Path(source_run).resolve(), Path(source_report).resolve()):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('Owned output must be separate from all historical sources')
    payload, cfg, train, test, normalization, provenance = load_source(source_run, source_report, checkpoint, neighbors)
    updates = 3 if smoke else updates
    snapshots = sorted({0, updates, *([1] if smoke else [step for step in (1000, 3000, 6000) if step <= updates])})
    plan = dict(**provenance, updates=updates, alpha=alpha, learning_rate=learning_rate, threads=threads,
        neighbors=neighbors, snapshots=snapshots, smoke=smoke, arms=list(ARM_NAMES),
        protocol=dict(R_only_ablation=True, exact_resume=False, Q_C_inference=False,
            physical_data_loaded=False, MI_calculation=False, device='cpu', precision='torch.float64',
            optimizer='两臂相同源 R 初值、各自 fresh Adam；固定学习率、full-batch、相同固定预算；无早停',
            closure='5一步MSE：所有系综及(g0→g1,g1→g2)等权；5 rollout：仅g0起点的R(g0)→g1、R²(g0)→g2两horizon等权，直接复用原函数',
            response='D·每K系综平均R(g_k) 对 D·每K平均g_(k+1)；每K及每尺度等权，除以TRAIN目标导数均方，floor1e-12；α预先固定',
            target_difference='保存的Q(full128)轨迹；原在线训练用两个half-subset Q平均。此为冻结已存轨迹的新R-only拟合，不是原在线QCR精确复现',
            K_use='raw K仅作局部响应坐标；不作g回归目标；TEST不参与训练归一化、邻域超参数、预算或选择',
            selection='固定0/1000/3000/6000节点及最终预算；不按指数、Kc或理论乘子调参或选根'))
    if dry_run:
        return plan
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'manifest.json', plan)
    torch.save(dict(normalization=normalization, train_D=train.D, test_D=test.D,
        train_K=train.knots, test_K=test.knots, train_target_derivative=train.target_derivative,
        test_target_derivative=test.target_derivative), output / 'response_preprocessing.pt')
    started = time.perf_counter()
    job = dict(status='running', pid=os.getpid(), current=None, completed_arms=[], plan=plan)
    write_json(output / 'training_job.json', job)
    try:
        base = SavedResidualR(payload['model'])
        active_config = deepcopy(cfg.to_dict())
        active_config['training'].update(train_modules=['R'], epochs=updates,
            learning_rate=learning_rate, lr_schedule='constant', lr_milestones=[], lr_cosine_updates=0)
        for arm in ARM_NAMES:
            arm_dir = output / arm; arm_dir.mkdir()
            model = deepcopy(base)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            arm_alpha = 0. if arm == 'control' else alpha
            metadata = deepcopy(payload['metadata'])
            metadata['config'] = active_config
            metadata.update(training_driver='ising_response_loss_ablation', arm=arm, alpha=arm_alpha,
                R_only_ablation=True, exact_resume=False, source_config=cfg.to_dict(),
                source_q_supervision=payload['metadata'].get('q_supervision'), q_supervision=None,
                optimized_modules=['R'], frozen_modules=['Q', 'C'], labels_used_in_training=bool(arm_alpha),
                K_label_use=plan['protocol']['K_use'], source_epoch=payload['epoch'], extra_R_updates=0,
                source_checkpoint_sha256=provenance['source_checkpoint_sha256'],
                encodings_sha256=provenance['encodings_sha256'], frozen_QC_sha256=provenance['frozen_QC_sha256'],
                initial_R_sha256=provenance['initial_R_sha256'], normalization=normalization.tolist(),
                protocol=plan['protocol'], learning_rate=learning_rate, requested_extra_R_updates=updates,
                source_checkpoint=provenance['source_checkpoint'], execution_device='cpu', precision='torch.float64')
            metadata.update(source_learning_rate_policy=payload['metadata'].get('learning_rate_policy'),
                learning_rate_policy=dict(schedule='constant', base_learning_rate=learning_rate, clock='extra_R_updates'),
                gradient_QC='Q/C frozen; their modules are not instantiated',
                gradient_R='5 one-step MSE + 5 g0-origin horizon-average rollout MSE + alpha * normalized local response')
            write_json(arm_dir / 'manifest.json', metadata)
            history = []
            arm_started = time.perf_counter()
            optimizer_seconds = 0.

            def save(step, name):
                state = {**payload['model'], **{f'theory_rg.{key}': value.detach().clone()
                         for key, value in model.theory_rg.state_dict().items()}}
                if state_sha256(state) != provenance['frozen_QC_sha256']:
                    raise AssertionError('Frozen source Q/C tensors changed')
                saved = {**payload, 'model': state, 'metadata': {**metadata, 'extra_R_updates': step},
                    'history': deepcopy(history), 'optimizer': optimizer.state_dict(),
                    'R_only_ablation': True, 'exact_resume': False, 'extra_R_updates': step}
                # Source epoch is preserved; additional R-only updates are a separate clock.
                torch.save(saved, arm_dir / name)

            for step in range(updates + 1):
                if step:
                    update_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    terms = closure_terms(model, train.direct, cfg)
                    loss = terms['closure']
                    if arm_alpha:
                        loss = loss + arm_alpha * response_terms(model, train, normalization)['response_normalized']
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite {arm} loss at R update {step}')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f'Nonfinite/missing {arm} R gradient at update {step}')
                    optimizer.step()
                    optimizer_seconds += time.perf_counter() - update_started
                if step % 100 == 0 or step in snapshots:
                    row = dict(step=step, extra_R_updates=step, source_epoch=payload['epoch'], alpha=arm_alpha,
                        **{f'{split}_{key}': value for split, data in [('train', train), ('test', test)]
                           for key, value in metrics(model, data, normalization, arm_alpha, cfg).items()})
                    history.append(row)
                    write_json(arm_dir / 'history_progress.json', history)
                    job['current'] = dict(arm=arm, extra_R_updates=step)
                    write_json(output / 'training_job.json', job)
                    print(json.dumps(dict(arm=arm, step=step, train_total=row['train_total'],
                                          test_total=row['test_total'])), flush=True)
                if step in snapshots:
                    save(step, f'training_step_{step}.pt')
            save(updates, 'checkpoint_last.pt')
            last = torch.load(arm_dir / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
            torch.save(last['model'], arm_dir / 'qcr_model.pt')
            metadata['extra_R_updates'] = updates
            write_json(arm_dir / 'manifest.json', metadata)
            write_json(arm_dir / 'summary.json', dict(config=active_config, contract=metadata,
                source_epoch=payload['epoch'], actual_epochs=payload['epoch'], actual_extra_R_updates=updates,
                requested_extra_R_updates=updates, history=history, frozen_QC_sha256=provenance['frozen_QC_sha256'],
                timing=dict(optimizer_seconds=optimizer_seconds, seconds_per_update=optimizer_seconds / updates,
                            arm_seconds=time.perf_counter() - arm_started)))
            job['completed_arms'].append(arm)
            write_json(output / 'training_job.json', job)
        for path, digest in provenance['source_sha256'].items():
            if file_fingerprint(path) != digest:
                raise ValueError('Historical source artifact changed during R-only fitting')
        job.update(status='complete', current=None, actual_extra_R_updates=updates,
                   elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'training_job.json', job)
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}',
                   elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'training_job.json', job)
        raise
    return job


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--source-report', type=Path, required=True)
    p.add_argument('--checkpoint', type=int, default=9000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--neighbors', type=int, default=7)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    result = run(**vars(args))
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
