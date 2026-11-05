"""完全无监督源模型的 R-only 物理 score 响应消融；Q/C 不参与优化。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from experiments.ising_response_loss_ablation import (
    closure_terms, state_sha256, tensor_sha256, write_json,
)
from scripts.diagnose_ising_mcrg import sha256_file


class FrozenResidualR(nn.Module):
    """只创建完整 R=g+MLP；解析保存的 dimg，不创建 Q 或 C。"""
    def __init__(self, state):
        super().__init__()
        weights = {key.removeprefix('theory_rg.'): value for key, value in state.items()
                   if key.startswith('theory_rg.')}
        required = {f'{layer}.{kind}' for layer in (0, 2, 4) for kind in ('weight', 'bias')}
        if set(weights) != required or any(value.dtype != torch.float64
                or not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError('需要有限 float64 三层残差 R')
        hidden, dim = weights['0.weight'].shape
        if dim not in (1, 2, 3) or weights['4.weight'].shape != (dim, hidden):
            raise ValueError('仅支持源 dimg=1/2/3')
        with torch.random.fork_rng(devices=[]):
            self.theory_rg = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, dim)).double()
        self.theory_rg.load_state_dict(weights, strict=True)

    def advance_theory(self, value):
        return value + self.theory_rg(value)

    def response_jvp(self, value, directions):
        """完整 DR 作用于多方向；保留关于 R 参数的导数，包含恒等项。"""
        first, middle, last = self.theory_rg[0], self.theory_rg[2], self.theory_rg[4]
        h1 = torch.tanh(first(value))
        h2 = torch.tanh(middle(h1))
        d1 = (first.weight @ directions) * (1-h1.square())[..., None]
        d2 = (middle.weight @ d1) * (1-h2.square())[..., None]
        return directions + last.weight @ d2


def tensors_for_split(cache, split, device):
    values = [cache[f'{split}_{name}'].detach().to(device=device, dtype=torch.float64)
              for name in ('direct', 'vin', 'vout', 'score_std')]
    direct, vin, vout, std = values
    if (direct.ndim != 4 or direct.shape[1] < 1 or direct.shape[2] != 3
            or vin.ndim != 5 or vin.shape != vout.shape
            or vin.shape[:3] != (*direct.shape[:2], 2) or vin.shape[-2] != direct.shape[-1]
            or vin.shape[-1] != 6 or std.shape != (*vin.shape[:3], 6)
            or any(not torch.isfinite(value).all() for value in values) or torch.any(std < 0)):
        raise ValueError('保存的物理系综、C抽样和配对响应形状不符')
    return values


def standardized_response(vin, vout, score_std):
    inverse = torch.where(score_std > 1e-12, score_std.reciprocal(), torch.zeros_like(score_std))
    return vin * inverse[..., None, :], vout * inverse[..., None, :]


def response_terms(model, direct, vin, vout, score_std, normalization):
    x, y = standardized_response(vin, vout, score_std)
    prediction = model.response_jvp(direct[:, :, :-1], x)
    residual = prediction - y
    # Equal ensemble, draw, operator and scale weights; coordinate reduction SUM.
    by_scale = residual.square().sum(-2).mean((0, 1, 3))
    return dict(response_mse_by_scale=by_scale,
        response_normalized=(by_scale / normalization).mean(),
        response_relative_norm_by_scale=(by_scale / normalization).sqrt())


def evaluate(model, direct, vin, vout, score_std, normalization, response_weight):
    with torch.no_grad():
        closure = closure_terms(model, direct.flatten(0, 1))
        response = response_terms(model, direct, vin, vout, score_std, normalization)
        values = {**closure, **response}
        values['total'] = closure['closure'] + response_weight * response['response_normalized']
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('非有限 R-only 指标')
    return {key: value.detach().cpu().tolist() for key, value in values.items()}


def train(source_run, cache_path, output, *, device='cuda:2', updates=6000,
          learning_rate=.001, smoke=False):
    if Path(output).is_symlink() or Path(cache_path).is_symlink():
        raise ValueError('不接受输出或缓存符号链接')
    source_run, cache_path, output = map(lambda path: Path(path).resolve(),
                                       (source_run, cache_path, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖、重复启动或隐式续训')
    for protected in (source_run, cache_path.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('新训练目录必须与历史源和缓存目录分离')
    if (type(updates) is not int or updates < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError('需要正的预定更新数与学习率')
    torch.set_num_threads(1)
    source_path = source_run/'training_step_12000.pt'
    source_sha = sha256_file(source_path)
    payload = torch.load(source_path, map_location='cpu', weights_only=True)
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    receipt = json.loads((cache_path.parent/'receipt.json').read_text())
    metadata = payload['metadata']
    if (payload.get('epoch') != 12000 or metadata.get('labels_used_in_training') is not False
            or metadata.get('q_supervision') is not None
            or receipt.get('source_checkpoint_sha256') != source_sha
            or receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True
            or receipt.get('smoke') != smoke):
        raise ValueError('需要匹配的完整无监督源权重及专用物理响应缓存')
    if receipt.get('response_cache_sha256') != sha256_file(cache_path):
        raise ValueError('物理响应缓存指纹不匹配')
    frozen_sha = state_sha256(payload['model'])
    if (receipt.get('frozen_QC_sha256_before') != frozen_sha
            or receipt.get('frozen_QC_sha256_after') != frozen_sha
            or receipt.get('labels_accessed') is not False
            or receipt.get('Kc_selection') is not False
            or receipt.get('c_repeats') != (1 if smoke else 5)):
        raise ValueError('缓存冻结权重、无标签来源或C抽样数不符')
    train_data = tensors_for_split(cache, 'train', device)
    test_data = tensors_for_split(cache, 'test', device)
    if not smoke and (len(train_data[0]), len(test_data[0])) != (516, 252):
        raise ValueError('完整K划分应有516 TRAIN及252 TEST独立系综')
    dimension = metadata['config']['model']['theory_dim']
    if (receipt.get('theory_dim') != dimension or any(
            value[0].shape[1:] != ((1 if smoke else 5), 3, dimension)
            for value in (train_data, test_data))):
        raise ValueError('响应缓存维数或抽样轴与源架构不同')
    base = FrozenResidualR(payload['model']).to(device)
    _, target = standardized_response(*train_data[1:])
    raw_normalization = target.square().sum(-2).mean((0, 1, 3)).detach()
    normalization = raw_normalization.clamp_min(1e-12)
    initial = evaluate(base, *train_data, normalization, 0.)
    # Predeclared TRAIN-only initial balancing. Never tune to TEST or ν.
    response_weight = initial['closure'] / max(initial['response_normalized'], 1e-12)
    if not math.isfinite(response_weight) or response_weight <= 0:
        raise ValueError('TRAIN初值不支持正的响应权重，须人工检查而非自动改规则')
    updates = 3 if smoke else updates
    snapshots = sorted({0, updates, *[step for step in (1000, 3000, 6000) if step <= updates]})
    plan = dict(source_run=str(source_run), source_report=str(source_run.parent/'report'),
        cache_path=str(cache_path), source_checkpoint_sha256=source_sha,
        response_cache_sha256=sha256_file(cache_path), source_epoch=12000,
        updates=updates, snapshots=snapshots, learning_rate=learning_rate,
        normalization=normalization.cpu().tolist(), response_weight=response_weight,
        raw_normalization=raw_normalization.cpu().tolist(), normalization_floor=1e-12,
        normalization_floor_active=(raw_normalization < 1e-12).cpu().tolist(),
        response_balance_floor=1e-12,
        response_balance_floor_active=initial['response_normalized'] < 1e-12,
        frozen_QC_sha256=frozen_sha, initial_R_sha256=state_sha256(payload['model'], residual=True),
        train_only_calibration=initial, device=device, smoke=smoke,
        labels_used_in_training=False, Kc_or_exponent_target=False,
        optimized_modules=['R'], frozen_modules=['Q','C'], exact_resume=False,
        protocol='同源R、fresh Adam、固定预算；5一步+5微观起点多步闭合；响应权重仅按TRAIN源模型初始两项loss等值校准并冻结。',
        response='六个偶算符的同尺度配对经验指数倾斜响应，按各系综score标准差归一；无K路径割线、无伪逆目标、无理论倍率标签。',
        independent_physical_ensembles=True, C_draws_not_independent_physical_replicas=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plan.json', plan)
    job = dict(status='running', current=None, completed_arms=[], plan=plan)
    write_json(output/'training_job.json', job)
    try:
        for arm in ('control', 'response'):
            folder = output/arm; folder.mkdir()
            model = deepcopy(base)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            weight = 0. if arm == 'control' else response_weight
            history = []
            arm_metadata = deepcopy(metadata)
            arm_metadata.update(training_driver='ising_physical_response_ablation',
                source_epoch=12000, requested_extra_R_updates=updates, extra_R_updates=0,
                optimized_modules=['R'], frozen_modules=['Q', 'C'], R_only_ablation=True,
                exact_resume=False, source_optimizer_state_restored=False,
                labels_used_in_training=False, K_label_access_in_training=False,
                Kc_or_exponent_target=False, response_weight=weight,
                source_checkpoint_sha256=source_sha, frozen_QC_sha256=frozen_sha,
                response_cache_sha256=plan['response_cache_sha256'],
                normalization=plan['normalization'], alpha_calibration='TRAIN源模型初始loss平衡；预先固定算法')
            write_json(folder/'manifest.json', arm_metadata)
            start = time.monotonic()
            for step in range(updates+1):
                if step:
                    optimizer.zero_grad(set_to_none=True)
                    direct, vin, vout, std = train_data
                    loss = closure_terms(model, direct.flatten(0,1))['closure']
                    if weight:
                        loss = loss + weight*response_terms(model, direct, vin, vout, std, normalization)['response_normalized']
                    if not torch.isfinite(loss):
                        raise FloatingPointError('非有限训练目标')
                    loss.backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters()):
                        raise FloatingPointError('R梯度缺失或非有限')
                    optimizer.step()
                if step % 100 == 0 or step in snapshots:
                    row = dict(step=step, extra_R_updates=step, source_epoch=12000,
                        **{f'{split}_{key}': value for split, data in [('train',train_data),('test',test_data)]
                           for key, value in evaluate(model,*data,normalization,weight).items()})
                    history.append(row)
                    write_json(folder/'history_progress.json', history)
                    job['current'] = dict(arm=arm, extra_R_updates=step)
                    write_json(output/'training_job.json', job)
                    print(json.dumps(dict(arm=arm,step=step,train_loss=row['train_total'],
                        test_response=row['test_response_normalized'])),flush=True)
                if step in snapshots:
                    state = {**payload['model'], **{f'theory_rg.{key}': value.detach().cpu().clone()
                             for key,value in model.theory_rg.state_dict().items()}}
                    if state_sha256(state) != frozen_sha:
                        raise AssertionError('冻结Q/C张量发生变化')
                    saved = dict(model=state, metadata={**arm_metadata,'extra_R_updates':step},
                        epoch=12000, source_epoch=12000, extra_R_updates=step,
                        optimizer=optimizer.state_dict(), history=deepcopy(history))
                    torch.save(saved,folder/f'training_step_{step}.pt')
            arm_metadata['extra_R_updates'] = updates
            write_json(folder/'manifest.json',arm_metadata)
            write_json(folder/'summary.json',dict(source_epoch=12000,actual_extra_R_updates=updates,
                requested_extra_R_updates=updates,frozen_QC_sha256=frozen_sha,
                elapsed_seconds=time.monotonic()-start,labels_used_in_training=False))
            job['completed_arms'].append(arm)
            write_json(output/'training_job.json',job)
        if sha256_file(source_path) != source_sha:
            raise ValueError('历史源权重发生变化')
        job.update(status='complete',current=None,actual_extra_R_updates=updates)
        write_json(output/'training_job.json',job)
    except BaseException as error:
        job.update(status='failed',requires_user_action=True,error=f'{type(error).__name__}: {error}')
        write_json(output/'training_job.json',job)
        raise
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--cache-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--updates',type=int,default=6000)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--smoke',action='store_true')
    train(**vars(parser.parse_args()))


if __name__=='__main__':
    main()
