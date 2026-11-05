"""冻结 Q/C 的有限物理扰动 R 消融；训练缓存只含编码和响应，不含耦合标签。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from experiments.ising_physical_response_ablation import (
    FrozenResidualR, response_terms, standardized_response,
)
from experiments.ising_response_loss_ablation import closure_terms, state_sha256, write_json
from scripts.diagnose_ising_mcrg import sha256_file


ARM_NAMES = ('A', 'B', 'C', 'D', 'E')
ARM_PROTOCOL = {
    'A': ('baseline', 'old'), 'B': ('baseline', 'white'),
    'C': ('enhanced', 'old'), 'D': ('enhanced', 'white'),
    'E': ('enhanced', 'none'),
}
NORMALIZATION_FLOOR = 1e-12
NOISE_MULTIPLIER = 2.
MIN_RELATIVE_SINGULAR_VALUE = .01
BOOTSTRAPS = 1000
BATCH_PHYSICAL_ENSEMBLES = 16
MAX_CUDA_BYTES = 8 * (1 << 30)
BENCHMARK_WARMUP_STEPS = 10
BENCHMARK_MEASURED_STEPS = 100
BENCHMARK_DIAGNOSTIC_WARMUP = 1
BENCHMARK_DIAGNOSTIC_MEASURED = 3
BENCHMARK_FIXED_OVERHEAD_SECONDS = 300.
ALLOWED_CACHE_KEYS = {
    'schema_version', 'direct', 'vin', 'vout', 'score_std', 'train_mask',
    'baseline_mask', 'condition_index', 'replica_index', 'condition_opaque_id',
    'size', 'logical_size', 'depth', 'c_draw_seeds', 'smoke', 'axes', 'labels_used_in_training',
}


def _tensor(value, name, *, dtype):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{name} 必须是 Tensor')
    return value.detach().to(device='cpu', dtype=dtype).clone()


def validate_cache(raw, *, depth, smoke=False):
    """白名单防止训练器意外取得 K、扰动幅度或条件的物理含义。"""
    if not isinstance(raw, dict) or set(raw) - ALLOWED_CACHE_KEYS:
        raise ValueError('训练缓存含白名单外字段；禁止耦合/幅度/条件物理标签')
    result = {name: _tensor(raw[name], name, dtype=torch.float64)
              for name in ('direct', 'vin', 'vout', 'score_std')}
    direct, vin, vout, std = (result[k] for k in ('direct', 'vin', 'vout', 'score_std'))
    if (direct.ndim != 4 or tuple(direct.shape[2:]) != (depth + 1, 3)
            or direct.shape[0] < 1 or direct.shape[1] < 1
            or vin.shape != (*direct.shape[:2], depth, 3, 6)
            or vout.shape != vin.shape or std.shape != (*direct.shape[:2], depth, 6)
            or any(not torch.isfinite(v).all() for v in result.values())
            or torch.any(std < 0)):
        raise ValueError('需要有限的独立物理系综/C抽样/尺度/三维编码/六方向响应轴')
    if not smoke and direct.shape[1] != 5:
        raise ValueError('正式缓存必须保留五次条件 C 抽样')
    for name in ('train_mask', 'baseline_mask'):
        if not isinstance(raw.get(name), torch.Tensor) or raw[name].dtype != torch.bool:
            raise ValueError(f'{name} 必须是布尔 Tensor')
        result[name] = _tensor(raw[name], name, dtype=torch.bool)
    for name in ('condition_index', 'replica_index'):
        if not isinstance(raw.get(name), torch.Tensor) or raw[name].dtype != torch.int64:
            raise ValueError(f'{name} 必须是 int64 opaque ID Tensor')
        result[name] = _tensor(raw[name], name, dtype=torch.int64)
    if any(result[k].shape != (len(direct),) for k in
           ('train_mask', 'baseline_mask', 'condition_index', 'replica_index')):
        raise ValueError('缓存物理系综 mask/opaque ID 长度不符')
    if torch.any(result['condition_index'] < 0) or torch.any(result['replica_index'] < 0):
        raise ValueError('opaque ID 不能为负')
    pairs = torch.stack((result['condition_index'], result['replica_index']), -1)
    if len(torch.unique(pairs, dim=0)) != len(pairs):
        raise ValueError('同一条件/物理链重复，不可当独立物理系综')
    for condition in torch.unique(result['condition_index']):
        selected = result['condition_index'] == condition
        if len(torch.unique(result['baseline_mask'][selected])) != 1:
            raise ValueError('同一 opaque 条件 baseline 标记不一致')
    if depth == 1 and torch.any(result['train_mask']):
        raise ValueError('独立 L54 仅评价，不得参与优化或归一化')
    result['depth'] = depth
    result['size'] = int(raw.get('size', 108 if depth == 2 else 54))
    return result


def data_slice(data, mask, device='cpu'):
    return tuple(data[key][mask].detach().to(device=device, dtype=torch.float64)
                 for key in ('direct', 'vin', 'vout', 'score_std'))


def reliable_projectors(data, *, seed, bootstraps=BOOTSTRAPS, allow_single_chain_smoke=False):
    """TRAIN 每条件、先 C 平均后按独立物理链 bootstrap；不看 TEST。"""
    if type(bootstraps) is not int or bootstraps < 20:
        raise ValueError('bootstrap 次数必须预先固定且不少于20')
    x, _ = standardized_response(data['vin'], data['vout'], data['score_std'])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows, projectors = [], {}
    train_mask = data['train_mask']
    for condition in torch.unique(data['condition_index'][train_mask], sorted=True).tolist():
        selected = train_mask & (data['condition_index'] == condition)
        samples = x[selected].mean(1)  # 物理链是重采样单位；C draw 不是独立样本。
        if len(samples) < 2 and not allow_single_chain_smoke:
            raise ValueError('可靠方向需每条件至少两条独立 TRAIN 链')
        mean = samples.mean(0)
        bootstrap_valid = len(samples) >= 2
        if bootstrap_valid:
            indices = torch.randint(len(samples), (bootstraps, len(samples)), generator=generator)
            centered = samples[indices].mean(1) - mean
        else:
            # 执行 smoke 只有1条TRAIN链：不伪复制样本、不称bootstrap可识别，白化方向全部跳过。
            centered = mean.new_zeros((1, *mean.shape))
        # 对均值矩阵整体扰动的谱范数给保守、与奇异向量选择无关的噪声上界。
        noise95 = torch.quantile(torch.linalg.matrix_norm(centered, ord=2), .95, dim=0)
        for transition in range(data['depth']):
            _, singular, vh = torch.linalg.svd(mean[transition], full_matrices=False)
            ratio = singular / singular[0] if singular[0] > 0 else torch.zeros_like(singular)
            keep = (singular > NOISE_MULTIPLIER * noise95[transition]) & (
                ratio >= MIN_RELATIVE_SINGULAR_VALUE)
            if not bootstrap_valid:
                keep = torch.zeros_like(keep)
            rank = int(keep.sum())
            # 保留真实 r 列，不补满秩、不创建最小范数 J 目标。
            projector = vh[keep].T / singular[keep] if rank else mean.new_empty((6, 0))
            projectors[(condition, transition)] = projector.detach()
            rows.append(dict(condition_index=condition, transition=transition,
                physical_train_ensembles=len(samples), C_draws_averaged_before_bootstrap=True,
                baseline=bool(data['baseline_mask'][selected][0]), reliable_rank=rank,
                mean_standardized_input_response=mean[transition].tolist(),
                singular_values=singular.tolist(), relative_singular_values=ratio.tolist(),
                noise_spectral_norm_q95=float(noise95[transition]) if bootstrap_valid else None,
                physical_chain_bootstrap_valid=bootstrap_valid,
                single_chain_smoke_has_no_reliable_directions=not bootstrap_valid,
                retained=keep.tolist(),
                right_projector_over_sigma=projector.tolist(),
                no_reliable_direction_action='跳过白化项并保留闭合' if rank == 0 else None))
    return projectors, rows


def identifiability(rows, *, smoke=False):
    """只对非零有限扰动 TRAIN 条件要求三维可识别；baseline 单列不挡 gate。"""
    enhanced = [row for row in rows if not row['baseline']]
    failed = [dict(condition_index=row['condition_index'], transition=row['transition'],
                   reliable_rank=row['reliable_rank']) for row in enhanced if row['reliable_rank'] < 3]
    passed = bool(enhanced) and not failed
    return dict(status='passed' if passed else 'insufficient_reliable_directions',
        scientific_gate_passed=passed, training_allowed=passed or smoke,
        smoke_bypasses_scientific_gate=bool(smoke and not passed),
        training_only=True, TEST_used=False, noise_multiplier=NOISE_MULTIPLIER,
        relative_singular_value_min=MIN_RELATIVE_SINGULAR_VALUE,
        scope='所有非零有限扰动 TRAIN 条件的两次转移；零扰动 baseline 只用可靠子空间，不要求满秩。',
        interpretation='数值满秩不等于可靠物理方向；未通过不强训、不调阈值、不扩大预算。',
        failed_nonbaseline_condition_transitions=failed, rows=rows,
        thermal_nu=None, thermal_mode_certified=False)


def whitened_response(model, direct, vin, vout, std, conditions, projectors):
    x, target = standardized_response(vin, vout, std)
    residual = model.response_jvp(direct[:, :, :-1], x) - target
    numerator, denominator = residual.sum() * 0., 0
    zero_rank_groups = 0
    for condition in torch.unique(conditions, sorted=True).tolist():
        mask = conditions == condition
        for transition in range(x.shape[2]):
            projector = projectors[(condition, transition)].to(device=x.device, dtype=x.dtype)
            rank = projector.shape[1]
            if rank == 0:
                zero_rank_groups += 1
                continue
            transformed = residual[mask, :, transition] @ projector
            # 先每个物理系综/C抽样对坐标和可靠方向求和并除 r，再物理系综等权。
            numerator = numerator + transformed.square().sum((-2, -1)).div(rank).sum()
            denominator += int(mask.sum()) * direct.shape[1]
    value = numerator / denominator if denominator else numerator
    return dict(response_white=value, white_valid_ensemble_draw_transitions=denominator,
                white_zero_rank_groups=zero_rank_groups)


def train_normalization(data, mask):
    _, target = standardized_response(data['vin'][mask], data['vout'][mask], data['score_std'][mask])
    raw = target.square().sum(-2).mean((0, 1, 3)).detach()
    return raw.clamp_min(NORMALIZATION_FLOOR), raw


def objective(model, values, conditions, *, response_kind, normalization, alpha, projectors):
    direct, vin, vout, std = values
    closure = closure_terms(model, direct.flatten(0, 1))
    result = {**closure, **response_terms(model, direct, vin, vout, std, normalization)}
    if response_kind == 'white':
        white = whitened_response(model, direct, vin, vout, std, conditions, projectors)
        result['response_white'] = white['response_white']
        penalty = white['response_white']
    else:
        penalty = result['response_normalized'] if response_kind == 'old' else closure['closure'] * 0.
    result['response_penalty'] = penalty
    result['total'] = closure['closure'] + alpha * penalty
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('非有限 R-only 目标或指标')
    return result


def evaluate(model, values, conditions, *, response_kind, normalization, alpha, projectors):
    with torch.no_grad():
        terms = objective(model, values, conditions, response_kind=response_kind,
                          normalization=normalization, alpha=alpha, projectors=projectors)
    return {key: value.detach().cpu().tolist() for key, value in terms.items()}


def model_statistics(model, data, device, source_train_std):
    """只用已保存编码/响应，保存完整 DR、全部谱和 FD；不创建 Q/C 或搜根。"""
    direct, vin, vout, std = data_slice(data, torch.ones(len(data['direct']), dtype=torch.bool), device)
    with torch.no_grad():
        identity = torch.eye(3, dtype=direct.dtype, device=device).expand(*direct.shape[:-1], 3, 3)
        jacobian = model.response_jvp(direct, identity)
        h = 1e-5
        finite = torch.stack([(model.advance_theory(direct + h * identity[..., j]) -
                               model.advance_theory(direct - h * identity[..., j])) / (2*h)
                              for j in range(3)], -1)
        eigenvalues = torch.linalg.eigvals(jacobian)
        r_of_g = model.advance_theory(direct)
        mean_g = direct.mean(1)
        r_of_mean = model.advance_theory(mean_g)
        all_pred = torch.zeros((*direct.shape[:2], data['depth'] + 1, data['depth'], 3),
                               dtype=direct.dtype, device=device)
        all_residual = torch.zeros_like(all_pred)
        valid = torch.zeros((data['depth'] + 1, data['depth']), dtype=torch.bool, device=device)
        for start in range(data['depth']):
            current = direct[:, :, start]
            for horizon in range(1, data['depth'] - start + 1):
                current = model.advance_theory(current)
                all_pred[:, :, start, horizon - 1] = current
                all_residual[:, :, start, horizon - 1] = current - direct[:, :, start + horizon]
                valid[start, horizon - 1] = True
        x, target = standardized_response(vin, vout, std)
        prediction = model.response_jvp(direct[:, :, :-1], x)
        residual = prediction - target
        singular = torch.linalg.svdvals(x)
        best = target @ torch.linalg.pinv(x, rtol=1e-8)
        best_residual = best @ x - target
        denominator = torch.linalg.matrix_norm(target, ord='fro', dim=(-2, -1))
        denominator_valid = denominator > 1e-15
        safe = denominator.clamp_min(1e-15)
        weights = singular.square()
        normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-300)
        entropy = -(normalized * normalized.clamp_min(1e-300).log()).sum(-1)
        rank = (singular > singular[..., :1] * 1e-8).sum(-1)
        condition = singular[..., 0] / singular[..., -1].clamp_min(1e-300)
        source_std = torch.as_tensor(source_train_std, dtype=direct.dtype, device=device)
        scale_offset = 0 if data['depth'] == 2 else 1
        targets_std = torch.zeros_like(valid, dtype=direct.dtype)
        per_ensemble_rmse = all_residual.square().mean((1, -1)).sqrt()
        coordinate_rmse = all_residual.square().mean(1).sqrt()
        for start, horizon in torch.nonzero(valid).tolist():
            targets_std[start, horizon] = source_std[scale_offset + start + horizon + 1]
        per_ensemble_nrmse = per_ensemble_rmse / targets_std.clamp_min(1e-300)
        result = dict(direct=direct, condition_index=data['condition_index'],
            replica_index=data['replica_index'], train_mask=data['train_mask'],
            baseline_mask=data['baseline_mask'], full_DR=jacobian, full_DR_fd_h1e_5=finite,
            full_DR_eigenvalues_real=eigenvalues.real, full_DR_eigenvalues_imag=eigenvalues.imag,
            r_of_g=r_of_g, displacement=r_of_g-direct, mean_g=mean_g,
            R_of_mean_g=r_of_mean, mean_R_of_g=r_of_g.mean(1),
            all_start_predictions=all_pred, all_start_residuals=all_residual,
            all_start_valid=valid, closure_rmse_per_physical_ensemble=per_ensemble_rmse,
            closure_nrmse_per_physical_ensemble=per_ensemble_nrmse,
            closure_coordinate_rmse_per_physical_ensemble=coordinate_rmse,
            closure_target_source_TRAIN_pooled_std=targets_std,
            standardized_input_response=x, standardized_output_response=target,
            DR_response_prediction=prediction, DR_response_residual=residual,
            response_target_norm=denominator, response_relative_norm_valid=denominator_valid,
            existing_R_relative_error=torch.linalg.matrix_norm(residual, ord='fro')/safe,
            best_J=best, best_J_residual=best_residual,
            best_J_relative_error=torch.linalg.matrix_norm(best_residual, ord='fro')/safe,
            response_input_singular_values=singular, response_input_numeric_rank=rank,
            response_input_effective_rank=torch.where(weights.sum(-1) > 0, entropy.exp(),
                                                      torch.zeros_like(entropy)),
            response_input_condition_number=condition,
            response_input_condition_number_valid=rank == 3,
            best_J_minimum_norm_extension_not_physical_spectrum=rank < 3)
    for name, value in result.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'非有限最终统计：{name}')
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def _write_npz(path, arrays):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _cuda_peak(device):
    if not str(device).startswith('cuda:'):
        return dict(allocated_bytes=0, reserved_bytes=0)
    peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                reserved_bytes=torch.cuda.max_memory_reserved(device))
    if max(peak.values()) > MAX_CUDA_BYTES:
        raise RuntimeError('训练自身 CUDA 显存峰值超过预定8GiB预算；不自动改变设备/配置')
    return peak


def _uuid_text(value):
    if isinstance(value, bytes):
        value = value.decode('ascii')
    value = str(value).strip().lower()
    if value.startswith('gpu-'):
        value = value[4:]
    value = value.replace('-', '')
    if not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('无法取得标准 GPU UUID，拒绝用名称猜测物理卡映射')
    return value


def validate_device_identity(device):
    """只核对设备，不更改 CUDA 映射、驱动或其它进程。"""
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('不接受 CUDA_VISIBLE_DEVICES 映射；cuda:N 必须与物理索引一致')
    if device not in ('cpu', 'cuda:2', 'cuda:3'):
        raise ValueError('仅允许 cpu 或已授权的物理 GPU2/3')
    if device == 'cpu':
        return dict(device='cpu', gpu_used=False, CUDA_VISIBLE_DEVICES_remapping=False)
    if not torch.cuda.is_available():
        raise ValueError('请求的 CUDA 设备不可用；不自动切换设备')
    index = int(device.split(':')[1])
    properties = torch.cuda.get_device_properties(index)
    actual_uuid = getattr(properties, 'uuid', None)
    if actual_uuid is None:
        raise ValueError('当前 torch 不提供设备 UUID；无法验证物理卡，拒绝猜测映射')
    query = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=uuid',
                            '--format=csv,noheader'], check=True, capture_output=True,
                           text=True, timeout=15)
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or _uuid_text(lines[0]) != _uuid_text(actual_uuid):
        raise ValueError('torch CUDA UUID 与 nvidia-smi 物理索引不符；不改映射或迁移')
    return dict(device=device, gpu_used=True, physical_index=index,
        physical_uuid=lines[0], torch_uuid=str(actual_uuid), name=str(properties.name),
        total_memory_bytes=int(properties.total_memory), CUDA_VISIBLE_DEVICES_remapping=False)


def _synchronize(device):
    if device != 'cpu':
        torch.cuda.synchronize(device)


def load_source_only(source_run):
    """只打开源检查点；不打开物理数据、训练缓存、标签或 Q/C。"""
    path = source_run/'training_step_12000.pt'
    if path.is_symlink() or not path.is_file():
        raise ValueError('需要非符号链接的源12000步检查点')
    digest = sha256_file(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    config = metadata['config']
    if (payload.get('epoch') != 12000 or metadata.get('labels_used_in_training') is not False
            or metadata.get('q_supervision') is not None
            or config['model']['theory_dim'] != 3 or config['model']['block_factor'] != 2):
        raise ValueError('仅允许完成的无监督12000步历史b2三维原表示源')
    return payload, digest, state_sha256(payload['model']), state_sha256(payload['model'], residual=True)


def synthetic_benchmark_panel(*, depth, generator):
    """与正式评价同轴大小的纯随机 panel；无物理 Hamiltonian、构型或耦合。"""
    conditions = 41 if depth == 2 else 27
    ensembles, draws = conditions * 12, 5
    direct = .2 * torch.randn(ensembles, draws, depth+1, 3, generator=generator, dtype=torch.float64)
    std = 1. + .2 * torch.rand(ensembles, draws, depth, 6, generator=generator, dtype=torch.float64)
    canonical = torch.cat((torch.eye(3, dtype=torch.float64), torch.zeros(3, 3, dtype=torch.float64)), -1)
    x = canonical.expand(ensembles, draws, depth, 3, 6).clone()
    y = 1.2 * x + .01 * torch.randn(*x.shape, generator=generator, dtype=torch.float64)
    ids = torch.arange(conditions).repeat_interleave(12)
    replicas = torch.arange(12).repeat(conditions)
    train = (ids < 14) & (replicas < 8) if depth == 2 else torch.zeros(ensembles, dtype=torch.bool)
    return validate_cache(dict(schema_version=1, direct=direct, vin=x*std[..., None, :],
        vout=y*std[..., None, :], score_std=std, condition_index=ids, replica_index=replicas,
        train_mask=train, baseline_mask=ids < 2, size=108 if depth == 2 else 54, depth=depth),
        depth=depth, smoke=False)


def _benchmark_diagnostic_times(model, *, device, generator, projectors):
    """实测正式额外 R 评价成本；不优化、不保存这些 synthetic 统计。"""
    training = synthetic_benchmark_panel(depth=2, generator=generator)
    independent = synthetic_benchmark_panel(depth=1, generator=generator)
    def periodic_round():
        for domain, kind in ARM_PROTOCOL.values():
            mask = training['train_mask'] & (training['baseline_mask'] if domain == 'baseline'
                                             else torch.ones_like(training['train_mask']))
            evaluate(model, data_slice(training, mask, device), training['condition_index'][mask].to(device),
                response_kind=kind, normalization=torch.ones(2, dtype=torch.float64, device=device),
                alpha=1., projectors=projectors)
            mask = ~training['train_mask']
            evaluate(model, data_slice(training, mask, device), training['condition_index'][mask].to(device),
                response_kind='old', normalization=torch.ones(2, dtype=torch.float64, device=device),
                alpha=0., projectors={})
            evaluate(model, data_slice(independent, torch.ones(len(independent['direct']), dtype=torch.bool), device),
                independent['condition_index'].to(device), response_kind='old',
                normalization=torch.ones(1, dtype=torch.float64, device=device), alpha=0., projectors={})
    def statistics_pair():
        # 包括全 DR、六次中心差分前向、全复谱、逐样本pinv/SVD和主机数组转换。
        for panel in (training, independent):
            arrays = model_statistics(model, panel, device, [1., 1., 1.])
            del arrays
    result = {}
    for name, callback in (('five_arm_periodic_evaluation_round', periodic_round),
                           ('two_size_final_statistics_pair', statistics_pair)):
        measured = []
        for repeat in range(BENCHMARK_DIAGNOSTIC_WARMUP + BENCHMARK_DIAGNOSTIC_MEASURED):
            _synchronize(device)
            tick = time.perf_counter()
            callback()
            _synchronize(device)
            elapsed = time.perf_counter()-tick
            _cuda_peak(device)
            if repeat >= BENCHMARK_DIAGNOSTIC_WARMUP:
                measured.append(elapsed)
        result[name] = dict(warmup_iterations=BENCHMARK_DIAGNOSTIC_WARMUP,
            measured_iterations=BENCHMARK_DIAGNOSTIC_MEASURED, wall_seconds=measured,
            max_wall_seconds=max(measured), mean_wall_seconds=float(np.mean(measured)))
    return result


def benchmark(source_run, output, *, device='cpu', seed=91017):
    """固定 synthetic 执行检查，不读真实缓存、不保存拟合权重或产生科学更新。"""
    paths = [Path(source_run), Path(output)]
    if any(path.is_symlink() for path in paths):
        raise ValueError('benchmark 源/输出不接受符号链接')
    source_run, output = [path.resolve() for path in paths]
    if output.exists():
        raise FileExistsError('拒绝覆盖或重复 benchmark')
    if output == source_run or output in source_run.parents or source_run in output.parents:
        raise ValueError('benchmark 输出必须与历史源分离')
    if type(seed) is not int:
        raise ValueError('benchmark 需要固定整数 seed')
    identity = validate_device_identity(device)
    torch.set_num_threads(1)
    payload, source_sha, frozen_sha, initial_r_sha = load_source_only(source_run)
    if device != 'cpu':
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', workflow_complete=False, scientific_execution=False,
        scientific_R_updates=0, source_checkpoint_sha256=source_sha, frozen_QC_sha256=frozen_sha,
        initial_R_sha256=initial_r_sha, Q_C_created_or_called=False, physical_data_opened=False,
        cache_opened=False, labels_read=False, seed=seed, device=device, device_identity=identity,
        compute_threads=1, requires_user_action=False, error=None)
    write_json(output/'status.json', status)
    started = time.perf_counter()
    try:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        direct = (.2 * torch.randn(16, 5, 3, 3, generator=generator, dtype=torch.float64)).to(device)
        std = (1. + .2 * torch.rand(16, 5, 2, 6, generator=generator, dtype=torch.float64)).to(device)
        canonical = torch.cat((torch.eye(3, dtype=torch.float64), torch.zeros(3, 3, dtype=torch.float64)), -1)
        x = canonical.expand(16, 5, 2, 3, 6).clone().to(device)
        y = (1.2 * canonical.expand(16, 5, 2, 3, 6) + .01 * torch.randn(
            16, 5, 2, 3, 6, generator=generator, dtype=torch.float64)).to(device)
        vin, vout = x * std[..., None, :], y * std[..., None, :]
        values = tuple(value.detach() for value in (direct, vin, vout, std))
        # 非物理 opaque condition，仅覆盖实际每批至多14组的白化调度开销。
        conditions = (torch.arange(16, dtype=torch.int64) % 14).to(device)
        projectors = {(condition, transition):canonical.T.clone() for condition in range(14)
                      for transition in range(2)}
        normalization = y.square().sum(-2).mean((0,1,3)).clamp_min(NORMALIZATION_FLOOR)
        base = FrozenResidualR(payload['model']).to(device)
        timings = {}
        for kind in ('old', 'white', 'none'):
            model = deepcopy(base)
            copied_state = {f'theory_rg.{key}':value for key,value in model.theory_rg.state_dict().items()}
            if state_sha256(copied_state, residual=True) != initial_r_sha:
                raise AssertionError('benchmark 三目标未从同一源 R 初始化')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            first = evaluate(model, values, conditions, response_kind=kind,
                normalization=normalization, alpha=0., projectors=projectors)
            alpha = 0. if kind == 'none' else first['closure'] / max(first['response_penalty'], 1e-12)
            elapsed = []
            for step in range(BENCHMARK_WARMUP_STEPS + BENCHMARK_MEASURED_STEPS):
                _synchronize(device)
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                terms = objective(model, values, conditions, response_kind=kind,
                    normalization=normalization, alpha=alpha, projectors=projectors)
                terms['total'].backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise FloatingPointError('benchmark R 梯度缺失或非有限')
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError('benchmark R 参数非有限')
                _synchronize(device)
                duration = time.perf_counter()-tick
                _cuda_peak(device)
                if step >= BENCHMARK_WARMUP_STEPS:
                    elapsed.append(duration)
            steps = sorted({int(item['step']) for item in optimizer.state.values()})
            if steps != [BENCHMARK_WARMUP_STEPS + BENCHMARK_MEASURED_STEPS]:
                raise AssertionError('benchmark 各参数更新数不符')
            timings[kind] = dict(warmup_steps=BENCHMARK_WARMUP_STEPS,
                measured_steps=BENCHMARK_MEASURED_STEPS, fresh_adam=True,
                initial_R_sha256=initial_r_sha, source_only_initialization=True,
                step_seconds=elapsed, mean_step_seconds=float(np.mean(elapsed)),
                max_step_seconds=float(max(elapsed)), min_step_seconds=float(min(elapsed)),
                every_adam_parameter_steps=steps, alpha_synthetic_initial_balance=alpha)
        diagnostic_timings = _benchmark_diagnostic_times(base, device=device, generator=generator,
                                                         projectors=projectors)
        _synchronize(device)
        total_seconds = time.perf_counter()-started
        optimizer_projection = 2. * 6000 * (2*timings['old']['max_step_seconds'] +
            2*timings['white']['max_step_seconds'] + timings['none']['max_step_seconds'])
        periodic_projection = 2. * 61 * diagnostic_timings['five_arm_periodic_evaluation_round']['max_wall_seconds']
        statistics_projection = 2. * 6 * diagnostic_timings['two_size_final_statistics_pair']['max_wall_seconds']
        projected = optimizer_projection + periodic_projection + statistics_projection + BENCHMARK_FIXED_OVERHEAD_SECONDS
        if (sha256_file(source_run/'training_step_12000.pt') != source_sha
                or state_sha256(payload['model']) != frozen_sha):
            raise ValueError('benchmark 后历史源检查点指纹变化')
        result = {**status, 'timings':timings, 'status':'complete', 'workflow_complete':True}
        result.update(source_checkpoint_unchanged=True, synthetic_white_rows_orthonormal_rank=3,
            frozen_QC_sha256_before=frozen_sha, frozen_QC_sha256_after=state_sha256(payload['model']),
            optimizer='fresh Adam', learning_rate=.001,
            synthetic_white_projector_not_scientific=True, synthetic_batch_physical_ensembles=16,
            synthetic_C_draws=5, direct_shape=list(direct.shape), vin_shape=list(vin.shape),
            target_shape=list(vout.shape), score_std_positive=bool(torch.all(std > 0)),
            warmup_optimizer_steps=3*BENCHMARK_WARMUP_STEPS,
            measured_optimizer_steps=3*BENCHMARK_MEASURED_STEPS,
            execution_check_optimizer_steps=3*(BENCHMARK_WARMUP_STEPS+BENCHMARK_MEASURED_STEPS),
            total_elapsed_seconds=total_seconds, elapsed_gpu_budget_seconds=total_seconds if device != 'cpu' else 0.,
            diagnostic_timings=diagnostic_timings,
            projected_training_wall_seconds_with_margin=projected,
            benchmark_module_sha256=sha256_file(Path(__file__)),
            projected_training_gpu_seconds_with_margin=projected if device != 'cpu' else 0.,
            projected_optimizer_seconds_with_margin=optimizer_projection,
            projected_periodic_evaluation_seconds_with_margin=periodic_projection,
            projected_final_statistics_seconds_with_margin=statistics_projection,
            fixed_IO_initialization_bootstrap_wall_seconds_reserve=BENCHMARK_FIXED_OVERHEAD_SECONDS,
            periodic_evaluation_rounds=61, all_model_statistics_two_size_pairs=6,
            periodic_evaluation_diagnostics_per_round=15,
            projection_rule='2×6000×(2×old最慢单步＋2×white最慢单步＋none最慢单步)'
                '＋2×61×五臂一轮完整评价最慢耗时＋2×6×两尺寸末统计最慢耗时＋300秒固定余量',
            fixed_reserve_scope='覆盖GPU子阶段内CPU bootstrap/启动/来源哈希/20个checkpoint/历史JSON与6次增长NPZ写入；'
                'NAS或主机负载可变，实测投影不是硬耗时保证，不允许靠缩短科学预算弥补超时。',
            scientific_R_updates=0, fitted_weights_saved=False, cuda_peak=_cuda_peak(device),
            max_cuda_memory_bytes=MAX_CUDA_BYTES, thermal_nu=None)
        write_json(output/'benchmark.json', result)
        write_json(output/'status.json', result)
        return result
    except BaseException as error:
        status.update(status='failed', workflow_complete=False, requires_user_action=True,
                      error=f'{type(error).__name__}: {error}')
        write_json(output/'status.json', status)
        raise


def _receipt_value(receipt, *names):
    found = [receipt[name] for name in names if name in receipt]
    if not found or any(value != found[0] for value in found):
        raise ValueError(f'缓存回执缺失或别名不一致：{names}')
    return found[0]


def load_inputs(source_run, cache_root, *, smoke=False):
    source_path = source_run / 'training_step_12000.pt'
    paths = [source_path, cache_root/'receipt.json', cache_root/'training_L108.pt',
             cache_root/'evaluation_L54.pt']
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('需要非符号链接的完整源及无标签缓存')
    source_sha = sha256_file(source_path)
    receipt = json.loads(paths[1].read_text())
    if (receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True
            or receipt.get('smoke') is not smoke
            or receipt.get('source_checkpoint_sha256') != source_sha
            or receipt.get('no_K_labels_in_training_cache') is not True):
        raise ValueError('缓存完成、无标签、smoke 或来源指纹回执不符')
    for path in paths[2:]:
        if receipt.get('cache_sha256', {}).get(path.name) != sha256_file(path):
            raise ValueError(f'无标签缓存指纹不符：{path.name}')
    payload = torch.load(source_path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    config = metadata['config']
    if (payload.get('epoch') != 12000 or metadata.get('labels_used_in_training') is not False
            or metadata.get('q_supervision') is not None
            or config['model']['theory_dim'] != 3 or config['model']['block_factor'] != 2):
        raise ValueError('仅允许完成的无监督12000步历史b2三维原表示源')
    frozen_sha = state_sha256(payload['model'])
    residual_sha = state_sha256(payload['model'], residual=True)
    if (_receipt_value(receipt, 'source_frozen_QC_sha256_before', 'frozen_QC_sha256_before') != frozen_sha
            or _receipt_value(receipt, 'source_frozen_QC_sha256_after', 'frozen_QC_sha256_after') != frozen_sha
            or _receipt_value(receipt, 'source_R_sha256', 'initial_R_sha256') != residual_sha):
        raise ValueError('冻结 Q/C 或初始 R 来源指纹不符')
    training = validate_cache(torch.load(paths[2], map_location='cpu', weights_only=True), depth=2, smoke=smoke)
    evaluation = validate_cache(torch.load(paths[3], map_location='cpu', weights_only=True), depth=1, smoke=smoke)
    train_mask = training['train_mask']
    baseline = train_mask & training['baseline_mask']
    if not train_mask.any() or not baseline.any() or not (~train_mask).any():
        raise ValueError('需要非空 TRAIN、TRAIN baseline 以及独立留出物理链')
    counts = [int((train_mask & (training['condition_index'] == c)).sum())
              for c in torch.unique(training['condition_index'][train_mask])]
    if not smoke and (int(train_mask.sum()) != 112 or int(baseline.sum()) != 16
                      or len(counts) != 14 or set(counts) != {8}):
        raise ValueError('正式 TRAIN 固定14条件×8链，baseline固定2条件×8链')
    return payload, training, evaluation, receipt, source_sha, frozen_sha, residual_sha


def source_coordinate_normalization(source_run):
    path = source_run.parent/'report/metrics.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('需要冻结原 TRAIN 坐标标准差的小 JSON 回执')
    metrics = json.loads(path.read_text())
    rows = sorted(metrics['train_representation'], key=lambda row: row['scale'])
    if [row['scale'] for row in rows] != [0, 1, 2]:
        raise ValueError('原 TRAIN 必须完整提供108/54/27三尺度标准差')
    pooled = [float(row['pooled_coordinate_std']) for row in rows]
    if any(not math.isfinite(v) or v <= 0 for v in pooled):
        raise ValueError('原 TRAIN pooled std 必须正且有限')
    coordinates = np.asarray([row['coordinate_std'] for row in rows], dtype=np.float64)
    if coordinates.shape != (3, 3) or not np.isfinite(coordinates).all() or (coordinates <= 0).any():
        raise ValueError('原 TRAIN coordinate std 必须为正有限3×3数组')
    return dict(path=str(path), sha256=sha256_file(path), source_lattices=[108, 54, 27],
                pooled_coordinate_std=pooled, coordinate_std=coordinates.tolist())


def train(source_run, cache_root, output, *, device='cpu', updates=6000, seed=91017, smoke=False):
    originals = [Path(value) for value in (source_run, cache_root, output)]
    if any(path.is_symlink() for path in originals):
        raise ValueError('源/缓存/输出路径不接受符号链接')
    source_run, cache_root, output = [p.resolve() for p in originals]
    if output.exists():
        raise FileExistsError('拒绝覆盖、重复启动或隐式续训')
    for protected in (source_run, cache_root):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('训练输出必须与源、缓存目录分离')
    if device not in ('cpu', 'cuda:2', 'cuda:3') or updates != 6000 or type(seed) is not int:
        raise ValueError('固定6000步，只允许cpu/cuda:2/cuda:3和整数seed')
    device_identity = validate_device_identity(device)
    torch.set_num_threads(1)
    payload, data, independent, receipt, source_sha, frozen_sha, initial_r_sha = load_inputs(
        source_run, cache_root, smoke=smoke)
    source_normalization = source_coordinate_normalization(source_run)
    if device != 'cpu':
        torch.cuda.reset_peak_memory_stats(device)
    base = FrozenResidualR(payload['model']).to(device)
    if any(not name.startswith('theory_rg.') for name, _ in base.named_parameters()):
        raise AssertionError('R-only训练不得创建 Q/C 参数')
    actual_updates = 2 if smoke else updates
    snapshots = sorted({0, actual_updates, *[s for s in (1000, 3000, 6000) if s <= actual_updates]})
    projectors, rows = reliable_projectors(data, seed=seed + 1, bootstraps=100 if smoke else BOOTSTRAPS,
                                          allow_single_chain_smoke=smoke)
    gate = identifiability(rows, smoke=smoke)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'identifiability.json', {**gate, 'bootstrap_count': 100 if smoke else BOOTSTRAPS,
                                               'bootstrap_seed': seed + 1})
    plan = dict(source_run=str(source_run), cache_root=str(cache_root), source_epoch=12000,
        source_checkpoint_sha256=source_sha, frozen_QC_sha256=frozen_sha, initial_R_sha256=initial_r_sha,
        cache_sha256=receipt['cache_sha256'], updates=actual_updates, requested_extra_R_updates=updates,
        learning_rate=.001, optimizer='fresh Adam', snapshots=snapshots, seed=seed, smoke=smoke,
        arms={name: dict(domain=domain, response=response) for name, (domain, response) in ARM_PROTOCOL.items()},
        batch_physical_ensembles=BATCH_PHYSICAL_ENSEMBLES, all_C_draws_per_selected_ensemble=True,
        paired_batch_randomization='各臂重置同seed；A/B同批，C/D/E同批；每批独立物理系综不重复。',
        labels_used_in_training=False, physical_couplings_read=False, Kc_or_exponent_target=False,
        Q_C_created_or_called=False, optimized_modules=['R'], frozen_modules=['Q','C'], MI_estimates=0,
        compute_threads=1, max_cuda_memory_bytes=MAX_CUDA_BYTES, exact_resume=False,
        device_identity=device_identity,
        closure='5一步＋5微观起点all-horizon闭合，直接复用原函数。',
        old_response='六偶score标准差归一后，按各域TRAIN目标响应平方范数逐转移归一。',
        white_response='TRAIN按条件C先平均/物理链bootstrap；仅可靠右奇异方向，按1/sigma均衡。',
        alpha_calibration='各臂所属TRAIN域上源R初始closure/response_penalty，一次冻结；E=0。',
        TEST_only_metrics='标准六向响应与闭合；不把未见条件映射到TRAIN白化投影。',
        closure_NRMSE_source=source_normalization,
        smoke_lattice_normalization_substitute=bool(smoke),
        thermal_nu=None, thermal_mode_certified=False)
    write_json(output/'plan.json', plan)
    job = dict(status='running', workflow_complete=False, training_started=False, current=None,
               completed_arms=[], requires_user_action=False, error=None, smoke=smoke)
    def save_job():
        write_json(output/'job.json', job)
        write_json(output/'status.json', job)
    save_job()
    try:
        arrays = {}
        for label, cache in (('L108', data), ('L54', independent)):
            arrays.update({f'initial_{label}_{key}': value for key, value in
                           model_statistics(base, cache, device,
                               source_normalization['pooled_coordinate_std']).items()})
        _write_npz(output/'final_statistics.npz', arrays)
        if not gate['training_allowed']:
            job.update(status='scientific_stop', workflow_complete=True, training_started=False,
                       reason='新增非零有限扰动 TRAIN 域未能在每条件/两转移支持三个可靠方向；不强训。',
                       source_checkpoint_unchanged=sha256_file(source_run/'training_step_12000.pt') == source_sha,
                       thermal_nu=None)
            if not job['source_checkpoint_unchanged']:
                raise ValueError('历史源权重发生变化')
            save_job()
            return job
        job['training_started'] = True
        save_job()
        for arm in ARM_NAMES:
            domain, response_kind = ARM_PROTOCOL[arm]
            mask = data['train_mask'] & (data['baseline_mask'] if domain == 'baseline'
                                         else torch.ones_like(data['train_mask']))
            indices = torch.where(mask)[0]
            values = data_slice(data, mask, device)
            conditions = data['condition_index'][mask].to(device)
            normalization, raw_normalization = train_normalization(data, mask)
            normalization = normalization.to(device)
            model = deepcopy(base)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            initial = evaluate(model, values, conditions, response_kind=response_kind,
                normalization=normalization, alpha=0., projectors=projectors)
            penalty = initial['response_penalty']
            alpha = 0. if response_kind == 'none' or penalty <= NORMALIZATION_FLOOR else initial['closure'] / penalty
            if not math.isfinite(alpha) or alpha < 0:
                raise ValueError('TRAIN 初值不支持有限正响应平衡权重')
            folder = output/arm
            folder.mkdir()
            metadata = deepcopy(payload['metadata'])
            metadata.update(training_driver='ising_finite_perturbation_ablation', source_epoch=12000,
                source_checkpoint_sha256=source_sha, requested_extra_R_updates=actual_updates,
                extra_R_updates=0, optimized_modules=['R'], frozen_modules=['Q','C'],
                Q_C_created_or_called=False, labels_used_in_training=False, K_label_access_in_training=False,
                Kc_or_exponent_target=False, source_optimizer_state_restored=False, exact_resume=False,
                arm=arm, domain=domain, response_kind=response_kind, response_weight=alpha,
                frozen_QC_sha256=frozen_sha, initial_R_sha256=initial_r_sha,
                normalization=normalization.cpu().tolist(), raw_normalization=raw_normalization.tolist(),
                normalization_floor=NORMALIZATION_FLOOR, alpha_calibration=initial,
                response_penalty_zero_or_below_floor=penalty <= NORMALIZATION_FLOOR,
                skip_response_if_no_reliable_direction=True, compute_threads=1,
                batch_physical_ensembles=min(len(indices), BATCH_PHYSICAL_ENSEMBLES), smoke=smoke)
            write_json(folder/'manifest.json', metadata)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            history, started = [], time.monotonic()
            for step in range(actual_updates + 1):
                if step:
                    batch_local = torch.randperm(len(indices), generator=generator)[:BATCH_PHYSICAL_ENSEMBLES]
                    batch_ids = indices[batch_local]
                    batch = data_slice(data, batch_ids, device)
                    batch_conditions = data['condition_index'][batch_ids].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, batch, batch_conditions, response_kind=response_kind,
                        normalization=normalization, alpha=alpha, projectors=projectors)['total']
                    loss.backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                           for parameter in model.parameters()):
                        raise FloatingPointError('R 梯度缺失或非有限')
                    optimizer.step()
                    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                        raise FloatingPointError('R 参数非有限')
                    _cuda_peak(device)
                if step % 100 == 0 or step in snapshots:
                    row = dict(step=step, source_epoch=12000, extra_R_updates=step,
                        train=evaluate(model, values, conditions, response_kind=response_kind,
                                       normalization=normalization, alpha=alpha, projectors=projectors))
                    test_mask = ~data['train_mask']
                    row['test'] = evaluate(model, data_slice(data, test_mask, device),
                        data['condition_index'][test_mask].to(device), response_kind='old',
                        normalization=normalization, alpha=0., projectors={})
                    independent_normalizer = normalization[-1:]
                    row['independent_L54'] = evaluate(model,
                        data_slice(independent, torch.ones(len(independent['direct']), dtype=torch.bool), device),
                        independent['condition_index'].to(device), response_kind='old',
                        normalization=independent_normalizer, alpha=0., projectors={})
                    history.append(row)
                    write_json(folder/'history_progress.json', history)
                    job['current'] = dict(arm=arm, extra_R_updates=step)
                    save_job()
                    print(json.dumps(dict(arm=arm, step=step, source_epoch=12000,
                        extra_R_updates=step, train_loss=row['train']['total'],
                        test_standardized_response=row['test']['response_normalized']), allow_nan=False), flush=True)
                if step in snapshots:
                    state = {**payload['model'], **{f'theory_rg.{key}': value.detach().cpu().clone()
                             for key, value in model.theory_rg.state_dict().items()}}
                    if state_sha256(state) != frozen_sha:
                        raise AssertionError('保存模型时冻结 Q/C 张量发生变化')
                    saved = dict(model=state, metadata={**metadata, 'extra_R_updates':step},
                        epoch=12000, source_epoch=12000, extra_R_updates=step,
                        optimizer=optimizer.state_dict(), history=deepcopy(history))
                    torch.save(saved, folder/f'training_step_{step}.pt')
            peak = _cuda_peak(device)
            metadata.update(extra_R_updates=actual_updates, cuda_peak=peak)
            write_json(folder/'manifest.json', metadata)
            summary = dict(status='complete', workflow_complete=True, arm=arm, source_epoch=12000,
                requested_extra_R_updates=actual_updates, actual_extra_R_updates=actual_updates,
                every_adam_parameter_updates=sorted({int(s['step']) for s in optimizer.state.values()}),
                frozen_QC_sha256=frozen_sha, initial_R_sha256=initial_r_sha,
                final_R_sha256=state_sha256({f'theory_rg.{k}':v for k,v in model.theory_rg.state_dict().items()}, residual=True),
                response_weight=alpha, elapsed_seconds=time.monotonic()-started, cuda_peak=peak,
                labels_used_in_training=False, thermal_nu=None, final_metrics=history[-1])
            if summary['every_adam_parameter_updates'] != [actual_updates]:
                raise AssertionError('各 R 参数 Adam 更新数不同于固定预算')
            for label, cache in (('L108', data), ('L54', independent)):
                arrays.update({f'{arm}_{label}_{key}':value for key,value in
                               model_statistics(model, cache, device,
                                   source_normalization['pooled_coordinate_std']).items()})
            _write_npz(output/'final_statistics.npz', arrays)
            write_json(folder/'summary.json', summary)
            write_json(folder/'status.json', summary)
            job['completed_arms'].append(arm)
            save_job()
        if sha256_file(source_run/'training_step_12000.pt') != source_sha:
            raise ValueError('历史源权重发生变化')
        job.update(status='complete', workflow_complete=True, current=None,
            actual_extra_R_updates_per_arm=actual_updates, source_checkpoint_unchanged=True,
            cuda_peak=_cuda_peak(device), thermal_nu=None)
        save_job()
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}')
        save_job()
        raise
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:2', 'cuda:3'), default='cpu')
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=91017)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = vars(parser.parse_args(argv))
    is_benchmark = args.pop('benchmark')
    if is_benchmark:
        if args.pop('cache_root') is not None or args.pop('smoke'):
            raise ValueError('benchmark 不允许物理缓存或smoke；它是固定 synthetic 执行检查')
        args.pop('updates')
        benchmark(**args)
    else:
        if args['cache_root'] is None:
            raise ValueError('正式或smoke训练必须显式提供 --cache-root')
        train(**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
