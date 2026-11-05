"""独立初始尺寸、相同末尺度的固定 Q/C/R 只读诊断。

本脚本不训练、不估计 MI、不生成物理构型。原始 L=54 和 L=108 系综
分别来自独立链；C 抽样是条件重复，不增加独立物理样本数。相同 K、相同
末尺度但不同 RG 历史的编码差异，并不单独证明 Q 错误或物理闭包失败。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_ising_mcrg import (
    OPERATOR_DEFINITIONS, OPERATOR_NAMES, ORIENTATION_NAMES, c_draw,
    c_probability_table, covariance, full_r_jacobian, local_feature_means,
    model_sha256, operator_values, orientation_operator_values, sha256_file,
    weighted_q_response,
)
from scripts.diagnose_ising_dimg_response import (
    RESPONSE_RCOND, best_local_jacobian, full_r_finite_difference,
)


Q_FD_T_VALUES = (.001, .003)
R_FD_H = 1e-5


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def c_seed(seed, size, k_value, replica, draw, transition):
    """Namespaced reproducible C noise, paired across models but not lattice sizes."""
    text = (f'matched-sizes-C-v1|{int(seed)}|{int(size)}|{float(k_value).hex()}|'
            f'{int(replica)}|{int(draw)}|{int(transition)}')
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def _new_output(output, protected=()):
    if Path(output).is_symlink():
        raise FileExistsError(f'Refusing symbolic-link diagnostic output: {output}')
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing overwrite or retry of an existing diagnostic: {output}')
    for path in protected:
        path = Path(path).resolve()
        if path == output or path in output.parents or output in path.parents:
            raise ValueError('Diagnostic output must be separate from all source artifacts')
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'status.json', dict(status='running', workflow_complete=False,
        no_training=True, no_MI=True, no_new_physical_sampling=True))
    return output


def _failed(output, error, started, model=None, before=None):
    state = dict(status='failed', workflow_complete=False,
        error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
        no_training=True, no_MI=True, no_new_physical_sampling=True,
        retry_policy='保留失败目录；不覆盖、不自动重跑。')
    if model is not None:
        state.update(model_sha256_before=before, model_sha256_after=model_sha256(model),
            all_parameter_grads_none=all(p.grad is None for p in model.parameters()))
    _write_json(output / 'status.json', state)


def _validate_arrays(configurations, K, chain_seeds, *, size, depth, smoke):
    if type(size) is not int or type(depth) is not int or depth not in (1, 2):
        raise ValueError('Require explicit integer size and depth one or two')
    if size != (4 if smoke else 27) * 2 ** depth:
        raise ValueError('Independent size must end at L=27 (formal) or L=4 (smoke)')
    if (not isinstance(configurations, torch.Tensor) or configurations.dtype != torch.int8
            or configurations.ndim != 5 or tuple(configurations.shape[-2:]) != (size, size)
            or configurations.shape[0] != 3):
        raise ValueError('Require int8 configurations [3 K, independent replicas, samples, L, L]')
    if (not isinstance(K, torch.Tensor) or K.dtype != torch.float64 or K.shape != (3,)
            or not torch.isfinite(K).all() or not torch.all(K[1:] > K[:-1])):
        raise ValueError('Require three finite increasing float64 evaluation K values')
    nrep, samples = configurations.shape[1:3]
    if samples < 2 or nrep < 1 or (not smoke and (nrep != 12 or samples != 128)):
        raise ValueError('Formal data require 12 independent ensembles x128 configurations per K')
    if (not isinstance(chain_seeds, torch.Tensor) or chain_seeds.dtype != torch.int64
            or tuple(chain_seeds.shape) != (3, nrep)
            or torch.unique(chain_seeds).numel() != chain_seeds.numel()):
        raise ValueError('Require a unique int64 production-chain seed for each physical ensemble')
    return int(nrep), int(samples)


def validate_data(data_dir, *, smoke=False):
    """Validate small completed receipts and tensor metadata; do not rehash data.pt."""
    data_dir = Path(data_dir)
    sidecar_path, status_path = data_dir / 'sidecar.json', data_dir / 'status.json'
    sidecar = json.loads(sidecar_path.read_text())
    status = json.loads(status_path.read_text())
    if (sidecar.get('schema_version') != 1 or sidecar.get('status') != 'complete'
            or sidecar.get('workflow_complete') is not True or status.get('status') != 'complete'
            or status.get('workflow_complete') is not True):
        raise ValueError('Independent physical sampling is not verifiably complete')
    fingerprint = sidecar.get('data_sha256')
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in '0123456789abcdef' for c in fingerprint)
            or status.get('data_sha256') != fingerprint
            or status.get('sidecar_sha256') != sha256_file(sidecar_path)):
        raise ValueError('Independent data small fingerprint receipts disagree')
    data = torch.load(data_dir / 'data.pt', map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Require the new independent-size schema1 diagnostic data')
    metadata_hash = hashlib.sha256(json.dumps(sidecar.get('provenance'), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if (not sidecar.get('metadata_fingerprint') or metadata_hash != sidecar['metadata_fingerprint']
            or status.get('metadata_fingerprint') != metadata_hash
            or data.get('metadata_fingerprint') != metadata_hash):
        raise ValueError('Payload/sidecar sampling metadata fingerprint mismatch')
    for key in ('size', 'target_size', 'depth', 'max_rg_steps', 'block_factor', 'seed', 'smoke'):
        if data.get(key) != sidecar.get(key):
            raise ValueError(f'Payload/sidecar {key} mismatch')
    if (sidecar.get('block_factor') != 2 or sidecar.get('target_size') != (4 if smoke else 27)
            or sidecar.get('smoke') is not smoke or sidecar.get('depth') != sidecar.get('max_rg_steps')
            or sidecar.get('diagnostic_only') is not True or data.get('diagnostic_only') is not True):
        raise ValueError('Require explicit b2 independent-size diagnostic-only geometry')
    _validate_arrays(data.get('configurations'), data.get('K'), data.get('chain_seeds'),
        size=sidecar['size'], depth=sidecar['depth'], smoke=smoke)
    provenance = sidecar['provenance']
    if (provenance.get('K_values') != data['K'].tolist()
            or provenance.get('chain_seeds') != data['chain_seeds'].tolist()
            or provenance.get('global_flip_seeds') != data.get('global_flip_seeds', torch.empty(0)).tolist()):
        raise ValueError('Payload sampling K or independent-chain seed maps disagree with small receipts')
    options = provenance.get('sampling_options', {})
    for key in ('size', 'seed', 'replicas', 'samples', 'max_rg_steps', 'block_factor', 'target_size'):
        if options.get(key) != sidecar.get(key):
            raise ValueError(f'Canonical sampling options/sidecar {key} mismatch')
    if tuple(sidecar.get('configuration_shape', ())) != tuple(data['configurations'].shape):
        raise ValueError('Independent configuration shape receipt mismatch')
    return data, sidecar, status


def _allocate(k_count, nrep, c_repeats, samples, depth, dimension):
    prefix, levels = (k_count, nrep, c_repeats), depth + 1
    arrays = dict(
        operators=np.empty(prefix+(levels, samples, 7)),
        orientation_operators=np.empty(prefix+(levels, samples, 7)),
        means=np.empty(prefix+(levels, 7)),
        covariance=np.empty(prefix+(levels, 7, 7)),
        cross_covariance=np.empty(prefix+(depth, 7, 7)),
        g=np.empty(prefix+(levels, dimension)),
        full_DR=np.empty(prefix+(levels, dimension, dimension)),
        r_fd_jacobian=np.empty(prefix+(levels, dimension, dimension)),
        r_of_g=np.empty(prefix+(levels, dimension)),
        r_displacement=np.empty(prefix+(levels, dimension)),
        one_step_pred=np.empty(prefix+(depth, dimension)),
        one_step_residual=np.empty(prefix+(depth, dimension)),
        all_start_horizon_pred=np.zeros(prefix+(levels, levels, dimension)),
        all_start_horizon_residual=np.zeros(prefix+(levels, levels, dimension)),
        all_start_horizon_valid=np.triu(np.ones((levels, levels), dtype=bool)),
        q_response_in=np.empty(prefix+(depth, dimension, 7)),
        q_response_out=np.empty(prefix+(depth, dimension, 7)),
        q_response_predicted=np.empty(prefix+(depth, dimension, 7)),
        q_fd_response_in=np.empty(prefix+(depth, 2, dimension, 7)),
        q_fd_response_out=np.empty(prefix+(depth, 2, dimension, 7)),
        score_std=np.empty(prefix+(levels, 7)),
        Cdraw_seeds=np.empty(prefix+(depth,), dtype=np.int64))
    for name, count in (('even', 6), ('all', 7)):
        arrays.update({f'best_J_{name}': np.empty(prefix+(depth, dimension, dimension)),
            f'response_input_rank_{name}': np.empty(prefix+(depth,), dtype=np.int64),
            f'response_input_singular_values_{name}': np.empty(prefix+(depth, dimension)),
            f'response_input_effective_rank_{name}': np.empty(prefix+(depth,)),
            f'response_input_projector_{name}': np.empty(prefix+(depth, dimension, dimension)),
            f'response_input_condition_{name}': np.empty(prefix+(depth,)),
            f'best_response_residual_{name}': np.empty(prefix+(depth, dimension, count))})
    return arrays


def _collect_model(model, configurations, K, chain_seeds, output, *, size, depth,
                   seed, c_repeats, smoke, provenance, started):
    nrep, samples = _validate_arrays(configurations, K, chain_seeds,
        size=size, depth=depth, smoke=smoke)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Require an explicit nonnegative integer diagnostic seed below 2^63')
    if type(c_repeats) is not int or c_repeats < 1 or (not smoke and c_repeats != 5):
        raise ValueError('Formal collection requires the predeclared five conditional C draws')
    torch.set_num_threads(1)
    model = model.cpu().double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    before_hash = model_sha256(model)
    try:
        patterns, probabilities = c_probability_table(model)
        # Determine d from the actual saved Q head, not a guessed decoder layout.
        first_features = local_feature_means(model, configurations[0, 0].double())
        with torch.no_grad():
            first_g = model.theory_encoder.head(first_features.mean(0))
        if first_g.ndim != 1 or first_g.numel() not in (1, 3):
            raise ValueError('Require a fixed phase2 dimg1 or dimg3 endpoint')
        dimension, levels = first_g.numel(), depth + 1
        arrays = _allocate(3, nrep, c_repeats, samples, depth, dimension)
        for ik, k_value in enumerate(K.tolist()):
            for replica in range(nrep):
                raw = configurations[ik, replica].double()
                raw_ops, raw_oriented = operator_values(raw), orientation_operator_values(raw)
                raw_features = (first_features if ik == replica == 0
                                else local_feature_means(model, raw))
                for draw in range(c_repeats):
                    ix = (ik, replica, draw)
                    fields = [raw]
                    for step in range(depth):
                        draw_seed = c_seed(seed, size, k_value, replica, draw, step)
                        arrays['Cdraw_seeds'][ix+(step,)] = draw_seed
                        generator = torch.Generator(device='cpu').manual_seed(draw_seed)
                        next_size = fields[-1].shape[-1] // 2
                        uniform = torch.rand((samples, next_size, next_size),
                            dtype=torch.float64, generator=generator)
                        fields.append(c_draw(fields[-1], probabilities, uniform))
                    ops = [raw_ops]+[operator_values(x) for x in fields[1:]]
                    oriented = [raw_oriented]+[orientation_operator_values(x) for x in fields[1:]]
                    features = [raw_features]+[local_feature_means(model, x) for x in fields[1:]]
                    arrays['operators'][ix] = torch.stack(ops).numpy()
                    arrays['orientation_operators'][ix] = torch.stack(oriented).numpy()
                    arrays['means'][ix] = torch.stack([op.mean(0) for op in ops]).numpy()
                    arrays['covariance'][ix] = torch.stack([covariance(op) for op in ops]).numpy()
                    arrays['cross_covariance'][ix] = torch.stack(
                        [covariance(ops[s+1], ops[s]) for s in range(depth)]).numpy()
                    for scale in range(levels):
                        with torch.no_grad():
                            g = model.theory_encoder.head(features[scale].mean(0))
                            rg = model.advance_theory(g)
                        arrays['g'][ix+(scale,)] = g.numpy()
                        arrays['r_of_g'][ix+(scale,)] = rg.numpy()
                        arrays['r_displacement'][ix+(scale,)] = (rg-g).numpy()
                        arrays['full_DR'][ix+(scale,)] = full_r_jacobian(model, g).numpy()
                        arrays['r_fd_jacobian'][ix+(scale,)] = full_r_finite_difference(model, g, R_FD_H).numpy()
                        arrays['score_std'][ix+(scale,)] = ops[scale].std(0, correction=0).numpy()
                    for step in range(depth):
                        _, vin, fin, std = weighted_q_response(model, features[step], ops[step],
                            t_values=Q_FD_T_VALUES)
                        _, vout, fout, _ = weighted_q_response(model, features[step+1], ops[step],
                            t_values=Q_FD_T_VALUES)
                        arrays['q_response_in'][ix+(step,)] = vin.numpy()
                        arrays['q_response_out'][ix+(step,)] = vout.numpy()
                        arrays['q_response_predicted'][ix+(step,)] = arrays['full_DR'][ix+(step,)] @ vin.numpy()
                        arrays['q_fd_response_in'][ix+(step,)] = fin.numpy()
                        arrays['q_fd_response_out'][ix+(step,)] = fout.numpy()
                        for name, count in (('even', 6), ('all', 7)):
                            fit = best_local_jacobian(vin[:, :count], vout[:, :count], std[:count])
                            for key, source in (('best_J', 'jacobian'), ('response_input_rank', 'input_rank'),
                                    ('response_input_singular_values', 'input_singular_values'),
                                    ('response_input_effective_rank', 'input_effective_rank'),
                                    ('response_input_projector', 'projector'),
                                    ('best_response_residual', 'residual')):
                                arrays[f'{key}_{name}'][ix+(step,)] = fit[source]
                            condition = fit['retained_condition_number']
                            arrays[f'response_input_condition_{name}'][ix+(step,)] = np.nan if condition is None else condition
                    with torch.no_grad():
                        for start in range(levels):
                            current = torch.from_numpy(arrays['g'][ix+(start,)].copy())
                            for end in range(start, levels):
                                if end > start:
                                    current = model.advance_theory(current)
                                arrays['all_start_horizon_pred'][ix+(start, end)] = current.numpy()
                                arrays['all_start_horizon_residual'][ix+(start, end)] = (
                                    current.numpy()-arrays['g'][ix+(end,)])
                    arrays['one_step_pred'][ix] = arrays['r_of_g'][ix][:-1]
                    arrays['one_step_residual'][ix] = arrays['one_step_pred'][ix]-arrays['g'][ix][1:]
                completed = ik*nrep+replica+1
                _write_json(output/'status.json', dict(status='running', workflow_complete=False,
                    size=size, depth=depth, theory_dim=dimension, completed_ensembles=completed,
                    total_ensembles=3*nrep, c_repeats=c_repeats, elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(size=size, K=k_value, replica=replica,
                    completed_ensembles=completed, total_ensembles=3*nrep)), flush=True)
        mean_g = arrays['g'].mean((1, 2))
        with torch.no_grad():
            arrays['r_of_mean_g'] = np.stack([np.stack([
                model.advance_theory(torch.from_numpy(row.copy())).numpy()
                for row in per_k]) for per_k in mean_g])
        arrays.update(K=K.detach().cpu().numpy(), lattice_sizes=np.array([size//2**s for s in range(levels)]),
            chain_seeds=chain_seeds.detach().cpu().numpy(), mean_g=mean_g,
            mean_r_of_g=arrays['r_of_g'].mean((1, 2)),
            c_draw_seeds=arrays['Cdraw_seeds'], operator_names=np.array(OPERATOR_NAMES),
            orientation_operator_names=np.array(ORIENTATION_NAMES), c_patterns=patterns.numpy(),
            c_p_plus=probabilities.numpy(), q_fd_t_values=np.array(Q_FD_T_VALUES), r_fd_h=np.array(R_FD_H))
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number) and 'response_input_condition_' not in key:
                if not np.isfinite(value).all():
                    raise RuntimeError(f'Nonfinite diagnostic array: {key}')
        after_hash = model_sha256(model)
        if before_hash != after_hash or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError('Read-only model state, requires_grad or parameter gradients changed')
        diagnostic = dict(status='complete', workflow_complete=True, schema_version=1,
            provenance=provenance or {}, source_run=(provenance or {}).get('source_run'),
            size=size, depth=depth, target_size=size//2**depth, theory_dim=dimension,
            K=K.tolist(), lattice_sizes=arrays['lattice_sizes'].tolist(),
            seed=seed, c_repeats=c_repeats, independent_replicas=nrep, samples_per_replica=samples,
            smoke=smoke, device='cpu', threads=1, no_training=True, no_optimizer=True,
            no_MI=True, no_new_physical_sampling=True, model_sha256_before=before_hash,
            model_sha256_after=after_hash, all_parameters_frozen=True, all_parameter_grads_none=True,
            source_data_rehashed=False, operator_names=OPERATOR_NAMES,
            operator_definitions=OPERATOR_DEFINITIONS, orientation_operator_names=ORIENTATION_NAMES,
            operator_normalization='广延算符逐构型求和；不作跨尺度面积归一。',
            covariance_definition='每独立物理系综、每固定 C 抽样分别计算 N-1 无偏协方差；不池化独立系综。',
            cross_covariance_definition='A=Cov(S_next,S_current)，矩阵行对应下一尺度算符；B=Cov(S_next,S_next)。',
            q_response_definition='同一当前尺度广延 score 对配对输入和输出构型的经验指数倾斜；Q头导数乘 ddof0 特征-score 协方差。',
            q_fd_definition='固定物理构型和 C 抽样，epsilon=t/population_std(score)，t=.001/.003；不是新采样或训练。',
            r_derivative_definition='完整 DR 包含恒等项；逐点中心差分 h=1e-5。',
            best_J_definition='每系综/抽样/transition 独立拟合六偶或七全 score 标准差归一响应；rcond=1e-6；未识别方向最小范数延拓无物理谱意义。',
            best_J_rcond=RESPONSE_RCOND, condition_nan_meaning='输入响应秩为0时保留条件数为 NaN，不伪造有限条件数。',
            closure_definition='all_start_horizon_pred 两个尺度轴是 start,end（不是 horizon）；仅 start<=end 有效，对角是原编码，下三角0；残差减 end 的真实编码。',
            r_mean_definition='r_of_mean_g=R(mean g)，mean_r_of_g=mean R(g)；两者均不同于真实下一尺度 mean g。',
            c_noise_definition='SHA256 命名空间 seed,size,K,replica,draw,transition；同物理数据跨模型共用 uniform，不同尺寸独立；不同 C 概率仍产生不同构型。',
            independence='每 K 的 R 个独立物理链是重采样单位；同一链128构型保留时间序，C重复及链内尺度均非额外独立物理样本；两个初始尺寸彼此独立。',
            arrays={key:list(value.shape) for key,value in arrays.items()},
            array_axes='前缀 K,independent_replica,C_draw；level=depth+1，transition=depth；响应随后 d,7，Q差分多2epsilon轴；闭合随后start,end,d。',
            limitations=['单训练 seed、有限尺寸、B0；有限偶算符基及数值截断；不认证独立磁指数。',
                '相同K和末尺度但不同RG历史不是相同重整化概率分布；编码差异不能单独归因Q或称闭包失败。',
                '不据最大特征值或接近理论筛选根/模式；输入响应秩不等于热模认证。',
                '这些新采样只用于固定模型验证；不调整模型、目标或训练终点。'],
            elapsed_seconds=time.monotonic()-started)
        np.savez_compressed(output/'operator_statistics.npz', **arrays)
        _write_json(output/'diagnostic.json', diagnostic)
        _write_json(output/'status.json', dict(status='complete', workflow_complete=True,
            size=size, depth=depth, theory_dim=dimension, completed_ensembles=3*nrep,
            model_sha256_before=before_hash, model_sha256_after=after_hash,
            all_parameter_grads_none=True, elapsed_seconds=time.monotonic()-started))
        return diagnostic
    except Exception as error:
        _failed(output, error, started, model, before_hash)
        raise


def collect_model(model, configurations, K, chain_seeds, output, *, size, depth,
                  seed=20261016, c_repeats=5, smoke=False, provenance=None):
    """Internal small-model test seam; production CLI always pins/validates source."""
    output, started = _new_output(output), time.monotonic()
    try:
        return _collect_model(model, configurations, K, chain_seeds, output,
            size=size, depth=depth, seed=seed, c_repeats=c_repeats, smoke=smoke,
            provenance=provenance, started=started)
    except Exception as error:
        # Preflight failures also leave a terminal receipt, never an implicit retry.
        if json.loads((output/'status.json').read_text()).get('status') != 'failed':
            _failed(output, error, started)
        raise


def collect(source_run, data_dir, output, expected_checkpoint_sha256, seed=20261016,
            c_repeats=5, smoke=False):
    source_run, data_dir = Path(source_run).resolve(), Path(data_dir).resolve()
    output, started = _new_output(output, (source_run, data_dir)), time.monotonic()
    try:
        if (not isinstance(expected_checkpoint_sha256, str)
                or len(expected_checkpoint_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_checkpoint_sha256)):
            raise ValueError('An explicit lowercase SHA256 pin of source checkpoint is mandatory')
        checkpoint = source_run/'training_step_12000.pt'
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != expected_checkpoint_sha256:
            raise ValueError('Source numbered checkpoint fingerprint mismatch')
        # Natural decoder must be reconstructed by its representation-aware loader.
        from scripts.report_ising_coupling_representation import load_representation_run, validate_formal_run
        loaded = load_representation_run(source_run, 12000)
        _, manifest, summary, representation = validate_formal_run(source_run, loaded, 12000)
        data, sidecar, status = validate_data(data_dir, smoke=smoke)
        provenance = dict(source_run=str(source_run), source_checkpoint=str(checkpoint), source_epoch=12000,
            source_checkpoint_sha256=checkpoint_sha, source_manifest_sha256=sha256_file(source_run/'manifest.json'),
            source_config=loaded.config, source_training_actual_updates=summary['actual_epochs'],
            source_labels_used_in_training=False, representation=representation,
            source_training_data_sha256_from_receipt=manifest['data_sha256'],
            source_training_metrics_file=str(source_run.parent/'report'/'metrics.json'),
            independent_data_dir=str(data_dir), independent_data_sha256_from_receipt=sidecar['data_sha256'],
            independent_sidecar_sha256=status['sidecar_sha256'],
            independent_sampling_metadata_fingerprint=sidecar['metadata_fingerprint'],
            independent_sampling_sidecar=sidecar, independent_data_rehashed=False)
        return _collect_model(loaded.model, data['configurations'], data['K'], data['chain_seeds'], output,
            size=sidecar['size'], depth=sidecar['depth'], seed=seed, c_repeats=c_repeats,
            smoke=smoke, provenance=provenance, started=started)
    except Exception as error:
        if json.loads((output/'status.json').read_text()).get('status') != 'failed':
            _failed(output, error, started)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--c-repeats', type=int, default=5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    collect(**vars(args))


if __name__ == '__main__':
    main()
