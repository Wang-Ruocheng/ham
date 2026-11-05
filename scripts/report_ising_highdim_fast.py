"""冻结 QCR 高维报告的等价 CPU 快路径与显式墙时预算。

不修改历史报告模块、模型、训练、数据、C 随机序列或求根科学口径。
旁路进度/数组位于输出目录之外；预算不足明确为 analysis_budget_hold，
不是无根、训练失败或正式完成。完成的 collection pass 可供显式恢复使用。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from scripts import report_ising_highdim as original

OPTIMIZATION_VERSION = 'ising-highdim-exact-cpu-v1'


class AnalysisBudgetHold(RuntimeError):
    """报告预算不足，不能解释成未找到根或完成。"""


class Deadline:
    def __init__(self, seconds, *, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('报告墙时预算必须为有限正数')
        self.clock, self.limit = clock, float(seconds)
        self.started = clock()

    @property
    def elapsed(self):
        return self.clock()-self.started

    def check(self):
        if self.elapsed >= self.limit:
            raise AnalysisBudgetHold(f'报告计算墙时达到 {self.limit:g} 秒预算')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(original.json_value(value), stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class Progress:
    def __init__(self, path, deadline):
        self.path, self.deadline = Path(path), deadline
        self.stage = None
        self.started = deadline.clock()
        self.stage_started = self.started
        self.timings = {}
        self.latest = {}

    def __call__(self, stage, completed=0, total=None, extra=None):
        now = self.deadline.clock()
        if stage != self.stage:
            if self.stage is not None:
                self.timings[self.stage] = self.timings.get(self.stage, 0.)+now-self.stage_started
            self.stage, self.stage_started = stage, now
        duration = now-self.stage_started
        eta = duration*(total-completed)/completed if total and completed else None
        self.latest = dict(status='running', report_complete=False, stage=stage,
            completed=completed, total=total, elapsed_seconds=self.deadline.elapsed,
            stage_elapsed_seconds=duration, stage_eta_seconds=eta,
            max_seconds=self.deadline.limit, stage_seconds=dict(self.timings),
            optimization_version=OPTIMIZATION_VERSION, extra=extra or {})
        atomic_json(self.path, self.latest)
        suffix = f'，预计本阶段余 {eta:.1f} 秒' if eta is not None else ''
        print(f'报告进度：{stage} {completed}/{total if total is not None else "?"}'
              f'，累计 {self.deadline.elapsed:.1f} 秒{suffix}', flush=True)


def _residual_layers(model):
    network = getattr(model, 'theory_rg', None)
    if not isinstance(network, torch.nn.Sequential) or len(network) != 5:
        raise ValueError('快导数仅接受已冻结的三层 Linear/Tanh/Tanh 残差 R')
    if any(not isinstance(network[index], torch.nn.Linear) for index in (0, 2, 4)) or any(
            not isinstance(network[index], torch.nn.Tanh) for index in (1, 3)):
        raise ValueError('未知 R 结构；不猜测解析导数')
    layers = [network[index] for index in (0, 2, 4)]
    if any(layer.weight.device.type != 'cpu' or layer.weight.dtype != torch.float64
           for layer in layers):
        raise ValueError('CPU 快导数只接受 float64 冻结模型')
    if layers[0].in_features != layers[-1].out_features:
        raise ValueError('残差 R 输入输出维数不符')
    return [(layer.weight.detach().numpy(), layer.bias.detach().numpy()
             if layer.bias is not None else np.zeros(layer.out_features)) for layer in layers]


class AnalyticResidualJacobian:
    """DR = I + W3 D2 W2 D1 W1；保留恒等项，绝不截断谱。"""
    def __init__(self, model, deadline):
        self.model, self.deadline = model, deadline
        self.layers = _residual_layers(model)
        self.validations = []

    def __call__(self, model, point):
        self.deadline.check()
        if model is not self.model:
            raise ValueError('解析导数不能跨冻结模型复用')
        x = np.asarray(point, dtype=np.float64)
        (w1, b1), (w2, b2), (w3, _) = self.layers
        if x.shape != (w1.shape[1],) or not np.isfinite(x).all():
            raise ValueError('完整坐标维数或有限性错误')
        a = np.tanh(w1@x+b1)
        b = np.tanh(w2@a+b2)
        result = np.eye(len(x))+(w3*(1.-b*b))@(w2*(1.-a*a))@w1
        self.deadline.check()
        return result

    def verified(self, model, point, *, h=original.FD_STEP):
        self.deadline.check()
        x = np.asarray(point, dtype=np.float64)
        if h <= 0:
            raise ValueError('中心差分步长必须为正')
        analytic = self(model, x)
        tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        autograd = torch.autograd.functional.jacobian(
            model.advance_theory, tensor).detach().cpu().numpy()
        self.deadline.check()
        directions = np.eye(len(x))*h
        finite = ((original.mapped(model, x+directions)-
                   original.mapped(model, x-directions))/(2*h)).T
        if not all(np.isfinite(value).all() for value in (analytic, autograd, finite)):
            raise ValueError('完整 DR 检查产生非有限值')
        matches_autograd = bool(np.allclose(analytic, autograd, rtol=1e-12, atol=1e-12))
        matches_finite = bool(np.allclose(analytic, finite, rtol=1e-5, atol=1e-7))
        if not matches_autograd:
            raise ValueError('解析完整 DR 与独立 autograd 不一致')
        residual = float(np.linalg.norm(original.mapped(model, x)-x))
        rank = int(np.linalg.matrix_rank(analytic-np.eye(len(x))))
        validation = dict(analytic_autograd_max_abs_error=float(np.max(abs(analytic-autograd))),
            analytic_autograd_verified=matches_autograd)
        self.validations.append(validation)
        self.deadline.check()
        return dict(jacobian=analytic.tolist(), finite_difference_jacobian=finite.tolist(),
            finite_difference_h=h, derivative_max_abs_error=float(np.max(abs(analytic-finite))),
            derivative_verified=matches_finite and matches_autograd,
            residual_norm=residual, root_verified=bool(residual <= original.ROOT_TOLERANCE),
            displacement_jacobian_rank=rank, displacement_jacobian_nullity=len(x)-rank,
            derivative_definition='解析完整 DR；独立 autograd 与中心差分校验，恒等项保留。',
            **validation)


def find_roots(model, train, knots, means, *, derivative, deadline, progress,
               partial_path=None, block_factor=2, max_seeds=original.MAX_SEEDS,
               max_nfev=original.MAX_NFEV, max_roots=original.MAX_ROOTS):
    """与历史搜索同种子/边界/阈值/顺序；仅解析求导及增加预算检查。"""
    from scipy.optimize import least_squares
    if not 1 <= max_nfev <= original.MAX_NFEV or not 1 <= max_roots <= original.MAX_ROOTS:
        raise ValueError('求根不能扩大或偷偷缩小默认预算')
    inputs = original.training_inputs(train)
    dim = inputs.shape[-1]
    seeds, indices = original.select_training_seeds(train, max_seeds=max_seeds)
    span = np.maximum(np.ptp(inputs, axis=0), .1)
    lower, upper = inputs.min(0)-.25*span, inputs.max(0)+.25*span
    attempts, accepted, records = [], [], []
    dedup = max(float(np.linalg.norm(span))*1e-6, 1e-8)

    def save_partial():
        if partial_path is not None:
            atomic_json(partial_path, dict(status='partial', report_complete=False,
                attempts=attempts, all_accepted_root_coordinates=accepted, roots=records,
                search_box=[lower, upper], seed_flat_training_input_indices=indices,
                acceptance_residual_l2=original.ROOT_TOLERANCE,
                max_nfev_per_seed=max_nfev, max_analyzed_roots=max_roots,
                search_complete=False, exhaustive_search_claim=False, thermal_nu=None,
                caution='未完成全部预算时不得把空列表解释为不存在根。'))

    def residual(point):
        deadline.check()
        result = original.mapped(model, point)-point
        deadline.check()
        return result

    def jacobian(point):
        return derivative(model, point)-np.eye(dim)

    try:
        # Validate the actual frozen network before its analytic Jacobian can
        # influence even the first solver step. This is a TRAIN point, not a
        # new physical perturbation or a root chosen with theoretical labels.
        deadline.check()
        preflight = derivative.verified(model, seeds[0])
        if not preflight['derivative_verified']:
            raise ValueError('TRAIN种子处解析导数未通过独立autograd/中心差分预检')
        for ordinal, (index, guess) in enumerate(zip(indices, seeds)):
            deadline.check()
            progress('完整坐标求根', ordinal, len(seeds))
            solution = least_squares(residual, guess, jac=jacobian, bounds=(lower, upper),
                max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            point = np.asarray(solution.x)
            norm = float(np.linalg.norm(residual(point)))
            inside = bool(np.isfinite(point).all() and np.all(point >= lower) and np.all(point <= upper))
            verified = inside and math.isfinite(norm) and norm <= original.ROOT_TOLERANCE
            unique = verified and not any(np.linalg.norm(point-old) <= dedup for old in accepted)
            if unique:
                accepted.append(point)
            attempts.append(dict(training_flat_input_index=int(index), solver_success=bool(solution.success),
                status=int(solution.status), nfev=int(solution.nfev),
                njev=int(solution.njev) if solution.njev is not None else None,
                residual_norm=norm, inside_search_box=inside, full_coordinate_root_verified=verified,
                new_unique_root=bool(unique)))
            save_partial()
        progress('完整坐标求根', len(seeds), len(seeds))
        retained = accepted[:max_roots]
        for index, point in enumerate(retained):
            deadline.check()
            progress('完整导数与全谱', index, len(retained))
            derivative_record = derivative.verified(model, point)
            support = original.training_support(point, train)
            deadline.check()
            projections = [original.curve_projections(knots, means[:, scale], point)
                           for scale in range(means.shape[1])]
            spectra = original.eigenvalue_records(derivative_record['jacobian'], projections,
                block_factor=block_factor,
                verified=derivative_record['derivative_verified'] and derivative_record['root_verified'],
                support_candidate=support['geometric_support_candidate'])
            records.append(dict(root_id=f'F{index+1}', g=point.tolist(), **derivative_record,
                **support, **original.stability_counts(derivative_record['jacobian']),
                eigenvalues=spectra, curve_projections_by_scale=projections,
                selected_thermal_eigenvalue=None, thermal_nu=None, isolated_root_certified=False,
                root_qualification='全坐标残差验收点；可能属于非孤立根集；投影交点不作根证据。'))
            save_partial()
            deadline.check()
        progress('完整导数与全谱', len(retained), len(retained))
    except AnalysisBudgetHold:
        save_partial()
        raise
    return dict(roots=records, search_box=[lower.tolist(), upper.tolist()],
        bounds_source='TRAIN direct[:, :-1] only; 25% coordinate-span expansion; span floor 0.1',
        seed_count=len(seeds), seed_flat_training_input_indices=indices.tolist(),
        seed_rule='最多64个TRAIN输入按固定文件次序确定性等距抽样；不用标签、TEST或理论',
        max_seed_count=max_seeds, max_nfev_per_seed=max_nfev, max_analyzed_roots=max_roots,
        acceptance_residual_l2=original.ROOT_TOLERANCE, root_dedup_tolerance=dedup,
        solver_success_count=sum(row['solver_success'] for row in attempts), attempts=attempts,
        accepted_unique_root_count=len(accepted), retained_root_count=len(retained),
        all_accepted_root_coordinates=[point.tolist() for point in accepted],
        all_accepted_coordinates_retained=True, root_analysis_truncated=bool(len(accepted) > max_roots),
        truncated_unique_root_count=max(0, len(accepted)-max_roots),
        search_complete=False, exhaustive_search_claim=False, thermal_nu=None,
        note='非穷尽有界多起点搜索；没有发现根不证明不存在根。超过16个验收点时按种子遇到次序截断，不能声称无其他根。')


@contextmanager
def fast_report_context(model, deadline, progress, partial_path=None):
    """仅在当前报告进程临时接入；退出即还原历史模块对象。"""
    derivative = AnalyticResidualJacobian(model, deadline)
    names = ('_full_jacobian', 'full_r_jacobian', 'find_roots', 'plot_maps',
             'plot_losses', 'plot_representation', 'plot_spectrum', 'closure_details',
             'architecture_diagnostics', 'representation_statistics', 'energy_path_diagnostics')
    saved = {name: getattr(original, name) for name in names}

    def roots(*args, **kwargs):
        return find_roots(*args, derivative=derivative, deadline=deadline,
                          progress=progress, partial_path=partial_path, **kwargs)

    def guarded(function, name):
        def call(*args, **kwargs):
            deadline.check()
            progress(name, 0, 1)
            value = function(*args, **kwargs)
            deadline.check()
            progress(name, 1, 1)
            return value
        return call

    original._full_jacobian = derivative
    original.full_r_jacobian = derivative.verified
    original.find_roots = roots
    labels = dict(plot_maps='流场与耦合图', plot_losses='既有损失曲线',
        plot_representation='表示与闭合图', plot_spectrum='完整谱图',
        closure_details='闭合数值统计', architecture_diagnostics='架构秩诊断',
        representation_statistics='表示数值统计', energy_path_diagnostics='耦合路径诊断')
    for name, label in labels.items():
        setattr(original, name, guarded(saved[name], label))
    try:
        yield derivative
    finally:
        for name, function in saved.items():
            setattr(original, name, function)


def _assert_frozen(loaded, checkpoint_sha, model_sha):
    state_after = original.model_state_sha256(loaded.model)
    checkpoint_after = original.sha256(loaded.checkpoint)
    gradients_none = all(parameter.grad is None for parameter in loaded.model.parameters())
    if checkpoint_after != checkpoint_sha or state_after != model_sha or not gradients_none:
        raise ValueError('报告前后冻结指纹或 gradNone 不符')
    return dict(checkpoint_sha256_before=checkpoint_sha, checkpoint_sha256_after=checkpoint_after,
        checkpoint_unchanged=True, model_state_sha256_before=model_sha,
        model_state_sha256_after=state_after, model_state_unchanged=True,
        model_gradients_none=gradients_none)


def budget_hold(output, progress, deadline, *, partial_files, freeze_verification=None):
    """旧 write_report 可能写过 failed；这里明确纠正为预算门控并取消完成标记。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = dict(status='analysis_budget_hold', report_complete=False,
        requires_user_action=True, training_started=False, physical_sampling_started=False,
        MI_estimated=False, thermal_nu=None, max_seconds=deadline.limit,
        elapsed_seconds=deadline.elapsed, latest_stage=progress.latest,
        partial_files=[str(path) for path in partial_files if Path(path).is_file()],
        freeze_verification=freeze_verification,
        optimization_version=OPTIMIZATION_VERSION,
        caution='报告预算不足；不等于没有根、训练失败或完整科学报告。不得自动减样本、重复或求根预算。')
    atomic_json(output/'status.json', record)
    atomic_json(output/'report.json', record)
    atomic_json(progress.path, record)
    print('报告预算门控：已保留完成的编码和求根中间结果；正式报告尚未完成。', flush=True)
    return record


@contextmanager
def single_thread_control():
    """不安装依赖；线程限制不可用时只有已锁定启动环境才允许继续。"""
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')
    environment = {name: os.environ.get(name) for name in names}
    startup_locked = all(value == '1' for value in environment.values())
    try:
        from threadpoolctl import threadpool_limits
        control = threadpool_limits(limits=1)
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        if not startup_locked:
            raise RuntimeError('缺少可用threadpoolctl且启动线程环境未全部锁为1；拒绝不受控CPU报告') from error
        yield dict(method='已验证启动环境四项均为1，Torch显式单线程',
                   startup_environment=environment, torch_num_threads=torch.get_num_threads(),
                   optional_threadpoolctl=f'{type(error).__name__}: {error}')
    else:
        with control:
            yield dict(method='threadpoolctl与Torch显式单线程',
                       startup_environment=environment, torch_num_threads=torch.get_num_threads())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', default=12000, type=int)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--sampling-repeats', default=5, type=int)
    parser.add_argument('--max-seconds', default=900., type=float)
    parser.add_argument('--progress-path', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.threads != 1 or args.sampling_repeats != 5 or args.checkpoint != 12000:
        parser.error('正式快路径固定 CPU1线程、5次额外TEST C重复及实际12000终点')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('拒绝覆盖既有正式/部分报告；恢复须另行显式授权与新输出目录')
    if args.progress_path == args.output or args.output in args.progress_path.parents:
        parser.error('进度必须位于报告目录之外，避免预建正式目录')
    deadline = Deadline(args.max_seconds)
    progress = Progress(args.progress_path, deadline)
    torch.set_num_threads(1)
    # SciPy/NumPy LAPACK must not secretly create a many-core root solver.
    with single_thread_control() as thread_control:
        args.thread_control = thread_control
        return _run(args, deadline, progress)


def _run(args, deadline, progress):
    from scripts.ising_highdim_fast_evaluation import FrozenBinaryCollector
    completed, repeat_statistics, partial_files = {}, [], []
    prefix = args.progress_path.with_suffix('')
    collection_path = Path(str(prefix)+'.collection.npz')
    collection_receipt = Path(str(prefix)+'.collection.json')
    partial_root_path = Path(str(prefix)+'.roots.partial.json')
    partial_files.extend([collection_path, collection_receipt, partial_root_path])
    loaded = collector = None
    checkpoint_before = state_before = None
    data_sha256 = active_pass = None
    collection_labels = {}

    def save_collection():
        arrays = {f'{name}__{key}': value for name, draw in completed.items()
                  for key, value in draw.items()}
        arrays.update(collection_labels)
        # Partial active pass is useful evidence; only a collector with a saved
        # exact RNG/offset receipt may resume it, never silently restart it.
        if collector is not None:
            for key, value in getattr(collector, 'partial_arrays', {}).items():
                arrays[f'active__{key}'] = value
            rng_state = getattr(collector, 'partial_rng_state', None)
            if rng_state is not None:
                arrays['active__torch_cpu_rng_state'] = rng_state.cpu().numpy()
        if arrays:
            atomic_npz(collection_path, **arrays)
            atomic_json(collection_receipt, dict(status='partial_collection', report_complete=False,
                checkpoint_sha256=checkpoint_before, model_state_sha256=state_before,
                data_sha256=data_sha256, active_pass=active_pass,
                completed_passes=list(completed), base_seed=12345, repeat_seeds=list(range(5)),
                chunk_size=16, samples=128, array_sha256=original.sha256(collection_path),
                completed_repeat_statistics=repeat_statistics,
                partial_resume=getattr(collector, 'partial_options', None),
                caution='已完成pass可显式复用；活动pass只可用其精确RNG与offset恢复，不得当作完整数据。'))

    try:
        progress('校验并加载冻结终点', 0, 1)
        checkpoint_path = args.run/f'training_step_{args.checkpoint}.pt'
        checkpoint_before = original.sha256(checkpoint_path)
        deadline.check()
        loaded = load_run(args.run, args.checkpoint)
        cfg = RunConfig.from_dict(loaded.config)
        cfg.validate()
        if (cfg.model.dimension != 2 or cfg.model.theory_dim < 1 or cfg.model.categorical_states or
                cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',)):
            raise ValueError('仅接受已记录的单层 K-only 二维 Ising')
        summary = json.loads((args.run/'summary.json').read_text())
        original.validate_completed_run(cfg, summary, loaded.payload, args.checkpoint)
        metadata = loaded.payload.get('metadata', {})
        if metadata.get('labels_used_in_training') is not False or metadata.get('q_supervision') not in (None, {}):
            raise ValueError('要求明确的无标签训练来源')
        sidecar = json.loads(Path(cfg.data.cache).with_suffix('.json').read_text())
        manifest = json.loads((args.run/'manifest.json').read_text())
        if (sidecar.get('B') != 0. or sidecar.get('coupling_names') != ['K'] or
                not manifest.get('data_sha256') or sidecar.get('data_sha256') != manifest['data_sha256']):
            raise ValueError('B0 K-only 数据指纹不符')
        data_sha256 = manifest['data_sha256']
        state_before = original.model_state_sha256(loaded.model)
        _assert_frozen(loaded, checkpoint_before, state_before)
        if cfg.evaluation.samples != 128:
            raise ValueError('正式报告必须保持每系综128构型')
        deadline.check()
        train_data, test_data = original.report_datasets(cfg)
        if len(train_data.configurations) != 516 or len(test_data.configurations) != 252:
            raise ValueError('正式报告必须保留完整516 TRAIN及252 TEST系综')
        collection_labels.update(train_labels=train_data.couplings.numpy(),
                                 test_labels=test_data.couplings.numpy())
        deadline.check()
        progress('校验并加载冻结终点', 1, 1)
        current_pass = ['']

        def collection_progress(stage, completed_count, total, extra=None):
            progress('编码 '+current_pass[0], completed_count, total, extra)

        collector = FrozenBinaryCollector(loaded.model, progress=collection_progress, deadline=deadline)
        options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
            spin_flip_augment=cfg.data.spin_flip_augment, chunk_size=16)
        passes = [('train', train_data.configurations, 12345),
                  ('test', test_data.configurations, 12345)]+[
                      (f'repeat_{seed}', test_data.configurations, seed) for seed in range(5)]
        for name, configurations, seed in passes:
            deadline.check()
            current_pass[0] = name
            active_pass = name
            completed[name] = collector.collect(configurations, sampling_seed=seed, **options)
            if name.startswith('repeat_'):
                repeat_statistics.append(original.closure_details(loaded.model,
                    completed[name], completed['train']['direct']))
            save_collection()
            deadline.check()
        _assert_frozen(loaded, checkpoint_before, state_before)
        history = loaded.payload.get('history')
        if history is None:
            history = json.loads((args.run/'history_progress.json').read_text())
        history = [row for row in history if row.get('epoch', 0) <= args.checkpoint]
        progress('生成完整正式报告', 0, 1)
        with fast_report_context(loaded.model, deadline, progress, partial_root_path) as derivative:
            report = original.write_report(loaded, completed['train'], completed['test'],
                train_data.couplings.numpy(), test_data.couplings.numpy(), history, args.output,
                repeats=dict(seeds=list(range(5)), closure=repeat_statistics,
                    caution='固定模型的随机 C 重复，不是独立训练 seed、物理链或谱置信区间。'),
                data_sha256=manifest['data_sha256'], expected_checkpoint_sha256=checkpoint_before,
                lattice_sizes=[train_data.configurations.shape[2]//cfg.model.block_factor**scale
                    for scale in range(completed['train']['direct'].shape[1])])
        deadline.check()
        freeze = _assert_frozen(loaded, checkpoint_before, state_before)
        optimization = dict(version=OPTIMIZATION_VERSION, cpu_threads=1,
            thread_control=args.thread_control,
            elapsed_seconds=deadline.elapsed, max_seconds=deadline.limit,
            stage_seconds=progress.timings, collector='等价二元局部统计CPU快路径',
            unused_split_encoding_omitted=True, microscopic_encoding_reused=True,
            train_ensemble_count=516, test_ensemble_count=252, samples=128,
            chunk_size=16, base_seed=12345, extra_test_seeds=list(range(5)),
            analytic_DR_validations=derivative.validations,
            search_seed_cap=64, search_nfev_cap=200, analyzed_root_cap=16,
            budget_caution='若超时则analysis_budget_hold，不能解读为无根或正式完成。')
        report['optimization'] = optimization
        for filename in ('metrics.json', 'critical_exponents.json'):
            atomic_json(args.output/filename, report)
        for filename in ('status.json', 'report.json'):
            completion = json.loads((args.output/filename).read_text())
            completion['optimization'] = optimization
            completion.update(freeze)
            atomic_json(args.output/filename, completion)
        progress('正式报告完成', 1, 1)
        progress.latest.update(status='complete', report_complete=True)
        atomic_json(progress.path, progress.latest)
        deadline.check()
        return report
    except AnalysisBudgetHold:
        save_collection()
        freeze = (_assert_frozen(loaded, checkpoint_before, state_before)
                  if loaded is not None and state_before is not None else None)
        return budget_hold(args.output, progress, deadline, partial_files=partial_files,
                           freeze_verification=freeze)


if __name__ == '__main__':
    result = main()
    if isinstance(result, dict) and result.get('status') == 'analysis_budget_hold':
        raise SystemExit(75)
