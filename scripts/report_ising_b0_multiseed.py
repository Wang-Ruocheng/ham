"""B=0、d_g=5 的固定 12000 步多种子冻结报告。

不采样物理构型、不重新估计 MI，不 hash 历史大物理缓存。每个模型只
使用自己的 TRAIN 标准差归一化；不同训练种子的原始潜变量坐标不平均。
额外 C 重复先按物理链平均平方误差，不充当新的独立物理链。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from scripts import report_ising_highdim as original
from scripts.ising_highdim_fast_evaluation import FrozenBinaryCollector
from scripts.ising_qcr_evaluation import (
    HALF_SUBSET_ESTIMATOR, collection_estimator, encoding_contract,
    require_matching_estimators,
)
from scripts.report_ising_highdim_fast import (
    AnalysisBudgetHold, Deadline, Progress, _assert_frozen, atomic_json,
    atomic_npz, budget_hold, fast_report_context, single_thread_control,
)

BASE_C_SEED = 12345
EXTRA_C_SEEDS = (0, 1)
EXPECTED_SEEDS = (91, 92, 93, 94, 95)


class TrainingMatchedBinaryCollector(FrozenBinaryCollector):
    """同一随机 C 构型轨迹上，使用训练闭合的两半 Q 均值估计器。

    只改变冻结编码函数，不改变 C 输入、输出、调用次数或随机数消耗。
    固定划分跨物理链、尺度和 C 重复共用；不修改历史全体 Q collector。
    """

    def __init__(self, model, *, samples=128, **kwargs):
        self.encoding_protocol = encoding_contract(samples)
        self._sample_partition = np.asarray(self.encoding_protocol['sample_partition'], dtype=np.int64)
        super().__init__(model, **kwargs)

    def _semantics(self):
        return (super()._semantics(), HALF_SUBSET_ESTIMATOR,
                tuple(self.encoding_protocol['sample_partition']))

    def _encode(self, bits):
        samples = self.encoding_protocol['samples']
        if bits.shape[1] != samples:
            raise ValueError('半子集编码必须使用协议中的相同构型数；不允许隐式增广或全体 Q 回退')
        midpoint = samples // 2
        first = super()._encode(bits[:, self._sample_partition[:midpoint]])
        second = super()._encode(bits[:, self._sample_partition[midpoint:]])
        return .5 * (first + second)


def encoding_metadata():
    return dict(samples=128, encoding_estimator=HALF_SUBSET_ESTIMATOR,
                encoding_protocol=encoding_contract(128))


def encoding_array_metadata():
    metadata = encoding_metadata()
    return dict(samples=np.asarray(128),
        encoding_estimator=np.asarray(metadata['encoding_estimator']),
        encoding_protocol=np.asarray(json.dumps(metadata['encoding_protocol'],
            ensure_ascii=False, sort_keys=True)))


def validate_encoding_metadata(value):
    expected = encoding_metadata()
    if (collection_estimator(value) != HALF_SUBSET_ESTIMATOR or
            any(value.get(key) != item for key, item in expected.items())):
        raise ValueError('本轮要求固定128构型划分的训练匹配半子集估计器；禁止混入全体 Q 或其他划分')


def file_stat(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'需要既有普通文件：{path}')
    row = path.stat()
    return dict(size=row.st_size, inode=row.st_ino, device=row.st_dev,
                mtime_ns=row.st_mtime_ns)


def path_errors(model, trajectories):
    """每物理链、每路径、每坐标的平方误差；恒等基准为 g_start。"""
    direct = np.asarray(trajectories['direct'], dtype=np.float64)
    if direct.ndim != 3 or direct.shape[1] < 2 or not np.isfinite(direct).all():
        raise ValueError('需要有限的 [物理链,尺度,坐标] 编码')
    paths, predicted, identity = [], [], []
    for start in range(direct.shape[1] - 1):
        prediction = direct[:, start]
        for end in range(start + 1, direct.shape[1]):
            prediction = original.mapped(model, prediction)
            paths.append((start, end))
            predicted.append((prediction - direct[:, end]) ** 2)
            identity.append((direct[:, start] - direct[:, end]) ** 2)
    return dict(paths=paths, R=np.stack(predicted, axis=1),
                identity=np.stack(identity, axis=1))


def grouped_path_metrics(errors, reference_train):
    """先平均 C 的平方误差，再平均物理链/坐标；不平均跨模型坐标。"""
    if not errors:
        raise ValueError('至少需要一个完整的固定 C 重复')
    reference = np.asarray(reference_train, dtype=np.float64)
    paths = errors[0]['paths']
    shape = errors[0]['R'].shape
    if (reference.ndim != 3 or not np.isfinite(reference).all() or
            any(row['paths'] != paths or row['R'].shape != shape or
                row['identity'].shape != shape for row in errors)):
        raise ValueError('重复必须保留相同物理链顺序、完整路径及模型坐标')
    sigma = reference.std(axis=0, ddof=0)
    result = []
    for path_index, (start, end) in enumerate(paths):
        row = dict(start=start, end=end,
                   target_train_rms_coordinate_std=float(np.sqrt(np.mean(sigma[end] ** 2))))
        for method in ('R', 'identity'):
            per_chain = np.mean([draw[method][:, path_index] for draw in errors], axis=0)
            if not np.isfinite(per_chain).all() or np.any(per_chain < 0):
                raise ValueError('逐链平方误差必须有限且非负')
            coordinate = np.sqrt(per_chain.mean(axis=0))
            rmse = float(np.sqrt(per_chain.mean()))
            scale = row['target_train_rms_coordinate_std']
            row[method] = dict(rmse=rmse, nrmse=rmse / scale if scale > 1e-12 else None,
                coordinate_rmse=coordinate.tolist(),
                coordinate_nrmse=[float(x / y) if y > 1e-12 else None
                                  for x, y in zip(coordinate, sigma[end])])
        baseline = row['identity']['rmse']
        row['R_to_identity_rmse_ratio'] = row['R']['rmse'] / baseline if baseline > 1e-12 else None
        result.append(row)
    return dict(paths=result, physical_chain_count=shape[0], C_draw_count=len(errors),
        independent_sample_count=shape[0],
        TRAIN_target_coordinate_std=sigma.tolist(), normalization_source='本模型固定基准 C 编码的 TRAIN；ddof=0',
        aggregation='先在每条物理链内平均 C 重复的平方误差，再平均物理链及坐标并开方；不将 C 重复视为独立链。',
        identity_definition='每条路径用 g_start 预测 g_end；与同一 C 编码的 R 预测配对。')


def validate_provenance(cfg, summary, payload, manifest, sidecar):
    original.validate_completed_run(cfg, summary, payload, 12000)
    metadata = payload.get('metadata', {})
    if (cfg.schema_version != 5 or cfg.model.dimension != 2 or cfg.model.theory_dim != 5 or
            cfg.model.block_factor != 3 or cfg.model.categorical_states or
            cfg.model.fixed_kernel or cfg.model.kernel_initialization != 'random' or
            cfg.model.kernel_sampling != 'bernoulli' or cfg.model.kernel_layout != 'nonoverlapping' or
            cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',) or
            cfg.evaluation.samples != 128 or cfg.training.max_rg_steps != 2 or
            cfg.evaluation.max_rg_steps != 2 or cfg.training.gradient_mode != 'qc_r_separate' or
            cfg.training.seed not in EXPECTED_SEEDS or cfg.data.spin_flip_augment):
        raise ValueError('本轮报告要求 schema5、d_g=5、随机可训练 b3、深度2、128构型及独立梯度')
    weights = cfg.loss.closure_scale_weights
    if weights is not None and (len(weights) != 2 or not np.allclose(weights, weights[0]) or weights[0] <= 0):
        raise ValueError('本轮主基准要求均匀尺度权重')
    if (metadata.get('labels_used_in_training') is not False or
            manifest.get('labels_used_in_training') is not False or
            metadata.get('q_supervision') not in (None, {}) or manifest.get('q_supervision') not in (None, {}) or
            manifest.get('source') is not None or metadata.get('source') is not None or
            manifest.get('resumed_from') is not None or metadata.get('resumed_from') is not None or
            manifest.get('frozen_C') or metadata.get('frozen_C') or
            manifest.get('data_driven_initialization') or
            sidecar.get('B') != 0. or sidecar.get('coupling_names') != ['K'] or
            sidecar.get('sites') != 108 or sidecar.get('train_shape') != [516, 128, 108, 108, 1] or
            sidecar.get('test_shape') != [252, 128, 108, 108, 1] or
            not manifest.get('data_sha256') or sidecar.get('data_sha256') != manifest['data_sha256']):
        raise ValueError('要求全新无监督模型以及完整 B=0、L108 数据来源，不能复用监督或续训模型')
    if original.json_value(manifest.get('config')) != original.json_value(cfg.to_dict()):
        raise ValueError('manifest 配置与冻结终点配置不一致')


def sampling_quality(sidecar, *, strict=False):
    """仅读取公开侧车，不加载原始物理构型或重算自相关。"""
    ensembles = sidecar.get('ensembles', [])
    if strict and (len(ensembles) != 768 or any(
            not all(name in row and 'ess_estimate' in row[name] for name in
                    ('bond_sum_per_site', 'magnetization_squared')) for row in ensembles)):
        raise ValueError('公开采样侧车缺少全部768条链的原始 ESS 诊断；不能静默输出空质量报告')
    result = dict(equilibrium_certified=False,
        qualification=sidecar.get('uncertainty_note'), options=sidecar.get('sampling_options'),
        production_chain_count=len(ensembles), saved_hot_cold_pilots=sidecar.get('pilot'), observables={})
    for name in ('bond_sum_per_site', 'magnetization_squared'):
        values = [row[name]['ess_estimate'] for row in sidecar.get('ensembles', []) if name in row]
        if values:
            result['observables'][name] = dict(minimum_saved_ESS=float(min(values)),
                saved_ESS_below_32=sum(x < 32 for x in values), chain_count=len(values))
    return result


def saved_mi_diagnostics(manifest):
    """兼容已公开字段名，但缺少完整诊断时显式失败而非写 null。"""
    values = manifest.get('empirical_statistics')
    field = 'empirical_statistics'
    if values is None:
        values = manifest.get('empirical_mi_diagnostics')
        field = 'empirical_mi_diagnostics'
    if not isinstance(values, dict) or any(not isinstance(values.get(split), dict)
            or values[split].get('visible_sites') != 9
            or values[split].get('environment_sites') != 24
            or values[split].get('buffer') != 1 or values[split].get('states') != 2
            for split in ('train', 'test')):
        raise ValueError('缺少完整 b3/buffer1 的既有 TRAIN/TEST 经验 MI 诊断；不允许报告中重估 MI')
    return dict(source_field=field, values=values, reestimated=False,
        caution='精确经验直方图 MI 不是总体 MI；稀有环境类别可能引起有限样本向上偏差。')


def run_report(args, deadline, progress):
    output, run = Path(args.output), Path(args.run)
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖已有正式或部分报告')
    loaded = None
    writer_started = False
    checkpoint_before = state_before = None
    completed = {}
    partial = output.with_name(output.name + '.collection.partial.npz')
    try:
        progress('读取冻结正式终点', 0, 1)
        checkpoint = run / 'training_step_12000.pt'
        checkpoint_before = original.sha256(checkpoint)
        loaded = load_run(run, 12000)
        cfg = RunConfig.from_dict(loaded.config)
        cfg.validate()
        manifest = json.loads((run / 'manifest.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        sidecar = json.loads(Path(cfg.data.cache).with_suffix('.json').read_text())
        validate_provenance(cfg, summary, loaded.payload, manifest, sidecar)
        quality = sampling_quality(sidecar, strict=True)
        mi_diagnostics = saved_mi_diagnostics(manifest)
        source_files = [Path(cfg.data.cache)]
        source_files += [Path(row['path']) for row in manifest.get('execution_optimizations', {}).get(
            'mi_statistics_cache', {}).values() if row.get('path')]
        before_stat = {str(path): file_stat(path) for path in source_files}
        state_before = original.model_state_sha256(loaded.model)
        _assert_frozen(loaded, checkpoint_before, state_before)
        deadline.check()
        train_data, test_data = original.report_datasets(cfg)
        if (tuple(train_data.configurations.shape) != (516, 128, 108, 108, 1) or
                tuple(test_data.configurations.shape) != (252, 128, 108, 108, 1)):
            raise ValueError('必须评价完整516 TRAIN和252 TEST物理链，每链128构型')
        collector = TrainingMatchedBinaryCollector(loaded.model, samples=128, deadline=deadline,
            progress=lambda stage, n, total, extra: progress('固定模型编码', n, total, extra))
        options = dict(samples=128, max_rg_steps=2, spin_flip_augment=cfg.data.spin_flip_augment,
                       chunk_size=16, sampling_seed=BASE_C_SEED)
        completed['train'] = collector.collect(train_data.configurations, **options)
        completed['test'] = collector.collect(test_data.configurations, **options)
        train_errors = [path_errors(loaded.model, completed['train'])]
        test_errors = [path_errors(loaded.model, completed['test'])]
        repeat_closure = []
        for seed in EXTRA_C_SEEDS:
            deadline.check()
            draw = collector.collect(test_data.configurations, **dict(options, sampling_seed=seed))
            test_errors.append(path_errors(loaded.model, draw))
            repeat_closure.append(original.closure_details(loaded.model, draw, completed['train']['direct']))
            del draw
        _assert_frozen(loaded, checkpoint_before, state_before)
        history = loaded.payload.get('history') or json.loads((run / 'history_progress.json').read_text())
        history = [row for row in history if row.get('epoch', 0) <= 12000]
        if output.exists() or output.is_symlink():
            raise FileExistsError('编码期间输出目录被其他进程创建；保留其内容')
        writer_started = True
        with fast_report_context(loaded.model, deadline, progress):
            report = original.write_report(loaded, completed['train'], completed['test'],
                train_data.couplings.numpy(), test_data.couplings.numpy(), history, output,
                repeats=dict(seeds=list(EXTRA_C_SEEDS), closure=repeat_closure,
                    caution='仅两个额外固定模型 C 重复，不是额外物理链或训练种子。'),
                data_sha256=manifest['data_sha256'], expected_checkpoint_sha256=checkpoint_before,
                lattice_sizes=[108, 36, 12])
        grouped = dict(train=grouped_path_metrics(train_errors, completed['train']['direct']),
                       test=grouped_path_metrics(test_errors, completed['train']['direct']),
                       **encoding_metadata())
        freeze = _assert_frozen(loaded, checkpoint_before, state_before)
        if before_stat != {str(path): file_stat(path) for path in source_files}:
            raise ValueError('历史物理/MI文件 stat 发生变化；不重hash或覆盖')
        report.update(schema_version=2, experiment='ising2d_B0_multiseed_stageA_v1', training_seed=cfg.training.seed,
            grouped_closure_identity=grouped, thermal_nu=None,
            physical_sampling_quality=quality,
            saved_MI_diagnostics=mi_diagnostics,
            sampling=dict(base_C_seed=BASE_C_SEED, extra_TEST_C_seeds=list(EXTRA_C_SEEDS),
                TRAIN_chains=516, TEST_chains=252, configurations_per_chain=128,
                additional_independent_physical_chains=0),
            analysis=dict(max_seconds=deadline.limit, elapsed_seconds=deadline.elapsed,
                CPU_threads=1, raw_physical_data_rehashed=False, MI_reestimated=False,
                sampling_warning='固定数据的训练种子消融不涵盖物理采样种子不确定性。'),
            immutable_source_stat=before_stat, freeze_verification=freeze, **encoding_metadata())
        report['fixed_points'].update(encoding_metadata())
        atomic_json(output / 'fixed_points.json', report['fixed_points'])
        for name in ('encodings.npz', 'trajectories.npz', 'fixed_point_arrays.npz'):
            with np.load(output / name, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            atomic_npz(output / name, **values, **encoding_array_metadata())
        atomic_json(output / 'closure_identity.json', grouped)
        atomic_npz(output / 'closure_identity_arrays.npz', paths=np.asarray(test_errors[0]['paths']),
            C_seeds=np.asarray((BASE_C_SEED,) + EXTRA_C_SEEDS),
            TEST_R_squared_error=np.stack([row['R'] for row in test_errors]),
            TEST_identity_squared_error=np.stack([row['identity'] for row in test_errors]),
            TRAIN_target_coordinate_std=np.asarray(grouped['train']['TRAIN_target_coordinate_std']),
            **encoding_array_metadata())
        for name in ('metrics.json', 'critical_exponents.json'):
            atomic_json(output / name, report)
        deadline.check()
        public = sorted(path.name for path in output.iterdir() if path.is_file() and
                        path.suffix.lower() in ('.json', '.npz', '.png'))
        completion = dict(status='complete', report_complete=True, checkpoint_epoch=12000,
            theory_dim=5, training_seed=cfg.training.seed, public_files=public,
            training_started=False, physical_sampling_started=False, MI_estimated=False,
            thermal_nu=None, **freeze, **encoding_metadata())
        atomic_json(output / 'report.json', completion)
        atomic_json(output / 'status.json', completion)
        progress('正式报告完成', 1, 1)
        return report
    except AnalysisBudgetHold:
        if not writer_started and (output.exists() or output.is_symlink()):
            raise FileExistsError('预算门控时发现非本次创建的输出目录；不覆盖')
        if completed:
            atomic_npz(partial, **{f'{name}__{key}': value
                for name, arrays in completed.items() for key, value in arrays.items()},
                **encoding_array_metadata())
        freeze = (_assert_frozen(loaded, checkpoint_before, state_before)
                  if loaded is not None and state_before is not None else None)
        held = budget_hold(output, progress, deadline, partial_files=[partial], freeze_verification=freeze)
        held.update(encoding_metadata())
        for path in (output / 'report.json', output / 'status.json', progress.path):
            atomic_json(path, held)
        return held
    except Exception as error:
        # The reused writer may have finished before an outer stat/freeze check
        # fails; never leave that provisional completion as the formal result.
        if isinstance(error, FileExistsError) or (not writer_started and output.exists()):
            raise
        output.mkdir(parents=True, exist_ok=True)
        failed = dict(status='failed', report_complete=False, requires_user_action=True,
            error=f'{type(error).__name__}: {error}', training_started=False,
            physical_sampling_started=False, MI_estimated=False, thermal_nu=None, **encoding_metadata())
        atomic_json(output / 'report.json', failed)
        atomic_json(output / 'status.json', failed)
        raise


def _summary(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
        std_ddof1=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        min=float(min(values)) if values else None, max=float(max(values)) if values else None)


def compare_reports(specifications, output):
    """只汇总完整五个训练种子；不挑最佳，不平均不同种子的原始 g。"""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖已有比较报告')
    reports = {}
    for specification in specifications:
        name, folder = specification.split('=', 1)
        if not name or name in reports:
            raise ValueError('比较名称必须唯一且非空')
        folder = Path(folder)
        completion = json.loads((folder / 'report.json').read_text())
        report = json.loads((folder / 'metrics.json').read_text())
        validate_encoding_metadata(completion)
        validate_encoding_metadata(report)
        if (completion.get('status') != 'complete' or not completion.get('report_complete') or
                report.get('checkpoint_epoch') != 12000 or report.get('theory_dim') != 5 or
                report.get('experiment') != 'ising2d_B0_multiseed_stageA_v1' or
                report.get('labels_used_in_training') is not False or report.get('thermal_nu') is not None or
                not report.get('freeze_verification', {}).get('model_state_unchanged')):
            raise ValueError('需要完整、冻结且同为 d_g=5 的12000步无监督正式报告')
        reports[name] = report
    require_matching_estimators(reports.values())
    seeds = [row['training_seed'] for row in reports.values()]
    if len(reports) != 5 or sorted(seeds) != list(EXPECTED_SEEDS):
        raise ValueError('必须纳入全部五个预定种子91至95，不能挑选成功或最好者')
    if len({row['data_sha256'] for row in reports.values()}) != 1:
        raise ValueError('训练种子对照必须使用同一固定物理数据')
    signatures = [[(x['start'], x['end']) for x in row['grouped_closure_identity']['test']['paths']]
                  for row in reports.values()]
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError('所有种子的闭合路径必须一致')
    rows = []
    for index, (start, end) in enumerate(signatures[0]):
        path_rows = [r['grouped_closure_identity']['test']['paths'][index] for r in reports.values()]
        rows.append(dict(start=start, end=end,
            R_nrmse=_summary([r['R']['nrmse'] for r in path_rows]),
            identity_nrmse=_summary([r['identity']['nrmse'] for r in path_rows]),
            R_to_identity_rmse_ratio=_summary([r['R_to_identity_rmse_ratio'] for r in path_rows])))
    result = dict(schema_version=2, experiment='ising2d_B0_multiseed_stageA_v1',
        training_seeds=seeds, actual_updates_per_seed=12000, theory_dim=5,
        data_sha256=next(iter(reports.values()))['data_sha256'], TEST_path_summary=rows,
        all_seeds_included=True, selected_best_seed=None, thermal_nu=None,
        raw_coordinate_errors_pooled_across_models=False,
        normalization='每模型自己的 TRAIN 目标标准差；跨种子只汇总无量纲误差和相对恒等基准。',
        root_counts_by_seed={str(r['training_seed']): dict(
            analyzed_root_count=len(r.get('roots', [])),
            geometric_support_candidate_count=sum(bool(x.get('geometric_support_candidate'))
                                                   for x in r.get('roots', [])),
            search_complete=False, physical_support_certified=False)
            for r in reports.values()},
        all_reports=reports, **encoding_metadata(),
        limitations=['五个优化种子条件于同一物理缓存，不包含采样种子、尺寸及截断偏差。',
            '不同种子坐标不相同；不对原始 g、原坐标 RMSE、PCA 基向量或根坐标作跨模型平均。',
            '按各自 TRAIN 标准差归一仍不是任意非线性坐标变换的不变量，只作训练稳定性描述。',
            '根为最多64起点、每次200评价的全维有限搜索，未找到不等于不存在。',
            '完整 DR 的实、复、负特征值全部保留；不按理论邻近或最大值选热模，thermal_nu 为 null。'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'comparison.json', result)
    plot_comparison(reports, output)
    atomic_json(output / 'report.json', dict(status='complete', report_complete=True,
        training_seeds=seeds, public_files=['comparison.json', 'comparison.png', 'report.json'],
        thermal_nu=None, selected_best_seed=None, **encoding_metadata()))
    return result


def plot_comparison(reports, output):
    import matplotlib.pyplot as plt
    from scripts.report_ising_zh import chinese_figures
    with chinese_figures():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for report in sorted(reports.values(), key=lambda row: row['training_seed']):
            label = f'训练种子{report["training_seed"]}'
            paths = report['grouped_closure_identity']['test']['paths']
            axes[0].plot(range(len(paths)), [p['R']['nrmse'] for p in paths], 'o-', label=label)
            axes[1].plot(range(len(paths)), [p['R_to_identity_rmse_ratio'] for p in paths], 'o-', label=label)
            for root in report.get('roots', []):
                axes[2].scatter([report['training_seed']] * len(root['eigenvalues']),
                    [value['magnitude'] for value in root['eigenvalues']], s=18,
                    facecolors='C0' if root['geometric_support_candidate'] else 'none', edgecolors='C0')
        ticks = [f'{p["start"]}→{p["end"]}' for p in paths]
        for ax in axes[:2]:
            ax.set_xticks(range(len(ticks)), ticks)
            ax.set_xlabel('尺度路径')
            ax.legend(fontsize=8)
        axes[0].set(title='测试集归一闭合误差', ylabel='本模型训练集标准差归一 RMSE')
        axes[1].set(title='相对配对恒等基准', ylabel='R 的 RMSE／恒等映射 RMSE')
        axes[1].axhline(1., c='black', ls='--', lw=.8)
        axes[2].set(title='全部已分析根的完整谱；不是热指数', xlabel='训练种子', ylabel='特征值模长')
        axes[2].axhline(1., c='black', ls='--', lw=.8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('五个预定训练种子全部保留；不挑最佳、不混合潜变量坐标')
        fig.savefig(Path(output) / 'comparison.png', dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', nargs='+')
    parser.add_argument('--checkpoint', type=int, default=12000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--sampling-repeats', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--progress-path', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('拒绝覆盖或重新执行已有正式/部分输出')
    if args.compare:
        if args.run:
            parser.error('--run 与 --compare 不能同时使用')
        torch.set_num_threads(1)
        with single_thread_control():
            return compare_reports(args.compare, args.output)
    if args.run is None or args.threads != 1 or args.checkpoint != 12000 or args.sampling_repeats != 2:
        parser.error('正式报告固定12000终点、CPU单线程和两个额外固定 C 重复')
    args.progress_path = args.progress_path or args.output.with_name(args.output.name + '.progress.json')
    if args.progress_path.exists() or args.progress_path == args.output or args.output in args.progress_path.parents:
        parser.error('进度路径必须是输出目录之外的新文件')
    torch.set_num_threads(1)
    deadline = Deadline(args.max_seconds)
    with single_thread_control():
        return run_report(args, deadline, Progress(args.progress_path, deadline))


if __name__ == '__main__':
    result = main()
    if isinstance(result, dict) and result.get('status') == 'analysis_budget_hold':
        raise SystemExit(75)
