"""有限预算、冻结模型的二维 J1--J2 QCR 中文报告。

只读取本轮 label-free cache、外部评价元数据和最终 checkpoint。使用精确
二值查表，在同一条 C 轨迹上计算训练口径的两半 Q 均值与物理统计。
不训练、不计算 MI、不搜根，不从投影流场或单尺寸统计认证相变。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from scripts.ising_highdim_fast_evaluation import FrozenBinaryCollector, _model_digest
from scripts.ising_qcr_evaluation import HALF_SUBSET_ESTIMATOR, encoding_contract


OBSERVABLE_NAMES = ('m2', 'm4', 'stripe_x2', 'stripe_y2', 'stripe_total4',
                    'Gx_L4', 'Gy_L4', 'S00', 'Spi0', 'S0pi', 'nn', 'diagonal')
OBSERVABLE_LABELS = ('磁化平方', '磁化四次方', '横向条纹平方', '纵向条纹平方',
                     '条纹总序四次方', '横向远距关联', '纵向远距关联',
                     '零波矢结构因子', '横向条纹结构因子', '纵向条纹结构因子',
                     '最近邻关联', '对角近邻关联')
QUALIFICATION = ('仅为冻结模型的有限尺度评价。投影箭头是 R(g)-g 的主成分投影，'
    '不是完整空间的根或连续 RG 向量场；未搜索不动点、未认证相变或临界指数。'
    'C 重复不是独立物理链；同一 rho 与 group 的不同温度来自同一平行回火重复。'
    '物理读出是事后 TRAIN-only 线性探针，不反馈 Q/C/R；其误差包括读出限制。')


class ReportBudgetHold(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('报告预算必须是正有限秒数')
        self.seconds, self.started = float(seconds), time.monotonic()

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def check(self):
        if self.elapsed > self.seconds:
            raise ReportBudgetHold('报告达到固定时间预算；保留已写结果，不自动重跑')


def sha256(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            if deadline is not None:
                deadline.check()
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                inode=value.st_ino, device=value.st_dev)


def validate_metadata(metadata, datasets, *, actual_sha256, manifest_sha256):
    """Bind label-free bytes, external labels and exact split/row identities."""
    if (metadata.get('schema_version') != 1 or not isinstance(actual_sha256, str)
            or len(actual_sha256) != 64
            or actual_sha256 != metadata.get('data_sha256')
            or actual_sha256 != manifest_sha256):
        raise ValueError('外部元数据、训练 manifest 与实际数据 SHA256 不一致')
    mapping, identities, train_rhos, test_rows = {}, set(), set(), []
    rho_by_index, index_by_rho, condition_keys, parameter_keys = {}, {}, {}, {}
    for split, dataset in datasets.items():
        rows = metadata.get(split)
        groups = dataset.group_ids
        if (dataset.couplings is not None or not isinstance(rows, list)
                or len(rows) != len(dataset.configurations) or groups is None
                or tuple(groups.shape) != (len(rows), 3)
                or groups.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError('需要无标签 cache 及逐行完整的整数 [rho,T,group] 身份')
        for row, actual_group in zip(rows, groups.tolist()):
            expected = tuple(row.get(key) for key in ('rho_index', 'temperature_index', 'group'))
            if (any(type(item) is not int or item < 0 for item in expected)
                    or tuple(actual_group) != expected or expected in identities):
                raise ValueError('cache 身份与外部评价行顺序不符、重复或跨划分泄露')
            identities.add(expected)
            rho, temperature = row.get('rho'), row.get('temperature')
            if (isinstance(rho, bool) or isinstance(temperature, bool)
                    or not isinstance(rho, (int, float)) or not np.isfinite(rho) or rho < 0
                    or not isinstance(temperature, (int, float))
                    or not np.isfinite(temperature) or temperature <= 0
                    or not isinstance(row.get('condition_id'), str) or not row['condition_id']):
                raise ValueError('外部 rho、温度或 condition_id 无效')
            parameter = (float(rho), float(temperature), row['condition_id'])
            key = expected[:2]
            if key in mapping and mapping[key] != parameter:
                raise ValueError('同一参数索引对应冲突的评价参数')
            mapping[key] = parameter
            rid = expected[0]
            if ((rid in rho_by_index and rho_by_index[rid] != float(rho))
                    or (float(rho) in index_by_rho and index_by_rho[float(rho)] != rid)
                    or (row['condition_id'] in condition_keys and condition_keys[row['condition_id']] != key)
                    or (parameter[:2] in parameter_keys and parameter_keys[parameter[:2]] != key)):
                raise ValueError('rho、condition_id 或实际参数与索引不是一一对应')
            rho_by_index[rid], index_by_rho[float(rho)] = float(rho), rid
            condition_keys[row['condition_id']], parameter_keys[parameter[:2]] = key, key
            if split == 'train':
                if row.get('partition') != 'train':
                    raise ValueError('TRAIN 评价身份必须明确标记 train')
                train_rhos.add(expected[0])
            else:
                test_rows.append(row)
    for row in test_rows:
        partition = row.get('partition')
        if (partition not in ('same_rho_new_group', 'heldout_rho')
                or ((row['rho_index'] in train_rhos) != (partition == 'same_rho_new_group'))):
            raise ValueError('TEST 的已见／留出 rho 身份与实际 TRAIN 冲突')
    # A PT group must be entirely held out across all temperatures at a seen rho.
    train_groups = {(row['rho_index'], row['group']) for row in metadata['train']}
    test_groups = {(row['rho_index'], row['group']) for row in metadata['test']}
    if train_groups & test_groups:
        raise ValueError('同一平行回火重复不能跨 TRAIN/TEST 温度泄露')
    return dict(data_sha256=actual_sha256, exact_row_identity_verified=True,
                cache_couplings_are_none=True, whole_PT_group_split_verified=True)


def physical_observables(bits):
    """[chain,sample,L,L] hard bits -> [chain,12] sample means.

    Structure factors use |sum s exp(i q.r)|^2 / V^2, hence S00=m2.
    G axes use the actual integer distance max(1,L//4). No absolute-value
    replacement is made for antiferromagnetic correlations.
    """
    bits = np.asarray(bits)
    if bits.dtype != np.bool_ or bits.ndim != 4 or bits.shape[-2] != bits.shape[-1]:
        raise ValueError('物理统计需要正方形的硬二值 [链,构型,L,L]')
    size = bits.shape[-1]
    if size < 2:
        raise ValueError('统计尺寸至少为 2')
    parity = (-1.) ** np.arange(size)
    m = 2. * bits.mean(axis=(-2, -1)) - 1.
    mx = (2. * (bits.sum(-1) * parity).sum(-1) - size * parity.sum()) / size ** 2
    my = (2. * (bits.sum(-2) * parity).sum(-1) - size * parity.sum()) / size ** 2
    m2, mx2, my2 = m ** 2, mx ** 2, my ** 2
    def corr(dx, dy):
        return 1. - 2. * np.not_equal(bits, np.roll(bits, (-dx, -dy), (-2, -1))).mean((-2, -1))
    distance = max(1, size // 4)
    values = (m2, m2 ** 2, mx2, my2, (mx2 + my2) ** 2,
              corr(distance, 0), corr(0, distance), m2, mx2, my2,
              .5 * (corr(1, 0) + corr(0, 1)),
              .5 * (corr(1, 1) + corr(1, -1)))
    return np.stack([item.mean(1) for item in values], -1)


class J1J2Collector(FrozenBinaryCollector):
    """Same exact tables as existing CPU reports, without generic root search."""
    def __init__(self, model, *, samples=128, **kwargs):
        self.encoding_protocol = encoding_contract(samples)
        self.partition = np.asarray(self.encoding_protocol['sample_partition'])
        super().__init__(model, **kwargs)

    def encoded_pair(self, bits):
        if bits.shape[1] != self.encoding_protocol['samples']:
            raise ValueError('构型数与固定两半 Q 协议不符')
        half = len(self.partition) // 2
        first = super()._encode(bits[:, self.partition[:half]])
        second = super()._encode(bits[:, self.partition[half:]])
        return (.5 * (first + second)).numpy(), super()._encode(bits).numpy()

    @torch.no_grad()
    def trajectories(self, configurations, *, steps, draws=5, chunk_size=2,
                     seed=2026102601):
        if (configurations.ndim != 5 or configurations.shape[-1] != 1
                or configurations.shape[1] != self.encoding_protocol['samples']
                or configurations.device.type != 'cpu' or configurations.requires_grad
                or type(steps) is not int or steps < 1
                or type(draws) is not int or draws < 1
                or type(chunk_size) is not int or chunk_size < 1
                or any(length % (3 ** steps) for length in configurations.shape[2:4])):
            raise ValueError('需要完整 CPU 方形二值数据、正重复数及可整除的 b3 尺度')
        self._unchanged()
        result = {key: [] for key in ('direct', 'full_ensemble_direct', 'observables')}
        total = len(configurations)
        for start in range(0, total, chunk_size):
            self._check()
            chunk = configurations[start:start + chunk_size]
            values = {key: [] for key in result}
            for draw in range(draws):
                # Seed is fixed per original chunk, independent of labels.
                with isolated_rng('cpu', seed + start * 1000 + draw):
                    bits = self._binary_view(chunk, len(self.partition), False)
                    direct, full, observed = [], [], []
                    for scale in range(steps + 1):
                        self._check()
                        encoded, all_q = self.encoded_pair(bits)
                        direct.append(encoded); full.append(all_q)
                        observed.append(physical_observables(bits))
                        if scale < steps:
                            bits = self._coarsen(bits)
                    values['direct'].append(np.stack(direct, 1))
                    values['full_ensemble_direct'].append(np.stack(full, 1))
                    values['observables'].append(np.stack(observed, 1))
            for key in result:
                result[key].append(np.stack(values[key], 1))
            if self.progress is not None:
                self.progress('collect', min(start + chunk_size, total), total, {})
        result = {key: np.concatenate(value) for key, value in result.items()}
        result['R_g'] = self.model.advance_theory(torch.from_numpy(result['direct'])).numpy()
        result['R_of_mean_g'] = self.model.advance_theory(
            torch.from_numpy(result['direct'].mean(1))).numpy()
        result['mean_R_g_minus_R_of_mean_g'] = result['R_g'].mean(1) - result['R_of_mean_g']
        result['Q_full_minus_half'] = result['full_ensemble_direct'] - result['direct']
        if any(not np.isfinite(value).all() for value in result.values()):
            raise RuntimeError('冻结采集出现非有限数值')
        self._unchanged()
        return result


def transport_paths(model, direct):
    """Preserve rows and C draws; return all earlier-start/target paths."""
    paths, predictions, identities = [], [], []
    with torch.no_grad():
        for start in range(direct.shape[2] - 1):
            prediction = torch.from_numpy(direct[:, :, start]).clone()
            for end in range(start + 1, direct.shape[2]):
                prediction = model.advance_theory(prediction)
                paths.append((start, end)); predictions.append(prediction.numpy().copy())
                identities.append(direct[:, :, start])
    return np.asarray(paths, dtype=np.int64), np.stack(predictions, 2), np.stack(identities, 2)


def error_summary(error, sigma):
    """C-mean squared errors first, never count C draws as independent chains."""
    error = np.asarray(error, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    per_chain = np.mean(error ** 2, axis=1)
    rmse = float(np.sqrt(per_chain.mean()))
    denominator = float(np.sqrt(np.mean(sigma ** 2)))
    coordinate = np.sqrt(per_chain.mean(0))
    return dict(rmse=rmse, nrmse=rmse / denominator if denominator > 1e-12 else None,
                coordinate_rmse=coordinate.tolist(),
                coordinate_nrmse=[float(x / y) if y > 1e-12 else None
                                  for x, y in zip(coordinate, sigma)],
                physical_row_count=len(error), C_draw_count=error.shape[1])


def grouped_closure_metrics(direct, predictions, identities, paths, train_std, rows,
                           *, bootstrap=1000, seed=2026102602):
    """Stratified bootstrap of complete PT groups, paired across temperatures."""
    rng = np.random.default_rng(seed)
    by_rho = {}
    for index, row in enumerate(rows):
        by_rho.setdefault(row['rho_index'], {}).setdefault(row['group'], []).append(index)
    summaries = []
    for p, (start, end) in enumerate(paths):
        target = direct[:, :, end]
        scale = train_std[end]
        row = dict(start=int(start), end=int(end))
        errors = {'R': predictions[:, :, p] - target,
                  'identity': identities[:, :, p] - target}
        mses = {key: np.mean(value ** 2, axis=(1, 2)) for key, value in errors.items()}
        for key, value in errors.items():
            row[key] = error_summary(value, scale)
        values = []
        for _ in range(bootstrap):
            indices = []
            for groups in by_rho.values():
                keys = list(groups)
                for chosen in rng.integers(len(keys), size=len(keys)):
                    indices.extend(groups[keys[chosen]])
            values.append([np.sqrt(mses[key][indices].mean()) for key in ('R', 'identity')])
        if values:
            interval = np.percentile(values, [2.5, 97.5], axis=0)
            for k, key in enumerate(('R', 'identity')):
                row[key]['PT_group_bootstrap_rmse_95'] = interval[:, k].tolist()
        baseline = row['identity']['rmse']
        row['R_to_identity_rmse_ratio'] = row['R']['rmse'] / baseline if baseline > 1e-12 else None
        summaries.append(row)
    return dict(paths=summaries, PT_group_count=sum(map(len, by_rho.values())),
        bootstrap_repetitions=bootstrap,
        bootstrap_rule='按 rho 分层，整组重采 (rho_index,group)，保留同组全部温度；R 与恒等配对。',
        qualification='区间条件于固定模型和采样记录，不包含训练 seed、热化偏差或有限尺寸偏差。')


def fit_ridge(x, y, *, ridge=1e-4):
    """TRAIN-only fixed probe; no TEST tuning, QCR gradients or labels."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError('读出需要匹配的 TRAIN 矩阵')
    mean, scale, y_mean = x.mean(0), x.std(0, ddof=0), y.mean(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = (x - mean) / scale
    coefficient = np.linalg.solve(design.T @ design + ridge * len(x) * np.eye(x.shape[1]),
                                  design.T @ (y - y_mean))
    return dict(mean=mean, scale=scale, coefficient=coefficient, y_mean=y_mean,
                target_std=y.std(0, ddof=0), ridge=float(ridge))


def predict_ridge(probe, x):
    return (np.asarray(x) - probe['mean']) / probe['scale'] @ probe['coefficient'] + probe['y_mean']


def physical_readout_metrics(probes, values, predictions, identities, paths, mask=None):
    direct, observed = values['direct'], values['observables']
    if mask is None:
        mask = np.ones(len(direct), dtype=bool)
    result = {}
    for family, fitted in probes.items():
        rows = []
        for p, (start, end) in enumerate(paths):
            probe = fitted[end]
            target_scale = 0 if family == 'microscopic_information' else end
            target = observed[:, :, target_scale]
            outputs = dict(direct=predict_ridge(probe, direct[:, :, end]),
                           transport=predict_ridge(probe, predictions[:, :, p]),
                           identity=predict_ridge(probe, identities[:, :, p]))
            row = dict(start=int(start), end=int(end), target_scale=int(target_scale))
            for key, output in outputs.items():
                row[key] = error_summary((output - target)[mask], probe['target_std'])
            row['transport_minus_direct'] = error_summary(
                (outputs['transport'] - outputs['direct'])[mask], probe['target_std'])
            rows.append(row)
        result[family] = rows
    return result


def pca_train(direct):
    matrix = direct.mean(1).reshape(-1, direct.shape[-1])
    mean = matrix.mean(0)
    _, singular, vectors = np.linalg.svd(matrix - mean, full_matrices=False)
    # Fix arbitrary signs reproducibly; no TEST information is used.
    for vector in vectors:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    total = float(np.sum(singular ** 2))
    return dict(mean=mean, components=vectors,
                explained_variance_ratio=singular ** 2 / total if total > 0 else np.zeros_like(singular))


def make_figures(output, arrays, metadata, pca, history, *, font=None, deadline=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scripts.report_ising_zh import select_chinese_font
    selected = select_chinese_font(font)
    font_manager.fontManager.addfont(str(selected))
    style = {'font.family': font_manager.FontProperties(fname=str(selected)).get_name(),
             'axes.unicode_minus': False}
    def save(fig, name):
        if deadline is not None:
            deadline.check()
        fig.savefig(output / name, dpi=150, bbox_inches='tight'); plt.close(fig)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ratio = pca['explained_variance_ratio']
        ax.bar(np.arange(1, len(ratio) + 1), ratio)
        ax.set(xlabel='主成分编号', ylabel='方差占比', title='TRAIN 拟合的潜变量主成分占比')
        save(fig, 'pca_explained.png')
        d = arrays['test']['direct'].mean(1)
        r = arrays['test']['R_of_mean_g']
        coordinates = (d - pca['mean']) @ pca['components'].T
        displacement = (r - d) @ pca['components'].T
        dimension = d.shape[-1]
        fig, axes = plt.subplots(1, d.shape[1], figsize=(5 * d.shape[1], 4), squeeze=False,
                                 layout='constrained')
        for level, ax in enumerate(axes[0]):
            # Plot R at each actual physical row's C-mean state. Do NOT average
            # R(g) at different physical states and call it R(mean g).
            for rho in sorted({row['rho'] for row in metadata['test']}):
                ids = [i for i, row in enumerate(metadata['test']) if row['rho'] == rho]
                # Render at most 100 physical states per rho; saved arrays and
                # all quantitative evaluations still include every row.
                ids = ids[::max(1, int(np.ceil(len(ids) / 100)))]
                points, arrows = coordinates[ids, level], displacement[ids, level]
                if dimension == 1:
                    ax.scatter(points[:, 0], arrows[:, 0], s=12, label=rf'$\rho={rho:g}$')
                else:
                    ax.scatter(points[:, 0], points[:, 1], s=12, label=rf'$\rho={rho:g}$')
                    ax.quiver(points[:, 0], points[:, 1], arrows[:, 0], arrows[:, 1],
                              angles='xy', scale_units='xy', scale=1, width=.003)
            ax.set(xlabel='第一主成分', ylabel='投影位移 R(g)-g' if dimension == 1 else '第二主成分',
                   title=f'尺度 k={level}：C 均值处 R(g)-g（未搜根）')
            if dimension == 1:
                ax.axhline(0, color='gray', linewidth=.7)
            ax.legend(fontsize=7)
        save(fig, 'R_displacement.png')
        fig, axes = plt.subplots(1, min(2, dimension), figsize=(6 * min(2, dimension), 4),
                                 squeeze=False, layout='constrained')
        for pc, ax in enumerate(axes[0]):
            for rho in sorted({row['rho'] for row in metadata['test']}):
                ids = [i for i, row in enumerate(metadata['test']) if row['rho'] == rho]
                temperatures = np.asarray([metadata['test'][i]['temperature'] for i in ids])
                ax.scatter(temperatures, coordinates[ids, 0, pc], s=12, label=rf'$\rho={rho:g}$')
            ax.set(xlabel='温度 T', ylabel=f'第 {pc + 1} 主成分', title='微观耦合参数到潜变量的评价映射')
            ax.legend(fontsize=8)
        save(fig, 'coupling_to_g.png')
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        labels = {'train_Q': 'TRAIN 表示', 'test_Q': 'TEST 表示',
                  'train_C': 'TRAIN 粗粒化', 'test_C': 'TEST 粗粒化',
                  'train_R': 'TRAIN 闭合', 'test_R': 'TEST 闭合',
                  'train_separation': 'TRAIN motif 重构', 'test_separation': 'TEST motif 重构',
                  'train_anchor': 'TRAIN 微观子集一致性', 'test_anchor': 'TEST 微观子集一致性',
                  'train_coarse_consistency': 'TRAIN 粗尺度子集一致性',
                  'test_coarse_consistency': 'TEST 粗尺度子集一致性',
                  'train_kernel_noncollapse': 'TRAIN 方差防塌缩',
                  'test_kernel_noncollapse': 'TEST 方差防塌缩',
                  'train_kernel_sign': 'TRAIN 保色', 'test_kernel_sign': 'TEST 保色',
                  'train_rsmi': 'TRAIN 负经验互信息', 'test_rsmi': 'TEST 负经验互信息',
                  'train_commutator': 'TRAIN 一步闭合', 'test_commutator': 'TEST 一步闭合',
                  'train_rollout': 'TRAIN 多步闭合', 'test_rollout': 'TEST 多步闭合',
                  'train_all_start_anchor': 'TRAIN 全起点闭合',
                  'test_all_start_anchor': 'TEST 全起点闭合'}
        plotted = []
        for key, label in labels.items():
            points = [(row.get('epoch', row.get('step')), row.get(key)) for row in history]
            points = [(x, y) for x, y in points if isinstance(x, (int, float))
                      and isinstance(y, (int, float)) and np.isfinite(y)]
            if points:
                x, y = zip(*points); ax.plot(x, y, label=label); plotted.append(key)
        if not plotted:
            ax.text(.5, .5, '历史未记录已识别的分项；详见原始历史 JSON', ha='center', transform=ax.transAxes)
        else:
            ax.legend(fontsize=7); ax.set_yscale('symlog', linthresh=1e-5)
        ax.set(xlabel='训练更新步', ylabel='已记录损失', title='正式训练历史（不重新计算损失）')
        save(fig, 'loss_curves.png')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
        observed = arrays['test']['observables'].mean(1)
        for ax, index, title in zip(axes, (0, 2, 3), ('磁化平方', '横向条纹平方', '纵向条纹平方')):
            for level in range(observed.shape[1]):
                for rho in sorted({row['rho'] for row in metadata['test']}):
                    ids = [i for i, row in enumerate(metadata['test']) if row['rho'] == rho]
                    temperatures = np.asarray([metadata['test'][i]['temperature'] for i in ids])
                    ax.scatter(temperatures, observed[ids, level, index], s=9,
                               label=rf'$\rho={rho:g},\ k={level}$', alpha=.65)
            ax.set(xlabel='温度 T', ylabel=title, title=title + '：实际 C 轨迹各尺度')
        axes[-1].legend(fontsize=5, ncol=2)
        save(fig, 'physical_C_statistics.png')
    return str(selected)


def validate_completed_run(loaded, manifest, summary):
    cfg = RunConfig.from_dict(loaded.config)
    cfg.validate()
    metadata = loaded.payload.get('metadata', {})
    if (cfg.schema_version != 5 or cfg.model.dimension != 2
            or cfg.model.block_factor != 3 or cfg.model.fixed_kernel
            or cfg.model.kernel_initialization != 'random'
            or cfg.model.kernel_sampling != 'bernoulli'
            or cfg.model.kernel_layout != 'nonoverlapping'
            or cfg.model.width != 3 or cfg.model.categorical_states
            or cfg.training.gradient_mode != 'qc_r_separate'
            or cfg.loss.closure_objective != 'all_start_anchor'
            or cfg.loss.weights.commutator != 5 or cfg.loss.rollout_weight != 5
            or cfg.training.epochs != 12000 or loaded.payload.get('epoch') != 12000
            or summary.get('actual_epochs') != 12000
            or summary.get('stopped_by_wall_limit') or summary.get('stopped_by_training_limit')
            or cfg.evaluation.samples != 128 or cfg.data.spin_flip_augment
            or cfg.data.train_pool_replicas != 1 or cfg.data.test_pool_replicas != 1
            or manifest.get('labels_used_in_training') is not False
            or metadata.get('labels_used_in_training') is not False
            or manifest.get('q_supervision') or metadata.get('q_supervision')
            or manifest.get('q_future_physics') or metadata.get('q_future_physics')
            or manifest.get('source') or metadata.get('source')
            or manifest.get('data_sha256') != metadata.get('data_sha256')):
        raise ValueError('正式报告要求全新默认无监督 b3 全起点训练的完整 12000 步终点')
    if json.dumps(manifest.get('config'), sort_keys=True) != json.dumps(loaded.config, sort_keys=True):
        raise ValueError('manifest 与 checkpoint 实际配置不一致')
    return cfg


def run_report(args):
    deadline = Deadline(args.max_seconds)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖已有正式或部分报告')
    output.mkdir(parents=True)
    write_json(output / 'status.json', dict(status='running', report_complete=False))
    try:
        torch.set_num_threads(1)
        checkpoint = Path(args.run) / 'training_step_12000.pt'
        checkpoint_sha = sha256(checkpoint, deadline)
        loaded = load_run(args.run, 12000)
        manifest = json.loads((Path(args.run) / 'manifest.json').read_text())
        summary = json.loads((Path(args.run) / 'summary.json').read_text())
        cfg = validate_completed_run(loaded, manifest, summary)
        model = loaded.model.cpu().double().eval().requires_grad_(False)
        before = _model_digest(model)
        data_path = Path(cfg.data.cache)
        source_stats = {str(path): file_stat(path) for path in (checkpoint, data_path, Path(args.data_metadata))}
        metadata = json.loads(Path(args.data_metadata).read_text())
        datasets = dict(zip(('train', 'test'), load_ensembles(data_path, mmap=True)))
        binding = validate_metadata(metadata, datasets, actual_sha256=sha256(data_path, deadline),
                                    manifest_sha256=manifest['data_sha256'])
        collector = J1J2Collector(model, deadline=deadline, samples=128,
            progress=lambda stage, done, total, extra: print(f'冻结报告 {stage}：{done}/{total}', flush=True))
        steps = cfg.training.max_rg_steps
        arrays, transport, identity, paths = {}, {}, {}, None
        for split, dataset in datasets.items():
            arrays[split] = collector.trajectories(dataset.configurations, steps=steps,
                draws=args.draws, chunk_size=2, seed=2026102601 + (split == 'test') * 10000000)
            paths, transport[split], identity[split] = transport_paths(model, arrays[split]['direct'])
        train_std = arrays['train']['direct'].mean(1).std(0, ddof=0)
        masks = {'train': ('train', np.ones(len(metadata['train']), dtype=bool)),
                 'same_rho_new_group': ('test', np.asarray([row['partition'] == 'same_rho_new_group' for row in metadata['test']])),
                 'heldout_rho': ('test', np.asarray([row['partition'] == 'heldout_rho' for row in metadata['test']]))}
        probes = {name: [] for name in ('microscopic_information', 'native_C_statistics')}
        for level in range(steps + 1):
            for family in probes:
                target = 0 if family == 'microscopic_information' else level
                probes[family].append(fit_ridge(arrays['train']['direct'][:, :, level].mean(1),
                    arrays['train']['observables'][:, :, target].mean(1)))
        closure, physical = {}, {}
        for name, (split, mask) in masks.items():
            deadline.check()
            if not mask.any():
                raise ValueError('缺少预先指定的完整评价划分：' + name)
            rows = [row for row, selected in zip(metadata[split], mask) if selected]
            closure[name] = grouped_closure_metrics(arrays[split]['direct'][mask],
                transport[split][mask], identity[split][mask], paths, train_std, rows)
            physical[name] = physical_readout_metrics(probes, arrays[split], transport[split],
                                                      identity[split], paths, mask)
        pca = pca_train(arrays['train']['direct'])
        history = loaded.payload.get('history') or json.loads((Path(args.run) / 'history_progress.json').read_text())
        history = [row for row in history if row.get('epoch', row.get('step', 0)) <= 12000]
        font = make_figures(output, arrays, metadata, pca, history, font=args.font, deadline=deadline)
        saved = dict(paths=paths, TRAIN_target_coordinate_std=train_std,
            pca_mean=pca['mean'], pca_components=pca['components'],
            pca_explained_variance_ratio=pca['explained_variance_ratio'],
            observable_names=np.asarray(OBSERVABLE_NAMES),
            encoding_protocol=np.asarray(json.dumps(collector.encoding_protocol, sort_keys=True)))
        for split in arrays:
            saved.update({f'{key}_{split}': value for key, value in arrays[split].items()})
            saved[f'transport_{split}'], saved[f'identity_{split}'] = transport[split], identity[split]
            saved[f'group_ids_{split}'] = datasets[split].group_ids.numpy()
            saved[f'rho_temperature_{split}'] = np.asarray([[row['rho'], row['temperature']] for row in metadata[split]])
        for family, fitted in probes.items():
            for level, probe in enumerate(fitted):
                saved.update({f'readout_{family}_k{level}_{key}': value for key, value in probe.items()})
        deadline.check()
        np.savez_compressed(output / 'evaluation_arrays.npz', **saved)
        if (_model_digest(model) != before or any(p.grad is not None or p.requires_grad for p in model.parameters())
                or sha256(checkpoint, deadline) != checkpoint_sha
                or source_stats != {path: file_stat(path) for path in source_stats}):
            raise RuntimeError('冻结 checkpoint、模型或输入文件在报告期间发生变化')
        report = dict(schema_version=1, status='complete', report_complete=True,
            experiment='ising_j1j2_stageA', checkpoint_step=12000,
            checkpoint_sha256=checkpoint_sha, model_sha256_before=before, model_sha256_after=before,
            data_binding=binding, data_metadata_sha256=sha256(args.data_metadata, deadline),
            encoding_estimator=HALF_SUBSET_ESTIMATOR, encoding_protocol=collector.encoding_protocol,
            sampling=dict(C_draws=args.draws, CPU_threads=1, chunk_size=2,
                rows={split: len(data.configurations) for split, data in datasets.items()},
                rng_layout='固定 split/chunk/draw 种子；不使用 rho、T 或相标签'),
            closure=closure, physical_readouts=physical,
            physical_observables=dict(names=list(OBSERVABLE_NAMES), labels=list(OBSERVABLE_LABELS),
                lattice_sizes=[datasets['train'].configurations.shape[2] // 3 ** k for k in range(steps + 1)],
                G_distances=[max(1, datasets['train'].configurations.shape[2] // 3 ** k // 4) for k in range(steps + 1)],
                structure_factor_normalization='V^2；S00=m2，Spi0=stripe_x2，S0pi=stripe_y2；不是独立统计'),
            readout_contract=dict(type='固定 ridge=1e-4、TRAIN-only、各尺度独立仿射探针',
                trained_on='TRAIN 各物理行的 C 均值；无 T/rho 输入',
                target_families=['原始微观物理统计（跨尺度共用目标）', '实际 C 轨迹各尺度物理统计'],
                normalization='对应读出 TRAIN 目标 std，ddof=0；不使用 TEST 拟合或选超参数'),
            PCA=dict(fit_split='train', fit_values='C 均值、全部实际尺度',
                explained_variance_ratio=pca['explained_variance_ratio'].tolist(),
                qualification='方差占比不等于有效物理维度'),
            flow_plot_contract='每物理行 C 均值处直接计算 R(mean_C g)-mean_C g；每 rho 最多绘制100行，量化评价无下采样。',
            C_Jensen_diagnostic='NPZ 同时保存 mean_C R(g)、R(mean_C g) 及差值，不混同两种非线性估计器。',
            root_search_performed=False, fixed_points=[], thermal_nu=None,
            phase_transition_certified=False, physical_sampling_certified=False,
            training_updates=0, optimizer_created=False, MI_evaluations=0,
            chinese_font=font, qualification=QUALIFICATION,
            implemented='完整 TRAIN/TEST、留出 rho、尺度内全路径、物理事后读出与中文图',
            not_implemented=['独立更大尺寸测试', '跨模型或训练 seed 汇总', '相变温度认证',
                             '与其它算法比较', '相变类型及临界指数认证'],
            elapsed_seconds=deadline.elapsed, max_seconds=deadline.seconds)
        write_json(output / 'loss_history.json', history)
        write_json(output / 'report.json', report)
        files = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != 'status.json')
        write_json(output / 'status.json', dict(status='complete', report_complete=True,
            public_files=files, sha256={name: sha256(output / name) for name in files},
            elapsed_seconds=deadline.elapsed, root_search_performed=False, thermal_nu=None))
        return report
    except Exception as error:
        write_json(output / 'status.json', dict(status='budget_hold' if isinstance(error, ReportBudgetHold) else 'failed',
            report_complete=False, error=f'{type(error).__name__}: {error}',
            requires_user_action=True, elapsed_seconds=deadline.elapsed,
            training_updates=0, root_search_performed=False, thermal_nu=None))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--draws', type=int, default=5)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args(argv)
    if args.draws != 5:
        parser.error('正式报告固定 5 个 C 重复，不允许事后减少凑预算')
    run_report(args)


if __name__ == '__main__':
    main()
