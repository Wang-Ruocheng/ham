"""未热化 J1--J2 小规模诊断训练的冻结模型中文报告。

这是独立入口，不修改、放宽或重跑正式平衡态实验的报告。
复用正式报告的冻结采集、全路径误差、TRAIN-only 读出及绘图实现，
但本诊断只有已见 rho 的独立 PT 组留出，不伪造留出 rho 结果。
所有图和机器可读结果均保留平衡态采样未验证的显著警告。
不训练、不计算 MI、不搜根，不认证相变、不动点或临界指数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import report_ising_j1j2 as base


WARNING_LINES = (
    '未热化诊断：平衡态采样未验证',
    '仅检查训练行为；不得据此认证相变、不动点或临界指数',
)
QUALIFICATION = ('本报告来自用户授权的未热化诊断训练，不是正式平衡态物理结果。'
    '热化／混合质量警告非阻塞仅适用于本诊断，不代表质量门控通过。'
    '小规模有限链样本可能保留初态和采样时间依赖；低闭合误差不证明物理 RG 正确。'
    '全部 rho 均出现在 TRAIN，仅按独立平行回火组留出 TEST；没有留出 rho 或更大尺寸验证。'
    + base.QUALIFICATION)
FLAGS = dict(diagnostic_only=True, equilibrium_not_validated=True,
             physical_sampling_certified=False, phase_transition_certified=False,
             root_search_performed=False, thermal_nu=None)
REQUIRED_FIGURES = ('pca_explained.png', 'R_displacement.png', 'coupling_to_g.png',
                    'loss_curves.png', 'physical_C_statistics.png')


def warning_reference(path, deadline=None):
    path = Path(path)
    if path.suffix != '.json' or not path.is_file() or path.is_symlink():
        raise ValueError('需要明确的非符号链接采样质量警告 JSON 文件')
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or not record:
        raise ValueError('采样质量警告必须是非空 JSON 对象，不得静默省略')
    return record, dict(source=str(path.resolve()), source_sha256=base.sha256(path, deadline),
                        report_copy='sampling_quality_warning.json',
                        blocking_for_this_diagnostic=False,
                        warning='；'.join(WARNING_LINES))


def diagnostic_partitions(metadata):
    """No heldout-rho surrogate: this run tests new groups at seen rho only."""
    train = metadata.get('train', [])
    test = metadata.get('test', [])
    if (not train or not test or any(row.get('partition') != 'train' for row in train)
            or any(row.get('partition') != 'same_rho_new_group' for row in test)
            or {row['rho_index'] for row in train} != {row['rho_index'] for row in test}):
        raise ValueError('本诊断要求全部 rho 在 TRAIN 和独立组 TEST 中出现，不得伪造留出 rho')
    return {'train': ('train', np.ones(len(train), dtype=bool)),
            'same_rho_new_group': ('test', np.ones(len(test), dtype=bool))}


def add_warning_banners(output, *, font, deadline=None):
    """Add a separate red header to newly generated plots; preserve plot pixels."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.ft2font import FT2Font

    available = set(FT2Font(str(font)).get_charmap())
    missing = {c for text in WARNING_LINES for c in text
               if ord(c) > 127 and ord(c) not in available}
    if missing:
        raise RuntimeError('中文字体缺少诊断警告字符：' + ''.join(sorted(missing)))
    found = {path.name for path in Path(output).glob('*.png')}
    if found != set(REQUIRED_FIGURES):
        raise RuntimeError('诊断报告图不完整或出现未声明图，拒绝省略警告')
    headers = {}
    for name in REQUIRED_FIGURES:
        if deadline is not None:
            deadline.check()
        path = Path(output) / name
        with Image.open(path) as source:
            original = source.convert('RGB')
        if original.width < 80:
            raise ValueError('诊断图尺寸过小，无法清晰绘制警告')
        size = min(30, max(12, original.width // 32))
        while True:
            warning_font = ImageFont.truetype(str(font), size=size)
            boxes = [warning_font.getbbox(text) for text in WARNING_LINES]
            if max(box[2] - box[0] for box in boxes) <= original.width - 24:
                break
            size -= 1
            if size < 8:
                raise ValueError('诊断警告无法在图宽中清晰显示')
        line_height = max(box[3] - box[1] for box in boxes) + 8
        height = 24 + len(WARNING_LINES) * line_height
        canvas = Image.new('RGB', (original.width, original.height + height), '#fff0f0')
        canvas.paste(original, (0, height))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, original.width - 1, height - 1), outline='#a00000', width=3)
        for line, (text, box) in enumerate(zip(WARNING_LINES, boxes)):
            x = (original.width - (box[2] - box[0])) / 2 - box[0]
            y = 12 + line * line_height - box[1]
            draw.text((x, y), text, font=warning_font, fill='#a00000')
        canvas.save(path)
        headers[name] = dict(original_height=original.height, warning_header_height=height,
                             warning_lines=list(WARNING_LINES))
    return headers


def run_report(args):
    deadline = base.Deadline(args.max_seconds)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖已有正式或部分诊断报告')
    output.mkdir(parents=True)
    base.write_json(output / 'status.json', dict(status='running', report_complete=False, **FLAGS))
    try:
        if args.draws != 5:
            raise ValueError('诊断报告固定 5 个 C 重复，不允许减少凑预算')
        warning_record, warning = warning_reference(args.quality_warning, deadline)
        base.write_json(output / warning['report_copy'], dict(
            **FLAGS, quality_warning=warning, source_record=warning_record,
            qualification=QUALIFICATION))
        torch.set_num_threads(1)
        checkpoint = Path(args.run) / 'training_step_12000.pt'
        checkpoint_sha = base.sha256(checkpoint, deadline)
        loaded = base.load_run(args.run, 12000)
        manifest = json.loads((Path(args.run) / 'manifest.json').read_text())
        summary = json.loads((Path(args.run) / 'summary.json').read_text())
        cfg = base.validate_completed_run(loaded, manifest, summary)
        model = loaded.model.cpu().double().eval().requires_grad_(False)
        before = base._model_digest(model)
        data_path = Path(cfg.data.cache)
        source_stats = {str(path): base.file_stat(path) for path in
                        (checkpoint, data_path, Path(args.data_metadata), Path(args.quality_warning))}
        metadata = json.loads(Path(args.data_metadata).read_text())
        datasets = dict(zip(('train', 'test'), base.load_ensembles(data_path, mmap=True)))
        binding = base.validate_metadata(metadata, datasets,
            actual_sha256=base.sha256(data_path, deadline), manifest_sha256=manifest['data_sha256'])
        masks = diagnostic_partitions(metadata)
        collector = base.J1J2Collector(model, deadline=deadline, samples=128,
            progress=lambda stage, done, total, extra:
                print(f'未热化诊断报告 {stage}：{done}/{total}（非平衡态认证）', flush=True))
        steps = cfg.training.max_rg_steps
        arrays, transport, identity, paths = {}, {}, {}, None
        for split, dataset in datasets.items():
            arrays[split] = collector.trajectories(dataset.configurations, steps=steps,
                draws=args.draws, chunk_size=2, seed=2026102601 + (split == 'test') * 10000000)
            paths, transport[split], identity[split] = base.transport_paths(model, arrays[split]['direct'])
        train_std = arrays['train']['direct'].mean(1).std(0, ddof=0)
        probes = {name: [] for name in ('microscopic_information', 'native_C_statistics')}
        for level in range(steps + 1):
            for family in probes:
                target = 0 if family == 'microscopic_information' else level
                probes[family].append(base.fit_ridge(arrays['train']['direct'][:, :, level].mean(1),
                    arrays['train']['observables'][:, :, target].mean(1)))
        closure, physical = {}, {}
        for name, (split, mask) in masks.items():
            deadline.check()
            rows = [row for row, selected in zip(metadata[split], mask) if selected]
            closure[name] = base.grouped_closure_metrics(arrays[split]['direct'][mask],
                transport[split][mask], identity[split][mask], paths, train_std, rows)
            physical[name] = base.physical_readout_metrics(probes, arrays[split], transport[split],
                                                          identity[split], paths, mask)
        pca = base.pca_train(arrays['train']['direct'])
        history = loaded.payload.get('history') or json.loads((Path(args.run) / 'history_progress.json').read_text())
        history = [row for row in history if row.get('epoch', row.get('step', 0)) <= 12000]
        font = base.make_figures(output, arrays, metadata, pca, history, font=args.font, deadline=deadline)
        headers = add_warning_banners(output, font=font, deadline=deadline)
        saved = dict(paths=paths, TRAIN_target_coordinate_std=train_std,
            pca_mean=pca['mean'], pca_components=pca['components'],
            pca_explained_variance_ratio=pca['explained_variance_ratio'],
            observable_names=np.asarray(base.OBSERVABLE_NAMES),
            encoding_protocol=np.asarray(json.dumps(collector.encoding_protocol, sort_keys=True)),
            diagnostic_only=np.asarray(True), equilibrium_not_validated=np.asarray(True),
            warning=np.asarray('；'.join(WARNING_LINES)),
            quality_warning_sha256=np.asarray(warning['source_sha256']))
        for split in arrays:
            saved.update({f'{key}_{split}': value for key, value in arrays[split].items()})
            saved[f'transport_{split}'], saved[f'identity_{split}'] = transport[split], identity[split]
            saved[f'group_ids_{split}'] = datasets[split].group_ids.numpy()
            saved[f'rho_temperature_{split}'] = np.asarray([[row['rho'], row['temperature']]
                                                           for row in metadata[split]])
        for family, fitted in probes.items():
            for level, probe in enumerate(fitted):
                saved.update({f'readout_{family}_k{level}_{key}': value for key, value in probe.items()})
        deadline.check()
        np.savez_compressed(output / 'evaluation_arrays.npz', **saved)
        if (base._model_digest(model) != before
                or any(p.grad is not None or p.requires_grad for p in model.parameters())
                or base.sha256(checkpoint, deadline) != checkpoint_sha
                or base.sha256(args.quality_warning, deadline) != warning['source_sha256']
                or source_stats != {path: base.file_stat(path) for path in source_stats}):
            raise RuntimeError('冻结 checkpoint、模型或输入文件在诊断报告期间发生变化')
        report = dict(schema_version=1, status='complete', report_complete=True, **FLAGS,
            experiment='ising_j1j2_unthermalized_diagnostic', checkpoint_step=12000,
            checkpoint_sha256=checkpoint_sha, model_sha256_before=before, model_sha256_after=before,
            data_binding=binding, data_metadata_sha256=base.sha256(args.data_metadata, deadline),
            quality_warning=warning,
            encoding_estimator=base.HALF_SUBSET_ESTIMATOR, encoding_protocol=collector.encoding_protocol,
            sampling=dict(C_draws=args.draws, CPU_threads=1, chunk_size=2,
                rows={split: len(data.configurations) for split, data in datasets.items()},
                rng_layout='固定 split/chunk/draw 种子；不使用 rho、T 或相标签'),
            evaluation_partitions=list(masks),
            omitted_partitions={'heldout_rho': '本小规模诊断全部 rho 均在 TRAIN；未实施此评价，不伪造结果。'},
            closure=closure, physical_readouts=physical,
            physical_observables=dict(names=list(base.OBSERVABLE_NAMES), labels=list(base.OBSERVABLE_LABELS),
                lattice_sizes=[datasets['train'].configurations.shape[2] // 3 ** k for k in range(steps + 1)],
                G_distances=[max(1, datasets['train'].configurations.shape[2] // 3 ** k // 4)
                             for k in range(steps + 1)],
                structure_factor_normalization='V^2；S00=m2，Spi0=stripe_x2，S0pi=stripe_y2；不是独立统计'),
            readout_contract=dict(type='固定 ridge=1e-4、TRAIN-only、各尺度独立仿射探针',
                trained_on='TRAIN 各物理行的 C 均值；无 T/rho 输入；不反馈 Q/C/R',
                target_families=['原始微观物理统计（跨尺度共用目标）', '实际 C 轨迹各尺度物理统计'],
                normalization='对应读出 TRAIN 目标 std，ddof=0；不使用 TEST 拟合或选超参数'),
            PCA=dict(fit_split='train', fit_values='C 均值、全部实际尺度',
                explained_variance_ratio=pca['explained_variance_ratio'].tolist(),
                qualification='未热化诊断数据的方差占比不等于有效物理维度'),
            flow_plot_contract='每物理行 C 均值处直接计算 R(mean_C g)-mean_C g；每 rho 最多100行；不搜根。',
            C_Jensen_diagnostic='NPZ 分别保存 mean_C R(g)、R(mean_C g) 及差值，不混同非线性估计器。',
            fixed_points=[], training_updates=0, model_training_updates=12000,
            optimizer_created=False, MI_evaluations=0, chinese_font=font, warning_banners=headers,
            qualification=QUALIFICATION,
            implemented='未热化小规模诊断：TRAIN/独立 PT 组 TEST、全路径闭合、物理事后读出与中文警告图',
            not_implemented=['平衡态采样认证', '留出 rho 评价', '独立更大尺寸测试',
                '跨训练 seed 汇总', '相变、不动点或临界指数认证', '与其它算法比较'],
            elapsed_seconds=deadline.elapsed, max_seconds=deadline.seconds)
        # Keep the original history list interface and exact loss values.
        base.write_json(output / 'loss_history.json', history)
        base.write_json(output / 'report.json', report)
        files = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != 'status.json')
        base.write_json(output / 'status.json', dict(status='complete', report_complete=True, **FLAGS,
            quality_warning=warning, qualification=QUALIFICATION,
            public_files=files, sha256={name: base.sha256(output / name) for name in files},
            elapsed_seconds=deadline.elapsed))
        return report
    except Exception as error:
        base.write_json(output / 'status.json', dict(
            status='budget_hold' if isinstance(error, base.ReportBudgetHold) else 'failed',
            report_complete=False, **FLAGS, qualification=QUALIFICATION,
            error=f'{type(error).__name__}: {error}', requires_user_action=True,
            elapsed_seconds=deadline.elapsed, training_updates=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-metadata', type=Path, required=True)
    parser.add_argument('--quality-warning', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--draws', type=int, default=5)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args(argv)
    if args.draws != 5:
        parser.error('诊断报告固定 5 个 C 重复，不允许事后减少凑预算')
    run_report(args)


if __name__ == '__main__':
    main()
