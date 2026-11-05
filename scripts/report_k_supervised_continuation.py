"""固定累计检查点的 K 监督过程报告；全缓存单次 C 诊断及保存编码的 R 分析。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint
from scripts.report_ising_zh import chinese_figures
from scripts.report_ising_local_response import KC, polynomial_response, predict_and_derivative
from scripts.report_k_supervised_pilot import report_one


DATA_SHA = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'


def validate_checkpoint(loaded, manifest, epoch):
    """拒绝换配方、换监督目标、错检查点或不匹配的运行清单。"""
    cfg, meta = loaded.config, loaded.payload.get('metadata', {})
    if loaded.payload.get('epoch') != epoch:
        raise ValueError('Requested checkpoint differs from its actual saved epoch')
    if json.dumps(cfg, sort_keys=True) != json.dumps(manifest['config'], sort_keys=True):
        raise ValueError('Checkpoint config differs from the run manifest')
    model, training, evaluation = cfg['model'], cfg['training'], cfg['evaluation']
    expected_model = dict(dimension=2, theory_dim=1, block_factor=2, hidden=96,
        width=3, kernel_layout='nonoverlapping', fixed_kernel=False,
        kernel_initialization='random', kernel_sampling='bernoulli',
        affine_reconstruction=True, binary_lookup=True, categorical_states=0)
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError('Expected the unchanged scalar b2 probabilistic affine-Q architecture')
    expected_training = dict(max_rg_steps=2, system_batch=16, sample_batch=128,
        seed=91, gradient_mode='qc_r_separate', train_modules=['Q', 'C', 'R'],
        learning_rate=.001, lr_schedule='multistep', lr_gamma=.3)
    if (any(training.get(key) != value for key, value in expected_training.items()) or
            list(training.get('lr_milestones', ())) != [6000, 9000] or
            evaluation.get('samples') != 128 or evaluation.get('max_rg_steps') != 2 or
            cfg['data'].get('coupling_names') != ['K']):
        raise ValueError('Expected unchanged depth, full diagnostic samples and cumulative LR policy')
    loss = cfg['loss']
    expected_weights = dict(anchor=2., coarse_consistency=5., kernel_noncollapse=2.,
                            separation=10., commutator=5.)
    expected_loss = dict(rollout_weight=5., kernel_sign_weight=1.,
        kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
        balanced_reconstruction_weight=0., isotropy_weight=0., tangent_weight=0.,
        orientation_weight=0., kernel_neutral_weight=0.)
    if (any(loss.get(key) != value for key, value in expected_loss.items()) or
            any(loss.get('weights', {}).get(key) != value for key, value in expected_weights.items())):
        raise ValueError('Expected unchanged reconstruction, consistency, closure and C objectives')
    expected_meta = dict(training_driver='rsmi_separate_qcr', labels_used_in_training=True,
                         independent_c=True, estimator='histogram',
                         environment_mode='full_shell', buffer=1, rsmi_weight=2.,
                         precision='torch.float64', data_sha256=DATA_SHA)
    for record in (meta, manifest):
        if any(record.get(key) != value for key, value in expected_meta.items()):
            raise ValueError('Checkpoint must retain the documented full independent supervised recipe')
        supervision = record.get('q_supervision', {})
        if (supervision.get('weight') != 10. or supervision.get('scale') != 0 or
                supervision.get('fitting_split') != 'TRAIN only' or
                supervision.get('coupling_names') != ['K'] or
                supervision.get('target') != 'raw coupling labels, Q(full microscopic batch)' or
                supervision.get('gradient') != 'Q only; no labels for coarse levels or C/R'):
            raise ValueError('Expected unchanged weight-10 microscopic TRAIN K supervision')
    return cfg


def validate_full_data(train, test):
    for dataset, count in ((train, 516), (test, 252)):
        if (tuple(dataset.configurations.shape) != (count, 128, 108, 108, 1) or
                dataset.configurations.dtype != torch.float64 or
                dataset.couplings is None or dataset.couplings.numel() != count):
            raise ValueError('Expected the full original L108 cache, 768 ensembles x128 samples')
    a, b = train.couplings.numpy().ravel(), test.couplings.numpy().ravel()
    knots, counts = np.unique(np.concatenate((a, b)), return_counts=True)
    if (len(knots) != 64 or not np.all(counts == 12) or
            np.intersect1d(a, b).size or not np.isfinite(knots).all() or
            not np.any(np.isclose(knots, KC, atol=1e-12, rtol=0))):
        raise ValueError('Expected 64 complete K knots, 12 independent ensembles each, including Kc')


def response_from_encodings(model, encodings, epoch, *, bootstraps=1000):
    """仅调用已训练 R；同一次系综重采样同时用于各尺度及其 R 预测。"""
    if bootstraps < 1:
        raise ValueError('Positive bootstrap count required')
    arrays, labels = [], []
    for split in ('train', 'test'):
        values = np.asarray(encodings[split], dtype=np.float64)
        k = np.asarray(encodings[split + '_labels'], dtype=np.float64).ravel()
        if (values.ndim != 3 or values.shape[1:] != (3, 1) or len(k) != len(values) or
                not len(values) or not np.isfinite(values).all() or not np.isfinite(k).all()):
            raise ValueError('Finite complete three-scale scalar encodings and K labels required')
        arrays.append(values[:, :, 0]); labels.append(k)
    values, labels = np.concatenate(arrays), np.concatenate(labels)
    k, counts = np.unique(labels, return_counts=True)
    if len(k) < 5 or len(set(counts.tolist())) != 1 or counts[0] < 2:
        raise ValueError('Require a uniform independent ensemble count >=2 at each K')
    if not k[0] <= KC <= k[-1]:
        raise ValueError('Saved K knots must bracket Kc; no critical extrapolation')
    replicas = int(counts[0])
    blocks = np.stack([values[labels == value] for value in k])
    means = blocks.mean(1)
    model = model.cpu().double().eval()
    prediction, derivative = predict_and_derivative(model, blocks[:, :, :-1])
    mean_prediction = prediction.mean(1)
    r_at_mean, dr_at_mean = predict_and_derivative(model, means[:, :-1])
    rng = np.random.default_rng(12345)
    ids = rng.integers(0, replicas, (bootstraps, len(k), replicas))
    boot = blocks[np.arange(len(k))[None, :, None], ids].mean(2)
    rboot = prediction[np.arange(len(k))[None, :, None], ids].mean(2)
    windows = []
    for width in (.02, .03, .04):
        mask = abs(k - KC) <= width
        if mask.sum() >= 5:
            scaled = (k[mask] - KC) / width
            matrix = np.stack([scaled**i for i in range(4)], axis=1)
            weights = np.linalg.pinv(matrix)[1] / width
            slopes = np.einsum('n,nk->k', weights, means[mask])
            sampled_slopes = np.einsum('n,bnk->bk', weights, boot[:, mask])
            if (np.any(abs(slopes[:-1]) <= 1e-10) or
                    np.any(np.sum(abs(sampled_slopes[:, :-1]) > 1e-10, axis=0) == 0)):
                windows.append(dict(center=KC, half_width=width, K_count=int(mask.sum()),
                    status='undefined_input_derivative', transitions=[],
                    caution='输入导数为零或无有效自助抽样分母；不报告无定义的响应比。'))
                continue
        windows.append(polynomial_response(k, means, boot, mean_prediction, rboot, KC, width))
    critical_g = np.array([np.interp(KC, k, means[:, scale]) for scale in range(3)])
    critical_mean_r = np.array([np.interp(KC, k, mean_prediction[:, scale]) for scale in range(2)])
    critical_r, critical_dr = predict_and_derivative(model, critical_g[:-1])
    critical_mean_dr = [float(np.interp(KC, k, derivative.mean(1)[:, scale])) for scale in range(2)]
    return dict(status='complete', actual_epochs=epoch, K=k.tolist(),
        mean_g=means.tolist(), mean_R=mean_prediction.tolist(), R_of_mean=r_at_mean.tolist(),
        mean_full_DR=derivative.mean(1).tolist(), DR_at_mean=dr_at_mean.tolist(),
        jensen_gap=(mean_prediction-r_at_mean).tolist(), critical_windows=windows,
        critical=dict(Kc=KC, g_by_scale=critical_g.tolist(),
            mean_R_by_scale=critical_mean_r.tolist(), R_of_mean_by_scale=critical_r.tolist(),
            mean_full_DR_by_scale=critical_mean_dr, DR_at_mean_by_scale=critical_dr.tolist(),
            jensen_gap_by_scale=(critical_mean_r-critical_r).tolist(),
            reference='保存的 Kc 系综；若没有精确 Kc 节点，则仅在已有 K 节点间线性插值。'),
        bootstrap=dict(repeats=bootstraps, seed=12345, replicas_per_K=replicas,
            unit='每个固定 K 内的独立系综；相同重采样用于全部尺度及配对 R 预测',
            limitation='区间以固定模型和已保存 C 抽样为条件；不包含训练种子、有限尺寸或多项式偏差，不是热乘子或 ν 的置信区间。'),
        methodology='比较 d(E[g 的下一尺度])/dK 与 d(E[R(g)])/dK，再除以当前尺度 d(E[g])/dK。仅为 K 扫描路径响应，不认证热乘子。',
        Q_C_inference=False, MI_calculation=False, optimizer_updates=0)


def draw_response(report, model, out):
    means, prediction = np.asarray(report['mean_g']), np.asarray(report['mean_R'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for scale in range(2):
        ax = axes[0, scale]
        ax.plot(means[:, scale], means[:, scale + 1], '.-', label='数据平均轨迹')
        ax.plot(means[:, scale], prediction[:, scale], '--', label='系综配对的平均 R')
        grid = np.linspace(means[:, scale].min(), means[:, scale].max(), 301)
        mapped, _ = predict_and_derivative(model, grid)
        ax.plot(grid, mapped, color='gray', label='R 作用于平均 g')
        ax.plot(grid, grid, 'k:', lw=.9, label='恒等映射')
        ax.set(xlabel='输入 g', ylabel='下一尺度 g 或 R(g)', title=f'尺度 k={scale}→{scale + 1}：局部闭合')
        for key, name, style in (('empirical_response', '数据切线响应比', 'o-'),
                                 ('predicted_response', 'R 配对响应比', 's-')):
            x, y = [], []
            for window in report['critical_windows']:
                if window['status'] != 'complete':
                    continue
                row = window['transitions'][scale]
                x.append(window['half_width']); y.append(row[key])
                low, high = row[key + '_bootstrap_95']
                axes[1, scale].vlines(window['half_width'], low, high, color='gray', alpha=.6)
            axes[1, scale].plot(x, y, style, label=name)
        axes[1, scale].set(xlabel='K 窗口半宽', ylabel='局部响应比',
                            title=f'尺度 k={scale}→{scale + 1}：Kc 邻域三次拟合', xticks=[.02, .03, .04])
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f'累计 {report["actual_epochs"]} 次更新：局部响应诊断（非热乘子认证）')
    fig.supxlabel('竖线为固定模型和保存抽样条件下的系综自助抽样 95% 区间；不包含拟合偏差或训练种子不确定性。', fontsize=9)
    fig.savefig(out / 'local_response.png', dpi=160)
    plt.close(fig)


def check_trend_owner(root):
    """只更新本脚本在新报告目录内创建的累计趋势文件。"""
    owner = root / 'trend_owner.json'
    expected = dict(writer='scripts.report_k_supervised_continuation',
                    schema_version=1, data_sha256=DATA_SHA)
    if owner.exists():
        if json.loads(owner.read_text()) != expected:
            raise ValueError('Trend ownership receipt differs; refusing to replace its files')
    elif any((root / name).exists() for name in ('trend.json', 'trend.png')):
        raise FileExistsError('Existing trend files have no ownership receipt; refusing to overwrite')
    return owner, expected


def collect_trend(root, reference_report=None):
    """读取已完成节点，完整保留根，不把各节点 F 编号视为分支身份。"""
    snapshots, skipped = [], []
    if reference_report is not None:
        path = reference_report / 'critical_exponents.json'
        saved = json.loads(path.read_text())
        snapshots.append(dict(actual_epochs=saved['checkpoint_epoch'],
            roots=saved['roots'], reference=True, source=str(path.resolve()),
            source_sha256=file_fingerprint(path)))
    for path in sorted(root.glob('epoch_*/checkpoint_summary.json')):
        saved = json.loads(path.read_text())
        if saved.get('status') != 'complete':
            skipped.append(str(path.resolve()))
            continue
        snapshots.append(dict(actual_epochs=saved['actual_epochs'], roots=saved['roots'],
            reference=False, source=str(path.resolve()), source_sha256=file_fingerprint(path)))
    epochs = [row['actual_epochs'] for row in snapshots]
    if (any(not isinstance(epoch, int) or epoch < 1 for epoch in epochs) or
            len(epochs) != len(set(epochs))):
        raise ValueError('Trend checkpoints must have unique positive actual cumulative epochs')
    snapshots.sort(key=lambda row: row['actual_epochs'])
    return dict(snapshots=snapshots, skipped=skipped, block_factor=2, Kc=KC,
        root_matching_across_checkpoints=False, all_saved_roots_retained=True,
        note='所有根均按各自检查点独立散点显示；不跨检查点连接 F 编号，不按理论值或指数筛选。ν 仅沿用原报告的有条件结果。')


def draw_trend(report, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    labeled = [set(), set(), set()]
    for snapshot in report['snapshots']:
        epoch = snapshot['actual_epochs']
        for root in snapshot['roots']:
            multiplier = root.get('multiplier')
            stability = root.get('stability', 'unknown')
            outside = root.get('outside_training_support') is not False
            negative = multiplier is not None and multiplier < 0
            color = 'purple' if negative else 'forestgreen' if stability == 'stable' else 'darkorange'
            state = {'stable': '稳定', 'unstable': '不稳定', 'nonhyperbolic': '非双曲',
                     'saddle': '鞍点', 'unknown': '未知'}.get(stability, '未知')
            category = ('训练支持之外或未知' if outside else '训练支持之内') + '；' + state
            if negative:
                category += '；负 λ'
            if snapshot['reference']:
                category += '；短试验参照'
            values = ([multiplier], [root.get('nu')], root.get('microscopic_inverse_coupling', []))
            for panel, candidates in enumerate(values):
                finite = [float(value) for value in candidates
                          if value is not None and np.isfinite(float(value))]
                if not finite:
                    continue
                label = category if category not in labeled[panel] else None
                axes[panel].scatter([epoch] * len(finite), finite, marker='D' if snapshot['reference'] else 'o',
                    s=45, edgecolors=color, facecolors='none' if outside else color, label=label)
                labeled[panel].add(category)
    axes[0].axhline(2., color='black', ls=':', label='理论参照 λ=b=2')
    axes[1].axhline(1., color='black', ls=':', label='理论参照 ν=1')
    axes[2].axhline(KC, color='black', ls=':', label='已知微观输入 Kc')
    epochs = [row['actual_epochs'] for row in report['snapshots']]
    for ax, title, ylabel in zip(axes,
            ('所有保存根的完整 R 乘子', '有条件的热指数 ν；空值保持缺失', '全部微观逆耦合分支'),
            ('λ', '有条件的 ν', '微观 K')):
        ax.set(title=title, xlabel='累计更新次数', ylabel=ylabel, xticks=epochs)
        ax.tick_params(axis='x', labelrotation=25)
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('预先固定的过程节点：根按检查点独立显示，不跨节点匹配或连接 F 编号')
    fig.supxlabel('菱形为先前短试验参照；空心标记表示训练支持之外或未知。接近理论参照不构成热方向认证。', fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def update_trend(root, reference_report=None):
    owner, receipt = check_trend_owner(root)
    report = collect_trend(root, reference_report)
    if not owner.exists():
        with owner.open('x') as stream:
            json.dump(receipt, stream, indent=2)
    pending = root / 'trend.pending.png'
    with chinese_figures():
        draw_trend(report, pending)
    pending.replace(root / 'trend.png')
    pending_json = root / 'trend.json.tmp'
    pending_json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pending_json.replace(root / 'trend.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--trend-root', type=Path)
    parser.add_argument('--reference-report', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.checkpoint < 1:
        raise ValueError('Positive cumulative checkpoint required')
    if args.reference_report is not None and args.trend_root is None:
        raise ValueError('--reference-report requires --trend-root')
    if args.trend_root is not None:
        if args.output.parent.resolve() != args.trend_root.resolve():
            raise ValueError('Checkpoint output must be directly inside its new owned trend root')
        check_trend_owner(args.trend_root)
    torch.set_num_threads(2)
    loaded = load_run(args.run, args.checkpoint)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    cfg = validate_checkpoint(loaded, manifest, args.checkpoint)
    checkpoint_sha = file_fingerprint(loaded.checkpoint)
    if file_fingerprint(cfg['data']['cache']) != DATA_SHA:
        raise ValueError('Original full cache fingerprint changed')
    train, test = load_ensembles(cfg['data']['cache'])
    validate_full_data(train, test)
    args.output.mkdir(parents=True, exist_ok=False)
    full = args.output / 'full_report'
    with chinese_figures():
        one = report_one(args.run, args.checkpoint, full, train, test, args.device)
        with np.load(full / 'encodings.npz', allow_pickle=False) as saved:
            encodings = {name: saved[name].copy() for name in
                         ('train', 'test', 'train_labels', 'test_labels')}
        response = response_from_encodings(loaded.model, encodings, args.checkpoint)
        draw_response(response, loaded.model, args.output)
    if file_fingerprint(loaded.checkpoint) != checkpoint_sha:
        raise RuntimeError('Checkpoint changed during diagnostics; preserve this incomplete report')
    exponents = json.loads((full / 'critical_exponents.json').read_text())
    exponents['selection']['epoch_selection'] = '预先固定的累计更新检查点；保留全部根，不按指数筛选检查点或分支。'
    (full / 'critical_exponents.json').write_text(json.dumps(exponents, indent=2, allow_nan=False) + '\n')
    response['source_sha256'] = dict(checkpoint=checkpoint_sha,
                                    encodings=file_fingerprint(full / 'encodings.npz'))
    (args.output / 'local_response.json').write_text(json.dumps(response, indent=2, allow_nan=False) + '\n')
    summary = dict(status='complete', actual_epochs=args.checkpoint,
        checkpoint=str(loaded.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        data_sha256=DATA_SHA, q_coupling_weight=10., roots=exponents['roots'],
        all_roots_retained=True, root_matching_across_checkpoints=False,
        conditional_nu_only=True, critical_windows=response['critical_windows'],
        critical=response['critical'], supervision_mse=one['supervision_mse'],
        closure=one['closure'], diagnostic=dict(device=args.device, systems=[516, 252],
            samples=128, scales=[108, 54, 27], C_sampling_seed=12345,
            extra_C_sampling_repeats=0, bootstrap_repeats=1000),
        note='过程节点按累计更新次数预先确定；局部 F 编号不跨检查点匹配。监督仅用于微观训练 K，不设置理论指数目标。')
    (args.output / 'checkpoint_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if args.trend_root is not None:
        update_trend(args.trend_root, args.reference_report)
    print(json.dumps(dict(output=str(args.output.resolve()), actual_epochs=args.checkpoint,
                          roots=len(exponents['roots'])), ensure_ascii=False))
    return summary


if __name__ == '__main__':
    main()
