"""中文、只读的自然耦合 Q 与仿射概率 Q 同预算正式报告。

自然臂是有限七算符 Gibbs 先验下的单点条件似然，不是精确 RG，也不将
潜变量的最大特征值或接近理论的谱值自动认作热指数。旧报告器只被进程内
调用；不修改旧文件，不覆盖任何既有输出。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories
from scripts import report_ising_dimg_ablation as base
from scripts.report_ising_zh import chinese_figures


NATURAL_KIND = 'directional_even_gibbs_crossfit_pseudolikelihood_v1'
ARM_NAMES = ('dim1_control', 'dim1_natural', 'dim3_control', 'dim3_natural')
ARM_LABELS = {'dim1_control': '一维仿射概率对照', 'dim1_natural': '一维自然耦合',
              'dim3_control': '三维仿射概率对照', 'dim3_natural': '三维自然耦合'}
PATH_LABELS = ('108→54', '54→27', '108→27')


def read_json(path):
    return json.loads(Path(path).read_text())


def representation_description(manifest):
    """拒绝把非默认结构伪装成历史仿射概率解码器。"""
    override = manifest.get('representation_override')
    affine = manifest.get('config', {}).get('model', {}).get('affine_reconstruction')
    if override is not None:
        if not isinstance(override, dict) or override.get('kind') != NATURAL_KIND or affine is not False:
            raise ValueError('Unrecognized or contradictory natural representation provenance')
        return dict(arm='natural', kind=NATURAL_KIND, contract=override,
            reconstruction_units='交叉子集单点条件负对数似然，单位为每格点自然单位；不是概率箱均方误差。',
            prior='七个平移不变偶算符的有限范围定向 Gibbs 先验；g 仿射进入自然耦合。',
            exact_RG_claim=False, thermal_identity_certified=False)
    if affine is not True:
        raise ValueError('Control requires the original shared affine probability decoder')
    return dict(arm='control', kind='shared_affine_probability_mse',
        reconstruction_units='共享仿射局部 motif 概率重构；按概率箱求和的均方误差。',
        exact_RG_claim=False, thermal_identity_certified=False)


def load_representation_run(run, checkpoint=12000):
    description = representation_description(read_json(Path(run)/'manifest.json'))
    if description['arm'] == 'natural':
        from experiments.ising_coupling_representation import load_coupling_run
        return load_coupling_run(run, checkpoint)
    return load_run(run, checkpoint)


def validate_formal_run(run, loaded, checkpoint=12000):
    run = Path(run)
    cfg = RunConfig.from_dict(loaded.config)
    cfg.validate()
    manifest, summary = read_json(run/'manifest.json'), read_json(run/'summary.json')
    description = representation_description(manifest)
    base.validate_completed_run(cfg, summary, loaded.payload, checkpoint)
    metadata = loaded.payload.get('metadata', {})
    if (base.json_value(manifest.get('config')) != base.json_value(loaded.config)
            or base.json_value(metadata.get('config')) != base.json_value(loaded.config)):
        raise ValueError('Manifest/checkpoint configuration mismatch')
    if representation_description(metadata) != description:
        raise ValueError('Manifest/checkpoint representation contract mismatch')
    for receipt in (manifest, metadata):
        if receipt.get('labels_used_in_training') is not False or receipt.get('q_supervision') not in (None, {}):
            raise ValueError('Require fully unsupervised model provenance')
    if (cfg.schema_version != 5 or cfg.model.dimension != 2 or cfg.model.theory_dim not in (1, 3)
            or cfg.model.block_factor != 2 or cfg.model.width != 3 or cfg.model.categorical_states
            or cfg.model.kernel_layout != 'nonoverlapping' or cfg.model.kernel_initialization != 'random'
            or cfg.model.fixed_kernel or cfg.model.kernel_sampling != 'bernoulli'
            or cfg.training.max_rg_steps != 2 or cfg.evaluation.max_rg_steps != 2
            or tuple(cfg.training.train_modules) != ('Q', 'C', 'R')
            or cfg.training.gradient_mode != 'qc_r_separate'
            or cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',)):
        raise ValueError('Require the explicit unsupervised block-two dim1/3 depth-two QCR ablation')
    sidecar = read_json(Path(cfg.data.cache).with_suffix('.json'))
    if (sidecar.get('B') != 0. or sidecar.get('coupling_names') != ['K']
            or not manifest.get('data_sha256') or sidecar.get('data_sha256') != manifest['data_sha256']
            or metadata.get('data_sha256') != manifest['data_sha256']):
        raise ValueError('B0 source data fingerprint receipts mismatch')
    return cfg, manifest, summary, description


def root_location_records(roots, *, Kc=base.KC):
    """保留所有逆 K 分支；无交点时只报带距离的投影。"""
    result = []
    for root in roots:
        scales = []
        for scale, projection in enumerate(root.get('curve_projections_by_scale', [])):
            branches = [dict(K=float(value), absolute_K_error=abs(float(value)-Kc),
                relative_K_error=abs(float(value)-Kc)/Kc)
                for value in projection.get('exact_inverse_K_branches', [])]
            scales.append(dict(scale=scale, exact_inverse_K_branches=branches,
                exact_inverse_K_intervals=projection.get('exact_inverse_K_intervals', []),
                has_exact_inverse=bool(branches or projection.get('exact_inverse_K_intervals')),
                nearest_distance=projection.get('nearest_distance'),
                nearest_segments=projection.get('nearest_segments', []),
                projection_is_critical_point=False))
        # No cross-scale branch is selected as the physical continuation.
        drift = [dict(start=a['scale'], end=b['scale'], K_start=x['K'], K_end=y['K'],
                      signed_drift=y['K']-x['K'])
                 for a, b in zip(scales[:-1], scales[1:])
                 for x in a['exact_inverse_K_branches'] for y in b['exact_inverse_K_branches']]
        result.append(dict(root_id=root.get('root_id'), g=root.get('g'), scales=scales,
            all_adjacent_exact_branch_drifts=drift,
            caution='保留全部精确逆分支；多维最近投影不是插值临界点，不据理论邻近性挑根。'))
    return result


def plot_natural_losses(history, output):
    import matplotlib.pyplot as plt
    rows = [('anchor', '微观子集一致性'), ('coarse_consistency', '粗尺度子集一致性'),
            ('separation', 'Q 条件负对数似然'), ('commutator', '单步闭合'),
            ('rollout', '多步闭合'), ('kernel_noncollapse', 'C 方差下限'),
            ('kernel_sign', '颜色保持'), ('total', '总损失'), ('qcr_total', 'QCR 总损失')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for ax, split, title in zip(axes[0], ('train', 'test'), ('训练集', '测试集（仅评价）')):
        for key, label in rows:
            available = [row for row in history if split+'_'+key in row]
            if available:
                ax.plot([row.get('epoch', row.get('step')) for row in available],
                        [row[split+'_'+key] for row in available], label=label)
        ax.set(xlabel='实际训练更新次数', ylabel='未加权记录值', title=title, yscale='symlog')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    for split, style, label in (('train', '-', '训练集'), ('test', '--', '测试集（仅评价）')):
        available = [row for row in history if split+'_rsmi_bits' in row]
        if available:
            axes[1, 0].plot([row['epoch'] for row in available], [row[split+'_rsmi_bits'] for row in available],
                            style, label=label)
        keys = sorted({key for row in history for key in row if key.startswith(split+'_reconstruction_nll_level_')})
        for key in keys:
            scale = int(key.rsplit('_', 1)[-1])
            available = [row for row in history if key in row]
            axes[1, 1].plot([row['epoch'] for row in available], [row[key] for row in available],
                           style, color=f'C{scale}', label=label+f'：k={scale}')
    axes[1, 0].set(xlabel='实际训练更新次数', ylabel='经验 MI（比特）', title='既有经验互信息记录；未重算')
    axes[1, 1].set(xlabel='实际训练更新次数', ylabel='每格点条件负对数似然（自然单位）',
                  title='七算符自然耦合解码；不是概率重构均方误差')
    for ax in axes[1]:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle('固定更新预算；标签和理论指数不参与训练或检查点选择')
    fig.savefig(Path(output)/'loss_comparison.png', dpi=160)
    plt.close(fig)


def write_representation_report(loaded, train, test, train_labels, test_labels, history, output,
                                *, representation, **kwargs):
    """复用原有全谱/闭合报告，完成标志只在本封装全部成功后写入。"""
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError('Need a new empty report directory')
    original_write = base.write_json
    completion = []
    def annotated_write(path, value):
        name = Path(path).name
        if name == 'report.json':
            completion.append(value)
            return
        if name in ('metrics.json', 'critical_exponents.json'):
            value = dict(value, representation=representation,
                logged_loss_caution='MI 与 Q 分项仅取已有训练日志；自然臂为条件似然，对照为概率箱均方误差，单位不可直接横比。',
                critical_location=root_location_records(value.get('roots', [])))
        original_write(path, value)
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, 'write_json', annotated_write))
        if representation['arm'] == 'natural':
            stack.enter_context(patch.object(base, 'plot_losses', plot_natural_losses))
        result = base.write_report(loaded, train, test, train_labels, test_labels, history, output, **kwargs)
    result.update(representation=representation, critical_location=root_location_records(result.get('roots', [])))
    if representation['arm'] == 'natural':
        decoder = loaded.model.theory_encoder.distribution_decoder
        original_write(output/'coupling_decoder.json', dict(representation=representation,
            weight=decoder.linear.weight.detach().cpu().numpy(), bias=decoder.linear.bias.detach().cpu().numpy(),
            formula='H_g(s)=-sum_j(a_j+W_j g) S_j(s)',
            labels_used_for_coefficients=False, exact_RG_claim=False))
    if len(completion) != 1:
        raise RuntimeError('Missing unique underlying report completion')
    original_write(output/'report.json', dict(completion[0], representation=representation,
        report_driver='scripts.report_ising_coupling_representation',
        public_files=sorted(path.name for path in output.iterdir() if path.is_file())))
    return result


def report_run(run, output, *, checkpoint=12000, threads=1, sampling_repeats=5):
    run, output = Path(run), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refuse to overwrite or repeat a report')
    if threads != 1 or sampling_repeats != 5:
        raise ValueError('Formal stage-two reporting requires one CPU thread and five C repeats')
    torch.set_num_threads(1)
    loaded = load_representation_run(run, checkpoint)
    cfg, manifest, summary, description = validate_formal_run(run, loaded, checkpoint)
    train_data, test_data = base.report_datasets(cfg)
    options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
        spin_flip_augment=cfg.data.spin_flip_augment, sampling_seed=12345)
    train = collect_trajectories(loaded.model, train_data.configurations, **options)
    test = collect_trajectories(loaded.model, test_data.configurations, **options)
    repeats = [base.closure_details(loaded.model,
        collect_trajectories(loaded.model, test_data.configurations, **dict(options, sampling_seed=seed)), train['direct'])
        for seed in range(sampling_repeats)]
    history = loaded.payload.get('history', summary.get('history', []))
    output.mkdir(parents=True, exist_ok=False)
    return write_representation_report(loaded, train, test, train_data.couplings.numpy(), test_data.couplings.numpy(),
        history, output, representation=description,
        repeats=dict(seeds=list(range(sampling_repeats)), closure=repeats,
            caution='同一模型上固定物理构型的条件 C 抽样；不是独立物理系综或指数置信区间。'),
        data_sha256=manifest['data_sha256'],
        lattice_sizes=[train_data.configurations.shape[2]//cfg.model.block_factor**k for k in range(3)])


def response_run(run, data, output, *, c_repeats=5, bootstraps=1000, seed=12345,
                 expected_checkpoint_sha256=None):
    from scripts import diagnose_ising_dimg_response as response
    if c_repeats != 5 or bootstraps != 1000:
        raise ValueError('Formal response uses five conditional C draws and 1000 grouped bootstraps')
    loaded = load_representation_run(run, 12000)
    _, _, _, description = validate_formal_run(run, loaded, 12000)
    original_write = response.write_json
    def annotated_write(path, value):
        value = dict(value, representation=description,
            representation_report_driver='scripts.report_ising_coupling_representation')
        if Path(path).name == 'comparison.json':
            value['protocol'] = '各臂固定实际12000无监督终点；CPU单线程；既有物理构型和五次条件C抽样，12系综成组重采；不依理论挑谱。'
        original_write(path, value)
    with patch.object(response, 'load_run', load_representation_run), patch.object(response, 'write_json', annotated_write):
        return response.diagnose(source_run=run, data=data, output=output, k_count=3,
            c_repeats=c_repeats, seed=seed, expected_checkpoint_sha256=expected_checkpoint_sha256,
            smoke=False, bootstraps=bootstraps)


def comparison_inputs(specifications):
    reports, responses = {}, {}
    for specification in specifications:
        name, directory = specification.split('=', 1)
        if name not in ARM_NAMES or name in reports:
            raise ValueError('Need each of the four predeclared named arms exactly once')
        directory = Path(directory)
        report_dir = directory if (directory/'metrics.json').is_file() else directory/'report'
        complete = read_json(report_dir/'report.json')
        metrics = read_json(report_dir/'metrics.json')
        dim = 1 if name.startswith('dim1_') else 3
        arm = 'natural' if name.endswith('_natural') else 'control'
        if (complete.get('status') != 'complete' or complete.get('report_complete') is not True
                or metrics.get('checkpoint_epoch') != 12000 or metrics.get('theory_dim') != dim
                or metrics.get('block_factor') != 2 or metrics.get('representation', {}).get('arm') != arm
                or metrics.get('labels_used_in_training') is not False):
            raise ValueError('Comparison requires all completed, declared, unsupervised actual-12000 endpoints')
        response_dir = report_dir.parent/'response'
        status = read_json(response_dir/'status.json')
        physical = read_json(response_dir/'comparison.json')
        if (status.get('status') != 'complete' or status.get('workflow_complete') is not True
                or physical.get('status') != 'complete' or physical.get('theory_dim') != dim
                or physical.get('representation', {}).get('arm') != arm):
            raise ValueError('Each physical-response report must complete with matching representation')
        reports[name], responses[name] = metrics, physical
    if set(reports) != set(ARM_NAMES):
        raise ValueError('Need dim1/dim3 control/natural four-arm comparison')
    return {name: reports[name] for name in ARM_NAMES}, {name: responses[name] for name in ARM_NAMES}


def compare_reports(specifications, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refuse overwrite or repeat of a comparison')
    reports, responses = comparison_inputs(specifications)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='complete', actual_updates_per_arm=12000, reports=reports, physical_response=responses,
        critical_location={name: root_location_records(report.get('roots', [])) for name, report in reports.items()},
        selection=dict(all_roots_retained=True, all_spectra_retained=True, selected_thermal_eigenvalue=None,
                       labels_used_in_training=False, theory_used_for_selection=False),
        caution='自然臂是有限七算符条件似然先验；重构损失单位不同。跨模型坐标和标准差不同，NRMSE改善不等于同坐标绝对误差改善。所有热指数均须另行认证。')
    base.write_json(output/'comparison.json', result)
    import matplotlib.pyplot as plt
    with chinese_figures():
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), layout='constrained')
        for index, name in enumerate(ARM_NAMES):
            report, label = reports[name], ARM_LABELS[name]
            for root in result['critical_location'][name]:
                for scale in root['scales']:
                    offset = (scale['scale']-1)*.1
                    for branch in scale['exact_inverse_K_branches']:
                        axes[0].scatter(index+offset, branch['K'], color=f'C{index}', s=24)
                    if not scale['has_exact_inverse']:
                        for segment in scale['nearest_segments']:
                            axes[0].scatter(index+offset, segment['K'], edgecolor=f'C{index}', facecolor='none', marker='D', s=24)
            by_path = {(row['start'], row['end']): row for row in report['closure']['test']['all_start_horizon']}
            ordered = [by_path[pair] for pair in ((0, 1), (1, 2), (0, 2))]
            axes[1].plot(range(3), [row['nrmse'] for row in ordered], 'o-', label=label)
            for root in report.get('roots', []):
                for eigenvalue in root.get('eigenvalues', []):
                    axes[2].scatter(index-.04, eigenvalue['real'], color=f'C{index}', marker='o', s=22,
                        facecolors=f'C{index}' if root.get('geometric_support_candidate') else 'none')
                    axes[2].scatter(index+.04, eigenvalue['imag'], color=f'C{index}', marker='x', s=20)
        axes[0].axhline(base.KC, color='magenta', ls=':', label='Kc（仅评价）')
        axes[0].set(title='全部根的精确逆 K 与最近投影', ylabel='K；空心菱形仅为投影',
                    xticks=range(4), xticklabels=[ARM_LABELS[name] for name in ARM_NAMES])
        axes[0].legend(fontsize=8)
        axes[1].set(title='测试集各路径归一闭合误差', ylabel='按训练目标标准差归一的均方根误差',
                    xticks=range(3), xticklabels=PATH_LABELS, yscale='log')
        axes[1].legend(fontsize=8)
        axes[2].axhline(2., color='gray', ls=':', label='热倍率理论参考（非选择规则）')
        axes[2].set(title='全部根的完整雅可比谱；圆为实部，叉为虚部', ylabel='谱值实部／虚部',
                    xticks=range(4), xticklabels=[ARM_LABELS[name] for name in ARM_NAMES])
        axes[2].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.tick_params(axis='x', rotation=20)
        fig.suptitle('固定12000更新：自然耦合表示消融；几何接近与谱值接近理论均不认证热指数')
        fig.savefig(output/'comparison.png', dpi=160)
        plt.close(fig)
    base.write_json(output/'report.json', dict(status='complete', report_complete=True, arms=list(ARM_NAMES),
        actual_updates_per_arm=12000, physical_response_complete=True, workflow_complete=True,
        labels_used_in_training=False, selected_thermal_eigenvalue=None))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', '--mode', choices=('report', 'response', 'compare'), required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=12000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--sampling-repeats', type=int, default=5)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--c-repeats', type=int, default=5)
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--arms', '--compare', nargs='+')
    args = parser.parse_args(argv)
    if args.kind == 'compare':
        if not args.arms or args.run:
            parser.error('Comparison needs --arms and no --run')
        return compare_reports(args.arms, args.output)
    if not args.run or args.arms:
        parser.error('Per-arm report needs --run and no --arms')
    if args.kind == 'report':
        return report_run(args.run, args.output, checkpoint=args.checkpoint, threads=args.threads,
                          sampling_repeats=args.sampling_repeats)
    if not args.data or args.threads != 1 or args.checkpoint != 12000:
        parser.error('Response needs --data, one CPU thread, actual checkpoint12000')
    return response_run(args.run, args.data, args.output, c_repeats=args.c_repeats, bootstraps=args.bootstraps,
        seed=args.seed, expected_checkpoint_sha256=args.expected_checkpoint_sha256)


if __name__ == '__main__':
    main()
