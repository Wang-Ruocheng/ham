"""Short supervised pilot: matched controls, isolated inference, no launch in tests."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

from experiments.rsmi_separate_qcr import build_parser as training_parser
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import report_k_supervised_pilot as report
from scripts import run_ising2d_k_supervised_pilot as pipeline
from scripts import report_ising_zh as zh
from scripts.run_ising2d_block2_latest import differences
from scripts.run_scalar_lr_sweep import make_configs


def source_fixture(tmp_path):
    job = tmp_path / 'historical'
    source = job / 'run'
    source.mkdir(parents=True)
    cfg = make_configs(tmp_path / 'L108.pt')['multistep']
    cfg['model']['block_factor'] = 2
    records = {}
    mi = job / 'mi_cache'
    mi.mkdir()
    for split, key, count in (
        ('train', '53b4cb45bda6c9271a4d18348e9f20cf75bbcf9f63647e76e5ca401dc19f5bea', 516),
        ('test', '85992c4ab2ce55ab36a0c45842d3a6e9c695bdd418c4cc3688117d6a60ebd8b6', 252),
    ):
        path = mi / f'mi-v1-{key}.pt'
        path.write_bytes(b'plan-only fixture; never load statistics')
        records[split] = dict(cache=dict(path=str(path), key=key, hit=True, cache_version=1),
                              data_shape=[count, 128, 108, 108, 1])
    manifest = dict(config=cfg, data_sha256=pipeline.DATA_SHA,
        labels_used_in_training=False, source=None, training_driver='rsmi_separate_qcr',
        independent_c=True, precision='torch.float64',
        execution_optimizations=dict(mi_statistics_cache={
            split: row['cache'] for split, row in records.items()}))
    summary = dict(config=cfg, actual_epochs=12000, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False,
        lr_scheduler_state=LearningRatePolicy(RunConfig.from_dict(cfg).training).checkpoint_state(12000))
    for name, value in (('manifest', manifest), ('summary', summary)):
        (source / f'{name}.json').write_text(json.dumps(value))
    (source / 'training_step_3000.pt').write_bytes(b'preserved checkpoint; do not load in plan')
    (job / 'mi_preprocessing.json').write_text(json.dumps(dict(status='complete',
        data_sha256=pipeline.DATA_SHA, cache_version=1, device='cpu', model_initialized=False,
        optimizer_updates=0, sampling_draws=0, splits=records)))
    return source, cfg


def test_plan_only_changes_epochs_and_preserves_schedule_recipe_and_cache(tmp_path, monkeypatch):
    source, cfg = source_fixture(tmp_path)
    before = {p: p.read_bytes() for p in source.parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No launch in plan'))
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **kw: pytest.fail('No GPU query in plan'))
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('No tensor/checkpoint load in plan'))
    output = tmp_path / 'new_pilot'
    plan = pipeline.build_plan(source, output)
    assert differences(cfg, plan['config']) == ['training.epochs']
    assert cfg['training']['epochs'] == 12000
    assert plan['config']['training']['epochs'] == 3000
    assert plan['config']['data'] == cfg['data']
    assert plan['config']['model'] == cfg['model']
    assert plan['config']['loss'] == cfg['loss']
    assert plan['config']['training']['lr_milestones'] == [6000, 9000]
    policy = LearningRatePolicy(RunConfig.from_dict(plan['config']).training)
    assert policy.rate_for_update(1) == policy.rate_for_update(3000) == .001
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not output.exists()


def test_supervised_command_is_fresh_explicit_bounded_and_chinese(tmp_path):
    source, _ = source_fixture(tmp_path)
    plan = pipeline.build_plan(source, tmp_path / 'pilot')
    command = plan['command']
    assert command[3:6] == ['scripts.report_ising_zh', '--module', 'experiments.rsmi_separate_qcr']
    args = training_parser().parse_args(command[6:])
    assert args.source is None and args.resume_checkpoint is None
    assert args.q_coupling_weight == 10
    assert args.max_training_seconds == 900 and args.max_wall_seconds is None
    assert args.skip_final_report and args.independent_c
    assert args.device == 'cuda:0'
    assert args.mi_cache_dir == source.parent / 'mi_cache'
    assert not args.allow_nondefault_recipe and not args.mi_initialize
    assert args.freeze_c_checkpoint is None
    assert plan['report_command'][3:6] == [
        'scripts.report_ising_zh', '--module', 'scripts.report_k_supervised_pilot']
    assert 'scripts.report_k_supervised_pilot' in zh.SUPPORTED_MODULES
    assert plan['protocol']['theory_labels'] is False


def test_plan_cli_writes_only_stdout(tmp_path, monkeypatch, capsys):
    source, _ = source_fixture(tmp_path)
    output = tmp_path / 'pilot'
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No launch'))
    pipeline.main(['--source-run', str(source), '--output', str(output), '--plan'])
    assert json.loads(capsys.readouterr().out)['output'] == str(output)
    assert not output.exists()


def test_existing_output_and_historical_nested_paths_are_preserved(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    for output in (source, source / 'new', source.parent):
        with pytest.raises(ValueError, match='separate'):
            pipeline.build_plan(source, output)
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('user data')
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No launch'))
    with pytest.raises(FileExistsError):
        pipeline.main(['--source-run', str(source), '--output', str(output)])
    assert sentinel.read_text() == 'user data'


@pytest.mark.parametrize('change', ['supervised_source', 'incomplete', 'block', 'depth', 'milestones', 'missing_control'])
def test_wrong_controls_rejected_before_output(tmp_path, change):
    source, _ = source_fixture(tmp_path)
    manifest = json.loads((source / 'manifest.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if change == 'supervised_source': manifest['labels_used_in_training'] = True
    if change == 'incomplete': summary['actual_epochs'] = 2999
    if change == 'block': manifest['config']['model']['block_factor'] = 3
    if change == 'depth': manifest['config']['training']['max_rg_steps'] = 3
    if change == 'milestones': manifest['config']['training']['lr_milestones'] = [1500, 2250]
    if change == 'missing_control': (source / 'training_step_3000.pt').unlink()
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'summary.json').write_text(json.dumps(summary))
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError):
        pipeline.build_plan(source, output)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['wrong_key', 'missing_file'])
def test_missing_or_wrong_mi_cache_is_rejected_without_rebuilding(tmp_path, problem):
    source, _ = source_fixture(tmp_path)
    receipt_path = source.parent / 'mi_preprocessing.json'
    receipt = json.loads(receipt_path.read_text())
    if problem == 'wrong_key':
        receipt['splits']['train']['cache']['key'] = '0' * 64
        receipt_path.write_text(json.dumps(receipt))
    else:
        Path(receipt['splits']['train']['cache']['path']).unlink()
    with pytest.raises(ValueError, match='MI cache'):
        pipeline.build_plan(source, tmp_path / 'pilot')
    assert not (tmp_path / 'pilot').exists()


def test_remapped_cuda_is_rejected_before_creating_job(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No launch'))
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        pipeline.main(['--source-run', str(source), '--output', str(output)])
    assert not output.exists()


def test_busy_gpu_records_failed_job_and_exits_nonzero_without_launch(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **kw: '4321\n')
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No launch'))
    output = tmp_path / 'pilot'
    with pytest.raises(SystemExit) as error:
        pipeline.main(['--source-run', str(source), '--output', str(output)])
    assert error.value.code == 1
    job = json.loads((output / 'job.json').read_text())
    assert job['status'] == 'failed' and not job['workflow_complete']
    assert job['stages'] == [] and 'GPU0' in job['error']


@pytest.mark.parametrize('flip', [False, True])
def test_cpu_device_collection_matches_reference_rng_float64_and_closure(flip):
    torch.set_num_threads(1)
    torch.manual_seed(730)
    model = build_model(ModelConfig(dimension=2, block_factor=2, hidden=4), 2)
    generator = torch.Generator().manual_seed(34)
    configurations = (2 * torch.randint(2, (5, 4, 16, 16, 1), generator=generator) - 1).float()
    state = torch.get_rng_state().clone()
    actual = report.collect_on_device(model, configurations, samples=4, depth=2,
        device='cpu', flip=flip, seed=81, chunk_size=2)
    assert torch.equal(torch.get_rng_state(), state)
    expected = collect_trajectories(model, configurations, samples=4, max_rg_steps=2,
        spin_flip_augment=flip, chunk_size=2, sampling_seed=81)
    assert torch.equal(torch.get_rng_state(), state)
    for key in ('direct', 'rollout', 'one_step'):
        assert actual[key].dtype == np.float64
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-14, rtol=1e-13)
    assert actual['direct'].shape == (5, 3, 1)
    assert actual['one_step'].shape == (5, 2, 1)
    np.testing.assert_array_equal(actual['rollout'][:, 0], actual['direct'][:, 0])
    actual_metrics = closure_metrics(actual, reference_direct=expected['direct'])
    expected_metrics = closure_metrics(expected, reference_direct=expected['direct'])
    assert actual_metrics['normalization'] == expected_metrics['normalization']
    np.testing.assert_allclose(actual_metrics['target_sd'], expected_metrics['target_sd'],
                               atol=1e-14, rtol=1e-13)
    for kind in ('rollout', 'one_step'):
        for metric in ('rmse', 'nrmse'):
            np.testing.assert_allclose(actual_metrics[kind][metric], expected_metrics[kind][metric],
                                       atol=1e-14, rtol=1e-13)


def test_report_one_uses_saved_checkpoint_and_explicit_training_reference(tmp_path, monkeypatch):
    source, cfg = source_fixture(tmp_path)
    torch.manual_seed(71)
    model = build_model(ModelConfig(dimension=2, block_factor=2, hidden=4), 2)
    history = [dict(epoch=3000, **{split+'_'+name: .01 for split in ('train', 'test')
        for name in ('anchor', 'coarse_consistency', 'separation', 'commutator', 'rollout')})]
    loaded = SimpleNamespace(model=model, config=cfg, payload=dict(epoch=3000, history=history))
    calls = []
    def load(run, checkpoint):
        calls.append((run, checkpoint))
        return loaded
    monkeypatch.setattr(report, 'load_run', load)
    tr = dict(direct=np.array([[[.2], [.25], [.3]], [[.5], [.55], [.6]]]),
              rollout=np.array([[[.2], [.3], [.4]], [[.5], [.6], [.7]]]),
              one_step=np.array([[[.3], [.4]], [[.6], [.7]]]))
    te = {key: value + .02 for key, value in tr.items()}
    datasets = [SimpleNamespace(configurations=torch.ones(2, 4, 16, 16, 1, dtype=torch.float64),
        couplings=torch.tensor(labels, dtype=torch.float64)) for labels in ([.2, .5], [.3, .6])]
    trajectories = iter([tr, te])
    monkeypatch.setattr(report, 'collect_on_device', lambda *a, **kw: next(trajectories))
    fixed = dict(roots=[dict(g=[.4])], search_box=[[.1], [.8]])
    def plots(m, trajectory, labels, names, out, *, training_inputs):
        assert trajectory is te and names == ['K']
        np.testing.assert_array_equal(training_inputs, tr['direct'][:, :-1])
        (out / 'fixed_points.json').write_text(json.dumps(fixed))
    monkeypatch.setattr(report, 'plot_diagnostics', plots)
    def exponents(saved, metrics, saved_fixed, encodings, out):
        assert saved is loaded and saved_fixed == fixed
        assert metrics['closure'] == closure_metrics(te, reference_direct=tr['direct'])
        np.testing.assert_array_equal(encodings['train'], tr['direct'])
        return dict(checkpoint_epoch=3000, roots=deepcopy(fixed['roots']),
            coupling_means=dict(g_by_scale=[[.2, .25, .3], [.5, .55, .6]]))
    monkeypatch.setattr(report, 'write_exponent_report', exponents)
    captured_history = []
    monkeypatch.setattr(report, 'draw_losses', lambda h, out: captured_history.append(h))
    out = tmp_path / 'diagnostics'
    result = report.report_one(source, 3000, out, *datasets, 'cpu')
    assert calls == [(source, 3000)] and captured_history == [history]
    assert result['actual_epochs'] == 3000
    assert result['supervision_mse']['train'] == 0.
    assert result['supervision_mse']['test'] == pytest.approx(.0064)
    assert result['roots'][0]['nearest_training_input_distance'] == pytest.approx(.1)
    assert result['roots'][0]['distance_to_mean_encoding_curve_by_scale'] == [0., 0., 0.]
    assert (out / 'encodings.npz').is_file() and (out / 'metrics.json').is_file()
    with pytest.raises(FileExistsError):
        report.report_one(source, 3000, out, *datasets, 'cpu')


def test_pilot_loss_display_text_is_chinese(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    history = [dict(epoch=1, **{split+'_'+name: .01 for split in ('train', 'test')
        for name in ('anchor', 'coarse_consistency', 'separation', 'commutator', 'rollout',
                     'weighted_coupling_supervision')})]
    texts = []
    def save(figure, path, **kwargs):
        texts.extend(artist.get_text() for artist in figure.findobj(match=Text)
                     if artist.get_visible() and artist.get_text())
    monkeypatch.setattr(Figure, 'savefig', save)
    report.draw_losses(history, tmp_path)
    assert {'训练集', '测试集', '更新次数', '加权损失', '加权 K 监督', '加权 Q 重构',
            '加权子集一致性', '加权 R 闭合'} <= set(texts)
    assert all(zh.translate_text(text) == text for text in texts)


def test_synthetic_report_one_saves_real_chinese_figures_and_validated_roots(tmp_path, monkeypatch):
    """Exercise the actual plotters/save hook on a small saved-model diagnostic."""
    try:
        font = zh.select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    torch.set_num_threads(1)
    torch.manual_seed(81)
    spec = ModelConfig(dimension=2, block_factor=2, hidden=4)
    model = build_model(spec, 2)
    generator = torch.Generator().manual_seed(22)
    datasets = [SimpleNamespace(
        configurations=(2 * torch.randint(2, (3, 4, 16, 16, 1), generator=generator) - 1).double(),
        couplings=torch.tensor(labels, dtype=torch.float64)) for labels in (
            [.2, .5, .68], [.3, .4406867935, .6])]
    with torch.no_grad():
        center = model.theory_encoder(datasets[0].configurations).mean().item()
    with torch.no_grad():
        # Preserve the real checkpoint architecture and make a known root:
        # R(g)=g+.5*tanh(tanh(g-center)), so R'(center)=1.5.
        for parameter in model.theory_rg.parameters():
            parameter.zero_()
        model.theory_rg[0].weight[0, 0] = 1.
        model.theory_rg[0].bias[0] = -center
        model.theory_rg[2].weight[0, 0] = 1.
        model.theory_rg[4].weight[0, 0] = .5
    source, cfg = source_fixture(tmp_path)
    cfg['model'] = asdict(spec)
    cfg['training']['epochs'] = 3000
    cfg['evaluation']['samples'] = 4
    history = [dict(epoch=3000, **{split+'_'+name: .01 for split in ('train', 'test')
        for name in ('anchor', 'coarse_consistency', 'separation', 'commutator', 'rollout',
                     'weighted_coupling_supervision')})]
    torch.save(dict(model=qcr_state_dict(model), epoch=3000,
        history=history, metadata=dict(config=cfg)), source / 'training_step_3000.pt')
    # No data generation, MI building or optimization occurs in report_one.
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *a, **kw: pytest.fail('No training'))
    out = tmp_path / 'real_chinese_report'
    state = torch.get_rng_state().clone()
    with zh.chinese_figures(font):
        result = report.report_one(source, 3000, out, *datasets, 'cpu')
    assert torch.equal(torch.get_rng_state(), state)
    assert result['actual_epochs'] == 3000
    assert len(result['roots']) == 1
    root = result['roots'][0]
    assert root['multiplier'] == pytest.approx(1.5)
    assert root['derivative_verified'] and root['root_verified']
    assert root['inside_training_input_range'] and not root['nu_certified']
    exponents = json.loads((out / 'critical_exponents.json').read_text())
    assert exponents['block_factor'] == 2 and exponents['checkpoint_epoch'] == 3000
    assert len(exponents['coupling_means']['K']) == 6
    for name in ('R_vs_g.png', 'coupling_vs_g.png', 'trajectories.png',
                 'scalar_critical_summary.png', 'loss_comparison.png'):
        assert (out / name).stat().st_size > 1000
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('spawn_fails', [False, True])
def test_detach_reserves_before_one_spawn_and_never_retries_ambiguous_launch(
        tmp_path, monkeypatch, capsys, spawn_fails):
    source, _ = source_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for name in ('experiments/rsmi_separate_qcr.py', 'scripts/report_ising_zh.py',
                 'scripts/report_k_supervised_pilot.py', 'scripts/run_ising2d_k_supervised_pilot.py'):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'static code fingerprint fixture')
    output = tmp_path / 'detached_pilot'
    receipt = tmp_path / 'logs' / 'detached_pilot.launch.json'
    calls = []
    def spawn(command, **kwargs):
        calls.append(command)
        assert json.loads(receipt.read_text())['status'] == 'reserved'
        assert kwargs['start_new_session'] is True
        assert '--detach' not in command
        assert not output.exists()
        if spawn_fails:
            raise OSError('ambiguous launch fixture')
        return SimpleNamespace(pid=9876)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', spawn)
    args = ['--source-run', str(source), '--output', str(output), '--detach']
    if spawn_fails:
        with pytest.raises(OSError, match='ambiguous'):
            pipeline.main(args)
        assert json.loads(receipt.read_text())['status'] == 'reserved'
    else:
        pipeline.main(args)
        record = json.loads(capsys.readouterr().out)
        assert record == json.loads(receipt.read_text())
        assert record['status'] == 'launched' and record['parent_pid'] == 9876
        assert len(record['code_sha256']) == 4
    with pytest.raises(FileExistsError):
        pipeline.main(args)
    assert len(calls) == 1 and not output.exists()
