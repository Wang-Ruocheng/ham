"""No-GPU/no-SSH contracts for the explicitly requested reverse-weight arm."""
import ast
import inspect
from types import SimpleNamespace

import pytest

from scripts import run_ising_closure_scale_early as runner


INITIAL = {'Q': 'q', 'C': 'c', 'R': 'r'}


def fixture(tmp_path, monkeypatch):
    prior, source, code = (tmp_path / name for name in ('prior', 'source', 'code'))
    for folder in (prior / 'data', prior / 'benchmark', code / 'scripts', source / 'run_control/report',
                   prior / 'reference_control/report', prior / 'run_late/report', prior / 'report'):
        folder.mkdir(parents=True)
    (code / 'scripts/science.py').write_text('# unchanged science')
    (prior / 'data/ensembles.pt').write_bytes(b'new data, never tensor loaded')
    runner.save(prior / 'pairs.json', {'new': 'pairs'})
    inputs = {name: runner.file_sha(prior / name) for name in ('data/ensembles.pt', 'pairs.json')}
    receipt = dict(input_sha256=inputs, source=str(source))
    config = runner.pair.make_config(prior / 'data/ensembles.pt')
    config['loss']['closure_scale_weights'] = [.2, 1.8]
    for name in ('config_late.json', 'config.json'):
        runner.save(prior / name, config)
    scale = dict(weights=[.2, 1.8], normalized_weights=[.1, .9], target_levels=[1, 2],
        application=['one_step_commutator', 'microscopic_start_rollout'], labels_used=False)
    manifest = dict(config=config, initial_module_sha256=INITIAL, closure_scale_weighting=scale,
        labels_used_in_training=False, auxiliary_future_parameters=0,
        q_future_physics=dict(mode='paired_control', weight=0., labels_used=False),
        execution_optimizations=dict(mi_statistics_cache={split: {'hit': True} for split in ('train', 'test')}))
    summary = dict(config=config, actual_epochs=6000, requested_epochs=6000,
        contract=dict(closure_scale_weighting=scale, initial_module_sha256=INITIAL))
    benchmark = dict(history=[{}, {}, {}], timing=dict(steady_step_seconds=[.1] * 35,
        setup_seconds=80., logging_seconds=6., checkpoint_seconds=2.), actual_epochs=40)
    runner.save(prior / 'benchmark/summary.json', benchmark)
    runner.save(prior / 'benchmark/manifest.json', {'initial_module_sha256': INITIAL})
    for base in (source / 'run_control', prior / 'reference_control', prior / 'run_late'):
        runner.save(base / 'manifest.json', manifest)
        runner.save(base / 'summary.json', summary)
        (base / 'report/encodings.npz').write_bytes(b'public NPZ is not tensor data')
        runner.save(base / 'report/metrics.json', dict(status='complete', actual_steps=6000,
            scales=[108, 36, 12], C_draws=5, samples=64,
            encodings_sha256=runner.file_sha(base / 'report/encodings.npz'),
            model_sha256_before='same', model_sha256_after='same'))
    runner.save(prior / 'report/status.json', dict(status='complete', workflow_complete=True))
    runner.save(prior / 'plan.json', dict(source_code_sha256=runner.execution.source_hashes(code),
        configuration_sha256={name: runner.file_sha(prior / name) for name in ('config_late.json', 'config.json')},
        objective={'future_loss_weight': 0}, budget={'GPU_card_seconds': 3600}))
    runner.save(prior / 'job.json', dict(experiment=runner.PRIOR_EXPERIMENT, status='complete',
        workflow_complete=True, current=None, child_still_running=False, gpu_seconds_current_stage=0.,
        error=None, requires_user_action=False, source=str(source), device='cuda:2', gpu_uuid=runner.pair.GPU_UUID,
        stages=[dict(name=name, status='complete', returncode=0, pid=123) for name in
                ('train_late', 'collect_late', 'report')],
        parent_pid=456, gpu_seconds_completed=979.3171557988971, CPU_seconds_completed=1200.,
        campaign_peak_own_gpu_mib=6671, input_reuse=receipt))
    monkeypatch.setattr(runner.pair, 'validated_source', lambda path: receipt)
    def validated(path, name, steps):
        assert path == prior and (name, steps) in (('late', 6000), ('benchmark', 40))
        return summary if name == 'late' else benchmark
    monkeypatch.setattr(runner.pair, 'validate_arm', validated)
    monkeypatch.setattr(runner, 'prior_processes', lambda path: [])
    return prior, source, code


def test_prior_campaign_usage_carried_once_no_double_benchmark_charge(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    receipt, estimate = runner.validated_prior(prior, source, code)
    assert receipt['gpu_seconds'] == 979.3171557988971
    assert receipt['CPU_seconds'] == 1200.
    assert receipt['reference_previous_gpu_seconds'] == 979.3171557988971
    assert receipt['reference_previous_CPU_seconds'] == 1200.
    assert receipt['benchmark_updates_in_this_run'] == 0
    assert receipt['reused_model_or_optimizer'] is receipt['source_checkpoints_opened'] is False
    projection = runner.execution.remaining_projection(dict(gpu_seconds_completed=receipt['gpu_seconds'],
        CPU_seconds_completed=receipt['CPU_seconds']),
        estimate, arms=1, collections=1)
    assert projection['actual_gpu_seconds'] == receipt['gpu_seconds']
    assert projection['allowed_gpu_seconds'] == 3600
    assert projection['projected_total_gpu_seconds'] < 3600


@pytest.mark.parametrize('mutation', ['incomplete', 'nonzero', 'current', 'error', 'live', 'science',
    'input', 'config', 'reference', 'collection', 'missing_report'])
def test_rejects_changed_or_unfinished_prior(tmp_path, monkeypatch, mutation):
    prior, source, code = fixture(tmp_path, monkeypatch)
    job = runner.read(prior / 'job.json')
    if mutation == 'incomplete': job['workflow_complete'] = False
    elif mutation == 'nonzero': job['stages'][0]['returncode'] = 1
    elif mutation == 'current': job['current'] = {'name': 'train_late'}
    elif mutation == 'error': job['error'] = 'unexpected'
    elif mutation == 'live': monkeypatch.setattr(runner, 'prior_processes', lambda path: [{'pid': 123}])
    elif mutation == 'science': (code / 'scripts/science.py').write_text('# changed')
    elif mutation == 'input': (prior / 'data/ensembles.pt').write_bytes(b'changed')
    elif mutation == 'config': runner.save(prior / 'config_late.json', {'changed': True})
    elif mutation == 'reference': (prior / 'reference_control/summary.json').write_text('{}')
    elif mutation == 'collection': (prior / 'run_late/report/encodings.npz').write_bytes(b'changed')
    elif mutation == 'missing_report': runner.save(prior / 'report/status.json', {'status': 'running'})
    runner.save(prior / 'job.json', job)
    with pytest.raises(ValueError): runner.validated_prior(prior, source, code)


def test_only_exact_two_new_files_are_exempt_from_science_pin(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    for name in runner.NEW_FILES:
        (code / name).write_text('# early orchestration/report')
    runner.validated_prior(prior, source, code)
    (code / 'scripts/another_new_science.py').write_text('# disallowed')
    with pytest.raises(ValueError, match='science SHA'):
        runner.validated_prior(prior, source, code)


def test_prepare_changes_only_weights_and_cache_and_hardlinks_public_references(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    receipt, _ = runner.validated_prior(prior, source, code)
    root = tmp_path / 'new'; root.mkdir()
    monkeypatch.setattr(runner, 'fresh_initial_sha', lambda config: INITIAL)
    runner.prepare(root, prior, receipt)
    expected = runner.read(prior / 'config_late.json')
    expected['loss']['closure_scale_weights'] = [1.8, .2]
    expected['data']['cache'] = str(root / 'data/ensembles.pt')
    assert runner.read(root / 'config_early.json') == runner.read(root / 'config.json') == expected
    assert (root / 'data/ensembles.pt').stat().st_ino == (prior / 'data/ensembles.pt').stat().st_ino
    for destination, source_folder in [('reference_control', 'reference_control'), ('reference_late', 'run_late')]:
        for name in runner.REFERENCE_FILES:
            assert (root / destination / name).stat().st_ino == (prior / source_folder / name).stat().st_ino
    assert not list(root.rglob('checkpoint*'))
    assert sorted(path.name for path in (root / 'benchmark').iterdir()) == ['manifest.json', 'summary.json']


def test_fresh_initial_mismatch_stops_before_formal_training(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    receipt, _ = runner.validated_prior(prior, source, code)
    root = tmp_path / 'new'; root.mkdir()
    monkeypatch.setattr(runner, 'fresh_initial_sha', lambda config: {**INITIAL, 'Q': 'wrong'})
    with pytest.raises(ValueError, match='initialization differs'):
        runner.prepare(root, prior, receipt)
    assert not (root / 'run_early').exists()


def test_real_cpu_initialization_matches_completed_late_initial_fingerprints():
    config = runner.pair.make_config('/tmp/never-open-physical-data.pt')
    config['loss']['closure_scale_weights'] = [1.8, .2]
    assert runner.fresh_initial_sha(config) == dict(
        Q='27e390d450b93eb2120ac8ca4ffc3f588c100d2283a37a55c93a3912466d461c',
        C='53536aac70302d389018e421e0cbe9276399dce2be70918e645b25f50a883dc8',
        R='d34d3d15cb5b4ebb7ce874091308d2b4f7afd2f5349fdaa320bab8d2f4499009')


def test_early_collector_science_loop_is_ast_identical_to_previous_collector():
    def science(function):
        tree = ast.parse(inspect.getsource(function))
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == 'isolated_rng' for item in node.items):
                return ast.dump(node, include_attributes=False)
        raise AssertionError('Missing isolated five-draw collection')
    assert science(runner.collect) == science(runner.pair.collect)


def test_stage_keeps_reference_pins_and_separates_new_usage(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    receipt, _ = runner.validated_prior(prior, source, code)
    root = tmp_path / 'new'; root.mkdir()
    monkeypatch.setattr(runner, 'fresh_initial_sha', lambda config: INITIAL)
    runner.prepare(root, prior, receipt)
    job = dict(prior_execution=receipt, gpu_seconds_completed=receipt['gpu_seconds'],
               CPU_seconds_completed=receipt['CPU_seconds'])
    called = []
    def stage(*args, **kwargs):
        called.append((args[2], kwargs))
        args[1]['gpu_seconds_completed'] += 10.
        args[1]['CPU_seconds_completed'] += 3.
    monkeypatch.setattr(runner.pair, 'run_stage', stage)
    runner.run_stage(root, job, 'train_early', ['trainer'], gpu=True)
    assert called == [('train_early', {'gpu': True})]
    assert job['gpu_seconds_new_execution'] == pytest.approx(10.)
    assert job['CPU_seconds_new_execution'] == pytest.approx(3.)
    (root / 'reference_late/report/metrics.json').write_text('{}')
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        runner.run_stage(root, job, 'collect_early', ['collector'], gpu=True)
    assert len(called) == 1


def test_prior_process_matching_uses_actual_command_and_exact_output_not_old_pid(tmp_path, monkeypatch):
    prior = tmp_path / 'prior'
    lines = [f'456 1 python -m unrelated --output {prior}',
             f'123 1 python -m experiments.rsmi_separate_qcr --output {prior}x/run_late',
             f'999 1 python -m scripts.run_ising_closure_scale_early --prior-execution {prior}',
             f'790 1 python -m experiments.rsmi_separate_qcr --output {prior}/run_late',
             f'791 1 python -m scripts.run_ising_closure_scale_late --output {prior}']
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='\n'.join(lines)))
    assert [row['pid'] for row in runner.prior_processes(prior)] == [790, 791]


def test_validate_early_enforces_fresh_initial_and_no_future(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    root = tmp_path / 'new'; (root / 'run_early').mkdir(parents=True)
    config = runner.read(prior / 'config_late.json'); config['loss']['closure_scale_weights'] = [1.8, .2]
    config['data']['cache'] = str(root / 'data/ensembles.pt')
    runner.save(root / 'config_early.json', config)
    manifest = runner.read(prior / 'run_late/manifest.json')
    manifest['config'] = config
    manifest['closure_scale_weighting'].update(weights=[1.8, .2], normalized_weights=[.9, .1])
    summary = runner.read(prior / 'run_late/summary.json'); summary['config'] = config
    summary['contract']['closure_scale_weighting'] = manifest['closure_scale_weighting']
    runner.save(root / 'run_early/manifest.json', manifest)
    runner.save(root / 'run_early/summary.json', summary)
    runner.save(root / 'preflight_reuse.json', {'fresh_initial_module_sha256': INITIAL})
    runner.validate_early(root)
    manifest['q_future_physics']['weight'] = 1
    runner.save(root / 'run_early/manifest.json', manifest)
    with pytest.raises(RuntimeError, match='no-future'):
        runner.validate_early(root)


def test_public_whitelist_excludes_pt_logs_and_reference_checkpoints(tmp_path):
    for name in ('run_early/report', 'reference_late/report', 'data', 'report'):
        (tmp_path / name).mkdir(parents=True)
    for name in ('job.json', 'config_early.json', 'run_early/report/encodings.npz',
                 'reference_late/report/metrics.json', 'report/R_vs_g.png',
                 'run_early/checkpoint_last.pt', 'data/ensembles.pt', 'report/forbidden.pt', 'report/error.log'):
        (tmp_path / name).write_bytes(b'public or forbidden fixture')
    assert runner.public_files(tmp_path) == ['config_early.json', 'job.json', 'reference_late/report/metrics.json',
        'report/R_vs_g.png', 'run_early/report/encodings.npz']


def test_fresh_command_no_future_no_resume_no_old_arm_execution(tmp_path):
    command = runner.pair.trainer_command(tmp_path, tmp_path / 'run_early', tmp_path / 'config_early.json')
    assert '--q-future-control' in command and '--q-future-weight' not in command
    assert '--resume-checkpoint' not in command and 'cuda:2' in command
    assert 'run_late' not in str(command) and 'run_uniform' not in str(command)


def test_existing_output_is_never_restarted(tmp_path):
    root = tmp_path / 'old'; root.mkdir(); runner.save(root / 'job.json', {'status': 'failed'})
    with pytest.raises(FileExistsError, match='Fresh output'):
        runner.run(root, tmp_path / 'source', tmp_path / 'prior')


def test_mock_workflow_executes_only_new_early6000_collect_report(tmp_path, monkeypatch):
    prior, source, code = fixture(tmp_path, monkeypatch)
    root = tmp_path / 'new'
    monkeypatch.chdir(code)
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'thinklab-105-225')
    monkeypatch.setattr(runner.os, 'setpriority', lambda *args: None)
    monkeypatch.setattr(runner, 'fresh_initial_sha', lambda config: INITIAL)
    commands = []
    def stage(output, job, name, command, *, gpu=False):
        commands.append((name, command, gpu))
        job['stages'].append(dict(name=name, status='complete', returncode=0))
        if gpu:
            job['gpu_seconds_completed'] += 10.
        if name == 'train_early':
            target = output / 'run_early'; target.mkdir()
            config = runner.read(output / 'config_early.json')
            manifest = runner.read(prior / 'run_late/manifest.json')
            manifest['config'] = config
            manifest['closure_scale_weighting'].update(weights=[1.8, .2], normalized_weights=[.9, .1])
            summary = runner.read(prior / 'run_late/summary.json'); summary['config'] = config
            summary['contract']['closure_scale_weighting'] = manifest['closure_scale_weighting']
            runner.save(target / 'manifest.json', manifest); runner.save(target / 'summary.json', summary)
        elif name == 'report':
            (output / 'report').mkdir()
            runner.save(output / 'report/status.json', dict(status='complete', workflow_complete=True,
                scientific_result=True, qualified_reference_comparison_available=True))
    monkeypatch.setattr(runner.pair, 'run_stage', stage)
    runner.run(root, source, prior)
    job = runner.read(root / 'job.json')
    assert job['status'] == 'complete' and job['workflow_complete'] is True
    assert [row[0] for row in commands] == list(runner.STAGES)
    assert [row[2] for row in commands] == [True, True, False]
    assert job['gpu_seconds_prior_execution'] == 979.3171557988971
    assert job['gpu_seconds_new_execution'] == pytest.approx(20.)
    assert job['gpu_seconds_completed'] == pytest.approx(999.3171557988971)
    assert job['peak_own_gpu_mib'] == 0 and job['campaign_peak_own_gpu_mib'] == 6671
    assert '--q-future-control' in commands[0][1]
    assert 'scripts.run_ising_closure_scale_early' in commands[1][1]
    assert 'scripts.report_ising_closure_scale_early' in commands[2][1]
    assert not any('benchmark' == row.get('name') for row in job['stages'])


@pytest.mark.parametrize('arguments', [
    ['collect', '--output', '/tmp/test'],
    ['collect', '--output', '/tmp/test', '--run', '/tmp/test/run_early', '--launch'],
    ['--output', '/tmp/test', '--source', '/tmp/source'],
    ['--output', '/tmp/test', '--source', '/tmp/source', '--prior-execution', '/tmp/prior', '--run', '/tmp/run'],
])
def test_invalid_cli_never_launches(monkeypatch, arguments):
    monkeypatch.setattr(runner.sys, 'argv', ['runner'] + arguments)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
