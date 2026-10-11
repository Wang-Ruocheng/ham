import json

import pytest

from recurrent_rg.config import RunConfig
from scripts.benchmark_qcr_device import (
    benchmark_config, build_parser, compare_benchmarks, compare_execution_benchmarks, config_fingerprint,
    main, summarize_timing, validate_cache,
)


def timing(device='cpu'):
    return {
        'device': device, 'updates': 4, 'warmup_updates': 1,
        'measured_updates': 3, 'includes_eval_in_step': False,
        'step_seconds_all': [5., 1., 2., 3.], 'steady_step_seconds': [1., 2., 3.],
        'setup_seconds': 10., 'training_wall_seconds': 14.,
        'report_seconds': 7.,
    }


def result(device='cpu', scale=1., *, reference_execution=False):
    item = {
        'device': device, 'cache_sha256': 'cache', 'config_sha256': 'config',
        'dtype': 'torch.float64', 'actual_epochs': 4, 'warmup_updates': 1,
        'reference_execution': reference_execution,
        'execution_optimizations': {
            'enabled': not reference_execution, 'reference': reference_execution,
            'mi_statistics_cache': {'hit': not reference_execution},
        },
        'timing': summarize_timing(timing(device), updates=4, warmup=1, device=device),
    }
    item['timing']['steady_update_mean_seconds'] *= scale
    item['timing']['steady_update_median_seconds'] *= scale
    item['timing']['driver']['training_wall_seconds'] *= scale
    item['timing']['driver']['step_seconds_all'] = [
        value * scale for value in item['timing']['driver']['step_seconds_all']]
    item['timing']['driver']['steady_step_seconds'] = [
        value * scale for value in item['timing']['driver']['steady_step_seconds']]
    return item


def test_benchmark_preserves_locked_recipe(tmp_path):
    values = benchmark_config(tmp_path / 'cache.pt')
    cfg = RunConfig.from_dict(values)
    cfg.validate()
    assert cfg.schema_version == 5
    assert cfg.model.theory_dim == 2 and cfg.model.hidden == 96
    assert cfg.model.block_factor == 3 and cfg.model.kernel_initialization == 'random'
    assert cfg.model.kernel_sampling == 'bernoulli' and cfg.model.affine_reconstruction
    assert cfg.loss.weights.separation == 10 and cfg.loss.weights.kernel_noncollapse == 2
    assert cfg.loss.kernel_sign_weight == 1 and cfg.loss.kernel_variance_ratio == 1
    assert cfg.loss.weights.commutator == 5 and cfg.loss.rollout_weight == 5
    assert cfg.loss.geometry_tail_mix == 0 and cfg.loss.balanced_reconstruction_weight == 0
    assert cfg.training.gradient_mode == 'qc_r_separate'
    assert tuple(cfg.training.train_modules) == ('Q', 'C', 'R')
    assert cfg.training.system_batch == 8 and cfg.training.sample_batch == 64
    assert cfg.training.epochs == 60


def test_config_fingerprint_ignores_only_cache_path(tmp_path):
    first = benchmark_config(tmp_path / 'cache.pt')
    second = benchmark_config(tmp_path / 'other' / 'cache.pt')
    assert config_fingerprint(first) == config_fingerprint(second)
    second['model']['hidden'] = 97
    assert config_fingerprint(first) != config_fingerprint(second)


def test_timing_uses_real_updates_and_excludes_warmup():
    summary = summarize_timing(timing(), updates=4, warmup=1, device='cpu')
    assert summary['steady_update_median_seconds'] == 2
    assert summary['steady_update_mean_seconds'] == 2
    assert summary['steady_update_p90_seconds'] == 3
    assert summary['measured_updates'] == 3


@pytest.mark.parametrize('mutation', [
    {'device': 'cuda:0'}, {'updates': 5}, {'warmup_updates': 2},
    {'includes_eval_in_step': True}, {'measured_updates': 4},
    {'steady_step_seconds': [0., 2., 3.]},
    {'steady_step_seconds': [1., 3., 2.]},
    {'step_seconds_all': [float('nan'), 1., 2., 3.]},
])
def test_rejects_unfair_or_invalid_driver_timing(mutation):
    values = timing()
    values.update(mutation)
    with pytest.raises(ValueError):
        summarize_timing(values, updates=4, warmup=1, device='cpu')


def test_comparison_cpu_seconds_over_gpu_seconds():
    comparison = compare_benchmarks(result('cpu', 4.), result('cuda:0'))
    assert comparison['steady_update_median_speedup'] == 4
    assert comparison['steady_update_mean_speedup'] == 4
    assert comparison['training_wall_speedup'] == 4


@pytest.mark.parametrize('key,value', [
    ('cache_sha256', 'different'), ('config_sha256', 'different'),
    ('dtype', 'torch.float32'), ('actual_epochs', 5), ('warmup_updates', 2),
])
def test_comparison_refuses_changed_workload(key, value):
    other = result('cuda:0')
    other[key] = value
    with pytest.raises(ValueError, match=key):
        compare_benchmarks(result(), other)


def test_parser_separates_prepare_and_reuse():
    args = build_parser().parse_args(['--cache', 'shared.pt', '--output', 'new', '--device', 'cuda:0'])
    assert args.epochs == 60 and args.warmup == 10 and args.threads == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--prepare', '--cache', 'shared.pt', '--output', 'new'])


def test_refuses_overwrite_and_invalid_warmup(tmp_path):
    with pytest.raises(SystemExit):
        main(['--prepare', '--output', str(tmp_path)])
    with pytest.raises(SystemExit):
        main(['--prepare', '--output', str(tmp_path / 'new'), '--epochs', '10', '--warmup', '10'])
    assert not (tmp_path / 'new').exists()


def test_shared_cache_shape_and_complete_coupling_split(tmp_path):
    import torch

    tx = torch.ones((16, 64, 36, 36, 1), dtype=torch.int8)
    vx = -torch.ones((8, 64, 36, 36, 1), dtype=torch.int8)
    ty = torch.arange(16, dtype=torch.float64)
    vy = torch.arange(8, dtype=torch.float64) + 20
    path = tmp_path / 'cache.pt'
    torch.save((tx, ty, vx, vy), path)
    validated = validate_cache(path)
    assert validated['train_shape'] == list(tx.shape)
    assert validated['labels_used_in_training'] is False
    assert len(validated['cache_sha256']) == 64
    torch.save((tx, ty, vx, torch.arange(8, dtype=torch.float64)), path)
    with pytest.raises(ValueError, match='disjoint complete K points'):
        validate_cache(path)


@pytest.mark.parametrize('reverse', [False, True])
def test_execution_comparison_reference_over_optimized_without_mixing_setup(reverse):
    reference = result('cuda:0', 3., reference_execution=True)
    optimized = result('cuda:0')
    reference['timing']['driver']['setup_seconds'] = 100.
    optimized['timing']['driver']['setup_seconds'] = 2.
    reference['timing']['driver']['report_seconds'] = 20.
    optimized['timing']['driver']['report_seconds'] = 10.
    args = (optimized, reference) if reverse else (reference, optimized)
    comparison = compare_execution_benchmarks(*args)
    assert comparison['steady_update_median_speedup'] == 3
    assert comparison['steady_update_mean_speedup'] == 3
    assert comparison['update_only_wall_speedup'] == 3
    assert comparison['training_wall_speedup'] == 3
    assert comparison['setup_wall_ratio'] == 50
    assert comparison['report_wall_ratio'] == 2
    assert comparison['device'] == 'cuda:0'
    assert 'warm MI cache is startup savings' in comparison['limitations']
    assert comparison['reference_execution_optimizations']['reference'] is True
    assert comparison['optimized_execution_optimizations']['enabled'] is True


@pytest.mark.parametrize('key,value', [
    ('cache_sha256', 'different'), ('config_sha256', 'different'),
    ('dtype', 'torch.float32'), ('actual_epochs', 5), ('warmup_updates', 2),
])
def test_execution_comparison_refuses_changed_workload(key, value):
    optimized = result('cuda:0')
    optimized[key] = value
    with pytest.raises(ValueError, match=key):
        compare_execution_benchmarks(result('cuda:0', reference_execution=True), optimized)


@pytest.mark.parametrize('other_device', ['cpu', 'cuda:1'])
def test_execution_comparison_refuses_different_devices(other_device):
    with pytest.raises(ValueError, match='same device'):
        compare_execution_benchmarks(result('cuda:0', reference_execution=True), result(other_device))


@pytest.mark.parametrize('reference', [False, True])
def test_execution_comparison_needs_two_distinct_modes(reference):
    with pytest.raises(ValueError, match='one reference and one optimized'):
        compare_execution_benchmarks(result(reference_execution=reference),
                                     result(reference_execution=reference))


@pytest.mark.parametrize('mutation', [
    {'reference_execution': None}, {'reference_execution': 1},
    {'execution_optimizations': None},
    {'execution_optimizations': {'enabled': True, 'reference': True}},
    {'execution_optimizations': {'enabled': False, 'reference': False}},
])
def test_execution_comparison_refuses_missing_or_conflicting_mode_records(mutation):
    optimized = result()
    optimized.update(mutation)
    with pytest.raises(ValueError, match='execution'):
        compare_execution_benchmarks(result(reference_execution=True), optimized)


def test_execution_comparison_does_not_guess_historical_record_mode():
    reference = result(reference_execution=True)
    del reference['reference_execution']
    with pytest.raises(ValueError, match='explicit reference_execution'):
        compare_execution_benchmarks(reference, result())


@pytest.mark.parametrize('key', ['setup_seconds', 'report_seconds', 'training_wall_seconds'])
@pytest.mark.parametrize('value', [None, -1., float('nan')])
def test_execution_comparison_refuses_invalid_separate_durations(key, value):
    optimized = result()
    optimized['timing']['driver'][key] = value
    with pytest.raises(ValueError, match=key):
        compare_execution_benchmarks(result(reference_execution=True), optimized)


def test_execution_comparison_zero_duration_ratio_is_unavailable():
    optimized = result()
    optimized['timing']['driver']['setup_seconds'] = 0.
    optimized['timing']['driver']['report_seconds'] = 0.
    comparison = compare_execution_benchmarks(result(reference_execution=True), optimized)
    assert comparison['setup_wall_ratio'] is None
    assert comparison['report_wall_ratio'] is None


def test_parser_exposes_reference_execution_and_exclusive_comparisons():
    args = build_parser().parse_args([
        '--cache', 'shared.pt', '--output', 'new', '--device', 'cuda:0',
        '--reference-execution', '--compare-reference', 'optimized.json'])
    assert args.reference_execution and args.compare_reference.name == 'optimized.json'
    assert args.compare is None
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            '--cache', 'shared.pt', '--output', 'new',
            '--compare', 'cpu.json', '--compare-reference', 'reference.json'])


@pytest.mark.parametrize('extra', [
    ['--reference-execution'], ['--compare-reference', 'reference.json'],
    ['--compare', 'cpu.json'],
])
def test_prepare_refuses_training_only_execution_options(tmp_path, extra):
    output = tmp_path / 'new'
    with pytest.raises(SystemExit):
        main(['--prepare', '--output', str(output), *extra])
    assert not output.exists()


@pytest.mark.parametrize('reference_execution', [False, True])
def test_benchmark_forwards_reference_flag_and_records_actual_manifest(
        tmp_path, monkeypatch, reference_execution):
    import scripts.benchmark_qcr_device as benchmark_module

    cache = tmp_path / 'shared.pt'
    cache.write_bytes(b'unit-test placeholder, never loaded')
    output = tmp_path / 'new'
    optimizations = {
        'enabled': not reference_execution, 'reference': reference_execution,
        'mi_statistics_cache': {'hit': not reference_execution},
        'shared_q_table': not reference_execution,
    }
    commands = []

    def fake_command(command, log_path):
        commands.append(command)
        return {'command': command, 'wall_seconds': 1., 'log': str(log_path)}

    monkeypatch.setattr(benchmark_module, 'environment_record', lambda: {'git_sha': 'commit'})
    monkeypatch.setattr(benchmark_module, 'validate_cache',
                        lambda path: {'cache_sha256': 'cache'})
    monkeypatch.setattr(benchmark_module, 'timed_command', fake_command)
    monkeypatch.setattr(benchmark_module, 'verify_training', lambda run, device, epochs: (
        {'actual_epochs': 4, 'timing': timing()},
        {'dtype': 'torch.float64', 'execution_optimizations': optimizations}))
    argv = ['--cache', str(cache), '--output', str(output), '--epochs', '4', '--warmup', '1']
    if reference_execution:
        argv.append('--reference-execution')
    assert main(argv) == 0
    assert ('--reference-execution' in commands[0]) is reference_execution
    record = json.loads((output / 'benchmark.json').read_text())
    assert record['reference_execution'] is reference_execution
    assert record['execution_optimizations'] == optimizations
    assert record['verification']['execution_optimizations'] == optimizations


def test_benchmark_runs_same_device_comparison_without_training(tmp_path, monkeypatch):
    import scripts.benchmark_qcr_device as benchmark_module

    cache = tmp_path / 'shared.pt'
    cache.write_bytes(b'unit-test placeholder, never loaded')
    output = tmp_path / 'optimized'
    reference = result(reference_execution=True, scale=2.)
    reference['output'] = str(tmp_path / 'reference')
    reference['config_sha256'] = config_fingerprint(benchmark_config(cache, epochs=4))
    reference_file = tmp_path / 'reference.json'
    reference_file.write_text(json.dumps(reference))
    optimizations = {'enabled': True, 'reference': False, 'mi_statistics_cache': {'hit': True}}
    monkeypatch.setattr(benchmark_module, 'environment_record', lambda: {'git_sha': 'commit'})
    monkeypatch.setattr(benchmark_module, 'validate_cache',
                        lambda path: {'cache_sha256': 'cache'})
    monkeypatch.setattr(benchmark_module, 'timed_command', lambda command, log_path: {
        'command': command, 'wall_seconds': 1., 'log': str(log_path)})
    monkeypatch.setattr(benchmark_module, 'verify_training', lambda run, device, epochs: (
        {'actual_epochs': 4, 'timing': timing()},
        {'dtype': 'torch.float64', 'execution_optimizations': optimizations}))
    assert main(['--cache', str(cache), '--output', str(output), '--epochs', '4', '--warmup', '1',
                 '--compare-reference', str(reference_file)]) == 0
    record = json.loads((output / 'benchmark.json').read_text())
    assert record['execution_comparison']['steady_update_median_speedup'] == 2.
    assert record['execution_comparison']['reference_source'] == reference['output']
    assert 'comparison' not in record
