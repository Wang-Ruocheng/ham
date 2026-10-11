import pytest

from recurrent_rg.config import RunConfig
from scripts.benchmark_qcr_device import (
    benchmark_config, build_parser, compare_benchmarks, config_fingerprint,
    main, summarize_timing, validate_cache,
)


def timing(device='cpu'):
    return {
        'device': device, 'updates': 4, 'warmup_updates': 1,
        'measured_updates': 3, 'includes_eval_in_step': False,
        'step_seconds_all': [5., 1., 2., 3.], 'steady_step_seconds': [1., 2., 3.],
        'setup_seconds': 10., 'training_wall_seconds': 14.,
    }


def result(device='cpu', scale=1.):
    item = {
        'device': device, 'cache_sha256': 'cache', 'config_sha256': 'config',
        'dtype': 'torch.float64', 'actual_epochs': 4, 'warmup_updates': 1,
        'timing': summarize_timing(timing(device), updates=4, warmup=1, device=device),
    }
    item['timing']['steady_update_mean_seconds'] *= scale
    item['timing']['steady_update_median_seconds'] *= scale
    item['timing']['driver']['training_wall_seconds'] *= scale
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
