"""Retained uniform coupling grid is distinct from the auxiliary PT ladder."""

import numpy as np
import pytest

from scripts import run_ising_j1j2 as runner


def test_retained_conditions_are_twelve_uniform_k_values_in_original_range():
    grid = runner.coupling_grid()
    temperatures = np.asarray(grid['temperatures'])
    ids = np.asarray(grid['saved_temperature_indices'])
    k = np.asarray(grid['retained_K_values'])
    assert grid['training_coupling_grid'] == 'uniform_K'
    assert len(ids) == len(set(ids)) == 12
    np.testing.assert_array_equal(k, np.linspace(.25, 2.5, 12)[::-1])
    np.testing.assert_allclose(np.diff(k), -2.25 / 11, rtol=0, atol=5e-16)
    np.testing.assert_array_equal(temperatures[ids], grid['retained_temperature_values'])
    np.testing.assert_allclose(1 / temperatures[ids], k, rtol=2e-16, atol=0)
    assert (temperatures[0], temperatures[-1]) == (.4, 4.)
    assert (k[-1], k[0]) == (.25, 2.5)


def test_previous_auxiliary_ladder_is_preserved_but_only_targets_are_retained():
    grid = runner.coupling_grid()
    temperatures = np.asarray(grid['temperatures'])
    saved = set(grid['saved_temperature_indices'])
    auxiliary = set(grid['auxiliary_temperature_indices'])
    assert len(temperatures) == 83
    assert len(auxiliary) == 71
    assert saved.isdisjoint(auxiliary)
    assert saved | auxiliary == set(range(len(temperatures)))
    assert np.all(np.diff(temperatures) > 0)
    assert np.all(np.diff(grid['saved_temperature_indices']) > 0)
    assert set(np.geomspace(.4, 4., 73)).issubset(set(temperatures))
    # Saved slots are not evenly spaced integer indices into the PT ladder.
    assert not np.array_equal(grid['saved_temperature_indices'],
                              np.linspace(0, len(temperatures) - 1, 12, dtype=int))


def test_grid_is_deterministic_and_does_not_share_mutable_results():
    first, second = runner.coupling_grid(), runner.coupling_grid()
    assert first == second
    first['temperatures'][0] = 0
    assert runner.coupling_grid() == second


def test_prepare_pins_grid_without_changing_sampler_quality_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: runner.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(runner, 'source_hashes', lambda root: {'frozen.py': 'a' * 64})
    root = tmp_path / 'uniform_k'
    runner.prepare(root)
    plan = runner.read(root / 'plan.json')
    physical = plan['physical']
    for key, value in runner.coupling_grid().items():
        assert physical[key] == value
    assert physical['rhos'] == [0., .2, .35, .7, 1., .3, .8]
    assert physical['train_rhos'] == [0., .2, .35, .7, 1.]
    assert physical['heldout_rhos'] == [.3, .8]
    assert (physical['size'], physical['groups'], physical['train_groups'],
            physical['samples'], physical['burn_in'], physical['spacing'],
            physical['master_seed']) == (108, 12, 8, 128, 16384, 128, 20261026)
    assert physical['initials'] == ['hot', 'ferro', 'stripe_x', 'stripe_y']
    assert plan['pilot'] == dict(rhos=[0., .35, .7, 1.], burn_in=4096, spacing=32,
        samples=128, min_exchange=.01, min_ess=16, hot_cold_abs_threshold=.05,
        hot_cold_z_threshold=4., production_reuse=False,
        constant_series='warning; not evidence of mixing, not alone proof of failure')
    assert plan['protocol']['training_coupling_grid'] == 'uniform_K'
    assert plan['protocol']['auxiliary_temperatures_are_sampling_only'] is True
    assert plan['protocol']['training_seeds'] == [91]
    assert set(plan['configs']) == {'dg5_seed91'}
    config = plan['configs']['dg5_seed91']
    assert config['training']['epochs'] == 12000
    assert config['training']['learning_rate'] == .001
    assert config['training']['seed'] == 91
    assert config['loss']['closure_objective'] == 'all_start_anchor'
    assert config['loss']['closure_scale_weights'] is None
    policy = plan['resource_policy']
    assert policy['cpu_seconds'] == 48 * 3600
    assert policy['eligible_gpu_uuids'] == {'3': runner.GPU_UUIDS[3]}
    assert policy['gpu_sharing_authorized'] is True
    assert policy['gpu_utilization_percent_max'] == 10
    assert len([s for s in runner.stage_list(root, plan) if s[0].startswith('sample_')]) == 84
    assert runner.verify(root)[0] == plan
    # A grid change still requires a new campaign, never rewriting old plans.
    with pytest.raises(FileExistsError):
        runner.prepare(root)
