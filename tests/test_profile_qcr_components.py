"""Diagnostic timing does not change QCR mathematics, routing, or RNG draws."""
from dataclasses import replace
import json

import pytest
import torch

from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig,
)
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.rsmi import region_offsets
from scripts import profile_qcr_components as profile


def test_exclusive_clock_subtracts_nested_regions_without_double_count():
    ticks = iter([0., 2., 5., 9.])
    clock = profile.ExclusiveClock(now=lambda: next(ticks))
    with clock.region('parent'):
        with clock.region('child'):
            pass
    assert dict(clock.values) == {'child': 3., 'parent': 6.}
    assert sum(clock.values.values()) == 9.
    assert not clock.stack


def test_exclusive_clock_resets_and_unwinds_on_exception():
    clock = profile.ExclusiveClock()
    with pytest.raises(RuntimeError, match='probe'):
        with clock.region('parent'):
            with pytest.raises(RuntimeError, match='reset'):
                clock.reset()
            raise RuntimeError('probe')
    assert not clock.stack
    clock.reset()
    assert not clock.values


def _fixture(tmp_path):
    generator = torch.Generator().manual_seed(17)
    train = (2*torch.randint(2, (3, 4, 8, 8, 1), generator=generator)-1).double()
    test = train[:2].clone()
    data = tmp_path/'data.pt'
    torch.save((train, torch.arange(3), test, torch.arange(2)+10), data)
    cfg = RunConfig(data=DataConfig(str(data), 'ising2d', ('K',)),
        model=ModelConfig(dimension=2, theory_dim=2, hidden=4, block_factor=2),
        loss=LossConfig(kernel_sign_weight=1., kernel_noncollapse_mode='variance_floor_ensemble',
                        kernel_variance_ratio=1.),
        training=TrainingConfig(max_rg_steps=1, epochs=20, system_batch=2,
            sample_batch=4, threads=1, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=1, samples=4, systems=2))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg.to_dict()))
    digest = '1'*64
    _, environment = region_offsets(2, block=2, buffer=1)
    identity = lattice_statistics_cache_key(cfg, digest, 'train', train, environment,
        'binary_lattice_mi_statistics', dict(block=2))
    _, record = load_or_build_mi_statistics(tmp_path/'MI', identity,
        lambda: binary_lattice_mi_statistics(train, block=2))
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(data_sha256=digest,
        splits=dict(train=dict(cache=record, data_shape=list(train.shape))))))
    return cfg, config, receipt, record


def test_cpu_profile_exact_twin_and_json_only(tmp_path):
    _, config, receipt, _ = _fixture(tmp_path)
    output = tmp_path/'cost.json'
    result = profile.profile(config, output, mi_receipt=receipt,
        warmup=1, steps=2, mi_only_steps=1)
    assert result['status'] == 'complete'
    assert all(result['exact_twin_equivalence'].values())
    assert result['inputs']['independent_data_rehash'] is False
    assert result['labels_used_in_loss'] is False
    assert result['checkpoints_saved'] is False
    assert result['MI_only_probe']['optimizer_updates'] == 0
    assert result['MI_only_probe']['included_in_exclusive_update_sum'] is False
    measured = result['instrumented']
    assert measured['measured_updates'] == 2
    assert .95 < measured['accounted_fraction'] <= 1.000001
    assert set(measured['buckets']) == set(profile.BUCKETS)
    assert measured['buckets']['C_routed_backward']['seconds'] > 0.
    assert measured['buckets']['Q_encoding_forward']['seconds'] > 0.
    assert measured['buckets']['R_forward_including_legacy_diagnostics']['seconds'] > 0.
    assert json.loads(output.read_text())['status'] == 'complete'
    assert not list(tmp_path.glob('training_step_*.pt'))
    with pytest.raises(FileExistsError):
        profile.profile(config, output, mi_receipt=receipt, warmup=0, steps=1)


def test_cache_identity_mismatch_fails_without_rebuilding(tmp_path):
    cfg, _, receipt, record = _fixture(tmp_path)
    before = profile.stat_identity(record['path'])
    wrong = replace(cfg, model=replace(cfg.model, block_factor=4),
                    training=replace(cfg.training, max_rg_steps=1))
    # Geometry minimum is checked before cache identity for this artificial case.
    with pytest.raises(ValueError):
        profile.load_inputs(wrong, mi_receipt=receipt)
    assert profile.stat_identity(record['path']) == before
    with pytest.raises(ValueError, match='fingerprint'):
        profile.load_inputs(cfg, mi_receipt=receipt, data_sha256='2'*64)


def test_scoped_patches_restore_even_if_profiled_call_raises(tmp_path):
    cfg, _, _, _ = _fixture(tmp_path)
    train = profile._read_train(cfg.data.cache)
    model, _, _ = profile._make_model(cfg, train, torch.device('cpu'))
    old_gradient = torch.autograd.grad
    old_split = profile.losses.split_theory_views
    old_coarse = model.coarse_grain
    with pytest.raises(RuntimeError, match='probe'):
        with profile.scoped_instrumentation(profile.ExclusiveClock(), model):
            assert torch.autograd.grad is not old_gradient
            raise RuntimeError('probe')
    assert torch.autograd.grad is old_gradient
    assert profile.losses.split_theory_views is old_split
    assert model.coarse_grain == old_coarse
