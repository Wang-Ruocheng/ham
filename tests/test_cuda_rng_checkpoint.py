"""Diagnostic RNG isolation and same-device checkpoint RNG continuation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import (
    build_model, checkpoint_payload, isolated_rng, load_run, model_device,
    restore_checkpoint_rng,
)
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.evaluation import collect_trajectories
from recurrent_rg.training import evaluate, train


def configuration():
    cfg = RunConfig(data=DataConfig('unused', 'test'),
                    model=ModelConfig(dimension=2, hidden=4))
    return replace(cfg, training=replace(cfg.training, max_rg_steps=1))


def checkpoint(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(91)
    return checkpoint_payload(model, optimizer, generator, 1,
                              metadata={'config': cfg.to_dict()})


def fail_cuda_call(*args, **kwargs):
    raise AssertionError('CPU operation touched CUDA RNG')


def test_cpu_rng_isolation_repeats_draws_without_touching_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', fail_cuda_call)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', fail_cuda_call)
    before = torch.get_rng_state()
    with isolated_rng('cpu', 12345):
        first = torch.rand(8)
    assert torch.equal(before, torch.get_rng_state())
    with isolated_rng('cpu', 12345):
        assert torch.equal(first, torch.rand(8))
    assert torch.equal(before, torch.get_rng_state())


def test_cpu_evaluation_preserves_rng_and_repeats_metrics(monkeypatch):
    torch.set_num_threads(1)
    model = build_model(configuration().model, 1)
    x = (2*torch.randint(2, (2, 4, 12, 12, 1))-1).double()
    options = dict(kernel_noncollapse_mode='variance_floor_ensemble',
                   kernel_variance_ratio=1., kernel_sign_weight=1.)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', fail_cuda_call)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', fail_cuda_call)
    before = torch.get_rng_state()
    first = evaluate(model, x, plaquette_distribution, 4,
                     spin_flip_augment=False, objective_kwargs=options)
    assert torch.equal(before, torch.get_rng_state())
    assert first == evaluate(model, x, plaquette_distribution, 4,
                             spin_flip_augment=False, objective_kwargs=options)
    assert torch.equal(before, torch.get_rng_state())


def test_cpu_report_trajectories_preserve_rng_without_touching_cuda(monkeypatch):
    torch.set_num_threads(1)
    model = build_model(configuration().model, 1)
    model.train()
    x = (2*torch.randint(2, (2, 4, 12, 12, 1))-1).double()
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', fail_cuda_call)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', fail_cuda_call)
    before = torch.get_rng_state()
    first = collect_trajectories(model, x, samples=4, chunk_size=1)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    repeated = collect_trajectories(model, x, samples=4, chunk_size=1)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    assert model.training
    assert torch.equal(before, torch.get_rng_state())


def test_cpu_checkpoint_and_legacy_rng_restore_remain_compatible(monkeypatch):
    cfg = configuration()
    model = build_model(cfg.model, 1)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', fail_cuda_call)
    payload = checkpoint(model, cfg)
    assert 'cuda_rng_state' not in payload and 'cuda_device_index' not in payload
    expected = torch.rand(8)
    torch.rand(8)
    restore_checkpoint_rng({'torch_rng_state': payload['torch_rng_state']})
    assert torch.equal(expected, torch.rand(8))


@pytest.mark.parametrize('payload,device,message', [
    ({}, 'cuda:0', 'saved CUDA RNG state'),
    ({'cuda_rng_state': torch.empty(0, dtype=torch.uint8), 'cuda_device_index': 1},
     'cuda:0', 'same CUDA device'),
    ({'cuda_rng_state': torch.empty(0, dtype=torch.uint8), 'cuda_device_index': 0},
     'cpu', 'cannot resume exactly on CPU'),
])
def test_wrong_device_resume_fails_before_changing_cpu_rng(payload, device, message):
    before = torch.get_rng_state()
    payload = dict(payload, torch_rng_state=before)
    with pytest.raises(ValueError, match=message):
        restore_checkpoint_rng(payload, device=device)
    assert torch.equal(before, torch.get_rng_state())


def test_load_run_keeps_cpu_report_default_and_does_not_restore_rng(tmp_path, monkeypatch):
    cfg = configuration()
    model = build_model(cfg.model, 1)
    payload = checkpoint(model, cfg)
    # CUDA continuation metadata must not make an ordinary report use CUDA.
    payload.update(cuda_rng_state=torch.empty(0, dtype=torch.uint8), cuda_device_index=0)
    path = tmp_path/'checkpoint.pt'
    torch.save(payload, path)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', fail_cuda_call)
    before = torch.get_rng_state()
    loaded = load_run(path)
    assert model_device(loaded.model) == torch.device('cpu')
    assert torch.equal(before, torch.get_rng_state())


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')


@requires_cuda
def test_cuda_evaluation_accepts_cpu_inputs_and_preserves_both_rngs():
    torch.set_num_threads(1)
    model = build_model(configuration().model, 1).to('cuda:0')
    x = (2*torch.randint(2, (2, 4, 12, 12, 1))-1).double()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)
    first = evaluate(model, x, plaquette_distribution, 4, spin_flip_augment=False)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))
    assert first == evaluate(model, x, plaquette_distribution, 4, spin_flip_augment=False)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))


@requires_cuda
def test_cuda_checkpoint_restores_cpu_and_cuda_draws_and_loads_reports_on_cpu(tmp_path):
    cfg = configuration()
    model = build_model(cfg.model, 1).to('cuda:0')
    payload = checkpoint(model, cfg)
    assert payload['cuda_device_index'] == 0
    expected_cpu = torch.rand(8)
    expected_cuda = torch.rand(8, device='cuda:0')
    torch.rand(8)
    torch.rand(8, device='cuda:0')
    restore_checkpoint_rng(payload, device='cuda:0')
    assert torch.equal(expected_cpu, torch.rand(8))
    assert torch.equal(expected_cuda, torch.rand(8, device='cuda:0'))
    path = tmp_path/'checkpoint.pt'
    torch.save(payload, path)
    x = (2*torch.randint(2, (2, 4, 12, 12, 1))-1).double()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)
    loaded = load_run(path)
    assert model_device(loaded.model) == torch.device('cpu')
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))
    first = collect_trajectories(loaded.model, x, samples=4)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))
    repeated = collect_trajectories(loaded.model, x, samples=4)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))


@requires_cuda
def test_historical_generic_training_does_not_implicitly_enable_cuda():
    model = build_model(configuration().model, 1).to('cuda:0')
    with pytest.raises(ValueError, match='historical generic trainer remains CPU-only'):
        train(model, None, None, pattern_fn=None, epochs=1, system_batch=2,
              sample_batch=4, eval_samples=4, eval_systems=2,
              learning_rate=.001, seed=91)
