"""Low-RAM loading preserves bytes, validation coverage, labels and RNG."""
import hashlib

import pytest
import torch

from recurrent_rg.config import RunConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig, EvaluationConfig
from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from recurrent_rg.runner import prepared_data
from recurrent_rg.tensor_io import load_cpu_tensors, tensor_chunks, all_finite


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, categorical=False, legacy=False, versioned=False):
    generator = torch.Generator().manual_seed(91)
    shape = (4, 8, 8, 8)
    values = torch.randint(2, shape, generator=generator)
    if categorical:
        x = torch.nn.functional.one_hot(values, 2).double()
    else:
        x = (values * 2 - 1).double().unsqueeze(-1)
    y = torch.arange(4).double()
    cache = tmp_path / 'data.pt'
    payload = (x[:2], y[:2], x[2:], y[2:])
    if versioned:
        payload = dict(schema_version=1,
                       train=dict(configurations=x[:2], couplings=y[:2]),
                       test=dict(configurations=x[2:], couplings=y[2:]))
    torch.save(payload, cache, _use_new_zipfile_serialization=not legacy)
    config = RunConfig(DataConfig(str(cache), 'synthetic', ('K',)),
        model=ModelConfig(dimension=2, block_factor=2, hidden=4,
                          categorical_states=2 if categorical else 0,
                          kernel_sampling='categorical' if categorical else 'bernoulli'),
        loss=LossConfig(kernel_noncollapse_mode='variance_floor_ensemble'),
        training=TrainingConfig(max_rg_steps=1, sample_batch=4, threads=1),
        evaluation=EvaluationConfig(max_rg_steps=1, samples=4, systems=2))
    return cache, config


@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 1, 31), (), (0, 5)])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_bounded_chunks_are_views_cover_every_element(shape, noncontiguous):
    x = torch.arange(max(1, int(torch.tensor(shape).prod()))).reshape(shape) if all(shape) else torch.empty(shape)
    if noncontiguous and x.ndim > 1:
        x = x.transpose(0, -1)
    chunks = list(tensor_chunks(x, max_elements=7))
    assert sum(chunk.numel() for chunk in chunks) == x.numel()
    assert all(chunk.numel() <= 7 for chunk in chunks)
    assert sum(float(chunk.double().sum()) for chunk in chunks) == float(x.double().sum())
    assert all(chunk.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for chunk in chunks)


def test_chunks_keep_channel_axis_for_onehot_predicate():
    x = torch.nn.functional.one_hot(torch.arange(60).reshape(3, 4, 5) % 3, 3).double()
    chunks = list(tensor_chunks(x, max_elements=13, keep_last_dim=True))
    assert all(chunk.shape[-1] == 3 and chunk.numel() <= 13 for chunk in chunks)
    assert sum(chunk.numel() for chunk in chunks) == x.numel()
    assert all(bool((chunk.sum(-1) == 1).all()) for chunk in chunks)
    with pytest.raises(ValueError, match='channel'):
        list(tensor_chunks(x, max_elements=2, keep_last_dim=True))


@pytest.mark.parametrize('limit', [0, -1, True, 2.5])
def test_invalid_chunk_size_rejected(limit):
    with pytest.raises(ValueError, match='positive integer'):
        list(tensor_chunks(torch.ones(2), max_elements=limit))


def test_finite_validation_is_bounded_and_covers_last_element(monkeypatch):
    x = torch.ones(101, 53, dtype=torch.double).t()
    original = torch.isfinite
    sizes = []
    def observe(value):
        sizes.append(value.numel())
        return original(value)
    monkeypatch.setattr(torch, 'isfinite', observe)
    before = torch.get_rng_state().clone()
    assert all_finite(x, max_elements=31)
    x[-1, -1] = float('nan')
    assert not all_finite(x, max_elements=31)
    assert max(sizes) <= 31
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('versioned', [False, True])
def test_mapped_and_eager_datasets_exact_bytes_and_rng(tmp_path, legacy, versioned, monkeypatch):
    cache, config = fixture(tmp_path, legacy=legacy, versioned=versioned)
    original_sha = fingerprint(cache)
    original = torch.load
    calls = []
    def observe(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)
    monkeypatch.setattr(torch, 'load', observe)
    before = torch.get_rng_state().clone()
    eager, mapped = prepared_data(config, mmap=False), prepared_data(config, mmap=True)
    assert calls[0] == dict(map_location='cpu', weights_only=True)
    assert calls[1] == dict(map_location='cpu', weights_only=True, **({} if legacy else {'mmap': True}))
    for a, b in zip(eager, mapped):
        assert torch.equal(a.configurations, b.configurations)
        assert a.configurations.dtype == b.configurations.dtype == torch.float64
        assert torch.equal(a.couplings, b.couplings)
    # Selected batches are copies; mutating one never changes mapped storage.
    selected = mapped[0].configurations[torch.tensor([1, 0])]
    selected.zero_()
    assert torch.equal(eager[0].configurations, mapped[0].configurations)
    assert torch.equal(before, torch.get_rng_state())
    assert fingerprint(cache) == original_sha


def test_private_mapping_never_writes_through_to_synthetic_cache(tmp_path):
    cache, _ = fixture(tmp_path)
    original_sha = fingerprint(cache)
    train, _ = load_ensembles(cache, mmap=True)
    train.configurations[0, 0, 0, 0, 0] = 99
    assert fingerprint(cache) == original_sha
    assert load_ensembles(cache)[0].configurations[0, 0, 0, 0, 0] != 99


def test_corrupt_modern_cache_is_not_retried_eager(tmp_path, monkeypatch):
    cache = tmp_path / 'bad.pt'
    cache.write_bytes(b'PK\x03\x04not a valid archive')
    original = torch.load
    calls = []
    def observe(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)
    monkeypatch.setattr(torch, 'load', observe)
    with pytest.raises((RuntimeError, ValueError)):
        load_cpu_tensors(cache, mmap=True)
    assert calls == [dict(map_location='cpu', weights_only=True, mmap=True)]


def test_shared_global_mapping_option_is_rejected_without_mutating_it(tmp_path, monkeypatch):
    import mmap
    cache, _ = fixture(tmp_path)
    monkeypatch.setattr(torch.serialization, 'get_default_mmap_options', lambda: mmap.MAP_SHARED, raising=False)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Shared mapping must not load'))
    with pytest.raises(ValueError, match='private, not shared'):
        load_cpu_tensors(cache, mmap=True)
    assert torch.serialization.get_default_mmap_options() == mmap.MAP_SHARED


@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('bad', ['nonfinite', 'alphabet', 'geometry', 'labels'])
def test_prepared_data_still_rejects_invalid_content(tmp_path, categorical, bad):
    cache, config = fixture(tmp_path, categorical=categorical)
    tx, ty, vx, vy = torch.load(cache, weights_only=True)
    if bad == 'nonfinite':
        vx[-1, -1, -1, -1, -1] = float('inf')
    elif bad == 'alphabet':
        vx[-1, -1, -1, -1, -1] = 0.5
    elif bad == 'geometry':
        tx, vx = tx.unsqueeze(-1), vx.unsqueeze(-1)
    else:
        vy = ty.clone()
    torch.save((tx, ty, vx, vy), cache)
    with pytest.raises(ValueError):
        prepared_data(config)


def test_onehot_sum_checks_survive_chunk_boundaries(tmp_path, monkeypatch):
    cache, config = fixture(tmp_path, categorical=True)
    import recurrent_rg.runner as runner
    monkeypatch.setattr(runner, 'tensor_chunks', lambda x, **kw: tensor_chunks(x, max_elements=17, **kw))
    prepared_data(config)
    tx, ty, vx, vy = torch.load(cache, weights_only=True)
    vx[-1, -1, -1, -1, :] = 1
    torch.save((tx, ty, vx, vy), cache)
    with pytest.raises(ValueError, match='one-hot'):
        prepared_data(config)


def test_finite_metadata_and_group_split_still_checked(tmp_path):
    cache, _ = fixture(tmp_path)
    tx, ty, vx, vy = torch.load(cache, weights_only=True)
    torch.save(dict(schema_version=1,
        train=dict(configurations=tx, couplings=ty, group_ids=torch.tensor([0, 1])),
        test=dict(configurations=vx, couplings=vy, group_ids=torch.tensor([1, 2]))), cache)
    with pytest.raises(ValueError, match='overlap'):
        load_ensembles(cache, mmap=True)
    with pytest.raises(ValueError, match='nonfinite'):
        EnsembleDataset(tx, torch.tensor([0., float('nan')])).validate()
