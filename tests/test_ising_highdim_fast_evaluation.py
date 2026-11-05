"""Numerical equivalence, RNG isolation and bounded specialized report path."""

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.evaluation import collect_trajectories
from scripts.ising_highdim_fast_evaluation import FrozenBinaryCollector, _model_digest


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def case(dim=5, sampling='bernoulli'):
    torch.manual_seed(81)
    model = build_model(ModelConfig(dimension=2, theory_dim=dim, hidden=96,
        block_factor=2, width=3, binary_lookup=True, kernel_sampling=sampling), 2)
    data = (torch.randint(0, 2, (5, 6, 12, 12, 1)) * 2 - 1).double()
    return model, data


@pytest.mark.parametrize('dim', [1, 5, 10, 100])
@pytest.mark.parametrize('augment', [False, True])
def test_all_report_arrays_match_original_with_partial_last_chunk(dim, augment):
    model, data = case(dim)
    before, rng = _model_digest(model), torch.get_rng_state().clone()
    old_training = model.training
    expected = collect_trajectories(model, data, samples=4, chunk_size=2,
        sampling_seed=12345, spin_flip_augment=augment)
    progress = []
    collector = FrozenBinaryCollector(model, progress=lambda *args: progress.append(args))
    actual = collector.collect(data, samples=4, chunk_size=2,
        sampling_seed=12345, spin_flip_augment=augment)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert [item[1] for item in progress] == [2, 4, 5]
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training == old_training
    assert _model_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4, 12345])
def test_repeated_c_passes_keep_original_draw_order_and_cache_only_micro(seed):
    model, data = case()
    collector = FrozenBinaryCollector(model)
    collector.collect(data, samples=4, chunk_size=2, sampling_seed=12345)
    expected = collect_trajectories(model, data, samples=4, chunk_size=2, sampling_seed=seed)
    actual = collector.collect(data, samples=4, chunk_size=2, sampling_seed=seed)
    assert collector.cache_hits == 3 and collector.cache_misses == 3
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-12)


def test_microscopic_cache_does_not_cross_data_or_mutations():
    model, data = case()
    collector = FrozenBinaryCollector(model)
    collector.collect(data, samples=4, chunk_size=2)
    other = data.clone()
    other[:2].neg_()
    actual = collector.collect(other, samples=4, chunk_size=2)
    assert collector.cache_hits == 0 and collector.cache_misses == 6
    expected = collect_trajectories(model, other, samples=4, chunk_size=2)
    np.testing.assert_allclose(actual['direct'], expected['direct'], rtol=0, atol=1e-12)
    # A NumPy alias mutation bypasses PyTorch's version counter; the bit digest
    # still invalidates precisely the changed microscopic chunk.
    other.numpy()[0, 0, 0, 0, 0] *= -1
    actual = collector.collect(other, samples=4, chunk_size=2)
    assert collector.cache_hits == 2 and collector.cache_misses == 7
    expected = collect_trajectories(model, other, samples=4, chunk_size=2)
    np.testing.assert_allclose(actual['direct'], expected['direct'], rtol=0, atol=1e-12)


def test_original_field_head_receives_identical_logits_shapes_rng_and_ste(monkeypatch):
    model, data = case()
    sample = model.rg_kernel.field_head.sample_logits
    records = []
    def record(logits):
        before = torch.get_rng_state().clone()
        value = sample(logits)
        records.append((logits.clone(), before, value.clone()))
        return value
    monkeypatch.setattr(model.rg_kernel.field_head, 'sample_logits', record)
    collect_trajectories(model, data, samples=4, chunk_size=2)
    reference = records[:]
    records.clear()
    FrozenBinaryCollector(model).collect(data, samples=4, chunk_size=2)
    assert len(records) == len(reference) == 6
    for old, new in zip(reference, records):
        assert all(torch.equal(a, b) for a, b in zip(old, new))


def test_no_materialized_patch_or_block_extraction_and_no_unused_half_q(monkeypatch):
    model, data = case()
    collector = FrozenBinaryCollector(model)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unused split/patch implementation called')
    monkeypatch.setattr(type(model.geometry), 'extract_patches', forbidden)
    monkeypatch.setattr(type(model.geometry), 'extract_blocks', forbidden)
    monkeypatch.setattr(model.theory_encoder, 'forward', forbidden)
    arrays = collector.collect(data, samples=4, chunk_size=2)
    assert arrays['direct'].shape == (5, 3, 5)


def test_frozen_parameter_and_sampling_mutations_refused():
    model, data = case()
    collector = FrozenBinaryCollector(model)
    with torch.no_grad():
        next(model.parameters()).add_(.1)
    with pytest.raises(RuntimeError, match='model changed'):
        collector.collect(data, samples=4)
    model, data = case()
    collector = FrozenBinaryCollector(model)
    model.rg_kernel.field_head.relaxation_temperature = 2.
    with pytest.raises(RuntimeError, match='model changed'):
        collector.collect(data, samples=4)


def test_deadline_retains_only_completed_prefix_and_restores_rng():
    model, data = case()
    class Deadline:
        stopped = False
        def check(self):
            if self.stopped:
                raise TimeoutError('Report walltime budget')
    deadline = Deadline()
    def progress(*args):
        deadline.stopped = True
    collector = FrozenBinaryCollector(model, deadline=deadline, progress=progress)
    rng, digest = torch.get_rng_state().clone(), _model_digest(model)
    with pytest.raises(TimeoutError):
        collector.collect(data, samples=4, chunk_size=2)
    assert collector.partial_arrays['direct'].shape == (2, 3, 5)
    assert collector.partial_options['completed_ensembles'] == 2
    assert collector.partial_rng_state is not None
    assert torch.equal(rng, torch.get_rng_state())
    assert _model_digest(model) == digest and model.training


def test_deterministic_sampling_and_zero_depth_match():
    model, data = case(sampling='deterministic')
    collector = FrozenBinaryCollector(model)
    for steps in (0, 1, 2):
        expected = collect_trajectories(model, data, samples=4, chunk_size=2, max_rg_steps=steps)
        actual = collector.collect(data, samples=4, chunk_size=2, max_rg_steps=steps)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)


def test_nonbinary_or_unsupported_model_refused():
    model, data = case()
    collector = FrozenBinaryCollector(model)
    data[0, 0, 0, 0, 0] = .5
    with pytest.raises(ValueError, match='nonbinary'):
        collector.collect(data, samples=4)
    model.theory_encoder.binary_lookup = False
    with pytest.raises(ValueError, match='Fast report requires'):
        FrozenBinaryCollector(model)
