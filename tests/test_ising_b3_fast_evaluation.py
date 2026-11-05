"""Frozen b=3 reporting matches the reference geometry, draws and Q capacity."""
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


def case(*, size=36, depth=2, features=12, sampling='bernoulli'):
    torch.manual_seed(81)
    model = build_model(ModelConfig(dimension=2, theory_dim=5, hidden=8,
        local_features=features, block_factor=3, width=3, binary_lookup=True,
        kernel_sampling=sampling), depth)
    data = (torch.randint(0, 2, (3, 4, size, size, 1)) * 2 - 1).double()
    return model, data


@pytest.mark.parametrize('features', [12, 24])
@pytest.mark.parametrize('augment', [False, True])
def test_L36_depth2_arrays_and_state_match_reference(features, augment):
    model, data = case(features=features)
    before, rng = _model_digest(model), torch.get_rng_state().clone()
    expected = collect_trajectories(model, data, samples=4, chunk_size=2,
        sampling_seed=29, spin_flip_augment=augment)
    collector = FrozenBinaryCollector(model)
    actual = collector.collect(data, samples=4, chunk_size=2,
        sampling_seed=29, spin_flip_augment=augment)
    assert collector._q_values.shape == (512, features)
    assert collector._c_logits.shape == (512, 1)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert actual['direct'].shape == (3, 3, 5)
    assert torch.equal(rng, torch.get_rng_state())
    assert _model_digest(model) == before
    assert model.training and all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('features', [12, 24])
def test_L108_depth3_small_configuration_panel_matches_reference(features):
    model, data = case(size=108, depth=3, features=features)
    expected = collect_trajectories(model, data, samples=2, chunk_size=1, sampling_seed=37)
    actual = FrozenBinaryCollector(model).collect(data, samples=2, chunk_size=1, sampling_seed=37)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert actual['direct'].shape == (3, 4, 5)


@pytest.mark.parametrize('sampling', ['bernoulli', 'deterministic'])
def test_b3_block_offset_bit_order_logits_STE_and_rng_match(sampling, monkeypatch):
    model, _ = case(sampling=sampling)
    # Exactly one of every 512 block patterns, tiled as [32,16] distinct blocks.
    codes = np.arange(512, dtype=np.uint16).reshape(32, 16)
    bits = np.zeros((1, 2, 96, 48), dtype=np.bool_)
    for bit in range(9):
        row, column = divmod(bit, 3)
        bits[:, :, row::3, column::3] = (codes >> bit) & 1
    configurations = torch.from_numpy(2 * bits.astype(np.float64) - 1)[..., None]
    collector = FrozenBinaryCollector(model)
    rng = torch.get_rng_state().clone()
    record = []
    head = model.rg_kernel.field_head
    sample = head.sample_logits
    def capture(logits):
        before = torch.get_rng_state().clone()
        value = sample(logits)
        record.append((logits.clone(), before, value.clone()))
        return value
    monkeypatch.setattr(head, 'sample_logits', capture)
    with torch.no_grad():
        expected = model.rg_kernel(configurations)
        reference_rng = torch.get_rng_state().clone()
        reference = list(record)
        record.clear()
        torch.set_rng_state(rng)
        actual = collector._coarsen(bits)
        actual_rng = torch.get_rng_state().clone()
    np.testing.assert_array_equal(actual, expected.numpy()[..., 0] > 0)
    assert torch.equal(reference_rng, actual_rng)
    if sampling == 'bernoulli':
        assert len(record) == len(reference) == 1
        assert all(torch.equal(a, b) for a, b in zip(record[0], reference[0]))
        # Every b=3 pattern, including index 511, must address its own logits.
        assert torch.equal(record[0][0][0, 0, ..., 0].flatten(), collector._c_logits[:, 0])


@pytest.mark.parametrize('steps', [0, 1, 2])
def test_b3_evaluation_depth_and_cached_micro_passes_keep_independent_C_draws(steps):
    model, data = case(features=24)
    collector = FrozenBinaryCollector(model)
    collector.collect(data, samples=4, chunk_size=2, sampling_seed=29)
    expected = collect_trajectories(model, data, samples=4, chunk_size=2,
        sampling_seed=91, max_rg_steps=steps)
    actual = collector.collect(data, samples=4, chunk_size=2,
        sampling_seed=91, max_rg_steps=steps)
    assert collector.cache_hits == 2 and collector.cache_misses == 2
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)


def test_b3_invalid_depth_divisibility_rejected_without_sampling():
    model, data = case()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='b=3'):
        FrozenBinaryCollector(model).collect(data, samples=4, max_rg_steps=3)
    assert torch.equal(rng, torch.get_rng_state())
