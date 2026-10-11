"""Exact-count motif pooling versus the original one-hot implementation."""
import pytest
import torch
from torch.nn import functional as F

from recurrent_rg.data.statistics import (
    plaquette_distribution,
    two_spin_distribution,
)


DEVICES = [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA runtime is unavailable')),
]


def old_distribution(configurations, dimensions):
    bits = (configurations.squeeze(-1) > 0).long()
    if dimensions == 1:
        pattern = 2 * bits + torch.roll(bits, -1, dims=2)
        return F.one_hot(pattern, 4).to(configurations.dtype).mean(dim=(1, 2))
    pattern = (8 * bits + 4 * torch.roll(bits, -1, dims=2)
               + 2 * torch.roll(bits, -1, dims=3)
               + torch.roll(torch.roll(bits, -1, dims=2), -1, dims=3))
    return F.one_hot(pattern, 16).to(configurations.dtype).mean(dim=(1, 2, 3))


def descriptor(dimensions):
    return two_spin_distribution if dimensions == 1 else plaquette_distribution


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dimensions', [1, 2])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32,
                                   torch.float16, torch.bfloat16])
def test_count_distribution_matches_one_hot(device, dimensions, dtype):
    generator = torch.Generator().manual_seed(981)
    lattice = (11,) if dimensions == 1 else (7, 11)
    x = (2 * torch.randint(2, (3, 5, *lattice, 1), generator=generator) - 1)
    x = x.to(device=device, dtype=dtype)
    result = descriptor(dimensions)(x)
    assert result.dtype == dtype and result.device == x.device
    assert result.shape == (3, 4 if dimensions == 1 else 16)
    torch.testing.assert_close(result, old_distribution(x, dimensions),
                               rtol=2 * torch.finfo(dtype).eps,
                               atol=2 * torch.finfo(dtype).eps)
    torch.testing.assert_close(result.sum(-1), torch.ones(3, device=device, dtype=dtype))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dimensions', [1, 2])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_uniform_ensembles_keep_original_pattern_codes(device, dimensions, sign):
    lattice = (9,) if dimensions == 1 else (5, 7)
    x = torch.full((2, 3, *lattice, 1), sign, device=device, dtype=torch.float64)
    result = descriptor(dimensions)(x)
    expected = torch.zeros_like(result)
    expected[:, 0 if sign < 0 else -1] = 1.
    torch.testing.assert_close(result, expected, rtol=0., atol=0.)
    torch.testing.assert_close(result, old_distribution(x, dimensions), rtol=0., atol=0.)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dimensions', [1, 2])
def test_noncontiguous_slices_preserve_threshold_and_probabilities(device, dimensions):
    generator = torch.Generator().manual_seed(982)
    lattice = (15,) if dimensions == 1 else (9, 15)
    # The descriptor historically thresholds at zero, rather than validating
    # strictly +/-1 fields. Keep that behavior for callers and diagnostics.
    x = torch.randn((4, 8, *lattice, 1), generator=generator, dtype=torch.float64)
    x[0].zero_()
    x = x.to(device)
    x = x[::2, 1::2, ::2] if dimensions == 1 else x[::2, 1::2, ::2, ::3]
    assert not x.is_contiguous()
    result = descriptor(dimensions)(x)
    torch.testing.assert_close(result, old_distribution(x, dimensions),
                               rtol=1e-15, atol=1e-15)
    assert not result.requires_grad


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dimensions', [1, 2])
@pytest.mark.parametrize('empty_axis', [0, 1, 2])
def test_empty_axes_match_historical_mean(device, dimensions, empty_axis):
    shape = [2, 3, 5, 1] if dimensions == 1 else [2, 3, 5, 7, 1]
    shape[empty_axis] = 0
    x = torch.empty(shape, device=device, dtype=torch.float64)
    result = descriptor(dimensions)(x)
    expected = old_distribution(x, dimensions)
    assert result.shape == expected.shape and result.dtype == expected.dtype
    torch.testing.assert_close(result, expected, rtol=0., atol=0., equal_nan=True)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_singleton_channels_not_required_by_historical_pooling(dimensions):
    generator = torch.Generator().manual_seed(983)
    lattice = (7,) if dimensions == 1 else (5, 7)
    x = (2 * torch.randint(2, (2, 3, *lattice, 2), generator=generator) - 1).double()
    torch.testing.assert_close(descriptor(dimensions)(x), old_distribution(x, dimensions),
                               rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_integer_input_still_rejected_like_mean(dimensions):
    lattice = (7,) if dimensions == 1 else (5, 7)
    x = torch.ones((2, 3, *lattice, 1), dtype=torch.int64)
    with pytest.raises(RuntimeError, match='mean'):
        old_distribution(x, dimensions)
    with pytest.raises(RuntimeError, match='mean'):
        descriptor(dimensions)(x)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_half_precision_counts_do_not_overflow_before_normalization(dimensions):
    lattice = (70000,) if dimensions == 1 else (250, 280)
    x = torch.ones((1, 1, *lattice, 1), dtype=torch.float16)
    torch.testing.assert_close(descriptor(dimensions)(x), old_distribution(x, dimensions),
                               rtol=0., atol=0.)


@pytest.mark.parametrize('device', DEVICES)
def test_descriptors_do_not_consume_random_state(device):
    x = torch.ones((2, 4, 7, 7, 1), device=device, dtype=torch.float64)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    plaquette_distribution(x)
    two_spin_distribution(x[:, :, :, 0])
    assert torch.equal(torch.get_rng_state(), before_cpu)
    if before_cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state(), before_cuda)
