from math import prod

import pytest
import torch
from torch import nn

from recurrent_rg.adapters import ContinuousFieldHead
from recurrent_rg.geometry import LatticeGeometry, chain_geometry, square_geometry
from recurrent_rg.model_nd import LocalCoarseGrainer, RecurrentRGNetwork1D, RecurrentRGNetwork2D


@pytest.mark.parametrize("shape,factors", [((8,), 2), ((4, 6), 2), ((4, 4, 4), 2), ((6, 4), (3, 2))])
def test_blocks_cover_each_site_once_with_channels_and_gradients(shape, factors):
    geometry = LatticeGeometry(len(shape), block_factor=factors)
    x = torch.arange(2 * 3 * prod(shape) * 2, dtype=torch.float64).reshape(2, 3, *shape, 2)
    x.requires_grad_()
    blocks = geometry.extract_blocks(x)
    coarse_shape = tuple(length // factor for length, factor in zip(shape, geometry.factors))
    assert blocks.shape == (2, 3, *coarse_shape, geometry.block_volume * 2)
    # No sites are omitted, duplicated, or mixed across systems/samples.
    assert torch.equal(blocks.flatten(2).sort().values, x.flatten(2).sort().values)
    first_block = x[(0, 0) + tuple(slice(0, f) for f in geometry.factors) + (slice(None),)]
    assert torch.equal(blocks[(0, 0) + (0,) * len(shape)], first_block.flatten())
    blocks.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("shape", [(8,), (4, 4), (4, 4, 4)])
def test_kernel_only_uses_its_own_block_and_downsamples_once(shape):
    geometry = LatticeGeometry(len(shape))
    head = ContinuousFieldHead(geometry.block_volume, channels=1).double()
    kernel = LocalCoarseGrainer(geometry, 1, geometry.block_volume, head).double()
    # A block sum makes exact input dependencies visible without STE/tanh.
    kernel.local_encoder = nn.Identity()
    with torch.no_grad():
        head.projection.weight.fill_(1.)
        head.projection.bias.zero_()
    x = torch.arange(prod(shape), dtype=torch.float64).reshape(1, 1, *shape, 1).requires_grad_()
    coarse = kernel(x)
    assert coarse.shape == (1, 1, *(length // 2 for length in shape), 1)
    assert torch.equal(coarse, geometry.extract_blocks(x).sum(dim=-1, keepdim=True))
    coarse[(0, 0) + (0,) * len(shape) + (0,)].backward()
    expected = torch.zeros_like(x)
    expected[(0, 0) + (slice(0, 2),) * len(shape) + (0,)] = 1.
    assert torch.equal(x.grad, expected)


@pytest.mark.parametrize("constructor,c_inputs,q_inputs", [(RecurrentRGNetwork1D, 2, 3), (RecurrentRGNetwork2D, 4, 9)])
def test_default_c_block_is_independent_of_q_window(constructor, c_inputs, q_inputs):
    model = constructor(hidden=8)
    assert model.rg_kernel.local_encoder[0].in_features == c_inputs
    assert model.theory_encoder.local_mlp[0].in_features == q_inputs
    assert model.action_network.local_energy[0].in_features == q_inputs + 1
    wider = constructor(hidden=8, width=5)
    assert wider.rg_kernel.local_encoder[0].in_features == c_inputs
    legacy = constructor(hidden=8, kernel_layout="overlapping")
    assert legacy.rg_kernel.local_encoder[0].in_features == q_inputs


def test_legacy_overlap_forward_matches_previous_definition():
    model = RecurrentRGNetwork1D(hidden=8, kernel_layout="overlapping").double()
    x = torch.randn(2, 4, 8, 1, dtype=torch.float64)
    kernel = model.rg_kernel
    expected = kernel.geometry.downsample(kernel.field_head(
        kernel.local_encoder(kernel.geometry.extract_patches(x))))
    assert torch.equal(kernel(x), expected)


def test_nondivisible_blocks_fail_without_dropping_sites():
    with pytest.raises(ValueError, match="divisible"):
        chain_geometry().extract_blocks(torch.zeros(1, 1, 7, 1))
    with pytest.raises(ValueError, match="divisible"):
        square_geometry().extract_blocks(torch.zeros(1, 1, 4, 5, 1))


def test_recurrent_1d_shapes_and_gradients():
    model = RecurrentRGNetwork1D(hidden=8, levels=4).double()
    trajectory = model(torch.randn(2, 4, 16, 1, dtype=torch.float64))
    assert [level.shape[2] for level in trajectory.configurations] == [16, 8, 4, 2]
    trajectory.direct_theory.square().sum().backward()
    assert model.rg_kernel.local_encoder[0].weight.grad is not None
