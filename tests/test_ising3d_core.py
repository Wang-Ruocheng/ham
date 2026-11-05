"""The explicit 3D cube representation does not reinterpret old geometries."""

import copy
from itertools import product

import pytest
import torch

from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig, TrainingConfig, EvaluationConfig
from recurrent_rg.data.statistics import cube_distribution
from recurrent_rg.execution import reference_binary_pattern_distribution
from recurrent_rg.geometry import LatticeGeometry, anchored_cubic_geometry, chain_geometry, square_geometry


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def binary(shape, seed=17):
    return (2 * torch.randint(2, shape, generator=torch.Generator().manual_seed(seed)) - 1).double()


def test_anchored_cube_order_periodicity_and_nonoverlapping_blocking():
    geometry = anchored_cubic_geometry()
    x = torch.arange(4**3, dtype=torch.double).reshape(1, 1, 4, 4, 4, 1)
    assert geometry.patch_offsets() == tuple(product(range(2), repeat=3))
    patches = geometry.extract_patches(x)
    assert patches.shape == (1, 1, 4, 4, 4, 8)
    for position in product(range(4), repeat=3):
        expected = torch.stack([x[(0, 0) + tuple((i+d) % 4 for i, d in zip(position, offset)) + (0,)]
                                for offset in geometry.patch_offsets()])
        torch.testing.assert_close(patches[(0, 0) + position], expected, atol=0, rtol=0)
    blocks = geometry.extract_blocks(x)
    assert blocks.shape == (1, 1, 2, 2, 2, 8)
    torch.testing.assert_close(blocks, patches[:, :, ::2, ::2, ::2], atol=0, rtol=0)


def test_cube_motifs_match_independent_brute_force_all_256_patterns():
    # Enumerate every oriented cube as a distinct ensemble; count all periodic
    # translations independently (translations permute a cube's bit positions).
    values = torch.arange(256)
    spins = (2 * ((values[:, None] // 2**torch.arange(7, -1, -1)) % 2) - 1).double()
    x = spins.reshape(256, 1, 2, 2, 2, 1)
    expected = torch.zeros(256, 256, dtype=torch.double)
    offsets = tuple(product(range(2), repeat=3))
    for ensemble in range(256):
        for anchor in product(range(2), repeat=3):
            code = 0
            for offset in offsets:
                position = tuple((i+d) % 2 for i, d in zip(anchor, offset))
                code = 2 * code + int(x[(ensemble, 0) + position + (0,)] > 0)
            expected[ensemble, code] += 1 / 8
    actual = cube_distribution(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual.mean(0), torch.full((256,), 1/256, dtype=torch.double), atol=0, rtol=0)
    assert actual[0, 0] == actual[255, 255] == 1


def test_cube_counted_and_reference_distributions_match_exactly():
    x = binary((3, 4, 4, 6, 8, 1))
    torch.testing.assert_close(cube_distribution(x), reference_binary_pattern_distribution(x), atol=0, rtol=0)
    torch.testing.assert_close(cube_distribution(x).sum(1), torch.ones(3, dtype=torch.double))
    with pytest.raises(ValueError, match='3D fields'):
        cube_distribution(x[..., 0])


def test_3d_model_shapes_random_probabilistic_c_and_affine_256_decoder():
    torch.manual_seed(91)
    config = ModelConfig(dimension=3, width=2, hidden=8, theory_dim=1)
    model = build_model(config, 2)
    assert model.theory_encoder.local_mlp[0].in_features == 8
    assert model.rg_kernel.local_encoder[0].in_features == 8
    assert model.rg_kernel.field_head.sampling_mode == 'bernoulli'
    decoder = model.theory_encoder.distribution_decoder
    assert decoder.in_features == 1 and decoder.out_features == 256
    torch.testing.assert_close(decoder.weight.norm(dim=0), torch.ones(1, dtype=torch.double))
    trajectory = model(binary((2, 4, 16, 16, 16, 1)))
    assert [tuple(level.shape[2:-1]) for level in trajectory.configurations] == [(16,)*3, (8,)*3, (4,)*3]
    assert trajectory.direct_theory.shape == trajectory.rollout_theory.shape == (2, 3, 1)
    for level in trajectory.configurations:
        # Straight-through hard + soft - detached-soft can differ by roundoff.
        torch.testing.assert_close(level.abs(), torch.ones_like(level), atol=1e-12, rtol=0)
    assert all(parameter.dtype == torch.double for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.rg_kernel.parameters())


def test_3d_lookup_preserves_q_c_r_values_and_input_parameter_gradients():
    torch.manual_seed(11)
    model = build_model(ModelConfig(dimension=3, width=2, hidden=5, theory_dim=1,
        binary_lookup=False, affine_reconstruction=False, kernel_sampling='deterministic'), 2)
    fast = copy.deepcopy(model)
    fast.theory_encoder.binary_lookup = fast.rg_kernel.binary_lookup = True
    x = binary((2, 4, 8, 8, 8, 1)).requires_grad_()
    y = x.detach().clone().requires_grad_()
    outputs, gradients = [], []
    for current, data in [(model, x), (fast, y)]:
        trajectory = current(data)
        loss = trajectory.direct_theory.square().sum() + trajectory.rollout_theory.square().sum()
        parameters = tuple(p for module in (current.theory_encoder, current.rg_kernel, current.theory_rg)
                           for p in module.parameters())
        gradients.append(torch.autograd.grad(loss, (data,) + parameters))
        outputs.append(torch.cat((trajectory.direct_theory.flatten(), trajectory.rollout_theory.flatten())))
    torch.testing.assert_close(*outputs, atol=1e-12, rtol=1e-10)
    for reference, optimized in zip(*gradients):
        assert torch.isfinite(reference).all() and torch.isfinite(optimized).all()
        torch.testing.assert_close(reference, optimized, atol=1e-11, rtol=1e-8)


def test_3d_checkpoint_state_roundtrip_and_config_serialization():
    config = RunConfig(data=DataConfig(cache='unused', physical_model='3D Ising'),
        model=ModelConfig(dimension=3, width=2, hidden=5),
        training=TrainingConfig(max_rg_steps=2), evaluation=EvaluationConfig(max_rg_steps=2))
    config.validate()
    restored = RunConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    original = build_model(config.model, 2)
    second = build_model(restored.model, 2)
    load_model_state(second, {'model': qcr_state_dict(original)})
    for key, value in qcr_state_dict(original).items():
        torch.testing.assert_close(value, qcr_state_dict(second)[key], atol=0, rtol=0)


@pytest.mark.parametrize('dimension', [1, 2])
def test_old_centered_geometries_and_decoder_shapes_are_unchanged(dimension):
    geometry = chain_geometry() if dimension == 1 else square_geometry()
    assert geometry.patch_offsets() == tuple(product(range(-1, 2), repeat=dimension))
    with pytest.raises(ValueError, match='odd'):
        LatticeGeometry(dimension, patch_width=2)
    old = build_model(ModelConfig(dimension=dimension, hidden=5), 1)
    assert old.theory_encoder.local_mlp[0].in_features == 3**dimension
    assert old.theory_encoder.distribution_decoder.out_features == 2**(2**dimension)
    invalid = RunConfig(data=DataConfig(cache='unused', physical_model='test'),
                        model=ModelConfig(dimension=dimension, width=2))
    with pytest.raises(ValueError, match='odd'):
        invalid.validate()


def test_3d_does_not_silently_reinterpret_centered_width3():
    with pytest.raises(ValueError, match='anchored'):
        build_model(ModelConfig(dimension=3, width=3), 1)
    config = RunConfig(data=DataConfig(cache='unused', physical_model='test'),
                       model=ModelConfig(dimension=3, width=3))
    with pytest.raises(ValueError, match='anchored'):
        config.validate()
