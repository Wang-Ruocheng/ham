"""Synthetic implementation checks, not a physical phase-transition claim."""
from itertools import product

import pytest
import torch

from recurrent_rg.geometry import LatticeGeometry
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.model_nd import EnsembleTheoryEncoder
from recurrent_rg.phi4_aligned_qcr import (
    FEATURE_DIM,
    AlignedPhi4QCR,
    RawPatchTheoryEncoder,
    aligned_objectives,
    soft_plaquette_features,
)
from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives, soft_local_features


@pytest.fixture(autouse=True)
def isolated_rng_and_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        yield
    torch.set_num_threads(previous)


def _parity_fields():
    patterns = torch.tensor(list(product((-1., 1.), repeat=4)), dtype=torch.double)
    fields = patterns.reshape(16, 2, 2).repeat(1, 2, 2).unsqueeze(-1)
    parity = patterns.prod(-1)
    return torch.stack((fields[parity == 1], fields[parity == -1]))


def _reference_joint(x, scale):
    values = x[..., 0] / scale
    centers = torch.arange(-2, 3, dtype=x.dtype)
    probability = torch.softmax(-.5 * ((values[..., None] - centers) / .5).square(), dim=-1)
    distributions = [torch.roll(probability, shifts=(-i, -j), dims=(2, 3))
                     for i, j in ((0, 0), (1, 0), (0, 1), (1, 1))]
    joint = torch.einsum("enija,enijb,enijc,enijd->eabcd", *distributions)
    return joint.flatten(1) / (x.shape[1] * x.shape[2] * x.shape[3])


@pytest.mark.parametrize("chunk_size", [1, 7, 128, 4096])
def test_complete_soft_joint_matches_direct_reference(chunk_size):
    x = torch.randn(2, 3, 4, 4, 1, dtype=torch.double)
    actual = soft_plaquette_features(x, 1.7, chunk_size=chunk_size)
    assert actual.shape == (2, FEATURE_DIM)
    torch.testing.assert_close(actual, _reference_joint(x, 1.7), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, dtype=torch.double))
    assert bool((actual >= 0).all())


def test_soft_joint_has_correct_marginals_and_differentiable_inputs():
    x = torch.randn(2, 2, 3, 3, 1, dtype=torch.double, requires_grad=True)
    distribution = soft_plaquette_features(x, 1., chunk_size=13).reshape(2, 5, 5, 5, 5)
    probability = torch.softmax(-2. * (x[..., 0, None]
                                      - torch.arange(-2, 3, dtype=x.dtype)).square(), dim=-1)
    single = probability.mean((1, 2, 3))
    for keep in range(4):
        reduce = tuple(axis + 1 for axis in range(4) if axis != keep)
        torch.testing.assert_close(distribution.sum(reduce), single)
    gradient, = torch.autograd.grad(distribution.square().sum(), x)
    assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0)


@pytest.mark.parametrize("scale", [1., .1])
def test_soft_joint_does_not_clip_physical_amplitudes_or_overflow_finite_tails(scale):
    maximum = torch.finfo(torch.double).max
    x = torch.tensor([maximum, -maximum, .1, -.3], dtype=torch.double).reshape(1, 1, 2, 2, 1)
    distribution = soft_plaquette_features(x, scale, chunk_size=3)
    assert bool(torch.isfinite(distribution).all())
    torch.testing.assert_close(distribution.sum(-1), torch.ones(1, dtype=torch.double))
    assert x.flatten()[0].item() == maximum


def test_joint_descriptor_removes_old_four_point_parity_collision():
    x = _parity_fields()
    old = soft_local_features(x, 1.)
    torch.testing.assert_close(old[0], old[1], rtol=0, atol=1e-15)
    joint = soft_plaquette_features(x, 1., chunk_size=16)
    assert (joint[0] - joint[1]).abs().sum().item() > .5
    encoder = RawPatchTheoryEncoder(1., hidden=8, local_features=5, patch_chunk_size=31)
    encoded = encoder(x)
    # Raw local Q no longer factors through the colliding old descriptor.
    assert (encoded[0] - encoded[1]).abs().max().item() > 1e-9


@pytest.mark.parametrize("chunk_size", [1, 11, 4096])
def test_raw_encoder_exact_discrete_architecture_and_patch_order(chunk_size):
    discrete = EnsembleTheoryEncoder(LatticeGeometry(2, 3, 3), 1, 2, 8, 5).double()
    continuous = RawPatchTheoryEncoder(1., hidden=8, theory_dim=2,
                                      local_features=5, patch_chunk_size=chunk_size)
    continuous.local_mlp.load_state_dict(discrete.local_mlp.state_dict())
    continuous.head.load_state_dict(discrete.head.state_dict())
    x = torch.randn(2, 3, 4, 4, 1, dtype=torch.double)
    actual, expected = continuous(x), discrete(x)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    actual_parameters = list(continuous.local_mlp.parameters()) + list(continuous.head.parameters())
    expected_parameters = list(discrete.parameters())
    actual_gradient = torch.autograd.grad(actual.square().sum(), actual_parameters)
    expected_gradient = torch.autograd.grad(expected.square().sum(), expected_parameters)
    for gradient, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(gradient, reference, rtol=1e-11, atol=1e-13)


def test_raw_encoder_uses_one_frozen_scale_and_preserves_input_derivative():
    encoder = RawPatchTheoryEncoder(2., hidden=7, theory_dim=2, patch_chunk_size=7)
    unit = RawPatchTheoryEncoder(1., hidden=7, theory_dim=2, patch_chunk_size=13)
    unit.local_mlp.load_state_dict(encoder.local_mlp.state_dict())
    unit.head.load_state_dict(encoder.head.state_dict())
    x = torch.randn(1, 2, 3, 3, 1, dtype=torch.double, requires_grad=True)
    actual = encoder(x)
    expected = unit(x / 2.)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    gradient, = torch.autograd.grad(actual.sum(), x)
    assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0)
    assert "scale" in dict(encoder.named_buffers())
    assert "scale" not in dict(encoder.named_parameters())


def test_encoder_eval_and_checkpoint_training_agree_and_bound_patch_calls():
    encoder = RawPatchTheoryEncoder(1., hidden=8, patch_chunk_size=13)
    x = torch.randn(2, 3, 4, 4, 1, dtype=torch.double)
    sizes = []
    handle = encoder.local_mlp[0].register_forward_pre_hook(
        lambda module, args: sizes.append(args[0].numel() // 9))
    train_value = encoder(x)
    torch.autograd.grad(train_value.sum(), list(encoder.local_mlp.parameters()))
    with torch.no_grad():
        eval_value = encoder(x)
    handle.remove()
    torch.testing.assert_close(train_value, eval_value)
    assert max(sizes) <= 13


def test_decoder_is_raw_affine_with_post_step_column_projection_only():
    encoder = RawPatchTheoryEncoder(1., hidden=8, theory_dim=2)
    torch.testing.assert_close(encoder.decoder.weight.norm(dim=0), torch.ones(2, dtype=torch.double))
    with torch.no_grad():
        encoder.decoder.weight.mul_(3.)
    g = torch.randn(3, 2, dtype=torch.double)
    torch.testing.assert_close(encoder.decode(g), encoder.decoder(g))
    torch.testing.assert_close(encoder.decoder.weight.norm(dim=0), torch.full((2,), 3., dtype=torch.double))
    gradient, = torch.autograd.grad(encoder.decode(g).sum(), encoder.decoder.weight)
    torch.testing.assert_close(gradient, g.sum(0).expand(FEATURE_DIM, -1))
    encoder.normalize_decoder_()
    torch.testing.assert_close(encoder.decoder.weight.norm(dim=0), torch.ones(2, dtype=torch.double))
    # No rank restoration is imposed on zero columns.
    with torch.no_grad():
        encoder.decoder.weight[:, 1].zero_()
    encoder.normalize_decoder_()
    assert encoder.decoder.weight[:, 1].count_nonzero().item() == 0


def test_new_objective_routes_each_component_exclusively_and_has_all_start_identity():
    model = AlignedPhi4QCR(1., hidden=6, theory_dim=2, local_features=4, patch_chunk_size=128)
    x = torch.randn(2, 4, 9, 9, 1, dtype=torch.double)
    q_loss, c_loss, r_loss, metrics = aligned_objectives(
        model, x, generator=torch.Generator().manual_seed(37), information_pairs=4)
    for name, loss in (("Q", q_loss), ("C", c_loss), ("R", r_loss)):
        for other in ("Q", "C", "R"):
            gradient = torch.autograd.grad(loss, list(getattr(model, other).parameters()),
                                           retain_graph=True, allow_unused=True)
            if name == other:
                assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradient)
                assert any(bool(value.abs().max() > 0) for value in gradient)
            else:
                assert all(value is None for value in gradient)
    torch.testing.assert_close(q_loss, q_loss.new_tensor(
        10. * metrics["q_reconstruction"] + 20. * metrics["q_subset_micro"]
        + 5. * metrics["q_subset_coarse"]))
    levels = model.coarse_grain(x, generator=torch.Generator().manual_seed(37))
    direct = torch.stack([(.5 * (model.Q(level[:, :2].detach())
                                  + model.Q(level[:, 2:].detach()))).detach()
                          for level in levels], dim=1)
    reference, _, _ = all_start_anchor_closure_loss(model, direct)
    torch.testing.assert_close(r_loss, 10. * reference)
    actual_gradient = torch.autograd.grad(r_loss, list(model.R.parameters()))
    expected_gradient = torch.autograd.grad(10. * reference, list(model.R.parameters()))
    for actual, expected in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, expected)
    assert metrics["r_total"] == r_loss.item()


def test_continuous_c_objective_unchanged_in_value_and_gradient():
    old = Phi4QCR(1.3, hidden=6, theory_dim=2)
    new = AlignedPhi4QCR(1.3, hidden=6, theory_dim=2, patch_chunk_size=128)
    new.C.load_state_dict(old.C.state_dict())
    x = torch.randn(1, 4, 9, 9, 1, dtype=torch.double)
    _, old_loss, _, old_metrics = continuous_objectives(
        old, x, generator=torch.Generator().manual_seed(7), information_pairs=5)
    _, new_loss, _, new_metrics = aligned_objectives(
        new, x, generator=torch.Generator().manual_seed(7), information_pairs=5)
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    for old_name, new_name in (("information_proxy", "c_information"),
                               ("variance_floor", "c_variance"),
                               ("uniform_amplitude", "c_uniform")):
        assert old_metrics[old_name] == new_metrics[new_name]
    old_gradient = torch.autograd.grad(old_loss, list(old.C.parameters()))
    new_gradient = torch.autograd.grad(new_loss, list(new.C.parameters()))
    for actual, expected in zip(new_gradient, old_gradient):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": float("nan")},
                                    {"scale": 1., "theory_dim": 0},
                                    {"scale": 1., "hidden": True},
                                    {"scale": 1., "local_features": 0},
                                    {"scale": 1., "patch_chunk_size": 0}])
def test_invalid_architecture_rejected(kwargs):
    with pytest.raises(ValueError):
        AlignedPhi4QCR(**kwargs)


def test_objective_rejects_odd_subsets_and_incompatible_sizes():
    model = AlignedPhi4QCR(1., hidden=4)
    with pytest.raises(ValueError, match="even"):
        aligned_objectives(model, torch.randn(1, 3, 9, 9, 1, dtype=torch.double))
    with pytest.raises(ValueError, match="divisible by 9"):
        model.coarse_grain(torch.randn(1, 4, 12, 12, 1, dtype=torch.double))
