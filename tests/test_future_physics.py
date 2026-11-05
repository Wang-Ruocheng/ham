"""Synthetic contract tests; no experiment data, labels or checkpoints."""

import copy
import math

import pytest
import torch
from torch import nn

from recurrent_rg.future_physics import (
    FuturePhysicsPredictor,
    binary2d_future_observables,
    fit_future_normalizer,
    paired_future_physics_loss,
    validate_independent_pairs,
)


def predictor(*, paths=((0, 1), (1, 2), (0, 2)), observable_count=5):
    return FuturePhysicsPredictor(
        1, paths=paths, target_mean=torch.zeros(len(paths), observable_count, dtype=torch.double),
        target_std=torch.ones(len(paths), observable_count, dtype=torch.double), hidden=4,
    )


def spins(shape=(4, 3, 12, 12, 1)):
    return (2 * torch.randint(0, 2, shape) - 1).double()


def slow_observables(level):
    """Independent complex-exponential reference, one configuration at a time."""
    result = []
    for ensemble in level[..., 0]:
        measured = []
        for configuration in ensemble:
            h, w = configuration.shape
            row = torch.arange(h, dtype=torch.double)[:, None]
            column = torch.arange(w, dtype=torch.double)[None, :]
            values = []
            for r in (h//6, h//3):
                total = 0.
                for i in range(h):
                    for j in range(w):
                        total += float(configuration[i, j] * (
                            configuration[(i+r) % h, j] + configuration[i, (j+r) % w]))
                values.append(total / (2*h*w))
            for n in (1, 2):
                qx = torch.exp(2j*torch.pi*n*row/h)
                qy = torch.exp(2j*torch.pi*n*column/w)
                values.append(float(((configuration*qx).sum().abs().square()
                                     + (configuration*qy).sum().abs().square())/(2*(h*w)**2)))
            value = sum((configuration*torch.exp(2j*torch.pi*(row/h+sign*column/w))).sum().abs().square()
                        for sign in (1, -1)) / (2*(h*w)**2)
            values.append(float(value))
            measured.append(values)
        result.append(torch.tensor(measured, dtype=torch.double).mean(0))
    return torch.stack(result)


def test_binary_observables_match_independent_reference_and_channel_forms():
    torch.manual_seed(101)
    level = spins((2, 3, 12, 12, 1)).requires_grad_()
    actual = binary2d_future_observables(level)
    torch.testing.assert_close(actual, slow_observables(level.detach()), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(actual, binary2d_future_observables(level[..., 0]))
    assert actual.shape == (2, 5) and not actual.requires_grad


def test_observables_uniform_spin_flip_translation():
    constant = torch.ones(2, 3, 12, 12, 1, dtype=torch.double)
    actual = binary2d_future_observables(constant)
    torch.testing.assert_close(actual[:, :2], torch.ones(2, 2, dtype=torch.double))
    assert float(actual[:, 2:].abs().max()) < 1e-28
    level = spins()
    actual = binary2d_future_observables(level)
    torch.testing.assert_close(actual, binary2d_future_observables(-level))
    torch.testing.assert_close(actual, binary2d_future_observables(torch.roll(level, (2, 3), (2, 3))))


@pytest.mark.parametrize("side", [12, 36, 108])
def test_fractional_distances_remain_distinct_at_all_b3_scales(side):
    # A half-positive/half-negative stripe has distinct correlations at
    # r=L/6 and L/3: after averaging axes, 2/3 and 1/3 respectively.
    level = torch.ones(1, 1, side, side, 1, dtype=torch.double)
    level[:, :, :side//2] = -1
    actual = binary2d_future_observables(level)
    torch.testing.assert_close(actual[0, :2], torch.tensor([2/3, 1/3], dtype=torch.double))


@pytest.mark.parametrize("bad", [
    torch.ones(2, 3, 12, 1),  # 1D, not a supported adapter
    torch.ones(2, 3, 8, 8, 1),
    torch.ones(2, 3, 14, 14, 1),
    torch.ones(2, 3, 12, 18, 1),
    torch.zeros(2, 3, 12, 12, 1),
    torch.full((2, 3, 12, 12, 1), float("nan")),
    torch.ones(2, 3, 12, 12, 2),
])
def test_observables_reject_unsupported_or_nonbinary_data(bad):
    with pytest.raises(ValueError):
        binary2d_future_observables(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_observables_accept_only_hard_st_roundoff_and_canonicalize(dtype):
    hard = spins((2, 3, 12, 12, 1)).to(dtype=dtype)
    eps = torch.finfo(dtype).eps
    perturbed = (hard * (1 + 4*eps)).requires_grad_()
    actual = binary2d_future_observables(perturbed)
    torch.testing.assert_close(actual, binary2d_future_observables(hard), atol=0, rtol=0)
    assert not actual.requires_grad
    for soft in (hard*.99, hard*(1 + 16*eps)):
        with pytest.raises(ValueError, match="hard"):
            binary2d_future_observables(soft)


def test_normalizer_train_only_frozen_population_moments_and_explicit_floor():
    targets = torch.tensor([[[1., 2.]], [[3., 2.]]], dtype=torch.double, requires_grad=True)
    mean, std = fit_future_normalizer(targets, split="TRAIN")
    torch.testing.assert_close(mean, torch.tensor([[2., 2.]], dtype=torch.double))
    torch.testing.assert_close(std, torch.tensor([[1., 0.]], dtype=torch.double))
    assert not mean.requires_grad and not std.requires_grad
    module = FuturePhysicsPredictor(1, paths=((0, 1),), target_mean=mean,
                                    target_std=std, std_floor=.01)
    torch.testing.assert_close(module.target_std, torch.tensor([[1., .01]], dtype=torch.double))
    torch.testing.assert_close(module.target_std_unfloored, std)
    with torch.no_grad():
        targets.add_(100)
    torch.testing.assert_close(module.target_mean, torch.tensor([[2., 2.]], dtype=torch.double))
    with pytest.raises(ValueError, match="TRAIN"):
        fit_future_normalizer(targets, split="TEST")


def test_json_moments_preserve_float64_and_invalid_moments_fail():
    value = 0.1234567890123456
    module = FuturePhysicsPredictor(1, paths=((0, 1),), target_mean=[[value]], target_std=[[.3]])
    assert module.target_mean.dtype == torch.float64 and module.target_mean.item() == value
    for mean, std in (([[float("nan")]], [[1.]]), ([[0.]], [[-1.]]), ([[0.]], [[float("inf")]])):
        with pytest.raises(ValueError):
            FuturePhysicsPredictor(1, paths=((0, 1),), target_mean=mean, target_std=std)


class MeanEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(1, 1).double()
        self.calls = 0

    def forward(self, level):
        self.calls += 1
        return self.readout(level.flatten(1).mean(-1, keepdim=True))


def test_loss_updates_only_q_and_auxiliary_not_c_r_or_targets():
    torch.manual_seed(87)
    c = nn.Parameter(torch.tensor(.3, dtype=torch.double))
    r = nn.Linear(1, 1).double()
    q, auxiliary = MeanEncoder(), predictor()
    # Mimic a hard stochastic C with a straight-through graph, without drawing C.
    levels = [spins() + c-c.detach() for _ in range(3)]
    initial_rng = torch.get_rng_state().clone()
    loss, terms = paired_future_physics_loss(auxiliary, q, levels, [1, 0, 3, 2],
                                           [0, 0, 1, 1], [0, 1, 0, 1], weight=2.)
    loss.backward()
    assert q.calls == 2  # sources 0 and 1 only; no redundant Q or C forward
    assert c.grad is None and all(p.grad is None for p in r.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in q.parameters())
    assert all(p.grad is not None for p in auxiliary.parameters())
    assert not auxiliary.target_mean.requires_grad and not auxiliary.target_std.requires_grad
    torch.testing.assert_close(loss, 2 * terms["future_physics"])
    torch.testing.assert_close(terms["future_physics_nrmse"].square(), terms["future_physics"])
    assert torch.equal(initial_rng, torch.get_rng_state())


def test_unequal_paths_and_partner_targets_have_equal_path_coordinate_weight():
    paths = ((0, 2), (1, 3))
    auxiliary = FuturePhysicsPredictor(1, paths=paths, hidden=3,
                                      target_mean=[[1.], [2.]], target_std=[[2.], [4.]])
    with torch.no_grad():
        for head, bias in zip(auxiliary.heads, (3., -1.)):
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(bias)
    levels = [torch.tensor([[i+1.], [i+3.], [i+6.], [i+10.]], dtype=torch.double)
              for i in range(4)]
    # This custom observable adapter demonstrates that the predictor/loss is not spin-specific.
    partners = [1, 0, 3, 2]
    loss, terms = paired_future_physics_loss(auxiliary, MeanEncoder(), levels, partners,
                                           [0, 0, 1, 1], [0, 1, 0, 1], observable_fn=lambda x: x)
    expected = torch.stack([
        ((3 - (levels[2][partners, 0]-1)/2)**2).mean(),
        ((-1 - (levels[3][partners, 0]-2)/4)**2).mean(),
    ])
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(terms["future_physics_path_0_2"], expected[0])
    torch.testing.assert_close(terms["future_physics_path_1_3"], expected[1])
    physical_expected = torch.stack([
        ((3*2+1 - levels[2][partners, 0])**2).mean(),
        ((-1*4+2 - levels[3][partners, 0])**2).mean(),
    ])
    torch.testing.assert_close(terms["future_physics_raw_path_0_2"], physical_expected[0])
    torch.testing.assert_close(terms["future_physics_raw_path_1_3"], physical_expected[1])
    torch.testing.assert_close(terms["future_physics_raw_mse"], physical_expected.mean())
    torch.testing.assert_close(terms["future_physics_raw_rmse"], physical_expected.mean().sqrt())
    assert not terms["future_physics_raw_mse"].requires_grad
    assert not terms["future_physics_raw_path_0_2"].requires_grad


@pytest.mark.parametrize("partners,conditions,chains", [
    ([0, 1], [0, 0], [0, 1]),
    ([1, 0], [0, 0], [4, 4]),
    ([1, 0], [0, 1], [0, 1]),
    ([2, 0], [0, 0], [0, 1]),
    ([1., 0.], [0, 0], [0, 1]),
    ([1, 0], [0], [0, 1]),
])
def test_independent_pair_metadata_rejects_invalid_pairs(partners, conditions, chains):
    with pytest.raises(ValueError):
        validate_independent_pairs(partners, conditions, chains, 2)


def test_pair_metadata_allows_arbitrary_condition_ids_but_never_passes_them_to_heads():
    assert validate_independent_pairs(torch.tensor([1, 0]), ["condition-a"]*2, ["rep0", "rep1"], 2) == [1, 0]


def test_weight_zero_skips_every_forward_pairing_validation_and_rng_draw():
    auxiliary = predictor()
    q = MeanEncoder()
    before = torch.get_rng_state().clone()

    def forbidden(*_, **__):
        raise AssertionError("Zero weight must skip the observable adapter")

    q.forward_many = forbidden
    loss, terms = paired_future_physics_loss(auxiliary, q, [], [], [], [], weight=0, observable_fn=forbidden,
                                           encoding_batch_chunk=4, validated_binary_inputs=True)
    loss.backward()
    assert loss.item() == 0 and q.calls == 0 and all(p.grad is None for p in q.parameters())
    assert torch.equal(before, torch.get_rng_state())
    assert terms["future_physics"].item() == 0


def test_auxiliary_checkpoint_roundtrip_preserves_normalizer_heads_and_path_contract():
    torch.manual_seed(88)
    auxiliary = predictor(paths=((0, 2), (1, 3)))
    values = [torch.randn(3, 1, dtype=torch.double) for _ in range(4)]
    state = copy.deepcopy(auxiliary.state_dict())
    restored = predictor(paths=((0, 2), (1, 3)))
    restored.load_state_dict(state)
    torch.testing.assert_close(auxiliary(values), restored(values), atol=0, rtol=0)
    torch.testing.assert_close(auxiliary.physical_predictions(values), restored.physical_predictions(values), atol=0, rtol=0)
    incompatible = predictor(paths=((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="contract"):
        incompatible.load_state_dict(state)


def test_loss_rejects_nonfinite_values_and_bad_weights():
    auxiliary, q = predictor(), MeanEncoder()
    for weight in (-1., math.inf, math.nan):
        with pytest.raises(ValueError):
            paired_future_physics_loss(auxiliary, q, [], [], [], [], weight=weight)
    with torch.no_grad():
        auxiliary.heads[0][-1].bias.fill_(math.nan)
    with pytest.raises(FloatingPointError):
        paired_future_physics_loss(auxiliary, q, [spins() for _ in range(3)],
                                   [1, 0, 3, 2], [0, 0, 1, 1], [0, 1, 0, 1])


def test_optional_trajectory_sink_preserves_default_qcr_loss_and_rng():
    from recurrent_rg.checkpoints import build_model
    from recurrent_rg.config import ModelConfig
    from recurrent_rg.data.statistics import plaquette_distribution
    from recurrent_rg.losses import qcr_objective

    torch.set_num_threads(1)
    torch.manual_seed(79)
    config = ModelConfig(dimension=2, block_factor=3, hidden=4, theory_dim=1,
                         kernel_sampling="bernoulli", affine_reconstruction=True)
    model = build_model(config, 1)
    model.detach_kernel_representation = True
    batch = spins((2, 4, 9, 9, 1))
    state = torch.get_rng_state().clone()
    first_loss, first_terms = qcr_objective(model, batch, pattern_fn=plaquette_distribution)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    captured = []
    second_loss, second_terms = qcr_objective(model, batch, pattern_fn=plaquette_distribution, levels_sink=captured)
    assert len(captured) == 2 and captured[0] is batch
    torch.testing.assert_close(first_loss, second_loss, atol=0, rtol=0)
    assert set(first_terms) == set(second_terms)
    for key in first_terms:
        torch.testing.assert_close(first_terms[key], second_terms[key], atol=0, rtol=0)
    assert torch.equal(after, torch.get_rng_state())


def test_chunked_shared_q_preserves_values_gradients_rng_and_full_chain_layout(monkeypatch):
    from recurrent_rg.checkpoints import build_model
    from recurrent_rg.config import ModelConfig

    torch.manual_seed(903)
    model = build_model(ModelConfig(dimension=2, block_factor=3, hidden=4, theory_dim=1,
                                   kernel_sampling="bernoulli", affine_reconstruction=True), 2)
    q = model.theory_encoder
    chunked_q = copy.deepcopy(q)
    # Deliberately reverse path-source order; restore chunks by source, not path.
    auxiliary = predictor(paths=((1, 2), (0, 2), (0, 1)))
    chunked_auxiliary = copy.deepcopy(auxiliary)
    c = nn.Parameter(torch.zeros((), dtype=torch.double))
    levels = [spins((6, 64, 12, 12, 1)) + c-c.detach() for _ in range(3)]
    partners, conditions, chains = [1, 0, 3, 2, 5, 4], [0, 0, 1, 1, 2, 2], list(range(6))
    calls = []
    original_many = chunked_q.forward_many

    def spy(chunks, *, validated_binary_inputs=False):
        calls.append((chunks, validated_binary_inputs))
        return original_many(chunks, validated_binary_inputs=validated_binary_inputs)

    def no_many(*_, **__):
        pytest.fail("Default execution must retain the original per-source forward layout")

    monkeypatch.setattr(q, "forward_many", no_many)
    monkeypatch.setattr(chunked_q, "forward_many", spy)
    rng = torch.get_rng_state().clone()
    expected, expected_terms = paired_future_physics_loss(auxiliary, q, levels, partners, conditions, chains)
    unrelated_parameters = [c] + list(model.theory_rg.parameters())
    expected_parameters = list(q.parameters()) + list(auxiliary.parameters()) + unrelated_parameters
    expected_grads = torch.autograd.grad(expected, expected_parameters, allow_unused=True)
    actual, actual_terms = paired_future_physics_loss(chunked_auxiliary, chunked_q, levels, partners,
        conditions, chains, encoding_batch_chunk=4, validated_binary_inputs=True)
    actual_parameters = list(chunked_q.parameters()) + list(chunked_auxiliary.parameters()) + unrelated_parameters
    actual_grads = torch.autograd.grad(actual, actual_parameters, allow_unused=True)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-11)
    for name in expected_terms:
        torch.testing.assert_close(actual_terms[name], expected_terms[name], atol=1e-13, rtol=1e-11)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-10)
    assert all(gradient is None for gradient in expected_grads[-len(unrelated_parameters):])
    assert all(gradient is None for gradient in actual_grads[-len(unrelated_parameters):])
    assert torch.equal(rng, torch.get_rng_state())
    assert len(calls) == 1 and calls[0][1] is True
    chunks = calls[0][0]
    assert [tuple(chunk.shape) for chunk in chunks] == [(4, 64, 12, 12, 1), (2, 64, 12, 12, 1)] * 2
    assert all(not chunk.requires_grad for chunk in chunks)
    torch.testing.assert_close(torch.cat(chunks[:2]), levels[0].detach(), atol=0, rtol=0)
    torch.testing.assert_close(torch.cat(chunks[2:]), levels[1].detach(), atol=0, rtol=0)


def test_chunked_generic_encoder_fallback_preserves_values_gradients_and_rng():
    torch.manual_seed(904)
    q, auxiliary = MeanEncoder(), predictor()
    other_q, other_auxiliary = copy.deepcopy(q), copy.deepcopy(auxiliary)
    levels = [spins((6, 64, 12, 12, 1)) for _ in range(3)]
    layout = []
    original_forward = other_q.forward

    def forward(level):
        layout.append(tuple(level.shape))
        return original_forward(level)

    other_q.forward = forward
    args = (levels, [1, 0, 3, 2, 5, 4], [0, 0, 1, 1, 2, 2], list(range(6)))
    rng = torch.get_rng_state().clone()
    expected, _ = paired_future_physics_loss(auxiliary, q, *args)
    actual, _ = paired_future_physics_loss(other_auxiliary, other_q, *args,
                                         encoding_batch_chunk=4, validated_binary_inputs=True)
    expected.backward(); actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-11)
    for a, b in zip(list(other_q.parameters()) + list(other_auxiliary.parameters()),
                    list(q.parameters()) + list(auxiliary.parameters())):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-10)
    assert q.calls == 2 and other_q.calls == 4
    assert layout == [(4, 64, 12, 12, 1), (2, 64, 12, 12, 1)] * 2
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_chunk_execution_rejects_invalid_sizes(chunk):
    with pytest.raises(ValueError, match="batch chunk"):
        paired_future_physics_loss(predictor(), MeanEncoder(), [], [], [], [], encoding_batch_chunk=chunk)
