"""Implementation-only shared lookup must preserve Q/C/R objectives and VJPs."""
import copy

import pytest
import torch

from experiments.rsmi_separate_qcr import independent_component_gradients
from recurrent_rg.binary_pooling import binary_mean, binary_local, binary_pattern_table, _cached_pattern_table
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import qcr_objective


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA'))]


def model(device='cpu', *, independent=True):
    torch.set_num_threads(1)
    torch.manual_seed(18)
    result = build_model(ModelConfig(dimension=2, block_factor=3, hidden=5,
        theory_dim=2, binary_lookup=True, affine_reconstruction=True,
        kernel_sampling='bernoulli'), 2).to(device)
    result.detach_kernel_representation = independent
    return result


def binary(shape, device='cpu'):
    # Keep test fixtures independent of the C sampling stream.
    generator = torch.Generator().manual_seed(sum(shape)+17)
    return (2*torch.randint(2, shape, generator=generator)-1).to(device=device, dtype=torch.float64)


@pytest.mark.parametrize('device', DEVICES)
def test_forward_many_matches_values_parameter_gradients_and_reuses_only_one_table(device):
    reference = model(device).theory_encoder
    shared = copy.deepcopy(reference)
    levels = [binary((3, 4, side, side, 1), device) for side in (9, 3, 1)]
    halves = [level[:, :2] for level in levels]+[level[:, 2:] for level in levels]
    count = []
    hook = shared.local_mlp.register_forward_hook(lambda *args: count.append(1))
    a = [reference(level) for level in halves]
    b = shared.forward_many(halves)
    assert len(count) == 1
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    for outputs in (a, b):
        sum((k+1)*value.square().sum() for k, value in enumerate(outputs)).backward()
    assert len(count) == 1  # Standard autograd reuses, rather than reevaluates, the shared table.
    for left, right in zip(reference.parameters(), shared.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is None:  # Attached reconstruction decoder is not part of this encoder-only loss.
            continue
        torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)
    with torch.no_grad():
        shared.local_mlp[0].weight.add_(.13)
    updated = shared.forward_many(halves)
    assert len(count) == 2
    assert any(not torch.equal(left, right) for left, right in zip(b, updated))
    hook.remove()


@pytest.mark.parametrize('device', DEVICES)
def test_trusted_binary_mean_retains_input_vjp_and_parameter_gradients(device):
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                    torch.nn.Linear(4, 2)).double().to(device)
    trusted = copy.deepcopy(reference)
    x = binary((2, 3, 5, 3), device).requires_grad_()
    y = x.detach().clone().requires_grad_()
    a = binary_mean(x, reference)
    b = binary_mean(y, trusted, validated_binary_inputs=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(reference.parameters(), trusted.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('device', DEVICES)
def test_trusted_binary_local_retains_input_vjp_and_parameter_gradients(device):
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                    torch.nn.Linear(4, 1)).double().to(device)
    trusted = copy.deepcopy(reference)
    x = binary((2, 3, 5, 3), device).requires_grad_()
    y = x.detach().clone().requires_grad_()
    a = binary_local(x, reference, reference.parameters())
    b = binary_local(y, trusted, trusted.parameters(), validated_binary_inputs=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(reference.parameters(), trusted.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('device', DEVICES)
def test_qcr_shared_path_preserves_loss_all_parameter_gradients_and_c_rng(device):
    reference = model(device)
    shared = copy.deepcopy(reference)
    shared.share_binary_q_tables = True
    shared.fast_validated_statistics = True
    shared.rg_kernel.validated_binary_inputs = True
    x = binary((3, 8, 9, 9, 1), device)
    results = []
    for current in (reference, shared):
        torch.manual_seed(618)
        total, terms = qcr_objective(current, x, pattern_fn=plaquette_distribution,
            kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
            kernel_sign_weight=1.)
        closure = 5*(terms['commutator']+terms['rollout'])
        c_objective = 2*terms['kernel_noncollapse']+terms['kernel_sign']
        # The assignment agrees with selective Q, C and R objectives, not joint loss gradients.
        q = list(current.theory_encoder.parameters())
        c = list(current.rg_kernel.parameters())
        r = list(current.theory_rg.parameters())
        expected = [torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
                    for objective, parameters in ((total-closure, q), (c_objective, c), (closure, r))]
        independent_component_gradients(total, closure, c_objective, q, c, r)
        for parameters, gradients in zip((q, c, r), expected):
            for parameter, gradient in zip(parameters, gradients):
                torch.testing.assert_close(parameter.grad, gradient, atol=1e-11, rtol=1e-10)
        rng = torch.cuda.get_rng_state(device) if device != 'cpu' else torch.get_rng_state()
        results.append((total.detach(), {key: value.detach() for key, value in terms.items()},
                        {key: value.grad.clone() for key, value in current.named_parameters()
                         if value.grad is not None}, rng.clone()))
    a, b = results
    torch.testing.assert_close(a[0], b[0], atol=1e-12, rtol=1e-12)
    assert a[1].keys() == b[1].keys() and a[2].keys() == b[2].keys()
    for left, right in zip(a[1].values(), b[1].values()):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    for key in a[2]:
        torch.testing.assert_close(a[2][key], b[2][key], atol=1e-10, rtol=1e-9)
    assert torch.equal(a[3], b[3])


def test_forward_many_falls_back_for_input_vjp_and_rejects_invalid_untrusted_inputs():
    reference = model().theory_encoder
    shared = copy.deepcopy(reference)
    x = binary((2, 4, 3, 3, 1)).requires_grad_()
    y = x.detach().clone().requires_grad_()
    a = reference(x)
    b = shared.forward_many([y], validated_binary_inputs=True)[0]
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
    invalid = torch.zeros_like(x.detach())
    with pytest.raises(ValueError, match='binary patches'):
        shared.forward_many([invalid])
    with pytest.raises(ValueError, match='binary patches'):
        shared.forward_many([invalid.requires_grad_()], validated_binary_inputs=True)
    with pytest.raises(ValueError, match='binary patches'):
        binary_local(torch.zeros(2, 3, dtype=torch.double), lambda p: p[:, :1], ())
    assert shared.forward_many([]) == []


def test_forward_many_preserves_nonlookup_historical_path():
    encoder = model().theory_encoder
    encoder.binary_lookup = False
    x = torch.randn(2, 4, 3, 3, 1, dtype=torch.double)
    torch.testing.assert_close(encoder(x), encoder.forward_many([x])[0], atol=0, rtol=0)


def test_unflagged_or_nonindependent_qcr_does_not_use_shared_q(monkeypatch):
    for independent in (False, True):
        current = model(independent=independent)
        if not independent:
            current.share_binary_q_tables = True
        monkeypatch.setattr(current.theory_encoder, 'forward_many',
            lambda *args, **kwargs: pytest.fail('Historical Q path must remain unshared'))
        qcr_objective(current, binary((2, 4, 9, 9, 1)), pattern_fn=plaquette_distribution,
                      kernel_noncollapse_mode='variance_floor_ensemble')


def test_pattern_constant_cache_is_bounded_and_dtype_specific():
    reference = torch.empty(1, dtype=torch.double)
    _, a = binary_pattern_table(3, reference, cached=True)
    _, b = binary_pattern_table(3, reference, cached=True)
    _, c = binary_pattern_table(3, reference.float(), cached=True)
    assert a.data_ptr() == b.data_ptr()
    assert c.dtype == torch.float32 and a.dtype == torch.float64
    assert not a.requires_grad and _cached_pattern_table.cache_info().maxsize == 16
