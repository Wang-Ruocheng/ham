import copy

import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.empirical_mi import (
    categorical_local_pair_mi_statistics,
    lattice_mi_statistics,
)
from recurrent_rg.rsmi import kernel_probabilities


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA runtime is unavailable")),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("states,sparse_conditionals", [(2, False), (2, True), (3, True)])
def test_empirical_statistics_device_transfer_preserves_values_and_gradients(
        device, states, sparse_conditionals):
    generator = torch.Generator().manual_seed(741)
    colors = torch.randint(states, (2, 5, 7, 7), generator=generator)
    fields = ((2 * colors - 1).unsqueeze(-1).double() if states == 2 else
              torch.nn.functional.one_hot(colors, states).double())
    source = lattice_mi_statistics(fields, states=states,
                                   sparse_conditionals=sparse_conditionals)
    transferred = source.to(device)
    assert transferred is not source
    assert source.marginal.device.type == "cpu"
    assert transferred.marginal.device.type == device
    assert transferred.weights.is_sparse
    assert transferred.weights.is_coalesced()
    assert transferred.conditional.is_sparse == sparse_conditionals
    assert transferred.diagnostics == source.diagnostics
    torch.testing.assert_close(transferred.marginal.cpu(), source.marginal)
    torch.testing.assert_close(transferred.weights.cpu().to_dense(), source.weights.to_dense())
    torch.testing.assert_close(transferred.conditional.cpu().to_dense(),
                               source.conditional.to_dense())

    logits = torch.randn(source.marginal.shape[1], states, generator=generator,
                         dtype=torch.double, requires_grad=True)
    target_logits = logits.detach().to(device).clone().requires_grad_()
    reference = source.information(logits.softmax(-1))
    actual = transferred.information(target_logits.softmax(-1))
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-11, atol=1e-12)
    reference_gradient = torch.autograd.grad(reference.sum(), logits)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), target_logits)[0]
    torch.testing.assert_close(actual_gradient.cpu(), reference_gradient,
                               rtol=1e-10, atol=1e-12)


def test_statistics_dtype_transfer_keeps_sparse_layout():
    generator = torch.Generator().manual_seed(742)
    fields = (2 * torch.randint(2, (2, 3, 7, 7, 1), generator=generator) - 1).double()
    source = lattice_mi_statistics(fields, states=2, sparse_conditionals=True)
    transferred = source.to("cpu", dtype=torch.float32)
    assert source.marginal.dtype == torch.double
    for tensor in (transferred.conditional, transferred.weights, transferred.marginal):
        assert tensor.dtype == torch.float32
    assert transferred.conditional.is_sparse and transferred.weights.is_sparse


@pytest.mark.parametrize("device", DEVICES)
def test_local_pair_statistics_transfer(device):
    generator = torch.Generator().manual_seed(743)
    colors = torch.randint(3, (2, 3, 7, 7), generator=generator)
    source = categorical_local_pair_mi_statistics(
        torch.nn.functional.one_hot(colors, 3).double())
    transferred = source.to(device)
    assert transferred is not source
    assert all(group.marginal.device.type == device for group in transferred.groups)
    assert all(group.marginal.device.type == "cpu" for group in source.groups)
    logits = torch.randn(81, 3, generator=generator, dtype=torch.double,
                         requires_grad=True)
    target_logits = logits.detach().to(device).clone().requires_grad_()
    reference = source.information(logits.softmax(-1))
    actual = transferred.information(target_logits.softmax(-1))
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), target_logits)[0].cpu(),
        torch.autograd.grad(reference.sum(), logits)[0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [2, 3])
def test_binary_kernel_lookup_device_values_and_gradients(device, block):
    torch.manual_seed(744)
    source = build_model(ModelConfig(dimension=2, block_factor=block), 2).rg_kernel
    target = copy.deepcopy(source).to(device)
    visible = (2 * torch.randint(2, (2, 7, block ** 2)) - 1).double()
    reference = kernel_probabilities(source, visible)
    actual = kernel_probabilities(target, visible.to(device))
    assert actual.device.type == device and actual.dtype == torch.double
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-11, atol=1e-12)
    weights = torch.linspace(.1, 1., reference.numel(), dtype=torch.double).reshape_as(reference)
    reference_gradients = torch.autograd.grad((reference * weights).sum(), source.parameters())
    actual_gradients = torch.autograd.grad((actual * weights.to(device)).sum(), target.parameters())
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient.cpu(), reference_gradient,
                                   rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_kernel_lookup_rejects_mismatched_devices():
    kernel = build_model(ModelConfig(dimension=2, block_factor=2), 2).rg_kernel.to("cuda")
    with pytest.raises(ValueError, match="same device"):
        kernel_probabilities(kernel, torch.ones(1, 2, 4, dtype=torch.double))
