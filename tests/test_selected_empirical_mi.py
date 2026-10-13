import copy
import random

import numpy as np
import pytest
import torch

from recurrent_rg.empirical_mi import EmpiricalBinaryMI, LocalPairMI, lattice_mi_statistics
from recurrent_rg.mi_cache import load_or_build_mi_statistics


DEVICES = [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA runtime is unavailable')),
]


def statistics(*, sparse):
    # Shared distributions, unused global rows, and ragged ensemble supports
    # expose incorrect global-to-local remapping and ensemble weight pooling.
    conditional = torch.tensor([
        [1., 0., 0., 0.], [.25, .25, .25, .25], [0., 0., .5, .5],
        [.3, .2, .1, .4], [0., 1., 0., 0.], [0., 0., 1., 0.],
    ], dtype=torch.double)
    weights = torch.tensor([
        [.25, .5, .25, 0., 0., 0.], [0., .2, 0., .8, 0., 0.],
        [0., 0., 0., 0., 1., 0.], [.4, 0., .6, 0., 0., 0.],
    ], dtype=torch.double).to_sparse().coalesce()
    marginal = weights.to_dense() @ conditional
    return EmpiricalBinaryMI(conditional.to_sparse().coalesce() if sparse else conditional,
                             weights, marginal, {'states': 2, 'buffer': 1})


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('outputs', [2, 3])
@pytest.mark.parametrize('ids', [[3, 1, 3, 0], [2], [3, 2, 1, 0], [], [-1, 1, -1]])
def test_selected_values_and_kernel_gradients_match_full(device, sparse, outputs, ids):
    source = statistics(sparse=sparse)
    assert source.prepare_selection() is source
    selected = source.to(device)
    assert selected._selection is not None
    generator = torch.Generator().manual_seed(519)
    logits = torch.randn(4, outputs, generator=generator, dtype=torch.double,
                         requires_grad=True, device='cpu').to(device).detach().requires_grad_()
    selected_logits = logits.detach().clone().requires_grad_()
    ids = torch.tensor(ids, dtype=torch.long)
    expected = selected.information(logits.softmax(-1))[ids.to(device)]
    actual = selected.information_for(selected_logits.softmax(-1), ids)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    # Distinct output multipliers check repeated rows and original row order.
    multipliers = torch.arange(1, len(ids) + 1, device=device, dtype=torch.double)
    expected_gradient = torch.autograd.grad((expected * multipliers).sum(), logits)[0]
    actual_gradient = torch.autograd.grad((actual * multipliers).sum(), selected_logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('states,sparse,block', [(2, False, 2), (2, True, 3), (3, True, 2)])
def test_selected_real_full_shell_histograms(device, states, sparse, block):
    generator = torch.Generator().manual_seed(520)
    length = 9 if block == 3 else 7
    colors = torch.randint(states, (4, 4, length, length), generator=generator)
    fields = ((2 * colors - 1).unsqueeze(-1).double() if states == 2 else
              torch.nn.functional.one_hot(colors, states).double())
    source = lattice_mi_statistics(fields, states=states, block=block,
                                   sparse_conditionals=sparse)
    hist = source.prepare_selection().to(device)
    logits = torch.randn(source.marginal.shape[1], states, generator=generator,
                         dtype=torch.double).to(device).requires_grad_()
    other = logits.detach().clone().requires_grad_()
    ids = torch.tensor([2, 0, 2, 3])
    expected = hist.information(logits.softmax(-1))[ids.to(device)]
    actual = hist.information_for(other.softmax(-1), ids)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), other)[0],
                               torch.autograd.grad(expected.sum(), logits)[0],
                               rtol=1e-10, atol=1e-12)
    assert hist.diagnostics['buffer'] == 1
    assert hist.diagnostics['visible_sites'] == block ** 2


def test_prepare_and_selected_evaluation_preserve_statistics_and_rng():
    source = statistics(sparse=True)
    original = copy.deepcopy(source)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = copy.deepcopy(np.random.get_state())
    python_rng = random.getstate()
    table = torch.tensor([[.2, .8], [.4, .6], [.7, .3], [.1, .9]], dtype=torch.double)
    source.prepare_selection()
    selection = source._selection
    for ids in ([3, 0, 3], [0, 1], [2], []):
        source.information_for(table, ids)
        assert source._selection is selection
    assert torch.equal(torch_rng, torch.get_rng_state())
    actual_numpy_rng = np.random.get_state()
    assert actual_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(actual_numpy_rng[1], numpy_rng[1])
    assert actual_numpy_rng[2:] == numpy_rng[2:]
    assert random.getstate() == python_rng
    assert source.diagnostics == original.diagnostics
    for name in ('weights', 'conditional', 'marginal'):
        torch.testing.assert_close(getattr(source, name).to_dense(),
                                   getattr(original, name).to_dense(), rtol=0, atol=0)


def test_dtype_transfer_preserves_cpu_support_index_and_source():
    source = statistics(sparse=True).prepare_selection()
    moved = source.to('cpu', dtype=torch.float32)
    assert moved._selection.weight_offsets is source._selection.weight_offsets
    assert moved._selection.weight_conditionals is source._selection.weight_conditionals
    assert moved._selection.weights.dtype == torch.float32
    assert moved._selection.conditional.dtype == torch.float32
    assert source._selection.weights.dtype == torch.double
    table = torch.tensor([[.2, .8], [.4, .6], [.7, .3], [.1, .9]], dtype=torch.float32)
    torch.testing.assert_close(moved.information_for(table, [3, 1, 3]),
                               moved.information(table)[[3, 1, 3]], rtol=1e-5, atol=1e-7)


def test_legacy_cache_payload_and_hits_derive_selection_without_schema_changes(tmp_path):
    key = dict(data_sha256='a' * 64, split='train', states=2, dimension=2,
               block=2, buffer=1, environment_offsets=None, flip=False,
               pooling='probabilistic', dtype='torch.float64', config_data={})
    original = statistics(sparse=True).prepare_selection()
    _, miss = load_or_build_mi_statistics(tmp_path, key, lambda: original)
    payload = torch.load(miss['path'], map_location='cpu', weights_only=True)
    assert set(payload['statistics']) == {'kind', 'conditional', 'weights', 'marginal', 'diagnostics'}
    loaded, hit = load_or_build_mi_statistics(
        tmp_path, key, lambda: pytest.fail('Existing histogram cache must be reused'))
    assert hit['hit'] and loaded._selection is None
    table = torch.tensor([[.2, .8], [.4, .6], [.7, .3], [.1, .9]], dtype=torch.double)
    torch.testing.assert_close(loaded.information_for(table, [3, 1, 3]),
                               original.information(table)[[3, 1, 3]], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('device', DEVICES)
def test_local_pair_selected_values_and_gradients(device):
    groups = LocalPairMI([statistics(sparse=False), statistics(sparse=True)],
                         {'aggregation': 'arithmetic mean'})
    moved = groups.prepare_selection().to(device)
    logits = torch.arange(12, dtype=torch.double, device=device).reshape(4, 3).requires_grad_()
    other = logits.detach().clone().requires_grad_()
    ids = torch.tensor([3, 1, 3])
    expected = moved.information(logits.softmax(-1))[ids.to(device)]
    actual = moved.information_for(other.softmax(-1), ids)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), other)[0],
                               torch.autograd.grad(expected.sum(), logits)[0],
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('ids,error', [([4], IndexError), ([-5], IndexError),
                                      ([[1, 2]], ValueError), ([.5], TypeError),
                                      (torch.tensor([1.]), TypeError)])
def test_invalid_selection_indices_fail_clearly(ids, error):
    source = statistics(sparse=False)
    with pytest.raises(error, match='Selected empirical MI'):
        source.information_for(torch.full((4, 2), .5, dtype=torch.double), ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime is unavailable')
def test_cuda_selection_uses_cpu_ids_without_host_tensor_reads(monkeypatch):
    source = statistics(sparse=True).prepare_selection().to('cuda')
    table = torch.full((4, 2), .5, dtype=torch.double, device='cuda', requires_grad=True)
    original_cpu = torch.Tensor.cpu
    original_numpy = torch.Tensor.numpy
    original_to = torch.Tensor.to
    index_copies = []
    def no_cuda_cpu(tensor, *args, **kwargs):
        assert tensor.device.type != 'cuda', 'Selected MI must not read CUDA tensors on CPU'
        return original_cpu(tensor, *args, **kwargs)
    def no_cuda_numpy(tensor, *args, **kwargs):
        assert tensor.device.type != 'cuda', 'Selected MI must not read CUDA tensors on CPU'
        return original_numpy(tensor, *args, **kwargs)
    def nonblocking_indices(tensor, *args, **kwargs):
        if tensor.device.type == 'cpu' and kwargs.get('device') == table.device:
            index_copies.append(tensor)
            assert kwargs.get('non_blocking'), 'Selected MI index copies must be nonblocking'
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', no_cuda_cpu)
    monkeypatch.setattr(torch.Tensor, 'numpy', no_cuda_numpy)
    monkeypatch.setattr(torch.Tensor, 'to', nonblocking_indices)
    value = source.information_for(table, torch.tensor([3, 1, 3]))
    torch.autograd.grad(value.sum(), table)
    assert len(index_copies) == 5
    with pytest.raises(ValueError, match='remain on CPU'):
        source.information_for(table, torch.tensor([3], device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime is unavailable')
def test_unprepared_cuda_histogram_requires_cpu_setup():
    source = statistics(sparse=True).to('cuda')
    with pytest.raises(ValueError, match='before to'):
        source.information_for(torch.full((4, 2), .5, dtype=torch.double, device='cuda'), [0])
