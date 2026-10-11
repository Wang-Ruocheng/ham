"""Execution fast paths do not silently change the default QCR recipe."""
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.cli import main as cli
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import two_spin_distribution, plaquette_distribution
from recurrent_rg.execution import (
    configure_cuda_execution, reference_binary_pattern_distribution,
    resident_training_data, select_training_batch,
)
from recurrent_rg.representation import linear_distribution_reconstruction_loss


@pytest.mark.parametrize('shape,fast', [((3, 8, 12, 1), two_spin_distribution),
                                      ((3, 8, 9, 9, 1), plaquette_distribution)])
def test_reference_and_counted_motifs_are_exactly_equal(shape, fast):
    generator = torch.Generator().manual_seed(66)
    x = (2*torch.randint(2, shape, generator=generator)-1).double()
    torch.testing.assert_close(reference_binary_pattern_distribution(x), fast(x),
                               atol=0, rtol=0)


def model(**overrides):
    config = dict(dimension=2, theory_dim=2, hidden=5, block_factor=3,
                  binary_lookup=True, affine_reconstruction=True,
                  kernel_sampling='bernoulli')
    config.update(overrides)
    return build_model(ModelConfig(**config), 2)


def test_flags_default_off_on_cpu_and_reference_and_untrusted_routes():
    current = model()
    for device, reference, independent in [('cpu', False, True),
            ('cuda:0', True, True), ('cuda:0', False, False)]:
        metadata = configure_cuda_execution(current, device, reference=reference,
                                             independent_c=independent)
        assert not current.share_binary_q_tables
        assert not current.fast_validated_statistics
        assert not current.rg_kernel.validated_binary_inputs
        assert metadata['precision'] == 'torch.float64'
    metadata = configure_cuda_execution(current, 'cuda:0')
    assert metadata['enabled'] and metadata['shared_binary_Q_tables']
    assert current.share_binary_q_tables and current.fast_validated_statistics
    assert current.rg_kernel.validated_binary_inputs
    # Setting flags itself does not allocate CUDA or initialize its RNG.
    for changed in (model(binary_lookup=False), model(kernel_sampling='deterministic')):
        metadata = configure_cuda_execution(changed, 'cuda:0')
        assert not metadata['shared_binary_Q_tables']


def test_cpu_residency_and_batch_selection_preserve_raw_dtype_and_rng():
    x = torch.arange(3*8*2).reshape(3, 8, 2).to(torch.int8)
    rng = torch.get_rng_state().clone()
    cached, metadata = resident_training_data(x, 'cpu', enabled=True)
    assert cached is x and not metadata['resident']
    ids, indices = torch.tensor([2, 0]), torch.tensor([7, 2, 4])
    batch = select_training_batch(cached, ids, indices, 'cpu')
    torch.testing.assert_close(batch, x[ids][:, indices].double(), atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')
def test_cuda_residency_budget_fallback_and_selection():
    x = torch.arange(3*8*2).reshape(3, 8, 2).to(torch.int8)
    ids, indices = torch.tensor([2, 0]), torch.tensor([7, 2, 4])
    before = torch.cuda.get_rng_state().clone()
    cached, metadata = resident_training_data(x, 'cuda:0', enabled=True)
    assert cached.device.type == 'cuda' and cached.dtype == x.dtype
    assert metadata['resident']
    batch = select_training_batch(cached, ids, indices, 'cuda:0')
    torch.testing.assert_close(batch.cpu(), x[ids][:, indices].double(), atol=0, rtol=0)
    fallback, metadata = resident_training_data(x, 'cuda:0', enabled=True, max_bytes=0)
    assert fallback is x and metadata['reason'] == 'cache_exceeds_safe_budget'
    torch.testing.assert_close(select_training_batch(fallback, ids, indices, 'cuda:0'),
                               batch, atol=0, rtol=0)
    assert torch.equal(torch.cuda.get_rng_state(), before)


def test_reconstruction_public_validation_and_trusted_shape_checks():
    decoder = torch.nn.Linear(2, 4).double()
    theory = torch.ones(3, 2, dtype=torch.double)
    target = torch.ones(3, 4, dtype=torch.double)/4
    strict = linear_distribution_reconstruction_loss(decoder, theory, target)
    trusted = linear_distribution_reconstruction_loss(decoder, theory, target, validate=False)
    torch.testing.assert_close(strict, trusted, atol=0, rtol=0)
    with pytest.raises(ValueError, match='sum to one'):
        linear_distribution_reconstruction_loss(decoder, theory, target*2)
    with pytest.raises(ValueError, match='batch shapes'):
        linear_distribution_reconstruction_loss(decoder, theory, target[:2], validate=False)


def test_cli_forwards_execution_flags_without_legacy(monkeypatch):
    calls = []
    monkeypatch.setattr('experiments.rsmi_separate_qcr.main', lambda argv: calls.append(argv))
    cli(['train', '--config', 'c', '--output', 'o', '--device', 'cuda:0',
         '--reference-execution'])
    assert calls == [['--config', 'c', '--output', 'o', '--device', 'cuda:0',
                      '--reference-execution']]
    calls.clear()
    cli(['train', '--config', 'c', '--output', 'o', '--device', 'cuda:0',
         '--mi-cache-dir', 'cache'])
    assert calls[0][-2:] == ['--mi-cache-dir', 'cache']
    with pytest.raises(SystemExit):
        cli(['train', '--config', 'c', '--output', 'o', '--allow-legacy-training',
             '--reference-execution'])
