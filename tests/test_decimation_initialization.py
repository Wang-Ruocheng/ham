from dataclasses import replace
import pytest
import torch
from recurrent_rg.config import ModelConfig
from recurrent_rg.checkpoints import build_model, load_model_state, qcr_state_dict


@pytest.mark.parametrize('dimension', [1, 2])
@pytest.mark.parametrize('lookup', [False, True])
def test_positive_decimation_initial_map_and_trainability(dimension, lookup):
    cfg = ModelConfig(dimension=dimension, block_factor=2, theory_dim=2, hidden=12,
                      binary_lookup=lookup, kernel_initialization='positive_decimation', kernel_sampling='deterministic')
    torch.manual_seed(91)
    model = build_model(cfg, 2)
    volume = 2**dimension
    patterns = (2*((torch.arange(2**volume)[:, None] // (2**torch.arange(volume))) % 2)-1).double()
    x = patterns.reshape(2**volume, 1, *([2]*dimension), 1)
    torch.testing.assert_close(model.rg_kernel(x).flatten(), patterns[:, 0], atol=1e-14, rtol=0)
    random_x = (2*torch.randint(2, (2, 4, *([16]*dimension), 1))-1).double()
    expected = random_x[(slice(None), slice(None), *([slice(None, None, 4)]*dimension), slice(None))]
    torch.testing.assert_close(model.coarse_grain(random_x)[-1], expected)
    kernel = model.rg_kernel
    before = [p.detach().clone() for p in kernel.parameters()]
    optimizer = torch.optim.SGD(kernel.parameters(), lr=.1)
    loss = (kernel(x).flatten()+patterns[:, 0]).square().mean()
    loss.backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in kernel.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, kernel.parameters()))


def test_paired_qr_and_rng_and_weight_reload():
    cfg = ModelConfig(dimension=2, hidden=12, theory_dim=2)
    torch.manual_seed(91)
    random_model = build_model(cfg, 2)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    initialized = build_model(replace(cfg, kernel_initialization='positive_decimation'), 2)
    assert torch.equal(rng, torch.get_rng_state())
    for name in ('theory_encoder', 'theory_rg'):
        for a, b in zip(getattr(random_model, name).parameters(), getattr(initialized, name).parameters()):
            assert torch.equal(a, b)
    load_model_state(initialized, qcr_state_dict(random_model))
    for a, b in zip(random_model.rg_kernel.parameters(), initialized.rg_kernel.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize('kwargs', [dict(fixed_kernel=True), dict(kernel_layout='overlapping')])
def test_invalid_initialization_combination(kwargs):
    with pytest.raises(ValueError):
        build_model(ModelConfig(kernel_initialization='positive_decimation', **kwargs), 2)


@pytest.mark.parametrize('lookup', [False, True])
def test_three_by_three_recurrent_kernel(lookup):
    cfg = ModelConfig(dimension=2, theory_dim=2, hidden=12, block_factor=3,
                      binary_lookup=lookup, kernel_initialization='positive_decimation', kernel_sampling='deterministic')
    model = build_model(cfg, 2)
    patterns = (2*((torch.arange(512)[:, None] // (2**torch.arange(9))) % 2)-1).double()
    blocks = patterns.reshape(512, 1, 3, 3, 1)
    torch.testing.assert_close(model.rg_kernel(blocks).flatten(), patterns[:, 0], atol=1e-14, rtol=0)
    coords = torch.arange(36)
    checker = (1-2*((coords[:, None]+coords[None, :]) % 2)).double()
    x = torch.stack([checker, -checker, torch.ones_like(checker), -torch.ones_like(checker)])[:, None, :, :, None]
    chain = model.coarse_grain(x)
    assert [t.shape[2] for t in chain] == [36, 12, 4]
    for k, coarse in enumerate(chain):
        torch.testing.assert_close(coarse, x[:, :, ::3**k, ::3**k])
    model.rg_kernel(blocks).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.rg_kernel.parameters())
