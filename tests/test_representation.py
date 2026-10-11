import math

import pytest
import torch
from torch import nn

from recurrent_rg.representation import linear_distribution_reconstruction_loss


@pytest.mark.parametrize("mode", ["squared", "kl"])
def test_reconstruction_detaches_histogram_and_trains_encoder_decoder(mode):
    decoder = nn.Linear(2, 3).double()
    theory = torch.tensor([[0.3, -0.2], [0.8, 0.4]], dtype=torch.double, requires_grad=True)
    target = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=torch.double, requires_grad=True)
    loss = linear_distribution_reconstruction_loss(decoder, theory, target, mode=mode)
    loss.backward()
    assert target.grad is None
    assert theory.grad is not None and torch.isfinite(theory.grad).all()
    assert theory.grad.abs().sum() > 0
    assert decoder.weight.grad is not None and torch.isfinite(decoder.weight.grad).all()
    assert decoder.weight.grad.abs().sum() > 0


def test_multinomial_kl_is_finite_with_zero_target_probabilities():
    decoder = nn.Linear(1, 3).double()
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.zero_()
    theory = torch.tensor([[0.1]], dtype=torch.double, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.double)
    loss = linear_distribution_reconstruction_loss(decoder, theory, target, mode="kl")
    torch.testing.assert_close(loss, torch.tensor(math.log(3), dtype=torch.double))
    loss.backward()
    assert torch.isfinite(theory.grad).all()
    assert torch.isfinite(decoder.bias.grad).all()


@pytest.mark.parametrize("mode, expected", [("squared", 0.5), ("kl", math.log(2))])
def test_identical_latents_cannot_reconstruct_distinct_distributions(mode, expected):
    # The optimal common reconstruction is the mean of these targets. A
    # collision still incurs positive loss even with the optimal decoder.
    decoder = nn.Linear(1, 2).double()
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.fill_(0.5 if mode == "squared" else 0.0)
    theory = torch.zeros(2, 1, dtype=torch.double)
    target = torch.eye(2, dtype=torch.double)
    loss = linear_distribution_reconstruction_loss(decoder, theory, target, mode=mode)
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.double))


@pytest.mark.parametrize("bins", [2, 3, 8])
def test_squared_reconstruction_sums_bins_instead_of_averaging(bins):
    decoder = nn.Linear(1, bins).double()
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.fill_(1.0)
    theory = torch.zeros(2, 3, 1, dtype=torch.double)
    target = torch.zeros(2, 3, bins, dtype=torch.double)
    target[..., 0] = 1.0
    loss = linear_distribution_reconstruction_loss(decoder, theory, target)
    torch.testing.assert_close(loss, torch.tensor(float(bins - 1), dtype=torch.double))


@pytest.mark.parametrize("bad", [torch.tensor([[0.4, 0.4]]), torch.tensor([[-0.1, 1.1]])])
def test_reconstruction_rejects_invalid_histogram(bad):
    with pytest.raises(ValueError):
        linear_distribution_reconstruction_loss(nn.Linear(1, 2), torch.zeros(1, 1), bad)
