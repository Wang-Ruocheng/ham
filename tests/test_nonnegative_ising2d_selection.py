import pytest
import torch
from scripts.prepare_nonnegative_ising2d import select_nonnegative


def test_selection_keeps_zero_signed_b_and_sample_identity():
    x = torch.arange(20).reshape(5, 4)
    y = torch.tensor([[-.4, 0.], [0., -.5], [.4, -.5], [-.2, .5], [.4, .5]])
    a, b = select_nonnegative(x, y)
    assert torch.equal(a, x[[1, 2, 4]])
    assert torch.equal(b, y[[1, 2, 4]])


def test_empty_selection_rejected():
    with pytest.raises(ValueError, match='No nonnegative'):
        select_nonnegative(torch.ones(2, 3), torch.tensor([[-1., 0.], [-.5, 0.]]))
