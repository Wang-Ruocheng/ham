"""Synthetic alignment checks; these are not empirical phi4 failure attribution.

The parity counterexample establishes an information loss in the current
one-/two-point soft descriptor, even on binary fields. It does not establish
that this loss caused any particular physical-data training result.
"""
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.phi4_qcr import (
    PAIR_OFFSETS,
    Phi4QCR,
    continuous_objectives,
    soft_local_features,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def parity_ensembles():
    """Eight even-/odd-parity 2x2 patterns, tiled with exact period 2 on L12."""
    patterns = torch.tensor(list(product((-1., 1.), repeat=4)), dtype=torch.double)
    fields = patterns.reshape(16, 2, 2).repeat(1, 6, 6).unsqueeze(-1)
    parity = patterns.prod(-1)
    return torch.stack((fields[parity == 1], fields[parity == -1]))


@pytest.mark.parametrize("theory_dim", [1, 2])
def test_pair_descriptor_cannot_distinguish_four_point_parity(theory_dim):
    x = parity_ensembles()
    assert x.shape == (2, 8, 12, 12, 1)
    bits = (x[..., 0] > 0).long()
    one_point = torch.stack([(bits == state).double().mean((1, 2, 3))
                             for state in range(2)], dim=-1)
    torch.testing.assert_close(one_point, torch.full((2, 2), .5, dtype=torch.double))
    for offset in PAIR_OFFSETS:
        neighbor = torch.roll(bits, tuple(-value for value in offset), (2, 3))
        pair = 2 * bits + neighbor
        probability = torch.stack([(pair == state).double().mean((1, 2, 3))
                                   for state in range(4)], dim=-1)
        torch.testing.assert_close(probability, torch.full((2, 4), .25, dtype=torch.double))
    spins = x[..., 0]
    four_point = (spins * torch.roll(spins, -1, 2) * torch.roll(spins, -1, 3)
                  * torch.roll(spins, (-1, -1), (2, 3))).mean((1, 2, 3))
    torch.testing.assert_close(four_point, torch.tensor([1., -1.], dtype=torch.double))

    features = soft_local_features(x, 1.)
    torch.testing.assert_close(features[0], features[1], atol=1e-15, rtol=0)
    motif = plaquette_distribution(x)
    torch.testing.assert_close((motif[0] - motif[1]).abs().sum(),
                               torch.tensor(2., dtype=torch.double), atol=0, rtol=0)
    torch.manual_seed(17)
    model = Phi4QCR(1., hidden=8, theory_dim=theory_dim)
    encoding = model.Q(features)
    torch.testing.assert_close(encoding[0], encoding[1], atol=1e-15, rtol=0)


def test_continuous_objectives_have_independent_component_gradients():
    torch.manual_seed(7)
    model = Phi4QCR(1., hidden=8, theory_dim=2)
    fields = torch.randn(2, 4, 9, 9, 1, dtype=torch.double)
    q_loss, c_loss, r_loss, _ = continuous_objectives(
        model, fields, generator=torch.Generator().manual_seed(11), information_pairs=4)
    modules = {name: list(getattr(model, name).parameters()) for name in ("Q", "C", "R")}
    for loss_name, loss in (("Q", q_loss), ("C", c_loss), ("R", r_loss)):
        for module_name, parameters in modules.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            if loss_name == module_name:
                assert any(value is not None and bool(value.abs().max() > 0) for value in gradients)
                assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
            else:
                assert all(value is None for value in gradients)


def test_two_transition_continuous_closure_matches_discrete_all_start_and_gradient():
    torch.manual_seed(7)
    model = Phi4QCR(1., hidden=8, theory_dim=2)
    fields = torch.randn(2, 4, 9, 9, 1, dtype=torch.double)
    _, _, r_loss, metrics = continuous_objectives(
        model, fields, generator=torch.Generator().manual_seed(11), information_pairs=4)
    levels = model.coarse_grain(fields, generator=torch.Generator().manual_seed(11))
    direct = []
    for level in levels:
        first = model.Q(model.features(level[:, :2].detach()))
        second = model.Q(model.features(level[:, 2:].detach()))
        direct.append((.5 * (first + second)).detach())
    direct = torch.stack(direct, dim=1)
    unweighted, _, _ = all_start_anchor_closure_loss(
        SimpleNamespace(advance_theory=model.advance), direct)
    expected = 10. * unweighted
    torch.testing.assert_close(r_loss, expected, atol=1e-14, rtol=1e-14)
    from_metrics = (5. * metrics["closure_one_step_level_1"]
                    + 2.5 * metrics["closure_one_step_level_2"]
                    + 2.5 * metrics["closure_rollout_level_2"])
    assert abs(r_loss.item() - from_metrics) < 1e-13
    actual_gradients = torch.autograd.grad(r_loss, list(model.R.parameters()))
    expected_gradients = torch.autograd.grad(expected, list(model.R.parameters()))
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-13)
