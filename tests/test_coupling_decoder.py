"""A Gibbs decoder is conditional-likelihood exact only for its explicit finite basis."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recurrent_rg.coupling_decoder import (
    DirectionalCouplingDecoder, OPERATOR_NAMES, crossfit_loss,
    directional_operator_values, normalize_coupling_decoder_,
)


def spins(shape, seed=91):
    generator = torch.Generator().manual_seed(seed)
    return (2 * torch.randint(2, shape, generator=generator) - 1).to(torch.float64)


def decoder(dim=2):
    torch.manual_seed(917)
    return DirectionalCouplingDecoder(dim)


def test_seven_directional_operator_order_and_extensive_normalization():
    observed = torch.ones(2, 3, 5, 7, 1, dtype=torch.float64)
    result = directional_operator_values(observed)
    assert OPERATOR_NAMES == ('nn_x', 'nn_y', 'diagonal_plus', 'diagonal_minus',
        'axial_distance2_x', 'axial_distance2_y', 'plaquette')
    assert result.shape == (2, 3, 7)
    torch.testing.assert_close(result, torch.full((2, 3, 7), 35., dtype=torch.float64))


@pytest.mark.parametrize('coordinate', list(range(7)) + [None])
def test_local_field_matches_global_single_spin_flip_energy_difference(coordinate):
    net = decoder()
    if coordinate is not None:
        with torch.no_grad():
            net.linear.weight.zero_(); net.linear.bias.zero_()
            net.linear.bias[coordinate] = .37
    observed = spins((2, 3, 5, 5, 1))
    theory = torch.tensor([[.3, -.2], [.1, .7]], dtype=torch.float64)
    field = net.local_field(observed, theory)
    before = net.energy(observed, theory)
    for x, y in ((0, 0), (0, 4), (4, 0), (4, 4), (2, 3)):
        flipped = observed.clone()
        flipped[:, :, x, y, 0] *= -1
        delta = net.energy(flipped, theory) - before
        torch.testing.assert_close(delta, 2*observed[:, :, x, y, 0]*field[:, :, x, y],
                                   atol=1e-12, rtol=1e-12)


def test_pseudolikelihood_equals_brute_all_sites_global_flip_conditional():
    net = decoder(1)
    observed = spins((2, 2, 5, 5))
    theory = torch.tensor([[.4], [-.6]], dtype=torch.float64)
    before = net.energy(observed, theory)
    losses = []
    for x in range(5):
        for y in range(5):
            flipped = observed.clone()
            flipped[:, :, x, y] *= -1
            # NLL of the observed value, from its two exactly normalized energies.
            losses.append(F.softplus(before-net.energy(flipped, theory)))
    brute = torch.stack(losses, dim=-1).mean(dim=(1, 2))
    torch.testing.assert_close(net.pseudolikelihood(observed, theory, reduction='none'),
                               brute, atol=1e-12, rtol=1e-12)


def test_one_g_per_ensemble_is_broadcast_over_configurations_not_pooled():
    net = decoder(3)
    observed = spins((3, 4, 6, 7, 1))
    theory = torch.randn(3, 3, dtype=torch.float64)
    together = net.pseudolikelihood(observed, theory, reduction='none')
    separately = torch.stack([net.pseudolikelihood(observed[index:index+1], theory[index:index+1])
                              for index in range(3)])
    assert together.shape == (3,)
    torch.testing.assert_close(together, separately)
    # Different numbers of saved configurations do not alter the g broadcast rule.
    duplicated = observed.repeat_interleave(2, dim=1)
    torch.testing.assert_close(net.pseudolikelihood(duplicated, theory, reduction='none'), together)


def test_even_hamiltonian_and_conditional_loss_are_spin_flip_invariant():
    net = decoder(2)
    observed = spins((3, 4, 5, 6, 1))
    theory = torch.randn(3, 2, dtype=torch.float64)
    torch.testing.assert_close(directional_operator_values(observed), directional_operator_values(-observed))
    torch.testing.assert_close(net.energy(observed, theory), net.energy(-observed, theory))
    torch.testing.assert_close(net.local_field(observed, theory), -net.local_field(-observed, theory))
    torch.testing.assert_close(net.pseudolikelihood(observed, theory), net.pseudolikelihood(-observed, theory))


def test_spatial_translation_invariance_and_optional_channel_axis():
    net = decoder()
    observed = spins((2, 3, 5, 7))
    theory = torch.randn(2, 2, dtype=torch.float64)
    shifted = torch.roll(observed, (2, -3), dims=(-2, -1))
    torch.testing.assert_close(net.energy(observed, theory), net.energy(shifted, theory))
    torch.testing.assert_close(net.pseudolikelihood(observed, theory), net.pseudolikelihood(shifted, theory))
    torch.testing.assert_close(net.pseudolikelihood(observed, theory), net.pseudolikelihood(observed[..., None], theory))
    torch.testing.assert_close(net.energy(observed[:, 0], theory), net.energy(observed, theory)[:, 0])


def test_only_Q_latent_and_decoder_get_gradients_even_if_input_carried_C_graph():
    net = decoder(2)
    c_parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    observed = spins((2, 4, 5, 5)) * c_parameter
    # At the evaluated point these are still binary, but carrying a C graph.
    encoder = torch.nn.Linear(3, 2).double()
    features = torch.randn(2, 3, dtype=torch.float64)
    theory = encoder(features)
    loss = net.pseudolikelihood(observed, theory)
    loss.backward()
    assert c_parameter.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in (*net.parameters(), *encoder.parameters()))
    assert sum(float(parameter.grad.abs().sum()) for parameter in encoder.parameters()) > 0
    assert sum(float(parameter.grad.abs().sum()) for parameter in net.parameters()) > 0


def test_crossfit_is_symmetric_held_half_loss_with_no_extra_samples():
    net = decoder(1)
    left, right = spins((2, 3, 5, 5), 8), spins((2, 5, 5, 5), 9)
    g_left = torch.tensor([[.3], [-.7]], dtype=torch.float64, requires_grad=True)
    g_right = torch.tensor([[.4], [-.8]], dtype=torch.float64, requires_grad=True)
    expected = .5*(net.pseudolikelihood(right, g_left, reduction='none')
                   + net.pseudolikelihood(left, g_right, reduction='none'))
    actual = crossfit_loss(net, g_left, right, g_right, left, reduction='none')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(net.crossfit_loss(g_right, left, g_left, right), actual.mean())
    actual.mean().backward()
    assert g_left.grad is not None and g_right.grad is not None


def test_natural_parameters_are_affine_in_g_not_nonlin_energy_labels():
    net = decoder(3)
    first, second = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    fraction = .29
    torch.testing.assert_close(net.couplings(fraction*first+(1-fraction)*second),
        fraction*net.couplings(first)+(1-fraction)*net.couplings(second))
    assert net.couplings(first).shape == (4, 7)


def test_optional_column_gauge_preserves_rank_deficiency_and_zero_columns():
    net = decoder(3)
    with torch.no_grad():
        net.linear.weight[:, 1] = 2*net.linear.weight[:, 0]
        net.linear.weight[:, 2] = 0
    original_bias = net.linear.bias.detach().clone()
    assert torch.linalg.matrix_rank(net.linear.weight).item() == 1
    assert normalize_coupling_decoder_(net) is net
    torch.testing.assert_close(net.linear.weight.norm(dim=0), torch.tensor([1., 1., 0.], dtype=torch.float64))
    assert torch.linalg.matrix_rank(net.linear.weight).item() == 1
    torch.testing.assert_close(net.linear.bias, original_bias)


def test_invalid_shape_nonbinary_and_small_periodic_alias_are_rejected():
    net = decoder(1)
    theory = torch.ones(2, 1, dtype=torch.float64)
    for observed in (torch.zeros(2, 3, 5, 5), torch.ones(2, 3, 4, 5),
                     torch.full((2, 3, 5, 5), float('nan'))):
        with pytest.raises(ValueError):
            net.pseudolikelihood(observed, theory)
    with pytest.raises(ValueError, match='ensemble'):
        net.pseudolikelihood(spins((3, 2, 5, 5)), theory)
    with pytest.raises(ValueError, match='finite g'):
        net.pseudolikelihood(spins((2, 2, 5, 5)), theory[:, 0])
    with pytest.raises(ValueError, match='reduction'):
        net.pseudolikelihood(spins((2, 2, 5, 5)), theory, reduction='sum')
    with pytest.raises(ValueError, match='positive'):
        normalize_coupling_decoder_(net, epsilon=0)


def test_zero_action_has_log_two_loss_and_does_not_force_nonzero_couplings():
    net = decoder(2)
    with torch.no_grad():
        net.linear.weight.zero_(); net.linear.bias.zero_()
    theory = torch.randn(2, 2, dtype=torch.float64)
    loss = net.pseudolikelihood(spins((2, 4, 5, 5)), theory)
    assert float(loss) == pytest.approx(np.log(2.))
    normalize_coupling_decoder_(net)
    torch.testing.assert_close(net.couplings(theory), torch.zeros(2, 7, dtype=torch.float64))


def test_only_machine_roundoff_from_ST_binary_sample_is_canonicalized():
    net = decoder(1)
    observed = spins((2, 3, 5, 5))
    theory = torch.tensor([[.2], [.7]], dtype=torch.float64)
    roundoff = observed * (1 + 4e-15)
    torch.testing.assert_close(net.pseudolikelihood(roundoff, theory),
                               net.pseudolikelihood(observed, theory), atol=0., rtol=0.)
    torch.testing.assert_close(directional_operator_values(roundoff), directional_operator_values(observed),
                               atol=0., rtol=0.)
    relaxed = observed * .999
    with pytest.raises(ValueError, match='binary'):
        net.pseudolikelihood(relaxed, theory)
