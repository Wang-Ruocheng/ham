import numpy as np
import pytest
import torch

from scripts.export_ising_j1j2_pc12_field import (
    evaluate_displacement, plane_states, support_mask, validate_persisted_parameters,
)


def test_residual_map_is_not_subtracted_twice_and_is_batched():
    class Map:
        def advance_theory(self, g):
            assert not torch.is_grad_enabled()
            return g + 2. * g + .3
    states = np.arange(35, dtype=float).reshape(7, 5) / 7
    np.testing.assert_allclose(evaluate_displacement(Map(), states, chunk=2), 2 * states + .3)


def test_slice_fixes_unshown_principal_coordinates():
    mean = np.arange(5, dtype=float)
    basis = np.eye(5)[:2]
    xy = np.array([[.2, -.1], [.4, .5]])
    lifted = plane_states(xy, mean, basis)
    np.testing.assert_allclose((lifted - mean) @ basis.T, xy)
    np.testing.assert_array_equal(lifted[:, 2:], np.broadcast_to(mean[2:], (2, 3)))
    with pytest.raises(ValueError):
        plane_states(xy, mean, 2 * basis)


def test_support_does_not_fill_far_or_outside_convex_hull():
    inputs = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    queries = np.array([[.01, .01], [.5, .5], [1.01, 0.]])
    valid, scale = support_mask(inputs, queries)
    assert valid.tolist() == [True, False, False]
    np.testing.assert_allclose(scale, [.5, .5])


def test_persisted_weights_checked_but_unused_action_is_not_cross_load_evidence():
    model = torch.nn.Module()
    model.theory_rg = torch.nn.Linear(5, 5).double()
    model.action_network = torch.nn.Linear(5, 1).double()
    payload = {'model': {k: v.clone() for k, v in model.state_dict().items()
                         if not k.startswith('action_network.')}}
    with torch.no_grad():
        model.action_network.weight.add_(5.)
    assert validate_persisted_parameters(model, payload)
    with torch.no_grad():
        model.theory_rg.weight.add_(.01)
    with pytest.raises(ValueError, match='persisted'):
        validate_persisted_parameters(model, payload)
