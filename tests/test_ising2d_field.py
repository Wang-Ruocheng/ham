import itertools
import numpy as np
import pytest
from recurrent_rg.ising2d_field import sample_field_chain, field_observables


@pytest.mark.parametrize('b', [-.2, 0., .2])
def test_field_sampler_matches_exact_small_lattice(b):
    states = np.array(list(itertools.product((-1, 1), repeat=4))).reshape(-1, 2, 2)
    obs = field_observables(states)
    weights = np.exp(4*(.3*obs[:, 0]+b*obs[:, 1]))
    exact = (weights[:, None]*obs).sum(0)/weights.sum()
    draws = sample_field_chain(.3, b, 2, 20000, 200, 4, 91)
    np.testing.assert_allclose(field_observables(draws).mean(0), exact, atol=.025)


def test_zero_coupling_field_magnetization():
    draws = sample_field_chain(0., -.3, 8, 2000, 10, 2, 91)
    assert abs(draws.mean()-np.tanh(-.3)) < .01


def test_field_ensemble_views_do_not_erase_magnetization():
    import torch
    from experiments.joint_qcr_ising import ensemble_views
    data = torch.ones((2, 8, 4, 4, 1), dtype=torch.int8)
    assert ensemble_views(data, 8, spin_flip_augment=False).mean() == 1
    assert ensemble_views(data, 8, spin_flip_augment=True).mean() == 0
