import itertools
import numpy as np
from scripts.prepare_ising2d_qcr import chain, observations


def test_fixed_wolff_matches_exact_small_periodic_lattice():
    patterns = np.array(list(itertools.product((-1, 1), repeat=4))).reshape(-1, 2, 2)
    obs = observations(patterns)
    k = .3
    weights = np.exp(k*4*obs[:, 0])
    exact = (weights[:, None]*obs).sum(0)/weights.sum()
    draws = chain(k, 2, 10000, 100, 10, 91)
    np.testing.assert_allclose(observations(draws).mean(0), exact, atol=.035)
    assert set(np.unique(draws)) == {-1, 1}


def test_fixed_sampler_reproducible():
    np.testing.assert_array_equal(chain(.44, 8, 10, 100, 4, 12), chain(.44, 8, 10, 100, 4, 12))
