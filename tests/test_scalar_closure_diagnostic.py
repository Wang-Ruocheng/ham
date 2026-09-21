import numpy as np
import torch

from scripts.diagnose_scalar_closure import describe_pairs, match_pairs, observables


def test_correlations_use_current_lattice_distance():
    ordered = torch.ones(1, 2, 32, 1, dtype=torch.float64)
    alternating = ordered.clone()
    alternating[:, :, 1::2] = -1
    assert torch.equal(observables(ordered), torch.ones(1, 4, dtype=torch.float64))
    assert torch.equal(observables(alternating), torch.tensor([[-1., 1., 1., 1.]], dtype=torch.float64))


def test_matching_excludes_same_coupling_and_saturated_states():
    values = np.zeros((3, 2, 10))
    values[:, :, 4] = .5
    values[:, :, 0] = [[0., 0.], [1., .005], [2., .001]]
    values[2, 1, 4] = 1.
    pairs = match_pairs(values, np.array([.1, .2, .3]), .01)
    assert pairs == [(0, 0, 1, 1)]
    summary = describe_pairs(values, pairs)
    assert summary["g"]["median_abs_difference"] == .005


def test_empty_matches_are_reported():
    assert describe_pairs(np.zeros((1, 1, 10)), []) == {"count": 0}
