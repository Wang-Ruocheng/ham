"""Validate the free-fermion Born sampler used for phase-reference plots."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_tfim_exact_long_chain import bond_matrix, configuration_probability, sample_spins, sample_spins_blocked
from plot_tfim_exact_marginals import ground_state
from plot_tfim_phase_transition import diagnostics


@pytest.mark.parametrize('x', [.3, .95, 1., 1.05, 3.])
def test_all_born_probabilities_and_correlations(x):
    spins, probabilities, _ = ground_state(8, x)
    matrix = bond_matrix(8, x)
    inferred = np.array([configuration_probability(matrix, s) for s in spins])
    np.testing.assert_allclose(inferred, probabilities, atol=1e-10, rtol=0)
    for r in range(1, 5):
        exact = probabilities @ (spins[:, 0] * spins[:, r])
        np.testing.assert_allclose(np.linalg.det(matrix[:r, :r]), exact, atol=1e-10)


def test_critical_nearest_neighbor_exact_formula():
    for n in [16, 128, 1024]:
        matrix = bond_matrix(n, 1.)
        expected = 1 / (n * np.sin(np.pi / (2*n)))
        np.testing.assert_allclose(matrix[0, 0], expected, atol=1e-12, rtol=0)


def test_sampler_binary_and_reproducible():
    matrix = bond_matrix(16, 1.05)
    uniforms = np.random.default_rng(91).random(16)
    a = sample_spins(matrix, uniforms)
    np.testing.assert_array_equal(a, sample_spins(matrix, uniforms))
    assert np.all(np.isin(a, [-1, 1]))


def test_fully_ordered_binder_and_bootstrap():
    samples = np.ones((32, 16), dtype=np.int8)
    samples[:16] *= -1
    result = diagnostics(samples, 91)
    assert result['mean_abs_m'] == 1
    np.testing.assert_allclose(result['binder_U'], 2/3)
    np.testing.assert_allclose(result['binder_ci95'], [2/3, 2/3])


@pytest.mark.parametrize('x', [.95, 1., 1.05])
def test_blocked_sampler_same_random_numbers(x):
    matrix = bond_matrix(64, x)
    rng = np.random.default_rng(131)
    for _ in range(20):
        uniforms = rng.random(64)
        expected = sample_spins(matrix, uniforms)
        for size in [1, 7, 32, 80]:
            np.testing.assert_array_equal(sample_spins_blocked(matrix, uniforms, size), expected)
