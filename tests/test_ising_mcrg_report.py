"""Numerical safeguards for the independent finite-scale MCRG response audit.

These tests use synthetic matrices, not Ising's theoretical critical exponent.
They protect the operator-coordinate convention and retained-subspace spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.report_ising_mcrg import solve_response


def spectrum(result):
    pairs = np.asarray(result['eigenvalues'], dtype=float)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    return pairs[:, 0] + 1j * pairs[:, 1]


def assert_spectrum(result, expected, *, atol=2e-10, rtol=2e-10):
    actual = spectrum(result)
    expected = np.asarray(expected, dtype=complex)
    assert actual.shape == expected.shape
    # Match by distance instead of relying on arbitrary eigensolver ordering.
    remaining = list(actual)
    for value in expected:
        index = int(np.argmin(np.abs(np.asarray(remaining) - value)))
        found = remaining.pop(index)
        assert found == pytest.approx(value, abs=atol, rel=rtol)


@pytest.fixture
def response_matrices():
    B = np.array([[4., .3, .2], [.3, 2., .1], [.2, .1, 1.]])
    # B T=A has a genuinely nonsymmetric T; replacing it with a symmetric
    # matrix or diagonalizing A alone would generally give the wrong spectrum.
    T = np.array([[3.3, .7, -.4], [0., 2.01, .5], [0., 0., -.6]])
    return B @ T, B, T


def test_recovers_nonsymmetric_response_in_correct_matrix_order(response_matrices):
    A, B, T = response_matrices
    result = solve_response(A, B)
    assert_spectrum(result, np.linalg.eigvals(T))
    assert result['retained_rank'] == 3
    assert result['leading_positive_real'] == pytest.approx(3.3)
    assert result['residual'] < 1e-10
    assert np.isfinite(result['condition_number'])


@pytest.mark.parametrize('factor', [1e-12, 1e12])
def test_global_covariance_units_do_not_change_eigenvalues(response_matrices, factor):
    A, B, T = response_matrices
    assert_spectrum(solve_response(factor * A, factor * B), np.linalg.eigvals(T))


def test_changing_covariance_metric_does_not_change_known_T(response_matrices):
    _, _, T = response_matrices
    metric = np.array([[.05, .001, .002], [.001, 13., .3], [.002, .3, 400.]])
    assert np.all(np.linalg.eigvalsh(metric) > 0)
    assert_spectrum(solve_response(metric @ T, metric), np.linalg.eigvals(T))


def test_same_operator_redefinition_at_both_scales_is_a_similarity(response_matrices):
    A, B, T = response_matrices
    D = np.diag([1e-3, 7., 1e3])
    # S'=D S implies A'=D A D and B'=D B D, so T'=D^-1 T D.
    result = solve_response(D @ A @ D, D @ B @ D)
    assert_spectrum(result, np.linalg.eigvals(T), atol=2e-9, rtol=2e-9)
    assert result['retained_rank'] == 3


def test_separate_fine_coarse_volume_normalization_changes_spectrum(response_matrices):
    A, B, T = response_matrices
    fine_volume, coarse_volume = 108**2, 54**2
    extensive = solve_response(A, B)
    density = solve_response(A / (fine_volume * coarse_volume), B / coarse_volume**2)
    # This is deliberately *not* an allowed operator-coordinate similarity.
    # The solver must not manufacture a theoretical b factor to hide it.
    expected = np.linalg.eigvals(T) * coarse_volume / fine_volume
    assert_spectrum(density, expected)
    assert density['leading_positive_real'] == pytest.approx(
        extensive['leading_positive_real'] / 4.)


def test_rank_deficiency_returns_only_retained_modes_not_pseudoinverse_zeros():
    B = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 2.]])
    T = np.diag([1.5, 1.5, -.5])
    result = solve_response(B @ T, B)
    assert result['retained_rank'] == 2
    assert_spectrum(result, [1.5, -.5])
    assert result['leading_positive_real'] == pytest.approx(1.5)
    assert result['residual'] < 1e-10


def test_relative_cutoff_is_applied_to_preconditioned_retained_subspace():
    epsilon = 1e-9
    B = np.array([[1., 1. - epsilon], [1. - epsilon, 1.]])
    A = 1.7 * B
    loose = solve_response(A, B, rcond=1e-6)
    tight = solve_response(A, B, rcond=1e-12)
    assert loose['retained_rank'] == 1
    assert tight['retained_rank'] == 2
    assert_spectrum(loose, [1.7], atol=1e-7, rtol=1e-7)
    assert_spectrum(tight, [1.7, 1.7], atol=1e-6, rtol=1e-6)


def test_complex_negative_and_positive_modes_are_all_retained():
    T = np.array([[.4, -1.2, 0., 0.], [1.2, .4, 0., 0.],
                  [0., 0., -2.7, 0.], [0., 0., 0., 1.3]])
    result = solve_response(T, np.eye(4))
    assert_spectrum(result, [.4 + 1.2j, .4 - 1.2j, -2.7, 1.3])
    # The largest modulus is negative, and the complex modes have positive
    # real parts. Neither is the leading *positive real* candidate.
    assert result['leading_positive_real'] == pytest.approx(1.3)


def test_no_positive_real_mode_does_not_create_a_thermal_candidate():
    T = np.array([[.3, -1., 0.], [1., .3, 0.], [0., 0., -.8]])
    result = solve_response(T, np.eye(3))
    assert_spectrum(result, [.3 + 1j, .3 - 1j, -.8])
    assert result['leading_positive_real'] is None


def test_leading_mode_is_not_selected_by_closeness_to_theoretical_two():
    T = np.diag([4.2, 2.00001, .8])
    result = solve_response(T, np.eye(3))
    assert_spectrum(result, [4.2, 2.00001, .8])
    assert result['leading_positive_real'] == pytest.approx(4.2)
