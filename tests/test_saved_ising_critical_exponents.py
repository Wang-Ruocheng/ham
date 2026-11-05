"""离线指数换算、成组重采样与两步协方差的数值约定。"""
import unittest

import numpy as np

from scripts.estimate_saved_ising_critical_exponents import (
    clustered_bootstrap, direct_covariance, inverse_y, unique_even_candidate,
)
from scripts.report_ising_mcrg import solve_response


class SavedPhysicalEstimatesTests(unittest.TestCase):
    def test_inverse_y_uses_total_block_factor(self):
        self.assertAlmostEqual(inverse_y(2., 2), 1.)
        self.assertAlmostEqual(inverse_y(4., 4), 1.)
        self.assertAlmostEqual(inverse_y(4., 2), .5)
        for value in (-3., .9, 1., 2. + .01j):
            self.assertIsNone(inverse_y(value, 2))

    def test_no_selection_when_several_positive_relevant_modes_exist(self):
        result = solve_response(np.diag([2., 1.2, -.5]), np.eye(3))
        candidate = unique_even_candidate(result, 2)
        self.assertEqual(candidate['positive_real_relevant_count'], 2)
        self.assertIsNone(candidate['multiplier'])
        self.assertIsNone(candidate['conditional_inverse_y'])
        self.assertFalse(candidate['thermal_nu_certified'])

    def test_unique_candidate_ignores_negative_and_complex_modes(self):
        matrix = np.array([[2., 0., 0., 0.], [0., -3., 0., 0.],
                           [0., 0., .4, -.2], [0., 0., .2, .4]])
        result = solve_response(matrix, np.eye(4))
        candidate = unique_even_candidate(result, 2)
        self.assertAlmostEqual(candidate['multiplier'], 2.)
        self.assertAlmostEqual(candidate['conditional_inverse_y'], 1.)
        self.assertFalse(candidate['thermal_nu_certified'])

    def test_direct_covariance_is_within_ensemble_and_has_out_in_order(self):
        rng = np.random.default_rng(99)
        operators = rng.normal(size=(4, 5, 3, 128, 2))
        actual = direct_covariance(operators, 0, 2)
        expected = np.stack([np.stack([np.cov(operators[i, j, 2].T,
            operators[i, j, 0].T, ddof=1)[:2, 2:] for j in range(5)]) for i in range(4)])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # Arbitrary per-ensemble/per-draw/per-scale mean shifts cancel.
        offsets = rng.normal(size=(4, 5, 3, 1, 2)) * 1000
        np.testing.assert_allclose(direct_covariance(operators + offsets, 0, 2),
                                   expected, rtol=1e-10, atol=1e-10)

    def test_grouped_bootstrap_recovers_known_covariance_response(self):
        B = np.array([[2., .2], [.2, 1.]])
        T = np.array([[4., .3], [0., .2]])
        aa = np.repeat((B @ T)[None], 12, axis=0)
        bb = np.repeat(B[None], 12, axis=0)
        indices = np.random.default_rng(7).integers(0, 12, (20, 12))
        result = clustered_bootstrap(aa, bb, indices, 4)
        self.assertEqual(result['positive_relevant_count_histogram'], {'1': 20})
        self.assertEqual(result['unique_candidate_draws'], 20)
        np.testing.assert_allclose(result['multiplier_percentiles'], 4., atol=1e-12)
        np.testing.assert_allclose(result['conditional_inverse_y_percentiles'], 1., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
