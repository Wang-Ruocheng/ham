"""Analytic and trajectory checks for the illustrative one-loop RG figure.

These tests do not sample fields, load learned models, or claim quantitative
two-dimensional critical exponents.  Only NumPy and unittest are required by
the tests themselves.
"""
from __future__ import annotations

import unittest

import numpy as np

from scripts import plot_phi4_theoretical_rg as rg


class TheoreticalPhi4RGTests(unittest.TestCase):
    def test_gaussian_and_wilson_fisher_coordinates(self):
        for epsilon in (0.1, 0.5, 1.0):
            with self.subTest(epsilon=epsilon):
                gaussian, wilson_fisher = rg.fixed_points(epsilon=epsilon)
                np.testing.assert_allclose(gaussian, [0.0, 0.0], atol=0.0)
                expected = [-epsilon / (6.0 + epsilon),
                            12.0 * epsilon / (6.0 + epsilon) ** 2]
                np.testing.assert_allclose(wilson_fisher, expected, rtol=1e-13)

    def test_both_fixed_points_have_vanishing_beta(self):
        for epsilon in (0.1, 0.5, 1.0):
            for point in rg.fixed_points(epsilon=epsilon):
                with self.subTest(epsilon=epsilon, point=point):
                    np.testing.assert_allclose(
                        rg.beta(point, epsilon=epsilon), np.zeros(2),
                        atol=1e-13, rtol=0.0,
                    )

    def test_beta_is_the_declared_momentum_shell_equation(self):
        state = np.array([-0.3, 0.2])
        expected = [2.0 * state[0] + state[1] / (1.0 + state[0]),
                    state[1] - 3.0 * state[1] ** 2 / (1.0 + state[0]) ** 2]
        np.testing.assert_allclose(rg.beta(state), expected, rtol=1e-14)

    def test_jacobian_matches_central_finite_differences(self):
        for epsilon in (0.5, 1.0):
            for point in (np.array([0.0, 0.0]), np.array([-0.3, 0.2]),
                          np.array([0.2, 0.5]),
                          rg.fixed_points(epsilon=epsilon)[1]):
                with self.subTest(epsilon=epsilon, point=point):
                    step = 1e-6
                    columns = []
                    for axis in range(2):
                        offset = np.eye(2)[axis] * step
                        columns.append(
                            (rg.beta(point + offset, epsilon=epsilon)
                             - rg.beta(point - offset, epsilon=epsilon))
                            / (2.0 * step)
                        )
                    numerical = np.column_stack(columns)
                    np.testing.assert_allclose(
                        rg.jacobian(point, epsilon=epsilon), numerical,
                        rtol=2e-7, atol=2e-9,
                    )

    def test_gaussian_is_repelling_and_wilson_fisher_is_a_saddle(self):
        for epsilon in (0.1, 0.5, 1.0):
            gaussian, wilson_fisher = rg.fixed_points(epsilon=epsilon)
            gaussian_values = np.linalg.eigvals(
                rg.jacobian(gaussian, epsilon=epsilon))
            wf_values = np.linalg.eigvals(
                rg.jacobian(wilson_fisher, epsilon=epsilon))
            with self.subTest(epsilon=epsilon):
                self.assertTrue(np.all(np.isreal(gaussian_values)))
                self.assertTrue(np.all(np.real(gaussian_values) > 0.0))
                self.assertTrue(np.all(np.isreal(wf_values)))
                self.assertEqual(int(np.count_nonzero(wf_values > 0.0)), 1)
                self.assertEqual(int(np.count_nonzero(wf_values < 0.0)), 1)
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvals(rg.jacobian(wilson_fisher))),
            [(1.0 - np.sqrt(22.0)) / 3.0,
             (1.0 + np.sqrt(22.0)) / 3.0], rtol=1e-13,
        )

    def test_critical_branches_follow_the_stable_manifold_backwards(self):
        gaussian, wilson_fisher = rg.fixed_points()
        left, right = (np.asarray(branch) for branch in rg.critical_branches())
        for branch in (left, right):
            self.assertEqual(branch.ndim, 2)
            self.assertEqual(branch.shape[1], 2)
            self.assertGreater(len(branch), 20)
            self.assertTrue(np.isfinite(branch).all())
            self.assertTrue(np.all(branch[:, 0] > -1.0))
            self.assertTrue(np.all(branch[:, 1] >= 0.0))
            self.assertLess(np.linalg.norm(branch[0] - wilson_fisher), 1e-3)

        self.assertLess(np.linalg.norm(left[-1] - gaussian), 1e-2)
        self.assertLess(left[-1, 1], left[0, 1])
        self.assertGreater(right[-1, 1], right[0, 1] + 0.05)

        # Critical branches are traced away from WF in reverse RG time, so
        # their segment tangents should point opposite to the forward beta.
        for branch in (left, right):
            tangent = np.diff(branch, axis=0)
            midpoint = 0.5 * (branch[1:] + branch[:-1])
            forward = np.array([rg.beta(point) for point in midpoint])
            norms = np.linalg.norm(tangent, axis=1) * np.linalg.norm(forward, axis=1)
            valid = norms > 1e-14
            self.assertGreater(np.count_nonzero(valid), 10)
            cosines = np.sum(tangent[valid] * forward[valid], axis=1) / norms[valid]
            self.assertLess(float(np.median(cosines)), -0.99)

    def test_critical_connection_is_not_a_straight_line_between_fixed_points(self):
        gaussian, wilson_fisher = rg.fixed_points()
        left, _ = rg.critical_branches()
        chord = wilson_fisher - gaussian
        relative = np.asarray(left) - gaussian
        perpendicular_distance = np.abs(
            chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
        ) / np.linalg.norm(chord)
        self.assertGreater(float(np.max(perpendicular_distance)), 1e-4)


if __name__ == "__main__":
    unittest.main()
