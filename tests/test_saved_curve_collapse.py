"""保存曲线横向缩放估计的纯 NumPy 回归测试。"""
import json
import unittest

import numpy as np

from scripts.estimate_saved_curve_collapse import estimate_curve_collapse


class SavedCurveCollapseTests(unittest.TestCase):
    def setUp(self):
        self.Kc = .44
        self.knots = self.Kc + np.arange(-24, 25) * .005
        centered = self.knots - self.Kc
        means = np.stack([np.tanh(14 * 2**k * centered) for k in range(3)], axis=1)
        self.groups = np.repeat(means[:, None, :], 12, axis=1)

    def test_recovers_two_and_is_json_compatible(self):
        output = estimate_curve_collapse(self.knots, self.groups, critical_K=self.Kc,
                                        widths=(.02, .03), bootstraps=12)
        json.dumps(output, allow_nan=False)
        self.assertEqual(len(output['records']), 8)
        for row in output['records']:
            self.assertAlmostEqual(row['best_horizontal_multiplier'], 2., places=10)
            self.assertAlmostEqual(row['nominal_nu'], 1., places=10)
            self.assertLess(row['best_rmse'], 1e-12)
            self.assertFalse(row['thermal_nu_certified'])

    def test_anchored_offsets_are_paired(self):
        shifted = self.groups + np.array([0., .02, -.01])[None, None, :]
        output = estimate_curve_collapse(self.knots, shifted, critical_K=self.Kc,
                                        widths=(.03,), bootstraps=12)
        for row in output['records']:
            if row['anchored']:
                self.assertAlmostEqual(row['best_horizontal_multiplier'], 2., places=10)
                self.assertLess(row['best_rmse'], 1e-12)
            else:
                self.assertGreater(row['best_rmse'], .001)

    def test_constant_curve_is_unidentified(self):
        output = estimate_curve_collapse(self.knots, np.ones_like(self.groups),
                                        critical_K=self.Kc, widths=(.03,), bootstraps=4)
        for row in output['records']:
            self.assertIsNone(row['best_horizontal_multiplier'])
            self.assertIsNone(row['nominal_nu'])
            self.assertEqual(row['bootstrap_identified_count'], 0)

    def test_reproducible_bootstrap(self):
        rng = np.random.default_rng(98)
        noisy = self.groups + rng.normal(0., .002, self.groups.shape)
        first = estimate_curve_collapse(self.knots, noisy, critical_K=self.Kc,
                                       widths=(.03,), bootstraps=20)
        second = estimate_curve_collapse(self.knots, noisy, critical_K=self.Kc,
                                        widths=(.03,), bootstraps=20)
        self.assertEqual(first, second)

    def test_rejects_unsorted_or_missing_reference(self):
        with self.assertRaises(ValueError):
            estimate_curve_collapse(self.knots[::-1], self.groups[::-1], critical_K=self.Kc)
        with self.assertRaises(ValueError):
            estimate_curve_collapse(self.knots, self.groups, critical_K=self.Kc + .001)


if __name__ == '__main__':
    unittest.main()
