"""仅合成数组测试；不打开实验数据或加载QCR。"""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location("future_diagnostic", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic(groups_train=5, groups_test=1):
    n = (groups_train + groups_test) * 12
    rng = np.random.default_rng(12)
    split = np.repeat([0, 1], [groups_train * 12, groups_test * 12])
    groups = np.concatenate((np.repeat(np.arange(groups_train), 12), np.repeat(np.arange(groups_test), 12)))
    replica = np.tile(np.arange(12), groups_train + groups_test)
    gfull = rng.normal(size=(1, n, 5, 3))
    gfull = np.repeat(gfull, 2, axis=0)
    return dict(g_full=gfull, g=np.full((2, n, 5, 3, 2), 123456.),
        observables=rng.normal(size=(n, 5, 3, 2, 5)), motif_means=rng.normal(size=(n, 5, 3, 2, 16)),
        split=split, group_index=groups, replica_index=replica, model_names=np.array(["unsupervised", "supervised"]))


class DiagnosticTests(unittest.TestCase):
    def test_disjoint_independent_pairs_and_directions(self):
        arrays = synthetic()
        paired = MODULE.build_pairs(arrays)
        np.testing.assert_array_equal(paired["source_indices"][0], [0, 1])
        np.testing.assert_array_equal(paired["source_indices"][5], [10, 11])
        np.testing.assert_array_equal(paired["target_indices"], paired["source_indices"][:, ::-1])
        self.assertEqual(len(np.unique(paired["source_indices"])), len(arrays["split"]))
        for a, b in paired["source_indices"]:
            self.assertEqual(arrays["split"][a], arrays["split"][b])
            self.assertEqual(arrays["group_index"][a], arrays["group_index"][b])

    def test_mean_C_full_g_not_half_g(self):
        arrays = synthetic()
        paired = MODULE.build_pairs(arrays)
        self.assertAlmostEqual(paired["x"][0, 0, 0, 0], arrays["g_full"][0, 0, :, 0].mean())
        self.assertNotEqual(paired["x"][0, 0, 0, 0], arrays["g"][0, 0, :, 0].mean())

    def test_future_alignment_and_extra_only_current(self):
        arrays = synthetic()
        paired = MODULE.build_pairs(arrays)
        for path, (source, target) in enumerate(MODULE.TRANSITIONS):
            np.testing.assert_allclose(paired["truth"][0, 0, path], arrays["observables"][1, :, target].mean(axis=(0, 1)))
            np.testing.assert_allclose(paired["extra"][0, 0, path, :16], arrays["motif_means"][0, :, source].mean(axis=(0, 1)))

    def test_reject_duplicate_replica(self):
        arrays = synthetic(); arrays["replica_index"][0] = 1
        with self.assertRaises(ValueError):
            MODULE.build_pairs(arrays)

    def test_bootstrap_sums_and_fixed_groups(self):
        groups = np.repeat(np.arange(4), 6)
        weights = MODULE.bootstrap_weights(groups, 31, np.random.default_rng(1))
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        for group in range(4):
            np.testing.assert_allclose(weights[:, groups == group].sum(axis=1), .25)

    def test_rank_monotonic_invariance_and_reversal(self):
        x = np.linspace(-2, 2, 80)
        test = np.linspace(-2.2, 2.2, 29)
        y = np.column_stack((np.sin(x), x*x))
        reference = MODULE.fit_probe(x, y, "rank_rbf", .001, .5)
        expected = MODULE.predict_probe(reference, test)
        for transform in (lambda v: np.exp(v), lambda v: -v, lambda v: -np.exp(v)):
            fit = MODULE.fit_probe(transform(x), y, "rank_rbf", .001, .5)
            np.testing.assert_allclose(MODULE.predict_probe(fit, transform(test)), expected, atol=2e-12, rtol=2e-12)

    def test_midpoint_ties(self):
        np.testing.assert_allclose(MODULE.midpoint_cdf(np.array([1., 1., 3., 4.]), [0., 1., 2., 3., 5.]), [0., .25, .5, .625, 1.])

    def test_excluded_extra_ignores_extra_values(self):
        x = np.linspace(-1, 1, 30); y = np.sin(x)[:, None]
        extra = np.column_stack((x, x*x))
        fit = MODULE.fit_probe(x, y, "rank_rbf_extra", .1, 1., extra, None)
        np.testing.assert_array_equal(MODULE.predict_probe(fit, x, extra), MODULE.predict_probe(fit, x, extra * 1e9))

    def test_no_test_fit_leak_and_identical_models(self):
        arrays = synthetic()
        protocol = json.loads(MODULE.PROTOCOL_PATH.read_text())
        protocol["bootstrap"]["repeats"] = 20
        report, saved = MODULE.analyze(arrays, protocol, ridges=(.1,), bandwidths=(.5,))
        changed = {k: v.copy() for k, v in arrays.items()}
        test = changed["split"] == 1
        changed["g_full"][:, test] += 100.
        changed["observables"][test] += 50.
        changed["motif_means"][test] -= 80.
        report2, _ = MODULE.analyze(changed, protocol, ridges=(.1,), bandwidths=(.5,))
        for method in MODULE.METHODS:
            for first, second in zip(report["methods"][method]["fitted_models"], report2["methods"][method]["fitted_models"]):
                self.assertEqual(first["cross_validation"], second["cross_validation"])
                np.testing.assert_array_equal(first["fit"]["coefficient"], second["fit"]["coefficient"])
            np.testing.assert_array_equal(saved[f"{method}_prediction"][0], saved[f"{method}_prediction"][1])
            difference = report["methods"][method]["splits"]["TEST"]["supervised_minus_unsupervised"]["overall"]
            self.assertEqual(difference["NMSE"], 0.)
            np.testing.assert_array_equal(difference["NMSE_ci95"], [0., 0.])
        self.assertFalse(report["evidence"]["relative_supervised_better"])
        self.assertFalse(report["evidence"]["extra_feature_mechanism_supported"])


if __name__ == "__main__":
    unittest.main()
