"""仅 numpy 的小型单元测试；不加载 QCR、物理 PT 或远端状态。"""
import unittest

import numpy as np

import analyze as diagnostic


class TestFrozenAnalysis(unittest.TestCase):
    def test_stratified_physical_rep_bootstrap(self):
        groups = np.repeat(np.arange(3), 12)
        weights = diagnostic.stratified_bootstrap_weights(groups, np.random.default_rng(9), 80)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        for group in range(3):
            np.testing.assert_allclose(weights[:, groups == group].sum(axis=1), 1 / 3)
        paired = np.stack((np.arange(36.), np.arange(36.) + 2.))
        summary = diagnostic.paired_summary(paired, weights)
        np.testing.assert_allclose(summary["supervised_minus_unsupervised"]["mean"], 2.)
        np.testing.assert_allclose(summary["supervised_minus_unsupervised"]["ci95"], 2.)

    def test_rank_one_floor(self):
        matrix = np.diag([4., 3., 1.])
        residual, singular = diagnostic.rank_one_residual(matrix)
        self.assertAlmostEqual(float(residual), 10.)
        np.testing.assert_allclose(singular, [4., 3., 1.])
        rank1 = np.arange(16.)[:, None] * np.arange(6.)[None]
        residual, _ = diagnostic.rank_one_residual(rank1)
        self.assertAlmostEqual(float(residual), 0., places=7)

    def test_train_only_shared_response_readout(self):
        rng = np.random.default_rng(12)
        q = rng.normal(size=(2, 7, 2, 3, 1, 6))
        w = rng.normal(size=(16, 1))
        # 每个模型使用相同 Q，保证同 target 可供两者精确读出。
        q[1] = q[0]
        target = w[None, None, None, :, :] * q[0]
        train = np.arange(7) < 5
        fitted, prediction = diagnostic.fit_shared_response_readout(q, target, train)
        np.testing.assert_allclose(fitted[0], w, atol=1e-12)
        np.testing.assert_allclose(prediction, np.stack((target, target)), atol=1e-12)
        changed = target.copy()
        changed[~train] *= 800.
        fitted_changed, _ = diagnostic.fit_shared_response_readout(q, changed, train)
        np.testing.assert_array_equal(fitted, fitted_changed)

    def test_zero_target_not_perfect_relative_reconstruction(self):
        target = np.zeros((4, 2, 3, 16, 6))
        prediction = np.zeros((2,) + target.shape)
        lower, _ = diagnostic.rank_one_residual(target)
        metric = diagnostic.response_metrics(prediction, target, lower, 1e-14, np.ones(6) * 1e-14)
        self.assertTrue(np.isnan(metric["relative_frobenius"]).all())
        self.assertTrue(np.isnan(metric["direction_relative"]).all())
        np.testing.assert_array_equal(metric["raw_mse"], 0.)

    def test_linear_probe_and_group_cv(self):
        x = np.linspace(-2, 2, 120)
        y = np.stack((2 * x + 3, -4 * x - 1), axis=1)
        groups = np.repeat(np.arange(10), 12)
        fit, cv = diagnostic.choose_probe(x, y, groups, "linear")
        prediction = diagnostic.predict_probe(fit, x)
        np.testing.assert_allclose(prediction, y, atol=1e-5)
        self.assertEqual(fit["ridge"], 1e-6)
        self.assertEqual(len(cv["candidates"]), 5)

    def test_cross_half_target_variance_axis(self):
        rng = np.random.default_rng(21)
        truth = rng.normal(size=(24, 3, 2, 5))
        prediction = np.stack((truth + .2, truth + .1))
        train = np.arange(24) < 12
        test = ~train
        groups = np.repeat(np.arange(2), 12)
        weights = {name: diagnostic.stratified_bootstrap_weights(groups[mask], rng, 30) for name, mask in (("TRAIN", train), ("TEST", test))}
        report, error = diagnostic.probe_summary(prediction, truth, train, test, weights)
        np.testing.assert_allclose(error[0], .04)
        np.testing.assert_allclose(error[1], .01)
        result = report["splits"]["TEST"]["models"]["无监督"]
        np.testing.assert_allclose(result["rmse"]["value"], .2)
        expected_r2 = 1 - .04 / truth[test].var(axis=(0, 2))
        np.testing.assert_allclose(result["r2"]["value"], expected_r2)
        self.assertLess(report["splits"]["TEST"]["supervised_minus_unsupervised"]["all_scales_all_targets_nrmse"]["ci95"][1], 0)

    def test_synthetic_full_analysis_identical_models(self):
        rng = np.random.default_rng(301)
        n, draws = 120, 2
        groups = np.tile(np.repeat(np.arange(5), 12), 2)
        train = np.arange(n) < 60
        test = ~train
        target = rng.normal(size=(n, draws, 3, 16, 6)) * .02
        q = rng.normal(size=(2, n, draws, 3, 1, 6)) * .02
        q[1] = q[0]
        w = rng.normal(size=16)
        prediction = w[None, None, None, None, :, None] * q[..., 0, :][..., None, :]
        g = rng.normal(size=(2, n, draws, 3, 2))
        g[1] = g[0]
        arrays = dict(response_target=target, response_prediction=prediction, q_response=q,
                      response_target_halves=np.stack((target * 1.02, target * .98), axis=3),
                      score_cov=np.broadcast_to(np.eye(6), (n, draws, 3, 6, 6)),
                      score_std=np.ones((n, draws, 3, 6)), g=g,
                      observables=rng.normal(size=(n, draws, 3, 2, 5)))
        weights = {name: diagnostic.stratified_bootstrap_weights(groups[mask], rng, 30) for name, mask in (("TRAIN", train), ("TEST", test))}
        old_repeats = diagnostic.BOOTSTRAPS
        diagnostic.BOOTSTRAPS = 30
        try:
            response, _ = diagnostic.analyze_response(arrays, train, test, groups, weights, rng)
            probes, _ = diagnostic.analyze_probes(arrays, train, test, groups, weights)
        finally:
            diagnostic.BOOTSTRAPS = old_repeats
        diff = response["splits"]["TEST"]["methods"]["native_decoder"]["relative_frobenius"]["all_scales_equal_weight"]["supervised_minus_unsupervised"]
        self.assertEqual(float(diff["mean"]), 0.)
        np.testing.assert_array_equal(diff["ci95"], 0.)
        for kind in ("linear", "rbf"):
            diff = probes["methods"][kind]["splits"]["TEST"]["supervised_minus_unsupervised"]["all_scales_all_targets_nrmse"]
            self.assertEqual(float(diff["value"]), 0.)
            np.testing.assert_array_equal(diff["ci95"], 0.)

    def test_reliable_projection_does_not_read_test_values(self):
        rng = np.random.default_rng(72)
        train = np.arange(48) < 24
        groups = np.tile(np.repeat(np.arange(2), 12), 2)
        target = rng.normal(size=(48, 2, 3, 16, 6))
        first = diagnostic.fit_train_reliable_projection(target, train, groups, np.random.default_rng(81), repeats=30)
        changed = target.copy()
        changed[~train] *= 1e100
        second = diagnostic.fit_train_reliable_projection(changed, train, groups, np.random.default_rng(81), repeats=30)
        for key in ("projection", "singular_values", "bootstrap_noise_spectral_norm_p95", "reliable_rank"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_zero_train_projection_is_not_perfect(self):
        rng = np.random.default_rng(91)
        train = np.arange(48) < 24
        test = ~train
        groups = np.tile(np.repeat(np.arange(2), 12), 2)
        target = rng.normal(size=(48, 2, 3, 16, 6))
        target[train] = 0.
        prediction = np.zeros((2,) + target.shape)
        weights = {name: diagnostic.stratified_bootstrap_weights(groups[mask], rng, 30) for name, mask in (("TRAIN", train), ("TEST", test))}
        report, _ = diagnostic.analyze_train_projection(target, {"native_decoder": prediction}, train, test, groups, weights, rng)
        np.testing.assert_array_equal(report["fitted_train_metric"]["reliable_rank"], 0)
        for split in ("TRAIN", "TEST"):
            for metric in ("relative_frobenius", "raw_mse"):
                summary = report["splits"][split]["methods"]["native_decoder"][metric]["all_scales_equal_weight"]["models"]["无监督"]
                self.assertTrue(np.isnan(summary["mean"]))
                self.assertEqual(summary["valid_physical_replicates"], 0)


if __name__ == "__main__":
    unittest.main()
