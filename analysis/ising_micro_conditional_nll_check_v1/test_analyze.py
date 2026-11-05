"""纯合成数据单测；不打开历史 g、模型、PT 或远端数据。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_SPEC = importlib.util.spec_from_file_location("micro_nll_analyzer", Path(__file__).with_name("analyze.py"))
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)


def expected_counts(kappa):
    """固定 |h| 的三个类别；符号概率服从真 Ising conditional。"""
    kappa = np.asarray(kappa)
    plus2 = 1. / (1. + np.exp(-4. * kappa))
    plus4 = 1. / (1. + np.exp(-8. * kappa))
    return np.stack((.4 * (1. - plus4), .4 * (1. - plus2),
                     np.full_like(kappa, .2), .4 * plus2, .4 * plus4), axis=-1) * 100000.


class LikelihoodTests(unittest.TestCase):
    def test_zero_is_log_two_and_counts_scale_invariance(self):
        counts = np.arange(1., 21.).reshape(4, 5)
        np.testing.assert_allclose(analyze.conditional_nll(np.zeros(4), counts), np.log(2.))
        np.testing.assert_allclose(analyze.conditional_nll(np.arange(4) / 10, counts),
                                   analyze.conditional_nll(np.arange(4) / 10, counts * 17))

    def test_histogram_matches_direct_spin_score(self):
        rng = np.random.default_rng(13)
        spins = rng.choice([-1, 1], size=(3, 8, 8))
        field = sum(np.roll(spins, shift, axis) for axis in (1, 2) for shift in (-1, 1))
        product = spins * field
        counts = np.array([(product == t).sum() for t in analyze.T_VALUES])
        for kappa in (-.5, 0., .47, 4.):
            direct = np.logaddexp(0., -2 * kappa * product).mean()
            self.assertAlmostEqual(float(analyze.conditional_nll(np.array(kappa), counts)), direct, places=14)

    def test_gradient_and_hessian_finite_differences(self):
        rng = np.random.default_rng(14)
        design = np.column_stack((rng.normal(size=23), np.ones(23)))
        counts = rng.integers(1, 100, size=(23, 5))
        theta = np.array([.23, -.11])
        value, gradient, hessian = analyze.objective_gradient_hessian(theta, design, counts)
        self.assertTrue(np.isfinite(value))
        h = 1e-5
        fd_gradient = np.empty(2)
        fd_hessian = np.empty((2, 2))
        for i in range(2):
            delta = np.eye(2)[i] * h
            plus, grad_plus, _ = analyze.objective_gradient_hessian(theta + delta, design, counts)
            minus, grad_minus, _ = analyze.objective_gradient_hessian(theta - delta, design, counts)
            fd_gradient[i] = (plus - minus) / (2 * h)
            fd_hessian[:, i] = (grad_plus - grad_minus) / (2 * h)
        np.testing.assert_allclose(gradient, fd_gradient, atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(hessian, fd_hessian, atol=2e-8, rtol=2e-8)
        self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0.)

    def test_extreme_logits_remain_finite(self):
        values = analyze.conditional_nll(np.array([-1e3, 1e3]), np.ones((2, 5)))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[0], values[1])

    def test_affine_recovery(self):
        x = np.linspace(-1., 1., 48).reshape(24, 2)
        target = .31 * x + .22
        counts = expected_counts(target)
        train = np.arange(24) < 18
        fit = analyze.fit_affine_train_only(x, counts, train)
        self.assertAlmostEqual(fit["a"], .31, places=9)
        self.assertAlmostEqual(fit["b"], .22, places=9)
        self.assertLessEqual(fit["audit"]["gradient_inf_norm"], analyze.GRAD_TOLERANCE)
        np.testing.assert_allclose(analyze.apply_affine(x, fit), target, atol=2e-10)

    def test_constant_optimum(self):
        counts = expected_counts(np.full((20, 2), .43))
        fit = analyze.fit_constant_train_only(counts, np.arange(20) < 15)
        self.assertAlmostEqual(fit["b"], .43, places=9)

    def test_affine_gauge_equivalence(self):
        x = np.linspace(-1, 1, 40).reshape(20, 2)
        counts = expected_counts(.4 * x + .1)
        train = np.arange(20) < 15
        fit = analyze.fit_affine_train_only(x, counts, train)
        gauge_x = -3.7 * x + 11.
        gauge_fit = analyze.fit_affine_train_only(gauge_x, counts, train)
        np.testing.assert_allclose(analyze.apply_affine(x, fit), analyze.apply_affine(gauge_x, gauge_fit), atol=2e-10)
        self.assertGreater(abs(analyze.conditional_nll(x, counts).mean() - analyze.conditional_nll(gauge_x, counts).mean()), .01)

    def test_affine_and_constant_do_not_read_test_values(self):
        x = np.linspace(-1, 1, 40).reshape(20, 2)
        counts = expected_counts(.23 * x + .3)
        train = np.arange(20) < 15
        baseline = analyze.fit_affine_train_only(x, counts, train)
        constant = analyze.fit_constant_train_only(counts, train)
        changed_x, changed_counts = x.copy(), counts.copy()
        changed_x[~train] = 9999.
        changed_counts[~train] = np.array([1., 3., 17., 10., 2.])
        changed = analyze.fit_affine_train_only(changed_x, changed_counts, train)
        changed_constant = analyze.fit_constant_train_only(changed_counts, train)
        self.assertEqual(baseline["a"], changed["a"])
        self.assertEqual(baseline["b"], changed["b"])
        self.assertEqual(baseline["train_g_mean"], changed["train_g_mean"])
        self.assertEqual(baseline["train_g_std_ddof0"], changed["train_g_std_ddof0"])
        self.assertEqual(constant["b"], changed_constant["b"])

    def test_separation_is_rejected(self):
        counts = np.tile([0., 0., 1., 20., 100.], (10, 1))
        with self.assertRaisesRegex(ValueError, "双侧"):
            analyze.newton_fit(np.ones((10, 1)), counts)

    def test_partial_pure_rows_are_accepted_when_mixed_rows_span(self):
        x = np.linspace(-1., 1., 20)
        counts = expected_counts(.2 * x + .1)
        counts[0] = [0., 0., 100., 200., 300.]
        design = np.column_stack((x, np.ones_like(x)))
        theta, audit = analyze.newton_fit(design, counts)
        self.assertTrue(np.isfinite(theta).all())
        self.assertEqual(audit["mixed_sign_training_rows"], 19)
        self.assertEqual(audit["mixed_sign_design_rank"], 2)
        self.assertTrue(audit["converged"])

    def test_constant_with_opposite_pure_rows_has_finite_optimum(self):
        counts = np.array([[0., 0., 0., 0., 100.], [100., 0., 0., 0., 0.]])
        theta, audit = analyze.newton_fit(np.ones((2, 1)), counts)
        np.testing.assert_array_equal(theta, [0.])
        self.assertEqual(audit["mixed_sign_training_rows"], 0)

    def test_nonidentified_constant_g_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "常数"):
            analyze.fit_affine_train_only(np.ones((10, 2)), np.ones((10, 2, 5)), np.ones(10, dtype=bool))


class IdentityAndBootstrapTests(unittest.TestCase):
    @staticmethod
    def identities():
        # Conditions deliberately reuse their IDs in the two splits.
        split = np.repeat([0, 1], 24)
        groups = np.tile(np.repeat([0, 1], 12), 2)
        replicas = np.tile(np.arange(12), 4)
        return split, groups, replicas

    def test_pair_identity_disjointness_no_cross_split(self):
        split, groups, replicas = self.identities()
        rng = np.random.default_rng(9)
        permutation = rng.permutation(48)
        split, groups, replicas = split[permutation], groups[permutation], replicas[permutation]
        pairs = analyze.build_independent_pairs(split, groups, replicas)
        self.assertEqual(pairs.shape, (24, 2))
        self.assertEqual(len(np.unique(pairs)), 48)
        np.testing.assert_array_equal(split[pairs[:, 0]], split[pairs[:, 1]])
        np.testing.assert_array_equal(groups[pairs[:, 0]], groups[pairs[:, 1]])
        np.testing.assert_array_equal(replicas[pairs], np.tile(np.arange(12).reshape(6, 2), (4, 1)))

    def test_missing_or_duplicate_replica_is_rejected(self):
        split, groups, replicas = self.identities()
        replicas[0] = 1
        with self.assertRaisesRegex(ValueError, "0..11"):
            analyze.build_independent_pairs(split, groups, replicas)

    def test_pair_direction_uses_opposite_chain(self):
        split, groups, replicas = self.identities()
        pairs = analyze.build_independent_pairs(split, groups, replicas)
        source = np.arange(48)[pairs]
        target = np.arange(48)[pairs[:, ::-1]]
        np.testing.assert_array_equal(source[:, 0], target[:, 1])
        np.testing.assert_array_equal(source[:, 1], target[:, 0])
        self.assertTrue(np.all(source != target))

    def test_bootstrap_preserves_each_stratum_size(self):
        groups = np.repeat([0, 1, 2], 6)
        indices = analyze.stratified_bootstrap_indices(groups, 1000, 77)
        self.assertEqual(indices.shape, (1000, 18))
        for group in [0, 1, 2]:
            np.testing.assert_array_equal((groups[indices] == group).sum(axis=1), np.full(1000, 6))
        np.testing.assert_array_equal(indices, analyze.stratified_bootstrap_indices(groups, 1000, 77))

    def test_paired_difference_not_independent_model_resample(self):
        groups = np.repeat([0, 1], 6)
        indices = analyze.stratified_bootstrap_indices(groups, 1000, 8)
        first = np.arange(12, dtype=float)
        values = np.stack((first, first - .25))
        result = analyze.paired_summary(values, indices)
        self.assertEqual(result["supervised_minus_unsupervised"]["mean_nats_per_site"], -.25)
        np.testing.assert_array_equal(result["supervised_minus_unsupervised"]["bootstrap_percentile_ci95"], [-.25, -.25])

    def test_histogram_integer_totals_and_identity(self):
        identities = {key: np.arange(768) for key in analyze.IDENTITY_KEYS}
        histogram = {key: values.tolist() for key, values in identities.items()}
        counts = np.zeros((768, 2, 5), dtype=int)
        counts[:, :, 2] = 64 * 108 ** 2
        histogram["counts"] = counts.tolist()
        histogram["bins"] = [-4, -2, 0, 2, 4]
        np.testing.assert_array_equal(analyze.validate_histograms(histogram, identities), counts)
        histogram["replica_index"][0] = -1
        with self.assertRaisesRegex(ValueError, "replica_index"):
            analyze.validate_histograms(histogram, identities)
        histogram["replica_index"][0] = 0
        histogram["counts"][0][0][2] -= 1
        with self.assertRaisesRegex(ValueError, "64"):
            analyze.validate_histograms(histogram, identities)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(FileExistsError):
                analyze.run(Path("does_not_exist.npz"), Path("does_not_exist.json"), output)

    def test_histogram_receipt_must_be_complete_and_unchanged(self):
        receipt = {"status": "complete", "workflow_complete": True, "source_metadata_unchanged": True}
        analyze.validate_histogram_receipt(receipt)
        for key, invalid in (("status", "failed"), ("workflow_complete", False),
                             ("source_metadata_unchanged", False)):
            changed = dict(receipt)
            changed[key] = invalid
            with self.assertRaises(ValueError):
                analyze.validate_histogram_receipt(changed)
            del changed[key]
            with self.assertRaises(ValueError):
                analyze.validate_histogram_receipt(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
