"""独立保存数组测试；无 torch、模型、训练、MI 或物理采样。"""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import report_ising_matched_sizes as module


def metrics(dim=1):
    return dict(lattice_sizes=[108, 54, 27],
        train_representation=[dict(scale=k, pooled_coordinate_std=float(k + 1),
                                  coordinate_std=[float(k + 1)] * dim) for k in range(3)],
        fixed_points=dict(roots=[dict(root_id='F1', g=[0.] * dim,
             eigenvalues=[dict(real=1.4, imag=0., conditional_inverse_y=2.)],
             curve_projections_by_scale=[dict(exact_inverse_K_branches=[.41, .45])])]))


def arrays(sizes=(108, 54, 27), dim=1, nr=2, nc=2):
    """严格仿射 R(g)=g/2+.1；有限算符协方差由独立随机数组构造。"""
    rng = np.random.default_rng(234 + sizes[0])
    levels, nk = len(sizes), 3
    g = np.zeros((nk, nr, nc, levels, dim))
    g[..., 0, :] = rng.normal(size=(nk, nr, nc, dim))
    for scale in range(levels - 1):
        g[..., scale + 1, :] = .5 * g[..., scale, :] + .1
    full = np.broadcast_to(.5 * np.eye(dim), g.shape[:-1] + (dim, dim)).copy()
    pred = np.zeros(g.shape[:3] + (levels, levels, dim))
    for start in range(levels):
        pred[..., start, start, :] = g[..., start, :]
        for end in range(start + 1, levels):
            pred[..., start, end, :] = .5 * pred[..., start, end - 1, :] + .1
    operators = rng.normal(size=(nk, nr, nc, levels, 20, 7))
    center = operators - operators.mean(-2, keepdims=True)
    covariance = np.einsum('...sni,...snj->...sij', center, center) / 19
    cross = np.einsum('...sni,...snj->...sij', center[..., 1:, :, :], center[..., :-1, :, :]) / 19
    vin = rng.normal(size=(nk, nr, nc, levels - 1, dim, 7))
    result = dict(K=np.array([.43619047619047624, module.KC, .45142857142857146]),
        lattice_sizes=np.asarray(sizes), g=g, full_DR=full, r_fd_jacobian=full.copy(),
        r_of_g=.5 * g + .1, r_of_mean_g=(.5 * g + .1).mean((1, 2)),
        mean_r_of_g=(.5 * g + .1).mean((1, 2)),
        one_step_pred=.5 * g[..., :-1, :] + .1,
        all_start_horizon_pred=pred, operators=operators, orientation_operators=operators.copy(),
        covariance=covariance, cross_covariance=cross,
        q_response_in=vin, q_response_out=.5 * vin, q_response_predicted=.5 * vin,
        q_fd_response_in=np.stack([vin, vin], axis=-3),
        q_fd_response_out=np.stack([.5 * vin, .5 * vin], axis=-3),
        score_std=np.ones((nk, nr, nc, levels, 7)),
        best_J_even=full[..., :-1, :, :].copy(),
        chain_seeds=np.arange(nk * nr).reshape(nk, nr) + sizes[0] * 100)
    for kind in ('even', 'all'):
        result['response_input_rank_' + kind] = np.full((nk, nr, nc, levels - 1), dim)
        result['response_input_singular_values_' + kind] = np.ones((nk, nr, nc, levels - 1, dim))
        result['response_input_effective_rank_' + kind] = np.full((nk, nr, nc, levels - 1), float(dim))
        result['response_input_condition_' + kind] = np.ones((nk, nr, nc, levels - 1))
    return result


class MatchedSizeReportTests(unittest.TestCase):
    def test_fixed_train_norm_uses_actual_lattice(self):
        self.assertEqual(module.train_norm(metrics(), 54)['source_scale'], 1)
        self.assertEqual(module.train_norm(metrics(), 27)['pooled_coordinate_std'], 3.)
        with self.assertRaises(ValueError):
            module.train_norm(metrics(), 4)
        self.assertTrue(module.train_norm(metrics(), 4, smoke=True)['smoke_nearest_substitute'])

    def test_depth_one_and_two_linear_closure(self):
        for sizes, count in [((54, 27), 3), ((108, 54, 27), 9)]:
            rows = module.closure_analysis(arrays(sizes), metrics())
            self.assertEqual(len(rows), count)
            self.assertTrue(all(row['rmse'] < 1e-14 for row in rows))
            for row in rows:
                if row['end'] == row['start'] + 1:
                    np.testing.assert_allclose(row['jensen_mean_R_minus_R_of_mean'], 0., atol=1e-14)
        modified = arrays((54, 27))
        modified['all_start_horizon_pred'][..., 0, 1, :] += .3
        row = module.closure_analysis(modified, metrics())[0]
        self.assertAlmostEqual(row['rmse'], .3)
        self.assertAlmostEqual(row['nrmse'], .1)

    def test_independent_bootstrap_never_pairs_replica_numbers(self):
        left, right = np.array([[0.], [10.]]), np.array([[0.], [10.]])
        row = module.independent_difference(left, right, np.array([[0, 0], [0, 0]]),
                                            np.array([[1, 1], [1, 1]]))
        self.assertEqual(row['mean'][0], 0.)
        self.assertEqual(row['bootstrap_percentiles'][1, 0], -10.)
        self.assertFalse(np.array_equal(module.bootstrap_indices(12, 20, 1),
                                        module.bootstrap_indices(12, 20, 2)))

    def test_six_direction_response_and_fd_exact(self):
        for dim in (1, 3):
            saved = arrays(dim=dim)
            rows, fd = module.response_analysis(saved, module.bootstrap_indices(2, 4, 1))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['full_DR_relative_frobenius'] == 0. for r in rows))
            self.assertTrue(all(r['best_local_matrix_relative_frobenius'] == 0. for r in rows))
            self.assertTrue(all(r['existing_R_relative_error_mean'] == 0. for r in rows))
            self.assertTrue(all(r['best_J_relative_error_mean'] == 0. for r in rows))
            self.assertTrue(all(r['thermal_nu'] is None for r in rows))
            self.assertEqual(fd['in']['max_absolute_error'], 0.)
            self.assertEqual(fd['full_DR_h_1e_5']['max_absolute_error'], 0.)

    def test_complex_spectra_retained_and_zero_variance_unidentifiable(self):
        solved = module.solve_response(np.array([[0., -1.], [1., 0.]]), np.eye(2))
        self.assertEqual(len(solved['eigenvalues']), 2)
        self.assertEqual(sorted(abs(v[1]) for v in solved['eigenvalues']), [1., 1.])
        self.assertEqual(module.clean(np.array([1 + 2j]))[0], {'real': 1., 'imag': 2.})
        saved = arrays((54, 27))
        saved['covariance'][:] = 0.
        rows = module.spectra_analysis(saved, module.bootstrap_indices(2, 1, 1))
        self.assertTrue(any(r['status'] == 'unidentifiable' for r in rows))
        self.assertTrue(all(r['thermal_nu'] is None for r in rows))

    def test_shared_scale_density_and_coordinate_shapes(self):
        small, large = arrays((54, 27), dim=3), arrays(dim=3)
        rows = module.shared_scale_analysis(small, large, metrics(3),
            module.bootstrap_indices(2, 4, 1), module.bootstrap_indices(2, 4, 2))
        self.assertEqual({r['lattice'] for r in rows}, {54, 27})
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['g_larger_minus_smaller']['mean'].shape, (3,))
        self.assertEqual(rows[0]['physical_even_operator_density_larger_minus_smaller']['mean'].shape, (6,))

    def test_report_no_model_loading_no_thermal_certification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root, phase2, output = [Path(folder) / n for n in ('phase3', 'phase2', 'report')]
            for arm in module.ARMS:
                dim = 1 if arm.startswith('dim1') else 3
                target = phase2 / arm / 'report'
                target.mkdir(parents=True)
                (target / 'metrics.json').write_text(json.dumps(metrics(dim)))
                for size, lattices in [(8, (8, 4)), (16, (16, 8, 4))]:
                    collection = root / 'arms' / arm / f'L{size}' / 'collection'
                    collection.mkdir(parents=True)
                    np.savez_compressed(collection / 'operator_statistics.npz', **arrays(lattices, dim))
                    (collection / 'status.json').write_text(json.dumps(dict(status='complete', workflow_complete=True)))
                    (collection / 'diagnostic.json').write_text('{}')
            with patch.object(module, 'draw'), patch.object(module, 'chinese_figures', contextlib.nullcontext):
                result = module.report(root, phase2, output, bootstraps=1, smoke=True)
            self.assertTrue(result['no_model_loading'])
            self.assertEqual(result['new_training_updates'], 0)
            self.assertIsNone(result['thermal_nu'])
            reference = result['arms']['dim1_control']['phase2_reference_metrics']
            self.assertEqual(reference['fixed_points']['roots'][0]['curve_projections_by_scale'][0]['exact_inverse_K_branches'], [.41, .45])
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                module.report(root, phase2, output, bootstraps=1, smoke=True)


if __name__ == '__main__':
    unittest.main()
