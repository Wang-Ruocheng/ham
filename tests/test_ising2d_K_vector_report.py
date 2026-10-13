from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising2d_K_vector as reporter


class AnalyticRG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64))

    def advance_theory(self, point):
        x, y = point.unbind(-1)
        return torch.stack((self.multiplier * x + x ** 3, 2 * y + .25 * x), dim=-1)


def test_full_jacobian_includes_skip_and_verifies_all_directions_without_parameter_gradients():
    model = AnalyticRG()
    result = reporter.full_r_jacobian(model, [0., 0.])
    np.testing.assert_allclose(result['jacobian'], [[3., 0.], [.25, 2.]], atol=0)
    assert result['root_verified'] and result['derivative_verified']
    assert result['finite_difference_h'] == 1e-5
    assert result['derivative_max_abs_error'] < 1e-8
    assert model.multiplier.grad is None


def test_every_positive_relevant_eigenvalue_is_retained_without_thermal_selection():
    records = reporter.eigenvalue_records([[3., 0.], [0., 2.]], 3)
    assert len(records) == 2
    assert [record['conditional_nu'] for record in records] == pytest.approx([1., np.log(3) / np.log(2)])
    assert all(record['physical_sector'] == 'unidentified' and not record['nu_certified'] for record in records)


@pytest.mark.parametrize('matrix', [[[.5, 0], [0, -3]], [[1, 0], [0, 0]], [[0, -3], [3, 0]]])
def test_stable_negative_marginal_and_complex_directions_do_not_produce_nu(matrix):
    assert all(record['conditional_nu'] is None for record in reporter.eigenvalue_records(matrix, 3))


def test_unverified_root_or_jacobian_cannot_supply_exponent():
    assert all(record['conditional_nu'] is None
               for record in reporter.eigenvalue_records([[3, 0], [0, 2]], 3, verified=False))


def test_curve_inverse_keeps_folded_branches_and_flat_intervals_and_residuals():
    projection = reporter.curve_projections([.2, .3, .4, .5], [[0, 0], [1, 0], [0, 0], [1, 0]], [.5, 0])
    assert projection['exact_inverse_K_branches'] == pytest.approx([.25, .35, .45])
    assert len(projection['nearest_segments']) == 3
    plateau = reporter.curve_projections([.2, .3, .4], [[1, 0], [1, 0], [2, 0]], [1, 0])
    assert plateau['exact_inverse_K_intervals'] == [[.2, .3]]
    off_curve = reporter.curve_projections([.2, .4], [[0, 0], [1, 0]], [.5, .1])
    assert off_curve['exact_inverse_K_branches'] == []
    assert off_curve['nearest_distance'] == pytest.approx(.1)
    assert off_curve['nearest_segments'][0]['K'] == pytest.approx(.3)


def test_box_membership_of_thin_curves_is_not_a_support_certificate():
    train = np.array([[[-1, -1], [-1, -1], [-1, -1]],
                      [[1, 1], [1, 1], [1, 1]]], dtype=float)
    result = reporter.training_support([.5, -.5], train)
    assert result['inside_training_coordinate_box']
    assert result['inside_training_convex_hull'] is None
    assert not result['support_certified']
    assert result['nearest_training_input_distance'] > 1


def test_report_retains_all_roots_and_sectors_and_uses_train_covariance(tmp_path):
    model = AnalyticRG()
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=3),
                             checkpoint='numbered_checkpoint.pt', payload=dict(epoch=12000))
    train_direct = np.array([[[-1, 0], [-.8, 0], [-.6, 0]],
                             [[1, 0], [.8, 0], [.6, 0]]], dtype=float)
    test_direct = train_direct * .5
    def trajectory(direct):
        with torch.no_grad():
            one = model.advance_theory(torch.from_numpy(direct[:, :-1])).numpy()
        return dict(direct=direct, rollout=direct.copy(), one_step=one)
    fixed = dict(roots=[dict(g=[0., 0.]), dict(g=[.1, 0.])], search_box=[[-2, -2], [2, 2]])
    report = reporter.write_vector_report(loaded, trajectory(train_direct), trajectory(test_direct),
        [.2, .6], [.3, .5], fixed, tmp_path)
    assert len(report['roots']) == 2
    assert len(report['roots'][0]['eigenvalues']) == 2
    assert report['roots'][0]['root_verified']
    assert not report['roots'][1]['root_verified']
    assert all(item['conditional_nu'] is None for item in report['roots'][1]['eigenvalues'])
    assert report['roots'][0]['selected_thermal_eigenvalue'] is None
    assert not report['selection']['theory_used_for_selection']
    assert report['magnetic_exponents']['y_h'] is None
    assert report['participation_rank_by_scale'] == [1., 1., 1.]
    assert report['critical_encodings_by_scale'] != report['roots'][0]['g']
    assert (tmp_path / 'critical_exponents.json').is_file()
    assert (tmp_path / 'metrics.json').is_file()
    assert (tmp_path / 'vector_critical_summary.png').stat().st_size > 0
    assert (tmp_path / 'coupling_by_scale.png').stat().st_size > 0
    assert model.multiplier.grad is None
