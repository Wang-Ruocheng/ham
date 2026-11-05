"""Unified reports must retain every direction without selecting theory-like modes."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from scripts import report_ising_dimg_ablation as report


class LinearMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer('matrix', torch.as_tensor(matrix, dtype=torch.float64))

    def advance_theory(self, point):
        return point @ self.matrix.T


def trajectories(dim, n=18):
    random = np.random.default_rng(91)
    knots = np.linspace(.2, .68, n)
    x = knots-report.KC
    direct = np.stack([np.stack([x*(1+.2*scale) + .02*j*x*x +
        random.normal(0, .001, n) for j in range(dim)], axis=-1) for scale in range(3)], axis=1)
    return direct, knots


def make_trajectories(model, direct):
    rollout = [direct[:, 0]]
    for _ in range(direct.shape[1]-1):
        rollout.append(report.mapped(model, rollout[-1]))
    return dict(direct=direct, rollout=np.stack(rollout, axis=1),
                one_step=report.mapped(model, direct[:, :-1]))


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_full_jacobian_includes_identity_and_all_dimensions(dim):
    matrix = np.diag([2., .6, -.2][:dim])
    checked = report.full_r_jacobian(LinearMap(matrix), np.zeros(dim))
    assert checked['root_verified'] and checked['derivative_verified']
    np.testing.assert_allclose(checked['jacobian'], matrix)
    assert checked['finite_difference_h'] == 1e-5


def test_inverse_projection_preserves_all_scalar_branches_and_flat_intervals():
    knots = [0., 1., 2., 3., 4.]
    projected = report.curve_projections(knots, np.array([[0.], [1.], [0.], [0.], [1.]]), [0.])
    assert projected['exact_inverse_K_branches'] == [0., 2., 3.]
    assert projected['exact_inverse_K_intervals'] == [[2., 3.]]


def test_vector_projection_does_not_invent_inverse_from_one_coordinate():
    projected = report.curve_projections([0., 1.], [[0., 0.], [1., 0.]], [.5, .1])
    assert projected['exact_inverse_K_branches'] == []
    assert projected['nearest_distance'] == pytest.approx(.1)


def test_all_eigenvalues_retained_and_largest_transverse_not_thermal():
    projections = [dict(nearest_segments=[dict(tangent=[1., 0., 0.])])]
    rows = report.eigenvalue_records(np.diag([1.7, 4., -.8]), projections,
        block_factor=2, verified=True, support_candidate=True)
    assert len(rows) == 3
    assert [row['real'] for row in rows] == [1.7, 4., -.8]
    assert rows[0]['K_curve_tangent_alignment_by_scale'] == [[1.]]
    assert rows[1]['K_curve_tangent_alignment_by_scale'] == [[0.]]
    assert all(row['thermal_nu'] is None and not row['thermal_identity_certified'] for row in rows)
    assert rows[2]['conditional_inverse_y'] is None


def test_extrapolated_or_unverified_directions_get_no_inverse_y():
    projections = [dict(nearest_segments=[dict(tangent=[1.])])]
    for verified, support in ((False, True), (True, False)):
        row = report.eigenvalue_records([[2.]], projections, block_factor=2,
            verified=verified, support_candidate=support)[0]
        assert row['conditional_inverse_y'] is None


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_root_search_is_nonexhaustive_and_never_selects_thermal(dim):
    direct, knots = trajectories(dim)
    model = LinearMap(np.diag([1.7, .5, .3][:dim]))
    roots = report.find_roots(model, direct, knots, direct, seed_points=3)
    assert roots['search_complete'] is False
    assert len(roots['roots']) == 1
    assert roots['roots'][0]['selected_thermal_eigenvalue'] is None
    assert len(roots['roots'][0]['eigenvalues']) == dim
    np.testing.assert_allclose(roots['roots'][0]['g'], np.zeros(dim), atol=1e-8)


def test_collapsed_representation_not_silently_good_normalized_error():
    direct = np.zeros((8, 3, 2))
    stats = report.representation_statistics(direct)
    assert all(row['participation_rank'] is None and row['numerical_rank'] == 0 for row in stats)
    values = dict(direct=direct, rollout=np.ones_like(direct), one_step=np.ones((8, 2, 2)))
    closure = report.closure_details(LinearMap(np.eye(2)), values, direct)
    assert closure['one_step']['nrmse'] == [None, None]
    assert closure['one_step']['coordinate_nrmse'] == [[None, None], [None, None]]


def test_K_path_is_not_eigenvalue_estimator():
    knots = np.linspace(.39, .49, 15)
    x = knots-report.KC
    means = np.stack((np.column_stack((x, x*.5)),
                      np.column_stack((2*x, x)), np.column_stack((4*x, 2*x))), axis=1)
    result = report.energy_path_diagnostics(LinearMap(2*np.eye(2)), knots, means)
    assert result['status'] == 'complete'
    assert all(row['best_scalar_response'] == pytest.approx(2.) for row in result['transitions'])
    assert all(row['thermal_nu'] is None for row in result['transitions'])


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_end_to_end_public_report_is_chinese_and_no_checkpoints(tmp_path, dim):
    direct, labels = trajectories(dim)
    model = LinearMap(np.diag([1.7, .5, .3][:dim]))
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=2),
        payload=dict(epoch=12000), checkpoint=tmp_path.parent/'run'/'training_step_12000.pt')
    values = make_trajectories(model, direct)
    result = report.write_report(loaded, values, values, labels, labels,
        [dict(epoch=100, train_total=.2, test_total=.3, train_rsmi_bits=.1, test_rsmi_bits=.11,
              train_reconstruction_mse_level_0=.01, test_reconstruction_mse_level_0=.02)], tmp_path)
    assert result['checkpoint_epoch'] == 12000 and result['theory_dim'] == dim
    assert json.loads((tmp_path/'report.json').read_text())['report_complete'] is True
    assert len(list(tmp_path.glob('*.png'))) == 5
    assert not list(tmp_path.glob('*.pt'))
    with np.load(tmp_path/'encodings.npz') as saved:
        np.testing.assert_array_equal(saved['train'], direct)


def test_compare_requires_exact_dimensions_and_complete_fixed_endpoints(tmp_path):
    for dim in (1, 2, 3):
        arm = tmp_path/f'dim{dim}'
        arm.mkdir()
        report.write_json(arm/'report.json', dict(status='complete', report_complete=True))
        report.write_json(arm/'metrics.json', dict(checkpoint_epoch=9000, theory_dim=dim))
    with pytest.raises(ValueError, match='12000'):
        report.compare_reports([f'dim{dim}={tmp_path/f"dim{dim}"}' for dim in (1, 2, 3)], tmp_path/'comparison')


def test_complete_summary_guard_rejects_partial_or_early_stop():
    cfg = SimpleNamespace(training=SimpleNamespace(epochs=12000), to_dict=lambda: dict(test=True))
    summary = dict(config=dict(test=True), actual_epochs=12000, requested_epochs=12000,
                   history=[dict(epoch=12000)])
    payload = dict(epoch=12000, history=[dict(epoch=12000)])
    report.validate_completed_run(cfg, summary, payload, 12000)
    for changes in (dict(actual_epochs=11900), dict(stopped_by_train_plateau=True),
                    dict(stopped_by_wall_limit=True), dict(history=[dict(epoch=9000)])):
        with pytest.raises(ValueError):
            report.validate_completed_run(cfg, dict(summary, **changes), payload, 12000)


def test_completed_guard_uses_real_configuration_json_roundtrip():
    from recurrent_rg.config import RunConfig
    cfg = RunConfig.from_dict(dict(data=dict(cache='/not/read.pt', physical_model='ising2d'),
        training=dict(epochs=12000, lr_schedule='multistep', lr_milestones=[6000, 9000],
                      train_modules=['Q', 'C', 'R'])))
    assert cfg.training.lr_milestones == (6000, 9000)
    saved = json.loads(json.dumps(cfg.to_dict()))
    assert saved['training']['lr_milestones'] == [6000, 9000]
    assert saved != cfg.to_dict()  # The prior direct comparison would reject a real run.
    summary = dict(config=saved, actual_epochs=12000, requested_epochs=12000,
                   history=[dict(epoch=12000)])
    report.validate_completed_run(cfg, summary, dict(epoch=12000), 12000)
    bad = json.loads(json.dumps(summary))
    bad['config']['model']['theory_dim'] += 1
    with pytest.raises(ValueError, match='matching config'):
        report.validate_completed_run(cfg, bad, dict(epoch=12000), 12000)


def test_compare_accepts_arm_and_report_paths_and_retains_response(tmp_path):
    specifications = []
    for dim in (1, 2, 3):
        arm = tmp_path/f'dim{dim}'
        folder = arm/'report'
        folder.mkdir(parents=True)
        (arm/'response').mkdir()
        report.write_json(folder/'report.json', dict(status='complete', report_complete=True))
        report.write_json(folder/'metrics.json', dict(checkpoint_epoch=12000, theory_dim=dim,
            closure=dict(test=dict(one_step=dict(nrmse=[.1, .2]))),
            train_representation=[dict(participation_rank=1.)]*3, roots=[]))
        report.write_json(arm/'response'/'comparison.json', dict(independent_physical_response=True))
        specifications.append(f'dim{dim}={arm if dim%2 else folder}')
    destination = tmp_path/'comparison'
    report.compare_reports(specifications, destination)
    saved = json.loads((destination/'comparison.json').read_text())
    assert all(row['independent_physical_response'] for row in saved['physical_response'].values())
    assert json.loads((destination/'report.json').read_text())['report_complete'] is True
    with pytest.raises(FileExistsError):
        report.compare_reports(specifications, destination)


def test_json_never_exports_nan_or_infinity():
    assert report.json_value(dict(x=np.array([np.nan, np.inf, 1.]))) == dict(x=[None, None, 1.])


def test_cache_loader_mmaps_without_repeating_physical_validation(tmp_path, monkeypatch):
    from recurrent_rg.config import RunConfig
    path = tmp_path/'physical.pt'
    configurations = torch.ones((4, 8, 8, 8, 1), dtype=torch.float64)
    torch.save((configurations, torch.arange(4.)[:, None], configurations,
                torch.arange(4., 8.)[:, None]), path)
    cfg = RunConfig.from_dict(dict(data=dict(cache=str(path), physical_model='ising2d'),
        model=dict(dimension=2, block_factor=2), training=dict(sample_batch=8),
        evaluation=dict(samples=8, max_rg_steps=2)))
    def no_full_scan(*args, **kwargs):
        raise AssertionError('Report must not repeat physical-data audit')
    monkeypatch.setattr(torch, 'isfinite', no_full_scan)
    train, test = report.report_datasets(cfg)
    assert train.configurations.shape == test.configurations.shape == (4, 8, 8, 8, 1)
    np.testing.assert_array_equal(test.couplings.numpy().ravel(), np.arange(4., 8.))
