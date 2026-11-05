"""High-dimensional reports keep full coordinates and a fixed, non-exponential budget."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from scripts import report_ising_highdim as report


class LinearMap(torch.nn.Module):
    def __init__(self, matrix, offset=None):
        super().__init__()
        self.register_buffer('matrix', torch.as_tensor(matrix, dtype=torch.float64))
        self.register_buffer('offset', torch.as_tensor(
            np.zeros(len(matrix)) if offset is None else offset, dtype=torch.float64))

    def advance_theory(self, point):
        return point@self.matrix.T+self.offset


def trajectories(dimension, count=18):
    rng = np.random.default_rng(91)
    knots = np.linspace(.2, .68, count)
    t = knots-report.KC
    direct = np.stack([np.stack([t*(1+.2*scale)+.02*coordinate*t*t+
        rng.normal(0, .001, count) for coordinate in range(dimension)], -1)
        for scale in range(3)], 1)
    return direct, knots


def complete_trajectories(model, direct):
    rollout = [direct[:, 0]]
    for _ in range(direct.shape[1]-1):
        rollout.append(report.mapped(model, rollout[-1]))
    return dict(direct=direct, rollout=np.stack(rollout, 1),
                one_step=report.mapped(model, direct[:, :-1]))


@pytest.mark.parametrize('dimension', [1, 5, 10, 100])
def test_full_discrete_derivative_retains_identity_and_all_dimensions(dimension):
    diagonal = np.linspace(.2, .8, dimension)
    diagonal[0] = 1.7
    matrix = np.diag(diagonal)
    checked = report.full_r_jacobian(LinearMap(matrix), np.zeros(dimension))
    assert checked['root_verified'] and checked['derivative_verified']
    np.testing.assert_allclose(checked['jacobian'], matrix)
    np.testing.assert_allclose(checked['finite_difference_jacobian'], matrix)
    assert checked['finite_difference_h'] == 1e-5


def test_fake_saddle_and_negative_complex_modes_are_not_dropped():
    matrix = np.array([[1.7, 0., 0., 0.], [0., -.5, 0., 0.],
                       [0., 0., .2, -1.2], [0., 0., 1.2, .2]])
    counts = report.stability_counts(matrix)
    assert counts['stability'] == 'saddle'
    assert counts['expanding_direction_count'] == 3
    assert counts['contracting_direction_count'] == 1
    assert counts['nonreal_count'] == 2
    assert counts['negative_real_count'] == 1
    assert counts['positive_real_expanding_count'] == 1
    counts = report.stability_counts(np.diag([2., .5, 1.]))
    assert counts['stability'] == 'nonhyperbolic' and counts['neutral_direction_count'] == 1


def test_seed_budget_has_no_labels_or_final_output_scale():
    direct, _ = trajectories(100, count=45)
    seeds, indices = report.select_training_seeds(direct)
    assert seeds.shape == (64, 100)
    np.testing.assert_array_equal(seeds, direct[:, :-1].reshape(-1, 100)[indices])
    changed_final = direct.copy()
    changed_final[:, -1] = 1e9
    changed_seeds, changed_indices = report.select_training_seeds(changed_final)
    np.testing.assert_array_equal(seeds, changed_seeds)
    np.testing.assert_array_equal(indices, changed_indices)
    with pytest.raises(ValueError, match='64'):
        report.select_training_seeds(direct, max_seeds=65)


def test_dim100_solver_is_linear_budget_and_never_delaunay(tmp_path, monkeypatch):
    import itertools
    import scipy.optimize
    import scipy.spatial
    def forbidden(*args, **kwargs):
        raise AssertionError('No exponential grid or high-dimensional triangulation')
    monkeypatch.setattr(itertools, 'product', forbidden)
    monkeypatch.setattr(scipy.spatial, 'Delaunay', forbidden)
    original = scipy.optimize.least_squares
    attempts = []
    def bounded(fun, x0, **kwargs):
        assert kwargs['max_nfev'] == 200
        assert np.asarray(x0).shape == (100,)
        assert np.all(x0 >= kwargs['bounds'][0]) and np.all(x0 <= kwargs['bounds'][1])
        attempts.append(np.array(x0))
        return original(fun, x0, **kwargs)
    monkeypatch.setattr(scipy.optimize, 'least_squares', bounded)
    direct, knots = trajectories(100)
    matrix = np.diag([1.7]+[.5]*99)
    roots = report.find_roots(LinearMap(matrix), direct, knots, direct, max_seeds=2)
    assert len(attempts) == 2
    assert roots['search_complete'] is False
    assert roots['root_analysis_truncated'] is False
    assert len(roots['roots']) == 1
    row = roots['roots'][0]
    assert row['stability'] == 'saddle'
    assert row['expanding_direction_count'] == 1
    assert len(row['eigenvalues']) == 100
    assert row['selected_thermal_eigenvalue'] is None and row['thermal_nu'] is None
    assert row['convex_hull_computed'] is False
    assert row['local_pca']['input_scales'] == [0, 1]
    assert row['local_pca']['numerical_rank'] <= 35
    assert row['residual_norm'] <= 1e-8
    assert not row['physical_support_certified']


def test_projection_intersection_cannot_create_a_full_space_root():
    knots = np.linspace(.2, .68, 12)
    direct = np.stack([np.column_stack((knots, knots**2, np.zeros(12)))
                       for _ in range(3)], 1)
    model = LinearMap(np.eye(3), [0., 0., .1])
    pca = report.fit_training_pca(direct)
    delta = report.mapped(model, direct)-direct
    projected = report.project_displacement(delta, pca)
    np.testing.assert_allclose(projected[..., :2], 0., atol=1e-12)
    assert np.linalg.norm(delta[0, 0]) == pytest.approx(.1)
    roots = report.find_roots(model, direct, knots, direct, max_seeds=3)
    assert roots['roots'] == []
    assert all(row['residual_norm'] == pytest.approx(.1) for row in roots['attempts'])


def test_support_does_not_use_test_or_final_scale():
    direct, _ = trajectories(5)
    point = direct[5, 0]
    original = report.training_support(point, direct)
    direct[:, -1] = 1e12
    changed = report.training_support(point, direct)
    assert original == changed
    assert original['nearest_training_input_distance'] == 0.


def test_all_K_inverse_branches_and_nonzero_projection_distances_are_saved():
    projection = report.curve_projections([0., 1., 2., 3., 4.],
        [[0.], [1.], [0.], [0.], [1.]], [0.])
    assert projection['exact_inverse_K_branches'] == [0., 2., 3.]
    assert projection['exact_inverse_K_intervals'] == [[2., 3.]]
    projection = report.curve_projections([0., 1.], [[0., 0.], [1., 0.]], [.5, .1])
    assert projection['exact_inverse_K_branches'] == []
    assert projection['nearest_distance'] == pytest.approx(.1)


def test_root_cap_is_explicit_for_nonisolated_identity_roots():
    direct, knots = trajectories(1)
    roots = report.find_roots(LinearMap([[1.]]), direct, knots, direct,
                              max_seeds=8, max_roots=3)
    assert roots['accepted_unique_root_count'] == 8
    assert roots['retained_root_count'] == 3
    assert roots['root_analysis_truncated'] is True
    assert roots['truncated_unique_root_count'] == 5
    assert np.asarray(roots['all_accepted_root_coordinates']).shape == (8, 1)
    assert roots['all_accepted_coordinates_retained'] is True
    assert all(row['neutral_direction_count'] == 1 for row in roots['roots'])
    assert all(not row['isolated_root_certified'] for row in roots['roots'])


def test_architecture_finds_dim100_hidden96_affine_obstruction():
    model = torch.nn.Module()
    model.theory_rg = torch.nn.Sequential(torch.nn.Linear(100, 96), torch.nn.Tanh(),
                                          torch.nn.Linear(96, 100)).double()
    model.theory_encoder = torch.nn.Module()
    model.theory_encoder.distribution_decoder = torch.nn.Linear(100, 16).double()
    model.theory_encoder.head = torch.nn.Sequential(torch.nn.Linear(12, 96), torch.nn.Tanh(),
                                                    torch.nn.Linear(96, 100)).double()
    with torch.no_grad():
        last = model.theory_rg[-1]
        last.weight.zero_(); last.bias.zero_()
        last.weight[:96].copy_(torch.eye(96, dtype=torch.float64))
        last.bias[99] = .2
    values = report.architecture_diagnostics(model, theory_dim=100, hidden=96)
    residual = values['R_residual_output']
    assert residual['numerical_rank'] == 96
    assert residual['structural_unit_eigenvalue_count_lower_bound'] == 4
    assert residual['displacement_norm_lower_bound'] == pytest.approx(.2)
    assert residual['output_affine_span_allows_root_at_tolerance'] is False
    assert values['affine_decoder']['numerical_rank'] <= 16
    assert values['affine_decoder']['latent_nullity'] >= 84
    assert values['affine_decoder']['normalized_probability_target_degrees_of_freedom_upper_bound'] == 15
    assert values['Q_final_affine']['numerical_rank'] <= 96
    assert values['Q_final_affine']['pooled_local_feature_count'] == 12
    assert values['Q_final_affine']['local_differential_rank_upper_bound'] == 12


def test_train_only_pca_and_vectors_are_not_centered():
    direct, _ = trajectories(10)
    pca = report.fit_training_pca(direct)
    np.testing.assert_allclose(pca['center'], direct[:, :-1].reshape(-1, 10).mean(0))
    zero_vectors = np.zeros((5, 10))
    np.testing.assert_array_equal(report.project_displacement(zero_vectors, pca), np.zeros((5, 6)))
    shifted = direct.copy(); shifted[:, -1] = 1e8
    other = report.fit_training_pca(shifted)
    np.testing.assert_array_equal(pca['components'], other['components'])


def test_solver_seed_locations_do_not_depend_on_evaluation_labels(monkeypatch):
    import scipy.optimize
    direct, knots = trajectories(5)
    captured = []
    original = scipy.optimize.least_squares
    def observe(fun, x0, **kwargs):
        captured.append(np.array(x0))
        return original(fun, x0, **kwargs)
    monkeypatch.setattr(scipy.optimize, 'least_squares', observe)
    model = LinearMap(np.diag([1.7, .5, .3, .2, .1]))
    report.find_roots(model, direct, knots, direct, max_seeds=2)
    report.find_roots(model, direct, knots+50, direct[::-1], max_seeds=2)
    np.testing.assert_array_equal(captured[:2], captured[2:])


@pytest.mark.parametrize('dimension', [1, 5, 100])
def test_public_report_exports_all_coordinates_full_spectrum_and_freeze_checks(tmp_path, dimension, monkeypatch):
    direct, labels = trajectories(dimension)
    model = LinearMap(np.diag([1.7]+[.5]*(dimension-1)))
    path = tmp_path/'run'/'training_step_12000.pt'
    path.parent.mkdir()
    path.write_bytes(b'frozen synthetic source placeholder')
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=2, hidden=96),
        payload=dict(epoch=12000), checkpoint=path)
    values = complete_trajectories(model, direct)
    original = report.find_roots
    monkeypatch.setattr(report, 'find_roots', lambda *a, **kw: original(*a, **kw, max_seeds=2))
    out = tmp_path/'report'
    result = report.write_report(loaded, values, values, labels, labels,
        [dict(epoch=12000, train_total=.2, test_total=.3, train_rsmi_bits=.1, test_rsmi_bits=.11,
              train_reconstruction_mse_level_0=.01, test_reconstruction_mse_level_0=.02)],
        out, lattice_sizes=[108, 54, 27])
    completion = json.loads((out/'report.json').read_text())
    assert completion['report_complete'] is True
    assert result['theory_dim'] == dimension and result['thermal_nu'] is None
    assert result['freeze_verification']['model_state_unchanged'] is True
    assert result['freeze_verification']['model_gradients_none'] is True
    assert result['freeze_verification']['checkpoint_sha256_before'] == report.sha256(path)
    assert len(list(out.glob('*.png'))) == 7
    assert not list(out.glob('*.pt'))
    with np.load(out/'encodings.npz') as arrays:
        np.testing.assert_array_equal(arrays['train'], direct)
        assert arrays['train'].shape[-1] == dimension
    with np.load(out/'trajectories.npz') as arrays:
        assert arrays['mean_by_scale'].shape == (len(labels), 3, dimension)
        np.testing.assert_allclose(arrays['projected_true_displacement'],
            arrays['displacement_of_mean']@arrays['pca_components'].T)
    with np.load(out/'fixed_point_arrays.npz') as arrays:
        assert arrays['eigenvalues'].shape == (1, dimension)
        assert arrays['full_DR'].shape == (1, dimension, dimension)
        assert arrays['eigenvectors'].shape == (1, dimension, dimension)
    before = (out/'report.json').read_bytes()
    with pytest.raises(FileExistsError):
        report.write_report(loaded, values, values, labels, labels, [], out)
    assert (out/'report.json').read_bytes() == before


def test_existing_output_refused_before_loading_or_sampling(tmp_path, monkeypatch):
    out = tmp_path/'exists'; out.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('Should refuse before loading or collecting')
    monkeypatch.setattr(report, 'load_run', forbidden)
    monkeypatch.setattr(report, 'collect_trajectories', forbidden)
    with pytest.raises(FileExistsError):
        report.main(['--run', str(tmp_path/'missing'), '--output', str(out)])


def test_comparison_supports_arbitrary_completed_unique_dimensions_and_no_overwrite(tmp_path):
    specifications = []
    for dimension in (1, 5, 10, 100):
        root = tmp_path/f'dim{dimension}'
        folder = root/'report'; folder.mkdir(parents=True)
        report.write_json(folder/'report.json', dict(status='complete', report_complete=True))
        report.write_json(folder/'metrics.json', dict(checkpoint_epoch=12000, theory_dim=dimension,
            closure=dict(test=dict(one_step=dict(nrmse=[.1, .2]))),
            train_representation=[dict(participation_rank=1.)]*3, roots=[]))
        specifications.append(f'dim{dimension}={root if dimension == 5 else folder}')
    output = tmp_path/'comparison'
    report.compare_reports(specifications, output)
    assert json.loads((output/'report.json').read_text())['report_complete'] is True
    assert len(json.loads((output/'comparison.json').read_text())['reports']) == 4
    with pytest.raises(FileExistsError):
        report.compare_reports(specifications, output)
    report.compare_reports(specifications[:3], tmp_path/'three_arms')


def test_comparison_rejects_incomplete_or_partial_endpoint(tmp_path):
    folder = tmp_path/'arm'; folder.mkdir()
    report.write_json(folder/'report.json', dict(status='complete', report_complete=True))
    report.write_json(folder/'metrics.json', dict(checkpoint_epoch=9000, theory_dim=100))
    with pytest.raises(ValueError, match='12000'):
        report.compare_reports([f'control={folder}'], tmp_path/'comparison')
    assert not (tmp_path/'comparison').exists()


def test_source_mutation_fails_instead_of_publishing_complete_report(tmp_path, monkeypatch):
    direct, labels = trajectories(1)
    model = LinearMap([[1.7]])
    path = tmp_path/'source.pt'; path.write_bytes(b'before')
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=2),
        payload=dict(epoch=12000), checkpoint=path)
    values = complete_trajectories(model, direct)
    original = report.plot_spectrum
    def mutate(roots, output):
        original(roots, output)
        path.write_bytes(b'after')
    monkeypatch.setattr(report, 'plot_spectrum', mutate)
    with pytest.raises(ValueError, match='fingerprint'):
        report.write_report(loaded, values, values, labels, labels, [], tmp_path/'report')
    assert json.loads((tmp_path/'report'/'status.json').read_text())['status'] == 'failed'
    assert not (tmp_path/'report'/'report.json').exists()
