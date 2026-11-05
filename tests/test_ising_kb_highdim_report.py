"""Small synthetic contracts for KB high-dimensional reports; no saved physical PT."""
from types import SimpleNamespace
from contextlib import nullcontext
import json

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.evaluation import collect_trajectories
from scripts import report_ising_kb_highdim as report


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class LinearMap(torch.nn.Module):
    def __init__(self, matrix, offset=None):
        super().__init__()
        self.register_buffer('matrix', torch.as_tensor(matrix, dtype=torch.float64))
        self.register_buffer('offset', torch.as_tensor(
            np.zeros(len(matrix)) if offset is None else offset, dtype=torch.float64))

    def advance_theory(self, point):
        return point@self.matrix.T+self.offset


def synthetic_grid(dimension=5):
    labels = np.stack(np.meshgrid(np.linspace(.2, .68, 4), np.linspace(-.01, .01, 3), indexing='ij'), -1).reshape(-1, 2)
    labels = np.repeat(labels, 2, axis=0)
    K, B = labels.T
    direct = np.zeros((len(labels), 3, dimension))
    for scale in range(3):
        direct[:, scale, 0] = K-.44
        direct[:, scale, 1] = 10*B
        direct[:, scale, 2] = .1*(K-.44)**2
        for coordinate in range(3, dimension):
            direct[:, scale, coordinate] = .01*coordinate*K*B
    # Split complete cells, retaining both physical replicas together.
    train = np.repeat(np.arange(len(labels)//2)%3 != 1, 2)
    return direct[train], direct[~train], labels[train], labels[~train]


def complete(model, direct):
    rollout = [direct[:, 0]]
    for _ in range(direct.shape[1]-1):
        rollout.append(report.mapped(model, rollout[-1]))
    return dict(direct=direct, rollout=np.stack(rollout, 1), one_step=report.mapped(model, direct[:, :-1]))


@pytest.mark.parametrize('dimension', [5, 10])
def test_standard_b3_collection_exact_rng_order_and_progress(dimension):
    torch.manual_seed(91)
    model = build_model(ModelConfig(dimension=2, theory_dim=dimension, hidden=8,
        block_factor=3, width=3, binary_lookup=True, kernel_sampling='bernoulli'), 2)
    configurations = (torch.randint(0, 2, (3, 4, 9, 9, 1))*2-1).double()
    before = torch.get_rng_state().clone()
    digest = report.model_state_sha256(model)
    expected = collect_trajectories(model, configurations, samples=4, chunk_size=2,
        max_rg_steps=2, sampling_seed=12345)
    progress = []
    actual = report.collect_bounded(model, configurations, samples=4, chunk_size=2,
        max_rg_steps=2, sampling_seed=12345, progress=lambda *args: progress.append(args))
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert [entry[1] for entry in progress] == [0, 2, 3]
    assert torch.equal(before, torch.get_rng_state())
    assert report.model_state_sha256(model) == digest
    assert all(p.grad is None for p in model.parameters())


def test_collection_deadline_checked_between_chunks():
    class CounterDeadline:
        def __init__(self):
            self.calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 3:
                raise report.AnalysisBudgetHold('synthetic budget')
    deadline = CounterDeadline()
    chunks = report.ProgressChunks(torch.ones(3, 2, 3, 3, 1), deadline=deadline)
    iterator = chunks.split(1)
    next(iterator)
    with pytest.raises(report.AnalysisBudgetHold):
        next(iterator)


def test_grid_axes_keep_B_and_all_highdim_coordinates():
    train, test, train_labels, test_labels = synthetic_grid(10)
    axes, means, noise = report.grid_statistics(np.concatenate((train, test)), np.concatenate((train_labels, test_labels)))
    assert means.shape == (4, 3, 3, 10)
    assert noise['within_cell_covariance_degrees_of_freedom'] == 12
    assert np.asarray(noise['replicas_per_cell']).tolist() == [[2, 2, 2]]*4
    np.testing.assert_array_equal(noise['within_cell_covariance_by_scale'], np.zeros((3, 10, 10)))
    assert len(axes[0]) == 4 and len(axes[1]) == 3


def test_pca_and_local_geometry_do_not_use_final_output_for_chart():
    train, _, _, _ = synthetic_grid()
    pca = report.fit_training_pca(train, components=5)
    changed = train.copy(); changed[:, -1] = 1e8
    other = report.fit_training_pca(changed, components=5)
    np.testing.assert_array_equal(pca['components'], other['components'])
    geometry = report.local_geometry(LinearMap(np.diag([2., .5, .3, -.2, .1])), train)
    assert len(geometry) == 3
    assert geometry[0]['neighbors'] == min(24, len(train))
    assert np.asarray(geometry[0]['T2_compressed_DR']).shape == (len(train), 2, 2)
    assert len(geometry[0]['T2_normal_leakage_fraction']) == len(train)
    assert 'physical_dimension' not in geometry[0]


def test_KB_tangent_normal_residuals_recompose_full_error():
    K, B = np.linspace(-1, 1, 3), np.linspace(-.2, .2, 3)
    grid = np.stack(np.meshgrid(K, B, indexing='ij'), -1)
    means = np.zeros((3, 3, 3, 5))
    means[..., :2] = grid[:, :, None]
    model = LinearMap(np.eye(5), [0., 0., .2, 0., 0.])
    geometry = report.kb_tangent_residuals(model, (K, B), means)
    np.testing.assert_allclose(geometry['displacement_tangent'], 0., atol=1e-12)
    np.testing.assert_allclose(geometry['closure_residual_normal'], geometry['closure_residual'])
    np.testing.assert_allclose(geometry['closure_residual_tangent']+geometry['closure_residual_normal'],
                               geometry['closure_residual'])
    assert np.all(geometry['KB_diagnostic_tangent_rank'] == 2)


def test_full_coordinate_roots_keep_negative_and_complex_spectrum(tmp_path):
    train, test, train_labels, test_labels = synthetic_grid()
    axes, means, _ = report.grid_statistics(np.concatenate((train, test)), np.concatenate((train_labels, test_labels)))
    matrix = np.diag([1.7, -.5, .3, .2, .2])
    matrix[3, 4], matrix[4, 3] = -1.2, 1.2
    roots = report.find_roots(LinearMap(matrix), train, axes, means, max_seeds=3,
        partial_path=tmp_path/'roots.partial.json')
    assert roots['seed_count'] == 3 and roots['max_nfev_per_seed'] == 200
    assert roots['search_complete'] is False and roots['exhaustive_search_claim'] is False
    assert len(roots['roots']) == 1
    row = roots['roots'][0]
    assert len(row['eigenvalues']) == 5
    assert row['nonreal_count'] == 2 and row['negative_real_count'] == 1
    assert row['root_verified'] and row['derivative_verified']
    assert np.asarray(row['jacobian']).shape == (5, 5)
    assert 'nearest_grid_points_by_scale' in row
    assert 'exact_inverse_K_branches' not in row
    assert 'thermal_nu' not in row and 'conditional_inverse_y' not in row


def test_projection_zero_cannot_create_a_root_and_seed_budget_is_label_free():
    train, test, train_labels, test_labels = synthetic_grid()
    axes, means, _ = report.grid_statistics(np.concatenate((train, test)), np.concatenate((train_labels, test_labels)))
    model = LinearMap(np.eye(5), [0., 0., .2, 0., 0.])
    roots = report.find_roots(model, train, axes, means, max_seeds=2)
    assert roots['roots'] == []
    assert all(attempt['residual_norm'] == pytest.approx(.2) for attempt in roots['attempts'])
    changed = report.find_roots(model, train, (axes[0]+100, axes[1]), means, max_seeds=2)
    assert roots['seed_flat_training_input_indices'] == changed['seed_flat_training_input_indices']


def test_small_report_writes_KB_maps_full_arrays_losses_and_no_exponent_fields(tmp_path):
    model = LinearMap(np.diag([1.7, .5, .3, -.2, .1]))
    train, test, train_labels, test_labels = synthetic_grid()
    loaded = SimpleNamespace(model=model, checkpoint=tmp_path/'training_step_12000.pt',
        payload=dict(epoch=12000), model_config=SimpleNamespace(block_factor=3))
    history = [dict(epoch=12000, train_total=1., test_total=1.2, train_rsmi_bits=.1,
                    test_rsmi_bits=.09, train_reconstruction_mse_level_0=.02)]
    before = report.model_state_sha256(model)
    out = tmp_path/'report'
    result = report.write_report(loaded, complete(model, train), complete(model, test),
        train_labels, test_labels, history, out, lattice_sizes=[36, 12, 4],
        root_options=dict(max_seeds=2))
    for name in ('encodings.npz', 'grid_encodings.npz', 'fixed_points.json', 'loss_history.json',
                 'metrics.json', 'coupling_by_scale.png', 'coupling_all_coordinates.png',
                 'R_displacement.png', 'R_vs_g.png', 'representation_diagnostics.png', 'loss_comparison.png'):
        assert (out/name).is_file()
    arrays = np.load(out/'grid_encodings.npz')
    assert arrays['mean_by_scale'].shape == (4, 3, 3, 5)
    assert arrays['closure_residual'].shape == (4, 3, 2, 5)
    assert result['pca']['fit_source'].startswith('TRAIN')
    text = (out/'metrics.json').read_text()
    assert 'thermal_nu' not in text and 'conditional_inverse_y' not in text
    assert 'selected_thermal_eigenvalue' not in text
    assert json.loads((out/'loss_history.json').read_text()) == history
    assert report.model_state_sha256(model) == before
    with pytest.raises(FileExistsError):
        report.write_report(loaded, complete(model, train), complete(model, test),
            train_labels, test_labels, history, out)


def test_deadline_hold_never_claims_no_roots_or_complete(tmp_path, monkeypatch):
    args = SimpleNamespace(run=tmp_path/'run', output=tmp_path/'out',
        progress_path=tmp_path/'progress.json', checkpoint=12000)
    monkeypatch.setattr(report, 'sha256', lambda path: 'abc')
    def hold(*args, **kwargs):
        raise report.AnalysisBudgetHold('synthetic budget')
    monkeypatch.setattr(report, 'load_run', hold)
    deadline = report.Deadline(30)
    progress = report.Progress(args.progress_path, deadline)
    result = report._run(args, deadline, progress, {})
    assert result['status'] == 'analysis_budget_hold'
    assert result['report_complete'] is False
    assert json.loads((args.output/'report.json').read_text())['report_complete'] is False
    assert result['completed_encoding_passes'] == []


def test_CLI_budget_hold_returns_distinct_exit75(tmp_path, monkeypatch):
    monkeypatch.setattr(report, '_run', lambda *args: dict(status='analysis_budget_hold', report_complete=False))
    monkeypatch.setattr(report, 'single_thread_control', lambda: nullcontext({}))
    with pytest.raises(SystemExit) as held:
        report.main(['--run', str(tmp_path/'run'), '--output', str(tmp_path/'out'),
                     '--progress-path', str(tmp_path/'progress.json')])
    assert held.value.code == 75
