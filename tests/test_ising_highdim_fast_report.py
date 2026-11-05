"""CPU report optimization cannot weaken data, root or full-spectrum rules."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising_highdim as original
from scripts import report_ising_highdim_fast as fast


FROZEN_REPORT_FUNCTIONS = {name: getattr(original, name) for name in (
    '_full_jacobian', 'full_r_jacobian', 'find_roots', 'plot_maps', 'plot_losses',
    'plot_representation', 'plot_spectrum', 'closure_details', 'architecture_diagnostics',
    'representation_statistics', 'energy_path_diagnostics')}


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    for name, function in FROZEN_REPORT_FUNCTIONS.items():
        assert getattr(original, name) is function, f'测试前历史模块对象泄漏：{name}'
    yield
    for name, function in FROZEN_REPORT_FUNCTIONS.items():
        assert getattr(original, name) is function, f'测试后历史模块对象泄漏：{name}'


class ResidualMLP(torch.nn.Module):
    def __init__(self, dimension, hidden=9):
        super().__init__()
        self.theory_rg = torch.nn.Sequential(torch.nn.Linear(dimension, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, dimension)).double()

    def advance_theory(self, point):
        return point+self.theory_rg(point)


@pytest.mark.parametrize('dimension', [1, 5, 10, 100])
def test_analytic_full_DR_matches_autograd_and_centered_FD(dimension):
    torch.manual_seed(91)
    model = ResidualMLP(dimension, hidden=96)
    before = original.model_state_sha256(model)
    point = np.random.default_rng(91).normal(0, .2, dimension)
    derivative = fast.AnalyticResidualJacobian(model, fast.Deadline(30))
    checked = derivative.verified(model, point)
    expected = original.full_r_jacobian(model, point)
    assert checked['derivative_verified'] and checked['analytic_autograd_verified']
    assert checked['finite_difference_h'] == 1e-5
    np.testing.assert_allclose(checked['jacobian'], expected['jacobian'], rtol=1e-12, atol=1e-12)
    assert original.model_state_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    if dimension == 100:
        assert np.linalg.matrix_rank(np.asarray(checked['jacobian'])-np.eye(dimension)) <= 96


def test_unknown_architecture_rejected_not_silently_guessed():
    model = ResidualMLP(5)
    model.theory_rg[1] = torch.nn.ReLU()
    with pytest.raises(ValueError, match='未知 R'):
        fast.AnalyticResidualJacobian(model, fast.Deadline(30))


def test_deadline_expires_without_sleeping_and_cannot_mean_no_roots():
    clock = [10.]
    deadline = fast.Deadline(5., clock=lambda: clock[0])
    deadline.check()
    clock[0] = 15.
    with pytest.raises(fast.AnalysisBudgetHold):
        deadline.check()
    for value in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError):
            fast.Deadline(value)


def test_hold_overrides_old_failed_or_complete_and_preserves_partial(tmp_path):
    clock = [0.]
    deadline = fast.Deadline(1., clock=lambda: clock[0])
    progress = fast.Progress(tmp_path/'progress.json', deadline)
    progress('完整坐标求根', 3, 64)
    assert not (tmp_path/'report').exists()
    output = tmp_path/'report'
    output.mkdir()
    fast.atomic_json(output/'report.json', dict(status='complete', report_complete=True))
    partial = tmp_path/'partial.npz'
    fast.atomic_npz(partial, direct=np.ones((16, 3, 5)))
    clock[0] = 2.
    record = fast.budget_hold(output, progress, deadline, partial_files=[partial])
    assert record['status'] == 'analysis_budget_hold' and record['report_complete'] is False
    assert record['requires_user_action'] and record['thermal_nu'] is None
    assert record['partial_files'] == [str(partial)]
    for name in ('status.json', 'report.json'):
        assert json.loads((output/name).read_text())['report_complete'] is False
    np.testing.assert_array_equal(np.load(partial)['direct'], np.ones((16, 3, 5)))


def _trajectories(dimension):
    rng = np.random.default_rng(91)
    knots = np.linspace(.2, .68, 36)
    values = rng.normal(0, .03, (36, 3, dimension))
    return values, knots


@pytest.mark.parametrize('dimension', [1, 5, 10])
def test_same_seeds_bounds_caps_acceptance_and_roots_as_old_path(dimension, tmp_path):
    torch.manual_seed(91)
    model = ResidualMLP(dimension, hidden=max(12, dimension))
    for layer in model.theory_rg:
        if isinstance(layer, torch.nn.Linear):
            with torch.no_grad():
                layer.bias.zero_()
    direct, knots = _trajectories(dimension)
    old = original.find_roots(model, direct, knots, direct, max_seeds=3)
    deadline = fast.Deadline(60)
    progress = fast.Progress(tmp_path/'progress.json', deadline)
    derivative = fast.AnalyticResidualJacobian(model, deadline)
    new = fast.find_roots(model, direct, knots, direct, derivative=derivative,
        deadline=deadline, progress=progress, max_seeds=3,
        partial_path=tmp_path/'roots.partial.json')
    for key in ('seed_flat_training_input_indices', 'search_box', 'max_nfev_per_seed',
                'max_analyzed_roots', 'acceptance_residual_l2', 'max_seed_count'):
        assert new[key] == old[key]
    assert new['accepted_unique_root_count'] == old['accepted_unique_root_count']
    assert len(new['attempts']) == len(old['attempts']) == 3
    assert new['search_complete'] is False and new['exhaustive_search_claim'] is False
    for before, after in zip(old['roots'], new['roots']):
        np.testing.assert_allclose(before['g'], after['g'], atol=1e-6, rtol=0)
        assert after['derivative_verified'] and after['root_verified']
        assert after['thermal_nu'] is None and after['selected_thermal_eigenvalue'] is None
        assert len(after['eigenvalues']) == dimension


def test_root_search_timeout_saves_attempts_not_empty_scientific_conclusion(tmp_path):
    model = ResidualMLP(5)
    direct, knots = _trajectories(5)
    deadline = fast.Deadline(60)
    progress = fast.Progress(tmp_path/'progress.json', deadline)
    derivative = fast.AnalyticResidualJacobian(model, deadline)
    def stop(*args):
        raise fast.AnalysisBudgetHold('测试预算中断')
    with pytest.raises(fast.AnalysisBudgetHold):
        fast.find_roots(model, direct, knots, direct, derivative=derivative,
            deadline=deadline, progress=stop, partial_path=tmp_path/'roots.partial.json')
    partial = json.loads((tmp_path/'roots.partial.json').read_text())
    assert partial['status'] == 'partial' and partial['report_complete'] is False
    assert len(partial['seed_flat_training_input_indices']) == 64
    assert partial['max_nfev_per_seed'] == 200
    assert partial['max_analyzed_roots'] == 16
    assert partial['search_complete'] is False


def test_context_restores_frozen_module_functions_even_on_error(tmp_path):
    model = ResidualMLP(5)
    names = ('_full_jacobian', 'full_r_jacobian', 'find_roots', 'plot_maps', 'closure_details')
    saved = {name: getattr(original, name) for name in names}
    deadline = fast.Deadline(30)
    progress = fast.Progress(tmp_path/'progress.json', deadline)
    with pytest.raises(RuntimeError, match='exit'):
        with fast.fast_report_context(model, deadline, progress):
            assert original.find_roots is not saved['find_roots']
            original.full_r_jacobian(model, np.zeros(5))
            raise RuntimeError('exit')
    for name, value in saved.items():
        assert getattr(original, name) is value


def test_all_complex_negative_and_unit_modes_are_retained():
    dimension = 100
    model = ResidualMLP(dimension, hidden=dimension)
    matrix = np.diag(np.linspace(.2, .8, dimension))
    matrix[0, 0] = 1.7
    matrix[1, 1] = -.5
    matrix[2:4, 2:4] = [[.2, -1.2], [1.2, .2]]
    matrix[4, 4] = 1.
    with torch.no_grad():
        for index in (0, 2):
            model.theory_rg[index].weight.copy_(torch.eye(dimension))
            model.theory_rg[index].bias.zero_()
        model.theory_rg[4].weight.copy_(torch.from_numpy(matrix-np.eye(dimension)))
        model.theory_rg[4].bias.zero_()
    derivative = fast.AnalyticResidualJacobian(model, fast.Deadline(30))
    result = derivative.verified(model, np.zeros(dimension))
    counts = original.stability_counts(result['jacobian'])
    assert counts['nonreal_count'] == 2 and counts['negative_real_count'] == 1
    assert counts['neutral_direction_count'] == 1
    values = original.eigenvalue_records(result['jacobian'], [], block_factor=2,
        verified=True, support_candidate=False)
    assert len(values) == 100
    assert result['root_verified'] and result['derivative_verified']


def test_cli_rejects_silent_data_repeat_checkpoint_thread_changes(tmp_path):
    base = ['--run', str(tmp_path/'run'), '--output', str(tmp_path/'out'),
            '--progress-path', str(tmp_path/'progress.json')]
    for option, value in (('--sampling-repeats', '4'), ('--threads', '2'), ('--checkpoint', '8000')):
        with pytest.raises(SystemExit):
            fast.main(base+[option, value])
    with pytest.raises(SystemExit):
        fast.main(['--run', str(tmp_path/'run'), '--output', str(tmp_path/'out'),
                   '--progress-path', str(tmp_path/'out'/'progress.json')])


def test_full_existing_writer_still_produces_verifiable_public_report(tmp_path, monkeypatch):
    from scripts.run_ising_highdim import report_receipt
    torch.manual_seed(91)
    model = ResidualMLP(5, hidden=12)
    for layer in model.theory_rg:
        if isinstance(layer, torch.nn.Linear):
            with torch.no_grad():
                layer.bias.zero_()
    direct, labels = _trajectories(5)
    values = dict(direct=direct, one_step=original.mapped(model, direct[:, :-1]))
    rollout = [direct[:, 0]]
    for _ in range(2):
        rollout.append(original.mapped(model, rollout[-1]))
    values['rollout'] = np.stack(rollout, 1)
    path = tmp_path/'run'/'training_step_12000.pt'
    path.parent.mkdir()
    path.write_bytes(b'frozen synthetic model fixture')
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=2, hidden=12),
        payload=dict(epoch=12000), checkpoint=path)
    deadline = fast.Deadline(60)
    progress = fast.Progress(tmp_path/'progress.json', deadline)
    output = tmp_path/'report'
    with fast.fast_report_context(model, deadline, progress, tmp_path/'rootpartial.json'):
        # Small fixture only: the real CLI never exposes a reduced root budget.
        root_function = original.find_roots
        # The temporary fixture patch MUST exit before fast_report_context.
        # Otherwise pytest's later teardown restores this fast closure after
        # the production context has already restored the real historical API.
        with monkeypatch.context() as fixture_patch:
            fixture_patch.setattr(original, 'find_roots',
                lambda *a, **kw: root_function(*a, **kw, max_seeds=2))
            result = original.write_report(loaded, values, values, labels, labels,
                [dict(epoch=12000, train_total=.2, test_total=.3, train_rsmi_bits=.1, test_rsmi_bits=.11,
                      train_reconstruction_mse_level_0=.01, test_reconstruction_mse_level_0=.02)],
                output, lattice_sizes=[108, 54, 27])
    receipt = report_receipt(output, 5)
    assert len(receipt['files']) == 16
    assert result['freeze_verification']['model_gradients_none'] is True
    with np.load(output/'encodings.npz') as arrays:
        np.testing.assert_array_equal(arrays['train'], direct)
    assert len(list(output.glob('*.png'))) == 7


def test_thread_control_fails_closed_without_environment_or_optional_library(monkeypatch):
    import builtins
    ordinary_import = builtins.__import__
    def without_optional(name, *args, **kwargs):
        if name == 'threadpoolctl':
            raise ModuleNotFoundError('test no optional dependency')
        return ordinary_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_optional)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    with fast.single_thread_control() as receipt:
        assert receipt['torch_num_threads'] == 1
        assert all(value == '1' for value in receipt['startup_environment'].values())
        assert 'ModuleNotFoundError' in receipt['optional_threadpoolctl']
    monkeypatch.delenv('NUMBA_NUM_THREADS')
    with pytest.raises(RuntimeError, match='拒绝不受控CPU报告'):
        with fast.single_thread_control():
            pytest.fail('Uncontrolled reporting cannot run')
