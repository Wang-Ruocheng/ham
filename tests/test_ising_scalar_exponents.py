import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising_scalar_exponents as reporter


class PolynomialRG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.tensor(.5, dtype=torch.float64))

    def advance_theory(self, g):
        return g + self.coefficient * (g ** 3 - g)


class LinearRG(torch.nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def advance_theory(self, g):
        return self.multiplier * g


def test_formula_uses_spatial_block_factor_and_full_r_multiplier():
    result = reporter.exponent_from_multiplier(3., 3)
    assert result['y'] == pytest.approx(1.)
    assert result['nu'] == pytest.approx(1.)
    assert not result['nu_certified']
    derivative = reporter.full_r_derivative(PolynomialRG().double(), 1.)
    assert derivative['multiplier'] == pytest.approx(2.)
    assert derivative['displacement_derivative'] == pytest.approx(1.)
    assert derivative['finite_difference_h'] == 1e-5
    assert derivative['derivative_verified'] and derivative['root_verified']


@pytest.mark.parametrize('multiplier', [0., .5, -.5, 1., -1.])
def test_stable_and_marginal_roots_cannot_supply_thermal_nu(multiplier):
    result = reporter.exponent_from_multiplier(multiplier, 3)
    assert result['nu'] is None
    assert not result['nu_certified']


def test_negative_relevant_multiplier_cannot_certify_period_or_thermal_exponent():
    result = reporter.exponent_from_multiplier(-3., 3)
    assert result['y'] == pytest.approx(1.)
    assert result['stability'] == 'unstable'
    assert result['nu'] is None
    assert result['sign_reversing']
    assert not result['period_two_certified'] and not result['nu_certified']


def test_all_folded_inverse_coupling_branches_are_retained():
    branches = reporter.inverse_coupling_branches(
        np.array([0., .2, .4, .6]), np.array([0., 2., 0., 2.]), 1.)
    assert branches == pytest.approx([.1, .3, .5])


def test_flat_inverse_coupling_interval_is_reported_without_choosing_one_k():
    intervals = reporter.inverse_coupling_plateau_intervals(
        np.array([.2, .4, .6]), np.array([1., 1., 2.]), 1.)
    assert intervals == [[.2, .4]]


def test_known_kc_is_forward_interpolated_at_each_scale_and_may_drift():
    knots = np.array([.2, .6])
    means = np.array([[1., 2., 3.], [3., 4., 5.]])
    reference = reporter.critical_encoding_reference(LinearRG(.8), knots, means)
    first = 1 + 2 * (reporter.KC - .2) / .4
    assert reference['g_by_scale'] == pytest.approx([first, first + 1, first + 2])
    assert reference['direct_scale_drift'] == pytest.approx([1., 1.])
    assert reference['R_drift_by_scale'] == pytest.approx(-.2 * np.array(reference['g_by_scale']))
    assert not reference['R_root_equivalence_assumed']
    assert reference['numerical_fixed_point_by_scale'] == [False, False, False]


def test_unbracketed_kc_is_not_silently_clamped_or_extrapolated():
    reference = reporter.critical_encoding_reference(
        LinearRG(.8), np.array([0., .1]), np.array([[0., 0.], [1., 1.]]))
    assert reference['g_by_scale'] is None


def report_fixture(tmp_path):
    model = PolynomialRG().double()
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(
        dimension=2, theory_dim=1, categorical_states=0, block_factor=3),
        config=dict(data=dict(coupling_names=['K'])),
        checkpoint=tmp_path / 'checkpoint_last.pt', payload=dict(epoch=4400))
    train = np.array([[[-.4], [-.2], [-.1]], [[.4], [.2], [.1]]])
    test = np.array([[[-.2], [-.1], [-.05]], [[.2], [.1], [.05]]])
    encodings = dict(train=train, test=test, train_labels=np.array([[.2], [.6]]),
                     test_labels=np.array([[.3], [.5]]))
    fixed = dict(roots=[dict(g=[-1.], stability='unstable'), dict(g=[0.], stability='stable')],
                 search_box=[[-1.5], [1.5]], search_note='Previously bounded search')
    metrics = dict(fixed_points=dict(roots=[
        dict(g=[-1.], microscopic_inverse_coupling=[-.3, .3, .55]),
        dict(g=[0.], microscopic_inverse_coupling=[.4]),
        dict(g=[1.], stability='unstable', microscopic_inverse_coupling=[.65])]))
    return loaded, metrics, fixed, encodings


def test_report_keeps_all_roots_branches_and_marks_extrapolation_without_selection(tmp_path):
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    before = {name: value.clone() for name, value in loaded.model.state_dict().items()}
    report = reporter.write_exponent_report(loaded, metrics, fixed, encodings, tmp_path)
    assert [root['g'][0] for root in report['roots']] == [-1., 0., 1.]
    assert report['roots'][0]['microscopic_inverse_coupling'] == [-.3, .3, .55]
    assert [root['outside_training_support'] for root in report['roots']] == [True, False, True]
    assert report['roots'][1]['nu'] is None
    assert report['roots'][0]['nu'] is None
    assert report['roots'][0]['formal_nu_from_extrapolated_positive_multiplier'] > 0
    assert all(root['derivative_verified'] and root['root_verified'] for root in report['roots'])
    assert report['training_input_range'] == [-.4, .4]
    assert report['original_flow_bounds'] == [-1.5, 1.5]
    assert not report['selection']['test_used_for_selection']
    assert not report['selection']['theory_used_for_selection']
    assert report['checkpoint_epoch'] == 4400
    assert report['magnetic_exponents']['y_h'] is None
    assert report['magnetic_exponents']['beta'] is None
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in loaded.model.parameters())
    assert (tmp_path / 'scalar_critical_summary.png').stat().st_size > 0
    saved = json.loads((tmp_path / 'critical_exponents.json').read_text())
    assert len(saved['roots']) == 3


def test_test_encodings_cannot_choose_root_or_checkpoint(tmp_path, monkeypatch):
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    encodings['train'][:, 0, 0] = [-1.1, 1.1]
    monkeypatch.setattr(reporter, 'draw_summary', lambda *args: None)
    before = reporter.write_exponent_report(loaded, metrics, fixed, encodings, tmp_path)
    assert any(root['nu'] is not None for root in before['roots'])
    encodings['test'] *= 100
    after = reporter.write_exponent_report(loaded, metrics, fixed, encodings, tmp_path)
    assert before['checkpoint_epoch'] == after['checkpoint_epoch'] == 4400
    assert [root['g'] for root in before['roots']] == [root['g'] for root in after['roots']]
    assert [root['nu'] for root in before['roots']] == [root['nu'] for root in after['roots']]
    assert before['training_input_range'] == after['training_input_range']


def test_unverified_saved_root_has_no_exponent_inference(tmp_path, monkeypatch):
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    fixed['roots'].append(dict(g=[.8]))
    monkeypatch.setattr(reporter, 'draw_summary', lambda *args: None)
    report = reporter.write_exponent_report(loaded, metrics, fixed, encodings, tmp_path)
    root = next(root for root in report['roots'] if root['g'] == [.8])
    assert root['multiplier'] > 1
    assert not root['root_verified'] and root['nu'] is None


def test_main_reads_saved_reports_and_writes_only_new_outputs(tmp_path, monkeypatch):
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    run = tmp_path / 'run'
    out = run / 'full_report'
    out.mkdir(parents=True)
    (out / 'metrics.json').write_text(json.dumps(metrics))
    (out / 'fixed_points.json').write_text(json.dumps(fixed))
    np.savez_compressed(out / 'encodings.npz', **encodings)
    (run / 'checkpoint_last.pt').write_bytes(b'existing checkpoint')
    originals = {path: path.read_bytes() for path in out.iterdir()}
    calls = []

    def load_saved(source, checkpoint):
        calls.append((source, checkpoint))
        return loaded

    def draw_saved(*args):
        (args[-1] / 'scalar_critical_summary.png').write_bytes(b'test image')

    monkeypatch.setattr(reporter, 'load_run', load_saved)
    monkeypatch.setattr(reporter, 'draw_summary', draw_saved)
    report = reporter.main(['--run', str(run)])
    assert calls == [(run, 'checkpoint_last.pt')]
    assert report['input_sha256'].keys() == {'metrics.json', 'fixed_points.json', 'encodings.npz'}
    assert all(path.read_bytes() == before for path, before in originals.items())
    assert {path.name for path in out.iterdir()} == {
        'metrics.json', 'fixed_points.json', 'encodings.npz',
        'critical_exponents.json', 'scalar_critical_summary.png'}


def test_main_hashes_reports_on_python310_without_hashlib_file_digest(tmp_path, monkeypatch):
    import hashlib
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    run = tmp_path / 'run'
    out = run / 'full_report'
    out.mkdir(parents=True)
    (out / 'metrics.json').write_text(json.dumps(metrics))
    (out / 'fixed_points.json').write_text(json.dumps(fixed))
    np.savez_compressed(out / 'encodings.npz', **encodings)
    (run / 'checkpoint_last.pt').write_bytes(b'existing checkpoint')
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    monkeypatch.setattr(reporter, 'load_run', lambda *args: loaded)
    monkeypatch.setattr(reporter, 'draw_summary', lambda *args: None)
    report = reporter.main(['--run', str(run)])
    for name, digest in report['input_sha256'].items():
        assert digest == hashlib.sha256((out / name).read_bytes()).hexdigest()


@pytest.mark.parametrize('payload_epoch', [200, 400])
def test_preview_cli_uses_exact_numbered_checkpoint_and_isolated_report(
        tmp_path, monkeypatch, payload_epoch):
    loaded, metrics, fixed, encodings = report_fixture(tmp_path)
    loaded.payload['epoch'] = payload_epoch
    run = tmp_path / 'run'
    out = run / 'monitor_previews' / 'epoch_000200'
    out.mkdir(parents=True)
    (out / 'metrics.json').write_text(json.dumps(metrics))
    (out / 'fixed_points.json').write_text(json.dumps(fixed))
    np.savez_compressed(out / 'encodings.npz', **encodings)
    (run / 'checkpoint_last.pt').write_bytes(b'newer checkpoint must not be selected')
    originals = {path: path.read_bytes() for path in out.iterdir()}
    calls = []
    monkeypatch.setattr(reporter, 'load_run',
                        lambda source, checkpoint: calls.append((source, checkpoint)) or loaded)
    monkeypatch.setattr(reporter, 'draw_summary', lambda *args: None)
    argv = ['--run', str(run), '--checkpoint', '200', '--output', str(out)]
    if payload_epoch != 200:
        with pytest.raises(ValueError, match='epoch'):
            reporter.main(argv)
        assert not (out / 'critical_exponents.json').exists()
    else:
        report = reporter.main(argv)
        assert report['checkpoint_epoch'] == 200
        assert (out / 'critical_exponents.json').exists()
    assert calls == [(run, 200)]
    assert all(path.read_bytes() == before for path, before in originals.items())
    assert not (run / 'full_report').exists()
