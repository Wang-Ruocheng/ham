"""Tiny saved-array checks; no physical sampling, training, launch or server IO."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

from scripts import report_k_supervised_continuation as reporter
from scripts.report_ising_zh import chinese_figures, select_chinese_font


class LinearR(torch.nn.Module):
    def advance_theory(self, x):
        return 1.7 * x


def saved_arrays():
    k = np.linspace(reporter.KC - .055, reporter.KC + .055, 25)
    deviations = np.array([-.0003, -.0001, .0001, .0003])
    blocks = np.stack([np.stack([1.7**scale * (value - reporter.KC + deviations)
                                for scale in range(3)], axis=-1) for value in k])
    train = np.arange(len(k)) % 3 != 1
    return dict(train=blocks[train].reshape(-1, 3, 1),
        test=blocks[~train].reshape(-1, 3, 1),
        train_labels=np.repeat(k[train], len(deviations))[:, None],
        test_labels=np.repeat(k[~train], len(deviations))[:, None])


def checkpoint_fixture():
    cfg = dict(model=dict(dimension=2, theory_dim=1, block_factor=2, hidden=96,
        width=3, kernel_layout='nonoverlapping', fixed_kernel=False,
        kernel_initialization='random', kernel_sampling='bernoulli',
        affine_reconstruction=True, binary_lookup=True, categorical_states=0),
        training=dict(max_rg_steps=2, system_batch=16, sample_batch=128, seed=91,
            gradient_mode='qc_r_separate', train_modules=['Q', 'C', 'R'],
            learning_rate=.001, lr_schedule='multistep', lr_gamma=.3,
            lr_milestones=[6000, 9000]),
        evaluation=dict(samples=128, max_rg_steps=2), data=dict(coupling_names=['K']))
    cfg['loss'] = dict(rollout_weight=5., kernel_sign_weight=1.,
        kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
        balanced_reconstruction_weight=0., isotropy_weight=0., tangent_weight=0.,
        orientation_weight=0., kernel_neutral_weight=0.,
        weights=dict(anchor=2., coarse_consistency=5., kernel_noncollapse=2.,
                     separation=10., commutator=5.))
    metadata = dict(config=cfg, training_driver='rsmi_separate_qcr',
        labels_used_in_training=True, independent_c=True, estimator='histogram',
        environment_mode='full_shell', buffer=1, rsmi_weight=2., precision='torch.float64',
        data_sha256=reporter.DATA_SHA, q_supervision=dict(weight=10., scale=0,
            fitting_split='TRAIN only', coupling_names=['K'],
            target='raw coupling labels, Q(full microscopic batch)',
            gradient='Q only; no labels for coarse levels or C/R'))
    loaded = SimpleNamespace(config=cfg, payload=dict(epoch=4000, metadata=metadata))
    return loaded, deepcopy(metadata)


def test_saved_linear_map_responses_and_paired_bootstrap_are_exact():
    encodings = saved_arrays()
    originals = {key: value.copy() for key, value in encodings.items()}
    report = reporter.response_from_encodings(LinearR(), encodings, 4000, bootstraps=50)
    assert report['actual_epochs'] == 4000
    assert report['Q_C_inference'] is False and report['optimizer_updates'] == 0
    assert report['bootstrap']['replicas_per_K'] == 4
    for window in report['critical_windows']:
        assert window['status'] == 'complete'
        for row in window['transitions']:
            assert row['empirical_response'] == pytest.approx(1.7)
            assert row['predicted_response'] == pytest.approx(1.7)
            assert row['response_mismatch'] == pytest.approx(0., abs=1e-12)
            np.testing.assert_allclose(row['mismatch_bootstrap_95'], 0., atol=1e-12)
    np.testing.assert_allclose(report['jensen_gap'], 0., atol=1e-15)
    np.testing.assert_allclose(report['critical']['DR_at_mean_by_scale'], 1.7)
    for key in encodings:
        np.testing.assert_array_equal(encodings[key], originals[key])
    json.dumps(report, allow_nan=False)


def test_nonlinear_jensen_gap_uses_mean_predictions_not_prediction_of_mean():
    class NonlinearR(torch.nn.Module):
        def advance_theory(self, x):
            return 1.7*x + 2*x.square()
    encodings = saved_arrays()
    report = reporter.response_from_encodings(NonlinearR(), encodings, 6000, bootstraps=30)
    assert report['actual_epochs'] == 6000
    assert np.all(np.asarray(report['jensen_gap']) > 0)
    expected = 2*np.mean(np.array([-.0003, -.0001, .0001, .0003])**2)
    np.testing.assert_allclose(np.asarray(report['jensen_gap'])[:, 0], expected, atol=1e-15)


def test_collapsed_input_is_marked_undefined_without_fake_confidence_intervals():
    encodings = saved_arrays()
    for split in ('train', 'test'):
        encodings[split][:] = 0.
    report = reporter.response_from_encodings(LinearR(), encodings, 9000, bootstraps=10)
    for window in report['critical_windows']:
        assert window['status'] == 'undefined_input_derivative'
        assert window['transitions'] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('problem', ['epoch', 'labels', 'weight', 'coarse', 'block', 'samples', 'schedule', 'manifest', 'loss', 'target'])
def test_wrong_checkpoint_or_supervision_is_rejected(problem):
    loaded, manifest = checkpoint_fixture()
    if problem == 'epoch': loaded.payload['epoch'] = 3999
    if problem == 'labels': loaded.payload['metadata']['labels_used_in_training'] = False
    if problem == 'weight': loaded.payload['metadata']['q_supervision']['weight'] = 0.
    if problem == 'coarse': loaded.payload['metadata']['q_supervision']['scale'] = 1
    if problem == 'block': loaded.config['model']['block_factor'] = 3
    if problem == 'samples': loaded.config['evaluation']['samples'] = 16
    if problem == 'schedule': loaded.config['training']['lr_milestones'] = [3000, 4500]
    if problem == 'manifest': manifest['config']['model']['hidden'] = 32
    if problem == 'loss': loaded.config['loss']['weights']['separation'] = 0.
    if problem == 'target': loaded.payload['metadata']['q_supervision']['target'] = 'a theoretical multiplier'
    with pytest.raises(ValueError):
        reporter.validate_checkpoint(loaded, manifest, 4000)


def test_exact_supervised_checkpoint_passes_validation():
    loaded, manifest = checkpoint_fixture()
    assert reporter.validate_checkpoint(loaded, manifest, 4000) is loaded.config


def test_existing_output_is_refused_before_loading_any_data(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('existing evidence')
    monkeypatch.setattr(reporter, 'load_run', lambda *_: pytest.fail('Existing output must fail before checkpoint IO'))
    with pytest.raises(FileExistsError):
        reporter.main(['--run', str(tmp_path), '--checkpoint', '4000', '--output', str(output)])
    assert sentinel.read_text() == 'existing evidence'


def test_real_chinese_response_figure_renders_without_missing_glyphs(tmp_path):
    try:
        select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    report = reporter.response_from_encodings(LinearR(), saved_arrays(), 4000, bootstraps=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with chinese_figures():
            reporter.draw_response(report, LinearR(), tmp_path)
    assert (tmp_path / 'local_response.png').stat().st_size > 0
    assert not any('Glyph' in str(item.message) and 'missing' in str(item.message) for item in caught)
    assert not list(tmp_path.glob('*.pt'))


def trend_root_fixture(tmp_path):
    root = tmp_path / 'reports'
    root.mkdir()
    roots = [dict(root_id='F1', g=[.4], multiplier=1.7, nu=1.3,
        stability='unstable', outside_training_support=False,
        microscopic_inverse_coupling=[.42, .47]),
        dict(root_id='F2', g=[-.2], multiplier=.5, nu=None,
        stability='stable', outside_training_support=True,
        microscopic_inverse_coupling=[])]
    reference = tmp_path / 'old_pilot' / 'supervised_final'
    reference.mkdir(parents=True)
    reference_path = reference / 'critical_exponents.json'
    reference_path.write_text(json.dumps(dict(checkpoint_epoch=2870, roots=roots)))
    source_files = [reference_path]
    for epoch in (4000, 6000):
        folder = root / f'epoch_{epoch}'
        folder.mkdir()
        source = folder / 'checkpoint_summary.json'
        source.write_text(json.dumps(dict(status='complete', actual_epochs=epoch, roots=roots)))
        source_files.append(source)
    incomplete = root / 'epoch_9000'
    incomplete.mkdir()
    (incomplete / 'checkpoint_summary.json').write_text(json.dumps(dict(status='failed')))
    return root, reference, source_files


def test_trend_retains_all_roots_and_reference_without_cross_epoch_matching(tmp_path):
    root, reference, files = trend_root_fixture(tmp_path)
    before = {path: path.read_bytes() for path in files}
    trend = reporter.collect_trend(root, reference)
    assert [row['actual_epochs'] for row in trend['snapshots']] == [2870, 4000, 6000]
    assert [len(row['roots']) for row in trend['snapshots']] == [2, 2, 2]
    assert trend['snapshots'][0]['reference'] is True
    assert trend['root_matching_across_checkpoints'] is False
    assert len(trend['skipped']) == 1
    assert all(path.read_bytes() == original for path, original in before.items())


def test_unowned_existing_trend_is_never_replaced(tmp_path):
    root, _, _ = trend_root_fixture(tmp_path)
    sentinel = root / 'trend.json'
    sentinel.write_text('existing evidence')
    with pytest.raises(FileExistsError):
        reporter.update_trend(root)
    assert sentinel.read_text() == 'existing evidence'
    assert not (root / 'trend_owner.json').exists()


def test_owned_chinese_trend_can_accumulate_without_overwriting_checkpoint_files(tmp_path):
    try:
        select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    root, reference, files = trend_root_fixture(tmp_path)
    before = {path: path.read_bytes() for path in files}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        first = reporter.update_trend(root, reference)
        second = reporter.update_trend(root, reference)
    assert first == second
    assert (root / 'trend.png').stat().st_size > 0
    assert (root / 'trend_owner.json').is_file()
    assert not (root / 'trend.pending.png').exists()
    assert all(path.read_bytes() == original for path, original in before.items())
    assert not any('Glyph' in str(item.message) and 'missing' in str(item.message) for item in caught)
