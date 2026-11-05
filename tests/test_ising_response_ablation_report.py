"""Saved-target report checks: no physical data, Q/C inference or experiment launch."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig, TrainingConfig, EvaluationConfig
from recurrent_rg.runner import file_fingerprint
from scripts import report_ising_response_ablation as reporter
from scripts.report_ising_zh import chinese_figures, select_chinese_font


def saved_arrays():
    kc = float(np.log1p(np.sqrt(2)) / 2)
    knots = kc + np.linspace(-.09, .09, 25)
    deviations = np.linspace(-.0003, .0003, 12)
    blocks = np.stack([np.stack([1.7**scale * (k-kc+deviations)
                                for scale in range(3)], -1) for k in knots])
    train = np.arange(len(knots)) % 3 != 1
    return dict(train=blocks[train].reshape(-1, 3, 1), test=blocks[~train].reshape(-1, 3, 1),
        train_labels=np.repeat(knots[train], 12)[:, None],
        test_labels=np.repeat(knots[~train], 12)[:, None])


class Forbidden(torch.nn.Module):
    def forward(self, *args, **kwargs):
        pytest.fail('Q/C must never execute in an R-only saved-encoding report')


class PolynomialR(torch.nn.Module):
    def __init__(self, multiplier=1.7):
        super().__init__()
        self.multiplier = multiplier
        self.theory_encoder = Forbidden()
        self.rg_kernel = Forbidden()

    def advance_theory(self, g):
        return self.multiplier*g - 6*g**3


def test_all_roots_full_derivative_fd_support_and_conditional_nu_are_retained(tmp_path):
    try:
        select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    arrays = saved_arrays()
    before = {name: value.copy() for name, value in arrays.items()}
    loaded = SimpleNamespace(model=PolynomialR(), model_config=SimpleNamespace(
        dimension=2, theory_dim=1, categorical_states=0, block_factor=2),
        config=dict(data=dict(coupling_names=['K'])), payload=dict(epoch=9000),
        checkpoint=tmp_path / 'fixed_final.pt')
    with chinese_figures():
        result = reporter.analyze_arm(loaded, arrays, [-.5, .5], tmp_path / 'new_arm', 6000, bootstraps=20)
    roots = result['roots']
    assert len(roots) == 3
    center = min(roots, key=lambda item: abs(item['g'][0]))
    assert center['multiplier'] == pytest.approx(1.7)
    assert center['displacement_derivative'] == pytest.approx(.7)
    assert center['finite_difference_h'] == 1e-5
    assert center['derivative_verified'] and center['root_verified']
    assert center['nu'] == pytest.approx(np.log(2)/np.log(1.7))
    assert center['nu_certified'] is False
    assert all(root['nearest_training_input_distance'] >= 0 for root in roots)
    assert all('microscopic_inverse_coupling' in root for root in roots)
    assert sum(root['outside_training_support'] for root in roots) == 2
    assert all(root['nu'] is None for root in roots if root is not center)
    assert result['source_QCR_epoch'] == 9000 and result['extra_R_updates'] == 6000
    assert len(result['critical']['g_by_scale']) == 3
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])


def test_d7_test_metrics_use_train_normalizers_and_split_boundary_from_interior():
    from experiments.ising_response_loss_ablation import prepare_response_data
    arrays = saved_arrays()
    prepared = prepare_response_data(arrays['train'], arrays['train_labels'])
    normalization = prepared.target_derivative.square().mean(0)
    measured, diagnostics = reporter.training_response_metrics(
        dict(control=PolynomialR(), response=PolynomialR(1.8)), arrays,
        dict(alpha=1., normalization=normalization.tolist()))
    for arm in reporter.ARMS:
        for split in ('train', 'test'):
            metric = measured[arm][split]
            assert metric['response_mse'] >= 0
            assert metric['response_boundary_mse'] >= 0 and metric['response_interior_mse'] >= 0
            assert metric['response_boundary_normalized'] >= 0
            expected = np.mean(np.asarray(metric['response_mse_by_scale']) / normalization.numpy())
            assert metric['response_normalized'] == pytest.approx(expected)
            assert metric['total'] == pytest.approx(metric['closure'] +
                (metric['response_normalized'] if arm == 'response' else 0.))
    assert len(diagnostics['test']['boundary_mask']) == len(np.unique(arrays['test_labels']))
    with pytest.raises(ValueError, match='exclusively'):
        reporter.training_response_metrics(dict(control=PolynomialR()), arrays,
            dict(alpha=1., normalization=(normalization*2).tolist()))


def test_prediction_metrics_are_absolute_and_train_sd_normalized_without_encoding():
    from recurrent_rg.evaluation import closure_metrics
    arrays = saved_arrays()
    model = PolynomialR()
    predictions = reporter.predictions(model, arrays['test'])
    metric = closure_metrics(predictions, reference_direct=arrays['train'])
    rmse = np.sqrt(np.mean((predictions['one_step']-arrays['test'][:, 1:])**2, axis=(0, 2)))
    spread = np.sqrt(arrays['train'].var(0).mean(-1))[1:]
    np.testing.assert_allclose(metric['one_step']['rmse'], rmse)
    np.testing.assert_allclose(metric['one_step']['nrmse'], rmse/spread)


def test_only_r_state_is_allowed_to_change():
    source = {'theory_encoder.w': torch.tensor([1.]), 'rg_kernel.w': torch.tensor([2.]),
              'theory_rg.w': torch.tensor([3.])}
    candidate = deepcopy(source)
    candidate['theory_rg.w'] += 1
    assert reporter.frozen_state_matches(source, candidate)
    candidate['rg_kernel.w'] += 1
    assert not reporter.frozen_state_matches(source, candidate)


def experiment_fixture(tmp_path):
    from experiments.ising_response_loss_ablation import prepare_response_data, metrics
    root, source_dir = tmp_path / 'training', tmp_path / 'source'
    root.mkdir(); source_dir.mkdir()
    source_report = source_dir / 'saved_report'
    source_report.mkdir()
    arrays = saved_arrays()
    np.savez_compressed(source_report / 'encodings.npz', **arrays)
    cfg = RunConfig(data=DataConfig(cache=str(tmp_path / 'NEVER_OPEN_PHYSICAL_DATA.pt'),
                                    physical_model='ising2d', coupling_names=['K']),
        model=ModelConfig(dimension=2, theory_dim=1, block_factor=2, hidden=4,
                          affine_reconstruction=True, kernel_sampling='bernoulli'),
        training=TrainingConfig(max_rg_steps=2, epochs=9000),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=128))
    model = build_model(cfg.model, 2)
    with torch.no_grad():
        for parameter in model.theory_rg.parameters():
            parameter.zero_()
    source_state = qcr_state_dict(model)
    source_metadata = dict(config=cfg.to_dict())
    source_payload = dict(model=source_state, epoch=9000, metadata=source_metadata)
    source_checkpoint = source_dir / 'training_step_9000.pt'
    torch.save(source_payload, source_checkpoint)
    (source_dir / 'manifest.json').write_text(json.dumps(source_metadata))
    train = prepare_response_data(arrays['train'], arrays['train_labels'])
    test = prepare_response_data(arrays['test'], arrays['test_labels'])
    normalization = train.target_derivative.square().mean(0)
    plan = dict(source_run=str(source_dir), source_report=str(source_report),
        source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=file_fingerprint(source_checkpoint),
        source_epoch=9000, updates=3, alpha=1., normalization=normalization.tolist(),
        encodings_sha256=file_fingerprint(source_report / 'encodings.npz'))
    (root / 'plan.json').write_text(json.dumps(plan))
    for arm in reporter.ARMS:
        folder = root / arm
        folder.mkdir()
        alpha = 0. if arm == 'control' else 1.
        state = deepcopy(source_state)
        if arm == 'response':
            state['theory_rg.4.bias'] += .001
        metadata = dict(config=cfg.to_dict(), arm=arm, alpha=alpha, extra_R_updates=3,
            R_only_ablation=True, exact_resume=False,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'], encodings_sha256=plan['encodings_sha256'])
        torch.save(dict(model=source_state, metadata=metadata, epoch=9000), folder / 'training_step_0.pt')
        torch.save(dict(model=state, metadata=metadata, epoch=9000), folder / 'checkpoint_last.pt')
        (folder / 'manifest.json').write_text(json.dumps(metadata))
        (folder / 'summary.json').write_text(json.dumps(dict(actual_extra_R_updates=3)))
        model.load_state_dict(state, strict=False)
        row = {f'{split}_{key}': value for split, data in (('train', train), ('test', test))
               for key, value in metrics(model, data, normalization, alpha).items()}
        history = [dict(extra_R_updates=step, **row) for step in (0, 3)]
        (folder / 'history_progress.json').write_text(json.dumps(history))
    return root, source_report, cfg.model


def test_wrong_budget_or_changed_qc_is_rejected(tmp_path):
    from recurrent_rg.checkpoints import load_run
    root, _, _ = experiment_fixture(tmp_path)
    plan = json.loads((root / 'plan.json').read_text())
    source = load_run(Path(plan['source_checkpoint']))
    arms = {arm: load_run(root / arm, 'checkpoint_last.pt') for arm in reporter.ARMS}
    summaries = {arm: json.loads((root / arm / 'summary.json').read_text()) for arm in reporter.ARMS}
    reporter.validate_fixed_final(plan, source, arms, summaries, root)
    summaries['response']['actual_extra_R_updates'] = 2
    with pytest.raises(ValueError, match='fixed-budget'):
        reporter.validate_fixed_final(plan, source, arms, summaries, root)


def test_full_report_is_chinese_saved_target_only_and_refuses_overwrite(tmp_path, monkeypatch):
    try:
        select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    root, source_report, spec = experiment_fixture(tmp_path)
    probe = build_model(spec, 2)
    monkeypatch.setattr(type(probe.theory_encoder), 'forward', Forbidden().forward)
    monkeypatch.setattr(type(probe.rg_kernel), 'forward', Forbidden().forward)
    source_before = {path: path.read_bytes() for path in source_report.iterdir()}
    output = tmp_path / 'new_report'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = reporter.main(['--run', str(root), '--output', str(output), '--bootstraps', '10'])
    assert result['status'] == 'complete'
    assert result['source_QCR_epoch'] == 9000 and result['extra_R_updates_per_arm'] == 3
    assert result['frozen_Q_C_verified'] is True and result['Q_C_inference'] is False
    assert len(result['shared_representation_statistics']['train']['std']) == 3
    assert result['shared_representation_statistics']['train']['effective_rank'] == [1, 1, 1]
    assert result['MI_calculation'] is False and result['physical_sampling'] is False
    assert (output / 'comparison.json').is_file()
    assert len(list(output.glob('*.png'))) == 4
    assert not any('Glyph' in str(item.message) and 'missing' in str(item.message) for item in caught)
    assert all(path.read_bytes() == before for path, before in source_before.items())
    assert not Path(root.parent / 'NEVER_OPEN_PHYSICAL_DATA.pt').exists()
    comparison_before = (output / 'comparison.json').read_bytes()
    with pytest.raises(FileExistsError):
        reporter.main(['--run', str(root), '--output', str(output)])
    assert (output / 'comparison.json').read_bytes() == comparison_before
