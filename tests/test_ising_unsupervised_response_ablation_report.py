"""K-free training diagnostics and explicitly separated post-hoc label evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, load_run, qcr_state_dict
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig, TrainingConfig, EvaluationConfig
from recurrent_rg.runner import file_fingerprint
from scripts import report_ising_unsupervised_response_ablation as reporter
from scripts.report_ising_zh import select_chinese_font


class LinearR(torch.nn.Module):
    def __init__(self, slope=1.7):
        super().__init__()
        self.slope = slope

    def advance_theory(self, value):
        return self.slope*value


def arrays_fixture():
    kc = float(np.log1p(np.sqrt(2))/2)
    knots = kc + np.linspace(-.09, .09, 25)
    noise = np.linspace(-.007, .007, 12)
    blocks = np.stack([np.stack([1.7**scale*(k-kc+noise) for scale in range(3)], -1)
                       for k in knots])
    train = np.arange(len(knots)) % 3 != 1
    return dict(train=blocks[train].reshape(-1, 3, 1), test=blocks[~train].reshape(-1, 3, 1),
        train_labels=np.repeat(knots[train], 12)[:, None],
        test_labels=np.repeat(knots[~train], 12)[:, None])


def pair_plan(arrays):
    from experiments.ising_unsupervised_response_ablation import prepare_secant_data
    prepared = prepare_secant_data(arrays['train'])
    normalizer = torch.stack([value.square().mean() for value in prepared.target_secants]).clamp_min(1e-12)
    return dict(alpha=1., train_sigma=prepared.train_sigma.tolist(),
                normalization=normalizer.tolist()), prepared


def test_label_free_metrics_never_access_label_arrays_and_test_reuses_train_scaling():
    class NoLabelAccess(dict):
        def __getitem__(self, key):
            if 'label' in key:
                pytest.fail('A K label was accessed by the label-free response evaluator')
            return super().__getitem__(key)
    arrays = arrays_fixture()
    plan, train = pair_plan(arrays)
    measured, diagnostics = reporter.label_free_response_metrics(
        dict(control=LinearR(1.5), response=LinearR()), NoLabelAccess(arrays), plan)
    assert measured['response']['train']['response_mse'] == pytest.approx(0., abs=1e-25)
    assert measured['response']['test']['response_mse'] == pytest.approx(0., abs=1e-25)
    assert measured['control']['test']['response_mse'] == pytest.approx(.04)
    for scale in range(2):
        assert diagnostics['test'][scale]['train_sigma'] == pytest.approx(plan['train_sigma'][scale])
    assert measured['response']['train']['response_normalized'] < measured['control']['train']['response_normalized']
    bad = deepcopy(plan)
    bad['normalization'] = (np.array(plan['normalization'])*2).tolist()
    with pytest.raises(ValueError, match='only from frozen TRAIN'):
        reporter.label_free_response_metrics(dict(control=LinearR()), NoLabelAccess(arrays), bad)


def test_K_regression_is_separate_evaluation_only_and_never_fits_a_coordinate():
    arrays = arrays_fixture()
    result = reporter.microscopic_K_evaluation(arrays)
    for split in ('train', 'test'):
        expected = np.mean((arrays[split][:, 0, 0]-arrays[split+'_labels'].ravel())**2)
        assert result[split]['raw_g0_minus_K_mse'] == pytest.approx(expected)
        assert result[split]['training_use'] is False


def experiment_fixture(tmp_path):
    from experiments.ising_unsupervised_response_ablation import prepare_secant_data, metrics
    root, source_dir = tmp_path / 'training', tmp_path / 'unsupervised_source'
    root.mkdir(); source_dir.mkdir()
    source_report = source_dir / 'full_report'
    source_report.mkdir()
    arrays = arrays_fixture()
    np.savez_compressed(source_report / 'encodings.npz', **arrays)
    cfg = RunConfig(data=DataConfig(cache=str(tmp_path / 'NEVER_OPEN_PHYSICAL_DATA.pt'),
                    physical_model='ising2d', coupling_names=['K']),
        model=ModelConfig(dimension=2, theory_dim=1, block_factor=2, hidden=4,
                          affine_reconstruction=True, kernel_sampling='bernoulli'),
        training=TrainingConfig(max_rg_steps=2, epochs=12000),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=128))
    model = build_model(cfg.model, 2)
    with torch.no_grad():
        for parameter in model.theory_rg.parameters():
            parameter.zero_()
    state = qcr_state_dict(model)
    metadata = dict(config=cfg.to_dict(), labels_used_in_training=False)
    source_payload = dict(model=state, epoch=12000, metadata=metadata)
    source_checkpoint = source_dir / 'training_step_12000.pt'
    torch.save(source_payload, source_checkpoint)
    (source_dir / 'manifest.json').write_text(json.dumps(metadata))
    plan, train = pair_plan(arrays)
    test = prepare_secant_data(arrays['test'], train_sigma=plan['train_sigma'])
    plan.update(source_run=str(source_dir), source_report=str(source_report),
        source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=file_fingerprint(source_checkpoint),
        source_epoch=12000, updates=3, smoke=True,
        encodings_sha256=file_fingerprint(source_report / 'encodings.npz'))
    (root / 'plan.json').write_text(json.dumps(plan))
    normalization = torch.tensor(plan['normalization'], dtype=torch.float64)
    for arm in reporter.ARMS:
        path = root / arm
        path.mkdir()
        alpha = 0. if arm == 'control' else 1.
        final_state = deepcopy(state)
        if arm == 'response':
            final_state['theory_rg.4.bias'] += .001
        arm_meta = dict(config=cfg.to_dict(), arm=arm, alpha=alpha, extra_R_updates=3,
            R_only_ablation=True, exact_resume=False, labels_used_in_training=False,
            K_label_access=False, source_checkpoint_sha256=plan['source_checkpoint_sha256'],
            encodings_sha256=plan['encodings_sha256'])
        torch.save(dict(model=state, epoch=12000, metadata=arm_meta), path / 'training_step_0.pt')
        torch.save(dict(model=final_state, epoch=12000, metadata=arm_meta), path / 'checkpoint_last.pt')
        (path / 'manifest.json').write_text(json.dumps(arm_meta))
        (path / 'summary.json').write_text(json.dumps(dict(actual_extra_R_updates=3)))
        model.load_state_dict(final_state, strict=False)
        row = {f'{split}_{key}': value for split, data in (('train', train), ('test', test))
               for key, value in metrics(model, data, normalization, alpha).items()}
        history = [dict(extra_R_updates=step, **row) for step in (0, 3)]
        (path / 'history_progress.json').write_text(json.dumps(history))
    return root, source_report, cfg.model


@pytest.mark.parametrize('problem', ['source_supervised', 'new_supervised', 'label_access', 'wrong_source_epoch'])
def test_any_supervised_source_or_training_label_access_is_rejected(tmp_path, problem):
    root, _, _ = experiment_fixture(tmp_path)
    plan = json.loads((root / 'plan.json').read_text())
    source = load_run(Path(plan['source_checkpoint']))
    arms = {arm: load_run(root / arm, 'checkpoint_last.pt') for arm in reporter.ARMS}
    summaries = {arm: json.loads((root / arm / 'summary.json').read_text()) for arm in reporter.ARMS}
    if problem == 'source_supervised': source.payload['metadata']['labels_used_in_training'] = True
    if problem == 'new_supervised': arms['response'].payload['metadata']['labels_used_in_training'] = True
    if problem == 'label_access': arms['response'].payload['metadata']['K_label_access'] = True
    if problem == 'wrong_source_epoch': plan['source_epoch'] = 9000
    with pytest.raises(ValueError):
        reporter.validate_unsupervised_final(plan, source, arms, summaries, root)


def test_source_rejected_before_report_opens_label_arrays(tmp_path, monkeypatch):
    root, _, _ = experiment_fixture(tmp_path)
    source = load_run(Path(json.loads((root / 'plan.json').read_text())['source_checkpoint']))
    source.payload['metadata']['labels_used_in_training'] = True
    original = reporter.load_run
    monkeypatch.setattr(reporter, 'load_run', lambda path, *a: source if Path(path).is_file() else original(path, *a))
    monkeypatch.setattr(reporter, 'load_encodings', lambda *_: pytest.fail('Labels opened before source validation'))
    with pytest.raises(ValueError, match='without label supervision'):
        reporter.main(['--run', str(root), '--output', str(tmp_path / 'report')])


def test_complete_chinese_report_never_executes_QC_or_DK_training_loss(tmp_path, monkeypatch):
    try:
        select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))
    root, source_report, spec = experiment_fixture(tmp_path)
    probe = build_model(spec, 2)
    def forbidden(*args, **kwargs):
        pytest.fail('No Q/C execution or dK training-response evaluation permitted')
    monkeypatch.setattr(type(probe.theory_encoder), 'forward', forbidden)
    monkeypatch.setattr(type(probe.rg_kernel), 'forward', forbidden)
    from scripts import report_ising_response_ablation as old_report
    monkeypatch.setattr(old_report, 'training_response_metrics', forbidden)
    before = {path: path.read_bytes() for path in source_report.iterdir()}
    output = tmp_path / 'new_report'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = reporter.main(['--run', str(root), '--output', str(output), '--bootstraps', '10'])
    assert result['status'] == 'complete'
    assert result['source_QCR_epoch'] == 12000
    assert result['source_labels_used_in_training'] is False
    assert result['K_label_access_in_training'] is False
    assert result['labels_used_for_report_evaluation'] is True
    assert result['full_framework_retrained'] is False
    assert result['Q_C_inference'] is False and result['MI_calculation'] is False
    assert result['frozen_Q_C_verified'] is True
    assert len(list(output.glob('*.png'))) == 4
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not any('Glyph' in str(item.message) and 'missing' in str(item.message) for item in caught)
    summary_before = (output / 'comparison.json').read_bytes()
    with pytest.raises(FileExistsError):
        reporter.main(['--run', str(root), '--output', str(output)])
    assert (output / 'comparison.json').read_bytes() == summary_before
