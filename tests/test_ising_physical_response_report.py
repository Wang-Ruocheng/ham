"""R-only 物理响应报告不得通过 Q/C 重新生成数据或挑选理论接近的根。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising_physical_response as report


class ROnly(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer('matrix', torch.as_tensor(matrix, dtype=torch.float64))

    def advance_theory(self, value):
        return value @ self.matrix.T

    def coarse_grain(self, *args, **kwargs):
        raise AssertionError('Report must not invoke C')

    def theory_encoder(self, *args, **kwargs):
        raise AssertionError('Report must not invoke Q')


def fake_cache(dim=1, n=12):
    rng = np.random.default_rng(821)
    labels = np.linspace(.39, .49, n)
    g = np.stack([np.column_stack([(labels-report.KC)*(1+.25*k)+.001*j
                                  for j in range(dim)]) for k in range(3)], axis=1)
    direct = np.repeat(g[:, None], 5, axis=1)
    vin = rng.normal(size=(n, 5, 2, dim, 6))
    return dict(train_direct=torch.tensor(direct), test_direct=torch.tensor(direct),
        train_vin=torch.tensor(vin), test_vin=torch.tensor(vin),
        train_vout=torch.tensor(2*vin), test_vout=torch.tensor(2*vin),
        train_score_std=torch.ones(n, 5, 2, 6), test_score_std=torch.ones(n, 5, 2, 6)), labels


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_saved_response_jvp_includes_identity_and_zero_target_is_not_good_by_epsilon(dim):
    cache, _ = fake_cache(dim)
    split = report.cached_split(cache, 'test')
    result = report.response_diagnostics(ROnly(2*np.eye(dim)), split, [1., 1.])
    assert result['response_normalized'] < 1e-25
    assert result['response_relative_norm_by_scale'] == pytest.approx([0., 0.], abs=1e-12)
    assert result['physical_ensembles'] == 12 and result['C_draws'] == 5
    split['vout'].zero_()
    result = report.response_diagnostics(ROnly(2*np.eye(dim)), split, [1., 1.])
    assert result['response_relative_norm_by_scale'] == [None, None]
    assert result['confidence_interval'] is None


def test_zero_score_directions_are_explicitly_masked():
    cache, _ = fake_cache()
    split = report.cached_split(cache, 'train')
    split['score_std'].zero_()
    result = report.response_diagnostics(ROnly([[3.]]), split, [1., 1.])
    assert result['score_zero_fraction_by_scale'] == [1., 1.]
    assert result['response_relative_norm_by_scale'] == [None, None]


def test_shape_guard_preserves_ensemble_and_C_draw_axes():
    cache, _ = fake_cache()
    cache['train_direct'] = cache['train_direct'].reshape(60, 3, 1)
    with pytest.raises(ValueError, match='ensemble'):
        report.cached_split(cache, 'train')


def test_only_original_public_labels_and_explicit_smoke_indices_are_used(tmp_path):
    np.savez(tmp_path/'encodings.npz', train_labels=np.arange(12), test_labels=np.arange(12))
    result = report.evaluation_labels(tmp_path, dict(train_source_indices=torch.tensor([4, 8])), 'train', 2)
    np.testing.assert_array_equal(result, [4, 8])
    with pytest.raises(ValueError, match='indices required'):
        report.evaluation_labels(tmp_path, {}, 'train', 2)
    with pytest.raises(ValueError, match='Invalid saved'):
        report.evaluation_labels(tmp_path, dict(train_source_indices=[2, 2]), 'train', 2)


def test_public_root_inverse_keeps_nonunique_branches_and_errors():
    roots = dict(roots=[dict(curve_projections_by_scale=[dict(exact_inverse_K_branches=[.4, .5]),
                      dict(exact_inverse_K_branches=[])])])
    result = report.augment_inverse_K(roots)['roots'][0]
    assert len(result['curve_projections_by_scale'][0]['exact_inverse_K_errors']) == 2
    assert result['inverse_K_all_branch_span'] == pytest.approx(.1)
    assert result['curve_projections_by_scale'][1]['exact_inverse_K_errors'] == []


def test_repeated_closure_all_three_paths_and_all_five_draws():
    cache, _ = fake_cache()
    split = report.cached_split(cache, 'train')
    result = report.repeated_closure(ROnly([[2.]]), split, split)
    assert len(result['by_C_draw']) == 5
    paths = result['all_five_draws']['all_start_horizon']
    assert [(row['start'], row['end']) for row in paths] == [(0, 1), (0, 2), (1, 2)]


def test_arm_guard_checks_actual_updates_not_source_config_epochs(tmp_path):
    loaded = SimpleNamespace(payload=dict(epoch=12000, extra_R_updates=6000, metadata=dict(frozen_QC_sha256='a')))
    (tmp_path/'summary.json').write_text(json.dumps(dict(actual_extra_R_updates=6000)))
    report.validate_arm(loaded, tmp_path, 6000, dict(frozen_QC_sha256='a'))
    with pytest.raises(ValueError, match='provenance'):
        report.validate_arm(loaded, tmp_path, 6000, dict(frozen_QC_sha256='b'))
    loaded.payload['extra_R_updates'] = 3000
    with pytest.raises(ValueError, match='additional updates'):
        report.validate_arm(loaded, tmp_path, 6000, {})


def test_fresh_directory_refuses_accidental_overwrite(tmp_path):
    with pytest.raises(ValueError, match='fresh report'):
        report.run_report(tmp_path/'not_read', tmp_path, updates=6000)


def test_report_source_has_no_configuration_or_sampling_calls():
    source = Path(report.__file__).read_text()
    assert '.coarse_grain(' not in source and '.theory_encoder(' not in source
    assert 'mmap_dataset(' not in source and 'sample_wolff' not in source


def test_arm_public_report_no_QC_calls_and_all_modes_kept(tmp_path, monkeypatch):
    cache, labels = fake_cache()
    model = ROnly([[1.7]])
    values = {split: report.cached_split(cache, split) for split in ('train', 'test')}
    loaded = SimpleNamespace(model=model, checkpoint=tmp_path/'source'/'training_step_12000.pt',
        payload=dict(epoch=12000, history=[]))
    # The scientific plotters are tested separately; avoid font rendering in this lightweight regression.
    monkeypatch.setattr(report, 'plot_maps', lambda *args: None)
    monkeypatch.setattr(report, 'plot_arm_diagnostics', lambda *args: None)
    result = report.arm_report('response', loaded, values, dict(train=labels, test=labels),
        dict(normalization=[1., 1.], response_weight=3.), tmp_path/'public', updates=6000)
    assert result['Q_C_inference_performed'] is False and result['additional_R_updates'] == 6000
    assert len(result['roots']) == 1
    assert result['roots'][0]['thermal_nu'] is None
    assert result['roots'][0]['eigenvalues'][0]['real'] == pytest.approx(1.7)
    assert not list((tmp_path/'public').glob('*.pt'))
    status = json.loads((tmp_path/'public'/'report.json').read_text())
    assert status['report_complete'] is True
    with np.load(tmp_path/'public'/'encodings.npz') as public:
        assert public['train_all_C_draws'].shape == (12, 5, 3, 1)
