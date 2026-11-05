"""Estimator guards on synthetic public reports; no historical report writes."""
import json

import numpy as np
import pytest

from scripts import report_ising_closure_scale_ablation as ablation
from scripts import report_ising_closure_scale_late as late
from scripts import report_ising_closure_scale_early as early
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays
from test_ising_closure_scale_report import fixture_run, write_json
from test_ising_closure_scale_late_report import single_fixture
from test_ising_closure_scale_early_report import triple_fixture


def mark_modern(root, folder):
    path = root / folder / 'report/encodings.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for split in ('train', 'test'):
        for key in ('direct', 'R_g', 'rollout'):
            draw_key = f'{key}_draws_{split}'
            if draw_key not in arrays:
                arrays[draw_key] = np.repeat(arrays[f'{key}_{split}'][:, None], 2, axis=1)
                if key == 'rollout':
                    arrays[draw_key][:, :, 0] = arrays[f'direct_draws_{split}'][:, :, 0]
            arrays[f'full_ensemble_{draw_key}'] = arrays[draw_key].copy()
            arrays[f'full_ensemble_{key}_{split}'] = arrays[f'{key}_{split}'].copy()
        arrays[f'Q_full_minus_half_draws_{split}'] = np.zeros_like(arrays[f'direct_draws_{split}'])
        arrays[f'Q_full_minus_half_{split}'] = np.zeros_like(arrays[f'direct_{split}'])
    np.savez_compressed(path, **arrays)
    metrics_path = path.with_name('metrics.json')
    metrics = json.loads(metrics_path.read_text())
    protocol = encoding_contract(4)
    metrics.update(schema_version=2, samples=4, C_draws=2, encoding_estimator=protocol['estimator'],
        encoding_protocol=protocol, encoding_matches_training=True, encodings_sha256=ablation.public_sha(path))
    write_json(metrics_path, metrics)
    return metrics, arrays


CASES = [(ablation, fixture_run, ['run_uniform', 'run_late']),
         (late, single_fixture, ['reference_control', 'run_late']),
         (early, triple_fixture, ['reference_control', 'reference_late', 'run_early'])]


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_report_refuses_mixed_estimators_before_comparison(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    mark_modern(tmp_path, folders[-1])
    with pytest.raises(ValueError, match='编码估计器不同'):
        report.build_report(tmp_path, expected_steps=6)


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_report_accepts_consistent_modern_estimators_and_preserves_diagnostics(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    for folder in folders:
        mark_modern(tmp_path, folder)
    result, panels = report.build_report(tmp_path, expected_steps=6)
    assert result['status'] == 'complete'
    assert all('Q_full_minus_half_draws_test' in arrays for arrays in panels.values())


@pytest.mark.parametrize('damage', ['missing_full', 'wrong_full_mean', 'wrong_gap', 'wrong_gap_mean',
                                   'wrong_draw_count', 'missing_draw_count'])
def test_diagnostic_claims_validated_not_just_metadata(tmp_path, damage):
    fixture_run(tmp_path)
    metrics, arrays = mark_modern(tmp_path, 'run_uniform')
    assert validate_encoding_arrays(metrics, arrays) == metrics['encoding_estimator']
    if damage == 'missing_full': arrays.pop('full_ensemble_R_g_draws_test')
    elif damage == 'wrong_full_mean': arrays['full_ensemble_direct_test'] += 1
    elif damage == 'wrong_gap': arrays['Q_full_minus_half_draws_test'] += 1
    elif damage == 'wrong_gap_mean': arrays['Q_full_minus_half_test'] += 1
    elif damage == 'wrong_draw_count': metrics['C_draws'] = 3
    elif damage == 'missing_draw_count': metrics.pop('C_draws')
    with pytest.raises((ValueError, AssertionError)):
        validate_encoding_arrays(metrics, arrays)
