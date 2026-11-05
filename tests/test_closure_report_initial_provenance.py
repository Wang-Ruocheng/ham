"""Report provenance checks use only temporary synthetic JSON/NPZ fixtures."""
import json

import pytest

from scripts import report_ising_closure_scale_ablation as ablation
from scripts import report_ising_closure_scale_late as late
from scripts import report_ising_closure_scale_early as early
from test_ising_closure_scale_report import fixture_run, write_json
from test_ising_closure_scale_late_report import single_fixture
from test_ising_closure_scale_early_report import triple_fixture


CASES = [
    (ablation, fixture_run, ['run_uniform', 'run_late']),
    (late, single_fixture, ['reference_control', 'run_late']),
    (early, triple_fixture, ['reference_control', 'reference_late', 'run_early']),
]
ACTUAL = dict(Q='a' * 64, C='b' * 64, R='c' * 64)
SCOPE = 'After calibration; before the first optimizer update'


def add_actual_start(root, folder, *, actual=None):
    run = root / folder
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['training_start_module_sha256'] = dict(ACTUAL if actual is None else actual)
    manifest['initial_module_sha256_scope'] = 'Legacy pre-calibration alias'
    manifest['module_initialization_provenance'] = dict(training_start_scope=SCOPE)
    write_json(run / 'manifest.json', manifest)
    summary = json.loads((run / 'summary.json').read_text())
    summary['contract']['training_start_module_sha256'] = manifest['training_start_module_sha256']
    write_json(run / 'summary.json', summary)


def comparisons(result):
    if 'pairwise_coordinate_comparison' in result:
        return list(result['pairwise_coordinate_comparison'].values())
    return [result['coordinate_comparison']]


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_legacy_hash_never_becomes_actual_training_start(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    result, panels = report.build_report(tmp_path, expected_steps=6)
    for arm in panels:
        assert result['arms'][arm]['actual_training_start_fingerprints_available'] is False
        assert result['arms'][arm]['training_start_module_sha256'] is None
    for comparison in comparisons(result):
        assert comparison['same_training_start_module_fingerprints'] is None
        assert '不证明首步参数相同' in comparison['initial_fingerprint_comparison_scope']


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_actual_start_scope_is_saved_and_same_hashes_are_comparable(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    for folder in folders:
        add_actual_start(tmp_path, folder)
    result, panels = report.build_report(tmp_path, expected_steps=6)
    for arm in panels:
        info = result['arms'][arm]
        assert info['actual_training_start_fingerprints_available'] is True
        assert info['training_start_module_sha256'] == ACTUAL
        assert info['training_start_module_sha256_scope'] == SCOPE
        assert info['initial_module_sha256_scope'] == 'Legacy pre-calibration alias'
    assert all(row['same_training_start_module_fingerprints'] is True for row in comparisons(result))


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_one_missing_actual_start_stays_unknown(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    add_actual_start(tmp_path, folders[-1])
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert all(row['same_training_start_module_fingerprints'] is None for row in comparisons(result))


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_actual_start_hashes_must_match_even_when_legacy_hashes_match(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    for folder in folders:
        add_actual_start(tmp_path, folder)
    add_actual_start(tmp_path, folders[-1], actual=dict(ACTUAL, Q='d' * 64))
    with pytest.raises(ValueError, match='实际训练起点.*不同'):
        report.build_report(tmp_path, expected_steps=6)


@pytest.mark.parametrize('report,fixture,folders', CASES)
def test_actual_start_requires_matching_summary_contract(tmp_path, report, fixture, folders):
    fixture(tmp_path)
    add_actual_start(tmp_path, folders[-1])
    path = tmp_path / folders[-1] / 'summary.json'
    summary = json.loads(path.read_text())
    summary['contract']['training_start_module_sha256']['Q'] = 'd' * 64
    write_json(path, summary)
    with pytest.raises(ValueError, match='实际训练起点.*summary.contract'):
        report.build_report(tmp_path, expected_steps=6)
