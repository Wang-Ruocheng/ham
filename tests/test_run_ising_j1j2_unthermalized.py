from pathlib import Path
import json

import pytest

from scripts import run_ising_j1j2_unthermalized as diagnostic
from scripts import run_ising_j1j2 as formal


def test_diagnostic_stage_order_and_no_old_pilot_or_quality_gate():
    plan = dict(physical=dict(rhos=[0., .35, .7, 1.], group_ids=[0, 1, 8, 9]),
                resource_policy=dict(report_seconds_per_arm=1800))
    stages = diagnostic.stage_list(Path('/new'), plan)
    assert len(stages) == 21
    assert sum(row[2] for row in stages) == 2
    assert [row[0] for row in stages[:4]] == [f'sample_r00_g{g:02d}' for g in (0, 1, 8, 9)]
    assert all('independent_pilot' not in row[0] and 'production_quality_gate' not in row[0]
               for row in stages)
    assert stages[-1][1][3] == 'scripts.report_ising_j1j2_unthermalized'
    assert '--quality-warning' in stages[-1][1]


def test_diagnostic_training_reuses_original_wrapper_without_persistent_mutation(monkeypatch):
    original = formal.verify
    def fake(root, arm):
        assert formal.verify is diagnostic.verify
        return root, arm
    monkeypatch.setattr(formal, 'train_child', fake)
    assert diagnostic.train_child('/new', 'profile') == ('/new', 'profile')
    assert formal.verify is original
    with pytest.raises(RuntimeError):
        with diagnostic.diagnostic_training_context():
            raise RuntimeError('test')
    assert formal.verify is original


def test_sampling_projection_counts_all_auxiliary_temperatures_and_keeps_hard_budget():
    plan = dict(physical=dict(rhos=[0, .35, .7, 1], group_ids=[0, 1, 8, 9],
        temperatures=[1., 2., 3.], size=108, burn_in=4096, samples=128, spacing=32),
        reference=dict(worst_cpu_seconds_per_visited_site=1e-9),
        resource_policy=dict(sampling_projection_factor=2., MI_cpu_reserve_seconds=10.,
            max_training_seconds=100., report_seconds_per_arm=10., cpu_seconds=1000.))
    result = diagnostic.sampling_projection(plan)
    assert result['projected_sampling_cpu_seconds'] == pytest.approx(4 * 4 * 3 * 108**2 * 8192 * 2e-9)
    assert result['budget_passed']
    plan['resource_policy']['cpu_seconds'] = 100.
    assert not diagnostic.sampling_projection(plan)['budget_passed']


def test_prepare_uses_fresh_recipe_and_never_changes_stopped_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.socket, 'gethostname', lambda: formal.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(formal, 'source_hashes', lambda code: {'sampler.py': 'unchanged'})
    reference = tmp_path / 'stopped'
    (reference / 'data/benchmark').mkdir(parents=True)
    old = dict(source_sha256={'sampler.py': 'unchanged'},
        pilot=dict(min_exchange=.01, min_ess=16, hot_cold_abs_threshold=.05,
                   hot_cold_z_threshold=4.), resource_policy={})
    (reference / 'plan.json').write_text(json.dumps(old))
    (reference / 'job.json').write_text(json.dumps(dict(status='scientific_stop', training_started=False)))
    (reference / 'data/benchmark/receipt.json').write_text(json.dumps(
        dict(measured_worst_cpu_seconds_per_visited_site=1e-9)))
    before = {str(p): p.read_bytes() for p in reference.rglob('*') if p.is_file()}
    root = tmp_path / 'new'
    plan = diagnostic.prepare(root, reference)
    assert plan['physical']['group_ids'] == [0, 1, 8, 9]
    assert plan['physical']['heldout_rhos'] == []
    assert plan['diagnostic']['quality_override_authorized']
    cfg = plan['configs']['dg5_seed91']
    assert cfg['training']['epochs'] == 12000
    assert cfg['training']['seed'] == 91
    assert cfg['model']['block_factor'] == 3
    assert cfg['loss']['closure_objective'] == 'all_start_anchor'
    assert cfg['loss']['closure_scale_weights'] is None
    assert diagnostic.verify(root)[1]['training_started'] is False
    assert before == {str(p): p.read_bytes() for p in reference.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        diagnostic.prepare(root, reference)
