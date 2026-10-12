from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recurrent_rg.config import RunConfig
from scripts.run_scalar_lr_sweep import build_plan, ensure_data_unchanged, main, make_configs


def arguments(tmp_path, **changes):
    values = dict(cache=tmp_path / 'cache.pt', output=tmp_path / 'sweep', device='cuda:0',
                  epochs=12000, seed=91, threads=2, max_wall_seconds_per_run=2700.)
    return Namespace(**(values | changes))


def test_only_the_preregistered_lr_policy_changes(tmp_path):
    configs = make_configs(tmp_path / 'cache.pt')
    assert list(configs) == ['constant', 'multistep', 'cosine']
    stripped = []
    for cfg in configs.values():
        RunConfig.from_dict(cfg).validate()
        cfg = deepcopy(cfg)
        for name in ('lr_schedule', 'lr_milestones', 'lr_gamma', 'lr_min_factor', 'lr_cosine_updates'):
            cfg['training'].pop(name)
        stripped.append(cfg)
    assert stripped[0] == stripped[1] == stripped[2]
    cfg = configs['constant']
    assert cfg['schema_version'] == 5
    assert cfg['model']['block_factor'] == 3
    assert cfg['model']['theory_dim'] == 1
    assert cfg['training']['gradient_mode'] == 'qc_r_separate'
    assert cfg['loss']['kernel_sign_weight'] == 1
    assert cfg['loss']['kernel_noncollapse_mode'] == 'variance_floor_ensemble'
    assert cfg['loss']['geometry_tail_mix'] == cfg['loss']['geometry_tail_add_weight'] == 0
    assert configs['multistep']['training']['lr_milestones'] == [6000, 9000]
    assert configs['cosine']['training']['lr_cosine_updates'] == 12000


def test_commands_preserve_default_driver_and_all_root_final_reports(tmp_path):
    plan = build_plan(arguments(tmp_path))
    assert len(plan['cases']) == 3
    for case in plan['cases']:
        commands = case['commands']
        assert commands[0]['command'][2] == 'experiments.rsmi_separate_qcr'
        assert '--device' in commands[0]['command']
        assert '--max-wall-seconds' in commands[0]['command']
        for entry in commands:
            assert not any(flag in entry['command'] for flag in (
                '--allow-nondefault-recipe', '--allow-legacy-training', '--source',
                '--resume-checkpoint', '--mi-initialize', '--joint-train-plateau'))
        assert commands[-1]['command'][2] == 'scripts.report_ising_scalar_exponents'
        assert '--checkpoint' not in commands[-1]['command']
    assert plan['protocol']['labels_used_in_training'] is False


def test_plan_only_does_not_access_data_write_files_or_run_training(tmp_path, capsys):
    args = arguments(tmp_path)
    plan = main(['--cache', str(args.cache), '--output', str(args.output), '--plan-only'])
    assert not args.output.exists()
    assert json.loads(capsys.readouterr().out) == plan


@pytest.mark.parametrize('changes', [dict(epochs=2), dict(device='cuda'),
    dict(threads=0), dict(max_wall_seconds_per_run=float('inf'))])
def test_reject_invalid_plan(tmp_path, changes):
    with pytest.raises(ValueError):
        build_plan(arguments(tmp_path, **changes))


def test_never_overwrite_existing_results(tmp_path):
    out = tmp_path / 'sweep'
    out.mkdir()
    with pytest.raises(FileExistsError):
        main(['--cache', str(tmp_path / 'missing.pt'), '--output', str(out)])


def test_mutated_data_cannot_mix_training_and_report_stages(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_scalar_lr_sweep.file_fingerprint', lambda path: 'changed')
    with pytest.raises(ValueError, match='Shared data changed'):
        ensure_data_unchanged(tmp_path / 'cache.pt', 'original')


def test_pipeline_detects_cache_mutation_after_training_before_any_report(tmp_path, monkeypatch):
    from scripts import run_scalar_lr_sweep as pipeline
    state = dict(digest='original', stages=[])
    monkeypatch.setattr(pipeline, 'check_data', lambda path: dict(sha256='original'))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: state['digest'])

    def fake_stage(out, status, name, stage, command):
        state['stages'].append(stage)
        assert stage == 'training'
        state['digest'] = 'mutated_after_training'

    monkeypatch.setattr(pipeline, 'run_stage', fake_stage)
    out = tmp_path / 'sweep'
    with pytest.raises(ValueError, match='Shared data changed'):
        pipeline.main(['--cache', str(tmp_path / 'cache.pt'), '--output', str(out)])
    assert state['stages'] == ['training']
    assert json.loads((out / 'pipeline.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('column_labels', [False, True])
def test_actual_sampler_scalar_label_layout_passes_production_validation(
        tmp_path, monkeypatch, column_labels):
    """Production-sized metadata without allocating a 256-MiB spin tensor."""
    from scripts import run_scalar_lr_sweep as pipeline
    from scripts.prepare_ising2d_lr_data import coupling_knots
    knots = coupling_knots().tolist()
    train_k = [k for i, k in enumerate(knots) if i % 3 != 1]
    test_k = [k for i, k in enumerate(knots) if i % 3 == 1]
    ensembles = []
    for count, values in ((129, train_k), (63, test_k)):
        labels = torch.tensor([k for k in values for _ in range(3)], dtype=torch.float64)
        if column_labels:
            labels = labels[:, None]
        ensembles.append(SimpleNamespace(
            configurations=SimpleNamespace(shape=(count, 128, 36, 36, 1)), couplings=labels))
    metadata = dict(sampling_options=dict(size=36, knots=64, samples=128, replicas=3,
        equil=4096, gap_low=512, gap_high=128, pilot_samples=1024),
        B=0., coupling_names=['K'], data_sha256='digest', train_K=train_k, test_K=test_k)
    cache = tmp_path / 'cache.pt'
    cache.with_suffix('.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: 'digest')
    monkeypatch.setattr(pipeline, 'load_ensembles', lambda path: tuple(ensembles))
    report = pipeline.check_data(cache)
    assert report['train_shape'] == [129, 128, 36, 36, 1]
    # Multiple K values per ensemble still fail closed.
    ensembles[0].couplings = ensembles[0].couplings.flatten()[:, None].expand(-1, 2)
    with pytest.raises(ValueError, match='K-only labels'):
        pipeline.check_data(cache)
