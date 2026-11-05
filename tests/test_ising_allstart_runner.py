"""Pure runner contracts; real CUDA execution remains an explicit experiment."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ising_allstart_closure as runner


def policy():
    return dict(cpu_threads=2, profile_safety_factor=1.5,
        train_final_reserve_seconds=120., report_reserve_seconds=1200.,
        gpu_seconds=7200., cpu_seconds=14400., max_training_seconds_per_arm=7200.)


def test_paired_configs_only_loss_mode_differs(tmp_path):
    configs = runner.configs(tmp_path / 'physical.pt')
    old, new = deepcopy(configs['control']), deepcopy(configs['anchor'])
    assert old['loss'].pop('closure_objective') == 'one_step_and_rollout'
    assert new['loss'].pop('closure_objective') == 'all_start_anchor'
    assert old == new
    assert old['schema_version'] == 5
    assert old['model']['block_factor'] == 2
    assert old['model']['local_features'] == 12
    assert old['model']['theory_dim'] == 5
    assert old['model']['affine_reconstruction'] is True
    assert old['model']['kernel_initialization'] == 'random'
    assert old['loss']['geometry_mode'] == 'absolute'
    assert old['loss']['closure_scale_weights'] is None
    assert old['training']['seed'] == 91
    assert old['training']['epochs'] == 12000
    assert old['training']['max_rg_steps'] == 3
    assert old['training']['gradient_mode'] == 'qc_r_separate'
    assert old['training']['lr_milestones'] == [6000, 9000]


def test_profile_projection_separates_setup_steady_logging_and_writes():
    timing = dict(steady_step_seconds=[.01, .02], setup_seconds=10.,
                  logging_seconds=2., checkpoint_seconds=1.)
    summaries = {a: {'timing': deepcopy(timing)} for a in runner.ARMS}
    projected = runner.profile_projection(summaries, 40., 70., {'resource_policy': policy()})
    expected = 1.5*(.02*12000 + 10 + 61 + 14) + 120
    assert projected['arms']['control']['projected_gpu_seconds'] == expected
    assert projected['projected_total_gpu_seconds_conservative'] == 40 + 2*expected
    assert projected['projected_total_cpu_seconds_conservative'] == 70 + 4*expected + 1200
    assert projected['budget_passed'] is True
    assert projected['profile_is_scientific_result'] is False
    summaries['anchor']['timing']['steady_step_seconds'] = [1.]
    assert not runner.profile_projection(summaries, 40., 70., {'resource_policy': policy()})['budget_passed']


@pytest.mark.parametrize('bad', [[], [0.], [-1.], [float('nan')]])
def test_profile_rejects_nonpositive_or_missing_real_updates(bad):
    summaries = {a: {'timing': dict(steady_step_seconds=bad, setup_seconds=10.,
        logging_seconds=2., checkpoint_seconds=1.)} for a in runner.ARMS}
    with pytest.raises(ValueError):
        runner.profile_projection(summaries, 40., 70., {'resource_policy': policy()})


def test_commands_never_add_broad_recipe_override(tmp_path):
    command = runner.training_command(tmp_path, 'anchor', profile=True)
    assert '--train-child' in command and '--profile' in command
    assert '--allow-nondefault-recipe' not in command
    assert '--allow-legacy-training' not in command


def test_public_whitelist_never_exports_pt_or_mi(tmp_path):
    (tmp_path / 'comparison').mkdir()
    (tmp_path / 'mi_cache').mkdir()
    (tmp_path / 'mi_cache/private.pt').write_bytes(b'private')
    (tmp_path / 'comparison/public.npz').write_bytes(b'public')
    (tmp_path / 'comparison/report.json').write_text(json.dumps(
        dict(status='complete', public_files=['public.npz'])))
    records = runner.public_files(tmp_path)
    assert {r['path'] for r in records} == {'comparison/report.json', 'comparison/public.npz'}
    assert all(not r['path'].endswith('.pt') for r in records)
    (tmp_path / 'comparison/report.json').write_text(json.dumps(
        dict(status='complete', public_files=['../mi_cache/private.pt'])))
    with pytest.raises(ValueError):
        runner.public_files(tmp_path)


def test_public_whitelist_refuses_symlinks(tmp_path):
    (tmp_path / 'comparison').mkdir()
    (tmp_path / 'private.json').write_text('{}')
    (tmp_path / 'comparison/public.json').symlink_to(tmp_path / 'private.json')
    (tmp_path / 'comparison/report.json').write_text(json.dumps(
        dict(status='complete', public_files=['public.json'])))
    with pytest.raises(ValueError):
        runner.public_files(tmp_path)


def test_sha_and_stat_identity(tmp_path):
    target = tmp_path / 'input.json'
    target.write_bytes(b'abc')
    assert runner.sha(target) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert runner.stat_identity(target)['size'] == 3
    link = tmp_path / 'link.json'
    link.symlink_to(target)
    with pytest.raises(ValueError):
        runner.stat_identity(link)


def test_physical_contract_validates_disjoint_K_without_training(monkeypatch, tmp_path):
    import torch
    import recurrent_rg.runner as data_runner
    control = runner.configs(tmp_path / 'physical.pt')['control']
    def dataset(count, first):
        return SimpleNamespace(configurations=SimpleNamespace(
            shape=(count*3, 128, 32, 32, 1), dtype=torch.float64),
            couplings=torch.arange(first, first+count, dtype=torch.float64).repeat_interleave(3)[:, None])
    train, test = dataset(43, 1), dataset(21, 100)
    monkeypatch.setattr(data_runner, 'prepared_data', lambda cfg: (train, test))
    contract = runner.physical_contract(control)
    assert contract['K_splits_disjoint']
    assert contract['labels_used_in_training'] is False
    assert len(contract['train_K_values']) == 43
    test.couplings = train.couplings[:63].clone()
    with pytest.raises(ValueError, match='disjoint'):
        runner.physical_contract(control)


def test_physical_contract_rejects_wrong_replication(monkeypatch, tmp_path):
    import torch
    import recurrent_rg.runner as data_runner
    control = runner.configs(tmp_path / 'physical.pt')['control']
    train = SimpleNamespace(configurations=SimpleNamespace(
        shape=(129, 128, 32, 32, 1), dtype=torch.float64),
        couplings=torch.ones(129, 1, dtype=torch.float64))
    monkeypatch.setattr(data_runner, 'prepared_data', lambda cfg: (train, train))
    with pytest.raises(ValueError, match='three'):
        runner.physical_contract(control)
