"""Stage A pins one data/recipe and five seeds without implicit continuations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_ising_b0_multiseed as runner


def reference():
    path = Path(__file__).resolve().parents[1] / 'configs/ising2d_exponent_ablation12000_seed91/baseline_control.json'
    cfg = json.loads(path.read_text())
    cfg['data']['cache'] = str(runner.BASE / 'data/ising2d_positiveK_L108_b3_replicas12_seed91.pt')
    return cfg


def test_five_fresh_configs_differ_only_in_seed_without_changing_source():
    source = reference()
    old = deepcopy(source)
    configs = runner.configs(source)
    assert source == old and list(configs) == ['91', '92', '93', '94', '95']
    base = deepcopy(configs['91'])
    for seed, cfg in configs.items():
        current = deepcopy(cfg)
        assert current['training'].pop('seed') == int(seed)
        expected = deepcopy(base)
        expected['training'].pop('seed')
        assert current == expected
        assert cfg['model']['block_factor'] == 3 and cfg['model']['theory_dim'] == 5
        assert cfg['model']['local_features'] == 12
        assert cfg['loss']['closure_scale_weights'] is None
        assert cfg['training']['epochs'] == 12000
        assert cfg['training']['lr_milestones'] == (6000, 9000)
        assert cfg['training']['lr_schedule'] == 'multistep'
        assert cfg['training']['max_rg_steps'] == cfg['evaluation']['max_rg_steps'] == 2


def test_wrong_block_or_data_rejected_before_any_io():
    for section, key, value in [('model', 'block_factor', 2), ('data', 'cache', 'other.pt')]:
        cfg = reference()
        cfg[section][key] = value
        with pytest.raises(ValueError, match='audited'):
            runner.configs(cfg)


def policy():
    return dict(resource_policy=dict(profile_safety_factor=1.5,
        train_final_reserve_per_seed=120, cpu_threads=2, report_reserve_per_seed=1800,
        gpu_seconds=28800, cpu_seconds=86400, max_training_seconds_per_arm=7200))


def test_profile_accounts_for_all_formal_logs_checkpoints_and_reserves():
    summary = dict(timing=dict(steady_step_seconds=[.1, .2], setup_seconds=10,
        logging_seconds=4, checkpoint_seconds=1))
    report = runner.profile_projection(summary, 60, policy())
    arm = 1.5 * (.2*12000 + 10 + 2*61 + 14) + 120
    assert report['projected_seconds_per_training_arm'] == arm
    assert report['projected_total_gpu_seconds_conservative'] == 60+5*arm
    assert report['projected_total_cpu_seconds_conservative'] == (60+5*arm)*2+5*1800
    assert report['budget_passed'] and not report['profile_is_scientific_result']


def test_slow_profile_holds_instead_of_reducing_fixed_endpoint():
    summary = dict(timing=dict(steady_step_seconds=[1.], setup_seconds=10,
        logging_seconds=4, checkpoint_seconds=1))
    assert not runner.profile_projection(summary, 60, policy())['budget_passed']
    assert runner.UPDATES == 12000 and len(runner.SEEDS) == 5


@pytest.mark.parametrize('values', [[], [0.], [-1.], [float('nan')]])
def test_profile_refuses_invalid_measurements(values):
    with pytest.raises(ValueError):
        runner.profile_projection(dict(timing=dict(steady_step_seconds=values)), 0, policy())


def test_training_and_report_commands_stay_in_new_output(tmp_path):
    command = runner.training_command(tmp_path, {}, 93)
    assert '--train-child' in command and str(tmp_path / 'seed93/run') in command
    assert '--resume-checkpoint' not in command and '--allow-nondefault-recipe' not in command
    profile = runner.training_command(tmp_path, {}, None)
    assert '--profile' in profile and str(tmp_path / 'profile_run') in profile


def test_exclusive_json_and_stat_guards(tmp_path):
    path = tmp_path / 'a.json'
    runner.save(path, {'x': '中文'}, exclusive=True)
    assert runner.read(path) == {'x': '中文'}
    with pytest.raises(FileExistsError):
        runner.save(path, {}, exclusive=True)
    identity = runner.stat_identity(path)
    assert identity['size'] > 0
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        runner.stat_identity(link)


def test_public_files_never_include_physical_or_model_cache(tmp_path):
    for name in ('plan.json', 'diagram.png', 'arrays.npz', 'checkpoint.pt', 'data.pt', 'job.json'):
        (tmp_path / name).write_bytes(b'content')
    (tmp_path / 'profile_run').mkdir()
    (tmp_path / 'profile_run/summary.json').write_text('{}')
    assert {row['path'] for row in runner.public_files(tmp_path)} == {'plan.json'}
    report = tmp_path / 'seed91/report'
    report.mkdir(parents=True)
    runner.save(report / 'report.json', {'public_files': ['arrays.npz', 'report.json']})
    (report / 'arrays.npz').write_bytes(b'public arrays')
    assert {row['path'] for row in runner.public_files(tmp_path)} == {
        'plan.json', 'seed91/report/report.json', 'seed91/report/arrays.npz'}
    runner.save(report / 'report.json', {'public_files': ['../../checkpoint.pt']})
    with pytest.raises(ValueError, match='namespace'):
        runner.public_files(tmp_path)
