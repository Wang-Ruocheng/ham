"""Small no-GPU/no-SSH tests for the explicitly requested single late arm."""
import hashlib

import pytest

from scripts import run_ising_closure_scale_late as runner


def prior_fixture(tmp_path, monkeypatch):
    prior, source, code = (tmp_path / name for name in ('prior', 'source', 'code'))
    for folder in (prior / 'data', prior / 'benchmark', code / 'scripts', source): folder.mkdir(parents=True)
    (code / 'scripts/science.py').write_text('# unchanged science')
    (prior / 'data/ensembles.pt').write_bytes(b'new data; never tensor loaded')
    runner.save(prior / 'pairs.json', {'new': 'pairs'})
    inputs = {n: runner.file_sha(prior / n) for n in ('data/ensembles.pt', 'pairs.json')}
    receipt = dict(input_sha256=inputs, source=str(source))
    late = runner.pair.make_config(prior / 'data/ensembles.pt')
    late['loss']['closure_scale_weights'] = [.2, 1.8]
    runner.save(prior / 'config_late.json', late)
    summary = dict(history=[{}, {}, {}], timing=dict(steady_step_seconds=[.1] * 35,
        setup_seconds=80., logging_seconds=6., checkpoint_seconds=2.), actual_epochs=40)
    runner.save(prior / 'benchmark/summary.json', summary)
    runner.save(prior / 'benchmark/manifest.json', {'initial_module_sha256': {'Q': 'q', 'C': 'c', 'R': 'r'}})
    runner.save(prior / 'plan.json', dict(source_code_sha256=runner.execution.source_hashes(code),
        configuration_sha256={'config_late.json': runner.file_sha(prior / 'config_late.json')},
        objective={'future_loss_weight': 0}, budget={'GPU_card_seconds': 3600}))
    runner.save(prior / 'job.json', dict(experiment=runner.pair.EXPERIMENT, status='budget_hold',
        workflow_complete=False, current=None, child_still_running=False, gpu_seconds_current_stage=0.,
        source=str(source), device='cuda:2', gpu_uuid=runner.pair.GPU_UUID,
        stages=[dict(name='benchmark', status='complete', returncode=0, pid=None)],
        parent_pid=None, gpu_seconds_completed=41.152130860835314, CPU_seconds_completed=61.918145,
        campaign_peak_own_gpu_mib=6671, input_reuse=receipt))
    monkeypatch.setattr(runner.pair, 'validated_source', lambda src: receipt)
    def validate(root, name, steps):
        assert root == prior and name == 'benchmark' and steps == 40
        if runner.read(root / 'benchmark/summary.json')['actual_epochs'] != steps:
            raise ValueError('not completed40')
        return summary
    monkeypatch.setattr(runner.pair, 'validate_arm', validate)
    monkeypatch.setattr(runner, 'alive', lambda pid: False)
    return prior, source, code


def test_reuse_carries_actual_once_and_single_arm_fits_original_budget(tmp_path, monkeypatch):
    prior, source, code = prior_fixture(tmp_path, monkeypatch)
    receipt, estimate = runner.validated_prior(prior, source, code)
    assert receipt['gpu_seconds'] == 41.152130860835314
    assert receipt['CPU_seconds'] == 61.918145
    assert receipt['reused_model_or_optimizer'] is False
    projection = runner.execution.remaining_projection(dict(gpu_seconds_completed=receipt['gpu_seconds'],
        CPU_seconds_completed=receipt['CPU_seconds']), estimate, arms=1, collections=1)
    assert projection['allowed_gpu_seconds'] == 3600
    assert projection['projected_total_gpu_seconds'] < 3600


@pytest.mark.parametrize('mutation', ['formal', 'live', 'incomplete', 'science', 'input', 'config', 'partial40'])
def test_rejects_unsafe_preflight_reuse(tmp_path, monkeypatch, mutation):
    prior, source, code = prior_fixture(tmp_path, monkeypatch)
    if mutation == 'formal': (prior / 'run_late').mkdir()
    elif mutation == 'live': monkeypatch.setattr(runner, 'alive', lambda pid: True)
    elif mutation == 'science': (code / 'scripts/science.py').write_text('# modified old science')
    elif mutation == 'input': (prior / 'data/ensembles.pt').write_bytes(b'changed')
    elif mutation == 'config': runner.save(prior / 'config_late.json', {'modified': True})
    elif mutation == 'partial40': runner.save(prior / 'benchmark/summary.json', {'actual_epochs': 20})
    else:
        job = runner.read(prior / 'job.json'); job['workflow_complete'] = True; runner.save(prior / 'job.json', job)
    with pytest.raises(ValueError): runner.validated_prior(prior, source, code)


def test_only_new_runner_report_are_excluded_from_science_comparison(tmp_path, monkeypatch):
    prior, source, code = prior_fixture(tmp_path, monkeypatch)
    for name in runner.NEW_FILES: (code / name).write_text('# new orchestration/report only')
    runner.validated_prior(prior, source, code)
    (code / 'scripts/new_other_science.py').write_text('# not excluded')
    with pytest.raises(ValueError, match='science SHA'): runner.validated_prior(prior, source, code)


def test_command_remains_fresh_late_no_future_predictor_or_control_execution(tmp_path):
    command = runner.pair.trainer_command(tmp_path, tmp_path / 'run_late', tmp_path / 'config_late.json')
    assert '--q-future-control' in command and '--q-future-weight' not in command
    assert '--resume-checkpoint' not in command and '--source' not in command
    assert 'cuda:2' in command and 'run_uniform' not in str(command)


def test_existing_output_is_never_restarted(tmp_path):
    root = tmp_path / 'old'; root.mkdir(); runner.save(root / 'job.json', {'status': 'failed'})
    with pytest.raises(FileExistsError, match='Fresh output'):
        runner.run(root, tmp_path / 'source', tmp_path / 'prior')
