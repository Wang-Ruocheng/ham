"""No SSH, GPU, physical sampling, or trained-model inference."""
import json

import pytest

from scripts import run_ising_large_size_probe as runner


def test_budget_caps_are_strict_and_independent_of_historical_campaign():
    assert runner.budget_ok(14400, 900, 2 * 1024**3)
    for args in ((14400.01, 0, 0), (0, 900.01, 0), (0, 0, 2 * 1024**3 + 1)):
        assert not runner.budget_ok(*args)
    assert runner.PRIOR_CPU_RESERVE > 0


def test_public_files_never_includes_raw_spins_or_checkpoint(tmp_path):
    for folder in ('code', 'data', 'report'):
        (tmp_path / folder).mkdir()
    for name in ('job.json', 'data/status.json', 'data/spins.npy', 'data/checkpoint.pt',
                 'report/arrays.npz', 'report/fig.png', 'code/source.py', 'code/input.json'):
        (tmp_path / name).write_bytes(b'fixture')
    assert runner.public_files(tmp_path) == ['data/status.json', 'job.json', 'report/arrays.npz', 'report/fig.png']


def test_resource_gate_requires_two_separated_gpu_observations(tmp_path, monkeypatch):
    base = dict(gpu_uuid=runner.GPU_UUID, free_mib=24000, processes=[], mem_available_gib=90,
                load1=9, cpu_count=24, io_full_avg10=.1)
    times = iter([0., .1, .2, 10.4, 10.5, 10.6])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(times))
    sleeps = []
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    monkeypatch.setattr(runner, 'snapshot', lambda: dict(base))
    job = dict(gpu_seconds_completed=0.)
    runner.wait_resources(tmp_path, job, gpu=True)
    assert sleeps == [10.1]
    assert len(job['resource_checks']) == 2
    assert 10 < job['gpu_seconds_completed'] < 11


def test_uuid_mismatch_stops_without_touching_other_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'snapshot', lambda: {'gpu_uuid': 'different'})
    with pytest.raises(RuntimeError, match='UUID'):
        runner.wait_resources(tmp_path, {}, gpu=True)


def test_launch_guards_reject_historical_outputs_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: runner.HOST)
    (tmp_path / 'code').mkdir()
    (tmp_path / 'job.json').write_text('{"historical": true}')
    with pytest.raises(FileExistsError):
        runner.run(tmp_path)
    assert json.loads((tmp_path / 'job.json').read_text()) == {'historical': True}


def test_regular_source_files_must_stay_in_scope(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{}')
    assert runner.ordinary(path, tmp_path) == path
    link = tmp_path / 'link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError): runner.ordinary(link, tmp_path)
    with pytest.raises(ValueError): runner.ordinary(path, tmp_path / 'subdir')


def test_source_metadata_no_training_or_mi_command_is_constructed(tmp_path):
    cmd = runner.command(tmp_path, 'scripts.ising_large_size_probe_sampling', 'sample', '--condition-index', 0)
    assert cmd[1:] == ['-m', 'scripts.ising_large_size_probe_sampling', '--root', str(tmp_path),
                       'sample', '--condition-index', '0']


def test_child_import_path_pins_new_probe_and_frozen_model(tmp_path):
    value = runner.stage_environment(tmp_path)
    assert value['PYTHONPATH'].split(runner.os.pathsep) == [str(tmp_path / 'code'), str(runner.FROZEN_CODE)]
    assert value['PYTHONDONTWRITEBYTECODE'] == '1'


def test_parent_projection_charges_actual_non_sampling_cpu_and_remaining_setup():
    benchmark = {'budget_gate': {'conditions': [dict(condition_index=k,
        projected_cpu_seconds=1000 * (k + 1), process_setup_cpu_seconds=50) for k in range(3)]}}
    projection = runner.cpu_projection_gate(benchmark, 1, actual_cpu=200)
    assert projection['remaining_sampling_cpu_seconds'] == 5100
    assert projection['projected_total_cpu_seconds'] == 6380
    assert projection['passed']
    assert not runner.cpu_projection_gate(benchmark, 0, actual_cpu=13000)['passed']


@pytest.mark.parametrize('stage,relative', [
    ('sampling_preflight', 'benchmark/status.json'),
    ('sample_c001', 'data/c001/status.json'),
    ('evaluation_preflight', 'gpu_preflight.json'),
    ('frozen_collection', 'collection/status.json'),
])
def test_explicit_budget_receipts_classified_without_conflating_failure(tmp_path, stage, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status":"budget_hold","requires_user_action":true}')
    assert runner.stage_budget_hold(tmp_path, stage)['requires_user_action']
    path.write_text('{"status":"failed","error":"Traceback"}')
    assert runner.stage_budget_hold(tmp_path, stage) is None
