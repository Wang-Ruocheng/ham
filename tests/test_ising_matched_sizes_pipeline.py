import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts import run_ising_matched_sizes as pipeline


def test_no_historical_large_cache_hash(tmp_path):
    path = tmp_path/'large.pt'
    with path.open('wb') as stream:
        stream.truncate(33 * (1 << 20))
    with pytest.raises(ValueError, match='大物理数据'):
        pipeline.fingerprint(path)


def test_hash_refuses_symlink(tmp_path):
    original = tmp_path/'public.json'; original.write_text('{}')
    link = tmp_path/'alias.json'; link.symlink_to(original)
    with pytest.raises(ValueError):
        pipeline.fingerprint(link)


def test_atomic_status_and_preserved_fields(tmp_path):
    pipeline.save(tmp_path/'job.json', dict(status='phase2_complete', phase1_complete=True, old=[1, 2]))
    pipeline.update_sequence(tmp_path, status='phase3_running', workflow_complete=False)
    result = pipeline.read(tmp_path/'job.json')
    assert result == dict(status='phase3_running', phase1_complete=True, old=[1, 2], workflow_complete=False)
    assert not (tmp_path/'job.json.writing').exists()


def fake_source(tmp_path):
    source, seq = tmp_path/'source', tmp_path/'sequence'
    source.mkdir(); seq.mkdir(); (source/'comparison').mkdir()
    pipeline.save(seq/'job.json', dict(status='phase2_complete', phase2_complete=True))
    pipeline.save(source/'comparison/report.json', dict(report_complete=True, workflow_complete=True,
                                                       actual_updates_per_arm=12000))
    training = {}
    for arm in pipeline.ARMS:
        for part in ('run', 'report', 'response'):
            (source/arm/part).mkdir(parents=True)
        cfg = dict(schema_version=5, model=dict(block_factor=2, width=3,
            kernel_layout='nonoverlapping', fixed_kernel=False, kernel_initialization='random'),
            training=dict(max_rg_steps=2))
        pipeline.save(source/arm/'run/summary.json', dict(actual_epochs=12000, requested_epochs=12000, config=cfg))
        pipeline.save(source/arm/'report/report.json', dict(report_complete=True, checkpoint_epoch=12000))
        pipeline.save(source/arm/'response/status.json', dict(status='complete', workflow_complete=True))
        pipeline.save(source/arm/'response/diagnostic.json', dict(K=list(pipeline.K_VALUES)))
        training[arm] = dict(actual_epochs=12000, new_updates=12000, labels_used_in_training=False)
    pipeline.save(source/'job.json', dict(status='complete', workflow_complete=True, smoke_only=False,
                                         training=training))
    return source, seq


def test_formal_source_gate_and_duplicate_claim(tmp_path):
    source, seq = fake_source(tmp_path)
    assert pipeline.validate_phase2(source, seq)[1]['status'] == 'complete'
    (seq/'phase3.claim.json').write_text('{}')
    with pytest.raises(ValueError, match='重复'):
        pipeline.validate_phase2(source, seq)
    pipeline.validate_phase2(source, seq, smoke=True)


@pytest.mark.parametrize('failure', ['training', 'labels', 'geometry', 'response', 'wrong_k', 'early_stop'])
def test_source_gate_rejects_incomplete_or_changed_protocol(tmp_path, failure):
    source, seq = fake_source(tmp_path); arm = pipeline.ARMS[0]
    if failure in ('training', 'labels'):
        path = source/'job.json'; j = pipeline.read(path)
        j['training'][arm]['actual_epochs' if failure == 'training' else 'labels_used_in_training'] = 11999 if failure == 'training' else True
    elif failure in ('geometry', 'early_stop'):
        path = source/arm/'run/summary.json'; j = pipeline.read(path)
        if failure == 'geometry': j['config']['model']['block_factor'] = 3
        else: j['stopped_by_wall_limit'] = True
    elif failure == 'response':
        path = source/arm/'response/status.json'; j = pipeline.read(path); j['workflow_complete'] = False
    else:
        path = source/arm/'response/diagnostic.json'; j = dict(K=[.433, .4406867935, .448])
    pipeline.save(path, j)
    with pytest.raises(ValueError): pipeline.validate_phase2(source, seq)


def test_no_training_or_gpu_commands_declared():
    assert len(pipeline.ARMS) == 4
    assert pipeline.K_VALUES[2] - pipeline.K_VALUES[1] != pytest.approx(pipeline.K_VALUES[1] - pipeline.K_VALUES[0])
    code = Path(pipeline.__file__).read_text()
    assert "'new_training_updates':" not in code  # all budgets are an explicit zero, not a user flag
    assert 'terminate(' not in code and 'kill(' not in code
    assert 'cuda:' not in code


def test_plan_all_eleven_stages_budget_and_no_writes(tmp_path, monkeypatch):
    source, seq = fake_source(tmp_path)
    j = pipeline.read(source/'job.json')
    for tr in j['training'].values():
        tr['checkpoint_sha256'] = 'a' * 64
    pipeline.save(source/'job.json', j)
    monkeypatch.setattr(pipeline, 'fingerprint', lambda path: 'a' * 64)
    output = tmp_path/'formal'
    plan = pipeline.build_plan(source, seq, output)
    assert len(plan['commands']) == 11
    assert [s['kind'] for s in plan['commands']].count('sampling') == 2
    assert [s['kind'] for s in plan['commands']].count('collection') == 8
    assert plan['production_saved_configurations'] == 9216
    assert plan['total_wolff_single_cluster_updates'] == 2310144
    assert plan['initial_sizes'] == [54, 108] and plan['terminal_size'] == 27
    assert plan['new_training_updates'] == plan['MI_estimates'] == 0
    assert all('train' not in s['command'] and '--resume' not in s['command'] for s in plan['commands'])
    assert not output.exists() and not (seq/'phase3.claim.json').exists()
    smoke = pipeline.build_plan(source, seq, tmp_path/'smoke', smoke=True)
    assert smoke['production_saved_configurations'] == 96
    assert smoke['total_wolff_single_cluster_updates'] == 960
    assert all('--smoke' in s['command'] for s in smoke['commands'])


def test_failure_never_retries_or_updates_formal_sequence_for_smoke(tmp_path, monkeypatch):
    root, sequence = tmp_path/'smoke', tmp_path/'sequence'
    sequence.mkdir()
    pipeline.save(sequence/'job.json', dict(status='phase2_complete', phase1_complete=True))
    before = (sequence/'job.json').read_bytes()
    spec = dict(output=str(root), sequence_root=str(sequence), source_sha256={}, code_sha256={},
                commands=[dict(name='first', kind='sampling', command=['first'], marker='unused'),
                          dict(name='second', kind='sampling', command=['second'], marker='unused')])
    monkeypatch.setattr(pipeline, 'build_plan', lambda *args, **kwargs: spec)
    monkeypatch.setattr(pipeline, 'ensure_cpu_policy', lambda: None)
    monkeypatch.setattr(pipeline, 'resource_snapshot', lambda: dict(ready=True))
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **options):
            calls.append(command)
            assert options['env']['OMP_NUM_THREADS'] == '1'
        def wait(self): return 7
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Child)
    args = SimpleNamespace(source_root=tmp_path/'source', sequence_root=sequence,
                           output=root, seed=20261016, smoke=True)
    assert pipeline.execute(args) == 1
    assert calls == [['first']]
    job = pipeline.read(root/'job.json')
    assert job['status'] == 'failed' and job['requires_user_action'] is True
    assert job['stages'][0]['returncode'] == 7
    assert (sequence/'job.json').read_bytes() == before
    assert not (sequence/'phase3.claim.json').exists()
    with pytest.raises(FileExistsError): pipeline.execute(args)
    assert calls == [['first']]


def test_dry_run_does_not_create_output_or_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path/'formal'
    monkeypatch.setattr(pipeline, 'build_plan', lambda *args, **kwargs: dict(output=str(root)))
    assert pipeline.main(['--source-root', str(tmp_path/'source'), '--sequence-root', str(tmp_path/'sequence'),
                          '--output', str(root), '--dry-run']) == 0
    assert not root.exists() and not (tmp_path/'logs').exists()


def test_detach_receipt_is_exclusive_and_environment_is_bounded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, 'build_plan', lambda *args, **kwargs: {})
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **options):
            calls.append(command)
            assert options['env']['NUMBA_NUM_THREADS'] == options['env']['OPENBLAS_NUM_THREADS'] == '1'
            assert command[:3] == ['nice', '-n', '10']
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Child)
    argv = ['--source-root', str(tmp_path/'source'), '--sequence-root', str(tmp_path/'sequence'),
            '--output', str(tmp_path/'formal'), '--detach']
    assert pipeline.main(argv) == 0
    with pytest.raises(FileExistsError): pipeline.main(argv)
    assert len(calls) == 1
    assert pipeline.read(tmp_path/'logs/formal.launch.json')['parent_pid'] == 123


def test_foreground_policy_only_lowers_own_priority(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.os, 'getpriority', lambda which, pid: 0)
    monkeypatch.setattr(pipeline.os, 'setpriority', lambda *args: calls.append(args))
    pipeline.ensure_cpu_policy()
    assert calls == [(pipeline.os.PRIO_PROCESS, 0, 10)]
    calls.clear()
    monkeypatch.setattr(pipeline.os, 'getpriority', lambda which, pid: 15)
    pipeline.ensure_cpu_policy()
    assert calls == []
