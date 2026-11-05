"""Independent phase-two queues, immutable evidence, truthful natural override."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising_coupling_representation_ablation as pipeline
from scripts.run_scalar_lr_sweep import make_configs


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    source, sequence = tmp_path / 'baseline' / 'run', tmp_path / 'sequence'
    data = tmp_path / 'physical.pt'
    data.write_bytes(b'existing physical cache must not be loaded or rehashed in planning')
    write(data.with_suffix('.json'), dict(data_sha256=pipeline.baseline.DATA_SHA))
    monkeypatch.setattr(pipeline.baseline, 'SIDECAR_SHA', pipeline.fingerprint(data.with_suffix('.json')))
    cfg = make_configs(data)['multistep']; cfg['model']['block_factor'] = 2
    records = {}
    for split, key in pipeline.baseline.MI_KEYS.items():
        path = source.parent / 'mi_cache' / f'mi-v1-{key}.pt'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'existing MI')
        records[split] = dict(hit=True, key=key, path=str(path), cache_version=1)
    metadata = dict(config=cfg, data_sha256=pipeline.baseline.DATA_SHA,
        labels_used_in_training=False, source=None,
        execution_optimizations=dict(mi_statistics_cache=records))
    write(source / 'manifest.json', metadata)
    write(source / 'summary.json', dict(config=cfg, actual_epochs=12000, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False))
    write(source.parent / 'job.json', dict(status='complete', workflow_complete=True,
        training=dict(actual_epochs=12000, manifest_sha256=pipeline.fingerprint(source / 'manifest.json'))))
    write(source.parent / 'mi_preprocessing.json', dict(status='complete', data_sha256=pipeline.baseline.DATA_SHA,
        model_initialized=False, optimizer_updates=0, splits={key: dict(cache=value) for key, value in records.items()}))
    old_code = tmp_path / 'phase_one_immutable.py'; old_code.write_text('unchanged stage one\n')
    old_source = tmp_path / 'old_source_receipt.json'; write(old_source, dict(status='complete'))
    write(sequence / 'plan.json', dict(smoke=False,
        code_sha256={str(old_code): pipeline.fingerprint(old_code)},
        source_sha256={str(old_source): pipeline.fingerprint(old_source)}))
    write(sequence / 'job.json', dict(status='phase1_complete', phase1_complete=True,
        workflow_complete=False, parent_pid=123, requires_user_action=False,
        stages=[dict(name='phase1_evidence', status='complete')], pending_experiments=[2, 3]))
    for dimension in (1, 3):
        write(sequence / f'phase1/dim{dimension}/report/report.json', dict(status='complete', report_complete=True))
        write(sequence / f'phase1/dim{dimension}/report/comparison.json', dict(phase1=dimension))
    write(sequence / 'phase1/comparison.json', dict(status='complete'))
    return source, sequence, cfg


def test_readonly_plan_exact_four_arms_budgets_and_only_GPU2_3(tmp_path, monkeypatch):
    source, sequence, base = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    monkeypatch.setattr(torch, 'load', lambda *_a, **_k: pytest.fail('Planning must not load tensors'))
    result = pipeline.build_plan(source, sequence, sequence / 'phase2')
    assert [arm['name'] for arm in result['arms']] == ['dim1_control', 'dim1_natural', 'dim3_control', 'dim3_natural']
    assert [arm['device'] for arm in result['arms']] == ['cuda:2', 'cuda:2', 'cuda:3', 'cuda:3']
    for arm in result['arms']:
        assert arm['config']['training']['epochs'] == 12000
        assert arm['config']['training']['threads'] == 2
        assert arm['config']['model']['block_factor'] == 2
        assert arm['config']['model']['theory_head'] == base['model']['theory_head'] == 'nonlinear'
        assert arm['config']['model']['affine_reconstruction'] == (arm['arm'] == 'control')
        allowed = []
        if arm['theory_dim'] != 1: allowed.append('model.theory_dim')
        if arm['arm'] == 'natural': allowed.append('model.affine_reconstruction')
        assert arm['changes'] == sorted(allowed)
        command = arm['commands']['training']
        assert command[command.index('--arm') + 1] == arm['arm']
        assert command[command.index('--device') + 1] == arm['device']
        assert command[command.index('--data-sha256') + 1] == pipeline.baseline.DATA_SHA
        flattened = ' '.join(' '.join(value) for value in arm['commands'].values())
        for forbidden in ('cuda:0', 'cuda:1', '--allow-nondefault-recipe', '--allow-legacy-training',
                          '--resume-checkpoint', 'prepare_qcr_mi', '--max-wall-seconds'):
            assert forbidden not in flattened
        assert arm['commands']['report'][:3] == arm['commands']['response'][:3] == ['nice', '-n', '10']
    assert result['protocol']['automatic_retry'] is False
    assert result['protocol']['Kc_or_exponent_target'] is False
    assert result['cache'] not in result['source_sha256']
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert not (sequence / 'phase2').exists()


def test_independent_smoke_all_four_on_GPU3_no_formal_reports_or_sequence_mutation(tmp_path, monkeypatch):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    before = (sequence / 'job.json').read_bytes()
    result = pipeline.build_plan(source, sequence, tmp_path / 'separate_smoke',
                                 smoke=True, smoke_device='cuda:3')
    assert result['updates'] == 2 and result['smoke_only'] is True
    assert result['comparison_command'] is None
    for arm in result['arms']:
        assert arm['device'] == 'cuda:3'
        assert arm['config']['training']['epochs'] == 2
        assert arm['config']['training']['log_every'] == arm['config']['training']['checkpoint_every'] == 1
        assert set(arm['commands']) == {'training'}
    pipeline.update_sequence(result, dict(status='complete'), complete=True)
    assert (sequence / 'job.json').read_bytes() == before


@pytest.mark.parametrize('kwargs', [dict(smoke_device='cuda:3'), dict(smoke=True, smoke_device='cuda:0'),
                                  dict(smoke=True, smoke_device='cpu')])
def test_smoke_device_cannot_change_formal_GPU_map(tmp_path, monkeypatch, kwargs):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        pipeline.build_plan(source, sequence, tmp_path / 'new', **kwargs)


@pytest.mark.parametrize('mutation', ['incomplete', 'duplicate', 'failed', 'smoke_source', 'changed_code', 'changed_source'])
def test_phase1_evidence_guard_prevents_invalid_or_duplicate_formal_start(tmp_path, monkeypatch, mutation):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    job = json.loads((sequence / 'job.json').read_text())
    spec = json.loads((sequence / 'plan.json').read_text())
    if mutation == 'incomplete': job['phase1_complete'] = False
    elif mutation == 'duplicate': job['phase2'] = dict(output='already')
    elif mutation == 'failed': job['requires_user_action'] = True
    elif mutation == 'smoke_source': spec['smoke'] = True
    elif mutation == 'changed_code': Path(next(iter(spec['code_sha256']))).write_text('different')
    else: Path(next(iter(spec['source_sha256']))).write_text('different')
    write(sequence / 'job.json', job); write(sequence / 'plan.json', spec)
    with pytest.raises(ValueError):
        pipeline.build_plan(source, sequence, sequence / 'phase2')
    assert not (sequence / 'phase2').exists()


def test_guards_output_nesting_and_stat_identity_without_rehashing_data(tmp_path, monkeypatch):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    for output in (sequence, sequence.parent, source / 'bad'):
        with pytest.raises((ValueError, FileExistsError)):
            pipeline.build_plan(source, sequence, output)
    result = pipeline.build_plan(source, sequence, sequence / 'phase2')
    pipeline.ensure_unchanged(result)
    Path(result['cache']).write_bytes(b'changed only metadata checked')
    with pytest.raises(ValueError, match='身份'):
        pipeline.ensure_unchanged(result)


def test_host_pressure_requires_memory_load_and_IO_budget_and_two_idle_check_contract(tmp_path):
    resources = dict(available_memory_bytes=75 * (1 << 30), cpu_count=24, load_average=[21.6, 18, 12],
                     io_pressure=dict(full=dict(avg10=2.43)))
    assert pipeline.host_ready(resources)
    for changed in (dict(available_memory_bytes=35 * (1 << 30)), dict(load_average=[31., 20, 12]),
                    dict(io_pressure=dict(full=dict(avg10=20.))), dict(cpu_count=None)):
        assert not pipeline.host_ready({**resources, **changed})
    path = tmp_path / 'io'; path.write_text('some avg10=3.10 avg60=1.20 avg300=0.40 total=123\nfull avg10=2.43 avg60=1.10 avg300=0.20 total=12\n')
    assert pipeline.io_pressure(path)['full']['avg10'] == 2.43
    path.write_text('some avg10=0.0\n')
    with pytest.raises(ValueError, match='I/O'):
        pipeline.io_pressure(path)
    assert pipeline.CHECK_INTERVAL == 10.


def test_sequence_update_preserves_phase1_evidence_and_stops_before_stage3(tmp_path, monkeypatch):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, sequence, sequence / 'phase2')
    original = json.loads((sequence / 'job.json').read_text())
    pipeline.update_sequence(plan, dict(status='running'))
    job = json.loads((sequence / 'job.json').read_text())
    assert job['status'] == 'phase2_running' and job['phase1_complete'] is True
    assert job['stages'] == original['stages'] and job['phase1_parent_pid'] == 123
    pipeline.update_sequence(plan, dict(status='complete'), complete=True)
    job = json.loads((sequence / 'job.json').read_text())
    assert job['status'] == 'phase2_complete' and job['phase2_complete'] is True
    assert job['workflow_complete'] is False and job['awaiting_scientific_review'] is True
    assert job['pending_experiments'] == [3] and job['next_experiment'] == 3
    assert job['stages'] == original['stages']


def initialization(kind):
    return dict(status='complete', fresh=True, source_checkpoint=None, seed=91, arm=kind,
        labels_used_in_training=False, C_tensor_sha256='c' * 64, Q_backbone_tensor_sha256='b' * 64,
        Q_tensor_sha256=('a' if kind == 'control' else 'd') * 64, R_tensor_sha256='e' * 64,
        data_reuse=dict(data_rehashed=False, data_full_audit_performed=False, coupling_labels_accessed=False))


def test_paired_initialization_requires_backbone_C_R_not_full_decoder_equality():
    rows = {f'dim{dimension}_{kind}': dict(initialization=initialization(kind))
            for dimension in (1, 3) for kind in ('control', 'natural')}
    result = pipeline.check_paired_initialization(rows)
    assert all(value['full_Q_equal'] is False for value in result.values())
    assert result['dim1']['Q_backbone_tensor_sha256'] is True
    rows['dim3_natural']['initialization']['R_tensor_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='随机初值'):
        pipeline.check_paired_initialization(rows)


def training_fixture(tmp_path, monkeypatch, kind='natural', dimension=1):
    from experiments.ising_coupling_representation import build_coupling_model, representation_contract
    from recurrent_rg.checkpoints import build_model, qcr_state_dict
    from recurrent_rg.representation import normalize_affine_decoder_
    from recurrent_rg.coupling_decoder import normalize_coupling_decoder_
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, sequence, tmp_path / 'smoke', smoke=True, smoke_device='cuda:3')
    arm = next(value for value in plan['arms'] if value['arm'] == kind and value['theory_dim'] == dimension)
    config = RunConfig.from_dict(arm['config'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model = (build_coupling_model if kind == 'natural' else build_model)(config.model, 2).double()
    parameters = list(model.theory_encoder.parameters()) + list(model.rg_kernel.parameters()) + list(model.theory_rg.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.001)
    for _ in range(2):
        for parameter in parameters: parameter.grad = torch.ones_like(parameter)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if kind == 'natural': normalize_coupling_decoder_(model.theory_encoder.distribution_decoder)
        else: normalize_affine_decoder_(model.theory_encoder)
    records = {split: dict(hit=True, key=key, cache_version=1,
        path=str(Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt')) for split, key in pipeline.baseline.MI_KEYS.items()}
    manifest = dict(config=arm['config'], config_source=arm['config_path'], data_sha256=pipeline.baseline.DATA_SHA,
        training_driver='ising_coupling_representation' if kind == 'natural' else 'rsmi_separate_qcr',
        precision='torch.float64', estimator='histogram', environment_mode='full_shell', buffer=1,
        independent_c=True, labels_used_in_training=False, q_supervision=None, rsmi_weight=2.,
        execution_device=arm['device'], source=None, execution_optimizations=dict(mi_statistics_cache=records))
    if kind == 'natural':
        manifest.update(underlying_training_driver='rsmi_separate_qcr', representation_override=representation_contract())
    policy = LearningRatePolicy(config.training)
    summary = dict(config=arm['config'], actual_epochs=2, requested_epochs=2, final_report_skipped=True,
        training_limit_seconds=None, lr_scheduler_state=policy.checkpoint_state(2),
        history=[dict(epoch=step, learning_rate=.001) for step in (1, 2)])
    run = Path(arm['run']); run.mkdir(parents=True)
    write(Path(arm['config_path']), arm['config'])
    write(run / 'manifest.json', manifest); write(run / 'summary.json', summary)
    write(Path(arm['root']) / 'initialization.json', initialization(kind))
    torch.save(dict(epoch=2, metadata=manifest, model=qcr_state_dict(model), optimizer=optimizer.state_dict()),
               run / 'training_step_2.pt')
    return plan, arm, manifest, summary


@pytest.mark.parametrize('kind,dimension', [('control', 1), ('natural', 1), ('control', 3), ('natural', 3)])
def test_training_receipt_validates_decoder_optimizer_actual_updates_and_cache(tmp_path, monkeypatch, kind, dimension):
    plan, arm, _, _ = training_fixture(tmp_path, monkeypatch, kind, dimension)
    receipt = pipeline.check_training(arm, plan)
    assert receipt['actual_epochs'] == receipt['optimizer_each_parameter_updates'] == 2
    assert receipt['decoder_weight_shape'] == [7 if kind == 'natural' else 16, dimension]
    assert receipt['initialization']['fresh'] is True
    assert receipt['labels_used_in_training'] is False


@pytest.mark.parametrize('kind', ['control', 'natural'])
def test_checkpoint_config_dataclass_tuples_equal_serialized_plan_lists(tmp_path, monkeypatch, kind):
    plan, arm, _, _ = training_fixture(tmp_path, monkeypatch, kind)
    path = Path(arm['run']) / 'training_step_2.pt'
    payload = torch.load(path, weights_only=True)
    # The real trainer stores RunConfig.to_dict() in the binary checkpoint,
    # while plan/manifest/summary pass through JSON and retain lists instead.
    checkpoint_config = RunConfig.from_dict(arm['config']).to_dict()
    assert isinstance(checkpoint_config['training']['lr_milestones'], tuple)
    assert isinstance(arm['config']['training']['lr_milestones'], list)
    assert checkpoint_config != arm['config']
    payload['metadata']['config'] = checkpoint_config
    torch.save(payload, path)
    receipt = pipeline.check_training(arm, plan)
    assert receipt['actual_epochs'] == receipt['optimizer_each_parameter_updates'] == 2
    assert pipeline.canonical_configuration(checkpoint_config) == arm['config']


def test_checkpoint_canonical_config_still_rejects_semantic_recipe_change(tmp_path, monkeypatch):
    plan, arm, _, _ = training_fixture(tmp_path, monkeypatch)
    path = Path(arm['run']) / 'training_step_2.pt'
    payload = torch.load(path, weights_only=True)
    payload['metadata']['config'] = RunConfig.from_dict(arm['config']).to_dict()
    payload['metadata']['config']['loss']['rollout_weight'] = 6.
    torch.save(payload, path)
    with pytest.raises(ValueError, match='配置不符'):
        pipeline.check_training(arm, plan)


@pytest.mark.parametrize('mutation', ['label', 'wrong_override', 'cache_miss', 'wrong_steps', 'wrong_lr', 'data_audit', 'optimizer_decoder'])
def test_training_validation_rejects_bad_contract_or_decoder_optimizer(tmp_path, monkeypatch, mutation):
    plan, arm, manifest, summary = training_fixture(tmp_path, monkeypatch)
    run = Path(arm['run'])
    if mutation == 'label': manifest['labels_used_in_training'] = True
    elif mutation == 'wrong_override': manifest['representation_override']['kind'] = 'implicit'
    elif mutation == 'cache_miss': manifest['execution_optimizations']['mi_statistics_cache']['train']['hit'] = False
    elif mutation == 'wrong_steps': summary['actual_epochs'] = 1
    elif mutation == 'wrong_lr': summary['history'][-1]['learning_rate'] = .0003
    elif mutation == 'data_audit':
        value = initialization('natural'); value['data_reuse']['data_rehashed'] = True
        write(Path(arm['root']) / 'initialization.json', value)
    else:
        path = run / 'training_step_2.pt'; checkpoint = torch.load(path, weights_only=True)
        checkpoint['optimizer']['param_groups'][0]['params'].pop()
        torch.save(checkpoint, path)
    write(run / 'manifest.json', manifest); write(run / 'summary.json', summary)
    with pytest.raises(ValueError):
        pipeline.check_training(arm, plan)


def _execute_smoke_mock(tmp_path, monkeypatch, fail_first=False):
    source, sequence, _ = fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, sequence, tmp_path / 'smoke', smoke=True, smoke_device='cuda:3')
    old = (sequence / 'job.json').read_bytes()
    now = iter(range(0, 100000, 11))
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: next(now))
    monkeypatch.setattr(pipeline.time, 'sleep', lambda *_a: None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(pipeline.baseline, 'gpu_snapshot', lambda gpu:
        dict(compute_pids=[], free_mib=24000, gpu=gpu))
    monkeypatch.setattr(pipeline, 'host_resources', lambda:
        dict(available_memory_bytes=75 * (1 << 30), cpu_count=24, load_average=[10., 10., 10.],
             io_pressure=dict(full=dict(avg10=2.))))
    live, starts = set(), []
    class Child:
        def __init__(self, command, **_kwargs):
            assert command[command.index('--device') + 1] == 'cuda:3'
            assert not live
            self.pid = len(starts) + 400
            self.command = command; self.polled = 0
            live.add(self.pid); starts.append(command)
        def poll(self):
            self.polled += 1
            if self.polled < 2: return None
            live.discard(self.pid)
            return 7 if fail_first and self.pid == 400 else 0
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Child)
    monkeypatch.setattr(pipeline, 'check_training', lambda arm, _plan:
        dict(status='complete', actual_epochs=2, initialization=initialization(arm['arm'])))
    result = pipeline.execute(plan)
    assert (sequence / 'job.json').read_bytes() == old
    return result, plan, starts


def test_smoke_scheduler_never_overlaps_GPU3_and_no_reports_or_product_mutation(tmp_path, monkeypatch):
    code, plan, starts = _execute_smoke_mock(tmp_path, monkeypatch)
    assert code == 0 and len(starts) == 4
    assert all('experiments.ising_coupling_representation' in command for command in starts)
    job = json.loads((Path(plan['output']) / 'job.json').read_text())
    assert job['status'] == 'smoke_complete' and job['workflow_complete'] is True
    assert len(job['training']) == 4 and not job['reports']


def test_failure_no_retry_no_next_training_and_failure_receipt(tmp_path, monkeypatch):
    code, plan, starts = _execute_smoke_mock(tmp_path, monkeypatch, fail_first=True)
    assert code == 1 and len(starts) == 1
    job = json.loads((Path(plan['output']) / 'job.json').read_text())
    assert job['status'] == 'failed' and job['requires_user_action'] is True
    assert job['workflow_complete'] is False and 'returncode=7' in job['error']
    assert len(job['stages']) == 1
