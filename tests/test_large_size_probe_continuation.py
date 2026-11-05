"""Synthetic continuation checks: no SSH, MC, GPU, or model inference."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys
import types

import pytest
import scripts

from scripts import continue_ising_large_size_probe as continuation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    monkeypatch.setattr(continuation, 'PRIOR_ROOT', old)
    monkeypatch.setattr(continuation, 'ROOT', new)
    monkeypatch.setattr(continuation, 'live_probe_processes', lambda root: [])
    monkeypatch.setattr(continuation.base, 'assert_frozen', lambda root, plan: None)
    codes = {}
    for relative in continuation.ORIGINAL_FILES:
        for root in (old, new):
            path = root / 'code' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable original ' + relative)
        codes[relative] = continuation.base.sha(old / 'code' / relative)
    controller = new / 'code' / continuation.CONTROLLER
    controller.write_text('# independent continuation')
    source = dict(module_sha256={'Q': 'q'*64, 'C': 'c'*64, 'R': 'r'*64}, source_json_sha256={})
    for name in ('job.json', 'plan.json', 'manifest.json', 'summary.json'):
        write(old / 'source_record' / name, {'historical': name})
        source['source_json_sha256'][name] = continuation.base.sha(old / 'source_record' / name)
    sampling = dict(K=[.42,.44,.46], size=1080, replicas=2, samples=64, equil=4096, gap=128,
        initial_conditions=['hot','cold'], extra_pilots=0, independent_production_chains=True,
        no_equilibrium_certificate=True, seed=20261018)
    plan = dict(source=source, new_code_sha256=codes, sampling=sampling)
    job = dict(status='budget_hold', workflow_complete=False, current=None, child_still_running=False,
        training_updates=0, MI_computation=False, stages=[dict(name=name, status=status, returncode=code) for
            name,status,code in [('sampling_preflight','complete',0),('sample_c000','complete',0),
                                 ('evaluation_preflight','budget_hold',3)]],
        budget=dict(CPU_core_hours=4,GPU_card_seconds=900,GPU_peak_mib=8192,output_bytes=2*1024**3),
        CPU_seconds_completed=continuation.PRIOR_CPU, gpu_seconds_completed=continuation.PRIOR_GPU,
        gpu_seconds_current_stage=0.)
    write(old / 'job.json', job); write(old / 'plan.json', plan)
    benchmark = dict(status='complete', workflow_complete=True, budget_gate=dict(passed=True, conditions=[
        dict(condition_index=i, projected_cpu_seconds=50, process_setup_cpu_seconds=2) for i in range(3)]))
    write(old / 'benchmark/receipt.json', benchmark)
    write(old / 'benchmark/status.json', dict(status='complete',receipt_sha256=continuation.base.sha(old / 'benchmark/receipt.json')))
    spins = old / 'data/c000/spins.npy'
    spins.parent.mkdir(parents=True); spins.write_bytes(b'not really a spin array; metadata only synthetic test')
    digest = continuation.base.sha(spins)
    monkeypatch.setattr(continuation, 'C000_SHA', digest)
    sidecar = dict(status='complete',workflow_complete=True,spins_sha256=digest,
        spin_shape=[2,64,1080,1080],spin_dtype='int8',K0=.42,target_delta6=[0.]*6,
        sampling_budget=dict(replicas=2,samples=64,equil=4096,gap=128,independent_pilot_chains=0),
        quality={'warnings':['autocorrelation evidence only']})
    write(old / 'data/c000/sidecar.json', sidecar)
    write(old / 'data/c000/status.json', dict(status='complete',workflow_complete=True,spins_sha256=digest,
        sidecar_sha256=continuation.base.sha(old / 'data/c000/sidecar.json')))
    gate = dict(status='budget_hold',workflow_complete=False,benchmark_only=True,scientific_result=False,
        physical_predictions_saved=False,checkpoint_sha256=continuation.base.CHECKPOINT_SHA,
        physical_input_sha256=digest,gpu_budget_seconds=900,training_updates=0,
        mutual_information_evaluations=0,optimizer_created=False,all_parameters_frozen=True,all_grad_none=True,
        model_sha256_before='a'*64,model_sha256_after='a'*64,module_sha256_before=source['module_sha256'],
        module_sha256_after=source['module_sha256'],training_depth_unchanged=2,evaluation_max_rg_steps=3,
        projected_remaining_gpu_seconds=2228.5707053393126,elapsed_seconds=80.2541447672993,
        prior_gpu_seconds=10.276356780901551,projected_total_gpu_seconds=2319.1012068875134,
        measured_chain_draw_seconds=30.129859171807766,peak_gpu_allocated_bytes=4*1024**3,
        peak_gpu_reserved_bytes=5*1024**3)
    write(old / 'gpu_preflight.json', gate)
    return old,new


def test_exact_prefix_is_reused_without_mutating_original(prefix):
    old,new = prefix
    before = {str(path):continuation.base.sha(path) for path in old.rglob('*') if path.is_file()}
    state = continuation.validate_prior(old,new)
    derived = continuation.reuse_prefix(new,state)
    after = {str(path):continuation.base.sha(path) for path in old.rglob('*') if path.is_file()}
    assert before == after
    assert derived['status'] == 'complete'
    assert derived['benchmark_only'] and not derived['scientific_result']
    assert derived['no_new_execution_precheck']
    assert derived['elapsed_seconds'] == 0
    assert derived['original_elapsed_seconds'] == 80.2541447672993
    assert derived['prior_gpu_seconds'] == continuation.PRIOR_GPU
    assert derived['projected_total_gpu_seconds'] == continuation.PRIOR_GPU + 2228.5707053393126
    assert derived['gpu_budget_seconds'] == 2700
    assert continuation.base.read(new / 'data/c000/sidecar.json')['quality']['warnings']
    assert not (new / 'data/c001').exists()


@pytest.mark.parametrize('field,value', [('status','failed'),('child_still_running',True),
    ('CPU_seconds_completed',300),('gpu_seconds_completed',101),('training_updates',1)])
def test_invalid_prior_state_cannot_continue(prefix,field,value):
    old,new = prefix
    job = continuation.base.read(old / 'job.json'); job[field] = value
    write(old / 'job.json',job)
    with pytest.raises(ValueError): continuation.validate_prior(old,new)


def test_actual_process_guard_not_historical_pid_only(prefix,monkeypatch):
    old,new = prefix
    monkeypatch.setattr(continuation,'live_probe_processes',lambda root:[dict(pid=44)])
    with pytest.raises(RuntimeError,match='running processes'): continuation.validate_prior(old,new)


@pytest.mark.parametrize('field,value', [('all_grad_none',False),('all_parameters_frozen',False),
    ('scientific_result',True),('physical_predictions_saved',True),('training_depth_unchanged',3),
    ('physical_input_sha256','changed'),('peak_gpu_reserved_bytes',8*1024**3+1)])
def test_nonbudget_precheck_issue_never_readmitted(prefix,field,value):
    old,new = prefix
    gate = continuation.base.read(old / 'gpu_preflight.json'); gate[field] = value
    write(old / 'gpu_preflight.json',gate)
    with pytest.raises(ValueError): continuation.validate_prior(old,new)


def test_new_cap_remains_a_gate_not_forced_completion(prefix):
    old,new = prefix
    gate = continuation.base.read(old / 'gpu_preflight.json'); gate['projected_remaining_gpu_seconds'] = 2700
    write(old / 'gpu_preflight.json',gate)
    with pytest.raises(continuation.base.BudgetHold): continuation.validate_prior(old,new)


def test_original_workers_must_remain_byte_identical(prefix):
    old,new = prefix
    (new / 'code' / continuation.ORIGINAL_FILES[2]).write_text('# altered')
    with pytest.raises(ValueError,match='byte-for-byte'): continuation.validate_prior(old,new)


def test_copy_refuses_existing_files_and_never_overwrites(prefix):
    old,new = prefix
    state = continuation.validate_prior(old,new)
    continuation.reuse_prefix(new,state)
    with pytest.raises(FileExistsError): continuation.reuse_prefix(new,state)


def test_budget_adapters_keep_cpu_prior_once_and_combined_directory(prefix):
    old,new = prefix
    state = continuation.validate_prior(old,new)
    prior_limit = continuation.base.GPU_LIMIT
    expected_size = continuation.base.output_bytes(old) + continuation.base.output_bytes(new)
    with continuation.budget_adapters(state):
        assert continuation.base.GPU_LIMIT == 2700
        assert continuation.base.PRIOR_CPU_RESERVE == continuation.PRIOR_CPU + 30
        assert continuation.base.output_bytes(new) == expected_size
        assert continuation.base.NEW_FILES == continuation.ORIGINAL_FILES + (continuation.CONTROLLER,)
        projection = continuation.base.cpu_projection_gate(continuation.base.read(old / 'benchmark/receipt.json'),
            1,actual_cpu=continuation.PRIOR_CPU+30)
        assert projection['evaluation_and_report_cpu_reserve_seconds'] == 2880
        assert projection['passed']
    assert continuation.base.GPU_LIMIT == prior_limit


def test_commands_never_repeat_completed_stages(prefix):
    _,new = prefix
    stages = continuation.continuation_commands(new,continuation.PRIOR_GPU)
    assert [name for name,cmd,gpu in stages] == ['sample_c001','sample_c002','frozen_collection','report']
    assert stages[2][2]
    assert '--collect-worker' in stages[2][1]
    assert '--prior-gpu-seconds' in stages[2][1]
    assert not any('benchmark' in cmd for name,cmd,gpu in stages)


def test_collect_adapter_calls_original_function_not_old_main(prefix,monkeypatch):
    old,new = prefix
    plan = dict(experiment=continuation.EXPERIMENT,budget={'GPU_card_seconds':2700},
        continuation=dict(prior_root=str(old),prior_CPU_seconds_completed=continuation.PRIOR_CPU))
    job = dict(status='running',current=dict(name='frozen_collection',pid=continuation.os.getpid()),
        budget={'GPU_card_seconds':2700},training_updates=0,MI_computation=False,
        gpu_seconds_completed=continuation.PRIOR_GPU+11)
    write(new / 'plan.json',plan); write(new / 'job.json',job)
    @contextlib.contextmanager
    def noop(state): yield
    monkeypatch.setattr(continuation,'budget_adapters',noop)
    captured = []
    class Parser:
        def parse_args(self,argv):
            captured.append(argv)
            return argparse.Namespace(gpu_budget_seconds=float(argv[argv.index('--gpu-budget-seconds')+1]),
                prior_gpu_seconds=float(argv[argv.index('--prior-gpu-seconds')+1]))
    evaluator = types.ModuleType('scripts.ising_large_size_probe_evaluation')
    evaluator.build_parser = Parser
    evaluator.collect = lambda args: dict(budget=args.gpu_budget_seconds,prior=args.prior_gpu_seconds)
    evaluator.main = lambda: pytest.fail('Historical CLI900s admission must not be invoked')
    monkeypatch.setitem(sys.modules,'scripts.ising_large_size_probe_evaluation',evaluator)
    monkeypatch.setattr(scripts,'ising_large_size_probe_evaluation',evaluator,raising=False)
    result = continuation.collect_worker(new,job['gpu_seconds_completed'])
    assert result == dict(budget=2700.,prior=continuation.PRIOR_GPU+11)
    assert captured[0][0] == 'collect' and captured[0][captured[0].index('--device')+1] == 'cuda:2'
    with pytest.raises(ValueError,match='accounting'):
        continuation.collect_worker(new,continuation.PRIOR_GPU)

    real_read = continuation.base.read
    calls, sleeps = [], []
    def publish_race(path):
        if Path(path) == new / 'job.json':
            calls.append(True)
            return dict(job,current=None,status='waiting_resources') if len(calls) == 1 else job
        return real_read(path)
    monkeypatch.setattr(continuation.base,'read',publish_race)
    monkeypatch.setattr(continuation.time,'sleep',sleeps.append)
    assert continuation.collect_worker(new,job['gpu_seconds_completed']) == result
    assert sleeps == [.05]
    monkeypatch.setattr(continuation.base,'read',real_read)
    write(new / 'job.json',dict(job,current=dict(name='frozen_collection',pid=-3)))
    with pytest.raises(ValueError,match='running continuation'):
        continuation.collect_worker(new,job['gpu_seconds_completed'])
    assert len(sleeps) == 201


def test_live_process_inventory_requires_actual_module_and_scope(tmp_path):
    old = tmp_path / 'old'; old.mkdir()
    proc = tmp_path / 'proc'; proc.mkdir()
    for pid,argv,cwd in ((101,['python','-m','scripts.run_ising_large_size_probe','--root',str(old)],old),
        (102,['grep',str(old)],old), (103,['python','-m','scripts.run_ising_large_size_probe','--root','/other'],tmp_path)):
        directory = proc / str(pid); directory.mkdir()
        (directory / 'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
        (directory / 'cwd').symlink_to(cwd,target_is_directory=True)
    assert [row['pid'] for row in continuation.live_probe_processes(old,proc=proc)] == [101]


def test_existing_job_is_not_overwritten(prefix,monkeypatch):
    _,new = prefix
    monkeypatch.setattr(continuation.socket,'gethostname',lambda:continuation.base.HOST)
    write(new / 'job.json',{'immutable':True})
    with pytest.raises(FileExistsError): continuation.run(new)
    assert continuation.base.read(new / 'job.json') == {'immutable':True}
