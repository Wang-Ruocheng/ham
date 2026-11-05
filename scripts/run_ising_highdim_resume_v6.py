"""Authorized report-only optimization handoff; completed training is immutable."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import socket
import sys
import time

from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from scripts import run_ising_highdim_resume_v5 as v5

EXECUTION_VERSION = 'resume_v6'
OLD_ROOT = Path('/mnt/nas-new/home/yange/ham/results/ising2d_b2_highdim_seed91_vm225_20261021_resume_v5')
OUTPUT_ROOT = Path('/mnt/nas-new/home/yange/ham/results/ising2d_b2_highdim_seed91_vm225_20261021_resume_v6')
INTERRUPT_RECEIPT_PATH = Path('/mnt/nas-new/home/yange/ham/logs/ising2d_b2_highdim_seed91_vm225_20261021_resume_v5.report_optimization_interrupt.json')
HANDOFF_NAME = 'report_optimization_v6_handoff.json'
PREVIOUS_ERROR = 'RuntimeError: report_dim5 returncode=-15; no retry/repair/restart'
EXPECTED_OLD_STAGES = ['train_dim5', 'report_dim5']
ALLOWED_NEW_MODULES = {'scripts/run_ising_highdim_resume_v6.py',
    'scripts/report_ising_highdim_fast.py', 'scripts/ising_highdim_fast_evaluation.py'}
REPORT_WALL_SECONDS = 900
REPORT_SECONDS = 840
canonical = v5.canonical
checkpoint_receipt = v5.checkpoint_receipt


class AnalysisBudgetHold(RuntimeError):
    """An explicit incomplete-analysis stop, never an absent-root conclusion."""


def prepare_handoff(interrupted_root, output):
    root, output = Path(interrupted_root).resolve(), Path(output).resolve()
    if (root != OLD_ROOT.resolve() or output != OUTPUT_ROOT.resolve()
            or root == output or root in output.parents or output in root.parents
            or output.exists() or output.is_symlink()):
        raise RuntimeError('Require the unique authorized v5-to-v6 report optimization handoff')
    plan, job = original.read(root / 'plan.json'), original.read(root / 'job.json')
    interruption = original.read(INTERRUPT_RECEIPT_PATH)
    stages = job.get('stages', [])
    if (plan.get('execution_version') != 'resume_v5' or job.get('execution_version') != 'resume_v5'
            or Path(plan.get('output', '')).resolve() != root
            or job.get('status') != 'failed' or job.get('workflow_complete') is not False
            or job.get('current') != 'report_dim5' or job.get('error') != PREVIOUS_ERROR
            or job.get('requires_user_action') is not True or job.get('completed_arms') != ['dim1']
            or job.get('host') != original.HOST or job.get('device') != 'cuda:2'
            or job.get('gpu_uuid') != original.GPU_UUID
            or job.get('labels_used_in_training') is not False or job.get('physical_sampling_updates') != 0
            or [s.get('name') for s in stages] != EXPECTED_OLD_STAGES
            or [(s.get('status'), s.get('returncode')) for s in stages] != [('complete', 0), ('failed', -15)]
            or interruption.get('status') != 'interrupted' or interruption.get('stage') != 'report_dim5'
            or interruption.get('job_error') != PREVIOUS_ERROR
            or any(interruption.get(key) is not True for key in ('user_authorized',
                'only_owned_report_session', 'completed_training_preserved',
                'other_processes_untouched', 'dim5_checkpoint_unchanged'))
            or interruption.get('parent', {}).get('pid') != job.get('parent_pid')
            or interruption.get('report_child', {}).get('pid') != stages[-1].get('pid')):
        raise RuntimeError('Prior report was not the explicitly authorized optimization interruption')
    for pid in {job.get('parent_pid'), *(s.get('pid') for s in stages)}:
        if type(pid) is not int or pid <= 0 or low.process_identity(pid) is not None:
            raise RuntimeError('A prior recorded process is still present; fail closed')
    profile_path = root / 'profiles/report_dim5.json'
    profile = original.read(profile_path)
    if (profile.get('status') != 'failed' or profile.get('returncode') != -15
            or profile.get('violation') or profile.get('monitor_error')
            or str(profile.get('child_starttime_ticks')) != str(interruption['report_child']['starttime_ticks'])):
        raise RuntimeError('Interrupted report profile differs from user-stop evidence')
    if [a.get('name') for a in plan.get('arms', [])] != ['dim1', 'dim5', 'dim10', 'dim100']:
        raise RuntimeError('Authorized four-arm comparison changed')
    for arm in plan['arms'][2:]:
        if any((Path(arm['root']) / name).exists() for name in ('run', 'report', 'initialization.json')):
            raise RuntimeError('A remaining fresh arm already started')
    old_report_marker = root / 'dim5/report/report.json'
    if old_report_marker.exists():
        marker = original.read(old_report_marker)
        if marker.get('status') == 'complete' and marker.get('report_complete') is True:
            raise RuntimeError('Prior dim5 report already completed; do not silently rerun it')
    original.ensure_unchanged(plan, root / 'code', plan['source_code_sha256'], plan['configuration_sha256'])
    v5.ensure_source(plan)
    with v5.continuation_resources():
        low.profile_receipt(original.read(root / 'profiles/train_dim5.json'))
        completed = v5.validate_training(plan, plan['arms'][1])
    if canonical(completed) != canonical(job.get('receipts', {}).get('train_dim5')):
        raise RuntimeError('Completed dim5 training differs from its authoritative receipt')
    checkpoint = root / 'dim5/run/training_step_12000.pt'
    selected = checkpoint_receipt(checkpoint, plan['arms'][1]['config'], step=12000)
    if (selected['sha256'] != completed.get('final_checkpoint_sha256')
            or selected['sha256'] != interruption.get('completed_dim5_checkpoint_sha256')):
        raise RuntimeError('Completed dim5 checkpoint changed')
    dim1 = original.report_receipt(Path(plan['arms'][0]['root']) / 'report', 1)
    if canonical(dim1) != canonical(plan['arms'][0]['inherited_report_receipt']):
        raise RuntimeError('Inherited dim1 report changed')
    inherited = dict(plan['interrupted_launch']['inherited_public_sha256'])
    for arm in plan['arms']:
        init = root / 'smoke' / arm['name'] / 'initialization.json'
        inherited[str(init)] = original.sha(init)
    for name, digest in dim1['files'].items():
        inherited[str(Path(dim1['path']) / name)] = digest
    for path in (checkpoint, root / 'dim5/run/manifest.json', root / 'dim5/run/summary.json',
                 root / 'dim5/run/history_progress.json', root / 'dim5/resume.json',
                 root / 'profiles/train_dim5.json', root / 'measured_resource_gate.json'):
        inherited[str(path)] = original.sha(path)
    for path, digest in inherited.items():
        if original.sha(path) != digest:
            raise RuntimeError('Inherited evidence changed')
    return dict(schema_version=1, status='approved_report_optimization_continuation',
        authorized_request='中断报告、优化生成效率并确保高维时间可控', user_authorized=True,
        old_directory_preserved=True, termination_confirmed=True, new_root=str(output),
        inherited_completed_arms=['dim1'], inherited_training_arms=['dim1', 'dim5'],
        old_plan_sha256=original.sha(root / 'plan.json'), old_job_sha256=original.sha(root / 'job.json'),
        old_profile_sha256=original.sha(profile_path), old_manifest_sha256=original.sha(root / 'dim5/run/manifest.json'),
        final_checkpoint_sha256=selected['sha256'], interrupt_receipt_sha256=original.sha(INTERRUPT_RECEIPT_PATH),
        inherited_public_sha256=inherited, inherited_dim5_training_receipt=completed,
        inherited_dim1_report_receipt=dim1,
        checkpoint={k:v for k,v in selected.items() if k not in ('history', 'metadata')},
        parent_pid=job['parent_pid'], child_pid=stages[-1]['pid'],
        report_wall_seconds=REPORT_WALL_SECONDS, report_compute_seconds=REPORT_SECONDS,
        scientific_training_unchanged=True, physical_sampling_updates=0)


def verify_handoff(interrupted_root, output):
    root = Path(interrupted_root).resolve()
    path = root / HANDOFF_NAME
    receipt = original.read(path)
    if canonical(receipt) != canonical(prepare_handoff(root, output)):
        raise RuntimeError('Report optimization handoff evidence or authorization changed')
    return dict(root=str(root), receipt=receipt, receipt_sha256=original.sha(path),
                inherited_public_sha256=receipt['inherited_public_sha256'])


def ensure_source(plan):
    record = plan['interrupted_launch']; root = Path(record['root'])
    paths = dict(old_plan_sha256=root/'plan.json', old_job_sha256=root/'job.json',
        old_profile_sha256=root/'profiles/report_dim5.json', old_manifest_sha256=root/'dim5/run/manifest.json',
        final_checkpoint_sha256=root/'dim5/run/training_step_12000.pt',
        interrupt_receipt_sha256=INTERRUPT_RECEIPT_PATH)
    if original.sha(root/HANDOFF_NAME) != record['receipt_sha256']:
        raise RuntimeError('Approved v6 handoff changed')
    for key, path in paths.items():
        if original.sha(path) != record['receipt'][key]:
            raise RuntimeError('Interrupted report/completed training evidence changed')
    for path, digest in record['inherited_public_sha256'].items():
        if original.sha(path) != digest:
            raise RuntimeError('Inherited evidence changed')
    old = original.read(root/'plan.json')
    original.ensure_unchanged(old, root/'code', old['source_code_sha256'], old['configuration_sha256'])
    v5.ensure_source(old)


def validate_code_delta(plan, project):
    old = original.read(Path(plan['interrupted_launch']['root'])/'plan.json')['source_code_sha256']
    current = original.source_hashes(project)
    if any(current.get(name) != digest for name,digest in old.items()) or set(current)-set(old) != ALLOWED_NEW_MODULES:
        raise RuntimeError('Only the three new report/orchestration modules may be added; all v5 science stays frozen')
    return dict(existing_changed_files=[], added_files=sorted(ALLOWED_NEW_MODULES), all_v5_files_unchanged=True)


def report_command(run, output, progress):
    return ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', f'{REPORT_WALL_SECONDS}s',
        sys.executable, '-u', '-m', 'scripts.report_ising_highdim_fast', '--run', str(run),
        '--output', str(output), '--checkpoint', '12000', '--threads', '1', '--sampling-repeats', '5',
        '--max-seconds', str(REPORT_SECONDS), '--progress-path', str(progress)]


def build_plan(interrupted_root, output):
    record = verify_handoff(interrupted_root, output)
    old = original.read(Path(record['root'])/'plan.json')
    plan = original.build_plan(old['source_run'], output)
    keys = ('source_run','source_sha256','immutable_stat_identity','data_sha256','mi_cache_dir','mi_cache_keys')
    if (canonical([a['config'] for a in plan['arms']]) != canonical([a['config'] for a in old['arms']])
            or any(canonical(plan[k]) != canonical(old[k]) for k in keys)):
        raise RuntimeError('Scientific configurations or input identities changed')
    root = Path(plan['output'])
    for arm, previous in zip(plan['arms'],old['arms']):
        arm.pop('smoke')
        inherited = arm['name']=='dim1'
        arm.update(inherited=inherited, training_inherited=arm['name'] in ('dim1','dim5'),
                   resume_from=None, segment_start_update=12000 if arm['name'] in ('dim1','dim5') else 0,
                   new_updates=0 if arm['name'] in ('dim1','dim5') else 12000)
        if inherited:
            arm.update(root=previous['root'], config_path=previous['config_path'], training=None, report=None,
                run_path=str(Path(previous['root'])/'run'), inherited_report_receipt=record['receipt']['inherited_dim1_report_receipt'])
            continue
        if arm['training_inherited']:
            arm.update(training=None, config_path=previous['config_path'],
                       run_path=str(Path(previous['root'])/'run'))
        else:
            arm['run_path']=str(Path(arm['root'])/'run')
            command=arm['training'];command[command.index('scripts.run_ising_highdim')]='scripts.run_ising_highdim_resume_v5'
            command.extend(['--benchmark-warmup','0'])
        arm['report']=report_command(arm['run_path'],Path(arm['root'])/'report',Path(arm['root'])/'report_progress.json')
    plan.update(schema_version=6,execution_version=EXECUTION_VERSION,interrupted_launch=record,
        inherited_completed_arms=['dim1'],inherited_training_arms=['dim1','dim5'],
        low_ram_execution=deepcopy(old['low_ram_execution']),
        measured_resource_gate=deepcopy(original.read(Path(record['root'])/'measured_resource_gate.json')))
    plan['protocol'].update(resume=True,automatic_retry=False,
        continuation='继承dim1完成报告和dim5累计12000训练；只重报dim5，dim10/100全新各12000',
        initialization='dim1/5仅继承已完成模型，不做新更新；dim10/100原配置随机Q/C/R与fresh Adam',
        source_usage='继承的小型checkpoint和回执只读；正式训练复用原物理data/data-only MI，身份核验仅stat，不重采/重hash/重建MI',
        reporting='全量数据和原5次C重复/seed/chunk均保留；整数模式统计和解析DR经等价验证；不减求根标准',
        report_budget=dict(compute_seconds=REPORT_SECONDS,wall_seconds=REPORT_WALL_SECONDS,kill_grace_seconds=5,
            action='analysis_budget_hold/incomplete；保留中间结果，不当作无根，不自动重试',resources_wait_excluded=True))
    plan['protocol']['resource_gate']=deepcopy(old['protocol']['resource_gate'])
    plan['compare']=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',f'{REPORT_WALL_SECONDS}s',
        sys.executable,'-u','-m','scripts.report_ising_highdim','--compare',
        *(f"{a['name']}={a['root']}/report" for a in plan['arms']),'--output',str(root/'comparison')]
    return plan


def validate_training(plan,arm):
    if arm['training_inherited']:
        raise RuntimeError('Completed training must not be re-executed')
    receipt=low.validate_training(plan,arm,smoke=False)
    saved=checkpoint_receipt(Path(arm['run_path'])/'training_step_12000.pt',arm['config'],step=12000)
    receipt['all_adam_parameter_steps']=saved['all_adam_parameter_steps']
    return receipt


def stage(root,job,name,command,*,gpu,plan,fingerprints,configs,check):
    ensure_source(plan)
    def checked():
        ensure_source(plan);result=check();ensure_source(plan);return result
    try:
        return low.stage(root,job,name,command,gpu=gpu,plan=plan,fingerprints=fingerprints,configs=configs,check=checked)
    except BaseException as error:
        row=job['stages'][-1] if job.get('stages') else {}
        timed_out=(row.get('returncode')==124 or (row.get('returncode')==137
            and row.get('elapsed_seconds',0)>=REPORT_WALL_SECONDS))
        if row.get('name')==name and not gpu and timed_out:
            raise AnalysisBudgetHold(f'{name} exceeded explicit report wall budget; analysis incomplete') from error
        progress=root/name.removeprefix('report_')/'report_progress.json'
        if not gpu and progress.exists() and original.read(progress).get('status')=='analysis_budget_hold':
            raise AnalysisBudgetHold(f'{name} reached its explicit compute budget; analysis incomplete') from error
        raise


def execute(plan):
    root=Path(plan['output'])
    if root.exists() or root.is_symlink():raise FileExistsError('Refuse duplicate continuation or overwrite')
    if socket.gethostname()!=original.HOST or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Require host225 and native physicalGPU2 indexing')
    if os.getpriority(os.PRIO_PROCESS,0)!=10:raise RuntimeError('Require single scheduled nice10')
    if verify_handoff(plan['interrupted_launch']['root'],root)!=plan['interrupted_launch']:
        raise RuntimeError('Handoff changed after planning')
    project=Path(__file__).resolve().parents[1];delta=validate_code_delta(plan,project)
    fingerprints=original.source_hashes(project)
    if not Path('/usr/bin/timeout').is_file():raise RuntimeError('Require independent GNU timeout wall guard')
    root.mkdir(parents=True,exist_ok=False)
    for folder in ('logs','configs','smoke','code','profiles'):(root/folder).mkdir()
    for name,digest in fingerprints.items():
        target=root/'code'/name;target.parent.mkdir(parents=True,exist_ok=True)
        original.shutil.copy2(project/name,target)
        if original.sha(target)!=digest:raise RuntimeError('Frozen code snapshot mismatch')
    original.save(root/'code/fingerprints.json',fingerprints,exclusive=True)
    configs={};old_root=Path(plan['interrupted_launch']['root'])
    for arm in plan['arms']:
        if not arm['inherited']:Path(arm['root']).mkdir()
        if not arm['training_inherited']:original.save(arm['config_path'],arm['config'],exclusive=True)
        configs[arm['config_path']]=original.sha(arm['config_path'])
        target=root/'smoke'/arm['name'];target.mkdir()
        original.shutil.copy2(old_root/'smoke'/arm['name']/'initialization.json',target/'initialization.json')
    plan.update(source_code_sha256=fingerprints,configuration_sha256=configs,source_code_delta=delta)
    original.save(root/'plan.json',plan,exclusive=True)
    original.save(root/'measured_resource_gate.json',plan['measured_resource_gate'],exclusive=True)
    original.save(root/'inherited_evidence.json',plan['interrupted_launch'],exclusive=True)
    job=dict(status='running',workflow_complete=False,started_at=original.now(),parent_pid=os.getpid(),
        host=original.HOST,device='cuda:2',gpu_uuid=original.GPU_UUID,actual_nice=10,current=None,
        stages=[],GPU_stage_seconds=0.,physical_sampling_updates=0,labels_used_in_training=False,
        completed_arms=['dim1'],inherited_completed_arms=['dim1'],inherited_training_arms=['dim1','dim5'],
        error=None,requires_user_action=False,public_files=[],execution_version=EXECUTION_VERSION,
        all_smoke_checks_passed=True,smoke_evidence_inherited=True,
        measured_resource_gate=plan['measured_resource_gate'],interrupted_launch=plan['interrupted_launch'],
        inherited_receipts={'train_dim5':plan['interrupted_launch']['receipt']['inherited_dim5_training_receipt'],
                            'report_dim1':plan['interrupted_launch']['receipt']['inherited_dim1_report_receipt']})
    original.save(root/'job.json',job,exclusive=True)
    started,cpu_start=time.monotonic(),original.cpu_seconds()
    try:
        with v5.continuation_resources():
            for arm in plan['arms']:
                if arm['inherited']:continue
                if not arm['training_inherited']:
                    stage(root,job,'train_'+arm['name'],arm['training'],gpu=True,plan=plan,
                        fingerprints=fingerprints,configs=configs,check=lambda a=arm:validate_training(plan,a))
                stage(root,job,'report_'+arm['name'],arm['report'],gpu=False,plan=plan,
                    fingerprints=fingerprints,configs=configs,
                    check=lambda a=arm:original.report_receipt(Path(a['root'])/'report',a['theory_dim']))
                job['completed_arms'].append(arm['name']);original.save(root/'job.json',job)
            stage(root,job,'comparison',plan['compare'],gpu=False,plan=plan,
                fingerprints=fingerprints,configs=configs,check=lambda:original.report_receipt(root/'comparison'))
        ensure_source(plan)
        public=['job.json','plan.json','measured_resource_gate.json','inherited_evidence.json']
        public += [str(p.relative_to(root)) for p in sorted(root.rglob('*'))
            if p.is_file() and not p.is_symlink() and 'code' not in p.relative_to(root).parts
            and p.suffix in ('.json','.png','.npz') and str(p.relative_to(root)) not in public]
        job.update(status='complete',workflow_complete=True,current=None,ended_at=original.now(),public_files=public)
    except BaseException as error:
        job.update(status='analysis_budget_hold' if isinstance(error,AnalysisBudgetHold) else 'failed',
            workflow_complete=False,requires_user_action=True,error=f'{type(error).__name__}: {error}',ended_at=original.now())
        raise
    finally:
        job.update(elapsed_seconds=time.monotonic()-started,CPU_core_seconds=original.cpu_seconds()-cpu_start)
        original.save(root/'job.json',job)
    return job


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interrupted-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv);plan=build_plan(args.interrupted_root,args.output)
    if args.plan:
        print(canonical(plan));return plan
    return execute(plan)


if __name__=='__main__':main()
