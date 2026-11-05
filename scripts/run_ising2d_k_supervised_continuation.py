"""从已完成 K 监督短实验精确续训；固定累计节点报告，不按指数选终点。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_ising2d_k_supervised_pilot import DATA_SHA, MI_KEYS, write_job

ENDPOINTS = (4000, 6000, 9000, 12000)
SOURCE_EPOCH = 2870
WEIGHT = 10.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(source, control, output):
    """只读取已有 JSON；不载入数据/模型、不创建目录或计算 MI。"""
    source, control, output = (Path(p).resolve() for p in (source, control, output))
    if any(output == p or p in output.parents or output in p.parents for p in (source, control)):
        raise ValueError('New output must be separate from historical experiments')
    manifest = json.loads((source/'manifest.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    cfg = manifest['config']
    control_manifest = json.loads((control/'manifest.json').read_text())
    control_summary = json.loads((control/'summary.json').read_text())
    expected_cfg = deepcopy(control_manifest['config'])
    expected_cfg['training']['epochs'] = 3000
    if (control_summary.get('actual_epochs') != 12000
            or control_manifest.get('data_sha256') != DATA_SHA
            or control_manifest.get('labels_used_in_training') is not False
            or cfg != expected_cfg):
        raise ValueError('Pilot must retain the entire completed latest b2 control configuration')
    sup = manifest.get('q_supervision', {})
    if (manifest.get('data_sha256') != DATA_SHA or not manifest.get('labels_used_in_training')
            or sup.get('weight') != WEIGHT or sup.get('scale') != 0
            or sup.get('fitting_split') != 'TRAIN only' or sup.get('coupling_names') != ['K']
            or summary.get('actual_epochs') != SOURCE_EPOCH
            or summary.get('config') != cfg or cfg.get('schema_version') != 5
            or cfg['model']['block_factor'] != 2 or cfg['model']['dimension'] != 2
            or cfg['model']['theory_dim'] != 1 or cfg['training']['max_rg_steps'] != 2
            or cfg['evaluation']['max_rg_steps'] != 2
            or cfg['training']['seed'] != 91 or cfg['training']['epochs'] != 3000
            or cfg['training']['lr_milestones'] != [6000, 9000]
            or cfg['data']['coupling_names'] != ['K']
            or manifest.get('execution_device') != 'cuda:0'
            or manifest.get('precision') != 'torch.float64' or not manifest.get('independent_c')):
        raise ValueError('Unexpected source: require the completed 2870-step supervised GPU0 pilot')
    source_job = json.loads((source.parent/'job.json').read_text())
    if source_job.get('status') != 'complete' or not source_job.get('workflow_complete'):
        raise ValueError('Historical pilot workflow must already be complete')
    initial = source/'checkpoint_last.pt'
    if not initial.is_file():
        raise ValueError('Saved source checkpoint missing; never restart from scratch')
    mi = control.parent/'mi_cache'
    receipt = json.loads((control.parent/'mi_preprocessing.json').read_text())
    if receipt.get('status') != 'complete' or receipt.get('data_sha256') != DATA_SHA:
        raise ValueError('Existing b2 MI receipt mismatch')
    for split, key in MI_KEYS.items():
        if (receipt.get('splits', {}).get(split, {}).get('cache', {}).get('key') != key
                or not (mi/f'mi-v1-{key}.pt').is_file()):
            raise ValueError('Existing b2 MI cache missing; no preprocessing in this continuation')
    reference = source.parent/'reports'/'supervised_final'
    if not (reference/'critical_exponents.json').is_file():
        raise ValueError('The completed 2870-step reference report must be retained')
    stages, checkpoint = [], initial
    for endpoint in ENDPOINTS:
        run = output/'segments'/f'epoch_{endpoint}'
        config_path = output/'configs'/f'epoch_{endpoint}.json'
        segment_cfg = deepcopy(cfg)
        segment_cfg['training']['epochs'] = endpoint
        training = [sys.executable, '-u', '-m', 'scripts.report_ising_zh', '--module',
            'experiments.rsmi_separate_qcr', '--config', str(config_path), '--output', str(run),
            '--resume-checkpoint', str(checkpoint), '--q-coupling-weight', str(WEIGHT),
            '--device', 'cuda:0', '--mi-cache-dir', str(mi), '--skip-final-report']
        report = [sys.executable, '-u', '-m', 'scripts.report_ising_zh', '--module',
            'scripts.report_k_supervised_continuation', '--run', str(run), '--checkpoint', str(endpoint),
            '--output', str(output/'reports'/f'epoch_{endpoint}'), '--device', 'cuda:0',
            '--trend-root', str(output/'reports'), '--reference-report', str(reference)]
        stages.append(dict(endpoint=endpoint, run=str(run), resume_checkpoint=str(checkpoint),
                           config_path=str(config_path), config=segment_cfg,
                           training_command=training, report_command=report))
        checkpoint = run/f'training_step_{endpoint}.pt'
    return dict(source_run=str(source), control_run=str(control), output=str(output),
        initial_checkpoint=str(initial), stages=stages, reference_report=str(reference),
        mi_cache_dir=str(mi), protocol=dict(start_updates=SOURCE_EPOCH, final_updates=12000,
            additional_updates=12000-SOURCE_EPOCH, endpoints=list(ENDPOINTS),
            q_coupling_weight=WEIGHT, labels='TRAIN raw K for microscopic Q only; no coarse-K or exponent target',
            initialization='Exact same-device Q/C/R, Adam, batch/MI/CPU/CUDA RNG continuation; no reinitialization',
            learning_rate='Original cumulative milestones6000/9000 retained; 0.001→0.0003→0.00009',
            wall_cap=None, automatic_retry=False,
            reports='GPU0 reports between exact-resume segments; one C diagnostic draw seed12345 per saved model; Chinese figures',
            selection='Fixed endpoints; no root or epoch selection by closeness to nu1; F labels are not matched across checkpoints',
            limitations='12000 is a comparison endpoint, not a convergence guarantee; finite lattice, one training seed and B0 only'))


def ensure_source(plan, source_sha):
    if sha(plan['initial_checkpoint']) != source_sha:
        raise ValueError('Historical source checkpoint changed')


def validate_source_payload(plan):
    import torch
    path = Path(plan['initial_checkpoint'])
    payload = torch.load(path, map_location='cpu', weights_only=True)
    manifest = json.loads((Path(plan['source_run'])/'manifest.json').read_text())
    if (payload.get('epoch') != SOURCE_EPOCH
            or json.dumps(payload.get('metadata'), sort_keys=True, allow_nan=False) !=
               json.dumps(manifest, sort_keys=True, allow_nan=False)
            or payload.get('lr_scheduler_state', {}).get('completed_updates') != SOURCE_EPOCH
            or payload.get('cuda_device_index') != 0
            or any(name not in payload for name in ('optimizer', 'torch_rng_state', 'cuda_rng_state',
                                                     'batch_generator_state', 'rsmi_generator_state'))):
        raise ValueError('Source checkpoint is not the documented complete 2870-step Adam/RNG state')
    summary = json.loads((Path(plan['source_run'])/'summary.json').read_text())
    if payload.get('history') != summary['history']:
        raise ValueError('Source checkpoint history differs from its completed summary')
    return sha(path)


def ensure_code(record):
    for name, expected in record.items():
        if sha(name) != expected:
            raise ValueError(f'Deployed source changed during continuation: {name}')


def wait_gpu(output, job, next_stage):
    consecutive = 0
    while consecutive < 2:
        rows = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        consecutive = consecutive+1 if not rows else 0
        job['current'] = dict(name='wait_gpu', next_stage=next_stage, busy_pids=rows.splitlines(),
                              consecutive_idle=consecutive)
        write_job(output, job)
        if consecutive < 2:
            time.sleep(10 if not rows else 30)


def stage(output, job, name, command, run, endpoint):
    wait_gpu(output, job, name)
    started = time.monotonic()
    record = dict(name=name, run=run, endpoint=endpoint, command=command,
                  log=str(output/'logs'/f'{name}.log'), status='running')
    with Path(record['log']).open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        job['current'] = record
        write_job(output, job)
        rc = child.wait()
    record.update(returncode=rc, seconds=time.monotonic()-started,
                  status='complete' if rc == 0 else 'failed')
    job['stages'].append(record)
    job['current'] = None
    write_job(output, job)
    if rc:
        raise RuntimeError(f'{name} failed with returncode{rc}; no automatic retry')


def check_training(row, previous):
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy
    run = Path(row['run'])
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if (summary.get('actual_epochs') != row['endpoint'] or summary.get('stopped_by_training_limit')
            or summary.get('stopped_by_wall_limit') or summary['config'] != row['config']
            or manifest.get('config') != row['config']
            or manifest.get('data_sha256') != DATA_SHA
            or manifest.get('resumed_from') != row['resume_checkpoint']
            or not manifest.get('labels_used_in_training') or manifest['q_supervision']['weight'] != WEIGHT):
        raise ValueError('Continuation endpoint, recipe, data or existing-cache receipt mismatch')
    caches = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    if set(caches) != set(MI_KEYS) or any(not caches[s].get('hit') or
            caches[s].get('key') != key for s, key in MI_KEYS.items()):
        raise ValueError('Both exact existing b2 MI cache keys must be hits')
    inherited = json.loads((Path(previous)/'summary.json').read_text())['history']
    if summary['history'][:len(inherited)] != inherited:
        raise ValueError('Inherited training history changed')
    policy = LearningRatePolicy(RunConfig.from_dict(row['config']).training)
    if summary.get('lr_scheduler_state') != policy.checkpoint_state(row['endpoint']):
        raise ValueError('Cumulative learning-rate scheduler receipt differs')
    if any(r['learning_rate'] != policy.rate_for_update(r['epoch']) for r in summary['history']):
        raise ValueError('A saved history row has a changed learning rate')
    if not (run/f'training_step_{row["endpoint"]}.pt').is_file():
        raise ValueError('Expected immutable numbered endpoint checkpoint missing')
    return dict(actual_epochs=summary['actual_epochs'], run=str(run),
                checkpoint=str(run/f'training_step_{row["endpoint"]}.pt'),
                learning_rate=summary['history'][-1]['learning_rate'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--control-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args(argv)
    plan = build_plan(args.source_run, args.control_run, args.output)
    if args.plan:
        print(json.dumps(plan, indent=2)); return plan
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Do not remap physical GPU0 with CUDA_VISIBLE_DEVICES')
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing continuation')
    if args.detach:
        logs = Path.cwd()/'logs'; logs.mkdir(exist_ok=True)
        receipt, log_path = logs/(output.name+'.launch.json'), logs/(output.name+'.log')
        with receipt.open('x') as handle:
            json.dump(dict(status='reserved', output=str(output)), handle)
        command = [sys.executable, '-u', '-m', 'scripts.run_ising2d_k_supervised_continuation',
            '--source-run', plan['source_run'], '--control-run', plan['control_run'], '--output', str(output)]
        with log_path.open('x') as log, open(os.devnull) as stdin:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=stdin, start_new_session=True)
        record = dict(status='launched', parent_pid=child.pid, output=str(output), log=str(log_path),
                      command=command, code_sha256={name: sha(name) for name in (
                          'experiments/rsmi_separate_qcr.py', 'scripts/report_ising_zh.py',
                          'scripts/report_k_supervised_pilot.py', 'scripts/report_k_supervised_continuation.py',
                          'scripts/run_ising2d_k_supervised_continuation.py')})
        receipt.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2)); return record
    source_sha = validate_source_payload(plan)
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir(); (output/'configs').mkdir(); (output/'segments').mkdir(); (output/'reports').mkdir()
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for row in plan['stages']:
        Path(row['config_path']).write_text(json.dumps(row['config'], indent=2)+'\n')
    code_sha = {name: sha(name) for name in ('experiments/rsmi_separate_qcr.py', 'scripts/report_ising_zh.py',
                'scripts/report_k_supervised_pilot.py', 'scripts/report_k_supervised_continuation.py',
                'scripts/run_ising2d_k_supervised_continuation.py')}
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(), stages=[], current=None,
               started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               source_checkpoint=plan['initial_checkpoint'], source_checkpoint_sha256=source_sha,
               code_sha256=code_sha,
               protocol=plan['protocol'], training=dict(actual_epochs=SOURCE_EPOCH), reports=[])
    write_job(output, job)
    previous = plan['source_run']
    try:
        for row in plan['stages']:
            ensure_source(plan, source_sha)
            for name, command in (('training', row['training_command']), ('report', row['report_command'])):
                ensure_code(code_sha)
                stage(output, job, f'{name}_{row["endpoint"]}', command, row['run'], row['endpoint'])
                if name == 'training':
                    job['training'] = check_training(row, previous)
                else:
                    report = output/'reports'/f'epoch_{row["endpoint"]}'/'checkpoint_summary.json'
                    if not report.is_file():
                        raise ValueError('Completed report did not save its checkpoint summary')
                    job['reports'].append(dict(epoch=row['endpoint'], directory=str(report.parent),
                        summary_sha256=sha(report)))
                write_job(output, job)
            previous = row['run']
        ensure_source(plan, source_sha)
        job.update(status='complete', workflow_complete=True, current=None,
                   ended_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    except Exception as error:
        job.update(status='failed', error=f'{type(error).__name__}: {error}', requires_user_action=True)
    write_job(output, job)
    if job['status'] != 'complete':
        raise SystemExit(1)
    return job


if __name__ == '__main__':
    main()
