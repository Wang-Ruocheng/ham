"""显式 K 标签监督短消融；只启动一次，失败保留证据，不自动重试。"""
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

DATA_SHA = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
WEIGHT = 10.
UPDATES = 3000
TRAINING_SECONDS = 900
MI_KEYS = dict(train='53b4cb45bda6c9271a4d18348e9f20cf75bbcf9f63647e76e5ca401dc19f5bea',
               test='85992c4ab2ce55ab36a0c45842d3a6e9c695bdd418c4cc3688117d6a60ebd8b6')


def build_plan(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('New output must be separate from historical source')
    manifest = json.loads((source/'manifest.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    cfg = deepcopy(manifest['config'])
    if (manifest['data_sha256'] != DATA_SHA or manifest['labels_used_in_training']
            or summary['actual_epochs'] != 12000 or cfg['model']['block_factor'] != 2
            or cfg['model']['theory_dim'] != 1 or cfg['model']['dimension'] != 2
            or cfg['training']['max_rg_steps'] != 2 or cfg['training']['seed'] != 91
            or cfg['training']['lr_milestones'] != [6000, 9000]
            or cfg['data']['coupling_names'] != ['K']):
        raise ValueError('Unexpected completed latest b2 control')
    if not (source/'training_step_3000.pt').is_file():
        raise ValueError('The matched 3000-update control must already exist')
    mi = source.parent/'mi_cache'
    receipt = json.loads((source.parent/'mi_preprocessing.json').read_text())
    if not mi.is_dir() or receipt.get('status') != 'complete':
        raise ValueError('Completed existing b2 MI preprocessing required')
    for split, key in MI_KEYS.items():
        cache = receipt.get('splits', {}).get(split, {}).get('cache', {})
        if cache.get('key') != key or not (mi/f'mi-v1-{key}.pt').is_file():
            raise ValueError('Existing b2 MI cache missing or mismatched; never rebuild in this pilot')
    cfg['training']['epochs'] = UPDATES
    config_path, run = output/'config.json', output/'run'
    command = [sys.executable, '-u', '-m', 'scripts.report_ising_zh', '--module',
        'experiments.rsmi_separate_qcr', '--config', str(config_path), '--output', str(run),
        '--device', 'cuda:0', '--mi-cache-dir', str(mi), '--q-coupling-weight', str(WEIGHT),
        '--max-training-seconds', str(TRAINING_SECONDS), '--skip-final-report']
    report = [sys.executable, '-u', '-m', 'scripts.report_ising_zh', '--module',
        'scripts.report_k_supervised_pilot', '--run', str(run), '--control-run', str(source),
        '--output', str(output/'reports'), '--device', 'cuda:0']
    return dict(source=str(source), output=str(output), config=cfg, command=command,
        report_command=report, protocol=dict(q_coupling_weight=WEIGHT,
            target='TRAIN microscopic Q(D) = raw K only; coarse g has no K target',
            independent_gradients=True, fresh_seed=91, target_updates=UPDATES,
            training_limit_seconds=TRAINING_SECONDS, theory_labels=False,
            learning_rate='Original milestones6000/9000 retained; no decay within3000',
            report='All768 ensembles x128; one independent CUDA C draw seed12345; no MI computation or five-draw audit',
            comparison='Existing unsupervised matched update checkpoint, not just the12000 endpoint',
            scope='Supervised coordinate ablation, not a claim of unbiased exponent estimation'))


def write_job(output, job):
    job['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    pending = output/'job.json.tmp'
    pending.write_text(json.dumps(job, indent=2)+'\n')
    pending.replace(output/'job.json')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--detach', action='store_true', help='Reserve a unique launch receipt and start one detached parent')
    args = p.parse_args(argv)
    plan = build_plan(args.source_run, args.output)
    if args.plan:
        print(json.dumps(plan, indent=2)); return
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('CUDA_VISIBLE_DEVICES must not remap the explicitly requested physical GPU0')
    output = Path(plan['output'])
    if args.detach:
        if output.exists():
            raise FileExistsError('Refusing to overwrite an existing experiment')
        logs = Path.cwd()/'logs'
        logs.mkdir(exist_ok=True)
        receipt = logs/(output.name+'.launch.json')
        log_path = logs/(output.name+'.log')
        # Reserve before spawn: an ambiguous SSH outcome must never start a second parent.
        with receipt.open('x') as handle:
            json.dump(dict(status='reserved', output=str(output)), handle)
        command = [sys.executable, '-u', '-m', 'scripts.run_ising2d_k_supervised_pilot',
                   '--source-run', plan['source'], '--output', str(output)]
        with log_path.open('x') as log, open(os.devnull) as stdin:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=stdin, start_new_session=True)
        code = ('experiments/rsmi_separate_qcr.py', 'scripts/report_ising_zh.py',
                'scripts/report_k_supervised_pilot.py', 'scripts/run_ising2d_k_supervised_pilot.py')
        record = dict(status='launched', parent_pid=child.pid, command=command,
                      output=str(output), log=str(log_path),
                      started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in code})
        receipt.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'config.json').write_text(json.dumps(plan['config'], indent=2)+'\n')
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(),
               started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               stages=[], current=None, protocol=plan['protocol'])
    write_job(output, job)
    try:
        # Only inspect the GPU; never stop another process or alter its state.
        for check in range(2):
            rows = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if rows:
                raise RuntimeError('GPU0 no longer idle; no training launched')
            if not check: time.sleep(10)
        for name, command in [('training', plan['command']), ('report', plan['report_command'])]:
            started = time.monotonic()
            with (output/'logs'/f'{name}.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                job['current'] = dict(name=name, pid=child.pid, command=command)
                write_job(output, job)
                rc = child.wait()
            job['stages'].append(dict(name=name, returncode=rc, seconds=time.monotonic()-started))
            if rc:
                raise RuntimeError(f'{name} failed with returncode {rc}; no automatic retry')
            write_job(output, job)
        summary = json.loads((output/'run'/'summary.json').read_text())
        manifest = json.loads((output/'run'/'manifest.json').read_text())
        if (not manifest['labels_used_in_training'] or manifest['source'] is not None
                or manifest['q_supervision']['weight'] != WEIGHT
                or summary['actual_epochs'] > UPDATES
                or not all(v['hit'] for v in manifest['execution_optimizations']['mi_statistics_cache'].values())):
            raise ValueError('Supervision, fresh initialization, update budget or MI cache receipt mismatch')
        job.update(status='complete', workflow_complete=True, current=None,
                   actual_epochs=summary['actual_epochs'])
    except Exception as error:
        job.update(status='failed', error=f'{type(error).__name__}: {error}', requires_user_action=True)
    write_job(output, job)
    if job['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
