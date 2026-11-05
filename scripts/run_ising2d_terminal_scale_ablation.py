"""Single new L108 -> 36 -> 12 arm against the existing L108 depth-3 endpoint.

Reuse exact physical data/MI cache; only change train/eval RG depth. Fresh
seed91, all 12000 updates, unchanged default independent-gradient recipe.
Wait for an unused selected GPU, fail on errors without retries, retain old runs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint
from recurrent_rg.rsmi import region_offsets
from recurrent_rg.mi_cache import lattice_statistics_cache_key, _cache_identity, load_or_build_mi_statistics
from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from scripts.run_ising2d_exponent_ablation import check_data, check_training, check_reports, stage
from scripts.run_scalar_lr_gpu_pipeline import save_status, utc_now

DATA_SHA='f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
SIDECAR_SHA='0d2d7591b37aa27757bde1d1d964059d38b64ffb6374df1d3d7724993e801e2a'


def differences(a,b,prefix=''):
    result=[]
    for key in a.keys() | b.keys():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(a.get(key),dict) and isinstance(b.get(key),dict):
            result.extend(differences(a[key],b[key],name))
        elif a.get(key)!=b.get(key): result.append(name)
    return sorted(result)


def matched_config(manifest):
    cfg=deepcopy(manifest['config'])
    if (manifest['data_sha256']!=DATA_SHA or cfg['training']['max_rg_steps']!=3
            or cfg['evaluation']['max_rg_steps']!=3 or cfg['training']['epochs']!=12000
            or cfg['training']['seed']!=91 or cfg['model']['theory_dim']!=1):
        raise ValueError('Unexpected historical depth-3 control')
    cfg['training']['max_rg_steps']=2
    cfg['evaluation']['max_rg_steps']=2
    assert differences(manifest['config'],cfg)==['evaluation.max_rg_steps','training.max_rg_steps']
    parsed=RunConfig.from_dict(cfg); parsed.validate()
    args=build_parser().parse_args(['--config','unused.json','--output','unused'])
    validate_recipe(args,parsed)
    return cfg


def cached_mi_receipt(cfg, manifest, data):
    _, offsets=region_offsets(2,block=3,buffer=1)
    records={}
    for split in ('train','test'):
        previous=manifest['execution_optimizations']['mi_statistics_cache'][split]
        shape=data[split+'_shape']
        proxy=SimpleNamespace(dtype=torch.float64,shape=shape)
        identity=lattice_statistics_cache_key(cfg,DATA_SHA,split,proxy,offsets,
            'binary_lattice_mi_statistics',dict(block=3))
        _,key=_cache_identity(identity)
        path=Path(previous['path'])
        if key!=previous['key'] or path.name!=f'mi-v1-{key}.pt' or not path.is_file():
            raise ValueError('Expected existing identical MI cache; rebuilding is not authorized here')
        def forbid_build():
            raise RuntimeError('Missing MI cache: no automatic recomputation')
        statistics,record=load_or_build_mi_statistics(path.parent,identity,forbid_build)
        if not record['hit']: raise ValueError('Expected cache hit')
        records[split]=dict(cache=record,data_shape=shape)
        del statistics
    return dict(status='complete',data_sha256=DATA_SHA,device='cpu',model_initialized=False,
        optimizer_updates=0,sampling_draws=0,cache_version=1,splits=records)


def gpu_snapshot(gpu):
    query=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.free',
                                   '--format=csv,noheader,nounits'],text=True).strip()
    uuid,name,free=[s.strip() for s in query.split(',')]
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    return dict(index=gpu,uuid=uuid,name=name,free_mib=int(free),compute_pids=processes.splitlines() if processes else [])


def wait_gpu(output,status,gpu):
    started=time.monotonic(); ready=0
    while True:
        snapshot=gpu_snapshot(gpu)
        available=not snapshot['compute_pids'] and snapshot['free_mib']>=12288
        ready=ready+1 if available else 0
        status.update(status='waiting_gpu',current=dict(stage='wait_gpu',**snapshot,
            checked_at=utc_now(),elapsed_seconds=time.monotonic()-started,
            condition='No compute processes on selected GPU and >=12 GiB free for two checks 60s apart'))
        save_status(output,status)
        if ready>=2:
            status['gpu_wait_seconds']=time.monotonic()-started
            status['execution_gpu']=snapshot
            status.update(status='running',current=None)
            save_status(output,status)
            return
        time.sleep(60)


def run(source,output,gpu=0):
    if gpu<0: raise ValueError('GPU index must be nonnegative')
    torch.set_num_threads(2)
    if output.exists(): raise FileExistsError(f'Refusing overwrite or implicit resume: {output}')
    manifest=json.loads((source/'manifest.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    cfg=matched_config(manifest)
    if summary.get('actual_epochs')!=12000 or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit'):
        raise ValueError('Control did not complete 12000 updates')
    cache=Path(cfg['data']['cache']); parsed=RunConfig.from_dict(cfg)
    if file_fingerprint(cache.with_suffix('.json'))!=SIDECAR_SHA:
        raise ValueError('Historical data sidecar changed')
    output.mkdir(parents=True,exist_ok=False); (output/'logs').mkdir(); (output/'configs').mkdir()
    config_path=output/'configs/drop_final4.json'
    config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    script_names=('scripts/run_ising2d_terminal_scale_ablation.py','scripts/report_ising_local_response.py')
    source_files={str(source/name):file_fingerprint(source/name) for name in
        ('checkpoint_last.pt','manifest.json','summary.json','full_report/encodings.npz')}
    config_hash=file_fingerprint(config_path)
    status=dict(status='preflight',pid=os.getpid(),started_at=utc_now(),source_run=str(source),
        hostname=subprocess.check_output(['hostname'],text=True).strip(),device=f'cuda:{gpu}',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        added_script_sha256={name:file_fingerprint(name) for name in script_names},
        source_sha256=source_files,config_sha256=config_hash,stages=[],current=dict(name='preflight'),
        protocol=dict(changed_fields=differences(manifest['config'],cfg),lattice_sizes=[108,36,12],
            control_lattice_sizes=[108,36,12,4],seed=91,updates=12000,
            initialization='Fresh random Q/C/R; not checkpoint continuation',data_reused=True,
            note='Same data and seed, but changed depth alters loss averaging, all-horizon pairs and RNG consumption; not identical stochastic trajectories.',
            selection='Final actual12000 only; all bounded roots, no theory-based selection',
            sampling='No new physical samples or MI statistics; formal report uses five independent diagnostic C seeds',
            wall_cap=None,retries=False),training={},reports={})
    save_status(output,status); started=time.monotonic()
    try:
        data=check_data(cache,size=108,max_rg_steps=3,expected_digest=DATA_SHA)
        status['data']=data
        receipt=cached_mi_receipt(parsed,manifest,data)
        receipt.update(config=str(config_path),config_sha256=config_hash)
        (output/'mi_preprocessing.json').write_text(json.dumps(receipt,indent=2)+'\n')
        status.update(status='running',current=None); save_status(output,status)
        stage(output,status,'source_local_response',['nice','-n','10',sys.executable,'-u','-m',
            'scripts.report_ising_local_response','--run',str(source),'--output',str(output/'source_local_response')])
        wait_gpu(output,status,gpu)
        for path,digest in source_files.items():
            if file_fingerprint(path)!=digest: raise ValueError('Historical source changed while waiting')
        if file_fingerprint(cache)!=DATA_SHA or file_fingerprint(cache.with_suffix('.json'))!=SIDECAR_SHA:
            raise ValueError('Physical data changed while waiting')
        if file_fingerprint(config_path)!=config_hash: raise ValueError('Owned config changed while waiting')
        for row in receipt['splits'].values():
            if not Path(row['cache']['path']).is_file(): raise FileNotFoundError('MI cache disappeared while waiting')
        run_path=output/'run'
        mi_dir=Path(receipt['splits']['train']['cache']['path']).parent
        command=[sys.executable,'-u','-m','experiments.rsmi_separate_qcr','--config',str(config_path),
                 '--output',str(run_path),'--device',f'cuda:{gpu}','--mi-cache-dir',str(mi_dir)]
        arm=dict(name='drop_final4',run=str(run_path),config=cfg,config_path=str(config_path),start_updates=0)
        stage(output,status,'training',command)
        status['training']['drop_final4']=check_training(arm,data,require_mi_hits=True); save_status(output,status)
        for name,module,extra in [('phase_report','scripts.report_scalar_joint',['--threads','1','--sampling-repeats','5']),
            ('loss_report','scripts.report_mi_training',['--threads','1']),
            ('exponent_report','scripts.report_ising_scalar_exponents',[])]:
            stage(output,status,name,[sys.executable,'-u','-m',module,'--run',str(run_path),*extra])
        status['reports']['drop_final4']=check_reports(arm)
        stage(output,status,'new_local_response',['nice','-n','10',sys.executable,'-u','-m',
            'scripts.report_ising_local_response','--run',str(run_path),'--output',str(output/'new_local_response')])
        for path,digest in source_files.items():
            if file_fingerprint(path)!=digest: raise ValueError('Historical source changed')
        status.update(status='complete',workflow_complete=True,current=None,ended_at=utc_now(),
                      elapsed_seconds=time.monotonic()-started)
        save_status(output,status)
    except Exception as error:
        status.update(status='failed',requires_user_action=True,error=f'{type(error).__name__}: {error}',
                      ended_at=utc_now(),elapsed_seconds=time.monotonic()-started)
        save_status(output,status)
        raise
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    if args.plan:
        original=json.loads((args.source_run/'manifest.json').read_text())
        print(json.dumps(dict(config=matched_config(original),changes=['evaluation.max_rg_steps','training.max_rg_steps'],
            gpu_policy=f'Wait until GPU{args.gpu} unused and at least12 GiB free; never stop other jobs'),indent=2))
    else:
        run(args.source_run.resolve(),args.output.resolve(),args.gpu)
