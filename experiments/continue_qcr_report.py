"""Continue a locked QCR run, then generate the scalar phase/loss reports."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,required=True,help='Cumulative target updates, not additional updates')
    parser.add_argument('--log-prefix',type=Path,required=True)
    args=parser.parse_args()
    cfg=RunConfig.load(args.config)
    cfg=replace(cfg,training=replace(cfg.training,epochs=args.epochs))
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    old=RunConfig.from_dict(payload['metadata']['config'])
    old=replace(old,training=replace(old.training,epochs=args.epochs))
    if cfg!=old or file_fingerprint(cfg.data.cache)!=payload['metadata']['data_sha256']:
        raise ValueError('Continuation configuration or data differs from checkpoint')
    if args.epochs<=payload['epoch'] or cfg.model.theory_dim!=1:
        raise ValueError('Requires a later cumulative endpoint and a scalar report')
    if args.output.exists():raise FileExistsError(args.output)
    prefix=args.log_prefix
    prefix.parent.mkdir(parents=True,exist_ok=True)
    status_file=prefix.with_name(prefix.name+'_pipeline.json')
    if status_file.exists():raise FileExistsError(status_file)
    status=dict(pipeline_pid=os.getpid(),started_at=time.time(),phase='starting',
                source_checkpoint=str(args.checkpoint.resolve()),source_epoch=int(payload['epoch']),
                target_epoch=args.epochs,output=str(args.output.resolve()),
                data_sha256=payload['metadata']['data_sha256'],config_identical=True,
                training='Locked original QCR objective; restore optimizer and all RNG states; no wall limit')
    commands=[('training',[sys.executable,'-m','experiments.rsmi_separate_qcr',
                          '--config',str(args.config),'--output',str(args.output),
                          '--resume-checkpoint',str(args.checkpoint),'--epochs',str(args.epochs)]),
              ('scalar_report',[sys.executable,'-m','scripts.report_scalar_joint','--run',str(args.output)]),
              ('phase_report',[sys.executable,'-m','scripts.plot_dual_ising_scalar','--run',str(args.output)]),
              ('loss_report',[sys.executable,'-m','scripts.report_mi_training','--run',str(args.output)])]
    try:
        for name,command in commands:
            log=prefix.with_name(prefix.name+'_'+name+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                status.update(phase=name,child_pid=child.pid,stage_log=str(log.resolve()),stage_started_at=time.time())
                status_file.write_text(json.dumps(status,indent=2)+'\n')
                print(f'{name}: pid={child.pid}, log={log}',flush=True)
                returncode=child.wait()
            if returncode:
                status.update(phase='failed',failed_stage=name,returncode=returncode,ended_at=time.time())
                status_file.write_text(json.dumps(status,indent=2)+'\n')
                return returncode
        summary=json.loads((args.output/'summary.json').read_text())
        if summary['actual_epochs']!=args.epochs:
            raise RuntimeError('Training did not reach requested cumulative endpoint')
        status.update(phase='completed',actual_epochs=summary['actual_epochs'],ended_at=time.time())
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        print('Completed training and reports:',args.output,flush=True)
        return 0
    except Exception as error:
        status.update(phase='failed',error=repr(error),ended_at=time.time())
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        raise


if __name__=='__main__':sys.exit(main())
