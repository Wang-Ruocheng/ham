"""Fresh default scalar Ising QCR training, TRAIN plateau check and reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--log-prefix',type=Path,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=2400.)
    a=p.parse_args()
    cfg=RunConfig.load(a.config)
    if (cfg.model.dimension!=2 or cfg.model.theory_dim!=1 or cfg.model.categorical_states or
            tuple(cfg.data.coupling_names)!=('K',) or not cfg.model.affine_reconstruction):
        raise ValueError('Requires scalar zero-field binary 2D Ising with affine reconstruction')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.log_prefix.parent.mkdir(parents=True,exist_ok=True)
    status_path=a.log_prefix.with_name(a.log_prefix.name+'_pipeline.json')
    if status_path.exists():
        raise FileExistsError(status_path)
    status=dict(phase='starting',pipeline_pid=os.getpid(),started_at=time.time(),
        config=str(a.config.resolve()),output=str(a.output.resolve()),
        data_sha256=file_fingerprint(cfg.data.cache),update_cap=cfg.training.epochs,
        training_wall_cap_seconds=a.max_wall_seconds,
        convention='Fresh Q/C/R; independent simultaneous gradients; unchanged default objectives; TRAIN-only stopping',
        block_factor=cfg.model.block_factor,theory_dim=cfg.model.theory_dim)
    commands=[
        ('training',[sys.executable,'-m','experiments.rsmi_separate_qcr','--config',str(a.config),
            '--output',str(a.output),'--joint-train-plateau','--max-wall-seconds',str(a.max_wall_seconds)]),
        ('scalar_report',[sys.executable,'-m','scripts.report_scalar_joint','--run',str(a.output),'--threads','1']),
        ('phase_report',[sys.executable,'-m','scripts.plot_dual_ising_scalar','--run',str(a.output),'--positive-ising']),
        ('loss_report',[sys.executable,'-m','scripts.report_mi_training','--run',str(a.output),'--threads','1']),
        ('exponent_report',[sys.executable,'-m','scripts.report_ising_scalar_exponents','--run',str(a.output)]),
    ]
    try:
        for name,command in commands:
            log=a.log_prefix.with_name(a.log_prefix.name+'_'+name+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                status.update(phase=name,child_pid=child.pid,stage_log=str(log.resolve()),stage_started_at=time.time())
                status_path.write_text(json.dumps(status,indent=2)+'\n')
                print(f'{name}: pid={child.pid}, log={log}',flush=True)
                returncode=child.wait()
            if returncode:
                status.update(phase='failed',failed_stage=name,returncode=returncode,ended_at=time.time())
                status_path.write_text(json.dumps(status,indent=2)+'\n')
                return returncode
            if name=='training':
                summary=json.loads((a.output/'summary.json').read_text())
                status.update(actual_epochs=summary['actual_epochs'],stop_reason=summary['stop_reason'],
                              train_plateau_reached=summary['stopped_by_train_plateau'])
        status.update(phase='completed',ended_at=time.time())
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        return 0
    except Exception as error:
        status.update(phase='failed',error=repr(error),ended_at=time.time())
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        raise


if __name__=='__main__':
    sys.exit(main())
