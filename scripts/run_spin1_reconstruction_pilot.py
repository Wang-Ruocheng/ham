"""Bounded unsupervised Q distribution-reconstruction ablation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from recurrent_rg.config import RunConfig

def main():
    root=Path('results/spin1_signedK_g1_reconstructJS_240_seed91').resolve()
    if root.exists():raise FileExistsError(root)
    config=Path('configs/spin1_signedK_g1_reconstructJS_240_seed91.json').resolve()
    cfg=RunConfig.load('configs/spin1_signedK_g1_block3_locked_recipe_h96_seed91.json').to_dict()
    cfg['model']['reconstruction_hidden']=32
    cfg['training'].update(epochs=240,checkpoint_every=60,log_every=30)
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    cmds=[
        [sys.executable,'-u','-m','experiments.rsmi_separate_qcr','--config',str(config),'--output',str(root)],
        [sys.executable,'scripts/report_scalar_joint.py','--run',str(root)],
        [sys.executable,'scripts/report_mi_training.py','--run',str(root)],
        [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(root),'--signed-spin1']]
    start=time.monotonic()
    try:
        for stage,cmd in enumerate(cmds):
            with Path(f'logs/spin1_reconstructJS240_stage{stage}.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,560-(time.monotonic()-start)))
        status=dict(status='complete',elapsed_seconds=time.monotonic()-start)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        status=dict(status='stopped',error=str(exc),elapsed_seconds=time.monotonic()-start)
    root.mkdir(parents=True,exist_ok=True)
    (root/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
