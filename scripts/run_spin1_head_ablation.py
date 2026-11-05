"""Paired short Q-head ablation with locked default geometry and C/R recipe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from recurrent_rg.config import RunConfig

def main():
    parent=Path('results/spin1_head_ablation_150_seed91').resolve()
    parent.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    start=time.monotonic();status={}
    try:
        for kind in ['nonlinear','linear']:
            root=parent/kind
            cfg=RunConfig.load('configs/spin1_signedK_g1_block3_locked_recipe_h96_seed91.json').to_dict()
            cfg['model'].update(theory_head=kind,reconstruction_hidden=0)
            cfg['loss'].update(geometry_bounds='interval',geometry_distance='euclidean',geometry_tail_mix=0.,geometry_tail_add_weight=0.)
            cfg['training'].update(epochs=150,checkpoint_every=50,log_every=30)
            config=Path(f'configs/spin1_head_{kind}_150_seed91.json').resolve()
            config.write_text(json.dumps(cfg,indent=2)+'\n')
            cmds=[
                [sys.executable,'-u','-m','experiments.rsmi_separate_qcr','--config',str(config),'--output',str(root)],
                [sys.executable,'scripts/report_scalar_joint.py','--run',str(root)],
                [sys.executable,'scripts/report_mi_training.py','--run',str(root)],
                [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(root),'--signed-spin1']]
            for stage,cmd in enumerate(cmds):
                with Path(f'logs/spin1_head_{kind}_150_stage{stage}.log').open('w') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                                   timeout=max(1,560-(time.monotonic()-start)))
            status[kind]='complete'
            print(f'{kind} complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
        status['status']='complete'
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        status.update(status='stopped',error=str(exc))
    status['elapsed_seconds']=time.monotonic()-start
    (parent/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
