"""Short, explicitly label-supervised microscopic Q diagnostic experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from recurrent_rg.config import RunConfig

def main():
    root=Path('results/spin1_signedK_g1_supervisedQ20_300_seed91').resolve()
    config=Path('configs/spin1_signedK_g1_supervisedQ20_300_seed91.json').resolve()
    cfg=RunConfig.load('configs/spin1_signedK_g1_block3_locked_recipe_h96_seed91.json').to_dict()
    cfg['training']['epochs']=300;cfg['training']['checkpoint_every']=100;cfg['training']['log_every']=50
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    cmds=[
        [sys.executable,'-u','-m','experiments.rsmi_separate_qcr','--config',str(config),'--output',str(root),'--q-coupling-weight','20'],
        [sys.executable,'scripts/report_scalar_joint.py','--run',str(root)],
        [sys.executable,'scripts/report_mi_training.py','--run',str(root)],
        [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(root),'--signed-spin1']]
    start=time.time()
    for stage,cmd in enumerate(cmds):
        with Path(f'logs/spin1_supervisedQ20_stage{stage}.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=np.load(root/'full_report/encodings.npz');h=json.loads((root/'history_progress.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');metrics={}
    for split in ['train','test']:
        K=z[split+'_labels'].ravel();g=z[split][:,0,0]
        metrics[split]=dict(K_rmse=float(np.sqrt(np.mean((g-K)**2))),K_mae=float(np.mean(abs(g-K))),correlation=float(np.corrcoef(g,K)[0,1]))
        axes[0].scatter(K,g,s=15,label=split)
        axes[1].plot([r['epoch'] for r in h],[r[split+'_coupling_supervision'] for r in h],label=split)
    axes[0].plot([-.9,.9],[-.9,.9],'k:');axes[0].set(xlabel='K',ylabel='Q(D)',title='Microscopic label supervision')
    axes[1].set(xlabel='epoch',ylabel='unweighted K MSE',yscale='log',title='Supervision loss')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'full_report/supervision.png',dpi=160)
    metrics.update(elapsed_seconds=time.time()-start,supervision='20*MSE(Q(D),K), microscopic only; no coarse-scale labels')
    (root/'supervision_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
