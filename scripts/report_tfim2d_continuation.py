"""Compare scalar R maps across the unchanged 2D TFIM continuation."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def main():
    torch.set_num_threads(1)
    run=Path('results/joint_qcr_tfim2d_g1_L16_M640_continue6000_seed91')
    summary=json.loads((run/'summary.json').read_text())
    reports=json.loads((run/'scalar_report.json').read_text())
    cfg=summary['config'];fig,axs=plt.subplots(1,3,figsize=(15,4))
    lo=min(r['observed_support'][0] for r in reports.values())-.2
    hi=max(r['observed_support'][1] for r in reports.values())+.2
    xx=torch.linspace(lo,hi,1000,dtype=torch.float64)[:,None]
    result={}
    for step,r in reports.items():
        m=RecurrentRGNetwork2D(levels=cfg['levels'],hidden=cfg['hidden'],theory_dim=1).double().eval()
        state=torch.load(run/f'training_step_{step}.pt',weights_only=True)
        m.load_state_dict(state['model'],strict=False)
        with torch.no_grad():yy=m.advance_theory(xx).flatten().numpy()
        axs[0].plot(xx.numpy().ravel(),yy,label=step)
        axs[1].plot(xx.numpy().ravel(),yy-xx.numpy().ravel(),label=step)
        axs[2].plot(r['coupling'],np.array(r['g_mean'])[:,0],label=step)
        log=next(v for v in summary['history'] if v['epoch']==int(step))
        result[step]={'train_total':log['train_total'],'test_total':log['test_total'],
                      'test_nrmse':r['metrics']['test']['nrmse'],
                      'C_truth':r['C_truth'],'roots':r['roots'],
                      'g_at_reference_coupling_by_scale':r['g_at_Kc']}
    axs[0].plot([lo,hi],[lo,hi],'k--',label='identity')
    axs[1].axhline(0,color='black',ls='--')
    axs[2].axvline(cfg['reference_critical_coupling'],color='black',ls=':',label='reference')
    axs[0].set(xlabel='g',ylabel='R(g)');axs[1].set(xlabel='g',ylabel='R(g)-g')
    axs[2].set(xlabel='h/J',ylabel='Q(D)')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('2D TFIM, dim g=1: joint continuation; latent coordinates can move during training')
    fig.tight_layout();fig.savefig(run/'R_continuation_comparison.png',dpi=180);plt.close(fig)
    (run/'continuation_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
