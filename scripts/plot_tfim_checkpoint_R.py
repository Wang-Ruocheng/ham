"""Plot the expanded-depth 1000-update R directly from its checkpoint."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(2)
    run=Path(__file__).resolve().parents[1]/'results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91'
    checkpoint=torch.load(run/'training_step_1000.pt',weights_only=True)
    model=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=1).double().eval()
    inc=model.load_state_dict(checkpoint['model'],strict=False)
    assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
    def residual(v):
        with torch.no_grad():
            return float(model.advance_theory(torch.tensor([[v]],dtype=torch.float64))[0,0])-v
    grid=np.linspace(-1.3,1.6,1501)
    with torch.no_grad():
        pred=model.advance_theory(torch.tensor(grid[:,None])).numpy()[:,0]
    roots=[]
    for a,b,fa,fb in zip(grid[:-1],grid[1:],(pred-grid)[:-1],(pred-grid)[1:]):
        if fa*fb<0:
            root=brentq(residual,a,b)
            x=torch.tensor([[root]],dtype=torch.float64,requires_grad=True)
            slope=float(torch.autograd.grad(model.advance_theory(x).sum(),x)[0])
            roots.append(dict(g=root,derivative=slope,stability='stable' if abs(slope)<1 else 'unstable'))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(grid,pred,lw=2,label='R(g)')
    axes[0].plot(grid,grid,'k--',lw=1,label='identity')
    axes[1].plot(grid,pred-grid,lw=2)
    axes[1].axhline(0,color='black',ls='--',lw=1)
    for i,r in enumerate(roots):
        color='tab:green' if r['stability']=='stable' else 'tab:red'
        label=f"F{i+1}: {r['g']:.4f}, {r['stability']}"
        axes[0].scatter(r['g'],r['g'],color=color,s=55,zorder=4,label=label)
        axes[1].scatter(r['g'],0,color=color,s=55,zorder=4)
        axes[1].annotate(f"F{i+1}\nR'={r['derivative']:.3f}",(r['g'],0),xytext=(0,15 if i%2==0 else -35),textcoords='offset points',ha='center',color=color)
    axes[0].set(xlabel='g',ylabel='R(g)',title='Learned RG map')
    axes[1].set(xlabel='g',ylabel='R(g) - g',title='Fixed points: zeros of the residual')
    axes[0].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.2); ax.set_xlim(grid[0],grid[-1])
    fig.suptitle('TFIM dim(g)=1, supervised k=0..5, seed 91 — 1000 updates')
    fig.text(.5,.015,'Fixed points of the learned map; physical identification and long-scale validity are not established.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(run/'R_vs_g_step1000.png',dpi=200)
    (run/'R_vs_g_step1000.json').write_text(json.dumps(dict(checkpoint='training_step_1000.pt',plot_range=[-1.3,1.6],roots=roots),indent=2)+'\n')
    print(json.dumps(roots,indent=2))


if __name__=='__main__': main()
