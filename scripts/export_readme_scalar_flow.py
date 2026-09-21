"""Replot the actual joint scalar checkpoint; no training or image editing."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(1)
    source=Path('results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91')
    model=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=1).double().eval()
    model.load_state_dict(torch.load(source/'training_step_1000.pt',weights_only=True)['model'],strict=False)
    roots=json.loads((source/'R_vs_g_step1000.json').read_text())['roots']
    x=torch.linspace(-1.3,1.6,1000,dtype=torch.float64)[:,None]
    with torch.no_grad():y=model.advance_theory(x).numpy().ravel()
    xx=x.numpy().ravel();fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    axs[0].plot(xx,y,label='R(g)');axs[0].plot(xx,xx,'k--',label='identity')
    axs[1].plot(xx,y-xx,label='R(g)-g');axs[1].axhline(0,color='black',ls='--')
    for r in roots:
        g=r['g'];z=torch.tensor([[g]],dtype=torch.float64,requires_grad=True)
        prediction=model.advance_theory(z)
        assert abs(prediction.item()-g)<1e-8
        derivative=torch.autograd.grad(prediction.sum(),z)[0].item()
        assert abs(derivative-r['derivative'])<1e-6
        c='tab:green' if r['stability']=='stable' else 'tab:red'
        axs[0].scatter(g,g,color=c,label=f"{g:.3f}: {r['stability']}")
        axs[1].scatter(g,0,color=c)
    for ax in axs:ax.set_xlabel('g');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('R(g)');axs[1].set_ylabel('R(g)-g')
    fig.suptitle('1D TFIM | dim g=1 | joint configuration-only QCR | seed 91 | 1000 updates')
    fig.tight_layout();out=Path('docs/figures');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'tfim_g1_joint_R.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
