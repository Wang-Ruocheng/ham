"""Standalone scalar R(g) maps for the short-chain comparison, no retraining."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import brentq
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]/'results/tfim_short_vs_long_g1_seed91'
    out=root/'R_maps';out.mkdir(exist_ok=True)
    arms=[('long_L128_rep3','Long chain: L128, 3 replicas, k <= 5'),
          ('short_L16_rep3','Short chains: L16, 3 replicas, k <= 2'),
          ('short_L16_rep24','Short chains: L16, 24 replicas, k <= 2'),
          ('short_L16_pooled8','Short chains: L16, 24 replicas pooled 8-to-1')]
    bounds=(-1.5,2.0);grid=np.linspace(*bounds,2501);records={}
    fig,axes=plt.subplots(2,2,figsize=(12,11),sharex=True,sharey=True)
    for (name,title),ax in zip(arms,axes.flat):
        model=RecurrentRGNetwork1D(levels=6 if name.startswith('long') else 3,
                                   hidden=60,theory_dim=1).double().eval()
        result=model.load_state_dict(torch.load(root/name/'qcr_model.pt',weights_only=True),strict=False)
        assert not result.unexpected_keys and all(k.startswith('action_network.') for k in result.missing_keys)
        def full_R(g):
            with torch.no_grad():
                return model.advance_theory(torch.tensor([[g]],dtype=torch.float64)).item()
        with torch.no_grad(): values=model.advance_theory(torch.from_numpy(grid[:,None])).numpy().ravel()
        residual=values-grid
        roots=[]
        for i in np.flatnonzero(residual[:-1]*residual[1:]<0):
            g=brentq(lambda t:full_R(t)-t,grid[i],grid[i+1],xtol=1e-13)
            x=torch.tensor([[g]],dtype=torch.float64,requires_grad=True)
            slope=torch.autograd.grad(model.advance_theory(x).sum(),x)[0].item()
            roots.append(dict(g=g,derivative=slope,stable=abs(slope)<1))
        with np.load(root/name/'trajectories.npz') as a:
            inputs=a['training_direct'][:,:-1]
            support=(float(inputs.min()),float(inputs.max()))
        records[name]=dict(plotted_range=bounds,training_R_input_range=support,roots=roots)
        def draw(axis):
            axis.axvspan(*support,color='#eef5eb',zorder=0)
            axis.plot(grid,grid,'--',color='#444444',lw=1.4)
            inside=(grid>=support[0])&(grid<=support[1])
            axis.plot(grid,np.where(inside,values,np.nan),color='#1764ac',lw=2.4)
            axis.plot(grid,np.where(~inside,values,np.nan),color='#1764ac',lw=2,ls=':')
            for r in roots:
                g=r['g'];stable=r['stable']
                axis.scatter(g,g,s=65,marker='o' if stable else 'x',
                             color='#258443' if stable else '#ce3030',zorder=5)
                axis.annotate(f"g*={g:.3f}\nR'(g*)={r['derivative']:.3f}",
                              (g,g),xytext=(-12 if g>1 else 10,-32 if stable else 16),textcoords='offset points',
                              ha='right' if g>1 else 'left',
                              fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
            axis.set(xlim=bounds,ylim=bounds,xlabel='g',ylabel='R(g)',title=title)
            axis.set_aspect('equal',adjustable='box');axis.grid(alpha=.18)
            axis.legend(handles=[Line2D([],[],color='#1764ac',lw=2,label='R(g)'),
                                 Line2D([],[],color='#444444',ls='--',label='R(g) = g'),
                                 Line2D([],[],color='#258443',marker='o',ls='',label="Stable: |R'| < 1"),
                                 Line2D([],[],color='#ce3030',marker='x',ls='',label="Unstable: |R'| > 1")],
                        loc='upper left',fontsize=8)
        draw(ax)
        single,sax=plt.subplots(figsize=(7,7));draw(sax)
        single.suptitle('1D TFIM, dim g = 1, seed 91, 2,000 updates',fontsize=12)
        single.text(.5,.015,'Pale green: observed training R-input range. Dotted blue: extrapolation.',ha='center',fontsize=9)
        single.tight_layout(rect=(0,.035,1,.96));single.savefig(out/f'{name}_R.png',dpi=180);plt.close(single)
    fig.suptitle('1D TFIM: R(g) maps after 2,000 updates (dim g = 1, seed 91)',fontsize=15)
    fig.text(.5,.018,'Pale green: observed training R-input range; dotted blue: extrapolation.\nLatent coordinates are learned separately; roots shown only in the plotted interval.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(out/'R_comparison_2x2.png',dpi=180);plt.close(fig)
    (out/'roots.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
