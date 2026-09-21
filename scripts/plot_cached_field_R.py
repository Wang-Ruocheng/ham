"""Readable single-panel R field from an already evaluated checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();directory=args.run/'step2000'
    d=json.loads((directory/'field_diagnostics.json').read_text());cfg=d['config']
    x=np.load(directory/'field_flow_data.npz')['direct']
    m=RecurrentRGNetwork2D(levels=cfg['levels'],hidden=cfg['hidden'],theory_dim=2,
                           learnable_kernel=cfg['learnable_kernel']).double().eval()
    m.load_state_dict(torch.load(args.run/'qcr_model.pt',weights_only=True),strict=False)
    lower,upper=np.array(d['plot_box']);xx,yy=np.meshgrid(np.linspace(lower[0],upper[0],29),np.linspace(lower[1],upper[1],29))
    grid=np.column_stack([xx.ravel(),yy.ravel()])
    with torch.no_grad():delta=m.theory_rg(torch.from_numpy(grid)).numpy()
    speed=np.linalg.norm(delta,axis=1)
    arrow=delta/np.maximum(speed[:,None],1e-12)*.025*min(upper-lower)
    near=cKDTree(x[:,:-1].reshape(-1,2)).query(grid)[0]<=d['near_sample_radius']
    fig,ax=plt.subplots(figsize=(8.5,8))
    ax.quiver(*grid[~near].T,*arrow[~near].T,color='.84',angles='xy',scale_units='xy',scale=1,width=.003)
    q=ax.quiver(*grid[near].T,*arrow[near].T,speed[near],cmap='viridis',angles='xy',scale_units='xy',scale=1,width=.004)
    for k in range(x.shape[1]):ax.scatter(x[:,k,0],x[:,k,1],s=3,alpha=.18,label=f'k={k}')
    for i,f in enumerate(d['fixed_points']):
        ax.scatter(*f['g'],marker='*',color='black',s=140,zorder=5)
        right=f['g'][0]>(lower[0]+upper[0])/2
        ax.annotate(f"F{i+1}: {f['stability']}",f['g'],xytext=(-10,12) if right else (10,12),
                    textcoords='offset points',ha='right' if right else 'left',fontsize=11,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1.5})
    ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),xlabel='$g_1$',ylabel='$g_2$',
           title='Original Q + fixed decimation: uniform (K,B) data\nDiscrete R(g) flow, seed 91, step 2000')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=8)
    fig.colorbar(q,ax=ax,shrink=.7,label=r'$\|R(g)-g\|$')
    fig.text(.5,.025,'Arrow direction: R(g)-g; lengths normalized. Gray: far from sampled R inputs.\n'
             'Numerical roots are not identified physical RG fixed points.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(directory/'R_vector_field_clean.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
