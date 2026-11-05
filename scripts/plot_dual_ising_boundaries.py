"""Map exact physical critical lines through a bilinear Q-grid interpolant."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories


def main():
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91'))
    folder=p.parse_args().run;out=folder/'report'
    run=load_run(folder);m=run.model.eval();tr,te=load_ensembles(run.config['data']['cache'])
    labels=torch.cat([tr.couplings,te.couplings]).numpy()
    t=collect_trajectories(m,torch.cat([tr.configurations,te.configurations]),samples=run.config['evaluation']['samples'],max_rg_steps=2)
    d=t['direct'];axes=[np.unique(labels[:,i]) for i in range(2)]
    values=np.stack([np.stack([d[(labels==[a,b]).all(1)].mean(0) for b in axes[1]]) for a in axes[0]])
    kc=np.log(1+np.sqrt(2))/2;u=np.linspace(0,1,401)
    parameters=[np.column_stack([np.full_like(u,kc),u]),np.column_stack([u,np.full_like(u,kc)])]
    interp=[RegularGridInterpolator(tuple(axes),values[:,:,k],bounds_error=True) for k in range(3)]
    curves=[[f(p) for p in parameters] for f in interp]
    centers=[f([[kc,kc]])[0] for f in interp]
    old=json.loads((out/'fixed_points.json').read_text());lo,hi=np.array(old['search_box'])
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],29),np.linspace(lo[1],hi[1],29));grid=np.column_stack([gx.ravel(),gy.ravel()])
    with torch.no_grad():delta=m.advance_theory(torch.from_numpy(grid)).numpy()-grid
    norm=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(norm[:,None],1e-12)
    def draw(ax,k):
        q=ax.quiver(*grid.T,*direction.T,norm,cmap='viridis',alpha=.55)
        ax.scatter(*d[:,k].T,s=6,c='.35',alpha=.3,label=f'Q(C^{k}D) samples')
        for j,(curve,color) in enumerate(zip(curves[k],['#d81b60','#0072b2'])):
            line=ax.plot(*curve.T,c=color,lw=2.8,label=f'K{j+1} = Kc, mapped at k={k}')[0]
            line.set_path_effects([pe.Stroke(linewidth=4.5,foreground='white'),pe.Normal()])
            for v in [0.,.25,1.]:
                z=curve[round(v*400)];ax.scatter(*z,c=color,s=17,zorder=5)
                ax.annotate(f'K{2-j}={v:g}',z,xytext=(5,4 if j==0 else -12),textcoords='offset points',fontsize=7,color=color)
        ax.scatter(*centers[k],s=75,c='black',marker='D',label='mapped (Kc,Kc)',zorder=6)
        for root in old['roots']:ax.scatter(*root['g'],marker='*',s=130,c='orange' if root['stability']=='unstable' else 'purple',edgecolors='black',linewidths=.4,label='R '+root['stability'],zorder=6)
        ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.grid(alpha=.12)
        return q
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained');q=draw(ax,0)
    ax.set_title('R(g)-g with interpolated microscopic critical lines\nBilinear Q map; NOT fitted R separatrices')
    ax.legend(fontsize=8,loc='lower left');fig.colorbar(q,ax=ax,label='|R(g)-g|; normalized arrows')
    fig.savefig(out/'R_with_theory_boundaries.png',dpi=180)
    fig,axs=plt.subplots(1,3,figsize=(17,5.6),layout='constrained')
    for k,ax in enumerate(axs):draw(ax,k);ax.set_title(f'Critical ensembles mapped at RG step k={k}')
    axs[0].legend(fontsize=6);fig.savefig(out/'theory_boundaries_by_scale.png',dpi=160)
    report=dict(critical_K=kc,interpolated_crossings_by_scale=[z.tolist() for z in centers],
                microscopic_crossing_distance_to_R_roots=[dict(stability=r['stability'],distance=float(np.linalg.norm(centers[0]-r['g']))) for r in old['roots']],
                method='Bilinear interpolation of three-replica mean Q encodings on the 11x11 grid; training and heldout cells used for diagnostic only.',
                caveat='K grid spacing=.1; finite L16,8,4; no direct data exactly at Kc. Mapped lines need not be stationary nor coincide with R separatrices. Off-data arrows are extrapolation.')
    (out/'theory_boundaries.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
