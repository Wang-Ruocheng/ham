"""Final two-system relative/local geometry pilot overview."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run


def main():
    torch.set_num_threads(2)
    tf=Path('results/tfim1d_g2_relative_local_6000_seed91')
    ising=Path('results/ising2d_g2_relative_local_6000_seed91')
    out=Path('results/relative_local_two_systems_seed91');out.mkdir(exist_ok=True)
    t=json.loads((tf/'step6000/report.json').read_text())
    a=json.loads((tf/'dimension_audit_step6000/report.json').read_text())
    b=json.loads((ising/'dimension_audit_step6000/report.json').read_text())
    im=json.loads((ising/'full_report/metrics.json').read_text())
    data=np.load(tf/'step6000/trajectories.npz');g=data['direct'];labels=data['couplings'].ravel()
    all_labels=np.r_[data['train_couplings'].ravel(),labels]
    all_g=np.r_[data['train_direct'][:,0],g[:,0]]
    knots=np.unique(all_labels);curve=np.array([all_g[all_labels==v].mean(0) for v in knots]);segments=np.diff(curve,axis=0)
    for root in t['fixed_points']['roots']:
        frac=np.clip(np.sum((np.array(root['g'])-curve[:-1])*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-20),0,1)
        dist=np.linalg.norm(curve[:-1]+frac[:,None]*segments-root['g'],axis=1);idx=dist.argmin()
        root['micro_curve_projection_h_over_J']=float(knots[idx]+frac[idx]*(knots[idx+1]-knots[idx]))
        root['micro_curve_projection_residual']=float(dist[idx])
    report=dict(tfim=dict(diagnostics=t,dimension=a),ising=dict(diagnostics=im,dimension=b,
        fixed_points=json.loads((ising/'full_report/fixed_points.json').read_text())))
    for name,path in [('tfim',tf),('ising',ising)]:
        report[name]['final_history']=json.loads((path/'history_progress.json').read_text())[-1]
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    ax=axs[0];run=load_run(tf,6000);cloud=g.reshape(-1,2)
    lo,hi=cloud.min(0)-.25,cloud.max(0)+.25
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],27),np.linspace(lo[1],hi[1],27))
    grid=np.stack([xx.ravel(),yy.ravel()],1)
    with torch.no_grad():delta=run.model.advance_theory(torch.from_numpy(grid)).numpy()-grid
    unit=delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-12)
    ax.quiver(*grid.T,*unit.T,color='gray',alpha=.5)
    sc=ax.scatter(*cloud.T,c=np.repeat(labels,g.shape[1]),cmap='coolwarm',vmin=0,vmax=2,s=9)
    fig.colorbar(sc,ax=ax,label='h/J (all scales)')
    for root in t['fixed_points']['roots']:
        ax.scatter(*root['g'],marker='*',s=160,edgecolors='black',c='orange' if root['stability']=='saddle' else 'green')
    ax.set(xlabel='g1',ylabel='g2',title='1D TFIM: normalized R(g)-g',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    ax=axs[1];z=np.load(ising/'dimension_audit_step6000/grid.npz');means=z['g']
    for row in means:ax.plot(*row.T,c='steelblue',alpha=.7,lw=.8)
    for col in means.transpose(1,0,2):ax.plot(*col.T,c='darkorange',alpha=.7,lw=.8)
    ax.scatter(*means.reshape(-1,2).T,s=4,color='black',alpha=.2)
    ax.plot([],[],color='steelblue',label='vary B, fixed K');ax.plot([],[],color='darkorange',label='vary K, fixed B')
    ax.set(xlabel='g1',ylabel='g2',title='2D Ising: microscopic K/B grid in g');ax.legend(fontsize=9)
    fig.savefig(out/'overview.png',dpi=180);plt.close(fig)
    print(json.dumps({name:dict(weak_fraction=report[name]['dimension']['test_scales'][0]['weak_fraction'],
        loss=report[name]['final_history']['test_total']) for name in ['tfim','ising']},indent=2))


if __name__=='__main__':main()
