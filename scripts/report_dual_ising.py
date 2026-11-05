"""Known critical couplings are annotations only, not training targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.runner import evaluate_run
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,closure_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'report';evaluate_run(a.run,output=out)
    run=load_run(a.run);tr,te=load_ensembles(run.config['data']['cache'])
    samples=run.config['evaluation']['samples']
    train=collect_trajectories(run.model,tr.configurations,samples=samples,max_rg_steps=2)
    test=collect_trajectories(run.model,te.configurations,samples=samples,max_rg_steps=2)
    K=torch.cat([tr.couplings,te.couplings])[:,0].numpy();knots=np.unique(K)
    d=np.concatenate([train['direct'],test['direct']]);avg=np.stack([d[K==v].mean(0) for v in knots])
    kc=np.log(1+np.sqrt(2))/2;critical=[kc,2*kc]
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for j in range(2):
        for k in range(3):axs[j].plot(knots,avg[:,k,j],'.-',ms=3,label=f'k={k}')
        for v in critical:axs[j].axvline(v,c='k',ls='--',lw=1)
        axs[j].set(xlabel='initial K=J1/T',ylabel=f'g{j+1}',title='Dashed: exact microscopic critical K')
    for k in range(3):axs[2].plot(*avg[:,k].T,'.-',label=f'k={k}')
    pts=[]
    for i,v in enumerate(critical):
        g=[float(np.interp(v,knots,avg[:,0,j])) for j in range(2)];pts.append(g)
        axs[2].scatter(*g,s=85,marker='x',c='k');axs[2].annotate(f'initial Kc{i+1}',g,xytext=(5,5),textcoords='offset points')
    axs[2].set(xlabel='g1',ylabel='g2',title='Initial critical encodings are NOT assumed fixed points')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'known_boundaries_and_Q.png',dpi=170)
    roots=json.loads((out/'fixed_points.json').read_text())
    lo,hi=np.array(roots['search_box'])
    mesh=np.meshgrid(*[np.linspace(a,b,31) for a,b in zip(lo,hi)])
    grid=np.stack(mesh,-1).reshape(-1,2)
    with torch.no_grad():delta=run.model.advance_theory(torch.from_numpy(grid)).numpy()-grid
    speed=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(speed[:,None],1e-12)
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    ax.quiver(*grid.T,*direction.T,color='.6',alpha=.65)
    sc=ax.scatter(*d[:,0].T,c=K,cmap='viridis',s=12,alpha=.7)
    ax.plot(*avg[:,0].T,c='.25',lw=1,label='Microscopic Q(K, K/2) curve')
    for i,g in enumerate(pts):
        ax.scatter(*g,marker='D',s=80,c=['crimson','deepskyblue'][i],edgecolors='k',
                   label=f'Interpolated Q(Kc{i+1}), K={critical[i]:.4f}')
    for i,r in enumerate(roots['roots']):
        ax.scatter(*r['g'],marker='*',s=130,c='gold',edgecolors='k')
        ax.annotate(f'F{i+1}: {r["stability"]}',r['g'],xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
           title='R(g)-g: normalized arrows; off-data flow is extrapolation\nCritical markers annotate input data, not fitted basin boundaries')
    ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Initial K=J1/T')
    fig.savefig(out/'R_with_two_transitions.png',dpi=170);plt.close(fig)
    h=json.loads((a.run/'history_progress.json').read_text());fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for split in ['train','test']:ax.plot([r['epoch'] for r in h],[r[f'{split}_total'] for r in h],label=split)
    ax.set_yscale('symlog',linthresh=.01)
    ax.set(xlabel='training step',ylabel='total loss');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'loss.png',dpi=170)
    report=dict(exact_microscopic_critical_K=critical,interpolated_initial_critical_g=pts,
                heldout_closure=closure_metrics(test,reference_direct=train['direct']),
                covariance_eigenvalues_by_scale=[np.linalg.eigvalsh(np.cov(d[:,k].T,bias=True)).tolist() for k in range(3)],
                caveat='Finite L16 data. Only two RG steps trained. Initial critical encodings need not be RG fixed points; markers alone do not verify phase recovery.')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
