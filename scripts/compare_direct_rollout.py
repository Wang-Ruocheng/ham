"""Compare full-ensemble Q(C^k D) with R^k(Q(D)); never iterate past available k."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);model=load_run(args.run).model
    z=np.load(args.run/'encodings.npz');d=z['test'];labels=z['test_labels']
    with torch.no_grad():
        r=[torch.from_numpy(d[:,0])]
        for _ in range(d.shape[1]-1):r.append(model.advance_theory(r[-1]))
        r=torch.stack(r,1).numpy()
    out=args.run/'direct_vs_rollout';out.mkdir(exist_ok=True)
    ks=[.32,.41,.50];bs=[-.002,0,.002];colors=['#2166ac','#333333','#b2182b']
    selected=[]
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for col,k in enumerate(ks):
        for b,color in zip(bs,colors):
            mask=np.isclose(labels[:,0],k)&np.isclose(labels[:,1],b)
            assert mask.sum()==2
            dm=d[mask].mean(0);rm=r[mask].mean(0)
            selected.append(dict(K=k,B=b,replicas=int(mask.sum()),direct=dm.tolist(),rollout=rm.tolist()))
            for j in range(2):
                ax=axs[j,col]
                ax.plot(range(3),dm[:,j],'-o',color=color,lw=1.7,ms=5)
                ax.plot(range(3),rm[:,j],'--x',color=color,lw=1.7,ms=6)
                for individual in d[mask]:ax.plot(range(3),individual[:,j],'-',color=color,alpha=.15,lw=.7)
                for individual in r[mask]:ax.plot(range(3),individual[:,j],'--',color=color,alpha=.15,lw=.7)
                ax.set(xlabel='k (L=16,8,4)',ylabel=f'g{j+1}',xticks=[0,1,2],title=f'K={k:.2f}');ax.grid(alpha=.2)
    handles=[Line2D([],[],color=c,label=f'B={b:g}') for b,c in zip(bs,colors)]+[Line2D([],[],color='black',marker='o',label='Q(C^k D), direct'),Line2D([],[],color='black',ls='--',marker='x',label='R^k(Q(D)), rollout')]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=9)
    fig.suptitle('Fixed C, covariance20: held-out full-ensemble encodings\nBold: mean of two replica trajectories; faint: individual replicas')
    fig.savefig(out/'coordinates_vs_k.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,k in zip(axs,ks):
        for item,color in zip([v for v in selected if v['K']==k],colors):
            dm=np.array(item['direct']);rm=np.array(item['rollout'])
            ax.plot(*dm.T,'-o',color=color,lw=1.8);ax.plot(*rm.T,'--x',color=color,lw=1.8)
            for step in (0,1,2):ax.annotate(str(step),dm[step],xytext=(4,4),textcoords='offset points',fontsize=8,color=color)
            for a,b in zip(dm[1:],rm[1:]):ax.plot([a[0],b[0]],[a[1],b[1]],':',color=color,alpha=.6)
        ax.set(xlabel='g1',ylabel='g2',title=f'K={k:.2f}',aspect='equal');ax.grid(alpha=.2)
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=9)
    fig.suptitle('Latent trajectories; numbers label direct k; dotted connectors show disagreement')
    fig.savefig(out/'latent_trajectories.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for j in range(2):
        for col,k in enumerate((1,2)):
            ax=axs[j,col];sc=ax.scatter(d[:,k,j],r[:,k,j],c=labels[:,0],cmap='viridis',s=12,vmin=.2,vmax=.68)
            limits=[min(d[:,k,j].min(),r[:,k,j].min()),max(d[:,k,j].max(),r[:,k,j].max())];ax.plot(limits,limits,'k--')
            ax.set(xlabel=f'direct g{j+1}',ylabel=f'rollout g{j+1}',title=f'k={k}, all held-out ensembles',aspect='equal');fig.colorbar(sc,ax=ax,label='K')
    fig.savefig(out/'all_test_parity.png',dpi=180);plt.close(fig)
    sd=z['train'][:,1:].var(0).mean(-1)**.5
    rmse=np.mean((r[:,1:]-d[:,1:])**2,axis=(0,2))**.5
    report=dict(convention='Full-ensemble direct encodings; not midpoint of two half-ensemble encodings used in training loss. Roll out each replica separately, then average for selected plots.',selected=selected,all_test=dict(ensembles=len(d),rmse=rmse.tolist(),nrmse_training_target_sd=(rmse/sd).tolist(),rmse_by_coordinate=np.mean((r[:,1:]-d[:,1:])**2,axis=0)**.5))
    report['all_test']['rmse_by_coordinate']=report['all_test']['rmse_by_coordinate'].tolist()
    np.savez(out/'trajectories.npz',direct=d,rollout=r,labels=labels)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['all_test'],indent=2))


if __name__=='__main__':main()
