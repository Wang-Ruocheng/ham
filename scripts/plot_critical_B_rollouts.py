"""Direct and recurrent trajectories across B=0 near the Ising critical K."""
import argparse
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('results/ising2d_g2_relative_local_6000_seed91'))
    path=parser.parse_args().run
    model=load_run(path).model
    z=np.load(path/'full_report/encodings.npz')
    g=np.concatenate([z['train'],z['test']]);labels=np.concatenate([z['train_labels'],z['test_labels']])
    kc=.5*np.log(1+np.sqrt(2));ks=np.unique(labels[:,0]);lo=ks[ks<kc].max();hi=ks[ks>kc].min();w=(kc-lo)/(hi-lo)
    def cell(k,b):return g[np.isclose(labels[:,0],k)&np.isclose(labels[:,1],b)]
    def rollout(initial):
        with torch.no_grad():
            r=[torch.from_numpy(initial.copy())]
            for _ in range(2):r.append(model.advance_theory(r[-1]))
        return torch.stack(r,1).numpy()
    out=path/'critical_B_rollouts';out.mkdir(exist_ok=True)
    reports={}
    for mode in ['observed_K044','interpolated_Kc']:
        rows={}
        for b in [-.005,-.002,-.001,0.,.001,.002,.005]:
            if mode=='observed_K044':
                d=cell(lo,b);r=rollout(d[:,0])
                rows[b]=dict(direct=d.mean(0),rollout=r.mean(0),direct_replicas=d,rollout_replicas=r)
            else:
                d=(1-w)*cell(lo,b).mean(0)+w*cell(hi,b).mean(0)
                r=rollout(d[None,0])[0]
                rows[b]=dict(direct=d,rollout=r)
        subtitle='Observed K=0.44; average of two independently rolled replicas' if mode=='observed_K044' else 'Kc interpolation of mean encodings; R iterates interpolated mean g0'
        fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
        for col,b in enumerate([.001,.002,.005]):
            for row,key in enumerate(['direct','rollout']):
                ax=axes[row,col]
                for bb,c in [(-b,'#2166ac'),(0.,'gray'),(b,'#b2182b')]:
                    v=rows[bb][key];ax.plot(*v.T,'o-',color=c,label=f'B={bb:+g}',lw=1.5)
                    for k,point in enumerate(v):ax.annotate(str(k),point,xytext=(4,4),textcoords='offset points',fontsize=9,color=c)
                    for k in range(2):
                        ax.annotate('',v[k+1],v[k],arrowprops=dict(arrowstyle='->',color=c,lw=1))
                ax.set(xlabel='g1',ylabel='g2',title=('Q(C^k D)' if key=='direct' else 'R^k(g0)')+f'; |B|={b:g}')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            points=np.concatenate([rows[bb][key] for bb in [-b,0.,b] for key in ['direct','rollout']])
            pad=np.maximum(.04,.12*np.ptp(points,axis=0))
            for ax in axes[:,col]:ax.set(xlim=(points[:,0].min()-pad[0],points[:,0].max()+pad[0]),ylim=(points[:,1].min()-pad[1],points[:,1].max()+pad[1]))
        fig.suptitle(subtitle+'\nLabels 0,1,2 are RG scales (L=16,8,4); same axis limits within each column')
        fig.savefig(out/f'{mode}_latent.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained');metrics=[]
        for b,c in [(.001,'#66a61e'),(.002,'#7570b3'),(.005,'#e6ab02')]:
            item=dict(B_abs=b)
            for key,style in [('direct','-o'),('rollout','--x')]:
                delta=rows[b][key]-rows[-b][key]
                norms=np.linalg.norm(delta,axis=1);gap=abs(delta[:,1])
                item[key]=dict(pair_distance=norms.tolist(),g2_gap=gap.tolist(),g2_signed_half_gap=(delta[:,1]/2).tolist())
                axes[0].plot(range(3),norms/norms[0],style,c=c,label=f'{key}, |B|={b:g}')
                axes[1].plot(range(3),gap/gap[0],style,c=c)
            metrics.append(item)
        for ax,title in zip(axes,['Full g pair distance / initial','|g2(+B)-g2(-B)| / initial']):
            ax.set(xlabel='k',ylabel='Relative separation',title=title,xticks=[0,1,2]);ax.axhline(1,c='gray',lw=.8);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(subtitle)
        fig.savefig(out/f'{mode}_separation.png',dpi=170);plt.close(fig)
        reports[mode]=dict(rows={str(b):{key:value.tolist() for key,value in row.items()} for b,row in rows.items()},metrics=metrics)
    reports['conventions']=dict(Kc=kc,interpolation_K=[lo,hi],weight=w,
        note='No exact-Kc samples. Interpolated direct targets are not evaluated Q of a sampled Kc ensemble. Finite-data diagnostics, not exact derivatives; no trajectories beyond trained k2.')
    (out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps({mode:reports[mode]['metrics'] for mode in ['observed_K044','interpolated_Kc']},indent=2))


if __name__=='__main__':main()
