"""B=0 trajectories on either side of Kc; only observed coarse scales."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);model=load_run(args.run).model;z=np.load(args.run/'encodings.npz')
    direct=np.concatenate((z['train'],z['test']));labels=np.concatenate((z['train_labels'],z['test_labels']))
    is_test=np.r_[np.zeros(len(z['train']),dtype=bool),np.ones(len(z['test']),dtype=bool)]
    ks=[.41,.44,.47,.5];colors=['#2166ac','#67a9cf','#ef8a62','#b2182b'];rows=[]
    for k in ks:
        mask=np.isclose(labels[:,0],k)&(labels[:,1]==0);d=direct[mask]
        with torch.no_grad():
            r=[torch.from_numpy(d[:,0])]
            for _ in range(d.shape[1]-1):r.append(model.advance_theory(r[-1]))
            r=torch.stack(r,1).numpy()
        rows.append(dict(K=k,B=0,split='test' if is_test[mask].all() else 'train',direct=d.tolist(),rollout=r.tolist()))
    out=args.run/'B0_critical_rollout';out.mkdir(exist_ok=True)
    for shifted in (False,True):
        fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        for row,color in zip(rows,colors):
            d=np.array(row['direct']);r=np.array(row['rollout'])
            if shifted:
                r=r-d[:,0:1];d=d-d[:,0:1]
            for j,ax in enumerate(axs):
                ax.plot(range(3),r.mean(0)[:,j],'-o',color=color,lw=2,label=f'K={row["K"]:.2f} ({row["split"]})')
                ax.plot(range(3),d.mean(0)[:,j],'--x',color=color,alpha=.55,lw=1)
                for replica in r:ax.plot(range(3),replica[:,j],color=color,alpha=.17,lw=.7)
                ax.set(xlabel='k (L=16,8,4)',ylabel=(f'g{j+1}(k)-g{j+1}(0)' if shifted else f'g{j+1}'),xticks=[0,1,2]);ax.grid(alpha=.2)
        axs[0].legend(fontsize=9)
        fig.suptitle('B=0; Kc=0.4406868; blue below Kc, red above Kc\nSolid circles: R rollout; faint dashed crosses: direct Q(C^kD); mean of two replicas')
        fig.savefig(out/('displacements.png' if shifted else 'rollout.png'),dpi=180);plt.close(fig)
    distances=[]
    for ia,ib in [(1,2),(0,3)]:
        a=rows[ia];b=rows[ib]
        distances.append(dict(K_pair=[a['K'],b['K']],rollout_mean_distance=np.linalg.norm(np.mean(b['rollout'],0)-np.mean(a['rollout'],0),axis=-1).tolist(),direct_mean_distance=np.linalg.norm(np.mean(b['direct'],0)-np.mean(a['direct'],0),axis=-1).tolist()))
    report=dict(Kc=float(.5*np.log(1+np.sqrt(2))),rows=rows,pair_distances=distances,note='No critical-coupling interpolation; full-ensemble encodings. Rollout each replica separately then average. Available data only k0..2. K=.44 is below, not exactly at, Kc.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(distances,indent=2))


if __name__=='__main__':main()
