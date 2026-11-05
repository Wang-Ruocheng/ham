"""Held-out cross-scale aliasing audit with split-noise diagnostic thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.evaluation import collect_trajectories, closure_metrics


def audit(path):
    run=load_run(path);m=run.model.eval()
    train,test=load_ensembles(run.config['data']['cache'])
    with torch.no_grad():
        levels=m.coarse_grain(test.configurations.double())
        halves=[];freq=[]
        for z in levels:
            a,b=z.chunk(2,dim=1)
            halves.append(torch.stack([m.theory_encoder(a),m.theory_encoder(b)]))
            freq.append(torch.stack([categorical_plaquette_distribution(a),categorical_plaquette_distribution(b)]))
        q=torch.stack(halves,2).numpy()  # split, system, scale, coordinate
        p=torch.stack(freq,2).numpy()
        ref=collect_trajectories(m,train.configurations,samples=128,max_rg_steps=2)
        traj=collect_trajectories(m,test.configurations,samples=128,max_rg_steps=2)
    # Only cross-scale pairs k=0 versus k=1, on wholly held-out K groups.
    g=q.mean(0);qn=.25*((q[0]-q[1])**2).sum(-1)
    f=p.mean(0);pn=.25*((p[0]-p[1])**2).sum(-1)
    latent=np.linalg.norm(g[:,0,None]-g[None,:,1],axis=-1)
    future=np.linalg.norm(f[:,1,None]-f[None,:,2],axis=-1)
    allowance=2*np.sqrt(pn[:,1,None]+pn[None,:,2])
    resolved=future>allowance
    near=latent<=2*np.sqrt(qn[:,0,None]+qn[None,:,1])
    sd=np.sqrt(np.var(g[:,:2].reshape(-1,g.shape[-1]),axis=0).sum())
    near_fixed=latent<=.05*sd
    report={'closure':closure_metrics(traj,reference_direct=ref['direct']),
            'cross_scale_pairs':int(latent.size),
            'near_pairs_split_noise':int(near.sum()),
            'resolved_future_conflicts_split_noise':int((near & resolved).sum()),
            'near_pairs_5pct_sd':int(near_fixed.sum()),
            'resolved_future_conflicts_5pct_sd':int((near_fixed & resolved).sum()),
            'mean_resolved_future_distance_near_5pct_sd':float(np.maximum(future-allowance,0)[near_fixed].mean()) if near_fixed.any() else None,
            'latent_rms_spread':float(sd),
            'Q_half_split_rms':float(np.sqrt(qn[:,:2].mean())),
            'note':'Pair counts are dependent, not independent tests. Split-noise bounds are heuristic, not confidence intervals. Future local-frequency distinguishability diagnoses retained information, not complete distributional or infinite-scale RG closure.'}
    return report,latent/sd,np.maximum(future-allowance,0),traj,test.couplings[:,0].numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    baseline=Path('results/ashkin_teller2d_g2_middle_dense_fixed_decimation_1200_seed91')
    out=args.run/'future_audit';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');result={}
    for j,(name,path) in enumerate([('baseline',baseline),('future_geometry',args.run)]):
        report,x,y,traj,K=audit(path);result[name]=report
        axes[0,j].scatter(x.flatten(),y.flatten(),s=5,alpha=.3)
        axes[0,j].axvline(.05,color='red',ls='--')
        axes[0,j].set(xlim=(0,.2),xlabel='cross-scale latent distance / RMS spread',ylabel='future frequency distance above noise',title=name)
        direct=traj['direct']
        for k in range(3):
            order=np.argsort(K);axes[1,j].plot(*direct[order,k].T,'.-',ms=3,lw=.6,label=f'k={k}')
        axes[1,j].set(xlabel='g1',ylabel='g2');axes[1,j].legend()
    fig.savefig(out/'closure.png',dpi=170)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
