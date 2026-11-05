"""Generate a six-state clock cache with held-out K cells and sampling diagnostics."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from recurrent_rg.data.clock import sample_clock, observables
from recurrent_rg.data.ensembles import EnsembleDataset, grouped_split


def summarize(x, q):
    obs=observables(x,q);result={k:float(v.mean()) for k,v in obs.items()}
    for key in ('bond','magnetization','locking'):
        v=obs[key]
        result[key+'_lag1']=float(np.corrcoef(v[:-1],v[1:])[0,1]) if v.std()>1e-12 else None
        batches=v[:len(v)//32*32].reshape(-1,32).mean(1)
        result[key+'_batch_se']=float(batches.std(ddof=1)/np.sqrt(len(batches))) if len(batches)>1 else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--q',type=int,default=6);p.add_argument('--length',type=int,default=16)
    p.add_argument('--k-min',type=float,default=.6);p.add_argument('--k-max',type=float,default=2.)
    p.add_argument('--couplings',type=int,default=64);p.add_argument('--replicas',type=int,default=3)
    p.add_argument('--samples',type=int,default=128);p.add_argument('--seed',type=int,default=91)
    p.add_argument('--burn',type=int,default=1024);p.add_argument('--spacing',type=int,default=32)
    p.add_argument('--probe-samples',type=int,default=1024)
    a=p.parse_args();torch.set_num_threads(2)
    if a.length<8 or a.length%4 or a.couplings<3 or a.replicas<1 or a.samples<64 or not 0<=a.k_min<a.k_max:
        raise ValueError('Invalid grid or lattice for two RG steps')
    a.output.mkdir(parents=True,exist_ok=False)
    grid=np.linspace(a.k_min,a.k_max,a.couplings);xs=[];ks=[];audit=[]
    for i,K in enumerate(grid):
        for r in range(a.replicas):
            x=sample_clock(K,a.length,a.samples,a.seed+1000*i+r,q=a.q,burn=a.burn,spacing=a.spacing)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)),a.q).to(torch.uint8));ks.append([K])
            audit.append(dict(K=float(K),replica=r,**summarize(x,a.q)))
        if i%8==0:print(f'Generated {i+1}/{len(grid)} K cells',flush=True)
    tr,te=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(ks,dtype=torch.double)))
    torch.save(dict(schema_version=1,train=asdict(tr),test=asdict(te)),a.output/'ensembles.pt')
    probes=[]
    for i,K in enumerate([a.k_min,1.11,1.27,1.43,a.k_max]):
        for ordered in (False,True):
            x=sample_clock(K,a.length,a.probe_samples,a.seed+100000+2*i+int(ordered),q=a.q,burn=2*a.burn,spacing=a.spacing,ordered=ordered)
            probes.append(dict(K=K,ordered_start=ordered,**summarize(x,a.q)))
    report=dict(q=a.q,L=a.length,K_grid=grid.tolist(),replicas=a.replicas,samples=a.samples,
                burn=a.burn,spacing=a.spacing,probe_samples=a.probe_samples,seed=a.seed,
                train_ensembles=len(tr.configurations),test_ensembles=len(te.configurations),audit=audit,probes=probes,
                convention='H/J=-sum cos(theta_i-theta_j), K=J/T. Embedded reflection multicluster sweeps; finite-size correlated MCMC samples, not exact critical estimates.')
    (a.output/'sampling.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,['bond','magnetization','locking']):
        for r in range(a.replicas):
            rows=[z for z in audit if z['replica']==r]
            ax.plot([z['K'] for z in rows],[z[key] for z in rows],'.-',alpha=.6)
        ax.set(xlabel='K=J/T',ylabel=key,title=f'Finite L={a.length} sampling');ax.grid(alpha=.2)
    fig.savefig(a.output/'sampling.png',dpi=160)
    print(json.dumps(dict(cache=str(a.output),probes=probes),indent=2))


if __name__=='__main__':main()
