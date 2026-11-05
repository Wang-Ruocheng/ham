"""Zero-crystal-field three-state Ising, signed K, grouped by |K|."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.data.blume_capel import sample_signed_spin1, observable_series
from recurrent_rg.data.ensembles import EnsembleDataset, grouped_split, load_ensembles


def observables(x, K):
    eta=1-2*(np.indices(x.shape[-2:]).sum(0)%2)
    obs=observable_series(x,delta=0.,exchange=K)
    obs['abs_staggered_magnetization']=np.abs((x*eta).mean((1,2)))
    result={}
    for name,v in obs.items():
        blocks=np.array([b.mean() for b in np.array_split(v,min(16,len(v)))])
        result[name]=dict(mean=float(v.mean()),block_se=float(blocks.std(ddof=1)/np.sqrt(len(blocks))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,default=36)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--replicas',type=int,default=2)
    p.add_argument('--positive-points',type=int,default=23)
    p.add_argument('--k-max',type=float,default=.9)
    p.add_argument('--burn',type=int,default=2048)
    p.add_argument('--spacing',type=int,default=16)
    p.add_argument('--seed',type=int,default=91)
    a=p.parse_args();torch.set_num_threads(2)
    if a.positive_points<3 or a.k_max<=0 or a.replicas<1:raise ValueError('Invalid grid')
    a.output.mkdir(parents=True,exist_ok=False)
    ids=np.arange(-a.positive_points,a.positive_points+1)
    grid=ids*a.k_max/a.positive_points
    xs,labels,groups,records=[],[],[],[]
    for i,(cell,K) in enumerate(zip(ids,grid)):
        for rep in range(a.replicas):
            seed=a.seed+1000*i+rep
            x=sample_signed_spin1(float(K),a.length,a.samples,seed,burn=a.burn,spacing=a.spacing)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)+1),3).to(torch.uint8))
            labels.append([K]);groups.append([abs(int(cell))])
            records.append(dict(K=float(K),replica=rep,seed=seed,observables=observables(x,K)))
        print(f'Generated {i+1}/{len(grid)} K cells',flush=True)
    train,test=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(labels,dtype=torch.double),torch.tensor(groups)))
    torch.save(dict(schema_version=1,train=asdict(train),test=asdict(test)),a.output/'ensembles.pt')
    load_ensembles(a.output/'ensembles.pt')
    probes=[]
    for j,K in enumerate([.58,.62]):
        for k,start in enumerate(['random','plus','zero']):
            x=sample_signed_spin1(K,a.length,256,a.seed+100000+10*j+k,burn=4096,spacing=a.spacing,start=start)
            probes.append(dict(K=K,start=start,observables=observables(x,K)))
    meta=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        model='P(s) proportional to exp(K sum_<ij> s_i s_j); s=-1,0,+1; Delta=B=0',
        state_encoding=[-1,0,1],K_grid=grid.tolist(),records=records,probes=probes,
        train_shape=list(train.configurations.shape),test_shape=list(test.configurations.shape),
        split='Hold out entire |K| cells including both signs and all replicas; no gauge-partner leakage',
        sampler='Heat bath + occupied-spin SW at |K|; staggered gauge for K<0; independent seeds at each sign',
        note='Labels and probe observables are diagnostic only. MCMC draws are correlated; block SE is approximate.')
    (a.output/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axs,['abs_magnetization','abs_staggered_magnetization','occupancy']):
        vals=np.array([r['observables'][key]['mean'] for r in records]).reshape(-1,a.replicas)
        ax.plot(grid,vals.mean(1));ax.fill_between(grid,vals.min(1),vals.max(1),alpha=.2)
        for kc in [-1/1.693,1/1.693]:ax.axvline(kc,c='grey',ls=':')
        ax.set(xlabel='K=J/T',ylabel=key);ax.grid(alpha=.2)
    fig.savefig(a.output/'sampling.png',dpi=160);plt.close(fig)
    print(meta['train_shape'],meta['test_shape'],flush=True)


if __name__=='__main__':main()
