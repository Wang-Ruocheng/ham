"""Generate fixed-Delta Blume-Capel ensembles, held out by temperature."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.data.blume_capel import sample_blume_capel,sampling_observables
from recurrent_rg.data.ensembles import EnsembleDataset,grouped_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--delta',type=float,default=1.)
    p.add_argument('--length',type=int,default=16)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--replicas',type=int,default=3)
    p.add_argument('--temperatures',type=int,default=64)
    p.add_argument('--t-min',type=float,default=.6)
    p.add_argument('--t-max',type=float,default=2.4)
    p.add_argument('--burn',type=int,default=1024)
    p.add_argument('--spacing',type=int,default=8)
    p.add_argument('--seed',type=int,default=91)
    a=p.parse_args()
    if not 0<a.t_min<a.t_max or a.temperatures<3 or not 1<=a.replicas<1000 or a.samples<2:raise ValueError('Invalid sampling grid')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    xs,labels,records=[],[],[]
    grid=np.linspace(a.t_min,a.t_max,a.temperatures)
    for i,t in enumerate(grid):
        for rep in range(a.replicas):
            x=sample_blume_capel(float(t),a.length,a.samples,a.seed+1000*i+rep,delta=a.delta,burn=a.burn,spacing=a.spacing)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)+1),3).to(torch.uint8))
            labels.append([t]);records.append(dict(T=float(t),replica=rep,observables=sampling_observables(x,a.delta)))
        if i%8==0:print(f'Generated {i+1}/{len(grid)} temperatures',flush=True)
    train,test=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(labels,dtype=torch.double)))
    probes=[]
    # Independent starts and more samples audit equilibration, not train data.
    for j,t in enumerate(np.linspace(a.t_min,a.t_max,5)):
        for k,start in enumerate(['random','plus','zero']):
            x=sample_blume_capel(float(t),a.length,1024,a.seed+100000+j*10+k,delta=a.delta,burn=max(2048,a.burn),spacing=a.spacing,start=start)
            probes.append(dict(T=float(t),start=start,observables=sampling_observables(x,a.delta)))
    torch.save(dict(schema_version=1,train=asdict(train),test=asdict(test)),a.output/'ensembles.pt')
    meta=dict(model='H=-sum_<ij> s_i s_j + Delta sum_i s_i^2; zero field',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        state_encoding={'0':-1,'1':0,'2':1},sampler='N random-site heat-bath draws + occupied-spin Swendsen-Wang sign sweep',
        temperature_grid=grid.tolist(),seed_formula='seed+1000*temperature_index+replica',
        train_ensembles=len(train.configurations),test_ensembles=len(test.configurations),
        audit=records,probes=probes,note='MCMC draws, not assumed independent; block SE is approximate. Probes are not training data.')
    (a.output/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axs,['abs_magnetization','occupancy','energy']):
        vals=np.array([r['observables'][key]['mean'] for r in records]).reshape(-1,a.replicas)
        ax.plot(grid,vals.mean(1));ax.fill_between(grid,vals.min(1),vals.max(1),alpha=.2)
        ax.set(xlabel='T/J',ylabel=key,title=f'Delta/J={a.delta:g}, L={a.length}');ax.grid(alpha=.2)
    fig.savefig(a.output/'sampling.png',dpi=160);plt.close(fig)
    print(a.output,flush=True)


if __name__=='__main__':main()
