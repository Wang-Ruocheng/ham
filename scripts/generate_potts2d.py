"""Generate a held-out-coupling Potts cache and reproducible sampling audit."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from recurrent_rg.data.potts import sample_potts
from recurrent_rg.data.ensembles import EnsembleDataset, grouped_split


def observables(x,q=3):
    bond = ((x==np.roll(x,1,axis=1)).mean((1,2))+(x==np.roll(x,1,axis=2)).mean((1,2)))/2
    occupancy = np.stack([(x==a).mean((1,2)) for a in range(q)],axis=-1)
    order = (q*occupancy.max(-1)-1)/(q-1)
    lag = float(np.corrcoef(bond[:-1],bond[1:])[0,1]) if bond.std()>0 else None
    return dict(bond_equal=float(bond.mean()),order=float(order.mean()),bond_lag1=lag,
                color_fractions=occupancy.mean(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--q',type=int,default=3)
    p.add_argument('--length',type=int,default=16)
    p.add_argument('--k-min',type=float,default=.4)
    p.add_argument('--k-max',type=float,default=1.6)
    p.add_argument('--burn',type=int,default=512)
    p.add_argument('--spacing',type=int,default=8)
    p.add_argument('--probe-samples',type=int,default=512)
    p.add_argument('--probe-burn',type=int,default=1024)
    p.add_argument('--include-critical',action='store_true')
    args=p.parse_args()
    if not 2<=args.q<=127 or args.length<6 or args.k_min<0 or args.k_max<=args.k_min or min(args.spacing,args.probe_samples)<1 or min(args.burn,args.probe_burn)<0:
        raise ValueError('Invalid Potts sampling configuration')
    args.output.mkdir(parents=True,exist_ok=False)
    xs,ks,audit=[],[],[]
    grid=np.linspace(args.k_min,args.k_max,64);kc=float(np.log1p(np.sqrt(args.q)))
    if args.include_critical:
        if not args.k_min<kc<args.k_max:raise ValueError('Critical coupling outside grid')
        grid[np.argmin(abs(grid-kc))]=kc
    for i,K in enumerate(grid):
        for replica in range(3):
            x=sample_potts(float(K),args.length,128,91+1000*i+replica,q=args.q,burn=args.burn,spacing=args.spacing)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)),args.q).to(torch.uint8))
            ks.append([K]);audit.append(dict(K=float(K),replica=replica,**observables(x,args.q)))
        if i%8==0:print(f'Generated {i+1}/64 couplings',flush=True)
    train,test=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(ks,dtype=torch.double)))
    torch.save(dict(schema_version=1,train=asdict(train),test=asdict(test)),args.output/'ensembles.pt')
    probes=[];probe_arrays={}
    probe_grid=(0.,.8,kc,1.2,4.) if args.q==3 else (args.k_min,kc-.04,kc,kc+.04,args.k_max)
    for i,K in enumerate(probe_grid):
        for ordered in (False,True):
            x=sample_potts(K,args.length,args.probe_samples,192+int(K*1000)+int(ordered),q=args.q,burn=args.probe_burn,spacing=args.spacing,ordered=ordered)
            energy=-((x==np.roll(x,1,axis=1)).mean((1,2))+(x==np.roll(x,1,axis=2)).mean((1,2)))
            probe_arrays[f'energy_{i}_{int(ordered)}']=energy
            batches=energy[:len(energy)//32*32].reshape(-1,32).mean(1) if len(energy)>=64 else energy
            probes.append(dict(K=K,ordered_start=ordered,energy_mean=float(energy.mean()),energy_batch_se=float(batches.std(ddof=1)/np.sqrt(len(batches))),**observables(x,args.q)))
    np.savez_compressed(args.output/'probe_energies.npz',K=np.array(probe_grid),**probe_arrays)
    (args.output/'sampling.json').write_text(json.dumps(dict(q=args.q,L=args.length,K_grid=grid.tolist(),replicas=3,
        samples=128,burn_sweeps=args.burn,spacing_sweeps=args.spacing,probe_samples=args.probe_samples,probe_burn=args.probe_burn,seed_formula='91+1000*K_index+replica',
        train_ensembles=len(train.configurations),test_ensembles=len(test.configurations),
        audit=audit,probes=probes),indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
