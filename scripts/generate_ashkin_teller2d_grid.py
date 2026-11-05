"""Uniform independent J/T,U/T grid; no labels enter QCR training batches."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.data.ashkin_teller import sample_ashkin_teller_couplings,observables
from recurrent_rg.data.ensembles import EnsembleDataset,grouped_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-j',type=float,help='Fix dimensionless J/T; labels then contain only U/T')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    if args.fixed_j is not None and (not np.isfinite(args.fixed_j) or args.fixed_j<0):
        raise ValueError('fixed-j must be finite and nonnegative')
    js=np.linspace(0,.6,11) if args.fixed_j is None else np.array([args.fixed_j])
    us=np.linspace(0,1.2,11 if args.fixed_j is None else 48)
    xs=[];ys=[];audit=[]
    for i,(J,U) in enumerate((j,u) for j in js for u in us):
        for replica in range(3):
            x=sample_ashkin_teller_couplings(J,U,16,128,91+1000*i+replica)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)),4).to(torch.uint8))
            ys.append([J,U] if args.fixed_j is None else [U])
            audit.append(dict(J=float(J),U=float(U),replica=replica,
                              **{k:float(v.mean()) for k,v in observables(x).items() if k!='energy'}))
        if i%8==0:print(f'Generated {i+1}/{len(js)*len(us)} coupling cells',flush=True)
    tr,te=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(ys,dtype=torch.double)))
    torch.save(dict(schema_version=1,train=asdict(tr),test=asdict(te)),args.output/'ensembles.pt')
    probes=[]
    probe_points=([(0.,.5),(.3,0.),(.2,.4),(.24,.48),(.27465,.27465),(.4,.8)] if args.fixed_j is None
                  else [(args.fixed_j,u) for u in [.2,.38,.42,.46,.6,1.]])
    for i,(J,U) in enumerate(probe_points):
        for ordered in (False,True):
            x=sample_ashkin_teller_couplings(J,U,16,1024,800000+2*i+ordered,burn=2048,ordered=ordered)
            row=dict(J=J,U=U,ordered_start=ordered)
            for k,v in observables(x).items():
                if k!='energy':row[k]=dict(mean=float(v.mean()),batch_se=float(v.reshape(-1,32).mean(1).std(ddof=1)/np.sqrt(32)))
            probes.append(row)
    meta=dict(model='Ashkin-Teller',dimensionless=True,fixed_J=args.fixed_j,
              coupling_names=['J/T','U/T'] if args.fixed_j is None else ['U/T'],J_grid=js.tolist(),U_grid=us.tolist(),L=16,
              samples=128,replicas=3,seed=91,burn=1024,spacing=16,train_ensembles=len(tr.configurations),
              test_ensembles=len(te.configurations),audit=audit,probes=probes,
              sampler='conditional SW: sigma/product for U>=J, sigma/tau for U<J; no negative bonds')
    (args.output/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Data and hot/cold probes complete',flush=True)


if __name__=='__main__':main()
