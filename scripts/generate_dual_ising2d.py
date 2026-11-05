"""Generate the two-critical-point decoupled Ising benchmark."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from recurrent_rg.data.dual_ising import sample_dual_ising,sample_dual_ising_couplings
from recurrent_rg.data.ashkin_teller import observables
from recurrent_rg.data.ensembles import EnsembleDataset,grouped_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid',action='store_true',help='Independent Cartesian K1,K2 grid')
    p.add_argument('--grid-points',type=int,default=11,help='Points per grid axis; historical default 11')
    p.add_argument('--grid-max',type=float,default=1.,help='Upper limit for both independent axes; historical default 1')
    p.add_argument('--seed',type=int,default=91)
    p.add_argument('--length',type=int,default=16)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--replicas',type=int,default=3)
    p.add_argument('--points',type=int,default=48)
    p.add_argument('--scan-axis',choices=['K1','K2'],default='K1')
    p.add_argument('--ratio',type=float,default=.5,help='Other coupling / scanned coupling for ratio lines')
    p.add_argument('--skip-probes',action='store_true',help='Skip extra convergence probe samples; keep sampling audit')
    a=p.parse_args()
    if min(a.length,a.samples,a.replicas,a.points)<1 or not np.isfinite(a.ratio) or a.ratio<=0:
        raise ValueError('Positive lattice/sample/grid settings and ratio required')
    if a.grid_points<2 or not np.isfinite(a.grid_max) or a.grid_max<=0:
        raise ValueError('Independent grid requires at least two points and a finite positive upper limit')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    grid=np.linspace(0,a.grid_max,a.grid_points) if a.grid else np.linspace(0,1.3,a.points)
    cells=([(k1,k2) for k1 in grid for k2 in grid] if a.grid else
           [(k,a.ratio*k) if a.scan_axis=='K1' else (a.ratio*k,k) for k in grid])
    xs=[];ys=[];audit=[]
    for i,(K,K2) in enumerate(cells):
        for replica in range(a.replicas):
            x=sample_dual_ising_couplings(K,K2,a.length,a.samples,a.seed+1000*i+replica)
            scan=K if a.scan_axis=='K1' else K2
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)),4).to(torch.uint8));ys.append([K,K2] if a.grid else [scan])
            audit.append(dict(K=float(K),K2=float(K2),replica=replica,**{k:float(v.mean()) for k,v in observables(x).items() if k!='energy'}))
        if i%8==0:print(f'Generated {i+1}/{len(cells)} coupling cells',flush=True)
    tr,te=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(ys,dtype=torch.double)))
    torch.save(dict(schema_version=1,train=asdict(tr),test=asdict(te)),a.output/'ensembles.pt')
    probes=[]
    probes_grid=([(.2,.2),(.6,.2),(.2,.6),(.6,.6),(.44,.44),(.44,.6)] if a.grid else
                 [(k,a.ratio*k) if a.scan_axis=='K1' else (a.ratio*k,k) for k in [.3,.44,.6,.8,.88,1.05]])
    if a.skip_probes:probes_grid=[]
    for i,(K,K2) in enumerate(probes_grid):
        for ordered in [False,True]:
            x=sample_dual_ising_couplings(K,K2,a.length,1024,900000+2*i+ordered,burn=2048,ordered=ordered)
            row=dict(K=K,K2=K2,ordered_start=ordered)
            for name,v in observables(x).items():
                if name!='energy':row[name]=dict(mean=float(v.mean()),batch_se=float(v.reshape(-1,32).mean(1).std(ddof=1)/np.sqrt(32)))
            probes.append(row)
    kc=np.log(1+np.sqrt(2))/2
    scan_critical=sorted([kc,kc/a.ratio])
    meta=dict(model='two uncoupled square Ising layers',ratio=None if a.grid else a.ratio,
              scan_axis=None if a.grid else a.scan_axis,coupling_cells=[list(c) for c in cells],
              grid_mode='independent' if a.grid else 'ratio_line',K_grid=grid.tolist(),
              coupling_names=['K1','K2'] if a.grid else [a.scan_axis],critical_K=[kc,kc] if a.grid else scan_critical,
              L=a.length,samples=a.samples,replicas=a.replicas,seed=a.seed,burn=1024,spacing=16,
              train_ensembles=len(tr.configurations),test_ensembles=len(te.configurations),audit=audit,probes=probes)
    (a.output/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n');print('Sampling complete',flush=True)


if __name__=='__main__':main()
