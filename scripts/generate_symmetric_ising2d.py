"""Symmetric signed K and focused signed B grid; labels only for diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import torch
from recurrent_rg.ising2d_field import sample_signed_field_chain, chain_statistics


def symmetric_grid():
    positive_k = np.round(np.arange(23)*.03+.02, 8)
    positive_b = np.array([.0005,.001,.002,.005,.01,.02,.05,.2,.5])
    return np.r_[-positive_k[::-1],0.,positive_k], np.r_[-positive_b[::-1],0.,positive_b]


def uniform_field_grid(b_max=1.0):
    """Original positive-K grid and 21 equally weighted symmetric B cells."""
    if not np.isfinite(b_max) or b_max <= 0:
        raise ValueError('b_max must be finite and positive')
    return np.round(.2+.03*np.arange(17), 8), np.round(np.linspace(-b_max, b_max, 21), 8)


def held_out(i, j, k_center=23, b_center=9):
    return (abs(i-k_center)+abs(j-b_center))%3 == 1


def sample_cell(task):
    i,j,k,b,seed,*sizes = task
    sites = sizes[0] if sizes else 16
    k_center, b_center = sizes[1:] if len(sizes) == 3 else (23, 9)
    result=[]
    for replica in range(2):
        chain_seed=seed*100000+i*1000+j*2+replica
        x=sample_signed_field_chain(k,b,sites,64,512,16,chain_seed,1 if replica==0 else -1)
        eta=1-2*(np.indices((sites,sites)).sum(0)%2)
        result.append((x,dict(K=k,B=b,replica=replica,seed=chain_seed,test=held_out(i,j,k_center,b_center),
                            staggered_m2=float(((x*eta).mean((-2,-1))**2).mean()),**chain_statistics(x))))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=91)
    p.add_argument('--sites',type=int,default=16)
    p.add_argument('--grid', choices=['signed-focused', 'positive-uniform-B'], default='signed-focused')
    p.add_argument('--b-max', type=float, default=1.0, help='Half-range of positive-uniform-B grid')
    args=p.parse_args()
    if args.sites < 2 or args.sites % 2:
        p.error('--sites must be positive and even for signed-K periodic sampling')
    if not np.isfinite(args.b_max) or args.b_max <= 0:
        p.error('--b-max must be finite and positive')
    if args.grid == 'signed-focused' and args.b_max != 1.0:
        p.error('--b-max only applies to positive-uniform-B')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    ks,bs = symmetric_grid() if args.grid == 'signed-focused' else uniform_field_grid(args.b_max)
    k_center,b_center = (23,9) if args.grid == 'signed-focused' else (0,10)
    start=time.perf_counter()
    tx,ty,vx,vy,records=[],[],[],[],[]
    tasks=[(i,j,float(k),float(b),args.seed,args.sites,k_center,b_center) for i,k in enumerate(ks) for j,b in enumerate(bs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,rows in enumerate(pool.map(sample_cell,tasks),1):
            for x,r in rows:
                data,labels=(vx,vy) if r['test'] else (tx,ty)
                data.append(torch.from_numpy(x).unsqueeze(-1));labels.append([r['K'],r['B']]);records.append(r)
            if n%95==0:print(f'{n}/{len(tasks)} cells; {time.perf_counter()-start:.1f}s',flush=True)
    tensors=(torch.stack(tx),torch.tensor(ty,dtype=torch.float64),torch.stack(vx),torch.tensor(vy,dtype=torch.float64))
    torch.save(tensors,args.output/'ensembles.pt')
    audit=dict(K=ks.tolist(),B=bs.tolist(),grid=args.grid,sites=args.sites,draws=64,replicas=2,seed=args.seed,
               equilibration_sweeps=512,sweeps_between=16,train_shape=list(tensors[0].shape),test_shape=list(tensors[2].shape),
               sampler='SW at K>=0; checkerboard gauge SW with staggered cluster field at K<0; output physical spins',
               split=f'(abs(K_index-{k_center})+abs(B_index-{b_center}))%3==1; entire sign-related cells and replicas kept in same split',
               elapsed_seconds=time.perf_counter()-start,ensembles=records)
    (args.output/'sampling_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print({k:v for k,v in audit.items() if k!='ensembles'})


if __name__=='__main__':main()
