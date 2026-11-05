"""Independent data for coverage and finite-size diagnostics (no training here)."""
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.dual_ising import sample_dual_ising_couplings
from recurrent_rg.evaluation import collect_trajectories

SOURCE=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
OUT=SOURCE/'followup_four_directions'


def encode(model, states, depth):
    n,rep,samples,L,_=states.shape
    x=torch.nn.functional.one_hot(torch.from_numpy(states.astype(np.int64)).reshape(n*rep,samples,L,L),4).to(torch.uint8)
    t=collect_trajectories(model,x,samples=samples,max_rg_steps=depth,chunk_size=1)
    return t['direct'].reshape(n,rep,depth+1,2)


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    model=load_run(SOURCE).model.eval();kc=np.log(1+np.sqrt(2))/2
    cells=[]
    for axis in range(2):
        for other in [.2,.4,kc,.48,.7]:
            for offset in [-.05,-.025,.025,.05]:
                p=[kc,kc];p[axis]+=offset;p[1-axis]=other
                if p not in cells:cells.append(p)
    for a in [-.05,-.025,.025,.05]:
        for b in [-.05,-.025,.025,.05]:
            p=[kc+a,kc+b]
            if p not in cells:cells.append(p)
    # All coupling pairs distinct from the earlier exact-Kc audit and original grid.
    audit=np.load(SOURCE/'critical_sampling_audit'/'trajectories.npz')['couplings']
    assert all(not np.isclose(audit,p).all(1).any() for p in cells)
    cache=OUT/'dense_samples.npz'
    if not cache.exists():
        states=[]
        for i,p in enumerate(cells):
            states.append(np.stack([sample_dual_ising_couplings(*p,16,128,850000+i*10+r,burn=2048,spacing=16,ordered=bool(r%2)) for r in range(4)]))
            if i%5==0:print('dense sampling',i+1,len(cells),flush=True)
        np.savez_compressed(cache,states=np.stack(states),couplings=cells)
    data=np.load(cache)
    if not (OUT/'dense_encodings.npz').exists():
        direct=encode(model,data['states'],2)
        np.savez_compressed(OUT/'dense_encodings.npz',direct=direct,couplings=data['couplings'])
    print('DENSE READY',flush=True)
    probes=np.array([[kc-.03,kc-.03],[kc,kc],[kc+.03,kc+.03],[.2,kc],[.7,kc]])
    for L in [16,32,64]:
        dest=OUT/f'size_L{L}.npz'
        if dest.exists():continue
        states=[]
        for i,p in enumerate(probes):
            states.append(np.stack([sample_dual_ising_couplings(*p,L,128,900000+L*1000+i*10+r,burn=2048,spacing=16,ordered=bool(r%2)) for r in range(4)]))
            print('size sampling',L,i+1,len(probes),flush=True)
        states=np.stack(states);depth=int(np.log2(L//4))
        direct=encode(model,states,depth)
        np.savez_compressed(dest,states=states,couplings=probes,direct=direct)
        print('SIZE READY',L,flush=True)
    (OUT/'sampling.json').write_text(json.dumps(dict(dense_cells=len(cells),dense_replicas=4,samples=128,burn=2048,spacing=16,size_lengths=[16,32,64],size_cells=probes.tolist(),size_replicas=4,checkpoint=str(SOURCE/'qcr_model.pt'),note='All new samples independent of training and previous audit chains; dense training excludes audit coupling pairs.'),indent=2)+'\n')


if __name__=='__main__':main()
