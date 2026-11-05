"""Extract zero-field signed-K ensembles without resampling or changing splits."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from recurrent_rg.data.ensembles import EnsembleDataset,load_ensembles
from recurrent_rg.runner import file_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nonnegative',action='store_true',help='Retain only K>=0 within the zero-field slice')
    a=p.parse_args();torch.set_num_threads(2)
    train,test=load_ensembles(a.source)
    selected=[];audit={}
    for name,ds in [('train',train),('test',test)]:
        mask=ds.couplings[:,1]==0
        if a.nonnegative:mask=mask & (ds.couplings[:,0]>=0)
        part=EnsembleDataset(ds.configurations[mask],ds.couplings[mask,:1]).validate()
        selected.append(part)
        K=part.couplings[:,0]
        L=part.configurations.shape[2]
        coords=torch.arange(L)
        eta=(1-2*((coords[:,None]+coords[None,:])%2))
        x=part.configurations[...,0].double()
        audit[name]=dict(shape=list(part.configurations.shape),K=sorted(K.unique().tolist()),
                        mean_abs_magnetization_per_ensemble=x.mean((-2,-1)).abs().mean(1).tolist(),
                        mean_abs_staggered_magnetization_per_ensemble=(x*eta).mean((-2,-1)).abs().mean(1).tolist(),
                        ensemble_K=K.tolist())
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema_version=1,train=asdict(selected[0]),test=asdict(selected[1])),a.output/'ensembles.pt')
    load_ensembles(a.output/'ensembles.pt')
    audit.update(source=str(a.source.resolve()),source_sha256=file_fingerprint(a.source),
                 operation='Exact B=0 slice; physical spins unchanged; original parameter-cell splits retained',
                 nonnegative_K_only=a.nonnegative,
                 note='K and magnetization statistics are metadata for diagnosis, never optimizer targets.')
    (a.output/'sampling_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print({s:audit[s]['shape'] for s in ['train','test']},flush=True)


if __name__=='__main__':main()
