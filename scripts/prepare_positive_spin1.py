"""Select strictly positive K from the existing spin-1 cache; preserve splits."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import EnsembleDataset,load_ensembles
from recurrent_rg.runner import file_fingerprint

def main():
    torch.set_num_threads(2)
    source=Path('data_cache/spin1_signedK_delta0_L36_n64_seed91/ensembles.pt').resolve()
    dest=Path('data_cache/spin1_positiveK_delta0_L36_n64_seed91').resolve()
    dest.mkdir(exist_ok=False)
    selected=[];meta={}
    for name,ds in zip(['train','test'],load_ensembles(source)):
        ids=torch.where(ds.couplings[:,0]>0)[0]
        sub=EnsembleDataset(ds.configurations[ids],ds.couplings[ids],None if ds.group_ids is None else ds.group_ids[ids]).validate()
        selected.append(sub)
        meta[name]=dict(source_indices=ids.tolist(),shape=list(sub.configurations.shape),K_grid=torch.unique(sub.couplings).tolist())
    cache=dest/'ensembles.pt'
    torch.save(dict(schema_version=1,train=asdict(selected[0]),test=asdict(selected[1])),cache)
    load_ensembles(cache)
    meta.update(source=str(source),source_sha256=file_fingerprint(source),data_sha256=file_fingerprint(cache),selection='K>0, excludes K=0; unchanged configurations and original split; labels not used by training')
    (dest/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n')
    cfg=RunConfig.load('configs/spin1_signedK_g1_block3_locked_recipe_h96_seed91.json').to_dict()
    cfg['data']['cache']=str(cache)
    cfg['data']['physical_model']='Square spin-1 Ising, strictly positive K, zero crystal and longitudinal fields'
    Path('configs/spin1_positiveK_g1_block3_notail_3000_seed91.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
