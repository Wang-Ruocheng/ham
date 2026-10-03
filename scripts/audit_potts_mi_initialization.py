"""Training-data MI references and permutation-null diagnostic; no training."""
import argparse,json
from pathlib import Path
import torch
from recurrent_rg.runner import prepared_data
from recurrent_rg.config import RunConfig
from recurrent_rg.empirical_mi import lattice_mi_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    meta=json.loads((args.run/'manifest.json').read_text());cfg=RunConfig.from_dict(meta['config'])
    tr,te=prepared_data(cfg);q=cfg.model.categorical_states
    codes=(torch.arange(q**4)[:,None]//(q**torch.arange(4)))%q
    candidates={'selected_initial_rule':torch.nn.functional.one_hot(torch.tensor(meta['data_driven_initialization']['selected_outputs']),q).double()}
    for i in range(4):candidates[f'decimation_site_{i}']=torch.nn.functional.one_hot(codes[:,i],q).double()
    counts=torch.nn.functional.one_hot(codes,q).sum(1)
    # Tie break by first occurrence in the block, not numeric color label.
    plurality=[]
    for row,c in zip(codes,counts):plurality.append(next(int(a) for a in row if c[a]==c.max()))
    candidates['plurality_first_occurrence_tie']=torch.nn.functional.one_hot(torch.tensor(plurality),q).double()
    result={}
    for split,data in [('train',tr),('test',te)]:
        result[split]={}
        for kind,seed in [('observed',None),('shuffled_pairing',12345)]:
            stats=lattice_mi_statistics(data.configurations,states=q,shuffle_visible_seed=seed)
            result[split][kind]={name:float(stats.information(table).mean()/torch.log(torch.tensor(2.))) for name,table in candidates.items()}
            print(split,kind,result[split][kind],flush=True)
    result['note']='Same finite cache; shuffle V/E pairings within each ensemble, retaining both marginals. Nonzero shuffled plug-in MI measures finite-sample bias, not true dependence. Reference rules never enter training or checkpoint selection.'
    (args.run/'mi_initialization_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
