"""Compare real versus within-ensemble shuffled empirical MI, post training."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.empirical_mi import lattice_mi_statistics
from recurrent_rg.rsmi import categorical_kernel_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);r=load_run(a.run);_,test=load_ensembles(r.config['data']['cache']);q=r.config['model']['categorical_states']
    k=test.couplings[:,0].numpy();kc=np.log1p(np.sqrt(q));ids=[int(np.argmin(abs(k-v))) for v in [k.min(),kc,k.max()]]
    x=test.configurations[ids];actual=lattice_mi_statistics(x,states=q);shuffled=lattice_mi_statistics(x,states=q,shuffle_visible_seed=551)
    codes=(torch.arange(q**4)[:,None]//(q**torch.arange(4)))%q
    patterns=torch.nn.functional.one_hot(codes,q).double().flatten(1)[None]
    with torch.no_grad():
        table=categorical_kernel_probabilities(r.model.rg_kernel,patterns)[0]
        real=actual.information(table)/math.log(2);null=shuffled.information(table)/math.log(2)
    result=dict(K=k[ids].tolist(),MI_bits=real.tolist(),shuffled_MI_bits=null.tolist(),
        actual_statistics=actual.diagnostics,shuffled_statistics=shuffled.diagnostics,
        note='Shuffle breaks V/E pairing within each ensemble. Large shuffled empirical MI exposes plug-in bias; this is not population MI or a correction used in training.')
    (a.run/'full_report/mi_bias_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
