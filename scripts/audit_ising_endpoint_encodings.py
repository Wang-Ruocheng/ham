"""Exact infinite-ensemble Q encodings for independent and aligned spins."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);run=load_run(args.run);model=run.model
    if run.model_config.dimension!=2:raise ValueError('This diagnostic expects 2D Ising')
    width=model.geometry.patch_volume
    patterns=2*((torch.arange(2**width)[:,None]//(2**torch.arange(width)))%2).double()-1
    roots=json.loads((args.run/'full_report/fixed_points.json').read_text())['roots']
    result={'method':'Exact local-pattern expectation before Q head, infinite-ensemble limit. No new training or Monte Carlo sampling.',
            'caveat':'K=infinity,B=0 uses equal mixture of all-plus/all-minus; pure phases are different distributions.',
            'endpoints':{}}
    with torch.no_grad():
        features=model.theory_encoder.local_mlp(patterns)
        inputs={'K0_B0':features.mean(0),'Kinf_B0_symmetric_mixture':(features[0]+features[-1])/2,
                'pure_minus':features[0],'pure_plus':features[-1]}
        for name,feature in inputs.items():
            g=model.theory_encoder.head(feature);rg=model.advance_theory(g)
            distances=[float(np.linalg.norm(g.numpy()-r['g'])) for r in roots];i=int(np.argmin(distances))
            result['endpoints'][name]=dict(g=g.tolist(),R_g=rg.tolist(),displacement=(rg-g).tolist(),
                displacement_norm=float((rg-g).norm()),nearest_fixed_point=f'F{i+1}',
                distance_to_fixed_point=distances[i])
        blockpatterns=(2*((torch.arange(16)[:,None]//(2**torch.arange(4)))%2)-1).double()
        truth=model.rg_kernel(blockpatterns.reshape(16,1,2,2,1)).flatten()
        result['C']=dict(truth_table=truth.tolist(),iid_half_output_plus_probability=float((truth>0).double().mean()),
                         all_minus_output=float(truth[0]),all_plus_output=float(truth[-1]))
    output=args.run/'full_report/endpoint_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
