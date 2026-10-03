"""Compare learned-C TFIM runs with coordinate-mean versus coordinate-sum errors."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from recurrent_rg.training import evaluate
from recurrent_rg.data.statistics import two_spin_distribution


def main():
    root=Path(__file__).resolve().parents[1];results=root/'results'
    runs={'mean':results/'joint_qcr_tfim_1d_g2_levels6_rollout5_2000_seed91',
          'sum':results/'joint_qcr_tfim_1d_g2_coordsum_levels6_2000_seed91'}
    cache=root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt'
    tx,_,vx,_=torch.load(cache,weights_only=True);torch.set_num_threads(2);report={}
    for name,p in runs.items():
        model=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=2).double().eval()
        mismatch=model.load_state_dict(torch.load(p/'qcr_model.pt',weights_only=True),strict=False)
        assert not mismatch.unexpected_keys and all(k.startswith('action_network.') for k in mismatch.missing_keys)
        model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
        item={}
        for split,x in [('train',tx),('test',vx)]:
            x=x[torch.linspace(0,len(x)-1,24).long()]
            item[split]={r:evaluate(model,x,two_spin_distribution,48,objective_kwargs=dict(objective_mode='data_geometry',geometry_scale=4.,noise_multiplier=2.,rollout_weight=5.,coordinate_reduction=r)) for r in ('mean','sum')}
        f=np.load(p/'vector_flow_data.npz');h=f['h_over_J'];d=f['direct'];knots=np.unique(h)
        k=knots[np.argmin(abs(knots-.01))];g0=d[h==k,0].mean(0);actual=d[h==k].mean(0)
        z=torch.tensor(g0);trajectory=[g0.tolist()]
        with torch.no_grad():
            for i in range(50):
                z=model.advance_theory(z)
                if i+1 in (1,5,20,50):trajectory.append(z.numpy().tolist())
        item['low_field_probe']={'h_over_J':float(k),'direct_means':actual.tolist(),'rollout_steps':[0,1,5,20,50],'rollout':trajectory,
                                'caveat':'Steps beyond5 are untrained extrapolation.'}
        item['vector_summary']=json.loads((p/'vector_flow_summary.json').read_text())
        report[name]=item
    (runs['sum']/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report.items():
        print(name,'test original objective',r['test']['mean']['total'],'test sum objective',r['test']['sum']['total'])
        print('roots',r['vector_summary']['fixed_points_in_plot_box'])
        print('NRMSE',r['vector_summary']['metrics']['test']['rollout_normalized_rmse'])
        print('low field',r['low_field_probe'])


if __name__=='__main__':main()
