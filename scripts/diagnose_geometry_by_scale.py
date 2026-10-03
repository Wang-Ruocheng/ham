"""Read-only model diagnostic: geometry on direct Q versus autonomous R."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def describe(g, pa, pb, mask=None):
    p = (pa+pb)/2
    noise = (pa-pb).square().sum(-1)/4
    i,j = torch.triu_indices(len(g),len(g),1)
    d = (p[i]-p[j]).norm(dim=-1)
    allowance = 2*(noise[i]+noise[j]).sqrt()
    lo,hi = 4*(d-allowance).clamp_min(0),4*(d+allowance)
    z = torch.pdist(g)
    low = (lo-z).clamp_min(0).square()
    high = (z-hi).clamp_min(0).square()
    if mask is None:
        mask = torch.ones(len(z),dtype=torch.bool)
    def avg(t): return float(t[mask].mean())
    return dict(loss=avg(low+high), too_close=avg(low), too_far=avg(high),
                resolved_fraction=avg((lo>0).double()), lower=avg(lo), upper=avg(hi),
                latent_distance=avg(z), target=avg(4*d), allowance=avg(4*allowance))


def inspect(model, raw, cross_mask=None):
    level = balanced_views(raw,128)
    records=[]
    with torch.no_grad():
        for k in range(6):
            a,b=level[:,:128],level[:,128:]
            g=(model.theory_encoder(a)+model.theory_encoder(b))/2
            r=g if k==0 else model.advance_theory(r)
            pa,pb=two_spin_distribution(a),two_spin_distribution(b)
            direct,pred=describe(g,pa,pb),describe(r,pa,pb)
            exact,_=distribution_geometry_loss(g,pa,pb)
            assert abs(float(exact)-direct['loss'])<1e-12
            rec=dict(k=k,sites=level.shape[2],direct=direct,rollout=pred,
                     closure_mse=float((r-g).square().mean()))
            if cross_mask is not None:
                rec['cross_pair_direct']=describe(g,pa,pb,cross_mask)
                rec['cross_pair_rollout']=describe(r,pa,pb,cross_mask)
            records.append(rec)
            if k<5: level=model.rg_kernel(level)
    return records


def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    run=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_rollout5_seed91'
    model=RecurrentRGNetwork1D(levels=4,hidden=60,theory_dim=1).double().eval()
    inc=model.load_state_dict(torch.load(run/'qcr_model.pt',weights_only=True),strict=False)
    assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
    model.theory_encoder.binary_lookup=True
    model.rg_kernel.binary_lookup=True
    tx,th,vx,vh=torch.load(root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    x,h=torch.cat((tx,vx)),torch.cat((th,vh))
    values=[h[(h-t).abs().argmin()] for t in [.95,1.05]]
    ids=torch.cat([torch.where(h==v)[0] for v in values])
    i,j=torch.triu_indices(len(ids),len(ids),1)
    cross=(h[ids[i]]!=h[ids[j]])
    summary=json.loads((run/'summary.json').read_text())
    last=summary['history'][-1]
    logged={split:[{key:last[f'{split}_{key}_level_{k}'] for key in
                  ['data_geometry','resolved_fraction','target_distance','noise_allowance','latent_distance']} for k in range(4)] for split in ['train','test']}
    report=dict(logged_24_systems_48_samples=logged,
                full_test_63_systems_128_samples=inspect(model,vx),
                selected_h_over_J=[float(v) for v in values],
                selected_six_ensembles=inspect(model,x[ids],cross),
                notes='Geometry only trains direct Q, k=0..3. R geometry is diagnostic, not optimized. k=4,5 are outside training. Each trained scale contributes 10/4 times its direct loss. Cross-pair statistics average all nine cross-side replica pairs; labels select diagnostics only.')
    (run/'geometry_by_scale.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
