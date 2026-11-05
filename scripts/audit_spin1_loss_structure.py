"""Frozen-model audit of triplet geometry and categorical Q lookup correctness."""
import json
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution,plaquette_distribution

def main():
    torch.set_num_threads(2);report={}
    for name,run in [('spin1','results/spin1_signedK_g1_block3_locked_recipe_h96_3000_seed91'),('binary','results/ising2d_signedK_B0_g1_block3_locked_recipe_h96_3000_seed91')]:
        loaded=load_run(Path(run));model=loaded.model
        tr,te=load_ensembles(loaded.config['data']['cache'])
        x=torch.cat([tr.configurations,te.configurations]);K=torch.cat([tr.couplings,te.couplings]).ravel()
        ids=[int((K-v).abs().argmin()) for v in [K.min(),torch.tensor(0.),K.max()]]
        selected=x[ids].double();fn=categorical_plaquette_distribution if name=='spin1' else plaquette_distribution
        with torch.no_grad():
            a=fn(selected[:,:32]);b=fn(selected[:,32:]);p=(a+b)/2
            g=model.theory_encoder(selected).ravel();d={}
            for metric in ['euclidean','hellinger']:
                phi=p if metric=='euclidean' else p.sqrt()/2**.5
                noise=((a-b)**2).sum(-1)/4 if metric=='euclidean' else ((a.sqrt()-b.sqrt())**2).sum(-1)/8
                pairs=[]
                for i,j in [(0,1),(1,2),(0,2)]:
                    dist=(phi[i]-phi[j]).norm();allow=2*(noise[i]+noise[j]).sqrt()
                    pairs.append(dict(pair=[i,j],distance=float(dist),lower=float(4*(dist-allow).clamp_min(0)),upper=float(4*(dist+allow)),latent=float(abs(g[i]-g[j]))))
                feasible=[]
                for longest in range(3):
                    others=[i for i in range(3) if i!=longest]
                    lo=max(pairs[longest]['lower'],sum(pairs[i]['lower'] for i in others))
                    hi=min(pairs[longest]['upper'],sum(pairs[i]['upper'] for i in others))
                    feasible.append(lo<=hi)
                d[metric]=dict(pairs=pairs,any_collinear_interval_solution=any(feasible))
        result=dict(K=K[ids].tolist(),g=g.tolist(),geometry=d)
        if name=='spin1':
            encoder=model.theory_encoder;small=selected[:,:2,:6,:6]
            fast=encoder(small);params=list(encoder.parameters())
            gf=torch.autograd.grad(fast.sum(),params)
            patches=encoder.geometry.extract_patches(small)
            dense=encoder.head(encoder.local_mlp(patches).mean(tuple(range(1,patches.ndim-1))))
            gd=torch.autograd.grad(dense.sum(),params)
            result['lookup_forward_max_error']=float((fast-dense).abs().max())
            result['lookup_parameter_gradient_max_error']=max(float((a-b).abs().max()) for a,b in zip(gf,gd))
            # Observable checks use labels only to select diagnostic ensembles.
            s=selected.argmax(-1).double()-1
            result['occupancy']=(s.square().mean((1,2,3))).tolist()
            result['nearest_neighbor']=(s*torch.roll(s,-1,2)).mean((1,2,3)).tolist()
        report[name]=result
    out=Path('results/spin1_loss_structure_audit');out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
