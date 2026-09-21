"""Paired fixed-C QCR control: latent utilization and common balanced metric."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.joint_qcr_ising2d_field import build_model
from experiments.joint_qcr_fixed_cache import halves
from experiments.supervised_q_ising2d_multiscale import patterns_for
from experiments.joint_qcr_ising import distribution_geometry_loss


def covariance(x):
    centered=x-x.mean(0)
    return centered.T@centered/(len(x)-1)


def balanced_metric(cov,s2):
    with torch.no_grad():
        values,vectors=torch.linalg.eigh(cov)
        return (vectors/values.clamp_min(.1*s2))@vectors.T


def quadratic(error,metric):
    return ((error@metric)*error).sum(-1).mean()/error.shape[-1]


def objective(model,ha,hb,patterns,marginal,metric,s2):
    features=model.theory_encoder.local_mlp(patterns)
    a=model.theory_encoder.head(ha@features)
    b=model.theory_encoder.head(hb@features)
    direct=.5*(a+b)
    pa,pb=ha@marginal,hb@marginal
    geometry=torch.stack([distribution_geometry_loss(direct[:,k],pa[:,k],pb[:,k])[0] for k in range(4)]).mean()
    cov=covariance(direct[:,0])
    use=(cov/s2-torch.eye(2,dtype=cov.dtype)).square().sum()/2
    same=quadratic(a-b,metric)
    step=quadratic(model.advance_theory(direct[:,:-1])-direct[:,1:],metric)
    roll=direct[:,0];errors=[]
    for k in range(1,4):
        roll=model.advance_theory(roll)
        errors.append(quadratic(roll-direct[:,k],metric))
    rollout=torch.stack(errors).mean()
    terms=dict(geometry=geometry,use=use,same=same,step=step,rollout=rollout)
    total=10*geometry+10*use+5*(same+step+rollout)
    return total,terms,cov.detach()


def main():
    root=Path(__file__).resolve().parents[2]
    old=root/'results/joint_qcr_ising2d_uniform_field_g2_h60_L32_fixedC_2000_seed91_cached'
    out=root/'results/joint_qcr_ising2d_uniform_balanced_g2_seed91'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(91)
    model=build_model(False)
    patterns=patterns_for(model.theory_encoder)
    bits=(patterns>0).long()
    marginal=nn.functional.one_hot(8*bits[:,4]+4*bits[:,7]+2*bits[:,5]+bits[:,8],16).double()
    counts=[np.load(root/f'results/supervised_Q_structure_uniform_KB_seed91/{s}_sample_counts.npy',mmap_mode='r') for s in ('train','test')]
    cached=torch.load(root/'results/supervised_Q_ising2d_uniform_KB_multiscale_h60_seed91/exact_patch_histograms.pt',weights_only=True)
    # E ||4(p_i-p_j)||^2 = 32 trace(Cov(p)); divide by 2d for a fixed target variance.
    s2=float(8*covariance(cached['train'][:,0]@marginal).trace())
    ema=s2*torch.eye(2,dtype=torch.float64)
    params=[p for module in (model.theory_encoder,model.theory_rg) for p in module.parameters()]
    optimizer=torch.optim.Adam(params,lr=.001)
    generator=torch.Generator().manual_seed(91)
    evaluation=[halves(c,np.linspace(0,len(c)-1,48).astype(int),np.arange(128)) for c in counts]
    config=json.loads((old/'run_config.json').read_text())
    config['config']['objective_kwargs']={'objective_mode':'balanced_geometry','s2':s2,'ema_decay':.99,'eigenvalue_floor_ratio':.1}
    config['training_contract']['loss_weights']={'geometry':10,'use':10,'same':5,'step':5,'rollout':5}
    config['training_contract']['separation']='Original geometry plus k0 covariance utilization; shared stop-gradient EMA Mahalanobis metric across all scales'
    config['training_contract']['kernel_noncollapse']='Fixed C; no learnable kernel loss'
    config['training_contract']['utilization']='||Cov_i((Q(Da)+Q(Db))/2)/s2-I||F^2 / d; k0 only'
    config['training_contract']['same']='Equal-scale balanced subset consistency replaces old anchor/coarse terms'
    config['config']['threads']=2
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    history=[];start=time.monotonic()
    def evaluate(epoch):
        metric=balanced_metric(ema,s2)
        row={'epoch':epoch,'seconds':time.monotonic()-start}
        with torch.no_grad():
            for split,(ha,hb) in zip(('train','test'),evaluation):
                loss,terms,_=objective(model,ha,hb,patterns,marginal,metric,s2)
                row[split+'_total']=float(loss)
                row.update({split+'_'+k:float(v) for k,v in terms.items()})
            g=model.theory_encoder.head(cached['train'][:,0]@model.theory_encoder.local_mlp(patterns))
            eig=torch.linalg.eigvalsh(covariance(g))
            row['train_weak_variance_fraction']=float(eig[0]/eig.sum())
        history.append(row)
        (out/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(1,2001):
        ids=torch.randperm(len(counts[0]),generator=generator)[:32]
        samples=torch.randperm(256,generator=generator)
        ha,hb=halves(counts[0],ids.numpy(),samples.numpy())
        metric=balanced_metric(ema,s2)
        optimizer.zero_grad(set_to_none=True)
        loss,terms,cov=objective(model,ha,hb,patterns,marginal,metric,s2)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        ema=.99*ema+.01*cov
        if epoch%40==0:evaluate(epoch)
        if epoch%500==0:
            state={k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')}
            torch.save({'epoch':epoch,'model':state,'optimizer':optimizer.state_dict(),'ema_cov':ema,'s2':s2,
                        'batch_generator_state':generator.get_state()},out/f'training_step_{epoch}.pt')
    torch.save(state,out/'qcr_model.pt')
    config['history']=history
    (out/'summary.json').write_text(json.dumps(config,indent=2)+'\n')
    # Exact full-ensemble fixed-C forward evaluation, independent of half-split training outputs.
    olddata=np.load(old/'step2000/field_flow_data.npz')
    h=torch.cat((cached['train'],cached['test']))
    with torch.no_grad():
        direct=model.theory_encoder.head(h@model.theory_encoder.local_mlp(patterns))
        roll=[direct[:,0]]
        for k in range(1,4):roll.append(model.advance_theory(roll[-1]))
        rollout=torch.stack(roll,1)
    payload={k:olddata[k] for k in olddata.files if k not in ('direct','rollout','means')}
    y=olddata['KB'];ks=olddata['K_grid'];bs=olddata['B_grid']
    payload.update(direct=direct.numpy(),rollout=rollout.numpy(),
        means=np.array([[direct.numpy()[np.isclose(y[:,0],K)&np.isclose(y[:,1],B,atol=1e-12)].mean(0) for B in bs] for K in ks]))
    np.savez_compressed(out/'field_flow_data.npz',**payload)


if __name__=='__main__':
    main()
