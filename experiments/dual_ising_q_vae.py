"""Explicit microscopic Q-only VAE/TC diagnostic, not a default QCR recipe.

One observation is a physical ensemble. The encoder sees its complete joint
plaquette histogram, not coupling values. Cross-subset reconstruction is a
normalized statistical prediction objective, not a full-configuration ELBO.
No C/R modules are constructed, evaluated, or optimized.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from recurrent_rg.data.ensembles import load_ensembles


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def per_configuration_histograms(x):
    """Exact periodic q=4 plaquette counts, keeping the configuration axis."""
    s=x.argmax(-1)
    code=64*s+16*s.roll(-1,2)+4*s.roll(-1,3)+s.roll(-1,2).roll(-1,3)
    flat=code.reshape(-1,code.shape[-1]*code.shape[-2])
    offsets=torch.arange(len(flat))[:,None]*256
    counts=torch.bincount((flat+offsets).flatten(),minlength=len(flat)*256)
    return counts.reshape(*x.shape[:2],256).double()/flat.shape[1]


def cache_quarters(x, seed):
    gen=torch.Generator().manual_seed(seed); out=[]
    if x.shape[1]!=64 or x.shape[-1]!=4:raise ValueError('Expected 64 configurations, four local states')
    for start in range(0,len(x),8):
        p=per_configuration_histograms(x[start:start+8])
        for row in p:out.append(row[torch.randperm(64,generator=gen)].reshape(4,16,256).mean(1))
    return torch.stack(out)


class EnsembleVAE(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer('center',center.clone());self.register_buffer('scale',scale.clone())
        self.encoder=nn.Sequential(nn.Linear(256,96),nn.LeakyReLU(.1),nn.Linear(96,96),nn.LeakyReLU(.1))
        self.mean=nn.Linear(96,2);self.logvar=nn.Linear(96,2)
        self.decoder=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.1),nn.Linear(64,64),nn.LeakyReLU(.1),nn.Linear(64,256))
        self.double()
        with torch.no_grad():
            self.mean.weight.normal_(0,.02);self.mean.bias.zero_()
            self.logvar.weight.zero_();self.logvar.bias.fill_(-2.)
            self.decoder[-1].weight.normal_(0,.02)
            self.decoder[-1].bias.copy_(center.clamp_min(1e-6).log())

    def encode(self,p):
        h=self.encoder((p-self.center)/self.scale)
        return self.mean(h),self.logvar(h).clamp(-8,3)


def gaussian_kl(mu,lv):
    return .5*(mu.square()+lv.exp()-lv-1).sum(-1).mean()


def mixture_tc(z,mu,lv):
    """Exact finite-mixture densities, Monte Carlo expectation (may be negative).

    ALL TRAIN ensemble components participate; this is not an uncorrected
    minibatch density approximation or an adversarial FactorVAE estimator.
    Multiple views get total weight one per physical ensemble.
    """
    lp=-.5*((z[:,None]-mu[None]).square()*(-lv[None]).exp()+lv[None]+math.log(2*math.pi))
    log_q=torch.logsumexp(lp.sum(-1),1)-math.log(len(mu))
    log_product=(torch.logsumexp(lp,1)-math.log(len(mu))).sum(-1)
    return (log_q-log_product).mean()


def objective(model,a,b,eps,view_choice,kl_weight,tc_weight):
    mu,lv=model.encode(torch.cat([a,b]));z=mu+(lv*.5).exp()*eps
    target=torch.cat([b,a]).detach()
    rec=-(target*model.decoder(z).log_softmax(-1)).sum(-1).mean()
    kl=gaussian_kl(mu,lv)
    selected=torch.arange(len(a))+view_choice*len(a)
    tc=mixture_tc(z[selected],mu,lv) if tc_weight else rec.new_zeros(())
    loss=rec+kl_weight*kl+tc_weight*tc
    return loss,dict(cross_entropy=rec,KL=kl,TC=tc)


@torch.no_grad()
def evaluate(model,quarters):
    p=quarters.mean(1);mu,lv=model.encode(p);pred=model.decoder(mu).softmax(-1)
    ma,_=model.encode(quarters[:,:2].mean(1));mb,_=model.encode(quarters[:,2:].mean(1))
    variance=mu.var(0,unbiased=False)
    metrics=dict(sum_bin_mse=float((pred-p).square().sum(-1).mean()),
        mean_decode_cross_entropy=float(-(p*pred.clamp_min(1e-30).log()).sum(-1).mean()),
        KL=float(gaussian_kl(mu,lv)),mean_coordinate_variance=variance.tolist(),
        mean_posterior_variance=lv.exp().mean(0).tolist(),
        subset_mean_squared_difference=float((ma-mb).square().sum(-1).mean()),
        logvar_at_numeric_bounds_fraction=float(((lv<=-8)|(lv>=3)).double().mean()))
    return metrics,dict(g=mu.numpy(),logvar=lv.numpy(),predicted_histogram=pred.numpy(),target_histogram=p.numpy(),half_a=ma.numpy(),half_b=mb.numpy())


def train_arm(arm, initial, train_quarters, test_quarters, settings, output, started):
    """No coupling-label argument; fixed endpoint, no model selection."""
    model=copy.deepcopy(initial);optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    gen=torch.Generator().manual_seed(settings['seed']+1);history=[]
    for step in range(settings['steps']+1):
        if time.monotonic()-started>settings['wall_limit_seconds']:raise TimeoutError('Declared pilot wall limit; no automatic extension')
        if step:
            order=torch.randperm(4,generator=gen)
            a=train_quarters[:,order[:2]].mean(1);b=train_quarters[:,order[2:]].mean(1)
            eps=torch.randn(2*len(a),2,generator=gen,dtype=torch.double)
            choose=torch.randint(2,(len(a),),generator=gen)
            ramp=min(step/settings['warmup_steps'],1.)
            optimizer.zero_grad(set_to_none=True)
            loss,terms=objective(model,a,b,eps,choose,ramp*arm['KL_weight'],ramp*arm['TC_weight'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();optimizer.step()
        if step%settings['log_every']==0 or step==settings['steps']:
            row=dict(step=step,train=evaluate(model,train_quarters)[0],test=evaluate(model,test_quarters)[0])
            if step:row.update(sampled_objective=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},
                               effective_KL_weight=ramp*arm['KL_weight'],effective_TC_weight=ramp*arm['TC_weight'])
            history.append(row);write(output/'history.json',history)
            print(json.dumps(dict(arm=arm['name'],step=step,test_mse=row['test']['sum_bin_mse'],mean_variance=row['test']['mean_coordinate_variance'])),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),settings=settings,arm=arm),output/'checkpoint_final.pt')
    metrics,arrays={},{}
    for split,data in [('train',train_quarters),('test',test_quarters)]:
        metrics[split],values=evaluate(model,data)
        arrays.update({split+'_'+key:value for key,value in values.items()})
    # Posterior-sampling variability is not an active latent dimension.
    mu,lv=model.encode(train_quarters.mean(1))
    with torch.no_grad():
        gen_eval=torch.Generator().manual_seed(settings['seed']+5000)
        tc_values=[]
        for _ in range(16):
            z=mu+(lv*.5).exp()*torch.randn(mu.shape,generator=gen_eval,dtype=torch.double)
            tc_values.append(float(mixture_tc(z,mu,lv)))
    metrics['TRAIN_TC_MC']=dict(mean=float(np.mean(tc_values)),standard_error=float(np.std(tc_values,ddof=1)/4),draws=16,
        note='Monte Carlo error conditional on fixed physical ensembles and model; not physical-chain uncertainty')
    write(output/'metrics.json',metrics);np.savez_compressed(output/'evaluation_arrays.npz',**arrays)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text());root=Path(plan['output']);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(plan['seed'])
    start=time.monotonic();state=dict(status='preparing',workflow_complete=False,pid=os.getpid(),current=None,error=None)
    write(root/'job.json',state);write(root/'plan.json',plan)
    try:
        for rel,digest in plan['source_sha256'].items():
            if sha(rel)!=digest:raise ValueError('Source fingerprint mismatch: '+rel)
        if sha(plan['cache'])!=plan['data_sha256']:raise ValueError('Data fingerprint mismatch')
        train,test=load_ensembles(plan['cache'],mmap=True)
        tq=cache_quarters(train.configurations,9191);vq=cache_quarters(test.configurations,9192)
        p_train=tq.mean(1);center=p_train.mean(0);scale=p_train.std(0,unbiased=False).clamp_min(.001)
        initial=EnsembleVAE(center,scale)
        write(root/'manifest.json',dict(**plan,train_chains=len(tq),test_chains=len(vq),
            CPU_threads=1,no_C_or_R_instantiated=True,labels_used_for_training=False,
            input='Complete joint 2x2 categorical motif histogram, 256 bins; not the previous 3x3 local-feature Q',
            quarter_views='Four disjoint fixed random groups of 16 configurations per chain, randomly re-paired each step',
            posterior='diagonal Gaussian; g=posterior mean; log variance numerical bounds [-8,3]',
            decoder='2-64-64-256 LeakyReLU(.1), softmax; no spectral cap, no PCA initialization',
            TC='Exact finite Gaussian mixture over ALL TRAIN chain/view components; MC expectation; no discriminator',
            probability_likelihood='Normalized cross-subset histogram cross entropy; not full configuration likelihood',
            parameter_count=sum(p.numel() for p in initial.parameters()),
            initial_model_fingerprint=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in initial.state_dict().values())).hexdigest()))
        for arm in plan['arms']:
            out=root/arm['name'];out.mkdir();state.update(status='training',current=arm['name']);write(root/'job.json',state)
            train_arm(arm,initial,tq,vq,plan,out,start)
        # Labels enter only this post-training report. No selection or refitting of Q.
        state.update(status='reporting',current='report');write(root/'job.json',state)
        from scripts.report_dual_ising_q_vae import report
        report(root,train.couplings.numpy(),test.couplings.numpy())
        state.update(status='complete',workflow_complete=True,current=None,elapsed_seconds=time.monotonic()-start,
            public_files=[dict(path=str(f.relative_to(root)),sha256=sha(f)) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix in ('.json','.png','.npz') and f.name!='job.json'])
        write(root/'job.json',state)
    except Exception as exc:
        state.update(status='budget_hold' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc),requires_user_action=True,
                     elapsed_seconds=time.monotonic()-start);write(root/'job.json',state);raise


if __name__=='__main__':main()
