"""Paired loss ablation from an expanded Q; fixed decimation, same minibatches."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run,qcr_state_dict,build_model
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.plotting.reports import plot_diagnostics
from experiments.ising2d_q_only import statistics,encode,readout


def components(model,hist,patterns,partition,geometry_weight=10.,isotropy_weight=10.):
    a,b=partition[:2],partition[2:]
    ga,gb=encode(model,hist[:,a].mean(1)),encode(model,hist[:,b].mean(1))
    g=.5*(ga+gb);err=(ga-gb).square().sum(-1).mean(0)
    geo=torch.stack([distribution_geometry_loss(g[:,k],patterns[:,a,k].mean(1),patterns[:,b,k].mean(1))[0] for k in range(3)]).mean()
    c=g[:,0]-g[:,0].mean(0);cov=c.T@c/len(c)
    one=model.advance_theory(g[:,:-1]);r=model.advance_theory(g[:,0]);r2=model.advance_theory(r)
    return dict(split=20*err[0]+5*err[1:].mean(),geometry=geometry_weight*geo,
        isotropy=isotropy_weight*.5*(cov-torch.eye(2)).square().sum(),
        step=5*(one-g[:,1:]).square().sum(-1).mean(),
        rollout=5*(torch.stack((r,r2),1)-g[:,1:]).square().sum(-1).mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--warm-r',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--initialization',choices=('expanded','random'),default='expanded')
    p.add_argument('--geometry-weight',type=float,default=10.)
    p.add_argument('--isotropy-weight',type=float,default=10.)
    p.add_argument('--arm',default=None,help='Run only the named arm')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(91);start=time.perf_counter()
    loaded=load_run(args.source/'q_only.pt');base=loaded.model
    source=json.loads((args.source/'report.json').read_text());tx,ty,vx,vy=torch.load(source['cache'],weights_only=True)
    tr=statistics(base,tx);te=statistics(base,vx)
    cached=np.load(args.source/'encodings.npz')
    with torch.no_grad():torch.testing.assert_close(encode(base,tr[0].mean(1)),torch.from_numpy(cached['train']),atol=1e-12,rtol=1e-10)
    warm=load_run(args.warm_r/'qcr_model.pt').model
    for n,v in base.theory_encoder.state_dict().items():assert torch.equal(v,warm.theory_encoder.state_dict()[n])
    if args.initialization=='random':
        torch.manual_seed(91)
        base=build_model(loaded.model_config,2)
    arms=[]
    for iso in (True,False):
        for closure in ('none','step','rollout','both'):
            arms.append(dict(name=f'{"iso" if iso else "noiso"}_{closure}',iso=iso,closure=closure,split=True,warm=False))
    arms += [dict(name='noiso_nosplit_none',iso=False,closure='none',split=False,warm=False),
             dict(name='noiso_both_warmR',iso=False,closure='both',split=True,warm=True)]
    if args.initialization=='random':
        arms=[a for a in arms if a['name'] in ('iso_none','iso_both','noiso_none','noiso_both')]
    if args.arm is not None:
        arms=[a for a in arms if a['name']==args.arm]
        if not arms:raise ValueError('Unknown arm for selected initialization')
    if args.geometry_weight<=0:raise ValueError('geometry-weight must be positive')
    if args.isotropy_weight<0:raise ValueError('isotropy-weight must be nonnegative')
    report=dict(seed=91,steps=args.steps,source=str(args.source.resolve()),cache=source['cache'],
        protocol='All arms start from identical expanded Q; C fixed positive decimation. Adam .001 fresh state, batch64, identical seed92 minibatch and quarter-partition sequence, 2000 updates by default. R initialized identically from original Q-only checkpoint except explicitly warmR arm. No coupling supervision. Closure targets receive gradients. Labels only for posthoc readouts. One seed; continuation stability, not training from scratch.',arms={})
    report['initialization']=args.initialization
    report['geometry_weight']=args.geometry_weight
    report['isotropy_weight']=args.isotropy_weight
    if args.initialization=='random':
        report['protocol']='Paired from-scratch seed91 Q/R; fixed positive decimation. Adam .001, fresh optimizer, same seed92 batch64/minibatch and sample-partition sequence in all four arms. Covariance on/off crossed with closure none/both. No labels in loss. One seed.'
    (args.output/'protocol.json').write_text(json.dumps(dict(report,arms=arms),indent=2)+'\n')
    for arm in arms:
        model=copy.deepcopy(warm if arm['warm'] else base)
        for par in model.parameters():par.requires_grad_(False)
        modules=[model.theory_encoder]+([model.theory_rg] if arm['closure']!='none' else [])
        params=[par for module in modules for par in module.parameters()]
        for par in params:par.requires_grad_(True)
        def active(t):
            loss=t['geometry']
            if arm['split']:loss=loss+t['split']
            if arm['iso']:loss=loss+t['isotropy']
            for term in ('step','rollout'):
                if arm['closure'] in (term,'both'):loss=loss+t[term]
            return loss
        def evaluate(data):
            with torch.no_grad():
                terms=components(model,*data[:2],torch.arange(4),geometry_weight=args.geometry_weight,isotropy_weight=args.isotropy_weight)
                g=encode(model,data[0].mean(1));eigs=torch.stack([torch.linalg.eigvalsh(torch.cov(g[:,k].T,correction=0)) for k in range(3)])
            return dict(loss=float(active(terms)),components={k:float(v) for k,v in terms.items()},eigenvalues=eigs.tolist(),weak_fraction=(eigs[:,0]/eigs.sum(-1)).tolist())
        # Instantaneous gradient diagnostic at this common starting point:
        # directional change of full-batch microscopic weak covariance eigenvalue.
        terms=components(model,*tr[:2],torch.arange(4),geometry_weight=args.geometry_weight,isotropy_weight=args.isotropy_weight);g=encode(model,tr[0].mean(1))[:,0]
        weak=torch.linalg.eigvalsh(torch.cov(g.T,correction=0))[0]
        qparams=list(model.theory_encoder.parameters());gradweak=torch.autograd.grad(weak,qparams)
        directional={}
        for name,term in terms.items():
            grad=torch.autograd.grad(term,qparams,retain_graph=True)
            directional[name]=-sum(float((a*b).sum()) for a,b in zip(gradweak,grad))
        opt=torch.optim.Adam(params,lr=.001);gen=torch.Generator().manual_seed(92);history=[]
        for step in range(args.steps+1):
            if step:
                idx=torch.randperm(len(tx),generator=gen)[:64];partition=torch.randperm(4,generator=gen)
                opt.zero_grad();loss=active(components(model,tr[0][idx],tr[1][idx],partition,geometry_weight=args.geometry_weight,isotropy_weight=args.isotropy_weight));assert torch.isfinite(loss);loss.backward();opt.step()
            if step%200==0 or step==args.steps:
                row=dict(step=step,train=evaluate(tr),test=evaluate(te));history.append(row)
                print(arm['name'],step,'test weak fractions',row['test']['weak_fraction'],'loss',row['test']['loss'],flush=True)
        out=args.output/arm['name'];out.mkdir()
        with torch.no_grad():
            gt=encode(model,tr[0].mean(1)).numpy();gv=encode(model,te[0].mean(1)).numpy()
            roll=[torch.from_numpy(gv[:,0])]
            for _ in range(2):roll.append(model.advance_theory(roll[-1]))
            rollout=torch.stack(roll,1).numpy()
        readouts=[readout(gt[:,k],gv[:,k],ty.numpy(),vy.numpy())[0] for k in range(3)]
        for n,v in base.state_dict().items():
            if not n.startswith(('theory_encoder.','theory_rg.')):assert torch.equal(v,model.state_dict()[n]),n
        record=dict(settings=arm,history=history,heldout_quadratic_K_B_R2=readouts,initial_weak_eigenvalue_gradient_descent_direction=directional,
                    gradient_note='-grad(lambda_min) dot grad(weighted loss term), Q parameters only, full training data, raw infinitesimal SGD direction, not actual Adam trajectory.')
        report['arms'][arm['name']]=record
        torch.save(dict(model=qcr_state_dict(model),metadata=loaded.payload['metadata']),out/'qcr_model.pt')
        np.savez(out/'encodings.npz',train=gt,test=gv,train_labels=ty.numpy(),test_labels=vy.numpy())
        if arm['closure']!='none':plot_diagnostics(model,dict(direct=gv,rollout=rollout),vy.numpy(),('K','B'),out,training_inputs=gt[:,:-1])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for name,r in report['arms'].items():
        h=r['history'];x=[v['step'] for v in h]
        axs[0].plot(x,[v['test']['weak_fraction'][0]*100 for v in h],label=name)
        axs[1].plot(x,[v['test']['eigenvalues'][0][0] for v in h],label=name)
    axs[0].set(xlabel='updates',ylabel='weak variance fraction (%)',yscale='log');axs[1].set(xlabel='updates',ylabel='weak covariance eigenvalue',yscale='log')
    axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/'collapse_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(18,7))
    for ax,arm in zip(axs.flat,arms):
        z=np.load(args.output/arm['name']/'encodings.npz');gv=z['test'][:,0]
        ax.scatter(*gv.T,c=vy[:,0],cmap='viridis',s=8,vmin=.2,vmax=.68)
        ax.set(title=arm['name'],xlabel='g1',ylabel='g2',aspect='equal')
    for ax in list(axs.flat)[len(arms):]:ax.axis('off')
    fig.suptitle('Held-out microscopic Q, colored by K (same color range)');fig.tight_layout();fig.savefig(args.output/'latent_comparison.png',dpi=180);plt.close(fig)
    report['seconds']=time.perf_counter()-start
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',report['seconds'],flush=True)


if __name__=='__main__':main()
