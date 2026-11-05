"""Refit the existing R with frozen Q/C and disjoint coupling-cell evaluation."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics


def predictions(model, direct):
    z = direct[:,0]
    roll = [z]
    for _ in range(direct.shape[1]-1):
        z = model.advance_theory(z); roll.append(z)
    return model.advance_theory(direct[:,:-1]), torch.stack(roll,1)


def objective(model, direct):
    one, roll = predictions(model,direct)
    return 5*((one-direct[:,1:]).square().sum(-1).mean() +
              (roll[:,1:]-direct[:,1:]).square().sum(-1).mean())


def trajectory(model, direct):
    with torch.no_grad():
        one,roll = predictions(model,direct)
    return dict(direct=direct.numpy(),rollout=roll.numpy(),one_step=one.numpy())


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(91)
    source=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
    out=source/'R_only_refit_6000';out.mkdir(exist_ok=False)
    run=load_run(source);base=run.model.eval();model=copy.deepcopy(base)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    tr,te=load_ensembles(run.config['data']['cache'])
    targets=[]
    for name,data in [('train',tr),('test',te)]:
        t=collect_trajectories(base,data.configurations,samples=128,max_rg_steps=2)
        targets.append(torch.from_numpy(t['direct']))
        print('encoded',name,len(data.configurations),flush=True)
    train,test=targets
    torch.save(dict(train=train,test=test,train_couplings=tr.couplings,test_couplings=te.couplings),out/'fixed_targets.pt')
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6000,eta_min=1e-5)
    history=[]
    def log(step):
        with torch.no_grad():row=dict(step=step,train=float(objective(model,train)),test=float(objective(model,test)))
        history.append(row);print(row,flush=True)
    log(0)
    for step in range(1,6001):
        opt.zero_grad();loss=objective(model,train)
        assert torch.isfinite(loss)
        loss.backward();opt.step();scheduler.step()
        if step%200==0:log(step)
    for name,value in base.state_dict().items():
        if not name.startswith('theory_rg.'):
            assert torch.equal(value,model.state_dict()[name]),name
    torch.save(dict(model=qcr_state_dict(model),source=str(source.resolve()),extra_R_updates=6000),out/'qcr_model.pt')
    report=dict(source=str(source.resolve()),R_parameters=sum(p.numel() for p in model.theory_rg.parameters()),
        settings='Original pretrained R, unchanged architecture; Q/C frozen bitwise; 6000 full-batch Adam updates, lr .001 cosine to .00001. Original 5*one-step + 5*rollout squared Euclidean loss, coordinate sum. Cached 128-sample ensemble encodings rather than stochastic 64-sample batches. Original disjoint coupling-cell split; no labels enter fitting; final checkpoint, no test-based selection.',history=history)
    for name,m in [('before',base),('after',model)]:
        report[name]={}
        for split,d in [('train',train),('test',test)]:
            t=trajectory(m,d)
            report[name][split]=closure_metrics(t,reference_direct=train.numpy())
        plot_diagnostics(m,trajectory(m,test),te.couplings.numpy(),['K1','K2'],out/name,training_inputs=train[:,:-1].numpy())
    audit=np.load(source/'critical_sampling_audit'/'trajectories.npz')
    ad=torch.from_numpy(audit['direct'].reshape(-1,3,2))
    # Audit targets are chain-mean Q values, consistent for before/after comparison.
    labels=audit['couplings'];critical=np.log(1+np.sqrt(2))/2
    center=np.flatnonzero(np.isclose(labels,critical).all(1))[0]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,axis in zip(axs,range(2)):
        for offset,color in zip([-.03,0,.03],['#0072b2','black','#d55e00']):
            cell=np.array([critical,critical]);cell[axis]+=offset
            idx=np.flatnonzero(np.isclose(labels,cell).all(1))[0]
            d=audit['direct'][idx];mean=d.mean(0)
            ax.plot(*mean.T,'o-',c=color,label=f'K{axis+1}=Kc{offset:+.2f}: direct')
            for name,m,style in [('before',base,':'),('after',model,'--')]:
                t=trajectory(m,torch.from_numpy(d));r=t['rollout'].mean(0)
                ax.plot(*r.T,style,c=color,marker='x',label=name if offset==0 else None)
        ax.set(title=f'Vary K{axis+1}; other layer at Kc',xlabel='g1',ylabel='g2');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'critical_comparison.png',dpi=180)
    for name,m in [('before',base),('after',model)]:
        t=trajectory(m,ad)
        report[name]['fresh_critical_audit']=closure_metrics(t,reference_direct=train.numpy())
        direct=audit['direct'][center]
        r=trajectory(m,torch.from_numpy(direct))['rollout']
        report[name]['double_critical']=dict(direct_mean=direct.mean(0).tolist(),rollout_mean=r.mean(0).tolist(),
            error_of_mean=np.linalg.norm((r-direct).mean(0),axis=1).tolist())
        t=trajectory(m,torch.cat([train,test]));ys=torch.cat([tr.couplings,te.couplings]).numpy()
        sel=(ys>critical).all(1)
        report[name]['ordered_region']=dict(direct_mean=t['direct'][sel].mean(0).tolist(),rollout_mean=t['rollout'][sel].mean(0).tolist())
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for key in ['train','test']:ax.semilogy([h['step'] for h in history],[h[key] for h in history],label=key)
    ax.set(xlabel='Additional R-only updates',ylabel='5 one-step + 5 rollout loss');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'loss.png',dpi=180)
    report['elapsed_seconds']=time.perf_counter()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)


if __name__=='__main__':main()
