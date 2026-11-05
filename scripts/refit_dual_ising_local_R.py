"""Matched-budget local R experiment; fixed Q/C, no coupling-label training."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run,qcr_state_dict
from recurrent_rg.local_flow import LocalRBFDisplacement
from recurrent_rg.evaluation import closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics
from refit_dual_ising_R import objective,trajectory


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(91)
    source=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
    prior=source/'R_only_refit_6000';out=source/'R_local32_refit_6000';out.mkdir(exist_ok=False)
    targets=torch.load(prior/'fixed_targets.pt',weights_only=True)
    train,test=targets['train'],targets['test']
    base=load_run(source).model.eval();mlp=copy.deepcopy(base)
    mlp.load_state_dict(torch.load(prior/'qcr_model.pt',weights_only=True)['model'],strict=False)
    model=copy.deepcopy(base)
    model.theory_rg=LocalRBFDisplacement.from_training_inputs(base.theory_rg,train[:,:-1],32)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    torch.testing.assert_close(model.advance_theory(train),base.advance_theory(train),rtol=0,atol=0)
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6000,eta_min=1e-5)
    history=[]
    for step in range(6001):
        if step:
            opt.zero_grad();loss=objective(model,train);assert torch.isfinite(loss)
            loss.backward();opt.step();scheduler.step()
        if step%200==0:
            with torch.no_grad():row=dict(step=step,train=float(objective(model,train)),test=float(objective(model,test)))
            history.append(row);print(row,flush=True)
    for name,value in base.state_dict().items():
        if not name.startswith('theory_rg.'):assert torch.equal(value,model.state_dict()[name]),name
    payload=dict(model=qcr_state_dict(model),architecture='LocalRBFDisplacement',centers=32,
        source=str(source.resolve()),extra_R_updates=6000,
        loading='Build source model; replace theory_rg with LocalRBFDisplacement.from_training_inputs(source.theory_rg, fixed_targets train[:,:-1], 32); load model state.')
    torch.save(payload,out/'qcr_model.pt')
    audit=np.load(source/'critical_sampling_audit'/'trajectories.npz')
    ad=torch.from_numpy(audit['direct'].reshape(-1,3,2));kc=np.log(1+np.sqrt(2))/2
    center=np.flatnonzero(np.isclose(audit['couplings'],kc).all(1))[0]
    report=dict(settings='Frozen Q/C; same cached 128-sample targets, coupling-cell split, seed91, 6000 Adam steps and cosine schedule, original Euclidean 5 step + 5 rollout loss. Local model starts from ORIGINAL R, not the refitted MLP. 32 fixed farthest-point centers from training inputs; fixed widths; zero-initialized affine corrections; global network trainable. Final iterate, no validation selection.',
        parameters=dict(mlp=sum(p.numel() for p in mlp.theory_rg.parameters()),local=sum(p.numel() for p in model.theory_rg.parameters())),history=history)
    for name,m in [('mlp',mlp),('local',model)]:
        report[name]={split:closure_metrics(trajectory(m,d),reference_direct=train.numpy()) for split,d in [('train',train),('test',test),('fresh_audit',ad)]}
        direct=audit['direct'][center];roll=trajectory(m,torch.from_numpy(direct))['rollout']
        report[name]['double_critical_error']=np.linalg.norm((roll-direct).mean(0),axis=1).tolist()
        # Noise-resolved directions: target displacement > 3 SE across independent chains.
        obs=audit['direct'][:,:,1:]-audit['direct'][:,:,:-1]
        actual=obs.mean(1);noise=np.linalg.norm(obs.std(1,ddof=1)/np.sqrt(obs.shape[1]),axis=-1)
        pred=trajectory(m,ad)['one_step'].reshape(len(audit['couplings']),4,2,2)-audit['direct'][:,:,:-1]
        predicted=pred.mean(1);norm=np.linalg.norm(actual,axis=-1);pnorm=np.linalg.norm(predicted,axis=-1)
        mask=norm>3*noise
        cosine=(predicted*actual).sum(-1)/np.maximum(norm*pnorm,1e-15)
        report[name]['resolved_direction']=dict(count=int(mask.sum()),total=int(mask.size),median_angle_degrees=float(np.median(np.degrees(np.arccos(np.clip(cosine[mask],-1,1))))),opposite_fraction=float((cosine[mask]<0).mean()))
        d=trajectory(m,test);sel=(targets['test_couplings'].numpy()>=.7).all(1)
        report[name]['deep_ordered']=dict(count=int(sel.sum()),direct_step_norm_mean=float(np.linalg.norm(d['direct'][sel,1:]-d['direct'][sel,:-1],axis=-1).mean()),predicted_step_norm_mean=float(np.linalg.norm(d['one_step'][sel]-d['direct'][sel,:-1],axis=-1).mean()),step_error_norm_mean=float(np.linalg.norm(d['one_step'][sel]-d['direct'][sel,1:],axis=-1).mean()))
    plot_diagnostics(model,trajectory(model,test),targets['test_couplings'].numpy(),['K1','K2'],out/'flow',training_inputs=train[:,:-1].numpy())
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for axis,ax in enumerate(axs):
        for offset,color in zip([-.03,0,.03],['#0072b2','black','#d55e00']):
            cell=np.array([kc,kc]);cell[axis]+=offset
            idx=np.flatnonzero(np.isclose(audit['couplings'],cell).all(1))[0];d=audit['direct'][idx]
            ax.plot(*d.mean(0).T,'o-',c=color,label=f'K{axis+1}=Kc{offset:+.2f}: Q')
            for name,m,style in [('MLP',mlp,':'),('local R',model,'--')]:
                r=trajectory(m,torch.from_numpy(d))['rollout'].mean(0)
                ax.plot(*r.T,style,c=color,marker='x',label=name if offset==0 else None)
        ax.set(xlabel='g1',ylabel='g2',title=f'Other layer at Kc; vary K{axis+1}');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'critical_comparison.png',dpi=180)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    old=json.loads((prior/'report.json').read_text())['history']
    for name,h in [('MLP',old),('local',history)]:
        for split,style in [('train','-'),('test','--')]:ax.semilogy([v['step'] for v in h],[v[split] for v in h],style,label=name+' '+split)
    ax.set(xlabel='R-only updates',ylabel='5 one-step + 5 rollout loss');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'loss_comparison.png',dpi=180)
    report['elapsed_seconds']=time.perf_counter()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)


if __name__=='__main__':main()
