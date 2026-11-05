"""Separate data-coverage and noise-floor relative-displacement ablations."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run,qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics
from refit_dual_ising_R import objective,trajectory,predictions
from dual_ising_followup_data import SOURCE,OUT


def evaluate(model,train,test,audit,center):
    result={name:closure_metrics(trajectory(model,d),reference_direct=train.numpy()) for name,d in [('test',test),('critical_audit',audit)]}
    t=trajectory(model,audit)
    errors=(t['rollout']-t['direct']).reshape(25,4,3,2)
    result['double_critical_error']=np.linalg.norm(errors[center].mean(0),axis=1).tolist()
    result['common_train_loss']=float(objective(model,train).detach())
    result['common_test_loss']=float(objective(model,test).detach())
    return result


def main():
    torch.set_num_threads(2);torch.manual_seed(91);OUT.mkdir(exist_ok=True)
    base=load_run(SOURCE).model.eval()
    saved=torch.load(SOURCE/'R_only_refit_6000'/'fixed_targets.pt',weights_only=True)
    train,test=saved['train'],saved['test']
    audit_np=np.load(SOURCE/'critical_sampling_audit'/'trajectories.npz')
    audit=torch.from_numpy(audit_np['direct'].reshape(-1,3,2))
    kc=np.log(1+np.sqrt(2))/2;center=np.flatnonzero(np.isclose(audit_np['couplings'],kc).all(1))[0]
    # Noise for a 128-sample increment estimated from disjoint 64-sample halves.
    noise_path=OUT/'training_increment_noise.pt'
    if noise_path.exists():noise=torch.load(noise_path,weights_only=True)
    else:
        tr,_=load_ensembles(load_run(SOURCE).config['data']['cache'])
        x=tr.configurations.reshape(-1,64,*tr.configurations.shape[2:])
        d=collect_trajectories(base,x,samples=64,max_rg_steps=2,chunk_size=4)['direct'].reshape(len(train),2,3,2)
        increments=np.diff(d,axis=2)
        noise=torch.from_numpy(.25*((increments[:,0]-increments[:,1])**2).sum(-1))
        torch.save(noise,noise_path)
    epsilon=.01*train[:,1:].std().item()
    denom=(train[:,1:]-train[:,:-1]).square().sum(-1)+noise+epsilon**2
    weight=denom.reciprocal();weight=(weight/weight.mean()).clamp(max=10);weight=weight/weight.mean()
    report=dict(settings='6000 Adam .001 cosine to 1e-5 updates from same ORIGINAL pretrained MLP; unchanged Q/C; final iterate; no audit labels or trajectories used in optimization.',
        relative_loss='Original loss + 5*mean(w*one-step squared norm). w=1/(increment squared norm + split-half increment noise variance + epsilon^2), divided by mean, capped at 10, then renormalized. Fixed detached weights; mean one. The cap applies before final normalization.',epsilon=epsilon,weight_range=[float(weight.min()),float(weight.max())])
    models={}
    baseline=copy.deepcopy(base);baseline.load_state_dict(torch.load(SOURCE/'R_only_refit_6000'/'qcr_model.pt',weights_only=True)['model'],strict=False)
    models['baseline']=baseline;report['baseline']=evaluate(baseline,train,test,audit,center)
    for name in ['relative_loss','dense_data']:
        dest=OUT/name;dest.mkdir(exist_ok=True)
        if name=='dense_data':
            new=np.load(OUT/'dense_encodings.npz');extra=torch.from_numpy(new['direct'].reshape(-1,3,2))
            fit=torch.cat([train,extra]);report['dense_count']=len(extra)
        else:fit=train
        model=copy.deepcopy(base)
        for p in model.parameters():p.requires_grad_(False)
        for p in model.theory_rg.parameters():p.requires_grad_(True)
        opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6000,eta_min=1e-5)
        history=[]
        for step in range(6001):
            if step:
                opt.zero_grad();loss=objective(model,fit)
                if name=='relative_loss':
                    one,_=predictions(model,fit)
                    loss=loss+5*(weight*(one-fit[:,1:]).square().sum(-1)).mean()
                assert torch.isfinite(loss);loss.backward();opt.step();scheduler.step()
            if step%400==0:
                with torch.no_grad():h=dict(step=step,fit_common_loss=float(objective(model,fit)),test_common_loss=float(objective(model,test)))
                history.append(h);print(name,h,flush=True)
        for key,value in base.state_dict().items():
            if not key.startswith('theory_rg.'):assert torch.equal(value,model.state_dict()[key]),key
        torch.save(dict(model=qcr_state_dict(model),source=str(SOURCE.resolve()),ablation=name),dest/'qcr_model.pt')
        result=evaluate(model,train,test,audit,center);result['history']=history
        report[name]=result;models[name]=model
        plot_diagnostics(model,trajectory(model,test),saved['test_couplings'].numpy(),['K1','K2'],dest/'flow',training_inputs=fit[:,:-1].numpy())
        (OUT/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for axis,ax in enumerate(axs):
        for offset,color in zip([-.03,0,.03],['#0072b2','black','#d55e00']):
            point=np.array([kc,kc]);point[axis]+=offset
            idx=np.flatnonzero(np.isclose(audit_np['couplings'],point).all(1))[0]
            d=torch.from_numpy(audit_np['direct'][idx]);ax.plot(*d.mean(0).numpy().T,'o-',c=color,label=f'Q: offset {offset:+.2f}')
            for name,style in [('baseline',':'),('relative_loss','--'),('dense_data','-.')]:
                r=trajectory(models[name],d)['rollout'].mean(0)
                ax.plot(*r.T,style,c=color,marker='x',label=name if offset==0 else None)
        ax.set(xlabel='g1',ylabel='g2',title=f'Vary K{axis+1}; other=Kc');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'fit_critical_comparison.png',dpi=180)
    print(json.dumps({k:v for k,v in report.items() if k not in models},indent=2),flush=True)


if __name__=='__main__':main()
