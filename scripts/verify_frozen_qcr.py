"""Freeze Q/C: controlled R-only optimization, capacity and state-collision audit."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def predictions(net,g):
    z=g[:,0:1]
    seq=[z]
    for _ in range(g.shape[1]-1):
        z=z+net(z)
        seq.append(z)
    return torch.cat(seq,1)


def objective(net,g):
    one=(g[:,:-1]+net(g[:,:-1,None]).squeeze(-1)-g[:,1:]).square().mean()
    multi=(predictions(net,g)[:,1:]-g[:,1:]).square().mean()
    return 5*(one+multi)


def metrics(net,g):
    with torch.no_grad():
        p=predictions(net,g)
        rmse=(p-g).square().mean(0).sqrt()
        one=(g[:,:-1]+net(g[:,:-1,None]).squeeze(-1)-g[:,1:]).square().mean(0).sqrt()
    return dict(objective=float(objective(net,g)),rmse=rmse.tolist(),
                nrmse=(rmse/g.std(0,unbiased=False)).tolist(),one_step_rmse=one.tolist())


def collisions(g,h):
    # Select nearest cross-scale inputs using two replicas; evaluate on third.
    hs=np.unique(h)
    replicas=np.stack([g[h==x] for x in hs])
    assert replicas.shape[1]==3
    result={}
    for tolerance in [.005,.01,.02]:
        records=[]
        for held in range(3):
            avg=np.mean(np.delete(replicas,held,axis=1),axis=1)
            for k in range(3):
                for l in range(k+1,3):
                    for i in range(len(hs)):
                        candidates=np.where(np.arange(len(hs))!=i)[0]
                        j=candidates[np.argmin(abs(avg[candidates,l]-avg[i,k]))]
                        if abs(avg[i,k]-avg[j,l])>tolerance: continue
                        a,b=replicas[i,held],replicas[j,held]
                        records.append(dict(held=held,h_a=float(hs[i]),h_b=float(hs[j]),k=k,l=l,
                                            selection_gap=float(abs(avg[i,k]-avg[j,l])),
                                            held_input_gap=float(abs(a[k]-b[l])),
                                            held_target_gap=float(abs(a[k+1]-b[l+1]))))
        validated=[r for r in records if r['held_input_gap']<=tolerance]
        result[str(tolerance)]=dict(selected=len(records),held_validated=len(validated),
            median_target_gap=float(np.median([r['held_target_gap'] for r in validated])) if validated else None,
            p90_target_gap=float(np.quantile([r['held_target_gap'] for r in validated],.9)) if validated else None,
            largest_examples=sorted(validated,key=lambda r:r['held_target_gap'],reverse=True)[:5])
    return result


def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    source=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_rollout5_2000_seed91'
    out=root/'results/tfim_frozen_qc_R_audit_seed91'
    out.mkdir(exist_ok=True)
    state=torch.load(source/'training_step_1000.pt',weights_only=True)['model']
    model=RecurrentRGNetwork1D(levels=4,hidden=60,theory_dim=1).double().eval()
    inc=model.load_state_dict(state,strict=False)
    assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
    for p in model.parameters(): p.requires_grad_(False)
    model.theory_encoder.binary_lookup=True
    model.rg_kernel.binary_lookup=True
    tx,th,vx,vh=torch.load(root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    def encode(x):
        rows=[]
        with torch.no_grad():
            for chunk in x.split(8):
                level=balanced_views(chunk,128)
                gs=[]
                for k in range(6):
                    gs.append(model.theory_encoder(level)[:,0])
                    if k<5: level=model.rg_kernel(level)
                rows.append(torch.stack(gs,1))
        return torch.cat(rows)
    tr,te=encode(tx),encode(vx)
    torch.save(dict(train=tr,test=te,train_h=th,test_h=vh),out/'fixed_encodings.pt')
    report=dict(source=str(source/'training_step_1000.pt'),
        protocol='Q/C frozen; fixed full-ensemble encodings; train original train split only; one-step and rollout weights 5; supervised k<=3; no test checkpoint selection; Adam cosine lr .001 to .00001, 3000 full-batch updates; seed91.',
        baseline=dict(train=metrics(model.theory_rg,tr[:,:4]),test=metrics(model.theory_rg,te[:,:4])),
        collisions=collisions(torch.cat([tr,te]).numpy(),torch.cat([th,vh]).numpy()),runs={})
    nets={'original':copy.deepcopy(model.theory_rg)}
    for label,width,warm in [('same60_warm',60,True),('same60_fresh',60,False),('wide128_fresh',128,False)]:
        torch.manual_seed(91)
        net=(copy.deepcopy(model.theory_rg) if warm else nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1)).double())
        for p in net.parameters(): p.requires_grad_(True)
        opt=torch.optim.Adam(net.parameters(),lr=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,3000,eta_min=.00001)
        history=[]
        for step in range(1,3001):
            opt.zero_grad()
            loss=objective(net,tr[:,:4])
            loss.backward()
            opt.step()
            schedule.step()
            if step==1 or step%100==0:
                history.append(dict(step=step,train=float(objective(net,tr[:,:4])),test=float(objective(net,te[:,:4]))))
            if step%1000==0: print(label,step,history[-1],flush=True)
        nets[label]=net.eval()
        report['runs'][label]=dict(parameters=sum(p.numel() for p in net.parameters()),history=history,
            train=metrics(net,tr[:,:4]),test=metrics(net,te[:,:4]),
            test_extrapolation=metrics(net,te),state_file=label+'.pt')
        torch.save(net.state_dict(),out/(label+'.pt'))
    # Check Q and C are untouched, not merely detached targets.
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for label,run in report['runs'].items():
        ax[0,0].semilogy([r['step'] for r in run['history']],[r['train'] for r in run['history']],label=label)
    for label,net in nets.items():
        m=metrics(net,te[:,:4])
        ax[0,1].plot([1,2,3],m['nrmse'][1:],'o-',label=label)
    for k in range(3):
        ax[1,0].scatter(tr[:,k],tr[:,k+1],s=10,alpha=.5,label=f'train k={k}')
    grid=torch.linspace(float(tr[:,:3].min()),float(tr[:,:3].max()),500)[:,None].double()
    with torch.no_grad():
        ax[1,0].plot(grid[:,0],(grid+nets['wide128_fresh'](grid))[:,0],'k-',label='wide R fit')
        for label,net in nets.items():
            x=torch.cat([tr,te]); h=torch.cat([th,vh])
            for j,target in enumerate([.95,1.05]):
                val=h[(h-target).abs().argmin()]
                actual=x[h==val].mean(0)
                pred=predictions(net,x[h==val]).mean(0)
                if label=='original': ax[1,1].plot(range(6),actual,'o-',color=f'C{j}',label=f'Q, h/J={float(val):.3f}')
                if label in ['original','wide128_fresh']:
                    ax[1,1].plot(range(6),pred,':' if label=='original' else '--',color=f'C{j}',label=f'{label}, {float(val):.3f}')
    ax[0,0].set(title='R-only training objective',xlabel='Update',ylabel='5(one-step + multistep MSE)')
    ax[0,1].set(title='Full-test rollout NRMSE',xlabel='Horizon k',ylabel='RMSE / frozen-Q std',xticks=[1,2,3])
    ax[1,0].set(title='Frozen input/target pairs across scales',xlabel='g_k',ylabel='g_(k+1)')
    ax[1,1].set(title='High-scale check (no supervision beyond k=3)',xlabel='k',ylabel='g')
    ax[1,1].axvspan(3.05,5.1,color='gray',alpha=.1)
    for a in ax.flat: a.grid(alpha=.2); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'frozen_R_audit.png',dpi=180)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{s:v[s] for s in ['parameters','train','test']} for k,v in report['runs'].items()},indent=2))
    print('COLLISIONS',json.dumps(report['collisions'],indent=2))


if __name__=='__main__': main()
