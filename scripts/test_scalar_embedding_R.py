"""Controlled R-only dimension lift: fixed scalar Q/C targets, no label loss."""
import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._utils import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from fit_frozen_qcr_R import errors, metrics


class Flow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 60), nn.Tanh(),
                                 nn.Linear(60, 60), nn.Tanh(), nn.Linear(60, dim))

    def advance_theory(self, x):
        return x + self.net(x)


def lift(scalar, exact=False):
    result = Flow(2).double()
    with torch.no_grad():
        result.net[0].weight[:, :1].copy_(scalar.net[0].weight)
        result.net[0].bias.copy_(scalar.net[0].bias)
        result.net[2].load_state_dict(scalar.net[2].state_dict())
        result.net[4].weight[:1].copy_(scalar.net[4].weight)
        result.net[4].bias[:1].copy_(scalar.net[4].bias)
        if exact:
            result.net[0].weight[:, 1].zero_()
            result.net[4].weight[1].zero_()
            result.net[4].bias[1].zero_()
    return result


def rollout(model, start, n=50):
    with torch.no_grad():
        zs=[start]; z=start
        for _ in range(n):
            z=model.advance_theory(z); zs.append(z)
        return torch.stack(zs, 1).numpy()


def audit_roots(output):
    from scipy.optimize import root
    torch.set_num_threads(1)
    targets=torch.load(output/'fixed_scalar_targets.pt',weights_only=True)
    g=torch.cat(list(targets.values()));lo=float(g.min())-.15;hi=float(g.max())+.15
    result={}
    for dim in [1,2]:
        model=Flow(dim).double();model.load_state_dict(torch.load(output/f'fit_{dim}d.pt',weights_only=True))
        roots=[]
        def residual(x):
            with torch.no_grad(): return model.net(torch.tensor(x,dtype=torch.float64)).numpy()
        for x in np.linspace(lo,hi,35):
            for y in ([-.2,0.,.2] if dim==2 else [0.]):
                sol=root(residual,[x,y] if dim==2 else [x])
                if np.linalg.norm(residual(sol.x))>1e-8 or not lo<=sol.x[0]<=hi:continue
                if dim==2 and abs(sol.x[1])>.35:continue
                if any(np.linalg.norm(sol.x-np.array(r['g']))<1e-5 for r in roots):continue
                z=torch.tensor(sol.x,dtype=torch.float64,requires_grad=True)
                jac=torch.autograd.functional.jacobian(model.advance_theory,z).detach().numpy()
                eig=np.linalg.eigvals(jac);mods=np.abs(eig)
                kind='stable' if np.all(mods<1) else ('unstable' if np.all(mods>1) else 'saddle')
                roots.append({'g':sol.x.tolist(),'eigenvalues':[[float(e.real),float(e.imag)] for e in eig],
                              'stability':kind,'residual_norm':float(np.linalg.norm(residual(sol.x)))})
        result[f'fit_{dim}d']=sorted(roots,key=lambda r:r['g'][0])
    result['caveat']='Numerical roots in plotting box only. Not a proof of physical RG or critical exponents.'
    (output/'root_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--source-step',type=int,choices=[1000,2000],default=2000)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/tfim_scalar_QC_embedding_R_seed91'))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if a.audit_only:
        audit_roots(a.output)
        return
    torch.set_num_threads(1); torch.manual_seed(91)
    source=Path('results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91')
    cache=Path('data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt')
    base=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=1).double().eval()
    checkpoint=source/('training_step_1000.pt' if a.source_step==1000 else 'qcr_model.pt')
    state=torch.load(checkpoint,weights_only=True)
    base.load_state_dict(state['model'] if a.source_step==1000 else state,strict=False)
    base.theory_encoder.binary_lookup=base.rg_kernel.binary_lookup=True
    for v in base.parameters(): v.requires_grad_(False)
    before={k:v.clone() for k,v in base.state_dict().items()}
    tx,th,vx,vh=torch.load(cache,weights_only=True)
    with torch.no_grad():
        g=torch.cat([torch.stack([base.theory_encoder(z) for z in base.coarse_grain(balanced_views(x,128))],1)
                     for x in torch.cat([tx,vx]).split(8)])
    h=torch.cat([th,vh]).numpy(); train=g[:len(tx)]; test=g[len(tx):]
    g2=torch.cat([g,torch.zeros_like(g)],-1)
    torch.save({'train':train,'test':test},a.output/'fixed_scalar_targets.pt')
    original=Flow(1).double(); original.net.load_state_dict(base.theory_rg.state_dict())
    embedded=lift(original,exact=True)
    rs=rollout(original,g[:,0]); re=rollout(embedded,g2[:,0])
    np.testing.assert_allclose(re[:,:,0],rs[:,:,0],atol=1e-12,rtol=1e-12)
    assert np.max(abs(re[:,:,1]))==0
    # Match all shared scalar-path weights; extra input/output weights remain random.
    # All 2D parameters train freely. No invariant-line penalty or hard projection.
    torch.manual_seed(91); scalar=Flow(1).double(); vector=lift(scalar)
    torch.testing.assert_close(scalar.advance_theory(g[:,0]),vector.advance_theory(g2[:,0])[:,:1])
    report={'source':str(source),'source_checkpoint':str(checkpoint),'seed':91,'steps':a.steps,
            'contract':'Fixed scalar Q/C; full ensembles; paired shared-path initialization; all R weights free; 5*one-step + 5*rollout coordinate-sum MSE; Adam .001 cosine to .00001; final checkpoint, no test selection.',
            'train_shape':list(train.shape),'test_shape':list(test.shape),
            'exact_embedding_max_difference':float(np.max(abs(re[:,:,0]-rs[:,:,0]))),
            'baseline':{'train':metrics(original,train),'test':metrics(original,test)}}
    models={'baseline':original}; trajectories={'baseline':rs}
    for name,model,targets in [('fit_1d',scalar,g),('fit_2d',vector,g2)]:
        tr,te=targets[:len(tx)],targets[len(tx):]
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-5)
        rows=[{'step':0,'train':metrics(model,tr),'test':metrics(model,te)}]
        for step in range(1,a.steps+1):
            opt.zero_grad(set_to_none=True); one,multi=errors(model,tr)
            loss=5*(one.mean()+multi.mean()); loss.backward();opt.step();sched.step()
            if step%500==0:
                row={'step':step,'train':metrics(model,tr),'test':metrics(model,te)};rows.append(row)
                print(name,step,row['train']['closure_loss'],row['test']['closure_loss'],flush=True)
        torch.save(model.state_dict(),a.output/(name+'.pt'))
        models[name]=model; trajectories[name]=rollout(model,targets[:,0])
        report[name]={'train':metrics(model,tr),'test':metrics(model,te),'history':rows,
                      'parameters':sum(v.numel() for v in model.parameters())}
        if name=='fit_2d':
            r=trajectories[name][len(tx):]
            report[name]['test_transverse_rms_by_k']=np.sqrt(np.mean(r[:,:,1]**2,axis=0)).tolist()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(torch.equal(v,before[k]) for k,v in base.state_dict().items())
    report['QC_unchanged']=True
    pair={}
    for aa,bb in [(.95,1.05),(.5,1.5),(.01,2.)]:
        ha=float(h[np.argmin(abs(h-aa))]);hb=float(h[np.argmin(abs(h-bb))]);ia=h==ha;ib=h==hb
        pair[str(aa)]={'h_over_J':[ha,hb],'direct_distance':abs(g.numpy()[ia,:,0].mean(0)-g.numpy()[ib,:,0].mean(0)).tolist(),
                      'rollout_distance':{n:np.linalg.norm(r[ia].mean(0)-r[ib].mean(0),axis=-1).tolist() for n,r in trajectories.items()}}
    report['pairs']=pair
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(a.output/'trajectories.npz',h_over_J=h,direct=g.numpy(),**trajectories)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for n in ['fit_1d','fit_2d']:
        rows=report[n]['history']
        for split,style in [('train','-'),('test','--')]:
            axs[0].semilogy([r['step'] for r in rows],[r[split]['closure_loss'] for r in rows],style,label=n+' '+split)
    for n in models: axs[1].plot(range(1,6),report[n]['test']['rollout_nrmse'],'o-',label=n)
    for n,r in pair['0.95']['rollout_distance'].items():axs[2].plot(range(6),r[:6],'o-',label=n)
    axs[2].plot(range(6),pair['0.95']['direct_distance'],'k--',label='Q target')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[0].set(xlabel='Updates',ylabel='Common closure loss');axs[1].set(xlabel='k',ylabel='Test NRMSE')
    axs[2].set(xlabel='k',ylabel='Near-critical pair distance')
    fig.tight_layout();fig.savefig(a.output/'fit_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    qmin,qmax=float(g.min()),float(g.max());xx=np.linspace(qmin-.15,qmax+.15,400)
    with torch.no_grad():
        for n,m in models.items():
            inp=torch.tensor(xx[:,None]);inp=inp if n!='fit_2d' else torch.cat([inp,torch.zeros_like(inp)],1)
            axs[0].plot(xx,m.advance_theory(inp)[:,0].numpy(),label=n)
    axs[0].plot(xx,xx,'k--',label='identity');axs[0].set(xlabel='g (or g1 on g2=0)',ylabel='R1(g,0)');axs[0].legend()
    for k in range(6):
        hu=np.unique(h);means=np.array([g.numpy()[h==v,k,0].mean() for v in hu])
        axs[1].plot(hu,means,label=f'k={k}')
    axs[1].axvline(1,color='k',ls=':');axs[1].set(xlabel='h/J (display only)',ylabel='Frozen scalar Q / embedded g1');axs[1].legend()
    fig.tight_layout();fig.savefig(a.output/'R_and_coupling.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    yy=np.linspace(-.35,.35,21);xgrid=np.linspace(qmin-.15,qmax+.15,35);X,Y=np.meshgrid(xgrid,yy)
    z=torch.tensor(np.stack([X.ravel(),Y.ravel()],-1))
    with torch.no_grad():d=(vector.advance_theory(z)-z).numpy()
    speed=np.linalg.norm(d,axis=-1);dd=d/np.maximum(speed[:,None],1e-12)
    quiv=axs[0].quiver(X,Y,dd[:,0].reshape(X.shape),dd[:,1].reshape(X.shape),speed.reshape(X.shape),cmap='viridis')
    fig.colorbar(quiv,ax=axs[0],label='|R(g)-g| (arrows normalized)')
    axs[0].plot(g.numpy().ravel(),np.zeros(g.numel()),'r.',ms=2,label='all Q targets: g2=0')
    for target in [.5,.95,1.05,1.5]:
        hv=h[np.argmin(abs(h-target))];r=trajectories['fit_2d'][h==hv].mean(0)
        axs[0].plot(r[:6,0],r[:6,1],'o-',ms=3,label=f'h/J={hv:.3f}, k<=5')
    axs[0].set(xlabel='g1',ylabel='g2',title='2D R-only: vector field R(g)-g');axs[0].legend(fontsize=6)
    for n,r in pair['0.95']['rollout_distance'].items():axs[1].plot(range(51),r,label=n)
    axs[1].plot(range(6),pair['0.95']['direct_distance'],'ko--',label='Q target')
    axs[1].axvspan(5,50,color='gray',alpha=.12);axs[1].set(xlabel='k',ylabel='Near-critical pair distance',title='k>5: untrained extrapolation');axs[1].legend()
    fig.tight_layout();fig.savefig(a.output/'vector_flow_and_extrapolation.png',dpi=170);plt.close(fig)
    print('DONE',a.output,flush=True)


if __name__=='__main__':main()
