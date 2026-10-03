"""Second stage: regress autonomous R on verified, frozen Q/decimation targets."""
import argparse
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
from recurrent_rg.checkpoints import load_run,qcr_state_dict
from recurrent_rg.plotting.reports import plot_diagnostics


def predictions(model,g):
    one=model.advance_theory(g[:,:-1])
    rollout=[g[:,0]]
    for _ in range(g.shape[1]-1): rollout.append(model.advance_theory(rollout[-1]))
    return one,torch.stack(rollout,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=6000)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(91);started=time.perf_counter()
    loaded=load_run(args.source/'q_only.pt');model=loaded.model
    cfg=json.loads((args.source/'report.json').read_text())
    data=np.load(args.source/'encodings.npz')
    tr=torch.from_numpy(data['train']);te=torch.from_numpy(data['test'])
    tx,_,vx,_=torch.load(cfg['cache'],weights_only=True)
    with torch.no_grad():
        for x,target in ((tx[:4],tr[:4]),(vx[:4],te[:4])):
            actual=torch.stack([model.theory_encoder(level) for level in model.coarse_grain(x.double())],1)
            torch.testing.assert_close(actual,target,atol=1e-12,rtol=1e-10)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    frozen={n:v.clone() for n,v in model.state_dict().items() if not n.startswith('theory_rg.')}
    cov=torch.stack([torch.cov(tr[:,k].T,correction=0) for k in range(1,3)])
    eig,axes=torch.linalg.eigh(cov)
    def metrics(g,identity=False):
        with torch.no_grad():
            one,roll=(g[:,:-1],g[:,0:1].expand_as(g)) if identity else predictions(model,g)
            result={}
            for name,pred in [('step',one),('rollout',roll[:,1:])]:
                err=pred-g[:,1:]
                pc=torch.einsum('nki,kij->nkj',err,axes)
                result[name]=dict(mse_sum_coordinates=float(err.square().sum(-1).mean()),
                    nrmse_by_k=(err.square().mean((0,2))/eig.mean(-1)).sqrt().tolist(),
                    pc_nrmse_weak_strong=(pc.square().mean(0)/eig).sqrt().tolist(),
                    rmse_by_k_coordinate=err.square().mean(0).sqrt().tolist())
            result['total']=5*(result['step']['mse_sum_coordinates']+result['rollout']['mse_sum_coordinates'])
            return result
    report=dict(source=str(args.source.resolve()),seed=91,steps=args.steps,
        contract='Only R optimized; original Q and fixed positive decimation bitwise frozen; verified full-ensemble cached targets. Labels for plotting only. All available scales k=0,1,2, no scale input to R.',
        loss='5 mean_{ensembles,k=0,1} ||R(g_k)-g_{k+1}||^2 + 5 mean_{ensembles,k=1,2} ||R^k(g_0)-g_k||^2; coordinate sum, ordinary Euclidean metric.',
        optimizer='Full-batch Adam lr0.001; cosine decay to 0.00001; original R architecture and unchanged Q-only checkpoint R initialization.',
        optimized_parameters=sum(p.numel() for p in model.theory_rg.parameters()),
        identity_baseline=dict(train=metrics(tr,True),test=metrics(te,True)),
        normalization='Per-scale training target covariance; fixed for all reports',history=[])
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
    for step in range(args.steps+1):
        if step:
            opt.zero_grad();one,roll=predictions(model,tr)
            loss=5*((one-tr[:,1:]).square().sum(-1).mean()+(roll[:,1:]-tr[:,1:]).square().sum(-1).mean())
            assert torch.isfinite(loss)
            loss.backward();opt.step();scheduler.step()
        if step%200==0 or step==args.steps:
            row=dict(step=step,train=metrics(tr),test=metrics(te));report['history'].append(row)
            print(step,'loss',row['train']['total'],row['test']['total'],'test rollout NRMSE',row['test']['rollout']['nrmse_by_k'],flush=True)
    for n,v in frozen.items():assert torch.equal(v,model.state_dict()[n]),n
    report['train']=metrics(tr);report['test']=metrics(te)
    report['seconds_training_and_metrics']=time.perf_counter()-started
    report['frozen_parameters_bitwise_verified']=True
    torch.save(dict(model=qcr_state_dict(model),metadata=loaded.payload['metadata']),args.output/'qcr_model.pt')
    with torch.no_grad():
        one,roll=predictions(model,te)
    trajectories=dict(direct=te.numpy(),rollout=roll.numpy(),one_step=one.numpy())
    np.savez(args.output/'test_trajectories.npz',**trajectories,labels=data['test_labels'])
    plot_diagnostics(model,trajectories,data['test_labels'],('K','B'),args.output,training_inputs=tr[:,:-1].numpy())
    report['fixed_points']=json.loads((args.output/'fixed_points.json').read_text())
    # Dedicated flow with frozen microscopic K/B grid and B=0 reference curve.
    g=np.concatenate((data['train'],data['test']));labels=np.concatenate((data['train_labels'],data['test_labels']))
    ks=np.unique(labels[:,0]);bs=np.unique(labels[:,1])
    grid=np.array([[g[np.all(labels==[k,b],axis=1),0].mean(0) for b in bs] for k in ks])
    bounds=np.asarray(report['fixed_points']['search_box']);lo,hi=bounds
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],31),np.linspace(lo[1],hi[1],31));points=np.column_stack((xx.ravel(),yy.ravel()))
    with torch.no_grad():delta=(model.advance_theory(torch.from_numpy(points))-torch.from_numpy(points)).numpy()
    speed=np.linalg.norm(delta,axis=1);arrows=delta/np.maximum(speed[:,None],1e-12)*.035*min(hi-lo)
    fig,ax=plt.subplots(figsize=(8,7))
    ax.quiver(*points.T,*arrows.T,color='.65',angles='xy',scale_units='xy',scale=1,width=.0025)
    for line in grid:ax.plot(*line.T,color='.6',alpha=.3,lw=.5)
    sc=ax.scatter(*grid.reshape(-1,2).T,c=np.tile(bs,len(ks)),cmap='coolwarm',s=12,vmin=-.5,vmax=.5)
    ax.plot(*grid[:,np.argmin(abs(bs))].T,'k-',lw=1.5,label='B=0: microscopic Q')
    for i,item in enumerate(report['fixed_points']['roots'],1):
        ax.scatter(*item['g'],s=100,marker='*',color='green' if item['stability']=='stable' else 'purple')
        ax.annotate(f'F{i}: {item["stability"]}',item['g'],xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='Frozen Q/C, trained R: normalized R(g)-g',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
    ax.legend();fig.colorbar(sc,ax=ax,label='B');fig.tight_layout();fig.savefig(args.output/'R_with_B0.png',dpi=180);plt.close(fig)
    h=report['history'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for split in ('train','test'):
        axs[0].plot([r['step'] for r in h],[r[split]['total'] for r in h],label=split)
    axs[0].set(yscale='log',xlabel='R updates',ylabel='weighted loss');axs[0].legend()
    for mode in ('step','rollout'):
        axs[1].plot([1,2],100*np.array(report['test'][mode]['nrmse_by_k']),'o-',label=mode)
        axs[1].plot([1,2],100*np.array(report['identity_baseline']['test'][mode]['nrmse_by_k']),'--',label=f'identity {mode}')
    axs[1].set(xticks=[1,2],xlabel='target k',ylabel='test NRMSE (%)');axs[1].legend()
    fig.tight_layout();fig.savefig(args.output/'loss_and_errors.png',dpi=180);plt.close(fig)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('train','test','fixed_points','seconds_training_and_metrics')},indent=2),flush=True)


if __name__=='__main__':main()
