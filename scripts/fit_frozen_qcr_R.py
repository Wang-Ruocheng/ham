"""Fixed-target R-only regression; no coupling labels enter optimization."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from recurrent_rg.data.views import balanced_views


def errors(model, g):
    one = (model.advance_theory(g[:,:-1])-g[:,1:]).square().sum(-1)
    z=g[:,0]; terms=[]
    for k in range(1,g.shape[1]):
        z=model.advance_theory(z)
        terms.append((z-g[:,k]).square().sum(-1))
    return one,torch.stack(terms,1)


def metrics(model,g):
    with torch.no_grad():
        one,multi=errors(model,g)
        scale=g[:,1:].var(0,unbiased=False).sum(-1)
        return dict(closure_loss=float(5*(one.mean()+multi.mean())),
                    one_mse=one.mean(0).tolist(),rollout_mse=multi.mean(0).tolist(),
                    rollout_nrmse=(multi.mean(0)/scale).sqrt().tolist(),
                    rollout_max_error=multi.max().sqrt().item())


def refresh_summaries(output):
    report=json.loads((output/'report.json').read_text())
    original=json.loads((Path(report['source'])/'summary.json').read_text())
    for mode in ('mean','tail'):
        cfg=dict(original['config'],output_dir=str(output/mode),epochs=report['steps'],
                 threads=1,system_batch=129,sample_batch=128,eval_samples=128,eval_systems=63)
        summary={'config':cfg,'training_contract':{'optimized_modules':['R'],
            'frozen_modules':['Q','C'],'couplings_used_in_loss':False,
            'target':'fixed full-ensemble Q trajectories','mode':mode},
            'history':[dict(epoch=r['step'],train_total=r['train']['closure_loss'],
                            test_total=r['test']['closure_loss']) for r in report[mode]['history']],
            'history_meaning':'common 5*one-step+5*rollout evaluation, not total joint-QCR loss',
            'source_joint_model':report['source']}
        (output/mode/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pair=report['pairs']['0.95']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,r in pair['rollout_distance'].items():
        axes[0].plot(range(6),r[:6],'o-',label=name)
        axes[1].plot(range(51),r,label=name)
    for ax in axes:
        ax.plot(range(6),pair['direct_distance'],'ko--',label='actual Q target')
        ax.set(xlabel='k',ylabel='distance between ensemble-mean coordinates');ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('Trained horizons: k=0..5')
    axes[1].set_title('Long extrapolation: not constrained after k=5')
    axes[1].axvspan(5,50,color='gray',alpha=.1)
    fig.tight_layout();fig.savefig(output/'near_critical_fit.png',dpi=170);plt.close(fig)
    # Exact duplicate-input lower bound for deterministic one-step regression.
    targets=torch.load(output/'fixed_targets.pt',weights_only=True)['train'].numpy()
    x=targets[:,:-1].reshape(-1,2);y=targets[:,1:].reshape(-1,2)
    unique,group,count=np.unique(x,axis=0,return_inverse=True,return_counts=True)
    best=np.zeros_like(y);witness=None;largest=-1.
    for i in range(len(unique)):
        ids=np.where(group==i)[0];best[ids]=y[ids].mean(0)
        for a in ids:
            for b in ids:
                gap=float(np.linalg.norm(y[a]-y[b]))
                if gap>largest:
                    largest=gap;witness=dict(train_ensemble_a=int(a//5),scale_a=int(a%5),
                        train_ensemble_b=int(b//5),scale_b=int(b%5),input=x[a].tolist(),
                        target_a=y[a].tolist(),target_b=y[b].tolist(),target_distance=gap)
    audit=dict(input_count=len(x),distinct_inputs=len(unique),duplicate_groups=int((count>1).sum()),
               one_step_mse_lower_bound=float(np.mean(np.sum((y-best)**2,axis=-1))),witness=witness,
               caveat='Finite empirical targets. Does not prove the infinite-ensemble physical process is non-Markovian.')
    (output/'input_collision_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--output',type=Path,default=Path('results/tfim_g2_frozen_QC_Ronly_seed91'))
    parser.add_argument('--refresh-summaries',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.refresh_summaries:
        refresh_summaries(args.output)
        return
    source=Path('results/joint_qcr_tfim_1d_g2_coordsum_levels6_2000_seed91')
    cache=Path('data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt')
    torch.set_num_threads(1);torch.manual_seed(91)
    base=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=2).double().eval()
    base.load_state_dict(torch.load(source/'qcr_model.pt',weights_only=True),strict=False)
    base.theory_encoder.binary_lookup=base.rg_kernel.binary_lookup=True
    for p in base.parameters():p.requires_grad_(False)
    tx,th,vx,vh=torch.load(cache,weights_only=True);h=torch.cat([th,vh]).numpy()
    with torch.no_grad():
        g=torch.cat([torch.stack([base.theory_encoder(z) for z in base.coarse_grain(balanced_views(x,128))],1)
                     for x in torch.cat([tx,vx]).split(8)])
    old=np.load(source/'vector_flow_data.npz')
    np.testing.assert_allclose(g.numpy(),old['direct'],rtol=1e-12,atol=1e-12)
    train,test=g[:len(tx)],g[len(tx):]
    torch.save(dict(train=train,test=test),args.output/'fixed_targets.pt')
    report={'source':str(source),'steps':args.steps,'seed':91,
            'contract':'Freeze Q/C; full-ensemble fixed targets; original R architecture/init; full-batch Adam cosine lr .001 to .00001; no labels; no test selection.',
            'baseline':{s:metrics(base,x) for s,x in [('train',train),('test',test)]}}
    models={'baseline':base};history={}
    for mode in ('mean','tail'):
        model=copy.deepcopy(base)
        for p in model.theory_rg.parameters():p.requires_grad_(True)
        params=list(model.theory_rg.parameters());optimizer=torch.optim.Adam(params,lr=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=1e-5)
        rows=[]
        for i in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            one,multi=errors(model,train)
            loss=5*(one.mean()+multi.mean())
            if mode=='tail':
                # Mean plus worst 10% sample/horizon errors, separately per term.
                n=max(1,int(np.ceil(one.numel()*.1)))
                loss=loss+5*(one.flatten().topk(n).values.mean()+multi.flatten().topk(n).values.mean())
            loss.backward();optimizer.step();scheduler.step()
            if i==0 or (i+1)%500==0:
                row={'step':i+1,'optimized_loss_before_update':loss.item(),
                     'train':metrics(model,train),'test':metrics(model,test)}
                rows.append(row)
                print(mode,i+1,'train/test closure',row['train']['closure_loss'],row['test']['closure_loss'],flush=True)
        for name,p in model.state_dict().items():
            if not name.startswith('theory_rg.'):
                assert torch.equal(p,base.state_dict()[name]),name
        dest=args.output/mode;dest.mkdir(exist_ok=True)
        torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},dest/'qcr_model.pt')
        source_summary=json.loads((source/'summary.json').read_text())
        config={'config':source_summary['config'], 'training_contract':{
            'optimized_modules':['R'], 'frozen_modules':['Q','C'], 'couplings_used_in_loss':False,
            'target':'fixed full-ensemble Q trajectories', 'mode':mode},
            'history':[dict(epoch=r['step'],train_total=r['train']['closure_loss'],
                            test_total=r['test']['closure_loss']) for r in rows],
            'history_meaning':'common 5*one-step+5*rollout evaluation, not total joint-QCR loss'}
        config['config']['output_dir']=str(dest)
        config['config']['epochs']=args.steps
        config['R_only_fit']={'steps':args.steps,'mode':mode,'frozen_QC':True,'train_targets_shape':list(train.shape)}
        (dest/'summary.json').write_text(json.dumps(config,indent=2)+'\n')
        report[mode]={s:metrics(model,x) for s,x in [('train',train),('test',test)]}
        report[mode]['history']=rows;models[mode]=model;history[mode]=rows
    trajectories={}
    with torch.no_grad():
        for name,model in models.items():
            z=g[:,0];zs=[z.numpy().copy()]
            for k in range(50):z=model.advance_theory(z);zs.append(z.numpy().copy())
            trajectories[name]=np.stack(zs,1)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for mode,rows in history.items():
        for split,style in [('train','-'),('test','--')]:
            axes[0].semilogy([r['step'] for r in rows],[r[split]['closure_loss'] for r in rows],style,label=mode+' '+split)
    for name in models:
        axes[1].plot(range(1,6),report[name]['test']['rollout_nrmse'],'o-',label=name)
    pairs={}
    for target_a,target_b in [(.95,1.05),(.5,1.5),(.01,2.)]:
        a=float(h[np.argmin(abs(h-target_a))]);b=float(h[np.argmin(abs(h-target_b))]);ia=h==a;ib=h==b
        direct=np.linalg.norm(g.numpy()[ia].mean(0)-g.numpy()[ib].mean(0),axis=-1)
        item={'h_over_J':[a,b],'direct_distance':direct.tolist(),'rollout_distance':{}}
        for name,r in trajectories.items():
            gap=np.linalg.norm(r[ia].mean(0)-r[ib].mean(0),axis=-1)
            item['rollout_distance'][name]=gap.tolist()
            if target_a==.95:axes[2].plot(range(51),gap,label=name)
        if target_a==.95:axes[2].plot(range(6),direct,'ko',label='actual Q')
        pairs[str(target_a)]=item
    axes[0].set(title='Common closure loss (not total QCR loss)',xlabel='R-only update',ylabel='5 one-step + 5 rollout')
    axes[1].set(title='Full-test rollout error',xlabel='k',ylabel='NRMSE')
    axes[2].set(title='Near-critical pair: mean-coordinate distance',xlabel='k',ylabel='Euclidean distance')
    axes[2].axvspan(5,50,color='grey',alpha=.08,label='untrained horizon')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'R_fit_comparison.png',dpi=170);plt.close(fig)
    report['pairs']=pairs
    np.savez(args.output/'trajectories.npz',h_over_J=h,direct=g.numpy(),**trajectories)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    refresh_summaries(args.output)
    print('DONE',str(args.output),flush=True)


if __name__=='__main__':main()
