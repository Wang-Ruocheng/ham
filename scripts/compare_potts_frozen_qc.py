"""Matched fresh-data diagnostics of joint vs frozen-QC continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import collect_trajectories,closure_metrics,scalar_flow


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint',type=Path,default=Path('results/potts2d_g1_neutral0_2400_seed91'))
    p.add_argument('--frozen',type=Path,default=Path('results/potts2d_g1_freezeQC800_R2400_seed91'))
    p.add_argument('--output',type=Path,default=Path('results/potts2d_frozenQC800_comparison_seed91'))
    p.add_argument('--soft',type=Path,help='Optional completed cosine soft-freeze run')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    fresh=torch.load('results/potts2d_early_stopping_seed91/fresh_ensembles.pt',weights_only=True)
    ks=fresh['K'].numpy();kc=float(np.log1p(np.sqrt(3)));reports={};models={}
    for name,path,step in [('start800',args.joint,800),('joint2400',args.joint,2400),('frozen2400',args.frozen,2400)]:
        models[name]=load_run(path,step).model
    if args.soft:models['soft2400']=load_run(args.soft,2400).model
    for name in models['start800'].state_dict():
        if name.startswith(('theory_encoder.','rg_kernel.')):
            assert torch.equal(models['start800'].state_dict()[name],models['frozen2400'].state_dict()[name]),name
    fig,axs=plt.subplots(2,len(models),figsize=(5*len(models),8),layout='constrained');reference=None
    for col,(name,m) in enumerate(models.items()):
        trajectories=[collect_trajectories(m,x,samples=64,max_rg_steps=2,sampling_seed=12345+i)
                      for i,x in enumerate(fresh['audit'])]
        joined={key:np.concatenate([v[key] for v in trajectories]) for key in trajectories[0]}
        if reference is None:reference=joined['direct']
        if name=='frozen2400':np.testing.assert_array_equal(reference,joined['direct'])
        g=np.stack([v['direct'][:,:,0] for v in trajectories]).mean(0)
        lo,hi=joined['direct'][:,:-1,0].min(),joined['direct'][:,:-1,0].max()
        pad=max(.4,.4*(hi-lo));f=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
        wide=scalar_flow(m,float(lo-2*(hi-lo+1)),float(hi+2*(hi-lo+1)),points=12001)
        for root in wide['roots']:
            z=root['g'][0];hits=[]
            for ka,kb,ga,gb in zip(ks[:-1],ks[1:],g[:-1,0],g[1:,0]):
                if min(ga,gb)<=z<=max(ga,gb) and ga!=gb:hits.append(float(ka+(kb-ka)*(z-ga)/(gb-ga)))
            root['microscopic_inverse_K']=hits;root['inside_observed_inputs']=bool(lo<=z<=hi)
        reports[name]=dict(closure=closure_metrics(joined),roots=wide['roots'],search_range=wide['search_range'],
            g_at_Kc=float(np.interp(kc,ks,g[:,0])),observed_range=[float(lo),float(hi)])
        ax=axs[0,col];ax.plot(f['grid'],f['mapped'],label='R(g)');ax.plot(f['grid'],f['grid'],'k--',label='identity')
        ax.axvspan(lo,hi,color='green',alpha=.1,label='Observed inputs');ax.axvline(reports[name]['g_at_Kc'],color='purple',ls=':',label='Q(Kc)')
        for r in f['roots']:
            z=r['g'][0];ax.scatter(z,z,c='green' if r['stability']=='stable' else 'red')
        ax.set(title=name,xlabel='g',ylabel='R(g)');ax.legend(fontsize=7);ax.grid(alpha=.2)
        ax=axs[1,col]
        for k in range(3):ax.plot(ks,g[:,k],label=f'Q(C^{k}D)')
        ax.axvline(kc,color='purple',ls=':',label='Kc')
        for r in wide['roots']:
            if r['stability']=='unstable':ax.axhline(r['g'][0],color='red',ls='--',label='Repeller')
        ax.set(xlabel='K',ylabel='g');ax.legend(fontsize=7);ax.grid(alpha=.2)
        np.savez_compressed(args.output/f'{name}_trajectories.npz',K=np.tile(ks,3),**joined)
    fig.savefig(args.output/'R_and_K_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    paths=[('joint',args.joint),('frozen QC',args.frozen)]
    if args.soft:paths.append(('soft QC',args.soft))
    for label,path in paths:
        hist=json.loads((path/'history_progress.json').read_text());hist=[r for r in hist if r['epoch']>=(400 if args.soft else 800)]
        for split,style in [('train','-'),('test','--')]:
            axs[0].plot([r['epoch'] for r in hist],[r[f'{split}_total'] for r in hist],style,label=f'{label} {split}')
            axs[1].plot([r['epoch'] for r in hist],[5*(r[f'{split}_commutator']+r[f'{split}_rollout']) for r in hist],style,label=f'{label} {split}')
    axs[0].set(title='Full QCR objective',ylabel='loss');axs[1].set(title='R-dependent loss: 5 one-step + 5 rollout',ylabel='loss')
    for ax in axs:ax.set(xlabel='Total training updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(args.output/'loss_comparison.png',dpi=170);plt.close(fig)
    reports['contract']=dict(hard_freeze_QC_bitwise_equal=True,hard_freeze_fresh_direct_targets_bitwise_equal=True,
        note='Same three fresh audit-chain families for all models; not used in training or selection. Microscopic K inverse is a coordinate diagnostic, not an effective coupling.')
    (args.output/'comparison.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
