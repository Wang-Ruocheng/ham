"""Scalar joint-QCR diagnostics from a versioned run, without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--output',type=Path,help='Separate diagnostic output directory; never changes the run')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--sampling-repeats',type=int,default=5,help='Extra stochastic-C audits; zero for lightweight previews')
    args=p.parse_args()
    if args.threads<1 or args.sampling_repeats<0:p.error('Invalid diagnostic threads/repeats')
    torch.set_num_threads(args.threads)
    loaded=load_run(args.run,args.checkpoint);cfg=loaded.config;model=loaded.model
    if cfg['model']['theory_dim']!=1:
        raise ValueError('Scalar reporter requires theory_dim=1')
    train,test=load_ensembles(cfg['data']['cache'])
    options=dict(samples=cfg['evaluation']['samples'],max_rg_steps=cfg['training']['max_rg_steps'],
                 spin_flip_augment=cfg['data']['spin_flip_augment'])
    tr=collect_trajectories(model,train.configurations,**options)
    te=collect_trajectories(model,test.configurations,**options)
    out=args.output or args.run/('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}');out.mkdir(parents=True,exist_ok=True)
    plot_diagnostics(model,te,test.couplings.numpy(),cfg['data']['coupling_names'],out,
                     training_inputs=tr['direct'][:,:-1])
    np.savez_compressed(out/'encodings.npz',train=tr['direct'],test=te['direct'],
                        test_rollout=te['rollout'],train_labels=train.couplings.numpy(),test_labels=test.couplings.numpy())
    g=np.concatenate([tr['direct'],te['direct']])[:,:,0]
    labels=np.concatenate([train.couplings.numpy(),test.couplings.numpy()]).reshape(-1)
    knots=np.unique(labels)
    means=np.stack([g[labels==v].mean(0) for v in knots])
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for k in range(g.shape[1]):ax.plot(knots,means[:,k],label=f'k={k}')
    ax.set(xlabel=cfg['data']['coupling_names'][0],ylabel='g',title='Q(C^k D): mean over ensembles')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'coupling_by_scale.png',dpi=170);plt.close(fig)
    # Numbered checkpoints include their own history: do not race a live writer.
    h=loaded.payload.get('history') if args.checkpoint is not None else None
    if h is None:h=json.loads((args.run/'history_progress.json').read_text())
    if args.checkpoint is not None:h=[r for r in h if r['epoch']<=args.checkpoint]
    weights=dict(cfg['loss']['weights']);weights.update(rollout=cfg['loss']['rollout_weight'],isotropy=cfg['loss']['isotropy_weight'],tangent=cfg['loss'].get('tangent_weight',0))
    weights['kernel_sign'] = cfg['loss'].get('kernel_sign_weight',0.)
    weights['kernel_neutral'] = cfg['loss'].get('kernel_neutral_weight',0.)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,split in zip(axs,['train','test']):
        for key,w in weights.items():
            if w:ax.plot([r['epoch'] for r in h],[w*r[split+'_'+key] for r in h],label=key)
        ax.plot([r['epoch'] for r in h],[r.get(split+'_qcr_total',r[split+'_total']) for r in h],'k-',label='QCR total (MI excluded if present)')
        ax.set(xlabel='updates',ylabel='weighted loss',title=split)
        ax.set_yscale('symlog',linthresh=1e-3);ax.set_ylim(bottom=0);ax.legend(fontsize=7)
    fig.savefig(out/'loss.png',dpi=170);plt.close(fig)
    volume=model.rg_kernel.geometry.block_volume
    categorical=cfg['model'].get('categorical_states',0)
    if cfg['model'].get('fixed_kernel'):
        kernel_report=dict(type='fixed positive decimation',sampling='deterministic',
                           note='Retains the complete state at the first site of each block')
    elif categorical:
        exhaustive=categorical**volume<=65536
        states=((torch.arange(categorical**volume)[:,None]//(categorical**torch.arange(volume)))%categorical
                if exhaustive else torch.arange(categorical)[:,None].expand(-1,volume))
        patterns=torch.nn.functional.one_hot(states,categorical).double().flatten(1)
        with torch.no_grad():
            probabilities=torch.softmax(model.rg_kernel.field_head.logits(model.rg_kernel.local_encoder(patterns)),dim=-1)
        kernel_report=dict(patterns=states.tolist(),probabilities=probabilities.tolist(),
            uniform_retention=[float(probabilities[(states==a).all(1),a].item()) for a in range(categorical)],
            sampling='Independent categorical at each block and scale, also during evaluation',
            exhaustive_pattern_report=exhaustive,total_possible_patterns=categorical**volume,
            note='All patterns' if exhaustive else 'Uniform blocks only; full large truth table omitted from report, not from training MI')
    else:
        patterns=(2*((torch.arange(2**volume)[:,None]//(2**torch.arange(volume)))%2)-1).double()
        with torch.no_grad():outputs=model.rg_kernel(patterns.reshape(2**volume,1,*model.rg_kernel.geometry.factors,1)).flatten().tolist()
        kernel_report=dict(patterns=patterns.tolist(),outputs=outputs)
    report=dict(closure=closure_metrics(te,reference_direct=tr['direct']),
                kernel=kernel_report,
                parameters={n:sum(p.numel() for p in module.parameters()) for n,module in [('Q',model.theory_encoder),('C',model.rg_kernel),('R',model.theory_rg)]},
                fixed_points=json.loads((out/'fixed_points.json').read_text()))
    if cfg['model'].get('kernel_sampling') in ('bernoulli','categorical'):
        if not categorical:
            with torch.no_grad():
                logits=model.rg_kernel.field_head.logits(model.rg_kernel.local_encoder(patterns))
            report['kernel']['outputs_note']='One random draw per pattern, NOT a deterministic truth table'
            report['kernel']['p_plus']=torch.sigmoid(2*logits).flatten().tolist()
            report['kernel']['sampling']='Independent Bernoulli at each block and scale, also during evaluation'
        sampling=[]
        for seed in range(args.sampling_repeats):
            draw=collect_trajectories(model,test.configurations,**options,sampling_seed=seed)
            sampling.append(closure_metrics(draw,reference_direct=tr['direct']))
        report['sampling_repeats']=dict(seeds=list(range(args.sampling_repeats)),closure=sampling,
            note='Sampling-noise audit of one trained model, not independent training seeds')
    for root in report['fixed_points']['roots']:
        target=root['g'][0];hits=[]
        for x,y,a,b in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]):
            if min(a,b)<=target<=max(a,b) and a!=b:
                hits.append(float(x+(y-x)*(target-a)/(b-a)))
        root['microscopic_inverse_coupling']=hits
        root['coupling_name']=cfg['data']['coupling_names'][0]
        if not categorical:
            root['microscopic_inverse_h_over_J']=hits  # Historical report compatibility.
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
