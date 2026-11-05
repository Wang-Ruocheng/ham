"""Explicit R-only continuation with frozen Q/C and label-free cached targets."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run, qcr_state_dict, checkpoint_payload
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, scalar_flow
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss
from recurrent_rg.plotting.reports import plot_diagnostics
from recurrent_rg.runner import prepared_data, file_fingerprint


def r_objective(model, direct, cfg):
    """Same one-step/all-horizon closure terms as joint QCR; no labels."""
    one=latent_squared_error(model.advance_theory(direct[:,:-1]),direct[:,1:],
                             cfg.loss.coordinate_reduction)
    rollout,_=multistep_closure_loss(model,direct,cfg.loss.coordinate_reduction)
    return cfg.loss.weights.commutator*one+cfg.loss.rollout_weight*rollout


def continued_r_adam(model, payload):
    """Extract R's moments and step counters from the original Q/C/R Adam."""
    saved=payload['optimizer']
    if len(saved['param_groups'])!=1:
        raise ValueError('R continuation currently requires a single Adam parameter group')
    cfg=payload['metadata']['config']
    modules=set(cfg['training']['train_modules'])
    frozen_c=bool(payload['metadata'].get('frozen_C'))
    q=list(model.theory_encoder.parameters()) if 'Q' in modules else []
    c=list(model.rg_kernel.parameters()) if 'C' in modules and not frozen_c else []
    r=list(model.theory_rg.parameters())
    ids=saved['param_groups'][0]['params']
    if 'R' not in modules or len(ids)!=len(q)+len(c)+len(r):
        raise ValueError('Source optimizer parameter layout does not match Q/C/R driver')
    opt=torch.optim.Adam(r,lr=saved['param_groups'][0]['lr'])
    group=copy.deepcopy(saved['param_groups'][0]);group['params']=list(range(len(r)))
    state={i:copy.deepcopy(saved['state'][old]) for i,old in enumerate(ids[-len(r):])
           if old in saved['state']}
    opt.load_state_dict(dict(state=state,param_groups=[group]))
    return opt


def predictions(model, direct):
    with torch.no_grad():
        one=model.advance_theory(direct[:,:-1])
        z=direct[:,0];roll=[z]
        for _ in range(direct.shape[1]-1):
            z=model.advance_theory(z);roll.append(z)
    return dict(direct=direct.numpy(),one_step=one.numpy(),rollout=torch.stack(roll,1).numpy())


def scalar_report(model, train, test, train_labels, test_labels, output):
    plot_diagnostics(model,predictions(model,test),test_labels,['K'],output,
                     training_inputs=train[:,:-1].numpy())
    direct=torch.cat([train,test]).numpy()[:,:,0]
    labels=np.concatenate([train_labels,test_labels]).reshape(-1)
    knots=np.unique(labels)
    means=np.stack([direct[labels==k].mean(0) for k in knots])
    report=json.loads((output/'fixed_points.json').read_text())
    inputs=train[:,:-1].numpy().reshape(-1)
    for root in report['roots']:
        target=root['g'][0];hits=[]
        for a,b,x,y in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]):
            if x!=y and min(x,y)<=target<=max(x,y):
                hits.append(float(a+(b-a)*(target-x)/(y-x)))
        root['microscopic_inverse_coupling']=hits
        root['inside_training_input_range']=bool(inputs.min()<=target<=inputs.max())
    (output/'fixed_points.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def comparison_figure(base,model,train,test,train_labels,test_labels,history,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    direct=torch.cat([train,test]).numpy()[:,:,0]
    labels=np.concatenate([train_labels,test_labels]).reshape(-1)
    knots=np.unique(labels);means=np.stack([direct[labels==k].mean(0) for k in knots])
    inputs=train[:,:-1].numpy().reshape(-1)
    pad=max(.2,.2*np.ptp(direct));lower=direct.min()-pad;upper=direct.max()+pad
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for name,m,color in [('Before',base,'#999999'),('R-only continuation',model,'#0072b2')]:
        flow=scalar_flow(m,lower,upper)
        axs[0,0].plot(flow['grid'],flow['mapped'],color=color,label=name)
        axs[0,1].plot(flow['grid'],flow['mapped']-flow['grid'],color=color,label=name)
        for root in flow['roots']:
            g=root['g'][0]
            axs[0,1].scatter(g,0,c=color,marker='o' if root['stability']=='stable' else 'x')
    axs[0,0].plot([lower,upper],[lower,upper],'k--',label='Identity')
    for k in range(means.shape[1]-1):
        axs[0,1].plot(means[:,k],means[:,k+1]-means[:,k],'o--',ms=3,alpha=.7,
                      label=f'Mean direct Q drift, k={k}')
    axs[0,1].axhline(0,color='black',lw=.8)
    for ax in axs[0]:ax.axvspan(inputs.min(),inputs.max(),alpha=.08,color='green')
    axs[0,0].set(xlabel='g',ylabel='R(g)',title='Same Q/C and R architecture')
    axs[0,1].set(xlabel='g',ylabel='R(g)-g',title='Zeros and empirical coarse-graining drift')
    for k in range(means.shape[1]):
        axs[1,0].plot(knots,means[:,k],'o-',ms=3,label=f'Q(C^k D), k={k}')
    axs[1,0].set(xlabel='K (evaluation only)',ylabel='g',title='Frozen coupling-to-g map')
    for split,style in [('train','-'),('test','--')]:
        axs[1,1].semilogy([r['extra_R_updates'] for r in history],
                         [r[split+'_closure'] for r in history],style,label=split)
    axs[1,1].set(xlabel='Additional R-only updates',ylabel='5 one-step + 5 rollout',
                 title='Cached targets; no test-based selection')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output/'R_continuation_comparison.png',dpi=170);plt.close(fig)


def run(source,output,updates,*,max_wall_seconds=None):
    started=time.perf_counter();source=Path(source);output=Path(output)
    if updates<1 or (max_wall_seconds is not None and
                     (not np.isfinite(max_wall_seconds) or max_wall_seconds<=0)):
        raise ValueError('Positive update count and finite positive optional wall limit required')
    loaded=load_run(source/'checkpoint_last.pt');cfg=RunConfig.from_dict(loaded.config)
    if cfg.model.theory_dim!=1 or cfg.loss.closure_metric!='euclidean':
        raise ValueError('This continuation diagnostic requires scalar g and Euclidean closure')
    if not loaded.payload.get('metadata',{}).get('independent_c'):
        raise ValueError('Requires an independent-gradient source run')
    if file_fingerprint(cfg.data.cache)!=loaded.payload['metadata']['data_sha256']:
        raise ValueError('Source data fingerprint changed')
    if (cfg.data.train_pool_replicas,cfg.data.test_pool_replicas)!=(1,1):
        raise ValueError('Saved target order currently requires unpooled ensembles')
    torch.set_num_threads(cfg.training.threads);torch.manual_seed(cfg.training.seed)
    base=loaded.model.eval();model=copy.deepcopy(base)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    opt=continued_r_adam(model,loaded.payload)
    tx,vx=prepared_data(cfg)
    cache=source/'full_report'/'encodings.npz'
    arrays=np.load(cache)
    train=torch.from_numpy(arrays['train'].copy()).double()
    test=torch.from_numpy(arrays['test'].copy()).double()
    train_labels=tx.couplings.numpy();test_labels=vx.couplings.numpy()
    for split,target,ds,labels in [('train',train,tx,train_labels),('test',test,vx,test_labels)]:
        if target.shape!=(len(ds.configurations),cfg.training.max_rg_steps+1,1):
            raise ValueError('Target shape differs from source data/scales')
        np.testing.assert_array_equal(labels,arrays[split+'_labels'])
        if not torch.isfinite(target).all():raise ValueError('Nonfinite fixed target')
    # Check the cached full-report target really comes from this checkpoint.
    options=dict(samples=cfg.evaluation.samples,max_rg_steps=cfg.training.max_rg_steps,
                 spin_flip_augment=cfg.data.spin_flip_augment)
    verified=collect_trajectories(base,tx.configurations,**options)
    np.testing.assert_allclose(verified['direct'],train.numpy(),atol=1e-12,rtol=1e-12)
    output.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(cfg.to_dict());config['training']['train_modules']=['R']
    config['training']['epochs']=updates
    metadata=dict(config=config,data_sha256=file_fingerprint(cfg.data.cache),
        source=str(source.resolve()),source_checkpoint_sha256=file_fingerprint(loaded.checkpoint),
        source_epoch=int(loaded.payload['epoch']),target_cache_sha256=file_fingerprint(cache),
        training_driver='frozen_qc_r_continue',independent_c=True,
        optimized_modules=['R'],frozen_modules=['Q','C'],labels_used_in_training=False,
        targets='Original full-report TRAIN ensemble trajectories; no coupling labels; fixed C draws',
        differences_from_source='Q/C frozen; full-batch cached 64-sample full-ensemble Q trajectories rather than fresh stochastic system/sample batches and the mean of two half-subset Q outputs',
        R='Original weights, architecture, learning rate, Adam moments and step counters; no scheduler',
        closure='Original weighted one-step and all-horizon rollout squared Euclidean loss',
        selection='Final completed update, not test-selected',requested_extra_R_updates=updates,
        max_wall_seconds=max_wall_seconds)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    torch.save(dict(train=train,test=test),output/'fixed_targets.pt')
    initial_steps=[float(opt.state[p]['step']) for p in model.theory_rg.parameters()]
    before_state={k:v.clone() for k,v in qcr_state_dict(base).items()}
    history=[];snapshots={};gen=torch.Generator().manual_seed(cfg.training.seed)

    def log(step):
        row={'extra_R_updates':step,'epoch':step}
        for split,d in [('train',train),('test',test)]:
            with torch.no_grad():row[split+'_closure']=float(r_objective(model,d,cfg))
            row[split+'_metrics']=closure_metrics(predictions(model,d),reference_direct=train.numpy())
        history.append(row)
        (output/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
        print(f'R-only {step}/{updates}: train closure={row["train_closure"]:.7f}, test closure={row["test_closure"]:.7f}',flush=True)
    log(0);completed=0
    for step in range(1,updates+1):
        opt.zero_grad(set_to_none=True);loss=r_objective(model,train,cfg)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite R loss at {step}')
        loss.backward();opt.step();completed=step
        stopped=max_wall_seconds is not None and time.perf_counter()-started>=max_wall_seconds
        if step%250==0 or step==updates or stopped:log(step)
        if step in (1000,3000,6000) or step==updates or stopped:
            torch.save(checkpoint_payload(model,opt,gen,step,metadata=metadata,history=history),
                       output/f'training_step_{step}.pt')
            snapshots[str(step)]=scalar_report(model,train,test,train_labels,test_labels,
                                               output/f'step{step}')
        if stopped:break
    frozen_difference={}
    for prefix in ('theory_encoder.','rg_kernel.'):
        differences=[float((v-qcr_state_dict(model)[k]).abs().max())
                     for k,v in before_state.items() if k.startswith(prefix)]
        frozen_difference[prefix]=max(differences,default=0.)
        if any(x!=0 for x in differences):raise AssertionError('Frozen Q/C changed')
    torch.save(checkpoint_payload(model,opt,gen,completed,metadata=metadata,history=history),
               output/'checkpoint_last.pt')
    torch.save(qcr_state_dict(model),output/'qcr_model.pt')
    report=dict(contract=metadata,actual_extra_R_updates=completed,
        R_parameters=sum(p.numel() for p in model.theory_rg.parameters()),
        initial_R_Adam_steps=initial_steps,
        final_R_Adam_steps=[float(opt.state[p]['step']) for p in model.theory_rg.parameters()],
        frozen_max_parameter_difference=frozen_difference,history=history,snapshots=snapshots)
    for name,m in [('before',base),('after',model)]:
        report[name]={split:closure_metrics(predictions(m,d),reference_direct=train.numpy())
                      for split,d in [('train',train),('test',test)]}
        report[name]['fixed_points']=scalar_report(m,train,test,train_labels,test_labels,output/name)
    # Fresh C draws test whether better regression transfers beyond the cached targets.
    report['fresh_C_sampling_audit']={}
    for seed in (9001,9002,9003):
        fresh=collect_trajectories(base,vx.configurations,**options,sampling_seed=seed)
        direct=torch.from_numpy(fresh['direct'])
        report['fresh_C_sampling_audit'][str(seed)]={name:closure_metrics(predictions(m,direct),
            reference_direct=train.numpy()) for name,m in [('before',base),('after',model)]}
    comparison_figure(base,model,train,test,train_labels,test_labels,history,output)
    report['elapsed_seconds']=time.perf_counter()-started
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(config=config,contract=metadata,
        actual_epochs=completed,history=history,elapsed_seconds=report['elapsed_seconds']),indent=2)+'\n')
    print('DONE',output,report['elapsed_seconds'],flush=True)
    return report
