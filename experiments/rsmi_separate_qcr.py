"""Simultaneous QC/critic versus R gradients, replacing C auxiliaries by RSMI."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import torch
from recurrent_rg.checkpoints import build_model, checkpoint_payload, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data, file_fingerprint, evaluate_run
from recurrent_rg.training import evaluate, separate_qc_r_gradients
from recurrent_rg.losses import qcr_objective
from recurrent_rg.rsmi import EnvironmentCritic, region_offsets, sample_regions, kernel_probabilities, enumerated_infonce
from recurrent_rg.rsmi import binary_chain_joint_histograms, histogram_mutual_information
from recurrent_rg.rsmi import best_binary_pair_kernel, initialize_kernel_from_table
from recurrent_rg.rsmi import categorical_kernel_probabilities
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics,best_deterministic_binary_rule,categorical_lattice_mi_statistics,optimize_categorical_rule
from experiments.potts_frozen_qc_continue import pattern_function


def independent_component_gradients(total,closure,c_objective,q,c,r,critic=()):
    """Each trainable component gets its own objective, on one forward graph."""
    c=list(c)
    cgrads=torch.autograd.grad(c_objective,c,retain_graph=True,allow_unused=True)
    separate_qc_r_gradients(total,closure,list(q)+list(critic),list(r))
    for parameter,gradient in zip(c,cgrads):parameter.grad=gradient


def evaluate_information(model,critic,x,*,flip,pairs=64,repeats=8):
    gen=torch.Generator().manual_seed(818)
    values=[];shuffled=[]
    with torch.no_grad():
        for _ in range(repeats):
            v,e=sample_regions(x,min(32,len(x)),pairs,gen,buffer=1,flip=flip)
            probabilities=kernel_probabilities(model.rg_kernel,v)
            scores=critic(e)
            values.append(enumerated_infonce(probabilities,scores))
            perm=torch.stack([torch.randperm(pairs,generator=gen) for _ in range(len(e))])
            shuffled.append(enumerated_infonce(probabilities,scores.gather(1,perm[:,:,None].expand_as(scores))))
    return float(torch.cat(values).mean()),float(torch.cat(shuffled).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--config',type=Path,help='Standalone configuration; initialize a fresh model')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--theory-dim',type=int,help='Override latent dimension; all modules are newly initialized unless resuming')
    p.add_argument('--rsmi-weight',type=float,default=2.)
    p.add_argument('--estimator',choices=['infonce','histogram'],default='infonce')
    p.add_argument('--variance-weight',type=float,default=0.)
    p.add_argument('--color-weight',type=float,default=0.,help='Uniform-block label-preservation NLL weight; zero preserves historical behavior')
    p.add_argument('--variance-ratio',type=float,default=.25)
    p.add_argument('--variance-per-ensemble',action='store_true')
    p.add_argument('--mi-initialize',action='store_true',help='Initialize trainable C from the best empirical-MI binary rule')
    p.add_argument('--independent-c',action='store_true',help='C receives only MI and variance loss; Q only representation, R only closure')
    p.add_argument('--resume-checkpoint',type=Path,help='Resume this driver with identical objectives and RNG/Adam state')
    p.add_argument('--epochs',type=int,help='Total updates, including updates already completed when resuming')
    args=p.parse_args()
    if not math.isfinite(args.rsmi_weight) or args.rsmi_weight<=0:raise ValueError('Positive RSMI weight required')
    if not math.isfinite(args.variance_weight) or args.variance_weight<0:raise ValueError('Nonnegative variance weight required')
    if not math.isfinite(args.color_weight) or args.color_weight<0:raise ValueError('Nonnegative color weight required')
    if not 0<args.variance_ratio<=1:raise ValueError('Variance ratio must be in (0,1]')
    cfg=(RunConfig.load(args.config) if args.config else RunConfig.from_dict(json.loads((args.source/'manifest.json').read_text())['config']))
    if args.epochs is not None:
        if args.epochs<1:raise ValueError('epochs must be positive')
        cfg=replace(cfg,training=replace(cfg.training,epochs=args.epochs))
    if args.theory_dim is not None:
        if args.theory_dim<1:raise ValueError('theory-dim must be positive')
        cfg=replace(cfg,model=replace(cfg.model,theory_dim=args.theory_dim))
    cfg=replace(cfg,training=replace(cfg.training,gradient_mode='qc_r_separate'),
        loss=replace(cfg.loss,kernel_sign_weight=args.color_weight,kernel_neutral_weight=0.,kernel_distinct_alpha=0.,
            kernel_noncollapse_mode='variance_floor_ensemble' if args.variance_per_ensemble else 'variance_floor',kernel_variance_ratio=args.variance_ratio,
            weights=replace(cfg.loss.weights,kernel_noncollapse=args.variance_weight)))
    categorical=bool(cfg.model.categorical_states)
    if cfg.model.fixed_kernel or cfg.model.kernel_sampling not in ('bernoulli','categorical') or cfg.model.block_factor!=2 or cfg.model.kernel_layout!='nonoverlapping':
        raise ValueError('Requires learnable factor-two finite-alphabet probabilistic C')
    cfg.validate();t,e=cfg.training,cfg.evaluation
    exact=args.estimator=='histogram'
    if categorical and not exact:raise ValueError('Categorical experiment requires histogram MI')
    if args.mi_initialize and not exact:raise ValueError('MI initialization requires histogram estimator')
    sparse=exact and cfg.model.dimension==2
    torch.set_num_threads(t.threads);torch.manual_seed(t.seed)
    m=build_model(cfg.model,t.max_rg_steps);tx,vx=prepared_data(cfg);fn=pattern_function(cfg)
    m.detach_kernel_representation=args.independent_c
    _,env=region_offsets(cfg.model.dimension,buffer=1)
    # Extra critic initialization must not change QCR's sampling RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(t.seed+1);critic=EnvironmentCritic(len(env),64).double()
    qc=list(m.theory_encoder.parameters())+list(m.rg_kernel.parameters())
    r=list(m.theory_rg.parameters());cp=[] if exact else list(critic.parameters())
    opt=torch.optim.Adam(qc+r,lr=t.learning_rate);co=None if exact else torch.optim.Adam(cp,lr=t.learning_rate)
    statistics=categorical_lattice_mi_statistics if categorical else binary_lattice_mi_statistics if sparse else binary_chain_joint_histograms
    histograms=({split:statistics(x,flip=cfg.data.spin_flip_augment)
                 for split,x in [('train',tx.configurations),('test',vx.configurations)]} if exact else None)
    resume=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=True) if args.resume_checkpoint else None
    initialization=None
    if resume is not None:
        old=resume['metadata']
        old_config=json.loads(json.dumps(old['config']))
        old_config['training']['epochs']=cfg.training.epochs
        if old_config!=cfg.to_dict() or old['estimator']!=args.estimator or old.get('independent_c',False)!=args.independent_c or old['rsmi_weight']!=args.rsmi_weight:
            raise ValueError('Resume requires identical configuration, MI estimator/weight, and gradient routing')
        if t.epochs<=resume['epoch']:raise ValueError('Resume total epochs must exceed checkpoint epoch')
        initialization=old.get('data_driven_initialization')
        if bool(initialization)!=args.mi_initialize:raise ValueError('Resume initialization flag mismatch')
        m.load_state_dict(resume['model'],strict=False);opt.load_state_dict(resume['optimizer'])
        if co is not None:critic.load_state_dict(resume['critic']);co.load_state_dict(resume['critic_optimizer'])
    elif args.mi_initialize:
        table,initialization=(optimize_categorical_rule(histograms['train'],cfg.model.categorical_states,seed=t.seed) if categorical else
            (best_deterministic_binary_rule if sparse else best_binary_pair_kernel)(histograms['train']))
        initialization.update(initialize_kernel_from_table(m.rg_kernel,table))
        print('Data-driven MI initialization:',initialization,flush=True)
    visible_sites=2**cfg.model.dimension
    states=cfg.model.categorical_states or 2
    codes=(torch.arange(states**visible_sites)[:,None]//(states**torch.arange(visible_sites)))%states
    patterns=(torch.nn.functional.one_hot(codes,states).flatten(1).double() if categorical else (2*codes-1).double())[None]
    probabilities=categorical_kernel_probabilities if categorical else kernel_probabilities
    gen=torch.Generator().manual_seed(t.seed);rg=torch.Generator().manual_seed(t.seed+2)
    pairs=min(64,tx.configurations.shape[1],vx.configurations.shape[1]);systems=min(16,len(tx.configurations))
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(training_driver='rsmi_separate_qcr',rsmi_weight=args.rsmi_weight,rsmi_scales=[0],buffer=1,
        pairs=pairs,systems=systems,critic_hidden=64,critic_parameters=sum(p.numel() for p in cp),
        QCR_parameters=sum(p.numel() for p in qc+r),gradient_QC='grad(L_QC - rsmi_weight*I_NCE)',
        gradient_R='grad(5 L_commutator + 5 L_rollout); no feedback to QC',
        gradient_critic='grad(-rsmi_weight*I_NCE)',negatives='Within ensemble only',
        kernel_auxiliaries='All disabled; RSMI replaces C loss',
        labels_used_in_training=False,source=str(args.source) if args.source else None,config_source=str(args.config) if args.config else None,selection='Final checkpoint; one seed',
        note='Extra region batches from the same cache, not extra generated data. Empirical InfoNCE is not exact MI.')
    contract['estimator']=args.estimator
    contract['data_driven_initialization']=initialization
    contract['independent_c']=args.independent_c
    contract['kernel_auxiliaries']=f'Multiscale {cfg.loss.kernel_noncollapse_mode} weight={args.variance_weight}, ratio={args.variance_ratio}; uniform-color NLL weight={args.color_weight}; neutral/distinct disabled'
    if exact:
        contract.update(critic_hidden=None,gradient_critic=None,negatives=None,pairs=None,
            gradient_QC='grad(L_QC - rsmi_weight*I_histogram)',
            note='Exact MI of empirical per-ensemble 4x4 P(V,E), all cached samples/translations, with original flip augmentation. Not exact population MI. No critic or coupling labels.')
    if args.independent_c:
        contract['gradient_QC']='Q: representation losses only; C: -rsmi_weight*I + variance_weight*variance_floor + color_weight*uniform_color_NLL only'
    if sparse:
        contract['empirical_statistics']={split:hist.diagnostics for split,hist in histograms.items()}
        contract['note']='Exact full-shell empirical MI via compressed P(V|E), per-ensemble weights, all cached samples/translations; no critic or coupling labels. Large finite-sample bias is possible.'
    metadata=dict(config=cfg.to_dict(),data_sha256=file_fingerprint(cfg.data.cache),**contract)
    if resume is not None:
        if resume['metadata']['data_sha256']!=metadata['data_sha256']:raise ValueError('Resume data changed')
        metadata['resumed_from']=str(args.resume_checkpoint)
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    train_eval=tx.configurations[torch.linspace(0,len(tx.configurations)-1,min(e.systems,len(tx.configurations))).long()]
    test_eval=vx.configurations[torch.linspace(0,len(vx.configurations)-1,min(e.systems,len(vx.configurations))).long()]
    history=[] if resume is None else resume['history'];started=time.perf_counter()
    if resume is not None:
        gen.set_state(resume['batch_generator_state']);rg.set_state(resume['rsmi_generator_state']);torch.set_rng_state(resume['torch_rng_state'])

    def save(step,name):
        state=checkpoint_payload(m,opt,gen,step,metadata=metadata,history=history)
        state.update(rsmi_generator_state=rg.get_state())
        if co is not None:state.update(critic=critic.state_dict(),critic_optimizer=co.state_dict())
        torch.save(state,args.output/name)

    for step in range(1 if resume is None else resume['epoch']+1,t.epochs+1):
        m.train();critic.train()
        ids=torch.randperm(len(tx.configurations),generator=gen)[:t.system_batch]
        indices=torch.randperm(tx.configurations.shape[1],generator=gen)[:t.sample_batch]
        selected=tx.configurations[ids][:,indices].double();half=t.sample_batch//2
        a,b=selected[:,:half],selected[:,half:]
        batch=torch.cat((a,-a,b,-b),1) if cfg.data.spin_flip_augment else selected
        opt.zero_grad(set_to_none=True)
        if co is not None:co.zero_grad(set_to_none=True)
        base,terms=qcr_objective(m,batch,pattern_fn=fn,**cfg.loss.objective_kwargs())
        if exact:
            selected_hist=torch.randperm(len(tx.configurations),generator=rg)[:systems]
            table=probabilities(m.rg_kernel,patterns)[0]
            info=(histograms['train'].information(table)[selected_hist].mean() if sparse else
                  histogram_mutual_information(histograms['train'][selected_hist],table).mean())
        else:
            v,envs=sample_regions(tx.configurations,systems,pairs,rg,buffer=1,flip=cfg.data.spin_flip_augment)
            info=enumerated_infonce(kernel_probabilities(m.rg_kernel,v),critic(envs)).mean()
        total=base-args.rsmi_weight*info
        closure=cfg.loss.weights.commutator*terms['commutator']+cfg.loss.rollout_weight*terms['rollout']
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite at {step}')
        if args.independent_c:
            independent_component_gradients(total,closure,
                -args.rsmi_weight*info+args.variance_weight*terms['kernel_noncollapse']+args.color_weight*terms.get('kernel_sign',0.),
                m.theory_encoder.parameters(),m.rg_kernel.parameters(),r,cp)
        else:
            separate_qc_r_gradients(total,closure,qc+cp,r)
        opt.step()
        if co is not None:co.step()
        if step==1 or step%t.log_every==0 or step==t.epochs:
            row=dict(epoch=step)
            m.eval();critic.eval()
            for split,x,full in [('train',train_eval,tx.configurations),('test',test_eval,vx.configurations)]:
                metrics=evaluate(m,x,fn,e.samples,spin_flip_augment=cfg.data.spin_flip_augment,objective_kwargs=cfg.loss.objective_kwargs())
                if exact:
                    with torch.no_grad():
                        table=probabilities(m.rg_kernel,patterns)[0]
                        hist=histograms[split]
                        if sparse:
                            info=float(hist.information(table).mean());shuffled=0.
                        else:
                            info=float(histogram_mutual_information(hist,table).mean())
                            independent=hist.sum(2,keepdim=True)*hist.sum(1,keepdim=True)
                            shuffled=float(histogram_mutual_information(independent,table).mean())
                else:
                    info,shuffled=evaluate_information(m,critic,full,flip=cfg.data.spin_flip_augment,pairs=pairs)
                metrics.update(qcr_total=metrics['total'],rsmi=-info,rsmi_bits=info/math.log(2),shuffled_rsmi_bits=shuffled/math.log(2))
                metrics['total']-=args.rsmi_weight*info
                row.update({f'{split}_{key}':value for key,value in metrics.items()})
            history.append(row)
            (args.output/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(f'epoch {step}/{t.epochs}: test total={row["test_total"]:.5f}, I_{args.estimator}={row["test_rsmi_bits"]:.4f} bits',flush=True)
        if step%t.checkpoint_every==0:save(step,f'training_step_{step}.pt')
    save(t.epochs,'checkpoint_last.pt');torch.save(qcr_state_dict(m),args.output/'qcr_model.pt')
    (args.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),contract=contract,history=history,
        elapsed_seconds=time.perf_counter()-started),indent=2)+'\n')
    evaluate_run(args.output,output=args.output/'report')


if __name__=='__main__':main()
