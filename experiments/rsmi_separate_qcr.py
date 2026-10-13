"""Default Q/C/R-independent training: empirical MI + variance + color C loss."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import torch
from recurrent_rg.checkpoints import build_model, checkpoint_payload, qcr_state_dict, load_run, load_model_state
from recurrent_rg.checkpoints import isolated_rng, restore_checkpoint_rng
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
from recurrent_rg.empirical_mi import categorical_local_pair_mi_statistics


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


def microscopic_coupling_loss(encoder, batch, labels):
    """Explicit ablation only; labels never become targets for coarse scales."""
    predicted=encoder(batch)
    target=labels.detach().to(predicted).reshape(len(predicted),-1)
    if target.shape!=predicted.shape:raise ValueError('Coupling target shape mismatch')
    return (predicted-target).square().sum(-1).mean()


def train_component_losses(row, cfg, args):
    """Actual independently routed objectives; TEST never enters stopping."""
    w=cfg.loss.weights
    return dict(epoch=row['epoch'],
        component_Q=w.anchor*row['train_anchor']+
                    w.coarse_consistency*row['train_coarse_consistency']+
                    w.separation*row['train_separation'],
        component_C=args.rsmi_weight*row['train_rsmi']+
                    args.variance_weight*row['train_kernel_noncollapse']+
                    args.color_weight*row.get('train_kernel_sign',0.),
        component_R=w.commutator*row['train_commutator']+
                    cfg.loss.rollout_weight*row['train_rollout'])


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--config',type=Path,help='Standalone configuration; initialize a fresh model')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',help='Explicit execution device; CPU remains the historical default')
    p.add_argument('--benchmark-warmup',type=int,help='Measure real updates; omit this many first updates from steady timing, without adding extra training')
    p.add_argument('--reference-execution',action='store_true',help='Disable execution-only GPU accelerations for a same-recipe comparison')
    p.add_argument('--mi-cache-dir',type=Path,help='GPU CPU-statistics cache directory; defaults to .qcr_mi_cache beside input data')
    p.add_argument('--theory-dim',type=int,help='Override latent dimension; all modules are newly initialized unless resuming')
    p.add_argument('--rsmi-weight',type=float,default=2.)
    p.add_argument('--estimator',choices=['infonce','histogram'],default='histogram')
    p.add_argument('--environment',choices=['full_shell','local_pairs'],default='full_shell')
    p.add_argument('--variance-weight',type=float,default=2.)
    p.add_argument('--color-weight',type=float,default=1.,help='Uniform-block label-preservation NLL weight')
    p.add_argument('--variance-ratio',type=float,default=1.)
    p.add_argument('--variance-per-ensemble',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mi-initialize',action='store_true',help='Initialize trainable C from the best empirical-MI binary rule')
    p.add_argument('--independent-c',action=argparse.BooleanOptionalAction,default=True,help='C: MI/variance/color; Q: representation; R: closure')
    p.add_argument('--allow-nondefault-recipe',action='store_true',help='Explicit opt-in for a user-requested ablation; never enable automatically')
    p.add_argument('--resume-checkpoint',type=Path,help='Resume this driver with identical objectives and RNG/Adam state')
    p.add_argument('--epochs',type=int,help='Total updates, including updates already completed when resuming')
    p.add_argument('--max-wall-seconds',type=float,help='Optional setup+training wall limit; finish the current step and save actual progress')
    p.add_argument('--max-training-seconds',type=float,help='Optional training-only wall limit after completed setup; save actual progress, never reduce configured epochs')
    p.add_argument('--joint-train-plateau',action='store_true',
                   help='Stop complete default Q/C/R training on separate TRAIN-only objective plateaus; no LR or gradient changes')
    p.add_argument('--freeze-c-checkpoint',type=Path,help='Explicit ablation: warm-start Q/C/R weights and freeze this learned probabilistic C; reset Adam')
    p.add_argument('--affine-pca-init',action='store_true',help='Explicit frozen-C ablation: initialize trainable affine decoder from TRAIN histograms only')
    p.add_argument('--q-coupling-weight',type=float,default=0.,help='Explicit supervised ablation: microscopic Q(D) fits coupling labels only')
    p.add_argument('--q-only-weight-sweep',action='store_true',help='Explicit ablation: frozen cached source C, fresh paired Q, separation weights 10/30/100')
    p.add_argument('--q-only-candidates',action='store_true',help='Explicit paired Q representation-objective ablation')
    p.add_argument('--q-balanced-reconstruction',action='store_true',help='Explicit paired Q-only affine plus signal-balanced reconstruction ablation; source C/R frozen')
    p.add_argument('--q-last-layer-audit',action='store_true',help='Explicit ablation: freeze Q backbone/C/R and train only final Q linear layer')
    p.add_argument('--q-head-structure-audit',action='store_true',help='Explicit ablation: linear/whitened/monotone Q heads on frozen features')
    p.add_argument('--q-feature-bottleneck',action='store_true',help='Explicit ablation: frozen features, reconstruction bottleneck without geometry')
    p.add_argument('--q-original-plateau',action='store_true',help='Explicit ablation: resume original full Q with train-only plateau LR schedule')
    p.add_argument('--q-initialization-pilot',action='store_true',help='Explicit paired Q-only initialization ablation')
    p.add_argument('--q-unfolding',action='store_true',help='Explicit Q-only graph-coordinate / linear distribution reconstruction ablation')
    p.add_argument('--r-only-continue',action='store_true',help='Explicit ablation: freeze source Q/C, continue existing R and Adam on saved TRAIN trajectories; --epochs is additional R updates')
    p.add_argument('--r-only-refit',action='store_true',help='Explicit ablation: refit original R to a paired balanced-Q checkpoint, Q/C frozen and fresh R Adam')
    p.add_argument('--r-only-plateau',action='store_true',help='Explicit ablation: freeze final Q/C, preserve online half-subset targets and R Adam, stop on a TRAIN plateau; --epochs is a safety cap on additional R updates')
    p.add_argument('--frozen-cache',type=Path,help='Existing Q-only run containing frozen_levels.pt')
    return p


def validate_recipe(args, cfg):
    """Fail closed: a historical flag must not silently change the default recipe."""
    if args.allow_nondefault_recipe:
        return
    expected=dict(estimator='histogram',environment='full_shell',rsmi_weight=2.,
                  variance_weight=2.,color_weight=1.,variance_ratio=1.,
                  variance_per_ensemble=True,independent_c=True,mi_initialize=False)
    changed=[key for key,value in expected.items() if getattr(args,key)!=value]
    if cfg.model.kernel_initialization!='random':changed.append('kernel_initialization')
    if cfg.model.fixed_kernel:changed.append('fixed_kernel')
    if cfg.model.kernel_sampling not in ('bernoulli','categorical'):changed.append('kernel_sampling')
    modules={'Q','R'} if args.freeze_c_checkpoint else {'Q','C','R'}
    if set(cfg.training.train_modules)!=modules:changed.append('train_modules')
    if changed:
        raise ValueError('Locked default recipe changed: '+', '.join(changed)+
                         '; use --allow-nondefault-recipe only for an explicitly requested ablation')


def default_representation(cfg, raw_config, *, continuing=False):
    """Fresh runs default to affine reconstruction; preserve explicit/history choices."""
    model=raw_config.get('model',{})
    if continuing or 'affine_reconstruction' in model or model.get('reconstruction_hidden',0):
        return cfg
    return replace(cfg,model=replace(cfg.model,affine_reconstruction=True))


def selected_mi_execution(execution, sparse, resume):
    """Fresh optimized runs may select supports; exact old resumes stay old."""
    if not execution['enabled'] or not sparse:
        return False
    if resume is None:
        return True
    return bool(resume.get('metadata',{}).get('execution_optimizations',{}).get(
        'selected_empirical_mi',False))


def validate_training_clock(args):
    if args.max_training_seconds is None:
        return
    if not math.isfinite(args.max_training_seconds) or args.max_training_seconds<=0:
        raise ValueError('max-training-seconds must be finite and positive')
    if args.max_wall_seconds is not None:
        raise ValueError('Choose either max-training-seconds or max-wall-seconds, not both')
    if args.source is not None:
        raise ValueError('Training-only clock requires a standalone default config')


def main(argv=None):
    wall_started=time.perf_counter()
    args=build_parser().parse_args(argv)
    validate_training_clock(args)
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type=='cpu' and device.index is not None):
        raise ValueError('Supported execution devices are cpu and cuda:<index>')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA was explicitly requested but is unavailable')
        device=torch.device('cuda',0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        # Historical optional drivers have not been audited for device/RNG routing.
        cuda_ablations=('source','allow_nondefault_recipe','mi_initialize','freeze_c_checkpoint',
            'affine_pca_init','q_coupling_weight','q_only_weight_sweep','q_only_candidates',
            'q_balanced_reconstruction','q_last_layer_audit','q_head_structure_audit',
            'q_feature_bottleneck','q_original_plateau','q_initialization_pilot',
            'q_unfolding','r_only_continue','r_only_refit','r_only_plateau','joint_train_plateau')
        if any(getattr(args,key) for key in cuda_ablations) or args.estimator!='histogram' or not args.independent_c:
            raise ValueError('CUDA execution currently supports fresh/default QCR or identical-device resume, not optional historical ablations')
    if args.benchmark_warmup is not None and args.benchmark_warmup<0:
        raise ValueError('benchmark-warmup must be nonnegative')
    if args.mi_cache_dir and (device.type!='cuda' or args.reference_execution):
        raise ValueError('mi-cache-dir requires optimized CUDA execution')
    if args.joint_train_plateau:
        joint_conflicts=('r_only_plateau','r_only_refit','r_only_continue',
            'q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
            'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck',
            'q_original_plateau','q_initialization_pilot','q_unfolding',
            'freeze_c_checkpoint','affine_pca_init','q_coupling_weight','frozen_cache')
        if any(getattr(args,key) for key in joint_conflicts):
            raise ValueError('Joint TRAIN plateau cannot be combined with component-only ablations')
    if args.r_only_plateau:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
                   'q_initialization_pilot','q_unfolding','r_only_continue','r_only_refit',
                   'theory_dim','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in conflicts):
            raise ValueError('R-only plateau continuation requires a final source run and no other ablations')
        if args.epochs is not None and args.epochs < 1:
            raise ValueError('Positive update count required')
        loaded_source=load_run(args.source/'checkpoint_last.pt')
        saved_metadata=loaded_source.payload['metadata']
        source_config=RunConfig.from_dict(saved_metadata['source_config']
            if saved_metadata.get('training_driver') == 'frozen_qc_r_plateau' else loaded_source.config)
        validate_recipe(args,source_config)
        from experiments.frozen_qc_r_plateau import run
        return run(args.source,args.output,30000 if args.epochs is None else args.epochs,
                   max_wall_seconds=args.max_wall_seconds)
    if args.q_balanced_reconstruction:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck',
                   'q_original_plateau','q_initialization_pilot','q_unfolding',
                   'r_only_continue','r_only_refit','theory_dim','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args, key) for key in conflicts):
            raise ValueError('Balanced Q-only pilot requires --source and no other ablations')
        if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
            raise ValueError('Positive finite wall limit required')
        if args.epochs is not None and args.epochs < 1:
            raise ValueError('Positive update count required')
        source_config=RunConfig.from_dict(json.loads((args.source/'manifest.json').read_text())['config'])
        validate_recipe(args,source_config)
        from experiments.q_balanced_reconstruction import run
        return run(args.source,args.output,args.epochs or 400,
                   max_wall_seconds=args.max_wall_seconds or 540.)
    if args.r_only_refit:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
                   'q_initialization_pilot','q_unfolding','r_only_continue','theory_dim','frozen_cache',
                   'mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in conflicts):
            raise ValueError('R-only refit requires a Q-only checkpoint and no other ablations')
        if args.epochs is not None and args.epochs<1:
            raise ValueError('Positive update count required')
        if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0):
            raise ValueError('Positive finite wall limit required')
        source_config=RunConfig.from_dict(load_run(args.source).config)
        validate_recipe(args,replace(source_config,training=replace(source_config.training,train_modules=('Q','C','R'))))
        from experiments.frozen_balanced_qc_r_refit import run
        return run(args.source,args.output,args.epochs or 6000,
                   max_wall_seconds=args.max_wall_seconds or 540.)
    if args.r_only_continue:
        other_ablations=('freeze_c_checkpoint','affine_pca_init','resume_checkpoint','theory_dim',
            'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_last_layer_audit',
            'q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
            'q_initialization_pilot','q_unfolding','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in other_ablations):
            raise ValueError('R-only continuation requires --source and no other ablations or overrides')
        from experiments.frozen_qc_r_continue import run
        return run(args.source,args.output,args.epochs or 6000,max_wall_seconds=args.max_wall_seconds)
    if args.freeze_c_checkpoint and (args.resume_checkpoint or not args.config or not args.independent_c or args.mi_initialize or args.q_coupling_weight):
        raise ValueError('Frozen C requires a standalone QR config, independent gradients, and no resume/kernel search/supervision')
    if args.affine_pca_init and not args.freeze_c_checkpoint:
        raise ValueError('PCA decoder initialization is an explicit frozen-C QR ablation')
    if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0):
        raise ValueError('Positive finite wall limit required')
    if args.q_unfolding:
        if args.source is None or args.frozen_cache is None or args.q_initialization_pilot or args.q_original_plateau or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q unfolding ablation requires source/cache and no other ablations')
        from experiments.q_unfolding import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_initialization_pilot:
        if args.source is None or args.frozen_cache is None or args.q_original_plateau or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q initialization pilot requires source/cache and no other ablations')
        from experiments.q_initialization_pilot import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 400)
    if args.q_original_plateau:
        if args.source is None or args.frozen_cache is None or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Original Q plateau audit requires source/cache and no other ablations')
        from experiments.q_original_plateau import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 30000)
    if args.q_feature_bottleneck:
        if args.source is None or args.frozen_cache is None or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Feature bottleneck requires source/cache and no other ablations')
        from experiments.q_feature_bottleneck import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 2400)
    if args.q_head_structure_audit:
        if args.source is None or args.frozen_cache is None or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Head structure audit requires source/cache and no other ablations')
        from experiments.q_head_structure_audit import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_last_layer_audit:
        if args.source is None or args.frozen_cache is None or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Last-layer audit requires source/cache and no other training ablations')
        from experiments.q_last_layer_audit import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_only_candidates:
        if args.source is None or args.frozen_cache is None or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Candidate sweep requires source/cache and no other ablations')
        from experiments.q_only_candidates import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 2400)
    if args.q_only_weight_sweep:
        if args.source is None or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q-only sweep needs a source, no resume or supervision')
        from experiments.q_only_cached_sweep import run
        return run(args.source,args.output,args.epochs or 240)
    if not math.isfinite(args.q_coupling_weight) or args.q_coupling_weight<0:
        raise ValueError('Nonnegative finite Q coupling weight required')
    if args.q_coupling_weight and args.resume_checkpoint:
        raise ValueError('Supervised pilot currently supports fresh runs only')
    if not math.isfinite(args.rsmi_weight) or args.rsmi_weight<=0:raise ValueError('Positive RSMI weight required')
    if not math.isfinite(args.variance_weight) or args.variance_weight<0:raise ValueError('Nonnegative variance weight required')
    if not math.isfinite(args.color_weight) or args.color_weight<0:raise ValueError('Nonnegative color weight required')
    if not 0<args.variance_ratio<=1:raise ValueError('Variance ratio must be in (0,1]')
    raw_config=(json.loads(args.config.read_text()) if args.config else
                json.loads((args.source/'manifest.json').read_text())['config'])
    cfg=(RunConfig.load(args.config) if args.config else RunConfig.from_dict(raw_config))
    cfg=default_representation(cfg,raw_config,continuing=bool(args.resume_checkpoint or args.freeze_c_checkpoint))
    if args.affine_pca_init and not cfg.model.affine_reconstruction:
        raise ValueError('PCA initialization requires affine_reconstruction=true')
    validate_recipe(args,cfg)
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
    if args.environment!='full_shell' and (not categorical or cfg.model.dimension!=2 or args.estimator!='histogram' or args.mi_initialize):
        raise ValueError('Local-pair MI requires 2D categorical histogram training without kernel search')
    if cfg.model.fixed_kernel or cfg.model.kernel_sampling not in ('bernoulli','categorical') or cfg.model.kernel_layout!='nonoverlapping':
        raise ValueError('Requires learnable nonoverlapping finite-alphabet probabilistic C')
    block=cfg.model.block_factor
    if block!=2 and (cfg.model.dimension!=2 or args.estimator!='histogram' or args.environment!='full_shell' or args.mi_initialize):
        raise ValueError('Non-two blocks require 2D full-shell histogram MI without kernel search')
    if (cfg.model.categorical_states or 2)**(block**cfg.model.dimension)>262144:
        raise ValueError('Empirical MI kernel table exceeds 262144 input patterns')
    cfg.validate();t,e=cfg.training,cfg.evaluation
    from recurrent_rg.learning_rate import LearningRatePolicy
    lr_policy=LearningRatePolicy(t)
    plateau_settings=dict(minimum=4000,window=2000,confirmations=3,check_every=200,
                          rtol=.01,rebound_rtol=.05,loss_floor=1e-3)
    if args.joint_train_plateau:
        conflicts=('freeze_c_checkpoint','affine_pca_init','q_coupling_weight',
                   'allow_nondefault_recipe')
        if (any(getattr(args,key) for key in conflicts) or not args.independent_c or
                args.estimator!='histogram' or set(t.train_modules)!={'Q','C','R'} or
                not cfg.model.affine_reconstruction or cfg.loss.balanced_reconstruction_weight or cfg.loss.isotropy_weight or
                t.log_every!=plateau_settings['check_every'] or
                t.checkpoint_every!=plateau_settings['check_every']):
            raise ValueError('Joint plateau requires the complete default affine Q/C/R recipe and log/checkpoint_every=200')
    exact=args.estimator=='histogram'
    if categorical and not exact:raise ValueError('Categorical experiment requires histogram MI')
    if args.mi_initialize and not exact:raise ValueError('MI initialization requires histogram estimator')
    sparse=exact and cfg.model.dimension==2
    torch.set_num_threads(t.threads);torch.manual_seed(t.seed)
    m=build_model(cfg.model,t.max_rg_steps).to(device);tx,vx=prepared_data(cfg);fn=pattern_function(cfg)
    if device.type=='cuda' and cfg.loss.balanced_reconstruction_weight:
        raise ValueError('Balanced historical ablation is not yet CUDA-adapted')
    m.detach_kernel_representation=args.independent_c
    from recurrent_rg.execution import configure_cuda_execution, resident_training_data
    from recurrent_rg.execution import reference_binary_pattern_distribution, select_training_batch
    execution=configure_cuda_execution(m,device,reference=args.reference_execution,
                                      independent_c=args.independent_c)
    if args.reference_execution and not categorical:
        fn=reference_binary_pattern_distribution
    data_sha256=file_fingerprint(cfg.data.cache)
    frozen_source=None
    if args.freeze_c_checkpoint:
        frozen_source=load_run(args.freeze_c_checkpoint)
        if frozen_source.model_config!=cfg.model or frozen_source.config['training']['max_rg_steps']!=t.max_rg_steps:
            raise ValueError('Frozen-C warm start requires the same Q/C/R architecture and scales')
        old_data=frozen_source.payload.get('metadata',{}).get('data_sha256')
        if old_data is None:
            old_data=file_fingerprint(frozen_source.config['data']['cache'])
        if old_data!=file_fingerprint(cfg.data.cache):
            raise ValueError('Frozen-C warm start data fingerprint differs')
        m.load_state_dict(qcr_state_dict(frozen_source.model),strict=False)
        for parameter in m.rg_kernel.parameters():parameter.requires_grad_(False)
    _,env=region_offsets(cfg.model.dimension,block=block,buffer=1)
    # Extra critic initialization must not change QCR's sampling RNG stream.
    with isolated_rng('cpu',t.seed+1):
        critic=EnvironmentCritic(len(env),64).double().to(device)
    qc=list(m.theory_encoder.parameters())+([] if frozen_source else list(m.rg_kernel.parameters()))
    r=list(m.theory_rg.parameters());cp=[] if exact else list(critic.parameters())
    opt=torch.optim.Adam(qc+r,lr=t.learning_rate);co=None if exact else torch.optim.Adam(cp,lr=t.learning_rate)
    statistics=categorical_lattice_mi_statistics if categorical else binary_lattice_mi_statistics if sparse else binary_chain_joint_histograms
    if args.environment=='local_pairs':statistics=categorical_local_pair_mi_statistics
    statistics_kwargs={'block':block} if sparse and args.environment=='full_shell' else {}
    histograms=None
    execution['mi_statistics_cache']={}
    resume=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=True) if args.resume_checkpoint else None
    if exact:
        histograms={}
        for split,x in [('train',tx.configurations),('test',vx.configurations)]:
            build=lambda x=x:statistics(x,flip=cfg.data.spin_flip_augment,**statistics_kwargs)
            if execution['enabled']:
                from recurrent_rg.mi_cache import load_or_build_mi_statistics, lattice_statistics_cache_key
                key_data=lattice_statistics_cache_key(cfg,data_sha256,split,x,env,
                    statistics.__name__,statistics_kwargs,environment=args.environment)
                cache_dir=args.mi_cache_dir or Path(cfg.data.cache).parent/'.qcr_mi_cache'
                hist,cache_record=load_or_build_mi_statistics(cache_dir,key_data,build)
                histograms[split]=hist
                execution['mi_statistics_cache'][split]=cache_record
                print(f'MI statistics {split}: '+('cache hit' if cache_record['hit'] else 'built and cached'),flush=True)
            else:
                histograms[split]=build()
    # Histogram construction/compression stays on CPU; only its existing tensors
    # move once. No change to the empirical MI estimator or sampled ensembles.
    execution['selected_empirical_mi']=selected_mi_execution(execution,sparse,resume)
    if histograms is not None and execution['selected_empirical_mi']:
        histograms['train'].prepare_selection()
    if histograms is not None:
        histograms={split:hist.to(device) for split,hist in histograms.items()}
    train_configurations,residency=resident_training_data(tx.configurations,device,
                                                        enabled=execution['enabled'])
    execution['training_data']=residency
    if resume is not None and bool(resume['metadata'].get('joint_train_plateau'))!=args.joint_train_plateau:
        raise ValueError('Exact resume must preserve the joint TRAIN plateau stopping convention')
    initialization=None
    if resume is not None:
        old=resume['metadata']
        if device.type=='cuda':
            # Older CUDA checkpoints predate these accelerations. Preserve their
            # original execution via --reference-execution for an exact resume.
            old_reference=old.get('execution_optimizations',{}).get('reference',True)
            if old_reference!=args.reference_execution:
                raise ValueError('Exact CUDA resume must preserve reference-execution; '
                                 'older unoptimized CUDA checkpoints require --reference-execution')
        old_config=RunConfig.from_dict(old['config']).to_dict()
        old_config['training']['epochs']=cfg.training.epochs
        if old_config!=cfg.to_dict() or old['estimator']!=args.estimator or old.get('environment_mode','full_shell')!=args.environment or old.get('independent_c',False)!=args.independent_c or old['rsmi_weight']!=args.rsmi_weight:
            raise ValueError('Resume requires identical configuration, MI estimator/weight, and gradient routing')
        if t.epochs<=resume['epoch']:raise ValueError('Resume total epochs must exceed checkpoint epoch')
        initialization=old.get('data_driven_initialization')
        if bool(initialization)!=args.mi_initialize:raise ValueError('Resume initialization flag mismatch')
        load_model_state(m,resume);opt.load_state_dict(resume['optimizer'])
        if co is not None:critic.load_state_dict(resume['critic']);co.load_state_dict(resume['critic_optimizer'])
        lr_policy.validate_resume(resume,opt)
        if co is not None:lr_policy.validate_resume(resume,co)
    elif args.mi_initialize:
        table,initialization=(optimize_categorical_rule(histograms['train'],cfg.model.categorical_states,seed=t.seed) if categorical else
            (best_deterministic_binary_rule if sparse else best_binary_pair_kernel)(histograms['train']))
        initialization.update(initialize_kernel_from_table(m.rg_kernel,table))
        print('Data-driven MI initialization:',initialization,flush=True)
    visible_sites=block**cfg.model.dimension
    states=cfg.model.categorical_states or 2
    codes=(torch.arange(states**visible_sites,device=device)[:,None]//(states**torch.arange(visible_sites,device=device)))%states
    patterns=(torch.nn.functional.one_hot(codes,states).flatten(1).double() if categorical else (2*codes-1).double())[None]
    probabilities=categorical_kernel_probabilities if categorical else kernel_probabilities
    frozen_information=None
    if frozen_source:
        with torch.no_grad():
            fixed_table=probabilities(m.rg_kernel,patterns)[0]
            frozen_information={split:float(hist.information(fixed_table).mean() if sparse else
                                histogram_mutual_information(hist,fixed_table).mean())
                                for split,hist in histograms.items()}
    gen=torch.Generator().manual_seed(t.seed);rg=torch.Generator().manual_seed(t.seed+2)
    pairs=min(64,tx.configurations.shape[1],vx.configurations.shape[1]);systems=min(16,len(tx.configurations))
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(training_driver='rsmi_separate_qcr',rsmi_weight=args.rsmi_weight,rsmi_scales=[0],buffer=1,
        pairs=pairs,systems=systems,critic_hidden=64,critic_parameters=sum(p.numel() for p in cp),
        QCR_parameters=sum(p.numel() for p in m.theory_encoder.parameters())+sum(p.numel() for p in m.rg_kernel.parameters())+sum(p.numel() for p in r),gradient_QC='grad(L_QC - rsmi_weight*I_NCE)',
        gradient_R='grad(5 L_commutator + 5 L_rollout); no feedback to QC',
        gradient_critic='grad(-rsmi_weight*I_NCE)',negatives='Within ensemble only',
        kernel_auxiliaries='All disabled; RSMI replaces C loss',
        labels_used_in_training=bool(args.q_coupling_weight),source=str(args.source) if args.source else None,config_source=str(args.config) if args.config else None,selection='Final checkpoint; one seed',
        note='Extra region batches from the same cache, not extra generated data. Empirical InfoNCE is not exact MI.')
    contract['estimator']=args.estimator
    contract['environment_mode']=args.environment
    contract['data_driven_initialization']=initialization
    contract['independent_c']=args.independent_c
    contract['execution_device']=str(device)
    contract['precision']='torch.float64'
    contract['execution_optimizations']=execution
    contract['learning_rate_policy']=lr_policy.description()
    contract['device_note']='CPU histogram construction; model, selected batches, kernel table and MI tensor algebra on execution_device. Reports reload on CPU.'
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
        if args.environment=='local_pairs':contract['note']='Mean of four buffered two-site environment MI objectives, not full-shell joint MI. Finite-cache bias must be audited independently.'
    if frozen_source:
        contract['gradient_QC']='Q: representation only; C: frozen learned weights; R: one-step/multistep closure only'
        contract['frozen_C']=dict(checkpoint=str(frozen_source.checkpoint),
            checkpoint_sha256=file_fingerprint(frozen_source.checkpoint),
            warm_start='Q/C/R weights; Adam reset; affine decoder optionally reinitialized',
            stochastic_sampling='Fresh independent C draws at each training step, not a frozen trajectory cache',
            empirical_MI='Constant diagnostic only; no C updates',labels_used=False)
        contract['trainable_parameters']=sum(p.numel() for p in qc+r)
    metadata=dict(config=cfg.to_dict(),data_sha256=data_sha256,**contract)
    if args.joint_train_plateau:
        metadata['joint_train_plateau']=dict(**plateau_settings,
            stopping_split='TRAIN only; TEST and critical exponents not used',
            panel='Fixed evenly spaced TRAIN ensembles; fixed half-subsets; C sampling seed12345; full TRAIN empirical MI',
            objectives='Q: affine reconstruction+subset; C: -2 MI+2 variance+color; R: 5 one-step+5 all-horizon rollout',
            optimizer=('Original constant Adam learning rate; no scheduler, freezing or gradient changes'
                       if lr_policy.kind=='constant' else
                       f'Original Adam parameters/moments with deterministic cumulative {lr_policy.kind} learning rates; no freezing or gradient changes'),
            interpretation='Operational local plateau, not global optimality or exact physical scaling')
        if resume is not None and resume['metadata'].get('joint_train_plateau')!=metadata['joint_train_plateau']:
            raise ValueError('Joint plateau exact resume requires identical stopping settings')
    if cfg.model.affine_reconstruction:
        pca_initialization=None
        if args.affine_pca_init:
            from recurrent_rg.representation import initialize_affine_decoder_pca_
            with isolated_rng(device,t.seed+9000), torch.no_grad():
                hist=[]
                for x in tx.configurations:
                    hist.extend(fn(level) for level in m.coarse_grain(x[None].to(device=device,dtype=torch.float64)))
                pca_initialization=initialize_affine_decoder_pca_(m.theory_encoder,torch.cat(hist))
                pca_initialization.update(fitting_split='train',all_scales=True,sampling_seed=t.seed+9000)
            print('TRAIN-only affine PCA initialization:',pca_initialization,flush=True)
        elif resume is None and not frozen_source:
            # Only a bias calibration from TRAIN distributions at random C;
            # no kernel/encoder pretraining, labels, or change to training RNG.
            with isolated_rng(device,t.seed+9000), torch.no_grad():
                levels=m.coarse_grain(tx.configurations[:min(32,len(tx.configurations))].to(device=device,dtype=torch.float64))
                center=torch.stack([fn(level).mean(0) for level in levels]).mean(0)
                m.theory_encoder.distribution_decoder.bias.copy_(center)
        metadata['q_reconstruction']=dict(
            objective='Shared affine probability reconstruction sum-bin MSE at all scales; replaces geometric distances',
            weight=cfg.loss.weights.separation,parameters=sum(p.numel() for p in m.theory_encoder.distribution_decoder.parameters()),
            gauge='Each decoder column norm1, projected after each optimizer update; no full-rank or isotropy constraint',
            initialization='Random decoder direction; bias=train-only mean motif distribution at initially random C; preserves sampling RNG',
            gradient='Q and auxiliary decoder only; target detached; default C/R objectives unchanged',
            logging='separation/data_geometry_level_* record affine reconstruction MSE; reconstruction_mse_level_* explicit')
        if pca_initialization:
            metadata['q_reconstruction']['initialization']='TRAIN-only histogram PCA direction and mean at the frozen learned C; no Q pretraining'
            metadata['q_reconstruction']['PCA']=pca_initialization
    if cfg.model.reconstruction_hidden:
        metadata['q_reconstruction']=dict(
            objective='Shared decoder JS to detached same-scale local-pattern histograms; replaces geometry at every scale',
            weight=cfg.loss.weights.separation,hidden=cfg.model.reconstruction_hidden,
            parameters=sum(p.numel() for p in m.theory_encoder.distribution_decoder.parameters()),
            gradient='Q and auxiliary decoder only; C/R objectives unchanged',
            logging='separation/data_geometry_level_* now record reconstruction JS, not distance loss')
    if cfg.loss.balanced_reconstruction_weight:
        if not args.independent_c or args.q_coupling_weight or args.affine_pca_init:
            raise ValueError('Balanced full-QCR test requires independent gradients, no labels/PCA initialization')
        metadata['q_balanced_reconstruction']=dict(
            weight=cfg.loss.balanced_reconstruction_weight, additive_to_original_MSE=True,
            max_directions=cfg.model.theory_dim, signal_noise_threshold=5., max_weight_ratio=20.,
            fitting_split='All TRAIN ensembles; no TEST or coupling labels',
            refresh_every=t.checkpoint_every, sampling_seed=9191,
            gradient='Q/affine decoder only; current-C targets and metric detached',
            initialization='No change to model initialization; no PCA weight initialization',
            metric='Same signal-gated observable covariance metric as paired Q-only pilot; '
                   'refreshed periodically because C is now trainable')
    if args.q_coupling_weight:
        for ds in (tx,vx):
            if ds.couplings is None or ds.couplings.reshape(len(ds.configurations),-1).shape[1]!=cfg.model.theory_dim:
                raise ValueError('Supervision requires coupling dimension equal to dim g')
        metadata['q_supervision']=dict(weight=args.q_coupling_weight,scale=0,target='raw coupling labels, Q(full microscopic batch)',gradient='Q only; no labels for coarse levels or C/R')
    if resume is not None:
        if resume['metadata']['data_sha256']!=metadata['data_sha256']:raise ValueError('Resume data changed')
        metadata['resumed_from']=str(args.resume_checkpoint)
    metric_history=[]
    if cfg.loss.balanced_reconstruction_weight:
        from recurrent_rg.balanced_reconstruction import refresh_training_metrics
        if resume is not None:
            if not hasattr(m,'balanced_reconstruction_metrics'):
                raise ValueError('Balanced exact resume requires saved TRAIN metric state')
        else:
            refresh_training_metrics(m,tx.configurations,fn,samples=t.sample_batch,
                                     fitted_epoch=0)
        metric_history.append(dict(fitted_epoch=m.balanced_metric_fitted_epoch,
            scales=[metric.description() for metric in m.balanced_reconstruction_metrics]))
        print('TRAIN-only balanced metric ranks: '+str([metric.axes.shape[1]
              for metric in m.balanced_reconstruction_metrics]),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if metric_history:
        (args.output/'balanced_metric_history.json').write_text(json.dumps(metric_history,indent=2)+'\n')
    train_eval=tx.configurations[torch.linspace(0,len(tx.configurations)-1,min(e.systems,len(tx.configurations))).long()]
    test_eval=vx.configurations[torch.linspace(0,len(vx.configurations)-1,min(e.systems,len(vx.configurations))).long()]
    history=[] if resume is None else resume['history'];started=time.perf_counter()
    plateau_rows=[] if resume is None else resume.get('joint_plateau_rows',[])
    if args.joint_train_plateau and resume is not None and not plateau_rows:
        raise ValueError('Missing joint plateau history in resume checkpoint')
    plateau_diagnostics=None
    if resume is not None:
        gen.set_state(resume['batch_generator_state']);rg.set_state(resume['rsmi_generator_state'])
        restore_checkpoint_rng(resume,device=device)

    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)

    timing_enabled=args.benchmark_warmup is not None
    step_seconds=[];logging_seconds=0.;checkpoint_seconds=0.
    if timing_enabled or args.max_training_seconds is not None:synchronize()
    setup_seconds=time.perf_counter()-wall_started
    training_started=time.perf_counter()

    def save(step,name):
        state=checkpoint_payload(m,opt,gen,step,metadata=metadata,history=history)
        state.update(rsmi_generator_state=rg.get_state(),
                     lr_scheduler_state=lr_policy.checkpoint_state(step))
        if args.joint_train_plateau:
            state.update(joint_plateau_rows=plateau_rows,
                         joint_plateau_diagnostics=plateau_diagnostics)
        if co is not None:state.update(critic=critic.state_dict(),critic_optimizer=co.state_dict())
        torch.save(state,args.output/name)

    completed_epoch=0 if resume is None else resume['epoch']
    stopped_by_wall_limit=False
    stopped_by_training_limit=False
    stopped_by_train_plateau=False
    for step in range(1 if resume is None else resume['epoch']+1,t.epochs+1):
        if cfg.loss.balanced_reconstruction_weight and step>1 and (step-1)%t.checkpoint_every==0:
            refresh_training_metrics(m,tx.configurations,fn,samples=t.sample_batch,
                                     fitted_epoch=step-1)
            metric_history.append(dict(fitted_epoch=step-1,
                scales=[metric.description() for metric in m.balanced_reconstruction_metrics]))
            (args.output/'balanced_metric_history.json').write_text(json.dumps(metric_history,indent=2)+'\n')
            print(f'Balanced metric refreshed at {step-1}; TRAIN ranks '+str([
                metric.axes.shape[1] for metric in m.balanced_reconstruction_metrics]),flush=True)
        if timing_enabled:synchronize()
        update_started=time.perf_counter()
        # Constant uses the historical Adam path without even rewriting LR.
        # Scheduling uses cumulative update numbers, not elapsed wall time.
        if lr_policy.kind!='constant':
            lr_policy.apply_update(opt,step)
            if co is not None:lr_policy.apply_update(co,step)
        m.train();critic.train()
        ids=torch.randperm(len(tx.configurations),generator=gen)[:t.system_batch]
        indices=torch.randperm(tx.configurations.shape[1],generator=gen)[:t.sample_batch]
        selected=select_training_batch(train_configurations,ids,indices,device);half=t.sample_batch//2
        a,b=selected[:,:half],selected[:,half:]
        batch=torch.cat((a,-a,b,-b),1) if cfg.data.spin_flip_augment else selected
        opt.zero_grad(set_to_none=True)
        if co is not None:co.zero_grad(set_to_none=True)
        base,terms=qcr_objective(m,batch,pattern_fn=fn,**cfg.loss.objective_kwargs())
        if args.q_coupling_weight:
            supervised=microscopic_coupling_loss(m.theory_encoder,batch,tx.couplings[ids])
            base=base+args.q_coupling_weight*supervised
        if frozen_source:
            info=frozen_information['train']
        elif exact:
            selected_hist=torch.randperm(len(tx.configurations),generator=rg)[:systems]
            table=probabilities(m.rg_kernel,patterns)[0]
            info=(histograms['train'].information_for(table,selected_hist).mean()
                  if execution['selected_empirical_mi'] else
                  histograms['train'].information(table)[selected_hist].mean() if sparse else
                  histogram_mutual_information(histograms['train'][selected_hist],table).mean())
        else:
            v,envs=sample_regions(tx.configurations,systems,pairs,rg,buffer=1,flip=cfg.data.spin_flip_augment)
            info=enumerated_infonce(kernel_probabilities(m.rg_kernel,v),critic(envs)).mean()
        total=base-args.rsmi_weight*info
        closure=cfg.loss.weights.commutator*terms['commutator']+cfg.loss.rollout_weight*terms['rollout']
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite at {step}')
        if frozen_source:
            separate_qc_r_gradients(total,closure,list(m.theory_encoder.parameters()),r)
        elif args.independent_c:
            independent_component_gradients(total,closure,
                -args.rsmi_weight*info+args.variance_weight*terms['kernel_noncollapse']+args.color_weight*terms.get('kernel_sign',0.),
                m.theory_encoder.parameters(),m.rg_kernel.parameters(),r,cp)
        else:
            separate_qc_r_gradients(total,closure,qc+cp,r)
        opt.step()
        from recurrent_rg.representation import normalize_affine_decoder_
        normalize_affine_decoder_(m.theory_encoder)
        if co is not None:co.step()
        if timing_enabled:
            synchronize();step_seconds.append(time.perf_counter()-update_started)
        completed_epoch=step
        stopped_by_wall_limit=args.max_wall_seconds is not None and time.perf_counter()-wall_started>=args.max_wall_seconds
        stopped_by_training_limit=args.max_training_seconds is not None and time.perf_counter()-training_started>=args.max_training_seconds
        if step==1 or step%t.log_every==0 or step==t.epochs or stopped_by_wall_limit or stopped_by_training_limit:
            logging_started=time.perf_counter()
            row=dict(epoch=step,learning_rate=opt.param_groups[0]['lr'])
            m.eval();critic.eval()
            for split,x,full in [('train',train_eval,tx.configurations),('test',test_eval,vx.configurations)]:
                metrics=evaluate(m,x,fn,e.samples,spin_flip_augment=cfg.data.spin_flip_augment,objective_kwargs=cfg.loss.objective_kwargs())
                if args.q_coupling_weight:
                    ds=tx if split=='train' else vx
                    eval_ids=torch.linspace(0,len(ds.configurations)-1,min(e.systems,len(ds.configurations))).long()
                    with torch.no_grad():
                        sup=float(microscopic_coupling_loss(m.theory_encoder,x[:,:e.samples].to(device=device,dtype=torch.float64),ds.couplings[eval_ids]))
                    metrics['coupling_supervision']=sup
                    metrics['weighted_coupling_supervision']=args.q_coupling_weight*sup
                    metrics['total']+=args.q_coupling_weight*sup
                if frozen_source:
                    info=frozen_information[split];shuffled=0.
                elif exact:
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
            print(f'epoch {step}/{t.epochs}: test total={row["test_total"]:.5f}, I_{args.estimator}={row["test_rsmi_bits"]:.4f} bits'+
                  (f', balanced={row["test_balanced_reconstruction"]:.5f}' if cfg.loss.balanced_reconstruction_weight else ''),flush=True)
            if args.joint_train_plateau:
                from recurrent_rg.joint_plateau import joint_plateau_reached
                component_row=train_component_losses(row,cfg,args)
                plateau_rows.append(component_row)
                stopped_by_train_plateau,plateau_diagnostics=joint_plateau_reached(plateau_rows,**plateau_settings)
                (args.output/'joint_plateau.json').write_text(json.dumps(dict(
                    history=plateau_rows,diagnostics=plateau_diagnostics),indent=2)+'\n')
                print('TRAIN components: '+str(component_row)+
                      '; plateau='+str(stopped_by_train_plateau),flush=True)
            if timing_enabled:synchronize()
            logging_seconds+=time.perf_counter()-logging_started
        if step%t.checkpoint_every==0:
            checkpoint_started=time.perf_counter();save(step,f'training_step_{step}.pt')
            checkpoint_seconds+=time.perf_counter()-checkpoint_started
        if stopped_by_train_plateau:
            print(f'Separate TRAIN Q/C/R objectives reached a local plateau at {step}; saving final endpoint.',flush=True)
            break
        if stopped_by_wall_limit:
            print(f'Wall limit reached after actual epoch {step}; saving completed training, not restarting.',flush=True)
            break
        if stopped_by_training_limit:
            print(f'Training-only limit reached after actual epoch {step}; saving completed training, not restarting.',flush=True)
            break
    if timing_enabled or args.max_training_seconds is not None:synchronize()
    training_wall_seconds=time.perf_counter()-training_started
    checkpoint_started=time.perf_counter()
    save(completed_epoch,'checkpoint_last.pt');torch.save(qcr_state_dict(m),args.output/'qcr_model.pt')
    checkpoint_seconds+=time.perf_counter()-checkpoint_started
    warmup=0 if args.benchmark_warmup is None else args.benchmark_warmup
    timing=dict(device=str(device),precision='torch.float64',setup_seconds=setup_seconds,
        training_wall_seconds=training_wall_seconds,updates=len(step_seconds),
        warmup_updates=min(warmup,len(step_seconds)),measured_updates=max(0,len(step_seconds)-warmup),
        step_seconds_all=step_seconds,steady_step_seconds=step_seconds[warmup:],
        logging_seconds=logging_seconds,checkpoint_seconds=checkpoint_seconds,
        includes_eval_in_step=False,enabled=timing_enabled)
    (args.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),contract=contract,history=history,
        elapsed_seconds=time.perf_counter()-started,setup_and_training_seconds=time.perf_counter()-wall_started,
        actual_epochs=completed_epoch,requested_epochs=t.epochs,stopped_by_wall_limit=stopped_by_wall_limit,
        stopped_by_training_limit=stopped_by_training_limit,training_limit_seconds=args.max_training_seconds,
        timing=timing,lr_scheduler_state=lr_policy.checkpoint_state(completed_epoch)),indent=2)+'\n')
    if args.joint_train_plateau:
        summary=json.loads((args.output/'summary.json').read_text())
        summary.update(stopped_by_train_plateau=stopped_by_train_plateau,
            stop_reason='train_plateau' if stopped_by_train_plateau else 'wall_limit' if stopped_by_wall_limit else 'training_limit' if stopped_by_training_limit else 'update_limit',
            joint_plateau_diagnostics=plateau_diagnostics)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report_started=time.perf_counter()
    evaluate_run(args.output,output=args.output/'report')
    summary=json.loads((args.output/'summary.json').read_text())
    summary['timing']['report_seconds']=time.perf_counter()-report_started
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
