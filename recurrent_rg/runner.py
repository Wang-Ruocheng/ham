"""Configuration-driven cache -> QCR training -> diagnostics pipeline."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import RunConfig, ScaleConfig
from .checkpoints import build_model, load_run, load_model_state, qcr_state_dict
from .data.ensembles import load_ensembles
from .data.statistics import two_spin_distribution, plaquette_distribution, categorical_plaquette_distribution
from .training import train
from .evaluation import collect_trajectories, closure_metrics


def file_fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def prepared_data(config):
    config.validate()
    tx,vx=load_ensembles(config.data.cache)
    if config.data.train_pool_replicas!=1: tx=tx.pool(config.data.train_pool_replicas)
    if config.data.test_pool_replicas!=1: vx=vx.pool(config.data.test_pool_replicas)
    for ds in (tx,vx):
        x=ds.configurations
        if config.model.categorical_states:
            if x.ndim!=config.model.dimension+3 or x.shape[-1]!=config.model.categorical_states or not torch.all((x==0)|(x==1)) or not torch.all(x.sum(-1)==1):
                raise ValueError('Expected one-hot categorical configurations')
        elif x.ndim!=config.model.dimension+3 or x.shape[-1]!=1 or not torch.all((x==1)|(x==-1)):
            raise ValueError('Expected binary single-channel configurations')
        if config.data.coupling_names and (ds.couplings is None or ds.couplings.reshape(len(x),-1).shape[1]!=len(config.data.coupling_names)):
            raise ValueError('Coupling names do not match label columns')
        if config.evaluation.samples>x.shape[1]: raise ValueError('Not enough evaluation samples')
        ScaleConfig(tuple(x.shape[2:-1]),config.training.max_rg_steps,config.model.block_factor).shapes()
    if config.training.sample_batch>tx.configurations.shape[1]: raise ValueError('Not enough training samples')
    return tx,vx


def run_training(config: RunConfig, output, *, resume=None, initialize_from=None, allow_legacy_training=False):
    if not allow_legacy_training:
        raise ValueError('This legacy runner omits empirical MI and independent C gradients. '
                         'Use python -m recurrent_rg train for the locked default recipe; '
                         'allow_legacy_training=True is only for explicitly requested legacy experiments.')
    if resume and initialize_from: raise ValueError('Resume and weight initialization are different operations')
    config=replace(config,data=replace(config.data,cache=str(Path(config.data.cache).resolve())))
    tx,vx=prepared_data(config)
    metadata={'config':config.to_dict(),'data_sha256':file_fingerprint(config.data.cache)}
    torch.set_num_threads(config.training.threads);torch.manual_seed(config.training.seed)
    model=build_model(config.model,config.training.max_rg_steps)
    state=None;history=None
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state.get('metadata',{}).get('training_driver') in ('rsmi_separate_qcr','at_predictive_q'):
            raise ValueError('Specialized checkpoint requires its custom trainer; generic resume would omit its objective/state')
        if not state.get('metadata',{}).get('config'):
            raise ValueError('Exact CLI resume requires a versioned checkpoint; legacy weights can use initialize_from')
        previous=json.loads(json.dumps(state['metadata']['config']))
        # Older versioned checkpoints predate this optional initialization field.
        previous['model'].setdefault('kernel_initialization', 'random')
        previous['model'].setdefault('categorical_states', 0)
        previous['model'].setdefault('kernel_sampling', 'deterministic')
        previous['model'].setdefault('kernel_temperature', 1.0)
        previous['model'].setdefault('affine_reconstruction', False)
        previous['loss'].setdefault('orientation_weight', 0.0)
        previous['loss'].setdefault('orientation_radius', 0.02)
        previous['loss'].setdefault('kernel_sign_weight', 0.0)
        previous['loss'].setdefault('kernel_neutral_weight', 0.0)
        previous['loss'].setdefault('kernel_neutral_max_patterns', 65536)
        previous['loss'].setdefault('kernel_distinct_alpha', 0.0)
        previous['loss'].setdefault('kernel_distinct_margin', 0.8)
        previous['training'].setdefault('gradient_mode', 'joint')
        current=json.loads(json.dumps(config.to_dict()))
        previous['training']['epochs']=current['training']['epochs']
        if previous!=current or state['metadata']['data_sha256']!=metadata['data_sha256']:
            raise ValueError('Exact resume allows only extending epochs; configuration/data changed')
        # A previous stopping endpoint is not an extra point in the continued
        # run's fixed logging schedule. Preserve the original run unchanged.
        history=[row for row in state.get('history',[])
                 if row['epoch']==1 or row['epoch']%config.training.log_every==0]
    elif initialize_from:
        load_model_state(model,torch.load(initialize_from,map_location='cpu',weights_only=True))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(config=config.to_dict(),data_sha256=metadata['data_sha256'],
                  train_shape=list(tx.configurations.shape),test_shape=list(vx.configurations.shape),
                  resume=str(resume) if resume else None,initialize_from=str(initialize_from) if initialize_from else None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    t,e=config.training,config.evaluation
    history=train(model,tx.configurations,vx.configurations,
        pattern_fn=(categorical_plaquette_distribution if config.model.categorical_states else
                    two_spin_distribution if config.model.dimension==1 else plaquette_distribution),
        epochs=t.epochs,system_batch=t.system_batch,sample_batch=t.sample_batch,
        eval_samples=e.samples,eval_systems=e.systems,learning_rate=t.learning_rate,seed=t.seed,
        progress_dir=out,spin_flip_augment=config.data.spin_flip_augment,objective_kwargs=config.loss.objective_kwargs(),
        resume_state=state,initial_history=history,train_modules=t.train_modules,
        checkpoint_every=t.checkpoint_every,checkpoint_metadata=metadata,record_every=t.log_every,
        gradient_mode=t.gradient_mode)
    torch.save(qcr_state_dict(model),out/'qcr_model.pt')
    (out/'summary.json').write_text(json.dumps(dict(config=config.to_dict(),history=history),indent=2)+'\n')
    return out


def evaluate_run(path, *, checkpoint=None, cache=None, max_rg_steps=None, output=None, make_plots=True):
    run=load_run(path,checkpoint)
    if 'data' not in run.config or 'training' not in run.config:
        raise ValueError('Generic reports require a versioned manifest; legacy runs remain supported by load_run and old reporters')
    cfg=RunConfig.from_dict(run.config);cfg.validate();torch.set_num_threads(cfg.training.threads)
    manifest_path=run.checkpoint.parent/'manifest.json'
    if manifest_path.exists():
        expected=json.loads(manifest_path.read_text()).get('data_sha256')
        if expected and file_fingerprint(cfg.data.cache)!=expected:
            raise ValueError('Training cache content changed since this run')
    tx,vx=prepared_data(cfg)
    if cache:
        _,vx=load_ensembles(cache)
    e=cfg.evaluation;steps=e.max_rg_steps if max_rg_steps is None else max_rg_steps
    ScaleConfig(tuple(vx.configurations.shape[2:-1]),steps,cfg.model.block_factor).shapes()
    training=collect_trajectories(run.model,tx.configurations,samples=e.samples,
        max_rg_steps=cfg.training.max_rg_steps,spin_flip_augment=cfg.data.spin_flip_augment)
    target=collect_trajectories(run.model,vx.configurations,samples=e.samples,
        max_rg_steps=steps,spin_flip_augment=cfg.data.spin_flip_augment)
    out=Path(output) if output else run.checkpoint.parent/'evaluation'/run.checkpoint.stem
    out.mkdir(parents=True,exist_ok=True)
    labels={} if vx.couplings is None else {'couplings':vx.couplings.numpy()}
    np.savez_compressed(out/'trajectories.npz',**target,**labels,training_inputs=training['direct'][:,:-1])
    report={'checkpoint':str(run.checkpoint),'evaluation_cache':str(cache or cfg.data.cache),
            'evaluation_cache_sha256':file_fingerprint(cache or cfg.data.cache),
            'max_rg_steps':steps,'metrics':closure_metrics(target)}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    if make_plots:
        from .plotting import plot_diagnostics
        plot_diagnostics(run.model,target,None if vx.couplings is None else vx.couplings.numpy(),cfg.data.coupling_names,out,
                         training_inputs=training['direct'][:,:-1])
    return out
