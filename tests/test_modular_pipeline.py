"""Architecture contracts, complete resume, and real package-level workflows."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from recurrent_rg.config import RunConfig, DataConfig, ModelConfig, TrainingConfig, EvaluationConfig, ScaleConfig
from recurrent_rg.checkpoints import build_model, load_run, load_model_state, qcr_state_dict
from recurrent_rg.data.ensembles import EnsembleDataset, grouped_split, load_ensembles
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, discrete_fixed_point, scalar_flow
from recurrent_rg.runner import run_training as legacy_run_training, evaluate_run
from functools import partial
# These tests deliberately exercise the historical trainer/resume contract.
run_training = partial(legacy_run_training, allow_legacy_training=True)
from recurrent_rg.training import train


def tiny_config(tmp_path, dimension=1):
    torch.set_num_threads(1);torch.manual_seed(72)
    shape=(16,) if dimension==1 else (8,8)
    tx=(2*torch.randint(2,(4,4)+shape+(1,))-1).to(torch.int8)
    vx=(2*torch.randint(2,(2,4)+shape+(1,))-1).to(torch.int8)
    cache=tmp_path/'data.pt'
    torch.save((tx,torch.tensor([.2,.2,.8,.8]),vx,torch.tensor([.5,.5])),cache)
    return RunConfig(DataConfig(str(cache),'synthetic test',('h/J',)),
                     model=ModelConfig(dimension=dimension,block_factor=2,theory_dim=dimension,hidden=4),
                     training=TrainingConfig(max_rg_steps=1,epochs=4,system_batch=3,sample_batch=4,threads=1,checkpoint_every=2,log_every=1),
                     evaluation=EvaluationConfig(max_rg_steps=1,samples=4,systems=2))


def test_legacy_imports_are_exact_aliases():
    import experiments.joint_qcr_ising as old
    import recurrent_rg.losses as new
    from experiments._utils import balanced_views as old_views
    from recurrent_rg.data.views import balanced_views
    assert old.qcr_objective is new.qcr_objective
    assert old.train is train
    assert old_views is balanced_views


def test_scale_config_and_explicit_symmetry(tmp_path):
    assert ScaleConfig((128,),5).shapes()==[(128,),(64,),(32,),(16,),(8,),(4,)]
    with pytest.raises(ValueError): ScaleConfig((16,),5).shapes()
    with pytest.raises(ValueError): ScaleConfig((18,),2).shapes()
    cfg=tiny_config(tmp_path)
    with pytest.raises(ValueError,match='symmetric'):
        replace(cfg,data=replace(cfg.data,spin_flip_augment=True)).validate()
    with pytest.raises(ValueError): RunConfig.from_dict({'data':{},'typo':1})
    path=tmp_path/'config.json';values=cfg.to_dict();values['data']['cache']='data.pt'
    path.write_text(json.dumps(values))
    assert Path(RunConfig.load(path).data.cache)==tmp_path/'data.pt'
    values['loss']={'weights':{'commutator':8.}}
    modified=RunConfig.from_dict(values)
    assert modified.loss.weights.commutator==8.
    assert modified.loss.weights.anchor==2. and modified.loss.weights.separation==10.


def test_pooling_and_split_preserve_parameter_groups(tmp_path):
    x=torch.arange(12*4*8).reshape(12,4,8,1)
    labels=torch.arange(3).repeat_interleave(4)
    ds=EnsembleDataset(x,labels)
    a,b=grouped_split(ds)
    assert not set(a.couplings.tolist()) & set(b.couplings.tolist())
    pooled=ds.pool(2)
    assert pooled.configurations.shape==(6,8,8,1)
    assert torch.equal(pooled.configurations[0],x[:2].flatten(0,1))
    with pytest.raises(ValueError): ds.pool(3)
    file=tmp_path/'leaked.pt';torch.save((x,labels,x,labels),file)
    with pytest.raises(ValueError,match='overlap'): load_ensembles(file)


def test_evaluation_does_not_mutate_depth_or_training_mode():
    torch.manual_seed(91)
    m=build_model(ModelConfig(hidden=4),1).train()
    x=(2*torch.randint(2,(3,4,32,1))-1).double()
    result=collect_trajectories(m,x,samples=4,max_rg_steps=3,spin_flip_augment=False)
    assert result['direct'].shape==(3,4,1)
    assert m.levels==2 and m.training
    metrics=closure_metrics(result)
    assert len(metrics['rollout']['nrmse'])==4
    assert metrics['rollout']['rmse'][0]==0
    collapsed={k:np.zeros_like(v) for k,v in result.items()}
    assert all(v is None for v in closure_metrics(collapsed)['rollout']['nrmse'])


def test_fixed_model_preserves_paired_QR_initialization():
    torch.manual_seed(91);a=build_model(ModelConfig(hidden=4),2)
    torch.manual_seed(91);b=build_model(ModelConfig(hidden=4,fixed_kernel=True,kernel_sampling='deterministic'),2)
    for k,v in a.state_dict().items():
        if k.startswith(('theory_encoder.','theory_rg.')):
            assert torch.equal(v,b.state_dict()[k])


@pytest.mark.parametrize('dimension,fixed',[(1,False),(2,False),(2,True)])
def test_legacy_checkpoint_loading(tmp_path,dimension,fixed):
    m=build_model(ModelConfig(dimension=dimension,block_factor=2,hidden=4,fixed_kernel=fixed,kernel_sampling='deterministic'),2)
    torch.save(qcr_state_dict(m),tmp_path/'qcr_model.pt')
    # Missing dimension/layout are recovered from weight shapes, not directory name.
    (tmp_path/'summary.json').write_text(json.dumps({'config':{'levels':3,'hidden':4,'fixed_kernel':fixed}}))
    before=torch.get_rng_state().clone();run=load_run(tmp_path)
    assert run.model_config.kernel_sampling == 'deterministic'
    assert torch.equal(before,torch.get_rng_state())
    for k,v in qcr_state_dict(m).items():assert torch.equal(v,run.model.state_dict()[k])
    broken=qcr_state_dict(m);broken.pop('theory_rg.0.bias')
    with pytest.raises(ValueError,match='Incompatible'):load_model_state(m,broken)


@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('log_every',[1,3])
@pytest.mark.parametrize('gradient_mode',['joint','qc_r_separate'])
def test_complete_pipeline_resume_and_figures(tmp_path,dimension,log_every,gradient_mode):
    cfg=tiny_config(tmp_path,dimension)
    cfg=replace(cfg,training=replace(cfg.training,log_every=log_every,gradient_mode=gradient_mode))
    full=run_training(cfg,tmp_path/'full')
    short=run_training(replace(cfg,training=replace(cfg.training,epochs=2)),tmp_path/'short')
    resumed=run_training(cfg,tmp_path/'resumed',resume=short/'checkpoint_last.pt')
    a,b=load_run(full),load_run(resumed)
    for k,v in qcr_state_dict(a.model).items():assert torch.equal(v,b.model.state_dict()[k])
    assert json.loads((full/'summary.json').read_text())['history']==json.loads((resumed/'summary.json').read_text())['history']
    # Intermediate checkpoint includes its logged history and resumes identically.
    intermediate=run_training(cfg,tmp_path/'intermediate',resume=full/'training_step_2.pt')
    for k,v in qcr_state_dict(a.model).items():assert torch.equal(v,load_run(intermediate).model.state_dict()[k])
    report=evaluate_run(full,max_rg_steps=1,make_plots=True)
    assert (report/('R_vs_g.png' if dimension==1 else 'R_vector_field.png')).exists()
    assert (report/'coupling_vs_g.png').exists()
    changed=replace(cfg,training=replace(cfg.training,sample_batch=2))
    with pytest.raises(ValueError,match='configuration/data'):run_training(changed,tmp_path/'bad',resume=short/'checkpoint_last.pt')


def test_fixed_decimation_separate_qr_resume(tmp_path):
    cfg=tiny_config(tmp_path)
    cfg=replace(cfg,model=replace(cfg.model,fixed_kernel=True,kernel_sampling='deterministic'),
        training=replace(cfg.training,train_modules=('Q','R'),gradient_mode='qc_r_separate'))
    torch.manual_seed(cfg.training.seed)
    learnable=build_model(replace(cfg.model,fixed_kernel=False,kernel_sampling='bernoulli'),1)
    torch.manual_seed(cfg.training.seed)
    fixed=build_model(cfg.model,1)
    assert not list(fixed.rg_kernel.parameters())
    for key,value in qcr_state_dict(learnable).items():
        if key.startswith(('theory_encoder.','theory_rg.')):assert torch.equal(value,fixed.state_dict()[key])
    x=load_ensembles(cfg.data.cache)[0].configurations
    torch.testing.assert_close(fixed.rg_kernel(x),x[:,:,::2])
    full=run_training(cfg,tmp_path/'fixed_full')
    short=run_training(replace(cfg,training=replace(cfg.training,epochs=2)),tmp_path/'fixed_short')
    resumed=run_training(cfg,tmp_path/'fixed_resumed',resume=short/'checkpoint_last.pt')
    a,b=load_run(full).model,load_run(resumed).model
    for key,value in qcr_state_dict(a).items():assert torch.equal(value,b.state_dict()[key])


def test_frozen_modules_and_invalid_batch(tmp_path):
    cfg=tiny_config(tmp_path);tx,vx=load_ensembles(cfg.data.cache)
    torch.manual_seed(91);m=build_model(cfg.model,1)
    before={k:v.clone() for k,v in qcr_state_dict(m).items()}
    kw=dict(pattern_fn=two_spin_distribution,epochs=2,system_batch=3,sample_batch=4,
            eval_samples=4,eval_systems=2,learning_rate=.001,seed=91,spin_flip_augment=False)
    train(m,tx.configurations,vx.configurations,train_modules=('R',),**kw)
    for k,v in before.items():
        if k.startswith(('theory_encoder.','rg_kernel.')):assert torch.equal(v,m.state_dict()[k])
    assert any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items() if k.startswith('theory_rg.'))
    with pytest.raises(ValueError,match='Sample batch'):
        train(m,tx.configurations,vx.configurations,**dict(kw,sample_batch=6))


def test_discrete_stability_not_displacement():
    class Map:
        def advance_theory(self,x):return -1.2*x
    assert discrete_fixed_point(Map(),[0.])['stability']=='unstable'
    flow=scalar_flow(Map(),-1,1)
    assert len(flow['roots'])==1 and flow['roots'][0]['g']==[0.]
    class Identity:
        def advance_theory(self,x):return x
    assert scalar_flow(Identity(),-1,1)['roots']==[]


def test_unlabeled_data_can_train_and_plot(tmp_path):
    cfg=tiny_config(tmp_path)
    tx,ty,vx,vy=torch.load(cfg.data.cache,weights_only=True)
    torch.save({'schema_version':1,
                'train':{'configurations':tx,'group_ids':torch.tensor([0,0,1,1])},
                'test':{'configurations':vx,'group_ids':torch.tensor([2,2])}},cfg.data.cache)
    cfg=replace(cfg,data=replace(cfg.data,coupling_names=()),training=replace(cfg.training,epochs=1))
    output=run_training(cfg,tmp_path/'unlabeled')
    figures=evaluate_run(output)
    assert (figures/'R_vs_g.png').exists()
    assert not (figures/'coupling_vs_g.png').exists()
    assert load_ensembles(cfg.data.cache)[0].pool(2).couplings is None


def test_edited_cache_is_rejected_by_report(tmp_path):
    cfg=tiny_config(tmp_path)
    output=run_training(replace(cfg,training=replace(cfg.training,epochs=1)),tmp_path/'run')
    tx,ty,vx,vy=torch.load(cfg.data.cache,weights_only=True)
    tx[0,0,0]*=-1;torch.save((tx,ty,vx,vy),cfg.data.cache)
    with pytest.raises(ValueError,match='cache content changed'):evaluate_run(output)
