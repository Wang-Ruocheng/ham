"""Single-pipeline R refinement: routing, RNG, clocks, resume and reports."""
import copy
from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.losses import qcr_objective
from recurrent_rg.r_refinement import closure_on_targets, refine_r_on_targets, component_optimizer_steps
from experiments import rsmi_separate_qcr as driver
from experiments.ising_cnn_short_trial import step, optimizers, validate, REFINED_RECIPE


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return RunConfig.load('configs/ising2d_B0_cnn_b3_g1_refinedR_seed91.json')


def same(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:same(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    else:assert left==right


def test_new_preset_changes_only_R_frequency_and_historical_configs_stay_single_step():
    old=RunConfig.load('configs/ising2d_B0_cnn_b3_g1_short_seed91.json')
    cfg=config();cfg.validate()
    assert old.training.r_updates_per_batch==1 and cfg.training.r_updates_per_batch==10
    assert replace(cfg,training=replace(cfg.training,r_updates_per_batch=1))==old
    raw=cfg.to_dict();raw['training'].pop('r_updates_per_batch')
    assert RunConfig.from_dict(raw).training.r_updates_per_batch==1
    args=driver.build_parser().parse_args(['--config','unused.json','--output','unused','--device','cuda:0','--configuration-cnn-short-trial'])
    validate(args,cfg);validate(args,old)
    changed=replace(cfg,training=replace(cfg.training,learning_rate=.002))
    with pytest.raises(ValueError,match='pinned'):validate(args,changed)


@pytest.mark.parametrize('ratio',[True,0,-1,1.5,33,'10'])
def test_reject_invalid_counts(ratio):
    cfg=config()
    with pytest.raises(ValueError,match='r_updates_per_batch'):
        replace(cfg,training=replace(cfg.training,r_updates_per_batch=ratio)).validate()


def test_reject_incompatible_recipe():
    cfg=config()
    for value in [replace(cfg,training=replace(cfg.training,gradient_mode='joint')),
                  replace(cfg,loss=replace(cfg.loss,closure_objective='one_step_and_rollout'))]:
        with pytest.raises(ValueError,match='R inner updates'):value.validate()
    args=driver.build_parser().parse_args(['--config','unused','--output','unused','--no-independent-c'])
    with pytest.raises(ValueError,match='independent C'):driver.validate_recipe(args,cfg)


def test_detached_snapshot_is_exact_original_objective_including_recurrent_gradients():
    torch.manual_seed(3);cfg=config();m=build_model(cfg.model,2);m.detach_kernel_representation=True
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1;targets=[]
    _,terms=qcr_objective(m,x,pattern_fn=plaquette_distribution,closure_targets_sink=targets,**cfg.loss.objective_kwargs())
    assert len(targets)==1 and not targets[0].requires_grad
    direct=targets[0].clone().requires_grad_()
    expected=terms['closure_total'];actual=closure_on_targets(m,direct,cfg.loss)
    same(expected,actual)
    parameters=list(m.theory_rg.parameters())
    same(torch.autograd.grad(expected,parameters,retain_graph=True),torch.autograd.grad(actual,parameters,retain_graph=True))
    actual.backward();assert direct.grad is None
    assert all(p.grad is None for mod in [m.theory_encoder,m.rg_kernel] for p in mod.parameters())


def test_online_CNN_QC_trajectory_rng_and_Adam_identical_with_ten_R_updates(monkeypatch):
    torch.manual_seed(8);cfg=config();a=build_model(cfg.model,2);a.detach_kernel_representation=True
    b=copy.deepcopy(a);oa=optimizers(a);ob=optimizers(b)
    old=replace(cfg,training=replace(cfg.training,r_updates_per_batch=1))
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    ids=torch.tensor([0,1]);calls=[];original=b.coarse_grain
    def tracked(batch):calls.append(1);return original(batch)
    monkeypatch.setattr(b,'coarse_grain',tracked)
    for iteration in range(2):
        rng=torch.get_rng_state()
        before=step(a,x,hist,ids,patterns,old,oa);after_rng=torch.get_rng_state()
        torch.set_rng_state(rng)
        after=step(b,x,hist,ids,patterns,cfg,ob)
        same(after_rng,torch.get_rng_state())
        for name in ['Q','C']:
            mod='theory_encoder' if name=='Q' else 'rg_kernel'
            same(getattr(a,mod).state_dict(),getattr(b,mod).state_dict())
            same(oa[name].state_dict(),ob[name].state_dict())
            assert all(p.grad is None for p in getattr(b,mod).parameters())
        assert before['q_total']==after['q_total'] and before['c_total']==after['c_total']
        assert all(int(s['step'])==(iteration+1)*10 for s in ob['R'].state.values())
        assert np.isfinite(after['r_batch_after'])
    assert len(calls)==2  # Not 20 C forwards, and no replay from old batches.
    assert any(not torch.equal(x,y) for x,y in zip(a.theory_rg.parameters(),b.theory_rg.parameters()))


def test_shared_Adam_extra_updates_skip_QC_and_keep_moments(monkeypatch):
    torch.manual_seed(7);cfg=config();m=build_model(cfg.model,2)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    # Give every parameter nonzero optimizer state and stale grads; helper
    # must clear them to None before touching shared Adam again.
    sum(p.square().sum() for p in m.parameters()).backward();opt.step()
    frozen={name:copy.deepcopy(mod.state_dict()) for name,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]}
    states={p:copy.deepcopy(opt.state[p]) for mod in [m.theory_encoder,m.rg_kernel] for p in mod.parameters()}
    monkeypatch.setattr(m,'coarse_grain',lambda *a:pytest.fail('Extra C forward'))
    monkeypatch.setattr(m.theory_encoder,'forward',lambda *a:pytest.fail('Extra Q forward'))
    direct=torch.randn(4,3,1,dtype=torch.float64,requires_grad=True)
    rng=torch.get_rng_state();refine_r_on_targets(m,opt,direct,cfg.loss,9)
    same(rng,torch.get_rng_state());assert direct.grad is None
    for name,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]:same(frozen[name],mod.state_dict())
    for p,value in states.items():same(value,opt.state[p]);assert p.grad is None
    clocks=component_optimizer_steps(m,opt)
    assert clocks['R']['min']==clocks['R']['max']==10
    assert clocks['Q']['min']==clocks['Q']['max']==1
    assert clocks['C']['min']==clocks['C']['max']==1


def test_zero_extra_steps_is_strict_noop():
    m=build_model(config().model,2);opt=torch.optim.Adam(m.parameters(),lr=.001)
    before=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
    assert refine_r_on_targets(m,opt,None,None,0) is None
    same(before,opt.state_dict());same(rng,torch.get_rng_state())


@pytest.mark.parametrize('schedule',[{},dict(lr_schedule='multistep',lr_milestones=[2],lr_gamma=.5)])
def test_standard_single_pipeline_exact_resume_and_actual_clocks(tmp_path,schedule):
    generator=torch.Generator().manual_seed(41)
    train=(2*torch.randint(2,(4,8,36,36,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(4,8,36,36,1),generator=generator)-1).double()
    cache=tmp_path/'synthetic.pt';torch.save((train,torch.arange(4),test,torch.arange(4)+.5),cache)
    raw=config().to_dict();raw['data']['cache']=str(cache)
    raw['training'].update(epochs=3,system_batch=2,sample_batch=4,log_every=1,checkpoint_every=1,**schedule)
    raw['evaluation'].update(samples=4,systems=2)
    path=tmp_path/'config.json';path.write_text(json.dumps(raw))
    common=['--config',str(path),'--skip-final-report']
    for name,extra in [('full',[]),('first',['--epochs','1']),
                       ('resumed',['--resume-checkpoint',str(tmp_path/'first/checkpoint_last.pt')])]:
        driver.main(common+['--output',str(tmp_path/name)]+extra)
    a=torch.load(tmp_path/'full/checkpoint_last.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/checkpoint_last.pt',weights_only=True)
    for key in ['model','optimizer','torch_rng_state','batch_generator_state','rsmi_generator_state','optimizer_steps','history','lr_scheduler_state']:
        same(a[key],b[key])
    assert a['epoch']==3
    for key,expected in [('Q',3),('C',3),('R',30)]:
        assert a['optimizer_steps'][key]['min']==a['optimizer_steps'][key]['max']==expected
    assert a['metadata']['R_optimization']['separate_R_only_stage'] is False
    summary=json.loads((tmp_path/'full/summary.json').read_text())
    same(summary['optimizer_steps'],a['optimizer_steps'])
    assert 'r_batch_after' in summary['history'][-1]
    # Config changes must never silently reinterpret a trained optimizer.
    raw['training']['r_updates_per_batch']=9;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='identical configuration'):
        driver.main(common+['--output',str(tmp_path/'bad'),'--resume-checkpoint',str(tmp_path/'first/checkpoint_last.pt')])


def test_refined_report_uses_actual_module_clocks(tmp_path):
    from scripts.report_ising_cnn_short_trial import render
    from scripts.report_phi4_hopping_qcr import closure_metrics
    rng=np.random.default_rng(7);g=rng.normal(size=(192,3,1))
    arrays=dict(direct_g=g,one_step_g=g[:,1:]+.01,rollout_g=np.concatenate([g[:,:1],g[:,1:]+.02],axis=1),
        all_start_g=np.stack([g[:,1]+.01,g[:,2]+.02,g[:,2]+.01],axis=1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=np.repeat(np.linspace(.2,.68,64),3)[:,None],splits=np.array(['train']*129+['test']*63),
        flow_grid=np.linspace(-3,3,501)[:,None],flow_next=np.linspace(-3,3,501)[:,None]*1.1)
    cfg=config().to_dict();cfg['training']['epochs']=25
    clocks={n:dict(min=v,max=v) for n,v in [('Q',25),('C',25),('R',250)]}
    for name,value in [('manifest',dict(recipe=REFINED_RECIPE,config=cfg)),
        ('summary',dict(actual_steps=25,closure=closure_metrics(arrays),optimizer_steps=clocks)),
        ('history',[dict(step=25,q_total=.1,c_total=-.2,r_total=.3,r_batch_after=.1)]),('roots',dict(roots=[]))]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    np.savez(tmp_path/'evaluation_arrays.npz',**arrays)
    result=render(tmp_path)
    assert result['r_updates_per_batch']==10 and result['optimizer_steps']==clocks
    assert len(list((tmp_path/'report').glob('*.png')))==4
