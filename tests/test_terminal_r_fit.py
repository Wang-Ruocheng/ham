import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.config import RunConfig, TrainingConfig, LossConfig
from recurrent_rg.checkpoints import build_model, load_run
from recurrent_rg.terminal_r_fit import collect_train_targets, fit_r, fingerprint, plateau_reached
from recurrent_rg.r_refinement import component_optimizer_steps
from experiments import rsmi_separate_qcr as driver


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def cfg():
    return RunConfig.load('configs/ising2d_B0_cnn_b3_g1_refinedR_seed91.json')


def assert_state_equal(a,b):
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_state_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_state_equal(x,y)
    else: assert a==b


def test_fresh_default_and_historical_resume_policy():
    assert TrainingConfig().terminal_r_fit
    c=cfg();raw=c.to_dict();raw['training'].pop('terminal_r_fit')
    historical=RunConfig.from_dict(raw)
    assert not historical.training.terminal_r_fit
    assert driver.default_terminal_r(historical,raw).training.terminal_r_fit
    assert not driver.default_terminal_r(historical,raw,historical=True).training.terminal_r_fit
    resume={'metadata':{'config':raw}}
    assert not driver.default_terminal_r(historical,raw,resume=resume).training.terminal_r_fit
    for marker in ['terminal_R_stage','terminal_R_fit_completed']:
        with pytest.raises(ValueError,match='terminal'):
            driver.default_terminal_r(historical,raw,resume={'metadata':{marker:True}})


def test_fixed_cache_private_rng_and_chain_accounting():
    model=build_model(cfg().model,2)
    x=torch.arange(5*8,dtype=torch.double).reshape(5,8,1)
    def encode(batch):
        z=batch.mean(1)+torch.rand(len(batch),1,dtype=torch.double)
        return torch.stack([z,z*.9,z*.8],1)
    state=torch.get_rng_state()
    a,receipt=collect_train_targets(model,x,encode,samples=4,systems=2,repeats=3)
    assert_state_equal(state,torch.get_rng_state())
    b,_=collect_train_targets(model,x,encode,samples=4,systems=2,repeats=3)
    assert_state_equal(a,b)
    assert not a.requires_grad and receipt['physical_chains']==5 and receipt['cached_rows']==15
    assert receipt['chain_index']==list(range(5))*3
    assert receipt['labels_used'] is False


def test_shared_Adam_terminal_freezes_QC_and_full_cache_chunking(monkeypatch):
    torch.manual_seed(7);c=cfg();m=build_model(c.model,2)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    sum(p.square().sum() for p in m.parameters()).backward();opt.step()
    twin=copy.deepcopy(m);other=torch.optim.Adam(twin.parameters(),lr=.001)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    frozen={n:fingerprint(mod) for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]}
    states={p:copy.deepcopy(opt.state[p]) for mod in [m.theory_encoder,m.rg_kernel] for p in mod.parameters()}
    monkeypatch.setattr(m,'coarse_grain',lambda *a:pytest.fail('C forward during R fit'))
    monkeypatch.setattr(m.theory_encoder,'forward',lambda *a:pytest.fail('Q forward during R fit'))
    targets=torch.randn(7,3,1,dtype=torch.double,requires_grad=True)
    kwargs=dict(max_steps=3,check_every=1)
    result=fit_r(m,m.theory_rg,{'Q':m.theory_encoder,'C':m.rg_kernel},opt,targets,c.loss,chunk_size=2,**kwargs)
    fit_r(twin,twin.theory_rg,{'Q':twin.theory_encoder,'C':twin.rg_kernel},other,targets,c.loss,chunk_size=99,**kwargs)
    assert targets.grad is None and result['stop_reason']=='update_cap' and not result['train_plateau_reached']
    for x,y in zip(m.theory_rg.parameters(),twin.theory_rg.parameters()): torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
    for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]: assert frozen[n]==fingerprint(mod)
    for p,state in states.items(): assert_state_equal(state,opt.state[p]);assert p.grad is None
    clocks=component_optimizer_steps(m,opt)
    assert clocks['R']['min']==clocks['R']['max']==4
    assert clocks['Q']['min']==clocks['C']['min']==1


def test_plateau_requires_time_window_confirmations_and_no_rebound():
    rows=[dict(extra_R_updates=s,train_loss=1.,best_train_closure=1.) for s in range(0,4001,500)]
    assert not plateau_reached(rows[:-1])
    assert plateau_reached(rows)
    rows[-1]['train_loss']=1.1
    assert not plateau_reached(rows)


def tiny_config(tmp_path):
    raw=json.loads(Path('configs/remote_smoke_225.json').read_text())
    gen=torch.Generator().manual_seed(33)
    train=(2*torch.randint(2,(3,8,36,36,1),generator=gen)-1).double()
    test=train[:2].clone();cache=tmp_path/'data.pt'
    torch.save((train,torch.arange(3),test,torch.arange(2)+10),cache)
    raw['data']['cache']=str(cache);raw['model']['hidden']=5
    raw['training'].update(epochs=1,threads=1,system_batch=2,sample_batch=4,
        checkpoint_every=1,log_every=1,terminal_r_max_steps=3,terminal_r_cache_repeats=2)
    raw['training'].pop('terminal_r_fit',None)
    raw['loss']['closure_objective']='all_start_anchor'
    raw['evaluation'].update(samples=4,systems=2)
    p=tmp_path/'config.json';p.write_text(json.dumps(raw));return p


def test_standard_pipeline_final_model_and_preserved_joint(tmp_path):
    p=tiny_config(tmp_path);root=tmp_path/'run'
    driver.main(['--config',str(p),'--output',str(root),'--skip-final-report'])
    pipe=json.loads((root/'pipeline.json').read_text());assert pipe['workflow_complete']
    joint=torch.load(root/'checkpoint_last.pt',weights_only=True)
    final=torch.load(root/'r_only/checkpoint_last.pt',weights_only=True)
    assert joint['optimizer_steps']['R']['min']==1
    assert final['optimizer_steps']['R']['min']==4
    assert final['optimizer_steps']['Q']['min']==final['optimizer_steps']['C']['min']==1
    assert final['metadata']['R_optimization']['separate_R_only_stage']
    loaded=load_run(root);old=load_run(root/'checkpoint_last.pt')
    assert fingerprint(loaded.model.theory_encoder)==fingerprint(old.model.theory_encoder)
    assert fingerprint(loaded.model.rg_kernel)==fingerprint(old.model.rg_kernel)
    assert fingerprint(loaded.model.theory_rg)!=fingerprint(old.model.theory_rg)
    pipe['model_complete']=False;(root/'pipeline.json').write_text(json.dumps(pipe))
    with pytest.raises(ValueError,match='incomplete'):load_run(root)
    load_run(root/'checkpoint_last.pt')
    pipe.update(model_complete=True,final_checkpoint='../escape.pt')
    (root/'pipeline.json').write_text(json.dumps(pipe))
    with pytest.raises(ValueError):load_run(root)


def test_budget_exhaustion_never_silently_completes_joint_only(tmp_path):
    p=tiny_config(tmp_path);root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='budget'):
        driver.main(['--config',str(p),'--output',str(root),'--skip-final-report','--max-training-seconds','1e-12'])
    pipe=json.loads((root/'pipeline.json').read_text())
    assert not pipe['workflow_complete'] and pipe['status']=='incomplete'
    assert (root/'checkpoint_last.pt').is_file()


def test_xy_paired_evaluation_has_no_QC_forward(monkeypatch):
    from recurrent_rg.xy_qcr import XYQCR
    from experiments.xy_qcr_g2_r_only import paired_predictions,restore
    from experiments.xy_qcr_g2 import RECIPE
    from experiments.phi4_hopping_qcr import module_fingerprints
    model=XYQCR(seed=91,dim=2)
    opt=torch.optim.Adam(model.R.parameters(),lr=.003)
    sum(p.square().sum() for p in model.R.parameters()).backward();opt.step()
    for state in opt.state.values():state['step'].fill_(1250)
    payload=dict(recipe=RECIPE,step=1250,model=model.state_dict(),optimizers={'R':opt.state_dict()})
    restored,ropt=restore(payload,module_fingerprints(model),'cpu')
    assert all(int(v['step'])==1250 for v in ropt.state.values())
    monkeypatch.setattr(restored.Q,'forward',lambda *a:pytest.fail('Q evaluation'))
    monkeypatch.setattr(restored.C,'forward',lambda *a:pytest.fail('C evaluation'))
    rng=np.random.default_rng(4);g=rng.normal(size=(9,3,2))
    before=dict(direct_g=g,labels=rng.random((9,2)),splits=np.array(['train']*9),flow_grid=np.ones((2,2)))
    after=paired_predictions(restored,before)
    for name in ['direct_g','labels','splits']:np.testing.assert_array_equal(after[name],before[name])
    assert 'flow_grid' not in after and after['all_start_g'].shape==(9,3,2)


def test_xy_report_paired_metrics(tmp_path):
    from test_xy_qcr_g2 import mock_arrays
    from experiments.xy_qcr_g2 import flow_diagnostics
    from experiments.phi4_hopping_qcr import closure_metrics
    from recurrent_rg.xy_qcr import XYQCR
    from scripts.report_xy_qcr_g2_r_only import render
    a=mock_arrays();roots=flow_diagnostics(XYQCR(seed=91,dim=2),a,'cpu')
    np.savez(tmp_path/'evaluation_arrays.npz',**a);(tmp_path/'data').mkdir()
    summary=dict(actual_steps=1253,closure=closure_metrics(a),before_closure=closure_metrics(a),
        information_diagnostic={},C_final_exactly_matches_scalar=True,extra_R_updates=3,R_total_updates=1253,
        Q_C_unchanged=True,R_fit=dict(stop_reason='update_cap'))
    objects={'summary.json':summary,'roots.json':roots,'history.json':[dict(step=1250,q_total=.1,c_total=.1,r_total=.1)],
        'data/metadata.json':dict(warnings=[]),'reference_scalar.json':dict(comparison_step=1250),
        'r_history.json':[dict(extra_R_updates=0,train_loss=.1),dict(extra_R_updates=3,train_loss=.08)]}
    for name,value in objects.items():(tmp_path/name).write_text(json.dumps(value))
    report=render(tmp_path)
    assert report['R_total_updates']==1253 and not report['BKT_certified']
    assert (tmp_path/'report/joint_loss_curves.png').is_file()
    assert (tmp_path/'report/closure_comparison.png').is_file()
