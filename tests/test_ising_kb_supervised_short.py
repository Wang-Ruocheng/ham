"""Synthetic checks only; no physical sampling or scientific training."""
import copy
import json
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import ising_kb_supervised_short as run
from experiments import ising_kb_cnn_qcr as kb
from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.phi4_ru_flow import bounded_flow


@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)


def data(n=4):
    return SimpleNamespace(configurations=torch.randint(2,(n,4,18,18,1),dtype=torch.int8)*2-1,
        couplings=torch.tensor(np.column_stack([np.linspace(.3,.6,n),np.linspace(-.01,.01,n)]),dtype=torch.float64))


def test_strict_config_and_cli(tmp_path,monkeypatch):
    path='configs/ising2d_KB_cnn_supervised_short_seed91.json'
    assert run.validate_config(path)==run.preset()
    from recurrent_rg.cli import main
    calls=[];monkeypatch.setattr(run,'run',lambda *a:calls.append(a))
    main(['train','--config',path,'--output',str(tmp_path/'out')]);assert len(calls)==1
    for extra in (['--resume','x'],['--initialize-from','x'],['--allow-legacy-training'],['--device','cuda:2']):
        with pytest.raises(SystemExit):main(['train','--config',path,'--output','x']+extra)
    for key,value in [('supervision_weight',1),('hard_seconds',1201),('block_size',2),('seed',True)]:
        obj=run.preset();obj[key]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(obj))
        with pytest.raises(ValueError):run.validate_config(p)


def test_train_only_scaler_and_paired_draw():
    ds=data();scaler=run.label_scaler(ds.couplings)
    z=run.standardized(ds.couplings.requires_grad_(),scaler,'cpu')
    assert not z.requires_grad
    torch.testing.assert_close(z.mean(0),torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
    torch.testing.assert_close(z.std(0,unbiased=False),torch.ones(2,dtype=torch.float64))
    g1=torch.Generator().manual_seed(13);g2=torch.Generator().manual_seed(13)
    batch,labels=run.draw(ds,g1,'cpu',scaler)
    ids=torch.randperm(len(ds.configurations),generator=g2)[:32]
    samples=torch.randperm(4,generator=g2)[:64]
    torch.testing.assert_close(batch,ds.configurations[ids[:,None],samples[None,:]].double())
    torch.testing.assert_close(labels,z[ids])
    assert torch.equal(g1.get_state(),g2.get_state())
    with pytest.raises(ValueError):run.label_scaler(torch.ones((4,2)))


def test_supervision_updates_only_Q_and_zero_weight_replays_original():
    cfg=kb.build_config();model=binary.fresh(cfg,'cpu');ds=data(2)
    x=ds.configurations.double();scaler=run.label_scaler(ds.couplings)
    labels=run.standardized(ds.couplings,scaler,'cpu').requires_grad_()
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    old=copy.deepcopy(model);zero=copy.deepcopy(model)
    outputs=[]
    for m,w in ((model,10.),(zero,0.)):
        opts=kb.optimizers(m)
        with isolated(7):values=run.step(m,x,labels,hist,torch.arange(2),patterns,cfg,opts,weight=w)
        outputs.append(values);assert all(np.isfinite(v) for v in values.values())
        assert common.optimizer_step_counts(opts,1)['R']['max']==1
    with isolated(7):binary.step(old,x,hist,torch.arange(2),patterns,cfg,kb.optimizers(old))
    assert common.state_fingerprint(old)==common.state_fingerprint(zero)
    for name in ('C','R'):
        assert common.state_fingerprint(binary.modules(model)[name])==common.state_fingerprint(binary.modules(zero)[name])
    assert common.state_fingerprint(model.theory_encoder)!=common.state_fingerprint(zero.theory_encoder)
    assert labels.grad is None
    assert outputs[0]['q_total']==pytest.approx(outputs[0]['q_representation']+10*outputs[0]['q_supervision'])


def isolated(seed):return binary.isolated_rng('cpu',seed)


def test_budget_freezes_endpoint_from_timing_only():
    value=run.choose_steps(40,[20,1]+[.5]*8,.05,.04)
    assert 200<=value['formal_steps']<=3000 and value['formal_steps']%50==0
    assert value['conservative_total_seconds']<1080
    assert run.choose_steps(40,[200,10]+[.5]*8,.05,.04)==value
    with pytest.raises(common.BudgetHold):run.choose_steps(1000,[1.]*10,.05,.04)


def test_probe_frozen_and_training_labels_do_not_enter_closure():
    m=binary.fresh(kb.build_config(),'cpu');ds=data();sc=run.label_scaler(ds.couplings)
    before=common.state_fingerprint(m);rng=torch.get_rng_state().clone()
    first=run.probe(m,ds,'cpu',sc)
    changed=SimpleNamespace(configurations=ds.configurations,couplings=ds.couplings+1)
    second=run.probe(m,changed,'cpu',sc)
    assert first['weighted_closure']==second['weighted_closure']
    assert first['supervised_mse']!=second['supervised_mse']
    assert before==common.state_fingerprint(m) and torch.equal(rng,torch.get_rng_state())


def test_supervised_model_free_report_and_label_metrics(tmp_path):
    rng=np.random.default_rng(4)
    labels=np.array([[.2+.03*k,-.01+.001*b] for k in range(17) for b in range(21) for _ in range(2)])
    splits=np.array(['test' if (k+abs(b-10))%3==1 else 'train' for k in range(17) for b in range(21) for _ in range(2)])
    scaler=run.label_scaler(labels[splits=='train']);z=(labels-scaler['mean'])/scaler['std']
    g=np.stack([z,z*.9,z*.8],1)
    a=dict(direct_g=g,direct_full_g=g,one_step_g=g[:,1:]+.1,rollout_g=g+.1,
        all_start_g=np.stack([g[:,1]+.1,g[:,2]+.2,g[:,2]+.1],1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=labels,splits=splits,cell_ids=np.repeat(np.arange(357),2),train_direct=g[splits=='train'],test_direct=g[splits=='test'])
    metrics=run.supervision_metrics(a,scaler)
    assert metrics['test']['direct_full_g']['r2']==pytest.approx([1.,1.])
    roots,flow=bounded_flow(SimpleNamespace(advance=lambda x:1.1*x),a,'cpu')
    manifest=dict(recipe=run.RECIPE,actual_planned_steps=200,labels_used_in_training=True,
        test_used_in_training=False,label_scaler=scaler,q_supervision=dict(scale=0,coarse_labels=False))
    summary=dict(recipe=run.RECIPE,status='complete',actual_steps=200,supervision=metrics,
        closure=binary.closure_metrics(a),optimizer_steps={k:dict(min=200,max=200) for k in ('Q','C','R')})
    for name,value in [('manifest',manifest),('summary',summary),('roots',dict(roots=roots,flow_analysis=flow)),
        ('history',[dict(step=200,q_total=.1,c_total=.1,r_total=.1,q_supervision=.1)]),
        ('train_probes',[dict(step=0,weighted_closure=.2,supervised_mse=1.),dict(step=200,weighted_closure=.1,supervised_mse=.1)])]:
        common.write_json(tmp_path/(name+'.json'),value)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    from scripts.report_ising_kb_supervised_short import render
    result=render(tmp_path)
    assert result['actual_steps']==200 and result['thermal_nu'] is None
    assert len(list((tmp_path/'report').glob('*.png')))==6
