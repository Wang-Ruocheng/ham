"""Synthetic checks only; no physical sampling, GPU admission or new results."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import phi4_ru_cnn_qcr as run
from experiments import phi4_hopping_qcr as base
from experiments.phi4_qcr import FieldDataset
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
from recurrent_rg.phi4_aligned_qcr import aligned_objectives
from recurrent_rg.phi4_ru_flow import bounded_flow


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def data(count=4):
    return FieldDataset(torch.randn(count,4,18,18,1,dtype=torch.float64),
        np.array(['train']*(count-1)+['test']),np.arange(count),np.zeros((count,3)),
        'synthetic',[],0,'synthetic only')


def test_config_fixed_endpoint_and_learning_rate_only_changes_in_optimizer(tmp_path):
    cfg=run.validate_config('configs/phi4_ru_b3_cnn_g2_Rlr003_seed91.json')
    assert cfg['steps']==6000 and cfg['r_updates_per_batch']==1
    assert cfg['loss']==base.LOSS_POLICY and not cfg['early_stopping']
    model=CNNPhi4QCR(1.,theory_dim=2)
    opts=run.optimizers(model)
    assert {n:o.param_groups[0]['lr'] for n,o in opts.items()}=={'Q':.001,'C':.001,'R':.003}
    for key,value in [('steps',900),('r_updates_per_batch',3),('theory_dim',1),('hard_seconds',1200)]:
        changed=copy.deepcopy(cfg);changed[key]=value
        path=tmp_path/'bad.json';path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):run.validate_config(path)


def test_scalar_default_stays_bitwise_compatible():
    a=CNNPhi4QCR(.9);b=CNNPhi4QCR(.9,theory_dim=1)
    assert base.state_fingerprint(a)==base.state_fingerprint(b)
    for invalid in (True,0,-1,1.5):
        with pytest.raises(ValueError):CNNPhi4QCR(.9,theory_dim=invalid)


def test_two_dimensional_independent_graphs_and_allstart_formula():
    m=CNNPhi4QCR(1.,theory_dim=2)
    d=data()
    q,c,r,terms=aligned_objectives(m,d.configurations[:2],generator=torch.Generator().manual_seed(4),information_pairs=8)
    assert m.Q(d.configurations[:2]).shape==(2,2)
    assert m.Q.decoder.weight.shape==(625,2)
    for loss,forbidden in ((q,('C','R')),(c,('Q','R')),(r,('Q','C'))):
        ps=[p for name in forbidden for p in getattr(m,name).parameters()]
        assert all(g is None for g in torch.autograd.grad(loss,ps,allow_unused=True,retain_graph=True))
    expected=5*terms['closure_start_0_target_1']+2.5*(terms['closure_start_0_target_2']+terms['closure_start_1_target_2'])
    assert float(r)==pytest.approx(expected)


def test_higher_R_lr_does_not_change_QC_or_RNG_and_only_one_step():
    a=CNNPhi4QCR(1.,theory_dim=2);b=copy.deepcopy(a)
    oa=run.optimizers(a);ob=run.optimizers(b)
    ob['R'].param_groups[0]['lr']=.001
    before=base.module_fingerprints(a);d=data()
    for i in range(2):
        for model,opts in ((a,oa),(b,ob)):
            base.independent_step(model,d.configurations[:2],opts,
                generator=torch.Generator().manual_seed(40+i),information_pairs=8)
        for name in ('Q','C'):
            assert base.state_fingerprint(getattr(a,name))==base.state_fingerprint(getattr(b,name))
    assert base.state_fingerprint(a.R)!=base.state_fingerprint(b.R)
    clocks=base.optimizer_step_counts(oa,2)
    assert all(c['min']==c['max']==2 for c in clocks.values())
    assert all(before[n]!=base.module_fingerprints(a)[n] for n in before)


def test_train_probe_is_label_blind_and_preserves_model_gradients_and_rng():
    d=data(18);m=CNNPhi4QCR(1.,theory_dim=2)
    # Probe expects64 fields; small fixture must still have two32-field halves.
    d=replace(d,configurations=d.configurations.repeat(1,16,1,1,1))
    before=base.state_fingerprint(m);rng=torch.get_rng_state().clone()
    a=run.train_probe(m,d,'cpu')
    b=run.train_probe(m,replace(d,labels=d.labels+9000),'cpu')
    assert a==b and before==base.state_fingerprint(m)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())


def test_evaluation_is_dimension_generic_label_only_attached_after_inference():
    d=data();m=CNNPhi4QCR(1.,theory_dim=2)
    a=run.evaluate(m,d,'cpu',lambda:None)
    b=run.evaluate(m,replace(d,labels=d.labels+32),'cpu',lambda:None)
    assert a['direct_g'].shape==(4,3,2)
    assert a['all_start_g'].shape==(4,3,2)
    for key in a:
        if key!='labels':np.testing.assert_array_equal(a[key],b[key])
    assert a['train_direct'].shape==(3,3,2)
    assert base.closure_metrics(a)['test']['weighted_all_start_objective']>=0
    assert all(p.grad is None for p in m.parameters())


def test_standard_cli_routes_and_forbids_resume(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    config=tmp_path/'config.json';config.write_text(json.dumps(run.preset()))
    called=[]
    monkeypatch.setattr(run,'run',lambda *a:called.append(a))
    main(['train','--config',str(config),'--output',str(tmp_path/'out')])
    assert len(called)==1
    with pytest.raises(SystemExit):main(['train','--config',str(config),'--output','bad','--resume','x'])


def test_budget_projection_keeps_6000_and_counts_pilot_eval_probes():
    assert run.projected_budget(40,[10,8]+[1]*8,1,1,180)==40+9000+37.5+540
    assert run.projected_budget(40,[10,8]+[1]*8,1,1,180)>run.preset()['hard_seconds']


@pytest.mark.parametrize('nonlinear_decoder', [False, True, 'local', 'kappa_h'])
def test_model_free_report_complete_synthetic(tmp_path, nonlinear_decoder):
    from scripts.report_phi4_ru_cnn_qcr import render
    rng=np.random.default_rng(7);g=rng.normal(size=(180,3,2))
    labels=np.array([[r,u,0.] for r in np.linspace(-1.55,-.5,9) for u in np.linspace(.5,1,5) for _ in range(4)])
    splits=np.array(['test' if c%4==1 else 'train' for c in range(45) for _ in range(4)])
    a=dict(direct_g=g,one_step_g=g[:,1:]+.1,rollout_g=np.concatenate([g[:,:1],g[:,1:]+.2],axis=1),
        all_start_g=np.stack([g[:,1]+.1,g[:,2]+.2,g[:,2]+.1],axis=1),
        all_start_pairs=np.array([[0,1],[0,2],[1,2]]),labels=labels,splits=splits,
        cell_ids=np.repeat(np.arange(45),4),train_direct=g[splits=='train'],test_direct=g[splits=='test'])
    model=CNNPhi4QCR(1.,theory_dim=2)
    model.R=torch.nn.Linear(2,2).double()
    with torch.no_grad():model.R.weight.copy_(torch.eye(2)*.1);model.R.bias.zero_()
    roots,flow=bounded_flow(model,a,'cpu')
    assert len(roots)==1 and roots[0]['classification']=='repelling'
    from experiments.phi4_ru_cnn_nonlinear import preset as nonlinear_preset
    config=nonlinear_preset() if nonlinear_decoder else run.preset()
    if nonlinear_decoder == 'local':
        from experiments.phi4_ru_cnn_local_window import preset as local_preset
        config=local_preset('a'*64)
    manifest=dict(config,labels_used_in_training=False,evaluation_label_columns=['r','u','h'],data_warning_count=55)
    if nonlinear_decoder == 'kappa_h':
        from experiments.phi4_kappa_h_cnn import preset as kh_preset, KAPPA_VALUES,H_VALUES
        config=kh_preset('a'*64)
        manifest=dict(config,labels_used_in_training=False,
            evaluation_label_columns=['kappa','h','quartic_lambda'],data_warning_count=55)
        a['labels']=np.array([[k,h,1.] for k in KAPPA_VALUES for h in H_VALUES for _ in range(4)])
    summary=dict(recipe=config['recipe'],status='complete',actual_steps=6000,
        optimizer_steps={k:dict(min=6000,max=6000) for k in ('Q','C','R')},closure=base.closure_metrics(a))
    for name,obj in [('manifest',manifest),('summary',summary),('history',[dict(step=6000,q_loss=.1,c_loss=.2,r_loss=.3)]),
                     ('roots',dict(roots=roots,flow_analysis=flow)),('train_probes',[dict(step=0,weighted_closure=.1),dict(step=6000,weighted_closure=.01)])]:
        base.write_json(tmp_path/(name+'.json'),obj)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    report=render(tmp_path)
    assert report['status']=='complete' and report['thermal_nu'] is None
    assert len(list((tmp_path/'report').glob('*.png')))==5
