"""Synthetic CPU checks; never physical sampling or a scientific result."""
import copy
import numpy as np
import pytest
import torch
from recurrent_rg.phi4_cnn_qcr import ConfigurationCNNQ, CNNPhi4QCR
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, aligned_objectives
from experiments import phi4_hopping_qcr as old
from experiments import phi4_hopping_cnn_qcr as new


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_encoder_scalar_symmetry_and_chunks():
    q=ConfigurationCNNQ(.9,frame_chunk=8)
    x=torch.randn(2,4,12,12,1,dtype=torch.float64)
    with torch.no_grad():
        y=q(x)
        assert y.shape==(2,1)
        torch.testing.assert_close(y,q(x.flip(1)),atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(y,q(x.roll((2,3),(2,3))),atol=1e-12,rtol=1e-10)
        q.frame_chunk=3
        torch.testing.assert_close(y,q(x),atol=1e-12,rtol=1e-10)
    assert isinstance(q.decoder,torch.nn.Linear)
    torch.testing.assert_close(q.decoder.weight.norm(dim=0),torch.ones(1,dtype=torch.float64))


def test_historical_c_r_initialization_and_global_rng_preserved():
    torch.manual_seed(91)
    original=AlignedPhi4QCR(.9)
    rng=torch.get_rng_state().clone()
    model=CNNPhi4QCR(.9)
    assert torch.equal(rng,torch.get_rng_state())
    for name in ('C','R'):
        assert old.state_fingerprint(getattr(original,name))==old.state_fingerprint(getattr(model,name))


def test_independent_gradient_graphs_and_uniform_target_weights():
    model=CNNPhi4QCR(.9,frame_chunk=8)
    x=torch.randn(2,4,18,18,1,dtype=torch.float64)
    q,c,r,terms=aligned_objectives(model,x,generator=torch.Generator().manual_seed(8),information_pairs=8)
    for loss,forbidden in ((q,['C','R']),(c,['Q','R']),(r,['Q','C'])):
        ps=[p for name in forbidden for p in getattr(model,name).parameters()]
        assert all(g is None for g in torch.autograd.grad(loss,ps,allow_unused=True,retain_graph=True))
    expected=5*terms['closure_start_0_target_1']+2.5*(terms['closure_start_0_target_2']+terms['closure_start_1_target_2'])
    assert terms['r_total']==pytest.approx(expected)


def test_all_three_models_update_and_optimizer_clocks():
    model=CNNPhi4QCR(.9,frame_chunk=8)
    before=old.module_fingerprints(model)
    optim=new.optimizers(model)
    x=torch.randn(2,4,18,18,1,dtype=torch.float64)
    for i in range(2):
        metrics=old.independent_step(model,x,optim,generator=torch.Generator().manual_seed(9+i),information_pairs=8)
        assert all(np.isfinite(v) for v in metrics.values())
    counts=old.optimizer_step_counts(optim,2)
    assert all(counts[k]['min']==counts[k]['max']==2 for k in counts)
    assert all(before[k]!=old.module_fingerprints(model)[k] for k in before)


def test_cnn_forward_matches_previous_encoder_with_same_weights():
    from scripts.run_phi4_hopping_qonly_encoders import HierarchicalQ
    q=ConfigurationCNNQ(.9,frame_chunk=3)
    reference=HierarchicalQ(.9,copy.deepcopy(q.decoder),frame_chunk=3)
    reference.load_state_dict(q.state_dict())
    x=torch.randn(2,4,12,12,1,dtype=torch.float64)
    torch.testing.assert_close(q(x),reference.encode(x,None),rtol=0,atol=0)


def test_budget_endpoint_only_from_timing():
    assert new.choose_steps(100,1,100)==700
    assert new.choose_steps(20,.01,90)==3000
    with pytest.raises(old.BudgetHold):
        new.choose_steps(800,10,100)


def test_root_slope_full_map_not_residual():
    model=CNNPhi4QCR(1.)
    model.R=torch.nn.Linear(1,1).double()
    with torch.no_grad():
        model.R.weight.fill_(1.);model.R.bias.fill_(0.)
    a=dict(flow_grid=np.array([[-1.],[1.]]),direct_g=np.array([[[-.5],[.5],[.8]]]),splits=np.array(['train']))
    roots=new.root_diagnostics(model,a)
    assert len(roots['roots'])==1
    assert roots['roots'][0]['full_R_derivative']==2.
    assert roots['roots'][0]['inside_train_input_envelope']
    assert roots['thermal_nu'] is None


@pytest.mark.parametrize('ratio', [1, 10])
def test_report_synthetic_end_to_end(tmp_path, ratio):
    from scripts.report_phi4_hopping_cnn_qcr import render
    rng=np.random.default_rng(3)
    direct=rng.normal(size=(60,3,1))
    arrays=dict(direct_g=direct,one_step_g=direct[:,:-1]+.1,
                rollout_g=np.concatenate([direct[:,:1],direct[:,1:]+.1],axis=1),
                all_start_predictions=np.stack([direct[:,1]+.1,direct[:,2]+.1,direct[:,2]+.2],axis=1),
                all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
                splits=np.array(['train']*40+['test']*20),cell_ids=np.repeat(np.arange(15),4),
                labels=np.column_stack([np.repeat(np.linspace(.32,1.04,15),4),np.ones(60),np.zeros(60)]),
                flow_grid=np.linspace(-3,3,501)[:,None],flow_next=np.linspace(-3,3,501)[:,None]*1.1)
    # Use the evaluator's actual array names/semantics for closure metrics.
    arrays['all_start_g']=arrays.pop('all_start_predictions')
    base_manifest=dict(recipe=new.RECIPE if ratio == 1 else new.REFINED_RECIPE,
                       options=dict(steps=25,r_updates_per_batch=ratio),labels_used_in_training=False)
    summary=dict(actual_steps=25,optimizer_steps={k:dict(min=25*(ratio if k=='R' else 1),max=25*(ratio if k=='R' else 1)) for k in ('Q','C','R')},
                 closure=old.closure_metrics(arrays))
    for name,obj in [('manifest',base_manifest),('summary',summary),('history',[dict(step=25,q_total=.1,c_total=.2,r_total=.3,R_batch_after=.2)]),
                     ('roots',dict(roots=[]))]:
        old.write_json(tmp_path/f'{name}.json',obj)
    np.savez(tmp_path/'evaluation_arrays.npz',**arrays)
    result=render(tmp_path)
    assert result['status']=='complete'
    assert len(list((tmp_path/'report').glob('*.png')))==4


def test_refinement_preserves_qc_rng_and_exact_targets():
    from types import SimpleNamespace
    from recurrent_rg.config import LossConfig
    from recurrent_rg.r_refinement import closure_on_targets
    torch.manual_seed(163)
    batch=torch.randn(2,4,18,18,1,dtype=torch.double)
    models=[CNNPhi4QCR(.9,frame_chunk=8) for _ in range(2)]
    opts=[new.optimizers(m) for m in models]
    rngs=[torch.Generator().manual_seed(45) for _ in models]
    global_rng=torch.get_rng_state().clone()
    for i in range(2):
        for model,opt,rng,ratio in zip(models,opts,rngs,(1,10)):
            old.independent_step(model,batch,opt,generator=rng,information_pairs=8,r_updates_per_batch=ratio)
        assert torch.equal(rngs[0].get_state(),rngs[1].get_state())
        assert torch.equal(global_rng,torch.get_rng_state())
        for name in ('Q','C'):
            assert old.state_fingerprint(getattr(models[0],name))==old.state_fingerprint(getattr(models[1],name))
            assert all(p.grad is None for p in getattr(models[1],name).parameters())
            for a,b in zip(opts[0][name].state.values(),opts[1][name].state.values()):
                for key in a:
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    counts=old.optimizer_step_counts(opts[1],2,r_updates_per_batch=10)
    assert counts['R']['min']==counts['R']['max']==20
    assert counts['Q']['min']==counts['C']['max']==2
    model=models[0]
    targets=[]
    _,_,r,_=aligned_objectives(model,batch,generator=torch.Generator().manual_seed(50),
                               information_pairs=8,closure_targets_sink=targets)
    assert len(targets)==1 and not targets[0].requires_grad
    adapter=SimpleNamespace(theory_rg=model.R,advance_theory=model.advance_theory)
    loss=closure_on_targets(adapter,targets[0],LossConfig())
    torch.testing.assert_close(loss,r,rtol=0,atol=0)
    a=torch.autograd.grad(loss,list(model.R.parameters()))
    b=torch.autograd.grad(r,list(model.R.parameters()))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_refined_config_and_standard_dispatch(monkeypatch,tmp_path):
    from recurrent_rg.cli import main
    from pathlib import Path
    preset=Path(__file__).resolve().parents[1]/'configs/phi4_hopping_b3_cnn_refinedR_seed91.json'
    assert new.validate_refined_config(preset)==new.refined_config()
    calls=[]
    monkeypatch.setattr(new,'main',lambda argv:calls.append(argv))
    main(['train','--config',str(preset),'--output',str(tmp_path/'run')])
    assert calls==[['--config',str(preset),'--output',str(tmp_path/'run'),'--execute']]
    with pytest.raises(SystemExit):
        main(['train','--config',str(preset),'--output',str(tmp_path/'run'),'--resume','old.pt'])
    invalid=new.refined_config();invalid['r_updates_per_batch']=1
    old.write_json(tmp_path/'invalid.json',invalid)
    with pytest.raises(ValueError):
        new.validate_refined_config(tmp_path/'invalid.json')
