import copy
import json
import numpy as np
import pytest
import torch

from experiments import phi4_hopping_local_current as new
from experiments import phi4_hopping_qcr as base
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, aligned_objectives
from scripts.report_phi4_hopping_local_current import crossings, analyze_arrays


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_same_initial_model_free_gain_only_and_no_global_rng_change():
    torch.manual_seed(91)
    old = AlignedPhi4QCR(.9, patch_chunk_size=131072)
    rng = torch.get_rng_state().clone()
    model = new.CurrentPhi4QCR(.9)
    assert torch.equal(rng, torch.get_rng_state())
    assert base.state_fingerprint(old) == base.state_fingerprint(model)
    with torch.no_grad(): model.Q.decoder.weight.mul_(2)
    model.Q.normalize_decoder_()
    torch.testing.assert_close(model.Q.decoder.weight.norm(dim=0), torch.tensor([2.],dtype=torch.double))
    old.Q.normalize_decoder_()
    torch.testing.assert_close(old.Q.decoder.weight.norm(dim=0), torch.ones(1,dtype=torch.double))


def test_gradient_separation_updates_clocks_and_no_labels():
    model = new.CurrentPhi4QCR(.9, patch_chunk_size=1024)
    x = torch.randn(2,4,18,18,1,dtype=torch.double)
    q,c,r,metrics = aligned_objectives(model,x,generator=torch.Generator().manual_seed(7),information_pairs=8)
    for loss,others in ((q,['C','R']),(c,['Q','R']),(r,['Q','C'])):
        parameters = [p for name in others for p in getattr(model,name).parameters()]
        assert all(v is None for v in torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True))
    assert float(r) == pytest.approx(5*metrics['closure_start_0_target_1'] + 2.5*(metrics['closure_start_0_target_2']+metrics['closure_start_1_target_2']))
    before = base.module_fingerprints(model)
    opts = new.optimizers(model)
    assert {k:o.param_groups[0]['lr'] for k,o in opts.items()} == {'Q':.001,'C':.001,'R':.003}
    for _ in range(2):
        base.independent_step(model,x,opts,generator=torch.Generator().manual_seed(8),information_pairs=8)
    clocks = base.optimizer_step_counts(opts,2)
    assert all(clocks[n]['min']==clocks[n]['max']==2 for n in clocks)
    assert all(before[n] != base.module_fingerprints(model)[n] for n in before)


def test_all_interpolation_branches_and_flat_nonidentifiability():
    assert crossings([0,1,2],[0,2,0],1)==[.5,1.5]
    assert crossings([0,1,2],[0,0,0])==[]
    assert crossings([0,1,2],[-1,0,1])==[1.]


def test_budget_keeps_fixed_length():
    assert new.fixed_budget(20,[.2]*10,1,1)['formal_steps']==1250
    with pytest.raises(base.BudgetHold):new.fixed_budget(20,[2.]*10,1,1)


def test_exact_configuration_and_cli_dispatch(tmp_path,monkeypatch):
    p=tmp_path/'config.json';p.write_text(json.dumps(new.preset()))
    assert new.validate_config(p)['supervision_weight']==0
    from recurrent_rg.cli import main
    calls=[]
    monkeypatch.setattr(new,'run',lambda *a:calls.append(a))
    main(['train','--config',str(p),'--output',str(tmp_path/'out')])
    assert calls==[(str(p),str(tmp_path/'out'))]
    with pytest.raises(SystemExit):main(['train','--config',str(p),'--output',str(tmp_path/'out'),'--allow-legacy-training'])


def test_array_report_end_to_end(tmp_path):
    from scripts.report_phi4_hopping_local_current import render
    direct=np.linspace(-1,1,60)[:,None,None]*np.array([1.,1.1,1.2])[None,:,None]
    a=dict(direct_g=direct,direct_full_g=direct.copy(),one_step_g=direct[:,1:]+.01,
           rollout_g=np.concatenate((direct[:,:1],direct[:,1:]+.01),axis=1),
           all_start_g=np.stack([direct[:,1]+.01,direct[:,2]+.01,direct[:,2]+.01],axis=1),
           all_start_pairs=np.array([[0,1],[0,2],[1,2]]),splits=np.array(['train']*40+['test']*20),
           cell_ids=np.repeat(np.arange(15),4),labels=np.column_stack([np.repeat(np.linspace(.32,1.04,15),4),np.ones(60),np.zeros(60)]),
           flow_grid=np.linspace(-2,2,101)[:,None],flow_next=1.1*np.linspace(-2,2,101)[:,None])
    manifest=dict(recipe=new.RECIPE,options=dict(steps=1250,r_updates_per_batch=1),labels_used_in_training=False)
    summary=dict(actual_steps=1250,optimizer_steps={n:dict(min=1250,max=1250) for n in ('Q','C','R')},closure=base.closure_metrics(a))
    roots=dict(roots=[dict(g=0.,full_R_derivative=1.1,inside_train_input_envelope=True,numerical_stability='repeller')])
    for name,value in [('manifest',manifest),('summary',summary),('history',[dict(step=1250,q_total=.1,c_total=.1,r_total=.1)]),('roots',roots)]:base.write_json(tmp_path/f'{name}.json',value)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    result=render(tmp_path)
    assert len(result['diagnostics']['roots'][0]['kappa_interpolation_by_scale'][0])==1
    assert result['thermal_nu'] is None
    assert (tmp_path/'report/R_flow.png').is_file()
