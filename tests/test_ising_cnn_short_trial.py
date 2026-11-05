from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from recurrent_rg.config import ModelConfig,RunConfig
from recurrent_rg.checkpoints import build_model,load_model_state,qcr_state_dict
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.rsmi import kernel_probabilities
from experiments.ising_cnn_short_trial import objectives,step,optimizers,RECIPE
from scripts.report_phi4_hopping_qcr import closure_metrics

@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)

def config():return RunConfig.load('configs/ising2d_B0_cnn_b3_g1_short_seed91.json')

def test_source_pins_skip_virtual_modules(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    from experiments.ising_cnn_short_trial import loaded_source_files
    fake=SimpleNamespace(__file__=str(tmp_path/'_ops.py'))
    monkeypatch.setitem(sys.modules,'recurrent_rg._virtual_test',fake)
    assert loaded_source_files(tmp_path)==[]

def test_architecture_old_default_and_seed_compatibility():
    cfg=config();torch.manual_seed(91);a=build_model(cfg.model,2)
    torch.manual_seed(91);b=build_model(replace(cfg.model,q_encoder='local_motif'),2)
    for name in ['rg_kernel','theory_rg']:
        assert all(torch.equal(x,y) for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters()))
    assert ModelConfig().q_encoder=='local_motif'
    assert a.theory_encoder.distribution_decoder.weight.shape==(16,1)
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1
    assert a.theory_encoder(x).shape==(2,1)
    torch.testing.assert_close(a.theory_encoder(x),a.theory_encoder(x.flip(1)))
    load_model_state(a,dict(model=qcr_state_dict(a)))
    with pytest.raises(ValueError):ModelConfig(q_encoder='configuration_cnn',dimension=1)

def test_actual_mi_and_independent_gradients():
    from itertools import product
    cfg=config();m=build_model(cfg.model,2);m.detach_kernel_representation=True
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    ids=torch.tensor([0,1]);table=kernel_probabilities(m.rg_kernel,patterns)[0]
    torch.testing.assert_close(hist.information_for(table,ids),hist.information(table))
    losses,terms,_=objectives(m,x,hist,ids,patterns,cfg)
    assert float(losses['R'])==pytest.approx(float(5*terms['all_start_pair_0_1']+2.5*(terms['all_start_pair_0_2']+terms['all_start_pair_1_2'])))
    assert all(g is None for g in torch.autograd.grad(losses['Q'],list(m.rg_kernel.parameters())+list(m.theory_rg.parameters()),allow_unused=True,retain_graph=True))
    assert all(g is None for g in torch.autograd.grad(losses['C'],list(m.theory_encoder.parameters())+list(m.theory_rg.parameters()),allow_unused=True,retain_graph=True))
    before={n:[p.detach().clone() for p in mod.parameters()] for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel),('R',m.theory_rg)]}
    opts=optimizers(m);metrics=step(m,x,hist,ids,patterns,cfg,opts)
    assert all(np.isfinite(v) for v in metrics.values())
    for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel),('R',m.theory_rg)]:
        assert any(not torch.equal(p,b) for p,b in zip(mod.parameters(),before[n]))
    assert all(int(s['step'])==1 for opt in opts.values() for s in opt.state.values())

def test_discrete_report(tmp_path):
    from scripts.report_ising_cnn_short_trial import render
    rng=np.random.default_rng(7);g=rng.normal(size=(192,3,1))
    arrays=dict(direct_g=g,one_step_g=g[:,1:]+.01,rollout_g=np.concatenate([g[:,:1],g[:,1:]+.02],axis=1),
        all_start_g=np.stack([g[:,1]+.01,g[:,2]+.02,g[:,2]+.01],axis=1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=np.repeat(np.linspace(.2,.68,64),3)[:,None],splits=np.array(['train']*129+['test']*63),
        flow_grid=np.linspace(-3,3,501)[:,None],flow_next=np.linspace(-3,3,501)[:,None]*1.1)
    cfg=config().to_dict();cfg['training']['epochs']=25
    for name,value in [('manifest',dict(recipe=RECIPE,config=cfg)),('summary',dict(actual_steps=25,closure=closure_metrics(arrays),optimizer_steps={n:dict(min=25,max=25) for n in ['Q','C','R']})),
                       ('history',[dict(step=25,q_total=.1,c_total=.2,r_total=.3)]),('roots',dict(roots=[]))]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    np.savez(tmp_path/'evaluation_arrays.npz',**arrays)
    assert render(tmp_path)['status']=='complete'
    assert len(list((tmp_path/'report').glob('*.png')))==4
