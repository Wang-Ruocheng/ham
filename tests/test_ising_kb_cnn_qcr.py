"""Synthetic regression only: no new physical data or formal scientific run."""
import copy
import json
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import ising_kb_cnn_qcr as run
from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from recurrent_rg.checkpoints import build_model, load_model_state, qcr_state_dict
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.phi4_ru_flow import bounded_flow


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def data(n=4):
    return SimpleNamespace(configurations=torch.randint(2,(n,4,18,18,1),dtype=torch.int8)*2-1,
        couplings=torch.tensor([[.44, b] for b in np.linspace(-.01,.01,n)],dtype=torch.float64))


def test_strict_config_and_standard_recipe(tmp_path):
    cfg=run.build_config()
    assert cfg.schema_version==5 and cfg.model.theory_dim==2 and cfg.model.block_factor==3
    assert cfg.loss.closure_objective=='all_start_anchor' and cfg.loss.closure_scale_weights is None
    assert cfg.training.epochs==6000 and cfg.training.r_updates_per_batch==1
    assert not cfg.data.spin_flip_augment and not cfg.data.spin_flip_symmetric
    value=run.validate_config('configs/ising2d_KB_cnn_b3_g2_Rlr003_seed91.json')
    for key,v in [('steps',100),('theory_dim',1),('r_updates_per_batch',3),('hard_seconds',1200),('seed',True)]:
        bad=copy.deepcopy(value); bad[key]=v
        path=tmp_path/'bad.json'; path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):run.validate_config(path)


def test_g2_encoder_preserves_random_C_R_initialization_and_serialization():
    cfg=run.build_config()
    torch.manual_seed(91);a=build_model(cfg.model,2)
    torch.manual_seed(91);b=build_model(replace(cfg.model,q_encoder='local_motif'),2)
    for name in ('rg_kernel','theory_rg'):
        assert common.state_fingerprint(getattr(a,name))==common.state_fingerprint(getattr(b,name))
    assert a.theory_encoder.distribution_decoder.weight.shape==(16,2)
    x=data(2).configurations.to(torch.float64)
    assert a.theory_encoder(x).shape==(2,2)
    torch.testing.assert_close(a.theory_encoder(x),a.theory_encoder(x.flip(1)))
    load_model_state(a,dict(model=qcr_state_dict(a)))


def test_independent_graphs_loss_formula_one_step_clocks_and_R_lr_only():
    cfg=run.build_config();m=binary.fresh(cfg,'cpu');x=run.draw_batch(data(2).configurations,torch.Generator().manual_seed(3),'cpu')
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    ids=torch.arange(2)
    losses,terms,_=binary.objectives(m,x,hist,ids,patterns,cfg)
    for name,forbidden in [('Q',('C','R')),('C',('Q','R'))]:
        ps=[p for n in forbidden for p in binary.modules(m)[n].parameters()]
        assert all(g is None for g in torch.autograd.grad(losses[name],ps,allow_unused=True,retain_graph=True))
    expected=5*terms['all_start_pair_0_1']+2.5*(terms['all_start_pair_0_2']+terms['all_start_pair_1_2'])
    assert float(losses['R'])==pytest.approx(float(expected))
    # The shared objective can retain Q in its graph. The existing trainer
    # differentiates R only w.r.t. R parameters (not total_loss.backward()).
    # Changing R's update below must leave Q and C bitwise identical.
    other=copy.deepcopy(m); opts=run.optimizers(m); oldopts=run.optimizers(other)
    oldopts['R'].param_groups[0]['lr']=.001
    assert {n:o.param_groups[0]['lr'] for n,o in opts.items()}=={'Q':.001,'C':.001,'R':.003}
    for model,opt in [(m,opts),(other,oldopts)]:
        with binary.isolated_rng('cpu',372):
            values=binary.step(model,x,hist,ids,patterns,cfg,opt)
        assert all(np.isfinite(v) for v in values.values())
    clocks=common.optimizer_step_counts(opts,1)
    assert all(v['min']==v['max']==1 for v in clocks.values())
    for n in ('Q','C'):
        assert common.state_fingerprint(binary.modules(m)[n])==common.state_fingerprint(binary.modules(other)[n])
    assert common.state_fingerprint(m.theory_rg)!=common.state_fingerprint(other.theory_rg)


def test_bias_probe_label_blind_private_rng_and_frozen_collection():
    cfg=run.build_config();m=binary.fresh(cfg,'cpu');d=data(4)
    rng=torch.get_rng_state().clone()
    receipt=run.initialize_bias(m,d,'cpu')
    assert receipt['C_R_unchanged'] and torch.equal(rng,torch.get_rng_state())
    before=common.state_fingerprint(m)
    probe=run.train_probe(m,d,'cpu')
    changed=SimpleNamespace(configurations=d.configurations,couplings=d.couplings+100)
    assert probe==run.train_probe(m,changed,'cpu')
    train=SimpleNamespace(configurations=d.configurations[:2],couplings=d.couplings[:2])
    test=SimpleNamespace(configurations=d.configurations[2:],couplings=d.couplings[2:])
    a=run.collect(m,train,test,'cpu',lambda:None)
    assert a['direct_g'].shape==(4,3,2) and a['all_start_g'].shape==(4,3,2)
    assert len(np.unique(a['cell_ids']))==4  # Same K, distinct B must not merge.
    changed=SimpleNamespace(configurations=train.configurations,couplings=train.couplings+100)
    b=run.collect(m,changed,test,'cpu',lambda:None)
    np.testing.assert_array_equal(a['direct_g'],b['direct_g'])
    assert before==common.state_fingerprint(m) and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())
    empty=SimpleNamespace(configurations=d.configurations[:0],couplings=d.couplings[:0])
    tiny=run.collect(m,train,empty,'cpu',lambda:None)
    assert tiny['test_direct'].shape==(0,3,2)


def test_standard_cli_route_and_legacy_resume_forbidden(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    config=tmp_path/'config.json';config.write_text(json.dumps(run.preset()))
    called=[];monkeypatch.setattr(run,'run',lambda *args:called.append(args))
    main(['train','--config',str(config),'--output',str(tmp_path/'out')])
    assert len(called)==1
    for args in (['--resume','x'],['--allow-nondefault-recipe'],['--allow-legacy-training']):
        with pytest.raises(SystemExit):main(['train','--config',str(config),'--output','bad',*args])
    assert len(called)==1


def test_projection_fixed_6000_including_pilot_probe_and_full_eval():
    assert run.projected_seconds(40,[10,8]+[1]*8,1,1)==40+9000+37.5+1608
    assert run.projected_seconds(40,[10,8]+[1]*8,1,1)>run.preset()['hard_seconds']


def test_int8_cache_conversion_preserves_batch_values_and_rng():
    x=data(5).configurations
    first=torch.Generator().manual_seed(192);second=torch.Generator().manual_seed(192)
    actual=run.draw_batch(x,first,'cpu')
    expected=binary.draw(x.double(),second,'cpu')
    assert actual.dtype==torch.float64 and x.dtype==torch.int8
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(first.get_state(),second.get_state())


def test_model_free_chinese_report_complete_grid(tmp_path):
    from scripts.report_ising_kb_cnn_qcr import render
    rng=np.random.default_rng(3);g=rng.normal(size=(714,3,2))
    labels=np.array([[.2+.03*ki,-.01+.001*bi] for ki in range(17) for bi in range(21) for rep in range(2)])
    splits=np.array(['test' if (ki+abs(bi-10))%3==1 else 'train' for ki in range(17) for bi in range(21) for rep in range(2)])
    a=dict(direct_g=g,one_step_g=g[:,1:]+.1,rollout_g=np.concatenate([g[:,:1],g[:,1:]+.2],axis=1),
        all_start_g=np.stack([g[:,1]+.1,g[:,2]+.2,g[:,2]+.1],axis=1),
        all_start_pairs=np.array([[0,1],[0,2],[1,2]]),labels=labels,splits=splits,
        cell_ids=np.repeat(np.arange(357),2),train_direct=g[splits=='train'],test_direct=g[splits=='test'])
    roots,flow=bounded_flow(SimpleNamespace(advance=lambda x:1.1*x),a,'cpu')
    assert len(roots)==1 and roots[0]['classification']=='repelling'
    manifest=dict(recipe=run.RECIPE,contract=run.preset(),labels_used_in_training=False,
        test_used_in_training=False,evaluation_label_columns=['K','B'])
    summary=dict(recipe=run.RECIPE,status='complete',actual_steps=6000,final_MI=dict(train=.1,test=.1),
        optimizer_steps={k:dict(min=6000,max=6000) for k in ('Q','C','R')},closure=binary.closure_metrics(a))
    for name,obj in [('manifest',manifest),('summary',summary),('history',[dict(step=6000,q_total=.1,c_total=.2,r_total=.3)]),
            ('roots',dict(roots=roots,flow_analysis=flow)),('train_probes',[dict(step=0,weighted_closure=.1),dict(step=6000,weighted_closure=.01)])]:
        common.write_json(tmp_path/(name+'.json'),obj)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    assert render(tmp_path)['thermal_nu'] is None
    assert len(list((tmp_path/'report').glob('*.png')))==5
