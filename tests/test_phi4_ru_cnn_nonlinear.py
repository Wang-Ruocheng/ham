"""Synthetic tests: no physical sampling, CUDA allocation, or formal training."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_ru_cnn_nonlinear as spec
from experiments import phi4_ru_cnn_qcr as trainer
from experiments import phi4_hopping_qcr as base
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
from recurrent_rg.phi4_aligned_qcr import aligned_objectives


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_only_decoder_and_device_differ_from_same_data_control(tmp_path):
    from experiments.phi4_ru_cnn_saddle import preset
    a,b=preset(spec.DATA_SHA),spec.preset()
    a.pop('recipe');b.pop('recipe')
    for key in ('decoder','gpu_index','gpu_uuid'):b.pop(key)
    assert a==b
    config=tmp_path/'config.json';config.write_text(json.dumps(spec.preset()))
    assert trainer.validate_config(config)==spec.preset()
    for key,value in [('steps',3000),('gpu_index',3),('direction_balancing',True)]:
        changed=copy.deepcopy(spec.preset());changed[key]=value
        config.write_text(json.dumps(changed))
        with pytest.raises(ValueError):trainer.validate_config(config)


def test_initial_predictions_C_R_and_rng_match_then_roundtrip():
    affine=CNNPhi4QCR(1.,seed=91,theory_dim=2)
    nonlinear=copy.deepcopy(affine)
    rng=torch.get_rng_state().clone()
    spec.install_decoder(nonlinear)
    assert torch.equal(rng,torch.get_rng_state())
    for name in ('C','R'):
        assert base.state_fingerprint(getattr(affine,name))==base.state_fingerprint(getattr(nonlinear,name))
    x=torch.randn(2,4,9,9,1,dtype=torch.double)
    assert torch.equal(affine.Q(x),nonlinear.Q(x))
    torch.testing.assert_close(affine.Q.decode(affine.Q(x)),nonlinear.Q.decode(nonlinear.Q(x)),atol=1e-14,rtol=0)
    clone=spec.install_decoder(CNNPhi4QCR(1.,seed=91,theory_dim=2))
    clone.load_state_dict(nonlinear.state_dict(),strict=True)
    assert base.state_fingerprint(clone)==base.state_fingerprint(nonlinear)
    with pytest.raises(ValueError):spec.install_decoder(nonlinear)


def test_independent_objectives_and_parameter_clocks():
    a=CNNPhi4QCR(1.,seed=91,theory_dim=2)
    b=spec.install_decoder(copy.deepcopy(a))
    x=torch.randn(2,4,18,18,1,dtype=torch.double)
    oa,ob=trainer.optimizers(a),trainer.optimizers(b)
    for step in range(3):
        for m,o in ((a,oa),(b,ob)):
            base.independent_step(m,x,o,generator=torch.Generator().manual_seed(100+step),information_pairs=8)
    # C's entire optimization path must not depend on Q's reconstruction head.
    assert base.state_fingerprint(a.C)==base.state_fingerprint(b.C)
    assert all(v['min']==v['max']==3 for v in base.optimizer_step_counts(ob,3).values())
    q,c,r,t=aligned_objectives(b,x,generator=torch.Generator().manual_seed(5),information_pairs=8)
    for loss,forbidden in ((q,('C','R')),(c,('Q','R')),(r,('Q','C'))):
        ps=[p for n in forbidden for p in getattr(b,n).parameters()]
        assert all(v is None for v in torch.autograd.grad(loss,ps,allow_unused=True,retain_graph=True))
    assert float(r)==pytest.approx(5*t['closure_start_0_target_1']+2.5*(t['closure_start_0_target_2']+t['closure_start_1_target_2']))
    grads=torch.autograd.grad(q,list(b.Q.parameters()))
    assert all(torch.isfinite(v).all() for v in grads)
    assert b.Q.decoder.net[0].weight.grad.abs().sum()>0
    assert b.Q.decoder.bounds()['lipschitz_upper_bound']<=np.sqrt(2)+8+1e-10


def test_original_bias_initialization_before_nonlinear_upgrade():
    a=CNNPhi4QCR(1.,seed=91,theory_dim=2)
    x=torch.randn(3,4,9,9,1,dtype=torch.double)
    before=base.state_fingerprint(a.C)
    receipt=base.initialize_q_decoder(a,x,np.array([0,1]),seed=91)
    bias=a.Q.decoder.bias.detach().clone()
    spec.install_decoder(a)
    assert torch.equal(a.Q.decoder.affine.bias,bias)
    assert base.state_fingerprint(a.C)==before
    assert receipt['global_cpu_rng_unchanged']


def test_standard_cli_and_no_resume(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    config=tmp_path/'config.json';config.write_text(json.dumps(spec.preset()))
    calls=[]
    monkeypatch.setattr(trainer,'run',lambda *a:calls.append(a))
    main(['train','--config',str(config),'--output',str(tmp_path/'new')])
    assert len(calls)==1
    for flag in ('--resume','--initialize-from'):
        with pytest.raises(SystemExit):main(['train','--config',str(config),'--output','bad',flag,'bad'])


def test_supervisor_plan_is_training_only(tmp_path):
    from scripts.run_phi4_ru_cnn_nonlinear import build_plan
    plan=build_plan(tmp_path/'new',tmp_path)
    assert plan['config']['steps']==6000 and plan['gpu_index']==2
    assert not plan['physical_sampling'] and not plan['checkpoint_warm_start']
    assert not plan['extra_R_only_stage'] and not plan['direction_balancing']
    assert plan['budgets']['active_pipeline_seconds']==7200
    assert not (tmp_path/'new').exists()
