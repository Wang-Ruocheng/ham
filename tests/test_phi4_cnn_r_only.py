"""Synthetic CPU checks only; not a physical experiment."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import phi4_cnn_r_only as run
from experiments import phi4_hopping_qcr as base
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_original_adam_and_ten_steps_continue_exactly():
    model=CNNPhi4QCR(.9,frame_chunk=8)
    x=torch.randn(2,4,18,18,1,dtype=torch.float64)
    direct=run.online_targets(model,x,torch.Generator().manual_seed(1))
    assert not direct.requires_grad
    opt=torch.optim.Adam(model.R.parameters(),lr=.001)
    run.update(model,direct,opt)
    frozen=copy.deepcopy(model)
    frozen_opt=run.freeze_and_continue(frozen,dict(optimizers={'R':opt.state_dict()}),expected=10)
    before={k:base.state_fingerprint(getattr(frozen,k)) for k in ('Q','C')}
    run.update(model,direct,opt)
    run.update(frozen,direct,frozen_opt)
    for a,b in zip(model.R.parameters(),frozen.R.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert before=={k:base.state_fingerprint(getattr(frozen,k)) for k in ('Q','C')}
    assert all(p.grad is None and not p.requires_grad for name in ('Q','C') for p in getattr(frozen,name).parameters())
    assert base.optimizer_step_counts({'R':frozen_opt},20)['R']['parameters']==6


def test_exact_objective_recursive_gradients_and_online_targets():
    from recurrent_rg.phi4_aligned_qcr import aligned_objectives
    model=CNNPhi4QCR(.9,frame_chunk=8)
    x=torch.randn(2,4,18,18,1,dtype=torch.float64)
    targets=[]
    _,_,original,_=aligned_objectives(model,x,generator=torch.Generator().manual_seed(3),
                                     information_pairs=8,closure_targets_sink=targets)
    direct=run.online_targets(model,x,torch.Generator().manual_seed(3))
    torch.testing.assert_close(direct,targets[0],atol=1e-14,rtol=1e-12)
    params=list(model.R.parameters())
    expected_grads=torch.autograd.grad(original,params)
    actual_grads=torch.autograd.grad(run.r_loss(model,direct),params)
    for a,b in zip(expected_grads,actual_grads):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    direct.requires_grad_()
    run.r_loss(model,direct).backward()
    assert direct.grad is None


def test_paired_targets_and_report(tmp_path):
    from scripts.report_ising_cnn_r_only import render
    model=CNNPhi4QCR(.9)
    rng=np.random.default_rng(1)
    original=dict(direct_g=rng.normal(size=(60,3,1)), labels=np.repeat(np.linspace(.32,1.04,15),4)[:,None],
        splits=np.repeat(np.array(['train']*10+['test']*5),4),flow_grid=np.linspace(-3,3,4001)[:,None],
        all_start_pairs=np.array([[0,1],[0,2],[1,2]]))
    a=run.paired_predictions(model,original)
    np.testing.assert_array_equal(a['direct_g'],original['direct_g'])
    a['source_flow_next']=a['flow_next'].copy()
    values=dict(manifest=dict(recipe=run.RECIPE),roots=dict(roots=[]),
        summary=dict(source_joint_updates=900,source_R_updates=9000,actual_extra_R_updates=6000,
            reason='fixed_endpoint',frozen_before_sha256={'Q':'a','C':'b'},frozen_after_sha256={'Q':'a','C':'b'},
            additional_Q_updates=0,additional_C_updates=0,data_warning_count=12,
            before_closure=base.closure_metrics(a),after_closure=base.closure_metrics(a)),
        history=[dict(extra_R_updates=0,train={'total':.1}),dict(extra_R_updates=6000,train={'total':.01})])
    for name,value in values.items():
        base.write_json(tmp_path/(name+'.json'),value)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    report=render(tmp_path,family='phi4')
    assert report['Q_C_unchanged'] and report['thermal_nu'] is None
    assert 'physical_critical_K_certified' not in report
    assert len(list((tmp_path/'report').glob('*.png')))==3


def test_pinned_cli_rejects_extra_changes():
    from experiments.rsmi_separate_qcr import build_parser
    args=build_parser().parse_args(['--source',str(run.SOURCE),'--output','unused',
                                  '--device','cuda:0','--phi4-cnn-r-only'])
    run.validate_args(args)
    args.configuration_cnn_r_only=True
    with pytest.raises(ValueError):run.validate_args(args)
    args.configuration_cnn_r_only=False
    args.rsmi_weight=0
    with pytest.raises(ValueError):run.validate_args(args)


def test_source_not_complete_fails_before_checkpoint(tmp_path):
    base.write_json(tmp_path/'status.json',dict(status='running',workflow_complete=False,returncode=None,error=None))
    with pytest.raises(ValueError,match='incomplete'):run.validate_source(tmp_path)


def test_panel_does_not_consume_global_rng():
    model=CNNPhi4QCR(.9,frame_chunk=8)
    for p in model.parameters():p.requires_grad_(False)
    x=torch.randn(2,4,18,18,1,dtype=torch.float64)
    before=torch.get_rng_state().clone()
    panel=run.train_panel(model,x,torch.device('cpu'),lambda:None)
    assert panel.shape==(6,3,1) and not panel.requires_grad
    assert torch.equal(before,torch.get_rng_state())
