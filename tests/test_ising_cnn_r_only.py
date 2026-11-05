import copy
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import RunConfig
from experiments.ising_cnn_r_only import freeze_and_continue,r_loss,online_targets,update,paired_predictions,RECIPE
from scripts.report_phi4_hopping_qcr import closure_metrics


def test_continuation_and_objective():
    torch.set_num_threads(1);torch.manual_seed(91)
    cfg=RunConfig.load('configs/ising2d_B0_cnn_b3_g1_short_seed91.json')
    model=build_model(cfg.model,2);model.detach_kernel_representation=True
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1
    d=online_targets(model,x)
    assert not d.requires_grad
    old=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    update(model,d,old)
    payload=dict(epoch=1,optimizers={'R':old.state_dict()})
    frozen=copy.deepcopy(model);opt=freeze_and_continue(frozen,payload)
    before={k:v.clone() for k,v in frozen.state_dict().items() if not k.startswith('theory_rg.')}
    update(model,d,old);update(frozen,d,opt)
    for a,b in zip(model.theory_rg.parameters(),frozen.theory_rg.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for k,v in before.items():torch.testing.assert_close(v,frozen.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for m in [frozen.theory_encoder,frozen.rg_kernel] for p in m.parameters())
    assert all(int(s['step'])==2 for s in opt.state.values())
    a=frozen.advance_theory(d[:,0]);b=frozen.advance_theory(d[:,1]);c=frozen.advance_theory(a)
    expected=5*(a-d[:,1]).square().mean()+2.5*((b-d[:,2]).square().mean()+(c-d[:,2]).square().mean())
    torch.testing.assert_close(r_loss(frozen,d),expected)
    direct=d.clone().requires_grad_();r_loss(frozen,direct).backward();assert direct.grad is None


def test_json_only_report(tmp_path):
    from scripts.report_ising_cnn_r_only import render
    torch.set_num_threads(1)
    cfg=RunConfig.load('configs/ising2d_B0_cnn_b3_g1_short_seed91.json');model=build_model(cfg.model,2)
    rng=np.random.default_rng(1);g=rng.normal(size=(192,3,1))
    a=dict(direct_g=g,labels=np.repeat(np.linspace(.2,.68,64),3)[:,None],splits=np.array(['train']*129+['test']*63),
           flow_grid=np.linspace(-3,3,401)[:,None],all_start_pairs=np.array([[0,1],[0,2],[1,2]]))
    a=paired_predictions(model,a);a['source_flow_next']=a['flow_next'].copy()
    for name,value in [('manifest',dict(recipe=RECIPE)),('summary',dict(actual_extra_R_updates=500,reason='update_cap',
        frozen_before_sha256={'Q':'a','C':'b'},frozen_after_sha256={'Q':'a','C':'b'},additional_Q_updates=0,additional_C_updates=0,
        before_closure=closure_metrics(a),after_closure=closure_metrics(a))),
        ('history',[dict(extra_R_updates=0,train={'total':.1}),dict(extra_R_updates=500,train={'total':.01})]),('roots',dict(roots=[]))]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    assert render(tmp_path)['Q_C_unchanged']
    assert len(list((tmp_path/'report').glob('*.png')))==3
