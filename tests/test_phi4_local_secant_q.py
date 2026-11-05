import copy
import numpy as np
import pytest
import torch
from experiments import phi4_local_secant_q as c
from experiments.phi4_local_scale_normalization import objective as original
from experiments.phi4_ru_cnn_nonlinear import JointNonlinearQ
from recurrent_rg.phi4_cnn_qcr import ConfigurationCNNQ


def data():
    rng=torch.Generator().manual_seed(43)
    means=torch.randn(9,3,12,generator=rng,dtype=torch.double)
    p=means.repeat_interleave(4,0)+.001*torch.randn(36,3,12,generator=rng,dtype=torch.double)
    return p,torch.arange(9).repeat_interleave(4)


def test_graph_train_only_detached_and_no_self_edges():
    p,g=data();p.requires_grad_();graph=c.fit_graph(p,g)
    assert graph['axes'].shape==(3,9,12,6)
    assert all(not v.requires_grad for v in graph.values())
    assert (graph['gains']>=0).all() and (graph['gains']<=10).all()
    assert not (graph['neighbors']==torch.arange(9)[None,:,None]).any()
    perm=torch.randperm(36);other=c.fit_graph(p[perm],g[perm])
    assert torch.allclose(graph['gains'],other['gains'],atol=1e-9)
    assert torch.equal(graph['neighbors'],other['neighbors'])


def test_zero_signal_does_not_force_rank():
    p,g=data();graph=c.fit_graph(torch.ones_like(p),g)
    assert (graph['gains']==0).all()
    assert c.initial_local_loss(graph['means'],graph)==0
    with pytest.raises(ValueError):c.fit_graph(p[:12],g[:12])


def test_local_loss_exact_translation_invariant_and_gradients():
    d=torch.randn(3,2,7,dtype=torch.double,requires_grad=True)
    target=(d[:,0]-d[:,1]).detach().requires_grad_()
    axes=torch.eye(7,dtype=torch.double).repeat(3,1,1).requires_grad_();gains=torch.ones(3,7,requires_grad=True)
    assert c.local_loss(d,target,axes,gains)==0
    value=c.local_loss(d,target+1,axes,gains);value.backward()
    assert d.grad is not None and target.grad is None and axes.grad is None and gains.grad is None
    assert torch.allclose(value,c.local_loss(d+10,target+1,axes,gains))


def test_original_equivalence_and_gradient_isolation():
    torch.manual_seed(9);q=JointNonlinearQ(ConfigurationCNNQ(1.,theory_dim=2))
    levels=[torch.randn(2,4,s,s,1,dtype=torch.double,requires_grad=True) for s in (9,6,4)]
    targets=torch.softmax(torch.randn(2,3,4,625,dtype=torch.double),-1).requires_grad_()
    pair=torch.randn(4,4,4,4,1,dtype=torch.double,requires_grad=True)
    delta=torch.randn(2,625,dtype=torch.double,requires_grad=True);axes=torch.randn(2,625,3,dtype=torch.double,requires_grad=True)
    gains=torch.ones(2,3,dtype=torch.double,requires_grad=True)
    a,_=c.objective(q,levels,targets,pair,delta,axes,gains,0.)
    b,_=original(q,levels,targets,torch.ones(3,dtype=torch.double))
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a,tuple(q.parameters()),retain_graph=True);gb=torch.autograd.grad(b,tuple(q.parameters()))
    assert all(torch.allclose(x,y) for x,y in zip(ga,gb))
    opt=torch.optim.Adam(q.parameters(),lr=.001);before=copy.deepcopy(q.state_dict())
    c.update(q,opt,levels,targets,pair,delta,axes,gains,.001)
    assert all(x.grad is None for x in levels+[targets,pair,delta,axes,gains])
    assert any(not torch.equal(v,q.state_dict()[k]) for k,v in before.items())
    assert all(x.grad is None for x in q.parameters())


def test_gate_rejects_no_evidence_and_scale_harm():
    a=dict(mse=[1,1,1],weak_component_count=10,weak_relative_mse=1.)
    b=dict(mse=[.9,.9,.9],weak_component_count=10,weak_relative_mse=.8)
    assert c.gate(a,b,[1,1,1])['passed']
    assert not c.gate(a,dict(b,weak_component_count=0),[1,1,1])['passed']
    assert not c.gate(a,dict(b,mse=[1.3,.1,.1]),[1,1,1])['passed']
    assert not c.gate(a,dict(b,weak_relative_mse=.95),[1,1,1])['passed']


def test_validation_exact_reconstruction():
    p,g=data();graph=c.fit_graph(p,g)
    m=c.validation_metrics(p,p,g,graph)
    assert m['mse']==[0,0,0] and m['weak_relative_mse']==0


@pytest.mark.parametrize('r_passed',[False,True])
def test_report_end_to_end(tmp_path,r_passed):
    import json
    from scripts.report_phi4_local_secant_q import render
    rng=np.random.default_rng(1);p=rng.normal(size=(180,3,8));g=rng.normal(size=(180,3,2))
    cells=np.repeat(np.arange(45),4);splits=np.where(cells%4==0,'test','train');inner=splits.astype('<U10');inner[(cells%7==0)&(splits=='train')]='validation'
    a=dict(target=p,cell_ids=cells,splits=splits,inner_splits=inner,
        labels=np.repeat(np.array([(r,u,0) for r in np.arange(9)*.15 for u in np.arange(5)*.25]),4,axis=0))
    for name in ('source','fresh_initial')+c.ARMS:
        for key,value in dict(g=g,first=g,second=g,g_full=g,decoded=p).items():a[name+'_'+key]=value
    for name in (('original',)+c.ARMS if r_passed else ('original',)):
        for key,value in dict(one_step_decoded=p[:,1:],rollout_decoded=p[:,2],flow_grid=rng.normal(size=(441,2)),flow_displacement=rng.normal(size=(441,2))).items():a[name+'_'+key]=value
    np.savez_compressed(tmp_path/'evaluation_arrays.npz',**a)
    arms={n:dict(Q_steps=600,R_steps=1000 if r_passed else 0,Q_initial_sha256='same',R_initial_sha256='same') for n in c.ARMS}
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='complete',source_unchanged=True,C_updates=0,arms=arms,R_gate_passed=r_passed)))
    (tmp_path/'gate.json').write_text('{}')
    for n in c.ARMS:
        (tmp_path/(n+'_Q_history.json')).write_text(json.dumps([dict(step=1,reconstruction_raw=[1,1,1],base_total=1,q_total=1,local=.1)]))
        if r_passed:(tmp_path/(n+'_R_history.json')).write_text('[{"step":1,"loss":1}]')
    result=render(tmp_path)
    assert result['arms']['warm_local']['test']['nrmse']==[0,0,0]
    assert len(list((tmp_path/'report').glob('*.png')))==(5 if r_passed else 4)
