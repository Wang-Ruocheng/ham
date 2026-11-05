import copy
import numpy as np
import torch
from experiments import phi4_ru_qonly_decoder_balance as c
from recurrent_rg.phi4_cnn_qcr import ConfigurationCNNQ
from recurrent_rg.losses import all_start_anchor_closure_loss


def test_paired_initialization_and_decoder_bound():
    torch.manual_seed(91);q=ConfigurationCNNQ(1.,theory_dim=2)
    arms=c.make_arms(q);x=torch.randn(2,4,9,9,1,dtype=torch.double)
    for a in arms.values():
        assert torch.equal(a(x),q(x))
        assert torch.allclose(a.decode(a(x)),q.decode(q(x)),atol=1e-14,rtol=0)
    assert all(torch.equal(v,arms['C_balanced'].state_dict()[k]) for k,v in arms['B_nonlinear'].state_dict().items())
    d=arms['B_nonlinear'].decoder
    with torch.no_grad():
        for p in d.net.parameters():p.mul_(100)
        d.normalize_()
    assert d.bounds()['lipschitz_upper_bound']<=2**.5+8+1e-10


def test_metric_no_dg_rank_cap_and_initial_scale_match():
    torch.manual_seed(1)
    means=torch.randn(30,3,7,dtype=torch.double)
    groups=torch.arange(30).repeat_interleave(4)
    p=means.repeat_interleave(4,0)+.001*torch.randn(120,3,7,dtype=torch.double)
    d=p+.1*torch.randn_like(p)
    metrics=c.fit_metrics(p,groups,d)
    for k,m in enumerate(metrics):
        assert m['axes'].shape[1]>2
        assert m['gains'].max()<=10
        assert torch.allclose(c.metric_loss(d[:,k]-p[:,k],m),c.metric_loss(d[:,k]-p[:,k]))
        assert not any(v.requires_grad for v in m.values())


def test_gradient_isolation_and_nonzero_hidden_gradient_after_warmup():
    torch.manual_seed(1);q=c.make_arms(ConfigurationCNNQ(1.,theory_dim=2))['B_nonlinear']
    levels=[torch.randn(2,4,s,s,1,dtype=torch.double,requires_grad=True) for s in (9,6,4)]
    targets=torch.softmax(torch.randn(2,3,4,625,dtype=torch.double),-1).requires_grad_()
    opt=torch.optim.Adam(q.parameters(),lr=.001)
    for _ in range(3):c.update(q,opt,levels,targets)
    assert all(x.grad is None for x in levels) and targets.grad is None
    loss,_=c.objective(q,levels,targets);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
    assert q.decoder.net[0].weight.grad.abs().sum()>0


def test_R_exact_all_start_and_stopgrad():
    torch.manual_seed(2);r=c.RReadout();g=torch.randn(5,3,2,dtype=torch.double,requires_grad=True)
    expected,_,_=all_start_anchor_closure_loss(r,g.detach(),coordinate_reduction='sum')
    actual=c.r_objective(r,g)
    assert torch.allclose(actual,10*expected)
    actual.backward();assert g.grad is None
    assert all(p.grad is not None for p in r.parameters())


def test_report_uses_common_statistic_space(tmp_path):
    import json
    from scripts.report_phi4_ru_qonly_decoder_balance import scores, render
    rng=np.random.default_rng(2)
    p=rng.normal(size=(180,3,7));g=rng.normal(size=(180,3,2))
    cells=np.repeat(np.arange(45),4)
    train=cells%4!=1
    metrics=c.fit_metrics(torch.tensor(p[train]),cells[train],torch.tensor(p[train]+.01))
    m={f'{i}_{k}':v.numpy() for i,x in enumerate(metrics) for k,v in x.items()}
    labels=np.repeat(np.array([(r,u,0) for r in range(9) for u in range(5)]),4,axis=0)
    a=dict(target=p,splits=np.where(train,'train','test'),cell_ids=cells,labels=labels)
    for arm in ('source',)+c.ARMS:
        for key,val in dict(g=g,g_full=g,first=g,second=g,decoded=p,
                            one_step=g[:,1:],rollout=g[:,2],
                            one_step_decoded=p[:,1:],rollout_decoded=p[:,2]).items():a[arm+'_'+key]=val
    result=scores(a,m)
    assert result['C_balanced']['reconstruction'][0]['test_nrmse']==0
    assert result['B_nonlinear']['R_readout'][2]['test_decoded_statistics_nrmse']==0
    for arm in c.ARMS:
        a[arm+'_flow_grid']=rng.normal(size=(441,2));a[arm+'_flow_displacement']=rng.normal(size=(441,2))
        (tmp_path/(arm+'_history.json')).write_text(json.dumps([dict(step=1,reconstruction_raw=[1.,1.,1.])]))
        (tmp_path/(arm+'_R_history.json')).write_text(json.dumps([dict(step=1,loss=1.)]))
    np.savez_compressed(tmp_path/'evaluation_arrays.npz',**a)
    np.savez_compressed(tmp_path/'metric_arrays.npz',**m)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(source='synthetic_test_only',data_warning_count=0)))
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='complete',source_unchanged=True,C_updates=0,
        arms={x:dict(Q_steps=600,R_steps=1000) for x in c.ARMS})))
    report=render(tmp_path)
    assert report['status']=='complete' and len(list((tmp_path/'report').glob('*.png')))==6
