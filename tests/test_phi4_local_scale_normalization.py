import copy
import torch
from experiments import phi4_local_scale_normalization as c
from experiments.phi4_ru_cnn_nonlinear import JointNonlinearQ
from experiments.phi4_ru_qonly_decoder_balance import objective as original_objective
from recurrent_rg.phi4_cnn_qcr import ConfigurationCNNQ


def test_train_only_inverse_variance_and_initial_loss_match():
    torch.manual_seed(3)
    p=torch.randn(20,3,7,dtype=torch.double)*torch.tensor([1.,3.,6.])[None,:,None]
    d=p+.03*torch.randn_like(p)
    m=c.fit_scale_weights(p,d)
    assert torch.allclose(m['weights']*m['variance'],(m['weights']*m['variance']).mean().expand(3))
    assert torch.allclose(m['initial_raw'].mean(),m['initial_weighted'].mean())
    assert all(not v.requires_grad for v in m.values())
    assert m['weights'][0]>m['weights'][1]>m['weights'][2]


def test_original_objective_exact_and_gradients_isolated():
    torch.manual_seed(4)
    q=JointNonlinearQ(ConfigurationCNNQ(1.,theory_dim=2))
    levels=[torch.randn(2,4,s,s,1,dtype=torch.double,requires_grad=True) for s in (9,6,4)]
    targets=torch.softmax(torch.randn(2,3,4,625,dtype=torch.double),-1).requires_grad_()
    a,_=c.objective(q,levels,targets,torch.ones(3,dtype=torch.double))
    b,_=original_objective(q,levels,targets)
    assert torch.allclose(a,b,atol=1e-14,rtol=1e-14)
    opt=torch.optim.Adam(q.parameters(),lr=.001)
    before=copy.deepcopy(q.state_dict())
    for _ in range(3):c.update(q,opt,levels,targets,torch.tensor([2.,1.,.5]))
    assert all(x.grad is None for x in levels) and targets.grad is None
    assert any(not torch.equal(v,q.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in q.parameters())
    assert q.decoder.bounds()['lipschitz_upper_bound']<=2**.5+8+1e-10


def test_zero_variance_fails_closed():
    import pytest
    with pytest.raises(ValueError):c.fit_scale_weights(torch.ones(8,3,5),torch.zeros(8,3,5))


def test_report_identity_and_paired_bootstrap(tmp_path):
    import json
    import numpy as np
    from scripts.report_phi4_local_scale_normalization import scores,render
    rng=np.random.default_rng(3)
    p=rng.normal(size=(180,3,7));g=rng.normal(size=(180,3,2))
    cells=np.repeat(np.arange(45),4);splits=np.where(cells%4==1,'test','train')
    labels=np.repeat(np.array([(r,u,0) for r in range(9) for u in range(5)]),4,axis=0)
    a=dict(target=p,cell_ids=cells,splits=splits,labels=labels)
    for name in c.READOUTS:
        for key,val in dict(g=g,g_full=g,first=g,second=g,decoded=p).items():a[name+'_'+key]=val
    for name in ('original',)+c.READOUTS:
        for key,val in dict(one_step=g[:,1:],rollout=g[:,2],one_step_decoded=p[:,1:],
                rollout_decoded=p[:,2],flow_grid=rng.normal(size=(441,2)),flow_displacement=rng.normal(size=(441,2))).items():a[name+'_'+key]=val
    result=scores(a)
    assert result['arms']['B_scale_normalized']['reconstruction'][0]['test_nrmse']==0
    assert result['arms']['original']['closure'][0]['test_decoded_statistics_nrmse']==0
    assert result['paired_Q_chain_bootstrap']['0']['B_minus_A_nrmse_interval']==[0,0,0]
    np.savez_compressed(tmp_path/'evaluation_arrays.npz',**a)
    arms={name:dict(Q_steps=0 if name=='source' else 600,R_steps=1000,R_initial_sha256='same',Q_initial_sha256='same') for name in c.READOUTS}
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='complete',source_unchanged=True,C_updates=0,arms=arms)))
    (tmp_path/'scale_metric.json').write_text(json.dumps(dict(weights=[2,1,.5])))
    for name in c.ARMS:(tmp_path/(name+'_Q_history.json')).write_text(json.dumps([dict(step=1,reconstruction_raw=[1,2,3])]))
    for name in c.READOUTS:(tmp_path/(name+'_R_history.json')).write_text(json.dumps([dict(step=1,loss=1)]))
    render(tmp_path)
    assert len(list((tmp_path/'report').glob('*.png')))==4
