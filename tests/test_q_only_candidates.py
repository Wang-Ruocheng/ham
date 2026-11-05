import torch
import pytest
from experiments.q_only_candidates import collision_loss,ranking_loss,manifold_loss,graph_targets
from experiments.q_only_candidates import validate_frozen_levels

def test_cache_rejects_truncated_hard_states():
    x=torch.tensor([[0.,1.-1e-16,0.]],dtype=torch.double)
    with pytest.raises(ValueError):validate_frozen_levels([x.to(torch.uint8)])
    exact=torch.nn.functional.one_hot(x.argmax(-1),3).to(torch.uint8)
    validate_frozen_levels([exact])

def test_candidate_losses_detach_targets_and_scale_invariance():
    p=torch.tensor([[1.,0.,0.],[.9,.1,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.double,requires_grad=True)
    g=torch.tensor([[0.],[.01],[.02],[2.]],dtype=torch.double,requires_grad=True)
    for fn in [collision_loss,ranking_loss]:
        value=fn(g,p,p)
        assert torch.isfinite(value)
        value.backward(retain_graph=True)
        assert p.grad is None and torch.isfinite(g.grad).all()
        torch.testing.assert_close(value,fn(3*g+7,p,p),rtol=1e-8,atol=1e-9)

def test_train_graph_disconnections_and_no_test_shortcuts():
    p=torch.tensor([[1.,0.],[.9,.1],[.1,.9],[0.,1.]],dtype=torch.double)
    test=torch.tensor([[.5,.5],[.7,.3]],dtype=torch.double)
    path,_,info=graph_targets(p,test,neighbors=1)
    assert info['components']==2 and torch.isinf(path[0,3])
    other,_,_=graph_targets(p,test[:1],neighbors=1)
    torch.testing.assert_close(path,other)
    g=torch.arange(4,dtype=torch.double)[:,None].requires_grad_()
    loss=manifold_loss(g,path);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(g.grad).all()
