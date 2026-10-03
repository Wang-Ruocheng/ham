import copy
import pytest
import torch
from experiments.potts_soft_freeze import qc_multiplier,split_adam,assign_gradients


def test_cosine_schedule():
    assert qc_multiplier(400)==1
    assert qc_multiplier(800)==pytest.approx(.5)
    assert qc_multiplier(1200)==0
    assert qc_multiplier(2400)==0
    with pytest.raises(ValueError):qc_multiplier(1,2,2)


def test_gradient_routing():
    for gamma in (1.,.5,0.):
        q=torch.tensor(2.,requires_grad=True);r=torch.tensor(3.,requires_grad=True)
        lr=(r*q-1).square();lq=q.square()
        assign_gradients(lq+lr,lr,[q],[r],gamma)
        assert r.grad.item()==20.
        if gamma:assert q.grad.item()==pytest.approx(4+30*gamma)
        else:assert q.grad is None


def test_split_preserves_adam_update():
    a=torch.tensor(2.,requires_grad=True);b=torch.tensor(3.,requires_grad=True)
    o=torch.optim.Adam([a,b],lr=.001)
    for _ in range(3):
        o.zero_grad();(a*a+b*b).backward();o.step()
    c=a.detach().clone().requires_grad_();d=b.detach().clone().requires_grad_()
    s=split_adam({'optimizer':o.state_dict()},1)
    other=torch.optim.Adam([dict(params=[c]),dict(params=[d])],lr=.001);other.load_state_dict(s)
    o.zero_grad();other.zero_grad();(a*a+b*b).backward();(c*c+d*d).backward();o.step();other.step()
    assert torch.equal(a,c) and torch.equal(b,d)
