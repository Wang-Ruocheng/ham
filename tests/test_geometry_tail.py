import pytest
import torch
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.config import LossConfig


def test_tail_formula_and_gradients():
    p=torch.tensor([[0.],[1.],[2.],[3.]],dtype=torch.double,requires_grad=True)
    g=torch.tensor([[0.],[1.],[2.],[0.1]],dtype=torch.double,requires_grad=True)
    base,_=distribution_geometry_loss(g,p,p,mode='relative_local',scale=1)
    mixed,extra=distribution_geometry_loss(g,p,p,mode='relative_local',scale=1,tail_mix=.5,tail_fraction=.1)
    i,j=torch.triu_indices(4,4,1);d=(p[i]-p[j]).norm(dim=-1).detach()
    score=(torch.pdist(g)-d).square()/(d.square()+.05**2)
    torch.testing.assert_close(extra['geometry_tail'],score.max())
    torch.testing.assert_close(mixed,.5*base+.5*score.max())
    mixed.backward();assert p.grad is None and torch.isfinite(g.grad).all() and g.grad.abs().sum()>0


def test_no_reliable_pairs_falls_back_and_default_unchanged():
    g=torch.tensor([[0.],[1.],[2.]],dtype=torch.double,requires_grad=True)
    p=torch.ones(3,4,dtype=torch.double)/4
    base,_=distribution_geometry_loss(g,p,p,mode='relative_local')
    explicit,_=distribution_geometry_loss(g,p,p,mode='relative_local',tail_mix=0.)
    mixed,info=distribution_geometry_loss(g,p,p,mode='relative_local',tail_mix=.5)
    torch.testing.assert_close(base,explicit,rtol=0,atol=0)
    torch.testing.assert_close(base,mixed,rtol=0,atol=0)
    assert info['tail_eligible_fraction']==0
    mixed.backward();assert torch.isfinite(g.grad).all()


@pytest.mark.parametrize('options',[dict(geometry_tail_mix=-1),dict(geometry_tail_fraction=0),dict(geometry_tail_min_reliability=2),dict(geometry_tail_mix=.5)])
def test_invalid_config(options):
    with pytest.raises(ValueError):LossConfig(**options).objective_kwargs()


def test_additive_tail_preserves_base_and_detaches_targets():
    p=torch.tensor([[0.],[1.],[2.],[3.]],dtype=torch.double,requires_grad=True)
    g=torch.tensor([[0.],[1.],[2.],[.1]],dtype=torch.double,requires_grad=True)
    base,_=distribution_geometry_loss(g,p,p,mode='relative_local',scale=1)
    loss,info=distribution_geometry_loss(g,p,p,mode='relative_local',scale=1,tail_add_weight=.8)
    torch.testing.assert_close(loss,base+.8*info['geometry_tail'])
    loss.backward()
    assert p.grad is None and torch.isfinite(g.grad).all()
    flat=torch.ones_like(p)
    base,_=distribution_geometry_loss(g,flat,flat,mode='relative_local')
    added,info=distribution_geometry_loss(g,flat,flat,mode='relative_local',tail_add_weight=.8)
    torch.testing.assert_close(added,base,rtol=0,atol=0)
    assert info['geometry_tail']==0


@pytest.mark.parametrize('weight',[-1,float('nan'),float('inf')])
def test_invalid_additive_weight(weight):
    with pytest.raises(ValueError):
        LossConfig(geometry_mode='relative_local',geometry_tail_add_weight=weight).objective_kwargs()


def test_tail_modes_are_exclusive():
    with pytest.raises(ValueError):
        LossConfig(geometry_mode='relative_local',geometry_tail_mix=.8,geometry_tail_add_weight=.8).objective_kwargs()
