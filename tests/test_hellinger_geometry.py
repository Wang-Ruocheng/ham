import pytest
import torch
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.config import LossConfig

def test_hellinger_formula_noise_and_detach():
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.],[.2,.3,.5]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[.8,.2,0.],[0.,.9,.1],[.3,.2,.5]],dtype=torch.double,requires_grad=True)
    g=torch.tensor([[0.],[.2],[.8]],dtype=torch.double,requires_grad=True)
    loss,info=distribution_geometry_loss(g,a,b,distance_metric='hellinger',scale=1)
    m=(a.detach()+b.detach())/2;i,j=torch.triu_indices(3,3,1)
    d=((m[i].sqrt()-m[j].sqrt())**2).sum(-1).sqrt()/2**.5
    n=.125*(a.detach().sqrt()-b.detach().sqrt()).square().sum(-1)
    allowance=2*(n[i]+n[j]).sqrt();dist=torch.pdist(g)
    expected=((d-allowance-dist).relu().square()+(dist-d-allowance).relu().square()).mean()
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(info['noise_allowance'],allowance.mean())
    loss.backward();assert a.grad is None and b.grad is None and torch.isfinite(g.grad).all()

def test_disjoint_support_and_unused_categories():
    p=torch.eye(2,dtype=torch.double);g=torch.tensor([[0.],[.4]],dtype=torch.double)
    loss,info=distribution_geometry_loss(g,p,p,scale=1,distance_metric='hellinger')
    torch.testing.assert_close(info['target_distance'],torch.tensor(1.,dtype=torch.double))
    padded=torch.nn.functional.pad(p,(0,79))
    other,_=distribution_geometry_loss(g,padded,padded,scale=1,distance_metric='hellinger')
    torch.testing.assert_close(loss,other)

def test_config_and_bad_histogram():
    assert LossConfig(geometry_distance='hellinger').objective_kwargs()['geometry_distance']=='hellinger'
    with pytest.raises(ValueError):LossConfig(geometry_distance='bad').objective_kwargs()
    with pytest.raises(ValueError):
        distribution_geometry_loss(torch.zeros(2,1),torch.ones(2,3),torch.ones(2,3),distance_metric='hellinger')
