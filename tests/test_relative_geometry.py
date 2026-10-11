import pytest
import torch
from recurrent_rg.losses import distribution_geometry_loss, qcr_objective
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.checkpoints import build_model
from recurrent_rg.data.statistics import two_spin_distribution


def test_absolute_formula_unchanged():
    torch.manual_seed(17)
    g=torch.randn(6,2,dtype=torch.float64)
    p=torch.softmax(torch.randn(6,4,dtype=torch.float64),-1)
    q=torch.softmax(torch.randn(6,4,dtype=torch.float64),-1)
    loss,_=distribution_geometry_loss(g,p,q)
    m=(p+q)/2;n=((p-q)**2).sum(-1)/4
    i,j=torch.triu_indices(6,6,1)
    d=(m[i]-m[j]).norm(dim=-1);a=2*(n[i]+n[j]).sqrt()
    z=torch.pdist(g)
    expected=((4*(d-a).clamp_min(0)-z).relu()**2+(z-4*(d+a)).relu()**2).mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)


def test_relative_promotes_small_reliable_differences():
    g=torch.zeros(2,2,dtype=torch.float64)
    losses={}
    for mode in ['absolute','relative_local']:
        values=[]
        for distance in [.1,1.]:
            p=torch.tensor([[0.],[distance]],dtype=torch.float64)
            values.append(distribution_geometry_loss(g,p,p,scale=1,mode=mode)[0])
        losses[mode]=values[0]/values[1]
    assert losses['relative_local'] > 50*losses['absolute']


def test_geometry_detaches_data_and_is_rotation_invariant():
    torch.manual_seed(3)
    g=torch.randn(8,2,dtype=torch.float64,requires_grad=True)
    p=torch.randn(8,4,dtype=torch.float64,requires_grad=True)
    loss,metrics=distribution_geometry_loss(g,p,p,mode='relative_local')
    rotated=torch.stack([-g[:,1],g[:,0]],-1)+torch.tensor([3.,-4.])
    other,more=distribution_geometry_loss(rotated,p,p,mode='relative_local')
    torch.testing.assert_close(loss,other)
    torch.testing.assert_close(metrics['local_pair_fraction'],more['local_pair_fraction'])
    loss.backward()
    assert p.grad is None and torch.isfinite(g.grad).all()


def test_unresolved_and_identical_data_finite():
    g=torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    p=torch.ones(5,4,dtype=torch.float64)/4
    loss,metrics=distribution_geometry_loss(g,p,p,mode='relative_local')
    assert torch.isfinite(loss) and metrics['local_pair_fraction']==0
    loss.backward();assert torch.isfinite(g.grad).all()


def test_full_qcr_relative_objective_gradients():
    torch.manual_seed(3)
    model=build_model(ModelConfig(dimension=1,theory_dim=2,hidden=6,affine_reconstruction=False),2)
    x=(2*torch.randint(2,(8,12,16,1))-1).double()
    loss,terms=qcr_objective(model,x,pattern_fn=two_spin_distribution,geometry_mode='relative_local')
    assert 'isotropy' not in terms and 'relative_local_level_2' in terms
    loss.backward()
    for module in [model.theory_encoder,model.rg_kernel,model.theory_rg]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize('options',[{'mode':'bad'},{'relative_floor':0},{'neighbors':0},{'local_fraction':1.1}])
def test_invalid_geometry_options(options):
    with pytest.raises(ValueError):
        distribution_geometry_loss(torch.zeros(3,2),torch.zeros(3,4),torch.zeros(3,4),**options)
