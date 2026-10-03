import torch
import pytest
from recurrent_rg.losses import noise_aware_precision, metric_error, local_tangent_loss


def test_precision_detached_bounded_and_rotation_equivariant():
    torch.manual_seed(2)
    a=torch.randn(64,2,dtype=torch.double)*torch.tensor([1.,.1])
    a.requires_grad_()
    b=a+.01*torch.randn_like(a)
    w=noise_aware_precision(a,b)
    eig=torch.linalg.eigvalsh(w)
    assert not w.requires_grad
    assert torch.allclose(w.trace(),torch.tensor(2.,dtype=torch.double))
    assert eig[-1]/eig[0]<=10.0001
    u=torch.tensor([[.6,-.8],[.8,.6]],dtype=torch.double)
    assert torch.allclose(noise_aware_precision(a@u,b@u),u.T@w@u,atol=1e-10)


def test_zero_variance_safe():
    a=torch.zeros(8,2,dtype=torch.double)
    assert torch.equal(noise_aware_precision(a,a),torch.eye(2,dtype=torch.double))


def test_tangent_exact_and_no_resolved_pairs():
    x=torch.randn(12,2,dtype=torch.double,requires_grad=True)
    p=torch.softmax(torch.randn(12,16,dtype=torch.double),-1)
    exact=local_tangent_loss(x,2*x,2*x,p,p)
    assert exact.item()==0
    loss=local_tangent_loss(x,x,2*x,p,p)
    assert loss.item()>0
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert local_tangent_loss(x,x,2*x,p*0,p*0).item()==0


def test_metric_identity_matches_euclidean():
    x=torch.randn(12,2,dtype=torch.double)
    assert torch.allclose(metric_error(x,x*0,torch.eye(2,dtype=torch.double)),x.square().sum(-1).mean())


@pytest.mark.parametrize('options',[{'tangent_weight':5.},{'closure_metric':'noise_aware'}])
def test_joint_gradients_and_total(options):
    from recurrent_rg.config import ModelConfig, QCRWeights
    from recurrent_rg.checkpoints import build_model
    from recurrent_rg.data.statistics import two_spin_distribution
    from recurrent_rg.losses import qcr_objective
    torch.manual_seed(3)
    model=build_model(ModelConfig(dimension=1,theory_dim=2,hidden=6),2)
    x=(2*torch.randint(2,(8,12,16,1))-1).double()
    weights=QCRWeights(anchor=2.,separation=10.)
    loss,terms=qcr_objective(model,x,pattern_fn=two_spin_distribution,weights=weights,**options)
    expected=sum(getattr(weights,k)*terms[k] for k in ['anchor','coarse_consistency','kernel_noncollapse','separation','commutator'])+5*terms['rollout']+options.get('tangent_weight',0)*terms.get('tangent',0)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    for module in [model.theory_encoder,model.rg_kernel,model.theory_rg]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
