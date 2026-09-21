import torch
from archive.qcr_ablations.balanced_geometry import covariance,balanced_metric,quadratic,objective
from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import patterns_for


def test_metric_detached_bounded_and_rotation_invariant():
    c=torch.tensor([[2.,0.],[0.,1e-9]],dtype=torch.float64,requires_grad=True)
    m=balanced_metric(c,1.)
    assert not m.requires_grad
    torch.testing.assert_close(m,torch.diag(torch.tensor([.5,10.],dtype=torch.float64)))
    u=torch.tensor([[.6,-.8],[.8,.6]],dtype=torch.float64)
    e=torch.randn(12,2,dtype=torch.float64)
    torch.testing.assert_close(quadratic(e,m),quadratic(e@u.T,balanced_metric(u@c@u.T,1.)))


def test_objective_finite_gradients_and_no_kernel_parameters():
    model=build_model(False);p=patterns_for(model.theory_encoder)
    bits=(p>0).long();marginal=torch.nn.functional.one_hot(8*bits[:,4]+4*bits[:,7]+2*bits[:,5]+bits[:,8],16).double()
    h=torch.rand(8,4,512,dtype=torch.float64);h=h/h.sum(-1,keepdim=True)
    loss,terms,cov=objective(model,h,h,p,marginal,torch.eye(2,dtype=torch.float64),1.)
    assert terms['same']==0 and not cov.requires_grad
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for mod in (model.theory_encoder,model.theory_rg) for p in mod.parameters())
    assert not list(model.rg_kernel.parameters())
