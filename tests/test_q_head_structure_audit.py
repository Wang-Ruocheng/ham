import torch
from experiments.q_head_structure_audit import MonotoneSpline, paired_heads


def test_spline_identity_monotonicity_and_gradients():
    spline=MonotoneSpline();x=torch.linspace(-10,10,501,dtype=torch.double)[:,None]
    torch.testing.assert_close(spline(x),x)
    with torch.no_grad():spline.raw_slopes.normal_()
    y=spline(x)
    assert torch.all(y[1:]>y[:-1])
    y.square().mean().backward()
    assert torch.isfinite(spline.raw_slopes.grad).all()


def test_paired_heads_same_initial_function():
    torch.manual_seed(3);a=torch.randn(12,12,dtype=torch.double)
    W=a@a.T+torch.eye(12,dtype=torch.double);mu=torch.randn(12,dtype=torch.double)
    heads=paired_heads(mu,W,91);x=torch.randn(7,3,2,12,dtype=torch.double)
    for head in heads.values():torch.testing.assert_close(head(x),heads['linear'](x),atol=1e-10,rtol=1e-10)
