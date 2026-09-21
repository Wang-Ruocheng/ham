import torch
import pytest
from experiments.joint_qcr_ising2d_field import build_model
from experiments.joint_qcr_ising import qcr_objective, plaquette_distribution
from experiments.joint_qcr_fixed_cache import cached_objective
from experiments.supervised_q_ising2d_multiscale import frequencies


def test_cached_fixed_qcr_matches_all_terms_and_gradients():
    torch.manual_seed(91)
    model=build_model(False)
    x=(2*torch.randint(0,2,(3,8,16,16,1))-1).double()
    levels=model.coarse_grain(x)
    ha=torch.stack([frequencies(v[:,:4],model.geometry) for v in levels],1)
    hb=torch.stack([frequencies(v[:,4:],model.geometry) for v in levels],1)
    options=dict(objective_mode='data_geometry',geometry_scale=4.,noise_multiplier=2.,
                 rollout_weight=5.,kernel_noncollapse_mode='variance_floor',kernel_variance_ratio=.25)
    a,at=qcr_objective(model,x,pattern_fn=plaquette_distribution,**options)
    b,bt=cached_objective(model,ha,hb,**options)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    for key in at:torch.testing.assert_close(at[key],bt[key],rtol=1e-10,atol=1e-11)
    params=list(model.theory_encoder.parameters())+list(model.theory_rg.parameters())
    for u,v in zip(torch.autograd.grad(a,params),torch.autograd.grad(b,params)):
        torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-10)


def test_cached_qcr_rejects_learned_kernel():
    model=build_model(True)
    with pytest.raises(ValueError,match='fixed decimation'):
        cached_objective(model,None,None)
