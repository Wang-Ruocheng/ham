"""R-only 物理 score 损失的梯度与坐标约定回归。"""
from copy import deepcopy
import inspect

import pytest
import torch

from experiments.ising_physical_response_ablation import (
    FrozenResidualR, evaluate, response_terms, standardized_response,
)


def model(dim):
    torch.manual_seed(81)
    network=torch.nn.Sequential(torch.nn.Linear(dim,7),torch.nn.Tanh(),
        torch.nn.Linear(7,7),torch.nn.Tanh(),torch.nn.Linear(7,dim)).double()
    state={f'theory_rg.{key}':value for key,value in network.state_dict().items()}
    state['theory_encoder.example']=torch.ones(2,dtype=torch.float64)
    state['rg_kernel.example']=torch.ones(2,dtype=torch.float64)
    return FrozenResidualR(state)


@pytest.mark.parametrize('dim',[1,3])
def test_full_residual_jvp_matches_autograd_and_finite_difference(dim):
    net=model(dim)
    x=torch.randn(5,dim,dtype=torch.float64)
    directions=torch.randn(5,dim,6,dtype=torch.float64)
    prediction=net.response_jvp(x,directions)
    for j in range(6):
        _, expected=torch.autograd.functional.jvp(net.advance_theory,x,directions[...,j])
        torch.testing.assert_close(prediction[...,j],expected,rtol=1e-11,atol=1e-11)
        h=1e-5
        finite=(net.advance_theory(x+h*directions[...,j])-net.advance_theory(x-h*directions[...,j]))/(2*h)
        torch.testing.assert_close(prediction[...,j],finite,rtol=1e-7,atol=1e-8)


def test_jvp_gradient_of_parameters_matches_autograd_second_derivative():
    first=model(3); second=deepcopy(first)
    x=torch.randn(4,3,dtype=torch.float64)
    v=torch.randn(4,3,2,dtype=torch.float64)
    first.response_jvp(x,v).square().sum().backward()
    expected=[]
    for j in range(2):
        _, value=torch.autograd.functional.jvp(second.advance_theory,x,v[...,j],create_graph=True)
        expected.append(value)
    torch.stack(expected,-1).square().sum().backward()
    for a,b in zip(first.parameters(),second.parameters()):
        # The final bias has zero derivative in JVP but participates in closure.
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else: torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-10)


def test_zero_residual_weights_still_include_identity():
    net=model(3)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    x=torch.randn(3,3,dtype=torch.float64)
    v=torch.randn(3,3,6,dtype=torch.float64)
    torch.testing.assert_close(net.response_jvp(x,v),v)


def test_response_uses_sum_over_coordinates_and_two_scales():
    net=model(3)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    direct=torch.zeros(4,5,3,3,dtype=torch.float64)
    vin=torch.ones(4,5,2,3,6,dtype=torch.float64)
    target=torch.zeros_like(vin)
    std=torch.ones(4,5,2,6,dtype=torch.float64)
    result=response_terms(net,direct,vin,target,std,torch.ones(2,dtype=torch.float64))
    torch.testing.assert_close(result['response_mse_by_scale'],torch.full((2,),3.,dtype=torch.float64))
    assert float(result['response_normalized'])==3.


def test_metrics_and_invalid_score_zero_do_not_mutate_weights():
    net=model(1); state=deepcopy(net.state_dict())
    direct=torch.randn(4,5,3,1,dtype=torch.float64)
    vin=torch.randn(4,5,2,1,6,dtype=torch.float64)
    target=torch.randn_like(vin)
    std=torch.ones(4,5,2,6,dtype=torch.float64);std[...,0]=0
    x,y=standardized_response(vin,target,std)
    assert torch.equal(x[...,0],torch.zeros_like(x[...,0]))
    assert torch.equal(y[...,0],torch.zeros_like(y[...,0]))
    result=evaluate(net,direct,vin,target,std,torch.ones(2,dtype=torch.float64),.5)
    assert result['total']==pytest.approx(result['closure']+.5*result['response_normalized'])
    for key,value in net.state_dict().items():torch.testing.assert_close(value,state[key])
    assert all(p.grad is None for p in net.parameters())


def test_only_R_parameters_are_created_and_no_coupling_path_targets():
    net=model(3)
    assert all(name.startswith('theory_rg.') for name,_ in net.named_parameters())
    text=inspect.getsource(response_terms)
    assert 'pinv' not in text and 'Kc' not in text and 'labels' not in text
