import torch
from recurrent_rg.losses import binary_kernel_variance_floor
from recurrent_rg.rsmi import best_binary_pair_kernel,initialize_kernel_from_table,kernel_probabilities
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from experiments.rsmi_separate_qcr import independent_component_gradients


def test_per_ensemble_floor_detects_cancellation_and_routes_gradient():
    source=torch.tensor([[-1.,1.],[-1.,1.]],dtype=torch.double).reshape(2,2,1,1).requires_grad_()
    target=torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.double).reshape(2,2,1,1).requires_grad_()
    assert binary_kernel_variance_floor([source,target])==0
    loss=binary_kernel_variance_floor([source,target],per_ensemble=True)
    assert loss>0
    loss.backward()
    assert source.grad is None
    assert target.grad.abs().sum()>0


def test_full_variance_floor_distinguishes_and_from_decimation():
    # All four input blocks occur equally often. This is a loss unit test,
    # not a table used to initialize or supervise the learned kernel.
    blocks=torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],dtype=torch.double)
    source=blocks.reshape(1,4,2,1)
    and_output=torch.tensor([-1.,-1.,-1.,1.],dtype=torch.double).reshape(1,4,1,1)
    decimation=blocks[:,0].reshape(1,4,1,1)
    assert binary_kernel_variance_floor([source,and_output],.25,per_ensemble=True)==0
    torch.testing.assert_close(binary_kernel_variance_floor([source,and_output],1.,per_ensemble=True),torch.tensor(.25,dtype=torch.double))
    assert binary_kernel_variance_floor([source,decimation],1.,per_ensemble=True)==0
    # A genuinely constant input is allowed to remain constant, not forced
    # to generate noise merely to satisfy a globally balanced output prior.
    assert binary_kernel_variance_floor([torch.ones_like(source),torch.ones_like(decimation)],1.,per_ensemble=True)==0


def test_data_selected_rule_and_trainable_initialization():
    torch.set_num_threads(1)
    # E carries only the first visible spin; second visible spin is independent.
    joint=torch.zeros(1,4,4,dtype=torch.double)
    for v in range(4):joint[0,v,v%2]=.25
    table,report=best_binary_pair_kernel(joint)
    assert report['selected_code']==10
    torch.manual_seed(7)
    m=build_model(ModelConfig(hidden=8),2)
    saved={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('rg_kernel.')}
    fit=initialize_kernel_from_table(m.rg_kernel,table,steps=400)
    assert fit['max_probability_error']<.01
    assert all(p.requires_grad for p in m.rg_kernel.parameters())
    for k,v in saved.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)


def test_independent_c_excludes_representation_and_closure_gradients():
    q=torch.tensor(2.,requires_grad=True);c=torch.tensor(3.,requires_grad=True);r=torch.tensor(4.,requires_grad=True)
    representation=(q*c).square();closure=(r*q-c).square();c_loss=(c-1).square()
    independent_component_gradients(representation+closure+c_loss,closure,c_loss,[q],[c],[r])
    torch.testing.assert_close(q.grad,torch.tensor(36.))
    torch.testing.assert_close(c.grad,torch.tensor(4.))
    torch.testing.assert_close(r.grad,torch.tensor(20.))
