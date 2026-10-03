import pytest
import torch
from recurrent_rg.config import ModelConfig, LossConfig
from recurrent_rg.checkpoints import build_model
from recurrent_rg.losses import binary_kernel_sign_loss, qcr_objective
from recurrent_rg.data.statistics import two_spin_distribution


@pytest.mark.parametrize('dimension', [1, 2])
def test_exact_probability_no_rng_and_gradient(dimension):
    model = build_model(ModelConfig(dimension=dimension, hidden=6, kernel_initialization='positive_decimation'), 2)
    c = model.rg_kernel
    before = torch.get_rng_state()
    loss = binary_kernel_sign_loss(c)
    a = torch.tanh(torch.tanh(torch.tensor(1.,dtype=torch.double)))
    torch.testing.assert_close(loss, -torch.log(torch.sigmoid(2*a)))
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():c.field_head.projection.weight.neg_()
    flipped = binary_kernel_sign_loss(c)
    assert flipped > loss
    flipped.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
    assert c.field_head.projection.weight.grad.abs().sum() > 0


def test_total_accounting_and_joint_gradients():
    model=build_model(ModelConfig(hidden=6,theory_dim=2),2)
    x=(2*torch.randint(2,(8,12,16,1))-1).double()
    rng=torch.get_rng_state()
    base,_=qcr_objective(model,x,pattern_fn=two_spin_distribution)
    torch.set_rng_state(rng)
    new,terms=qcr_objective(model,x,pattern_fn=two_spin_distribution,kernel_sign_weight=.1)
    torch.testing.assert_close(new,base+.1*terms['kernel_sign'])
    new.backward()
    for module in (model.theory_encoder,model.rg_kernel,model.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_default_and_validation():
    assert LossConfig().kernel_sign_weight==0
    for weight in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):LossConfig(kernel_sign_weight=weight).objective_kwargs()
    c=build_model(ModelConfig(kernel_sampling='deterministic'),2).rg_kernel
    with pytest.raises(ValueError):binary_kernel_sign_loss(c)
