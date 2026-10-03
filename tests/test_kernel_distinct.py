import pytest
import torch
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig, QCRWeights
from recurrent_rg.kernel_regularizers import uniform_block_probabilities, kernel_distinct_loss
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import two_spin_distribution


def test_positive_and_negative_decimation_and_constant():
    c=build_model(ModelConfig(hidden=5,kernel_initialization='positive_decimation'),1).rg_kernel
    # Positive-decimation initialization preserves sign, but its Bernoulli
    # probabilities are initially soft. Increase confidence for this endpoint.
    with torch.no_grad():c.field_head.projection.weight.mul_(10)
    assert kernel_distinct_loss(c).item()==0
    with torch.no_grad():c.field_head.projection.weight.neg_()
    assert kernel_distinct_loss(c).item()==0
    with torch.no_grad():c.field_head.projection.weight.zero_()
    assert kernel_distinct_loss(c).item()==pytest.approx(.64)


@pytest.mark.parametrize('dimension,q',[(1,2),(2,2),(2,3),(2,4)])
def test_probabilities_gradient_and_output_permutation(dimension,q):
    torch.manual_seed(2)
    c=build_model(ModelConfig(dimension=dimension,hidden=5,categorical_states=q if q>2 else 0,
        kernel_sampling='categorical' if q>2 else 'bernoulli'),1).rg_kernel
    rng=torch.get_rng_state()
    probabilities=uniform_block_probabilities(c)
    assert probabilities.shape==(q,q)
    torch.testing.assert_close(probabilities.sum(-1),torch.ones(q,dtype=torch.float64))
    a=kernel_distinct_loss(c)
    if q==2:torch.testing.assert_close(a,torch.relu(.8-(probabilities[0,1]-probabilities[1,1]).abs()).square())
    a.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in c.parameters())
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():
        head=c.field_head.projection
        if q==2:head.weight.neg_();head.bias.neg_()
        else:head.weight.copy_(head.weight.roll(1,0));head.bias.copy_(head.bias.roll(1,0))
    torch.testing.assert_close(a,kernel_distinct_loss(c))


def test_objective_accounting_and_same_random_draws():
    torch.manual_seed(3)
    m=build_model(ModelConfig(hidden=5),2)
    x=(2*torch.randint(2,(4,8,16,1))-1).double()
    rng=torch.get_rng_state()
    base,_=qcr_objective(m,x,pattern_fn=two_spin_distribution)
    after=torch.get_rng_state()
    torch.set_rng_state(rng)
    augmented,terms=qcr_objective(m,x,pattern_fn=two_spin_distribution,kernel_distinct_alpha=1.)
    torch.testing.assert_close(augmented,base+2*terms['kernel_distinct'])
    assert torch.equal(after,torch.get_rng_state())


@pytest.mark.parametrize('margin',[-1,0,1.1,float('nan')])
def test_invalid_margin(margin):
    with pytest.raises(ValueError):kernel_distinct_loss(None,margin=margin)
