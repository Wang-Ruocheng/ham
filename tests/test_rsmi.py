import math
import pytest
import torch
from recurrent_rg.rsmi import (enumerated_infonce, region_offsets, sample_regions,
                               kernel_probabilities, reference_probabilities)
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig


@pytest.mark.parametrize('dimension,environment_size', [(1, 2), (2, 20)])
def test_regions_separated(dimension, environment_size):
    v, e = region_offsets(dimension)
    assert len(v) == 2**dimension and len(e) == environment_size
    assert torch.cdist(v.double(), e.double(), p=float('inf')).min() == 2
    x = torch.arange(3*8*8**dimension).reshape(3,8,*([8]*dimension),1)
    a,b = sample_regions(x, 2, 4, torch.Generator().manual_seed(4))
    assert a.shape == (2,4,2**dimension) and b.shape == (2,4,environment_size)
    with pytest.raises(ValueError):
        sample_regions(torch.ones(3,8,*([4]*dimension),1),2,4,torch.Generator())


def test_exact_enumeration_matches_explicit_anchors_and_gradients():
    torch.manual_seed(2)
    p = torch.softmax(torch.randn(3,7,2,dtype=torch.double),-1).requires_grad_()
    s = torch.randn_like(p, requires_grad=True)
    explicit = sum(p[:,i,h]*(s[:,i,h]-torch.logsumexp(s[:,:,h],1)+math.log(7))
                   for i in range(7) for h in range(2))/7
    actual = enumerated_infonce(p,s)
    torch.testing.assert_close(actual,explicit)
    for a,b in zip(torch.autograd.grad(actual.sum(),(p,s),retain_graph=True),
                   torch.autograd.grad(explicit.sum(),(p,s))):
        torch.testing.assert_close(a,b)


def test_no_cross_ensemble_negatives_and_relabel_invariance():
    p = torch.tensor([[[1.,0.],[1.,0.]],[[0.,1.],[0.,1.]]])
    s = 10*p
    torch.testing.assert_close(enumerated_infonce(p,s),torch.zeros(2))
    torch.testing.assert_close(enumerated_infonce(p,s),enumerated_infonce(p.flip(-1),s.flip(-1)))


def test_random_independent_output_cannot_get_positive_empirical_bound():
    s = torch.randn(4,50,2,dtype=torch.double)
    p = torch.full_like(s,.5)
    assert (enumerated_infonce(p,s) <= 1e-12).all()


def test_perfect_binary_dependence_gives_one_bit():
    p = torch.tensor([[[1.,0.],[0.,1.],[1.,0.],[0.,1.]]],dtype=torch.double)
    torch.testing.assert_close(enumerated_infonce(p,40*p),torch.tensor([math.log(2)],dtype=torch.double))


def test_binary_kernel_gradient_without_straight_through():
    torch.manual_seed(91)
    model = build_model(ModelConfig(dimension=2,block_factor=2),2)
    v = torch.tensor([[-1.,-1.,-1.,-1.],[1.,1.,1.,1.]],dtype=torch.double)[None]
    p = kernel_probabilities(model.rg_kernel,v)
    loss = -enumerated_infonce(p,torch.tensor([[[2.,-2.],[-2.,2.]]],dtype=torch.double)).mean()
    loss.backward()
    assert sum(float(x.grad.abs().sum()) for x in model.rg_kernel.parameters()) > 0
    assert all(x.grad is None for x in model.theory_encoder.parameters())
    torch.testing.assert_close(reference_probabilities(torch.tensor([[[-1.,1.]]]),'majority'),torch.tensor([[[.5,.5]]]))


def test_exact_chain_diagnostic_matches_full_enumeration():
    import itertools
    import numpy as np
    from scripts.report_rsmi_experiment import exact_chain_mi
    s=np.array(list(itertools.product((-1.,1.),repeat=8)))
    p=np.exp(.7*(s*np.roll(s,-1,axis=1)).sum(1));p/=p.sum()
    # Deterministic AND kernel, which also tests asymmetry of the hidden state.
    table=np.array([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    ei=(((s[:,[6,3]]+1)/2) @ np.array([1,2])).astype(int)
    vi=(((s[:,:2]+1)/2) @ np.array([1,2])).astype(int)
    he=np.zeros((2,4))
    for i in range(len(p)):he[:,ei[i]]+=p[i]*table[vi[i]]
    marginal=he.sum(1,keepdims=True)*he.sum(0,keepdims=True)
    value=(he*np.log2(he/marginal)).sum()
    assert exact_chain_mi(table,.7,length=8)==pytest.approx(value,abs=1e-12)
