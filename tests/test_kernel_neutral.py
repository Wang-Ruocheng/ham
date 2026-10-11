import pytest
import torch
from torch import nn
from recurrent_rg.adapters import BinaryFieldHead, CategoricalFieldHead
from recurrent_rg.model_nd import LocalCoarseGrainer
from recurrent_rg.geometry import LatticeGeometry
from recurrent_rg.kernel_regularizers import kernel_neutral_loss, uniform_reference_marginal
from recurrent_rg.config import ModelConfig, LossConfig, RunConfig, DataConfig
from recurrent_rg.checkpoints import build_model
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import two_spin_distribution, categorical_plaquette_distribution


@pytest.mark.parametrize('q,dimension,block',[(2,1,2),(2,2,2),(3,2,2),(4,1,3),(5,1,2),(2,3,2)])
def test_reference_matches_direct_enumeration_and_gradient(q,dimension,block):
    torch.manual_seed(4)
    head=BinaryFieldHead(5) if q==2 else CategoricalFieldHead(5,q)
    if q==2:head.sampling_mode='bernoulli'
    c=LocalCoarseGrainer(LatticeGeometry(dimension,3,block),1 if q==2 else q,5,head).double()
    sites=block**dimension
    states=(torch.arange(q**sites)[:,None]//(q**torch.arange(sites)))%q
    inputs=(2*states-1).double() if q==2 else torch.nn.functional.one_hot(states,q).double().flatten(1)
    logits=c.field_head.logits(c.local_encoder(inputs))
    probabilities=(torch.cat([1-torch.sigmoid(2*logits),torch.sigmoid(2*logits)],-1)
                   if q==2 else torch.softmax(logits,-1))
    rng=torch.get_rng_state()
    actual=uniform_reference_marginal(c,chunk_size=7)
    torch.testing.assert_close(actual,probabilities.mean(0))
    assert torch.equal(rng,torch.get_rng_state())
    loss=kernel_neutral_loss(c)
    torch.testing.assert_close(loss,.5*(probabilities.mean(0)-1/q).square().sum())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
    assert c.field_head.projection.bias.grad.abs().sum()>0


def test_binary_zero_uniform_random_and_positive_negative_decimation():
    c=build_model(ModelConfig(hidden=5,kernel_initialization='positive_decimation'),2).rg_kernel
    assert float(kernel_neutral_loss(c))<1e-25
    with torch.no_grad():c.field_head.projection.weight.neg_()
    assert float(kernel_neutral_loss(c))<1e-25
    with torch.no_grad():
        c.field_head.projection.weight.zero_();c.field_head.projection.bias.zero_()
    assert float(kernel_neutral_loss(c))==0
    with torch.no_grad():c.field_head.projection.bias.fill_(10)
    assert float(kernel_neutral_loss(c))==pytest.approx(.25,abs=1e-8)


def test_categorical_output_relabeling_invariant():
    c=build_model(ModelConfig(dimension=2,block_factor=2,hidden=5,categorical_states=3,kernel_sampling='categorical'),2).rg_kernel
    before=kernel_neutral_loss(c)
    with torch.no_grad():
        head=c.field_head.projection
        head.weight.copy_(head.weight[[2,0,1]].clone());head.bias.copy_(head.bias[[2,0,1]].clone())
    torch.testing.assert_close(before,kernel_neutral_loss(c))


@pytest.mark.parametrize('categorical',[False,True])
def test_total_accounting_and_rng_pairing(categorical):
    if categorical:
        model=build_model(ModelConfig(dimension=2,block_factor=2,hidden=5,categorical_states=3,kernel_sampling='categorical'),1)
        x=torch.nn.functional.one_hot(torch.randint(3,(4,8,8,8)),3).double()
        fn=categorical_plaquette_distribution
    else:
        model=build_model(ModelConfig(hidden=5),1)
        x=(2*torch.randint(2,(4,8,8,1))-1).double();fn=two_spin_distribution
    rng=torch.get_rng_state()
    base,_=qcr_objective(model,x,pattern_fn=fn,kernel_noncollapse_mode='variance_floor')
    after=torch.get_rng_state()
    torch.set_rng_state(rng)
    new,terms=qcr_objective(model,x,pattern_fn=fn,kernel_noncollapse_mode='variance_floor',kernel_neutral_weight=10.)
    assert torch.equal(after,torch.get_rng_state())
    torch.testing.assert_close(new,base+10*terms['kernel_neutral'])
    new.backward()
    for module in (model.theory_encoder,model.rg_kernel,model.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_defaults_limits_and_invalid_settings():
    assert LossConfig().kernel_neutral_weight==0
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    assert RunConfig.from_dict(cfg.to_dict()).to_dict()==cfg.to_dict()
    for weight in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):LossConfig(kernel_neutral_weight=weight).objective_kwargs()
    c=build_model(ModelConfig(),1).rg_kernel
    with pytest.raises(ValueError,match='exceeds'):uniform_reference_marginal(c,max_patterns=3)
    c.field_head.sampling_mode='deterministic'
    with pytest.raises(ValueError,match='Bernoulli'):kernel_neutral_loss(c)


def test_and_has_nonzero_neutral_penalty():
    c=LocalCoarseGrainer(LatticeGeometry(1,3,2),1,2,BinaryFieldHead(2)).double()
    c.field_head.sampling_mode='bernoulli'
    c.local_encoder=nn.Sequential(nn.Linear(2,2,bias=False)).double()
    with torch.no_grad():
        c.local_encoder[0].weight.copy_(torch.eye(2,dtype=torch.double))
        c.field_head.projection.weight.fill_(10);c.field_head.projection.bias.fill_(-10)
    assert float(kernel_neutral_loss(c))==pytest.approx(.0625,abs=1e-8)


def test_invalid_neutral_model_rejected_before_training():
    data=DataConfig(cache='unused',physical_model='test')
    for model in [ModelConfig(fixed_kernel=True,kernel_sampling='deterministic'),
                  ModelConfig(kernel_sampling='deterministic'),ModelConfig(dimension=2,block_factor=5)]:
        with pytest.raises(ValueError):RunConfig(data=data,model=model,loss=LossConfig(kernel_neutral_weight=10)).validate()
