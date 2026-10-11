import copy
import torch
import numpy as np
import pytest
from recurrent_rg.config import ModelConfig, LossConfig
from recurrent_rg.checkpoints import build_model
from recurrent_rg.categorical_pooling import CategoricalMean, CategoricalLocal
from recurrent_rg.losses import qcr_objective, categorical_kernel_preserve_loss, categorical_kernel_variance_floor
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.data.potts import sample_potts


@pytest.mark.parametrize('pooled',[False,True])
def test_lookup_matches_dense_values_and_gradients(pooled):
    torch.manual_seed(4)
    x=torch.nn.functional.one_hot(torch.randint(3,(3,5,4,4)),3).double().flatten(-2).requires_grad_()
    y=x.detach().clone().requires_grad_()
    a=torch.nn.Sequential(torch.nn.Linear(12,7),torch.nn.Tanh(),torch.nn.Linear(7,3)).double()
    b=copy.deepcopy(a)
    if pooled:
        out=CategoricalMean.apply(x,3,a,*a.parameters());ref=b(y).mean((1,2))
    else:
        out=CategoricalLocal.apply(x,3,a,*a.parameters());ref=b(y)
    torch.testing.assert_close(out,ref)
    upstream=torch.randn_like(out)
    (out*upstream).sum().backward();(ref*upstream).sum().backward()
    torch.testing.assert_close(x.grad,y.grad)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)


def test_categorical_joint_gradients_and_sampling():
    m=build_model(ModelConfig(dimension=2,block_factor=2,categorical_states=3,kernel_sampling='categorical',hidden=6),2)
    x=torch.nn.functional.one_hot(torch.randint(3,(6,8,16,16)),3).double()
    levels=m.coarse_grain(x)
    assert all(torch.allclose(v.sum(-1),torch.ones_like(v[...,0])) for v in levels)
    assert all(((v-v.round()).abs()<1e-10).all() for v in levels)
    pattern=categorical_plaquette_distribution(x)
    assert pattern.shape==(6,81)
    torch.testing.assert_close(pattern.sum(-1),torch.ones(6,dtype=torch.double))
    loss,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
        **LossConfig(geometry_mode='relative_local',kernel_noncollapse_mode='variance_floor',kernel_sign_weight=.1).objective_kwargs())
    loss.backward()
    for module in (m.rg_kernel,m.theory_encoder,m.theory_rg):
        assert all(p.grad is not None and p.grad.isfinite().all() for p in module.parameters())
    assert 'kernel_sign' in terms
    head=m.rg_kernel.field_head
    draws=head.sample_logits(torch.tensor([0.,1.,-1.],dtype=torch.double).expand(30000,3))
    torch.testing.assert_close(draws.mean(0),torch.softmax(torch.tensor([0.,1.,-1.],dtype=torch.double),0),atol=.012,rtol=0)


def test_color_preservation_and_collapse():
    m=build_model(ModelConfig(dimension=2,block_factor=2,categorical_states=3,kernel_sampling='categorical',hidden=6),2)
    c=m.rg_kernel
    patterns=torch.eye(3,dtype=torch.double).repeat(1,4)
    expected=-torch.log_softmax(c.field_head.logits(c.local_encoder(patterns)),-1).diag().mean()
    rng=torch.get_rng_state()
    torch.testing.assert_close(categorical_kernel_preserve_loss(c),expected)
    assert torch.equal(rng,torch.get_rng_state())
    source=torch.eye(3,dtype=torch.double).reshape(3,1,1,1,3)
    target=source[0:1].expand_as(source).clone().requires_grad_()
    loss=categorical_kernel_variance_floor([source,target]);assert loss>0
    loss.backward();assert target.grad.abs().sum()>0


def test_sampler_limits_and_reproducibility():
    a=sample_potts(0.,8,100,123,burn=4,spacing=1)
    assert np.array_equal(a,sample_potts(0.,8,100,123,burn=4,spacing=1))
    assert abs((a==np.roll(a,1,axis=1)).mean()-1/3)<.03
    b=sample_potts(10.,8,30,124,burn=100,spacing=2)
    assert (b==np.roll(b,1,axis=1)).mean()>.99


def test_sampler_matches_enumerated_small_lattice():
    from itertools import product
    states=np.array(list(product(range(3),repeat=4))).reshape(-1,2,2)
    # L=2 periodic directed right/down bonds count each parallel bond twice,
    # exactly as in the sampler; do not compare with an open four-edge square.
    bonds=(states==np.roll(states,1,axis=1)).sum((1,2))+(states==np.roll(states,1,axis=2)).sum((1,2))
    w=np.exp(.6*bonds);w=w/w.sum()
    expected=(bonds*w).sum()/8
    draws=sample_potts(.6,2,8000,734,burn=256,spacing=4)
    actual=((draws==np.roll(draws,1,axis=1)).mean()+(draws==np.roll(draws,1,axis=2)).mean())/2
    assert abs(actual-expected)<.015


def test_checkpoint_roundtrip_and_evaluation_rng(tmp_path):
    import json
    from recurrent_rg.config import RunConfig,DataConfig,TrainingConfig,EvaluationConfig
    from recurrent_rg.checkpoints import load_run,qcr_state_dict
    from recurrent_rg.evaluation import collect_trajectories
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='potts'),
        model=ModelConfig(dimension=2,block_factor=2,categorical_states=3,kernel_sampling='categorical',hidden=6),
        loss=LossConfig(kernel_noncollapse_mode='variance_floor'),
        training=TrainingConfig(max_rg_steps=1),evaluation=EvaluationConfig(max_rg_steps=1))
    cfg.validate();m=build_model(cfg.model,1)
    (tmp_path/'manifest.json').write_text(json.dumps({'config':cfg.to_dict()}))
    torch.save(qcr_state_dict(m),tmp_path/'qcr_model.pt')
    loaded=load_run(tmp_path)
    x=torch.nn.functional.one_hot(torch.randint(3,(3,4,8,8)),3).double()
    rng=torch.get_rng_state()
    a=collect_trajectories(loaded.model,x,samples=4)
    b=collect_trajectories(loaded.model,x,samples=4)
    assert np.array_equal(a['direct'],b['direct'])
    assert torch.equal(rng,torch.get_rng_state())
