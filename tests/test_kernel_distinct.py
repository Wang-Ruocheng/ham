"""Historical diagnostic fidelity and retirement from active training."""
import inspect
import pytest
import torch
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import DataConfig, LossConfig, ModelConfig, RunConfig
from recurrent_rg.kernel_regularizers import uniform_block_probabilities
from archive.qcr_ablations.kernel_distinct import kernel_distinct_loss
from recurrent_rg.losses import qcr_objective


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


def test_retired_loss_is_absent_from_active_objective():
    from recurrent_rg import kernel_regularizers
    assert not hasattr(kernel_regularizers, 'kernel_distinct_loss')
    assert 'kernel_distinct_alpha' not in inspect.signature(qcr_objective).parameters
    assert 'kernel_distinct_margin' not in inspect.signature(qcr_objective).parameters
    with pytest.raises(TypeError, match='kernel_distinct_alpha'):
        qcr_objective(None, None, pattern_fn=None, kernel_distinct_alpha=1.)


def test_zero_legacy_fields_do_not_reach_active_objective():
    kwargs = LossConfig(kernel_distinct_alpha=0., kernel_distinct_margin=.65).objective_kwargs()
    assert 'kernel_distinct_alpha' not in kwargs
    assert 'kernel_distinct_margin' not in kwargs
    assert set(kwargs) <= set(inspect.signature(qcr_objective).parameters)


def test_historical_nonzero_manifest_remains_readable_but_cannot_train(tmp_path):
    from recurrent_rg.runner import run_training
    cfg = RunConfig(data=DataConfig(cache=str(tmp_path/'missing.pt'), physical_model='ising1d'),
                    loss=LossConfig(kernel_distinct_alpha=1., kernel_distinct_margin=.8))
    historical = cfg.to_dict()
    loaded = RunConfig.from_dict(historical)
    loaded.validate()  # Report/config readers preserve the historical objective.
    assert loaded.to_dict() == historical
    with pytest.raises(ValueError, match='has been retired'):
        loaded.loss.objective_kwargs()
    with pytest.raises(ValueError, match='has been retired'):
        run_training(loaded, tmp_path/'not-created', allow_legacy_training=True)
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('settings',[
    {'kernel_distinct_alpha': -1.},
    {'kernel_distinct_alpha': float('nan')},
    {'kernel_distinct_margin': 0.},
    {'kernel_distinct_margin': 1.1},
])
def test_invalid_historical_metadata_is_not_silently_accepted(settings):
    with pytest.raises(ValueError, match='Invalid kernel distinct settings'):
        LossConfig(**settings).objective_kwargs(historical_metadata=True)


@pytest.mark.parametrize('margin',[-1,0,1.1,float('nan')])
def test_invalid_margin(margin):
    with pytest.raises(ValueError):kernel_distinct_loss(None,margin=margin)
