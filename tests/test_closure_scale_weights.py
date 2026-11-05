"""Target-scale closure ablation: normalized weights, no label dependence."""
from dataclasses import replace
import copy

import pytest
import torch

from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, RunConfig,
    TrainingConfig, validate_closure_scale_weights,
)
from recurrent_rg.losses import (
    closure_scale_mean, latent_squared_error, multistep_closure_loss,
    one_step_closure_loss, qcr_objective,
)


@pytest.mark.parametrize('weights', [
    [], (), [0, 0], [-1, 2], [1, float('nan')], [1, float('inf')],
    [True, 1], ['1', 2], '1,4', 2, {1: 4},
])
def test_invalid_scale_weights_rejected(weights):
    with pytest.raises(ValueError, match='closure_scale_weights'):
        LossConfig(closure_scale_weights=weights).objective_kwargs()


def test_configuration_roundtrip_and_depth_validation():
    config = RunConfig(
        data=DataConfig(cache='unused.pt', physical_model='ising1d'),
        model=ModelConfig(dimension=1),
        loss=LossConfig(closure_objective='one_step_and_rollout',
                        closure_scale_weights=(1., 4.)),
        training=TrainingConfig(max_rg_steps=2),
        evaluation=EvaluationConfig(max_rg_steps=2),
    )
    config.validate()
    loaded = RunConfig.from_dict(config.to_dict())
    assert loaded.loss.closure_scale_weights == (1., 4.)
    assert loaded.loss.objective_kwargs()['closure_scale_weights'] == (1., 4.)
    with pytest.raises(ValueError, match='length must match max_rg_steps'):
        replace(config, loss=replace(config.loss, closure_scale_weights=(1.,))).validate()
    assert validate_closure_scale_weights(None, 2) is None


class _LinearR:
    def __init__(self):
        self.a = torch.tensor(2., dtype=torch.double, requires_grad=True)

    def advance_theory(self, value):
        return self.a * value


def test_weighted_one_step_and_rollout_target_indexing():
    model = _LinearR()
    direct = torch.tensor([[[1.], [3.], [10.]]], dtype=torch.double)
    step, step_terms, raw = one_step_closure_loss(
        model.advance_theory(direct[:, :-1]), direct[:, 1:], 'sum', (1., 4.))
    rollout, rollout_terms = multistep_closure_loss(model, direct, 'sum', (1., 4.))
    torch.testing.assert_close(torch.stack(step_terms), torch.tensor([1., 16.], dtype=torch.double))
    torch.testing.assert_close(torch.stack(rollout_terms), torch.tensor([1., 36.], dtype=torch.double))
    assert raw.item() == 8.5
    assert step.item() == 13.
    assert rollout.item() == 29.
    rollout.backward()
    torch.testing.assert_close(model.a.grad, torch.tensor(-38.8, dtype=torch.double))


def test_zero_early_weight_keeps_late_rollout_gradient_through_early_R():
    model = _LinearR()
    direct = torch.tensor([[[1.], [3.], [10.]]], dtype=torch.double)
    rollout, _ = multistep_closure_loss(model, direct, closure_scale_weights=(0., 1.))
    assert rollout.item() == 36.
    rollout.backward()
    # d(a^2 - 10)^2/da = 4*a*(a^2 - 10), not a detached first-step derivative.
    assert model.a.grad.item() == -48.


def test_common_weight_rescaling_does_not_change_strength():
    terms = [torch.tensor(3., dtype=torch.double), torch.tensor(8., dtype=torch.double)]
    first = closure_scale_mean(terms, (1., 4.))
    assert torch.equal(first, closure_scale_mean(terms, (2., 8.)))
    assert torch.equal(first, closure_scale_mean(terms, (1e307, 4e307)))
    assert torch.equal(closure_scale_mean(terms, (1e308, 1e308)), torch.stack(terms).mean())
    with pytest.raises(ValueError, match='length must match max_rg_steps'):
        closure_scale_mean(terms, (1.,))


def _objective_fixture():
    from recurrent_rg.checkpoints import build_model
    from recurrent_rg.data.statistics import two_spin_distribution
    torch.manual_seed(113)
    model = build_model(ModelConfig(dimension=1, theory_dim=2, hidden=6), 2)
    microscopic = (2 * torch.randint(2, (6, 12, 16, 1)) - 1).double()
    return model, microscopic, two_spin_distribution


@pytest.mark.parametrize('metric', ['euclidean', 'noise_aware'])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_uniform_weights_are_bitwise_legacy_compatible(metric, reduction):
    model, microscopic, pattern = _objective_fixture()
    rng_state = torch.random.get_rng_state()
    expected = None
    for weights in (None, (1., 1.), (7., 7.)):
        candidate = copy.deepcopy(model)
        torch.random.set_rng_state(rng_state)
        loss, terms = qcr_objective(
            candidate, microscopic, pattern_fn=pattern, closure_metric=metric,
            closure_objective='one_step_and_rollout',
            coordinate_reduction=reduction, closure_scale_weights=weights)
        loss.backward()
        result = (loss, terms['commutator'], terms['rollout'],
                  [p.grad for p in candidate.parameters()], torch.random.get_rng_state())
        if expected is None:
            expected = result
        else:
            for actual, reference in zip(result[:3], expected[:3]):
                assert torch.equal(actual, reference)
            for actual, reference in zip(result[3], expected[3]):
                assert (actual is None and reference is None) or torch.equal(actual, reference)
            assert torch.equal(result[4], expected[4])


@pytest.mark.parametrize('metric', ['euclidean', 'noise_aware'])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_objective_uses_weights_for_both_closure_paths(metric, reduction):
    model, microscopic, pattern = _objective_fixture()
    loss, terms = qcr_objective(
        model, microscopic, pattern_fn=pattern, closure_metric=metric,
        closure_objective='one_step_and_rollout',
        coordinate_reduction=reduction, closure_scale_weights=(1., 4.))
    prefix = 'noise_aware_' if metric == 'noise_aware' else ''
    for actual, stem in [('commutator', 'one_step'), ('rollout', 'rollout')]:
        by_scale = [terms[f'{prefix}{stem}_level_{k}'] for k in (1, 2)]
        torch.testing.assert_close(terms[actual], (by_scale[0] + 4*by_scale[1])/5)
        torch.testing.assert_close(terms[f'weighted_{actual}'], terms[actual])
        torch.testing.assert_close(terms[f'unweighted_{actual}'], torch.stack(by_scale).mean())
    assert terms['closure_scale_weight_1'].item() == .2
    assert terms['closure_scale_weight_2'].item() == .8
    weights = LossConfig().weights
    expected = sum(getattr(weights, key)*terms[key] for key in (
        'anchor', 'coarse_consistency', 'kernel_noncollapse', 'separation', 'commutator'))
    torch.testing.assert_close(loss, expected + 5*terms['rollout'])
    loss.backward()
    for module in (model.theory_encoder, model.rg_kernel, model.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_uniform_one_step_keeps_vectorized_legacy_reduction(reduction):
    torch.manual_seed(2)
    predicted = torch.randn(7, 3, 2, dtype=torch.double)
    target = torch.randn_like(predicted)
    expected = latent_squared_error(predicted, target, reduction)
    for weights in (None, (1., 1., 1.)):
        actual, _, unweighted = one_step_closure_loss(predicted, target, reduction, weights)
        assert torch.equal(actual, expected)
        assert torch.equal(unweighted, expected)
