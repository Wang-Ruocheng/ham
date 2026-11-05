"""All-start anchored consistency and unchanged historical direct-call closure."""
import copy
from dataclasses import replace

import pytest
import torch

from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, QCRWeights,
    RunConfig, TrainingConfig,
)
from recurrent_rg.losses import (
    all_start_anchor_closure_loss,
    multistep_closure_loss, one_step_closure_loss, qcr_objective,
)


class _ScalarR:
    def __init__(self, value=1.7):
        self.a = torch.tensor(value, dtype=torch.double, requires_grad=True)

    def advance_theory(self, value):
        return self.a * value


class _NonlinearR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2).double()

    def advance_theory(self, value):
        return value + torch.tanh(self.layer(value))


def _old_closure(model, direct, reduction='sum'):
    step, _, _ = one_step_closure_loss(
        model.advance_theory(direct[:, :-1]), direct[:, 1:], reduction)
    rollout, _ = multistep_closure_loss(model, direct, reduction)
    return 5 * step + 5 * rollout


def test_config_default_roundtrip_and_legacy_kwargs():
    default = LossConfig()
    assert default.closure_objective == 'all_start_anchor'
    assert default.objective_kwargs()['closure_objective'] == 'all_start_anchor'
    historical = replace(default, closure_objective='one_step_and_rollout')
    assert 'closure_objective' not in historical.objective_kwargs()
    selected = default
    assert selected.objective_kwargs()['closure_objective'] == 'all_start_anchor'
    config = RunConfig(
        data=DataConfig(cache='unused.pt', physical_model='ising1d'),
        model=ModelConfig(dimension=1), loss=selected,
        training=TrainingConfig(max_rg_steps=3),
        evaluation=EvaluationConfig(max_rg_steps=3),
    )
    config.validate()
    loaded = RunConfig.from_dict(config.to_dict())
    assert loaded.loss == selected
    loaded.validate()


@pytest.mark.parametrize('changes', [
    {'closure_objective': 'unknown'},
    {'closure_metric': 'noise_aware'},
    {'closure_scale_weights': (1., 2., 1.)},
    {'closure_scale_weights': (0., 0., 0.)},
    {'rollout_weight': 0.},
    {'rollout_weight': 4.},
    {'rollout_weight': float('inf')},
    {'rollout_weight': float('nan')},
    {'weights': QCRWeights(commutator=0.)},
    {'weights': QCRWeights(commutator=float('inf'))},
    {'weights': QCRWeights(commutator=float('nan'))},
])
def test_config_rejects_ambiguous_all_start_objectives(changes):
    config = replace(LossConfig(closure_objective='all_start_anchor'), **changes)
    with pytest.raises(ValueError):
        config.objective_kwargs()


@pytest.mark.parametrize('steps', [1, 2])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_one_or_two_steps_equal_legacy_values_and_gradients(steps, reduction):
    torch.manual_seed(7)
    model = _NonlinearR()
    direct = torch.randn(5, steps + 1, 2, dtype=torch.double, requires_grad=True)
    old = _old_closure(model, direct, reduction)
    new, targets, pairs = all_start_anchor_closure_loss(model, direct, reduction)
    assert len(targets) == steps
    assert set(pairs) == {(start, target) for target in range(1, steps + 1)
                          for start in range(target)}
    torch.testing.assert_close(10 * new, old, rtol=1e-13, atol=1e-13)
    parameters = list(model.parameters()) + [direct]
    old_gradients = torch.autograd.grad(old, parameters, retain_graph=True)
    new_gradients = torch.autograd.grad(10 * new, parameters)
    for actual, expected in zip(new_gradients, old_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_three_steps_coefficients_and_new_intermediate_start():
    model = _ScalarR()
    direct = torch.tensor([[[1.], [3.], [10.], [21.]]], dtype=torch.double)
    loss, targets, e = all_start_anchor_closure_loss(model, direct)
    assert len(targets) == 3 and len(e) == 6
    torch.testing.assert_close(targets[0], e[0, 1])
    torch.testing.assert_close(targets[1], (e[0, 2] + e[1, 2]) / 2)
    torch.testing.assert_close(targets[2], (e[0, 3] + e[1, 3] + e[2, 3]) / 3)
    expected = ((10 / 3) * e[0, 1] + (5 / 3) * (e[0, 2] + e[1, 2])
                + (10 / 9) * (e[0, 3] + e[1, 3] + e[2, 3]))
    torch.testing.assert_close(10 * loss, expected)
    old = _old_closure(model, direct)
    delta = (10 * e[1, 3] - 5 * e[0, 3] - 5 * e[2, 3]) / 9
    torch.testing.assert_close(10 * loss - old, delta)
    assert abs(float(delta)) > 1.
    old_gradient = torch.autograd.grad(old, model.a, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(10 * loss, model.a)[0]
    assert not torch.allclose(old_gradient, new_gradient)


def test_recurrent_prefixes_are_not_detached():
    model = _ScalarR(value=2.)
    direct = torch.tensor([[[1.], [3.], [10.], [21.]]], dtype=torch.double)
    _, _, pairs = all_start_anchor_closure_loss(model, direct)
    # d(a^3 - 21)^2/da = 6*a^2*(a^3 - 21), including all three R calls.
    actual = torch.autograd.grad(pairs[0, 3], model.a, retain_graph=True)[0]
    expected = 6 * model.a.square() * (model.a.pow(3) - 21)
    torch.testing.assert_close(actual, expected)
    # The newly included path also retains both shared-parameter R calls.
    actual = torch.autograd.grad(pairs[1, 3], model.a)[0]
    expected = 12 * model.a * (3 * model.a.square() - 21)
    torch.testing.assert_close(actual, expected)


def test_uniform_scale_weights_do_not_change_all_start_math():
    direct = torch.tensor([[[1.], [3.], [10.], [21.]]], dtype=torch.double)
    model = _ScalarR()
    reference = all_start_anchor_closure_loss(model, direct)
    for weights in ((1., 1., 1.), (7., 7., 7.)):
        actual = all_start_anchor_closure_loss(model, direct, closure_scale_weights=weights)
        assert torch.equal(actual[0], reference[0])
        for a, b in zip(actual[1], reference[1]):
            assert torch.equal(a, b)
        for key in reference[2]:
            assert torch.equal(actual[2][key], reference[2][key])


@pytest.mark.parametrize('weights', [(1., 2., 1.), (1., 1.), (0., 0., 0.)])
def test_helper_rejects_nonuniform_or_wrong_depth_weights(weights):
    with pytest.raises(ValueError):
        all_start_anchor_closure_loss(
            _ScalarR(), torch.ones(2, 4, 1, dtype=torch.double),
            closure_scale_weights=weights)


@pytest.mark.parametrize('shape', [(3, 1), (3, 1, 2), (0, 4, 2), (3, 4, 0)])
def test_helper_rejects_invalid_trajectory_shape(shape):
    with pytest.raises(ValueError, match='encodings'):
        all_start_anchor_closure_loss(_ScalarR(), torch.ones(shape, dtype=torch.double))


def _objective_fixture(steps=3):
    from recurrent_rg.checkpoints import build_model
    from recurrent_rg.data.statistics import two_spin_distribution
    torch.manual_seed(13)
    model = build_model(ModelConfig(dimension=1, theory_dim=2, hidden=6), steps)
    microscopic = (2 * torch.randint(2, (6, 12, 4 * 2**steps, 1)) - 1).double()
    return model, microscopic, two_spin_distribution


def test_explicit_default_is_bitwise_legacy_and_adds_no_diagnostics():
    model, microscopic, pattern_fn = _objective_fixture()
    state = torch.random.get_rng_state()
    results = []
    for options in ({}, {'closure_objective': 'one_step_and_rollout'}):
        candidate = copy.deepcopy(model)
        torch.random.set_rng_state(state)
        total, terms = qcr_objective(candidate, microscopic, pattern_fn=pattern_fn, **options)
        total.backward()
        results.append((total, terms, [p.grad for p in candidate.parameters()],
                        torch.random.get_rng_state()))
        assert 'closure_total' not in terms
        assert 'all_start_anchor' not in terms
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1].keys() == results[1][1].keys()
    for key in results[0][1]:
        assert torch.equal(results[0][1][key], results[1][1][key])
    for actual, expected in zip(results[0][2], results[1][2]):
        assert (actual is None and expected is None) or torch.equal(actual, expected)
    assert torch.equal(results[0][3], results[1][3])


def test_all_start_objective_reuses_C_trajectory_and_common_diagnostics():
    model, microscopic, pattern_fn = _objective_fixture()
    state = torch.random.get_rng_state()
    outputs = []
    for mode in ('one_step_and_rollout', 'all_start_anchor'):
        torch.random.set_rng_state(state)
        levels = []
        total, terms = qcr_objective(
            model, microscopic, pattern_fn=pattern_fn, closure_objective=mode,
            levels_sink=levels)
        outputs.append((total, terms, levels, torch.random.get_rng_state()))
    old, new = outputs
    for actual, expected in zip(new[2], old[2]):
        assert torch.equal(actual, expected)
    assert torch.equal(new[3], old[3])
    for key, value in old[1].items():
        assert torch.equal(new[1][key], value)
    terms = new[1]
    weights = LossConfig().weights
    expected = sum(getattr(weights, key) * terms[key] for key in (
        'anchor', 'coarse_consistency', 'kernel_noncollapse', 'separation'))
    torch.testing.assert_close(new[0], expected + terms['closure_total'])
    torch.testing.assert_close(terms['closure_total'], 10 * terms['all_start_anchor'])
    assert 'all_start_pair_1_3' in terms
    assert {f'all_start_target_{target}' for target in (1, 2, 3)} <= terms.keys()


def test_actual_closure_routing_keeps_Q_representation_only():
    from recurrent_rg.training import separate_qc_r_gradients
    model, microscopic, pattern_fn = _objective_fixture()
    total, terms = qcr_objective(
        model, microscopic, pattern_fn=pattern_fn, closure_objective='all_start_anchor')
    weights = LossConfig().weights
    representation = sum(getattr(weights, key) * terms[key] for key in (
        'anchor', 'coarse_consistency', 'separation'))
    q_parameters = list(model.theory_encoder.parameters())
    r_parameters = list(model.theory_rg.parameters())
    expected_Q = torch.autograd.grad(
        representation, q_parameters, retain_graph=True, allow_unused=True)
    expected_R = torch.autograd.grad(
        terms['closure_total'], r_parameters, retain_graph=True, allow_unused=True)
    separate_qc_r_gradients(total, terms['closure_total'], q_parameters, r_parameters)
    for parameter, expected in zip(q_parameters + r_parameters, expected_Q + expected_R):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('steps', [1, 2])
def test_fresh_default_short_chain_value_and_separate_gradients_match_legacy(steps):
    from recurrent_rg.training import separate_qc_r_gradients
    model, microscopic, pattern_fn = _objective_fixture(steps)
    state = torch.random.get_rng_state()
    outputs = []
    for loss_config in (LossConfig(), LossConfig(closure_objective='one_step_and_rollout')):
        candidate = copy.deepcopy(model)
        torch.random.set_rng_state(state)
        total, terms = qcr_objective(
            candidate, microscopic, pattern_fn=pattern_fn, **loss_config.objective_kwargs())
        old_closure = (loss_config.weights.commutator * terms['commutator']
                       + loss_config.rollout_weight * terms['rollout'])
        actual_closure = terms.get('closure_total', old_closure)
        q_parameters = list(candidate.theory_encoder.parameters())
        r_parameters = list(candidate.theory_rg.parameters())
        separate_qc_r_gradients(total, actual_closure, q_parameters, r_parameters)
        outputs.append((total, actual_closure, [p.grad for p in q_parameters + r_parameters],
                        torch.random.get_rng_state()))
    new, old = outputs
    torch.testing.assert_close(new[0], old[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new[1], old[1], rtol=1e-12, atol=1e-12)
    for actual, expected in zip(new[2], old[2]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert torch.equal(new[3], old[3])


@pytest.mark.parametrize('options', [
    {'closure_objective': 'unknown'},
    {'closure_objective': 'all_start_anchor', 'closure_metric': 'noise_aware'},
    {'closure_objective': 'all_start_anchor', 'rollout_weight': 4.},
    {'closure_objective': 'all_start_anchor', 'closure_scale_weights': (1., 2., 1.)},
])
def test_direct_objective_rejects_invalid_mode_before_sampling(options):
    state = torch.random.get_rng_state()
    with pytest.raises(ValueError):
        qcr_objective(None, torch.ones(2, 4, 32, 1), pattern_fn=None, **options)
    assert torch.equal(torch.random.get_rng_state(), state)
