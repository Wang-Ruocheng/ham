"""The paired all-start closure gate is narrow and uses the actual R target."""
import copy
from dataclasses import replace

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.config import (
    DataConfig, EvaluationConfig, ModelConfig, RunConfig, TrainingConfig,
)


def _args(*extra, marked=True):
    command = ['--config', 'unused.json', '--output', 'unused']
    if marked:
        command.append('--all-start-closure-ablation')
    return driver.build_parser().parse_args(command + list(extra))


def _config(mode='all_start_anchor'):
    config = RunConfig(
        data=DataConfig(cache='unused.pt', physical_model='ising2d', coupling_names=('K',)),
        model=ModelConfig(dimension=2, theory_dim=5, hidden=96, block_factor=2),
        training=TrainingConfig(max_rg_steps=3, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=3),
    )
    return replace(config, loss=replace(config.loss, closure_objective=mode))


@pytest.mark.parametrize('mode', ['one_step_and_rollout', 'all_start_anchor'])
def test_named_control_and_anchor_pass_cpu_cuda_and_do_not_rewrite_config(mode):
    args, config = _args(), _config(mode)
    before_config = copy.deepcopy(config.to_dict())
    before_args = copy.deepcopy(vars(args))
    config.validate()
    driver.validate_all_start_closure_ablation(args, config)
    driver.validate_rsmi_options(args)
    driver.validate_cuda_options(args)
    driver.validate_recipe(args, config)
    assert config.to_dict() == before_config
    assert vars(args) == before_args
    assert not args.allow_nondefault_recipe


def test_all_start_is_accepted_without_the_historical_paired_marker():
    args, config = _args(marked=False), _config()
    before = copy.deepcopy(config.to_dict())
    driver.validate_recipe(args, config)
    assert config.to_dict() == before
    assert not args.all_start_closure_ablation
    assert not args.allow_nondefault_recipe


@pytest.mark.parametrize('extra', [
    ['--rsmi-weight', '0'], ['--variance-weight', '0'],
    ['--color-weight', '0'], ['--no-independent-c'],
    ['--estimator', 'infonce'], ['--environment', 'local_pairs'],
])
def test_new_default_without_paired_marker_does_not_relax_other_recipe_guards(extra):
    with pytest.raises(ValueError, match='recipe'):
        driver.validate_recipe(_args(*extra, marked=False), _config())


def test_unmarked_legacy_configuration_keeps_historical_path():
    config = _config('one_step_and_rollout')
    driver.validate_recipe(_args(marked=False), config)
    assert not _args(marked=False).all_start_closure_ablation


@pytest.mark.parametrize('extra', [
    ['--allow-nondefault-recipe'], ['--resume-checkpoint', 'old.pt'],
    ['--ising-b0-ablation', 'none'], ['--mi-initialize'],
    ['--freeze-c-checkpoint', 'old.pt'], ['--affine-pca-init'],
    ['--theory-dim', '5'], ['--q-coupling-weight', '1'],
    ['--q-future-weight', '1'], ['--q-future-control'],
    ['--q-future-pairs', 'pairs.json'], ['--q-future-normalizer', 'normalizer.json'],
    ['--q-only-weight-sweep'], ['--q-only-candidates'],
    ['--q-balanced-reconstruction'], ['--q-last-layer-audit'],
    ['--q-head-structure-audit'], ['--q-feature-bottleneck'],
    ['--q-original-plateau'], ['--q-initialization-pilot'], ['--q-unfolding'],
    ['--r-only-continue'], ['--r-only-refit'], ['--r-only-plateau'],
    ['--joint-train-plateau'], ['--frozen-cache', 'cache.pt'],
])
def test_named_gate_rejects_other_ablations_before_data(extra):
    with pytest.raises(ValueError, match='fresh standalone'):
        driver.validate_all_start_closure_ablation(_args(*extra))


def test_named_gate_rejects_source_without_reading_it():
    args = driver.build_parser().parse_args([
        '--source', 'old', '--output', 'new', '--all-start-closure-ablation'])
    with pytest.raises(ValueError, match='fresh standalone'):
        driver.validate_all_start_closure_ablation(args)


@pytest.mark.parametrize('extra', [
    ['--rsmi-weight', '0'], ['--variance-weight', '0'], ['--color-weight', '0'],
    ['--variance-ratio', '.5'], ['--no-variance-per-ensemble'],
    ['--no-independent-c'], ['--estimator', 'infonce'],
    ['--environment', 'local_pairs'],
])
def test_named_gate_locks_C_objective_and_independent_gradients(extra):
    with pytest.raises(ValueError, match='locked recipe'):
        driver.validate_all_start_closure_ablation(_args(*extra), _config())


@pytest.mark.parametrize('change', [
    {'block_factor': 3}, {'theory_dim': 1}, {'hidden': 32}, {'local_features': 24},
    {'fixed_kernel': True}, {'kernel_initialization': 'positive_decimation'},
    {'kernel_sampling': 'deterministic'}, {'affine_reconstruction': False},
])
def test_named_gate_locks_model_except_authorized_block_two(change):
    config = _config()
    config = replace(config, model=replace(config.model, **change))
    with pytest.raises(ValueError, match='locked recipe'):
        driver.validate_recipe(_args(), config)


@pytest.mark.parametrize('train_depth,eval_depth', [(2, 3), (3, 2), (4, 4)])
def test_named_gate_requires_exactly_four_scale_points(train_depth, eval_depth):
    config = _config()
    config = replace(config,
        training=replace(config.training, max_rg_steps=train_depth),
        evaluation=replace(config.evaluation, max_rg_steps=eval_depth))
    with pytest.raises(ValueError, match='four_scale_points'):
        driver.validate_recipe(_args(), config)


@pytest.mark.parametrize('change', [
    {'isotropy_weight': .1}, {'tangent_weight': .1}, {'orientation_weight': .1},
    {'balanced_reconstruction_weight': 1.}, {'kernel_neutral_weight': .1},
    {'closure_metric': 'noise_aware'}, {'closure_scale_weights': (1., 2., 1.)},
    {'closure_objective': 'prediction_pair'},
])
def test_named_gate_rejects_other_loss_changes(change):
    config = _config()
    config = replace(config, loss=replace(config.loss, **change))
    with pytest.raises(ValueError):
        driver.validate_recipe(_args(), config)


@pytest.mark.parametrize('mode', ['one_step_and_rollout', 'all_start_anchor'])
def test_explicit_uniform_target_weights_are_accepted(mode):
    config = _config(mode)
    config = replace(config, loss=replace(config.loss, closure_scale_weights=(7., 7., 7.)))
    driver.validate_recipe(_args(), config)


def test_component_R_uses_new_weighted_target_not_old_diagnostics_or_TEST():
    config, args = _config(), _args()
    row = dict(epoch=200, train_anchor=.1, train_coarse_consistency=.2,
        train_separation=.3, train_rsmi=-.4, train_kernel_noncollapse=.5,
        train_kernel_sign=.6, train_commutator=70., train_rollout=80.,
        train_closure_total=17.25, test_closure_total=-1e99)
    result = driver.train_component_losses(row, config, args)
    assert result == pytest.approx(dict(
        epoch=200, component_Q=4.2, component_C=.8, component_R=17.25))
    row['test_closure_total'] = 1e99
    assert driver.train_component_losses(row, config, args) == result
    row.pop('train_closure_total')
    assert driver.train_component_losses(row, config, args)['component_R'] == 750.


def test_actual_closure_routes_only_to_R_and_C_keeps_own_objective():
    q = torch.tensor(1., dtype=torch.double, requires_grad=True)
    c = torch.tensor(2., dtype=torch.double, requires_grad=True)
    r = torch.tensor(3., dtype=torch.double, requires_grad=True)
    representation = (q - 2).square()
    c_objective = (c - 3).square()
    closure_total = 10 * (r * q + c - 7).square()
    total = representation + c_objective + closure_total
    driver.independent_component_gradients(
        total, closure_total, c_objective, [q], [c], [r])
    torch.testing.assert_close(q.grad, torch.tensor(-2., dtype=torch.double))
    torch.testing.assert_close(c.grad, torch.tensor(-2., dtype=torch.double))
    torch.testing.assert_close(r.grad, torch.tensor(-40., dtype=torch.double))
