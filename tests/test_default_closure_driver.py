"""Fresh training adopts all-start closure without rewriting historical runs."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig,
)


def _raw():
    return dict(schema_version=5,
        data=dict(cache='unused.pt', physical_model='ising2d', coupling_names=['K']),
        model=dict(dimension=2, theory_dim=1, affine_reconstruction=True),
        training=dict(max_rg_steps=2, gradient_mode='qc_r_separate'),
        evaluation=dict(max_rg_steps=2))


def test_fresh_default_is_separate_from_historical_deserialization(tmp_path):
    assert LossConfig().closure_objective == 'all_start_anchor'
    raw = _raw()
    before = deepcopy(raw)
    loaded = RunConfig.from_dict(raw)
    assert loaded.loss.closure_objective == 'one_step_and_rollout'
    fresh = driver.default_closure(loaded, raw)
    assert fresh.loss.closure_objective == 'all_start_anchor'
    assert loaded.loss.closure_objective == 'one_step_and_rollout'
    assert raw == before
    path = tmp_path / 'historical.json'
    path.write_text(json.dumps(raw))
    assert RunConfig.load(path).loss.closure_objective == 'one_step_and_rollout'
    assert RunConfig.from_dict(fresh.to_dict()).loss.closure_objective == 'all_start_anchor'


@pytest.mark.parametrize('context', [dict(continuing=True), dict(historical=True),
                                  dict(continuing=True, historical=True)])
def test_continuations_and_explicit_historical_contexts_do_not_upgrade(context):
    raw = _raw()
    loaded = RunConfig.from_dict(raw)
    assert driver.default_closure(loaded, raw, **context) is loaded


@pytest.mark.parametrize('mode', ['one_step_and_rollout', 'all_start_anchor'])
@pytest.mark.parametrize('context', [dict(), dict(continuing=True), dict(historical=True)])
def test_explicit_objective_is_authoritative_in_every_context(mode, context):
    raw = _raw()
    raw['loss'] = dict(closure_objective=mode)
    loaded = RunConfig.from_dict(raw)
    assert driver.default_closure(loaded, raw, **context) is loaded
    assert loaded.loss.closure_objective == mode


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_current_b3_scalar_default_works_at_each_depth_without_ablation_flags(depth):
    cfg = RunConfig(data=DataConfig('unused.pt', 'ising2d', ('K',)),
        model=ModelConfig(dimension=2, theory_dim=1),
        training=TrainingConfig(max_rg_steps=depth, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=depth))
    args = driver.build_parser().parse_args(['--config', 'unused.json', '--output', 'unused'])
    cfg.validate()
    driver.validate_recipe(args, cfg)
    driver.validate_rsmi_options(args)
    driver.validate_cuda_options(args)
    assert cfg.model.block_factor == 3
    assert cfg.loss.closure_objective == 'all_start_anchor'
    assert not args.all_start_closure_ablation
    assert not args.allow_nondefault_recipe


def test_new_manifest_contract_describes_one_actual_R_objective():
    cfg = RunConfig(data=DataConfig('unused.pt', 'ising2d'))
    value = driver.closure_objective_contract(cfg)
    assert value['objective'] == 'all_start_anchor'
    assert value['total_weight'] == 10.
    assert value['target_scale_weights'] == 'uniform'
    assert value['starts_within_target'] == 'uniform over all earlier scales'
    assert value['prediction_prediction_penalty'] is False
    assert value['labels_used'] is False
    assert value['gradient_R'] == 'grad(10 L_all_start_anchor); no feedback to QC'
    assert 'nonlinear R' in value['interpretation']
    assert 'not a finite-dimensional linear Koopman operator' in value['interpretation']
    assert driver.closure_scale_contract(cfg) is None


def test_uniform_weight_contract_means_equal_endpoints_not_old_double_branches():
    cfg = RunConfig(data=DataConfig('unused.pt', 'ising2d'),
        training=TrainingConfig(max_rg_steps=3),
        loss=LossConfig(closure_scale_weights=(7., 7., 7.)))
    value = driver.closure_scale_contract(cfg)
    assert value['weights'] == [7., 7., 7.]
    assert value['normalized_weights'] == pytest.approx([1/3, 1/3, 1/3])
    assert value['target_levels'] == [1, 2, 3]
    assert value['application'] == ['all_start_anchor']
    assert value['total_weight'] == 10.
    assert value['labels_used'] is False
    assert 'mean over all earlier starts' in value['normalization']


def test_historical_manifest_contract_retains_old_branch_weights():
    cfg = RunConfig(data=DataConfig('unused.pt', 'ising2d'),
        loss=LossConfig(closure_objective='one_step_and_rollout', rollout_weight=0.))
    value = driver.closure_objective_contract(cfg)
    assert value['objective'] == 'one_step_and_rollout'
    assert value['one_step_weight'] == 5.
    assert value['rollout_weight'] == 0.
    assert value['gradient_R'] == 'grad(5 L_commutator + 0 L_rollout); no feedback to QC'
    assert value['labels_used'] is False


@pytest.mark.parametrize('mode', ['one_step_and_rollout', 'all_start_anchor'])
def test_omitted_resume_field_inherits_saved_actual_objective(mode):
    raw = _raw()
    cfg = RunConfig.from_dict(raw)
    saved = deepcopy(raw)
    saved['loss'] = dict(closure_objective=mode)
    resumed = driver.resume_closure(cfg, raw, dict(metadata=dict(config=saved)))
    assert resumed.loss.closure_objective == mode
    assert cfg.loss.closure_objective == 'one_step_and_rollout'
    assert 'loss' not in raw


def test_pre_policy_checkpoint_missing_mode_stays_historical_on_resume():
    raw = _raw()
    cfg = RunConfig(data=DataConfig('unused.pt', 'ising2d'))
    resumed = driver.resume_closure(cfg, raw, dict(metadata=dict(config=raw)))
    assert resumed.loss.closure_objective == 'one_step_and_rollout'


def test_explicit_resume_mode_is_not_silently_rewritten_from_checkpoint():
    raw = _raw()
    raw['loss'] = dict(closure_objective='one_step_and_rollout')
    cfg = RunConfig.from_dict(raw)
    saved = deepcopy(raw)
    saved['loss']['closure_objective'] = 'all_start_anchor'
    assert driver.resume_closure(cfg, raw, dict(metadata=dict(config=saved))) is cfg
    assert driver.resume_closure(cfg, raw, None) is cfg


def test_fresh_standard_b3_three_step_driver_uses_all_start_without_experimental_flag(tmp_path):
    import torch
    generator = torch.Generator().manual_seed(1729)
    train = (2 * torch.randint(2, (2, 4, 108, 108, 1), generator=generator) - 1).double()
    test = (2 * torch.randint(2, (2, 4, 108, 108, 1), generator=generator) - 1).double()
    cache = tmp_path / 'synthetic.pt'
    torch.save((train, torch.tensor([.2, .5]), test, torch.tensor([.3, .6])), cache)
    raw = _raw()
    raw['data']['cache'] = str(cache)
    raw['model']['hidden'] = 4
    raw['training'].update(max_rg_steps=3, epochs=2, system_batch=2,
        sample_batch=4, threads=1, log_every=1, checkpoint_every=1)
    raw['evaluation'].update(max_rg_steps=3, samples=4, systems=2)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(raw))
    output = tmp_path / 'fresh'
    driver.main(['--config', str(config), '--output', str(output), '--skip-final-report'])
    manifest = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert manifest['config']['loss']['closure_objective'] == 'all_start_anchor'
    assert manifest['closure_objective_contract']['total_weight'] == 10.
    assert manifest['gradient_R'] == 'grad(10 L_all_start_anchor); no feedback to QC'
    assert manifest['labels_used_in_training'] is False
    assert 'closure_ablation' not in manifest
    assert summary['actual_epochs'] == 2
    assert 'train_all_start_pair_1_3' in summary['history'][-1]
    assert summary['history'][-1]['train_closure_total'] == pytest.approx(
        10. * summary['history'][-1]['train_all_start_anchor'])
    assert 'loss' not in json.loads(config.read_text())
