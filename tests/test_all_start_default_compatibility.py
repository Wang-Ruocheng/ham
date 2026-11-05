"""Fresh closure defaults must not reinterpret any historical saved JSON."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, QCRWeights,
    RunConfig, TrainingConfig,
)


def _fresh_config():
    return RunConfig(
        data=DataConfig(cache='unused.pt', physical_model='ising1d'),
        model=ModelConfig(dimension=1),
        training=TrainingConfig(max_rg_steps=3),
        evaluation=EvaluationConfig(max_rg_steps=3),
    )


def test_fresh_loss_and_run_default_to_all_start_anchor():
    assert LossConfig().closure_objective == 'all_start_anchor'
    config = _fresh_config()
    config.validate()
    assert config.loss.objective_kwargs()['closure_objective'] == 'all_start_anchor'
    assert config.to_dict()['loss']['closure_objective'] == 'all_start_anchor'
    serialized = json.loads(json.dumps(config.to_dict()))
    restored = RunConfig.from_dict(serialized)
    assert json.loads(json.dumps(restored.to_dict())) == serialized


@pytest.mark.parametrize('schema_version', [None, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('include_loss', [False, True])
def test_every_historical_schema_missing_selection_keeps_old_loss(schema_version, include_loss):
    raw = {'data': {'cache': 'unused.pt', 'physical_model': 'ising1d'}}
    if schema_version is not None:
        raw['schema_version'] = schema_version
    if include_loss:
        raw['loss'] = {'geometry_mode': 'relative_local'}
    original = copy.deepcopy(raw)
    loaded = RunConfig.from_dict(raw)
    assert raw == original
    assert loaded.loss.closure_objective == 'one_step_and_rollout'
    assert 'closure_objective' not in loaded.loss.objective_kwargs()
    saved = json.loads(json.dumps(loaded.to_dict()))
    assert saved['loss']['closure_objective'] == 'one_step_and_rollout'
    assert json.loads(json.dumps(RunConfig.from_dict(saved).to_dict())) == saved


@pytest.mark.parametrize('mode', ['one_step_and_rollout', 'all_start_anchor'])
def test_explicit_selection_is_preserved_independently_of_schema(mode):
    for version in (1, 3, 5):
        raw = _fresh_config().to_dict()
        raw['schema_version'] = version
        raw['loss']['closure_objective'] = mode
        loaded = RunConfig.from_dict(raw)
        assert loaded.loss.closure_objective == mode
        assert loaded.to_dict()['loss']['closure_objective'] == mode


def test_omitted_historical_mode_keeps_noise_aware_and_nonuniform_ablation_readable():
    raw = _fresh_config().to_dict()
    raw['loss'].pop('closure_objective')
    raw['loss'].update(closure_metric='noise_aware', rollout_weight=0.,
                       closure_scale_weights=[1., 2., 3.])
    loaded = RunConfig.from_dict(raw)
    loaded.validate()
    kwargs = loaded.loss.objective_kwargs()
    assert 'closure_objective' not in kwargs
    assert kwargs['closure_metric'] == 'noise_aware'
    assert kwargs['rollout_weight'] == 0.
    assert kwargs['closure_scale_weights'] == (1., 2., 3.)


def test_repository_historical_schema5_config_keeps_original_formula():
    path = Path(__file__).resolve().parents[1] / 'configs' / (
        'ising2d_positiveK_g1_affineQ_block2_plateau_seed91.json')
    raw = json.loads(path.read_text())
    assert raw['schema_version'] == 5
    assert 'closure_objective' not in raw['loss']
    loaded = RunConfig.from_dict(raw)
    loaded.validate()
    assert loaded.loss.closure_objective == 'one_step_and_rollout'
    assert 'closure_objective' not in loaded.loss.objective_kwargs()
    assert RunConfig.from_dict(loaded.to_dict()).to_dict() == loaded.to_dict()


def test_historical_config_load_preserves_selection_when_resolving_data_path(tmp_path):
    raw = _fresh_config().to_dict()
    raw['loss'].pop('closure_objective')
    raw['data']['cache'] = '../inputs/data.pt'
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(raw))
    loaded = RunConfig.load(path)
    assert loaded.loss.closure_objective == 'one_step_and_rollout'
    assert loaded.data.cache == str((tmp_path / '../inputs/data.pt').resolve())


def test_fresh_mode_does_not_relax_equal_positive_weight_contract():
    with pytest.raises(ValueError, match='equal positive'):
        replace(LossConfig(), rollout_weight=0.).objective_kwargs()
    with pytest.raises(ValueError, match='equal positive'):
        replace(LossConfig(), weights=QCRWeights(commutator=4.)).objective_kwargs()
    with pytest.raises(ValueError, match='uniform'):
        replace(LossConfig(), closure_scale_weights=(1., 2.)).objective_kwargs()
    with pytest.raises(ValueError, match='euclidean'):
        replace(LossConfig(), closure_metric='noise_aware').objective_kwargs()
