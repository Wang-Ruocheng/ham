"""Explicit Q feature-capacity ablations preserve historical default models."""
from dataclasses import replace
import json

import pytest
import torch

from recurrent_rg.checkpoints import build_model, load_run, qcr_state_dict
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D, RecurrentRGNetwork3D


@pytest.mark.parametrize('schema_version', [1, 2, 3, 4, 5])
def test_old_missing_configuration_field_defaults_to_twelve(schema_version):
    config = RunConfig.from_dict(dict(schema_version=schema_version,
        data=dict(cache='not-opened.pt', physical_model='ising2d'),
        model=dict(dimension=2, theory_dim=5)))
    assert config.model.local_features == 12
    assert RunConfig.from_dict(config.to_dict()).model == config.model


@pytest.mark.parametrize('capacity', [12, 24, 48])
@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_binary_factory_wires_only_the_requested_Q_capacity(capacity, dimension):
    config = ModelConfig(dimension=dimension, width=2 if dimension == 3 else 3,
        theory_dim=5, hidden=8, local_features=capacity)
    model = build_model(config, 2)
    q = model.theory_encoder
    assert q.local_mlp[2].weight.shape == (capacity, 8)
    assert q.local_mlp[4].weight.shape == (capacity, capacity)
    assert q.head[0].weight.shape == (8, capacity)
    assert q.head[2].weight.shape == (5, 8)
    assert q.distribution_decoder.weight.shape == (2 ** {1: 2, 2: 4, 3: 8}[dimension], 5)
    assert model.theory_rg[0].weight.shape == (8, 5)


@pytest.mark.parametrize('capacity', [12, 24])
def test_categorical_factory_also_preserves_configured_capacity(capacity):
    config = ModelConfig(dimension=2, categorical_states=3, kernel_sampling='categorical',
        theory_dim=5, hidden=8, local_features=capacity)
    model = build_model(config, 2)
    assert model.theory_encoder.local_mlp[2].out_features == capacity
    assert model.theory_encoder.head[0].in_features == capacity
    assert model.theory_encoder.distribution_decoder.out_features == 3 ** 4


def test_omitted_and_explicit_default_are_bitwise_equal_including_RNG():
    config = ModelConfig(dimension=2, theory_dim=5, hidden=8)
    states, rng = [], []
    for spec in (config, replace(config, local_features=12)):
        torch.manual_seed(91)
        states.append(qcr_state_dict(build_model(spec, 2)))
        rng.append(torch.get_rng_state())
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
    assert torch.equal(rng[0], rng[1])


@pytest.mark.parametrize('cls,kwargs', [
    (RecurrentRGNetwork1D, {}), (RecurrentRGNetwork2D, {}),
    (RecurrentRGNetwork3D, {'width': 2}),
])
def test_convenience_constructors_accept_capacity_without_moving_old_parameters(cls, kwargs):
    model = cls(levels=3, hidden=8, theory_dim=5, local_features=24, **kwargs)
    assert model.theory_encoder.local_mlp[2].out_features == 24


@pytest.mark.parametrize('invalid', [0, -1, 12.0, True, None, '24'])
def test_invalid_capacity_rejected_before_architecture_construction(invalid):
    with pytest.raises(ValueError, match='local_features'):
        ModelConfig(local_features=invalid)
    with pytest.raises(ValueError, match='local_features'):
        RecurrentRGNetwork2D(local_features=invalid)


@pytest.mark.parametrize('capacity,omit_field', [(12, True), (24, False), (48, False)])
def test_versioned_checkpoint_round_trip_preserves_architecture(tmp_path, capacity, omit_field):
    config = RunConfig(data=DataConfig(cache='not-opened.pt', physical_model='ising2d'),
        model=ModelConfig(dimension=2, hidden=8, theory_dim=5, local_features=capacity),
        training=TrainingConfig(max_rg_steps=2))
    torch.manual_seed(91)
    original = build_model(config.model, 2)
    saved = config.to_dict()
    if omit_field:
        saved['model'].pop('local_features')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(config=saved)))
    torch.save(qcr_state_dict(original), tmp_path / 'qcr_model.pt')
    rng = torch.get_rng_state().clone()
    restored = load_run(tmp_path)
    assert restored.model_config.local_features == capacity
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, qcr_state_dict(restored.model)[name])
               for name, value in qcr_state_dict(original).items())


def test_unversioned_checkpoint_can_recover_unambiguous_feature_shape(tmp_path):
    config = ModelConfig(dimension=2, hidden=8, theory_dim=5, block_factor=3,
        kernel_sampling='deterministic', affine_reconstruction=False, local_features=24)
    model = build_model(config, 2)
    (tmp_path / 'run_config.json').write_text(json.dumps(dict(dimension=2, width=3,
        block_factor=3, levels=3, kernel_layout='nonoverlapping')))
    torch.save(qcr_state_dict(model), tmp_path / 'qcr_model.pt')
    restored = load_run(tmp_path)
    assert restored.model_config.local_features == 24
    assert restored.model.theory_encoder.head[0].in_features == 24


def test_versioned_checkpoint_rejects_conflicting_explicit_capacity(tmp_path):
    config = RunConfig(data=DataConfig(cache='not-opened.pt', physical_model='ising2d'),
        model=ModelConfig(dimension=2, hidden=8, theory_dim=5, local_features=24),
        training=TrainingConfig(max_rg_steps=2))
    model = build_model(config.model, 2)
    saved = config.to_dict()
    saved['model']['local_features'] = 12
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(config=saved)))
    torch.save(qcr_state_dict(model), tmp_path / 'qcr_model.pt')
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_run(tmp_path)


def test_nondefault_capacity_batched_binary_Q_matches_dense_values_and_gradients():
    torch.manual_seed(91)
    model = build_model(ModelConfig(dimension=2, hidden=8, theory_dim=5, local_features=24), 2)
    q = model.theory_encoder
    inputs = [(2 * torch.randint(2, (2, 4, size, size, 1)) - 1).double()
              for size in (9, 3)]
    parameters = tuple(q.local_mlp.parameters()) + tuple(q.head.parameters())
    actual = q.forward_many(inputs, validated_binary_inputs=True)
    gradients = torch.autograd.grad(sum(value.square().sum() for value in actual), parameters)
    q.binary_lookup = False
    expected = [q(value) for value in inputs]
    reference_gradients = torch.autograd.grad(sum(value.square().sum() for value in expected), parameters)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    for left, right in zip(gradients, reference_gradients):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)


def test_low_RAM_and_reference_training_preserve_custom_capacity_and_updates(tmp_path):
    from experiments import rsmi_separate_qcr as driver

    generator = torch.Generator().manual_seed(43)
    train = (2 * torch.randint(2, (3, 8, 36, 36, 1), generator=generator) - 1).double()
    test = (2 * torch.randint(2, (2, 8, 36, 36, 1), generator=generator) - 1).double()
    data = tmp_path / 'data.pt'
    torch.save((train, torch.tensor([.2, .3, .4]), test, torch.tensor([.5, .6])), data)
    config = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(data), physical_model='ising2d', coupling_names=['K']),
        model=dict(dimension=2, theory_dim=5, hidden=8, local_features=24,
                   kernel_sampling='bernoulli', affine_reconstruction=True),
        training=dict(max_rg_steps=2, epochs=1, system_batch=2, sample_batch=4,
            checkpoint_every=1, log_every=1, gradient_mode='qc_r_separate', threads=1),
        evaluation=dict(max_rg_steps=2, samples=4, systems=2)))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config.to_dict()))
    states = []
    for name, extra in [('reference', ['--reference-execution']), ('low_RAM', [])]:
        output = tmp_path / name
        driver.main(['--config', str(config_path), '--output', str(output), '--skip-final-report'] + extra)
        states.append(torch.load(output / 'training_step_1.pt', map_location='cpu', weights_only=True))
        assert states[-1]['metadata']['config']['model']['local_features'] == 24
        assert states[-1]['model']['theory_encoder.local_mlp.2.weight'].shape == (24, 8)
        assert not states[-1]['metadata']['labels_used_in_training']
    assert all(torch.equal(value, states[1]['model'][name])
               for name, value in states[0]['model'].items())
    assert states[0]['history'] == states[1]['history']
