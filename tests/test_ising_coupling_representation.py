"""Explicit natural-parameter Q ablation must not silently alter C/R or core files."""
from dataclasses import replace
from contextlib import nullcontext
import json

import pytest
import torch

from experiments import ising_coupling_representation as experiment
from experiments import rsmi_separate_qcr as trainer
from recurrent_rg import checkpoints, losses, representation
from recurrent_rg.config import ModelConfig, RunConfig, QCRWeights
from recurrent_rg.coupling_decoder import DirectionalCouplingDecoder
from recurrent_rg.data.statistics import plaquette_distribution


def model_config(dim=1):
    return ModelConfig(dimension=2, block_factor=2, width=3, theory_dim=dim, hidden=5,
        affine_reconstruction=True, kernel_sampling='bernoulli')


def batch():
    return (2*torch.randint(2, (3, 4, 20, 20, 1))-1).double()


def run_config():
    return RunConfig.from_dict(dict(schema_version=5,
        model=dict(dimension=2, block_factor=2, width=3, theory_dim=1, hidden=5,
                   affine_reconstruction=True, kernel_sampling='bernoulli'),
        data=dict(cache='/not/read.pt', physical_model='ising2d'),
        training=dict(max_rg_steps=2, epochs=2, seed=91, train_modules=['Q','C','R']),
        evaluation=dict(max_rg_steps=2),
        loss=dict(weights=dict(anchor=2., coarse_consistency=5., separation=10.,
                               commutator=5., kernel_noncollapse=2.), rollout_weight=5.)))


@pytest.mark.parametrize('dim', [1, 3])
def test_replacement_preserves_original_QCR_tensors_and_rng_without_unused_affine_decoder(dim):
    torch.manual_seed(91)
    original = checkpoints.build_model(model_config(dim), 2)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    natural = experiment.build_coupling_model(model_config(dim), 2)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in checkpoints.qcr_state_dict(original).items():
        if not name.startswith('theory_encoder.distribution_decoder.'):
            torch.testing.assert_close(value, checkpoints.qcr_state_dict(natural)[name], atol=0., rtol=0.)
    assert isinstance(natural.theory_encoder.distribution_decoder, DirectionalCouplingDecoder)
    keys = natural.theory_encoder.distribution_decoder.state_dict()
    assert set(keys) == {'linear.weight', 'linear.bias'}
    assert keys['linear.weight'].shape == (7, dim)


def test_hooks_are_restored_after_exceptions_and_forbid_nested_hooks():
    before = (checkpoints.build_model, trainer.build_model, trainer.default_representation,
              trainer.add_supervision_contract, trainer.checkpoint_payload,
              losses.data_geometry_objective, representation.normalize_affine_decoder_)
    with pytest.raises(RuntimeError, match='deliberate'):
        with experiment.coupling_representation_hooks():
            assert losses.data_geometry_objective is experiment.natural_data_geometry_objective
            with pytest.raises(RuntimeError, match='nested'):
                with experiment.coupling_representation_hooks():
                    pass
            raise RuntimeError('deliberate')
    after = (checkpoints.build_model, trainer.build_model, trainer.default_representation,
             trainer.add_supervision_contract, trainer.checkpoint_payload,
             losses.data_geometry_objective, representation.normalize_affine_decoder_)
    assert before == after


def test_effective_config_skips_old_histogram_bias_calibration_and_records_true_loss():
    cfg = run_config()
    with experiment.coupling_representation_hooks():
        effective = trainer.default_representation(cfg, cfg.to_dict(), continuing=False)
        assert effective.model.affine_reconstruction is False
        contract = {}
        trainer.add_supervision_contract(contract, 0.)
        assert contract['representation_override']['kind'] == experiment.OVERRIDE_KIND
        assert contract['q_reconstruction']['reconstruction_weight'] == 10.
        assert contract['q_reconstruction']['subset_effective_weights'] == [20., 5.]
        assert contract['labels_used_in_training'] is False and contract['q_supervision'] is None
        with pytest.raises(ValueError, match='cannot supervise'):
            trainer.add_supervision_contract({}, 1.)


@pytest.mark.parametrize('dim', [1, 3])
def test_reconstruction_only_change_keeps_original_C_and_R_gradients(dim):
    torch.set_num_threads(1)
    torch.manual_seed(711)
    observed = batch()
    outputs, gradients = [], []
    for natural in (False, True):
        torch.manual_seed(91)
        model = (experiment.build_coupling_model if natural else checkpoints.build_model)(model_config(dim), 2)
        model.detach_kernel_representation = True
        torch.manual_seed(31)
        context = experiment.coupling_representation_hooks() if natural else nullcontext()
        with context:
            total, terms = losses.qcr_objective(model, observed, pattern_fn=plaquette_distribution,
                weights=QCRWeights(anchor=2., separation=10., coarse_consistency=5., commutator=5., kernel_noncollapse=2.),
                kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1., kernel_sign_weight=1.)
            closure = 5*(terms['commutator']+terms['rollout'])
            trainer.independent_component_gradients(total, closure,
                2*terms['kernel_noncollapse']+terms['kernel_sign'],
                model.theory_encoder.parameters(), model.rg_kernel.parameters(), model.theory_rg.parameters())
        gradients.append([[None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]
                          for module in (model.rg_kernel, model.theory_rg)])
        outputs.append(terms)
        if natural:
            assert all(f'reconstruction_nll_level_{k}' in terms for k in range(3))
            assert not any(name.startswith('reconstruction_mse_') for name in terms)
            assert model.theory_encoder.distribution_decoder.linear.weight.grad.abs().sum() > 0
    for before, after in zip(*gradients):
        for a, b in zip(before, after):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for name in ('anchor', 'coarse_consistency', 'commutator', 'rollout', 'kernel_noncollapse', 'kernel_sign'):
        torch.testing.assert_close(outputs[0][name], outputs[1][name], atol=0., rtol=0.)


def test_crossfit_reconstruction_and_subset_weights_match_exact_declared_algebra():
    model = experiment.build_coupling_model(model_config(), 2)
    model.detach_kernel_representation = True
    levels = [batch(), batch()[:, :, :10, :10], batch()[:, :, :5, :5]]
    first = torch.randn(3, 3, 1, dtype=torch.float64, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    weights = QCRWeights(anchor=2., coarse_consistency=5., separation=10., kernel_noncollapse=2., commutator=5.)
    total, terms = experiment.natural_data_geometry_objective(model, levels, first, second,
        lambda _: (_ for _ in ()).throw(AssertionError('Natural reconstruction must not call histogram targets')),
        weights, 4., 2., kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.)
    expected_reconstruction = torch.stack([model.theory_encoder.distribution_decoder.crossfit_loss(
        first[:, k], level[:, 2:], second[:, k], level[:, :2]) for k, level in enumerate(levels)]).mean()
    torch.testing.assert_close(terms['separation'], expected_reconstruction)
    expected = (20*losses.latent_squared_error(first[:,0], second[:,0])
        + 5*losses.latent_squared_error(first[:,1:], second[:,1:])
        + 10*expected_reconstruction + 2*terms['kernel_noncollapse']
        + 5*terms['commutator'] + 5*terms['rollout'])
    torch.testing.assert_close(total, expected)


def test_independent_C_input_requirement_fails_closed():
    model = experiment.build_coupling_model(model_config(), 2)
    with pytest.raises(ValueError, match='detach C'):
        experiment.natural_data_geometry_objective(model, [batch()]*3,
            torch.zeros(3,3,1), torch.zeros(3,3,1), None, QCRWeights(), 4., 2.)


def test_custom_numbered_checkpoint_loader_roundtrips_without_changing_core(tmp_path):
    cfg = run_config()
    with experiment.coupling_representation_hooks():
        effective = trainer.default_representation(cfg, cfg.to_dict())
        model = trainer.build_model(effective.model, 2)
        contract = dict(config=effective.to_dict(), data_sha256='not-read')
        trainer.add_supervision_contract(contract, 0.)
        optimizer = torch.optim.Adam(model.parameters())
        payload = trainer.checkpoint_payload(model, optimizer, torch.Generator().manual_seed(1), 2,
            metadata=contract, history=[])
    torch.save(payload, tmp_path/'training_step_2.pt')
    (tmp_path/'manifest.json').write_text(json.dumps(contract))
    loaded = experiment.load_coupling_run(tmp_path, 2)
    assert loaded.model_config.affine_reconstruction is False
    assert loaded.model.theory_encoder.distribution_reconstruction_mode == experiment.MODE
    for key, value in checkpoints.qcr_state_dict(model).items():
        torch.testing.assert_close(value, checkpoints.qcr_state_dict(loaded.model)[key], atol=0., rtol=0.)
    with pytest.raises(ValueError):
        checkpoints.load_run(tmp_path, 2)


def test_optional_gauge_is_dispatched_without_affine_probability_assumptions():
    with experiment.coupling_representation_hooks():
        model = trainer.build_model(model_config(3), 2)
        with torch.no_grad():
            model.theory_encoder.distribution_decoder.linear.weight *= 3
        representation.normalize_affine_decoder_(model.theory_encoder)
        torch.testing.assert_close(model.theory_encoder.distribution_decoder.linear.weight.norm(dim=0),
                                   torch.ones(3, dtype=torch.float64))


def test_fresh_validation_rejects_extra_losses_or_wrong_recipe():
    cfg = run_config()
    experiment.validate_fresh_config(cfg)
    for changed in (replace(cfg, model=replace(cfg.model, fixed_kernel=True)),
                    replace(cfg, loss=replace(cfg.loss, tangent_weight=.1)),
                    replace(cfg, model=replace(cfg.model, block_factor=3))):
        with pytest.raises(ValueError):
            experiment.validate_fresh_config(changed)


def test_checkpoint_contract_guard_will_not_silently_save_an_unlabelled_override():
    with experiment.coupling_representation_hooks():
        model = trainer.build_model(model_config(), 2)
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError, match='contract'):
            trainer.checkpoint_payload(model, optimizer, torch.Generator(), 1, metadata={})


def test_cached_hooks_read_only_configuration_metadata_and_forbid_MI_rebuild(tmp_path, monkeypatch):
    data = tmp_path/'physical.pt'
    data.write_bytes(b'existing-test-receipt-placeholder')
    data.with_suffix('.json').write_text(json.dumps(dict(data_sha256=experiment.DATA_SHA256)))
    cfg = run_config()
    cfg = replace(cfg, data=replace(cfg.data, cache=str(data)))
    class Split(dict):
        def __getitem__(self, key):
            if key in ('couplings', 'group_ids'):
                raise AssertionError('Training must not read coupling/group labels')
            return super().__getitem__(key)
    # Expanded metadata has scalar storage; no huge test allocation or full-data scan.
    observed = torch.ones(1, dtype=torch.float64)
    payload = dict(schema_version=1,
        train=Split(configurations=observed.expand(516,128,108,108,1)),
        test=Split(configurations=observed.expand(252,128,108,108,1)))
    def read(path, **kwargs):
        assert kwargs == dict(map_location='cpu', weights_only=True, mmap=True)
        return payload
    monkeypatch.setattr(experiment.torch, 'load', read)
    key = next(iter(experiment.MI_KEYS))
    def existing(directory, key_data, build):
        return 'saved-statistics', dict(hit=True, key=key, cache_version=1)
    monkeypatch.setattr(experiment.mi_cache, 'load_or_build_mi_statistics', existing)
    with experiment.cached_data_hooks(cfg, experiment.DATA_SHA256) as receipt:
        training, test = trainer.prepared_data(cfg)
        assert training.couplings is None and test.couplings is None
        assert trainer.file_fingerprint(data) == experiment.DATA_SHA256
        assert receipt['data_rehashed'] is False and receipt['coupling_labels_accessed'] is False
        result, _ = experiment.mi_cache.load_or_build_mi_statistics(tmp_path, {},
            lambda: (_ for _ in ()).throw(AssertionError('Should never run user build')))
        assert result == 'saved-statistics'
    def missing(directory, key_data, build):
        build()
    monkeypatch.setattr(experiment.mi_cache, 'load_or_build_mi_statistics', missing)
    with experiment.cached_data_hooks(cfg, experiment.DATA_SHA256):
        with pytest.raises(RuntimeError, match='rebuilding is forbidden'):
            experiment.mi_cache.load_or_build_mi_statistics(tmp_path, {}, lambda: None)


@pytest.mark.parametrize('kind', ['tuple', 'list'])
def test_original_four_slot_cache_never_reads_either_coupling_label_slot(tmp_path, monkeypatch, kind):
    data = tmp_path/'original-four-slot.pt'
    data.write_bytes(b'original-audited-tuple-receipt-placeholder')
    data.with_suffix('.json').write_text(json.dumps(dict(data_sha256=experiment.DATA_SHA256)))
    cfg = run_config()
    cfg = replace(cfg, data=replace(cfg.data, cache=str(data)))
    class GuardTuple(tuple):
        def __getitem__(self, index):
            if index in (1, 3):
                raise AssertionError('K label slots must never be read in training')
            return super().__getitem__(index)
    class GuardList(list):
        def __getitem__(self, index):
            if index in (1, 3):
                raise AssertionError('K label slots must never be read in training')
            return super().__getitem__(index)
    scalar = torch.ones(1, dtype=torch.float64)
    contents = (scalar.expand(516,128,108,108,1), object(),
                scalar.expand(252,128,108,108,1), object())
    payload = (GuardTuple if kind == 'tuple' else GuardList)(contents)
    monkeypatch.setattr(experiment.torch, 'load', lambda *args, **kwargs: payload)
    with experiment.cached_data_hooks(cfg, experiment.DATA_SHA256):
        training, test = trainer.prepared_data(cfg)
        assert training.configurations.shape[0] == 516
        assert test.configurations.shape[0] == 252
        assert training.couplings is None and test.couplings is None


@pytest.mark.parametrize('payload', [(), [1, 2, 3], dict(schema_version=2)])
def test_unrecognized_physical_cache_layout_is_rejected(tmp_path, monkeypatch, payload):
    data = tmp_path/'bad-layout.pt'
    data.write_bytes(b'receipt-placeholder')
    data.with_suffix('.json').write_text(json.dumps(dict(data_sha256=experiment.DATA_SHA256)))
    cfg = run_config()
    cfg = replace(cfg, data=replace(cfg.data, cache=str(data)))
    monkeypatch.setattr(experiment.torch, 'load', lambda *args, **kwargs: payload)
    with experiment.cached_data_hooks(cfg, experiment.DATA_SHA256):
        with pytest.raises(ValueError, match='four-slot'):
            trainer.prepared_data(cfg)
