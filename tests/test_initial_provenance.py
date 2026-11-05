"""Synthetic-only initialization identities: no data/cache/checkpoint reads."""
import ast
import copy
import inspect

import pytest
import torch

import experiments.rsmi_separate_qcr as trainer
from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig


def _model():
    with isolated_rng('cpu', 917):
        return build_model(ModelConfig(dimension=1, theory_dim=2, hidden=5,
                                       affine_reconstruction=True), 2)


def test_calibrated_training_start_is_separate_from_legacy_constructor_alias():
    model = _model()
    constructor = trainer.qcr_module_sha256(model)
    pre_affine = trainer.qcr_module_sha256(model)
    # A TRAIN-only synthetic calibration changes only the decoder bias.
    with torch.no_grad():
        model.theory_encoder.distribution_decoder.bias.copy_(
            torch.tensor([.1, .2, .3, .4], dtype=torch.double))
    metadata = dict(initial_module_sha256=copy.deepcopy(pre_affine))
    trainer.record_initial_module_provenance(metadata, model,
        random_construction_sha256=constructor, pre_affine_sha256=pre_affine)

    assert metadata['initial_module_sha256'] == constructor
    assert metadata['random_construction_module_sha256'] == constructor
    assert metadata['pre_affine_initialization_module_sha256'] == pre_affine
    assert metadata['training_start_module_sha256'] == trainer.qcr_module_sha256(model)
    assert metadata['training_start_module_sha256']['Q'] != constructor['Q']
    for name in ('C', 'R'):
        assert metadata['training_start_module_sha256'][name] == constructor[name]
    assert metadata['module_initialization_provenance']['schema_version'] == 1
    assert 'legacy alias' in metadata['initial_module_sha256_scope']


def test_hashing_and_recording_preserve_values_gradients_and_rng():
    model = _model()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    hashes = trainer.qcr_module_sha256(model)
    metadata = {}
    trainer.record_initial_module_provenance(metadata, model,
        random_construction_sha256=hashes, pre_affine_sha256=hashes)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, tensors[name])
    for parameter, gradient in zip(model.parameters(), gradients):
        assert torch.equal(parameter.grad, gradient)
    # Metadata gets independent copies, not caller-owned mutable dictionaries.
    hashes['Q'] = 'caller changed it'
    assert metadata['random_construction_module_sha256']['Q'] != hashes['Q']


def test_warm_start_is_not_described_as_random_construction():
    model = _model()
    constructor = trainer.qcr_module_sha256(model)
    with torch.no_grad():
        next(model.theory_rg.parameters()).add_(.1)
    warm_start = trainer.qcr_module_sha256(model)
    metadata = dict(initial_module_sha256=copy.deepcopy(warm_start))
    trainer.record_initial_module_provenance(metadata, model,
        random_construction_sha256=constructor, pre_affine_sha256=warm_start)
    assert metadata['random_construction_module_sha256'] != warm_start
    assert metadata['pre_affine_initialization_module_sha256'] == warm_start
    assert metadata['training_start_module_sha256'] == warm_start
    assert metadata['initial_module_sha256'] == warm_start


def test_resume_preserves_initial_source_metadata_and_records_distinct_loaded_state():
    model = _model()
    hashes = trainer.qcr_module_sha256(model)
    source = dict(initial_module_sha256=copy.deepcopy(hashes))
    trainer.record_initial_module_provenance(source, model,
        random_construction_sha256=hashes, pre_affine_sha256=hashes)
    original = copy.deepcopy(source)
    with torch.no_grad():
        next(model.theory_encoder.parameters()).add_(.5)
    loaded_hashes = trainer.qcr_module_sha256(model)
    metadata = dict(initial_module_sha256=loaded_hashes)
    trainer.record_initial_module_provenance(metadata, model,
        resume=dict(epoch=23, metadata=source))
    for name in trainer._INITIAL_MODULE_METADATA_FIELDS:
        assert metadata[name] == original[name]
    assert metadata['resume_start_module_sha256'] == loaded_hashes
    assert metadata['training_start_module_sha256'] != loaded_hashes
    assert metadata['resume_start_provenance']['source_epoch'] == 23
    assert metadata['resume_start_provenance']['training_start_initial_state_available']
    assert 'not an initial model' in metadata['resume_start_provenance']['scope']
    metadata['module_initialization_provenance']['schema_version'] = 999
    assert source == original


@pytest.mark.parametrize('legacy_hash_available', [False, True])
def test_legacy_resume_does_not_invent_missing_constructor_or_training_start(legacy_hash_available):
    model = _model()
    source = ({'initial_module_sha256': {'Q': 'old-Q', 'C': 'old-C', 'R': 'old-R'}}
              if legacy_hash_available else {})
    metadata = dict(initial_module_sha256=trainer.qcr_module_sha256(model))
    trainer.record_initial_module_provenance(metadata, model,
        random_construction_sha256={'Q': 'throwaway'},
        pre_affine_sha256={'Q': 'trained'}, resume=dict(epoch=42, metadata=source))
    for name in trainer._INITIAL_MODULE_METADATA_FIELDS:
        if name in source:
            assert metadata[name] == source[name]
        else:
            assert name not in metadata
    record = metadata['resume_start_provenance']
    assert not record['training_start_initial_state_available']
    assert record['legacy_initial_scope'].startswith('Unknown legacy scope')
    assert metadata['resume_start_module_sha256'] == trainer.qcr_module_sha256(model)


def test_fresh_recording_requires_both_initialization_stages():
    with pytest.raises(ValueError, match='constructor and pre-affine'):
        trainer.record_initial_module_provenance({}, _model())


def test_summary_contract_inherits_fresh_actual_start_scope_by_value():
    model = _model()
    hashes = trainer.qcr_module_sha256(model)
    metadata = dict(initial_module_sha256=copy.deepcopy(hashes))
    trainer.record_initial_module_provenance(metadata, model,
        random_construction_sha256=hashes, pre_affine_sha256=hashes)
    contract = dict(unrelated='preserve', resume_start_provenance={'stale': True})
    trainer.sync_initial_module_provenance(contract, metadata)
    assert contract['unrelated'] == 'preserve'
    assert 'resume_start_provenance' not in contract
    for name in trainer._INITIAL_MODULE_METADATA_FIELDS:
        assert contract[name] == metadata[name]
    contract['module_initialization_provenance']['schema_version'] = 99
    assert metadata['module_initialization_provenance']['schema_version'] == 1


@pytest.mark.parametrize('legacy_hash_available', [False, True])
def test_summary_contract_removes_unavailable_initial_fields_on_legacy_resume(legacy_hash_available):
    model = _model()
    source = ({'initial_module_sha256': {'Q': 'old-Q', 'C': 'old-C', 'R': 'old-R'}}
              if legacy_hash_available else {})
    metadata = {}
    trainer.record_initial_module_provenance(metadata, model,
        resume=dict(epoch=42, metadata=source))
    contract = {name: 'stale constructor claim' for name in trainer._INITIAL_MODULE_METADATA_FIELDS}
    contract['unrelated'] = 'preserve'
    trainer.sync_initial_module_provenance(contract, metadata)
    for name in trainer._INITIAL_MODULE_METADATA_FIELDS:
        if name in metadata:
            assert contract[name] == metadata[name]
        else:
            assert name not in contract
    assert contract['unrelated'] == 'preserve'
    for name in ('resume_start_module_sha256', 'resume_start_provenance'):
        assert contract[name] == metadata[name]
        assert contract[name] is not metadata[name]
    contract['resume_start_provenance']['source_epoch'] = 0
    assert metadata['resume_start_provenance']['source_epoch'] == 42


def test_main_records_hashes_on_the_correct_sides_of_bias_calibration():
    # Guard call-site ordering without running any physical data or training.
    tree = ast.parse(inspect.getsource(trainer.main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_calls = lambda name: [node for node in calls
                              if isinstance(node.func, ast.Name) and node.func.id == name]
    constructor_hash_line = min(node.lineno for node in name_calls('qcr_module_sha256'))
    prepared_line = name_calls('prepared_data')[0].lineno
    calibration_line = next(node.lineno for node in calls
                            if isinstance(node.func, ast.Attribute) and node.func.attr == 'copy_'
                            and any(isinstance(arg, ast.Name) and arg.id == 'center'
                                    for arg in node.args))
    record_line = name_calls('record_initial_module_provenance')[0].lineno
    summary_sync_line = name_calls('sync_initial_module_provenance')[0].lineno
    train_loop_line = next(node.lineno for node in ast.walk(tree)
                           if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                           and node.target.id == 'step')
    assert constructor_hash_line < prepared_line < calibration_line < record_line < summary_sync_line < train_loop_line
