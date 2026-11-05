"""Fresh tiny cache only; no old data, checkpoints, model fitting, or inference."""
from copy import deepcopy
import json

import pytest
import torch

from experiments.prepare_ising_j1j2 import (
    campaign_configs, main, make_config, mi_preprocessing_command,
    protocol_contract, training_command, validate_label_free_cache,
)
from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.runner import file_fingerprint


def test_explicit_current_recipe_and_requested_campaign_only():
    configs = campaign_configs('/new/ensembles.pt')
    assert list(configs) == ['dg5_seed91', 'dg5_seed92', 'dg5_seed93']
    for raw in configs.values():
        cfg = RunConfig.from_dict(raw)
        cfg.validate()
        validate_recipe(build_parser().parse_args([
            '--config', 'new.json', '--output', 'new']), cfg)
        assert cfg.schema_version == 5
        assert cfg.data.physical_model == 'ising2d_j1j2' and not cfg.data.coupling_names
        assert not cfg.data.spin_flip_augment
        assert cfg.model.theory_dim == 5 and cfg.model.local_features == 12
        assert (cfg.model.hidden, cfg.model.width, cfg.model.block_factor) == (96, 3, 3)
        assert cfg.model.affine_reconstruction and not cfg.model.fixed_kernel
        assert cfg.model.kernel_initialization == 'random'
        assert cfg.model.kernel_sampling == 'bernoulli'
        assert cfg.loss.closure_objective == 'all_start_anchor'
        assert cfg.loss.closure_scale_weights is None
        assert cfg.loss.weights.commutator == cfg.loss.rollout_weight == 5
        assert cfg.loss.weights.separation == 10 and cfg.loss.weights.anchor == 2
        assert cfg.loss.kernel_noncollapse_mode == 'variance_floor_ensemble'
        assert cfg.loss.kernel_variance_ratio == cfg.loss.kernel_sign_weight == 1
        assert cfg.training.gradient_mode == 'qc_r_separate'
        assert cfg.training.train_modules == ('Q', 'C', 'R')
        assert (cfg.training.epochs, cfg.training.system_batch, cfg.training.sample_batch) == (12000, 16, 128)
        assert cfg.training.lr_schedule == 'multistep'
        assert cfg.training.lr_milestones == (6000, 9000) and cfg.training.lr_gamma == .3
        assert cfg.evaluation.systems == 24 and cfg.evaluation.samples == 128
        assert ScaleConfig((108, 108), cfg.training.max_rg_steps, 3).shapes() == [(108, 108), (36, 36), (12, 12)]
    ablation = campaign_configs('/new/cache.pt', seeds=(91,), theory_dims=(1, 2, 5))
    reference = deepcopy(ablation['dg5_seed91'])
    for value in ablation.values():
        value = deepcopy(value)
        value['model']['theory_dim'] = 5
        assert value == reference


def test_commands_use_only_standard_driver_and_no_labels_or_warm_start():
    cpu = training_command('/new/python', 'c.json', 'run')
    assert cpu == ['/new/python', '-m', 'experiments.rsmi_separate_qcr',
        '--config', 'c.json', '--output', 'run', '--device', 'cpu', '--skip-final-report']
    gpu = training_command('python', 'c.json', 'run', device='cuda:2',
                           mi_cache_dir='mi', benchmark_warmup=5)
    args = build_parser().parse_args(gpu[3:])
    validate_recipe(args, RunConfig.from_dict(make_config('cache.pt')))
    assert args.skip_final_report and args.mi_test_cpu and args.mi_cache_dir.name == 'mi'
    assert args.benchmark_warmup == 5 and not args.allow_nondefault_recipe
    assert args.source is None and args.resume_checkpoint is None
    assert args.q_coupling_weight == args.q_future_weight == 0
    assert not args.mi_initialize and not args.affine_pca_init
    assert mi_preprocessing_command('python', 'c.json', 'mi.json', cache_dir='mi') == [
        'python', '-m', 'scripts.prepare_qcr_mi_cache', '--config', 'c.json',
        '--output', 'mi.json', '--cache-dir', 'mi']
    with pytest.raises(ValueError, match='CPU execution'):
        training_command('python', 'c', 'r', mi_cache_dir='mi')
    for bad in ('cuda', 'cuda:-1', 'cuda:2;echo', 'mps'):
        with pytest.raises(ValueError, match='explicit physical'):
            training_command('python', 'c', 'r', device=bad)


def tiny_cache(tmp_path):
    cache = tmp_path / 'cache.pt'
    x = torch.ones((2, 4, 12, 12, 1), dtype=torch.int8)
    payload = dict(schema_version=1,
        train=dict(configurations=x, couplings=None,
                   group_ids=torch.tensor([[0, 0, 0], [0, 1, 0]])),
        test=dict(configurations=-x, couplings=None,
                  group_ids=torch.tensor([[0, 0, 8], [5, 0, 0]])))
    torch.save(payload, cache)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(make_config(cache, updates=2, threads=1,
        sample_batch=4, system_batch=2, max_rg_steps=1, evaluation_systems=2)))
    row = lambda r, t, g, rho, part: dict(rho_index=r, temperature_index=t,
        group=g, rho=rho, temperature=2., condition_id=f'{r}:{t}', partition=part)
    metadata = dict(schema_version=1, data_sha256=file_fingerprint(cache),
        train=[row(0, 0, 0, 0., 'train'), row(0, 1, 0, 0., 'train')],
        test=[row(0, 0, 8, 0., 'same_rho_new_group'), row(5, 0, 0, .3, 'heldout_rho')])
    external = tmp_path / 'cache.json'
    external.write_text(json.dumps(metadata))
    return cache, config, external, payload, metadata


def test_label_free_cache_uses_opaque_ids_and_external_physical_index(tmp_path):
    _, config, external, _, _ = tiny_cache(tmp_path)
    before = torch.get_rng_state().clone()
    receipt = validate_label_free_cache(config, external)
    assert torch.equal(before, torch.get_rng_state())
    assert receipt['status'] == 'complete'
    assert not receipt['labels_used_in_training'] and not receipt['model_initialized']
    assert receipt['optimizer_updates'] == 0 and not receipt['equilibrium_certified']
    assert receipt['splits']['train']['shape'] == [2, 4, 12, 12, 1]


@pytest.mark.parametrize('bad', ['labels', 'float', 'cross_temperature_leak',
    'wrong_partition', 'wrong_row', 'wrong_hash', 'missing_rows'])
def test_boundary_rejects_resealed_labels_or_split_leaks(tmp_path, bad):
    cache, config, external, payload, metadata = tiny_cache(tmp_path)
    if bad == 'labels':
        payload['train']['couplings'] = torch.tensor([[0., 2.], [0., 3.]])
    elif bad == 'float':
        payload['train']['configurations'] = payload['train']['configurations'].double()
    elif bad == 'cross_temperature_leak':
        # No duplicate [rho,T,repeat] row, but an entire PT run leaks splits.
        payload['test']['group_ids'][0] = torch.tensor([0, 2, 0])
        metadata['test'][0].update(temperature_index=2, group=0)
    elif bad == 'wrong_partition':
        metadata['test'][0]['partition'] = 'train'
    elif bad == 'wrong_row':
        metadata['train'][1]['temperature_index'] = 3
    elif bad == 'missing_rows':
        metadata.pop('test')
    torch.save(payload, cache)
    metadata['data_sha256'] = '0' * 64 if bad == 'wrong_hash' else file_fingerprint(cache)
    external.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate_label_free_cache(config, external)


def test_preparation_is_exclusive_and_does_not_initialize_models(tmp_path):
    output = tmp_path / 'new' / 'config.json'
    before = torch.get_rng_state().clone()
    main(['--cache', 'not_yet_created.pt', '--output', str(output)])
    assert torch.equal(before, torch.get_rng_state())
    assert RunConfig.load(output).loss.closure_objective == 'all_start_anchor'
    with pytest.raises(FileExistsError):
        main(['--cache', 'not_yet_created.pt', '--output', str(output)])
    contract = protocol_contract()
    assert contract['source_checkpoint'] is None and contract['fresh_QCR']
    assert not contract['future_dimension_ablation']['automatic_execution']
    assert not contract['labels_used_in_training'] and contract['thermal_nu'] is None
