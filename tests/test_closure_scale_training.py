"""Synthetic-only checks for the requested unsupervised scale-weight ablation."""
from dataclasses import replace
import json

import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.config import DataConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig, EvaluationConfig


def test_unspecified_scale_weights_do_not_add_an_ablation_contract():
    cfg = RunConfig(data=DataConfig(cache='unused', physical_model='synthetic'))
    assert driver.closure_scale_contract(cfg) is None


def test_paired_cli_scale_weights_change_r_not_q_c_or_sampling(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7101)
    payload = dict(schema_version=1)
    for split, group in [('train', 0), ('test', 100)]:
        spins = (2 * torch.randint(2, (2, 4, 108, 108, 1), generator=generator) - 1).to(torch.int8)
        payload[split] = dict(configurations=spins, couplings=None,
                              group_ids=torch.tensor([group, group]))
    cache = tmp_path / 'synthetic.pt'
    torch.save(payload, cache)
    pair_data = dict(schema_version=1, data_sha256=driver.file_fingerprint(cache))
    for split, offset in [('train', 0), ('test', 100)]:
        pair_data[split] = dict(condition_ids=[offset, offset], chain_ids=[offset, offset + 1],
                                pairs=[[0, 1]])
    pairs = tmp_path / 'pairs.json'
    pairs.write_text(json.dumps(pair_data))
    cfg = RunConfig(data=DataConfig(cache=str(cache), physical_model='synthetic'),
        model=ModelConfig(dimension=2, theory_dim=1, hidden=5, block_factor=3,
                          affine_reconstruction=True),
        loss=LossConfig(closure_objective='one_step_and_rollout'),
        training=TrainingConfig(max_rg_steps=2, epochs=2, system_batch=2, sample_batch=4,
            threads=1, checkpoint_every=1, log_every=1, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=4, systems=2))
    states = []
    for name, weights in [('uniform', (1., 1.)), ('late', (.2, 1.8))]:
        current = replace(cfg, loss=replace(cfg.loss, closure_scale_weights=weights))
        config = tmp_path / (name + '.json')
        config.write_text(json.dumps(current.to_dict()))
        output = tmp_path / name
        driver.main(['--config', str(config), '--output', str(output), '--skip-final-report',
                     '--q-future-control', '--q-future-pairs', str(pairs)])
        saved = torch.load(output / 'checkpoint_last.pt', map_location='cpu', weights_only=True)
        assert saved['epoch'] == 2
        metadata = saved['metadata']
        assert metadata['labels_used_in_training'] is False
        assert metadata['auxiliary_future_parameters'] == 0
        assert metadata['closure_scale_weighting']['weights'] == list(weights)
        assert metadata['closure_scale_weighting']['application'] == [
            'one_step_commutator', 'microscopic_start_rollout']
        assert 'future_physics_predictor_state' not in saved
        states.append(saved)
    uniform, late = states
    assert uniform['metadata']['initial_module_sha256'] == late['metadata']['initial_module_sha256']
    for key, value in uniform['model'].items():
        if key.startswith(('theory_encoder.', 'rg_kernel.')):
            torch.testing.assert_close(value, late['model'][key], rtol=0, atol=1e-13)
    assert any(not torch.equal(value, late['model'][key]) for key, value in uniform['model'].items()
               if key.startswith('theory_rg.'))
    for key in ('torch_rng_state', 'batch_generator_state', 'rsmi_generator_state'):
        assert torch.equal(uniform[key], late[key])
