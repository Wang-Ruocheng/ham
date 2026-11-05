"""Real default-driver integration, using synthetic spins and only two updates.

This is an execution regression, NOT a physical sampling/training result.
No SSH, GPU, historical data, production output or scientific report is used.
"""
from __future__ import annotations

import json
import sys

import torch

from experiments.prepare_ising_j1j2 import make_config, training_command
from experiments import rsmi_separate_qcr as trainer
from recurrent_rg.data.ensembles import load_ensembles
from scripts.report_ising_j1j2 import sha256
from scripts.report_ising_zh import chinese_figures


def test_actual_default_driver_label_free_two_update_smoke(tmp_path):
    """L36 ->12 ->4; exact ordinary driver, no bypass or legacy flags."""
    generator = torch.Generator().manual_seed(2026102609)
    cache = tmp_path / 'synthetic_only.pt'
    source = {}
    for split, groups in (
        ('train', [[0, 0, 0], [0, 1, 0]]),
        ('test', [[0, 0, 8], [5, 0, 0]]),
    ):
        values = 2 * torch.randint(2, (2, 8, 36, 36, 1), generator=generator) - 1
        source[split] = dict(configurations=values.to(torch.int8), couplings=None,
                             group_ids=torch.tensor(groups, dtype=torch.int64))
    torch.save(dict(schema_version=1, **source), cache)
    fingerprint = sha256(cache)
    config = make_config(cache, seed=191, updates=2, threads=2,
                         sample_batch=8, system_batch=2, max_rg_steps=2,
                         evaluation_systems=2)
    config_file = tmp_path / 'synthetic_only_config.json'
    config_file.write_text(json.dumps(config))
    output = tmp_path / 'synthetic_only_run'
    command = training_command(sys.executable, config_file, output, device='cpu')
    assert '--skip-final-report' in command
    assert '--source' not in command and '--resume' not in command
    assert not any('legacy' in argument or 'nondefault' in argument for argument in command)
    arguments = command[command.index('experiments.rsmi_separate_qcr') + 1:]
    with chinese_figures():
        trainer.main(arguments)
    summary = json.loads((output / 'summary.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    checkpoint = torch.load(output / 'training_step_2.pt', weights_only=True, map_location='cpu')
    assert summary['actual_epochs'] == summary['requested_epochs'] == 2
    assert not summary['stopped_by_training_limit'] and not summary['stopped_by_wall_limit']
    assert summary['config'] == json.loads(config_file.read_text())
    assert manifest['config']['schema_version'] == 5
    assert manifest['config']['loss']['closure_objective'] == 'all_start_anchor'
    assert manifest['config']['loss']['closure_scale_weights'] is None
    assert manifest['config']['training']['gradient_mode'] == 'qc_r_separate'
    assert manifest['independent_c'] is True
    assert manifest['labels_used_in_training'] is False
    assert checkpoint['metadata']['labels_used_in_training'] is False
    assert manifest['source'] is None and not manifest.get('resumed_from')
    assert not manifest.get('q_supervision') and not manifest.get('q_future_physics')
    assert not manifest.get('frozen_C') and not manifest.get('data_driven_initialization')
    assert manifest['estimator'] == 'histogram'
    assert manifest['environment_mode'] == 'full_shell' and manifest['buffer'] == 1
    assert manifest['rsmi_weight'] == 2.
    assert manifest['skip_final_report'] is True
    assert checkpoint['epoch'] == 2
    assert checkpoint['optimizer']['state']
    assert all(float(state['step']) == 2 for state in checkpoint['optimizer']['state'].values())
    assert sha256(cache) == manifest['data_sha256'] == checkpoint['metadata']['data_sha256'] == fingerprint
    assert not (output / 'full_report').exists()
    assert all(dataset.couplings is None for dataset in load_ensembles(cache))
