import hashlib
import json
import os
import sys

import pytest

from scripts import preview_latest_checkpoint as preview


def test_streaming_digest_is_python310_compatible(tmp_path, monkeypatch):
    data = b'checkpoint' * 200000
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(data)
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    assert preview.digest(path) == hashlib.sha256(data).hexdigest()


def test_default_lr_pilot_reports_scalar_exponents_and_loss_without_mi(tmp_path):
    stages = preview.preview_stages(tmp_path, 1000, tmp_path / 'out', 1,
        coupling_names=['K'], physical_model='ising2d')
    assert [name for name, _ in stages] == ['scalar', 'exponents', 'loss']
    assert 'scripts.report_scalar_joint' in stages[0][1]
    assert stages[0][1][-2:] == ['--sampling-repeats', '0']
    assert 'scripts.report_ising_scalar_exponents' in stages[1][1]
    assert all('scripts.plot_dual_ising_scalar' not in command for _, command in stages)
    assert all('experiments.rsmi_separate_qcr' not in command for _, command in stages)


def test_single_K_vector_preview_routes_to_single_layer_report_without_extra_sampling(tmp_path):
    stages = preview.preview_stages(tmp_path, 1000, tmp_path / 'out', 2,
        coupling_names=['K'], physical_model='ising2d')
    assert [name for name, _ in stages] == ['phase', 'loss']
    assert 'scripts.report_ising2d_K_vector' in stages[0][1]
    assert stages[0][1][-2:] == ['--sampling-repeats', '0']
    assert 'scripts.report_dual_ising_vector' not in stages[0][1]
    field_stages = preview.preview_stages(tmp_path, 1000, tmp_path / 'out', 2,
        coupling_names=['K', 'B'], physical_model='ising2d')
    assert 'scripts.report_ising2d_KB_vector' in field_stages[0][1]


def test_complete_preview_is_reused_without_running_any_report(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'training_step_1000.pt'
    checkpoint.write_bytes(b'checkpoint')
    os.utime(checkpoint, (1., 1.))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(config=dict(
        model=dict(theory_dim=1), data=dict(coupling_names=['K'], physical_model='ising2d')))))
    out = tmp_path / 'monitor_previews' / 'epoch_001000'
    out.mkdir(parents=True)
    (out / 'preview.json').write_text(json.dumps(dict(phase='completed',
        checkpoint_sha256=preview.digest(checkpoint))))
    monkeypatch.setattr(sys, 'argv', ['preview', '--run', str(tmp_path)])
    monkeypatch.setattr(preview.subprocess, 'run', lambda *a, **k: pytest.fail('Do not repeat a preview'))
    assert preview.main() == 0
