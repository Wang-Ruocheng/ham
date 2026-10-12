"""Detached orchestration must not overwrite outputs or retry failed stages."""
import json
from pathlib import Path

import pytest

from scripts import run_scalar_lr_gpu_pipeline as pipeline


def test_existing_job_is_not_resumed(tmp_path):
    output = tmp_path / 'job'
    output.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.main(['--cache', str(tmp_path / 'cache.pt'), '--output', str(output)])


def test_incomplete_cache_fails_before_creating_job(tmp_path):
    cache = tmp_path / 'cache.pt'
    cache.touch()
    output = tmp_path / 'job'
    with pytest.raises(ValueError, match='Incomplete'):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert not output.exists()


def test_prepare_then_sweep_is_single_existing_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'abc123\n')
    monkeypatch.setattr(pipeline, 'check_data', lambda cache: dict(sha256='audited'))
    calls = []
    monkeypatch.setattr(pipeline, 'stage', lambda out, status, name, command: calls.append((name, command)))
    output = tmp_path / 'job'
    status = pipeline.main(['--cache', str(tmp_path / 'cache.pt'), '--output', str(output)])
    assert [name for name, _ in calls] == ['data_generation', 'lr_sweep']
    assert calls[0][1][3] == 'scripts.prepare_ising2d_lr_data'
    assert calls[1][1][3] == 'scripts.run_scalar_lr_sweep'
    assert '--resume-checkpoint' not in calls[1][1]
    assert status['git_commit'] == 'abc123' and status['status'] == 'complete'
    assert json.loads((output / 'job.json').read_text())['data']['sha256'] == 'audited'


def test_failure_is_recorded_without_retry_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'abc123\n')
    calls = []

    def fail(out, status, name, command):
        calls.append(name)
        raise RuntimeError('sampler failed')

    monkeypatch.setattr(pipeline, 'stage', fail)
    output = tmp_path / 'job'
    with pytest.raises(RuntimeError, match='sampler failed'):
        pipeline.main(['--cache', str(tmp_path / 'cache.pt'), '--output', str(output)])
    status = json.loads((output / 'job.json').read_text())
    assert status['status'] == 'failed' and calls == ['data_generation']
