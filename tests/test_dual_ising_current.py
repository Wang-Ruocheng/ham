from pathlib import Path
import numpy as np
import pytest
from recurrent_rg.config import RunConfig
from scripts.run_dual_ising_current import eligible, verify_plan, sha256
from scripts.report_dual_ising_current import closure_metrics


def snapshot():
    return dict(mem_available_gib=64, load1=8, cpu_count=24, io_full_avg10=0,
                gpus=[dict(uuid='GPU-a', free_mib=22000, compute_process_present=False)])


def test_process_presence_rejects_gpu_even_with_free_memory():
    s = snapshot()
    assert len(eligible(s)) == 1
    s['gpus'][0]['compute_process_present'] = True
    assert not eligible(s)


def test_explicit_sharing_is_pinned_and_keeps_load_memory_guards():
    s = snapshot(); gpu = s['gpus'][0]
    gpu.update(compute_process_present=True, utilization_percent=0)
    policy = dict(uuid='GPU-a', max_utilization_percent=20, min_free_mib=20480)
    assert not eligible(s) and eligible(s, policy) == [gpu]
    gpu['uuid'] = 'GPU-other'
    assert not eligible(s, policy)
    gpu['uuid'] = 'GPU-a'; gpu['utilization_percent'] = 21
    assert not eligible(s, policy)
    gpu['utilization_percent'] = None
    assert not eligible(s, policy)
    gpu['utilization_percent'] = 0; gpu['free_mib'] = 20479
    assert not eligible(s, policy)
    gpu['free_mib'] = 22000; s['mem_available_gib'] = 31
    assert not eligible(s, policy)
    s['mem_available_gib'] = 64; s['load1'] = 31
    assert not eligible(s, policy)
    s['load1'] = 8; s['io_full_avg10'] = 20
    assert not eligible(s, policy)


@pytest.mark.parametrize('key,value', [('mem_available_gib', 31), ('load1', 31), ('io_full_avg10', 20)])
def test_host_pressure_rejects_admission(key, value):
    s = snapshot(); s[key] = value
    assert not eligible(s)


def test_snapshot_fingerprint_changes_fail_closed(tmp_path):
    code = tmp_path/'code.py'; code.write_text('frozen')
    data = tmp_path/'data.pt'; data.write_bytes(b'data')
    plan = dict(source_sha256={'code.py': sha256(code)}, data_path=str(data), data_sha256=sha256(data))
    verify_plan(plan, tmp_path)
    code.write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'): verify_plan(plan, tmp_path)


def test_heldout_nrmse_uses_train_target_scale():
    draws = np.zeros((2, 4, 3, 2))
    draws[:, 0, :, :] = -1; draws[:, 1, :, :] = 1
    draws[:, 2, :, :] = -100; draws[:, 3, :, :] = 100
    pred = draws[:, :, [1, 2, 2], :]+2
    result = closure_metrics(draws, pred, 2)
    assert all(r['rmse'] == 2 and r['nrmse'] == 2 for r in result['test'])


def test_fresh_config_has_current_recipe_and_terminal_fit():
    p = Path(__file__).resolve().parents[1]/'configs/dual_ising2d_current_g2_seed91_20261105.json'
    c = RunConfig.load(p); c.validate()
    assert c.model.theory_dim == 2 and c.model.block_factor == 3
    assert c.model.kernel_initialization == 'random' and not c.model.fixed_kernel
    assert c.loss.closure_objective == 'all_start_anchor' and c.loss.closure_scale_weights is None
    assert c.loss.balanced_reconstruction_weight == 0
    assert c.training.terminal_r_fit and c.training.terminal_r_cache_repeats == 8
    assert c.training.r_updates_per_batch == 1 and c.training.gradient_mode == 'qc_r_separate'


@pytest.mark.parametrize('pipeline', ['phi4_ru_ensemble_vae', 'ising_KB_ensemble_vae'])
def test_failed_VAE_numerical_preflight_never_starts_training(tmp_path, monkeypatch, pipeline):
    import json
    from types import SimpleNamespace
    from scripts import run_dual_ising_current as runner
    output = tmp_path/'fresh'
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(output=str(output), pipeline=pipeline,
        verify_vae_normalization=True, config='unused.json', active_wall_limit_seconds=3540)))
    monkeypatch.setattr(runner.sys, 'argv', ['runner', '--plan', str(plan)])
    monkeypatch.setattr(runner.os, 'uname', lambda: SimpleNamespace(nodename='thinklab-105-225'))
    monkeypatch.setattr(runner.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(runner, 'verify_plan', lambda *args: None)
    monkeypatch.setattr(runner, 'resource_snapshot', snapshot)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    commands = []

    def failed_preflight(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=123, wait=lambda **kwargs: 1)

    monkeypatch.setattr(runner.subprocess, 'Popen', failed_preflight)
    with pytest.raises(RuntimeError, match='numerical_preflight exited 1'):
        runner.main()
    assert len(commands) == 1 and 'scripts.verify_vae_normalization' in commands[0]
    assert not (output/'run').exists()
    job = json.loads((output/'job.json').read_text())
    assert job['status'] == 'failed' and job['requires_user_action']
