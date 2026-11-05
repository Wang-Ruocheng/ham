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
