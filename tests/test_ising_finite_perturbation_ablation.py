"""无标签有限扰动 R-only 五臂设计、可靠方向 gate 与完整残差谱回归。"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import ising_finite_perturbation_ablation as ablation
from experiments.ising_response_loss_ablation import state_sha256, write_json
from scripts.diagnose_ising_mcrg import sha256_file


def source_model_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Tanh(),
            torch.nn.Linear(7, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).double()
    state = {f'theory_rg.{key}': value.clone() for key, value in model.state_dict().items()}
    state['theory_encoder.frozen_example'] = torch.ones(2, dtype=torch.float64)
    state['rg_kernel.frozen_example'] = torch.ones(4, dtype=torch.float64)
    return state


def cache(*, depth=2, conditions=3, replicas=6, draws=2, rank=3, smoke=True):
    generator = torch.Generator().manual_seed(1223 + depth)
    ne = conditions * replicas
    direct = .1 * torch.randn(ne, draws, depth+1, 3, generator=generator, dtype=torch.float64)
    matrix = torch.zeros(3, 6, dtype=torch.float64)
    matrix[:rank, :rank] = torch.diag(torch.tensor([.8, .4, .2][:rank], dtype=torch.float64))
    vin = matrix.expand(ne, draws, depth, 3, 6).clone()
    vin += .001 * torch.randn(*vin.shape, generator=generator, dtype=torch.float64)
    if rank < 3:
        vin[..., rank:, :] = 0.
    vout = 1.4 * vin + .002
    condition = torch.arange(conditions).repeat_interleave(replicas)
    replica = torch.arange(replicas).repeat(conditions)
    train = replica < (replicas-2) if depth == 2 else torch.zeros(ne, dtype=torch.bool)
    return dict(schema_version=1, direct=direct, vin=vin, vout=vout,
        score_std=torch.ones(ne, draws, depth, 6, dtype=torch.float64),
        train_mask=train, baseline_mask=condition == 0,
        condition_index=condition, replica_index=replica,
        condition_opaque_id=[f'opaque_{i}' for i in condition.tolist()],
        c_draw_seeds=torch.ones(ne, draws, dtype=torch.int64),
        size=108 if depth == 2 else 54, logical_size=108 if depth == 2 else 54,
        depth=depth, smoke=smoke)


def checked_cache(**kwargs):
    raw = cache(**kwargs)
    return ablation.validate_cache(raw, depth=kwargs.get('depth', 2), smoke=kwargs.get('smoke', True))


def source_and_caches(tmp_path, *, smoke=True, rank=3):
    run, caches = tmp_path/'source'/'run', tmp_path/'cache'
    run.mkdir(parents=True); caches.mkdir()
    (run.parent/'report').mkdir()
    write_json(run.parent/'report/metrics.json', dict(train_representation=[
        dict(scale=i, pooled_coordinate_std=float(i+1), coordinate_std=[float(i+1)]*3)
        for i in range(3)]))
    state = source_model_state()
    payload = dict(epoch=12000, model=state,
        metadata=dict(labels_used_in_training=False, q_supervision=None,
            config=dict(schema_version=5, model=dict(theory_dim=3, block_factor=2, hidden=7))))
    torch.save(payload, run/'training_step_12000.pt')
    options = dict(conditions=3, replicas=6, draws=2, rank=rank, smoke=True) if smoke else dict(
        conditions=14, replicas=12, draws=5, rank=rank, smoke=False)
    training = cache(**options)
    if not smoke:
        training['train_mask'] = training['replica_index'] < 8
        training['baseline_mask'] = training['condition_index'] < 2
    torch.save(training, caches/'training_L108.pt')
    torch.save(cache(depth=1, conditions=2, replicas=4, draws=options['draws'], rank=rank, smoke=smoke),
               caches/'evaluation_L54.pt')
    receipt = dict(status='complete', workflow_complete=True, smoke=smoke,
        no_K_labels_in_training_cache=True,
        source_checkpoint_sha256=sha256_file(run/'training_step_12000.pt'),
        source_frozen_QC_sha256_before=state_sha256(state),
        source_frozen_QC_sha256_after=state_sha256(state),
        source_R_sha256=state_sha256(state, residual=True),
        cache_sha256={name:sha256_file(caches/name) for name in ('training_L108.pt','evaluation_L54.pt')})
    write_json(caches/'receipt.json', receipt)
    return run, caches


def test_cache_forbids_physics_labels_and_detaches_inputs():
    raw = cache()
    raw['direct'].requires_grad_()
    parsed = ablation.validate_cache(raw, depth=2, smoke=True)
    assert parsed['direct'].requires_grad is False
    assert parsed['direct'].data_ptr() != raw['direct'].data_ptr()
    for key in ('K', 'delta', 'epsilon', 'coupling', 'family', 'condition_parameters'):
        contaminated = {**raw, key:torch.ones(1)}
        with pytest.raises(ValueError, match='白名单'):
            ablation.validate_cache(contaminated, depth=2, smoke=True)


@pytest.mark.parametrize('failure', ['duplicate', 'mask_dtype', 'response_shape', 'nan', 'negative_std'])
def test_cache_rejects_bad_independence_axes_and_values(failure):
    raw = cache()
    if failure == 'duplicate': raw['replica_index'][1] = raw['replica_index'][0]
    elif failure == 'mask_dtype': raw['train_mask'] = raw['train_mask'].long()
    elif failure == 'response_shape': raw['vin'] = raw['vin'][..., :5]
    elif failure == 'nan': raw['direct'][0,0,0,0] = float('nan')
    else: raw['score_std'][0,0,0,0] = -1
    with pytest.raises(ValueError): ablation.validate_cache(raw, depth=2, smoke=True)


def test_independent_size_must_not_enter_training():
    raw = cache(depth=1)
    raw['train_mask'][0] = True
    with pytest.raises(ValueError, match='仅评价'):
        ablation.validate_cache(raw, depth=1, smoke=True)


def test_reliability_is_train_only_and_averages_C_before_physical_bootstrap():
    data = checked_cache()
    first, rows = ablation.reliable_projectors(data, seed=4, bootstraps=100)
    changed = deepcopy(data)
    changed['vin'][~changed['train_mask']] *= 1e9
    changed['vout'][~changed['train_mask']] *= -1e8
    second, other = ablation.reliable_projectors(changed, seed=4, bootstraps=100)
    assert rows == other
    assert all(row['physical_train_ensembles'] == 4 for row in rows)
    assert all(row['reliable_rank'] == 3 for row in rows)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
        assert first[key].shape == (6, 3)
    repeated = deepcopy(data)
    for name in ('direct', 'vin', 'vout', 'score_std'):
        repeated[name] = repeated[name].repeat_interleave(3, dim=1)
    _, repeated_rows = ablation.reliable_projectors(repeated, seed=4, bootstraps=100)
    for a, b in zip(rows, repeated_rows):
        assert a['physical_train_ensembles'] == b['physical_train_ensembles']
        assert a['noise_spectral_norm_q95'] == pytest.approx(b['noise_spectral_norm_q95'], rel=1e-10)


def test_gate_ignores_zero_baseline_but_requires_every_nonzero_condition_transition():
    rows = [dict(condition_index=c, transition=t, baseline=c==0,
                 reliable_rank=1 if c==0 else 3) for c in range(3) for t in range(2)]
    assert ablation.identifiability(rows)['training_allowed'] is True
    rows[-1]['reliable_rank'] = 2
    formal = ablation.identifiability(rows)
    assert formal['training_allowed'] is False
    assert formal['failed_nonbaseline_condition_transitions'] == [
        dict(condition_index=2, transition=1, reliable_rank=2)]
    assert ablation.identifiability(rows, smoke=True)['smoke_bypasses_scientific_gate'] is True
    assert ablation.identifiability([])['training_allowed'] is False


def test_whitening_retains_only_reliable_right_directions_and_R_parameter_gradients():
    data = checked_cache()
    model = ablation.FrozenResidualR(source_model_state())
    values = ablation.data_slice(data, data['train_mask'])
    conditions = data['condition_index'][data['train_mask']]
    projectors, _ = ablation.reliable_projectors(data, seed=3, bootstraps=100)
    result = ablation.whitened_response(model, *values, conditions, projectors)
    assert result['response_white'] > 0
    result['response_white'].backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(not v.requires_grad for v in values)
    text = inspect.getsource(ablation.whitened_response)
    assert 'pinv' not in text and 'Kc' not in text


def test_zero_reliable_rank_skips_without_division_or_invented_directions():
    data = checked_cache(rank=0)
    projectors, rows = ablation.reliable_projectors(data, seed=5, bootstraps=100)
    assert all(row['reliable_rank'] == 0 for row in rows)
    assert all(p.shape == (6, 0) for p in projectors.values())
    selected = data['train_mask']
    result = ablation.whitened_response(ablation.FrozenResidualR(source_model_state()),
        *ablation.data_slice(data, selected), data['condition_index'][selected], projectors)
    assert float(result['response_white']) == 0.
    assert result['white_valid_ensemble_draw_transitions'] == 0


def test_two_replica_smoke_single_train_chain_is_not_pseudoreplicated():
    raw = cache(replicas=2)
    raw['train_mask'] = raw['replica_index'] == 0
    data = ablation.validate_cache(raw, depth=2, smoke=True)
    with pytest.raises(ValueError, match='至少两条'):
        ablation.reliable_projectors(data, seed=2, bootstraps=100)
    projectors, rows = ablation.reliable_projectors(data, seed=2, bootstraps=100,
                                                   allow_single_chain_smoke=True)
    assert all(row['physical_train_ensembles'] == 1 for row in rows)
    assert all(row['physical_chain_bootstrap_valid'] is False for row in rows)
    assert all(row['noise_spectral_norm_q95'] is None for row in rows)
    assert all(row['reliable_rank'] == 0 for row in rows)
    assert all(p.shape == (6,0) for p in projectors.values())
    gate = ablation.identifiability(rows, smoke=True)
    assert gate['scientific_gate_passed'] is False
    assert gate['smoke_bypasses_scientific_gate'] is True


def test_standardized_old_response_uses_domain_train_target_scale_normalization():
    data = checked_cache()
    mask = data['train_mask']
    norm, raw = ablation.train_normalization(data, mask)
    _, expected = ablation.standardized_response(data['vin'][mask], data['vout'][mask],
                                                data['score_std'][mask])
    torch.testing.assert_close(raw, expected.square().sum(-2).mean((0,1,3)))
    torch.testing.assert_close(norm, raw.clamp_min(1e-12))
    changed = deepcopy(data); changed['vout'][~mask] = 1e12
    torch.testing.assert_close(ablation.train_normalization(changed, mask)[0], norm)


def test_complete_DR_identity_all_modes_fd_and_target_lattice_std():
    model = ablation.FrozenResidualR(source_model_state())
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    data = checked_cache(depth=1)
    stats = ablation.model_statistics(model, data, 'cpu', [1., 2., 4.])
    np.testing.assert_allclose(stats['full_DR'], np.broadcast_to(np.eye(3), stats['full_DR'].shape))
    np.testing.assert_allclose(stats['full_DR_fd_h1e_5'], stats['full_DR'], atol=1e-11)
    assert stats['full_DR_eigenvalues_real'].shape == data['direct'].shape
    assert np.all(stats['full_DR_eigenvalues_imag'] == 0)
    # 独立54的内部scale1是实际L27，用源scale2 std=4，不误用源scale1。
    assert stats['closure_target_source_TRAIN_pooled_std'][0,0] == 4.
    np.testing.assert_allclose(stats['closure_nrmse_per_physical_ensemble'][:,0,0],
                              stats['closure_rmse_per_physical_ensemble'][:,0,0]/4)
    assert stats['R_of_mean_g'].shape == (len(data['direct']), 2, 3)
    assert stats['existing_R_relative_error'].shape == (len(data['direct']),2,1)


def test_source_and_cache_hashes_and_frozen_state_are_checked(tmp_path):
    run, caches = source_and_caches(tmp_path)
    ablation.load_inputs(run, caches, smoke=True)
    raw = torch.load(caches/'training_L108.pt', weights_only=True)
    raw['direct'] += .001
    torch.save(raw, caches/'training_L108.pt')
    with pytest.raises(ValueError, match='指纹'):
        ablation.load_inputs(run, caches, smoke=True)


def test_scientific_stop_keeps_initial_evaluation_and_never_creates_optimizer(tmp_path, monkeypatch):
    run, caches = source_and_caches(tmp_path, smoke=False, rank=1)
    def forbidden(*args, **kwargs): raise AssertionError('scientific stop must precede optimizer')
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    result = ablation.train(run, caches, tmp_path/'stopped')
    assert result['status'] == 'scientific_stop'
    assert result['workflow_complete'] is True and result['training_started'] is False
    assert result['requires_user_action'] is False
    assert not any((tmp_path/'stopped'/name).exists() for name in ablation.ARM_NAMES)
    with np.load(tmp_path/'stopped/final_statistics.npz', allow_pickle=False) as saved:
        assert 'initial_L108_full_DR' in saved.files and 'initial_L54_full_DR' in saved.files
        assert not any(name.startswith('A_') for name in saved.files)


def test_smoke_five_paired_arms_fresh_adam_exact_two_updates_and_preserved_QC(tmp_path):
    run, caches = source_and_caches(tmp_path, rank=3)
    source_bytes = (run/'training_step_12000.pt').read_bytes()
    output = tmp_path/'smoke'
    result = ablation.train(run, caches, output, smoke=True)
    assert result['status'] == 'complete' and result['completed_arms'] == list(ablation.ARM_NAMES)
    source = torch.load(run/'training_step_12000.pt', weights_only=True)
    assert (run/'training_step_12000.pt').read_bytes() == source_bytes
    for arm in ablation.ARM_NAMES:
        folder = output/arm
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['actual_extra_R_updates'] == 2
        assert summary['every_adam_parameter_updates'] == [2]
        checkpoint = torch.load(folder/'training_step_2.pt', weights_only=True)
        initial = torch.load(folder/'training_step_0.pt', weights_only=True)
        assert checkpoint['epoch'] == checkpoint['source_epoch'] == 12000
        assert checkpoint['extra_R_updates'] == 2
        assert state_sha256(checkpoint['model']) == state_sha256(source['model'])
        assert state_sha256(initial['model'], residual=True) == state_sha256(source['model'], residual=True)
        assert not initial['optimizer']['state']
        assert checkpoint['metadata']['labels_used_in_training'] is False
        assert checkpoint['metadata']['Q_C_created_or_called'] is False
        assert len(checkpoint['history']) == 2
    with np.load(output/'final_statistics.npz', allow_pickle=False) as saved:
        assert all(f'{arm}_{size}_full_DR' in saved.files for arm in ablation.ARM_NAMES
                   for size in ('L108','L54'))
    with pytest.raises(FileExistsError): ablation.train(run, caches, output, smoke=True)


def test_fixed_budget_no_early_stopping_no_label_or_theory_selection():
    source = inspect.getsource(ablation.train)
    assert "updates != 6000" in source
    assert "actual_updates = 2 if smoke else updates" in source
    assert 'loss.backward()' in source and 'optimizer.step()' in source
    assert 'load_run(' not in source and 'early_stop' not in source
    assert ablation.ARM_PROTOCOL == {'A':('baseline','old'), 'B':('baseline','white'),
        'C':('enhanced','old'), 'D':('enhanced','white'), 'E':('enhanced','none')}


def test_device_identity_rejects_remapping_even_before_cuda_or_training(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(ValueError, match='映射'):
        ablation.validate_device_identity('cuda:2')
    with pytest.raises(ValueError, match='映射'):
        ablation.validate_device_identity('cpu')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert ablation.validate_device_identity('cpu')['gpu_used'] is False


def test_cuda_uuid_must_match_authorized_physical_index(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    uuid = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index:SimpleNamespace(
        uuid=uuid, name='RTX 3090', total_memory=24*(1<<30)))
    calls = []
    def query(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=uuid+'\n')
    monkeypatch.setattr(ablation.subprocess, 'run', query)
    result = ablation.validate_device_identity('cuda:2')
    assert result['physical_index'] == 2 and result['physical_uuid'] == uuid
    assert '--id=2' in calls[0][0]
    assert calls[0][1]['check'] is True and calls[0][1]['timeout'] == 15
    monkeypatch.setattr(ablation.subprocess, 'run', lambda *a, **k:SimpleNamespace(
        stdout='GPU-01234567-89ab-cdef-0123-456789abcdee\n'))
    with pytest.raises(ValueError, match='不符'):
        ablation.validate_device_identity('cuda:3')
    with pytest.raises(ValueError): ablation.validate_device_identity('cuda:0')


def test_cpu_benchmark_is_synthetic_bounded_and_never_saves_weights(tmp_path, monkeypatch):
    run, _ = source_and_caches(tmp_path)
    original = (run/'training_step_12000.pt').read_bytes()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(ablation, 'BENCHMARK_WARMUP_STEPS', 1)
    monkeypatch.setattr(ablation, 'BENCHMARK_MEASURED_STEPS', 2)
    monkeypatch.setattr(ablation, 'BENCHMARK_DIAGNOSTIC_WARMUP', 0)
    monkeypatch.setattr(ablation, 'BENCHMARK_DIAGNOSTIC_MEASURED', 1)
    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic benchmark must not load caches, Q/C or physical labels')
    monkeypatch.setattr(ablation, 'load_inputs', forbidden)
    monkeypatch.setattr(ablation, 'source_coordinate_normalization', forbidden)
    output = tmp_path/'benchmark'
    result = ablation.benchmark(run, output)
    assert result['status'] == 'complete' and result['scientific_R_updates'] == 0
    assert result['measured_optimizer_steps'] == 6
    assert result['warmup_optimizer_steps'] == 3
    assert result['execution_check_optimizer_steps'] == 9
    assert result['fitted_weights_saved'] is False
    assert result['physical_data_opened'] is False and result['cache_opened'] is False
    assert result['Q_C_created_or_called'] is False and result['labels_read'] is False
    assert result['device'] == 'cpu'
    assert result['direct_shape'] == [16,5,3,3] and result['vin_shape'] == [16,5,2,3,6]
    assert result['synthetic_white_rows_orthonormal_rank'] == 3
    timings = result['timings']
    assert set(timings) == {'old','white','none'}
    assert all(t['every_adam_parameter_steps'] == [3] for t in timings.values())
    assert len({t['initial_R_sha256'] for t in timings.values()}) == 1
    expected = 2*6000*(2*timings['old']['max_step_seconds'] +
        2*timings['white']['max_step_seconds'] + timings['none']['max_step_seconds'])
    assert result['projected_optimizer_seconds_with_margin'] == pytest.approx(expected)
    diagnostic = result['diagnostic_timings']
    periodic = 2*61*diagnostic['five_arm_periodic_evaluation_round']['max_wall_seconds']
    statistics = 2*6*diagnostic['two_size_final_statistics_pair']['max_wall_seconds']
    assert result['projected_periodic_evaluation_seconds_with_margin'] == pytest.approx(periodic)
    assert result['projected_final_statistics_seconds_with_margin'] == pytest.approx(statistics)
    assert result['fixed_IO_initialization_bootstrap_wall_seconds_reserve'] == 300.
    assert result['projected_training_wall_seconds_with_margin'] == pytest.approx(expected+periodic+statistics+300.)
    assert result['projected_training_gpu_seconds_with_margin'] == 0.
    assert result['elapsed_gpu_budget_seconds'] == 0.
    assert set(p.name for p in output.iterdir()) == {'benchmark.json','status.json'}
    assert (run/'training_step_12000.pt').read_bytes() == original
    with pytest.raises(FileExistsError): ablation.benchmark(run, output)


def test_diagnostic_synthetic_panels_match_all_actual_train_and_holdout_axes():
    generator = torch.Generator().manual_seed(91017)
    large = ablation.synthetic_benchmark_panel(depth=2, generator=generator)
    small = ablation.synthetic_benchmark_panel(depth=1, generator=generator)
    assert large['direct'].shape == (492,5,3,3)
    assert int(large['train_mask'].sum()) == 112
    assert int((large['train_mask'] & large['baseline_mask']).sum()) == 16
    assert int((~large['train_mask']).sum()) == 380
    assert small['direct'].shape == (324,5,2,3)
    assert not small['train_mask'].any()
    assert ablation.BENCHMARK_DIAGNOSTIC_WARMUP == 1
    assert ablation.BENCHMARK_DIAGNOSTIC_MEASURED == 3
    assert ablation.BENCHMARK_WARMUP_STEPS == 10
    assert ablation.BENCHMARK_MEASURED_STEPS == 100


def test_benchmark_cli_cannot_open_physical_cache_or_use_smoke(tmp_path, monkeypatch):
    args = ['--source-run',str(tmp_path/'source'),'--output',str(tmp_path/'check'),'--benchmark']
    with pytest.raises(ValueError, match='不允许'):
        ablation.main(args+['--cache-root',str(tmp_path/'physical')])
    with pytest.raises(ValueError, match='不允许'):
        ablation.main(args+['--smoke'])
    calls = []
    monkeypatch.setattr(ablation, 'benchmark', lambda **kwargs:calls.append(kwargs))
    assert ablation.main(args) == 0
    assert 'cache_root' not in calls[0] and 'smoke' not in calls[0]
