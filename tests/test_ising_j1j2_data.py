"""CPU-only tiny generated/fake shards; no historical data or GPU activity."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import scripts.ising_j1j2_data as pipeline
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.ising_j1j2_pt import (
    configuration_energy, energy_components, sample_ladder, seed_for_group,
    series_diagnostics,
)


def small_plan():
    return dict(physical=dict(size=36, rhos=[0., .7, .3], train_rhos=[0., .7],
        heldout_rhos=[.3], temperatures=[.5, 1., 2., 4.],
        saved_temperature_indices=[0, 2], groups=12, train_groups=8,
        samples=8, burn_in=4, spacing=2, master_seed=20261026,
        initials=['hot', 'ferro', 'stripe_x', 'stripe_y']),
        pilot=dict(rhos=[0., .7], burn_in=4, spacing=2, samples=8,
            min_exchange=.01, min_ess=1., hot_cold_abs_threshold=3., hot_cold_z_threshold=4.))


def fake_sampler(size, rho, temperatures, seed, burn_in, samples, spacing,
                 initial='hot', **kwargs):
    rng = np.random.default_rng(seed)
    x = 2 * rng.integers(0, 2, (len(temperatures), samples, size, size), dtype=np.int8) - 1
    nn, diagonal = energy_components(x)
    m = x.mean((-2, -1))
    sx = (x * (1 - 2 * (np.arange(size) % 2))[None, None, :, None]).mean((-2, -1))
    sy = (x * (1 - 2 * (np.arange(size) % 2))[None, None, None, :]).mean((-2, -1))
    energy = nn + rho * diagonal
    diagnostics = [dict(temperature=float(temp),
        energy_per_site=series_diagnostics(energy[t] / size ** 2),
        magnetization_squared=series_diagnostics(m[t] ** 2),
        stripe_x_squared=series_diagnostics(sx[t] ** 2),
        stripe_y_squared=series_diagnostics(sy[t] ** 2)) for t, temp in enumerate(temperatures)]
    return dict(size=size, rho=rho, temperatures=np.array(temperatures), seed=seed,
        initial=initial, configs=x, energies=energy, energy_nn=nn,
        energy_diagonal=diagonal, magnetization=m, stripe_x=sx, stripe_y=sy,
        swap_attempts=np.full(len(temperatures) - 1, 10),
        swap_accepts=np.full(len(temperatures) - 1, 5),
        local_attempts=np.full(len(temperatures), 10), local_accepts=np.full(len(temperatures), 5),
        temperature_visits=np.ones((len(temperatures), len(temperatures)), dtype=np.int64),
        round_trips=np.ones(len(temperatures), dtype=np.int64), diagnostics=diagnostics,
        max_component_tracking_drift=0, max_energy_tracking_drift=0.,
        visited_sites=len(temperatures) * (burn_in + samples * spacing) * size ** 2,
        total_sweeps=len(temperatures) * (burn_in + samples * spacing),
        sweeps_per_temperature=burn_in + samples * spacing,
        burn_in=burn_in, samples=samples, spacing=spacing, numba_used=False,
        trace=dict(sweeps=np.arange(samples), includes_burn_in=True,
                   energy_per_site=energy / size ** 2), cpu_seconds=.01, wall_seconds=.02)


def test_real_tiny_group_retains_raw_and_applies_only_exact_symmetries(tmp_path, monkeypatch):
    plan = small_plan()
    plan['physical']['size'] = 6
    monkeypatch.setattr(pipeline, 'sample_ladder', lambda *a, **kw: sample_ladder(*a, use_numba=False, **kw))
    result = pipeline.sample_group(plan, tmp_path, 0, 0)
    record = json.loads((tmp_path / 'data/shards/r00_g00.sidecar.json').read_text())
    payload = torch.load(result['data_path'], weights_only=True)
    assert payload['configurations'].shape == (2, 8, 6, 6)
    assert payload['configurations'].dtype == torch.int8
    assert payload['raw']['energies'].shape == (4, 8)
    assert payload['raw_trace']['includes_burn_in']
    seed = seed_for_group(plan['physical']['master_seed'], 6, 0., 'production', 0)
    raw = sample_ladder(6, 0., plan['physical']['temperatures'], seed, 4, 8, 2, use_numba=False)
    for slot, tid in enumerate(plan['physical']['saved_temperature_indices']):
        assert np.allclose(configuration_energy(payload['configurations'][slot].numpy(), 0.), raw['energies'][tid])
        for sample in range(8):
            expected = payload['symmetry_signs'][slot, sample].item() * np.rot90(raw['configs'][tid, sample],
                payload['symmetry_rotations'][slot, sample].item())
            np.testing.assert_array_equal(payload['configurations'][slot, sample].numpy(), expected)
    assert record['raw_diagnostics_before_symmetry_augmentation']
    assert record['symmetry_augmentation']['independent_samples_added'] == 0
    assert not record['symmetry_augmentation']['fixes_thermalization']
    assert record['seed'] == seed and record['initial'] == 'hot'
    with pytest.raises(FileExistsError):
        pipeline.sample_group(plan, tmp_path, 0, 0)


def generated_shards(tmp_path, monkeypatch, plan=None):
    plan = small_plan() if plan is None else plan
    monkeypatch.setattr(pipeline, 'sample_ladder', fake_sampler)
    for ri in range(len(plan['physical']['rhos'])):
        for group in range(12):
            pipeline.sample_group(plan, tmp_path, ri, group)
    from experiments.prepare_ising_j1j2 import make_config
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'dg5_seed91.json').write_text(json.dumps(make_config(
        tmp_path / 'data/training.pt', updates=2, threads=1, sample_batch=8,
        system_batch=2, max_rg_steps=1, evaluation_systems=2)))
    return plan


def test_assembly_preserves_whole_ladder_split_and_keeps_labels_external(tmp_path, monkeypatch):
    plan = generated_shards(tmp_path, monkeypatch)
    result = pipeline.assemble(plan, tmp_path)
    train, test = load_ensembles(result['data_path'])
    assert train.configurations.shape == (32, 8, 36, 36, 1)
    assert test.configurations.shape == (40, 8, 36, 36, 1)
    assert train.couplings is None and test.couplings is None
    assert train.group_ids.dtype == torch.int64
    train_repeats = {(r, g) for r, t, g in train.group_ids.tolist()}
    test_repeats = {(r, g) for r, t, g in test.group_ids.tolist()}
    assert train_repeats.isdisjoint(test_repeats)
    meta = json.loads((tmp_path / 'data/metadata.json').read_text())
    assert meta['data_sha256'] == pipeline._sha(result['data_path'])
    assert {r['partition'] for r in meta['train']} == {'train'}
    assert {r['partition'] for r in meta['test']} == {'same_rho_new_group', 'heldout_rho'}
    assert not meta['physical_labels_in_training_cache']
    assert meta['temperature_grid']['training_coupling_grid'] == 'historical_temperature_subset'
    assert result['optimizer_updates'] == 0 and not result['model_initialized']
    with pytest.raises(FileExistsError):
        pipeline.assemble(plan, tmp_path)


def test_uniform_k_assembly_saves_only_retained_conditions_and_keeps_k_external(tmp_path, monkeypatch):
    from scripts.run_ising_j1j2 import coupling_grid
    plan = small_plan()
    plan['physical'].update(coupling_grid())
    plan = generated_shards(tmp_path, monkeypatch, plan)
    result = pipeline.assemble(plan, tmp_path)
    payload = torch.load(result['data_path'], weights_only=True)
    metadata = json.loads((tmp_path / 'data/metadata.json').read_text())
    grid = metadata['temperature_grid']
    assert grid['training_coupling_grid'] == 'uniform_K'
    assert (grid['total_sampling_temperature_count'], grid['retained_condition_count'],
            grid['auxiliary_temperature_count']) == (83, 12, 71)
    assert grid['auxiliary_temperatures_are_sampling_only']
    assert grid['auxiliary_rows_in_training_cache'] == 0
    assert grid['physical_labels_external_only']
    assert grid['retained_K_values'] == plan['physical']['retained_K_values']
    saved = set(plan['physical']['saved_temperature_indices'])
    auxiliary = set(plan['physical']['auxiliary_temperature_indices'])
    assert result['train_ensembles'] == 2 * 12 * 8
    assert result['test_ensembles'] == 2 * 12 * 4 + 12 * 12
    for split in ('train', 'test'):
        assert set(payload[split]) == {'configurations', 'couplings', 'group_ids'}
        assert payload[split]['couplings'] is None
        ids = {row[1] for row in payload[split]['group_ids'].tolist()}
        assert ids == saved and ids.isdisjoint(auxiliary)
        for row in metadata[split]:
            assert row['temperature_index'] in saved
            assert row['K'] == 1 / row['temperature']
        np.testing.assert_allclose(sorted({r['K'] for r in metadata[split]}),
                                   np.linspace(.25, 2.5, 12), rtol=2e-16, atol=0)
    receipt = json.loads((tmp_path / 'data/shards/r00_g00.sidecar.json').read_text())
    assert receipt['temperature_grid'] == grid
    assert len(receipt['saved_K_values']) == 12
    assert receipt['configuration_shape'][0] == 12


@pytest.mark.parametrize('bad', ['missing', 'mode', 'nonuniform_k', 'retained_t',
                                'retained_slot', 'auxiliary_overlap', 'auxiliary_missing'])
def test_uniform_grid_contract_fails_before_output_or_sampling(tmp_path, monkeypatch, bad):
    from scripts.run_ising_j1j2 import coupling_grid
    plan = small_plan()
    p = plan['physical']
    p.update(coupling_grid())
    if bad == 'missing':
        del p['retained_K_values']
    elif bad == 'mode':
        p['training_coupling_grid'] = 'uniform_T'
    elif bad == 'nonuniform_k':
        p['retained_K_values'][4] += .001
    elif bad == 'retained_t':
        p['retained_temperature_values'][4] += .001
    elif bad == 'retained_slot':
        p['saved_temperature_indices'][4] += 1
    elif bad == 'auxiliary_overlap':
        p['auxiliary_temperature_indices'][0] = p['saved_temperature_indices'][0]
    elif bad == 'auxiliary_missing':
        p['auxiliary_temperature_indices'].pop()
    monkeypatch.setattr(pipeline, 'sample_ladder', lambda *a, **kw: pytest.fail('invalid grid sampled'))
    with pytest.raises(ValueError):
        pipeline.sample_group(plan, tmp_path, 0, 0)
    assert not (tmp_path / 'data').exists()


def test_production_quality_reads_only_json_and_scientific_stop_is_not_failure(tmp_path, monkeypatch):
    plan = generated_shards(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('quality gate opened a PT/configuration payload'))
    result = pipeline.quality_gate(plan, tmp_path)
    assert result['status'] == 'complete' and result['scientific_gate_passed']
    assert result['physical_groups'] == 36 and not result['equilibrium_certified']
    assert all(r['uncertainty'] == 'whole_repeat_means' for r in result['hot_cold'])
    with pytest.raises(FileExistsError):
        pipeline.quality_gate(plan, tmp_path)


@pytest.mark.parametrize('bad', ['shard', 'sidecar', 'seed', 'burn_in', 'initial', 'incomplete'])
def test_assembly_fails_closed_on_tampering_even_resealed_semantics(tmp_path, monkeypatch, bad):
    plan = generated_shards(tmp_path, monkeypatch)
    prefix = tmp_path / 'data/shards/r00_g00'
    sidecar, status = prefix.with_suffix('.sidecar.json'), prefix.with_suffix('.status.json')
    receipt, state = json.loads(sidecar.read_text()), json.loads(status.read_text())
    if bad == 'shard':
        with prefix.with_suffix('.pt').open('ab') as stream:
            stream.write(b'tamper')
    elif bad == 'sidecar':
        with sidecar.open('a') as stream:
            stream.write(' ')
    elif bad == 'incomplete':
        state['status'] = 'running'
        status.write_text(json.dumps(state))
    else:
        receipt[bad] = receipt[bad] + 1 if bad != 'initial' else 'stripe_x'
        sidecar.write_text(json.dumps(receipt))
        state['sidecar_sha256'] = pipeline._sha(sidecar)
        status.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        pipeline.assemble(plan, tmp_path)
    assert not (tmp_path / 'data/training.pt').exists()


def test_pilot_gate_and_benchmark_use_distinct_namespaces_and_no_models(tmp_path, monkeypatch):
    plan = small_plan()
    calls = []
    def capture(*a, **kw):
        calls.append((a, kw))
        return fake_sampler(*a, **kw)
    monkeypatch.setattr(pipeline, 'sample_ladder', capture)
    before = torch.get_rng_state().clone()
    bench = pipeline.benchmark(plan, tmp_path)
    pilot = pipeline.pilot(plan, tmp_path)
    assert torch.equal(before, torch.get_rng_state())
    assert bench['status'] == 'complete' and not bench['scientific_result']
    assert bench['conservative_production_cpu_seconds'] == 2 * bench['projected_production_cpu_seconds']
    assert bench['warmup_cpu_seconds'] > 0
    assert pilot['status'] == 'complete' and pilot['ladders'] == 5
    assert not pilot['equilibrium_certified'] and not pilot['automatic_resampling']
    pilot_seeds = {a[3] for a, kw in calls[3:]}
    production_seed = seed_for_group(20261026, 36, 0., 'production', 0)
    assert production_seed not in pilot_seeds
    with pytest.raises(FileExistsError):
        pipeline.benchmark(plan, tmp_path)
    with pytest.raises(FileExistsError):
        pipeline.pilot(plan, tmp_path)


def test_pilot_nonconstant_ess_and_exchange_fail_as_scientific_stop(tmp_path, monkeypatch):
    plan = small_plan()
    def poor(*a, **kw):
        result = fake_sampler(*a, **kw)
        result['swap_accepts'][:] = 0
        return result
    monkeypatch.setattr(pipeline, 'sample_ladder', poor)
    plan['pilot']['min_ess'] = 16
    result = pipeline.pilot(plan, tmp_path)
    assert result['status'] == 'scientific_stop'
    assert result['workflow_complete'] and not result['scientific_gate_passed']
    assert result['requires_user_action']
    assert {'low_exchange', 'low_ess'} <= {r['kind'] for r in result['failures']}


def test_hot_cold_requires_both_absolute_and_uncertainty_thresholds():
    criteria = small_plan()['pilot']
    criteria['hot_cold_abs_threshold'] = .05
    records = []
    for initial, means in [('hot', [.5, .51, .49]), ('ferro', [.1, .11, .09])]:
        for index, mean in enumerate(means):
            diag = dict(mean=mean, variance=.0001, ess=128)
            records.append(dict(rho=0., initial=initial, group=index,
                temperatures=[1.], diagnostics=[dict(energy_per_site=diag,
                    rotation_invariant_order=deepcopy(diag))]))
    checks, failed = pipeline._hot_cold(records, criteria, production=True)
    assert len(failed) == 2 and all(r['failed'] for r in checks)
    changed = deepcopy(records)
    for record in changed:
        if record['initial'] == 'ferro':
            for d in record['diagnostics'][0].values():
                d['mean'] += .39
    _, failed = pipeline._hot_cold(changed, criteria, production=True)
    assert not failed
