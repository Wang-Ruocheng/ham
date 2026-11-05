"""Tiny mocked execution checks, never a scientific or equilibrium result."""
import json

import numpy as np
import pytest
import torch

import scripts.ising_j1j2_unthermalized_data as pipeline
from recurrent_rg.data.ising_j1j2_pt import energy_components, seed_for_group, series_diagnostics
from scripts.run_ising_j1j2 import coupling_grid


def diagnostic_plan():
    return dict(diagnostic=dict(diagnostic_only=True, quality_override_authorized=True,
        equilibrium_certified=False, user_authorization='User explicitly continued diagnostic sampling/training'),
        physical=dict(size=108, rhos=[0., .35, .7, 1.], train_rhos=[0., .35, .7, 1.],
            heldout_rhos=[], groups=4, train_groups=2, group_ids=[0, 1, 8, 9],
            train_group_ids=[0, 1], test_group_ids=[8, 9], samples=128,
            burn_in=4096, spacing=32, master_seed=20261026, **coupling_grid()),
        pilot=dict(min_exchange=.01, min_ess=16,
            hot_cold_abs_threshold=.05, hot_cold_z_threshold=4.))


def tiny_plan(monkeypatch):
    """Only execution tests substitute tiny geometry, never the public validator."""
    plan = diagnostic_plan()
    pipeline.validate_plan(plan)
    plan['physical'].update(size=12, samples=8, burn_in=2, spacing=1)
    monkeypatch.setattr(pipeline, 'validate_plan', lambda value: value['physical'])
    return plan


def fake_sampler(size, rho, temperatures, seed, burn_in, samples, spacing, initial='hot'):
    rng = np.random.default_rng(seed)
    if initial == 'hot':
        x = 2 * rng.integers(0, 2, (len(temperatures), samples, size, size), dtype=np.int8) - 1
    else:
        x = np.ones((len(temperatures), samples, size, size), dtype=np.int8)
        if initial == 'stripe_x':
            x *= (1 - 2 * (np.arange(size) % 2))[None, None, :, None].astype(np.int8)
    nn, diagonal = energy_components(x)
    m = x.mean((-2, -1))
    sx = (x * (1 - 2 * (np.arange(size) % 2))[None, None, :, None]).mean((-2, -1))
    sy = (x * (1 - 2 * (np.arange(size) % 2))[None, None, None, :]).mean((-2, -1))
    energy = nn + rho * diagonal
    return dict(size=size, rho=rho, temperatures=np.array(temperatures), seed=seed,
        initial=initial, configs=x, energies=energy, energy_nn=nn, energy_diagonal=diagonal,
        magnetization=m, stripe_x=sx, stripe_y=sy, burn_in=burn_in, samples=samples,
        spacing=spacing, total_sweeps=len(temperatures) * (burn_in + samples * spacing),
        sweeps_per_temperature=burn_in + samples * spacing,
        visited_sites=len(temperatures) * (burn_in + samples * spacing) * size ** 2,
        cpu_seconds=.01, wall_seconds=.02, numba_used=False,
        max_component_tracking_drift=0, max_energy_tracking_drift=0.,
        swap_attempts=np.full(len(temperatures) - 1, 10),
        swap_accepts=np.zeros(len(temperatures) - 1, dtype=np.int64),
        local_attempts=np.full(len(temperatures), 10), local_accepts=np.full(len(temperatures), 5),
        temperature_visits=np.ones((len(temperatures), len(temperatures)), dtype=np.int64),
        round_trips=np.zeros(len(temperatures), dtype=np.int64),
        diagnostics=[dict(temperature=float(temp),
            energy_per_site=series_diagnostics(energy[t] / size ** 2))
            for t, temp in enumerate(temperatures)],
        trace=dict(sweeps=np.arange(samples), includes_burn_in=True,
                   energy_per_site=energy / size ** 2))


def generate(tmp_path, monkeypatch):
    plan = tiny_plan(monkeypatch)
    monkeypatch.setattr(pipeline, 'sample_ladder', fake_sampler)
    for ri in range(4):
        for group in [0, 1, 8, 9]:
            pipeline.sample_group(plan, tmp_path, ri, group)
    from experiments.prepare_ising_j1j2 import make_config
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'dg5_seed91.json').write_text(json.dumps(make_config(tmp_path / 'data/training.pt',
        updates=2, threads=1, sample_batch=8, system_batch=2, max_rg_steps=1, evaluation_systems=2)))
    return plan


def test_authorized_fixed_contract_and_distinct_seeds():
    plan = diagnostic_plan()
    assert pipeline.validate_plan(plan) is plan['physical']
    p = plan['physical']
    seeds = [seed_for_group(p['master_seed'], 108, rho, pipeline.PURPOSE, group)
             for rho in p['rhos'] for group in p['group_ids']]
    assert len(set(seeds)) == 16
    assert seeds[0] != seed_for_group(p['master_seed'], 108, 0., 'production', 0)
    assert seeds[0] != seed_for_group(p['master_seed'], 108, 0., 'pilot', 0)
    assert [pipeline.initial_for_group(.35, group) for group in p['group_ids']] == ['hot', 'ferro', 'hot', 'ferro']
    assert [pipeline.initial_for_group(.7, group) for group in p['group_ids']] == ['hot', 'stripe_x', 'hot', 'stripe_x']


@pytest.mark.parametrize('field,value', [
    ('diagnostic_only', False), ('quality_override_authorized', False),
    ('equilibrium_certified', True), ('user_authorization', ''),
])
def test_missing_authorization_refused_before_output(tmp_path, monkeypatch, field, value):
    plan = diagnostic_plan()
    plan['diagnostic'][field] = value
    monkeypatch.setattr(pipeline, 'sample_ladder', lambda *a, **kw: pytest.fail('must not sample'))
    with pytest.raises(ValueError, match='Explicit user-authorized'):
        pipeline.sample_group(plan, tmp_path, 0, 0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field,value', [
    ('size', 36), ('rhos', [0., .35, .7]), ('groups', 12),
    ('group_ids', [0, 1, 2, 3]), ('train_group_ids', [0, 8]),
    ('samples', 64), ('burn_in', 16384), ('spacing', 128), ('master_seed', 91),
])
def test_physical_expansion_or_plan_change_refused(field, value):
    plan = diagnostic_plan()
    plan['physical'][field] = value
    with pytest.raises(ValueError):
        pipeline.validate_plan(plan)


def test_grid_and_quality_threshold_changes_refused():
    plan = diagnostic_plan()
    plan['physical']['temperatures'][3] += 1e-7
    with pytest.raises(ValueError):
        pipeline.validate_plan(plan)
    plan = diagnostic_plan()
    plan['pilot']['min_ess'] = 1
    with pytest.raises(ValueError):
        pipeline.validate_plan(plan)


def test_shard_retains_raw_failed_quality_and_exact_symmetry(tmp_path, monkeypatch):
    plan = tiny_plan(monkeypatch)
    monkeypatch.setattr(pipeline, 'sample_ladder', fake_sampler)
    result = pipeline.sample_group(plan, tmp_path, 0, 0)
    record = json.loads((tmp_path / 'data/shards/r00_g00.sidecar.json').read_text())
    payload = torch.load(result['data_path'], weights_only=True)
    assert record['quality_failures']
    assert record['quality_override_authorized'] and record['diagnostic_only']
    assert record['equilibrium_certified'] is False
    assert record['raw_diagnostics_before_symmetry_augmentation']
    assert record['symmetry_augmentation']['independent_samples_added'] == 0
    assert not record['symmetry_augmentation']['fixes_thermalization']
    raw = fake_sampler(12, 0., plan['physical']['temperatures'], record['seed'], 2, 8, 1)
    for slot, temperature in enumerate(plan['physical']['saved_temperature_indices']):
        for sample in range(8):
            expected = payload['symmetry_signs'][slot, sample].item() * np.rot90(
                raw['configs'][temperature, sample], payload['symmetry_rotations'][slot, sample].item())
            np.testing.assert_array_equal(payload['configurations'][slot, sample].numpy(), expected)
    assert payload['raw']['energies'].shape == (83, 8)
    with pytest.raises(FileExistsError):
        pipeline.sample_group(plan, tmp_path, 0, 0)


@pytest.mark.parametrize('problem', ['energy', 'component', 'binary', 'identity'])
def test_physical_integrity_failures_remain_hard(tmp_path, monkeypatch, problem):
    plan = tiny_plan(monkeypatch)
    def corrupt(*args, **kwargs):
        result = fake_sampler(*args, **kwargs)
        if problem == 'energy':
            result['max_energy_tracking_drift'] = .1
        elif problem == 'component':
            result['max_component_tracking_drift'] = 1
        elif problem == 'binary':
            result['configs'][0, 0, 0, 0] = 0
        else:
            result['seed'] += 1
        return result
    monkeypatch.setattr(pipeline, 'sample_ladder', corrupt)
    with pytest.raises(ValueError):
        pipeline.sample_group(plan, tmp_path, 0, 0)
    assert not (tmp_path / 'data/shards/r00_g00.pt').exists()


def test_assembly_allows_quality_failure_but_keeps_split_and_labels(tmp_path, monkeypatch):
    plan = generate(tmp_path, monkeypatch)
    result = pipeline.assemble(plan, tmp_path)
    assert result['train_ensembles'] == result['test_ensembles'] == 96
    assert result['quality_failure_count'] > 0
    assert result['label_boundary_validation']['status'] == 'complete'
    assert result['report_boundary_validation']['whole_PT_group_split_verified']
    payload = torch.load(result['data_path'], weights_only=True)
    for split, groups in [('train', {0, 1}), ('test', {8, 9})]:
        assert set(payload[split]) == {'configurations', 'couplings', 'group_ids'}
        assert payload[split]['couplings'] is None
        assert payload[split]['configurations'].shape == (96, 8, 12, 12, 1)
        assert set(payload[split]['group_ids'][:, 2].tolist()) == groups
        assert set(payload[split]['group_ids'][:, 1].tolist()) == set(plan['physical']['saved_temperature_indices'])
    meta = json.loads((tmp_path / 'data/metadata.json').read_text())
    quality = json.loads((tmp_path / 'data/quality_warning.json').read_text())
    assert meta['quality_passed'] is False and quality['scientific_gate_passed'] is False
    assert quality['status'] == 'complete' and not quality['quality_failure_blocks_training']
    assert quality['physical_groups'] == 16 and quality['configurations'] == 16 * 12 * 8
    assert 'low_exchange' in meta['failed_criteria']
    assert 'hot_cold_disagreement' in meta['failed_criteria']
    assert meta['quality_warning_sha256'] == pipeline._sha(tmp_path / 'data/quality_warning.json')
    assert not meta['equilibrium_certified'] and meta['diagnostic_only']
    assert {row['partition'] for row in meta['test']} == {'same_rho_new_group'}
    assert all(row['raw_initial'] in ('hot', 'ferro', 'stripe_x') for row in meta['test'])
    with pytest.raises(FileExistsError):
        pipeline.assemble(plan, tmp_path)


def test_fingerprint_corruption_is_not_quality_override(tmp_path, monkeypatch):
    plan = generate(tmp_path, monkeypatch)
    shard = tmp_path / 'data/shards/r00_g00.pt'
    with shard.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='fingerprint'):
        pipeline.assemble(plan, tmp_path)
    assert not (tmp_path / 'data/training.pt').exists()


def test_sidecar_fingerprint_corruption_is_hard(tmp_path, monkeypatch):
    plan = generate(tmp_path, monkeypatch)
    sidecar = tmp_path / 'data/shards/r00_g00.sidecar.json'
    row = json.loads(sidecar.read_text())
    row['quality_failures'] = []
    sidecar.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='receipt'):
        pipeline.assemble(plan, tmp_path)


def test_existing_quality_warning_cannot_hide_failures(tmp_path, monkeypatch):
    plan = generate(tmp_path, monkeypatch)
    result = pipeline.quality_warning(plan, tmp_path)
    assert result['failures'] and not result['quality_passed']
    path = tmp_path / 'data/quality_warning.json'
    result.update(failures=[], quality_passed=True, scientific_gate_passed=True)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='quality warning'):
        pipeline.assemble(plan, tmp_path)
    assert not (tmp_path / 'data/training.pt').exists()


def test_unsupported_group_refused_before_output(tmp_path, monkeypatch):
    plan = diagnostic_plan()
    monkeypatch.setattr(pipeline, 'sample_ladder', lambda *a, **kw: pytest.fail('must not sample'))
    with pytest.raises(ValueError):
        pipeline.sample_group(plan, tmp_path, 0, 2)
    assert not list(tmp_path.iterdir())
