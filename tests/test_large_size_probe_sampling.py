"""Synthetic/mocked execution tests; never launch a physical L1080 chain."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ising_large_size_probe_sampling as probe


def fake_kernel_module(monkeypatch):
    from scripts import ising_finite_perturbation_data as frozen
    calls = []

    def run_chain(k, delta, size, draws, equil, gap, seed, cold=False):
        calls.append(dict(k=k, delta=delta.copy(), size=size, draws=draws,
                          equil=equil, gap=gap, seed=seed, cold=cold))
        rng = np.random.default_rng(seed)
        x = rng.choice(np.array([-1, 1], dtype=np.int8), size=(draws, size, size))
        proposals = equil + draws * gap
        return x, dict(cluster_proposals=proposals, local_sweeps=proposals // 32,
                       cluster_accepted=proposals, local_spin_proposals=(proposals // 32) * size * size)

    module = SimpleNamespace(__file__=probe.__file__, run_chain=run_chain,
                             _series_stats=frozen._series_stats,
                             _hot_cold_comparison=frozen._hot_cold_comparison,
                             quality_report=frozen.quality_report)
    monkeypatch.setattr(probe, '_kernel_module', lambda: module)
    return calls


def test_roll_observables_match_frozen_monomial_oracle():
    from scripts import ising_finite_perturbation_data as frozen
    rng = np.random.default_rng(321)
    x = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, 6, 6))
    actual = probe.raw_observable_series(x)
    expected = frozen.operator_values_numpy(x) / 36.
    for index, name in enumerate(frozen.EVEN_OPERATOR_NAMES):
        np.testing.assert_allclose(actual[name], expected[:, index], atol=1e-15)
    m = x.mean(axis=(1, 2))
    np.testing.assert_allclose(actual['magnetization'], m)
    np.testing.assert_allclose(actual['absolute_magnetization'], abs(m))
    np.testing.assert_allclose(actual['magnetization_squared'], m * m)


@pytest.mark.parametrize('bad', [np.zeros((1, 4, 4), dtype=np.int8),
                                np.ones((1, 4, 4), dtype=np.float64),
                                np.ones((0, 4, 4), dtype=np.int8),
                                np.ones((1, 3, 4), dtype=np.int8)])
def test_invalid_spins_rejected(bad):
    with pytest.raises(ValueError):
        probe.raw_observable_series(bad)


def test_namespaces_disjoint_and_deterministic():
    assert probe.validate_seed_namespace(20261018, 1080) == 24
    values = [probe.namespaced_seed(20261018, 1080, purpose, condition, replica)
              for purpose in probe.PURPOSES for condition in range(3) for replica in range(2)]
    assert len(set(values)) == len(values)
    assert values[0] == probe.namespaced_seed(20261018, 1080, 'benchmark', 0, 0)
    assert values[0] != probe.namespaced_seed(20261018, 108, 'benchmark', 0, 0)
    from scripts.ising_finite_perturbation_data import namespaced_seed as previous
    old = {previous(20261018, 1080, purpose, condition, replica)
           for purpose in ('production', 'benchmark_hot', 'benchmark_cold', 'global_sign')
           for condition in range(3) for replica in range(2)}
    assert not old.intersection(values)


@pytest.mark.parametrize('args', [(True, 1080, 'production', 0, 0),
                                  (1, 1080, 'unknown', 0, 0),
                                  (1, 1080, 'production', 3, 0),
                                  (1, 1080, 'production', 0, 2)])
def test_invalid_seed_namespace(args):
    with pytest.raises(ValueError):
        probe.namespaced_seed(*args)


def test_benchmark_fixed_cases_gate_and_no_reuse(tmp_path, monkeypatch):
    calls = fake_kernel_module(monkeypatch)
    receipt = probe.benchmark(tmp_path, size=27)
    assert receipt['status'] == 'complete'
    assert receipt['non_scientific'] and not receipt['scientific_samples_reused']
    assert len(calls) == 7
    assert {call['k'] for call in calls} == {.42, .44, .46}
    assert (calls[0]['draws'], calls[0]['equil'], calls[0]['gap']) == (1, 0, 1)
    assert all((call['draws'], call['equil'], call['gap']) == (1, 64, 128) for call in calls[1:])
    assert receipt['setup_warmup']['kernel']['cluster_proposals'] == 1
    assert all(not call['delta'].any() for call in calls)
    gate = receipt['budget_gate']
    assert len(gate['conditions']) == 3
    for row in gate['conditions']:
        assert row['production_cluster_proposals'] == 24576
        assert row['projected_cpu_seconds'] == pytest.approx(
            row['worst_cpu_seconds_per_cluster_proposal_including_sweeps'] * 24576 * 2)
    assert gate['projected_total_cpu_seconds'] == pytest.approx(
        gate['benchmark_actual_cpu_seconds'] + sum(row['projected_cpu_seconds'] for row in gate['conditions'])
        + gate['projected_sample_process_setup_cpu_seconds'] + 120)
    assert not list(tmp_path.rglob('*.npy'))
    assert not list(tmp_path.rglob('*.pt'))
    with pytest.raises(FileExistsError):
        probe.benchmark(tmp_path, size=27)
    assert len(calls) == 7


def test_budget_hold_never_starts_production(tmp_path, monkeypatch):
    calls = fake_kernel_module(monkeypatch)
    monkeypatch.setattr(probe, 'CPU_BUDGET_SECONDS', 0.)
    receipt = probe.benchmark(tmp_path, size=27)
    assert receipt['status'] == 'budget_hold'
    assert not receipt['budget_gate']['passed']
    with pytest.raises(ValueError, match='passing'):
        probe.sample(tmp_path, condition_index=0, size=27)
    assert len(calls) == 7
    assert not (tmp_path / 'data').exists()


def test_two_production_chains_fixed_budget_signs_and_raw_diagnostics(tmp_path, monkeypatch):
    calls = fake_kernel_module(monkeypatch)
    probe.benchmark(tmp_path, size=27)
    receipt = probe.sample(tmp_path, condition_index=0, size=27)
    assert receipt['status'] == 'complete'
    assert len(calls) == 9
    production = calls[7:]
    assert all((call['draws'], call['equil'], call['gap']) == (64, 4096, 128) for call in production)
    assert [call['cold'] for call in production] == [False, True]
    assert len({call['seed'] for call in calls}) == len(calls)
    assert receipt['sampling_budget']['independent_pilot_chains'] == 0
    assert receipt['training_updates'] == receipt['MI_estimates'] == 0
    path = tmp_path / 'data/c000/spins.npy'
    assert probe.file_sha256(path) == receipt['spins_sha256']
    data = np.load(path, allow_pickle=False)
    assert data.shape == (2, 64, 27, 27) and data.dtype == np.int8
    for replica, chain in enumerate(receipt['chains']):
        rng = np.random.default_rng(chain['seed'])
        raw = rng.choice(np.array([-1, 1], dtype=np.int8), size=(64, 27, 27))
        expected = probe.raw_observable_series(raw)
        for name, series in expected.items():
            np.testing.assert_allclose(chain['raw_observable_series'][name], series)
        np.testing.assert_array_equal(data[replica], raw * np.asarray(chain['global_signs'])[:, None, None])
    assert receipt['raw_diagnostics_before_global_Z2']
    quality = receipt['quality']
    assert quality['independent_pilot_chains'] == 0
    assert not any('pilot' in warning['kind'] for warning in quality['warnings'])
    assert 'pilot_ranges' not in quality
    with pytest.raises(FileExistsError):
        probe.sample(tmp_path, condition_index=0, size=27)
    assert len(calls) == 9


def test_preflight_required_matching_and_sequential(tmp_path, monkeypatch):
    fake_kernel_module(monkeypatch)
    with pytest.raises(FileNotFoundError):
        probe.sample(tmp_path, condition_index=0, size=27)
    assert not (tmp_path / 'data').exists()
    probe.benchmark(tmp_path, size=27)
    with pytest.raises(ValueError):
        probe.sample(tmp_path, condition_index=0, size=54)
    with pytest.raises(FileNotFoundError):
        probe.sample(tmp_path, condition_index=1, size=27)
    probe.sample(tmp_path, condition_index=0, size=27)
    assert probe.sample(tmp_path, condition_index=1, size=27)['status'] == 'complete'


def test_tampered_preflight_and_previous_stage_rejected(tmp_path, monkeypatch):
    fake_kernel_module(monkeypatch)
    probe.benchmark(tmp_path, size=27)
    path = tmp_path / 'benchmark/receipt.json'
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='fingerprint'):
        probe.sample(tmp_path, condition_index=0, size=27)
    path.write_bytes(original)
    probe.sample(tmp_path, condition_index=0, size=27)
    previous = tmp_path / 'data/c000/sidecar.json'
    previous.write_bytes(previous.read_bytes() + b' ')
    with pytest.raises(ValueError, match='fingerprint'):
        probe.sample(tmp_path, condition_index=1, size=27)


def test_failure_records_error_without_retry(tmp_path, monkeypatch):
    calls = fake_kernel_module(monkeypatch)
    probe.benchmark(tmp_path, size=27)
    module = probe._kernel_module()

    def fail(*args, **kwargs):
        raise RuntimeError('mock transition failure')

    module.run_chain = fail
    with pytest.raises(RuntimeError, match='mock transition'):
        probe.sample(tmp_path, condition_index=0, size=27)
    status = json.loads((tmp_path / 'data/c000/status.json').read_text())
    assert status['status'] == 'failed' and not status['workflow_complete']
    assert status['error']['message'] == 'mock transition failure'
    assert not (tmp_path / 'data/c000/spins.npy').exists()
    with pytest.raises(FileExistsError):
        probe.sample(tmp_path, condition_index=0, size=27)
    assert len(calls) == 7


def test_cli_exposes_no_budget_reduction_switch():
    with pytest.raises(SystemExit):
        probe.main(['--root', '/unused', 'sample', '--condition-index', '0', '--draws', '2'])
