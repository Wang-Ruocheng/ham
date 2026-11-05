"""Finite-coupling sampler tests; no source model or historical data is opened."""
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from scripts import ising_finite_perturbation_data as data
from scripts.diagnose_ising_mcrg import operator_values as reference_operators


def test_predeclared_condition_domain_and_fixed_sampling_budget():
    specs = data.condition_specs()
    assert len(specs) == 68
    assert [p['id'] for p in specs] == [f'c{i:03d}' for i in range(68)]
    assert [p['index'] for p in specs] == list(range(68))
    assert all(p['logical_size'] == p['size'] for p in specs)
    assert [(p['K0'], p['size']) for p in specs[:14]] == [(.42, 108)] * 7 + [(.46, 108)] * 7
    assert all(p['role'] == 'train_domain' for p in specs[:14])
    assert all(p['role'] == 'heldout_domain' and p['K0'] == .44 for p in specs[14:])
    assert sum(p['size'] == 108 for p in specs) == 41
    assert sum(p['size'] == 54 for p in specs) == 27
    assert [p['index'] for p in specs if p['role'] == 'train_domain' and p['family'] == 'zero'] == [0, 7]
    assert all(p['target_size'] == 27 and p['depth'] == (2 if p['size'] == 108 else 1) for p in specs)
    for start in (14, 41):
        held = specs[start:start + 27]
        assert sum(p['family'] == 'zero' for p in held) == 1
        assert sum(p['family'] == 'axis_unit' for p in held) == 6
        assert sum(p['family'] == 'axis_double' for p in held) == 6
        assert sum(p['family'] == 'mixed_unit' for p in held) == 8
        assert sum(p['family'] == 'unaugmented_axis' for p in held) == 6
        np.testing.assert_allclose([np.linalg.norm(p['unit_delta6']) for p in held if p['family'] == 'mixed_unit'], 1.)
    assert data.total_sampling_budget()['cluster_proposals'] == 26181632
    assert data.total_sampling_budget()['local_sweeps'] == 818176
    assert data.total_sampling_budget()['production_chains'] == 816
    assert data.total_sampling_budget()['production_configurations'] == 104448


def test_smoke_geometry_preserves_the_condition_design():
    formal, smoke = data.condition_specs(), data.condition_specs(smoke=True)
    for x, y in zip(formal, smoke):
        assert y['size'] == {108: 16, 54: 8}[x['size']]
        assert y['logical_size'] == x['size'] and y['depth'] == x['depth']
        assert y['target_size'] == 4 and y['unit_delta6'] == x['unit_delta6']
    with pytest.raises(ValueError):
        data.condition_specs(smoke=1)


def test_operator_values_match_existing_mcrg_including_both_rectangles():
    rng = np.random.default_rng(2)
    for side in (2, 4, 8):
        x = rng.choice(np.array([-1, 1], np.int8), size=(12, side, side))
        actual = data.operator_values_numpy(x)
        expected = reference_operators(torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(actual, expected[:, :6])
        np.testing.assert_array_equal(data.operator_values_numpy(-x), actual)
    aligned = data.operator_values_numpy(np.ones((1, 8, 8), np.int8))[0]
    np.testing.assert_array_equal(aligned, np.array([2, 2, 1, 2, 2, 2]) * 64)


@pytest.mark.parametrize('side', [2, 4, 8])
def test_single_spin_energy_increment_matches_total_action_oracle(side):
    rng = np.random.default_rng(5)
    terms = data.interaction_structure(side)
    x = rng.choice(np.array([-1, 1], np.int8), size=(side, side))
    coupling = np.array([.44, -.003, .002, .005, -.001, .004])
    before = data.operator_values_numpy(x[None])[0]
    for site in range(side * side):
        delta = data.single_spin_operator_delta(x, site, terms)
        changed = x.copy()
        changed.flat[site] *= -1
        oracle = data.operator_values_numpy(changed[None])[0] - before
        np.testing.assert_array_equal(delta, oracle)
        assert np.dot(delta, coupling) == pytest.approx(np.dot(oracle, coupling), abs=1e-13)


def _exact_small_wolff_proposal(side, k):
    """Enumerate FK bonds (parallel torus edges retained) and uniform seeds."""
    n = side * side
    states = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=np.int8)
    lookup = {tuple(row): i for i, row in enumerate(states)}
    edges = [(i * side + j, ((i + 1) % side) * side + j) for i in range(side) for j in range(side)]
    edges += [(i * side + j, i * side + (j + 1) % side) for i in range(side) for j in range(side)]
    p = -np.expm1(-2 * k)
    proposal = np.zeros((len(states), len(states)))
    for ix, s in enumerate(states):
        eligible = [(a, b) for a, b in edges if s[a] == s[b]]
        for occupied in itertools.product((False, True), repeat=len(eligible)):
            prob = np.prod([p if flag else 1 - p for flag in occupied])
            adjacency = [[] for _ in range(n)]
            for flag, (a, b) in zip(occupied, eligible):
                if flag:
                    adjacency[a].append(b)
                    adjacency[b].append(a)
            for seed in range(n):
                component, pending = {seed}, [seed]
                while pending:
                    for neighbor in adjacency[pending.pop()]:
                        if neighbor not in component:
                            component.add(neighbor)
                            pending.append(neighbor)
                changed = s.copy()
                changed[list(component)] *= -1
                proposal[ix, lookup[tuple(changed)]] += prob / n
    return states.reshape(-1, side, side), proposal


def test_small_exact_wolff_mh_kernel_satisfies_detailed_balance():
    x, proposal = _exact_small_wolff_proposal(2, .43)
    np.testing.assert_allclose(proposal.sum(1), 1., atol=5e-15)
    op = data.operator_values_numpy(x)
    delta = np.array([-.012, .021, -.017, .004, -.009, .003])
    log_weight = .43 * op[:, 0] + op @ delta
    target = np.exp(log_weight - log_weight.max())
    target /= target.sum()
    transition = proposal * np.exp(np.minimum(0., (op @ delta)[None] - (op @ delta)[:, None]))
    transition[np.diag_indices(len(x))] += 1 - transition.sum(1)
    np.testing.assert_allclose(target[:, None] * transition,
                               target[None, :] * transition.T, atol=4e-15)
    for state in x:
        for site in range(4):
            d = data.single_spin_operator_delta(state, site)
            changed = state.copy()
            changed.flat[site] *= -1
            np.testing.assert_array_equal(d, data.operator_values_numpy(changed[None])[0] -
                                          data.operator_values_numpy(state[None])[0])


def test_cluster_mh_delta_matches_total_action_for_arbitrary_even_terms():
    rng = np.random.default_rng(88)
    for side in (2, 4, 8):
        terms = data.interaction_structure(side)
        for count in (1, side * side // 2, side * side - 1, side * side):
            x = rng.choice(np.array([-1, 1], np.int8), size=(side, side))
            sites = rng.choice(side * side, size=count, replace=False)
            proposed = x.copy()
            proposed.flat[sites] *= -1
            np.testing.assert_array_equal(data.cluster_operator_delta(proposed, sites, terms),
                data.operator_values_numpy(proposed[None])[0] - data.operator_values_numpy(x[None])[0])


def test_zero_delta_accepts_every_cluster_and_counts_rejections_and_sweeps():
    x, receipt = data.run_chain(.44, np.zeros(6), 8, 8, 16, 2, 777)
    assert x.shape == (8, 8, 8) and x.dtype == np.int8
    assert receipt['cluster_proposals'] == receipt['cluster_accepted'] == 32
    assert receipt['local_sweeps'] == 1 and receipt['local_spin_proposals'] == 64
    assert np.all(np.abs(x) == 1)
    again, other = data.run_chain(.44, np.zeros(6), 8, 8, 16, 2, 777)
    np.testing.assert_array_equal(x, again)
    assert receipt == other


def test_actual_small_sampler_agrees_with_exact_target_not_reference_K0():
    """End-to-end sign check independent of the implementation's local oracle."""
    states = np.array(list(itertools.product((-1, 1), repeat=4)), dtype=np.int8).reshape(-1, 2, 2)
    operators = data.operator_values_numpy(states)
    k = .24
    delta = np.array([.16, .035, .07, 0., -.025, 0.])
    logp = k * operators[:, 0] + operators @ delta
    probability = np.exp(logp - logp.max())
    probability /= probability.sum()
    expected = probability @ operators / 4
    configurations, receipt = data.run_chain(k, delta, 2, 12000, 1024, 4, 8837)
    observed = data.operator_values_numpy(configurations).mean(0) / 4
    np.testing.assert_allclose(observed, expected, atol=.06, rtol=0.)
    assert 0 < receipt['cluster_accepted'] < receipt['cluster_proposals']
    assert receipt['cluster_proposals'] == 1024 + 12000 * 4


def test_seed_namespace_exhaustive_no_collisions_and_reserved_purposes():
    assert data.validate_seed_namespace(20261017, smoke=False) > 1000
    seeds = [data.namespaced_seed(20261017, p['size'], purpose, p['index'], rep)
             for p in data.condition_specs() for purpose in data.SEED_PURPOSES for rep in range(12)]
    assert len(seeds) == len(set(seeds))
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            data.namespaced_seed(bad, 108, 'production', 0, 0)


def _fake_chain(k, delta, size, draws, equil, gap, seed, cold=False):
    rng = np.random.default_rng(seed)
    x = rng.choice(np.array([-1, 1], np.int8), size=(draws, size, size))
    proposals = equil + draws * gap
    return x, dict(cluster_proposals=proposals, cluster_accepted=proposals,
                   local_sweeps=proposals // 32, local_spin_proposals=(proposals // 32) * size ** 2,
                   local_spin_accepted=(proposals // 32) * size ** 2,
                   cluster_sites_proposed=proposals, cluster_acceptance=1., local_acceptance=1.)


def _fake_benchmark(epsilon6, seed, smoke=False):
    return dict(passed=True, cpu_budget_hours=24., projected_cpu_hours_with_margin=.01,
                safety_factor=2., cases=[], non_scientific=True)


def test_benchmark_covers_all_six_active_axis_templates_and_full_mixed_table(monkeypatch):
    monkeypatch.setattr(data, 'run_chain', _fake_chain)
    monkeypatch.setattr(data, 'operator_values_numpy', lambda x: np.zeros((len(x), 6)))
    epsilon = np.arange(1, 7, dtype=np.float64) / 1000
    report = data.benchmark_budget(epsilon, 20261017, smoke=True)
    assert report['fixed_benchmark_cases'] == len(report['cases']) == 60
    assert report['non_scientific'] and report['benchmark_budget_not_in_scientific_production']
    assert report['total_scientific_sampling_budget'] == data.total_sampling_budget(smoke=True)
    assert len({row['seed'] for row in report['cases']}) == 60
    for k in (.42, .46):
        for initial in ('hot', 'cold'):
            rows = [r for r in report['cases'] if r['K0'] == k and r['initial'] == initial]
            assert len(rows) == 15
            axes = [r for r in rows if np.count_nonzero(r['delta6']) == 1]
            assert len(axes) == 12
            for axis in range(6):
                coefficients = [r['delta6'][axis] for r in axes if r['delta6'][axis] != 0]
                np.testing.assert_allclose(sorted(coefficients), [-2 * epsilon[axis], 2 * epsilon[axis]])
            mixed = [r for r in rows if np.count_nonzero(r['delta6']) == 6]
            assert len(mixed) == 2
            for row in mixed:
                np.testing.assert_allclose(np.abs(row['delta6']), 2 * epsilon / np.sqrt(6))


def test_calibration_reuses_four_pilots_and_produces_exclusive_smoke_data(tmp_path, monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_chain(*args, **kwargs)
    monkeypatch.setattr(data, 'run_chain', fake)
    monkeypatch.setattr(data, 'benchmark_budget', _fake_benchmark)
    calibration_dir = tmp_path / 'calibration'
    calibration = data.calibrate(calibration_dir, seed=20261017, smoke=True)
    assert len(calls) == 4
    assert len(calibration['pilot']) == 2 and calibration['budget_gate']['passed']
    variance = [p[initial]['operator_variance_extensive'] for p in calibration['pilot'] for initial in ('hot', 'cold')]
    np.testing.assert_allclose(calibration['sigma6'], np.sqrt(np.mean(variance, axis=0)))
    np.testing.assert_allclose(calibration['epsilon6'], .5 / np.array(calibration['sigma6']))
    out0 = tmp_path / 'c000'
    report = data.sample(calibration_dir, 0, out0, seed=20261017, smoke=True)
    assert len(calls) == 4 + 2  # zero-condition pilots are reused, never redrawn.
    assert report['pilot']['reused_from_calibration'] is True
    status = json.loads((out0 / 'status.json').read_text())
    payload = torch.load(out0 / 'data.pt', weights_only=True)
    assert status['status'] == 'complete' and status['workflow_complete']
    assert payload['configurations'].shape == (2, 8, 16, 16)
    assert payload['configurations'].dtype == torch.int8
    assert report['condition_id'] == 'c000' and report['condition_index'] == 0
    assert report['role'] == 'train_domain' and report['family'] == 'zero'
    assert report['pilot_in_production'] is False
    assert report['training_updates'] == report['MI_estimates'] == 0
    assert report['data_sha256'] == hashlib.sha256((out0 / 'data.pt').read_bytes()).hexdigest()
    assert status['sidecar_sha256'] == hashlib.sha256((out0 / 'sidecar.json').read_bytes()).hexdigest()
    assert payload['metadata_fingerprint'] == report['metadata_fingerprint']
    for rep, record in enumerate(report['ensembles']):
        raw = _fake_chain(.42, np.zeros(6), 16, 8, 16, 2, record['chain_seed'])[0]
        signs = np.random.default_rng(record['global_flip_seed']).choice(np.array([-1, 1], np.int8), size=(8, 1, 1))
        np.testing.assert_array_equal(payload['configurations'][rep].numpy(), raw * signs)
        assert record['raw_diagnostics'] == data.raw_diagnostics(raw)
    report1 = data.sample(calibration_dir, 1, tmp_path / 'c001', seed=20261017, smoke=True)
    assert len(calls) == 4 + 2 + 2 + 2
    assert report1['pilot']['reused_from_calibration'] is False
    np.testing.assert_allclose(report1['delta6'], np.array(data.condition_specs(True)[1]['unit_delta6']) * calibration['epsilon6'])
    for path, operation in [(calibration_dir, lambda: data.calibrate(calibration_dir, smoke=True)),
                            (out0, lambda: data.sample(calibration_dir, 0, out0, smoke=True))]:
        before = sorted(p.name for p in path.iterdir())
        with pytest.raises(FileExistsError):
            operation()
        assert before == sorted(p.name for p in path.iterdir())


def test_budget_gate_is_not_training_failure_and_blocks_production(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'run_chain', _fake_chain)
    def over(*args, **kwargs):
        return {**_fake_benchmark(*args, **kwargs), 'passed': False, 'projected_cpu_hours_with_margin': 30.}
    monkeypatch.setattr(data, 'benchmark_budget', over)
    directory = tmp_path / 'calibration'
    result = data.calibrate(directory, seed=20261017, smoke=True)
    status = json.loads((directory / 'status.json').read_text())
    assert result['status'] == status['status'] == 'complete'
    assert status['budget_gate']['passed'] is False
    with pytest.raises(ValueError, match='budget'):
        data.sample(directory, 1, tmp_path / 'not_created', seed=20261017, smoke=True)
    assert not (tmp_path / 'not_created').exists()


def test_invalid_calibration_or_seed_rejected_before_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'run_chain', _fake_chain)
    monkeypatch.setattr(data, 'benchmark_budget', _fake_benchmark)
    directory = tmp_path / 'calibration'
    data.calibrate(directory, seed=20261017, smoke=True)
    monkeypatch.setattr(data, 'run_chain', lambda *a, **kw: pytest.fail('Invalid inputs sampled'))
    with pytest.raises(ValueError, match='seed'):
        data.sample(directory, 1, tmp_path / 'wrong_seed', seed=20261018, smoke=True)
    with pytest.raises(ValueError, match='index'):
        data.sample(directory, 68, tmp_path / 'wrong_index', seed=20261017, smoke=True)
    with pytest.raises(ValueError, match='smoke'):
        data.sample(directory, 1, tmp_path / 'wrong_mode', seed=20261017, smoke=False)
    with (directory / 'calibration.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='SHA'):
        data.sample(directory, 1, tmp_path / 'wrong_hash', seed=20261017, smoke=True)


def test_sampler_failure_is_preserved_without_public_data_or_automatic_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'run_chain', _fake_chain)
    monkeypatch.setattr(data, 'benchmark_budget', _fake_benchmark)
    directory = tmp_path / 'calibration'
    data.calibrate(directory, seed=20261017, smoke=True)
    calls = []
    def broken(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('deliberate sampler test failure')
    monkeypatch.setattr(data, 'run_chain', broken)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='deliberate'):
        data.sample(directory, 0, output, seed=20261017, smoke=True)
    status = json.loads((output / 'status.json').read_text())
    assert len(calls) == 1 and status['status'] == 'failed' and not status['workflow_complete']
    assert not (output / 'data.pt').exists() and not (output / 'sidecar.json').exists()


def test_quality_is_descriptive_and_retains_six_ops_and_raw_magnetization():
    x = np.ones((16, 8, 8), np.int8)
    diagnostics = data.raw_diagnostics(x)
    assert all(name in diagnostics for name in data.EVEN_OPERATOR_NAMES)
    assert all(name in diagnostics for name in ('magnetization', 'magnetization_squared', 'absolute_magnetization'))
    report = data.quality_report(dict(hot=diagnostics, cold=diagnostics), [dict(replica=0, raw_diagnostics=diagnostics)],
                                 samples=16, pilot_samples=16)
    assert report['status'] == 'warning'
    assert 'never automatically' in report['action']


def test_cli_fixed_subcommands_no_device_or_budget_override():
    parser = data.build_parser()
    a = parser.parse_args(['calibrate', '--output', 'a', '--seed', '20261017'])
    assert a.command == 'calibrate' and not a.smoke
    b = parser.parse_args(['sample', '--calibration', 'a', '--condition-index', '12', '--output', 'b', '--smoke'])
    assert b.condition_index == 12 and b.seed == 20261017
    with pytest.raises(SystemExit):
        parser.parse_args(['sample', '--calibration', 'a', '--condition-index', '12', '--output', 'b', '--device', 'cuda:2'])
