import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import prepare_ising_matched_sizes as preparation


def fake_chain(k, size, draws, equil, gap, seed, cold=False):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(draws, size, size))


def test_formal_contract_exact_K_and_geometry():
    assert preparation.K_VALUES == (.43619047619047624, .44068679350977147, .45142857142857146)
    for size, depth, shapes in [(54, 1, [54, 27]), (108, 2, [108, 54, 27])]:
        options = preparation._options(size, 20261016, **preparation.FORMAL_BUDGET, smoke=False)
        assert options['max_rg_steps'] == depth
        assert options['lattice_sizes'] == shapes
        assert options['target_size'] == 27 and options['block_factor'] == 2
    args = preparation.build_parser().parse_args(['--output', 'new', '--size', '54'])
    assert args.seed == 20261016
    assert {name: getattr(args, name) for name in preparation.FORMAL_BUDGET} == preparation.FORMAL_BUDGET


@pytest.mark.parametrize('changes', [dict(size=36), dict(size=8), dict(size=True), dict(seed=-1),
    dict(seed=True), dict(replicas=2), dict(samples=8), dict(equil=16), dict(gap=2),
    dict(pilot_samples=16), dict(smoke=1)])
def test_formal_rejects_mixed_budgets_before_creating_directory(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(preparation, 'chain', lambda *args: pytest.fail('Invalid options sampled'))
    output = tmp_path / 'missing' / 'formal'
    with pytest.raises(ValueError):
        preparation.prepare_size(output, **{**dict(size=54), **changes})
    assert not output.parent.exists()


def test_seed_streams_are_independent_and_reproducible():
    values = [preparation.namespaced_seed(20261016, size, purpose, index, replica)
              for size in (54, 108) for purpose in preparation.SEED_PURPOSES
              for index in range(3) for replica in range(12)]
    assert len(set(values)) == len(values)
    assert preparation.namespaced_seed(20261016, 54, 'production', 2, 11) == values[3 * 12 * 2 + 2 * 12 + 11]
    assert preparation.namespaced_seed(20261017, 54, 'production', 2, 11) not in values
    with pytest.raises(ValueError):
        preparation.namespaced_seed(-1, 54, 'production', 0)


def test_smoke_payload_stats_global_symmetry_and_fingerprints(tmp_path, monkeypatch):
    calls = []
    def capture(*args):
        calls.append(args)
        return fake_chain(*args)
    monkeypatch.setattr(preparation, 'chain', capture)
    output = tmp_path / 'smoke'
    report = preparation.prepare_size(output, size=8, smoke=True)
    assert report == json.loads((output / 'sidecar.json').read_text())
    payload = torch.load(output / 'data.pt', weights_only=True)
    status = json.loads((output / 'status.json').read_text())
    assert payload['schema_version'] == 1
    assert payload['configurations'].shape == (3, 2, 8, 8, 8)
    assert payload['configurations'].dtype == torch.int8
    assert set(payload['configurations'].unique().tolist()) == {-1, 1}
    assert payload['K'].dtype == torch.float64
    assert payload['K'].tolist() == list(preparation.K_VALUES)
    assert payload['chain_seeds'].dtype == payload['global_flip_seeds'].dtype == torch.int64
    assert payload['chain_seeds'].shape == (3, 2)
    assert report['lattice_sizes'] == [8, 4]
    assert payload['depth'] == report['depth'] == payload['max_rg_steps'] == report['max_rg_steps'] == 1
    assert report['pilot_in_production'] is False
    assert report['scientific_result'] is False and payload['smoke'] is True
    assert len(calls) == 6 + 6
    assert len({row[5] for row in calls}) == len(calls)
    assert [row[-1] for row in calls[:6]] == [False, True] * 3
    assert all(row[3:5] == (16, 2) for row in calls)
    assert status['completed_pilot_chains'] == 6
    assert status['completed_production_chains'] == 6
    assert status['status'] == report['status'] == 'complete'
    assert status['workflow_complete'] is report['workflow_complete'] is True
    assert status['training_updates'] == payload['training_updates'] == report['training_updates'] == 0
    assert status['MI_estimates'] == payload['MI_estimates'] == report['MI_estimates'] == 0
    assert report['data_sha256'] == status['data_sha256'] == hashlib.sha256((output / 'data.pt').read_bytes()).hexdigest()
    assert status['sidecar_sha256'] == hashlib.sha256((output / 'sidecar.json').read_bytes()).hexdigest()
    assert payload['metadata_fingerprint'] == report['metadata_fingerprint'] == status['metadata_fingerprint']
    assert report['metadata_fingerprint'] == hashlib.sha256(json.dumps(report['provenance'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    for row in report['ensembles']:
        original = fake_chain(row['K'], 8, 8, 16, 2, row['chain_seed'])
        signs = np.random.default_rng(row['global_flip_seed']).choice(np.array([-1, 1], dtype=np.int8), size=(8, 1, 1))
        actual = payload['configurations'][row['K_index'], row['replica']].numpy()
        np.testing.assert_array_equal(actual, original * signs)
        assert row['raw_diagnostics'] == preparation.raw_diagnostics(original)
        for name in ('bond_sum_per_site', 'magnetization_squared'):
            assert row['raw_diagnostics'][name] == preparation.raw_diagnostics(actual)[name]
    assert not list(output.glob('.*.tmp'))


def test_data_reproducibility_new_size_and_seed_independence(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', fake_chain)
    reports = [preparation.prepare_size(tmp_path / name, size=size, seed=seed, smoke=True)
               for name, size, seed in [('a', 8, 20261016), ('b', 8, 20261016),
                                        ('c', 8, 20261017), ('d', 16, 20261016)]]
    assert reports[0]['data_sha256'] == reports[1]['data_sha256']
    assert reports[0]['ensembles'] == reports[1]['ensembles']
    assert reports[0]['data_sha256'] != reports[2]['data_sha256']
    assert reports[0]['data_sha256'] != reports[3]['data_sha256']
    assert reports[3]['lattice_sizes'] == [16, 8, 4]


def test_existing_directory_even_empty_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    monkeypatch.setattr(preparation, 'chain', lambda *args: pytest.fail('Existing target sampled'))
    with pytest.raises(FileExistsError):
        preparation.prepare_size(output, size=8, smoke=True)
    assert list(output.iterdir()) == []
    (output / 'status.json').write_text('original')
    with pytest.raises(FileExistsError):
        preparation.prepare_size(output, size=8, smoke=True)
    assert (output / 'status.json').read_text() == 'original'


@pytest.mark.parametrize('problem', ['shape', 'nonbinary', 'nonfinite'])
def test_bad_sampler_retains_failed_receipt_without_publishing_data(tmp_path, monkeypatch, problem):
    def invalid(*args):
        x = fake_chain(*args).astype(float)
        if problem == 'shape':
            return x[:, :-1]
        x[0, 0, 0] = np.nan if problem == 'nonfinite' else 0
        return x
    monkeypatch.setattr(preparation, 'chain', invalid)
    output = tmp_path / 'failed'
    with pytest.raises(ValueError):
        preparation.prepare_size(output, size=8, smoke=True)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed' and status['workflow_complete'] is False
    assert status['error']['type'] == 'ValueError'
    assert not (output / 'data.pt').exists() and not (output / 'sidecar.json').exists()


def test_quality_thresholds_do_not_adapt_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', lambda k, size, draws, *args: np.ones((draws, size, size), dtype=np.int8))
    output = tmp_path / 'warning'
    report = preparation.prepare_size(output, size=8, smoke=True)
    assert report['quality']['status'] == 'warning'
    assert report['quality']['warnings']
    assert report['production_saved_configurations'] == 3 * 2 * 8
    assert report['pilot_generated_configurations'] == 3 * 2 * 16
    assert report['pilot_saved_configurations'] == 0
    assert report['quality']['thresholds'] == preparation.QUALITY_THRESHOLDS


def test_cli_smoke_and_formal_budget_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', fake_chain)
    report = preparation.main(['--output', str(tmp_path / 'cli'), '--size', '8', '--smoke'])
    assert report['samples'] == 8 and report['replicas'] == 2
    with pytest.raises(SystemExit) as error:
        preparation.main(['--output', str(tmp_path / 'invalid'), '--size', '54', '--samples', '8'])
    assert error.value.code == 2
    assert not (tmp_path / 'invalid').exists()


def test_real_small_smoke_sampler(tmp_path):
    """The only real Monte Carlo test; a separate smoke root, never formal data."""
    report = preparation.prepare_size(tmp_path / 'real_smoke', size=8, smoke=True)
    x = torch.load(tmp_path / 'real_smoke' / 'data.pt', weights_only=True)['configurations']
    assert tuple(x.shape) == (3, 2, 8, 8, 8)
    assert bool(((x == -1) | (x == 1)).all())
    assert report['status'] == 'complete' and report['smoke'] is True
