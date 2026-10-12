import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.data.ensembles import load_ensembles
from scripts import prepare_ising2d_lr_data as preparation


def fake_chain(k, size, draws, equil, gap, seed, cold=False):
    """No numba compilation or real Monte Carlo in preparation tests."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(draws, size, size))


def small_options():
    return dict(knots=6, samples=8, pilot_samples=8, equil=0, gap_low=1, gap_high=2)


def test_production_defaults_and_complete_K_splits():
    options = preparation.SamplingOptions().validate()
    assert options == preparation.SamplingOptions(size=36, seed=91, knots=64,
        samples=128, replicas=3, equil=4096, gap_low=512, gap_high=128, pilot_samples=1024)
    values = preparation.coupling_knots()
    assert len(values) == 64
    assert values[0] == .2 and values[-1] == .68
    assert preparation.KC in values
    assert np.all(np.diff(values) > 0)
    assert sum(i % 3 != 1 for i in range(64)) * 3 == 129
    assert sum(i % 3 == 1 for i in range(64)) * 3 == 63
    parsed = preparation.build_parser().parse_args(['--output', 'cache.pt'])
    assert {k: getattr(parsed, k) for k in options.__dataclass_fields__} == vars(options)


@pytest.mark.parametrize('changes', [dict(size=32), dict(size=27), dict(size=40),
    dict(size=0), dict(seed=-1), dict(knots=5), dict(samples=7), dict(replicas=0),
    dict(equil=-1), dict(gap_low=0), dict(gap_high=0), dict(pilot_samples=7),
    dict(seed=True)])
def test_reject_invalid_settings_before_sampling_or_creating_outputs(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(preparation, 'chain', lambda *a: pytest.fail('Invalid input sampled'))
    output = tmp_path / 'not-created' / 'cache.pt'
    with pytest.raises(ValueError):
        preparation.prepare_data(output, **{**small_options(), **changes})
    assert not output.parent.exists()


def test_cache_shape_split_labels_and_actual_fingerprint(tmp_path, monkeypatch):
    calls = []
    def capture(*args):
        calls.append(args)
        return fake_chain(*args)
    monkeypatch.setattr(preparation, 'chain', capture)
    output = tmp_path / 'cache.pt'
    report = preparation.prepare_data(output, **small_options())
    persisted = json.loads(output.with_suffix('.json').read_text())
    assert report == persisted
    assert report['data_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    train, test = load_ensembles(output)
    assert train.configurations.shape == (12, 8, 36, 36, 1)
    assert test.configurations.shape == (6, 8, 36, 36, 1)
    assert train.configurations.dtype == test.configurations.dtype == torch.float64
    assert train.couplings.dtype == test.couplings.dtype == torch.float64
    assert report['lattice_sizes'] == [36, 12, 4]
    assert set(train.couplings.tolist()).isdisjoint(test.couplings.tolist())
    values = preparation.coupling_knots(6)
    for index, k in enumerate(values):
        target, other = (test, train) if index % 3 == 1 else (train, test)
        assert (target.couplings == k).sum().item() == 3
        assert (other.couplings == k).sum().item() == 0
    assert len(calls) == 6 + 6 * 3
    assert len({row[5] for row in calls}) == len(calls)
    assert [row[6] for row in calls[:6]] == [False, True] * 3
    assert all(row[6] is False for row in calls[6:])
    assert all(row[4] == (1 if row[0] <= .36 else 2) for row in calls)
    assert len(report['pilot']) == 3
    assert report['sampling_spin_flip']['training_spin_flip_augment'] is False
    assert 'never optimizer supervision' in report['labels_use']
    assert 'finite-chain estimates' in report['uncertainty_note']
    assert not list(tmp_path.glob('.*.tmp'))


def test_global_flip_is_per_configuration_and_keeps_even_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', fake_chain)
    output = tmp_path / 'cache.pt'
    report = preparation.prepare_data(output, **small_options())
    train_x, train_k, test_x, test_k = torch.load(output, weights_only=True)
    offsets = {False: 0, True: 0}
    for row in report['ensembles']:
        before = fake_chain(row['K'], 36, 8, 0, row['gap_cluster_updates'], row['seed'])
        flip_rng = np.random.default_rng(row['global_flip_seed'])
        expected = before * flip_rng.choice(np.array([-1, 1], dtype=np.int8), size=(8, 1, 1))
        x = test_x if row['test'] else train_x
        actual = x[offsets[row['test']], ..., 0].numpy()
        offsets[row['test']] += 1
        np.testing.assert_array_equal(actual, expected)
        assert row['bond_sum_per_site'] == preparation.stats(expected)['bond_sum_per_site']
        assert row['magnetization_squared'] == preparation.stats(expected)['magnetization_squared']


def test_identical_seed_gives_identical_data_and_changed_seed_changes_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', fake_chain)
    reports = []
    for name, seed in [('a', 91), ('b', 91), ('c', 92)]:
        reports.append(preparation.prepare_data(tmp_path / f'{name}.pt', seed=seed, **small_options()))
    assert reports[0]['data_sha256'] == reports[1]['data_sha256']
    assert reports[0]['data_sha256'] != reports[2]['data_sha256']
    assert reports[0]['ensembles'] == reports[1]['ensembles']
    assert reports[0]['historical_reproduction'] is False


@pytest.mark.parametrize('existing', ['cache.pt', 'cache.json'])
def test_existing_cache_or_sidecar_is_not_overwritten(tmp_path, monkeypatch, existing):
    path = tmp_path / existing
    path.write_bytes(b'original data')
    monkeypatch.setattr(preparation, 'chain', lambda *a: pytest.fail('Existing target sampled'))
    with pytest.raises(FileExistsError):
        preparation.prepare_data(tmp_path / 'cache.pt', **small_options())
    assert path.read_bytes() == b'original data'
    assert sorted(p.name for p in tmp_path.iterdir()) == [existing]


def test_concurrent_publication_cannot_clobber_existing_file(tmp_path):
    path = tmp_path / 'cache.pt'
    path.write_bytes(b'other writer')
    with pytest.raises(FileExistsError):
        preparation._publish_new_file(path, lambda stream: stream.write(b'new writer'))
    assert path.read_bytes() == b'other writer'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['cache.pt']


@pytest.mark.parametrize('problem', ['shape', 'nonbinary', 'nonfinite'])
def test_invalid_sampler_output_is_not_published(tmp_path, monkeypatch, problem):
    def broken(k, size, draws, equil, gap, seed, cold=False):
        x = fake_chain(k, size, draws, equil, gap, seed, cold).astype(float)
        if problem == 'shape':
            return x[:, :-1]
        x[0, 0, 0] = np.nan if problem == 'nonfinite' else 0
        return x
    monkeypatch.setattr(preparation, 'chain', broken)
    with pytest.raises(ValueError):
        preparation.prepare_data(tmp_path / 'cache.pt', **small_options())
    assert not list(tmp_path.iterdir())


def test_cli_small_knobs_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', fake_chain)
    output = tmp_path / 'cache.pt'
    report = preparation.main(['--output', str(output), '--size', '36', '--seed', '0',
        '--knots', '6', '--samples', '8', '--pilot-samples', '8', '--equil', '0',
        '--gap-low', '1', '--gap-high', '2'])
    assert report['seed'] == 0
    with pytest.raises(SystemExit) as error:
        preparation.main(['--output', str(tmp_path / 'bad.pt'), '--samples', '7'])
    assert error.value.code == 2
    assert not (tmp_path / 'bad.pt').exists()


def test_non_pt_target_rejected_before_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'chain', lambda *a: pytest.fail('Invalid output sampled'))
    with pytest.raises(ValueError, match='.pt'):
        preparation.prepare_data(tmp_path / 'cache.json', **small_options())
    assert not list(tmp_path.iterdir())
