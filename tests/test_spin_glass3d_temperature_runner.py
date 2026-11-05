"""Small receipt/cache fixtures only: no model, sampler, SSH, or real GPU probes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_spin_glass3d_temperature_supervised as runner


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    root = tmp_path / 'baseline'
    config = json.loads((Path(__file__).resolve().parents[1]
        / 'configs/spin_glass3d_b2_overlap_g1_seed91.json').read_text())
    cache = root / 'data/ensembles.pt'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'fixture data fingerprint only; never loaded as a dataset')
    monkeypatch.setattr(runner, 'SOURCE_DATA_SHA', runner.sha(cache))
    config['data']['cache'] = str(cache)
    fingerprints = {}
    for name in ('experiments/rsmi_separate_qcr.py', 'recurrent_rg/model_nd.py',
                 'recurrent_rg/losses.py'):
        path = root / 'code' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# immutable historical fixture: ' + name + '\n')
        fingerprints[name] = runner.sha(path)
    dump(root / 'code/fingerprints.json', fingerprints)
    dump(root / 'job.json', dict(status='complete', workflow_complete=True))
    dump(root / 'config.json', config)
    dump(root / 'run/manifest.json', dict(labels_used_in_training=False))
    dump(root / 'run/summary.json', dict(actual_epochs=6000))
    (root / 'run/checkpoint_last.pt').write_bytes(b'not a model; only fingerprinted')
    dump(root / 'data/evaluation_metadata.json', dict(labels_used_in_training=False,
        cache_sha256=runner.SOURCE_DATA_SHA,
        train=[dict(disorder_id=j) for j in (0, 2, 3, 5, 6, 8)],
        test=[dict(disorder_id=j) for j in (1, 4, 7)]))
    dump(root / 'mi_receipt.json', dict(status='complete', data_sha256=runner.SOURCE_DATA_SHA))
    dump(root / 'report/report.json', dict(status='complete'))
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: runner.HOST)
    monkeypatch.setattr(runner.os, 'getpriority', lambda *args: 0)
    return root


def test_baseline_validates_frozen_source_without_model_or_dataset_loading(baseline):
    config, metadata = runner.validate_baseline(baseline)
    assert config['training']['epochs'] == 6000
    assert config['data']['coupling_names'] == []
    assert len(metadata['train']) == 6
    config['data']['coupling_names'].append('T')
    assert runner.read(baseline / 'config.json')['data']['coupling_names'] == []


@pytest.mark.parametrize('change', ['unfinished', 'supervised', 'wrong_endpoint', 'wrong_model',
                                    'nonuniform', 'wrong_data', 'wrong_split', 'changed_source', 'no_mi'])
def test_baseline_mismatch_is_rejected_without_repair(baseline, change):
    if change == 'unfinished':
        dump(baseline / 'job.json', dict(status='running', workflow_complete=False))
    elif change == 'supervised':
        dump(baseline / 'run/manifest.json', dict(labels_used_in_training=True))
    elif change == 'wrong_endpoint':
        dump(baseline / 'run/summary.json', dict(actual_epochs=5999))
    elif change in ('wrong_model', 'nonuniform'):
        config = runner.read(baseline / 'config.json')
        if change == 'wrong_model':
            config['model']['theory_dim'] = 2
        else:
            config['loss']['closure_scale_weights'] = [.2, 1.8]
        dump(baseline / 'config.json', config)
    elif change == 'wrong_data':
        (baseline / 'data/ensembles.pt').write_bytes(b'changed physical inputs')
    elif change == 'wrong_split':
        metadata = runner.read(baseline / 'data/evaluation_metadata.json')
        metadata['test'][0]['disorder_id'] = 0
        dump(baseline / 'data/evaluation_metadata.json', metadata)
    elif change == 'changed_source':
        (baseline / 'code/recurrent_rg/model_nd.py').write_text('# unexpected change')
    else:
        dump(baseline / 'mi_receipt.json', dict(status='failed', data_sha256=runner.SOURCE_DATA_SHA))
    with pytest.raises((ValueError, RuntimeError)):
        runner.validate_baseline(baseline)


def test_stage_commands_keep_default_mi_and_fresh_training(tmp_path):
    commands = runner.stage_commands(tmp_path / 'new', tmp_path / 'baseline')
    assert [row[0] for row in commands] == ['prepare_temperature_labels', 'reuse_exact_MI_statistics',
                                          'gpu_preflight', 'train', 'report']
    for name, command, gpu in commands:
        assert '--resume-checkpoint' not in command
        assert '--source' not in command
        assert '--allow-nondefault-recipe' not in command
        assert '--allow-legacy-training' not in command
        assert not any('generate_spin_glass' in argument for argument in command)
        if name in ('gpu_preflight', 'train'):
            assert gpu and command[command.index('--device') + 1] == 'cuda:2'
            assert command[command.index('--q-coupling-weight') + 1] == '10.0'
            assert '--mi-cache-dir' in command and '--rsmi-weight' not in command
            assert '--skip-final-report' in command
            assert ('--epochs' in command) == (name == 'gpu_preflight')
    assert commands[2][1][commands[2][1].index('--epochs') + 1] == '5'


def test_run_copies_immutable_core_and_never_retries_failed_child(baseline, tmp_path, monkeypatch):
    root = tmp_path / 'new_run'
    calls = []
    monkeypatch.setattr(runner, 'admit', lambda *args: None)
    monkeypatch.setattr(runner, 'stage_commands', lambda *args: [('fixture_failure', ['not-executed'], False)])
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=77, wait=lambda: 1)
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    protected = {str(path): runner.sha(path) for path in baseline.rglob('*') if path.is_file()}
    job = runner.run(root, baseline)
    assert job['status'] == 'failed' and job['requires_user_action']
    assert len(calls) == len(job['stages']) == 1
    assert job['stages'][0]['returncode'] == 1
    for name in ('experiments/rsmi_separate_qcr.py', 'recurrent_rg/model_nd.py', 'recurrent_rg/losses.py'):
        assert (root / 'code' / name).read_bytes() == (baseline / 'code' / name).read_bytes()
        assert name not in runner.OVERRIDES
    assert {str(path): runner.sha(path) for path in baseline.rglob('*') if path.is_file()} == protected
    runner.verify_code(root, runner.read(root / 'plan.json')['source_fingerprints'])
    with pytest.raises(FileExistsError):
        runner.run(root, baseline)
    assert len(calls) == 1


def test_launch_lock_and_receipt_prevent_duplicate_spawn(baseline, tmp_path, monkeypatch):
    root = tmp_path / 'results/new_launch'
    root.parent.mkdir()
    calls = []
    monkeypatch.setattr(runner.subprocess, 'Popen',
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(pid=88))
    record = runner.launch(root, baseline)
    assert record['no_automatic_restart'] and record['device'] == 'cuda:2'
    assert record['gpu_uuid'] == runner.GPU_UUID
    assert len(calls) == 1 and '--run' in calls[0]
    with pytest.raises(FileExistsError):
        runner.launch(root, baseline)
    assert len(calls) == 1


def test_ambiguous_launch_error_is_not_retried(baseline, tmp_path, monkeypatch):
    root = tmp_path / 'results/ambiguous_launch'
    root.parent.mkdir()
    calls = []
    def fail_spawn(*args, **kwargs):
        calls.append(args)
        raise OSError('fixture ambiguous spawn')
    monkeypatch.setattr(runner.subprocess, 'Popen', fail_spawn)
    with pytest.raises(OSError):
        runner.launch(root, baseline)
    with pytest.raises(FileExistsError):
        runner.launch(root, baseline)
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['wrong_uuid', 'active', 'low_gpu', 'low_host', 'load', 'io', 'ready'])
def test_snapshot_gpu2_uuid_and_shared_host_gates(monkeypatch, mode):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: runner.HOST)
    gpu_uuid = 'GPU-wrong' if mode == 'wrong_uuid' else runner.GPU_UUID
    free_mib = 11000 if mode == 'low_gpu' else 24000
    def query(command, **kwargs):
        if '--query-gpu=index,uuid,memory.free' in command:
            return f'0, GPU-other, 2\n2, {gpu_uuid}, {free_mib}\n'
        assert '--query-compute-apps=gpu_uuid,pid,used_memory' in command
        return f'{runner.GPU_UUID}, 19, 5\n' if mode == 'active' else 'GPU-other, 20, 20000\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', query)
    def read_proc(path, *args, **kwargs):
        if str(path) == '/proc/meminfo':
            return 'MemAvailable: ' + ('20000000' if mode == 'low_host' else '50000000') + ' kB\n'
        if str(path) == '/proc/pressure/io':
            return 'full avg10=' + ('20.0' if mode == 'io' else '0.1') + ' avg60=0\n'
        pytest.fail('Unexpected real filesystem resource probe: ' + str(path))
    monkeypatch.setattr(Path, 'read_text', read_proc)
    monkeypatch.setattr(runner.os, 'cpu_count', lambda: 24)
    monkeypatch.setattr(runner.os, 'getloadavg', lambda: (31 if mode == 'load' else 10, 0, 0))
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda *args: SimpleNamespace(free=100 * 1024**3))
    if mode == 'wrong_uuid':
        with pytest.raises(RuntimeError, match='identity mismatch'):
            runner.snapshot()
    else:
        result = runner.snapshot()
        assert result['gpu_ready'] is (mode == 'ready')
        assert result['gpu_index'] == 2


def test_gpu_admission_requires_two_consecutive_checks_and_ten_second_gaps(tmp_path, monkeypatch):
    states = iter([False, True, False, True, True])
    sleeps = []
    monkeypatch.setattr(runner, 'snapshot', lambda: dict(host_ready=True, gpu_ready=next(states)))
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    job = {}
    runner.admit(tmp_path, job, gpu=True)
    assert len(job['resource_checks']) == 5
    assert sleeps == [10, 10, 10, 10]


def test_cpu_stage_requires_host_but_not_an_idle_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'snapshot', lambda: dict(host_ready=True, gpu_ready=False))
    monkeypatch.setattr(runner.time, 'sleep', lambda *args: pytest.fail('CPU stage waited for GPU'))
    job = {}
    runner.admit(tmp_path, job, gpu=False)
    assert len(job['resource_checks']) == 1
    assert job['resource_checks'][0]['required'] == 'host'


@pytest.fixture
def mi_pair(tmp_path, monkeypatch):
    """Tiny synthetic statistics are serialized, never inferred from real data."""
    from recurrent_rg.config import RunConfig
    from recurrent_rg.data.ensembles import EnsembleDataset
    from recurrent_rg.empirical_mi import EmpiricalBinaryMI, binary_lattice_mi_statistics
    from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
    from recurrent_rg.rsmi import region_offsets
    import recurrent_rg.runner as data_runner
    old_root, new_root = tmp_path / 'mi_baseline', tmp_path / 'mi_new'
    old_cache, new_cache = old_root / 'data/ensembles.pt', new_root / 'data/ensembles.pt'
    for cache, content in ((old_cache, b'fixture old label-free data'), (new_cache, b'fixture new T targets')):
        cache.parent.mkdir(parents=True)
        cache.write_bytes(content)
    monkeypatch.setattr(runner, 'SOURCE_DATA_SHA', runner.sha(old_cache))
    raw_config = json.loads((Path(__file__).resolve().parents[1]
        / 'configs/spin_glass3d_b2_overlap_g1_seed91.json').read_text())
    configs, datasets = [], []
    for root, cache, supervised in ((old_root, old_cache, False), (new_root, new_cache, True)):
        raw = deepcopy(raw_config)
        raw['data'].update(cache=str(cache), coupling_names=['T'] if supervised else [])
        dump(root / 'config.json', raw)
        configs.append(RunConfig.load(root / 'config.json'))
        pair = []
        for j in (0, 1):
            x = torch.ones((2, 4, 8, 8, 8, 1), dtype=torch.int8)
            x[1] = -1
            pair.append(EnsembleDataset(x, torch.tensor([[.75], [1.5]]) if supervised else None,
                torch.tensor([[j, 0], [j, 1]])))
        datasets.append(tuple(pair))
    monkeypatch.setattr(data_runner, 'prepared_data',
        lambda cfg: datasets[0] if cfg.data.cache == str(old_cache) else datasets[1])
    conditional = torch.zeros((2, 256), dtype=torch.float64)
    conditional[0, 0] = conditional[1, -1] = 1
    weights = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float64).to_sparse()
    statistic = EmpiricalBinaryMI(conditional, weights, weights.to_dense() @ conditional,
                                 dict(synthetic_fixture=True))
    receipt = dict(status='complete', data_sha256=runner.SOURCE_DATA_SHA, splits={})
    _, environment = region_offsets(3, block=2, buffer=1)
    for split, dataset in zip(('train', 'test'), datasets[0]):
        key = lattice_statistics_cache_key(configs[0], runner.SOURCE_DATA_SHA, split,
            dataset.configurations, environment, binary_lattice_mi_statistics.__name__, dict(block=2))
        _, record = load_or_build_mi_statistics(old_root / 'mi_cache', key, lambda: statistic)
        receipt['splits'][split] = dict(cache=record)
    dump(old_root / 'mi_receipt.json', receipt)
    return old_root, new_root, datasets


def test_mi_rebind_preserves_counts_with_new_data_sha_without_computation(mi_pair, monkeypatch):
    import recurrent_rg.empirical_mi as empirical
    from recurrent_rg.mi_cache import _cache_identity, _load_cache
    old, new, _ = mi_pair
    original_files = {str(p): runner.sha(p) for p in old.rglob('*') if p.is_file()}
    def binary_lattice_mi_statistics(*args, **kwargs):
        pytest.fail('MI statistics must not be recomputed')
    monkeypatch.setattr(empirical, 'binary_lattice_mi_statistics', binary_lattice_mi_statistics)
    result = runner.reuse_mi(new, old)
    assert result['status'] == 'complete' and not result['new_MI_histograms_computed']
    assert not result['model_initialized'] and result['sampling_draws'] == 0
    assert result['data_sha256'] == runner.sha(new / 'data/ensembles.pt')
    assert result['data_sha256'] != result['source_data_sha256']
    for split in ('train', 'test'):
        entry = result['splits'][split]
        assert entry['configurations_identical'] and entry['groups_identical'] and entry['source_unchanged']
        saved_new = torch.load(entry['cache']['path'], map_location='cpu', weights_only=True)
        saved_old = torch.load(entry['source_path'], map_location='cpu', weights_only=True)
        assert {k:v for k,v in saved_new['identity']['key_data'].items() if k != 'data_sha256'} == {
            k:v for k,v in saved_old['identity']['key_data'].items() if k != 'data_sha256'}
        assert saved_new['key'] != saved_old['key']
        for name in ('conditional', 'weights', 'marginal'):
            torch.testing.assert_close(saved_new['statistics'][name]['values'],
                                       saved_old['statistics'][name]['values'])
        identity, key = _cache_identity(saved_new['identity']['key_data'])
        loaded = _load_cache(Path(entry['cache']['path']), identity, key)
        assert loaded.diagnostics['synthetic_fixture']
    assert {str(p): runner.sha(p) for p in old.rglob('*') if p.is_file()} == original_files
    with pytest.raises(FileExistsError):
        runner.reuse_mi(new, old)


@pytest.mark.parametrize('change', ['configurations', 'groups', 'dtype', 'shape', 'flip', 'pooling',
                                    'region_geometry', 'dimension', 'source_receipt_key',
                                    'corrupt_source_cache', 'corrupt_destination_cache'])
def test_mi_rebind_rejects_input_region_or_source_changes(mi_pair, change):
    old, new, datasets = mi_pair
    if change == 'configurations':
        datasets[1][0].configurations[0, 0, 0, 0, 0, 0] = -1
    elif change == 'groups':
        datasets[1][0].group_ids[0, 0] = 5
    elif change == 'dtype':
        # torch.equal ignores dtype, so the full cache-key comparison is needed.
        datasets[1][0].configurations = datasets[1][0].configurations.double()
    elif change == 'shape':
        datasets[1][0].configurations = datasets[1][0].configurations[:, :, :4]
    elif change in ('flip', 'pooling', 'region_geometry', 'dimension'):
        config = runner.read(new / 'config.json')
        if change == 'flip':
            config['data']['spin_flip_augment'] = True
        elif change == 'pooling':
            config['data']['train_pool_replicas'] = 2
        elif change == 'region_geometry':
            config['model']['block_factor'] = 3
        else:
            config['model']['dimension'] = 2
        dump(new / 'config.json', config)
    elif change == 'source_receipt_key':
        receipt = runner.read(old / 'mi_receipt.json')
        receipt['splits']['train']['cache']['key'] = 'f' * 64
        dump(old / 'mi_receipt.json', receipt)
    elif change == 'corrupt_source_cache':
        source = Path(runner.read(old / 'mi_receipt.json')['splits']['train']['cache']['path'])
        source.write_bytes(b'corrupt fixture; do not replace')
    else:
        from recurrent_rg.config import RunConfig
        from recurrent_rg.mi_cache import lattice_statistics_cache_key, _cache_identity
        from recurrent_rg.rsmi import region_offsets
        _, environment = region_offsets(3, block=2, buffer=1)
        key = lattice_statistics_cache_key(RunConfig.load(new / 'config.json'),
            runner.sha(new / 'data/ensembles.pt'), 'train', datasets[1][0].configurations,
            environment, 'binary_lattice_mi_statistics', dict(block=2))
        _, digest = _cache_identity(key)
        source = new / 'mi_cache' / f'mi-v1-{digest}.pt'
        source.parent.mkdir()
        source.write_bytes(b'corrupt fixture; do not replace')
    corrupt_sha = runner.sha(source) if change.startswith('corrupt_') else None
    with pytest.raises((ValueError, RuntimeError)):
        runner.reuse_mi(new, old)
    assert not (new / 'mi_receipt.json').exists()
    if corrupt_sha is not None:
        assert runner.sha(source) == corrupt_sha
