"""有限扰动统计缓存：物理标签隔离、当前尺度响应、冻结及独立尺寸。"""
import hashlib
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from experiments import prepare_ising_finite_perturbation as prepare
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig


@pytest.fixture
def model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        value = build_model(ModelConfig(dimension=2, block_factor=2, theory_dim=3,
            hidden=5, width=3, binary_lookup=True, kernel_sampling='bernoulli'), 2).double().eval()
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    return value


def spins(n=13, size=16):
    generator = torch.Generator().manual_seed(361+size)
    return (2*torch.randint(0, 2, (n, size, size), generator=generator)-1).double()


def spec(index=0, logical_size=108, *, role='train_domain', family='zero'):
    depth = 2 if logical_size == 108 else 1
    return dict(condition_index=index, condition_id=f'c{index:03d}', id=f'c{index:03d}',
        role=role, family=family, size=4*2**depth, logical_size=logical_size,
        depth=depth, K0=.123456789, unit_delta6=[0.]*6)


def write_condition(root, condition):
    folder = root/'conditions'/condition['condition_id']
    folder.mkdir(parents=True)
    tensor = spins(16, condition['size']).reshape(2, 8, condition['size'], condition['size']).to(torch.int8)
    provenance = dict(condition=condition, seed=1234)
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    torch.save(dict(schema_version=1, configurations=tensor,
                    metadata_fingerprint=fingerprint, K0=torch.tensor(987.)), folder/'data.pt')
    sidecar = dict(condition, schema_version=1, status='complete', workflow_complete=True,
        smoke=True, configuration_shape=list(tensor.shape), data_sha256='b'*64,
        metadata_fingerprint=fingerprint, provenance=provenance, condition=condition)
    (folder/'sidecar.json').write_text(json.dumps(sidecar))
    status = dict(status='complete', workflow_complete=True, data_sha256='b'*64,
        metadata_fingerprint=fingerprint, sidecar_sha256=prepare.sha256_file(folder/'sidecar.json'))
    (folder/'status.json').write_text(json.dumps(status))
    return tensor, sidecar, status


def minimal_cache():
    return dict(schema_version=1, direct=torch.zeros(4, 1, 3, 3, dtype=torch.float64),
        vin=torch.zeros(4, 1, 2, 3, 6, dtype=torch.float64),
        vout=torch.zeros(4, 1, 2, 3, 6, dtype=torch.float64),
        score_std=torch.ones(4, 1, 2, 6, dtype=torch.float64),
        train_mask=torch.tensor([True, False, True, False]),
        baseline_mask=torch.tensor([True, True, False, False]),
        condition_index=torch.tensor([0, 0, 1, 1]), replica_index=torch.tensor([0, 1, 0, 1]),
        condition_opaque_id=['c000', 'c000', 'c001', 'c001'],
        c_draw_seeds=torch.arange(8).reshape(4, 1, 2), size=16, logical_size=108,
        depth=2, smoke=True)


def test_namespaced_C_streams_are_size_condition_replica_draw_transition_independent():
    seeds = [prepare.c_seed(20261017, size, point, replica, draw, step)
        for size in (54, 108) for point in range(3) for replica in range(2)
        for draw in range(5) for step in range(2)]
    assert len(seeds) == len(set(seeds)) and max(seeds) < 2**63
    assert prepare.c_seed(20261017, 108, 1, 0, 0, 0) == prepare.c_seed(20261017, 108, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        prepare.c_seed(20261017, 108, .44, 0, 0, 0)


@pytest.mark.parametrize('depth', [1, 2])
def test_real_paired_collection_keeps_frozen_state_and_complete_DR(model, depth):
    size = 4*2**depth
    raw = spins(size=size)
    table = prepare.q_feature_table(model)
    _, probabilities = prepare.c_probability_table(model)
    seeds = torch.arange(2*depth).reshape(2, depth)+1700
    before = prepare.model_sha256(model)
    row, reference, checks = prepare.collect_ensemble(model, raw, table, probabilities,
        depth=depth, seeds=seeds, spotcheck=True)
    assert row['direct'].shape == (2, depth+1, 3)
    assert row['vin'].shape == row['vout'].shape == (2, depth, 3, 6)
    assert row['score_std'].shape == (2, depth, 6)
    assert reference['source_full_DR'].shape == (2, depth+1, 3, 3)
    assert torch.allclose(reference['source_full_DR'], reference['source_r_fd_jacobian'], atol=1e-8)
    assert reference['source_all_start_horizon_pred'].shape == (2, depth+1, depth+1, 3)
    assert len(checks) == 2*depth
    assert all(not item.requires_grad and item.dtype == torch.float64
        for item in (*row.values(), *reference.values()))
    assert prepare.model_sha256(model) == before
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    # Reconstruct exactly the same C noise; score at transition1 must be current,
    # and not microscopic.  This also checks population (not ddof1) covariance.
    levels = [raw]
    for step in range(depth):
        side = levels[-1].shape[-1]//2
        generator = torch.Generator().manual_seed(int(seeds[0, step]))
        uniform = torch.rand((len(raw), side, side), dtype=torch.float64, generator=generator)
        levels.append(prepare.c_draw(levels[-1], probabilities, uniform))
    head = model.theory_encoder.head
    for step in range(depth):
        score = prepare.operator_values(levels[step])[:, :6]
        features = prepare.local_feature_means_device(levels[step+1], table)
        _, jacobian = prepare.head_value_jacobian(head, features.mean(0))
        expected = jacobian @ prepare.covariance(features, score, ddof=0)
        assert torch.allclose(row['vout'][0, step], expected, atol=1e-14)
        assert torch.allclose(row['score_std'][0, step], score.std(0, unbiased=False))
        if step:
            wrong = jacobian @ prepare.covariance(features, prepare.operator_values(raw)[:, :6], ddof=0)
            assert not torch.allclose(expected, wrong, atol=1e-8)


def test_training_cache_allowlist_excludes_any_generation_label_or_metadata():
    cache = minimal_cache()
    assert prepare.validate_training_cache(cache) is cache
    for field in ('K', 'K0', 'delta', 'epsilon', 'family', 'source_config', 'metadata', 'couplings'):
        with pytest.raises(ValueError, match='allowlist'):
            prepare.validate_training_cache(dict(cache, **{field: .44}))
    bad = dict(cache, condition_opaque_id=['K=.44']*4)
    with pytest.raises(ValueError, match='opaque'):
        prepare.validate_training_cache(bad)
    bad = dict(cache, direct=cache['direct'].float())
    with pytest.raises(ValueError, match='float64'):
        prepare.validate_training_cache(bad)
    bad = dict(cache, direct=cache['direct'].clone().requires_grad_(True))
    with pytest.raises(ValueError, match='detached'):
        prepare.validate_training_cache(bad)


def test_evaluation_only_independent_size_cannot_be_marked_training():
    cache = minimal_cache()
    cache.update(logical_size=54, size=8, depth=1,
        direct=cache['direct'][:, :, :2], vin=cache['vin'][:, :, :1],
        vout=cache['vout'][:, :, :1], score_std=cache['score_std'][:, :, :1],
        c_draw_seeds=cache['c_draw_seeds'][:, :, :1])
    with pytest.raises(ValueError, match='evaluation-only'):
        prepare.validate_training_cache(cache)
    cache['train_mask'].zero_()
    prepare.validate_training_cache(cache)


def test_condition_only_accesses_configurations_and_small_fingerprints(tmp_path, monkeypatch):
    condition = spec()
    raw, sidecar, status = write_condition(tmp_path, condition)
    class ConfigurationsOnly(dict):
        def __getitem__(self, key):
            if key not in ('configurations',):
                pytest.fail('Physical generation labels must never be indexed from data payload')
            return super().__getitem__(key)
    payload = ConfigurationsOnly(schema_version=1, configurations=raw, K0=torch.tensor(.44),
        metadata_fingerprint=sidecar['metadata_fingerprint'])
    def loader(path, **options):
        assert options == dict(map_location='cpu', weights_only=True, mmap=True)
        return payload
    monkeypatch.setattr(prepare.torch, 'load', loader)
    original_hash = prepare.sha256_file
    def small_only(path):
        assert PathName(path) != 'data.pt'
        return original_hash(path)
    monkeypatch.setattr(prepare, 'sha256_file', small_only)
    actual, mapping, receipt = prepare.validate_condition(tmp_path, condition, smoke=True)
    assert actual is raw and mapping == sidecar
    assert receipt['physical_data_rehashed'] is False


def PathName(path):
    from pathlib import Path
    return Path(path).name


@pytest.mark.parametrize('failure', ['sidecar_sha', 'metadata', 'role'])
def test_condition_receipt_mismatches_fail_before_collection(tmp_path, failure):
    condition = spec()
    write_condition(tmp_path, condition)
    folder = tmp_path/'conditions'/'c000'
    status = json.loads((folder/'status.json').read_text())
    if failure == 'sidecar_sha':
        status['sidecar_sha256'] = 'f'*64
    elif failure == 'metadata':
        status['metadata_fingerprint'] = 'f'*64
    else:
        condition = dict(condition, role='heldout_domain')
    (folder/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError):
        prepare.validate_condition(tmp_path, condition, smoke=True)


def test_output_refuses_overwrite_and_failed_directory_is_preserved(tmp_path):
    output = tmp_path/'failed'
    output.mkdir()
    (output/'status.json').write_text('{"status":"failed"}')
    with pytest.raises(FileExistsError):
        prepare.collect(tmp_path/'source', tmp_path/'data', output, 'a'*64)
    assert json.loads((output/'status.json').read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='separate'):
        prepare._new_output(tmp_path/'source'/'nested', (tmp_path/'source',))


def test_unauthorized_GPU_fails_without_source_or_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, '_validate_source', lambda *a: pytest.fail('Source must not be read'))
    with pytest.raises(ValueError, match='authorized'):
        prepare.collect(tmp_path/'source', tmp_path/'data', tmp_path/'failed', 'a'*64, device='cuda:0')
    status = json.loads((tmp_path/'failed'/'status.json').read_text())
    assert status['status'] == 'failed' and not status['workflow_complete']


def test_CUDA_memory_cap_is_recorded_and_never_automatically_repaired(monkeypatch):
    monkeypatch.setattr(prepare.torch.cuda, 'max_memory_allocated', lambda device: 7*1024**3)
    monkeypatch.setattr(prepare.torch.cuda, 'max_memory_reserved', lambda device: 7*1024**3)
    assert prepare._cuda_memory(torch.device('cuda:2'))['uses_cuda']
    monkeypatch.setattr(prepare.torch.cuda, 'max_memory_reserved', lambda device: 9*1024**3)
    with pytest.raises(MemoryError, match='8 GiB'):
        prepare._cuda_memory(torch.device('cuda:2'))


def test_complete_cache_has_only_detached_opaque_training_fields(tmp_path, monkeypatch, model):
    source, data, output = tmp_path/'source', tmp_path/'data', tmp_path/'cache'
    source.mkdir(); data.mkdir()
    conditions = [spec(index, 108 if index < 41 else 54,
        role='train_domain' if index < 14 else 'heldout_domain',
        family='zero' if index in (0, 7) else 'axis_unit') for index in range(68)]
    for condition in conditions:
        write_condition(data, condition)
    fake = SimpleNamespace(model=model)
    monkeypatch.setattr(prepare, '_validate_source', lambda *a: (fake, {}, dict(
        source_checkpoint_sha256='a'*64, source_epoch=12000, source_training_actual_updates=12000)))
    monkeypatch.setitem(sys.modules, 'scripts.ising_finite_perturbation_data',
        SimpleNamespace(condition_specs=lambda smoke: conditions))
    def cheap_collect(model, raw, table, probabilities, *, depth, seeds, spotcheck):
        draws = len(seeds)
        row = dict(direct=torch.ones(draws, depth+1, 3, dtype=torch.float64),
            vin=torch.ones(draws, depth, 3, 6, dtype=torch.float64),
            vout=torch.ones(draws, depth, 3, 6, dtype=torch.float64),
            score_std=torch.ones(draws, depth, 6, dtype=torch.float64))
        return row, dict(source_full_DR=torch.eye(3, dtype=torch.float64).expand(draws, depth+1, 3, 3)), []
    monkeypatch.setattr(prepare, 'collect_ensemble', cheap_collect)
    result = prepare.collect(source, data, output, 'a'*64, smoke=True)
    assert result['status'] == 'complete' and result['workflow_complete']
    assert result['no_K_labels_in_training_cache'] and result['all_parameter_grads_none']
    assert result['source_frozen_QC_sha256_before'] == result['source_frozen_QC_sha256_after']
    assert result['initial_R_sha256'] == result['source_R_sha256_after']
    for filename in ('training_L108.pt', 'evaluation_L54.pt'):
        cache = torch.load(output/filename, weights_only=True)
        assert set(cache) == prepare.CACHE_FIELDS
        assert result['cache_sha256'][filename] == prepare.sha256_file(output/filename)
        prepare.validate_training_cache(cache)
    training = torch.load(output/'training_L108.pt', weights_only=True)
    assert training['direct'].shape == (82, 1, 3, 3)
    assert training['train_mask'].sum() == 14
    assert training['baseline_mask'].sum() == 4
    assert training['replica_index'][training['train_mask']].unique().tolist() == [0]
    assert training['condition_index'][training['train_mask']].unique().tolist() == list(range(14))
    evaluation = torch.load(output/'evaluation_L54.pt', weights_only=True)
    assert evaluation['direct'].shape == (54, 1, 2, 3)
    assert not evaluation['train_mask'].any()
    assert not evaluation['baseline_mask'].any()
    metadata = json.loads((output/'metadata.json').read_text())
    assert metadata['generation_mapping_evaluation_only'] and len(metadata['conditions']) == 68
    assert metadata['conditions'][0]['K0'] == .123456789
    with np.load(output/'collected_statistics.npz', allow_pickle=False) as arrays:
        assert arrays['L108_direct'].shape == training['direct'].shape
        assert arrays['L54_direct'].shape == evaluation['direct'].shape
        assert arrays['L108_condition_opaque_id'].dtype.kind == 'U'
        assert 'L108_source_full_DR' in arrays


@pytest.mark.parametrize('visible', ['2,3', '3,2,1,0', 'GPU-abc', '-1', ''])
def test_cuda_remapping_is_rejected_without_modifying_environment(monkeypatch, visible):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(prepare.torch.cuda, 'get_device_properties',
        lambda *args: pytest.fail('Remapped GPU must not be initialized'))
    with pytest.raises(ValueError, match='remapping'):
        prepare.validate_device('cuda:2')
    assert prepare.os.environ['CUDA_VISIBLE_DEVICES'] == visible


def test_cuda_physical_uuid_must_equal_torch_ordinal(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    monkeypatch.setattr(prepare.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(prepare.torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(prepare.torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(uuid='GPU-abcd-1234'))
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='GPU-abcd-1234\n'))
    device, identity = prepare.validate_device('cuda:2')
    assert device == torch.device('cuda:2')
    assert identity['GPU_mapping_checked'] and identity['physical_gpu_index'] == 2
    monkeypatch.setattr(prepare.torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(uuid='GPU-wrong'))
    with pytest.raises(ValueError, match='UUID'):
        prepare.validate_device('cuda:2')


def test_fixed_execution_benchmark_saves_no_random_data_or_model(tmp_path, monkeypatch, model):
    source = tmp_path/'source'
    source.mkdir()
    fake = SimpleNamespace(model=model)
    monkeypatch.setattr(prepare, '_validate_source', lambda *a: (fake, {}, dict(
        source_checkpoint_sha256='a'*64, source_epoch=12000, source_training_actual_updates=12000)))
    calls = []
    def cheap_collect(model, synthetic, table, probabilities, *, depth, seeds, spotcheck):
        assert synthetic.shape == (128, 108, 108) and synthetic.dtype == torch.float64
        assert torch.all(synthetic.abs() == 1)
        assert depth == 2 and seeds.shape == (1, 2) and spotcheck
        calls.append(bool(torch.all(synthetic == 1)))
        return {}, {}, []
    monkeypatch.setattr(prepare, 'collect_ensemble', cheap_collect)
    before = prepare.model_sha256(model)
    result = prepare.benchmark(source, tmp_path/'benchmark', 'a'*64, device='cpu')
    assert len(calls) == 10
    assert calls == [False]*5+[True]*5
    assert result['warmup_iterations'] == 2 and result['measured_iterations'] == 8
    assert [row['case'] for row in result['cases']] == [
        'iid_binary', 'all_plus_binary_histogram_contention']
    assert result['warmup_iterations_per_case'] == 1 and result['measured_iterations_per_case'] == 4
    assert result['projected_collection_gpu_seconds_with_margin'] == 0.
    assert result['projected_collection_wall_seconds_with_margin'] == (
        max(result['measured_single_draw_wall_seconds']) * 816 * 5 * 2)
    assert result['projected_collection_cpu_seconds_with_margin'] == result['projected_collection_wall_seconds_with_margin']
    assert result['model_sha256_before'] == result['model_sha256_after'] == before
    assert result['all_parameter_grads_none'] and result['scientific_result'] is False
    assert sorted(path.name for path in (tmp_path/'benchmark').iterdir()) == ['benchmark.json', 'status.json']
    assert not result['saved_synthetic_data'] and not result['saved_model']
    with pytest.raises(FileExistsError):
        prepare.benchmark(source, tmp_path/'benchmark', 'a'*64)


def test_benchmark_cli_is_explicit_and_does_not_require_physical_data(monkeypatch):
    calls = []
    monkeypatch.setattr(prepare, 'benchmark', lambda **kwargs: calls.append(kwargs))
    prepare.main(['--benchmark', '--source-run', '/source', '--output', '/new',
                  '--expected-checkpoint-sha256', 'a'*64, '--device', 'cpu'])
    assert len(calls) == 1 and 'data_root' not in calls[0] and 'smoke' not in calls[0]
    with pytest.raises(SystemExit):
        prepare.main(['--benchmark', '--source-run', '/source', '--output', '/new',
            '--expected-checkpoint-sha256', 'a'*64, '--data-root', '/physical'])
