"""TRAIN-only score response preparation, with no coupling-label access."""
from dataclasses import asdict
import json

import pytest
import torch

from experiments import prepare_ising_physical_response as prepare
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import ModelConfig
from scripts.diagnose_ising_mcrg import local_feature_means


@pytest.fixture(params=[1, 3])
def model(request):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        result = build_model(ModelConfig(dimension=2, block_factor=2,
            theory_dim=request.param, hidden=5, width=3,
            binary_lookup=True, kernel_sampling='bernoulli'), 2).double().eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def spins(n=31, side=8, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(621)
    return (2 * torch.randint(0, 2, (n, side, side), generator=generator, device=device) - 1).double()


def make_source(tmp_path, dimension=1):
    source = tmp_path / 'source' / f'dim{dimension}' / 'run'
    source.mkdir(parents=True)
    data = tmp_path / 'existing.pt'
    # Expanded tiny storages preserve the full metadata shape without creating
    # any physical samples or a multi-gigabyte synthetic unit-test fixture.
    train = torch.ones((1, 1, 1, 1, 1), dtype=torch.int8).expand(516, 128, 108, 108, 1)
    test = torch.ones((1, 1, 1, 1, 1), dtype=torch.int8).expand(252, 128, 108, 108, 1)
    torch.save(dict(schema_version=1, train=dict(configurations=train, couplings=None),
                    test=dict(configurations=test, couplings=None)), data)
    model_cfg = ModelConfig(dimension=2, block_factor=2, theory_dim=dimension,
        hidden=5, width=3, binary_lookup=True, kernel_sampling='bernoulli')
    config = dict(schema_version=5, data=dict(cache=str(data)), model=asdict(model_cfg),
        training=dict(max_rg_steps=2, epochs=12000, train_modules=['Q', 'C', 'R'],
                      gradient_mode='qc_r_separate'), evaluation=dict(max_rg_steps=2))
    metadata = dict(config=config, data_sha256=prepare.EXPECTED_DATA_SHA256,
                    labels_used_in_training=False, q_supervision=None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        original = build_model(model_cfg, 2).double().eval()
    torch.save(dict(epoch=12000, metadata=metadata, model=qcr_state_dict(original)),
               source / 'training_step_12000.pt')
    (source / 'manifest.json').write_text(json.dumps(metadata))
    (source / 'summary.json').write_text(json.dumps(dict(config=config,
        actual_epochs=12000, requested_epochs=12000, final_report_skipped=True)))
    job = dict(status='complete', workflow_complete=True,
        training={f'dim{dimension}': dict(status='complete', actual_epochs=12000,
            manifest_sha256=prepare.sha256_file(source / 'manifest.json'))})
    (source.parent.parent / 'job.json').write_text(json.dumps(job))
    return source, data, original


def test_gpu_safe_pattern_table_matches_existing_exact_cpu_Q(model):
    samples = spins()
    before = prepare.model_sha256(model)
    table = prepare.q_feature_table(model)
    actual = prepare.local_feature_means_device(samples, table)
    expected = local_feature_means(model, samples)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    with torch.no_grad():
        encoded = model.theory_encoder(samples[None, ..., None])[0]
        pooled = model.theory_encoder.head(actual.mean(0))
    assert torch.allclose(encoded, pooled, atol=2e-15, rtol=2e-14)
    assert prepare.model_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonlinear_head_covariance_is_exact_empirical_tilt_derivative(model):
    samples = spins()
    features = prepare.local_feature_means_device(samples, prepare.q_feature_table(model))
    scores = prepare.operator_values(samples)[:, :6]
    before = prepare.model_sha256(model)
    g, response, std, jacobian = prepare.physical_response(model.theory_encoder.head, features, scores)
    record = prepare.validate_derivatives(model.theory_encoder.head, features, scores, jacobian, response)
    assert g.shape == (model.theory_encoder.head[-1].out_features,)
    assert response.shape == (len(g), 6)
    assert std.shape == (6,)
    assert record['head']['max_absolute_error'] < 1e-8
    assert record['physical_response']['relative_frobenius_error'] < 1e-6
    assert prepare.model_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_population_covariance_and_current_level_score_not_micro_score():
    head = torch.nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2., -1.], [.3, 4.]], dtype=torch.float64))
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    features = torch.tensor([[1., 3.], [2., -1.], [4., 2.]], dtype=torch.float64)
    current = torch.tensor([[3., 4.], [1., 0.], [2., -2.]], dtype=torch.float64)
    _, response, _, _ = prepare.physical_response(head, features, current)
    centered = (features - features.mean(0)).T @ (current - current.mean(0)) / len(features)
    assert torch.allclose(response, head.weight @ centered)
    assert not torch.allclose(response, head.weight @ prepare.covariance(features, current, ddof=1))
    microscopic = current.square()
    _, wrong, _, _ = prepare.physical_response(head, features, microscopic)
    assert not torch.allclose(response, wrong)


def test_zero_score_has_zero_response_no_division_or_nan(model):
    features = torch.arange(31 * 5, dtype=torch.float64).reshape(31, 5) / 100
    width = model.theory_encoder.head[0].in_features
    features = features[:, :1].expand(31, width)
    score = torch.ones(31, 6, dtype=torch.float64)
    _, response, std, _ = prepare.physical_response(model.theory_encoder.head, features, score)
    finite = prepare.finite_difference_response(model.theory_encoder.head, features, score)
    assert torch.equal(response, torch.zeros_like(response))
    assert torch.equal(std, torch.zeros_like(std))
    assert torch.equal(finite, response)


class ConfigurationsOnly(dict):
    def __getitem__(self, key):
        if key == 'couplings':
            pytest.fail('Coupling labels must not be accessed')
        return super().__getitem__(key)


def test_mmap_reads_configuration_fields_but_never_coupling_arrays(monkeypatch):
    train = torch.ones(1, 1, 1, 1, 1).expand(516, 128, 108, 108, 1)
    test = torch.ones(1, 1, 1, 1, 1).expand(252, 128, 108, 108, 1)
    payload = dict(schema_version=1, train=ConfigurationsOnly(configurations=train),
                   test=ConfigurationsOnly(configurations=test))
    def fake_load(path, **kwargs):
        assert kwargs == dict(map_location='cpu', weights_only=True, mmap=True)
        return payload
    monkeypatch.setattr(prepare.torch, 'load', fake_load)
    assert prepare.mmap_configurations('unread.pt')['train'] is train
    assert prepare.mmap_configurations('unread.pt')['test'] is test


def test_mmap_historical_tuple_reads_only_indices_zero_and_two(monkeypatch):
    class Archive(tuple):
        def __getitem__(self, index):
            if index in (1, 3):
                pytest.fail('Must not index historical coupling arrays')
            return super().__getitem__(index)
    train = torch.ones(1, 1, 1, 1, 1).expand(516, 128, 108, 108, 1)
    test = torch.ones(1, 1, 1, 1, 1).expand(252, 128, 108, 108, 1)
    monkeypatch.setattr(prepare.torch, 'load', lambda *_args, **_kwargs: Archive((train, None, test, None)))
    result = prepare.mmap_configurations('unread.pt')
    assert result['train'] is train and result['test'] is test


def test_mmap_rejects_wrong_shape_without_fallback(monkeypatch):
    monkeypatch.setattr(prepare.torch, 'load', lambda *_a, **_k:
        dict(schema_version=1, train=dict(configurations=torch.ones(2, 3)),
             test=dict(configurations=torch.ones(2, 3))))
    with pytest.raises(ValueError, match='shape'):
        prepare.mmap_configurations('missing.pt')


def test_draw_streams_are_index_based_separate_and_do_not_use_global_rng():
    values = [prepare.draw_seed(20261016, split, index, draw)
              for split in ('train', 'test') for index in (0, 1, 515) for draw in range(5)]
    assert len(set(values)) == len(values)
    assert prepare.draw_seed(20261016, 'train', 0, 0) == 20261016
    assert prepare.draw_seed(20261016, 'test', 0, 0) == 1020261016


@pytest.mark.parametrize('dimension', [1, 3])
def test_collect_smoke_cache_layout_receipts_and_no_R_or_label_inference(tmp_path, monkeypatch, dimension):
    source, data, _ = make_source(tmp_path, dimension)
    output = tmp_path / 'separate_smoke'
    calls = []
    def fake_collect(model, samples, table, probabilities, seeds, *, spotcheck):
        assert tuple(samples.shape) == (128, 108, 108)
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        assert len(seeds) == 1
        calls.append((seeds[0], spotcheck))
        return dict(direct=torch.ones(1, 3, dimension, dtype=torch.float64),
            vin=torch.ones(1, 2, dimension, 6, dtype=torch.float64),
            vout=2 * torch.ones(1, 2, dimension, 6, dtype=torch.float64),
            score_std=torch.ones(1, 2, 6, dtype=torch.float64)), []
    monkeypatch.setattr(prepare, 'collect_ensemble', fake_collect)
    # Hashing the 9GB physical source is forbidden; only its saved identity is read.
    original_hash = prepare.sha256_file
    def guarded_hash(path):
        assert str(path) != str(data)
        return original_hash(path)
    monkeypatch.setattr(prepare, 'sha256_file', guarded_hash)
    receipt = prepare.collect(source, output, device='cpu', smoke=True)
    assert receipt['status'] == 'complete' and receipt['workflow_complete'] is True
    assert receipt['source_data_rehashed'] is False and receipt['K_label_access'] is False
    assert receipt['no_R_inference'] and receipt['no_optimizer'] and receipt['no_MI']
    assert receipt['c_repeats'] == 1 and receipt['requested_c_repeats'] == 5
    assert receipt['model_sha256_before'] == receipt['model_sha256_after']
    assert receipt['frozen_QC_sha256_before'] == receipt['frozen_QC_sha256_after']
    payload = torch.load(source / 'training_step_12000.pt', weights_only=True)
    assert receipt['frozen_QC_sha256_before'] == prepare.state_sha256(payload['model'])
    assert receipt['split_counts'] == dict(train=2, test=2)
    cache = torch.load(output / 'response_cache.pt', weights_only=True)
    for split in ('train', 'test'):
        assert cache[f'{split}_direct'].shape == (2, 1, 3, dimension)
        assert cache[f'{split}_vin'].shape == cache[f'{split}_vout'].shape == (2, 1, 2, dimension, 6)
        assert cache[f'{split}_score_std'].shape == (2, 1, 2, 6)
        assert cache[f'{split}_source_indices'].tolist() == [0, 1]
        assert cache[f'{split}_c_draw_seeds'].shape == (2, 1)
    assert [spotcheck for _, spotcheck in calls] == [True, False, True, False]
    assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    assert not (output / 'response_cache.pt.tmp').exists()


def test_existing_output_and_nested_source_rejected_before_loading(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    monkeypatch.setattr(prepare, 'load_run', lambda *_args: pytest.fail('Must guard first'))
    with pytest.raises(FileExistsError):
        prepare.collect(source, source, device='cpu')
    with pytest.raises(ValueError, match='separate'):
        prepare.collect(source, source / 'child', device='cpu')


def test_checkpoint_pin_rejected_before_loading_model(tmp_path, monkeypatch):
    source, _, _ = make_source(tmp_path)
    monkeypatch.setattr(prepare, 'load_run', lambda *_args: pytest.fail('Must check fingerprint first'))
    with pytest.raises(ValueError, match='fingerprint'):
        prepare.collect(source, tmp_path / 'new', device='cpu', expected_checkpoint_sha256='incorrect')


def test_incomplete_parent_receipt_rejected_without_output(tmp_path):
    source, _, _ = make_source(tmp_path)
    path = source.parent.parent / 'job.json'
    job = json.loads(path.read_text()); job['workflow_complete'] = False
    path.write_text(json.dumps(job))
    output = tmp_path / 'new'
    with pytest.raises(ValueError, match='workflow'):
        prepare.collect(source, output, device='cpu', smoke=True)
    assert not output.exists()


def test_failed_collection_preserves_error_without_cache_or_retry(tmp_path, monkeypatch):
    source, _, _ = make_source(tmp_path)
    output = tmp_path / 'new'
    def failed(*_args, **_kwargs):
        raise RuntimeError('test collection interruption')
    monkeypatch.setattr(prepare, 'collect_ensemble', failed)
    with pytest.raises(RuntimeError, match='interruption'):
        prepare.collect(source, output, device='cpu', smoke=True)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed' and status['workflow_complete'] is False
    assert 'interruption' in status['error']
    assert not (output / 'response_cache.pt').exists()
    with pytest.raises(FileExistsError):
        prepare.collect(source, output, device='cpu', smoke=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA for device-index regression')
def test_cuda_offsets_features_and_weighted_response_match_cpu(model):
    samples = spins()
    table = prepare.q_feature_table(model)
    cpu = prepare.local_feature_means_device(samples, table)
    model = model.cuda()
    gpu = prepare.local_feature_means_device(samples.cuda(), prepare.q_feature_table(model))
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-14, rtol=1e-12)
    scores = prepare.operator_values(samples.cuda())[:, :6]
    _, response, _, jacobian = prepare.physical_response(model.theory_encoder.head, gpu, scores)
    checked = prepare.validate_derivatives(model.theory_encoder.head, gpu, scores, jacobian, response)
    assert checked['physical_response']['relative_frobenius_error'] < 1e-6
