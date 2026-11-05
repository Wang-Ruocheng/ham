"""新独立尺寸诊断的只读、配对响应与不覆盖契约；无训练或物理采样。"""
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from scripts import diagnose_ising_matched_sizes as diagnostic


class FakeModel(nn.Module):
    """Small analytic surrogate, so collector tests never load formal checkpoints."""
    def __init__(self, dimension):
        super().__init__()
        self.theory_encoder = nn.Module()
        self.theory_encoder.head = nn.Linear(4, dimension, bias=False, dtype=torch.float64)
        self.theory_rg = nn.Linear(dimension, dimension, dtype=torch.float64)
        self.c_bias = nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        with torch.no_grad():
            self.theory_encoder.head.weight.copy_(torch.arange(1, 4*dimension+1).reshape(dimension, 4)/13.)
            self.theory_rg.weight.copy_(torch.eye(dimension, dtype=torch.float64)*.2)
            self.theory_rg.bias.fill_(.03)

    def advance_theory(self, g):
        return g+torch.tanh(self.theory_rg(g))


@pytest.fixture(autouse=True)
def fake_local_operations(monkeypatch):
    def features(model, spins):
        s = spins.double()
        mean = s.mean((-2, -1))
        return torch.stack((mean, (s*torch.roll(s, -1, -1)).mean((-2, -1)),
            (s*torch.roll(s, (-1, -1), (-2, -1))).mean((-2, -1)), mean.square()), -1)

    def probabilities(model):
        patterns = 2*((torch.arange(16)[:, None] >> torch.arange(4)) % 2).double()-1
        return patterns, torch.sigmoid(patterns.sum(-1)*.2+model.c_bias.detach())

    monkeypatch.setattr(diagnostic, 'local_feature_means', features)
    monkeypatch.setattr(diagnostic, 'c_probability_table', probabilities)


def small_data(depth=1, replicas=2, samples=12):
    side = 4*2**depth
    generator = torch.Generator().manual_seed(91+side)
    spins = (2*torch.randint(0, 2, (3, replicas, samples, side, side), generator=generator)-1).to(torch.int8)
    K = torch.tensor([.43, .4406867935, .45], dtype=torch.float64)
    seeds = torch.arange(3*replicas, dtype=torch.int64).reshape(3, replicas)+1000*side
    return spins, K, seeds


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('depth', [1, 2])
def test_dimension_depth_shapes_freezing_closure_and_response(tmp_path, dimension, depth):
    model = FakeModel(dimension)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = diagnostic.model_sha256(model)
    spins, K, seeds = small_data(depth)
    output = tmp_path/'new'
    result = diagnostic.collect_model(model, spins, K, seeds, output,
        size=4*2**depth, depth=depth, c_repeats=2, smoke=True)
    assert result['status'] == 'complete'
    assert result['model_sha256_before'] == result['model_sha256_after'] == before
    assert result['independent_replicas'] == 2
    assert result['no_optimizer'] and result['no_training'] and result['no_MI']
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with np.load(output/'operator_statistics.npz', allow_pickle=False) as arrays:
        prefix, levels = (3, 2, 2), depth+1
        assert arrays['g'].shape == prefix+(levels, dimension)
        assert arrays['full_DR'].shape == prefix+(levels, dimension, dimension)
        assert arrays['operators'].shape == prefix+(levels, 12, 7)
        assert arrays['orientation_operators'].shape == arrays['operators'].shape
        assert arrays['covariance'].shape == prefix+(levels, 7, 7)
        assert arrays['cross_covariance'].shape == prefix+(depth, 7, 7)
        assert arrays['q_response_in'].shape == prefix+(depth, dimension, 7)
        assert arrays['q_fd_response_out'].shape == prefix+(depth, 2, dimension, 7)
        assert arrays['score_std'].shape == prefix+(levels, 7)
        assert arrays['Cdraw_seeds'].shape == prefix+(depth,)
        assert np.array_equal(arrays['chain_seeds'], seeds.numpy())
        assert np.array_equal(arrays['lattice_sizes'], [4*2**(depth-i) for i in range(levels)])
        assert np.allclose(arrays['full_DR'], arrays['r_fd_jacobian'], atol=1e-9, rtol=1e-8)
        assert np.allclose(arrays['q_response_in'], arrays['q_fd_response_in'][..., 0, :, :], atol=1e-5, rtol=1e-4)
        assert np.allclose(arrays['q_response_predicted'], np.einsum('...ij,...jo->...io',
            arrays['full_DR'][..., :-1, :, :], arrays['q_response_in']))
        assert np.allclose(arrays['r_displacement'], arrays['r_of_g']-arrays['g'])
        assert np.allclose(arrays['one_step_pred'], arrays['r_of_g'][..., :-1, :])
        assert np.allclose(arrays['one_step_residual'], arrays['one_step_pred']-arrays['g'][..., 1:, :])
        assert np.allclose(arrays['mean_r_of_g'], arrays['r_of_g'].mean((1, 2)))
        mean_g = arrays['g'].mean((1, 2))
        with torch.no_grad():
            assert np.allclose(arrays['r_of_mean_g'], model.advance_theory(torch.from_numpy(mean_g)).numpy())
        for start in range(levels):
            for end in range(levels):
                assert arrays['all_start_horizon_valid'][start, end] == (start <= end)
                pred = arrays['all_start_horizon_pred'][..., start, end, :]
                if start <= end:
                    assert np.allclose(arrays['all_start_horizon_residual'][..., start, end, :],
                        pred-arrays['g'][..., end, :])
                    expected = torch.from_numpy(arrays['g'][..., start, :].copy())
                    with torch.no_grad():
                        for _ in range(end-start):
                            expected = model.advance_theory(expected)
                    assert np.allclose(pred, expected.numpy())
                else:
                    assert np.count_nonzero(pred) == 0
        for family, count in (('even', 6), ('all', 7)):
            assert arrays[f'best_J_{family}'].shape == prefix+(depth, dimension, dimension)
            assert arrays[f'best_response_residual_{family}'].shape == prefix+(depth, dimension, count)
            assert arrays[f'response_input_rank_{family}'].shape == prefix+(depth,)
            assert arrays[f'response_input_projector_{family}'].shape == prefix+(depth, dimension, dimension)
        # ddof0 score scaling is not the ddof1 physical covariance convention.
        op = arrays['operators'][0, 0, 0, 0]
        assert np.allclose(arrays['score_std'][0, 0, 0, 0], op.std(0, ddof=0))
        assert np.allclose(arrays['covariance'][0, 0, 0, 0], np.cov(op, rowvar=False, ddof=1))
    assert json.loads((output/'status.json').read_text())['workflow_complete']


@pytest.mark.parametrize('dimension', [1, 3])
def test_full_R_identity_term_and_paired_C_noise(tmp_path, dimension):
    spins, K, seeds = small_data(1, replicas=1, samples=8)
    models = [FakeModel(dimension), FakeModel(dimension)]
    for model in models:
        with torch.no_grad():
            model.theory_rg.weight.zero_()
            model.theory_rg.bias.zero_()
    outputs = [tmp_path/'one', tmp_path/'two']
    for model, output in zip(models, outputs):
        diagnostic.collect_model(model, spins, K, seeds, output, size=8, depth=1, c_repeats=2, smoke=True)
    with np.load(outputs[0]/'operator_statistics.npz') as a, np.load(outputs[1]/'operator_statistics.npz') as b:
        assert np.allclose(a['full_DR'], np.eye(dimension))
        assert np.allclose(a['r_fd_jacobian'], np.eye(dimension), atol=2e-11)
        assert np.array_equal(a['Cdraw_seeds'], b['Cdraw_seeds'])
        assert np.array_equal(a['operators'], b['operators'])
        assert np.count_nonzero(a['r_displacement']) == 0


def test_noise_streams_do_not_share_sizes_or_replicas_and_do_not_depend_on_model():
    seeds = [diagnostic.c_seed(20261016, size, .4406867935, rep, draw, step)
        for size in (8, 16, 54, 108) for rep in range(2) for draw in range(5) for step in range(2)]
    assert len(set(seeds)) == len(seeds)
    assert max(seeds) < 2**63
    assert diagnostic.c_seed(20261016, 54, .4406867935, 0, 0, 0) == diagnostic.c_seed(20261016, 54, .4406867935, 0, 0, 0)


def test_existing_output_is_never_reused_even_when_failed(tmp_path, monkeypatch):
    output = tmp_path/'failed'
    output.mkdir()
    (output/'status.json').write_text('{"status":"failed"}')
    monkeypatch.setattr(diagnostic, '_collect_model', lambda *a, **k: pytest.fail('must not infer'))
    spins, K, seeds = small_data()
    with pytest.raises(FileExistsError):
        diagnostic.collect_model(FakeModel(1), spins, K, seeds, output, size=8, depth=1, smoke=True)
    assert json.loads((output/'status.json').read_text())['status'] == 'failed'


def test_invalid_inputs_leave_failed_receipt_not_a_retry(tmp_path):
    spins, K, seeds = small_data()
    output = tmp_path/'bad_dtype'
    with pytest.raises(ValueError, match='int8'):
        diagnostic.collect_model(FakeModel(1), spins.double(), K, seeds, output,
            size=8, depth=1, smoke=True)
    assert json.loads((output/'status.json').read_text())['status'] == 'failed'
    with pytest.raises(FileExistsError):
        diagnostic.collect_model(FakeModel(1), spins, K, seeds, output, size=8, depth=1, smoke=True)


def test_formal_geometry_and_budget_cannot_be_silently_smoke(tmp_path):
    spins, K, seeds = small_data()
    with pytest.raises(ValueError, match='end at L=27'):
        diagnostic.collect_model(FakeModel(1), spins, K, seeds, tmp_path/'formal',
            size=8, depth=1, smoke=False)


def test_changed_model_or_parameter_grad_during_collection_is_rejected(tmp_path, monkeypatch):
    spins, K, seeds = small_data(replicas=1, samples=4)
    original = diagnostic.full_r_finite_difference
    def mutate(model, g, h):
        value = original(model, g, h)
        model.c_bias.grad = torch.zeros_like(model.c_bias)
        return value
    monkeypatch.setattr(diagnostic, 'full_r_finite_difference', mutate)
    output = tmp_path/'mutation'
    with pytest.raises(RuntimeError, match='Read-only'):
        diagnostic.collect_model(FakeModel(1), spins, K, seeds, output, size=8, depth=1, c_repeats=1, smoke=True)
    assert json.loads((output/'status.json').read_text())['status'] == 'failed'
    assert not (output/'operator_statistics.npz').exists()


def test_checkpoint_pin_is_checked_before_source_loading_and_preserves_failure(tmp_path, monkeypatch):
    source, data = tmp_path/'source', tmp_path/'data'
    source.mkdir(); data.mkdir()
    (source/'training_step_12000.pt').write_bytes(b'not a checkpoint')
    output = tmp_path/'result'
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        diagnostic.collect(source, data, output, '0'*64)
    state = json.loads((output/'status.json').read_text())
    assert state['status'] == 'failed'
    assert 'fingerprint mismatch' in state['error']


def test_source_output_overlap_is_rejected_before_creation(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    with pytest.raises(ValueError, match='separate'):
        diagnostic.collect(source, tmp_path/'data', source/'diagnostic', '0'*64)
    assert not (source/'diagnostic').exists()


def data_receipt(tmp_path):
    path = tmp_path/'independent'; path.mkdir()
    spins, K, seeds = small_data()
    flips = seeds+5000
    options = dict(size=8, seed=20261016, replicas=2, samples=12, max_rg_steps=1,
        block_factor=2, target_size=4)
    provenance = dict(sampling_options=options, chain_seeds=seeds.tolist(),
        global_flip_seeds=flips.tolist(), K_values=K.tolist())
    metadata_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    metadata = dict(**options, depth=1, smoke=True, diagnostic_only=True,
        metadata_fingerprint=metadata_fingerprint)
    payload = dict(schema_version=1, configurations=spins, K=K, chain_seeds=seeds,
        global_flip_seeds=flips, **metadata)
    torch.save(payload, path/'data.pt')
    sidecar = dict(schema_version=1, status='complete', workflow_complete=True,
        configuration_shape=list(spins.shape), data_sha256='a'*64, provenance=provenance, **metadata)
    (path/'sidecar.json').write_text(json.dumps(sidecar))
    status = dict(status='complete', workflow_complete=True, data_sha256='a'*64,
        metadata_fingerprint=metadata_fingerprint, sidecar_sha256=diagnostic.sha256_file(path/'sidecar.json'))
    (path/'status.json').write_text(json.dumps(status))
    return path


def test_new_data_validates_small_receipts_without_rehashing_physical_file(tmp_path, monkeypatch):
    path = data_receipt(tmp_path)
    original_hash = diagnostic.sha256_file
    def small_only(value):
        assert str(value).endswith('sidecar.json'), 'must not rehash physical data'
        return original_hash(value)
    monkeypatch.setattr(diagnostic, 'sha256_file', small_only)
    data, sidecar, status = diagnostic.validate_data(path, smoke=True)
    assert data['configurations'].shape == (3, 2, 12, 8, 8)
    assert sidecar['data_sha256'] == status['data_sha256']


@pytest.mark.parametrize('fault', ['status', 'sidecar_hash', 'data_sha', 'metadata', 'depth'])
def test_new_data_rejects_receipt_or_geometry_mismatch(tmp_path, fault):
    path = data_receipt(tmp_path)
    status_path = path/'status.json'
    state = json.loads(status_path.read_text())
    if fault == 'status':
        state['status'] = 'failed'
    elif fault == 'sidecar_hash':
        state['sidecar_sha256'] = 'b'*64
    elif fault == 'data_sha':
        state['data_sha256'] = 'b'*64
    else:
        sidecar_path = path/'sidecar.json'
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['metadata_fingerprint' if fault == 'metadata' else 'depth'] = 'bad'
        sidecar_path.write_text(json.dumps(sidecar))
        state['sidecar_sha256'] = diagnostic.sha256_file(sidecar_path)
    status_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        diagnostic.validate_data(path, smoke=True)
