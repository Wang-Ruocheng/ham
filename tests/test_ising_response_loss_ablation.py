"""Saved-encoding-only R ablation: local derivatives, paired objectives and provenance."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import ising_response_loss_ablation as ablation
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig, RunConfig


def fixture_source(tmp_path, hidden=5):
    source, report = tmp_path/'source', tmp_path/'reports'/'epoch_9000'/'full_report'
    source.mkdir(); report.mkdir(parents=True)
    cfg = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache='/does/not/exist/physical_data.pt', physical_model='ising2d', coupling_names=['K']),
        model=dict(dimension=2, theory_dim=1, hidden=hidden, block_factor=2, affine_reconstruction=True),
        loss=dict(rollout_weight=5., weights=dict(commutator=5.)),
        training=dict(max_rg_steps=2, epochs=12000, gradient_mode='qc_r_separate'),
        evaluation=dict(max_rg_steps=2)))
    torch.manual_seed(19)
    model = build_model(cfg.model, 2)
    checkpoint = source/'training_step_9000.pt'
    torch.save(dict(epoch=9000, model=model.state_dict(), optimizer=dict(source_only='never restore'),
        metadata=dict(config=cfg.to_dict(), independent_c=True, data_sha256='a'*64,
            labels_used_in_training=True, q_supervision=dict(weight=10., scale=0, coupling_names=['K']))), checkpoint)
    encodings = {}
    grid = np.linspace(.2, .68, 16)
    for split, knots in [('train', grid[::2]), ('test', grid[1::2])]:
        k = np.repeat(knots, 12)
        noise = np.tile(np.linspace(-.01, .01, 12), len(knots))
        encodings[split] = np.stack([k+noise, .7*k+.03+noise*.5, .5*k+.04+noise*.25], axis=1)[:, :, None]
        encodings[split+'_labels'] = k
    np.savez_compressed(report/'encodings.npz', **encodings)
    checkpoint_sha = ablation.file_fingerprint(checkpoint)
    (report.parent/'checkpoint_summary.json').write_text(json.dumps(dict(
        actual_epochs=9000, checkpoint_sha256=checkpoint_sha)))
    (report.parent/'local_response.json').write_text(json.dumps(dict(source_sha256=dict(
        checkpoint=checkpoint_sha, encodings=ablation.file_fingerprint(report/'encodings.npz')))))
    return source, report, checkpoint


def test_cubic_derivative_matrix_is_exact_including_one_sided_edges():
    k = np.array([.2, .213, .24, .28, .32, .4, .49, .55, .68])
    D, boundary, diagnostics = ablation.local_derivative_matrix(k)
    actual = D.numpy() @ (2-3*k+4*k*k-5*k**3)
    np.testing.assert_allclose(actual, -3+8*k-15*k*k, atol=1e-12, rtol=1e-12)
    assert boundary.tolist() == [True]*3 + [False]*3 + [True]*3
    assert all(len(ids) == 7 for ids in diagnostics['neighbor_indices'])
    assert np.isfinite(diagnostics['condition_numbers']).all()


def test_closure_uses_original_g0_rollout_not_all_start_pairs():
    model = SimpleNamespace(advance_theory=lambda x: 2*x)
    direct = torch.tensor([[[1.], [3.], [11.]]], dtype=torch.double)
    terms = ablation.closure_terms(model, direct)
    assert float(terms['one_step_mse']) == 13.
    assert float(terms['rollout_mse']) == 25.
    assert float(terms['closure']) == 190.


def test_response_uses_mean_of_each_ensemble_R_not_R_of_the_mean():
    k = np.repeat(np.linspace(.2, .68, 9), 12)
    noise = np.tile(np.linspace(-.1, .1, 12), 9) * k
    direct = np.stack([k+noise, .7*k, .5*k], 1)[:, :, None]
    data = ablation.prepare_response_data(direct, k)
    model = SimpleNamespace(advance_theory=lambda x: x.square())
    norm = data.target_derivative.square().mean(0).clamp_min(ablation.NORMALIZATION_FLOOR)
    actual = ablation.response_terms(model, data, norm)
    predicted = data.D @ data.mean_by_K(data.direct[:, :-1, 0].square())
    residual = (predicted-data.target_derivative).square()
    torch.testing.assert_close(actual['response_mse'], residual.mean())
    torch.testing.assert_close(actual['response_normalized'], (residual/norm).mean())
    wrong = data.D @ data.mean_by_K(data.direct[:, :-1, 0]).square()
    assert not torch.allclose(predicted, wrong, atol=1e-8, rtol=1e-8)
    assert data.direct.requires_grad is False and data.target_derivative.requires_grad is False


def test_response_only_R_parameters_receive_gradients(tmp_path):
    source, report, _ = fixture_source(tmp_path)
    payload, cfg, train, test, norm, provenance = ablation.load_source(source, report, 9000)
    model = ablation.SavedResidualR(payload['model'])
    loss = ablation.closure_terms(model, train.direct, cfg)['closure'] + ablation.response_terms(model, train, norm)['response_normalized']
    loss.backward()
    assert all(name.startswith('theory_rg.') for name, _ in model.named_parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    expected = train.target_derivative.square().mean(0).clamp_min(ablation.NORMALIZATION_FLOOR)
    torch.testing.assert_close(norm, expected)
    changed_test = ablation.prepare_response_data(test.direct*100, test.labels)
    assert not torch.allclose(changed_test.target_derivative.square().mean(0), norm)
    torch.testing.assert_close(norm, expected)
    assert provenance['train_fingerprint']['encodings'] == ablation.tensor_sha256(train.direct)


def test_dry_run_is_read_only_and_does_not_instantiate_any_model(tmp_path, monkeypatch):
    source, report, _ = fixture_source(tmp_path)
    monkeypatch.setattr(ablation, 'SavedResidualR', lambda *a: pytest.fail('No model during dry run'))
    output = tmp_path/'new'
    plan = ablation.run(source, output, source_report=report, checkpoint=9000, dry_run=True)
    assert plan['updates'] == 6000 and plan['alpha'] == 1.
    assert plan['snapshots'] == [0, 1000, 3000, 6000]
    assert plan['protocol']['physical_data_loaded'] is False
    assert plan['protocol']['Q_C_inference'] is False
    assert not output.exists()


def test_paired_smoke_preserves_QC_and_source_starts_same_R_and_fresh_adam(tmp_path, monkeypatch):
    source, report, checkpoint = fixture_source(tmp_path)
    # A shared Q/C loader must not be needed, even though the physical-data path is nonexistent.
    monkeypatch.setattr('recurrent_rg.checkpoints.build_model', lambda *a, **k: pytest.fail('Never create Q/C'))
    before = {path: path.read_bytes() for path in [checkpoint, report/'encodings.npz']}
    output = tmp_path/'ablation'
    job = ablation.run(source, output, source_report=report, checkpoint=9000, smoke=True)
    assert job['status'] == 'complete' and job['completed_arms'] == ['control', 'response']
    original = torch.load(checkpoint, map_location='cpu', weights_only=True)
    initial = []
    for arm, alpha in [('control', 0.), ('response', 1.)]:
        directory = output/arm
        zero = torch.load(directory/'training_step_0.pt', map_location='cpu', weights_only=True)
        final = torch.load(directory/'checkpoint_last.pt', map_location='cpu', weights_only=True)
        initial.append(ablation.state_sha256(zero['model'], residual=True))
        assert initial[-1] == ablation.state_sha256(original['model'], residual=True)
        assert not zero['optimizer']['state']
        assert set(float(item['step']) for item in final['optimizer']['state'].values()) == {3.}
        assert final['epoch'] == 9000 and final['extra_R_updates'] == 3
        assert final['metadata']['arm'] == arm and final['metadata']['alpha'] == alpha
        assert final['metadata']['R_only_ablation'] is True and final['metadata']['exact_resume'] is False
        assert final['metadata']['source_config'] == original['metadata']['config']
        assert final['metadata']['config']['training']['train_modules'] == ['R']
        for name, value in original['model'].items():
            if not name.startswith('theory_rg.'):
                assert torch.equal(value, final['model'][name])
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['actual_extra_R_updates'] == 3
        assert [row['extra_R_updates'] for row in summary['history']] == [0, 1, 3]
        for row in summary['history']:
            assert row['train_total'] == pytest.approx(row['train_closure'] + alpha*row['train_response_normalized'])
            assert 'test_response_boundary_normalized' in row and 'test_response_interior_mse' in row
    assert initial[0] == initial[1]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert json.loads((output/'training_job.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('change', ['report_sha', 'encodings_sha', 'overlap', 'replicas', 'nonfinite'])
def test_invalid_sources_fail_before_creating_output(tmp_path, change):
    source, report, _ = fixture_source(tmp_path)
    if change == 'report_sha':
        path = report.parent/'checkpoint_summary.json'
        receipt = json.loads(path.read_text()); receipt['checkpoint_sha256'] = 'b'*64
        path.write_text(json.dumps(receipt))
    else:
        with np.load(report/'encodings.npz') as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        if change == 'overlap': arrays['test_labels'] = arrays['train_labels'].copy()
        if change == 'replicas':
            arrays['train'] = arrays['train'][:-1]; arrays['train_labels'] = arrays['train_labels'][:-1]
        if change == 'nonfinite': arrays['train'][0, 0, 0] = np.nan
        if change == 'encodings_sha': arrays['test'][0, 0, 0] += .01
        np.savez_compressed(report/'encodings.npz', **arrays)
        if change != 'encodings_sha':
            path = report.parent/'local_response.json'; receipt = json.loads(path.read_text())
            receipt['source_sha256']['encodings'] = ablation.file_fingerprint(report/'encodings.npz')
            path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        ablation.run(source, tmp_path/'new', source_report=report, checkpoint=9000, smoke=True)
    assert not (tmp_path/'new').exists()


def test_existing_output_and_nonfinite_loss_fail_without_retry(tmp_path, monkeypatch):
    source, report, _ = fixture_source(tmp_path)
    output = tmp_path/'existing'; output.mkdir()
    with pytest.raises(FileExistsError):
        ablation.run(source, output, source_report=report, checkpoint=9000, smoke=True)
    actual = ablation.closure_terms
    calls = []
    def broken(model, direct, cfg=None):
        result = actual(model, direct, cfg)
        calls.append(1)
        result['closure'] = result['closure'] * float('nan')
        return result
    monkeypatch.setattr(ablation, 'closure_terms', broken)
    with pytest.raises(FloatingPointError):
        ablation.run(source, tmp_path/'failed', source_report=report, checkpoint=9000, smoke=True)
    assert len(calls) == 1
    job = json.loads((tmp_path/'failed'/'training_job.json').read_text())
    assert job['status'] == 'failed' and job['requires_user_action'] is True
