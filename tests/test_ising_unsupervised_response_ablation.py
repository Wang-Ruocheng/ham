"""Strictly label-free saved-g secants, frozen-source identity and paired R fits."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiments import ising_unsupervised_response_ablation as ablation
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import RunConfig


def fixture_source(tmp_path, *, test_factor=1.):
    source, report = tmp_path/'source'/'run', tmp_path/'source'/'run'/'full_report'
    report.mkdir(parents=True)
    cfg = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache='/physical/data/is/never/opened.pt', physical_model='ising2d', coupling_names=['K']),
        model=dict(dimension=2, theory_dim=1, hidden=5, block_factor=2, affine_reconstruction=True),
        loss=dict(rollout_weight=5., weights=dict(commutator=5.)),
        training=dict(max_rg_steps=2, epochs=12000, gradient_mode='qc_r_separate'),
        evaluation=dict(max_rg_steps=2)))
    torch.manual_seed(23)
    model = build_model(cfg.model, 2)
    metadata = dict(config=cfg.to_dict(), labels_used_in_training=False, q_supervision=None,
                    independent_c=True, data_sha256='a'*64)
    payload = dict(epoch=12000, metadata=metadata, model=model.state_dict(),
                   optimizer=dict(source_only='do not restore'))
    checkpoint, last = source/'training_step_12000.pt', source/'checkpoint_last.pt'
    torch.save(payload, checkpoint); torch.save(payload, last)
    manifest = source/'manifest.json'; manifest.write_text(json.dumps(metadata))
    (source/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(), actual_epochs=12000)))
    arrays = {}
    for split, count, factor in [('train', 516, 1.), ('test', 252, test_factor)]:
        x = np.linspace(.1, .9, count) * factor
        arrays[split] = np.stack([x, .7*x+.02*np.sin(5*x), .5*x+.01*np.sin(9*x)], 1)[:, :, None]
        # These object arrays cannot be loaded with allow_pickle=False. Access is forbidden.
        arrays[split+'_labels'] = np.array([dict(forbidden='Any coupling labels must stay unopened')], dtype=object)
    np.savez_compressed(report/'encodings.npz', **arrays)
    receipt = dict(status='complete', workflow_complete=True,
        training=dict(actual_epochs=12000, manifest_sha256=ablation.file_fingerprint(manifest)),
        reports=dict(files={'full_report/encodings.npz':ablation.file_fingerprint(report/'encodings.npz')}))
    (source.parent/'job.json').write_text(json.dumps(receipt))
    return source, report, checkpoint


def test_pair_selection_keeps_large_index_only_edge_and_canonical_deduplication():
    x = np.array([.1, .11, 0.])
    direct = np.stack([x, 2*x+3, 4*x+9], 1)[:, :, None]
    data = ablation.prepare_secant_data(direct, neighbors=1, min_distance=.01, max_distance=3.)
    assert data.pairs[0].tolist() == [[0, 1], [0, 2]]
    assert data.pairs[1].tolist() == [[0, 1], [0, 2]]
    assert data.direct.requires_grad is False
    assert all(target.requires_grad is False for target in data.target_secants)


def test_fixed_distance_bounds_population_sigma_and_isolated_nodes_are_preserved():
    x = np.r_[np.linspace(0, .1, 20), 1.]
    direct = np.stack([x, .7*x, .5*x], 1)[:, :, None]
    data = ablation.prepare_secant_data(direct)
    torch.testing.assert_close(data.train_sigma, torch.from_numpy(direct[:, :-1, 0]).std(0, unbiased=False))
    for scale, (edges, diagnostics) in enumerate(zip(data.pairs, data.diagnostics)):
        dx = (data.direct[edges[:, 0], scale, 0] - data.direct[edges[:, 1], scale, 0]).abs()
        assert torch.all(dx >= .05*data.train_sigma[scale])
        assert torch.all(dx <= .25*data.train_sigma[scale])
        assert len(torch.unique(edges, dim=0)) == len(edges)
        assert diagnostics['isolated_nodes'] == [20]
        assert diagnostics['isolated_node_count'] == 1
        assert diagnostics['coverage'] == pytest.approx(20/21)
        assert diagnostics['outgoing_neighbor_count']['max'] <= 8
        assert set(diagnostics['pair_distance']) == {'min','q10','median','q90','max'}
        assert 0 <= diagnostics['largest_10pct_target_square_share'] <= 1


def test_secant_objective_is_exact_for_affine_full_R_at_both_scales():
    x = np.linspace(.1, .9, 100)
    direct = np.stack([x, 2*x+3, 4*x+9], 1)[:, :, None]
    data = ablation.prepare_secant_data(direct)
    model = type('AffineR', (), {'advance_theory':lambda self, x:2*x+3})()
    normalization = torch.stack([target.square().mean() for target in data.target_secants])
    terms = ablation.response_terms(model, data, normalization)
    assert float(terms['response_mse']) < 1e-24
    assert float(terms['response_normalized']) < 1e-24
    assert float(ablation.closure_terms(model, data.direct)['closure']) < 1e-24


def test_loader_never_reads_poison_labels_and_test_uses_only_train_scales(tmp_path, monkeypatch):
    source, report, _ = fixture_source(tmp_path)
    actual_load, accessed = ablation.np.load, []
    class OnlyEncodingArrays:
        def __init__(self, path, **kwargs): self.arrays = actual_load(path, **kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): self.arrays.close()
        def __getitem__(self, key):
            accessed.append(key)
            assert key in ('train', 'test'), 'Core attempted to read a label array'
            return self.arrays[key]
    monkeypatch.setattr(ablation.np, 'load', OnlyEncodingArrays)
    payload, cfg, train, test, normalization, plan = ablation.load_source(source, report, 12000)
    assert accessed == ['train', 'test']
    torch.testing.assert_close(test.train_sigma, train.train_sigma)
    expected = torch.stack([target.square().mean() for target in train.target_secants]).clamp_min(1e-12)
    torch.testing.assert_close(normalization, expected)
    assert plan['K_label_access'] is plan['labels_used_in_training'] is False
    assert plan['accessed_encoding_arrays'] == ['train', 'test']
    assert plan['source_checkpoint_models_equal'] is True


def test_dry_run_is_read_only_and_does_not_create_Q_C_or_R(tmp_path, monkeypatch):
    source, report, _ = fixture_source(tmp_path)
    monkeypatch.setattr(ablation, 'SavedResidualR', lambda *a:pytest.fail('No model during dry-run'))
    plan = ablation.run(source, tmp_path/'new', source_report=report, dry_run=True)
    assert plan['updates'] == 6000 and plan['alpha'] == 1.
    assert plan['neighbors'] == 8 and plan['min_distance'] == .05 and plan['max_distance'] == .25
    assert plan['snapshots'] == [0,1000,3000,6000]
    assert not (tmp_path/'new').exists()


def test_paired_smoke_uses_same_R_fresh_adam_bitwise_frozen_QC_and_no_labels(tmp_path, monkeypatch):
    source, report, checkpoint = fixture_source(tmp_path)
    before = checkpoint.read_bytes()
    monkeypatch.setattr('recurrent_rg.checkpoints.build_model', lambda *a, **k:pytest.fail('Core never creates Q/C'))
    output = tmp_path/'fit'
    job = ablation.run(source, output, source_report=report, smoke=True)
    assert job['status'] == 'complete' and job['completed_arms'] == ['control','response']
    original = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for arm, alpha in [('control',0.),('response',1.)]:
        directory = output/arm
        initial = torch.load(directory/'training_step_0.pt', map_location='cpu', weights_only=True)
        final = torch.load(directory/'checkpoint_last.pt', map_location='cpu', weights_only=True)
        assert not initial['optimizer']['state']
        assert set(float(value['step']) for value in final['optimizer']['state'].values()) == {3.}
        assert final['epoch'] == 12000 and final['extra_R_updates'] == 3
        assert final['metadata']['alpha'] == alpha
        assert final['metadata']['labels_used_in_training'] is final['metadata']['K_label_access'] is False
        assert final['metadata']['R_only_ablation'] is True and final['metadata']['exact_resume'] is False
        assert ablation.state_sha256(initial['model'], residual=True) == ablation.state_sha256(original['model'], residual=True)
        for name, value in original['model'].items():
            if not name.startswith('theory_rg.'): assert torch.equal(value, final['model'][name])
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['actual_extra_R_updates'] == 3
        for row in summary['history']:
            assert row['train_total'] == pytest.approx(row['train_closure'] + alpha*row['train_response_normalized'])
    assert checkpoint.read_bytes() == before


def test_test_encodings_cannot_change_training_pairs_normalizers_or_final_R(tmp_path):
    first_root, second_root = tmp_path/'one', tmp_path/'two'
    first_root.mkdir(); second_root.mkdir()
    one = fixture_source(first_root, test_factor=1.)
    two = fixture_source(second_root, test_factor=.8)
    jobs = [ablation.run(source, root/'fit', source_report=report, smoke=True)
            for (source,report,_),root in zip((one,two),(first_root,second_root))]
    assert jobs[0]['plan']['train_sigma'] == jobs[1]['plan']['train_sigma']
    assert jobs[0]['plan']['normalization'] == jobs[1]['plan']['normalization']
    assert jobs[0]['plan']['pair_diagnostics']['train'] == jobs[1]['plan']['pair_diagnostics']['train']
    for arm in ('control','response'):
        a = torch.load(first_root/'fit'/arm/'checkpoint_last.pt', map_location='cpu', weights_only=True)
        b = torch.load(second_root/'fit'/arm/'checkpoint_last.pt', map_location='cpu', weights_only=True)
        for key in a['model']: assert torch.equal(a['model'][key], b['model'][key])


@pytest.mark.parametrize('where', ['manifest','payload','last'])
def test_any_supervised_source_is_rejected_before_output_creation(tmp_path, where):
    source, report, checkpoint = fixture_source(tmp_path)
    if where == 'manifest':
        path = source/'manifest.json'; value = json.loads(path.read_text())
        value['labels_used_in_training'] = True; path.write_text(json.dumps(value))
    else:
        path = checkpoint if where == 'payload' else source/'checkpoint_last.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        value['metadata']['labels_used_in_training'] = True
        value['metadata']['q_supervision'] = dict(weight=10.)
        torch.save(value, path)
    with pytest.raises(ValueError, match='strictly unsupervised'):
        ablation.run(source, tmp_path/'bad', source_report=report, smoke=True)
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('change', ['model','receipt','encodings'])
def test_source_model_and_report_identity_guards(tmp_path, change):
    source, report, checkpoint = fixture_source(tmp_path)
    if change == 'model':
        value = torch.load(checkpoint, map_location='cpu', weights_only=True)
        value['model']['theory_rg.0.bias'].add_(.1); torch.save(value, checkpoint)
    elif change == 'receipt':
        path = source.parent/'job.json'; value = json.loads(path.read_text())
        value['status'] = 'running'; path.write_text(json.dumps(value))
    else:
        with (report/'encodings.npz').open('ab') as stream: stream.write(b'changed artifact')
    with pytest.raises(ValueError):
        ablation.run(source, tmp_path/'bad', source_report=report, smoke=True)
    assert not (tmp_path/'bad').exists()


def test_existing_output_and_pair_free_scale_fail_without_fallback(tmp_path):
    source, report, _ = fixture_source(tmp_path)
    output = tmp_path/'exists'; output.mkdir()
    with pytest.raises(FileExistsError):
        ablation.run(source, output, source_report=report, smoke=True)
    with pytest.raises(ValueError, match='No admissible'):
        ablation.prepare_secant_data(np.array([[[0.],[0.],[0.]],[[1.],[1.],[1.]]]))


def test_nonfinite_loss_stops_first_arm_without_retry(tmp_path, monkeypatch):
    source, report, _ = fixture_source(tmp_path)
    actual, calls = ablation.closure_terms, []
    def broken(*args, **kwargs):
        result = actual(*args, **kwargs); calls.append(1)
        result['closure'] = result['closure'] * float('nan')
        return result
    monkeypatch.setattr(ablation, 'closure_terms', broken)
    with pytest.raises(FloatingPointError):
        ablation.run(source, tmp_path/'bad', source_report=report, smoke=True)
    job = json.loads((tmp_path/'bad'/'training_job.json').read_text())
    assert len(calls) == 1 and job['status'] == 'failed' and job['completed_arms'] == []
