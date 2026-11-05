"""Fast, synthetic checks; none of these are formal physical results."""
import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.ensembles import EnsembleDataset
from recurrent_rg.encoding import half_subset_theory
from scripts.report_ising_j1j2 import (
    Deadline, J1J2Collector, OBSERVABLE_NAMES, ReportBudgetHold,
    error_summary, fit_ridge, grouped_closure_metrics, pca_train,
    physical_observables, physical_readout_metrics, predict_ridge,
    transport_paths, validate_metadata,
)


def test_ferro_and_stripe_statistics_keep_signed_correlations():
    ferro = np.ones((2, 4, 12, 12), dtype=bool)
    stripe = np.broadcast_to((np.arange(12) % 2 == 0)[None, None, :, None], ferro.shape).copy()
    observed = physical_observables(ferro)
    np.testing.assert_allclose(observed[:, [0, 1, 7, 10, 11]], 1.)
    np.testing.assert_allclose(observed[:, [2, 3, 4, 8, 9]], 0.)
    observed = physical_observables(stripe)
    np.testing.assert_allclose(observed[:, [0, 1, 3, 7, 9, 10]], 0.)
    np.testing.assert_allclose(observed[:, [2, 4, 6, 8]], 1.)
    np.testing.assert_allclose(observed[:, [5, 11]], -1.)  # distance L//4=3
    assert observed.shape[-1] == len(OBSERVABLE_NAMES)


def test_observables_reject_float_soft_fields():
    with pytest.raises(ValueError):
        physical_observables(np.ones((2, 8, 12, 12)))


def metadata_fixture():
    def row(r, t, group, partition):
        return dict(rho_index=r, temperature_index=t, group=group,
                    rho=.2 * r, temperature=1. + t, condition_id=f'r{r}t{t}', partition=partition)
    train = [row(0, 0, 0, 'train'), row(0, 1, 0, 'train')]
    test = [row(0, 0, 1, 'same_rho_new_group'), row(1, 0, 0, 'heldout_rho')]
    def dataset(rows):
        groups = torch.tensor([[r['rho_index'], r['temperature_index'], r['group']] for r in rows])
        return EnsembleDataset(torch.ones((len(rows), 8, 12, 12, 1), dtype=torch.int8), None, groups)
    return dict(schema_version=1, data_sha256='a' * 64, train=train, test=test), {
        'train': dataset(train), 'test': dataset(test)}


def test_metadata_binds_fingerprint_rows_and_entire_PT_groups():
    metadata, datasets = metadata_fixture()
    receipt = validate_metadata(metadata, datasets, actual_sha256='a' * 64, manifest_sha256='a' * 64)
    assert receipt['whole_PT_group_split_verified']
    with pytest.raises(ValueError, match='SHA256'):
        validate_metadata(metadata, datasets, actual_sha256='b' * 64, manifest_sha256='a' * 64)
    metadata['test'][0]['temperature_index'] = 1
    with pytest.raises(ValueError, match='行顺序'):
        validate_metadata(metadata, datasets, actual_sha256='a' * 64, manifest_sha256='a' * 64)


def test_metadata_rejects_labels_and_cross_temperature_PT_leakage():
    metadata, datasets = metadata_fixture()
    datasets['train'].couplings = torch.zeros(2, 2)
    with pytest.raises(ValueError, match='无标签'):
        validate_metadata(metadata, datasets, actual_sha256='a' * 64, manifest_sha256='a' * 64)
    metadata, datasets = metadata_fixture()
    metadata['test'][0].update(group=0, temperature_index=2, temperature=3., condition_id='r0t2')
    datasets['test'].group_ids[0] = torch.tensor([0, 2, 0])
    with pytest.raises(ValueError, match='平行回火'):
        validate_metadata(metadata, datasets, actual_sha256='a' * 64, manifest_sha256='a' * 64)


def test_heldout_rho_cannot_be_spoofed_by_a_new_index():
    metadata, datasets = metadata_fixture()
    metadata['test'][1]['rho'] = 0.
    with pytest.raises(ValueError, match='一一对应'):
        validate_metadata(metadata, datasets, actual_sha256='a' * 64, manifest_sha256='a' * 64)


def test_collector_exact_dense_half_Q_and_same_C_statistics():
    torch.set_num_threads(1)
    with isolated_rng('cpu', 123):
        model = build_model(ModelConfig(dimension=2, theory_dim=2, hidden=8,
                            local_features=4, block_factor=3), 1).eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(5)
    spins = (2 * torch.randint(2, (2, 8, 12, 12, 1), generator=generator) - 1).to(torch.int8)
    collector = J1J2Collector(model, samples=8)
    before = torch.get_rng_state().clone()
    result = collector.trajectories(spins, steps=1, draws=2, chunk_size=2, seed=77)
    assert torch.equal(before, torch.get_rng_state())
    partition = torch.tensor(collector.encoding_protocol['sample_partition'])
    with torch.no_grad(), isolated_rng('cpu', 77):
        levels = model.coarse_grain(spins.double())
        reference = half_subset_theory(model, [x.index_select(1, partition) for x in levels])
        full = torch.stack([model.theory_encoder(x) for x in levels], 1)
    np.testing.assert_allclose(result['direct'][:, 0], reference.numpy(), atol=1e-12)
    np.testing.assert_allclose(result['full_ensemble_direct'][:, 0], full.numpy(), atol=1e-12)
    for k, field in enumerate(levels):
        expected = physical_observables(field[..., 0].numpy() > 0)
        np.testing.assert_allclose(result['observables'][:, 0, k], expected, atol=1e-12)
    assert result['direct'].shape == (2, 2, 2, 2)
    with torch.no_grad():
        expected_mean_R = model.advance_theory(torch.from_numpy(result['direct'].mean(1))).numpy()
    np.testing.assert_allclose(result['R_of_mean_g'], expected_mean_R)
    np.testing.assert_allclose(result['mean_R_g_minus_R_of_mean_g'],
        result['R_g'].mean(1) - result['R_of_mean_g'])
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_all_paths_identity_errors_and_PT_group_bootstrap():
    class Shift:
        @staticmethod
        def advance_theory(x):
            return x + 1.
    source = np.zeros((4, 3, 3, 2))
    source[:, :, 1] = 1.
    source[:, :, 2] = 2.
    paths, predicted, identity = transport_paths(Shift(), source)
    np.testing.assert_array_equal(paths, [[0, 1], [0, 2], [1, 2]])
    rows = [dict(rho_index=0, temperature_index=i % 2, group=i // 2) for i in range(4)]
    metrics = grouped_closure_metrics(source, predicted, identity, paths,
                                     np.ones((3, 2)), rows, bootstrap=20)
    assert metrics['PT_group_count'] == 2
    assert all(row['R']['rmse'] == 0 for row in metrics['paths'])
    assert metrics['paths'][1]['identity']['rmse'] == 2.
    assert metrics['paths'][1]['R']['PT_group_bootstrap_rmse_95'] == [0., 0.]


def test_C_draws_average_squared_errors_not_prediction_first():
    observed = error_summary(np.array([[[1., 0.], [-1., 0.]]]), np.ones(2))
    assert observed['rmse'] == pytest.approx(np.sqrt(.5))
    assert observed['physical_row_count'] == 1
    assert observed['C_draw_count'] == 2


def test_fixed_ridge_has_no_test_fit_and_decomposes_transport():
    x = np.arange(12.).reshape(6, 2)
    target = x[:, :1] * 2 + 3
    probe = fit_ridge(x, target)
    np.testing.assert_allclose(predict_ridge(probe, x), target, atol=.001)
    assert probe['ridge'] == 1e-4
    assert not np.array_equal(predict_ridge(probe, x + 2), predict_ridge(probe, x))
    g = np.stack([x, x + 1], 1)[:, None]
    values = dict(direct=g, observables=np.stack([target, target], 1)[:, None])
    result = physical_readout_metrics({'microscopic_information': [probe, probe]}, values,
        g[:, :, :1], g[:, :, :1], np.array([[0, 1]]))
    assert result['microscopic_information'][0]['target_scale'] == 0
    assert result['microscopic_information'][0]['transport_minus_direct']['rmse'] > 0


def test_PCA_fitted_from_TRAIN_C_means_and_budget_guard():
    x = np.arange(12.).reshape(6, 1, 2, 1)
    direct = np.concatenate([x, 2 * x], -1)
    pca = pca_train(direct)
    np.testing.assert_allclose(pca['explained_variance_ratio'], [1., 0.], atol=1e-15)
    deadline = Deadline(1.)
    deadline.started -= 2
    with pytest.raises(ReportBudgetHold):
        deadline.check()


def test_cli_defaults_are_formal_and_does_not_offer_root_search():
    import inspect
    from scripts import report_ising_j1j2 as report
    assert 'root_search_performed=False' in inspect.getsource(report.run_report)
    assert 'thermal_nu=None' in inspect.getsource(report.run_report)
    assert '--checkpoint' not in inspect.getsource(report.main)


def test_synthetic_end_to_end_frozen_report_and_no_overwrite(tmp_path):
    """Fake endpoint fixture verifies plumbing, NOT training/physics success."""
    import json
    from types import SimpleNamespace
    from recurrent_rg.config import DataConfig, EvaluationConfig, RunConfig, TrainingConfig
    from scripts.report_ising_j1j2 import run_report, sha256
    torch.set_num_threads(1)
    metadata, small = metadata_fixture()
    generator = torch.Generator().manual_seed(18)
    for dataset in small.values():
        dataset.configurations = (2 * torch.randint(2, (2, 128, 12, 12, 1), generator=generator) - 1).to(torch.int8)
    cache = tmp_path / 'data.pt'
    torch.save(dict(schema_version=1,
        **{key: dict(configurations=value.configurations, couplings=None, group_ids=value.group_ids)
           for key, value in small.items()}), cache)
    fingerprint = sha256(cache)
    metadata['data_sha256'] = fingerprint
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text(json.dumps(metadata))
    cfg = RunConfig(data=DataConfig(cache=str(cache), physical_model='ising2d'),
        model=ModelConfig(dimension=2, theory_dim=2, hidden=8, local_features=4),
        training=TrainingConfig(max_rg_steps=1, epochs=12000, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=1, samples=128, systems=2))
    cfg.validate()
    with isolated_rng('cpu', 22):
        model = build_model(cfg.model, 1)
    run = tmp_path / 'synthetic_fixture_run'; run.mkdir()
    contract = dict(config=cfg.to_dict(), labels_used_in_training=False, data_sha256=fingerprint)
    (run / 'manifest.json').write_text(json.dumps(contract))
    (run / 'summary.json').write_text(json.dumps(dict(actual_epochs=12000)))
    history = [dict(epoch=1, train_commutator=.01, test_commutator=.02),
               dict(epoch=12000, train_commutator=.001, test_commutator=.002)]
    torch.save(dict(model=model.state_dict(), epoch=12000, metadata=contract, history=history),
               run / 'training_step_12000.pt')
    output = tmp_path / 'report'
    args = SimpleNamespace(run=run, data_metadata=metadata_file, output=output,
                           max_seconds=60., draws=5, font=None)
    result = run_report(args)
    assert result['status'] == 'complete'
    assert result['root_search_performed'] is False
    assert result['phase_transition_certified'] is False
    assert result['training_updates'] == 0
    assert result['thermal_nu'] is None
    assert set(result['closure']) == {'train', 'same_rho_new_group', 'heldout_rho'}
    status = json.loads((output / 'status.json').read_text())
    assert status['report_complete']
    assert len(list(output.glob('*.png'))) == 5
    for filename, fingerprint in status['sha256'].items():
        assert sha256(output / filename) == fingerprint
    with np.load(output / 'evaluation_arrays.npz', allow_pickle=False) as saved:
        assert saved['direct_train'].shape == (2, 5, 2, 2)
        assert saved['observables_test'].shape == (2, 5, 2, 12)
    with pytest.raises(FileExistsError):
        run_report(args)
