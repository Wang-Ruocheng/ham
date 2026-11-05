"""The paper seed comparison never pools arbitrary coordinates or C draws."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.encoding import half_subset_theory
from scripts import report_ising_b0_multiseed as report


class LinearMap(torch.nn.Module):
    def __init__(self, factor=2.):
        super().__init__()
        self.factor = factor

    def advance_theory(self, points):
        return self.factor * points


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def config():
    raw = dict(schema_version=5,
        data=dict(cache='/unused/physical.pt', physical_model='ising2d', coupling_names=['K'],
                  spin_flip_augment=False),
        model=dict(dimension=2, theory_dim=5, block_factor=3, hidden=96, width=3,
                   kernel_layout='nonoverlapping', kernel_sampling='bernoulli',
                   kernel_initialization='random', fixed_kernel=False, affine_reconstruction=True),
        training=dict(epochs=12000, max_rg_steps=2, gradient_mode='qc_r_separate', seed=91),
        evaluation=dict(max_rg_steps=2, samples=128), loss=dict(closure_scale_weights=None))
    return RunConfig.from_dict(raw)


def provenance():
    cfg = config()
    summary = dict(actual_epochs=12000, requested_epochs=12000, config=cfg.to_dict(),
                   history=[dict(epoch=12000)])
    payload = dict(epoch=12000, history=[dict(epoch=12000)],
                   metadata=dict(labels_used_in_training=False))
    manifest = dict(config=cfg.to_dict(), labels_used_in_training=False, source=None, data_sha256='a' * 64,
        empirical_statistics={split: dict(visible_sites=9, environment_sites=24, buffer=1, states=2)
                              for split in ('train', 'test')})
    sidecar = dict(B=0., coupling_names=['K'], sites=108, data_sha256='a' * 64,
                   train_shape=[516, 128, 108, 108, 1], test_shape=[252, 128, 108, 108, 1],
                   ensembles=[{name: dict(ess_estimate=64.) for name in
                       ('bond_sum_per_site', 'magnetization_squared')} for _ in range(768)])
    return cfg, summary, payload, manifest, sidecar


def test_identity_baseline_is_start_encoding_for_each_path():
    draw = dict(direct=np.asarray([[[1.], [3.], [7.]], [[2.], [5.], [9.]]]))
    errors = report.path_errors(LinearMap(), draw)
    assert errors['paths'] == [(0, 1), (0, 2), (1, 2)]
    np.testing.assert_array_equal(errors['R'][:, :, 0], [[1., 9., 1.], [1., 1., 1.]])
    np.testing.assert_array_equal(errors['identity'][:, :, 0], [[4., 36., 16.], [9., 49., 16.]])


def test_C_averaged_squared_error_is_not_averaged_RMSE_or_extra_chains():
    errors = [dict(paths=[(0, 1)], R=np.full((2, 1, 1), value),
                   identity=np.full((2, 1, 1), 16.)) for value in (1., 9.)]
    reference = np.asarray([[[0.], [0.]], [[1.], [4.]]])
    metrics = report.grouped_path_metrics(errors, reference)
    path = metrics['paths'][0]
    assert path['R']['rmse'] == pytest.approx(np.sqrt(5.))
    assert path['R']['rmse'] != pytest.approx((1. + 3.) / 2)
    assert path['R']['nrmse'] == pytest.approx(np.sqrt(5.) / 2)
    assert path['identity']['rmse'] == 4.
    assert metrics['physical_chain_count'] == metrics['independent_sample_count'] == 2
    assert metrics['C_draw_count'] == 2


def test_target_std_uses_TRAIN_only_and_zero_std_is_explicit_null():
    errors = [dict(paths=[(0, 1)], R=np.ones((3, 1, 2)), identity=np.ones((3, 1, 2)))]
    reference = np.asarray([[[0., 1.], [0., 5.]], [[1., 1.], [4., 5.]]])
    result = report.grouped_path_metrics(errors, reference)['paths'][0]
    assert result['target_train_rms_coordinate_std'] == pytest.approx(np.sqrt(2))
    assert result['R']['coordinate_nrmse'] == [.5, None]


def test_C_repeats_reject_changed_physical_chain_shape():
    a = dict(paths=[(0, 1)], R=np.ones((2, 1, 1)), identity=np.ones((2, 1, 1)))
    b = dict(paths=[(0, 1)], R=np.ones((3, 1, 1)), identity=np.ones((3, 1, 1)))
    with pytest.raises(ValueError, match='相同物理链'):
        report.grouped_path_metrics([a, b], np.ones((2, 2, 1)))


def test_strict_endpoint_and_unsupervised_geometry_provenance():
    args = provenance()
    report.validate_provenance(*args)
    for location, key, value in [(1, 'actual_epochs', 11999), (2, 'epoch', 11999),
                                 (3, 'labels_used_in_training', True), (3, 'resumed_from', 'old.pt'),
                                 (4, 'B', .01), (4, 'sites', 36)]:
        changed = deepcopy(args)
        changed[location][key] = value
        with pytest.raises(ValueError):
            report.validate_provenance(*changed)


def test_nonuniform_k_and_b2_cannot_be_silent_default():
    values = provenance()
    for key, value in [('closure_scale_weights', [.2, 1.8])]:
        raw = values[0].to_dict()
        raw['loss'][key] = value
        cfg = RunConfig.from_dict(raw)
        summary = deepcopy(values[1]); summary['config'] = cfg.to_dict()
        manifest = deepcopy(values[3]); manifest['config'] = cfg.to_dict()
        with pytest.raises(ValueError, match='均匀'):
            report.validate_provenance(cfg, summary, values[2], manifest, values[4])
    raw = values[0].to_dict(); raw['model']['block_factor'] = 2
    cfg = RunConfig.from_dict(raw)
    summary = deepcopy(values[1]); summary['config'] = cfg.to_dict()
    with pytest.raises(ValueError, match='b3'):
        report.validate_provenance(cfg, summary, values[2], values[3], values[4])


def sample_report(seed):
    return dict(experiment='ising2d_B0_multiseed_stageA_v1', checkpoint_epoch=12000,
        theory_dim=5, training_seed=seed, data_sha256='a' * 64,
        labels_used_in_training=False, thermal_nu=None,
        freeze_verification=dict(model_state_unchanged=True), roots=[],
        **report.encoding_metadata(),
        grouped_closure_identity=dict(test=dict(paths=[dict(start=0, end=1,
            R=dict(rmse=float(seed), nrmse=.2 + (seed - 91) / 10),
            identity=dict(rmse=float(seed * 10), nrmse=1.), R_to_identity_rmse_ratio=.1)])))


def saved_reports(root, seeds=report.EXPECTED_SEEDS):
    specs = []
    for seed in seeds:
        folder = root / str(seed)
        dump(folder / 'metrics.json', sample_report(seed))
        dump(folder / 'report.json', dict(status='complete', report_complete=True,
                                         **report.encoding_metadata()))
        specs.append(f'种子{seed}={folder}')
    return specs


def test_compare_same_dim_includes_all_five_without_coordinate_pooling(tmp_path, monkeypatch):
    specs = saved_reports(tmp_path / 'runs')
    monkeypatch.setattr(report, 'plot_comparison', lambda rows, out: None)
    result = report.compare_reports(specs, tmp_path / 'comparison')
    assert result['all_seeds_included'] and result['selected_best_seed'] is None
    assert result['thermal_nu'] is None and not result['raw_coordinate_errors_pooled_across_models']
    assert result['TEST_path_summary'][0]['R_nrmse']['mean'] == pytest.approx(.4)
    assert result['TEST_path_summary'][0]['R_nrmse']['std_ddof1'] == pytest.approx(np.std([.2, .3, .4, .5, .6], ddof=1))
    assert len(result['all_reports']) == 5
    assert 'R_rmse' not in result['TEST_path_summary'][0]


@pytest.mark.parametrize('seeds', [(91, 92, 93), (91, 92, 93, 94, 96), (91, 92, 93, 94, 94)])
def test_compare_does_not_drop_missing_failed_or_repeated_seed(tmp_path, monkeypatch, seeds):
    monkeypatch.setattr(report, 'plot_comparison', lambda rows, out: None)
    with pytest.raises(ValueError):
        report.compare_reports(saved_reports(tmp_path / 'runs', seeds), tmp_path / 'comparison')


def test_compare_rejects_data_or_dimension_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'plot_comparison', lambda rows, out: None)
    specs = saved_reports(tmp_path / 'runs')
    changed = sample_report(95); changed['data_sha256'] = 'b' * 64
    dump(tmp_path / 'runs/95/metrics.json', changed)
    with pytest.raises(ValueError, match='同一固定物理数据'):
        report.compare_reports(specs, tmp_path / 'comparison')
    changed['theory_dim'] = 10
    dump(tmp_path / 'runs/95/metrics.json', changed)
    with pytest.raises(ValueError, match='d_g=5'):
        report.compare_reports(specs, tmp_path / 'comparison2')


def test_quality_is_saved_diagnostic_not_new_sampling():
    result = report.sampling_quality(dict(ensembles=[
        dict(bond_sum_per_site=dict(ess_estimate=8.3)),
        dict(bond_sum_per_site=dict(ess_estimate=50.))]))
    assert result['production_chain_count'] == 2
    assert not result['equilibrium_certified']
    assert result['observables']['bond_sum_per_site']['saved_ESS_below_32'] == 1
    with pytest.raises(ValueError, match='768'):
        report.sampling_quality({}, strict=True)


def test_saved_mi_diagnostics_accept_actual_field_and_never_silently_empty():
    manifest = provenance()[3]
    assert report.saved_mi_diagnostics(manifest)['source_field'] == 'empirical_statistics'
    alternative = dict(empirical_mi_diagnostics=manifest['empirical_statistics'])
    assert report.saved_mi_diagnostics(alternative)['source_field'] == 'empirical_mi_diagnostics'
    with pytest.raises(ValueError, match='经验 MI'):
        report.saved_mi_diagnostics({})


def test_stat_identity_never_hashes_large_input_and_rejects_symlink(tmp_path, monkeypatch):
    path = tmp_path / 'physical.pt'; path.write_bytes(b'not-loaded')
    monkeypatch.setattr(report.original, 'sha256', lambda _: pytest.fail('物理数据不得hash'))
    assert report.file_stat(path)['size'] == 10
    link = tmp_path / 'alias.pt'; link.symlink_to(path)
    with pytest.raises(ValueError, match='普通文件'):
        report.file_stat(link)


def test_cli_requires_fixed_endpoint_threads_and_two_C_repeats(tmp_path):
    for extra in [('--checkpoint', '1000'), ('--threads', '2'), ('--sampling-repeats', '5')]:
        with pytest.raises(SystemExit):
            report.main(['--run', 'unused', '--output', str(tmp_path / 'report'), *extra])
    assert not (tmp_path / 'report').exists()


def test_readonly_report_public_files_have_no_checkpoint_or_physical_data(tmp_path):
    source = Path(report.__file__).read_text()
    assert 'TrainingMatchedBinaryCollector' in source and 'fast_report_context' in source
    assert 'raw_physical_data_rehashed=False' in source and 'MI_reestimated=False' in source
    assert "('.json', '.npz', '.png')" in source
    assert 'torch.save(' not in source


def test_full_report_orchestration_keeps_only_base_encodings_and_grouped_errors(tmp_path, monkeypatch):
    cfg, summary, payload, manifest, sidecar = provenance()
    raw = cfg.to_dict(); raw['data']['cache'] = str(tmp_path / 'physical.pt')
    cfg = RunConfig.from_dict(raw)
    summary['config'] = manifest['config'] = cfg.to_dict()
    (tmp_path / 'physical.pt').write_bytes(b'never-open-physical')
    dump(tmp_path / 'physical.json', sidecar)
    run = tmp_path / 'run'; run.mkdir()
    checkpoint = run / 'training_step_12000.pt'; checkpoint.write_bytes(b'checkpoint')
    dump(run / 'manifest.json', manifest); dump(run / 'summary.json', summary)
    loaded = SimpleNamespace(checkpoint=checkpoint, config=cfg.to_dict(), payload=payload,
                             model=LinearMap())
    monkeypatch.setattr(report, 'load_run', lambda *args: loaded)
    counts = (516, 252)
    datasets = [SimpleNamespace(configurations=SimpleNamespace(shape=(n, 128, 108, 108, 1)),
                                couplings=torch.linspace(.2, .68, n)) for n in counts]
    monkeypatch.setattr(report.original, 'report_datasets', lambda cfg: datasets)
    calls = []
    class Collector:
        def __init__(self, *args, **kwargs):
            pass

        def collect(self, configurations, **options):
            calls.append((configurations.shape[0], options['sampling_seed']))
            n = configurations.shape[0]
            random = np.random.default_rng(options['sampling_seed'])
            direct = random.normal(size=(n, 3, 5))
            return dict(direct=direct, rollout=np.stack([direct[:, 0], 2 * direct[:, 0],
                4 * direct[:, 0]], 1), one_step=2 * direct[:, :-1])
    monkeypatch.setattr(report, 'TrainingMatchedBinaryCollector', Collector)
    monkeypatch.setattr(report, 'fast_report_context', lambda *args, **kwargs: nullcontext())
    def write(loaded, train, test, ty, vy, history, output, **kwargs):
        output.mkdir()
        assert train['direct'].shape == (516, 3, 5) and test['direct'].shape == (252, 3, 5)
        assert kwargs['lattice_sizes'] == [108, 36, 12]
        result = dict(theory_dim=5, checkpoint_epoch=12000, data_sha256='a' * 64,
                      labels_used_in_training=False, roots=[], fixed_points=dict(roots=[]))
        dump(output / 'metrics.json', result)
        dump(output / 'report.json', dict(status='complete'))
        dump(output / 'status.json', dict(status='complete'))
        for name in ('encodings.npz', 'trajectories.npz', 'fixed_point_arrays.npz'):
            np.savez(output / name, direct=train['direct'])
        return result
    monkeypatch.setattr(report.original, 'write_report', write)
    output = tmp_path / 'report'
    args = SimpleNamespace(run=run, output=output)
    deadline = report.Deadline(60)
    progress = report.Progress(tmp_path / 'progress.json', deadline)
    result = report.run_report(args, deadline, progress)
    assert calls == [(516, 12345), (252, 12345), (252, 0), (252, 1)]
    assert result['grouped_closure_identity']['test']['physical_chain_count'] == 252
    assert result['grouped_closure_identity']['test']['C_draw_count'] == 3
    assert result['sampling']['additional_independent_physical_chains'] == 0
    with np.load(output / 'closure_identity_arrays.npz') as arrays:
        assert arrays['TEST_R_squared_error'].shape == (3, 252, 3, 5)
        assert not any('direct' in name for name in arrays.files)
        assert arrays['encoding_estimator'].item() == report.HALF_SUBSET_ESTIMATOR
        assert json.loads(arrays['encoding_protocol'].item()) == report.encoding_contract(128)
    complete = json.loads((output / 'report.json').read_text())
    assert complete['report_complete'] and complete['model_gradients_none']
    assert complete['encoding_estimator'] == report.HALF_SUBSET_ESTIMATOR
    assert complete['encoding_protocol'] == report.encoding_contract(128)
    assert 'closure_identity_arrays.npz' in complete['public_files']
    assert not any(Path(name).suffix == '.pt' for name in complete['public_files'])


def test_budget_hold_is_not_complete_or_no_root_result(tmp_path, monkeypatch):
    cfg, summary, payload, manifest, sidecar = provenance()
    raw = cfg.to_dict(); raw['data']['cache'] = str(tmp_path / 'physical.pt')
    cfg = RunConfig.from_dict(raw); summary['config'] = manifest['config'] = cfg.to_dict()
    (tmp_path / 'physical.pt').write_bytes(b'not-loaded')
    dump(tmp_path / 'physical.json', sidecar)
    run = tmp_path / 'run'; run.mkdir()
    checkpoint = run / 'training_step_12000.pt'; checkpoint.write_bytes(b'checkpoint')
    dump(run / 'manifest.json', manifest); dump(run / 'summary.json', summary)
    loaded = SimpleNamespace(checkpoint=checkpoint, config=cfg.to_dict(), payload=payload,
                             model=LinearMap())
    monkeypatch.setattr(report, 'load_run', lambda *args: loaded)
    def hold(_):
        raise report.AnalysisBudgetHold('测试预算门控')
    monkeypatch.setattr(report.original, 'report_datasets', hold)
    deadline = report.Deadline(60)
    result = report.run_report(SimpleNamespace(run=run, output=tmp_path / 'report'), deadline,
                               report.Progress(tmp_path / 'progress.json', deadline))
    assert result['status'] == 'analysis_budget_hold' and not result['report_complete']
    assert result['requires_user_action'] and result['thermal_nu'] is None
    assert not result['MI_estimated'] and not result['physical_sampling_started']


def test_real_b3_two_half_estimator_matches_training_same_C_levels_without_new_rng(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(81)
        model = build_model(ModelConfig(dimension=2, theory_dim=5, hidden=12,
            block_factor=3, width=3, binary_lookup=True, kernel_sampling='bernoulli'), 2)
        # Amplify the nonlinearity so the difference from Q(full) is a visible
        # test signal rather than numerical roundoff of a nearly affine head.
        with torch.no_grad():
            model.theory_encoder.head[0].weight.mul_(15.)
        data = (torch.randint(0, 2, (2, 128, 36, 36, 1)) * 2 - 1).double()
        contract = report.encoding_contract(128)
        partition = torch.tensor(contract['sample_partition'], dtype=torch.long)
        sample = model.rg_kernel.field_head.sample_logits
        records = []
        def sampled(logits):
            before = torch.get_rng_state().clone()
            value = sample(logits)
            records.append((logits.clone(), before, value.clone(), torch.get_rng_state().clone()))
            return value
        monkeypatch.setattr(model.rg_kernel.field_head, 'sample_logits', sampled)
        global_before = torch.get_rng_state().clone()
        with isolated_rng('cpu', 12345), torch.no_grad():
            levels = model.coarse_grain(data)
            expected = half_subset_theory(model, [level.index_select(1, partition) for level in levels])
            full = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
        reference_calls = records[:]
        records.clear()
        collector = report.TrainingMatchedBinaryCollector(model)
        actual = collector.collect(data, samples=128, chunk_size=2, sampling_seed=12345)
        np.testing.assert_allclose(actual['direct'], expected.numpy(), rtol=0, atol=1e-12)
        assert np.max(abs(actual['direct'] - full.numpy())) > 1e-9
        assert len(records) == len(reference_calls) == 2
        for old, new in zip(reference_calls, records):
            assert all(torch.equal(left, right) for left, right in zip(old, new))
        assert torch.equal(global_before, torch.get_rng_state())
        assert all(parameter.grad is None for parameter in model.parameters())
        np.testing.assert_allclose(actual['one_step'],
            report.original.mapped(model, expected.numpy()[:, :-1]), rtol=0, atol=1e-12)
        # Microscopic cache holds the matching half-Q estimator, not full-Q.
        again = collector.collect(data, samples=128, chunk_size=2, sampling_seed=12345)
        assert collector.cache_hits == 1
        np.testing.assert_allclose(again['direct'], actual['direct'], rtol=0, atol=1e-12)
    finally:
        torch.set_num_threads(previous)


def test_compare_rejects_legacy_full_Q_or_different_half_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'plot_comparison', lambda rows, out: None)
    specifications = saved_reports(tmp_path / 'runs')
    value = sample_report(95)
    value['encoding_estimator'] = 'Q_of_full_ensemble_v1'
    dump(tmp_path / 'runs/95/metrics.json', value)
    with pytest.raises(ValueError, match='半子集'):
        report.compare_reports(specifications, tmp_path / 'comparison')
    value = sample_report(95)
    value['encoding_protocol'] = report.encoding_contract(128, split_seed=99)
    dump(tmp_path / 'runs/95/metrics.json', value)
    with pytest.raises(ValueError, match='划分'):
        report.compare_reports(specifications, tmp_path / 'comparison')
