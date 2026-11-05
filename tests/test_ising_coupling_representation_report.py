"""自然耦合正式报告不得误标损失单位或依理论筛谱。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising_coupling_representation as report


def description(arm='natural'):
    config = dict(model=dict(affine_reconstruction=arm == 'control'))
    manifest = dict(config=config)
    if arm == 'natural':
        manifest['representation_override'] = dict(kind=report.NATURAL_KIND, seven_even_operators=True)
    return report.representation_description(manifest)


@pytest.mark.parametrize('arm', ['control', 'natural'])
def test_explicit_representation_contract_and_loss_units(arm):
    row = description(arm)
    assert row['arm'] == arm and row['thermal_identity_certified'] is False
    assert row['exact_RG_claim'] is False
    if arm == 'natural':
        assert '条件' in row['reconstruction_units'] and '不是' in row['reconstruction_units']


@pytest.mark.parametrize('manifest', [dict(config=dict(model=dict(affine_reconstruction=False))),
    dict(config=dict(model=dict(affine_reconstruction=True)), representation_override=dict(kind=report.NATURAL_KIND)),
    dict(config=dict(model=dict(affine_reconstruction=False)), representation_override=dict(kind='unknown'))])
def test_unrecognized_or_contradictory_architecture_rejected(manifest):
    with pytest.raises(ValueError):
        report.representation_description(manifest)


def test_all_inverse_branches_and_projection_distance_retained():
    roots = [dict(root_id='F1', g=[.2], curve_projections_by_scale=[
        dict(exact_inverse_K_branches=[.41, .47], nearest_distance=0., nearest_segments=[]),
        dict(exact_inverse_K_branches=[.42, .48], nearest_distance=0., nearest_segments=[]),
        dict(exact_inverse_K_branches=[], nearest_distance=.003,
             nearest_segments=[dict(K=.44, distance=.003)])])]
    row = report.root_location_records(roots)[0]
    assert len(row['scales'][0]['exact_inverse_K_branches']) == 2
    assert len(row['all_adjacent_exact_branch_drifts']) == 4
    assert row['scales'][2]['has_exact_inverse'] is False
    assert row['scales'][2]['projection_is_critical_point'] is False
    assert row['scales'][2]['nearest_distance'] == .003
    assert row['scales'][0]['exact_inverse_K_branches'][0]['relative_K_error'] == pytest.approx(abs(.41-report.base.KC)/report.base.KC)


def test_loader_dispatch_preserves_original_control(tmp_path, monkeypatch):
    from experiments import ising_coupling_representation as train
    calls = []
    monkeypatch.setattr(report, 'load_run', lambda path, ckpt: calls.append(('control', path, ckpt)) or 'old')
    monkeypatch.setattr(train, 'load_coupling_run', lambda path, ckpt: calls.append(('natural', path, ckpt)) or 'new')
    for arm in ('control', 'natural'):
        folder = tmp_path/arm
        folder.mkdir()
        manifest = dict(config=dict(model=dict(affine_reconstruction=arm == 'control')))
        if arm == 'natural':
            manifest['representation_override'] = dict(kind=report.NATURAL_KIND)
        (folder/'manifest.json').write_text(json.dumps(manifest))
        assert report.load_representation_run(folder, 12000) == ('old' if arm == 'control' else 'new')
    assert [row[0] for row in calls] == ['control', 'natural']


def formal_receipts(tmp_path, arm='natural'):
    from recurrent_rg.config import RunConfig
    data = tmp_path/'data.pt'
    cfg = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(data), physical_model='ising2d', coupling_names=['K']),
        model=dict(dimension=2, theory_dim=1, block_factor=2, width=3,
                   affine_reconstruction=arm == 'control'),
        training=dict(epochs=12000, max_rg_steps=2, gradient_mode='qc_r_separate'),
        evaluation=dict(max_rg_steps=2)))
    canonical = json.loads(json.dumps(cfg.to_dict()))
    metadata = dict(config=canonical, labels_used_in_training=False, q_supervision=None, data_sha256='audited')
    if arm == 'natural':
        metadata['representation_override'] = dict(kind=report.NATURAL_KIND)
    loaded = SimpleNamespace(config=canonical, payload=dict(epoch=12000, metadata=deepcopy(metadata),
        history=[dict(epoch=12000)]))
    report.base.write_json(tmp_path/'manifest.json', metadata)
    report.base.write_json(tmp_path/'summary.json', dict(config=canonical, actual_epochs=12000,
        requested_epochs=12000, history=[dict(epoch=12000)]))
    report.base.write_json(data.with_suffix('.json'), dict(B=0., coupling_names=['K'], data_sha256='audited'))
    return loaded


@pytest.mark.parametrize('arm', ['control', 'natural'])
def test_formal_validation_accepts_explicit_original_or_custom_metadata(tmp_path, arm):
    loaded = formal_receipts(tmp_path, arm)
    _, _, _, actual = report.validate_formal_run(tmp_path, loaded)
    assert actual['arm'] == arm


@pytest.mark.parametrize('failure', ['partial', 'labels', 'sidecar', 'config'])
def test_formal_validation_rejects_early_stop_labels_or_fingerprint_mismatch(tmp_path, failure):
    loaded = formal_receipts(tmp_path)
    if failure == 'partial':
        summary = report.read_json(tmp_path/'summary.json')
        summary['actual_epochs'] = 11999
        report.base.write_json(tmp_path/'summary.json', summary)
    elif failure == 'labels':
        loaded.payload['metadata']['labels_used_in_training'] = True
    elif failure == 'sidecar':
        report.base.write_json(tmp_path/'data.json', dict(B=0., coupling_names=['K'], data_sha256='other'))
    else:
        loaded.payload['metadata']['config']['model']['theory_dim'] = 3
    with pytest.raises(ValueError):
        report.validate_formal_run(tmp_path, loaded)


class LinearMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theory_encoder = SimpleNamespace(distribution_decoder=SimpleNamespace(
            linear=SimpleNamespace(weight=torch.ones((7, 1), dtype=torch.float64),
                                   bias=torch.zeros(7, dtype=torch.float64))))

    def advance_theory(self, point):
        return 1.7*point


def test_natural_public_report_truthful_and_completion_deferred(tmp_path):
    model = LinearMap()
    labels = np.linspace(.2, .68, 12)
    x = (labels-report.base.KC)[:, None]
    direct = np.stack((x, 1.7*x, 1.7**2*x), axis=1)
    values = dict(direct=direct, one_step=1.7*direct[:, :-1], rollout=direct)
    loaded = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=2),
        checkpoint=tmp_path.parent/'run'/'training_step_12000.pt', payload=dict(epoch=12000))
    history = [dict(epoch=12000, train_total=.1, test_total=.2, train_rsmi_bits=.3,
        test_rsmi_bits=.31, train_reconstruction_nll_level_0=.4, test_reconstruction_nll_level_0=.41)]
    result = report.write_representation_report(loaded, values, values, labels, labels, history,
        tmp_path, representation=description())
    metrics = report.read_json(tmp_path/'metrics.json')
    completion = report.read_json(tmp_path/'report.json')
    assert completion['report_complete'] is True
    assert completion['representation']['arm'] == 'natural'
    assert metrics['representation']['reconstruction_units'] == description()['reconstruction_units']
    assert '单位不可直接横比' in metrics['logged_loss_caution']
    assert len(metrics['roots']) == 1 and metrics['roots'][0]['thermal_nu'] is None
    assert len(metrics['roots'][0]['eigenvalues']) == 1
    assert (tmp_path/'coupling_decoder.json').is_file()
    assert (tmp_path/'loss_comparison.png').is_file()
    assert not list(tmp_path.glob('*.pt'))
    assert result['representation']['exact_RG_claim'] is False


def test_failed_underlying_report_cannot_leave_complete_marker(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        report.base.write_json(tmp_path/'report.json', dict(status='complete', report_complete=True))
        raise RuntimeError('synthetic failure')
    original_writer = report.base.write_json
    monkeypatch.setattr(report.base, 'write_report', fail)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        report.write_representation_report(None, None, None, None, None, [], tmp_path,
                                           representation=description('control'))
    assert not (tmp_path/'report.json').exists()
    assert report.base.write_json is original_writer


def create_arm(tmp_path, name):
    arm = tmp_path/name
    (arm/'report').mkdir(parents=True)
    (arm/'response').mkdir()
    dim = 1 if name.startswith('dim1_') else 3
    representation = description('natural' if name.endswith('_natural') else 'control')
    metrics = dict(checkpoint_epoch=12000, theory_dim=dim, block_factor=2,
        labels_used_in_training=False, representation=representation,
        roots=[dict(root_id='F1', g=[0.]*dim, geometric_support_candidate=False,
            eigenvalues=[dict(real=1.7, imag=0.), dict(real=4., imag=0.), dict(real=-.5, imag=.2)][:dim],
            curve_projections_by_scale=[dict(exact_inverse_K_branches=[], nearest_distance=.003,
                nearest_segments=[dict(K=.44, distance=.003)])]*3)],
        closure=dict(test=dict(all_start_horizon=[dict(start=a, end=b, rmse=.002, nrmse=.01)
            for a, b in ((0, 1), (1, 2), (0, 2))])))
    report.base.write_json(arm/'report'/'metrics.json', metrics)
    report.base.write_json(arm/'report'/'report.json', dict(status='complete', report_complete=True))
    report.base.write_json(arm/'response'/'status.json', dict(status='complete', workflow_complete=True))
    report.base.write_json(arm/'response'/'comparison.json', dict(status='complete', theory_dim=dim,
        representation=representation, spectra=[dict(all_modes=[[1.9, 0.], [-.2, .3]])]))
    return arm


def test_four_arm_comparison_retains_every_root_spectrum_and_physical_response(tmp_path):
    arms = [f'{name}={create_arm(tmp_path, name)}' for name in report.ARM_NAMES]
    result = report.compare_reports(arms, tmp_path/'comparison')
    assert tuple(result['reports']) == report.ARM_NAMES
    assert result['selection']['selected_thermal_eigenvalue'] is None
    assert len(result['reports']['dim3_natural']['roots'][0]['eigenvalues']) == 3
    assert result['physical_response']['dim3_natural']['spectra'][0]['all_modes'] == [[1.9, 0.], [-.2, .3]]
    assert (tmp_path/'comparison'/'comparison.png').is_file()
    assert report.read_json(tmp_path/'comparison'/'report.json')['workflow_complete'] is True


@pytest.mark.parametrize('mutation', ['partial', 'wrong_dim', 'missing_response', 'wrong_representation'])
def test_comparison_fail_closed_without_all_four_valid_endpoints(tmp_path, mutation):
    paths = {name: create_arm(tmp_path, name) for name in report.ARM_NAMES}
    metrics_path = paths['dim3_natural']/'report'/'metrics.json'
    metrics = report.read_json(metrics_path)
    if mutation == 'partial':
        metrics['checkpoint_epoch'] = 11999
    elif mutation == 'wrong_dim':
        metrics['theory_dim'] = 2
    elif mutation == 'wrong_representation':
        metrics['representation']['arm'] = 'control'
    else:
        report.base.write_json(paths['dim3_natural']/'response'/'status.json', dict(status='running'))
    report.base.write_json(metrics_path, metrics)
    with pytest.raises(ValueError):
        report.comparison_inputs([f'{name}={path}' for name, path in paths.items()])


def test_response_customloader_and_writer_hooks_always_restored(monkeypatch):
    from scripts import diagnose_ising_dimg_response as response
    original_loader, original_write = response.load_run, response.write_json
    monkeypatch.setattr(report, 'load_representation_run', lambda *args: 'loaded')
    monkeypatch.setattr(report, 'validate_formal_run', lambda *args: (None, None, None, description()))
    def fail(**kwargs):
        assert response.load_run is report.load_representation_run
        assert kwargs['c_repeats'] == 5 and kwargs['bootstraps'] == 1000 and kwargs['smoke'] is False
        raise RuntimeError('failure')
    monkeypatch.setattr(response, 'diagnose', fail)
    with pytest.raises(RuntimeError):
        report.response_run('run', 'data', 'output')
    assert response.load_run is original_loader and response.write_json is original_write


def test_cli_compatibility_aliases(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(report, 'compare_reports', lambda arms, output: calls.append((arms, output)))
    report.main(['--mode', 'compare', '--compare', 'dim1_control=a', '--output', str(tmp_path/'output')])
    assert calls == [(['dim1_control=a'], tmp_path/'output')]


@pytest.mark.parametrize('options', [dict(threads=2), dict(sampling_repeats=0)])
def test_no_reduced_formal_report_protocol(tmp_path, options):
    with pytest.raises(ValueError):
        report.report_run(tmp_path/'not_read', tmp_path/'output', **options)
