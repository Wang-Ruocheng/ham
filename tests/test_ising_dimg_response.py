"""Dimension-agnostic read-only response diagnostics; no theoretical target tests."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from scripts import diagnose_ising_dimg_response as diagnostic


@pytest.fixture(params=[1, 2, 3])
def model(request):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1024)
        value = build_model(ModelConfig(dimension=2, block_factor=2, theory_dim=request.param,
                                       hidden=5, width=3, binary_lookup=True,
                                       kernel_sampling='bernoulli'), 2).double().eval()
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    return value


def binary_samples(n=31, side=8):
    generator = torch.Generator().manual_seed(224)
    return (torch.randint(0, 2, (n, side, side), generator=generator) * 2 - 1).double()


def fake_source(tmp_path, dim=2):
    source = tmp_path / 'arm' / 'run'
    source.mkdir(parents=True)
    data = tmp_path / 'never_rehash_9GB.pt'
    cfg = dict(schema_version=5, data=dict(cache=str(data)),
        model=asdict(ModelConfig(dimension=2, block_factor=2, theory_dim=dim, width=3,
                                kernel_sampling='bernoulli')),
        training=dict(max_rg_steps=2, epochs=12000, train_modules=('Q', 'C', 'R'),
                      gradient_mode='qc_r_separate'), evaluation=dict(max_rg_steps=2))
    metadata = dict(config=cfg, data_sha256=diagnostic.EXPECTED_DATA_SHA256,
                    labels_used_in_training=False, q_supervision=None)
    (source / 'manifest.json').write_text(json.dumps(metadata))
    loaded = SimpleNamespace(payload=dict(metadata=metadata, epoch=12000), config=cfg)
    return source, data, loaded, dict(config=cfg, actual_epochs=12000, requested_epochs=12000,
                                     final_report_skipped=True)


def test_head_and_R_jacobians_generalize_and_preserve_parameters(model):
    features = diagnostic.local_feature_means(model, binary_samples())
    scores = diagnostic.operator_values(binary_samples())
    before = diagnostic.model_sha256(model)
    g, response, fd, std = diagnostic.weighted_q_response(model, features, scores)
    dimension = model.theory_encoder.head[-1].out_features
    assert g.shape == (dimension,)
    assert response.shape == (dimension, 7)
    assert fd.shape == (2, dimension, 7)
    assert torch.allclose(response, fd[0], atol=1e-8, rtol=2e-5)
    full = diagnostic.full_r_jacobian(model, g)
    finite = diagnostic.full_r_finite_difference(model, g)
    assert full.shape == finite.shape == (dimension, dimension)
    assert torch.allclose(full, finite, atol=2e-10, rtol=2e-9)
    assert diagnostic.model_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_R_jacobian_retains_identity_in_every_dimension(model):
    dimension = model.theory_encoder.head[-1].out_features
    with torch.no_grad():
        for parameter in model.theory_rg.parameters():
            parameter.zero_()
    g = torch.linspace(-.2, .3, dimension, dtype=torch.float64)
    assert torch.equal(diagnostic.full_r_jacobian(model, g), torch.eye(dimension, dtype=torch.float64))
    assert torch.allclose(diagnostic.full_r_finite_difference(model, g),
                          torch.eye(dimension, dtype=torch.float64), atol=2e-12, rtol=0)


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_best_matrix_recovers_known_full_rank_response_without_scalar_assumption(dimension):
    rng = np.random.default_rng(17)
    incoming = rng.normal(size=(dimension, 7))
    matrix = rng.normal(size=(dimension, dimension))
    std = np.exp(rng.normal(size=7))
    fit = diagnostic.best_local_jacobian(incoming, matrix @ incoming, std)
    assert fit['input_rank'] == dimension
    assert np.allclose(fit['jacobian'], matrix, atol=1e-13, rtol=0)
    assert np.allclose(fit['projector'], np.eye(dimension), atol=2e-14)
    assert fit['relative_error'] < 1e-13
    assert fit['unidentifiable_output_fraction'] < 1e-13


def test_rank_deficiency_identifies_only_the_input_response_span():
    incoming = np.array([[1., 2., 3., 4., 5., 6.], [2., 4., 6., 8., 10., 12.], [0., 0., 0., 0., 0., 0.]])
    matrix = np.array([[3., 2., 9.], [-1., .5, 11.], [4., 2., -5.]])
    fit = diagnostic.best_local_jacobian(incoming, matrix @ incoming, np.ones(6))
    assert fit['input_rank'] == 1
    assert fit['input_effective_rank'] == pytest.approx(1.)
    assert np.allclose(fit['jacobian'] @ incoming, matrix @ incoming)
    assert np.allclose(fit['jacobian'], matrix @ fit['projector'])
    # The unused third column is not an inferred physical zero-eigenmode.
    assert np.allclose(fit['jacobian'][:, 2], 0)
    assert not np.allclose(fit['jacobian'], matrix)


def test_best_matrix_uses_unit_score_normalization_not_large_operator_magnitude():
    incoming = np.array([[1., 1.]])
    outgoing = np.array([[1., 3.]])
    fit = diagnostic.best_local_jacobian(incoming, outgoing, [1., 2.])
    assert fit['jacobian'][0, 0] == pytest.approx((1. + 3. / 4.) / (1. + 1. / 4.))
    # Simultaneously rescaling one physical score and its SD is immaterial.
    equivalent = diagnostic.best_local_jacobian(incoming * [100., .02], outgoing * [100., .02], [100., .04])
    assert np.allclose(fit['jacobian'], equivalent['jacobian'])


def test_zero_response_is_rank_zero_and_finite_not_fake_identity():
    fit = diagnostic.best_local_jacobian(np.zeros((3, 7)), np.ones((3, 7)), np.ones(7))
    assert fit['input_rank'] == 0
    assert fit['input_effective_rank'] == 0
    assert fit['retained_condition_number'] is None
    assert fit['relative_error'] == pytest.approx(1)
    assert np.isfinite(fit['jacobian']).all()
    assert np.allclose(fit['projector'], 0)


def test_all_spectral_modes_retained_even_if_energy_overlaps_nonleading_mode():
    matrix = np.diag([4.2, 2.00001, -.8])
    modes = diagnostic.energy_mode_diagnostics(matrix, [0., 1., 0.])
    assert np.allclose(np.asarray(modes['eigenvalues'])[:, 0], [4.2, 2.00001, -.8])
    assert modes['energy_right_overlap'] == [0., 1., 0.]
    assert modes['energy_modal_excitation_weights'] == [0., 1., 0.]
    assert 'thermal_eigenvalue' not in modes
    # Energy is unchanged while the largest-modulus eigenvalue is negative.
    modes = diagnostic.energy_mode_diagnostics(np.diag([-9., 1.3]), [0., 1.])
    assert modes['eigenvalues'][0][0] == -9
    assert modes['energy_right_overlap'] == [0., 1.]


def test_complex_modes_and_zero_energy_not_dropped_or_called_thermal():
    modes = diagnostic.energy_mode_diagnostics(np.array([[.3, -1.], [1., .3]]), [0., 0.])
    values = np.asarray(modes['eigenvalues'])
    assert np.allclose(values[:, 0], .3)
    assert sorted(values[:, 1]) == pytest.approx([-1., 1.])
    assert modes['energy_response_norm'] == 0
    assert modes['energy_right_overlap'] == [0., 0.]


def test_source_guard_accepts_tuple_serialization_and_skipped_historical_plotter(tmp_path):
    source, data, loaded, summary = fake_source(tmp_path)
    assert not data.exists()  # no full-data scan or fresh hash in the guard
    manifest = diagnostic.validate_source(source, data, loaded, summary)
    assert manifest['config']['training']['train_modules'] == ['Q', 'C', 'R']
    assert isinstance(loaded.config['training']['train_modules'], tuple)


@pytest.mark.parametrize('kind', ['checkpoint_label', 'manifest_label', 'supervision', 'hash', 'wrong_config', 'epoch', 'wall_cap'])
def test_source_guard_rejects_supervision_fingerprint_config_and_incomplete_endpoints(tmp_path, kind):
    source, data, loaded, summary = fake_source(tmp_path)
    if kind == 'checkpoint_label':
        loaded.payload['metadata']['labels_used_in_training'] = True
    elif kind == 'manifest_label':
        path = source / 'manifest.json'
        payload = json.loads(path.read_text()); payload['labels_used_in_training'] = True
        path.write_text(json.dumps(payload))
    elif kind == 'supervision':
        loaded.payload['metadata']['q_supervision'] = dict(weight=10)
    elif kind == 'hash':
        loaded.payload['metadata']['data_sha256'] = 'newhash'
    elif kind == 'wrong_config':
        loaded.config['model']['theory_dim'] = 4
    elif kind == 'epoch':
        summary['actual_epochs'] = 11999
    else:
        summary['stopped_by_wall_limit'] = True
    with pytest.raises(ValueError):
        diagnostic.validate_source(source, data, loaded, summary)


def test_output_and_checkpoint_pin_guards_run_before_any_model_inference(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    monkeypatch.setattr(diagnostic, 'load_run', lambda *_: pytest.fail('Must reject before loading'))
    with pytest.raises(FileExistsError):
        diagnostic.diagnose(source, tmp_path / 'data.pt', source)
    with pytest.raises(ValueError, match='separate'):
        diagnostic.diagnose(source, tmp_path / 'data.pt', source / 'child')
    (source / 'training_step_12000.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='fingerprint'):
        diagnostic.diagnose(source, tmp_path / 'data.pt', tmp_path / 'output', expected_checkpoint_sha256='wrong')


def synthetic_arrays(dimension):
    rng = np.random.default_rng(27)
    # K, original physical replicas, C draws, transition, d, operators
    vin = rng.normal(size=(1, 3, 2, 2, dimension, 7))
    matrix = np.diag(np.arange(1, dimension + 1) + .3)
    vout = np.einsum('ij,...jp->...ip', matrix, vin)
    full = np.broadcast_to(matrix, (1, 3, 2, 3, dimension, dimension)).copy()
    std = np.ones((1, 3, 2, 3, 7))
    shape = (1, 3, 2, 2)
    arrays = dict(K=np.array([diagnostic.KC]), q_response_in=vin, q_response_out=vout,
        q_response_predicted=vout.copy(), full_DR=full, r_fd_jacobian=full.copy(),
        score_std=std, g=rng.normal(size=(1, 3, 2, 3, dimension)),
        q_fd_response_in=np.stack([vin, vin], axis=-3), q_fd_response_out=np.stack([vout, vout], axis=-3),
        best_J_even=np.empty(shape + (dimension, dimension)),
        response_input_rank_even=np.empty(shape, int), response_input_effective_rank_even=np.empty(shape),
        response_input_singular_values_even=np.empty(shape + (dimension,)),
        best_response_residual_even=np.empty(shape + (dimension, 6)))
    for index in np.ndindex(shape):
        fit = diagnostic.best_local_jacobian(vin[index][..., :6], vout[index][..., :6], std[index][..., :6])
        arrays['best_J_even'][index] = fit['jacobian']
        arrays['response_input_rank_even'][index] = fit['input_rank']
        arrays['response_input_effective_rank_even'][index] = fit['input_effective_rank']
        arrays['response_input_singular_values_even'][index] = fit['input_singular_values']
        arrays['best_response_residual_even'][index] = fit['residual']
    return arrays


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_analysis_axes_keep_dimensionality_and_cluster_physical_replicas(dimension):
    arrays = synthetic_arrays(dimension)
    indices = np.random.default_rng(2).integers(0, 3, (100, 3))
    rows, fd = diagnostic.response_analysis(arrays, indices)
    assert len(rows) == 2
    for row in rows:
        assert len(row['mean_g']) == dimension
        assert np.asarray(row['full_DR_mean']).shape == (dimension, dimension)
        assert len(row['existing_R_relative_error_per_replica']) == 3  # not 3*2 independent draws
        assert row['existing_R_relative_error_mean'] < 1e-13
        assert row['best_J_relative_error_mean'] < 1e-13
        assert row['input_response_rank_counts'] == {str(dimension):6}
        assert len(row['full_DR_modes_per_replica_draw']) == 6
    assert all(value['max_absolute_error'] == 0 for value in fd.values())


def test_invalid_response_geometry_fails_not_silently_broadcast():
    with pytest.raises(ValueError):
        diagnostic.best_local_jacobian(np.ones((2, 7)), np.ones((1, 7)), np.ones(7))
    with pytest.raises(ValueError):
        diagnostic.best_local_jacobian(np.ones((2, 7)), np.ones((2, 7)), -np.ones(7))


def test_physical_energy_projection_keeps_all_modes_and_uses_similarity_coordinates():
    rng = np.random.default_rng(113)
    covariance = np.diag([9., 1., 2., 3., 4., 5., 6.])
    transform = np.diag([1.3, 4.2, .8, .6, -.4, .2, .1])
    arrays = dict(covariance=np.broadcast_to(covariance, (1, 2, 2, 3, 7, 7)),
        cross_covariance=np.broadcast_to(covariance @ transform, (1, 2, 2, 2, 7, 7)),
        orientation_operators=rng.normal(size=(1, 2, 2, 3, 128, 7)))
    rows = [dict(K_index=0, transition=0, family='isotropic', operator_indices=[0, 1], rcond=1e-6)]
    actual = diagnostic.add_physical_energy_projections(rows, arrays)[0]
    assert np.allclose(np.asarray(actual['energy_projection_eigenvalues'])[:, 0], [4.2, 1.3])
    assert actual['energy_right_overlap'] == pytest.approx([0., 1.])
    assert actual['energy_modal_excitation_weights'] == pytest.approx([0., 1.])
    assert actual['energy_perturbation_retained_fraction'] == pytest.approx(1.)
    assert 'thermal_eigenvalue' not in actual


def test_real_chinese_wrapper_draws_all_public_figures(tmp_path):
    arrays = synthetic_arrays(3)
    indices = np.random.default_rng(2).integers(0, 3, (100, 3))
    rows, fd = diagnostic.response_analysis(arrays, indices)
    spectra = [dict(family='isotropic', K_index=0, transition=step, basis=str(n), rcond=rcond,
        leading_positive_real=1.7, bootstrap=dict(leading_positive_real_percentiles=[1.5, 1.7, 1.9]))
        for step in range(2) for n in (1, 3, 5, 6) for rcond in (1e-8, 1e-6, 1e-4)]
    diagnostic.draw_report(dict(theory_dim=3, K=arrays['K'], spectra=spectra, q_response=rows), tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        'mcrg_basis.png', 'vector_response.png', 'response_conditioning.png'}
