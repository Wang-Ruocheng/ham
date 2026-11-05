"""Read-only MCRG sufficient-statistic collection and exact empirical responses."""
import numpy as np
import json
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from scripts import diagnose_ising_mcrg as diagnostic


@pytest.fixture
def model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1024)
        value = build_model(ModelConfig(dimension=2, block_factor=2, theory_dim=1,
                                       hidden=5, width=3, binary_lookup=True,
                                       kernel_sampling='bernoulli'), 2).double().eval()
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    return value


def binary_samples(n=13, side=8):
    generator = torch.Generator().manual_seed(224)
    return (torch.randint(0, 2, (n, side, side), generator=generator) * 2 - 1).double()


def test_operator_extensive_normalization_and_even_odd_separation():
    samples = binary_samples()
    operators = diagnostic.operator_values(samples)
    opposite = diagnostic.operator_values(-samples)
    assert torch.equal(operators[:, :6], opposite[:, :6])
    assert torch.equal(operators[:, 6], -opposite[:, 6])
    uniform = diagnostic.operator_values(torch.ones(2, 8, 8))
    assert torch.equal(uniform, torch.tensor([[128., 128., 64., 128., 128., 128., 64.]] * 2))
    half = diagnostic.operator_values(torch.ones(2, 4, 4))
    assert torch.equal(uniform, 4 * half)


def test_operator_directional_groups_exactly_reconstruct_isotropic_entries():
    samples = binary_samples()
    all_ops = diagnostic.operator_values(samples)
    directions = diagnostic.orientation_operator_values(samples)
    assert torch.equal(all_ops[:, 0], directions[:, 0] + directions[:, 1])
    assert torch.equal(all_ops[:, 1], directions[:, 2] + directions[:, 3])
    assert torch.equal(all_ops[:, 2], directions[:, 6])
    assert torch.equal(all_ops[:, 3], directions[:, 4] + directions[:, 5])


def test_covariance_orientation_and_different_finite_sample_denominators():
    fine = torch.tensor([[1., 3.], [2., 7.], [5., 1.]], dtype=torch.float64)
    coarse = torch.tensor([[9.], [2.], [4.]], dtype=torch.float64)
    actual = diagnostic.covariance(coarse, fine)
    expected = np.cov(coarse.numpy().T, fine.numpy().T, ddof=1)[0:1, 1:]
    assert np.allclose(actual, expected)
    assert torch.allclose(diagnostic.covariance(coarse, fine, ddof=0), actual * 2 / 3)
    with pytest.raises(ValueError):
        diagnostic.covariance(coarse[:1], fine[:1])


def test_probability_lookup_matches_saved_C_forward_distribution_and_rng(model):
    spins = binary_samples(n=3)
    _, p = diagnostic.c_probability_table(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        actual = model.rg_kernel(spins[None, ..., None])[0, ..., 0]
        torch.manual_seed(44)
        uniform = torch.rand(actual.shape, dtype=torch.float64)
        expected = diagnostic.c_draw(spins, p, uniform)
    # The original STE hard + soft - soft can retain machine-epsilon roundoff;
    # the diagnostic stores exact hard spins, never differentiates through C.
    assert torch.equal(actual.sign(), expected)
    assert torch.allclose(actual, expected, atol=1e-15, rtol=0)


def test_complemented_uniform_exact_spin_flip_for_symmetric_kernel(model):
    patterns, _ = diagnostic.c_probability_table(model)
    logits = .4 * patterns.sum(1)
    p = torch.sigmoid(2 * logits)
    assert torch.allclose(p + p.flip(0), torch.ones_like(p))
    spins = binary_samples(n=4)
    generator = torch.Generator().manual_seed(5)
    u = torch.rand(4, 4, 4, generator=generator, dtype=torch.float64)
    a = diagnostic.c_draw(spins, p, u)
    b = diagnostic.c_draw(-spins, p, 1 - u)
    assert torch.equal(a, -b)


def test_C_D4_enumeration_detects_anisotropy(model):
    patterns, _ = diagnostic.c_probability_table(model)
    symmetric = torch.sigmoid(patterns.sum(1))
    assert diagnostic.d4_probability_errors(patterns, symmetric).abs().max() == 0
    anisotropic = torch.sigmoid(patterns[:, 0])
    errors = diagnostic.d4_probability_errors(patterns, anisotropic)
    assert errors.shape == (8, 16)
    assert errors.abs().max() > .4
    assert torch.equal(errors[0], torch.zeros(16, dtype=torch.float64))


def test_local_feature_lookup_retains_sample_axis_and_matches_Q(model):
    spins = binary_samples(n=7)
    actual = diagnostic.local_feature_means(model, spins)
    with torch.no_grad():
        patches = model.geometry.extract_patches(spins[None, ..., None])
        expected = model.theory_encoder.local_mlp(patches).mean((2, 3))[0]
        direct = model.theory_encoder(spins[None, ..., None])[0]
    assert actual.shape[0] == len(spins)
    assert torch.allclose(actual, expected, atol=2e-15, rtol=0)
    g, _ = diagnostic.head_jacobian(model, actual.mean(0))
    assert torch.allclose(g, direct, atol=2e-15, rtol=0)


def test_empirical_weighted_Q_response_uses_population_covariance_and_matches_FD(model):
    spins = binary_samples(n=31)
    features = diagnostic.local_feature_means(model, spins)
    score = diagnostic.operator_values(spins)
    before = diagnostic.model_sha256(model)
    g, analytic, finite, std = diagnostic.weighted_q_response(model, features, score)
    _, jac = diagnostic.head_jacobian(model, features.mean(0))
    independent = jac @ ((features - features.mean(0)).T @ (score - score.mean(0)) / len(score))
    assert torch.allclose(analytic, independent, atol=1e-15, rtol=0)
    assert torch.allclose(finite[0], analytic, atol=1e-8, rtol=2e-5)
    assert torch.allclose(finite[1], analytic, atol=1e-7, rtol=2e-4)
    assert torch.allclose(std, score.std(0, correction=0))
    assert diagnostic.model_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_constant_score_has_exact_zero_empirical_Q_response(model):
    features = diagnostic.local_feature_means(model, binary_samples())
    score = torch.ones(len(features), 7, dtype=torch.float64)
    _, analytic, finite, std = diagnostic.weighted_q_response(model, features, score)
    assert analytic.abs().max() == 0
    assert finite.abs().max() == 0
    assert std.abs().max() == 0


def test_full_R_derivative_includes_identity(model):
    with torch.no_grad():
        for parameter in model.theory_rg.parameters():
            parameter.zero_()
    assert torch.equal(diagnostic.full_r_jacobian(model, torch.tensor([.2], dtype=torch.float64)),
                       torch.ones(1, 1, dtype=torch.float64))


def test_selection_is_posthoc_exact_Kc_plus_adjacent_original_cells():
    knots = np.array([.2, .4, diagnostic.KC, .45, .5])
    actual = diagnostic.select_points(np.repeat(knots, 12), 3)
    assert np.array_equal(actual, knots[1:4])
    with pytest.raises(ValueError, match='exact Kc'):
        diagnostic.select_points([.3, .4, .5], 3)
    with pytest.raises(ValueError, match='odd'):
        diagnostic.select_points(knots, 2)


def test_mmap_loader_preserves_original_split_and_does_not_validate_whole_cache(tmp_path, monkeypatch):
    x = binary_samples(n=3)[:, None, ..., None]
    path = tmp_path / 'cache.pt'
    torch.save((x, torch.tensor([.2, .3, .4]), x, torch.tensor([.5, .6, .7])), path)
    from recurrent_rg.data.ensembles import EnsembleDataset
    monkeypatch.setattr(EnsembleDataset, 'validate', lambda *_: pytest.fail('Full data audit is forbidden'))
    train, test = diagnostic.mmap_dataset(path)
    assert train[2] == 'train' and test[2] == 'test'
    assert torch.equal(train[0], x)


def test_output_overlap_and_overwrite_rejected_before_any_inference(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(diagnostic, 'load_run', lambda *_: pytest.fail('Must reject before loading'))
    with pytest.raises(FileExistsError):
        diagnostic.diagnose(source, tmp_path / 'data.pt', source)
    with pytest.raises(ValueError, match='separate'):
        diagnostic.diagnose(source, tmp_path / 'data.pt', source / 'child')


def test_selected_spin_validation_rejects_invalid_values():
    bad = binary_samples()
    bad[0, 0, 0] = .2
    with pytest.raises(ValueError, match='binary'):
        diagnostic.operator_values(bad)


def test_source_receipt_and_config_cache_guards_do_not_read_physical_data(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    # Deliberately absent: provenance validation must not open/re-hash this.
    data = tmp_path / 'never_read_9GB_cache.pt'
    metadata = dict(config=dict(data=dict(cache=str(data))), data_sha256=diagnostic.EXPECTED_DATA_SHA256,
                    labels_used_in_training=False)
    manifest_path = source / 'manifest.json'
    manifest_path.write_text(json.dumps(metadata))
    summary = dict(actual_epochs=12000)
    diagnostic.validate_source_provenance(source, data, metadata, summary)
    with pytest.raises(ValueError, match='physical cache'):
        diagnostic.validate_source_provenance(source, tmp_path / 'other.pt', metadata, summary)
    bad = dict(metadata, data_sha256='wrong')
    with pytest.raises(ValueError, match='fingerprint'):
        diagnostic.validate_source_provenance(source, data, bad, summary)
    receipt_path = tmp_path / 'job.json'
    receipt = dict(status='complete', workflow_complete=True, training=dict(actual_epochs=12000,
        data_sha256=diagnostic.EXPECTED_DATA_SHA256, manifest_sha256=diagnostic.sha256_file(manifest_path)))
    receipt_path.write_text(json.dumps(receipt))
    diagnostic.validate_source_provenance(source, data, metadata, summary)
    receipt['workflow_complete'] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='verifiably complete'):
        diagnostic.validate_source_provenance(source, data, metadata, summary)


def test_provenance_config_serialized_tuple_list_equivalence_not_semantic_changes(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    data = tmp_path / 'never_read_9GB_cache.pt'
    metadata = dict(config=dict(data=dict(cache=str(data)),
                               training=dict(train_modules=('Q', 'C', 'R'))),
                    data_sha256=diagnostic.EXPECTED_DATA_SHA256,
                    labels_used_in_training=False)
    path = source / 'manifest.json'
    path.write_text(json.dumps(metadata))
    manifest = json.loads(path.read_text())
    assert isinstance(metadata['config']['training']['train_modules'], tuple)
    assert isinstance(manifest['config']['training']['train_modules'], list)
    diagnostic.validate_source_provenance(source, data, metadata, dict(actual_epochs=12000))
    manifest['config']['training']['train_modules'] = ['R']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='provenance mismatch'):
        diagnostic.validate_source_provenance(source, data, metadata, dict(actual_epochs=12000))
