"""Synthetic CPU checks of common readouts, not real-data baseline experiments."""
import json

import numpy as np
import pytest
import torch

from recurrent_rg.baselines.readout import (
    AffineReadout, PCAReadout, Q_MOTIF_ORDERING, TARGET_MOTIF_ORDERING,
    build_r, fit_affine_readout, fit_train_pca, fixed_coarsen,
    motif_probabilities, path_metrics, physical_error_decomposition, validate_spins,
)
from recurrent_rg.binary_pooling import binary_frequencies
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.geometry import square_geometry
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def spins(shape, seed=91):
    return np.random.default_rng(seed).choice(np.array([-1, 1], dtype=np.int8), size=shape)


def probabilities(shape=(10, 3, 512), seed=92):
    values = np.random.default_rng(seed).uniform(.1, 1., shape)
    return values / values.sum(-1, keepdims=True)


def test_fixed_decimation_matches_geometry_with_arbitrary_prefix_axes():
    x = spins((2, 3, 4, 12, 12))
    before = x.copy()
    actual = fixed_coarsen(x, "decimation")
    np.testing.assert_array_equal(actual, x[..., ::3, ::3])
    tensor = torch.from_numpy(x.reshape(6, 4, 12, 12, 1))
    expected = square_geometry(3, 3).downsample(tensor)[..., 0].numpy().reshape(actual.shape)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, before)
    assert actual.dtype == x.dtype
    actual[...] = -actual
    np.testing.assert_array_equal(x, before)


def test_fixed_majority_is_exact_odd_spin_flip_equivariant_no_tie_rng():
    x = spins((2, 3, 18, 12))
    expected = np.sign(x.reshape(2, 3, 6, 3, 4, 3).sum(axis=(3, 5)))
    actual = fixed_coarsen(x, "majority")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fixed_coarsen(-x, "majority"), -actual)
    tensor = torch.from_numpy(x).double()
    state = torch.random.get_rng_state().clone()
    result = fixed_coarsen(tensor, "majority")
    assert isinstance(result, torch.Tensor) and result.dtype == torch.float64
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("value,kwargs", [
    (np.ones((2, 8, 8)), dict(rule="majority", block=2)),
    (np.ones((2, 8, 9)), dict(rule="decimation")),
    (np.zeros((2, 9, 9)), dict(rule="decimation")),
    (np.ones((2, 9, 9)) * .9, dict(rule="majority")),
    (np.ones((2, 9, 9)), dict(rule="other")),
    (np.ones((2, 9, 9)), dict(rule="majority", block=True)),
])
def test_fixed_rules_reject_invalid_data_or_implicit_ties(value, kwargs):
    with pytest.raises(ValueError):
        fixed_coarsen(value, **kwargs)


def test_binary_roundoff_is_accepted_but_graph_bool_empty_and_nonfinite_rejected():
    validate_spins(np.array([[-1. + 1e-15, 1. - 1e-15]]))
    for value in (np.ones((2, 2), dtype=bool), np.empty((0, 2, 2)),
                  np.full((2, 2), np.nan), np.ones((2, 2), dtype=complex),
                  torch.ones(2, 2, requires_grad=True)):
        with pytest.raises(ValueError):
            validate_spins(value)


def test_three_motif_order_matches_existing_Q_binary_lookup_exactly():
    x = torch.from_numpy(spins((4, 6, 9, 12))).double()
    expected, _ = binary_frequencies(square_geometry(3, 3).extract_patches(x[..., None]))
    actual = motif_probabilities(x, 3)
    np.testing.assert_array_equal(actual, expected.numpy())
    np.testing.assert_array_equal(actual.sum(-1), np.ones(4))
    assert Q_MOTIF_ORDERING != TARGET_MOTIF_ORDERING


def test_two_motif_order_matches_existing_affine_decoder_plaquette_exactly():
    x = torch.from_numpy(spins((4, 5, 9, 12))).double()
    expected = plaquette_distribution(x[..., None])
    np.testing.assert_array_equal(motif_probabilities(x, 2), expected.numpy())
    # Distinct probabilities ensure a permutation bug is actually detectable.
    assert len(np.unique(expected.numpy()[0])) > 4


def test_motif_prefix_sample_axis_and_periodic_translation():
    x = spins((2, 3, 5, 9, 12))
    actual = motif_probabilities(x)
    assert actual.shape == (2, 3, 512)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(actual[i, j], motif_probabilities(x[i, j]))
    shifted = np.roll(x, (2, -4), axis=(-2, -1))
    np.testing.assert_array_equal(actual, motif_probabilities(shifted))
    moved = np.moveaxis(x, 2, 0)
    np.testing.assert_array_equal(actual, motif_probabilities(moved, sample_axis=0))
    for kwargs in (dict(width=4), dict(sample_axis=-1), dict(sample_axis=7)):
        with pytest.raises(ValueError):
            motif_probabilities(x, **kwargs)


def test_pca_is_shared_joint_train_fit_with_stable_sign_and_roundtrip_receipt():
    p = probabilities()
    fitted = fit_train_pca(p)
    flat = p.reshape(-1, 512)
    np.testing.assert_allclose(fitted.center, flat.mean(0), atol=0., rtol=0.)
    np.testing.assert_allclose(fitted.basis.T @ fitted.basis, np.eye(5), atol=1e-12)
    for column in fitted.basis.T:
        assert column[np.argmax(abs(column))] >= 0
    encoded = fitted.transform(p)
    assert encoded.shape == (10, 3, 5)
    np.testing.assert_allclose(encoded.mean((0, 1)), 0., atol=1e-17)
    again = fit_train_pca(p)
    np.testing.assert_array_equal(fitted.basis, again.basis)
    receipt = json.loads(json.dumps(fitted.to_dict(), allow_nan=False))
    restored = PCAReadout.from_dict(receipt)
    np.testing.assert_array_equal(restored.transform(p), encoded)
    np.testing.assert_allclose(restored.inverse_transform(encoded), fitted.inverse_transform(encoded))
    assert fitted.train_shape == p.shape and fitted.sample_count == 30
    assert not fitted.center.flags.writeable and not fitted.basis.flags.writeable


def test_pca_never_refits_on_test_and_reports_rank_deficiency_without_forcing_rank():
    base = np.ones((8, 3, 512)) / 512
    amplitude = np.linspace(-.001, .001, 24).reshape(8, 3)
    base[..., 0] += amplitude
    base[..., 1] -= amplitude
    fitted = fit_train_pca(base)
    assert fitted.rank == 1 and fitted.dimension == 5
    assert any("rank" in warning for warning in fitted.warnings)
    assert any("weak" in warning for warning in fitted.warnings)
    snapshot = fitted.to_dict()
    fitted.transform(probabilities((2, 3, 512)))
    assert snapshot == fitted.to_dict()
    with pytest.raises(ValueError, match="TRAIN-only"):
        fit_train_pca(base, split="test")
    for wrong in (base[..., :16], np.ones_like(base), base[:1, :1]):
        with pytest.raises(ValueError):
            fit_train_pca(wrong)


def test_constant_pca_warns_but_does_not_invent_data_rank():
    fitted = fit_train_pca(np.ones((6, 512)) / 512)
    assert fitted.rank == 0
    assert fitted.to_dict()["explained_variance_ratio"] == [0.] * 6
    np.testing.assert_allclose(fitted.transform(np.ones((6, 512)) / 512), 0.)


def affine_example():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(10, 3, 5))
    weight = rng.normal(scale=1e-3, size=(5, 16))
    weight -= weight.mean(axis=1, keepdims=True)
    y = x @ weight + 1 / 16
    return x, y, weight


def test_affine_exact_fit_shared_scales_and_raw_unclipped_extrapolation():
    x, y, weight = affine_example()
    fitted = fit_affine_readout(x, y)
    np.testing.assert_allclose(fitted.weight, weight, atol=1e-16)
    np.testing.assert_allclose(fitted.predict(x), y, atol=1e-16)
    extrapolated = fitted.predict(np.full((1, 5), 1e5))
    assert (extrapolated < 0).any() and (extrapolated > 1).any()
    restored = AffineReadout.from_dict(json.loads(json.dumps(fitted.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.predict(x), fitted.predict(x))
    assert fitted.rank == 5 and fitted.sample_count == 30


def test_affine_minimum_norm_rank_deficiency_and_ridge_closed_form():
    x, y, _ = affine_example()
    x[..., 4] = 2 * x[..., 0]
    fitted = fit_affine_readout(x, y)
    assert fitted.rank == 4
    flat_x, flat_y = x.reshape(-1, 5), y.reshape(-1, 16)
    xc, yc = flat_x - flat_x.mean(0), flat_y - flat_y.mean(0)
    expected = np.linalg.pinv(xc, rcond=1e-12) @ yc
    np.testing.assert_allclose(fitted.weight, expected, atol=1e-16)
    np.testing.assert_allclose(fitted.weight[4], 2 * fitted.weight[0], atol=1e-16)
    regularized = fit_affine_readout(x, y, ridge=2.)
    expected_ridge = np.linalg.solve(xc.T @ xc + 2 * np.eye(5), xc.T @ yc)
    np.testing.assert_allclose(regularized.weight, expected_ridge, atol=1e-16)
    assert np.linalg.norm(regularized.weight) < np.linalg.norm(fitted.weight)


def test_affine_degenerate_intercept_only_and_invalid_fit_contracts():
    x = np.ones((3, 5))
    y = probabilities((3, 16))
    fitted = fit_affine_readout(x, y)
    assert fitted.rank == 0
    np.testing.assert_allclose(fitted.predict(x), np.broadcast_to(y.mean(0), y.shape))
    for kwargs in (dict(split="test"), dict(ridge=-1), dict(rcond=0), dict(ridge=True)):
        with pytest.raises(ValueError):
            fit_affine_readout(x, y, **kwargs)
    with pytest.raises(ValueError):
        fit_affine_readout(x, y[..., :4])
    with pytest.raises(ValueError):
        fit_affine_readout(x, y * 2)


def test_residual_flow_matches_current_R_architecture_and_preserves_seed_stream():
    state = torch.random.get_rng_state().clone()
    flow = build_r(seed=91)
    torch.testing.assert_close(state, torch.random.get_rng_state())
    other = build_r(seed=91)
    for a, b in zip(flow.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)
        assert a.dtype == torch.float64 and a.device.type == "cpu"
    original = RecurrentRGNetwork2D(levels=3, hidden=96, theory_dim=5, width=3, block_factor=3).double()
    assert [type(layer) for layer in flow.theory_rg] == [type(layer) for layer in original.theory_rg]
    assert [p.shape for p in flow.theory_rg.parameters()] == [p.shape for p in original.theory_rg.parameters()]
    original.theory_rg.load_state_dict(flow.theory_rg.state_dict())
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    torch.testing.assert_close(flow(x), original.advance_theory(x))
    torch.testing.assert_close(flow.advance_theory(x), flow(x))


def test_path_metrics_first_average_C_squared_error_not_error_of_C_mean():
    direct = np.zeros((2, 2, 3, 1))
    direct[:, :, 1, 0] = 1.
    direct[:, :, 2, 0] = 2.
    one = direct[:, :, 1:].copy()
    one[:, 0] += 1.
    one[:, 1] -= 1.
    rollout = direct.copy()
    rollout[:, 0, 1:] += 1.
    rollout[:, 1, 1:] -= 1.
    train = np.array([[[-1.], [-1.], [0.]], [[1.], [3.], [4.]]])
    result = path_metrics(direct, one, rollout, train)
    assert [(p["start"], p["end"]) for p in result["paths"]] == [(0, 1), (0, 2), (1, 2)]
    assert result["independent_sample_count"] == 2 and result["C_draw_count"] == 2
    for row in result["paths"]:
        assert row["R"]["rmse"] == 1.
        assert row["R"]["nrmse"] == .5
    assert result["paths"][0]["identity"]["rmse"] == 1.
    assert result["paths"][1]["identity"]["rmse"] == 2.
    moved = path_metrics(np.moveaxis(direct, 1, 2), np.moveaxis(one, 1, 2),
                         np.moveaxis(rollout, 1, 2), train, c_axis=2)
    assert moved == result


def test_path_metrics_zero_variance_single_draw_and_wrong_reference_rejected():
    direct = np.arange(12, dtype=float).reshape(2, 3, 2)
    one = direct[:, 1:]
    result = path_metrics(direct, one, direct, np.ones_like(direct), c_axis=None)
    assert result["C_draw_count"] == 1
    assert result["paths"][0]["R"]["nrmse"] is None
    with pytest.raises(ValueError):
        path_metrics(direct[:, None], one[:, None], direct[:, None], np.ones((2, 1, 3, 2)))
    bad_rollout = direct[:, None].copy()
    bad_rollout[:, :, 0] += 1
    with pytest.raises(ValueError, match="start"):
        path_metrics(direct[:, None], one[:, None], bad_rollout, direct)


def test_physical_error_decomposition_keeps_cross_term_and_chain_units():
    observed = np.zeros((3, 2, 4, 16))
    decoded = np.ones_like(observed)
    predicted = 3 * np.ones_like(observed)
    result = physical_error_decomposition(observed, decoded, predicted)
    assert result["total_mse"] == 9.
    assert result["readout_mse"] == 1.
    assert result["R_transport_mse"] == 4.
    assert result["cross_term"] == 4.
    assert result["independent_sample_count"] == 3
    assert result["C_averaged_before_error"] is False
    # Conditional means can cancel C noise, but are a different labeled metric.
    decoded[:, 1] = -1
    predicted[:, 1] = -3
    per_draw = physical_error_decomposition(observed, decoded, predicted)
    means = physical_error_decomposition(observed, decoded, predicted, average_C_before_error=True)
    assert per_draw["total_mse"] == 9. and means["total_mse"] == 0.
    assert means["C_averaged_before_error"] is True
    assert result["clipped"] is False and result["renormalized"] is False


def test_receipts_reject_misdeclared_TEST_and_postprocessed_outputs():
    pca = fit_train_pca(probabilities()).to_dict()
    pca["fit_split"] = "test"
    with pytest.raises(ValueError):
        PCAReadout.from_dict(pca)
    x, y, _ = affine_example()
    affine = fit_affine_readout(x, y).to_dict()
    affine["clipped"] = True
    with pytest.raises(ValueError):
        AffineReadout.from_dict(affine)
