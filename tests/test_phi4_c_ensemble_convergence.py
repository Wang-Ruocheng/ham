"""Analytic/toy checks only: no model, checkpoint, or physical fields loaded."""
import itertools

import numpy as np
import pytest

from scripts.diagnose_phi4_c_ensemble_convergence import (
    aggregate, block_mean, cross_chain_distances, field_metrics, select_saved_fields,
)


def test_saved_selection_preserves_chain_and_sample_axes():
    x = np.arange(4 * 128 * 36 * 36, dtype=float).reshape(4, 128, 36, 36, 1)
    indices = tuple(range(0, 128, 8))
    y = select_saved_fields(x, indices)
    assert y.shape == (4, 16, 36, 36)
    for chain, saved_position in itertools.product(range(4), range(16)):
        np.testing.assert_array_equal(y[chain, saved_position], x[chain, indices[saved_position], :, :, 0])


def test_block_mean_batch_and_spatial_mean():
    x = np.arange(4 * 2 * 36 * 36, dtype=float).reshape(4, 2, 36, 36)
    y = block_mean(x)
    expected = np.stack([np.stack([
        np.array([[row[3*i:3*i+3, 3*j:3*j+3].mean() for j in range(12)] for i in range(12)])
        for row in chain]) for chain in x])
    np.testing.assert_allclose(y, expected)
    np.testing.assert_allclose(block_mean(y).mean(axis=(-2, -1)), x.mean(axis=(-2, -1)))


def test_z2_sectors_differ_raw_but_match_after_fixed_sign_alignment():
    signs = np.array([[1., -1.], [-1., 1.], [1., -1.], [-1., 1.]])
    x = np.broadcast_to(signs[..., None, None], (4, 2, 3, 3)).copy()
    raw = cross_chain_distances(x, 1.)
    aligned = cross_chain_distances(x, 1., signs)
    assert raw["relative_rms"] == pytest.approx(np.sqrt(2.))
    assert aligned["absolute_rms_over_s"] == pytest.approx(0.)
    metrics = field_metrics(x, 1., signs)
    np.testing.assert_allclose(metrics["q"], 1.)
    np.testing.assert_allclose(metrics["majority_fraction"], 1.)
    np.testing.assert_allclose(metrics["relative_spatial_fluctuation"], 0.)


def test_relative_metrics_invariant_to_common_positive_gain():
    x = np.random.default_rng(91).normal(size=(4, 5, 6, 6)) + .8
    signs = np.sign(x.mean(axis=(-2, -1)))
    first = field_metrics(x, .9, signs)
    second = field_metrics(3 * x, .9, signs)
    for key in ("q", "majority_fraction", "relative_spatial_fluctuation"):
        np.testing.assert_allclose(first[key], second[key])
    d1 = cross_chain_distances(x, .9, signs)
    d2 = cross_chain_distances(3 * x, .9, signs)
    assert d2["relative_rms"] == pytest.approx(d1["relative_rms"])
    assert d2["absolute_rms_over_s"] == pytest.approx(3 * d1["absolute_rms_over_s"])


def test_cross_chain_only_matches_brute_force_per_site_and_decomposition():
    x = np.random.default_rng(92).normal(size=(4, 3, 3, 3))
    signs = np.sign(x.mean(axis=(-2, -1)))
    aligned = x * signs[..., None, None]
    squared = []
    for a, b in itertools.combinations(range(4), 2):
        for i, j in itertools.product(range(3), repeat=2):
            squared.append(np.mean((aligned[a, i] - aligned[b, j]) ** 2) / .9**2)
    result = cross_chain_distances(x, .9, signs)
    assert len(result["chain_pairs"]) == 6
    assert result["mse_over_s2"] == pytest.approx(np.mean(squared))
    assert result["mse_over_s2"] == pytest.approx(result["amplitude_mse_over_s2"] + result["texture_mse_over_s2"])


def test_sign_retention_uses_original_sign_without_reorienting_outputs():
    x = np.ones((4, 2, 3, 3))
    original = np.ones((4, 2))
    np.testing.assert_allclose(field_metrics(-x, 1., original)["original_sign_retention"], 0.)


def test_draws_averaged_as_metrics_not_as_new_physical_chains():
    x = np.random.default_rng(93).normal(size=(4, 3, 3, 3)) + .3
    y = x + np.random.default_rng(94).normal(size=x.shape) * .2
    signs = np.sign(x.mean(axis=(-2, -1)))
    result = aggregate([x, y], 1., signs)
    expected = (cross_chain_distances(x, 1., signs)["relative_rms"] + cross_chain_distances(y, 1., signs)["relative_rms"]) / 2
    assert result["aligned_relative_rms_distance"]["mean"] == pytest.approx(expected)
    assert len(result["q"]["chain_means"]) == 4
    assert len(result["aligned_relative_rms_distance"]["chain_pair_means"]) == 6


def test_invalid_geometry_and_zero_rms_are_rejected():
    with pytest.raises(ValueError):
        block_mean(np.zeros((4, 2, 10, 10)))
    with pytest.raises(ValueError):
        field_metrics(np.zeros((4, 2, 3, 3)), 1., np.ones((4, 2)))
