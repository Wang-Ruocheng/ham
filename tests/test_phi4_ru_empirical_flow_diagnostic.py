import numpy as np

from scripts.phi4_ru_empirical_flow_diagnostic import (
    bootstrap_cell_means, chain_and_cell_means, paired_neighborhoods, vector_metrics,
)


def test_repeats_are_averaged_before_physical_chains():
    a = np.array([[[0.], [2.], [10.], [14.]],
                  [[2.], [4.], [12.], [16.]]])
    chain, grouped, means, ids = chain_and_cell_means(a, [3, 3, 9, 9])
    np.testing.assert_allclose(chain[:, 0], [1, 3, 11, 15])
    assert grouped.shape == (2, 2, 1)
    np.testing.assert_allclose(means[:, 0], [2, 13])
    np.testing.assert_array_equal(ids, [3, 9])


def test_bootstrap_keeps_successive_scales_paired():
    grouped = np.arange(24.).reshape(2, 4, 3, 1)
    result, index = bootstrap_cell_means(grouped, count=50)
    assert result.shape == (50, 2, 3, 1)
    np.testing.assert_allclose(np.diff(result, axis=2), 1.)
    np.testing.assert_allclose(result[7, 1], grouped[1, index[7, 1]].mean(0))


def test_exact_drift_has_zero_error_and_no_rotation():
    delta = np.array([[1., 2.], [-2., 1.]])
    metrics, angles = vector_metrics(delta, delta)
    assert metrics['rmse'] == 0.
    assert metrics['relative_displacement_error'] == 0.
    np.testing.assert_allclose(angles, 0., atol=2e-6)
    assert metrics['opposite_fraction'] == 0.


def test_reverse_drift_has_180_degree_rotation():
    delta = np.array([[1., 0.], [0., 1.]])
    metrics, angles = vector_metrics(delta, -delta)
    np.testing.assert_allclose(angles, 180.)
    assert metrics['relative_displacement_error'] == 2.
    assert metrics['opposite_fraction'] == 1.


def test_steep_single_valued_map_is_not_mismatch():
    # Very nearby inputs can have distant successors under a legitimate map.
    g = np.array([[[1., 0.], [100., 0.], [10000., 0.]],
                  [[1.001, 0.], [100.1, 0.], [10010., 0.]],
                  [[1.002, 0.], [100.2, 0.], [10020., 0.]]])
    boot = np.broadcast_to(g, (20, *g.shape)).copy()
    prediction = g[:, 1:]
    bp = np.broadcast_to(prediction, (20, *prediction.shape)).copy()
    rows = paired_neighborhoods(g, boot, prediction, bp, np.arange(3), ['train'] * 3,
                               np.ones(2), selection_g=g)
    assert max(r['required_gain_lower_descriptive'] for r in rows) > 99.
    assert all(r['successor_contrast_error_norm'] == 0. for r in rows)
    assert not any(r['contrast_error_resolved_descriptive'] for r in rows)
