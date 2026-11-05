import numpy as np

from scripts.phi4_ru_physics_phase_diagnostic import physical_metrics


def test_binder_uses_pooled_moments_not_mean_of_chain_ratios():
    moments = np.array([[0., 1., 1., 1.], [0., 2., 4., 16.]])
    pooled = physical_metrics(moments.mean(axis=0))
    expected = 1. - 8.5 / (3. * 2.5 ** 2)
    np.testing.assert_allclose(pooled[1], expected)
    assert not np.isclose(pooled[1], physical_metrics(moments)[:, 1].mean())


def test_connected_and_absolute_subtraction_are_distinct():
    moments = np.array([0., 1., 1., 1.])
    absolute_m, binder, connected, absolute_subtracted = physical_metrics(moments)
    assert absolute_m == 1.
    np.testing.assert_allclose(binder, 2. / 3.)
    assert connected == 36. ** 2
    assert absolute_subtracted == 0.
