import numpy as np
import pytest

from scripts.plot_phi4_ru_scan_theory import beta_ru, truncated_fixed_points
from scripts.plot_phi4_theoretical_rg import beta


def test_conversion_to_rescaled_dimensionless_equations():
    cutoff = np.pi
    points = np.array([[-1.55, .5], [-.5, 1.], [0., 0.], [-2., 5.]])
    conversion = np.array([1/cutoff**2, 3/(2*np.pi*cutoff**2)])
    np.testing.assert_allclose(beta_ru(points)*conversion,
                               beta(points*conversion, epsilon=2), atol=1e-14)


def test_analytic_fixed_points_are_zeros_outside_actual_scan():
    roots = truncated_fixed_points()
    np.testing.assert_allclose(beta_ru(roots), 0, atol=1e-13)
    assert not np.any(np.all((roots >= [-1.55, .5]) & (roots <= [-.5, 1.]), axis=1))


def test_actual_scan_has_finite_nonzero_IR_vectors():
    r, u = np.meshgrid(np.linspace(-1.55, -.5, 101), np.linspace(.5, 1., 101))
    field = beta_ru(np.stack([r, u], axis=-1))
    assert np.isfinite(field).all()
    assert (field[..., 0] < 0).all()
    assert (field[..., 1] > 0).all()


@pytest.mark.parametrize("state,cutoff", [([-2., .5], 1.), ([0., .5], 0.),
                                           ([0., .5], -1.), ([np.nan, .5], np.pi)])
def test_invalid_symmetric_expansion_rejected(state, cutoff):
    with pytest.raises(ValueError):
        beta_ru(np.array(state), cutoff)
