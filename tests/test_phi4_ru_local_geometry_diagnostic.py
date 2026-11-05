import numpy as np

from scripts.phi4_ru_local_geometry_diagnostic import (
    bootstrap_chains, chart_statistics, local_chart, stencil,
)


def test_normalized_chart_of_affine_map_has_correct_axes_and_units():
    r, u = np.linspace(-1.5, -.5, 9), np.linspace(.5, 1., 5)
    rn, un = np.meshgrid((r-r[0])/np.ptp(r), (u-u[0])/np.ptp(u), indexing="ij")
    g = np.stack((2*rn+3*un, -rn+4*un), axis=-1)
    g = np.repeat(g[:, :, None, :], 3, axis=2)
    for stride in (1, 2):
        jac, interior, _ = local_chart(g, r, u, stride=stride)
        np.testing.assert_allclose(jac, np.broadcast_to([[2, 3], [-1, 4]], jac.shape))
        assert interior.sum() == (21 if stride == 1 else 5)


def test_rank_one_regular_chart_has_vanishing_second_singular_value():
    r, u = np.linspace(-1.5, -.5, 9), np.linspace(.5, 1., 5)
    rn, un = np.meshgrid((r-r[0])/np.ptp(r), (u-u[0])/np.ptp(u), indexing="ij")
    t = rn+un
    g = np.stack((t, 2*t), axis=-1)[:, :, None, :]
    chart, _, _ = local_chart(g, r, u)
    _, _, _, rank2 = chart_statistics(chart, np.repeat(chart[None], 20, axis=0))
    assert not rank2.any()


def test_curved_one_dimensional_image_can_have_spurious_finite_grid_rank2():
    # This control documents why finite-grid rank2 is not intrinsic dimension.
    r, u = np.linspace(0., 1., 9), np.linspace(0., 1., 5)
    rn, un = np.meshgrid(r, u, indexing="ij")
    t = rn+un
    g = np.stack((t, t**3), axis=-1)[:, :, None, :]
    chart, _, _ = local_chart(g, r, u)
    sigma, _, _, _ = chart_statistics(chart, np.repeat(chart[None], 20, axis=0))
    assert sigma[..., 1].max() > .01


def test_chain_bootstrap_preserves_scale_pairing_and_explicit_indices():
    g = np.arange(2*3*4*3*2.).reshape(2, 3, 4, 3, 2)
    boot, index = bootstrap_chains(g, count=20)
    assert boot.shape == (20, 2, 3, 3, 2)
    np.testing.assert_allclose(boot[8, 1, 2], g[1, 2, index[8, 1, 2]].mean(axis=0))
    np.testing.assert_allclose(np.diff(boot, axis=-2), 2.)


def test_noise_is_operatornorm_and_resolved_rule_has_factor_two():
    chart = np.diag([3., 1.])[None, None, None]
    boot = np.repeat(chart[None], 20, axis=0)
    boot[..., 0, 0] += .6
    sigma, noise, ratio, flag = chart_statistics(chart, boot)
    np.testing.assert_allclose(sigma, [[[[3., 1.]]]])
    np.testing.assert_allclose(noise, .6)
    np.testing.assert_allclose(ratio, 1/3)
    assert not flag.any()


def test_boundary_stencil_never_wraps_to_other_side():
    assert stencil(0, 5, 1) == (0, 1, False)
    assert stencil(4, 5, 1) == (3, 4, False)
    assert stencil(2, 5, 2) == (0, 4, True)
    assert stencil(1, 5, 2) == (1, 3, False)
