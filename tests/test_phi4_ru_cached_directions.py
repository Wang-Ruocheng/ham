"""Plot geometry tests; no models or physical configurations are read."""
import numpy as np
import pytest

from scripts.plot_phi4_ru_cached_directions import directions, geometry


def test_equal_arrows_preserve_direction_and_skip_numerical_zero():
    raw = np.array([[3., 4.], [-.003, .004], [0., 0.], [1e-15, 0.]])
    scaled, keep = directions(raw, .2)
    np.testing.assert_array_equal(keep, [True, True, False, False])
    np.testing.assert_allclose(np.linalg.norm(scaled[keep], axis=1), .2)
    np.testing.assert_allclose(scaled[0], [.12, .16])
    np.testing.assert_allclose(scaled[1], [-.12, .16])
    np.testing.assert_array_equal(scaled[~keep], 0)


def example():
    rng = np.random.default_rng(3)
    direct = rng.normal(size=(4, 24, 3, 2))
    labels = np.array([[36 if i < 15 else 108, float(i), .5] for i in range(24)])
    predictions = {seed: direct[:, :, :2] + [seed / 100., -.2] for seed in (91, 92, 93)}
    groups = [dict(size=int(labels[i, 0]), r=float(i), u=.5, physical_chains=1,
                   g_mean=direct[:, i].mean(0).tolist(), models={
                       f"selected_seed{seed}": dict(mean_R_minus_g=(predictions[seed][:, i] - direct[:, i, :2]).mean(0).tolist())
                       for seed in (91, 92, 93)}) for i in range(24)]
    return direct, labels, predictions, groups


def test_geometry_uses_cached_mean_delta_without_interpolation_or_final_input():
    direct, labels, predictions, groups = example()
    clouds, origins, raw, identities = geometry(direct, labels, predictions, groups)
    assert len(clouds) == 48 and origins.shape == (48, 2)
    assert {x["input_scale"] for x in identities} == {0, 1}
    assert all(value.shape == (4, 2) for value in clouds)
    np.testing.assert_allclose(raw[91], np.tile([.91, -.2], (48, 1)))
    np.testing.assert_allclose(origins[0], direct[:, 0, 0].mean(0))
    groups[0]["models"]["selected_seed91"]["mean_R_minus_g"][0][0] += .1
    with pytest.raises(ValueError, match="位移"):
        geometry(direct, labels, predictions, groups)
