import numpy as np
import pytest
import torch

from recurrent_rg.phi4_r_calibration import FrozenResidualMap
from scripts import plot_phi4_ru_full_grid_directions as plot


def test_uniform_grid_covers_frame_and_leaves_room_for_equal_arrows():
    bounds = np.array([[-.08, .01], [-.03, .09]])
    grid, length = plot.make_grid(bounds, 19, 29)
    assert grid.shape == (19 * 29, 2)
    for axis, count in enumerate((19, 29)):
        unique = np.unique(grid[:, axis])
        np.testing.assert_allclose(np.diff(unique), np.ptp(bounds[axis]) / count)
        assert unique[0] - length / 2 > bounds[axis, 0]
        assert unique[-1] + length / 2 < bounds[axis, 1]


def test_coverage_does_not_fill_gaps_between_condition_clouds():
    box = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    mask = plot.coverage_mask(np.array([[.5, .5], [5., .5], [10.5, .5]]), [box, box + [10., 0.]])
    np.testing.assert_array_equal(mask, [True, False, True])


def test_frozen_map_restore_forward_and_fingerprint_reject_tampering(tmp_path):
    torch.manual_seed(7)
    model = FrozenResidualMap(2, 4, center=[.3, -.2], scale=[.04, .2]).eval().requires_grad_(False)
    expected = plot.fingerprint(model)
    key = "selected_seed91"
    state = {f"{key}.{k}": v.detach().numpy().copy() for k, v in model.state_dict().items()}
    path = tmp_path / "R_parameters.npz"
    np.savez(path, **state)
    restored = plot.restore_model(path, key, expected)
    points = np.array([[.1, .4], [-.2, .7]])
    np.testing.assert_allclose(plot.mapped(restored, points), plot.mapped(model, points), rtol=0, atol=0)
    assert plot.fingerprint(restored) == expected
    assert all(not p.requires_grad and p.grad is None for p in restored.parameters())
    state[f"{key}.net.0.bias"][0] += .1
    np.savez(path, **state)
    with pytest.raises(ValueError, match="指纹"):
        plot.restore_model(path, key, expected)
