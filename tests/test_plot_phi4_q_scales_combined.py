import numpy as np
import pytest

from scripts.plot_phi4_saved_q_scales_combined import visual_interpolation


def test_visual_grid_keeps_all_measured_anchors():
    x, y = visual_interpolation([.60, .65, .62])
    assert len(x) == 201
    assert np.allclose(x[[0, 100, 200]], [0, 1, 2])
    assert np.allclose(y[[0, 100, 200]], [.60, .65, .62])
    assert np.all(y >= .60) and np.all(y <= .65)
    assert x.min() == 0 and x.max() == 2


@pytest.mark.parametrize("values", [[.6, .7], [.6, np.nan, .7], [.6, .7, np.inf]])
def test_invalid_measured_values_rejected(values):
    with pytest.raises(ValueError, match="three measured"):
        visual_interpolation(values)


@pytest.mark.parametrize("points", [2, 200, 5.5])
def test_grid_must_include_integer_anchors(points):
    with pytest.raises(ValueError, match="odd grid"):
        visual_interpolation([.6, .7, .8], points)
