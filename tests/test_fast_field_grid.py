import numpy as np
from experiments.fast_qcr_ising2d_field import focused_grid


def test_focused_grid_and_holdout():
    ks, bs = focused_grid()
    assert len(ks)==17 and len(bs)==19
    np.testing.assert_allclose(bs, -bs[::-1])
    assert bs[9]==0
    assert bs[10]==.0005
    assert np.isclose(ks[8], .44)
    split = np.array([[(i+abs(j-9))%3==1 for j in range(len(bs))] for i in range(len(ks))])
    np.testing.assert_array_equal(split, split[:,::-1])
    assert split.any() and not split.all()
