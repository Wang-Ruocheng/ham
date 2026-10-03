import numpy as np
import pytest
from scripts.generate_symmetric_ising2d import uniform_field_grid, held_out


def test_uniform_field_grid_and_split():
    ks, bs = uniform_field_grid()
    assert len(ks) == 17 and len(bs) == 21
    np.testing.assert_allclose(np.diff(ks), .03)
    np.testing.assert_allclose(np.diff(bs), .1)
    np.testing.assert_allclose(bs, -bs[::-1])
    assert ks[0] == .2 and ks[-1] == .68
    assert bs[0] == -1 and bs[10] == 0 and bs[-1] == 1
    for i in range(len(ks)):
        for j in range(len(bs)):
            assert held_out(i,j,0,10) == held_out(i,20-j,0,10)


@pytest.mark.parametrize('limit',[.1,.01])
def test_narrow_uniform_field(limit):
    ks,bs = uniform_field_grid(limit)
    np.testing.assert_allclose(np.diff(bs),limit/10)
    np.testing.assert_array_equal(ks,uniform_field_grid()[0])
    assert bs[0]==-limit and bs[-1]==limit and bs[10]==0 and len(bs)==21


@pytest.mark.parametrize('limit',[0,-1,float('nan'),float('inf')])
def test_invalid_field_range(limit):
    with pytest.raises(ValueError):
        uniform_field_grid(limit)
