import numpy as np
import pytest

from scripts.prepare_ising2d_field_qcr import parameter_grid


@pytest.mark.parametrize('step,count',[(.05,21),(.1,11)])
def test_uniform_grid_and_symmetric_cell_split(step,count):
    ks,bs=parameter_grid(step)
    assert len(ks)==33 and len(bs)==count
    np.testing.assert_allclose(np.diff(ks),.015,atol=1e-14)
    np.testing.assert_allclose(np.diff(bs),step,atol=1e-14)
    np.testing.assert_allclose([ks[0],ks[-1],bs[0],bs[-1]],[.2,.68,-.5,.5])
    np.testing.assert_allclose(bs,-bs[::-1],atol=1e-14)
    assert bs[count//2]==0
    split=np.array([[(i+abs(j-count//2))%3==1 for j in range(count)] for i in range(33)])
    np.testing.assert_array_equal(split,split[:,::-1])
    assert split.sum()==len(ks)*len(bs)//3


def test_reject_nonuniform_option():
    with pytest.raises(ValueError):
        parameter_grid(.03)
