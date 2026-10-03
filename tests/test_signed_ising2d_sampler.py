import numpy as np
import pytest
from recurrent_rg.ising2d_field import sample_signed_field_chain, sample_field_chain, field_observables
from scripts.generate_symmetric_ising2d import symmetric_grid, held_out


@pytest.mark.parametrize('k,b',[(-.3,0.),(-.3,.4),(-.6,-.5)])
def test_signed_sampler_exact_4x4(k,b):
    n=16
    states=(2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1).reshape(-1,4,4)
    obs=field_observables(states)
    eta=1-2*(np.indices((4,4)).sum(0)%2)
    obs=np.c_[obs,(states*eta).mean((-2,-1))**2]
    logw=n*(k*obs[:,0]+b*obs[:,1]);w=np.exp(logw-logw.max());exact=w@obs/w.sum()
    x=sample_signed_field_chain(k,b,4,16000,512,4,91)
    measured=np.r_[field_observables(x).mean(0),((x*eta).mean((-2,-1))**2).mean()]
    np.testing.assert_allclose(measured,exact,atol=.035)


def test_positive_branch_and_invalid_odd_lattice():
    np.testing.assert_array_equal(sample_signed_field_chain(.3,.1,4,20,20,2,91),sample_field_chain(.3,.1,4,20,20,2,91))
    with pytest.raises(ValueError):sample_signed_field_chain(-.3,0.,3,10,10,2,91)


def test_symmetric_grid_and_group_split():
    ks,bs=symmetric_grid()
    np.testing.assert_allclose(ks,-ks[::-1]);np.testing.assert_allclose(bs,-bs[::-1])
    assert ks[23]==0 and bs[9]==0
    assert all(np.isclose(ks,k).any() for k in np.linspace(.2,.68,17))
    mask=np.array([[held_out(i,j) for j in range(len(bs))] for i in range(len(ks))])
    np.testing.assert_array_equal(mask,mask[::-1]);np.testing.assert_array_equal(mask,mask[:,::-1])
