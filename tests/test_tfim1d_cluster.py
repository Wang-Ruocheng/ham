import numpy as np
import pytest
from recurrent_rg.tfim1d_cluster import sample_tfim1d_chain


def test_reproducibility_and_values():
    args = dict(h=1., sites=8, slices=8, draws=20, equil=20, gap=2)
    a = sample_tfim1d_chain(**args)
    np.testing.assert_array_equal(a, sample_tfim1d_chain(**args))
    assert a.shape == (20, 8)
    assert set(np.unique(a)) <= {-1, 1}
    with pytest.raises(ValueError):
        sample_tfim1d_chain(-1.)


def test_exact_small_trotter_measure():
    # Enumerate all 2^16 spacetime states, not the continuum quantum measure.
    x = (2*((np.arange(65536)[:, None] >> np.arange(16)) & 1)-1).reshape(-1,4,4)
    dtau, h = .3, 1.2
    energy = dtau*(x*np.roll(x,1,axis=2)).sum((1,2))
    energy += -.5*np.log(np.tanh(dtau*h))*(x*np.roll(x,1,axis=1)).sum((1,2))
    p = np.exp(energy-energy.max()); p /= p.sum()
    expected = [p @ (x*np.roll(x,1,axis=2)).mean((1,2)),
                p @ (x.mean(2)**2).mean(1)]
    z = sample_tfim1d_chain(h,4,4,dtau,draws=30000,equil=512,gap=4,seed=193)
    observed = [(z*np.roll(z,1,axis=1)).mean(), (z.mean(1)**2).mean()]
    np.testing.assert_allclose(observed, expected, atol=.015, rtol=0)
