import itertools
import numpy as np
import pytest
from recurrent_rg.data.blume_capel import sample_blume_capel,conditional_probabilities,observable_series


def test_conditional_distribution_and_validation():
    p=conditional_probabilities(np.arange(-4,5),1.4,1.)
    np.testing.assert_allclose(p.sum(-1),1)
    np.testing.assert_allclose(p,p[::-1,::-1])
    np.testing.assert_allclose(conditional_probabilities(0,1.,1.),np.array([np.exp(-1),1,np.exp(-1)])/(1+2*np.exp(-1)))
    for kwargs in [dict(temperature=0),dict(exchange=-1),dict(start='bad')]:
        a=dict(temperature=1.,length=4,samples=4,seed=91);a.update(kwargs)
        with pytest.raises(ValueError):sample_blume_capel(**a)


def test_reproducible_sampler_and_independent_limit():
    x=sample_blume_capel(1.,8,1000,91,exchange=0.,burn=50,spacing=2)
    np.testing.assert_array_equal(x,sample_blume_capel(1.,8,1000,91,exchange=0.,burn=50,spacing=2))
    assert set(np.unique(x))=={-1,0,1}
    assert abs((x*x).mean()-2*np.exp(-1)/(1+2*np.exp(-1)))<.012
    assert abs(x.mean())<.015


def test_sampler_against_exact_periodic_two_by_two():
    states=np.array(list(itertools.product([-1,0,1],repeat=4))).reshape(-1,2,2)
    obs=observable_series(states,delta=1.)
    t=1.4;p=np.exp(-4*obs['energy']/t);p/=p.sum()
    x=sample_blume_capel(t,2,30000,912,delta=1.,burn=500,spacing=2)
    estimated=observable_series(x,delta=1.)
    for key in ['energy','occupancy','magnetization_squared']:
        assert abs(estimated[key].mean()-np.dot(p,obs[key]))<.018
