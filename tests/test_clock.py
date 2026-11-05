from itertools import product
import numpy as np
import pytest
from recurrent_rg.data.clock import sample_clock, observables


def test_limits_reproducibility_and_rotation():
    a=sample_clock(0., 8, 1000, 121, burn=100, spacing=4)
    assert np.array_equal(a,sample_clock(0.,8,1000,121,burn=100,spacing=4))
    assert abs(observables(a)['bond'].mean())<.015
    assert np.max(abs(np.bincount(a.ravel(),minlength=6)/a.size-1/6))<.01
    b=sample_clock(8.,8,100,122,burn=256,spacing=8)
    assert observables(b)['bond'].mean()>.98
    for key, values in observables(a).items():
        np.testing.assert_allclose(values,observables((a+2)%6)[key],atol=1e-10)


@pytest.mark.parametrize('K',[.4,1.2,1.8])
def test_sampler_matches_exact_periodic_2x2(K):
    states=np.array(list(product(range(6),repeat=4))).reshape(-1,2,2)
    obs=observables(states)
    # L2 periodic right/down bonds include parallel edges, as in sampler.
    logw=-4*K*obs['energy'];w=np.exp(logw-logw.max());w/=w.sum()
    draws=observables(sample_clock(K,2,16000,740,burn=512,spacing=4))
    for name in ('bond','magnetization','locking'):
        assert abs(draws[name].mean()-np.dot(w,obs[name]))<.025


def test_validation():
    for kw in [dict(K=-1),dict(length=1),dict(q=128),dict(spacing=0)]:
        args=dict(K=1.,length=4,samples=2,seed=1);args.update(kw)
        with pytest.raises(ValueError):sample_clock(**args)
