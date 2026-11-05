import numpy as np
import pytest
from recurrent_rg.data.ashkin_teller import sample_ashkin_teller, observables
from recurrent_rg.data.ashkin_teller import sample_ashkin_teller_couplings


@pytest.mark.parametrize('J,U',[(0.,.4),(.3,0.),(.3,.1),(.2,.4),(.25,.25)])
def test_independent_couplings_against_exact(J,U):
    states=((np.arange(256)[:,None]//(4**np.arange(4)))%4).reshape(-1,2,2)
    s,t=2*(states//2)-1,2*(states%2)-1;p=s*t
    logw=sum((J*(s*np.roll(s,1,axis)+t*np.roll(t,1,axis))+U*p*np.roll(p,1,axis)).sum((1,2)) for axis in (1,2))
    w=np.exp(logw-logw.max());w/=w.sum()
    x=sample_ashkin_teller_couplings(J,U,2,16000,82,spacing=4)
    exact=observables(states);sampled=observables(x)
    for key in ['m_sigma','m_tau','m_product']:
        assert abs(sampled[key].mean()-w@exact[key])<.04


def test_new_sampler_preserves_ratio_two_path():
    np.testing.assert_array_equal(sample_ashkin_teller(.2,4,32,5),
        sample_ashkin_teller_couplings(.2,.4,4,32,5,burn=512,spacing=8))
    with pytest.raises(ValueError):sample_ashkin_teller_couplings(.2,-.1,4,32,5)


def test_against_exact_small_torus():
    # Enumerate 4^4 states, counting the same periodic bonds as the sampler.
    states = ((np.arange(256)[:,None] // (4**np.arange(4))) % 4).reshape(-1,2,2)
    exact = observables(states)
    K = .23
    weights = np.exp(-K*4*exact['energy']); weights /= weights.sum()
    x = sample_ashkin_teller(K, 2, 12000, 81, spacing=4)
    sampled = observables(x)
    for key in exact:
        assert abs(sampled[key].mean() - weights@exact[key]) < .07


def test_independent_limit_and_reproducibility():
    x = sample_ashkin_teller(0., 8, 256, 12)
    assert np.array_equal(x, sample_ashkin_teller(0., 8, 256, 12))
    assert max(abs(np.bincount(x.ravel(), minlength=4)/x.size-.25)) < .02
    with pytest.raises(ValueError): sample_ashkin_teller(.2, 8, 10, 1, ratio=.5)


def test_middle_dense_grid_keeps_total_size():
    from scripts.generate_ashkin_teller2d import coupling_grid
    uniform=coupling_grid();dense=coupling_grid('middle_dense')
    assert len(dense)==len(uniform)==48
    assert np.all(np.diff(dense)>0)
    assert dense[0]==uniform[0] and dense[-1]==uniform[-1]
    assert np.sum((dense>=.19)&(dense<=.22))==16
    assert np.sum((uniform>=.19)&(uniform<=.22))==2
