import numpy as np
import pytest
from recurrent_rg.data.dual_ising import sample_dual_ising
from recurrent_rg.data.dual_ising import sample_dual_ising_couplings


@pytest.mark.parametrize('K1,K2',[(0.,.5),(.5,0.),(.2,.6),(.6,.2)])
def test_independent_axes_against_exact(K1,K2):
    states=((np.arange(256)[:,None]//4**np.arange(4))%4).reshape(-1,2,2)
    s,t=2*(states//2)-1,2*(states%2)-1
    lw=sum((K1*s*np.roll(s,1,a)+K2*t*np.roll(t,1,a)).sum((1,2)) for a in [1,2])
    w=np.exp(lw-lw.max());w/=w.sum()
    actual=observables(sample_dual_ising_couplings(K1,K2,2,16000,32,spacing=4));expected=observables(states)
    for k in ['m_sigma','m_tau','m_product']:assert abs(actual[k].mean()-w@expected[k])<.035
from recurrent_rg.data.ashkin_teller import observables


@pytest.mark.parametrize('K',[0.,.3,.6])
def test_dual_exact_small_torus(K):
    states=((np.arange(256)[:,None]//4**np.arange(4))%4).reshape(-1,2,2)
    s,t=2*(states//2)-1,2*(states%2)-1
    logw=sum((K*s*np.roll(s,1,a)+.5*K*t*np.roll(t,1,a)).sum((1,2)) for a in [1,2])
    w=np.exp(logw-logw.max());w/=w.sum()
    x=sample_dual_ising(K,2,16000,31,spacing=4)
    actual=observables(x);expected=observables(states)
    for key in ['m_sigma','m_tau','m_product']:
        assert abs(actual[key].mean()-w@expected[key])<.035
    assert abs(np.cov(actual['m_sigma'],actual['m_tau'])[0,1])<.015


def test_dual_reproducible_and_validation():
    np.testing.assert_array_equal(sample_dual_ising(.3,4,16,3),sample_dual_ising(.3,4,16,3))
    with pytest.raises(ValueError):sample_dual_ising(.3,4,16,3,ratio=-1)
