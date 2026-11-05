import numpy as np
import pytest
from recurrent_rg.data.blume_capel import sample_signed_spin1, observable_series


def test_signed_sampler_gauge_and_energy():
    plus=sample_signed_spin1(.6,6,8,91,burn=10,spacing=2)
    minus=sample_signed_spin1(-.6,6,8,91,burn=10,spacing=2)
    eta=1-2*(np.indices((6,6)).sum(0)%2)
    np.testing.assert_array_equal(minus,plus*eta)
    np.testing.assert_array_equal(observable_series(plus,0,.6)['energy'],observable_series(minus,0,-.6)['energy'])
    assert set(np.unique(plus)) <= {-1,0,1}


def test_signed_sampler_zero_and_validation():
    x=sample_signed_spin1(0.,12,128,91,burn=20,spacing=4)
    p=np.bincount((x+1).ravel(),minlength=3)/x.size
    assert np.max(np.abs(p-1/3))<.025
    for K,L in [(np.nan,6),(-.6,5)]:
        with pytest.raises(ValueError):sample_signed_spin1(K,L,4,91)
