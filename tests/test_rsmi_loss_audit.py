import itertools
import numpy as np
from scripts.audit_ising1d_rsmi_loss import exact_patterns


def test_exact_stochastic_block_transfer_matches_full_enumeration():
    j=.7;length=8
    table=np.array([[.9,.1],[.3,.7],[.6,.4],[.05,.95]])
    actual,_=exact_patterns(j,table,length=length,steps=1)
    states=np.array(list(itertools.product(range(2),repeat=length)))
    spins=2*states-1
    weight=np.exp(j*(spins*np.roll(spins,-1,axis=1)).sum(-1));weight/=weight.sum()
    expected=np.zeros(8)
    for row,w in zip(states,weight):
        ids=row[::2]+2*row[1::2]
        for i,pattern in enumerate(itertools.product(range(2),repeat=3)):
            expected[i]+=w*np.prod([table[ids[k],pattern[k]] for k in range(3)])
    np.testing.assert_allclose(actual[1],expected,rtol=1e-12,atol=1e-12)


def test_AND_and_decimation_zero_field_means():
    and_table=np.array([[1,0],[1,0],[1,0],[0,1]])
    _,m=exact_patterns(0,and_table)
    np.testing.assert_allclose(m,[2*.5**(2**k)-1 for k in range(6)],atol=1e-12)
    dec=np.array([[1,0],[0,1],[1,0],[0,1]])
    _,m=exact_patterns(1,dec)
    np.testing.assert_allclose(m,0,atol=1e-12)
