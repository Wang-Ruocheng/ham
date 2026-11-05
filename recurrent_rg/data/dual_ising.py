"""Two uncoupled square Ising layers, encoded as a joint four-state field."""
import numpy as np
from numba import njit
from .ashkin_teller import _conditional


@njit(cache=True)
def _sample(k1,k2,length,samples,seed,burn,spacing,ordered):
    np.random.seed(seed)
    n=length*length
    s=np.ones(n,np.int64) if ordered else 2*np.random.randint(0,2,n)-1
    t=np.ones(n,np.int64) if ordered else 2*np.random.randint(0,2,n)-1
    dummy=np.zeros(n,np.int64)
    out=np.empty((samples,length,length),np.int8)
    for step in range(burn+samples*spacing):
        _conditional(s,dummy,length,k1,0.)
        _conditional(t,dummy,length,k2,0.)
        if step>=burn and (step-burn+1)%spacing==0:
            out[(step-burn)//spacing]=((s+1)//2*2+(t+1)//2).reshape(length,length)
    return out


def sample_dual_ising(K,length,samples,seed,ratio=.5,burn=1024,spacing=16,ordered=False):
    """K=J1/T, ratio=J2/J1. No inter-layer coupling; two independent SW sweeps."""
    if not np.isfinite(K) or K<0 or not np.isfinite(ratio) or ratio<=0:
        raise ValueError('Require K>=0 and ratio>0, both finite')
    if length<2 or samples<1 or burn<0 or spacing<1:
        raise ValueError('Invalid sampling settings')
    return _sample(K,ratio*K,length,samples,seed,burn,spacing,ordered)


def sample_dual_ising_couplings(K1,K2,length,samples,seed,burn=1024,spacing=16,ordered=False):
    """Independent nonnegative couplings, including either zero-coupling axis."""
    if not np.isfinite(K1) or not np.isfinite(K2) or min(K1,K2)<0:
        raise ValueError('Couplings must be finite and nonnegative')
    if length<2 or samples<1 or burn<0 or spacing<1:
        raise ValueError('Invalid sampling settings')
    return _sample(K1,K2,length,samples,seed,burn,spacing,ordered)
