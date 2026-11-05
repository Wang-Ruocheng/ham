"""Square ferromagnetic Ashkin--Teller sampling, U/J >= 1.

Use p=sigma*tau. Conditional sigma bonds have coupling K(1+p_i*p_j),
and conditional p bonds have coupling K(ratio+sigma_i*sigma_j).
Both are nonnegative, allowing exact conditional Swendsen--Wang sweeps.
"""
import numpy as np
from numba import njit
from .potts import _root


@njit(cache=True)
def _conditional(spins, other, length, base, interaction):
    n = length * length
    parent = np.arange(n)
    for i in range(n):
        row, col = i // length, i % length
        for j in (row*length+(col+1)%length, ((row+1)%length)*length+col):
            coupling = base + interaction*other[i]*other[j]
            if spins[i] == spins[j] and np.random.random() < -np.expm1(-2*coupling):
                parent[_root(parent, i)] = _root(parent, j)
    colors = 2*np.random.randint(0, 2, n)-1
    for i in range(n):
        spins[i] = colors[_root(parent, i)]


@njit(cache=True)
def _sample(K, ratio, length, samples, seed, burn, spacing, ordered):
    np.random.seed(seed)
    n = length*length
    sigma = np.ones(n, np.int64) if ordered else 2*np.random.randint(0, 2, n)-1
    product = np.ones(n, np.int64) if ordered else 2*np.random.randint(0, 2, n)-1
    result = np.empty((samples, length, length), np.int8)
    for step in range(burn + samples*spacing):
        _conditional(sigma, product, length, K, K)
        _conditional(product, sigma, length, ratio*K, K)
        if step >= burn and (step-burn+1) % spacing == 0:
            tau = product*sigma
            result[(step-burn)//spacing] = ((sigma+1)//2*2+(tau+1)//2).reshape(length, length)
    return result


def sample_ashkin_teller(K, length, samples, seed, ratio=2., burn=512, spacing=8, ordered=False):
    """Return MCMC draws encoded as 0:(--), 1:(-+), 2:(+-), 3:(++)."""
    if not np.isfinite(K) or K < 0 or not np.isfinite(ratio) or ratio < 1:
        raise ValueError('Require K>=0 and U/J>=1')
    if length < 2 or samples < 1 or burn < 0 or spacing < 1:
        raise ValueError('Invalid lattice or sampling settings')
    return _sample(K, ratio, length, samples, seed, burn, spacing, ordered)


@njit(cache=True)
def _sample_couplings(J, U, length, samples, seed, burn, spacing, ordered):
    np.random.seed(seed)
    n=length*length
    sigma=np.ones(n,np.int64) if ordered else 2*np.random.randint(0,2,n)-1
    other=np.ones(n,np.int64) if ordered else 2*np.random.randint(0,2,n)-1
    result=np.empty((samples,length,length),np.int8)
    for step in range(burn+samples*spacing):
        if U>=J:
            # other = sigma*tau; both conditional bond families nonnegative.
            _conditional(sigma,other,length,J,J)
            _conditional(other,sigma,length,U,J)
            tau=other*sigma
        else:
            # other = tau; J+U*s_i*s_j >= 0 in this region.
            _conditional(sigma,other,length,J,U)
            _conditional(other,sigma,length,J,U)
            tau=other
        if step>=burn and (step-burn+1)%spacing==0:
            result[(step-burn)//spacing]=((sigma+1)//2*2+(tau+1)//2).reshape(length,length)
    return result


def sample_ashkin_teller_couplings(J,U,length,samples,seed,burn=1024,spacing=16,ordered=False):
    """Independent dimensionless J/T,U/T >=0, with valid conditional SW updates."""
    if not np.isfinite(J) or not np.isfinite(U) or J<0 or U<0:
        raise ValueError('Require finite nonnegative dimensionless J and U')
    if length<2 or samples<1 or burn<0 or spacing<1:
        raise ValueError('Invalid lattice or sampling settings')
    return _sample_couplings(J,U,length,samples,seed,burn,spacing,ordered)


def observables(x, ratio=2.):
    sigma, tau = 2*(x//2)-1, 2*(x%2)-1
    product = sigma*tau
    energy = np.zeros(len(x))
    for axis in (1, 2):
        energy -= (sigma*np.roll(sigma, 1, axis)+tau*np.roll(tau, 1, axis)
                   +ratio*product*np.roll(product, 1, axis)).mean((1, 2))
    return dict(m_sigma=np.abs(sigma.mean((1, 2))), m_tau=np.abs(tau.mean((1, 2))),
                m_product=np.abs(product.mean((1, 2))), energy=energy)
