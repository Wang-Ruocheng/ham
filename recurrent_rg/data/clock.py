"""Ferromagnetic square-lattice clock model; embedded-reflection cluster sampling.

H/J = -sum_<ij> cos(2*pi*(a_i-a_j)/q), K=J/T, periodic boundaries.
One sweep chooses a reflection a -> (m-a) mod q, builds embedded Ising
clusters and independently reflects each cluster with probability 1/2.
"""
import numpy as np
from numba import njit
from .potts import _root


@njit(cache=True)
def _sweep(spins, length, q, K):
    n = length*length
    m = np.random.randint(q)
    # Normal component to the reflection axis alpha=pi*m/q.
    projection = np.sin(np.pi*(2*np.arange(q)-m)/q)
    parent = np.arange(n)
    for i in range(n):
        row, col = i//length, i%length
        for j in (row*length+(col+1)%length, ((row+1)%length)*length+col):
            product = projection[spins[i]]*projection[spins[j]]
            if product > 0 and np.random.random() < -np.expm1(-2*K*product):
                parent[_root(parent, i)] = _root(parent, j)
    flip = np.random.random(n) < .5
    for i in range(n):
        if flip[_root(parent, i)]:
            spins[i] = (m-spins[i]) % q


@njit(cache=True)
def _sample(K, length, samples, seed, q, burn, spacing, ordered):
    np.random.seed(seed)
    spins = np.zeros(length*length, np.int64) if ordered else np.random.randint(0, q, length*length)
    for _ in range(burn): _sweep(spins, length, q, K)
    result = np.empty((samples, length, length), np.int8)
    for t in range(samples):
        for _ in range(spacing): _sweep(spins, length, q, K)
        result[t] = spins.reshape(length, length)
    return result


def sample_clock(K, length, samples, seed, q=6, burn=1024, spacing=32, ordered=False):
    if not np.isfinite(K) or K < 0 or not 2 <= q <= 127:
        raise ValueError('Requires finite ferromagnetic K and 2<=q<=127')
    if any(not isinstance(v, (int, np.integer)) for v in (length, samples, seed, q, burn, spacing)):
        raise ValueError('Counts and seed must be integers')
    if length < 2 or samples < 1 or burn < 0 or spacing < 1:
        raise ValueError('Invalid lattice size or sampling counts')
    return _sample(float(K), length, samples, seed, q, burn, spacing, ordered)


def observables(x, q=6):
    theta = 2*np.pi*np.asarray(x)/q
    magnet = np.exp(1j*theta).mean((1, 2))
    bond = sum(np.cos(theta-np.roll(theta, 1, ax)).mean((1, 2)) for ax in (1, 2))/2
    # At vanishing magnetization the angle is undefined; suppress roundoff.
    locking = np.where(abs(magnet)>1e-12, np.cos(q*np.angle(magnet)), 0.)
    corr = np.cos(theta-np.roll(theta, x.shape[1]//2, 1)).mean((1, 2))
    return dict(bond=bond, energy=-2*bond, magnetization=abs(magnet), locking=locking, half_box_correlation=corr)
