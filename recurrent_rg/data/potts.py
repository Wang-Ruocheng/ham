"""Zero-field ferromagnetic square Potts sampling via Swendsen-Wang sweeps."""
import numpy as np
from numba import njit


@njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _sweep(spins, length, q, probability):
    n = length*length
    parent = np.arange(n)
    for i in range(n):
        row, col = i//length, i%length
        for j in (row*length+(col+1)%length, ((row+1)%length)*length+col):
            if spins[i] == spins[j] and np.random.random() < probability:
                a, b = _root(parent,i), _root(parent,j)
                parent[a] = b
    colors = np.random.randint(0,q,n)
    for i in range(n):
        spins[i] = colors[_root(parent,i)]


@njit(cache=True)
def sample_potts(K, length, samples, seed, q=3, burn=512, spacing=8, ordered=False):
    """Each sweep bonds equal neighbors with p=1-exp(-K), then recolors clusters.

    Returned configurations are MCMC draws (not claimed independent).
    """
    np.random.seed(seed)
    spins = np.zeros(length*length,np.int64) if ordered else np.random.randint(0,q,length*length)
    probability = -np.expm1(-K)
    for _ in range(burn):
        _sweep(spins,length,q,probability)
    result = np.empty((samples,length,length),np.int8)
    for sample in range(samples):
        for _ in range(spacing):
            _sweep(spins,length,q,probability)
        result[sample] = spins.reshape(length,length)
    return result
