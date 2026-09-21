"""Finite-field Swendsen-Wang sampling: cluster heat-bath recoloring.

P(s) proportional to exp(K sum_nn s_i s_j + B sum_i s_i).
Bond probability is 1-exp(-2K); a cluster of n spins is recolored + with
probability sigmoid(2 B n). Fixed sweep counts, no global flip augmentation.
"""
import numpy as np
from numba import njit


@njit
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit
def sample_field_chain(k, b, size, draws, equil, gap, seed, initial=0):
    if k < 0 or size < 2 or draws < 1 or equil < 0 or gap < 1:
        raise ValueError('Invalid ferromagnetic sampler arguments')
    np.random.seed(seed)
    n = size*size
    spins = np.empty(n, dtype=np.int8)
    for i in range(n):
        spins[i] = initial if initial != 0 else (1 if np.random.random() < .5 else -1)
    result = np.empty((draws, size, size), dtype=np.int8)
    parent, counts = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    colors = np.empty(n, dtype=np.int8)
    probability = 1-np.exp(-2*k)
    for step in range(equil+draws*gap):
        for i in range(n):
            parent[i] = i
            counts[i] = 1
        for i in range(n):
            row, col = i//size, i % size
            for j in (row*size+(col+1) % size, ((row+1) % size)*size+col):
                if spins[i] == spins[j] and np.random.random() < probability:
                    a, c = _root(parent, i), _root(parent, j)
                    if a != c:
                        if counts[a] < counts[c]:
                            a, c = c, a
                        parent[c] = a
                        counts[a] += counts[c]
        for i in range(n):
            if parent[i] == i:
                # Stable logistic, avoiding overflow for large ordered clusters.
                z = 2*b*counts[i]
                plus = 1/(1+np.exp(-z)) if z >= 0 else np.exp(z)/(1+np.exp(z))
                colors[i] = 1 if np.random.random() < plus else -1
        for i in range(n):
            spins[i] = colors[_root(parent, i)]
        if step >= equil and (step-equil+1) % gap == 0:
            result[(step-equil)//gap] = spins.reshape(size, size)
    return result


def field_observables(x):
    m = x.mean(axis=(-2, -1))
    bond = (x*np.roll(x, 1, -1)+x*np.roll(x, 1, -2)).mean(axis=(-2, -1))
    return np.stack((bond, m, m*m), axis=-1)


def chain_statistics(x):
    result = {}
    for name, z in zip(('bond_per_site', 'm', 'm2'), field_observables(x).T):
        centered = z-z.mean()
        variance = (centered**2).mean()
        acf = [float(np.mean(centered[:-lag]*centered[lag:])/variance) if variance else 0.
               for lag in range(1, min(65, len(z)//2))]
        tau = 1.
        for i in range(0, len(acf)-1, 2):
            pair = acf[i]+acf[i+1]
            if pair <= 0:
                break
            tau += 2*pair
        result[name] = {'mean': float(z.mean()), 'se': float(np.sqrt(variance*tau/len(z))),
                        'lag1': acf[0], 'ess_estimate': len(z)/tau}
    return result
