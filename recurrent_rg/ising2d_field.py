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


@njit
def sample_signed_field_chain(k, b, size, draws, equil, gap, seed, initial=0):
    """Signed K on an even periodic square lattice, with uniform physical B.

    For K<0 set s_i=eta_i*t_i, eta=(-1)^(row+col). The t variables
    have ferromagnetic |K| bonds and staggered fields B*eta_i. A cluster
    is recolored + with sigmoid(2*B*sum_cluster eta). initial +/-1 names
    the two ordered states of t (checkerboards in physical spins).
    Positive K delegates to the unchanged historical sampler.
    """
    if initial not in (-1, 0, 1):
        raise ValueError('initial must be -1, 0 or 1')
    if k >= 0:
        return sample_field_chain(k, b, size, draws, equil, gap, seed, initial)
    if size < 2 or size % 2 or draws < 1 or equil < 0 or gap < 1:
        raise ValueError('Antiferromagnetic gauge sampler needs an even periodic lattice')
    np.random.seed(seed)
    n = size*size
    eta = np.empty(n, dtype=np.int8)
    spins = np.empty(n, dtype=np.int8)
    for i in range(n):
        eta[i] = 1 if (i//size+i%size)%2 == 0 else -1
        spins[i] = initial if initial != 0 else (1 if np.random.random()<.5 else -1)
    result = np.empty((draws, size, size), dtype=np.int8)
    parent, counts = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    charges = np.empty(n, dtype=np.int64)
    colors = np.empty(n, dtype=np.int8)
    probability = 1-np.exp(2*k)
    for step in range(equil+draws*gap):
        for i in range(n):
            parent[i] = i
            counts[i] = 1
            charges[i] = 0
        for i in range(n):
            row, col = i//size, i%size
            for j in (row*size+(col+1)%size, ((row+1)%size)*size+col):
                if spins[i] == spins[j] and np.random.random()<probability:
                    a, c = _root(parent, i), _root(parent, j)
                    if a != c:
                        if counts[a]<counts[c]: a, c = c, a
                        parent[c] = a
                        counts[a] += counts[c]
        for i in range(n):
            charges[_root(parent, i)] += eta[i]
        for i in range(n):
            if parent[i] == i:
                z = 2*b*charges[i]
                plus = 1/(1+np.exp(-z)) if z>=0 else np.exp(z)/(1+np.exp(z))
                colors[i] = 1 if np.random.random()<plus else -1
        for i in range(n): spins[i] = colors[_root(parent, i)]
        if step>=equil and (step-equil+1)%gap == 0:
            result[(step-equil)//gap] = (spins*eta).reshape(size, size)
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
