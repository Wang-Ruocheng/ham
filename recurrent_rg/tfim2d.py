"""Square-lattice TFIM snapshots from a periodic 2+1D Trotter Ising model.

Pauli convention H=-sum_<ij> Z_i Z_j - h sum_i X_i; J=1.
K_space=dtau; K_time=-log(tanh(dtau*h))/2. Only one random equal-time
slice is saved per Markov-chain observation, not all correlated time slices.
Fixed-count Swendsen-Wang sweeps avoid state-dependent observation times.
Finite beta and dtau are not claimed to be ground-state/continuum converged.
"""
import numpy as np
from numba import njit
from recurrent_rg.ising2d_field import _root


def forward_neighbors(size, slices):
    if size < 2 or slices < 2:
        raise ValueError('Spatial size and time slices must be at least two')
    index = np.arange(slices * size * size).reshape(slices, size, size)
    return np.stack([np.roll(index, -1, axis=a).ravel() for a in (1, 2, 0)], axis=1)


@njit(nogil=True)
def _sample(neighbors, size, slices, h, dtau, draws, equil, gap, seed, initial):
    np.random.seed(seed)
    n = len(neighbors)
    spins = np.empty(n, dtype=np.int8)
    for i in range(n):
        spins[i] = initial if initial else (1 if np.random.random() < .5 else -1)
    parent = np.empty(n, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)
    colors = np.empty(n, dtype=np.int8)
    result = np.empty((draws, size, size), dtype=np.int8)
    probabilities = np.array([-np.expm1(-2*dtau), -np.expm1(-2*dtau), 1-np.tanh(dtau*h)])
    for step in range(equil + draws*gap):
        for i in range(n):
            parent[i], counts[i] = i, 1
        for i in range(n):
            for axis in range(3):
                j = neighbors[i, axis]
                if spins[i] == spins[j] and np.random.random() < probabilities[axis]:
                    a, b = _root(parent, i), _root(parent, j)
                    if a != b:
                        if counts[a] < counts[b]:
                            a, b = b, a
                        parent[b] = a
                        counts[a] += counts[b]
        for i in range(n):
            if parent[i] == i:
                colors[i] = 1 if np.random.random() < .5 else -1
        for i in range(n):
            spins[i] = colors[_root(parent, i)]
        if step >= equil and (step-equil+1) % gap == 0:
            t = np.random.randint(slices)
            result[(step-equil)//gap] = spins[t*size*size:(t+1)*size*size].reshape(size, size)
    return result


def sample_tfim2d_chain(h, size=16, slices=80, dtau=.1, draws=128,
                        equil=256, gap=8, seed=91, initial=0):
    if not np.isfinite(h) or h <= 0 or not np.isfinite(dtau) or dtau <= 0:
        raise ValueError('h and dtau must be finite and positive')
    if draws < 1 or equil < 0 or gap < 1 or initial not in (-1, 0, 1):
        raise ValueError('Invalid sampling count or initial state')
    return _sample(forward_neighbors(size, slices), size, slices, h, dtau,
                   draws, equil, gap, seed, initial)
