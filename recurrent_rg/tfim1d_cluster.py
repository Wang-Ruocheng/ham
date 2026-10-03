"""Fixed-sweep Swendsen-Wang sampling of the 1+1D Trotter TFIM.

Same finite-beta/discrete-time measure as tfim.sample_tfim_ensembles,
but uses full cluster sweeps, not Wolff single-cluster updates.
"""
import numpy as np
from numba import njit
from recurrent_rg.ising2d_field import _root


@njit(nogil=True, cache=True)
def _sample(h, sites, slices, dtau, draws, equil, gap, seed, initial):
    np.random.seed(seed)
    n = sites * slices
    spins = np.empty(n, dtype=np.int8)
    for i in range(n):
        spins[i] = initial if initial else (1 if np.random.random() < .5 else -1)
    parent = np.empty(n, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)
    colors = np.empty(n, dtype=np.int8)
    result = np.empty((draws, sites), dtype=np.int8)
    ps, pt = -np.expm1(-2*dtau), 1-np.tanh(dtau*h)
    for step in range(equil + draws*gap):
        for i in range(n):
            parent[i], counts[i] = i, 1
        for i in range(n):
            t, s = i//sites, i % sites
            for axis in range(2):
                j = t*sites+(s+1) % sites if axis == 0 else ((t+1) % slices)*sites+s
                p = ps if axis == 0 else pt
                if spins[i] == spins[j] and np.random.random() < p:
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
            result[(step-equil)//gap] = spins[t*sites:(t+1)*sites]
    return result


def sample_tfim1d_chain(h, sites=16, slices=32, dtau=.5, draws=128,
                        equil=512, gap=16, seed=91, initial=0):
    if not np.isfinite(h) or h < 0 or not np.isfinite(dtau) or dtau <= 0:
        raise ValueError('Need finite h >= 0 and dtau > 0')
    if sites < 4 or slices < 4 or draws < 1 or equil < 0 or gap < 1:
        raise ValueError('Invalid geometry or sampling counts')
    if initial not in (-1, 0, 1):
        raise ValueError('Invalid initial state')
    return _sample(h, sites, slices, dtau, draws, equil, gap, seed, initial)
