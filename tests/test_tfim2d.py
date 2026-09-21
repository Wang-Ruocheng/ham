import numpy as np
import pytest
from recurrent_rg.tfim2d import forward_neighbors, sample_tfim2d_chain


def test_topology_and_contract():
    neighbors = forward_neighbors(4, 8)
    assert neighbors.shape == (128, 3)
    assert neighbors[0].tolist() == [4, 1, 16]
    assert neighbors[-1].tolist() == [115, 124, 15]
    kw = dict(h=3., size=4, slices=8, draws=8, equil=8, gap=2, seed=7)
    x = sample_tfim2d_chain(**kw)
    assert x.shape == (8, 4, 4) and set(np.unique(x)) <= {-1, 1}
    np.testing.assert_array_equal(x, sample_tfim2d_chain(**kw))
    with pytest.raises(ValueError):
        sample_tfim2d_chain(h=0)


def test_against_exact_trotter_distribution():
    # L=M=2 has parallel periodic bonds; enumeration uses the same bonds.
    h, dtau = 1.7, .3
    states = 2*((np.arange(256)[:,None] >> np.arange(8)) & 1)-1
    neighbors = forward_neighbors(2, 2)
    strengths = np.array([dtau, dtau, -.5*np.log(np.tanh(dtau*h))])
    logw = ((states[:,:,None]*states[:,neighbors])*strengths).sum((1,2))
    w = np.exp(logw-logw.max()); w /= w.sum()
    target = w @ states[:,:4].mean(1)**2
    x = sample_tfim2d_chain(h, size=2, slices=2, dtau=dtau,
                           draws=30000, equil=256, gap=2, seed=123)
    assert abs((x.mean((1,2))**2).mean()-target) < .025


def test_small_dtau_against_quantum_exact_diagonalization():
    # Pauli Z=+/-1 and periodic L=2 parallel bonds, as in the sampler.
    states = 2*((np.arange(16)[:,None] >> np.arange(4)) & 1)-1
    square = states.reshape(16,2,2)
    bonds = (square*np.roll(square,1,1)+square*np.roll(square,1,2)).sum((1,2))
    hamiltonian = np.diag(-bonds.astype(float))
    for i in range(16):
        for bit in range(4):
            hamiltonian[i,i^(1<<bit)] = -3.
    energy,vectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-(energy-energy.min()))
    diagonal = vectors**2 @ weights / weights.sum()
    expected = diagonal @ states.mean(1)**2
    x = sample_tfim2d_chain(3., size=2, slices=40, dtau=.025,
                           draws=20000, equil=256, gap=2, seed=321)
    assert abs((x.mean((1,2))**2).mean()-expected) < .02
