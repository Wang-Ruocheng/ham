"""Independent checks for the isolated PT / Houdayer comparison kernel."""

from itertools import product

import numpy as np
import pytest

from recurrent_rg.data.spin_glass import energy, generate_bonds, make_periodic_lattice
from recurrent_rg.data.spin_glass_houdayer import (
    _flip_disagreement_component,
    houdayer_cluster_update,
    sample_houdayer_parallel_tempering,
)


def test_cluster_preserves_overlap_joint_energy_and_is_an_involution_for_real_bonds_and_field():
    rng = np.random.default_rng(931)
    bonds = rng.normal(size=(2, 3, 4))
    for _ in range(20):
        a = rng.choice(np.array([-1, 1], dtype=np.int8), size=(3, 4))
        b = rng.choice(np.array([-1, 1], dtype=np.int8), size=(3, 4))
        first = houdayer_cluster_update(a, b, bonds, seed=341, field=-0.173)
        np.testing.assert_array_equal(first["spins_a"] * first["spins_b"], a * b)
        assert first["energies_after"].sum() == pytest.approx(first["energies_before"].sum(), abs=2e-13)
        assert abs(first["joint_energy_residual"]) < 2e-13
        reverse = houdayer_cluster_update(first["spins_a"], first["spins_b"], bonds, seed=341, field=-0.173)
        np.testing.assert_array_equal(reverse["spins_a"], a)
        np.testing.assert_array_equal(reverse["spins_b"], b)
        np.testing.assert_array_equal(first["cluster_indices"], reverse["cluster_indices"])


def test_cluster_is_complete_periodic_nearest_neighbor_component_and_does_not_mutate_inputs():
    bonds = np.ones((2, 4, 4))
    a = np.ones((4, 4), dtype=np.int8)
    b = a.copy()
    b[0, 0] = b[3, 0] = b[2, 2] = -1
    original = b.copy()
    components = set()
    for seed in range(20):
        result = houdayer_cluster_update(a, b, bonds, seed=seed)
        component = frozenset(result["cluster_indices"].tolist())
        assert component in (frozenset([0, 12]), frozenset([10]))
        components.add(component)
    assert len(components) == 2
    np.testing.assert_array_equal(a, np.ones((4, 4), dtype=np.int8))
    np.testing.assert_array_equal(b, original)
    identity = houdayer_cluster_update(a, a, bonds, seed=4)
    assert identity["cluster_size"] == 0
    np.testing.assert_array_equal(identity["spins_a"], a)


def test_cluster_kernel_exact_detailed_balance_with_state_weighted_component_selection():
    # For every selected site, q and the complete component are unchanged.
    # Therefore the number of sites selecting this transition is unchanged in
    # reverse, and equal-temperature pair Boltzmann weights are identical.
    # This checks all site choices for every 3x3 first-copy state against
    # several distinct second copies, using the exact Hamiltonian.
    rng = np.random.default_rng(31)
    bonds = rng.normal(scale=0.3, size=(2, 3, 3))
    forward, backward = make_periodic_lattice((3, 3))
    states = np.asarray(list(product((-1, 1), repeat=9)), dtype=np.int8)
    for a0 in states:
        for b0 in states[[0, 83, 511]]:
            disagreeing = np.flatnonzero(a0 != b0)
            if not disagreeing.size:
                continue
            before = float(energy(np.stack([a0, b0]).reshape(2, 3, 3), bonds, field=0.13).sum())
            for chosen in disagreeing:
                a, b = a0.copy(), b0.copy()
                queue = np.empty(9, dtype=np.int64)
                marked = np.empty(9, dtype=np.bool_)
                count = _flip_disagreement_component(a, b, bonds.reshape(2, 9), forward, backward, chosen, queue, marked)
                assert count > 0
                after = float(energy(np.stack([a, b]).reshape(2, 3, 3), bonds, field=0.13).sum())
                np.testing.assert_array_equal(a * b, a0 * b0)
                proposal = count / disagreeing.size
                reverse_proposal = np.sum(marked) / np.sum(a != b)
                assert np.exp(-before / 1.7) * proposal == pytest.approx(
                    np.exp(-after / 1.7) * reverse_proposal, rel=2e-13,
                )
                _flip_disagreement_component(a, b, bonds.reshape(2, 9), forward, backward, chosen, queue, marked)
                np.testing.assert_array_equal(a, a0)
                np.testing.assert_array_equal(b, b0)


def test_zero_bonds_disconnect_components_and_periodic_nonzero_bond_reconnects():
    bonds = np.zeros((2, 3, 3))
    a = np.ones((3, 3), dtype=np.int8)
    b = -a
    for seed in range(8):
        result = houdayer_cluster_update(a, b, bonds, seed=seed, field=0.3)
        assert result["cluster_size"] == 1
        assert result["joint_energy_residual"] == pytest.approx(0.0)
    # A single periodic edge joins sites (2,0) and (0,0), while their other
    # geometric neighbors remain disconnected even though all q values are -1.
    bonds[0, 2, 0] = -0.71
    seen = set()
    for seed in range(30):
        result = houdayer_cluster_update(a, b, bonds, seed=seed, field=-0.2)
        sites = frozenset(result["cluster_indices"].tolist())
        assert sites == frozenset([0, 6]) or (len(sites) == 1 and not sites & {0, 6})
        seen.add(sites)
        assert abs(result["joint_energy_residual"]) < 1e-13
    assert frozenset([0, 6]) in seen


@pytest.mark.parametrize("algorithm", ["pt", "houdayer_pt"])
@pytest.mark.parametrize("initial", ["hot", "cold"])
def test_sampler_determinism_energy_tracking_counters_and_traces(algorithm, initial):
    bonds = generate_bonds((3, 4), 51, distribution="gaussian", scale=0.4)
    original = bonds.copy()
    options = dict(samples=7, burn_in=5, spacing=2, seed=34, replicas=4,
                   algorithm=algorithm, initial=initial, exchange_interval=2,
                   cluster_interval=3, trace_interval=4, field=0.17)
    first = sample_houdayer_parallel_tempering(bonds, [0.8, 1.5, 3.0], **options)
    second = sample_houdayer_parallel_tempering(bonds, [0.8, 1.5, 3.0], **options)
    for key, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second[key])
    np.testing.assert_array_equal(bonds, original)
    assert first["configurations"].shape == (3, 4, 7, 3, 4)
    assert first["configurations"].dtype == np.int8
    np.testing.assert_allclose(first["energies"], energy(first["configurations"], bonds, 0.17), atol=1e-12)
    assert first["max_energy_tracking_drift"] < 1e-12
    assert first["max_cluster_joint_energy_residual"] < 1e-12
    assert not first["equilibration_certified"]
    np.testing.assert_array_equal(first["trace_sweeps"], [0, 4, 8, 12, 16, 19])
    assert first["trace_energy_density"].shape == first["trace_q2"].shape == (3, 6)
    assert np.all((first["trace_q2"] >= 0) & (first["trace_q2"] <= 1))
    np.testing.assert_array_equal(first["site_visits"], np.full((4, 3), 19 * 12))
    np.testing.assert_array_equal(first["temperature_visits"].sum(axis=2), np.full((4, 3), 20))
    np.testing.assert_array_equal(first["temperature_visits"].sum(axis=1), np.full((4, 3), 20))
    for prefix in ("local", "swap"):
        assert np.all(first[f"{prefix}_accepts"] <= first[f"{prefix}_attempts"])
    attempts = 12 if algorithm == "houdayer_pt" else 0
    np.testing.assert_array_equal(first["cluster_attempts"], np.full(3, attempts))
    np.testing.assert_array_equal(first["cluster_size_histogram"].sum(axis=1), first["cluster_attempts"])
    np.testing.assert_array_equal(first["cluster_size_histogram"][:, 1:].sum(axis=1), first["cluster_updates"])
    np.testing.assert_array_equal(
        first["cluster_size_histogram"] @ np.arange(13), first["cluster_size_sum"],
    )


@pytest.mark.parametrize("algorithm", ["pt", "houdayer_pt"])
def test_tiny_lattice_stationary_energy_and_overlap_match_exact_enumeration(algorithm):
    bonds = generate_bonds((3, 3), 51, distribution="gaussian", scale=0.2)
    states = np.asarray(list(product((-1, 1), repeat=9)), dtype=np.int8).reshape(512, 3, 3)
    exact_energy = energy(states, bonds, field=0.11)
    temperatures = np.array([0.9, 1.7])
    result = sample_houdayer_parallel_tempering(
        bonds, temperatures, samples=4096, burn_in=128, spacing=2, seed=99,
        replicas=4, algorithm=algorithm, field=0.11, trace_interval=128,
    )
    pair_q2 = (states.reshape(512, 9) @ states.reshape(512, 9).T / 9.0) ** 2
    for temperature, T in enumerate(temperatures):
        weights = np.exp(-(exact_energy - exact_energy.min()) / T)
        weights /= weights.sum()
        expected_energy = weights @ exact_energy
        expected_q2 = weights @ pair_q2 @ weights
        assert result["energies"][temperature].mean() == pytest.approx(expected_energy, abs=0.045)
        samples = result["configurations"][temperature].reshape(4, 4096, 9)
        observed_q2 = np.mean((np.sum(samples[0] * samples[1], axis=1) / 9.0) ** 2)
        assert observed_q2 == pytest.approx(expected_q2, abs=0.012)


@pytest.mark.parametrize("algorithm", ["pt", "houdayer_pt"])
def test_zero_energy_lazy_kernel_and_odd_replica_matching_remain_ergodic(algorithm):
    result = sample_houdayer_parallel_tempering(
        np.zeros((2, 3, 3)), [0.7, 1.4, 2.8], samples=512, burn_in=0,
        spacing=2, seed=98, replicas=3, algorithm=algorithm,
    )
    np.testing.assert_array_equal(result["local_accepts"], result["local_attempts"])
    np.testing.assert_array_equal(result["swap_accepts"], result["swap_attempts"])
    np.testing.assert_array_equal(result["energies"], np.zeros((3, 3, 512)))
    assert np.all(result["round_trips"] > 0)
    assert np.max(np.abs(result["configurations"].mean(axis=(1, 2)))) < 0.12
    if algorithm == "houdayer_pt":
        np.testing.assert_array_equal(result["cluster_attempts"], np.full(3, 1024))
        assert np.all(result["cluster_size_max"] == 1)


@pytest.mark.parametrize("change", [
    {"algorithm": "other"}, {"initial": "mixed"}, {"replicas": 1},
    {"samples": True}, {"burn_in": -1}, {"spacing": 0},
    {"cluster_interval": 0}, {"trace_interval": 0}, {"seed": -1},
    {"field": float("nan")}, {"max_output_bytes": 1},
    {"samples": 2**63}, {"samples": 2**62, "spacing": 8},
])
def test_invalid_settings_fail_before_sampling(change):
    options = dict(samples=2, burn_in=0, spacing=1, seed=9)
    options.update(change)
    with pytest.raises(ValueError):
        sample_houdayer_parallel_tempering(np.ones((2, 3, 3)), [1.0, 2.0], **options)


@pytest.mark.parametrize("temperatures", [[1.0], [2.0, 1.0], [0.0, 2.0], [1.0, np.inf], [[1.0, 2.0]]])
def test_invalid_temperatures(temperatures):
    with pytest.raises(ValueError):
        sample_houdayer_parallel_tempering(
            np.ones((2, 3, 3)), temperatures, samples=2, burn_in=0, spacing=1, seed=9,
        )


def test_rejects_unsupported_geometry_and_bad_spins():
    with pytest.raises(ValueError, match="2D"):
        sample_houdayer_parallel_tempering(np.ones((1, 3)), [1., 2.], samples=1, burn_in=0, spacing=1, seed=0)
    with pytest.raises(ValueError, match="shape"):
        houdayer_cluster_update(np.ones(9), np.ones((3, 3)), np.ones((2, 3, 3)), seed=0)
    with pytest.raises(ValueError, match="-1 and \\+1"):
        houdayer_cluster_update(np.zeros((3, 3)), np.ones((3, 3)), np.ones((2, 3, 3)), seed=0)
