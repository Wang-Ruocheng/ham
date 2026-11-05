"""Independent energy, symmetry, periodic-geometry and target-law tests."""

import numpy as np
import pytest

from recurrent_rg.data.spin_glass import energy, generate_bonds, make_periodic_lattice
from recurrent_rg.data.spin_glass3d import icm_cluster_update, sample_icm_parallel_tempering


def test_3d_lattice_has_six_periodic_neighbors_and_full_energy_matches_explicit_rolls():
    shape = (3, 4, 5)
    forward, backward = make_periodic_lattice(shape)
    indices = np.arange(np.prod(shape)).reshape(shape)
    assert forward.shape == backward.shape == (3, 60)
    for axis in range(3):
        np.testing.assert_array_equal(forward[axis], np.roll(indices, -1, axis).ravel())
        np.testing.assert_array_equal(backward[axis], np.roll(indices, 1, axis).ravel())
    rng = np.random.default_rng(551)
    spins = rng.choice([-1, 1], size=(6, *shape)).astype(np.int8)
    bonds = rng.normal(size=(3, *shape))
    explicit = np.zeros(6)
    for axis in range(3):
        explicit -= np.sum(bonds[axis] * spins * np.roll(spins, -1, axis + 1), axis=(1, 2, 3))
    np.testing.assert_allclose(energy(spins, bonds), explicit, atol=2e-14)


@pytest.mark.parametrize("count", [0, 1, 7, 13, 14, 20, 26, 27])
def test_minority_and_majority_cluster_are_energy_preserving_involutions_without_folding_q(count):
    rng = np.random.default_rng(351 + count)
    bonds = rng.normal(size=(3, 3, 3, 3))
    a = rng.choice([-1, 1], size=(3, 3, 3)).astype(np.int8)
    b = a.copy()
    b.ravel()[rng.permutation(27)[:count]] *= -1
    original = b.copy()
    for seed in range(8):
        first = icm_cluster_update(a, b, bonds, seed=seed)
        np.testing.assert_array_equal(first["spins_a"] * first["spins_b"], a * b)
        assert first["raw_disagreement_count"] == count
        assert first["minority_disagreement_count"] == min(count, 27 - count)
        assert first["gauge_flip_used"] == (count > 13)
        assert first["gauge_flip_undone"]
        assert first["cluster_size"] <= min(count, 27 - count)
        assert abs(first["joint_energy_residual"]) < 6e-14
        second = icm_cluster_update(first["spins_a"], first["spins_b"], bonds, seed=seed)
        np.testing.assert_array_equal(second["spins_a"], a)
        np.testing.assert_array_equal(second["spins_b"], b)
        np.testing.assert_array_equal(second["cluster_indices"], first["cluster_indices"])
    np.testing.assert_array_equal(b, original)


def test_periodic_complete_cluster_zero_bonds_and_true_winding_are_distinguished():
    a = np.ones((4, 4, 4), dtype=np.int8)
    b = a.copy()
    b[0, 0, 0] = b[3, 0, 0] = -1
    bonds = np.ones((3, 4, 4, 4))
    result = icm_cluster_update(a, b, bonds, seed=0)
    assert set(result["cluster_indices"]) == {0, 48}
    assert not result["winding_any"]  # a seam edge is not a torus cycle
    b[:, 0, 0] = -1
    result = icm_cluster_update(a, b, bonds, seed=0)
    assert result["cluster_size"] == 4
    np.testing.assert_array_equal(result["winding_axes"], [True, False, False])
    bonds[:] = 0
    result = icm_cluster_update(a, b, bonds, seed=0)
    assert result["cluster_size"] == 1
    assert not result["winding_any"]
    assert result["joint_energy_residual"] == 0


@pytest.mark.parametrize("algorithm", ["pt", "icm_pt"])
def test_sampler_reproducible_energy_audited_and_temperature_mask_is_fixed(algorithm):
    bonds = generate_bonds((3, 4, 3), 553, distribution="gaussian", scale=.4)
    options = dict(samples=7, burn_in=5, spacing=2, seed=99, replicas=4,
                   algorithm=algorithm, exchange_interval=2, cluster_interval=3,
                   trace_interval=4, icm_max_temperature=1.0)
    first = sample_icm_parallel_tempering(bonds, [.5, 1., 2.], **options)
    second = sample_icm_parallel_tempering(bonds, [.5, 1., 2.], **options)
    for key, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second[key])
    assert first["configurations"].shape == (3, 4, 7, 3, 4, 3)
    assert first["configurations"].dtype == np.int8
    np.testing.assert_allclose(first["energies"], energy(first["configurations"], bonds), atol=2e-13)
    assert first["max_energy_tracking_drift"] < 2e-13
    assert first["max_cluster_joint_energy_residual"] < 2e-13
    np.testing.assert_array_equal(first["cluster_attempts"], [12, 12, 0] if algorithm == "icm_pt" else [0, 0, 0])
    np.testing.assert_array_equal(first["cluster_size_histogram"].sum(1), first["cluster_attempts"])
    np.testing.assert_array_equal(first["trace_sweeps"], [0, 4, 8, 12, 16, 19])
    assert np.all(first["minority_disagreement_sum"] <= first["raw_disagreement_sum"])
    assert not first["equilibration_certified"]
    np.testing.assert_array_equal(first["temperature_visits"].sum(1), np.full((4, 3), 20))


@pytest.mark.parametrize("algorithm", ["pt", "icm_pt"])
def test_3d_isolated_bond_matches_exact_boltzmann_and_q2_including_pt_sign(algorithm):
    # Exactly one coupled pair plus 25 free spins: E=-J*tanh(J/T),
    # E[q^2]=(N+2*tanh(J/T)^2)/N^2 for two independent equilibrium copies.
    bonds = np.zeros((3, 3, 3, 3))
    bonds[0, 0, 0, 0] = .7
    temperatures = np.array([.4, .9, 2.2])
    result = sample_icm_parallel_tempering(
        bonds, temperatures, samples=8192, burn_in=256, spacing=1,
        seed=137, replicas=4, algorithm=algorithm, trace_interval=512,
    )
    for t, T in enumerate(temperatures):
        assert result["energies"][t].mean() == pytest.approx(-.7 * np.tanh(.7 / T), abs=.025)
        spins = result["configurations"][t].reshape(4, 8192, 27)
        q = np.mean(spins[0] * spins[1], axis=1)
        assert q.mean() == pytest.approx(0.0, abs=.015)
        assert np.mean(q * q) == pytest.approx((27 + 2 * np.tanh(.7 / T) ** 2) / 27**2, abs=.004)


def test_zero_bond_lazy_pt_accepts_all_and_has_unfolded_overlap_signs():
    result = sample_icm_parallel_tempering(np.zeros((3, 3, 3, 3)), [.7, 1., 2.],
        samples=512, burn_in=0, spacing=2, seed=61, replicas=3)
    np.testing.assert_array_equal(result["local_accepts"], result["local_attempts"])
    np.testing.assert_array_equal(result["swap_accepts"], result["swap_attempts"])
    assert np.all(result["energies"] == 0)
    assert np.all(result["round_trips"] > 0)
    assert np.all(result["cluster_size_max"] == [1, 1, 0])
    spins = result["configurations"].reshape(3, 3, 512, 27)
    q = np.mean(spins[:, 0] * spins[:, 1], axis=-1)
    assert np.all((q < 0).any(1)) and np.all((q > 0).any(1))


@pytest.mark.parametrize("changes", [
    {"field": .1}, {"field": float("nan")}, {"icm_max_temperature": 0},
    {"icm_max_temperature": float("nan")}, {"algorithm": "houdayer_pt"},
    {"initial": "mixed"}, {"replicas": 1}, {"samples": True},
    {"samples": 2**63}, {"max_output_bytes": 1},
])
def test_invalid_settings_rejected_before_sampling(changes):
    options = dict(samples=2, burn_in=0, spacing=1, seed=0)
    options.update(changes)
    with pytest.raises(ValueError):
        sample_icm_parallel_tempering(np.ones((3, 3, 3, 3)), [.5, 1.5], **options)


def test_global_symmetry_shortcut_rejects_field_and_non3d():
    with pytest.raises(ValueError, match="zero field"):
        icm_cluster_update(np.ones((3, 3, 3)), np.ones((3, 3, 3)), np.ones((3, 3, 3, 3)), seed=0, field=.1)
    with pytest.raises(ValueError, match="3D"):
        sample_icm_parallel_tempering(np.ones((2, 3, 3)), [.5, 1.5], samples=1, burn_in=0, spacing=1, seed=0)
