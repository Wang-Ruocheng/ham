"""Tiny, independent checks of the Edwards--Anderson sampling conventions."""

from itertools import product

import numpy as np
import pytest

from recurrent_rg.data.spin_glass import (
    energy,
    generate_bonds,
    make_periodic_lattice,
    sample_parallel_tempering,
    swap_log_acceptance,
)


def _states(shape):
    return np.asarray(list(product((-1, 1), repeat=int(np.prod(shape)))), dtype=np.int8).reshape(
        (-1, *shape)
    )


def _explicit_energy(spins, bonds, field=0.0):
    """Count each forward bond once, without using the production lattice."""
    total = -field * float(np.sum(spins))
    for site in np.ndindex(spins.shape):
        for axis, length in enumerate(spins.shape):
            neighbor = list(site)
            neighbor[axis] = (neighbor[axis] + 1) % length
            total -= bonds[(axis, *site)] * spins[site] * spins[tuple(neighbor)]
    return total


def _flip_delta(spins, bonds, site, field=0.0):
    local_field = field
    for axis, length in enumerate(spins.shape):
        following = list(site)
        preceding = list(site)
        following[axis] = (following[axis] + 1) % length
        preceding[axis] = (preceding[axis] - 1) % length
        local_field += bonds[(axis, *site)] * spins[tuple(following)]
        local_field += bonds[(axis, *preceding)] * spins[tuple(preceding)]
    return 2.0 * spins[site] * local_field


@pytest.mark.parametrize("shape", [(3,), (3, 4), (3, 3, 4), (3, 3, 3, 3)])
def test_periodic_neighbors_match_coordinates_and_are_inverses(shape):
    forward, backward = make_periodic_lattice(shape)
    sites = int(np.prod(shape))
    assert forward.shape == backward.shape == (len(shape), sites)
    assert np.issubdtype(forward.dtype, np.integer)
    assert np.issubdtype(backward.dtype, np.integer)
    for site in np.ndindex(shape):
        index = np.ravel_multi_index(site, shape)
        for axis, length in enumerate(shape):
            following = list(site)
            preceding = list(site)
            following[axis] = (following[axis] + 1) % length
            preceding[axis] = (preceding[axis] - 1) % length
            assert forward[axis, index] == np.ravel_multi_index(following, shape)
            assert backward[axis, index] == np.ravel_multi_index(preceding, shape)
            assert backward[axis, forward[axis, index]] == index
            assert forward[axis, backward[axis, index]] == index


@pytest.mark.parametrize("shape", [(3,), (3, 3), (3, 3, 3)])
def test_energy_counts_forward_bonds_once_in_one_two_and_three_dimensions(shape):
    bonds = np.ones((len(shape), *shape), dtype=np.float64)
    sites = int(np.prod(shape))
    states = np.ones((2, *shape), dtype=np.int8)
    states[1] = -1
    field = 0.25
    np.testing.assert_allclose(
        energy(states, bonds, field=field),
        [-len(shape) * sites - field * sites, -len(shape) * sites + field * sites],
        rtol=0,
        atol=1e-12,
    )


def test_frustrated_triangle_has_hand_computed_energy():
    bonds = np.array([[0.7, -1.1, 0.4]])
    spins = np.array([1, -1, 1], dtype=np.int8)
    # Products along the three stored forward bonds are (-1, -1, +1).
    assert energy(spins, bonds, field=0.2) == pytest.approx(-1.0)


@pytest.mark.parametrize("shape", [(3,), (3, 3)])
def test_energy_matches_full_enumeration_and_preserves_batch_axes(shape):
    spins = _states(shape)
    bonds = generate_bonds(shape, seed=521, distribution="gaussian", mean=0.1)
    exact = np.array([_explicit_energy(state, bonds, 0.13) for state in spins])
    np.testing.assert_allclose(energy(spins, bonds, field=0.13), exact, rtol=0, atol=2e-12)
    batched = spins.reshape((2, len(spins) // 2, *shape))
    assert energy(batched, bonds, field=0.13).shape == (2, len(spins) // 2)
    np.testing.assert_allclose(energy(batched, bonds, field=0.13).ravel(), exact, atol=2e-12)


@pytest.mark.parametrize("shape", [(3,), (3, 3), (3, 3, 3)])
def test_single_spin_delta_matches_full_energy_difference(shape):
    rng = np.random.default_rng(305)
    bonds = rng.normal(size=(len(shape), *shape))
    for _ in range(3):
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)
        before = energy(spins, bonds, field=-0.17)
        assert before == pytest.approx(_explicit_energy(spins, bonds, -0.17), abs=2e-12)
        for site in np.ndindex(shape):
            flipped = spins.copy()
            flipped[site] *= -1
            assert energy(flipped, bonds, field=-0.17) - before == pytest.approx(
                _flip_delta(spins, bonds, site, -0.17), abs=3e-12
            )


def test_single_site_metropolis_detailed_balance_for_every_triangle_transition():
    bonds = np.array([[0.7, -1.1, 0.4]])
    states = _states((3,))
    beta, field = 1.0 / 1.4, 0.13
    for spins in states:
        before = _explicit_energy(spins, bonds, field)
        for site in np.ndindex(spins.shape):
            flipped = spins.copy()
            flipped[site] *= -1
            after = _explicit_energy(flipped, bonds, field)
            delta = _flip_delta(spins, bonds, site, field)
            forward = 0.5 * min(1.0, np.exp(-beta * delta))
            backward = 0.5 * min(1.0, np.exp(beta * delta))
            assert np.exp(-beta * before) * forward == pytest.approx(
                np.exp(-beta * after) * backward, rel=2e-13
            )


def test_exchange_sign_and_detailed_balance_for_every_pair_of_triangle_states():
    assert swap_log_acceptance(1.3, 0.4, -5.0, 2.0) == pytest.approx(-6.3)
    bonds = np.array([[0.7, -1.1, 0.4]])
    energies = np.array([_explicit_energy(spins, bonds, 0.13) for spins in _states((3,))])
    beta_left, beta_right = 1.3, 0.4
    for left, right in product(energies, repeat=2):
        log_ratio = swap_log_acceptance(beta_left, beta_right, left, right)
        original_weight = np.exp(-beta_left * left - beta_right * right)
        exchanged_weight = np.exp(-beta_left * right - beta_right * left)
        assert np.exp(log_ratio) == pytest.approx(exchanged_weight / original_weight)
        forward = min(1.0, np.exp(log_ratio))
        reverse_log_ratio = swap_log_acceptance(beta_left, beta_right, right, left)
        backward = min(1.0, np.exp(reverse_log_ratio))
        assert original_weight * forward == pytest.approx(exchanged_weight * backward)


@pytest.mark.parametrize("distribution", ["bimodal", "gaussian"])
def test_bonds_have_fixed_dtype_shape_and_reproducible_disorder(distribution):
    options = dict(shape=(3, 4, 3), seed=319, distribution=distribution, scale=1.7, mean=0.2)
    first = generate_bonds(**options)
    assert first.shape == (3, 3, 4, 3)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, generate_bonds(**options))
    assert not np.array_equal(first, generate_bonds(**{**options, "seed": 320}))
    assert np.all(np.isfinite(first))
    if distribution == "bimodal":
        assert np.all(np.isclose(first, 0.2 - 1.7) | np.isclose(first, 0.2 + 1.7))


@pytest.mark.parametrize("probability, expected", [(0.0, -1.5), (1.0, 2.5)])
def test_bimodal_probability_endpoints(probability, expected):
    bonds = generate_bonds((3, 3), seed=4, scale=2.0, mean=0.5, positive_probability=probability)
    np.testing.assert_array_equal(bonds, np.full((2, 3, 3), expected))


@pytest.mark.parametrize("distribution", ["bimodal", "gaussian"])
def test_zero_scale_bonds_are_the_requested_mean(distribution):
    bonds = generate_bonds((3,), seed=4, distribution=distribution, scale=0.0, mean=-0.4)
    np.testing.assert_array_equal(bonds, np.full((1, 3), -0.4))


def test_sampler_reproducibility_shared_disorder_and_independent_real_replicas():
    bonds = generate_bonds((3, 3), seed=502)
    original_bonds = bonds.copy()
    options = dict(
        temperatures=[0.9, 1.6, 3.2], samples=12, burn_in=8, spacing=2, seed=601,
        replicas=2, exchange_interval=2, field=0.11, initial="hot",
    )
    first = sample_parallel_tempering(bonds, **options)
    second = sample_parallel_tempering(bonds, **options)
    required = {
        "configurations", "energies", "swap_attempts", "swap_accepts", "local_attempts",
        "local_accepts", "site_visits", "round_trips", "temperature_visits", "temperatures",
        "replica_seeds",
    }
    assert required <= first.keys()
    for key in required:
        np.testing.assert_array_equal(first[key], second[key])
    configurations = first["configurations"]
    assert configurations.shape == (3, 2, 12, 3, 3)
    assert configurations.dtype == np.int8
    assert set(np.unique(configurations)) <= {-1, 1}
    assert not np.array_equal(configurations[:, 0], configurations[:, 1])
    assert first["energies"].shape == (3, 2, 12)
    assert first["energies"].dtype == np.float64
    np.testing.assert_array_equal(first["temperatures"], options["temperatures"])
    assert first["replica_seeds"].shape == (2,)
    assert first["replica_seeds"].dtype == np.uint32
    assert len(np.unique(first["replica_seeds"])) == 2
    assert first["proposal_flip_probability"] == 0.5
    np.testing.assert_allclose(first["energies"], energy(configurations, bonds, field=0.11), atol=1e-10)
    np.testing.assert_array_equal(bonds, original_bonds)
    changed = sample_parallel_tempering(bonds, **{**options, "seed": 602})
    assert not np.array_equal(configurations, changed["configurations"])
    total_sweeps = options["burn_in"] + options["samples"] * options["spacing"]
    assert first["local_attempts"].shape == first["local_accepts"].shape == (2, 3)
    np.testing.assert_array_equal(first["site_visits"], np.full((2, 3), total_sweeps * 9))
    assert np.all(first["local_attempts"] > 0)
    assert np.all(first["local_attempts"] < first["site_visits"])
    assert first["swap_attempts"].shape == first["swap_accepts"].shape == (2, 2)
    for prefix in ("local", "swap"):
        assert np.all(first[f"{prefix}_accepts"] >= 0)
        assert np.all(first[f"{prefix}_accepts"] <= first[f"{prefix}_attempts"])
    assert first["round_trips"].shape == (2, 3)
    assert np.all(first["round_trips"] >= 0)
    assert first["temperature_visits"].shape == (2, 3, 3)
    np.testing.assert_array_equal(first["temperature_visits"].sum(axis=2), np.full((2, 3), total_sweeps + 1))
    np.testing.assert_array_equal(first["temperature_visits"].sum(axis=1), np.full((2, 3), total_sweeps + 1))


@pytest.mark.parametrize("shape", [(3,), (3, 3, 3)])
@pytest.mark.parametrize("replicas", [1, 2])
def test_sampler_accepts_one_and_three_dimensional_lattices(shape, replicas):
    bonds = generate_bonds(shape, seed=8)
    result = sample_parallel_tempering(
        bonds, [1.0, 2.0], samples=3, burn_in=2, spacing=1, seed=9, replicas=replicas,
    )
    assert result["configurations"].shape == (2, replicas, 3, *shape)
    np.testing.assert_allclose(result["energies"], energy(result["configurations"], bonds), atol=1e-10)


def test_zero_energy_exchanges_all_accept_and_track_walker_round_trips():
    result = sample_parallel_tempering(
        np.zeros((1, 3)), [0.7, 1.4, 2.8], samples=12, burn_in=0, spacing=1,
        seed=18, replicas=2,
    )
    np.testing.assert_array_equal(result["swap_attempts"], np.full((2, 2), 6))
    np.testing.assert_array_equal(result["swap_accepts"], result["swap_attempts"])
    assert np.all(result["round_trips"] > 0)
    np.testing.assert_array_equal(result["energies"], np.zeros((3, 2, 12)))


def test_zero_bonds_are_uniform_even_when_sampling_every_two_sweeps():
    # Flipping every visited site unconditionally would freeze each replica
    # when spacing is even. The fixed lazy proposal must remove that cycle.
    result = sample_parallel_tempering(
        np.zeros((1, 3)), [1.0, 2.0], samples=512, burn_in=0, spacing=2,
        seed=98, replicas=2,
    )
    for draws in result["configurations"]:
        state_ids = ((draws.reshape(-1, 3) + 1) // 2) @ np.array([4, 2, 1])
        probabilities = np.bincount(state_ids, minlength=8) / len(state_ids)
        assert np.max(np.abs(probabilities - 1.0 / 8)) < 0.06
    np.testing.assert_array_equal(result["local_accepts"], result["local_attempts"])
    np.testing.assert_array_equal(result["site_visits"], np.full((2, 2), 512 * 2 * 3))


def test_tiny_frustrated_triangle_matches_exact_boltzmann_distribution():
    # This is a tiny verification run, not a production data-generation job.
    bonds = np.array([[0.7, -1.05, 0.5]])
    temperatures = np.array([0.9, 1.8, 3.0])
    field = 0.17
    states = _states((3,))
    exact_energies = np.array([_explicit_energy(spins, bonds, field) for spins in states])
    result = sample_parallel_tempering(
        bonds, temperatures, samples=3000, burn_in=300, spacing=3, seed=185,
        replicas=2, field=field,
    )
    bit_weights = 2 ** np.arange(2, -1, -1)
    for temperature_index, temperature in enumerate(temperatures):
        log_weights = -exact_energies / temperature
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        draws = result["configurations"][temperature_index].reshape(-1, 3)
        state_ids = ((draws + 1) // 2) @ bit_weights
        observed = np.bincount(state_ids, minlength=8) / len(draws)
        # A broad fixed bound absorbs finite-chain correlation without masking a
        # reversed exchange exponent or a doubled/omitted bond convention.
        assert np.max(np.abs(observed - weights)) < 0.045
        assert abs(result["energies"][temperature_index].mean() - weights @ exact_energies) < 0.10


@pytest.mark.parametrize("shape", [(), (2,), (3, 2), (3.0,), (True,), (0, 3)])
def test_lattice_and_bond_generation_reject_invalid_shapes(shape):
    with pytest.raises(ValueError):
        make_periodic_lattice(shape)
    with pytest.raises(ValueError):
        generate_bonds(shape, seed=1)


@pytest.mark.parametrize(
    "options",
    [
        {"distribution": "uniform"}, {"scale": -0.1}, {"scale": np.nan},
        {"scale": np.inf}, {"mean": np.nan}, {"mean": np.inf},
        {"positive_probability": -0.1}, {"positive_probability": 1.1},
        {"positive_probability": np.nan}, {"positive_probability": np.inf},
        {"seed": -1}, {"seed": 2**32}, {"seed": 0.5}, {"seed": True},
    ],
)
def test_bond_generation_rejects_invalid_parameters(options):
    with pytest.raises(ValueError):
        generate_bonds((3,), **{"seed": 1, **options})


@pytest.mark.parametrize(
    "spins,bonds,field",
    [
        (np.array([1, 0, -1]), np.ones((1, 3)), 0.0),
        (np.array([1, np.nan, -1]), np.ones((1, 3)), 0.0),
        (np.array([1, np.inf, -1]), np.ones((1, 3)), 0.0),
        (np.ones(4), np.ones((1, 3)), 0.0),
        (np.ones(3), np.ones((2, 3)), 0.0),
        (np.ones(3), np.array([[1.0, np.nan, 1.0]]), 0.0),
        (np.ones(3), np.array([[1.0, np.inf, 1.0]]), 0.0),
        (np.ones(3), np.ones((1, 3)), np.nan),
        (np.ones(3), np.ones((1, 3)), np.inf),
        (np.ones(2), np.ones((1, 2)), 0.0),
    ],
)
def test_energy_rejects_nonfinite_malformed_or_nonising_inputs(spins, bonds, field):
    with pytest.raises(ValueError):
        energy(spins, bonds, field=field)


@pytest.mark.parametrize(
    "temperatures",
    [[], [1.0], [0.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0],
     [1.0, np.nan], [1.0, np.inf], [[1.0, 2.0]]],
)
def test_sampler_rejects_invalid_temperature_ladders(temperatures):
    with pytest.raises(ValueError):
        sample_parallel_tempering(np.ones((1, 3)), temperatures, samples=2, burn_in=0, spacing=1, seed=1)


@pytest.mark.parametrize(
    "options",
    [
        {"samples": 0}, {"samples": 1.5}, {"samples": True},
        {"burn_in": -1}, {"burn_in": 0.5}, {"burn_in": True},
        {"spacing": 0}, {"spacing": 1.5}, {"spacing": True},
        {"replicas": 0}, {"replicas": 1.5}, {"replicas": True},
        {"exchange_interval": 0}, {"exchange_interval": 1.5}, {"exchange_interval": True},
        {"exchange_interval": 10**30},
        {"seed": -1}, {"seed": 2**32}, {"seed": 1.5}, {"seed": True},
        {"field": np.nan}, {"field": np.inf}, {"initial": "unknown"},
    ],
)
def test_sampler_rejects_invalid_counts_seeds_fields_and_initialization(options):
    arguments = dict(samples=2, burn_in=0, spacing=1, seed=1)
    arguments.update(options)
    with pytest.raises(ValueError):
        sample_parallel_tempering(np.ones((1, 3)), [1.0, 2.0], **arguments)


def test_real_replica_seed_collision_is_rejected_before_sampling(monkeypatch):
    class Child:
        def generate_state(self, *args, **kwargs):
            return np.array([7], dtype=np.uint32)
    class CollidingSeeds:
        def __init__(self, *args):
            pass
        def spawn(self, count):
            return [Child() for _ in range(count)]
    monkeypatch.setattr(np.random, "SeedSequence", CollidingSeeds)
    with pytest.raises(ValueError, match="seeds collide"):
        sample_parallel_tempering(np.ones((1, 3)), [1., 2.],
            samples=2, burn_in=0, spacing=1, seed=1, replicas=2)


@pytest.mark.parametrize(
    "bonds",
    [np.ones((1, 2)), np.ones((2, 3)), np.array([[1.0, np.nan, 1.0]]), np.array([[1.0, np.inf, 1.0]])],
)
def test_sampler_rejects_invalid_bonds(bonds):
    with pytest.raises(ValueError):
        sample_parallel_tempering(bonds, [1.0, 2.0], samples=2, burn_in=0, spacing=1, seed=1)
