"""Independent target-law, PT sign, tracking and finite-array sampler tests."""

import itertools

import numpy as np
import pytest

from recurrent_rg.data.ising_j1j2_pt import (
    configuration_energy,
    energy_components,
    local_delta_energy,
    proposal_log_swap_ratio,
    sample_ladder,
    seed_for_group,
    series_diagnostics,
)


def test_energy_matches_explicit_undirected_bonds_and_global_sign_symmetry():
    rng = np.random.default_rng(902)
    spins = rng.choice([-1, 1], size=(5, 6, 6)).astype(np.int8)
    original = spins.copy()
    expected = []
    for state in spins:
        value = 0.0
        for x, y in itertools.product(range(6), repeat=2):
            value -= state[x, y] * (state[(x + 1) % 6, y] + state[x, (y + 1) % 6])
            value += .7 * state[x, y] * (
                state[(x + 1) % 6, (y + 1) % 6] + state[(x + 1) % 6, (y - 1) % 6]
            )
        expected.append(value)
    np.testing.assert_allclose(configuration_energy(spins, .7), expected, atol=1e-13)
    np.testing.assert_array_equal(configuration_energy(spins, .7), configuration_energy(-spins, .7))
    np.testing.assert_array_equal(spins, original)
    nn, _ = energy_components(spins)
    np.testing.assert_array_equal(configuration_energy(spins, 0), nn)


@pytest.mark.parametrize("rho", [0, .3, .5, 1.0, 2.3])
def test_all_local_delta_energies_equal_independent_total_recomputation(rho):
    rng = np.random.default_rng(2026)
    state = rng.choice([-1, 1], size=(6, 6)).astype(np.int8)
    before = configuration_energy(state, rho)
    for x, y in itertools.product(range(6), repeat=2):
        flipped = state.copy()
        flipped[x, y] *= -1
        assert local_delta_energy(state, x, y, rho) == pytest.approx(
            configuration_energy(flipped, rho) - before, abs=3e-14,
        )


def test_ferro_and_both_stripe_ground_state_energies_and_odd_block_periodicity():
    size = 18
    ferro = np.ones((size, size), dtype=np.int8)
    stripe_x = ferro * (1 - 2 * (np.arange(size) % 2))[:, None]
    stripe_y = stripe_x.T
    for rho in [.2, .5, 1.0]:
        assert configuration_energy(ferro, rho) == pytest.approx(size**2 * (-2 + 2 * rho))
        assert configuration_energy(stripe_x, rho) == pytest.approx(-2 * rho * size**2)
        assert configuration_energy(stripe_y, rho) == pytest.approx(-2 * rho * size**2)
    # Both majority and a fixed decimation example preserve period two for b=3.
    for state in (stripe_x, stripe_y):
        blocks = state.reshape(6, 3, 6, 3)
        majority = np.sign(blocks.sum(axis=(1, 3)))
        expected = state[::3, ::3]
        np.testing.assert_array_equal(majority, expected)
        assert set(np.unique(expected)) == {-1, 1}


def test_swap_sign_is_direct_boltzmann_ratio_and_cold_low_energy_swap_is_unfavorable():
    temperature_a, temperature_b = .7, 2.1
    energy_a, energy_b = -20., -4.
    before = -energy_a / temperature_a - energy_b / temperature_b
    after = -energy_b / temperature_a - energy_a / temperature_b
    result = proposal_log_swap_ratio(temperature_a, temperature_b, energy_a, energy_b)
    assert result == pytest.approx(after - before)
    assert result < 0
    assert proposal_log_swap_ratio(temperature_a, temperature_b, energy_b, energy_a) == -result


def test_seed_namespace_separates_size_rho_purpose_and_group():
    baseline = seed_for_group(91, 108, .7, "production", 0)
    assert baseline == seed_for_group(91, 108, .7, "production", 0)
    alternatives = [
        seed_for_group(92, 108, .7, "production", 0),
        seed_for_group(91, 216, .7, "production", 0),
        seed_for_group(91, 108, .8, "production", 0),
        seed_for_group(91, 108, .7, "pilot", 0),
        seed_for_group(91, 108, .7, "production", 1),
    ]
    assert len(set([baseline] + alternatives)) == 6


@pytest.mark.parametrize("initial", ["hot", "ferro", "stripe_x", "stripe_y"])
def test_sampler_reproducible_bounded_traces_and_exact_energy_tracking(initial):
    temperatures = np.array([.7, 1.4, 3.])
    original = temperatures.copy()
    options = dict(size=6, rho=.7, temperatures=temperatures, seed=229,
                   burn_in=13, samples=12, spacing=3, trace_interval=2,
                   max_trace_points=9, initial=initial)
    first = sample_ladder(**options)
    second = sample_ladder(**options)
    for key in ["configs", "energies", "energy_nn", "energy_diagonal", "round_trips",
                "swap_attempts", "swap_accepts", "temperature_visits"]:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(temperatures, original)
    assert first["configs"].shape == (3, 12, 6, 6)
    assert first["configs"].dtype == np.int8
    assert set(np.unique(first["configs"])) == {-1, 1}
    assert first["max_component_tracking_drift"] == 0
    assert first["max_energy_tracking_drift"] == 0
    np.testing.assert_array_equal(first["energies"], configuration_energy(first["configs"], .7))
    raw = first["configs"]
    np.testing.assert_array_equal(first["magnetization"], raw.mean(axis=(-2, -1)))
    sign = 1 - 2 * (np.arange(6) % 2)
    np.testing.assert_array_equal(first["stripe_x"], (raw * sign[None, None, :, None]).mean(axis=(-2, -1)))
    np.testing.assert_array_equal(first["stripe_y"], (raw * sign[None, None, None, :]).mean(axis=(-2, -1)))
    assert first["trace"]["sweeps"][0] == 0
    assert first["trace"]["sweeps"][-1] == 49
    assert len(first["trace"]["sweeps"]) <= 9
    assert first["total_sweeps"] == 147
    assert first["visited_sites"] == 147 * 36
    assert np.all(first["local_accepts"] <= first["local_attempts"])
    assert np.all(first["swap_accepts"] <= first["swap_attempts"])
    assert not first["equilibration_certified"]
    assert first["independent_unit"] == "whole_temperature_ladder_group"
    assert first["cpu_seconds"] > 0
    assert first["wall_seconds"] > 0
    np.testing.assert_array_equal(first["temperature_visits"].sum(1), [50] * 3)
    for labels in first["trace"]["temperature_to_replica"].T:
        np.testing.assert_array_equal(np.sort(labels), np.arange(3))


def test_python_fallback_matches_compiled_sampler_without_global_rng_changes():
    options = dict(size=4, rho=.8, temperatures=[.7, 2.], seed=98,
                   burn_in=7, samples=5, spacing=3)
    np.random.seed(12345)
    expected = np.random.random(8)
    np.random.seed(12345)
    compiled = sample_ladder(**options)
    fallback = sample_ladder(**options, use_numba=False)
    np.testing.assert_array_equal(np.random.random(8), expected)
    np.testing.assert_array_equal(compiled["configs"], fallback["configs"])
    np.testing.assert_array_equal(compiled["swap_accepts"], fallback["swap_accepts"])


@pytest.mark.parametrize("rho", [0., .3, 1.])
def test_small_lattice_pt_statistics_match_exact_enumeration(rho):
    # 4x4 has no repeated torus neighbors; enumerate all 2^16 configurations.
    indices = np.arange(2**16, dtype=np.uint32)
    states = (2 * ((indices[:, None] >> np.arange(16)) & 1).astype(np.int8) - 1).reshape(-1, 4, 4)
    energies = configuration_energy(states, rho)
    magnetization2 = states.mean(axis=(1, 2)) ** 2
    temperatures = np.array([1.3, 2.2, 4.0])
    sampled = sample_ladder(4, rho, temperatures, seed=135,
                            burn_in=1024, samples=8192, spacing=4, trace_interval=64)
    for t, temperature in enumerate(temperatures):
        log_weight = -energies / temperature
        weight = np.exp(log_weight - log_weight.max())
        weight /= weight.sum()
        assert sampled["energies"][t].mean() == pytest.approx(weight @ energies, abs=.3)
        assert np.mean(sampled["magnetization"][t] ** 2) == pytest.approx(weight @ magnetization2, abs=.025)
    assert np.all(sampled["round_trips"] > 0)


def test_single_temperature_and_no_burn_are_supported_without_fabricated_roundtrips():
    sampled = sample_ladder(4, .2, [2.], 8, burn_in=0, samples=5, spacing=2)
    assert sampled["configs"].shape == (1, 5, 4, 4)
    assert sampled["swap_attempts"].size == 0
    np.testing.assert_array_equal(sampled["round_trips"], [0])
    assert not sampled["equilibration_certified"]


def test_finite_series_diagnostics_do_not_call_constant_traces_independent_samples():
    constant = series_diagnostics(np.ones(128))
    assert constant["ess"] is None
    assert constant["iat"] is None
    rng = np.random.default_rng(24)
    iid = series_diagnostics(rng.normal(size=8192))
    assert 4096 < iid["ess"] <= 8192
    assert -0.05 < iid["lag1"] < .05
    correlated = np.empty(8192)
    correlated[0] = 0
    for index in range(1, correlated.size):
        correlated[index] = .9 * correlated[index - 1] + rng.normal()
    assert series_diagnostics(correlated)["ess"] < iid["ess"] / 5


@pytest.mark.parametrize("changes", [
    {"size": 3}, {"size": 5}, {"size": True}, {"rho": -.1}, {"rho": np.nan},
    {"temperatures": [1., 1.]}, {"temperatures": [2., 1.]},
    {"temperatures": [0., 1.]}, {"temperatures": [np.inf]},
    {"temperatures": [1e-320]}, {"rho": 1e308},
    {"seed": -1}, {"burn_in": -1}, {"samples": True}, {"samples": 0},
    {"spacing": 0}, {"initial": "cold"}, {"trace_interval": 0},
    {"swap_interval": 0}, {"max_trace_points": 1}, {"max_output_bytes": 1},
])
def test_invalid_settings_rejected_before_kernel(changes):
    options = dict(size=4, rho=.7, temperatures=[1., 2.], seed=7,
                   burn_in=0, samples=2, spacing=1)
    options.update(changes)
    with pytest.raises(ValueError):
        sample_ladder(**options)
