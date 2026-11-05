"""CPU-only tiny tests; no production data, model inference, training, or SSH."""
import itertools
import math

import numpy as np
import pytest

from recurrent_rg.data import phi4 as base
from recurrent_rg.data import phi4_hybrid as hybrid


class FixedRNG:
    def __init__(self, seed_site=0, values=()):
        self.seed_site = seed_site
        self.values = iter(values)
        self.calls = 0
    def integers(self, high):
        assert 0 <= self.seed_site < high
        return self.seed_site
    def random(self):
        self.calls += 1
        return next(self.values)


def test_edge_probability_matches_once_per_edge_action_and_alignment():
    left, right = .7, 1.3
    expected = 1 - np.exp(-2 * left * right)
    assert hybrid.bond_probability(left, right) == pytest.approx(expected, abs=1e-15)
    assert hybrid.bond_probability(-left, -right) == pytest.approx(expected, abs=1e-15)
    assert hybrid.bond_probability(left, -right) == 0.
    assert hybrid.bond_probability(0., right) == 0.
    assert hybrid.bond_probability(1e-12, 1e-12) == pytest.approx(2e-24, rel=1e-15)
    assert hybrid.bond_probability(100., 100.) == 1.


def test_actual_growth_uses_exact_threshold_not_doubled_coupling():
    phi = np.array([1., .3, -.5])
    probability = -np.expm1(-.6)
    joined, a = hybrid.cluster_step(phi, FixedRNG(values=[probability - 1e-10]))
    rejected, b = hybrid.cluster_step(phi, FixedRNG(values=[probability + 1e-10]))
    np.testing.assert_array_equal(joined, [-1., -.3, -.5])
    np.testing.assert_array_equal(rejected, [-1., .3, -.5])
    assert a["cluster_size"] == 2 and b["cluster_size"] == 1
    assert a["bond_tests"] == b["bond_tests"] == 1
    assert a["neighbor_visits"] == 4 and b["neighbor_visits"] == 2


@pytest.mark.parametrize("shape", [(3,), (4, 3), (3, 3, 3)])
def test_cluster_preserves_all_amplitudes_and_local_potential_without_input_mutation(shape):
    rng = np.random.default_rng(773)
    phi = rng.normal(size=shape)
    before = phi.copy()
    updated, receipt = hybrid.cluster_step(phi, rng)
    np.testing.assert_array_equal(phi, before)
    assert not np.shares_memory(updated, phi)
    np.testing.assert_array_equal(np.abs(updated), np.abs(before))
    np.testing.assert_array_equal(updated ** 2, before ** 2)
    for r, u in ((-.5, .2), (.3, .2), (.8, 0.)):
        old_local = .5 * r * before ** 2 + .25 * u * before ** 4
        new_local = .5 * r * updated ** 2 + .25 * u * updated ** 4
        # Some NumPy/libm SIMD paths round pow(+x,4) and pow(-x,4)
        # differently by one ULP. Amplitudes and squares above must remain
        # bitwise exact; the derived polynomial energy permits roundoff only.
        np.testing.assert_allclose(new_local, old_local,
            rtol=8 * np.finfo(np.float64).eps, atol=0.)
    assert 1 <= receipt["cluster_size"] <= phi.size
    assert receipt["changed_sites"] == receipt["cluster_size"]
    assert receipt["neighbor_visits"] == receipt["cluster_size"] * 2 * len(shape)
    assert receipt["activated_bonds"] == receipt["cluster_size"] - 1
    assert receipt["bond_tests"] == receipt["bond_random_draws"]
    assert not receipt["amplitude_changed"]


def test_zero_amplitudes_do_not_create_fake_bonds_or_fake_changed_sites():
    phi = np.zeros(3)
    rng = FixedRNG(seed_site=1)
    new, receipt = hybrid.cluster_step(phi, rng)
    np.testing.assert_array_equal(new, phi)
    assert receipt["cluster_size"] == 1 and receipt["changed_sites"] == 0
    assert receipt["bond_tests"] == rng.calls == 0


def _state_index(signs):
    return sum((int(sign) > 0) << index for index, sign in enumerate(signs))


def test_actual_tiny_ring_kernel_matches_fortuin_kasteleyn_enumeration_and_detailed_balance():
    # Enumerate the *actual* implementation's independent random draws using a
    # common partition at every possible edge probability. At most three
    # eligible edge tests occur on this size-3 periodic ring.
    amplitude = np.array([.35, .8, 1.1])
    edges = ((0, 1), (1, 2), (2, 0))
    probabilities = [1 - math.exp(-2 * amplitude[a] * amplitude[b]) for a, b in edges]
    cuts = sorted([0., *probabilities, 1.])
    intervals = [(float((a + b) / 2), float(b - a)) for a, b in zip(cuts[:-1], cuts[1:])]
    actual, reference = np.zeros((8, 8)), np.zeros((8, 8))
    weights = np.empty(8)
    for index in range(8):
        signs = np.array([1 if index & (1 << j) else -1 for j in range(3)])
        field = amplitude * signs
        weights[index] = math.exp(sum(field[a] * field[b] for a, b in edges))
        for seed in range(3):
            for draws in itertools.product(intervals, repeat=3):
                rng = FixedRNG(seed, [entry[0] for entry in draws])
                updated, _ = hybrid.cluster_step(field, rng)
                probability = math.prod(entry[1] for entry in draws) / 3
                actual[index, _state_index(np.sign(updated))] += probability
            edge_p = [p if signs[a] == signs[b] else 0.
                      for p, (a, b) in zip(probabilities, edges)]
            for occupied in itertools.product((False, True), repeat=3):
                probability = math.prod(p if yes else 1 - p for p, yes in zip(edge_p, occupied)) / 3
                connected = {seed}
                while True:
                    previous = len(connected)
                    for (a, b), yes in zip(edges, occupied):
                        if yes and (a in connected or b in connected):
                            connected.update((a, b))
                    if previous == len(connected):
                        break
                target = signs.copy()
                target[list(connected)] *= -1
                reference[index, _state_index(target)] += probability
    np.testing.assert_allclose(actual.sum(axis=1), 1., atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual, reference, atol=2e-15, rtol=2e-14)
    stationary = weights / weights.sum()
    flux = stationary[:, None] * actual
    np.testing.assert_allclose(flux, flux.T, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(stationary @ actual, stationary, atol=2e-15, rtol=0)


def _kwargs(**changes):
    values = dict(size=3, dimension=1, samples=8, burn_in=4, spacing=2,
                  step_size=.08, leapfrog_steps=4, seed=971, initial="hot")
    values.update(changes)
    return values


@pytest.mark.parametrize("initial", ["hot", "cold_positive", "cold_negative"])
def test_hybrid_records_raw_time_series_and_is_reproducible(initial):
    kwargs = _kwargs(initial=initial)
    first = hybrid.sample_hybrid_chain(-.3, .2, **kwargs)
    second = hybrid.sample_hybrid_chain(-.3, .2, **kwargs)
    names = ("configurations", "raw_observables", "accepted", "delta_h", "cluster_sizes",
             "cluster_changed_sites", "cluster_neighbor_visits", "cluster_bond_tests",
             "cluster_activated_bonds", "cluster_attempted", "trajectory_magnetization")
    for name in names:
        np.testing.assert_array_equal(first[name], second[name])
        assert np.isfinite(first[name]).all()
    assert first["configurations"].shape == (8, 3, 1)
    assert first["configurations"].dtype == np.float64
    assert first["raw_observables"].shape == (8, len(base.OBSERVABLES))
    assert first["raw_observable_names"] == list(base.OBSERVABLES)
    assert first["accepted"].shape == (20,)
    assert first["trajectory_magnetization"].shape == (20,)
    assert first["cluster_probability"] == .5
    np.testing.assert_array_equal(first["raw_observables"][:, 1], first["trajectory_magnetization"][5::2])
    assert first["schema"] == hybrid.SCHEMA
    assert first["artificial_global_sign_flips"] == 0
    assert not first["production_adaptation"] and not first["equilibrium_certified"]
    assert not first["symmetry_augmentation"]


def test_true_work_counts_match_actual_force_calls_and_cluster_rng_operations(monkeypatch):
    gradient, cluster = base._gradient, hybrid.cluster_step
    calls, receipts = [], []
    def counted_gradient(phi, *args):
        calls.append(phi.size)
        return gradient(phi, *args)
    def counted_cluster(phi, rng):
        value, receipt = cluster(phi, rng)
        receipts.append(receipt)
        return value, receipt
    monkeypatch.setattr(base, "_gradient", counted_gradient)
    monkeypatch.setattr(hybrid, "cluster_step", counted_cluster)
    result = hybrid.sample_hybrid_chain(.3, .2, **_kwargs(dimension=2), cluster_probability=1.)
    assert result["hybrid_cycles"] == result["hmc_trajectories"] == result["wolff_cluster_updates"] == 20
    assert len(receipts) == 20
    assert len(calls) == result["force_evaluations"] == 20 * 5
    assert sum(calls) == result["force_site_evaluations"] == 20 * 5 * 9
    for array_name, receipt_name, total_name in (
        ("cluster_sizes", "cluster_size", "total_cluster_sites"),
        ("cluster_changed_sites", "changed_sites", "total_cluster_changed_sites"),
        ("cluster_neighbor_visits", "neighbor_visits", "total_cluster_neighbor_visits"),
        ("cluster_bond_tests", "bond_tests", "total_cluster_bond_tests"),
        ("cluster_activated_bonds", "activated_bonds", "total_cluster_activated_bonds")):
        np.testing.assert_array_equal(result[array_name], [r[receipt_name] for r in receipts])
        assert result[total_name] == sum(r[receipt_name] for r in receipts)
    assert result["mean_production_cluster_fraction"] == pytest.approx(result["cluster_sizes"][4:].mean() / 9)


def test_hmc_rejections_are_not_retried_or_skipped_and_cluster_still_executes(monkeypatch):
    calls = []
    def bad_proposal(phi, momentum, *args):
        calls.append(1)
        return phi + 100., momentum.copy()
    monkeypatch.setattr(base, "_leapfrog", bad_proposal)
    result = hybrid.sample_hybrid_chain(.3, .2, **_kwargs(initial="cold_positive"), cluster_probability=1.)
    assert len(calls) == 20
    assert not result["accepted"].any()
    assert result["production_acceptance"] == 0.
    assert result["wolff_cluster_updates"] == 20
    np.testing.assert_array_equal(np.abs(result["configurations"]), 1.)
    np.testing.assert_array_equal(result["cluster_activated_bonds"], result["cluster_sizes"] - 1)
    assert any("do not repair amplitude rejection" in warning for warning in result["warnings"])


def test_lazy_default_counts_only_actual_attempts_and_records_identity_cycles(monkeypatch):
    cluster = hybrid.cluster_step
    actual_calls = []
    def tracked(phi, rng):
        actual_calls.append(1)
        return cluster(phi, rng)
    monkeypatch.setattr(hybrid, "cluster_step", tracked)
    result = hybrid.sample_hybrid_chain(.3, .2, **_kwargs(samples=64))
    attempted = result["cluster_attempted"]
    assert attempted.dtype == np.bool_
    assert attempted.any() and (~attempted).any()
    assert len(actual_calls) == result["wolff_cluster_updates"] == int(attempted.sum())
    assert result["hybrid_cycles"] == 4 + 64 * 2
    for name in ("cluster_sizes", "cluster_changed_sites", "cluster_neighbor_visits", "cluster_bond_tests",
                 "cluster_activated_bonds"):
        assert np.all(result[name][~attempted] == 0)
    np.testing.assert_array_equal(result["cluster_activated_bonds"][attempted],
                                  result["cluster_sizes"][attempted] - 1)


def test_explicit_zero_cluster_probability_is_bitwise_hmc_only_control():
    kwargs = _kwargs(samples=16)
    control = hybrid.sample_hybrid_chain(.3, .2, **kwargs, cluster_probability=0.)
    reference = base.sample_hmc_chain(.3, .2, **kwargs)
    for name in ("configurations", "raw_observables", "accepted", "delta_h"):
        np.testing.assert_array_equal(control[name], reference[name])
    assert control["hmc_only_control"]
    assert control["wolff_cluster_updates"] == 0
    assert control["total_cluster_sites"] == 0
    assert not control["cluster_attempted"].any()


def test_lazy_conditional_kernel_keeps_sign_stationarity_and_removes_period_two():
    # In the full-cluster limit W swaps the two global signs. Half identity
    # yields all eigenvalues (1,0), rather than the periodic eigenvalue -1.
    full_cluster_kernel = np.array([[0., 1.], [1., 0.]])
    lazy_kernel = .5 * np.eye(2) + .5 * full_cluster_kernel
    np.testing.assert_allclose(np.linalg.eigvalsh(lazy_kernel), [0., 1.], atol=1e-15)
    np.testing.assert_allclose(np.array([.5, .5]) @ lazy_kernel, [.5, .5], atol=1e-15)


def test_actual_full_cluster_even_spacing_alias_is_removed_by_frozen_laziness(monkeypatch):
    # Hold the amplitudes by deliberately rejecting HMC. For rho=10 all
    # aligned-edge probabilities round to one; non-lazy W alternates sectors.
    def bad_proposal(phi, momentum, *args):
        return phi + 100., momentum.copy()
    monkeypatch.setattr(base, "_leapfrog", bad_proposal)
    kwargs = _kwargs(samples=32, initial="cold_positive")
    periodic = hybrid.sample_hybrid_chain(-.5, .005, **kwargs, cluster_probability=1.)
    lazy = hybrid.sample_hybrid_chain(-.5, .005, **kwargs, cluster_probability=.5)
    assert not periodic["accepted"].any() and not lazy["accepted"].any()
    assert np.unique(np.sign(periodic["raw_observables"][:, 1])).size == 1
    assert np.unique(np.sign(lazy["raw_observables"][:, 1])).size == 2
    # Laziness repairs this sign-period alias, not rejected amplitude updates.
    np.testing.assert_array_equal(np.abs(lazy["configurations"]), 10.)


@pytest.mark.parametrize("dimension", [1, 2])
def test_small_gaussian_hybrid_recovers_dense_periodic_covariance(dimension):
    size, r = 3, .7
    shape = (size,) * dimension
    volume = math.prod(shape)
    precision = np.eye(volume) * r
    for source in np.ndindex(shape):
        i = np.ravel_multi_index(source, shape)
        for axis in range(dimension):
            neighbor = list(source)
            neighbor[axis] = (neighbor[axis] + 1) % size
            j = np.ravel_multi_index(tuple(neighbor), shape)
            edge = np.zeros(volume)
            edge[i], edge[j] = 1., -1.
            precision += np.outer(edge, edge)
    result = hybrid.sample_hybrid_chain(r, 0., size=size, dimension=dimension, samples=6000,
        burn_in=128, spacing=1, step_size=.12, leapfrog_steps=8, seed=10612 + dimension)
    flat = result["configurations"][..., 0].reshape(6000, volume)
    np.testing.assert_allclose(flat.mean(axis=0), 0., atol=.05, rtol=0)
    np.testing.assert_allclose(np.cov(flat, rowvar=False, ddof=0), np.linalg.inv(precision),
                               atol=.035, rtol=.10)
    assert result["production_acceptance"] > .8


@pytest.mark.parametrize("changes", [
    {"r": 0., "u": 0.}, {"r": -.2, "u": 0.}, {"u": -.1}, {"r": float("nan")},
    {"h": .1}, {"h": float("nan")}, {"size": 2}, {"dimension": 0}, {"dimension": 4},
    {"samples": 3}, {"burn_in": -1}, {"spacing": 0}, {"step_size": 0.},
    {"step_size": float("nan")}, {"leapfrog_steps": 0}, {"seed": -1}, {"initial": "wrong"},
    {"cluster_probability": -.1}, {"cluster_probability": 1.1},
    {"cluster_probability": float("nan")}, {"cluster_probability": True},
])
def test_invalid_hybrid_parameters_reject_before_rng_creation(monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid parameters must fail before RNG use")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    values = dict(r=.3, u=.2, **_kwargs())
    values.update(changes)
    with pytest.raises(ValueError):
        hybrid.sample_hybrid_chain(**values)


@pytest.mark.parametrize("phi", [np.ones(2), np.ones((3, 3, 3, 3)), np.array([1., np.nan, 2.]),
                                  np.ones(3, dtype=complex)])
def test_bad_cluster_field_rejects_without_rng_use(phi):
    with pytest.raises(ValueError):
        hybrid.cluster_step(phi, None)
