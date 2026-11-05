"""Tiny CPU tests of hopping conventions; no formal sampling or training."""
import json
import math

import numpy as np
import pytest

from recurrent_rg.data import phi4 as base
from recurrent_rg.data import phi4_hopping as hopping
from recurrent_rg.data import phi4_hybrid as hybrid


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_direct_hopping_action_gradient_finite_difference_and_no_input_mutation(dimension):
    phi = np.random.default_rng(774 + dimension).normal(scale=.4, size=(3,) * dimension)
    before = phi.copy()
    action, gradient = hopping.action_and_gradient(phi, .67, 1.13)
    numerical = np.empty_like(phi)
    epsilon = 1e-6
    for index in np.ndindex(phi.shape):
        plus, minus = phi.copy(), phi.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (hopping.action_and_gradient(plus, .67, 1.13)[0]
                            - hopping.action_and_gradient(minus, .67, 1.13)[0]) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=2e-8)
    np.testing.assert_array_equal(phi, before)
    assert math.isfinite(action) and not np.shares_memory(phi, gradient)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("kappa,quartic_lambda", [(.13, .2), (.68, 1.), (2.4, 3.2)])
def test_exact_gradient_map_including_action_constant_and_chain_rule(dimension, kappa, quartic_lambda):
    phi = np.random.default_rng(994 + dimension).normal(scale=.6, size=(3,) * dimension)
    receipt = hopping.gradient_parameters(kappa, quartic_lambda, dimension)
    psi = receipt["field_scale"] * phi
    physical_action, physical_gradient = hopping.action_and_gradient(phi, kappa, quartic_lambda)
    mapped_action, mapped_gradient = base.action_and_gradient(psi, receipt["r"], receipt["u"])
    assert physical_action == pytest.approx(mapped_action + quartic_lambda * phi.size, abs=5e-13)
    np.testing.assert_allclose(physical_gradient, mapped_gradient * receipt["field_scale"],
                               rtol=2e-14, atol=2e-14)
    assert receipt["physical_effective_mass_phi"] == kappa
    assert receipt["integrator_mass_psi"] == 1.
    assert receipt["physical_action_constant_density"] == quartic_lambda
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_hard_spin_action_matches_ising_with_each_forward_edge_once(dimension):
    signs = np.random.default_rng(128 + dimension).choice([-1., 1.], size=(3,) * dimension)
    kappa = .441
    bond_sum = sum(np.sum(signs * np.roll(signs, -1, axis=axis)) for axis in range(dimension))
    for quartic_lambda in (.1, 1., 100.):
        action, _ = hopping.action_and_gradient(signs, kappa, quartic_lambda)
        assert action == pytest.approx(-kappa * bond_sum + signs.size, abs=2e-14)


@pytest.mark.parametrize("kappa", [.15, .68, 1.3])
def test_mapped_cluster_probability_contains_correct_hopping_factor(kappa):
    left, right = .47, 1.21
    scale = hopping.gradient_parameters(kappa, 1., 2)["field_scale"]
    expected = -math.expm1(-2 * kappa * left * right)
    assert hybrid.bond_probability(scale * left, scale * right) == pytest.approx(expected, abs=1e-15)
    assert hybrid.bond_probability(-scale * left, -scale * right) == pytest.approx(expected, abs=1e-15)
    assert hybrid.bond_probability(scale * left, -scale * right) == 0.


def test_actual_cluster_uses_hopping_probability_threshold_not_doubled_convention():
    class ThresholdRNG:
        def __init__(self, value):
            self.value, self.calls = value, 0
        def integers(self, high):
            return 0
        def random(self):
            self.calls += 1
            assert self.calls == 1
            return self.value
    kappa = .68
    field = np.array([1., .3, -.5])
    scale = hopping.gradient_parameters(kappa, 1., 1)["field_scale"]
    probability = -math.expm1(-2 * kappa * .3)
    joined, a = hybrid.cluster_step(scale * field, ThresholdRNG(probability - 1e-10))
    rejected, b = hybrid.cluster_step(scale * field, ThresholdRNG(probability + 1e-10))
    np.testing.assert_array_equal(joined, scale * np.array([-1., -.3, -.5]))
    np.testing.assert_array_equal(rejected, scale * np.array([-1., .3, -.5]))
    assert a["cluster_size"] == 2 and b["cluster_size"] == 1
    assert a["bond_tests"] == b["bond_tests"] == 1


def _kwargs(**changes):
    values = dict(size=3, dimension=1, samples=8, burn_in=4, spacing=2,
                  step_size=.04, leapfrog_steps=3, seed=971, initial="hot")
    values.update(changes)
    return values


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("initial", ["hot", "cold_positive", "cold_negative"])
def test_tiny_chain_exact_kernel_replay_and_all_physical_units(dimension, initial):
    kwargs = _kwargs(dimension=dimension, initial=initial)
    kappa, quartic_lambda = .68, 1.
    mapping = hopping.gradient_parameters(kappa, quartic_lambda, dimension)
    kernel = hybrid.sample_hybrid_chain(mapping["r"], mapping["u"], **kwargs)
    actual = hopping.sample_chain(kappa, quartic_lambda, **kwargs)
    np.testing.assert_array_equal(actual["configurations"], kernel["configurations"] / math.sqrt(kappa))
    np.testing.assert_array_equal(actual["trajectory_magnetization"],
                                  kernel["trajectory_magnetization"] / math.sqrt(kappa))
    expected = kernel["raw_observables"].copy()
    expected[:, 0] += quartic_lambda
    expected[:, 1:3] /= math.sqrt(kappa)
    expected[:, (3, 5, 6, 7)] /= kappa
    expected[:, 4] /= kappa
    expected[:, 4] /= kappa
    np.testing.assert_array_equal(actual["raw_observables"], expected)
    for index, field in enumerate(actual["configurations"][..., 0]):
        direct, _ = hopping.action_and_gradient(field, kappa, quartic_lambda)
        assert actual["raw_observables"][index, 0] == pytest.approx(direct / field.size, abs=5e-14)
        assert actual["raw_observables"][index, 1] == pytest.approx(field.mean(), abs=1e-15)
        assert actual["raw_observables"][index, 2] == pytest.approx(abs(field.mean()), abs=1e-15)
        assert actual["raw_observables"][index, 3] == pytest.approx(field.mean() ** 2, abs=2e-15)
        assert actual["raw_observables"][index, 4] == pytest.approx(field.mean() ** 4, abs=5e-15)
        assert actual["raw_observables"][index, 5] == pytest.approx(np.mean(field ** 2), abs=5e-15)
    for name in ("accepted", "delta_h", "cluster_sizes", "cluster_changed_sites",
                 "cluster_neighbor_visits", "cluster_bond_tests", "cluster_activated_bonds", "cluster_attempted"):
        np.testing.assert_array_equal(actual[name], kernel[name])
    for name in ("production_acceptance", "burn_in_acceptance", "max_absolute_delta_h",
                 "saved_magnetization_sign_changes", "hmc_trajectories", "hybrid_cycles",
                 "wolff_cluster_updates", "force_evaluations", "force_site_evaluations",
                 "total_cluster_sites", "total_cluster_changed_sites", "total_cluster_neighbor_visits",
                 "total_cluster_bond_tests", "total_cluster_activated_bonds", "initial", "seed"):
        assert actual[name] == kernel[name]
    for index, name in enumerate(hopping.OBSERVABLES):
        assert actual["diagnostics"][name] == base.series_diagnostics(expected[:, index])
    assert set(kernel["warnings"]) <= set(actual["warnings"])
    assert actual["kernel_schema"] == hybrid.SCHEMA
    assert actual["schema"] == hopping.SCHEMA
    assert actual["cluster_probability"] == .5
    assert actual["lazy_cluster_skips"] + actual["wolff_cluster_updates"] == actual["hybrid_cycles"]
    assert actual["physical_effective_mass_phi"] == kappa
    assert not actual["production_adaptation"] and not actual["equilibrium_certified"]
    assert not actual["symmetry_augmentation"] and actual["artificial_global_sign_flips"] == 0


def test_wrapper_does_not_mutate_kernel_returned_arrays(monkeypatch):
    mapping = hopping.gradient_parameters(.71, 1.1, 1)
    original = hybrid.sample_hybrid_chain(mapping["r"], mapping["u"], **_kwargs())
    array_copies = {name: value.copy() for name, value in original.items() if isinstance(value, np.ndarray)}
    old_diagnostics = json.dumps(original["diagnostics"], sort_keys=True)
    monkeypatch.setattr(hybrid, "sample_hybrid_chain", lambda *args, **kwargs: original)
    result = hopping.sample_chain(.71, 1.1, **_kwargs())
    for name, before in array_copies.items():
        np.testing.assert_array_equal(original[name], before)
    assert json.dumps(original["diagnostics"], sort_keys=True) == old_diagnostics
    assert not np.shares_memory(result["configurations"], original["configurations"])
    assert not np.shares_memory(result["raw_observables"], original["raw_observables"])
    assert not np.shares_memory(result["trajectory_magnetization"], original["trajectory_magnetization"])


@pytest.mark.parametrize("changes", [
    {"kappa": 0.}, {"kappa": -.1}, {"kappa": True}, {"kappa": float("nan")},
    {"kappa": float("inf")}, {"quartic_lambda": 0.}, {"quartic_lambda": -.1},
    {"quartic_lambda": True}, {"quartic_lambda": float("nan")},
    {"dimension": 0}, {"dimension": 4}, {"dimension": True}, {"h": .1}, {"h": float("nan")},
    {"size": 2}, {"samples": 3}, {"burn_in": -1}, {"spacing": 0},
    {"step_size": 0.}, {"step_size": float("nan")}, {"leapfrog_steps": 0},
    {"seed": -1}, {"initial": "wrong"}, {"cluster_probability": -.1},
    {"cluster_probability": 1.1}, {"cluster_probability": float("nan")},
])
def test_bad_parameters_fail_before_rng_creation(monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid parameters must fail before allocating RNG")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    values = dict(kappa=.68, quartic_lambda=1., **_kwargs())
    values.update(changes)
    with pytest.raises(ValueError):
        hopping.sample_chain(**values)


@pytest.mark.parametrize("phi", [np.ones(2), np.ones((3, 3, 3, 3)),
    np.array([0., float("nan"), 1.]), np.ones(3, dtype=complex)])
def test_invalid_direct_fields_rejected(phi):
    with pytest.raises(ValueError):
        hopping.action_and_gradient(phi, .68, 1.)
