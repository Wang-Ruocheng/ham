"""Exact saved-array algebra for fixed-J overlap observables, with no sampler.

The enumerated small lattice is the *entire* uniform measure, not a Monte Carlo
estimate.  Replica pairs share replicas and saved times: they are diagnostic
averages, never additional independent chains.
"""

import itertools

import numpy as np
import pytest

from scripts.analyze_spin_glass_fixed_j_observables import (
    aggregate_observables,
    replica_observables,
    sample_observables,
)


def all_l2_fields():
    return np.array(list(itertools.product((-1, 1), repeat=8)), dtype=np.int8).reshape(-1, 2, 2, 2)


def stripe(side=4, axis=0):
    """Two positive then two negative slabs: nonzero first Fourier mode."""
    indices = np.indices((side,) * 3)[axis]
    return np.where(indices < side // 2, 1, -1).astype(np.int8)


def direct_structure_factors(fields):
    """Brute force, unnormalised sum divided by N; independent of FFT APIs."""
    side = fields.shape[1]
    sites = side ** 3
    coordinates = np.indices((side,) * 3).reshape(3, -1)
    flattened = fields.reshape(len(fields), sites).astype(np.float64)
    modes = [np.ones(sites, dtype=np.complex128)]
    modes.extend(np.exp(2j * np.pi * coordinates[axis] / side) for axis in range(3))
    return np.stack([np.abs(flattened @ mode) ** 2 / sites for mode in modes], axis=1)


@pytest.mark.parametrize("side", (2, 3, 4))
def test_moments_and_all_four_structure_factors_match_direct_dft(side):
    coords = np.indices((side,) * 3)
    fields = np.stack([
        np.ones((side,) * 3, dtype=np.int8),
        np.where((coords[0] + coords[1] + coords[2]) % 2, -1, 1),
        np.where((2 * coords[0] + coords[1] + coords[2]) % 3, -1, 1),
        np.where(coords[1] < (side + 1) // 2, -1, 1),
    ]).astype(np.int8)
    before = fields.copy()
    result = sample_observables(fields)
    q = fields.mean(axis=(1, 2, 3))
    assert set(result) >= {"q", "abs_q", "q2", "q4", "structure_factor"}
    np.testing.assert_allclose(result["q"], q, atol=1e-15)
    np.testing.assert_allclose(result["abs_q"], np.abs(q), atol=1e-15)
    np.testing.assert_allclose(result["q2"], q ** 2, atol=1e-15)
    np.testing.assert_allclose(result["q4"], q ** 4, atol=1e-15)
    np.testing.assert_allclose(result["structure_factor"], direct_structure_factors(fields), rtol=1e-13, atol=2e-13)
    np.testing.assert_allclose(result["structure_factor"][:, 0], side ** 3 * result["q2"], atol=2e-13)
    np.testing.assert_array_equal(fields, before)


def test_enumerated_infinite_temperature_measure_has_exact_known_moments():
    fields = all_l2_fields()
    samples = sample_observables(fields)
    result = aggregate_observables(samples, 2)
    sites = 8
    expected_q2 = 1 / sites
    expected_q4 = 3 / sites ** 2 - 2 / sites ** 3
    assert result["q_mean"] == pytest.approx(0.0, abs=1e-15)
    assert result["abs_q_mean"] == pytest.approx(np.mean(np.abs(fields.mean(axis=(1, 2, 3)))))
    assert result["q2_mean"] == pytest.approx(expected_q2)
    assert result["q4_mean"] == pytest.approx(expected_q4)
    assert result["chi_sg"] == pytest.approx(1.0)
    assert result["S0"] == pytest.approx(1.0)
    np.testing.assert_allclose(result["S_kmin_axes"], [1.0] * 3, atol=2e-14)
    assert result["S_kmin_mean"] == pytest.approx(1.0)
    assert result["U4"] == pytest.approx(3 - 2 / sites)
    assert result["binder"] == pytest.approx(1 / sites)
    assert result["second_moment_ratio"] == pytest.approx(0.0, abs=2e-14)
    assert result["xi_valid"]
    assert result["xi_over_L"] == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(result["xi_over_L_axes"], [0.0] * 3, atol=1e-7)


def test_binder_is_ratio_of_ensemble_moments_not_mean_per_sample_ratio():
    # q=(1, 1/2), so the Binder ratio must be computed only *after* averaging.
    fields = np.ones((2, 2, 2, 2), dtype=np.int8)
    fields[1].ravel()[:2] = -1
    result = aggregate_observables(sample_observables(fields), 2)
    q2 = (1 + .25) / 2
    q4 = (1 + .0625) / 2
    assert result["U4"] == pytest.approx(q4 / q2 ** 2)
    assert result["binder"] == pytest.approx((3 - q4 / q2 ** 2) / 2)
    assert result["binder"] != pytest.approx(1.0)
    assert result["chi_sg"] == pytest.approx(8 * q2)


@pytest.mark.parametrize("sign", (-1, 1))
def test_uniform_field_has_zero_kmin_denominator_not_a_finite_length(sign):
    fields = np.full((3, 4, 4, 4), sign, dtype=np.int8)
    result = aggregate_observables(sample_observables(fields), 4)
    assert result["q_mean"] == sign
    assert result["abs_q_mean"] == result["q2_mean"] == result["q4_mean"] == 1
    assert result["chi_sg"] == result["S0"] == 64
    assert result["U4"] == result["binder"] == 1
    assert not result["xi_valid"]
    assert np.isnan(result["xi_over_L"])
    assert isinstance(result["xi_reason"], str) and result["xi_reason"]
    assert np.all(np.isnan(result["xi_over_L_axes"]))


def test_negative_second_moment_ratio_is_preserved_and_not_clipped_to_zero():
    field = stripe()
    result = aggregate_observables(sample_observables(field[None]), 4)
    assert result["q2_mean"] == result["chi_sg"] == result["S0"] == 0
    assert result["S_kmin_axes"][0] > 0
    assert result["second_moment_ratio"] == pytest.approx(-1.0)
    assert not result["xi_valid"]
    assert np.isnan(result["xi_over_L"])
    assert isinstance(result["xi_reason"], str) and result["xi_reason"]
    assert np.isnan(result["binder"]) and np.isnan(result["U4"])


def test_three_axes_are_averaged_before_ratio_and_square_root():
    # ⟨S0⟩=32; ⟨Sx⟩=16, ⟨Sy⟩=⟨Sz⟩=0.  The mean-k ratio is 5,
    # not the x-axis ratio 1, and not a mean of three correlation lengths.
    fields = np.stack([np.ones((4, 4, 4), dtype=np.int8), stripe()])
    result = aggregate_observables(sample_observables(fields), 4)
    assert result["S0"] == pytest.approx(32)
    np.testing.assert_allclose(result["S_kmin_axes"], [16, 0, 0], atol=2e-13)
    assert result["S_kmin_mean"] == pytest.approx(16 / 3)
    assert result["second_moment_ratio"] == pytest.approx(5)
    assert result["xi_over_L"] == pytest.approx(np.sqrt(5) / (8 * np.sin(np.pi / 4)))
    assert result["xi_valid"]
    assert result["xi_over_L_axes"][0] == pytest.approx(1 / (8 * np.sin(np.pi / 4)))
    assert np.all(np.isnan(result["xi_over_L_axes"][1:]))


def test_axis_permutations_preserve_isotropic_estimate_and_permute_diagnostics():
    base = np.stack([np.ones((4, 4, 4), dtype=np.int8), stripe(axis=0)])
    original = aggregate_observables(sample_observables(base), 4)
    for target_axis in (1, 2):
        rotated = np.swapaxes(base, 1, target_axis + 1)
        result = aggregate_observables(sample_observables(rotated), 4)
        assert result["xi_over_L"] == pytest.approx(original["xi_over_L"])
        expected = np.zeros(3)
        expected[target_axis] = 16
        np.testing.assert_allclose(result["S_kmin_axes"], expected, atol=2e-13)


def test_tiny_negative_roundoff_ratio_may_be_zero_but_real_negative_may_not():
    samples = sample_observables(all_l2_fields())
    samples["structure_factor"] = np.ones((256, 4), dtype=np.float64)
    samples["structure_factor"][:, 1:] += 5e-13
    rounded = aggregate_observables(samples, 2)
    assert rounded["xi_valid"]
    assert rounded["xi_over_L"] == pytest.approx(0.0, abs=1e-15)
    # Preserve the raw estimator in the audit even when sqrt uses numerical 0.
    assert -1e-12 < rounded["second_moment_ratio"] < 0
    assert "tolerance" in rounded["xi_reason"]
    samples["structure_factor"][:, 1:] = 1 + 1e-6
    invalid = aggregate_observables(samples, 2)
    assert not invalid["xi_valid"]
    assert invalid["second_moment_ratio"] < -1e-7
    assert np.isnan(invalid["xi_over_L"])


def test_replica_pairs_preserve_time_alignment_and_primary_pair_01():
    spins = np.ones((3, 3, 4, 4, 4), dtype=np.int8)
    spins[1, 1] = stripe()
    spins[1, 2] = -1
    spins[2, 0] = stripe(axis=1)
    spins[2, 1] = -1
    before = spins.copy()
    result = replica_observables(spins)
    expected_pairs = list(itertools.combinations(range(3), 2))
    assert [tuple(pair) for pair in result["pairs"]] == expected_pairs
    expected = [sample_observables(spins[a] * spins[b]) for a, b in expected_pairs]
    for key in expected[0]:
        np.testing.assert_allclose(result["primary"][key], expected[0][key], atol=2e-13)
        np.testing.assert_allclose(result["pair_average"][key], np.mean([item[key] for item in expected], axis=0), atol=2e-13)
    np.testing.assert_array_equal(result["primary"]["q"], [1, 0, -1])
    np.testing.assert_allclose(result["pair_q"], np.stack([item["q"] for item in expected]))
    assert result["pair_q"].shape == (3, 3), "pairs must not be concatenated with saved time"
    np.testing.assert_array_equal(spins, before)


def test_six_pair_average_averages_moments_not_pair_mean_before_squaring():
    spins = np.ones((4, 5, 2, 2, 2), dtype=np.int8)
    spins[2:] = -1
    result = replica_observables(spins)
    assert len(result["pairs"]) == 6
    assert result["pair_q"].shape == (6, 5)
    np.testing.assert_array_equal(result["primary"]["q"], np.ones(5))
    np.testing.assert_allclose(result["pair_average"]["q"], -np.ones(5) / 3)
    for key in ("abs_q", "q2", "q4"):
        np.testing.assert_array_equal(result["pair_average"][key], np.ones(5))
    np.testing.assert_allclose(result["pair_average"]["structure_factor"][:, 0], 8)
    aggregate = aggregate_observables(result["pair_average"], 2)
    assert aggregate["q2_mean"] == aggregate["q4_mean"] == 1
    assert aggregate["chi_sg"] == 8
    assert aggregate["q2_mean"] != pytest.approx(aggregate["q_mean"] ** 2)


def test_common_local_gauge_transform_of_replicas_leaves_overlap_observables():
    fields = all_l2_fields()[:5]
    spins = np.stack([fields, np.roll(fields, 1, axis=1), np.roll(fields, 1, axis=2)])
    gauge = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int8).reshape(2, 2, 2)
    first = replica_observables(spins)
    second = replica_observables(spins * gauge)
    for group in ("primary", "pair_average"):
        for key in first[group]:
            np.testing.assert_allclose(second[group][key], first[group][key], rtol=0, atol=0)
    np.testing.assert_array_equal(second["pair_q"], first["pair_q"])


@pytest.mark.parametrize("fields", [
    np.ones((2, 2, 2)),
    np.ones((0, 2, 2, 2)),
    np.ones((2, 1, 1, 1)),
    np.ones((2, 2, 3, 2)),
    np.zeros((2, 2, 2, 2)),
    np.full((2, 2, 2, 2), np.nan),
    np.full((2, 2, 2, 2), np.inf),
])
def test_sample_input_shape_and_ising_values_are_validated(fields):
    with pytest.raises(ValueError):
        sample_observables(fields)


@pytest.mark.parametrize("spins", [
    np.ones((2, 2, 2, 2)),
    np.ones((1, 2, 2, 2, 2)),
    np.ones((2, 0, 2, 2, 2)),
    np.ones((2, 2, 2, 3, 2)),
    np.zeros((2, 2, 2, 2, 2)),
    np.full((2, 2, 2, 2, 2), np.nan),
])
def test_replica_input_shape_and_ising_values_are_validated(spins):
    with pytest.raises(ValueError):
        replica_observables(spins)


@pytest.mark.parametrize("side", (0, 1, -2, 2.0, True, "2"))
def test_aggregate_rejects_noninteger_or_invalid_lattice_size(side):
    with pytest.raises(ValueError):
        aggregate_observables(sample_observables(all_l2_fields()), side)


@pytest.mark.parametrize("bad_trace", ("empty", "different_lengths", "nonfinite", "spectrum_shape"))
def test_aggregate_rejects_invalid_trace_geometry_or_nonfinite_values(bad_trace):
    samples = sample_observables(all_l2_fields())
    if bad_trace == "empty":
        samples = {key: value[:0] for key, value in samples.items()}
    elif bad_trace == "different_lengths":
        samples["q"] = samples["q"][:-1]
    elif bad_trace == "nonfinite":
        samples["q4"][0] = np.nan
    elif bad_trace == "spectrum_shape":
        samples["structure_factor"] = samples["structure_factor"][:, :3]
    with pytest.raises(ValueError):
        aggregate_observables(samples, 2)


def test_aggregate_audits_S0_equals_sites_times_q2():
    samples = sample_observables(all_l2_fields())
    samples["structure_factor"][:, 0] *= 2
    with pytest.raises((ValueError, AssertionError)):
        aggregate_observables(samples, 2)
