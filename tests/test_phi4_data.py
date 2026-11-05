"""Tiny numerical and provenance checks for the continuous-field HMC sampler.

These tests generate only small, disposable lattices; they neither open earlier
physical datasets nor start training or a production sampling job.
"""

from collections import defaultdict
import hashlib
import json

import numpy as np
import pytest

from recurrent_rg.data import phi4 as data


def _options(**changes):
    values = dict(
        sizes=(3,), dimension=1, r_values=(.5, 1., 1.5), u_values=(0.,),
        chains=4, samples=8, burn_in=4, spacing=1, step_size=.1,
        leapfrog_steps=3, seed=553,
    )
    values.update(changes)
    return data.Phi4Options(**values)


def _independent_action(phi, r, u, h=0.):
    """Positive-direction bonds, each torus edge once, independent of force code."""
    value = .5 * r * np.sum(phi ** 2) + .25 * u * np.sum(phi ** 4) - h * np.sum(phi)
    for axis in range(phi.ndim):
        value += .5 * np.sum((phi - np.roll(phi, -1, axis=axis)) ** 2)
    return float(value)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_action_and_force_match_independent_periodic_bonds_and_finite_difference(dimension):
    phi = np.random.default_rng(554 + dimension).normal(scale=.45, size=(3,) * dimension)
    original = phi.copy()
    r, u, h = -.3, 1.1, .17
    action, gradient = data.action_and_gradient(phi, r, u, h=h)
    assert isinstance(action, (float, np.floating))
    assert gradient.shape == phi.shape and gradient.dtype == np.float64
    assert action == pytest.approx(_independent_action(phi, r, u, h), abs=1e-12)
    finite_difference = np.empty_like(phi)
    for index in np.ndindex(phi.shape):
        plus, minus = phi.copy(), phi.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference[index] = (
            _independent_action(plus, r, u, h) - _independent_action(minus, r, u, h)
        ) / 2e-6
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-8, rtol=1e-7)
    np.testing.assert_array_equal(phi, original)


def test_uniform_field_has_zero_kinetic_term_and_correct_source_sign():
    phi = np.full((3, 3), .7, dtype=np.float64)
    r, u, h = -.8, 1.3, .2
    action, gradient = data.action_and_gradient(phi, r, u, h=h)
    assert action == pytest.approx(phi.size * (.5 * r * .7 ** 2 + .25 * u * .7 ** 4 - h * .7))
    np.testing.assert_allclose(gradient, r * phi + u * phi ** 3 - h, atol=1e-14)
    zero_source, force = data.action_and_gradient(phi, r, u)
    flipped, flipped_force = data.action_and_gradient(-phi, r, u)
    assert zero_source == flipped
    np.testing.assert_array_equal(force, -flipped_force)


def test_leapfrog_is_reversible_without_hidden_momentum_flip_or_input_mutation():
    rng = np.random.default_rng(885)
    phi = rng.normal(scale=.3, size=(3, 3))
    momentum = rng.normal(scale=.5, size=phi.shape)
    old_phi, old_momentum = phi.copy(), momentum.copy()
    q, p = data.leapfrog(phi, momentum, -.2, 1., .06, 11, h=.08)
    recovered_q, recovered_minus_p = data.leapfrog(q, -p, -.2, 1., .06, 11, h=.08)
    np.testing.assert_allclose(recovered_q, phi, atol=2e-14, rtol=1e-14)
    np.testing.assert_allclose(recovered_minus_p, -momentum, atol=2e-14, rtol=1e-14)
    np.testing.assert_array_equal(phi, old_phi)
    np.testing.assert_array_equal(momentum, old_momentum)
    assert not np.shares_memory(q, phi)
    assert not np.shares_memory(p, momentum)


def test_leapfrog_fixed_trajectory_energy_error_has_second_order_step_convergence():
    errors = []
    for step_size, steps in ((.12, 8), (.06, 16)):
        sample_errors = []
        case_rng = np.random.default_rng(912)
        for _ in range(12):
            phi = case_rng.normal(scale=.4, size=3)
            momentum = case_rng.normal(scale=.6, size=3)
            initial = _independent_action(phi, .8, 1.2, .1) + .5 * np.sum(momentum ** 2)
            q, p = data.leapfrog(phi, momentum, .8, 1.2, step_size, steps, h=.1)
            final = _independent_action(q, .8, 1.2, .1) + .5 * np.sum(p ** 2)
            sample_errors.append(final - initial)
        errors.append(np.sqrt(np.mean(np.square(sample_errors))))
    assert errors[0] > 1e-6
    assert errors[1] < .35 * errors[0]


def test_gaussian_hmc_matches_exact_periodic_covariance_not_independent_site_variance():
    # For u=0, S=phi^T A phi/2, A=r I+the positive torus Laplacian.
    size, r = 3, .9
    eye = np.eye(size)
    precision = (r + 2.) * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    exact_covariance = np.linalg.inv(precision)
    sampled = data.sample_hmc_chain(
        r, 0., size=size, dimension=1, samples=2500, burn_in=128,
        spacing=2, step_size=.14, leapfrog_steps=9, seed=41822, initial="hot",
    )
    phi = sampled["configurations"][..., 0]
    assert phi.shape == (2500, 3) and phi.dtype == np.float64
    np.testing.assert_allclose(phi.mean(axis=0), 0., atol=.10, rtol=0.)
    np.testing.assert_allclose(np.cov(phi, rowvar=False, ddof=0), exact_covariance,
                               atol=.09, rtol=.12)
    assert exact_covariance[0, 1] > .2  # The analytic check is sensitive to the bonds.
    assert sampled["accepted"].mean() > .8


@pytest.mark.parametrize("initial", ["hot", "cold_positive", "cold_negative"])
def test_chain_records_raw_float64_fields_and_every_fixed_hmc_trajectory(initial):
    kwargs = dict(size=3, dimension=2, samples=8, burn_in=4, spacing=2,
                  step_size=.08, leapfrog_steps=4, seed=761, initial=initial)
    first = data.sample_hmc_chain(-.7, 1., **kwargs)
    second = data.sample_hmc_chain(-.7, 1., **kwargs)
    for key in ("configurations", "raw_observables", "accepted", "delta_h"):
        np.testing.assert_array_equal(first[key], second[key])
        assert np.all(np.isfinite(first[key]))
    assert first["configurations"].shape == (8, 3, 3, 1)
    assert first["configurations"].dtype == np.float64
    assert first["raw_observables"].shape == (8, 8)
    assert first["raw_observables"].dtype == np.float64
    assert first["accepted"].shape == first["delta_h"].shape == (4 + 8 * 2,)
    assert first["accepted"].dtype == np.bool_
    assert first["delta_h"].dtype == np.float64
    assert isinstance(first["diagnostics"], dict)
    assert isinstance(first["warnings"], list)
    assert not np.all(np.isin(first["configurations"], (-1., 1.)))


@pytest.mark.parametrize("changes", [
    {"sizes": (2,)}, {"sizes": ()}, {"dimension": 0}, {"dimension": 4},
    {"r_values": (float("nan"),)}, {"u_values": (-.1,)},
    {"r_values": (0.,), "u_values": (0.,)},
    {"r_values": (-.1,), "u_values": (0.,)},
    {"h": float("inf")}, {"chains": 2}, {"chains": 6}, {"samples": 3},
    {"burn_in": -1}, {"spacing": 0}, {"step_size": 0.},
    {"step_size": float("nan")}, {"leapfrog_steps": 0}, {"seed": -1},
])
def test_invalid_options_fail_before_creating_output(tmp_path, changes):
    target = tmp_path / "never_created"
    with pytest.raises(ValueError):
        data.generate_dataset(target, _options(**changes))
    assert not target.exists()


@pytest.mark.parametrize("cap", [
    "max_trajectories", "max_force_site_evaluations", "max_output_bytes", "max_memory_bytes",
])
def test_generation_caps_fail_closed_without_sampling_or_output(tmp_path, monkeypatch, cap):
    def forbidden(*args, **kwargs):
        raise AssertionError("A rejected plan must not sample.")

    monkeypatch.setattr(data, "sample_hmc_chain", forbidden)
    target = tmp_path / "over_budget"
    with pytest.raises(ValueError):
        data.generate_dataset(target, _options(**{cap: 1}))
    assert not target.exists()


def test_force_work_receipt_counts_actual_force_calls_including_both_half_steps(monkeypatch):
    original = data._gradient
    calls = []

    def counted(phi, *args):
        calls.append(phi.size)
        return original(phi, *args)

    monkeypatch.setattr(data, "_gradient", counted)
    result = data.sample_hmc_chain(
        .8, .7, size=3, dimension=2, samples=4, burn_in=2, spacing=2,
        step_size=.08, leapfrog_steps=3, seed=651,
    )
    expected = (2 + 4 * 2) * (3 + 1)
    assert len(calls) == result["force_evaluations"] == expected
    assert sum(calls) == expected * 9
    options = _options(sizes=(3, 4), dimension=2)
    plan = data.generation_plan(options)
    per_chain = options.burn_in + options.samples * options.spacing
    assert plan["total_hmc_trajectories"] == 3 * 2 * 4 * per_chain
    assert plan["total_force_site_evaluations"] == 3 * 4 * per_chain * (3 + 1) * (3 ** 2 + 4 ** 2)


def test_metropolis_rejections_retain_original_state_and_are_not_resampled(monkeypatch):
    calls = []

    def deliberately_bad(phi, momentum, *args):
        calls.append(1)
        return phi + 100., momentum.copy()

    monkeypatch.setattr(data, "_leapfrog", deliberately_bad)
    result = data.sample_hmc_chain(
        1., 1., size=3, dimension=1, samples=8, burn_in=2, spacing=2,
        step_size=.08, leapfrog_steps=3, seed=651, initial="cold_positive",
    )
    assert len(calls) == 2 + 8 * 2
    assert not result["accepted"].any()
    assert np.all(result["delta_h"] > 1e6)
    np.testing.assert_array_equal(result["configurations"], np.ones((8, 3, 1)))
    assert result["production_acceptance"] == 0.
    assert any("acceptance below" in warning for warning in result["warnings"])


def test_plan_is_deterministic_and_does_not_sample_or_create_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not sample.")

    monkeypatch.setattr(data, "sample_hmc_chain", forbidden)
    first = data.generation_plan(_options())
    second = data.generation_plan(_options())
    assert first == second
    assert not list(tmp_path.iterdir())
    json.dumps(first, allow_nan=False)


def test_resource_projection_includes_chain_diagnostics_not_only_small_field_arrays():
    small = data.generation_plan(_options(chains=4))
    large = data.generation_plan(_options(chains=40))
    assert large["projected_output_bytes"] - small["projected_output_bytes"] >= 3 * 36 * 32768
    assert large["projected_working_bytes"] - small["projected_working_bytes"] >= 36 * 65536


def _file_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_saved_shards_are_deterministic_separate_labels_and_never_overwrite(tmp_path):
    options = _options()
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = data.generate_dataset(first, options), data.generate_dataset(second, options)
    assert a["status"] == b["status"] == "complete"
    assert len(a["shards"]) == len(b["shards"]) == 3
    for row_a, row_b in zip(a["shards"], b["shards"]):
        with np.load(first / row_a["path"], allow_pickle=False) as xa, \
                np.load(second / row_b["path"], allow_pickle=False) as xb:
            assert set(xa.files) == {"configurations", "raw_observables", "accepted", "delta_h"}
            assert set(xb.files) == set(xa.files)
            for key in xa.files:
                np.testing.assert_array_equal(xa[key], xb[key])
            assert xa["configurations"].dtype == np.float64
            assert xa["configurations"].shape == (4, 8, 3, 1)
            assert xa["raw_observables"].shape == (4, 8, 8)
        assert hashlib.sha256((first / row_a["path"]).read_bytes()).hexdigest() == row_a["sha256"]
        sidecar_path = first / row_a["sidecar"]
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == row_a["sidecar_sha256"]
        assert json.loads(sidecar_path.read_text())["data_sha256"] == row_a["sha256"]
    snapshot = _file_hashes(first)
    with pytest.raises(FileExistsError):
        data.generate_dataset(first, options)
    assert _file_hashes(first) == snapshot


def test_existing_empty_directory_is_not_implicitly_adopted(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        data.generate_dataset(target, _options())
    assert not list(target.iterdir())


def test_split_is_by_entire_physical_parameter_cell_across_sizes_and_chain_initializations(tmp_path):
    options = _options(sizes=(3, 4), u_values=(0., .7))
    root = tmp_path / "multiple_sizes"
    manifest = data.generate_dataset(root, options)
    assert len(manifest["shards"]) == 12
    grouped = defaultdict(list)
    seeds = []
    for shard in manifest["shards"]:
        sidecar = json.loads((root / shard["sidecar"]).read_text())
        assert sidecar["size"] == shard["size"]
        assert sidecar["cell_index"] == shard["cell_index"]
        assert sidecar["split"] == shard["split"]
        assert sidecar["h"] == 0.
        assert not sidecar["labels_used_in_training"]
        assert not sidecar["symmetry_augmentation"]
        assert not sidecar["equilibrium_certified"]
        rows = sidecar["chains"]
        assert [row["initial"] for row in rows] == [
            "hot", "hot", "cold_positive", "cold_negative",
        ]
        assert all(not row["equilibrium_certified"] and not row["symmetry_augmentation"] for row in rows)
        seeds.extend(row["seed"] for row in rows)
        with np.load(root / shard["path"], allow_pickle=False) as arrays:
            assert arrays["configurations"].shape == (4, 8, shard["size"], 1)
        grouped[(sidecar["r"], sidecar["u"], sidecar["h"])].append(shard)
    assert len(seeds) == len(set(seeds))
    assert len(grouped) == 6
    for shards in grouped.values():
        assert {shard["size"] for shard in shards} == {3, 4}
        assert len({shard["split"] for shard in shards}) == 1
        cell = shards[0]["cell_index"]
        assert shards[0]["split"] == ("test" if cell % 3 == 1 else "train")
    assert {shard["split"] for shard in manifest["shards"]} == {"train", "test"}


def test_observations_and_diagnostics_use_saved_unaugmented_chain_and_retain_cold_sign():
    for initial, sign in (("cold_positive", 1), ("cold_negative", -1)):
        # Tiny fixed trajectory length preserves the chosen sector, so an
        # accidental Z2 augmentation or post-hoc symmetrization is detectable.
        result = data.sample_hmc_chain(
            -2., 1., size=3, dimension=2, samples=8, burn_in=0, spacing=1,
            step_size=1e-5, leapfrog_steps=1, seed=651, initial=initial,
        )
        phi = result["configurations"][..., 0]
        means = phi.mean(axis=(1, 2))
        indices = {name: index for index, name in enumerate(result["raw_observable_names"])}
        raw = result["raw_observables"]
        np.testing.assert_allclose(raw[:, indices["magnetization"]], means, atol=1e-14)
        np.testing.assert_allclose(raw[:, indices["absolute_magnetization"]], np.abs(means), atol=1e-14)
        np.testing.assert_allclose(raw[:, indices["magnetization_squared"]], means ** 2, atol=1e-14)
        np.testing.assert_allclose(raw[:, indices["magnetization_fourth"]], means ** 4, atol=1e-14)
        np.testing.assert_allclose(raw[:, indices["mean_field_squared"]], (phi ** 2).mean(axis=(1, 2)), atol=1e-14)
        np.testing.assert_allclose(raw[:, indices["structure_factor_zero"]], 9 * means ** 2, atol=1e-13)
        expected_actions = [_independent_action(field, -2., 1.) / 9 for field in phi]
        np.testing.assert_allclose(raw[:, indices["action_density"]], expected_actions, atol=1e-14)
        assert np.all(sign * means > 1.)
        assert result["saved_magnetization_sign_changes"] == 0
        assert not result["symmetry_augmentation"] and not result["equilibrium_certified"]
        assert result["diagnostics"]["magnetization"]["mean"] == pytest.approx(means.mean())
        assert any("symmetry-sector" in warning for warning in result["warnings"])


def test_seed_streams_are_physical_parameter_keyed_not_parameter_list_order():
    options = _options(sizes=(3, 4), u_values=(0., .7))
    reordered = _options(sizes=(4, 3), r_values=(1.5, .5, 1.), u_values=(.7, 0.))
    for size in options.sizes:
        for r_index, r in enumerate(options.r_values):
            for u_index, u in enumerate(options.u_values):
                cell = r_index * len(options.u_values) + u_index
                other_cell = reordered.r_values.index(r) * len(reordered.u_values) + reordered.u_values.index(u)
                for chain in range(options.chains):
                    assert data.chain_seed(options, size, cell, chain) == data.chain_seed(reordered, size, other_cell, chain)


def test_sampling_failure_preserves_partial_receipts_without_retry_or_resume(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["seed"])
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(data, "sample_hmc_chain", fail)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        data.generate_dataset(root, _options())
    assert len(calls) == 1
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed"
    assert not status["workflow_complete"]
    assert status["requires_user_action"]
    assert status["returncode"] == 1
    assert (root / "plan.json").is_file()
    assert not (root / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        data.generate_dataset(root, _options())
    assert len(calls) == 1


def _tiny_cli_arguments():
    return [
        "--dimension", "1", "--sizes", "3", "--r-values", ".5", "1.", "1.5",
        "--u-values", "0.", "--chains", "4", "--samples", "4", "--burn-in", "2",
        "--spacing", "1", "--step-size", ".1", "--leapfrog-steps", "3", "--seed", "553",
    ]


def test_cli_defaults_to_pure_plan_even_when_output_is_given(tmp_path, monkeypatch, capsys):
    from scripts.prepare_phi4 import main

    def forbidden(*args, **kwargs):
        raise AssertionError("Default CLI planning must not sample.")

    monkeypatch.setattr(data, "sample_hmc_chain", forbidden)
    target = tmp_path / "not_created"
    plan = main(["--output", str(target), *_tiny_cli_arguments()])
    assert not target.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == plan["schema"] == data.SCHEMA
    assert plan["training_updates"] == 0
    assert not plan["labels_used_in_training"]
    assert not plan["current_binary_qcr_compatible"]


def test_cli_execute_requires_an_explicit_destination(capsys):
    from scripts.prepare_phi4 import main

    with pytest.raises(SystemExit) as error:
        main(["--execute", *_tiny_cli_arguments()])
    assert error.value.code == 2
    assert "--execute requires --output" in capsys.readouterr().err


def test_cli_tiny_explicit_execution_writes_complete_status_not_training(tmp_path, capsys):
    from scripts.prepare_phi4 import main

    target = tmp_path / "tiny_cli_data"
    manifest = main(["--execute", "--output", str(target), *_tiny_cli_arguments()])
    assert manifest["status"] == "complete"
    assert len(manifest["shards"]) == 3
    status = json.loads((target / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"]
    assert status["returncode"] == 0 and status["error"] is None
    assert not status["requires_user_action"]
    final_line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final_line["training_updates"] == 0
    assert not final_line["current_binary_qcr_compatible"]
    assert final_line["conditions"] == 3

