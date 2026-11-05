"""Small correctness checks only; no formal data or model training."""
from dataclasses import replace
import json

import numpy as np
import pytest

from recurrent_rg.ising3d import (Wolff3DOptions, chain_seed, energy_3d, generate_dataset,
    generation_plan, periodic_neighbors, sample_wolff_chain)


def sample(coupling=.22, **kwargs):
    options = dict(size=3, samples=20, burn_in_clusters=32, clusters_between=3,
        seed=91, initial="hot")
    options.update(kwargs)
    return sample_wolff_chain(coupling, **options)


def test_energy_periodic_bonds_global_z2_and_axis_symmetry():
    rng = np.random.default_rng(23)
    spins = rng.choice(np.asarray([-1, 1], np.int8), size=(7, 3, 3, 3))
    expected = []
    for configuration in spins:
        value = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    value -= configuration[i, j, k] * (int(configuration[(i + 1) % 3, j, k])
                        + int(configuration[i, (j + 1) % 3, k]) + int(configuration[i, j, (k + 1) % 3]))
        expected.append(value)
    np.testing.assert_array_equal(energy_3d(spins), expected)
    np.testing.assert_array_equal(energy_3d(spins), energy_3d(-spins))
    np.testing.assert_array_equal(energy_3d(spins), energy_3d(spins.transpose(0, 3, 1, 2)))
    assert energy_3d(np.ones((3, 3, 3), np.int8)) == -81


def test_six_periodic_neighbors_and_size_two_parallel_bond_convention():
    neighbors = periodic_neighbors(3)
    assert neighbors.shape == (27, 6)
    assert set(neighbors[0]) == {1, 2, 3, 6, 9, 18}
    assert all(site in neighbors[other] for site, adjacent in enumerate(neighbors) for other in adjacent)
    assert periodic_neighbors(2).shape == (8, 6)
    assert energy_3d(np.ones((2, 2, 2), np.int8)) == -24
    assert len(set(periodic_neighbors(2)[0])) == 3


@pytest.mark.parametrize("coupling", (.08, .22, .45))
def test_tiny_exact_boltzmann_distribution(coupling):
    states = (2 * ((np.arange(256)[:, None] >> np.arange(8)) & 1) - 1).astype(np.int8).reshape(256, 2, 2, 2)
    energies = energy_3d(states)
    m = states.mean(axis=(1, 2, 3))
    weights = np.exp(-coupling * energies)
    weights /= weights.sum()
    observables = np.stack([energies / 8, m, m ** 2, abs(m)], axis=1)
    exact = weights @ observables
    result = sample(coupling, size=2, samples=24000, burn_in_clusters=512, clusters_between=3)
    np.testing.assert_allclose(result["raw_observables"].mean(axis=0), exact, atol=.035)
    encoded = (((result["configurations"].reshape(-1, 8) + 1) // 2) * (1 << np.arange(8))).sum(axis=1)
    observed = np.bincount(encoded, minlength=256) / len(encoded)
    assert abs(observed - weights).sum() / 2 < .065


def test_determinism_work_counts_and_raw_pre_augmentation_identity():
    raw, again, augmented = sample(), sample(), sample(augmentation_seed=1729)
    np.testing.assert_array_equal(raw["configurations"], again["configurations"])
    np.testing.assert_array_equal(raw["raw_observables"], again["raw_observables"])
    np.testing.assert_array_equal(raw["cluster_sizes"], augmented["cluster_sizes"])
    np.testing.assert_array_equal(raw["raw_observables"], augmented["raw_observables"])
    np.testing.assert_array_equal(augmented["configurations"], raw["configurations"] * augmented["augmentation_flips"][:, None, None, None])
    np.testing.assert_array_equal(energy_3d(augmented["configurations"]) / 27, raw["raw_observables"][:, 0])
    assert raw["cluster_proposals"] == 32 + 20 * 3
    assert len(raw["cluster_sizes"]) == raw["cluster_proposals"]
    assert raw["cluster_visited_sites"] == raw["cluster_sizes"].sum()
    assert raw["neighbor_inspections"] == 6 * raw["cluster_visited_sites"]
    assert raw["cluster_sizes"].min() >= 1 and raw["cluster_sizes"].max() <= 27
    assert raw["diagnostics_before_global_z2"] and not raw["equilibrium_certified"]
    assert not np.array_equal(raw["configurations"], sample(seed=92)["configurations"])


@pytest.mark.parametrize("bad", ({"coupling": -.1}, {"coupling": np.nan}, {"size": 1},
    {"size": True}, {"size": 2.5}, {"samples": 0}, {"burn_in_clusters": -1},
    {"clusters_between": 0}, {"seed": -1}, {"seed": 2 ** 32}, {"initial": "mixed"},
    {"cold_sign": 0}, {"augmentation_seed": 2 ** 32}, {"max_output_bytes": 1}))
def test_invalid_chain_inputs_rejected_before_sampling(bad):
    with pytest.raises(ValueError):
        sample(**bad)


def test_default_plan_fixed_work_split_seed_namespaces_and_preflight():
    options = Wolff3DOptions()
    plan = generation_plan(options)
    assert plan["total_cluster_proposals"] == 24 * 4 * (4096 + 128 * 64)
    assert plan["dimension"] == 3 and plan["sites"] == 4096
    assert len(plan["train_coupling_indices"]) == 16 and len(plan["test_coupling_indices"]) == 8
    assert not set(plan["train_coupling_indices"]) & set(plan["test_coupling_indices"])
    seeds = [chain_seed(options, i, c, purpose) for i in range(24) for c in range(4) for purpose in ("thermal", "global_z2")]
    assert len(set(seeds)) == 192
    assert seeds != [chain_seed(replace(options, seed=options.seed + 1), i, c, purpose)
        for i in range(24) for c in range(4) for purpose in ("thermal", "global_z2")]
    assert not plan["labels_used_in_training"] and not plan["equilibrium_certified"]
    with pytest.raises(ValueError, match="preflight"):
        generation_plan(replace(options, max_working_bytes=1))


@pytest.mark.parametrize("bad", ({"chains": 3}, {"samples": 5}, {"couplings": (.1, .1, .3)},
    {"couplings": (-.1, .2, .3)}, {"test_offset": 3}, {"seed": 2 ** 32},
    {"augment_global_z2": 1}, {"test_every": 1}))
def test_invalid_physical_plan(bad):
    with pytest.raises(ValueError):
        generation_plan(replace(Wolff3DOptions(), **bad))


def small_options():
    return Wolff3DOptions(size=3, couplings=(.1, .22, .34), chains=4,
        samples=8, burn_in_clusters=8, clusters_between=2)


def test_small_streamed_generator_receipts_no_overwrite_and_raw_observables(tmp_path):
    root = tmp_path / "physical"
    manifest = generate_dataset(root, small_options())
    status = json.loads((root / "status.json").read_text())
    sampling = json.loads((root / "sampling.json").read_text())
    assert manifest["status"] == status["status"] == "complete"
    assert status["workflow_complete"] and status["returncode"] == 0
    assert len(manifest["shards"]) == 3
    assert sampling["total_cluster_proposals"] == 3 * 4 * (8 + 8 * 2)
    for shard in manifest["shards"]:
        sidecar = json.loads((root / shard["sidecar"]).read_text())
        assert [row["initial"] for row in sidecar["chains"]] == ["hot", "hot", "cold", "cold"]
        assert [row["cold_sign"] for row in sidecar["chains"]][2:] == [1, -1]
        assert sidecar["raw_diagnostics_before_global_z2"] and not sidecar["equilibrium_certified"]
        assert all("hot_minus_cold" in item for item in sidecar["hot_cold_comparison"].values())
        with np.load(root / shard["path"], allow_pickle=False) as arrays:
            assert arrays["configurations"].shape == (4, 8, 3, 3, 3)
            np.testing.assert_array_equal(energy_3d(arrays["configurations"]) / 27, arrays["raw_observables"][..., 0])
    with pytest.raises(FileExistsError):
        generate_dataset(root, small_options())


def test_chain_private_numba_rng_does_not_reseed_python_numpy_rng():
    np.random.seed(19)
    expected = np.random.random(4)
    np.random.seed(19)
    sample()
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_generator_failure_preserves_partial_root_and_does_not_resume(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure, not a physical experiment")
    monkeypatch.setattr("recurrent_rg.ising3d.sample_wolff_chain", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        generate_dataset(output, small_options())
    for status_path in (output / "status.json", output / "conditions/c000/status.json"):
        status = json.loads(status_path.read_text())
        assert status["status"] == "failed" and status["requires_user_action"]
        assert not status["workflow_complete"] and "synthetic failure" in status["error"]
    assert (output / "plan.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        generate_dataset(output, small_options())


def test_cli_dry_run_does_not_allocate_or_create_output(tmp_path, monkeypatch):
    from scripts.generate_ising3d_wolff import main
    def fail(*args, **kwargs):
        raise AssertionError("dry run attempted sampling")
    monkeypatch.setattr("recurrent_rg.ising3d.sample_wolff_chain", fail)
    output = tmp_path / "dry_run"
    result = main(["--output", str(output), "--coupling-count", "3", "--dry-run"])
    assert result["total_cluster_proposals"] == 3 * 4 * 12288
    assert not output.exists()
