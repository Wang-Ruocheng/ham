"""Disposable tiny hybrid datasets only; no production sampling or training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from recurrent_rg.data import phi4 as base
from recurrent_rg.data import phi4_hybrid as kernel
from recurrent_rg.data import phi4_hybrid_pipeline as pipeline


def options(**changes):
    values = dict(sizes=(3,), dimension=1, r_values=(-.3, .1, .3), u_values=(.2,),
                  chains=4, samples=4, burn_in=2, spacing=2, step_size=.05,
                  leapfrog_steps=2, seed=780)
    values.update(changes)
    return pipeline.Phi4HybridOptions(**values)


def test_default_plan_pins_formal_physics_and_worst_case_work_without_sampling(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning cannot sample")
    monkeypatch.setattr(kernel, "sample_hybrid_chain", forbidden)
    plan = pipeline.generation_plan(pipeline.Phi4HybridOptions())
    assert plan["schema"] == "euclidean_phi4_hmc_embedded_wolff_v1"
    assert plan["sample_semantics"] == "independent_correlated_hybrid_markov_chains"
    assert plan["options"]["sizes"] == (36,)
    np.testing.assert_array_equal(plan["options"]["r_values"], np.linspace(-.5, .3, 12))
    assert plan["options"]["u_values"] == (.2,) and plan["options"]["h"] == 0.
    assert plan["options"]["samples"] == 128 and plan["options"]["chains"] == 4
    assert plan["options"]["burn_in"] == 1024 and plan["options"]["spacing"] == 8
    assert plan["options"]["step_size"] == .08 and plan["options"]["leapfrog_steps"] == 12
    assert plan["cluster_probability"] == .5
    assert plan["total_hmc_trajectories"] == 98304
    assert plan["total_force_evaluations"] == 1277952
    assert plan["total_force_site_evaluations"] == 1656225792
    assert plan["maximum_cluster_updates"] == 98304
    assert plan["expected_cluster_updates"] == 49152
    assert plan["maximum_cluster_sites"] == 98304 * 36 ** 2
    assert plan["maximum_cluster_neighbor_visits"] == 98304 * 36 ** 2 * 4
    assert plan["maximum_cluster_bond_tests"] == 98304 * 36 ** 2 * 2
    assert plan["maximum_cluster_activated_bonds"] == 98304 * (36 ** 2 - 1)
    assert plan["test_cell_indices"] == [1, 4, 7, 10]
    assert plan["configurations"] == 6144
    assert plan["projected_output_bytes"] < 1024 ** 3
    assert plan["projected_working_bytes"] < 512 * 1024 ** 2
    assert plan["source_pins"] == pipeline.source_pins()
    assert len(plan["source_pins"]) == 3
    assert not list(tmp_path.iterdir())
    json.dumps(plan, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"sizes": ()}, {"sizes": (2,)}, {"dimension": 4}, {"r_values": (-.3,), "u_values": (0.,)},
    {"u_values": (-.2,)}, {"h": .1}, {"cluster_probability": 1.}, {"cluster_probability": 0.},
    {"cluster_probability": True}, {"chains": 2}, {"samples": 3}, {"burn_in": -1}, {"spacing": 0},
    {"step_size": 0.}, {"leapfrog_steps": 0}, {"seed": -1}, {"test_offset": 3},
    {"max_force_site_evaluations": 2_000_000_001}, {"max_output_bytes": 1024 ** 3 + 1},
    {"max_trajectories": 1}, {"max_force_site_evaluations": 1}, {"max_cluster_updates": 1},
    {"max_cluster_neighbor_visits": 1}, {"max_output_bytes": 1}, {"max_memory_bytes": 1},
])
def test_invalid_or_over_budget_plan_never_creates_directory_or_samples(tmp_path, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected plan cannot sample")
    monkeypatch.setattr(kernel, "sample_hybrid_chain", forbidden)
    root = tmp_path / "not_created"
    with pytest.raises(ValueError):
        pipeline.generate_dataset(root, options(**changes))
    assert not root.exists()


def test_cluster_budget_uses_every_cycle_upper_bound_not_half_expected_number():
    current = options()
    plan = pipeline.generation_plan(current)
    assert plan["maximum_cluster_updates"] > plan["expected_cluster_updates"]
    with pytest.raises(ValueError, match="maximum cluster update"):
        pipeline.generation_plan(options(max_cluster_updates=int(plan["expected_cluster_updates"])))


def test_plan_array_byte_projection_includes_all_actual_per_cycle_arrays():
    plan = pipeline.generation_plan(options())
    one_chain = kernel.sample_hybrid_chain(-.3, .2, size=3, dimension=1, samples=4,
        burn_in=2, spacing=2, step_size=.05, leapfrog_steps=2, seed=780)
    expected = 4 * (3 + len(base.OBSERVABLES)) * 8 + 10 * 58
    assert sum(one_chain[name].nbytes for name in pipeline.ARRAY_MEMBERS) == expected
    assert plan["projected_output_bytes"] > expected * 4 * 3


def test_tiny_dataset_raw_arrays_hashes_labels_splits_and_actual_work_are_consistent(tmp_path):
    root = tmp_path / "fresh"
    current = options()
    pins_before = pipeline.source_pins()
    manifest = pipeline.generate_dataset(root, current)
    assert pins_before == pipeline.source_pins() == manifest["source_pins"]
    assert manifest["source_pins_match"] and not manifest["equilibrium_certified"]
    assert manifest["schema"] == kernel.SCHEMA and manifest["sample_semantics"] == pipeline.SAMPLE_SEMANTICS
    assert manifest["options"]["cluster_probability"] == .5
    assert len(manifest["shards"]) == 3
    totals = {name: 0 for name in pipeline.WORK_FIELDS}
    all_seeds = []
    for cell, shard in enumerate(manifest["shards"]):
        sidecar_path = root / shard["sidecar"]
        assert base.sha256_file(sidecar_path) == shard["sidecar_sha256"]
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["r"] == current.r_values[cell] and sidecar["u"] == .2 and sidecar["h"] == 0.
        assert sidecar["split"] == shard["split"] == ("test" if cell == 1 else "train")
        assert sidecar["training_member"] == "configurations"
        assert sidecar["source_pins"] == pins_before
        assert not sidecar["symmetry_augmentation"] and not sidecar["production_adaptation"]
        assert sidecar["artificial_global_sign_flips"] == 0
        assert [row["initial"] for row in sidecar["chains"]] == ["hot", "hot", "cold_positive", "cold_negative"]
        assert base.sha256_file(root / shard["path"]) == shard["sha256"] == sidecar["data_sha256"]
        with np.load(root / shard["path"], allow_pickle=False) as archive:
            assert set(archive.files) == set(pipeline.ARRAY_MEMBERS)
            configurations = archive["configurations"]
            raw = archive["raw_observables"]
            assert configurations.shape == (4, 4, 3, 1) and configurations.dtype == np.float64
            assert raw.shape == (4, 4, len(base.OBSERVABLES))
            for chain, row in enumerate(sidecar["chains"]):
                all_seeds.append(row["seed"])
                assert row["hybrid_cycles"] == row["hmc_trajectories"] == 10
                assert row["wolff_cluster_updates"] + row["lazy_cluster_skips"] == 10
                assert row["force_evaluations"] == 30 and row["force_site_evaluations"] == 90
                assert not any(isinstance(value, list) and name in pipeline.ARRAY_MEMBERS for name, value in row.items())
                attempted = archive["cluster_attempted"][chain]
                assert attempted.dtype == np.bool_
                assert int(attempted.sum()) == row["wolff_cluster_updates"]
                np.testing.assert_array_equal(archive["trajectory_magnetization"][chain, 3::2], raw[chain, :, 1])
                for array, total in (("cluster_sizes", "total_cluster_sites"),
                        ("cluster_neighbor_visits", "total_cluster_neighbor_visits"),
                        ("cluster_bond_tests", "total_cluster_bond_tests"),
                        ("cluster_activated_bonds", "total_cluster_activated_bonds")):
                    values = archive[array][chain]
                    assert int(values.sum()) == row[total]
                    assert np.all(values[~attempted] == 0)
                for sample, field in enumerate(configurations[chain, ..., 0]):
                    np.testing.assert_allclose(raw[chain, sample],
                        base._observations(field, sidecar["r"], sidecar["u"], 0.), atol=1e-13, rtol=1e-13)
                for name in totals:
                    totals[name] += row[name]
    assert len(all_seeds) == len(set(all_seeds)) == 12
    assert totals == manifest["actual_work"]
    quality = json.loads((root / manifest["quality_summary"]).read_text())
    assert base.sha256_file(root / manifest["quality_summary"]) == manifest["quality_summary_sha256"]
    assert quality["actual_work"] == totals
    assert quality["warning_count"] == manifest["warning_count"] > 0  # Tiny-chain ESS warnings are preserved.
    assert not quality["quality_is_execution_gate"] and quality["warnings_retained_without_resampling"]
    assert all(Path(path).suffix == ".json" and (root / path).is_file() for path in manifest["public_files"])
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"] and status["returncode"] == 0
    assert status["manifest_sha256"] == base.sha256_file(root / "manifest.json")
    assert not status["requires_user_action"] and status["actual_work"] == totals


def test_existing_directory_is_never_overwritten_or_adopted(tmp_path):
    root = tmp_path / "exists"
    root.mkdir()
    sentinel = root / "preserve.txt"
    sentinel.write_text("user data")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        pipeline.generate_dataset(root, options())
    assert list(root.iterdir()) == [sentinel]
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


def test_sampler_failure_is_recorded_and_not_automatically_retried(tmp_path, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("test injected sampler failure")
    monkeypatch.setattr(kernel, "sample_hybrid_chain", fail)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        pipeline.generate_dataset(root, options())
    assert calls == [1]
    assert not (root / "manifest.json").exists()
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and not status["workflow_complete"]
    assert status["returncode"] == 1 and status["requires_user_action"]
    assert (root / "plan.json").is_file()


def test_changed_source_pins_fail_closed_with_completed_data_preserved(tmp_path, monkeypatch):
    pins = pipeline.source_pins()
    responses = [pins, {**pins, "recurrent_rg/data/phi4_hybrid.py": "changed"}]
    monkeypatch.setattr(pipeline, "source_pins", lambda: responses.pop(0))
    root = tmp_path / "source_changed"
    with pytest.raises(RuntimeError, match="source changed"):
        pipeline.generate_dataset(root, options())
    assert len(list((root / "conditions").glob("*/data.npz"))) == 3
    assert not (root / "manifest.json").exists()
    assert json.loads((root / "status.json").read_text())["status"] == "failed"


def test_data_values_and_seed_streams_repeat_in_new_directory_without_reusing_old_files(tmp_path):
    current = options()
    first, second = tmp_path / "first", tmp_path / "second"
    one = pipeline.generate_dataset(first, current)
    two = pipeline.generate_dataset(second, current)
    for a, b in zip(one["shards"], two["shards"]):
        with np.load(first / a["path"], allow_pickle=False) as left, np.load(second / b["path"], allow_pickle=False) as right:
            for name in pipeline.ARRAY_MEMBERS:
                np.testing.assert_array_equal(left[name], right[name])


def test_size9_dataset_is_accepted_by_existing_loader_with_only_configurations_member_used(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from experiments.phi4_qcr import load_dataset
    root = tmp_path / "loader"
    pipeline.generate_dataset(root, options(sizes=(9,), dimension=2))
    original = np.load
    accessed = []
    class Archive:
        def __init__(self, target):
            self.target = target
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.target.close()
        def __getitem__(self, key):
            accessed.append(key)
            return self.target[key]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: Archive(original(*args, **kwargs)))
    loaded = load_dataset(root)
    assert loaded.configurations.shape == (12, 4, 9, 9, 1) and loaded.configurations.dtype == torch.float64
    assert accessed == ["configurations"] * 3
    assert loaded.sampler_schema == pipeline.SCHEMA and loaded.sample_semantics == pipeline.SAMPLE_SEMANTICS
    assert len(loaded.train_indices) == 8 and len(loaded.test_indices) == 4
    assert set(loaded.cell_ids[loaded.train_indices]).isdisjoint(set(loaded.cell_ids[loaded.test_indices]))
    assert loaded.scale > 0


def test_cli_defaults_to_plan_without_creating_output(tmp_path):
    from scripts.prepare_phi4_hybrid import main
    root = tmp_path / "not_created"
    plan = main(["--output", str(root)])
    assert plan["schema"] == pipeline.SCHEMA and plan["training_updates"] == 0
    assert not root.exists()


def test_cli_execute_requires_output_and_rejects_nonlazy_production():
    from scripts.prepare_phi4_hybrid import main
    with pytest.raises(SystemExit):
        main(["--execute"])
    with pytest.raises(SystemExit):
        main(["--cluster-probability", "1"])
