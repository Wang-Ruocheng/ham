"""Small synthetic execution checks, not physical production/equilibration claims."""
from dataclasses import replace
import json

import numpy as np
import pytest

from recurrent_rg.data import spin_glass_pipeline as pipeline
from recurrent_rg.data.spin_glass_pipeline import (
    GenerationOptions, generate_dataset, generation_plan, iter_spin_glass, series_diagnostics,
)
from scripts.generate_spin_glass import main


def tiny(**kwargs):
    options = GenerationOptions(shape=(3,), temperatures=(1., 3.), disorders=3,
        replicas=2, samples=8, burn_in=4, spacing=1, seed=123)
    return replace(options, **kwargs)


@pytest.mark.parametrize("shape", [(3,), (3, 4), (3, 3, 3), (3, 3, 3, 3)])
def test_dimension_independent_roundtrip_and_no_disorder_leakage(tmp_path, shape):
    options = tiny(shape=shape)
    manifest = generate_dataset(tmp_path / "new", options)
    assert manifest["status"] == "complete"
    assert manifest["training_updates"] == 0
    train = list(iter_spin_glass(tmp_path / "new", "train"))
    test = list(iter_spin_glass(tmp_path / "new", "test"))
    assert {r.disorder_id for r in train} == {0, 2}
    assert {r.disorder_id for r in test} == {1}
    for record in train + test:
        assert record.configurations.shape == (2, 2, 8, *shape)
        assert record.bonds.shape == (len(shape), *shape)
        assert record.spin_view().shape == (2, 8, *shape, 1)
        assert record.overlap_view().shape == (2, 8, *shape, 1)
        np.testing.assert_array_equal(record.overlap(),
            record.configurations[:, 0] * record.configurations[:, 1])
        assert np.isin(record.overlap_view(), (-1, 1)).all()
        # No metadata is appended as a model channel.
        assert record.spin_view().dtype == np.int8
        for pair in ((0, 0), (0, 2), (-1, 1)):
            with pytest.raises(ValueError):
                record.overlap(*pair)
        with pytest.raises(ValueError):
            record.spin_view(2)
    status = json.loads((tmp_path / "new/status.json").read_text())
    assert status["workflow_complete"] and not status["equilibrium_certified"]
    receipt = json.loads((tmp_path / "new/sampling.json").read_text())
    assert not receipt["equilibrium_certified"]
    assert "warning_count" in receipt


def test_seeded_shards_are_reproducible_and_bonds_stay_quenched(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    options = tiny(distribution="gaussian", bond_mean=.1, bond_scale=.8)
    generate_dataset(first, options)
    generate_dataset(second, options)
    a, b = list(iter_spin_glass(first)), list(iter_spin_glass(second))
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left.configurations, right.configurations)
        np.testing.assert_array_equal(left.bonds, right.bonds)
        np.testing.assert_array_equal(left.energies, right.energies)
    assert not np.array_equal(a[0].bonds, a[1].bonds)
    # J has no temperature, thermal-replica, or sample axis.
    assert a[0].bonds.shape == (1, 3)


@pytest.mark.parametrize("updates", [
    dict(shape=()), dict(shape=(2, 3)), dict(shape=(3.5,)), dict(disorders=2),
    dict(replicas=1), dict(samples=3), dict(seed=-1), dict(burn_in=-1), dict(spacing=0),
    dict(temperatures=(0., 1.)), dict(temperatures=(1., 1.)),
    dict(temperatures=(2., 1.)), dict(temperatures=(1., float("nan"))),
    dict(bond_scale=0), dict(bond_mean=.1), dict(positive_probability=2),
    dict(positive_probability=0), dict(positive_probability=1),
    dict(distribution="gaussian", positive_probability=.7),
    dict(distribution="unknown"), dict(field=float("inf")),
    dict(test_every=2, test_offset=2), dict(test_every=99, test_offset=8),
    dict(max_spin_proposals=1), dict(max_output_bytes=1), dict(max_memory_bytes=1),
    dict(max_sites=2), dict(shape=(True,)), dict(replicas=True),
    dict(disorders=10**12), dict(shape=(np.int64(2**22),) * 3),
    dict(disorders=100, max_memory_bytes=1_500_000),
])
def test_validation_and_budgets_are_before_sampling_or_output(tmp_path, monkeypatch, updates):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must run before sampler")
    monkeypatch.setattr(pipeline, "sample_parallel_tempering", forbidden)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "invalid", tiny(**updates))
    assert not (tmp_path / "invalid").exists()


def test_dry_run_and_work_accounting_do_not_sample_or_create_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "sample_parallel_tempering",
        lambda *a, **kw: pytest.fail("dry-run must not sample"))
    root = tmp_path / "dry"
    result = main(["--output", str(root), "--dimension", "3", "--length", "3",
                   "--temperatures", "1", "3", "--disorders", "3", "--samples", "4",
                   "--burn-in", "0", "--spacing", "1", "--dry-run"])
    assert result["dimension"] == 3 and result["sites"] == 27
    assert result["total_single_spin_visits"] == 3 * 2 * 2 * 4 * 27
    assert not root.exists()
    assert json.loads(capsys.readouterr().out)["dry_run"]
    with pytest.raises(SystemExit):
        main(["--output", str(root), "--dimension", "3", "--shape", "3", "3",
              "--temperatures", "1", "3", "--dry-run"])


def test_no_overwrite_existing_dataset(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "user-data.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        generate_dataset(root, tiny())
    assert sentinel.read_text() == "preserve"
    assert list(root.iterdir()) == [sentinel]


def test_sha_and_semantic_energy_validation(tmp_path):
    root = tmp_path / "new"
    generate_dataset(root, tiny())
    path = root / "shards/disorder_000000.npz"
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    data["energies"] = data["energies"] + .125
    np.savez(path, **data)
    with pytest.raises(ValueError, match="SHA256"):
        list(iter_spin_glass(root))
    with pytest.raises(ValueError, match="energies"):
        list(iter_spin_glass(root, verify_hashes=False))


def test_failed_shard_preserves_completed_data_and_does_not_resume(tmp_path, monkeypatch):
    original = pipeline.sample_parallel_tempering
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic sampling failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "sample_parallel_tempering", failing)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic"):
        generate_dataset(root, tiny())
    assert len(calls) == 2
    assert (root / "shards/disorder_000000.npz").exists()
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and not status["workflow_complete"]
    assert status["completed_disorders"] == 1 and status["requires_user_action"]
    with pytest.raises(ValueError, match="not complete"):
        list(iter_spin_glass(root))


def test_diagnostic_limits_are_explicit():
    assert series_diagnostics(np.ones(8))["ess"] is None
    assert series_diagnostics(np.arange(3))["ess"] is None
    for invalid in ([], [[1, 2]], [1, np.nan]):
        with pytest.raises(ValueError):
            series_diagnostics(invalid)
    plan = generation_plan(tiny())
    assert plan["training_updates"] == 0 and not plan["labels_used_in_training"]


def test_actual_bond_collision_is_rejected_without_reseeding(tmp_path, monkeypatch):
    calls = []
    original = pipeline.sample_parallel_tempering
    def sampling(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "sample_parallel_tempering", sampling)
    monkeypatch.setattr(pipeline, "generate_bonds",
                        lambda shape, *a, **kw: np.ones((len(shape), *shape)))
    root = tmp_path / "collision"
    with pytest.raises(ValueError, match="Identical quenched bonds"):
        generate_dataset(root, tiny())
    assert len(calls) == 1  # No thermal sampling of the cross-split duplicate.
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_disorders"] == 1


def test_loader_checks_all_split_fingerprints_before_first_yield(tmp_path):
    root = tmp_path / "new"
    generate_dataset(root, tiny())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][1]["bond_sha256"] = manifest["shards"][0]["bond_sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaked"):
        next(iter_spin_glass(root, split="train", verify_hashes=False))


def test_seed_namespaces_do_not_collide_within_dataset():
    seeds = [pipeline._seed(91, purpose, i, (3, 3)) for i in range(256)
             for purpose in ("quenched_bonds", "thermal_ladders")]
    assert len(set(seeds)) == len(seeds)


def test_cli_tiny_execution_is_not_legacy_training_data(tmp_path, capsys):
    root = tmp_path / "cli"
    manifest = main(["--output", str(root), "--shape", "3", "3",
        "--temperatures", "1", "3", "--disorders", "3", "--samples", "4",
        "--burn-in", "0", "--spacing", "1", "--distribution", "gaussian",
        "--bond-mean", ".25"])
    assert manifest["options"]["shape"] == (3, 3)
    assert manifest["options"]["distribution"] == "gaussian"
    assert len(list(iter_spin_glass(root))) == 3
    assert not (root / "ensembles.pt").exists()
    assert json.loads(capsys.readouterr().out)["training_updates"] == 0
