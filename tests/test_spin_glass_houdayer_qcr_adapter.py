"""Synthetic Houdayer-envelope adaptation; no production chain or training."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.spin_glass import energy
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, SCHEMA, _bond_sha, _json, _sha
from scripts.prepare_spin_glass_houdayer_qcr import prepare_overlap_cache
from scripts.report_spin_glass2d import _validate_metadata


def synthetic_source(tmp_path, *, samples=8):
    """Exact-energy random arrays exercise serialization, not physics evidence."""
    root = tmp_path / "synthetic_houdayer_source"
    (root / "shards").mkdir(parents=True)
    options = GenerationOptions(shape=(3, 3), temperatures=(.8, 2.), disorders=3,
        replicas=4, samples=samples, burn_in=0, spacing=1)
    sampler = dict(schema="houdayer_parallel_tempering_v1", algorithm="houdayer_pt",
        initial="hot", replicas=4, real_replicas_coupled=True, kernel_source_sha256="c" * 64,
        cluster_interval=1, exchange_interval=1,
        pairing="uniform random disjoint matching independently at each temperature and cluster event",
        target_joint_distribution="product of identical-Hamiltonian Boltzmann distributions at each temperature (stationary target, not equilibrium certification)")
    _json(root / "plan.json", dict(schema=SCHEMA, sampler=sampler, synthetic_fixture=True))
    _json(root / "sampling.json", dict(schema=SCHEMA, synthetic_fixture=True, equilibrium_certified=False))
    rng, originals, shards = np.random.default_rng(918), {}, []
    for disorder in range(3):
        bonds = np.ones((2, 3, 3))
        bonds[0, 0, disorder] = -1
        spins = rng.choice(np.array([-1, 1], np.int8), size=(2, 4, samples, 3, 3))
        # Intentionally force other pair statistics to differ; only pair0/1 is cached.
        spins[:, 2] = 1
        spins[:, 3] = -1
        energies = energy(spins, bonds)
        path = root / "shards" / f"disorder_{disorder:06d}.npz"
        np.savez(path, configurations=spins, bonds=bonds, energies=energies,
            temperatures=np.asarray(options.temperatures), disorder_id=np.array(disorder))
        originals[disorder] = spins
        shards.append(dict(disorder_id=disorder, split="test" if disorder == 1 else "train",
            path=str(path.relative_to(root)), sha256=_sha(path), bytes=path.stat().st_size,
            bond_sha256=_bond_sha(bonds)))
    manifest = dict(schema=SCHEMA, status="complete", options=asdict(options), shards=shards,
        plan_sha256=_sha(root / "plan.json"), sampling_sha256=_sha(root / "sampling.json"),
        sampler=sampler,
        equilibrium_certified=False)
    _json(root / "manifest.json", manifest)
    _json(root / "status.json", dict(schema=SCHEMA, status="complete", workflow_complete=True,
        manifest_sha256=_sha(root / "manifest.json"), equilibrium_certified=False))
    return root, manifest, originals


def update_manifest(source, manifest):
    plan = json.loads((source / "plan.json").read_text())
    plan["sampler"] = manifest.get("sampler")
    _json(source / "plan.json", plan)
    manifest["plan_sha256"] = _sha(source / "plan.json")
    _json(source / "manifest.json", manifest)
    status = json.loads((source / "status.json").read_text())
    status["manifest_sha256"] = _sha(source / "manifest.json")
    _json(source / "status.json", status)


@pytest.mark.parametrize("samples", (8, 512))
def test_houdayer_four_replica_adapter_only_pair01_no_labels_and_report_compatible(tmp_path, samples):
    source, manifest, originals = synthetic_source(tmp_path, samples=samples)
    output = tmp_path / "cache"
    metadata = prepare_overlap_cache(source, output)
    train, test = load_ensembles(output / "ensembles.pt")
    assert train.configurations.shape == (4, samples, 3, 3, 1)
    assert test.configurations.shape == (2, samples, 3, 3, 1)
    assert train.couplings is None and test.couplings is None
    assert train.group_ids.tolist() == [[0, 0], [0, 1], [2, 0], [2, 1]]
    assert test.group_ids.tolist() == [[1, 0], [1, 1]]
    for dataset in (train, test):
        for group, actual in zip(dataset.group_ids.tolist(), dataset.configurations):
            disorder, temperature = group
            expected = originals[disorder][temperature, 0] * originals[disorder][temperature, 1]
            np.testing.assert_array_equal(actual[..., 0].numpy(), expected)
    assert metadata["real_replica_count"] == 4
    assert metadata["overlap_pair"] == [0, 1]
    assert metadata["real_replicas_coupled_by_houdayer"]
    assert metadata["replica_pairs_are_not_independent"]
    assert metadata["training_ensemble_count_per_disorder_and_temperature"] == 1
    assert not metadata["source_thermal_independence_certified"]
    assert not metadata["labels_used_in_training"] and not metadata["equilibrium_certified"]
    assert metadata["cache_sha256"] == _sha(output / "ensembles.pt")
    assert all(row["samples"] == samples and row["overlap_replicas"] == [0, 1]
        for split in ("train", "test") for row in metadata[split])
    _validate_metadata([("train", train), ("test", test)], metadata)
    with pytest.raises(FileExistsError):
        prepare_overlap_cache(source, output)


@pytest.mark.parametrize("bad", ("pt", "no_receipt", "ordered", "independent_claim", "replica_count", "field", "equilibrium_claim", "incomplete", "tampered_source", "kernel_sha", "sampler_schema"))
def test_houdayer_adapter_rejects_wrong_or_unverified_sampler_contract(tmp_path, bad):
    source, manifest, _ = synthetic_source(tmp_path)
    if bad == "pt":
        manifest["sampler"]["algorithm"] = "pt"
    elif bad == "no_receipt":
        del manifest["sampler"]
    elif bad == "ordered":
        manifest["sampler"]["initial"] = "cold"
    elif bad == "independent_claim":
        manifest["sampler"]["real_replicas_coupled"] = False
    elif bad == "replica_count":
        manifest["sampler"]["replicas"] = 2
    elif bad == "field":
        manifest["options"]["field"] = .1
    elif bad == "equilibrium_claim":
        manifest["equilibrium_certified"] = True
    elif bad == "incomplete":
        manifest["status"] = "running"
    elif bad == "tampered_source":
        path = source / manifest["shards"][0]["path"]
        path.write_bytes(path.read_bytes() + b"corruption")
    elif bad == "kernel_sha":
        manifest["sampler"]["kernel_source_sha256"] = "invalid"
    elif bad == "sampler_schema":
        manifest["sampler"]["schema"] = "unsupported"
    update_manifest(source, manifest)
    output = tmp_path / "cache"
    with pytest.raises(ValueError):
        prepare_overlap_cache(source, output)
    assert not output.exists()


def test_houdayer_adapter_preserves_whole_disorder_split_checks(tmp_path):
    source, manifest, _ = synthetic_source(tmp_path)
    manifest["shards"][1]["split"] = "train"
    update_manifest(source, manifest)
    with pytest.raises(ValueError, match="split"):
        prepare_overlap_cache(source, tmp_path / "cache")


def test_houdayer_adapter_rejects_plan_manifest_sampler_mismatch(tmp_path):
    source, manifest, _ = synthetic_source(tmp_path)
    plan = json.loads((source / "plan.json").read_text())
    plan["sampler"]["cluster_interval"] = 2
    _json(source / "plan.json", plan)
    # The dedicated sampler-contract check precedes general plan hash checks.
    with pytest.raises(ValueError, match="differs between frozen plan"):
        prepare_overlap_cache(source, tmp_path / "cache")


def test_houdayer_adapter_rejects_odd_saved_count_before_cache(tmp_path):
    source, _, _ = synthetic_source(tmp_path, samples=5)
    with pytest.raises(ValueError, match="even saved sample"):
        prepare_overlap_cache(source, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
