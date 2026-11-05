"""Tiny newly generated data only: label boundary, split and integrity tests."""
import json

import numpy as np
import pytest

from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.ising3d import Wolff3DOptions, generate_dataset, sha256_file, write_json
from scripts.prepare_ising3d_qcr import prepare_cache


def small_options():
    return Wolff3DOptions(size=3, couplings=(.1, .22, .34), chains=4,
        samples=8, burn_in_clusters=8, clusters_between=2)


def test_adapter_keeps_all_physical_chains_but_no_coupling_training_labels(tmp_path):
    source, output = tmp_path / "physical", tmp_path / "cache"
    generate_dataset(source, small_options())
    metadata = prepare_cache(source, output)
    train, test = load_ensembles(output / "ensembles.pt")
    assert train.configurations.shape == (8, 8, 3, 3, 3, 1)
    assert test.configurations.shape == (4, 8, 3, 3, 3, 1)
    assert train.couplings is None and test.couplings is None
    assert train.group_ids.tolist() == [0] * 4 + [2] * 4
    assert test.group_ids.tolist() == [1] * 4
    assert [row["coupling_index"] for row in metadata["train"]] == train.group_ids.tolist()
    assert [row["K"] for row in metadata["test"]] == [.22] * 4
    assert metadata["dimension"] == 3 and metadata["lattice_shape"] == [3] * 3
    assert not metadata["labels_used_in_training"] and not metadata["couplings_in_training_cache"]
    assert metadata["independent_physical_chains_per_K"] == 4
    assert metadata["cache_sha256"] == sha256_file(output / "ensembles.pt")
    with pytest.raises(FileExistsError):
        prepare_cache(source, output)


@pytest.mark.parametrize("bad", ("incomplete", "manifest", "plan", "shard", "sidecar", "split"))
def test_adapter_rejects_incomplete_tampered_or_leaking_source_before_output(tmp_path, bad):
    source, output = tmp_path / "physical", tmp_path / "cache"
    manifest = generate_dataset(source, small_options())
    if bad == "incomplete":
        status = json.loads((source / "status.json").read_text())
        status["workflow_complete"] = False
        write_json(source / "status.json", status)
    elif bad == "manifest":
        manifest["dimension"] = 2
        write_json(source / "manifest.json", manifest)
    elif bad == "plan":
        (source / "plan.json").write_text("{}")
    elif bad in ("shard", "sidecar"):
        target = source / manifest["shards"][0]["path" if bad == "shard" else "sidecar"]
        target.write_bytes(target.read_bytes() + b"tampered")
    elif bad == "split":
        manifest["shards"][1]["split"] = "train"
        write_json(source / "manifest.json", manifest)
        status = json.loads((source / "status.json").read_text())
        status["manifest_sha256"] = sha256_file(source / "manifest.json")
        write_json(source / "status.json", status)
    with pytest.raises(ValueError):
        prepare_cache(source, output)
    assert not output.exists()


def _reseal(source, manifest, shard, sidecar):
    """Update hashes so semantic checks, not only outer fingerprints, are tested."""
    shard["sha256"] = sha256_file(source / shard["path"])
    sidecar["data_sha256"] = shard["sha256"]
    write_json(source / shard["sidecar"], sidecar)
    shard["sidecar_sha256"] = sha256_file(source / shard["sidecar"])
    write_json(source / "manifest.json", manifest)
    status = json.loads((source / "status.json").read_text())
    status["manifest_sha256"] = sha256_file(source / "manifest.json")
    write_json(source / "status.json", status)


@pytest.mark.parametrize("bad", ("energy", "magnetization", "nonbinary", "cluster_size",
    "chain_work", "thermal_seed", "initial", "raw_claim", "coupling"))
def test_adapter_semantic_physics_and_seed_checks_even_after_hashes_resealed(tmp_path, bad):
    source, output = tmp_path / "physical", tmp_path / "cache"
    manifest = generate_dataset(source, small_options())
    shard = manifest["shards"][0]
    sidecar = json.loads((source / shard["sidecar"]).read_text())
    with np.load(source / shard["path"], allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    if bad == "energy":
        values["raw_observables"][0, 0, 0] += .125
    elif bad == "magnetization":
        values["raw_observables"][0, 0, 1] += .125
    elif bad == "nonbinary":
        values["configurations"][0, 0, 0, 0, 0] = 0
    elif bad == "cluster_size":
        values["cluster_sizes"][0, 0] = 0
    elif bad == "chain_work":
        sidecar["chains"][0]["cluster_visited_sites"] += 1
    elif bad == "thermal_seed":
        sidecar["chains"][0]["thermal_seed"] += 1
    elif bad == "initial":
        sidecar["chains"][0]["initial"] = "cold"
    elif bad == "raw_claim":
        sidecar["raw_diagnostics_before_global_z2"] = False
    elif bad == "coupling":
        sidecar["K"] += .01
    np.savez(source / shard["path"], **values)
    _reseal(source, manifest, shard, sidecar)
    with pytest.raises(ValueError):
        prepare_cache(source, output)
    assert not output.exists()
