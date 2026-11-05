"""Synthetic hopping shards only; no physical sampling, training, or SSH."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.data import phi4_hopping_loader as loader


def _write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _semantics():
    return dict(schema=loader.SCHEMA, parameterization=loader.PARAMETERIZATION,
        evaluation_label_columns=list(loader.EVALUATION_LABEL_COLUMNS),
        field_representation="phi", internal_sampling_field="psi=sqrt(kappa)*phi",
        labels_used_in_training=False)


def _refresh(root, *, sidecar=None, plan=False):
    manifest = _read(root / "manifest.json")
    if sidecar is not None:
        shard = manifest["shards"][sidecar]
        shard["sidecar_sha256"] = loader._sha256_file(root / shard["sidecar"])
    if plan:
        manifest["plan_sha256"] = loader._sha256_file(root / "plan.json")
    _write(root / "manifest.json", manifest)
    status = _read(root / "status.json")
    status["manifest_sha256"] = loader._sha256_file(root / "manifest.json")
    _write(root / "status.json", status)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "hopping"
    root.mkdir()
    options = dict(dimension=2, sizes=[9], chains=4, samples=4,
        kappa_values=[.58, .68, .78], quartic_lambda=1., h=0.,
        test_every=3, test_offset=1)
    pins = {"new_sampler.py": "a" * 64, "old_sampler.py": "b" * 64}
    plan = dict(**_semantics(), schema_version=5, options=options,
        source_pins=pins, train_cell_indices=[0, 2], test_cell_indices=[1])
    _write(root / "plan.json", plan)
    shards = []
    for cell, kappa in enumerate(options["kappa_values"]):
        directory = root / f"c{cell}"
        directory.mkdir()
        fields = np.random.default_rng(cell + 971).normal(size=(4, 4, 9, 9, 1))
        if cell == 1:
            fields *= 100.  # TEST amplitude must not enter the TRAIN RMS.
        np.savez(directory / "data.npz", configurations=fields,
            raw_observables=np.full((4, 4, 8), np.nan),
            arbitrary_object_diagnostic=np.array([{"do_not_unpickle": True}], dtype=object))
        data_sha = loader._sha256_file(directory / "data.npz")
        split = "test" if cell == 1 else "train"
        sidecar = dict(**_semantics(), status="complete", cell_index=cell,
            size=9, dimension=2, kappa=kappa, quartic_lambda=1., h=0., split=split,
            source_pins=pins, independent_chains=4, samples_per_chain=4,
            data_sha256=data_sha)
        _write(directory / "sidecar.json", sidecar)
        shards.append(dict(cell_index=cell, size=9, split=split,
            path=f"c{cell}/data.npz", sha256=data_sha,
            sidecar=f"c{cell}/sidecar.json",
            sidecar_sha256=loader._sha256_file(directory / "sidecar.json")))
    manifest = dict(**_semantics(), status="complete", options=options, shards=shards,
        source_pins=pins, plan_sha256=loader._sha256_file(root / "plan.json"),
        warning_count=13, sample_semantics="independent_correlated_hybrid_markov_chains",
        qualification="Synthetic fixture only; not equilibrium certified")
    _write(root / "manifest.json", manifest)
    _write(root / "status.json", dict(schema=loader.SCHEMA, status="complete",
        workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        manifest_sha256=loader._sha256_file(root / "manifest.json")))
    return root


def test_loads_original_phi_only_and_keeps_kappa_labels_explicit(data_root, monkeypatch):
    original = np.load
    accessed = []
    class Archive:
        def __init__(self, path, **kwargs):
            assert kwargs == {"allow_pickle": False}
            self.archive = original(path, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            accessed.append(key)
            assert key == "configurations"
            return self.archive[key]
    monkeypatch.setattr(loader.np, "load", Archive)
    data = loader.load_dataset(data_root)
    assert accessed == ["configurations"] * 3
    assert data.configurations.shape == (12, 4, 9, 9, 1)
    assert data.configurations.dtype == torch.float64
    assert data.labels.shape == (12, 3)
    assert data.parameterization == "hopping_kappa_quartic_lambda"
    assert data.evaluation_label_columns == ("kappa", "quartic_lambda", "h")
    assert data.field_representation == "phi" and data.sampler_schema == loader.SCHEMA
    assert data.warning_count == 13
    np.testing.assert_array_equal(data.labels[:, 0], np.repeat([.58, .68, .78], 4))
    assert set(data.cell_ids[data.train_indices]) == {0, 2}
    assert set(data.cell_ids[data.test_indices]) == {1}
    assert data.scale == pytest.approx(float(data.configurations[data.train_indices].square().mean().sqrt()))
    assert .5 < data.scale < 2.


@pytest.mark.parametrize("file", ["manifest.json", "plan.json", "c0/data.npz", "c0/sidecar.json"])
def test_rejects_any_bound_file_hash_change(data_root, file):
    with (data_root / file).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        loader.load_dataset(data_root)


@pytest.mark.parametrize("changes", [
    {"schema": "euclidean_phi4_hmc_embedded_wolff_v1"}, {"status": "running"},
    {"workflow_complete": False}, {"returncode": 1}, {"returncode": False},
    {"error": "failure"}, {"requires_user_action": True},
])
def test_rejects_unknown_or_unsuccessful_execution(data_root, changes):
    status = _read(data_root / "status.json")
    status.update(changes)
    _write(data_root / "status.json", status)
    with pytest.raises(ValueError, match="successfully complete"):
        loader.load_dataset(data_root)


@pytest.mark.parametrize("target,changes", [
    ("manifest.json", {"parameterization": "r_u_h"}),
    ("manifest.json", {"evaluation_label_columns": ["r", "u", "h"]}),
    ("manifest.json", {"field_representation": "psi"}),
    ("manifest.json", {"labels_used_in_training": True}),
    ("c0/sidecar.json", {"kappa": .59}),
    ("c0/sidecar.json", {"quartic_lambda": 2.}),
    ("c0/sidecar.json", {"split": "test"}),
    ("c0/sidecar.json", {"field_representation": "psi"}),
    ("c0/sidecar.json", {"independent_chains": 8}),
    ("c0/sidecar.json", {"r": .58}),
])
def test_rejects_rebound_incorrect_semantics_or_provenance(data_root, target, changes):
    value = _read(data_root / target)
    value.update(changes)
    _write(data_root / target, value)
    _refresh(data_root, sidecar=0 if target.startswith("c0/") else None)
    with pytest.raises(ValueError):
        loader.load_dataset(data_root)


@pytest.mark.parametrize("changes", [
    {"dimension": 1}, {"sizes": [9, 27]}, {"sizes": [10]}, {"chains": 8},
    {"samples": 5}, {"kappa_values": [0., .68, .78]},
    {"kappa_values": [.58, .58, .78]}, {"quartic_lambda": 0.},
    {"h": .1}, {"test_every": 1}, {"test_offset": 3}, {"r_values": [.58, .68, .78]},
])
def test_rejects_invalid_geometry_physics_or_legacy_parameter_aliases(data_root, changes):
    plan, manifest = _read(data_root / "plan.json"), _read(data_root / "manifest.json")
    plan["options"].update(changes)
    manifest["options"].update(changes)
    _write(data_root / "plan.json", plan)
    _write(data_root / "manifest.json", manifest)
    _refresh(data_root, plan=True)
    with pytest.raises(ValueError):
        loader.load_dataset(data_root)


@pytest.mark.parametrize("path", ["../outside.npz", "/tmp/outside.npz", ""])
def test_rejects_paths_outside_root(data_root, path):
    manifest = _read(data_root / "manifest.json")
    manifest["shards"][0]["path"] = path
    _write(data_root / "manifest.json", manifest)
    _refresh(data_root)
    with pytest.raises(ValueError, match="path"):
        loader.load_dataset(data_root)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "bad_index"])
def test_rejects_incomplete_or_duplicated_kappa_cells(data_root, mode):
    manifest = _read(data_root / "manifest.json")
    if mode == "missing":
        manifest["shards"].pop()
    elif mode == "duplicate":
        manifest["shards"][2] = manifest["shards"][0]
    else:
        manifest["shards"][2]["cell_index"] = 99
    _write(data_root / "manifest.json", manifest)
    _refresh(data_root)
    with pytest.raises(ValueError):
        loader.load_dataset(data_root)


def test_rejects_legacy_loader_instead_of_silently_relabeling_kappa(data_root):
    from experiments.phi4_qcr import load_dataset as legacy_loader
    with pytest.raises(KeyError, match="r_values"):
        legacy_loader(data_root)


@pytest.mark.parametrize("mode", ["float32", "wrong_shape", "nan", "zero_train"])
def test_rejects_invalid_original_phi_arrays(data_root, mode):
    manifest = _read(data_root / "manifest.json")
    for cell in ([0, 2] if mode == "zero_train" else [0]):
        shard = manifest["shards"][cell]
        path = data_root / shard["path"]
        value = np.ones((4, 4, 9, 9, 1), dtype=np.float64)
        if mode == "float32":
            value = value.astype(np.float32)
        elif mode == "wrong_shape":
            value = value[..., 0]
        elif mode == "nan":
            value.flat[0] = np.nan
        else:
            value.fill(0.)
        np.savez(path, configurations=value)
        digest = loader._sha256_file(path)
        shard["sha256"] = digest
        sidecar_path = data_root / shard["sidecar"]
        metadata = _read(sidecar_path)
        metadata["data_sha256"] = digest
        _write(sidecar_path, metadata)
        shard["sidecar_sha256"] = loader._sha256_file(sidecar_path)
    _write(data_root / "manifest.json", manifest)
    _refresh(data_root)
    with pytest.raises(ValueError):
        loader.load_dataset(data_root)
