"""Strict configuration-only loader for the hopping-parameter scalar dataset.

This is deliberately separate from ``experiments.phi4_qcr.load_dataset``:
evaluation labels here are (kappa, quartic_lambda, h), never (r, u, h).  Arrays
contain the original field phi, not the condition-dependently rescaled sampling
field psi=sqrt(kappa)*phi.  Only ``configurations`` is decompressed; all physical
diagnostics and coupling labels remain provenance/evaluation information.

The returned tensors can be used by continuous-field QCR components, but this
module does not launch training or make historical r-axis reports compatible.
Such reports need an explicitly kappa-aware entry point.  The present training
geometry defaults to four independent chains at one 2D size with two b=3
steps. Explicit geometry arguments admit requested block/depth ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import re

import numpy as np
import torch


SCHEMA = "euclidean_phi4_hopping_hmc_embedded_wolff_v1"
PARAMETERIZATION = "hopping_kappa_quartic_lambda"
EVALUATION_LABEL_COLUMNS = ("kappa", "quartic_lambda", "h")
FIELD_REPRESENTATION = "phi"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset JSON metadata must be an object")
    return value


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Data manifest path must be a nonempty relative path")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Data manifest path escapes its root")
    return path


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(result)


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256 metadata: {name}")
    return value


def _semantics(metadata, *, label):
    if (metadata.get("schema") != SCHEMA
            or metadata.get("parameterization") != PARAMETERIZATION
            or metadata.get("evaluation_label_columns") != list(EVALUATION_LABEL_COLUMNS)
            or metadata.get("field_representation") != FIELD_REPRESENTATION):
        raise ValueError(f"Incompatible hopping parameter/field/schema semantics in {label}")
    if metadata.get("labels_used_in_training") is not False:
        raise ValueError(f"Coupling labels must be evaluation-only in {label}")


@dataclass
class HoppingFieldDataset:
    configurations: torch.Tensor
    splits: np.ndarray
    cell_ids: np.ndarray
    labels: np.ndarray
    manifest_sha256: str
    receipts: list
    warning_count: int
    qualification: str
    sampler_schema: str = SCHEMA
    sample_semantics: str = "independent_correlated_hybrid_markov_chains"
    evaluation_label_columns: tuple[str, ...] = EVALUATION_LABEL_COLUMNS
    parameterization: str = PARAMETERIZATION
    field_representation: str = FIELD_REPRESENTATION

    @property
    def train_indices(self):
        return np.flatnonzero(self.splits == "train")

    @property
    def test_indices(self):
        return np.flatnonzero(self.splits == "test")

    @property
    def scale(self):
        """TRAIN-only original-phi RMS, without coupling-dependent rescaling."""
        return float(self.configurations[self.train_indices].square().mean().sqrt())


def load_dataset(root, *, block_size=3, depth=2):
    """Validate a completed new-schema dataset and read only configurations.

    Root status binds manifest SHA; manifest binds plan and every shard/sidecar
    SHA.  Whole kappa cells, including all four physical chains, are assigned
    to the predeclared TRAIN/TEST split.  A finite-chain quality warning is not
    silently discarded or promoted into an equilibrium certification.
    """
    block_size = _integer(block_size, "block_size", 2)
    depth = _integer(depth, "depth", 1)
    root = Path(root).resolve()
    status = _json(root / "status.json")
    if (status.get("schema") != SCHEMA or status.get("status") != "complete"
            or status.get("workflow_complete") is not True
            or type(status.get("returncode")) is not int or status.get("returncode") != 0
            or status.get("error") is not None
            or status.get("requires_user_action") is not False):
        raise ValueError("Hopping physical dataset is not successfully complete")
    manifest_path = root / "manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != _sha(status.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("Physical manifest SHA256 mismatch")
    manifest = _json(manifest_path)
    _semantics(manifest, label="manifest")
    if manifest.get("status") != "complete":
        raise ValueError("Hopping physical manifest is not complete")
    plan_path = root / "plan.json"
    if _sha256_file(plan_path) != _sha(manifest.get("plan_sha256"), "plan_sha256"):
        raise ValueError("Physical plan SHA256 mismatch")
    plan = _json(plan_path)
    _semantics(plan, label="plan")
    if plan.get("schema_version") != 5:
        raise ValueError("Hopping data plan requires schema_version=5")
    options = manifest.get("options")
    if not isinstance(options, dict) or options != plan.get("options"):
        raise ValueError("Hopping plan and manifest options disagree")
    if "r_values" in options or "u_values" in options:
        raise ValueError("Hopping couplings must not be disguised as legacy r/u labels")
    pins = manifest.get("source_pins")
    if not isinstance(pins, dict) or not pins or pins != plan.get("source_pins"):
        raise ValueError("Hopping plan and manifest source pins disagree")
    for name, digest in pins.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid source pin name")
        _sha(digest, name)
    if _integer(options.get("dimension"), "dimension", 1) != 2:
        raise ValueError("This continuous QCR loader requires two spatial dimensions")
    sizes = options.get("sizes")
    if not isinstance(sizes, list) or len(sizes) != 1:
        raise ValueError("This continuous QCR loader requires one microscopic size")
    side = _integer(sizes[0], "size", 3)
    if side % (block_size**depth) or ((block_size, depth) != (3, 2) and side // (block_size**depth) < 3):
        raise ValueError("Microscopic side must support the explicitly requested nonoverlapping geometry with final side>=3")
    chains = _integer(options.get("chains"), "chains", 1)
    samples = _integer(options.get("samples"), "samples", 4)
    if chains != 4 or samples % 2:
        raise ValueError("Require four independent chains and an even saved-sample count")
    kappas = options.get("kappa_values")
    if not isinstance(kappas, list) or len(kappas) < 2:
        raise ValueError("Require a nonempty hopping scan with TRAIN and TEST cells")
    kappas = [_finite(value, "kappa") for value in kappas]
    if any(value <= 0. for value in kappas) or len(set(kappas)) != len(kappas):
        raise ValueError("Hopping scan requires distinct positive kappa values")
    quartic_lambda = _finite(options.get("quartic_lambda"), "quartic_lambda")
    h = _finite(options.get("h"), "h")
    if quartic_lambda <= 0. or h != 0.:
        raise ValueError("Hopping loader requires fixed quartic_lambda>0 and h=0")
    test_every = _integer(options.get("test_every"), "test_every", 2)
    test_offset = _integer(options.get("test_offset"), "test_offset", 0)
    if test_offset >= test_every:
        raise ValueError("Invalid whole-cell split offset")
    expected_cells = set(range(len(kappas)))
    expected_test = [cell for cell in sorted(expected_cells) if cell % test_every == test_offset]
    expected_train = [cell for cell in sorted(expected_cells) if cell % test_every != test_offset]
    if not expected_test or not expected_train:
        raise ValueError("Whole-cell split must retain both TRAIN and TEST cells")
    if (plan.get("train_cell_indices") != expected_train
            or plan.get("test_cell_indices") != expected_test):
        raise ValueError("Plan whole-cell indices disagree with split options")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(expected_cells):
        raise ValueError("Incomplete physical hopping grid")
    configurations, splits, cells, labels, receipts = [], [], [], [], []
    seen, seen_paths = set(), set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Physical shard metadata must be an object")
        cell = _integer(shard.get("cell_index"), "cell_index")
        if cell not in expected_cells or cell in seen or shard.get("size") != side:
            raise ValueError("Duplicate or unexpected microscopic hopping shard")
        seen.add(cell)
        data_path = _inside(root, shard.get("path"))
        sidecar_path = _inside(root, shard.get("sidecar"))
        if data_path in seen_paths or sidecar_path in seen_paths or data_path == sidecar_path:
            raise ValueError("Physical shard and sidecar paths must be unique")
        seen_paths.update((data_path, sidecar_path))
        data_sha, sidecar_sha = _sha256_file(data_path), _sha256_file(sidecar_path)
        if (data_sha != _sha(shard.get("sha256"), "shard.sha256")
                or sidecar_sha != _sha(shard.get("sidecar_sha256"), "sidecar_sha256")):
            raise ValueError("Physical shard/sidecar SHA256 mismatch")
        metadata = _json(sidecar_path)
        _semantics(metadata, label="sidecar")
        split = "test" if cell % test_every == test_offset else "train"
        if (metadata.get("status") != "complete" or metadata.get("cell_index") != cell
                or metadata.get("size") != side or metadata.get("dimension") != 2
                or metadata.get("data_sha256") != data_sha
                or metadata.get("split") != split or shard.get("split") != split
                or metadata.get("source_pins") != pins
                or metadata.get("independent_chains") != chains
                or metadata.get("samples_per_chain") != samples):
            raise ValueError("Inconsistent hopping whole-cell split or shard provenance")
        if "r" in metadata or "u" in metadata:
            raise ValueError("Hopping sidecar must not relabel kappa as r/u")
        expected_label = [kappas[cell], quartic_lambda, h]
        actual_label = [_finite(metadata.get(name), name) for name in EVALUATION_LABEL_COLUMNS]
        if actual_label != expected_label:
            raise ValueError("Sidecar hopping evaluation labels disagree with the manifest")
        with np.load(data_path, allow_pickle=False) as archive:
            value = archive["configurations"]
        if (value.shape != (chains, samples, side, side, 1)
                or value.dtype != np.float64 or not np.isfinite(value).all()):
            raise ValueError("Invalid original-phi continuous configuration array")
        configurations.append(torch.from_numpy(np.ascontiguousarray(value)))
        splits.extend([split] * chains)
        cells.extend([cell] * chains)
        labels.extend([actual_label] * chains)
        receipts.append(dict(path=shard["path"], sha256=data_sha,
            sidecar=shard["sidecar"], sidecar_sha256=sidecar_sha))
    if seen != expected_cells:
        raise ValueError("Incomplete physical hopping grid")
    warning_count = _integer(manifest.get("warning_count", 0), "warning_count")
    dataset = HoppingFieldDataset(torch.cat(configurations, dim=0), np.asarray(splits),
        np.asarray(cells, dtype=np.int64), np.asarray(labels, dtype=np.float64),
        manifest_sha, receipts, warning_count,
        str(manifest.get("qualification", "Finite chains do not certify equilibration or physical RG.")),
        sample_semantics=str(manifest.get("sample_semantics", "independent_correlated_hybrid_markov_chains")))
    if not math.isfinite(dataset.scale) or dataset.scale <= 0.:
        raise ValueError("TRAIN original-phi RMS must be finite and nonzero")
    return dataset
