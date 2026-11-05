"""Persistent CPU cache for *data-only* empirical mutual-information statistics.

This cache never stores a kernel, optimizer, or RNG state.  The caller supplies
the actual data-file SHA256 and every data/region convention used by its build
function; model parameters deliberately do not participate in the key.  The
training entry point remains responsible for moving the returned statistics to
the model device once.

Cache files contain only basic dictionaries and tensors, not pickled statistics
classes, and are loaded with ``weights_only=True``.  A corrupt existing file is
an explicit error: it is never silently overwritten.  Concurrent builders use
unique temporary files and an atomic no-clobber publication, so readers only
ever observe completed cache files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping

import torch

from .empirical_mi import EmpiricalBinaryMI, LocalPairMI
from .tensor_io import all_finite, load_cpu_tensors, tensor_chunks


# Bump when the statistics implementation or serialization convention changes.
# Existing cache files and historical training artifacts remain untouched.
MI_STATISTICS_CACHE_VERSION = 1


def lattice_statistics_cache_key(config, data_sha256, split, configurations,
                                 environment_offsets, statistics_name,
                                 statistics_kwargs, *, environment='full_shell'):
    """Shared identity for trainer and standalone data-only preprocessing.

    Keep the original v1 key unchanged: execution-only support indices do not
    alter empirical counts, the statistic, or its serialized payload.
    """
    return dict(data_sha256=data_sha256, split=split,
                states=config.model.categorical_states or 2,
                dimension=config.model.dimension, block=config.model.block_factor,
                buffer=1, environment_offsets=environment_offsets.tolist(),
                flip=config.data.spin_flip_augment,
                pooling=(config.data.train_pool_replicas if split == 'train'
                         else config.data.test_pool_replicas),
                dtype=str(configurations.dtype), shape=list(configurations.shape),
                environment=environment, statistics=statistics_name,
                statistics_kwargs=statistics_kwargs)


class MIStatisticsCacheError(RuntimeError):
    """An existing statistics cache cannot be safely or consistently reused."""


_REQUIRED_KEY_FIELDS = {
    "data_sha256", "split", "states", "dimension", "block", "buffer",
    "environment_offsets", "flip", "pooling", "dtype",
}


def _plain(value, *, location):
    """Accept only unambiguous JSON primitives (no tensors/model objects)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain a non-finite value")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, location=f"{location}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{location} must have string dictionary keys")
        return {key: _plain(item, location=f"{location}.{key}")
                for key, item in value.items()}
    raise TypeError(f"{location} must contain only JSON primitives, not {type(value).__name__}")


def _cache_identity(key_data):
    if not isinstance(key_data, Mapping):
        raise TypeError("key_data must be a mapping of data/region conventions")
    normalized = _plain(dict(key_data), location="key_data")
    missing = _REQUIRED_KEY_FIELDS - normalized.keys()
    if missing:
        raise ValueError(f"Missing empirical MI cache key fields: {', '.join(sorted(missing))}")
    fingerprint = normalized["data_sha256"]
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            any(char not in "0123456789abcdefABCDEF" for char in fingerprint)):
        raise ValueError("data_sha256 must be the actual data-file SHA256 hex digest")
    normalized["data_sha256"] = fingerprint.lower()
    if not isinstance(normalized["split"], str) or not normalized["split"]:
        raise ValueError("split must be a nonempty name of the actual cached data split")
    for name, minimum in (("states", 2), ("dimension", 1), ("block", 1), ("buffer", 0)):
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(normalized["flip"], bool):
        raise ValueError("flip must be an explicit boolean")
    if not isinstance(normalized["dtype"], str) or not normalized["dtype"]:
        raise ValueError("dtype must be an explicit string")
    offsets = normalized["environment_offsets"]
    if offsets is not None:
        if (not isinstance(offsets, list) or not offsets or
                any(not isinstance(offset, list) or len(offset) != normalized["dimension"]
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in offset)
                    for offset in offsets)):
            raise ValueError("environment_offsets must be None or a nonempty list of integer offsets")
    identity = {"cache_version": MI_STATISTICS_CACHE_VERSION, "key_data": normalized}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return identity, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tensor_payload(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Empirical MI cache tensor fields must be torch tensors")
    if tensor.device.type != "cpu" or tensor.requires_grad:
        raise ValueError("Empirical MI caches only accept static CPU tensors without gradients")
    if tensor.layout == torch.strided:
        return {"layout": "strided", "values": tensor.detach().contiguous()}
    if tensor.layout == torch.sparse_coo:
        coalesced = tensor.coalesce()
        return {"layout": "sparse_coo", "indices": coalesced.indices(),
                "values": coalesced.values(), "size": list(coalesced.shape)}
    raise ValueError(f"Unsupported empirical MI cache tensor layout: {tensor.layout}")


def _tensor_from_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Invalid empirical MI cache tensor dictionary")
    layout = payload.get("layout")
    if layout == "strided" and set(payload) == {"layout", "values"}:
        tensor = payload["values"]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("Invalid dense empirical MI cache tensor")
    elif layout == "sparse_coo" and set(payload) == {"layout", "indices", "values", "size"}:
        indices, values, size = payload["indices"], payload["values"], payload["size"]
        if (not isinstance(indices, torch.Tensor) or indices.layout != torch.strided
                or indices.dtype != torch.long or indices.ndim != 2
                or not isinstance(values, torch.Tensor) or values.layout != torch.strided
                or values.ndim != 1 or not isinstance(size, list) or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in size)
                or indices.shape != (len(size), len(values))):
            raise ValueError("Invalid sparse empirical MI cache tensor")
        if any(any(bool(((part < 0) | (part >= width)).any())
                   for part in tensor_chunks(indices[axis]))
               for axis, width in enumerate(size)):
            raise ValueError("Sparse empirical MI cache indices exceed tensor dimensions")
        tensor = torch.sparse_coo_tensor(indices, values, tuple(size)).coalesce()
    else:
        raise ValueError("Invalid empirical MI cache tensor layout or fields")
    if tensor.device.type != "cpu" or tensor.requires_grad:
        raise ValueError("Empirical MI cache tensors must be static CPU tensors")
    values = tensor.values() if tensor.layout == torch.sparse_coo else tensor
    if not all_finite(values):
        raise ValueError("Empirical MI cache contains non-finite tensor values")
    return tensor


def _statistics_payload(statistics):
    if isinstance(statistics, EmpiricalBinaryMI):
        return {"kind": "empirical_binary", "conditional": _tensor_payload(statistics.conditional),
                "weights": _tensor_payload(statistics.weights),
                "marginal": _tensor_payload(statistics.marginal),
                "diagnostics": _plain(statistics.diagnostics, location="diagnostics")}
    if isinstance(statistics, LocalPairMI):
        return {"kind": "local_pair", "groups": [_statistics_payload(group)
                                                  for group in statistics.groups],
                "diagnostics": _plain(statistics.diagnostics, location="diagnostics")}
    if isinstance(statistics, torch.Tensor):
        return {"kind": "tensor", "tensor": _tensor_payload(statistics)}
    raise TypeError("Empirical MI caches support EmpiricalBinaryMI, LocalPairMI, or a static CPU tensor")


def _statistics_from_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Invalid empirical MI cache statistics dictionary")
    kind = payload.get("kind")
    if kind == "empirical_binary" and set(payload) == {
            "kind", "conditional", "weights", "marginal", "diagnostics"}:
        conditional = _tensor_from_payload(payload["conditional"])
        weights = _tensor_from_payload(payload["weights"])
        marginal = _tensor_from_payload(payload["marginal"])
        if (conditional.ndim != 2 or weights.ndim != 2 or marginal.ndim != 2
                or weights.layout != torch.sparse_coo or marginal.layout != torch.strided
                or conditional.shape[1] != marginal.shape[1]
                or weights.shape != (len(marginal), conditional.shape[0])
                or conditional.dtype != weights.dtype or conditional.dtype != marginal.dtype):
            raise ValueError("Empirical MI cache statistics dimensions/dtypes are inconsistent")
        diagnostics = _plain(payload["diagnostics"], location="diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Empirical MI cache diagnostics must be a dictionary")
        return EmpiricalBinaryMI(conditional, weights, marginal, diagnostics)
    if kind == "local_pair" and set(payload) == {"kind", "groups", "diagnostics"}:
        if not isinstance(payload["groups"], list) or not payload["groups"]:
            raise ValueError("Local-pair empirical MI cache requires nonempty groups")
        groups = [_statistics_from_payload(group) for group in payload["groups"]]
        if (any(not isinstance(group, EmpiricalBinaryMI) for group in groups)
                or any(group.marginal.shape != groups[0].marginal.shape for group in groups)):
            raise ValueError("Local-pair empirical MI cache groups are inconsistent")
        diagnostics = _plain(payload["diagnostics"], location="diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Local-pair empirical MI cache diagnostics must be a dictionary")
        return LocalPairMI(groups, diagnostics)
    if kind == "tensor" and set(payload) == {"kind", "tensor"}:
        tensor = _tensor_from_payload(payload["tensor"])
        if tensor.layout != torch.strided:
            raise ValueError("Standalone empirical MI cache tensors must be dense")
        return tensor
    raise ValueError("Invalid empirical MI cache statistics kind or fields")


def _load_cache(path, identity, key, *, mmap=False):
    try:
        payload = load_cpu_tensors(path, mmap=mmap)
        if (not isinstance(payload, dict) or set(payload) != {"identity", "key", "statistics"}
                or payload["identity"] != identity or payload["key"] != key):
            raise ValueError("Cache identity does not match the requested data/region conventions")
        return _statistics_from_payload(payload["statistics"])
    except Exception as error:
        raise MIStatisticsCacheError(
            f"Cannot reuse empirical MI cache {path}: {error}. "
            "The existing file was not modified; choose a new cache directory or inspect it explicitly.") from error


def load_or_build_mi_statistics(cache_dir, key_data, build: Callable, *, mmap=False):
    """Return ``(statistics, metadata)`` without changing the MI definition.

    Required ``key_data`` fields: ``data_sha256``, ``split``, ``states``,
    ``dimension``, ``block``, ``buffer``, ``environment_offsets`` (None means
    the implementation's full shell), ``flip``, ``pooling`` and ``dtype``.
    Include additional JSON fields for any other data/config convention used
    by ``build`` (e.g. shuffled-visible seed, input encoding, or data config).
    The helper cannot independently verify a caller-supplied file fingerprint.

    ``build`` must return static CPU statistics; cache hits never call it.  This
    helper itself neither reads nor changes torch/NumPy/Python RNG states.

    ``mmap`` is an explicit execution-only opt-in. Modern archives are mapped
    privately and legacy serialization retains eager compatibility; malformed
    archives do not silently fall back or rebuild. The key, on-disk format,
    sparse coalescing and all statistic validations are unchanged.
    """
    identity, key = _cache_identity(key_data)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mi-v{MI_STATISTICS_CACHE_VERSION}-{key}.pt"
    started = time.perf_counter()
    if path.exists():
        statistics = _load_cache(path, identity, key, mmap=mmap)
        return statistics, {"hit": True, "path": str(path), "key": key,
                            "cache_version": MI_STATISTICS_CACHE_VERSION,
                            "build_seconds": 0.0, "load_seconds": time.perf_counter() - started,
                            "publication_race": False}
    started = time.perf_counter()
    statistics = build()
    build_seconds = time.perf_counter() - started
    serialized = _statistics_payload(statistics)
    # Validate before publishing, including shape/finite checks on dense and COO
    # statistics.  Do not make a malformed builder's result a persistent cache.
    _statistics_from_payload(serialized)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    race = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save({"identity": identity, "key": key, "statistics": serialized}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and unlike replace() does not
            # overwrite another builder's completed (or corrupt) cache file.
            os.link(temporary, path)
        except FileExistsError:
            race = True
            statistics = _load_cache(path, identity, key, mmap=mmap)
    finally:
        temporary.unlink(missing_ok=True)
    return statistics, {"hit": race, "path": str(path), "key": key,
                        "cache_version": MI_STATISTICS_CACHE_VERSION,
                        "build_seconds": build_seconds, "load_seconds": 0.0,
                        "publication_race": race}
