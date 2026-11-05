"""Opt-in mapped MI reads preserve old cache identities and exact statistics."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from recurrent_rg import mi_cache, tensor_io
from recurrent_rg.empirical_mi import EmpiricalBinaryMI, LocalPairMI


def key_data():
    return dict(data_sha256="ab" * 32, split="train", states=2, dimension=2,
        block=2, buffer=1, environment_offsets=None, flip=False, pooling=1,
        dtype="torch.float64", shape=[3, 4, 6, 6, 1],
        statistics="binary_lattice_mi_statistics", statistics_kwargs={"block": 2})


def statistics(*, sparse=False):
    conditional = torch.tensor([[1., 0., 0., 0.], [.2, .3, .1, .4],
        [0., 0., .5, .5], [.1, .6, .2, .1]], dtype=torch.float64)
    if sparse:
        conditional = conditional.to_sparse().coalesce()
    weights = torch.tensor([[.1, .2, .3, .4], [.4, .1, .2, .3],
        [.25, .25, .25, .25]], dtype=torch.float64).to_sparse().coalesce()
    marginal = weights.to_dense() @ conditional.to_dense()
    return EmpiricalBinaryMI(conditional, weights, marginal, dict(kind="small test", labels_used=False))


def write_existing(directory, original=None, *, legacy=False, mutate=None):
    directory.mkdir(parents=True, exist_ok=True)
    identity, key = mi_cache._cache_identity(key_data())
    original = statistics() if original is None else original
    payload = dict(identity=identity, key=key, statistics=mi_cache._statistics_payload(original))
    if mutate:
        mutate(payload)
    path = directory / f"mi-v1-{key}.pt"
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
    return path, original


def no_builder():
    pytest.fail("Existing caches must never rebuild or resample")


def assert_same(left, right):
    if isinstance(left, LocalPairMI):
        assert isinstance(right, LocalPairMI) and left.diagnostics == right.diagnostics
        assert len(left.groups) == len(right.groups)
        for a, b in zip(left.groups, right.groups):
            assert_same(a, b)
        return
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape and left.stride() == right.stride()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        return
    assert isinstance(left, EmpiricalBinaryMI) and left.diagnostics == right.diagnostics
    for name in ("conditional", "weights", "marginal"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype and a.layout == b.layout and a.shape == b.shape
        if a.is_sparse:
            assert a.is_coalesced() and b.is_coalesced()
            torch.testing.assert_close(a.indices(), b.indices(), rtol=0, atol=0)
            torch.testing.assert_close(a.values(), b.values(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_mapped_and_eager_loads_have_identical_values_layouts_and_gradients(tmp_path, sparse, legacy):
    original = statistics(sparse=sparse)
    path, _ = write_existing(tmp_path, original, legacy=legacy)
    before, before_stat = path.read_bytes(), path.stat()
    eager, eager_record = mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder)
    mapped, mapped_record = mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert_same(eager, original)
    assert_same(mapped, eager)
    assert eager_record["key"] == mapped_record["key"]
    assert eager_record["cache_version"] == mapped_record["cache_version"] == 1
    assert eager_record["hit"] is mapped_record["hit"] is True
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert path.stat().st_ino == before_stat.st_ino
    logits = torch.arange(8, dtype=torch.float64).reshape(4, 2).requires_grad_()
    mapped_logits = logits.detach().clone().requires_grad_()
    eager_result = eager.information(logits.softmax(-1))
    mapped_result = mapped.information(mapped_logits.softmax(-1))
    torch.testing.assert_close(eager_result, mapped_result, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(eager_result.sum(), logits)[0],
        torch.autograd.grad(mapped_result.sum(), mapped_logits)[0], rtol=0, atol=0)
    eager.prepare_selection()
    mapped.prepare_selection()
    ids = torch.tensor([2, 0, 2, 1], dtype=torch.long)
    torch.testing.assert_close(eager.information_for(logits.softmax(-1), ids),
        mapped.information_for(mapped_logits.softmax(-1), ids), rtol=0, atol=0)
    np.testing.assert_array_equal(eager._selection.weight_offsets, mapped._selection.weight_offsets)
    np.testing.assert_array_equal(eager._selection.weight_conditionals, mapped._selection.weight_conditionals)


@pytest.mark.parametrize("mmap,legacy", [(False, False), (False, True), (True, False), (True, True)])
def test_mapping_is_opt_in_and_only_modern_serialization_uses_mmap(tmp_path, monkeypatch, mmap, legacy):
    write_existing(tmp_path, legacy=legacy)
    original_load, calls = torch.load, []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        assert isinstance(args[0], str), "Older supported PyTorch mmap requires a string filename"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed)
    mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=mmap)
    assert len(calls) == 1
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is True
    assert calls[0].get("mmap", False) is (mmap and not legacy)


@pytest.mark.parametrize("kind", ["dense", "local_pair"])
def test_mapped_loading_keeps_other_supported_cache_kinds(tmp_path, kind):
    original = (torch.arange(36, dtype=torch.float64).reshape(6, 6).T if kind == "dense"
        else LocalPairMI([statistics(), statistics(sparse=True)], dict(aggregation="mean")))
    write_existing(tmp_path, original)
    eager, _ = mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder)
    mapped, _ = mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert_same(mapped, eager)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("corruption", ["identity", "dtype", "dimensions", "conditional_nan", "weights_inf",
                                        "negative_index", "past_end_index"])
def test_bad_existing_cache_is_not_rebuilt_overwritten_or_silently_relaxed(tmp_path, legacy, corruption):
    def corrupt(payload):
        stats = payload["statistics"]
        if corruption == "identity": payload["identity"]["key_data"]["buffer"] = 9
        if corruption == "dtype": stats["marginal"]["values"] = stats["marginal"]["values"].float()
        if corruption == "dimensions": stats["marginal"]["values"] = torch.ones(3, 3, dtype=torch.double)
        if corruption == "conditional_nan": stats["conditional"]["values"][-1, -1] = float("nan")
        if corruption == "weights_inf": stats["weights"]["values"][-1] = float("inf")
        if corruption == "negative_index": stats["weights"]["indices"][0, -1] = -1
        if corruption == "past_end_index": stats["weights"]["indices"][1, -1] = 4

    path, _ = write_existing(tmp_path, legacy=legacy, mutate=corrupt)
    before = path.read_bytes()
    with pytest.raises(mi_cache.MIStatisticsCacheError, match="existing file was not modified"):
        mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("prefix", [b"PK\x03\x04not-a-valid-archive", b"not-a-valid-legacy-payload"])
def test_truncated_cache_has_no_eager_retry_or_rebuild(tmp_path, monkeypatch, prefix):
    path, _ = write_existing(tmp_path)
    path.write_bytes(prefix)
    original_load, calls = torch.load, []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed)
    with pytest.raises(mi_cache.MIStatisticsCacheError):
        mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert len(calls) == 1
    assert calls[0].get("mmap", False) is prefix.startswith(b"PK\x03\x04")
    assert path.read_bytes() == prefix


def test_unrelated_modern_mapping_error_is_not_swallowed_by_eager_fallback(tmp_path, monkeypatch):
    path, _ = write_existing(tmp_path)
    before, calls = path.read_bytes(), []

    def failing_load(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic mapping permission failure")

    monkeypatch.setattr(torch, "load", failing_load)
    with pytest.raises(mi_cache.MIStatisticsCacheError, match="synthetic mapping permission failure"):
        mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert len(calls) == 1 and calls[0]["mmap"] is True
    assert path.read_bytes() == before


def test_chunk_validation_retains_all_elements_and_coalesce_semantics(tmp_path, monkeypatch):
    original_chunks = tensor_io.tensor_chunks
    seen_sizes = []

    def small_chunks(values, **kwargs):
        for part in original_chunks(values, max_elements=3):
            seen_sizes.append(part.numel())
            yield part

    monkeypatch.setattr(tensor_io, "tensor_chunks", small_chunks)
    monkeypatch.setattr(mi_cache, "tensor_chunks", small_chunks)
    original_finite = torch.isfinite

    def bounded_finite(values):
        assert values.numel() <= 3, "Validation must not allocate a full-tensor finite mask"
        return original_finite(values)

    monkeypatch.setattr(torch, "isfinite", bounded_finite)
    identity, key = mi_cache._cache_identity(key_data())
    source = statistics()
    payload = dict(identity=identity, key=key, statistics=mi_cache._statistics_payload(source))
    weight = payload["statistics"]["weights"]
    # Same final COO values, but intentionally duplicate the first coordinate
    # and reverse its order: the historic coalesce path must remain active.
    original_indices, original_values = weight["indices"], weight["values"]
    weight["indices"] = torch.cat((original_indices[:, 1:], original_indices[:, :1], original_indices[:, :1]), 1).flip(1)
    weight["values"] = torch.cat((original_values[1:], original_values[:1] / 2, original_values[:1] / 2)).flip(0)
    path = tmp_path / f"mi-v1-{key}.pt"
    torch.save(payload, path)
    mapped, _ = mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert_same(mapped, source)
    assert seen_sizes and max(seen_sizes) <= 3
    # A bad value at the last element must still be inspected after all chunks.
    payload["statistics"]["conditional"]["values"][-1, -1] = float("nan")
    torch.save(payload, path)
    with pytest.raises(mi_cache.MIStatisticsCacheError, match="non-finite"):
        mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)


def test_mapped_cache_hit_does_not_advance_any_rng(tmp_path):
    path, _ = write_existing(tmp_path)
    before_file = hashlib.sha256(path.read_bytes()).hexdigest()
    torch_state = torch.get_rng_state().clone()
    numpy_state = deepcopy(np.random.get_state())
    python_state = random.getstate()
    mi_cache.load_or_build_mi_statistics(tmp_path, key_data(), no_builder, mmap=True)
    assert torch.equal(torch.get_rng_state(), torch_state)
    actual_numpy = np.random.get_state()
    assert numpy_state[0] == actual_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
    assert numpy_state[2:] == actual_numpy[2:]
    assert random.getstate() == python_state
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_file
