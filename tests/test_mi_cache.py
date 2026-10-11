import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import random
import threading

import numpy as np
import pytest
import torch

from recurrent_rg.empirical_mi import EmpiricalBinaryMI, LocalPairMI
from recurrent_rg.mi_cache import MIStatisticsCacheError, load_or_build_mi_statistics


def key_data(**changes):
    result = dict(data_sha256=hashlib.sha256(b"actual-cache-fixture").hexdigest(),
                  split="train", states=2, dimension=2, block=2, buffer=1,
                  environment_offsets=None, flip=False, pooling="probabilistic",
                  dtype="torch.float64", config_data={"sample_count": 64, "encoding": "+/-1"})
    result.update(changes)
    return result


def statistics(*, sparse=False):
    conditional = torch.tensor([[1., 0., 0., 0.], [.25, .25, .25, .25],
                                [0., 0., .5, .5]], dtype=torch.double)
    if sparse:
        conditional = conditional.to_sparse().coalesce()
    weights = torch.tensor([[.25, .5, .25], [.2, .3, .5]], dtype=torch.double).to_sparse().coalesce()
    marginal = weights.to_dense() @ conditional.to_dense()
    return EmpiricalBinaryMI(conditional, weights, marginal,
                             {"states": 2, "buffer": 1, "note": "static test", "optional": None})


def assert_same_statistics(actual, expected):
    assert actual.diagnostics == expected.diagnostics
    for name in ("conditional", "weights", "marginal"):
        left, right = getattr(actual, name), getattr(expected, name)
        assert left.layout == right.layout
        assert left.dtype == right.dtype
        torch.testing.assert_close(left.to_dense(), right.to_dense(), rtol=0, atol=0)
        if left.is_sparse:
            assert left.is_coalesced()
            torch.testing.assert_close(left.indices(), right.indices(), rtol=0, atol=0)
            torch.testing.assert_close(left.values(), right.values(), rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [False, True])
def test_cache_round_trip_preserves_exact_values_and_kernel_gradients(tmp_path, sparse):
    original = statistics(sparse=sparse)
    built, miss = load_or_build_mi_statistics(tmp_path, key_data(), lambda: original)
    cached, hit = load_or_build_mi_statistics(
        tmp_path, key_data(), lambda: pytest.fail("Cache hit must never call build"))
    assert not miss["hit"] and hit["hit"]
    assert hit["build_seconds"] == 0
    assert miss["key"] == hit["key"]
    assert_same_statistics(built, original)
    assert_same_statistics(cached, original)
    logits = torch.tensor([[.4, -.7], [1., -.2], [-.2, .3], [.8, 1.]],
                          dtype=torch.double, requires_grad=True)
    cached_logits = logits.detach().clone().requires_grad_()
    expected = original.information(logits.softmax(-1))
    actual = cached.information(cached_logits.softmax(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), cached_logits)[0],
                               torch.autograd.grad(expected.sum(), logits)[0], rtol=0, atol=0)
    # Changing a model/kernel does not invalidate statistics or freeze its gradient.
    different_logits = (logits.detach() + torch.tensor([.1, -.2])).requires_grad_()
    reused, second_hit = load_or_build_mi_statistics(
        tmp_path, key_data(), lambda: pytest.fail("Kernel changes are not cache-key changes"))
    assert second_hit["hit"]
    assert torch.isfinite(torch.autograd.grad(reused.information(different_logits.softmax(-1)).sum(),
                                             different_logits)[0]).all()


def test_local_pair_and_dense_histogram_round_trip(tmp_path):
    groups = LocalPairMI([statistics(), statistics(sparse=True)], {"aggregation": "mean", "groups": [0, 1]})
    original, _ = load_or_build_mi_statistics(tmp_path, key_data(pooling="local_pairs"), lambda: groups)
    cached, metadata = load_or_build_mi_statistics(
        tmp_path, key_data(pooling="local_pairs"), lambda: pytest.fail("Unexpected build"))
    assert metadata["hit"] and cached.diagnostics == groups.diagnostics
    for actual, expected in zip(cached.groups, original.groups):
        assert_same_statistics(actual, expected)
    logits = torch.arange(8, dtype=torch.double).reshape(4, 2).requires_grad_()
    actual_logits = logits.detach().clone().requires_grad_()
    actual = cached.information(actual_logits.softmax(-1))
    expected = groups.information(logits.softmax(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), actual_logits)[0],
                               torch.autograd.grad(expected.sum(), logits)[0], rtol=0, atol=0)
    histogram = torch.arange(32, dtype=torch.double).reshape(2, 4, 4)
    _, _ = load_or_build_mi_statistics(tmp_path, key_data(dimension=1), lambda: histogram)
    loaded, tensor_hit = load_or_build_mi_statistics(
        tmp_path, key_data(dimension=1), lambda: pytest.fail("Unexpected histogram build"))
    assert tensor_hit["hit"]
    torch.testing.assert_close(loaded, histogram, rtol=0, atol=0)


@pytest.mark.parametrize("change", [
    {"data_sha256": hashlib.sha256(b"other-real-data").hexdigest()},
    {"split": "test"}, {"states": 3}, {"dimension": 1}, {"block": 3},
    {"buffer": 2}, {"environment_offsets": [[-2, 0], [-2, 1]]},
    {"flip": True}, {"pooling": "different"}, {"dtype": "torch.float32"},
    {"config_data": {"sample_count": 128}}, {"shuffle_visible_seed": 10},
])
def test_every_data_and_region_convention_invalidates_cache(tmp_path, change):
    _, first = load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    calls = []
    _, changed = load_or_build_mi_statistics(
        tmp_path, key_data(**change), lambda: calls.append(True) or statistics())
    assert calls == [True] and not changed["hit"]
    assert first["key"] != changed["key"]
    assert Path(first["path"]).exists() and Path(changed["path"]).exists()


def test_dictionary_order_and_tuple_offsets_are_canonical(tmp_path):
    first = key_data(environment_offsets=[[-2, 0], [-2, 1]])
    _, miss = load_or_build_mi_statistics(tmp_path, first, statistics)
    second = dict(reversed(list(first.items())))
    second["environment_offsets"] = ((-2, 0), (-2, 1))
    _, hit = load_or_build_mi_statistics(tmp_path, second, lambda: pytest.fail("Should hit same key"))
    assert hit["hit"] and hit["key"] == miss["key"]


def snapshot_rng():
    return torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate()


def assert_rng_unchanged(before):
    torch_state, numpy_state, python_state = before
    assert torch.equal(torch.get_rng_state(), torch_state)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    assert np.array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_cache_operations_do_not_consume_global_rng(tmp_path):
    before = snapshot_rng()
    load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    assert_rng_unchanged(before)
    load_or_build_mi_statistics(tmp_path, key_data(), lambda: pytest.fail("Unexpected build"))
    assert_rng_unchanged(before)


def test_weights_only_basic_dictionary_serialization(tmp_path, monkeypatch):
    _, metadata = load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    payload = torch.load(metadata["path"], map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and isinstance(payload["statistics"], dict)
    assert payload["statistics"]["kind"] == "empirical_binary"
    assert isinstance(payload["statistics"]["conditional"]["values"], torch.Tensor)
    original_load = torch.load
    seen = []
    def observed_load(*args, **kwargs):
        seen.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", observed_load)
    load_or_build_mi_statistics(tmp_path, key_data(), lambda: pytest.fail("Unexpected build"))
    assert seen == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("corruption", ["truncated", "identity", "dimensions", "nonfinite", "invalid_indices"])
def test_corrupt_cache_fails_clearly_without_rebuilding_or_overwriting(tmp_path, corruption):
    _, metadata = load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    path = Path(metadata["path"])
    if corruption == "truncated":
        path.write_bytes(b"not a completed torch cache")
    else:
        payload = torch.load(path, weights_only=True)
        if corruption == "identity":
            payload["identity"]["key_data"]["buffer"] = 100
        elif corruption == "dimensions":
            payload["statistics"]["marginal"]["values"] = torch.ones(4, 3, dtype=torch.double)
        elif corruption == "nonfinite":
            payload["statistics"]["marginal"]["values"][0, 0] = float("nan")
        else:
            payload["statistics"]["weights"]["indices"][0, 0] = 100
        torch.save(payload, path)
    unchanged = path.read_bytes()
    with pytest.raises(MIStatisticsCacheError, match="existing file was not modified"):
        load_or_build_mi_statistics(tmp_path, key_data(), lambda: pytest.fail("Must not rebuild bad cache"))
    assert path.read_bytes() == unchanged
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_builders_publish_only_one_complete_cache(tmp_path):
    barrier = threading.Barrier(2)
    def build():
        barrier.wait(timeout=10)
        return statistics(sparse=True)
    def call():
        return load_or_build_mi_statistics(tmp_path, key_data(), build)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future, second_future = executor.submit(call), executor.submit(call)
        first, second = first_future.result(timeout=20), second_future.result(timeout=20)
    assert first[1]["key"] == second[1]["key"]
    assert sorted((first[1]["publication_race"], second[1]["publication_race"])) == [False, True]
    assert len(list(tmp_path.glob("mi-v*.pt"))) == 1
    assert len(list(tmp_path.iterdir())) == 1
    assert_same_statistics(first[0], second[0])


def test_cache_version_is_part_of_identity(tmp_path, monkeypatch):
    _, first = load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    monkeypatch.setattr("recurrent_rg.mi_cache.MI_STATISTICS_CACHE_VERSION", 2)
    _, changed = load_or_build_mi_statistics(tmp_path, key_data(), statistics)
    assert not changed["hit"] and changed["key"] != first["key"]
    assert Path(first["path"]).exists() and Path(changed["path"]).exists()


@pytest.mark.parametrize("bad_key", [
    {"data_sha256": "not a data fingerprint"}, {"dimension": True},
    {"flip": 0}, {"environment_offsets": [[-1]]},
    {"model_weights": torch.ones(2)}, {"float": float("nan")},
])
def test_bad_cache_keys_rejected_before_build(tmp_path, bad_key):
    with pytest.raises((ValueError, TypeError)):
        load_or_build_mi_statistics(tmp_path, key_data(**bad_key), lambda: pytest.fail("Invalid key must not build"))
    assert not list(tmp_path.iterdir())


def test_missing_required_cache_convention_rejected(tmp_path):
    key = key_data()
    del key["pooling"]
    with pytest.raises(ValueError, match="Missing.*pooling"):
        load_or_build_mi_statistics(tmp_path, key, lambda: pytest.fail("Invalid key must not build"))


@pytest.mark.parametrize("bad_builder", [
    lambda: torch.ones(2, requires_grad=True),
    lambda: torch.tensor([float("inf")]),
    lambda: object(),
])
def test_invalid_builder_output_is_not_published(tmp_path, bad_builder):
    with pytest.raises((ValueError, TypeError)):
        load_or_build_mi_statistics(tmp_path, key_data(), bad_builder)
    assert not list(tmp_path.iterdir())
