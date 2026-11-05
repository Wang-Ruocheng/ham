"""Synthetic CPU contracts for repeated frozen phi4 Q/C cache collection.

No saved physical fields, checkpoints, source results or GPU are accessed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import phi4_ru_r_cache as cache
from experiments.phi4_qcr import state_fingerprint
from recurrent_rg.phi4_qcr import Phi4QCR


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def frozen_model_and_chain():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(733)
        model = Phi4QCR(scale=1.2, hidden=7, theory_dim=2).double().eval()
    model.requires_grad_(False)
    rng = torch.Generator(device="cpu").manual_seed(177)
    chain = torch.randn((128, 9, 9, 1), generator=rng, dtype=torch.float64) * .1
    # Different half-ensemble distributions expose mean(Q(features)) versus
    # Q(mean(features)); a near-linear input example would miss that error.
    chain[:32] -= 2.0
    chain[32:64] += 1.7
    return model, chain


def test_seed_domains_are_pinned_reproducible_and_disjoint():
    assert cache.derived_seed("fit", 0, 0, "subset") == 4350469251130976515
    assert cache.derived_seed("fit", 0, 0, "C") == 9118115528653796471
    assert cache.derived_seed("eval", 0, 0, "subset") == 1380491236970376078
    values = [cache.derived_seed(purpose, repeat, chain, stream)
              for purpose in ("fit", "eval") for repeat in range(4)
              for chain in range(180) for stream in ("subset", "C")]
    assert len(set(values)) == len(values)
    assert all(0 <= value < 2 ** 63 for value in values)


@pytest.mark.parametrize("purpose,repeat,chain,stream", [
    ("test", 0, 0, "C"), ("fit", -1, 0, "C"), ("fit", 4, 0, "C"),
    ("fit", True, 0, "C"), ("fit", 0, -1, "C"),
    ("fit", 0, True, "C"), ("fit", 0, 0, "other"),
])
def test_invalid_seed_identity_is_rejected(purpose, repeat, chain, stream):
    with pytest.raises(ValueError, match="seed identity"):
        cache.derived_seed(purpose, repeat, chain, stream)


def test_subset_keeps64_unique_ordered_samples_and_private_rng():
    global_rng = torch.get_rng_state().clone()
    fit = cache.subset_indices("fit", 0, 0)
    np.testing.assert_array_equal(fit[:12], [107, 28, 53, 42, 94, 81, 124, 106, 37, 104, 62, 47])
    assert fit.shape == (64,) and fit.dtype == np.int64
    assert len(np.unique(fit)) == 64 and np.all((fit >= 0) & (fit < 128))
    np.testing.assert_array_equal(fit, cache.subset_indices("fit", 0, 0))
    assert not np.array_equal(fit, cache.subset_indices("eval", 0, 0))
    assert not np.array_equal(fit, cache.subset_indices("fit", 1, 0))
    assert not np.array_equal(fit, np.sort(fit))
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_encoding_matches_same_stochastic_trajectory_and_two_half_q_definition():
    model, chain = frozen_model_and_chain()
    indices, c_seed = np.arange(64, dtype=np.int64), 9377
    before_state, before_chain = state_fingerprint(model), chain.clone()
    global_rng = torch.get_rng_state().clone()
    actual = cache.encode_chain(model, chain, indices, c_seed=c_seed)
    generator = torch.Generator(device="cpu").manual_seed(c_seed)
    x0 = chain[torch.from_numpy(indices)].unsqueeze(0)
    with torch.no_grad():
        x1 = model.C(x0, generator=generator)
        x2 = model.C(x1, generator=generator)
        direct, targets = [], []
        for field in (x0, x1, x2):
            first, second = model.features(field[:, :32]), model.features(field[:, 32:])
            direct.append((model.Q(first) + model.Q(second)) / 2)
            targets.append((first + second) / 2)
        expected_direct = torch.stack(direct, dim=1)[0]
        expected_target = torch.stack(targets, dim=1)[0]
        expected_one = torch.stack([model.advance(direct[0]), model.advance(direct[1])], dim=1)[0]
        p01 = model.advance(direct[0])
        expected_rollout = torch.stack([direct[0], p01, model.advance(p01)], dim=1)[0]
        wrong_pooled = model.Q(targets[0])[0]
    assert not torch.allclose(expected_direct[0], wrong_pooled, rtol=1e-9, atol=1e-9)
    for key, expected in (("direct", expected_direct), ("feature_target", expected_target),
                          ("source_one_step", expected_one), ("source_rollout", expected_rollout)):
        np.testing.assert_allclose(actual[key], expected.numpy(), rtol=0, atol=0)
        assert actual[key].dtype == np.float64 and np.isfinite(actual[key]).all()
    assert not np.allclose(actual["source_rollout"][2], actual["source_one_step"][1], atol=1e-10)
    assert state_fingerprint(model) == before_state
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(chain, before_chain, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_c_seed_changes_only_coarse_trajectory_and_is_reproducible():
    model, chain = frozen_model_and_chain()
    indices = cache.subset_indices("fit", 1, 7)
    first = cache.encode_chain(model, chain, indices, c_seed=120)
    repeated = cache.encode_chain(model, chain, indices, c_seed=120)
    other = cache.encode_chain(model, chain, indices, c_seed=121)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    np.testing.assert_array_equal(first["direct"][0], other["direct"][0])
    np.testing.assert_array_equal(first["feature_target"][0], other["feature_target"][0])
    assert not np.array_equal(first["direct"][1:], other["direct"][1:])
    assert not np.array_equal(first["feature_target"][1:], other["feature_target"][1:])


@pytest.mark.parametrize("condition", ["trainable", "existing_gradient"])
def test_encoding_rejects_unfrozen_or_existing_gradient_parameters(condition):
    model, chain = frozen_model_and_chain()
    parameter = next(model.Q.parameters())
    if condition == "trainable":
        parameter.requires_grad_(True)
    else:
        parameter.grad = torch.zeros_like(parameter)
    before_state = state_fingerprint(model)
    with pytest.raises(ValueError, match="frozen parameters"):
        cache.encode_chain(model, chain, np.arange(64), c_seed=1)
    assert state_fingerprint(model) == before_state


@pytest.mark.parametrize("indices", [
    np.arange(63), np.zeros(64, dtype=np.int64), np.arange(64, dtype=np.float64),
    np.arange(65, 129), np.arange(-1, 63),
])
def test_encoding_rejects_invalid_saved_sample_selection(indices):
    model, chain = frozen_model_and_chain()
    with pytest.raises(ValueError, match="64 unique"):
        cache.encode_chain(model, chain, indices, c_seed=1)
