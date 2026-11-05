"""Small CPU-only frozen-feature checks; no physical production or optimizer."""
import numpy as np
import pytest
import torch

from experiments.phi4_qcr import FieldDataset, evaluate_model, state_fingerprint
from recurrent_rg.phi4_qcr import FEATURE_DIM, Phi4QCR
from scripts.collect_phi4_frozen_q import (
    BudgetHold, TARGET_ATOL, chain_feature_arrays, collect_feature_arrays,
    verify_reference_targets,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def frozen_case():
    torch.manual_seed(9821)
    model = Phi4QCR(1.2, hidden=4).eval().requires_grad_(False)
    dataset = FieldDataset(torch.randn(4, 6, 9, 9, 1, dtype=torch.float64),
        np.array(["train", "test", "train", "test"]), np.array([0, 1, 2, 3]),
        np.array([[r, .2, 0.] for r in (-.3, -.1, .1, .3)]), "test-sha", [], 1,
        "Finite synthetic fixture, not physical sampling")
    return model, dataset


def test_per_configuration_statistics_equal_exact_pooled_halves(frozen_case):
    model, dataset = frozen_case
    before = state_fingerprint(model)
    values, first, second, difference = chain_feature_arrays(model,
        dataset.configurations[0], torch.device("cpu"),
        generator=torch.Generator().manual_seed(71))
    assert values.shape == (3, 6, FEATURE_DIM)
    assert first.shape == second.shape == (3, FEATURE_DIM)
    assert values.dtype == first.dtype == np.float64
    assert difference < TARGET_ATOL
    np.testing.assert_allclose(values[:, :3].mean(axis=1), first, atol=2e-16, rtol=2e-14)
    np.testing.assert_allclose(values[:, 3:].mean(axis=1), second, atol=2e-16, rtol=2e-14)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    assert state_fingerprint(model) == before


def test_c_path_is_ancestral_and_q_r_are_never_called(frozen_case, monkeypatch):
    model, dataset = frozen_case
    calls = []
    original = model.C.forward

    def tracked(x, generator=None):
        output = original(x, generator=generator)
        calls.append((x.detach().clone(), output.detach().clone()))
        return output

    def forbidden(*args, **kwargs):
        raise AssertionError("Q and R forward inference is outside feature collection")

    monkeypatch.setattr(model.C, "forward", tracked)
    monkeypatch.setattr(model.Q, "from_features", forbidden)
    monkeypatch.setattr(model.R, "forward", forbidden)
    chain_feature_arrays(model, dataset.configurations[0], torch.device("cpu"),
                         generator=torch.Generator().manual_seed(722))
    assert len(calls) == 2
    assert calls[0][0].shape == (1, 6, 9, 9, 1)
    assert calls[0][1].shape == calls[1][0].shape == (1, 6, 3, 3, 1)
    torch.testing.assert_close(calls[1][0], calls[0][1], rtol=0, atol=0)
    assert calls[1][1].shape == (1, 6, 1, 1, 1)


def test_cache_shapes_rng_order_and_official_evaluation_targets(frozen_case, tmp_path):
    model, dataset = frozen_case
    arrays, difference = collect_feature_arrays(model, dataset, torch.device("cpu"),
        train_seeds=(811, 812))
    assert arrays["train_per_sample"].shape == (2, 2, 3, 6, FEATURE_DIM)
    assert arrays["eval_train_first"].shape == arrays["eval_test_first"].shape == (2, 3, FEATURE_DIM)
    assert difference < TARGET_ATOL
    np.testing.assert_array_equal(arrays["train_indices"], [0, 2])
    np.testing.assert_array_equal(arrays["eval_test_chain_ids"], [1, 3])
    reference = evaluate_model(model, dataset, torch.device("cpu"))
    path = tmp_path / "reference.npz"
    np.savez(path, **reference)
    differences = verify_reference_targets(arrays, path)
    assert differences == {"train": 0., "test": 0.}
    repeated, _ = collect_feature_arrays(model, dataset, torch.device("cpu"), train_seeds=(811, 812))
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    # Conditional C randomness changes only the coarse levels, not physical
    # configurations or the fixed microscopic statistics.
    np.testing.assert_array_equal(arrays["train_per_sample"][0, :, 0],
                                  arrays["train_per_sample"][1, :, 0])
    assert not np.array_equal(arrays["train_per_sample"][0, :, 1:],
                              arrays["train_per_sample"][1, :, 1:])


def test_collection_rejects_nonfrozen_model(frozen_case):
    model, dataset = frozen_case
    model.Q.requires_grad_(True)
    with pytest.raises(ValueError, match="fully frozen"):
        collect_feature_arrays(model, dataset, torch.device("cpu"))


def test_collection_checks_explicit_budget_between_chains(frozen_case):
    model, dataset = frozen_case
    calls = []

    def exhausted():
        calls.append(None)
        raise BudgetHold("synthetic fixed cap")

    with pytest.raises(BudgetHold, match="synthetic fixed cap"):
        collect_feature_arrays(model, dataset, torch.device("cpu"), budget_check=exhausted)
    assert len(calls) == 1


def test_source_target_disagreement_is_not_silently_accepted(frozen_case, tmp_path):
    model, dataset = frozen_case
    arrays, _ = collect_feature_arrays(model, dataset, torch.device("cpu"), train_seeds=(811,))
    reference = evaluate_model(model, dataset, torch.device("cpu"))
    reference["train_feature_target"][0, 0, 0] += 1e-5
    path = tmp_path / "changed.npz"
    np.savez(path, **reference)
    with pytest.raises(RuntimeError, match="target mismatch"):
        verify_reference_targets(arrays, path)
