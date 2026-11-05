"""Small synthetic report checks; no physical data, SSH or model training."""
import copy

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import isolated_rng
from recurrent_rg.data.ensembles import EnsembleDataset
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays
from scripts.report_spin_glass1d import (
    DRAW_SEED, SPLIT_SEED, _model_state_sha256, _validate_metadata,
    collect_encodings, compute_closure_metrics, describe_coordinates,
)


def closure_arrays():
    arrays = {}
    for split, disorders in (("train", (0, 1)), ("test", (2, 3))):
        groups = np.array([[disorder, t] for disorder in disorders for t in (0, 1)])
        direct = np.empty((4, 2, 4, 1))
        for index, (_, temperature) in enumerate(groups):
            direct[index, :, :, 0] = np.arange(4)[None, :] + temperature + index // 2
        arrays[f"direct_draws_{split}"] = direct
        arrays[f"R_g_draws_{split}"] = direct + 1
        arrays[f"rollout_draws_{split}"] = direct.copy()
        arrays[f"group_ids_{split}"] = groups
        arrays[f"ring_flux_{split}"] = np.repeat((-1, 1), 2)
    return arrays


def test_closure_uses_the_corresponding_train_target_std_for_both_splits():
    arrays = closure_arrays()
    # Change TEST coordinate spread without changing its one-step error of 1.
    arrays["direct_draws_test"] *= 10
    arrays["R_g_draws_test"] = arrays["direct_draws_test"] + 11
    arrays["rollout_draws_test"] = arrays["direct_draws_test"].copy()
    metrics = compute_closure_metrics(arrays)
    assert metrics["target_scales"] == [1, 2, 3]
    for mode in ("Cmean", "per_C_draw"):
        np.testing.assert_allclose(metrics[mode]["train_target_std"], np.sqrt(.5))
        np.testing.assert_allclose(metrics[mode]["train"]["one_step"]["rmse"], 0)
        np.testing.assert_allclose(metrics[mode]["test"]["one_step"]["rmse"], 1)
        np.testing.assert_allclose(metrics[mode]["test"]["one_step"]["nrmse"], np.sqrt(2))
        np.testing.assert_allclose(metrics[mode]["test"]["microscopic_rollout"]["rmse"], 0)


def test_Cmean_and_per_draw_errors_keep_their_different_mean_square_order():
    arrays = closure_arrays()
    arrays["R_g_draws_test"][:, 0, :-1] += 2
    arrays["R_g_draws_test"][:, 1, :-1] -= 2
    metrics = compute_closure_metrics(arrays)
    np.testing.assert_allclose(metrics["Cmean"]["test"]["one_step"]["rmse"], 0)
    np.testing.assert_allclose(metrics["per_C_draw"]["test"]["one_step"]["rmse"], 2)


def test_duplicate_temperature_cells_do_not_reweight_independent_disorders():
    arrays = closure_arrays()
    arrays["R_g_draws_test"][0, :, :-1] += 4
    arrays["R_g_draws_test"][2, :, :-1] += 2
    before = compute_closure_metrics(arrays)
    # Repeat only the largest-error cell.  Its cell, T and J weights stay fixed.
    for key in list(arrays):
        if key.endswith("_test"):
            arrays[key] = np.concatenate((arrays[key], np.repeat(arrays[key][:1], 5, axis=0)))
    after = compute_closure_metrics(arrays)
    for mode in ("Cmean", "per_C_draw"):
        np.testing.assert_allclose(before[mode]["test"]["one_step"]["rmse"], np.sqrt(5))
        np.testing.assert_allclose(after[mode]["test"]["one_step"]["rmse"], np.sqrt(5))
        assert after[mode]["test"]["independent_disorders"] == 2
        sectors = after[mode]["test"]["one_step"]["by_ring_flux"]
        np.testing.assert_allclose(sectors["-1"]["rmse"], np.sqrt(8))
        np.testing.assert_allclose(sectors["1"]["rmse"], np.sqrt(2))


def test_collapsed_train_targets_keep_raw_errors_and_null_normalized_errors():
    arrays = closure_arrays()
    arrays["direct_draws_train"][:] = 0
    arrays["rollout_draws_train"][:] = 0
    arrays["R_g_draws_test"][:, :, :-1] += 3
    metrics = compute_closure_metrics(arrays)
    for mode in ("Cmean", "per_C_draw"):
        assert metrics[mode]["train_target_std"] == [0., 0., 0.]
        np.testing.assert_allclose(metrics[mode]["test"]["one_step"]["rmse"], 3)
        assert metrics[mode]["test"]["one_step"]["nrmse"] == [None, None, None]


@pytest.mark.parametrize("bad", ("shape", "nonfinite", "disorder_leak", "rollout_start"))
def test_closure_rejects_incompatible_arrays(bad):
    arrays = closure_arrays()
    if bad == "shape":
        arrays["R_g_draws_test"] = arrays["R_g_draws_test"][:, :, :-1]
    elif bad == "nonfinite":
        arrays["direct_draws_test"][0, 0, 0, 0] = np.nan
    elif bad == "disorder_leak":
        arrays["group_ids_test"][0, 0] = 0
    else:
        arrays["rollout_draws_test"][0, 0, 0, 0] += 1
    with pytest.raises((ValueError, AssertionError)):
        compute_closure_metrics(arrays)


class _MomentEncoder(torch.nn.Module):
    def forward(self, configurations):
        return configurations.mean(dim=(1, 2, 3)).square().unsqueeze(-1)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theory_encoder = _MomentEncoder()
        self.calls = 0

    def coarse_grain(self, batch, *, max_rg_steps):
        self.calls += 1
        levels = [batch]
        for _ in range(max_rg_steps):
            probabilities = .5 + .2 * levels[-1][:, :, ::2]
            levels.append(2 * torch.bernoulli(probabilities) - 1)
        return levels

    def advance_theory(self, value):
        return .1 + .7 * value + .2 * value.square()


def toy_datasets():
    samples = 8
    partition = encoding_contract(samples, split_seed=SPLIT_SEED)["sample_partition"]
    configurations = torch.ones((4, samples, 16, 1), dtype=torch.float64)
    configurations[:, partition[:samples // 2]] = -1
    datasets, metadata = [], {"schema_version": 1}
    for split, ids in (("train", (0, 1)), ("test", (2, 3))):
        groups = torch.tensor([[disorder, t] for disorder in ids for t in (0, 1)])
        dataset = EnsembleDataset(configurations.clone(), None, groups)
        datasets.append((split, dataset))
        metadata[split] = [dict(disorder_id=int(disorder), temperature_index=int(t),
            temperature=(.7, 2.)[int(t)], bond_sha256=str(int(disorder)) * 64,
            overlap_replicas=[0, 1], ring_flux=(-1, 1)[index // 2], samples=samples, sites=16)
            for index, (disorder, t) in enumerate(groups)]
    return datasets, metadata


def test_collection_reuses_C_and_preserves_half_full_coordinates_rng_and_public_boundary():
    datasets, metadata = toy_datasets()
    _validate_metadata(datasets, metadata)
    model = _ToyModel().eval()
    fingerprint = _model_state_sha256(model)
    saved_rng = torch.get_rng_state().clone()
    with isolated_rng("cpu", DRAW_SEED), torch.no_grad():
        arrays, contract = collect_encodings(model, datasets, metadata, "cpu",
            samples=8, draws=2, depth=2, batch_size=3)
    assert torch.equal(saved_rng, torch.get_rng_state())
    assert model.calls == 2 * 2 * 2  # splits * batches * C draws, shared by both Q estimators
    assert fingerprint == _model_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert contract["split_seed"] == SPLIT_SEED
    assert sorted(contract["sample_partition"]) == list(range(8))
    validate_encoding_arrays(dict(schema_version=2, samples=8, C_draws=2,
        encoding_estimator=contract["estimator"], encoding_protocol=contract), arrays)
    for split in ("train", "test"):
        direct = arrays[f"direct_draws_{split}"]
        assert direct.shape == (4, 2, 3, 1)
        # Nonlinear full-Q is different from the training half-Q estimator.
        np.testing.assert_allclose(direct[:, :, 0], 1)
        np.testing.assert_allclose(arrays[f"full_ensemble_direct_draws_{split}"][:, :, 0], 0)
        np.testing.assert_allclose(arrays[f"Q_full_minus_half_draws_{split}"][:, :, 0], -1)
        np.testing.assert_allclose(arrays[f"R_g_draws_{split}"], .1 + .7 * direct + .2 * direct ** 2)
        rollout = arrays[f"rollout_draws_{split}"]
        np.testing.assert_allclose(rollout[:, :, 0], direct[:, :, 0])
        np.testing.assert_allclose(rollout[:, :, 1:], .1 + .7 * rollout[:, :, :-1] + .2 * rollout[:, :, :-1] ** 2)
        assert arrays[f"bond_sha256_{split}"].dtype.kind == "U"
        assert arrays[f"group_ids_{split}"].shape == (4, 2)
    assert all(value.dtype.kind != "O" for value in arrays.values())
    assert not any(word in key.lower() for key in arrays for word in ("spins", "configurations", "parameters"))
    coordinates = describe_coordinates(arrays)
    assert {row["ring_flux"] for row in coordinates["train"]["by_temperature_and_ring_flux"]} == {"all", "-1", "1"}


@pytest.mark.parametrize("bad", ("labels", "group_order", "replica_pair", "bond", "flux", "temperature"))
def test_evaluation_metadata_rejects_training_labels_or_misaligned_physical_identity(bad):
    datasets, metadata = toy_datasets()
    metadata = copy.deepcopy(metadata)
    if bad == "labels":
        datasets[0][1].couplings = torch.ones(4)
    elif bad == "group_order":
        metadata["train"].reverse()
    elif bad == "replica_pair":
        metadata["train"][0]["overlap_replicas"] = [0, 2]
    elif bad == "bond":
        metadata["train"][1]["bond_sha256"] = "f" * 64
    elif bad == "flux":
        metadata["train"][1]["ring_flux"] = 1
    else:
        metadata["train"][0]["temperature"] = 0
    with pytest.raises(ValueError):
        _validate_metadata(datasets, metadata)
