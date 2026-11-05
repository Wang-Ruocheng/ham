"""Small synthetic 2D report checks: no training, SSH, sampling or roots."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import EnsembleDataset
from recurrent_rg.data.statistics import plaquette_distribution
from scripts.ising_qcr_evaluation import encoding_contract
from scripts.report_spin_glass2d import (
    DRAW_SEED, SPLIT_SEED, _model_state_sha256, _validate_metadata,
    collect_encodings, compute_closure_metrics, describe_coordinates,
)


def toy_datasets():
    samples, length = 8, 12
    partition = encoding_contract(samples, split_seed=SPLIT_SEED)["sample_partition"]
    configurations = torch.ones((4, samples, length, length, 1), dtype=torch.float64)
    configurations[:, partition[:samples // 2]] = -1
    datasets, metadata = [], dict(schema_version=1, dimension=2, lattice_shape=[length, length])
    for split, ids in (("train", (0, 1)), ("test", (2, 3))):
        groups = torch.tensor([[disorder, t] for disorder in ids for t in (0, 1)])
        datasets.append((split, EnsembleDataset(configurations.clone(), None, groups)))
        metadata[split] = [dict(disorder_id=int(disorder), temperature_index=int(t),
            temperature=(.7, 2.)[int(t)], bond_sha256=str(int(disorder)) * 64,
            overlap_replicas=[0, 1], dimension=2, lattice_shape=[length, length],
            samples=samples, sites=length ** 2, frustrated_plaquette_fraction=.4 + .01 * int(disorder),
            zero_plaquette_fraction=0.) for disorder, t in groups]
    return datasets, metadata


class _MomentEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution_decoder = torch.nn.Linear(1, 16).double()

    def forward(self, value):
        return value.mean(dim=(1, 2, 3, 4)).square().unsqueeze(-1)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theory_encoder = _MomentEncoder()
        self.calls = 0

    def coarse_grain(self, batch, *, max_rg_steps):
        self.calls += 1
        levels = [batch]
        for _ in range(max_rg_steps):
            probabilities = .5 + .2 * levels[-1][:, :, ::3, ::3]
            levels.append(2 * torch.bernoulli(probabilities) - 1)
        return levels

    def advance_theory(self, value):
        return .1 + .7 * value + .2 * value.square()


def test_2d_collection_reuses_C_preserves_RNG_and_correct_16_bin_target():
    datasets, metadata = toy_datasets()
    _validate_metadata(datasets, metadata)
    model = _ToyModel().eval().requires_grad_(False)
    fingerprint, rng = _model_state_sha256(model), torch.get_rng_state().clone()
    with isolated_rng("cpu", DRAW_SEED), torch.no_grad():
        arrays, contract = collect_encodings(model, datasets, metadata, "cpu",
            samples=8, draws=2, depth=1, batch_size=3)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.calls == 2 * 2 * 2  # splits * batches * C draws, not per-Q-estimator
    assert fingerprint == _model_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert contract["split_seed"] == SPLIT_SEED
    for split, dataset in datasets:
        direct = arrays[f"direct_draws_{split}"]
        assert direct.shape == (4, 2, 2, 1)
        np.testing.assert_allclose(direct[:, :, 0], 1)
        np.testing.assert_allclose(arrays[f"full_ensemble_direct_draws_{split}"][:, :, 0], 0)
        np.testing.assert_allclose(arrays[f"R_g_draws_{split}"], .1 + .7 * direct + .2 * direct ** 2)
        assert arrays[f"plaquette_distribution_target_draws_{split}"].shape == (4, 2, 2, 16)
        expected = plaquette_distribution(dataset.configurations).numpy()
        np.testing.assert_allclose(arrays[f"plaquette_distribution_target_draws_{split}"][:, 0, 0], expected)
        np.testing.assert_allclose(arrays[f"plaquette_distribution_target_draws_{split}"].sum(-1), 1)
        assert f"ring_flux_{split}" not in arrays
        assert f"frustrated_plaquette_fraction_{split}" in arrays
    assert all(value.dtype.kind != "O" for value in arrays.values())
    assert not any(word in key.lower() for key in arrays for word in ("configurations", "parameters", "spins"))
    coordinates = describe_coordinates(arrays)
    assert len(coordinates["train"]["by_temperature"]) == 2
    assert coordinates["test"]["local_frustration"]["disorder_ids"] == [2, 3]
    assert len(coordinates["train"]["plaquette_distribution_reconstruction_rmse_by_scale"]) == 2
    metrics = compute_closure_metrics(arrays)
    assert metrics["target_scales"] == [1]
    assert "by_ring_flux" not in metrics["Cmean"]["test"]["one_step"]


def test_actual_binary_b3_model_collects_smoke_shape_without_training():
    torch.set_num_threads(1)
    cfg = RunConfig.load(Path(__file__).resolve().parents[1] / "configs/spin_glass2d_overlap_g1_seed91.json")
    datasets, metadata = toy_datasets()
    model = build_model(cfg.model, 1).eval().requires_grad_(False)
    before = _model_state_sha256(model)
    with isolated_rng("cpu", DRAW_SEED), torch.no_grad():
        arrays, _ = collect_encodings(model, datasets, metadata, "cpu", samples=8, draws=2, depth=1)
    assert arrays["direct_draws_train"].shape == (4, 2, 2, 1)
    assert arrays["plaquette_distribution_reconstruction_draws_test"].shape[-1] == 16
    assert _model_state_sha256(model) == before
    assert np.isfinite(arrays["direct_draws_test"]).all()


@pytest.mark.parametrize("bad", ("root_dimension", "root_shape", "row_dimension", "row_shape",
    "sites", "samples", "flux", "frustration", "frustration_conflict", "zero", "labels",
    "group_order", "bond_conflict", "bond_leak", "replica", "temperature"))
def test_2d_metadata_rejects_1d_geometry_and_label_or_identity_errors(bad):
    datasets, metadata = toy_datasets()
    metadata = copy.deepcopy(metadata)
    row = metadata["train"][0]
    if bad == "root_dimension":
        metadata["dimension"] = 1
    elif bad == "root_shape":
        metadata["lattice_shape"] = [12]
    elif bad == "row_dimension":
        row["dimension"] = 1
    elif bad == "row_shape":
        row["lattice_shape"] = [12, 9]
    elif bad == "sites":
        row["sites"] = 12
    elif bad == "samples":
        row["samples"] = 64
    elif bad == "flux":
        row["ring_flux"] = 1
    elif bad == "frustration":
        row["frustrated_plaquette_fraction"] = 1.1
    elif bad == "frustration_conflict":
        row["frustrated_plaquette_fraction"] += .1
    elif bad == "zero":
        row["zero_plaquette_fraction"] = np.nan
    elif bad == "labels":
        datasets[0][1].couplings = torch.ones(4)
    elif bad == "group_order":
        metadata["train"].reverse()
    elif bad == "bond_conflict":
        row["bond_sha256"] = "f" * 64
    elif bad == "bond_leak":
        for test_row in metadata["test"][:2]:
            test_row["bond_sha256"] = "0" * 64
    elif bad == "replica":
        row["overlap_replicas"] = [0, 2]
    else:
        row["temperature"] = 0
    with pytest.raises(ValueError):
        _validate_metadata(datasets, metadata)


def test_collection_rejects_requested_unsaved_samples_and_wrong_decoder_target():
    datasets, metadata = toy_datasets()
    model = _ToyModel().eval().requires_grad_(False)
    with torch.no_grad(), pytest.raises(ValueError, match="actual saved"):
        collect_encodings(model, datasets, metadata, "cpu", samples=10, draws=2, depth=1)
    model.theory_encoder.distribution_decoder = torch.nn.Linear(1, 4).double().requires_grad_(False)
    with torch.no_grad(), pytest.raises(ValueError, match="16-bin"):
        collect_encodings(model, datasets, metadata, "cpu", samples=8, draws=2, depth=1)
