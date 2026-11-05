"""Label-free TRAIN mean-bias initialization; synthetic fields only."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_qcr as train
from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives


def _model_and_fields(device="cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    model = Phi4QCR(1., hidden=4).to(device=device)
    fields = torch.randn(5, 4, 9, 9, 1, dtype=torch.float64)
    return model, fields


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_only_bias_changes_and_rng_gradients_and_trainability_are_preserved(device):
    model, fields = _model_and_fields(device)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    receipt = train.initialize_q_decoder(model, fields, np.array([0, 2, 4]), seed=91)
    for key, value in model.state_dict().items():
        if key != "Q.decoder.bias":
            torch.testing.assert_close(value, before[key], atol=0., rtol=0.)
    assert not torch.equal(model.Q.decoder.bias, before["Q.decoder.bias"])
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert all(parameter.grad is None and parameter.requires_grad for parameter in model.parameters())
    assert model.training
    assert receipt["optimizer_updates"] == {"Q": 0, "C": 0, "R": 0}
    assert receipt["physical_ensemble_count"] == 3 and receipt["fields_per_ensemble"] == 4
    assert receipt["total_fields"] == 12 and receipt["scales"] == [0, 1, 2]
    assert receipt["generator_seed"] == 9091
    assert receipt["global_cpu_rng_unchanged"]
    assert receipt["global_cuda_rng_unchanged"] is (True if cuda_rng is not None else None)
    assert receipt["constructor_module_sha256"]["C"] == receipt["training_start_module_sha256"]["C"]
    assert receipt["constructor_module_sha256"]["R"] == receipt["training_start_module_sha256"]["R"]
    assert receipt["constructor_protected_sha256"] == receipt["training_start_protected_sha256"]


def test_bias_matches_streamed_ancestral_uniform_train_mean_and_not_test():
    model, fields = _model_and_fields()
    reference = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(9091)
    means = []
    with torch.no_grad():
        for index in (0, 2, 4):
            # The second coarse field descends from this chain's first field.
            levels = reference.coarse_grain(fields[index:index + 1], generator=generator)
            means.extend(reference.features(level)[0] for level in levels)
    expected = torch.stack(means).mean(0)

    class TrainFieldsOnly:
        shape = fields.shape

        def __len__(self):
            return len(fields)

        def __getitem__(self, index):
            assert index in (0, 2, 4), "TEST data entered initialization"
            return fields[index]

    class PoisonMetadata:
        configurations = TrainFieldsOnly()
        train_indices = np.array([0, 2, 4])

        @property
        def labels(self):
            raise AssertionError("Coupling labels entered initialization")

        @property
        def test_indices(self):
            raise AssertionError("TEST split entered initialization")

    dataset = PoisonMetadata()
    receipt = train.initialize_q_decoder(model, dataset.configurations, dataset.train_indices, seed=91)
    torch.testing.assert_close(model.Q.decoder.bias, expected, atol=2e-17, rtol=2e-15)
    assert receipt["mean_sha256"] == receipt["bias_after_sha256"]
    assert not receipt["labels_used"] and not receipt["test_data_used"]
    np.testing.assert_array_equal(receipt["mean"], model.Q.decoder.bias.detach().numpy())
    assert model.Q.decoder.bias.sum().item() == pytest.approx(1.)


def test_streaming_statistics_match_collective_statistics_when_C_noise_is_zero():
    model, fields = _model_and_fields()
    # Deterministic test-only C isolates linear pooling of feature statistics.
    model.C.noise_ratio.zero_()
    reference = copy.deepcopy(model)
    indices = np.array([0, 2, 4])
    with torch.no_grad():
        levels = reference.coarse_grain(fields[indices], generator=torch.Generator().manual_seed(9091))
        expected = torch.stack([reference.features(level).mean(0) for level in levels]).mean(0)
    train.initialize_q_decoder(model, fields, indices, seed=91)
    torch.testing.assert_close(model.Q.decoder.bias, expected, atol=2e-16, rtol=2e-14)


def test_random_legacy_does_not_read_fields_or_change_constructor_and_20_updates():
    model, fields = _model_and_fields()
    reference = copy.deepcopy(model)
    initial = train.state_fingerprint(model)
    # No statistics, metadata or RNG draw may be read in legacy mode.
    receipt = train.initialize_q_decoder(model, None, None, seed=91, mode="random_legacy")
    assert train.state_fingerprint(model) == initial
    assert receipt["constructor_state_sha256"] == receipt["training_start_state_sha256"]
    assert receipt["generator_seed"] is None and receipt["mean"] is None
    actual_optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                         for name in ("Q", "C", "R")}
    reference_optimizers = {name: torch.optim.Adam(getattr(reference, name).parameters(), lr=.001)
                            for name in ("Q", "C", "R")}
    actual_rng = torch.Generator().manual_seed(293)
    reference_rng = torch.Generator().manual_seed(293)
    for update in range(20):
        batch = fields[[update % 5, (update + 1) % 5]]
        actual = train.independent_step(model, batch, actual_optimizers,
                                       generator=actual_rng, information_pairs=4)
        losses = continuous_objectives(reference, batch, generator=reference_rng,
                                       information_pairs=4)
        gradients = {name: torch.autograd.grad(loss, tuple(getattr(reference, name).parameters()))
                     for name, loss in zip(("Q", "C", "R"), losses[:3])}
        for name in ("Q", "C", "R"):
            reference_optimizers[name].zero_grad(set_to_none=True)
            for parameter, gradient in zip(getattr(reference, name).parameters(), gradients[name]):
                parameter.grad = gradient
        for optimizer in reference_optimizers.values():
            optimizer.step()
        reference.Q.normalize_decoder_()
        assert actual["r_loss"] == pytest.approx(
            5 * actual["closure_one_step"] + 5 * actual["closure_rollout"], abs=1e-14)
    assert train.state_fingerprint(model) == train.state_fingerprint(reference)
    assert torch.equal(actual_rng.get_state(), reference_rng.get_state())
    assert train.optimizer_step_counts(actual_optimizers, 20)["Q"]["min"] == 20


def test_mean_bias_leaves_independent_C_training_exactly_paired_with_legacy():
    mean_model, fields = _model_and_fields()
    legacy_model = copy.deepcopy(mean_model)
    train.initialize_q_decoder(mean_model, fields, [0, 2, 4], seed=91)
    train.initialize_q_decoder(legacy_model, None, None, seed=91, mode="random_legacy")
    optimizers = [{name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                   for name in ("Q", "C", "R")} for model in (mean_model, legacy_model)]
    generators = [torch.Generator().manual_seed(293), torch.Generator().manual_seed(293)]
    for update in range(20):
        batch = fields[[update % 5, (update + 1) % 5]]
        for model, optimizer, generator in zip((mean_model, legacy_model), optimizers, generators):
            train.independent_step(model, batch, optimizer, generator=generator, information_pairs=4)
    assert train.state_fingerprint(mean_model.C) == train.state_fingerprint(legacy_model.C)
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    assert train.state_fingerprint(mean_model.Q) != train.state_fingerprint(legacy_model.Q)


def test_invalid_mode_and_invalid_train_indices_are_rejected(tmp_path):
    options = train.TrainingOptions("unused", str(tmp_path / "new"), q_initialization="labels")
    with pytest.raises(ValueError, match="q_initialization"):
        options.validate()
    model, fields = _model_and_fields()
    for indices in ([], [0, 0], [-1], [5], [[0, 1]]):
        with pytest.raises(ValueError, match="TRAIN"):
            train.initialize_q_decoder(model, fields, indices, seed=91)


def _write_tiny_dataset(root):
    root.mkdir()
    options = dict(dimension=2, sizes=[9], chains=2, samples=4,
                   r_values=[-.5, 0., .5], u_values=[.2], h=0.,
                   test_every=3, test_offset=1)
    shards = []
    for cell, r in enumerate(options["r_values"]):
        shard = root / f"c{cell}"
        shard.mkdir()
        fields = np.random.default_rng(91 + cell).normal(size=(2, 4, 9, 9, 1))
        np.savez(shard / "data.npz", configurations=fields)
        digest = train.sha256_file(shard / "data.npz")
        split = "test" if cell == 1 else "train"
        train.write_json(shard / "sidecar.json", dict(status="complete", dimension=2,
            cell_index=cell, size=9, r=r, u=.2, h=0., split=split, data_sha256=digest))
        shards.append(dict(cell_index=cell, size=9, split=split, path=f"c{cell}/data.npz",
            sha256=digest, sidecar=f"c{cell}/sidecar.json",
            sidecar_sha256=train.sha256_file(shard / "sidecar.json")))
    train.write_json(root / "manifest.json", dict(status="complete", options=options,
        shards=shards, warning_count=0, qualification="Synthetic, not physical"))


@pytest.mark.parametrize("mode", ["train_mean_bias", "random_legacy"])
def test_manifest_summary_and_checkpoint_record_actual_initialization(tmp_path, mode):
    data = tmp_path / "data"
    _write_tiny_dataset(data)
    output = tmp_path / mode
    summary = train.run_training(train.TrainingOptions(str(data), str(output), device="cpu",
        steps=2, system_batch=2, sample_batch=4, information_pairs=4, log_every=1,
        checkpoint_every=2, max_training_seconds=120., q_initialization=mode))
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    checkpoint = torch.load(output / "training_step_2.pt", map_location="cpu", weights_only=False)
    receipt = manifest["q_initialization_receipt"]
    assert manifest["q_initialization"] == summary["q_initialization"] == mode
    assert manifest["initial_state_sha256"] == manifest["training_start_state_sha256"]
    assert manifest["random_construction_state_sha256"] == receipt["constructor_state_sha256"]
    assert manifest["training_start_state_sha256"] == receipt["training_start_state_sha256"]
    assert summary["q_initialization_receipt"] == receipt
    assert set(summary["final_module_sha256"]) == {"Q", "C", "R"}
    assert checkpoint["manifest"] == manifest
    assert summary["optimizer_steps"]["C"]["min"] == summary["optimizer_steps"]["R"]["max"] == 2
    assert manifest["closure_objective"] == "all_start_anchor" and manifest["closure_scale_weights"] is None
    assert not manifest["labels_used_in_training"] and receipt["C_trainable"]
    assert summary["timing"]["initialization_seconds"] > 0
    assert summary["timing"]["compute_seconds"] >= sum(summary["timing"][key] for key in (
        "initialization_seconds", "training_seconds", "evaluation_seconds"))
    assert status["status"] == "complete" and status["returncode"] == 0
    if mode == "train_mean_bias":
        assert receipt["physical_ensemble_count"] == 4
        assert manifest["random_construction_state_sha256"] != manifest["initial_state_sha256"]
    else:
        assert manifest["random_construction_state_sha256"] == manifest["initial_state_sha256"]


def test_default_is_mean_bias_and_initialization_is_inside_budget(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_tiny_dataset(data)
    output = tmp_path / "held"
    options = train.TrainingOptions(str(data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, max_training_seconds=1e-12)
    assert options.q_initialization == "train_mean_bias"
    called = []
    original = train.initialize_q_decoder

    def initialize(*args, **kwargs):
        called.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "initialize_q_decoder", initialize)
    with pytest.raises(train.BudgetHold):
        train.run_training(options)
    status = json.loads((output / "status.json").read_text())
    assert called == ["train_mean_bias"]
    assert status["actual_steps"] == 0 and status["status"] == "budget_hold"
    assert not (output / "training_step_2.pt").exists()


def test_constructor_pin_fails_before_calibration_and_zero_updates(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_tiny_dataset(data)
    output = tmp_path / "mismatch"

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not run after a constructor mismatch")

    monkeypatch.setattr(train, "initialize_q_decoder", forbidden)
    options = train.TrainingOptions(str(data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4,
        expected_random_construction_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="construction SHA256 mismatch"):
        train.run_training(options)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["actual_steps"] == 0
    assert not (output / "manifest.json").exists()
