"""Synthetic CPU checks of the fresh paired phi4 b=2 training pipeline."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_b2_closure_comparison as training
from recurrent_rg.phi4_b2_qcr import Phi4B2QCR, continuous_objectives_b2


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def field_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    options = dict(dimension=2, sizes=[32], chains=2, samples=4,
                   r_values=[-.5, -.2, .3], u_values=[.2], h=0.,
                   test_every=3, test_offset=1)
    shards = []
    for cell, r in enumerate(options["r_values"]):
        path = root / f"cell_{cell}"
        path.mkdir()
        split = "test" if cell == 1 else "train"
        fields = np.random.default_rng(481 + cell).normal(size=(2, 4, 32, 32, 1))
        np.savez(path / "data.npz", configurations=fields,
                 raw_observables=np.full((2, 4, 8), np.nan),
                 forbidden_labels=np.full(3, np.nan))
        data_sha = training.sha256_file(path / "data.npz")
        training.write_json(path / "sidecar.json", dict(status="complete", dimension=2,
            cell_index=cell, size=32, r=r, u=.2, h=0., split=split, data_sha256=data_sha))
        shards.append(dict(cell_index=cell, size=32, split=split,
            path=f"cell_{cell}/data.npz", sha256=data_sha,
            sidecar=f"cell_{cell}/sidecar.json",
            sidecar_sha256=training.sha256_file(path / "sidecar.json")))
    training.write_json(root / "manifest.json", dict(status="complete", options=options,
        shards=shards, warning_count=7, qualification="Synthetic tests, no physical equilibrium claim",
        schema="synthetic_phi4_b2_test", sample_semantics="synthetic_fields"))
    return root


def read(path):
    return json.loads(path.read_text())


def optimizers(model):
    return {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
            for name in ("Q", "C", "R")}


def test_loader_reads_only_configurations_and_preserves_inputs_split_and_metadata(field_data, monkeypatch):
    before = {str(path): training.sha256_file(path) for path in field_data.rglob("*") if path.is_file()}
    original_load = np.load
    accessed = []

    class OnlyConfigurations:
        def __init__(self, path, **kwargs):
            assert kwargs == {"allow_pickle": False}
            self.archive = original_load(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            accessed.append(key)
            assert key == "configurations", "Physical diagnostics/labels entered optimization loader"
            return self.archive[key]

    monkeypatch.setattr(training.np, "load", OnlyConfigurations)
    dataset = training.load_dataset(field_data)
    assert accessed == ["configurations"] * 3
    assert dataset.configurations.shape == (6, 4, 32, 32, 1)
    assert dataset.configurations.dtype == torch.float64
    assert dataset.labels.shape == (6, 3)
    assert set(dataset.cell_ids[dataset.train_indices]) == {0, 2}
    assert set(dataset.cell_ids[dataset.test_indices]) == {1}
    expected = dataset.configurations[dataset.train_indices].square().mean().sqrt().item()
    assert dataset.scale == pytest.approx(expected)
    assert dataset.warning_count == 7
    assert dataset.sampler_schema == "synthetic_phi4_b2_test"
    assert dataset.sample_semantics == "synthetic_fields"
    assert before == {str(path): training.sha256_file(path) for path in field_data.rglob("*") if path.is_file()}


@pytest.mark.parametrize("target", ["data.npz", "sidecar.json"])
def test_loader_rejects_changed_data_or_sidecar(field_data, target):
    with (field_data / "cell_0" / target).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        training.load_dataset(field_data)


def test_loader_rejects_outside_path_and_cell_leakage(field_data):
    path = field_data / "manifest.json"
    original = read(path)
    wrong = copy.deepcopy(original)
    wrong["shards"][0]["path"] = "../outside.npz"
    training.write_json(path, wrong)
    with pytest.raises(ValueError, match="escapes"):
        training.load_dataset(field_data)
    wrong = copy.deepcopy(original)
    sidecar_path = field_data / wrong["shards"][0]["sidecar"]
    sidecar = read(sidecar_path)
    sidecar["split"] = "test"
    training.write_json(sidecar_path, sidecar)
    wrong["shards"][0]["sidecar_sha256"] = training.sha256_file(sidecar_path)
    training.write_json(path, wrong)
    with pytest.raises(ValueError, match="split"):
        training.load_dataset(field_data)


def test_four_scale_bias_uses_train_only_and_changes_no_kernel_encoder_weight_or_rng(field_data):
    dataset = training.load_dataset(field_data)
    torch.manual_seed(91)
    model = Phi4B2QCR(dataset.scale, hidden=4)
    reference = copy.deepcopy(model)
    state_before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(9091)
    with torch.no_grad():
        features = []
        for index in dataset.train_indices:
            levels = reference.coarse_grain(dataset.configurations[index].unsqueeze(0), generator)
            features.extend(reference.features(level)[0] for level in levels)
        expected_mean = torch.stack(features).mean(0)

    class TrainConfigurationsOnly:
        shape = dataset.configurations.shape

        def __getitem__(self, index):
            assert index in dataset.train_indices
            return dataset.configurations[index]

    class NoLabelsOrTest:
        train_indices = dataset.train_indices
        configurations = TrainConfigurationsOnly()

        @property
        def labels(self):
            raise AssertionError("Couplings entered mean-bias initialization")

        @property
        def test_indices(self):
            raise AssertionError("TEST split entered mean-bias initialization")

    checks = []
    receipt = training.initialize_decoder(model, NoLabelsOrTest(), torch.device("cpu"), 91,
                                          lambda: checks.append(True))
    assert len(checks) == len(dataset.train_indices)
    for key, tensor in model.state_dict().items():
        if key != "Q.decoder.bias":
            torch.testing.assert_close(tensor, state_before[key], rtol=0., atol=0.)
    torch.testing.assert_close(model.Q.decoder.bias, expected_mean, rtol=2e-14, atol=2e-16)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(parameter.grad is None and parameter.requires_grad for parameter in model.parameters())
    assert receipt["C_updates"] == 0
    assert receipt["scales"] == [0, 1, 2, 3]
    assert receipt["physical_ensembles"] == 4 and receipt["fields_per_ensemble"] == 4
    assert receipt["global_RNG_unchanged"] and receipt["all_parameter_gradients_none"]
    assert not receipt["labels_used"] and not receipt["test_data_used"]
    assert not receipt["pca_initialization"] and not receipt["warm_start"]


@pytest.mark.parametrize("mode", tuple(training.MODES.values()))
def test_independent_update_matches_simultaneous_reference_exactly(mode):
    torch.manual_seed(251)
    model = Phi4B2QCR(1., hidden=4)
    reference = copy.deepcopy(model)
    batch = torch.randn(2, 4, 32, 32, 1, dtype=torch.float64)
    actual_optimizers, reference_optimizers = optimizers(model), optimizers(reference)
    rng1, rng2 = torch.Generator().manual_seed(513), torch.Generator().manual_seed(513)
    metrics = training.independent_step(model, batch, actual_optimizers, rng1, mode, 4)
    losses = continuous_objectives_b2(reference, batch, rng2, 4, mode)[:3]
    gradients = {name: torch.autograd.grad(loss, tuple(getattr(reference, name).parameters()))
                 for name, loss in zip(("Q", "C", "R"), losses)}
    for name in ("Q", "C", "R"):
        for parameter, gradient in zip(getattr(reference, name).parameters(), gradients[name]):
            parameter.grad = gradient
        reference_optimizers[name].step()
    reference.Q.normalize_decoder_()
    assert training.state_fingerprint(model) == training.state_fingerprint(reference)
    assert torch.equal(rng1.get_state(), rng2.get_state())
    assert all(isinstance(value, float) and np.isfinite(value) for value in metrics.values())
    assert all(count["min"] == count["max"] == 1
               for count in training.optimizer_step_counts(actual_optimizers, 1).values())


def test_three_paired_updates_keep_q_c_identical_and_r_different():
    torch.manual_seed(91)
    legacy = Phi4B2QCR(1., hidden=4)
    all_start = copy.deepcopy(legacy)
    models = [legacy, all_start]
    ops = [optimizers(model) for model in models]
    rngs = [torch.Generator().manual_seed(293) for _ in models]
    batch = torch.randn(2, 4, 32, 32, 1, dtype=torch.float64)
    for _ in range(3):
        rows = [training.independent_step(model, batch, optimizer, rng, mode, 4)
                for model, optimizer, rng, mode in zip(models, ops, rngs, training.MODES.values())]
        assert rows[0]["q_loss"] == rows[1]["q_loss"]
        assert rows[0]["c_loss"] == rows[1]["c_loss"]
        assert torch.equal(rngs[0].get_state(), rngs[1].get_state())
        hashes = [training.module_fingerprints(model) for model in models]
        assert hashes[0]["Q"] == hashes[1]["Q"]
        assert hashes[0]["C"] == hashes[1]["C"]
        assert hashes[0]["R"] != hashes[1]["R"]
    for optimizer in ops:
        assert all(count["min"] == count["max"] == 3
                   for count in training.optimizer_step_counts(optimizer, 3).values())


def test_four_scale_evaluation_is_reproducible_frozen_and_correctly_decoded(field_data):
    dataset = training.load_dataset(field_data)
    torch.manual_seed(651)
    model = Phi4B2QCR(dataset.scale, hidden=4)
    before = training.state_fingerprint(model)
    arrays = training.evaluate(model, dataset, torch.device("cpu"), lambda: None)
    torch.randn(100)
    repeated = training.evaluate(model, dataset, torch.device("cpu"), lambda: None)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    assert arrays["train_direct"].shape == (4, 4, 1)
    assert arrays["test_one_step"].shape == (2, 3, 1)
    assert arrays["train_direct_full"].shape == (4, 4, 1)
    assert arrays["train_feature_target"].shape == (4, 4, 105)
    assert arrays["test_feature_reconstruction"].shape == (2, 4, 105)
    assert arrays["test_one_step_feature_prediction"].shape == (2, 3, 105)
    assert arrays["test_rollout_feature_prediction"].shape == (2, 4, 105)
    assert arrays["grid_mean_direct"].shape == (3, 4, 1)
    assert arrays["grid_R_of_mean"].shape == (3, 3, 1)
    assert arrays["grid_displacement"].shape == (3, 3, 1)
    np.testing.assert_array_equal(arrays["train_rollout"][:, 0], arrays["train_direct"][:, 0])
    np.testing.assert_allclose(arrays["grid_displacement"],
                               arrays["grid_R_of_mean"] - arrays["grid_mean_direct"][:, :-1])
    with torch.no_grad():
        for split in ("train", "test"):
            for source, decoded in (("direct", "feature_reconstruction"),
                                    ("one_step", "one_step_feature_prediction"),
                                    ("rollout", "rollout_feature_prediction")):
                expected = model.Q.decode(torch.from_numpy(arrays[f"{split}_{source}"])).numpy()
                np.testing.assert_allclose(arrays[f"{split}_{decoded}"], expected, atol=1e-14, rtol=1e-14)
    assert training.state_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_endpoint_roots_are_numerical_and_never_certified_physical_markers():
    class Affine:
        def advance(self, x):
            return .5 * x + .1

    direct = np.array([[[-.5], [0.], [.5], [1.]], [[0.], [.5], [1.], [1.5]]])
    arrays = {"train_direct": direct, "test_direct": direct}
    roots = training.flow_and_roots(Affine(), arrays, torch.device("cpu"))
    assert len(roots) == 1
    root = roots[0]
    assert root["g"] == pytest.approx(.2)
    assert root["R_derivative"] == pytest.approx(.5)
    assert root["classification"] == "attracting"
    assert root["physical_critical_point_certified"] is False and root["thermal_nu"] is None
    assert len(root["data_nearby"]) == 8
    assert set(point["scale"] for point in root["data_nearby"]) == {0, 1, 2, 3}
    np.testing.assert_allclose(arrays["flow_displacement"], arrays["flow_R"] - arrays["flow_g"])


def test_two_arm_two_update_cpu_endpoint_status_optimizer_and_array_contract(field_data, tmp_path):
    output = tmp_path / "endpoint"
    options = training.Options(str(field_data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, log_every=1, checkpoint_every=2)
    result = training.train(options)
    assert result["status"] == "complete" and result["actual_steps"] == 2
    assert result["actual_steps_by_arm"] == {"legacy": 2, "all_start": 2}
    assert result["lattice_sizes"] == [32, 16, 8, 4]
    assert result["C_Q_pairing_verified"] and result["gradient_routing_verified"]
    assert result["critical_r"] is None and result["thermal_nu"] is None
    assert not result["equilibrium_certified"] and not result["Q_closure_feedback"]
    status = read(output / "status.json")
    assert status["status"] == "complete" and status["workflow_complete"]
    assert status["returncode"] == 0 and status["error"] is None and not status["requires_user_action"]
    assert status["latest_checkpoint_step"] == 2
    assert status["summary_sha256"] == training.sha256_file(output / "summary.json")
    assert status["evaluation_arrays_sha256"] == training.sha256_file(output / "evaluation_arrays.npz")
    manifest = read(output / "manifest.json")
    assert manifest["schema_version"] == 5 and manifest["block_size"] == manifest["stride"] == 2
    assert not manifest["C_pretraining"] and not manifest["kernel_selection"] and not manifest["warm_start"]
    assert manifest["model_input_members"] == ["configurations"]
    assert manifest["q_initialization_receipt"]["C_updates"] == 0
    assert manifest["closure_modes"] == training.MODES
    assert manifest["closure_scale_weights"] is None
    fingerprints = []
    for name in training.MODES:
        assert (output / name / "training_step_2.pt").is_file()
        arm = result["arms"][name]
        assert arm["actual_steps"] == 2
        assert all(row["min"] == row["max"] == 2 for row in arm["optimizer_steps"].values())
        assert arm["closure_objective"] == training.MODES[name]
        assert arm["thermal_nu"] is None and arm["critical_r"] is None
        fingerprints.append(arm["final_module_sha256"])
    assert fingerprints[0]["Q"] == fingerprints[1]["Q"]
    assert fingerprints[0]["C"] == fingerprints[1]["C"]
    assert fingerprints[0]["R"] != fingerprints[1]["R"]
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        for split, count in (("train", 4), ("test", 2)):
            assert arrays[f"legacy_{split}_direct"].shape == (count, 4, 1)
            assert arrays[f"all_start_{split}_one_step"].shape == (count, 3, 1)
            np.testing.assert_array_equal(arrays[f"legacy_{split}_direct"], arrays[f"all_start_{split}_direct"])
            np.testing.assert_array_equal(arrays[f"legacy_{split}_feature_target"], arrays[f"all_start_{split}_feature_target"])


@pytest.mark.parametrize("steps,seed", [(19, 191), (20, 91), (2, 91), (21, 191)])
def test_execution_only_profile_gate_requires_seed191_exact20(steps, seed, tmp_path):
    output = tmp_path / "profile_gate"
    options = training.Options("unused", str(output), device="cpu", profile_only=True,
                               steps=steps, seed=seed)
    with pytest.raises(ValueError, match="seed191 and 20"):
        options.validate()
    assert not output.exists()


def test_execution_only_profile20_has_no_checkpoint_or_endpoint_inference(field_data, tmp_path, monkeypatch):
    output = tmp_path / "profile"

    def forbidden(*args, **kwargs):
        raise AssertionError("Execution-only profile invoked scientific endpoint evaluation")

    monkeypatch.setattr(training, "evaluate", forbidden)
    monkeypatch.setattr(training, "flow_and_roots", forbidden)
    result = training.train(training.Options(str(field_data), str(output), device="cpu",
        steps=20, seed=191, profile_only=True, system_batch=2, sample_batch=4,
        information_pairs=4, log_every=20, checkpoint_every=1))
    assert result["status"] == "complete" and result["returncode"] == 0
    assert result["profile_only"] and not result["scientific_result"]
    assert result["steps_per_arm"] == 20 and result["arms"] == 2 and result["seed"] == 191
    assert not result["warm_start_reused_for_formal"]
    for counts in result["optimizer_steps"].values():
        assert all(row["min"] == row["max"] == 20 for row in counts.values())
    assert not list(output.rglob("*.pt"))
    assert not (output / "evaluation_arrays.npz").exists()
    assert not (output / "summary.json").exists()
    assert (output / "profile.json").is_file()
    status = read(output / "status.json")
    assert status["workflow_complete"] and status["returncode"] == 0 and status["profile_only"]
    assert status["latest_checkpoint_step"] == 0
    assert status["actual_steps_by_arm"] == {"legacy": 20, "all_start": 20}


def test_budget_hold_preserves_status_without_shortening_or_restart(field_data, tmp_path):
    output = tmp_path / "budget_hold"
    result = training.train(training.Options(str(field_data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, cpu_budget_seconds=1e-12))
    assert result == {"status": "budget_hold"}
    status = read(output / "status.json")
    assert status["status"] == "budget_hold" and not status["workflow_complete"]
    assert status["returncode"] == 3 and status["requires_user_action"]
    assert status["actual_steps_by_arm"] == {"legacy": 0, "all_start": 0}
    assert not (output / "summary.json").exists()
    with pytest.raises(FileExistsError):
        training.train(training.Options(str(field_data), str(output), device="cpu"))


def test_corrupt_input_is_failed_not_unknown_or_budget_hold(field_data, tmp_path):
    with (field_data / "cell_0" / "data.npz").open("ab") as handle:
        handle.write(b"changed")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="SHA256"):
        training.train(training.Options(str(field_data), str(output), device="cpu"))
    status = read(output / "status.json")
    assert status["status"] == "failed" and status["returncode"] == 1
    assert not status["workflow_complete"] and status["requires_user_action"]
