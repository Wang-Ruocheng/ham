"""Small continuous-field training checks; no production sampling or GPU use."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import phi4_qcr as train
from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives


@pytest.fixture
def field_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    options = dict(dimension=2, sizes=[9], chains=4, samples=4,
                   r_values=[-.5, 0., .5], u_values=[1.], h=0.,
                   test_every=3, test_offset=1)
    shards = []
    for cell, r in enumerate(options["r_values"]):
        path = root / f"cell_{cell}"
        path.mkdir()
        split = "test" if cell == 1 else "train"
        fields = np.random.default_rng(491 + cell).normal(size=(4, 4, 9, 9, 1))
        # Non-configuration members exist but must never be loaded by training.
        np.savez(path / "data.npz", configurations=fields,
                 raw_observables=np.full((4, 4, 8), np.nan))
        data_sha = train.sha256_file(path / "data.npz")
        train.write_json(path / "sidecar.json", dict(status="complete", dimension=2,
            cell_index=cell, size=9, r=r, u=1., h=0., split=split, data_sha256=data_sha))
        shards.append(dict(cell_index=cell, size=9, split=split,
            path=f"cell_{cell}/data.npz", sha256=data_sha,
            sidecar=f"cell_{cell}/sidecar.json", sidecar_sha256=train.sha256_file(path / "sidecar.json")))
    train.write_json(root / "manifest.json", dict(status="complete", options=options,
                     shards=shards, warning_count=7, qualification="Not equilibrium certified"))
    return root


def test_loader_uses_only_fields_and_preserves_whole_cell_split(field_data, monkeypatch):
    original_load = np.load

    class OnlyConfigurations:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key == "configurations"
            return self.archive[key]

    monkeypatch.setattr(train.np, "load", OnlyConfigurations)
    dataset = train.load_dataset(field_data)
    assert dataset.configurations.shape == (12, 4, 9, 9, 1)
    assert dataset.configurations.dtype == torch.float64
    assert dataset.labels.shape == (12, 3)
    assert set(dataset.cell_ids[dataset.train_indices]) == {0, 2}
    assert set(dataset.cell_ids[dataset.test_indices]) == {1}
    expected_scale = dataset.configurations[dataset.train_indices].square().mean().sqrt()
    assert dataset.scale == pytest.approx(expected_scale.item())
    assert dataset.warning_count == 7


def test_loader_preserves_exact_gaussian_sampler_metadata(field_data):
    manifest = json.loads((field_data / "manifest.json").read_text())
    manifest.update(schema="euclidean_gaussian_fft_v1",
                    sample_semantics="independent_iid_groups_not_markov_chains",
                    qualification="Exact finite-lattice Gaussian draws up to PRNG/floating-point error")
    train.write_json(field_data / "manifest.json", manifest)
    dataset = train.load_dataset(field_data)
    assert dataset.sampler_schema == "euclidean_gaussian_fft_v1"
    assert dataset.sample_semantics == "independent_iid_groups_not_markov_chains"
    assert dataset.qualification == manifest["qualification"]


@pytest.mark.parametrize("which", ["data.npz", "sidecar.json"])
def test_loader_rejects_changed_shard_or_sidecar(field_data, which):
    with (field_data / "cell_0" / which).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        train.load_dataset(field_data)


def test_loader_rejects_manifest_paths_outside_data_root(field_data):
    manifest = json.loads((field_data / "manifest.json").read_text())
    manifest["shards"][0]["path"] = "../outside.npz"
    train.write_json(field_data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="escapes"):
        train.load_dataset(field_data)


def test_loader_rejects_cell_leakage_even_with_consistent_sidecar_sha(field_data):
    manifest = json.loads((field_data / "manifest.json").read_text())
    sidecar_path = field_data / manifest["shards"][0]["sidecar"]
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["split"] = "test"
    train.write_json(sidecar_path, sidecar)
    manifest["shards"][0]["sidecar_sha256"] = train.sha256_file(sidecar_path)
    train.write_json(field_data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="split"):
        train.load_dataset(field_data)


def _optimizers(model):
    return {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
            for name in ("Q", "C", "R")}


def test_step_matches_simultaneous_independent_reference_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(251)
    model = Phi4QCR(1., hidden=4)
    reference = copy.deepcopy(model)
    batch = torch.randn(2, 4, 9, 9, 1, dtype=torch.float64)
    actual_optimizers, reference_optimizers = _optimizers(model), _optimizers(reference)
    rng1 = torch.Generator().manual_seed(513)
    rng2 = torch.Generator().manual_seed(513)
    metrics = train.independent_step(model, batch, actual_optimizers, generator=rng1, information_pairs=4)
    losses = continuous_objectives(reference, batch, generator=rng2, information_pairs=4)[:3]
    gradients = {}
    for name, loss in zip(("Q", "C", "R"), losses):
        gradients[name] = torch.autograd.grad(loss, list(getattr(reference, name).parameters()))
    for name in ("Q", "C", "R"):
        for parameter, gradient in zip(getattr(reference, name).parameters(), gradients[name]):
            parameter.grad = gradient
        reference_optimizers[name].step()
    reference.Q.normalize_decoder_()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0., atol=0.)
    assert metrics["r_loss"] == pytest.approx(5 * metrics["closure_one_step"] + 5 * metrics["closure_rollout"])
    assert train.optimizer_step_counts(actual_optimizers, 1)["C"]["min"] == 1


def test_two_transition_closure_is_exactly_all_start_anchor():
    torch.set_num_threads(1)
    torch.manual_seed(728)
    model = Phi4QCR(1., hidden=4)
    batch = torch.randn(2, 4, 9, 9, 1, dtype=torch.float64)
    loss = continuous_objectives(model, batch, generator=torch.Generator().manual_seed(116),
                                 information_pairs=4)[2]
    levels = model.coarse_grain(batch, generator=torch.Generator().manual_seed(116))
    g = []
    for level in levels:
        a, b = level[:, :2].detach(), level[:, 2:].detach()
        g.append((.5 * (model.Q.from_features(model.features(a)) +
                         model.Q.from_features(model.features(b)))).detach())
    error = lambda a, b: (a - b).square().sum(dim=-1).mean()
    expected = 5 * error(model.advance(g[0]), g[1]) + 2.5 * (
        error(model.advance(model.advance(g[0])), g[2]) + error(model.advance(g[1]), g[2]))
    torch.testing.assert_close(loss, expected, atol=1e-13, rtol=1e-13)
    actual_grad = torch.autograd.grad(loss, tuple(model.R.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(model.R.parameters()))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_full_evaluation_uses_matching_half_encoder_and_is_rng_isolated(field_data):
    torch.set_num_threads(1)
    dataset = train.load_dataset(field_data)
    torch.manual_seed(651)
    model = Phi4QCR(dataset.scale, hidden=4)
    before = train.state_fingerprint(model)
    a = train.evaluate_model(model, dataset, torch.device("cpu"))
    torch.randn(1000)
    b = train.evaluate_model(model, dataset, torch.device("cpu"))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["train_direct"].shape == (8, 3, 1)
    assert a["test_one_step"].shape == (4, 2, 1)
    assert a["train_direct_full"].shape == a["train_direct"].shape
    assert a["train_feature_target"].shape == (8, 3, 105)
    assert a["test_feature_reconstruction"].shape == (4, 3, 105)
    assert a["test_one_step_feature_prediction"].shape == (4, 2, 105)
    assert a["test_rollout_feature_prediction"].shape == (4, 3, 105)
    with torch.no_grad():
        for split in ("train", "test"):
            for input_name, decoded_name in (("direct", "feature_reconstruction"),
                    ("one_step", "one_step_feature_prediction"),
                    ("rollout", "rollout_feature_prediction")):
                expected = model.Q.decode(torch.from_numpy(a[f"{split}_{input_name}"])).numpy()
                np.testing.assert_allclose(a[f"{split}_{decoded_name}"], expected, atol=1e-14, rtol=1e-14)
    assert np.max(np.abs(a["train_direct_full"] - a["train_direct"])) > 0
    np.testing.assert_array_equal(a["train_rollout"][:, 0], a["train_direct"][:, 0])
    assert train.state_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nrmse_uses_train_population_target_variance_and_zero_is_null():
    train_direct = np.array([[[0.], [1.], [2.]], [[0.], [3.], [2.]]])
    direct = np.array([[[0.], [2.], [2.]]])
    one = np.array([[[3.], [4.]]])
    rollout = np.array([[[0.], [3.], [4.]]])
    metric = train.closure_metrics(direct, one, rollout, train_direct)
    assert metric["one_step"][0]["rmse"] == 1
    assert metric["one_step"][0]["nrmse"] == 1
    assert metric["one_step"][1]["rmse"] == 2
    assert metric["one_step"][1]["nrmse"] is None


def test_scalar_flow_roots_are_discrete_map_roots_not_certified_critical_points():
    class Affine:
        def advance(self, x):
            return .5 * x + .1

    direct = np.array([[[-.5], [0.], [.5]], [[.5], [1.], [1.5]]])
    arrays = {"train_direct": direct, "test_direct": direct}
    roots = train.scalar_flow(Affine(), arrays, torch.device("cpu"))
    assert len(roots) == 1
    root = roots[0]
    assert root["g"] == pytest.approx(.2)
    assert root["R_derivative"] == pytest.approx(.5)
    assert root["classification"] == "attracting"
    assert root["physical_critical_point_certified"] is False and root["thermal_nu"] is None
    np.testing.assert_allclose(arrays["flow_displacement"], arrays["flow_R"] - arrays["flow_g"])


def test_cell_grid_distinguishes_R_of_mean_from_mean_R_without_terminal_extrapolation():
    class Nonlinear:
        def advance(self, x):
            return x.square()

    arrays = {}
    for split, ids, values in (("train", [0, 0], [1., 3.]), ("test", [1, 1], [2., 4.])):
        direct = np.tile(np.asarray(values)[:, None, None], (1, 3, 1))
        arrays.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, :2] ** 2,
                       f"{split}_cell_ids": np.asarray(ids), f"{split}_labels": np.zeros((2, 3))})
    train.cell_grid_evaluation(Nonlinear(), arrays, torch.device("cpu"))
    assert arrays["grid_R_of_mean"].shape == (2, 2, 1)
    np.testing.assert_array_equal(arrays["grid_R_of_mean"][:, 0, 0], [4., 9.])
    np.testing.assert_array_equal(arrays["grid_mean_R"][:, 0, 0], [5., 10.])
    np.testing.assert_array_equal(arrays["grid_jensen_difference"], np.ones((2, 2, 1)))


def test_tiny_training_records_steps_provenance_and_frozen_final_evaluation(field_data, tmp_path):
    output = tmp_path / "training"
    options = train.TrainingOptions(str(field_data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, log_every=1,
        checkpoint_every=1, max_training_seconds=120.)
    summary = train.run_training(options)
    status = json.loads((output / "status.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    history = json.loads((output / "history_progress.json").read_text())
    assert summary["actual_steps"] == status["actual_steps"] == 2
    assert status["workflow_complete"] and status["returncode"] == 0
    assert status["summary_sha256"] == train.sha256_file(output / "summary.json")
    assert manifest["recipe"] == train.RECIPE
    assert not manifest["labels_used_in_training"]
    assert manifest["closure_scale_weights"] is None
    assert manifest["closure_objective"] == "all_start_anchor"
    assert manifest["closure_target_scales"] == 2
    assert manifest["initial_state_sha256"] != summary["final_state_sha256"]
    assert summary["frozen_evaluation_verified"]
    assert summary["data_warning_count"] == 7 and not summary["equilibrium_certified"]
    assert summary["thermal_nu"] is None
    assert summary["timing"]["training_seconds"] > 0
    assert summary["timing"]["evaluation_seconds"] > 0
    assert summary["optimizer_steps"]["Q"]["min"] == summary["optimizer_steps"]["R"]["max"] == 2
    assert [row["step"] for row in history["history"]] == [1, 2]
    assert (output / "training_step_2.pt").is_file()
    assert status["evaluation_arrays_sha256"] == train.sha256_file(output / "evaluation_arrays.npz")
    old_hash = train.sha256_file(output / "manifest.json")
    with pytest.raises(FileExistsError):
        train.run_training(options)
    assert train.sha256_file(output / "manifest.json") == old_hash


def test_budget_hold_records_zero_actual_updates_without_rerun(field_data, tmp_path):
    output = tmp_path / "held"
    options = train.TrainingOptions(str(field_data), str(output), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, max_training_seconds=1e-12)
    with pytest.raises(train.BudgetHold):
        train.run_training(options)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "budget_hold"
    assert status["actual_steps"] == 0
    assert not status["workflow_complete"] and status["requires_user_action"]
    assert not (output / "summary.json").exists()


def test_invalid_data_records_failed_status_not_complete(field_data, tmp_path):
    (field_data / "cell_0" / "data.npz").write_bytes(b"bad")
    output = tmp_path / "failed"
    options = train.TrainingOptions(str(field_data), str(output), device="cpu")
    with pytest.raises(ValueError, match="SHA256"):
        train.run_training(options)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["returncode"] == 1
    assert status["actual_steps"] == 0 and "traceback" in status


def test_gpu_uuid_matching_is_exact_not_gpu_index_assumption(monkeypatch):
    monkeypatch.setattr(train.torch.cuda, "get_device_properties", lambda device:
        SimpleNamespace(uuid="GPU-12345678-1234-5678-1234-567812345678", name="test", total_memory=123))
    receipt = train._device_receipt(torch.device("cuda:2"), "12345678-1234-5678-1234-567812345678")
    assert receipt["index"] == 2
    with pytest.raises(RuntimeError, match="UUID"):
        train._device_receipt(torch.device("cuda:2"), "00000000-1234-5678-1234-567812345678")
