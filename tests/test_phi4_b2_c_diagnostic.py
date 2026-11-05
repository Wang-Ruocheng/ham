"""Synthetic checks only; no remote access, sampler or training is run."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import phi4_b2_c_diagnostic as diagnostic
from experiments import phi4_b2_closure_comparison as source
from experiments.phi4_qcr import module_fingerprints, sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_b2_qcr import Phi4B2QCR


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_raw_statistics_are_actual_amplitudes_without_sign_flip():
    field = torch.tensor([1., 2., 3., 4.], dtype=torch.double).reshape(1, 1, 2, 2, 1)
    result = diagnostic.raw_field_statistics(field)
    expected = [2.5, np.sqrt(1.25), np.sqrt(7.5), 2.5, 25., 0.]
    np.testing.assert_allclose(result.numpy(), expected, rtol=0., atol=1e-14)
    reverse = diagnostic.raw_field_statistics(-field)
    assert reverse[0] == -result[0]
    assert reverse[4] == -result[4]


def test_oddization_changes_only_even_mean_and_shares_noise():
    model = Phi4B2QCR(2., hidden=3).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.C.parameters():
            parameter.zero_()
        model.C.net[-1].bias.fill_(.3)
    field = torch.randn((1, 4, 8, 8, 1), dtype=torch.double)
    before = state_fingerprint(model)
    rng_a, rng_b = [torch.Generator().manual_seed(487) for _ in range(2)]
    original, odd, residual = diagnostic.paired_paths(model, field, rng_a, rng_b)
    assert len(original) == len(odd) == 4
    for first, second in zip(original[1:], odd[1:]):
        torch.testing.assert_close(first - second, torch.full_like(first, .6), rtol=0., atol=2e-16)
    np.testing.assert_allclose(residual.numpy(), np.tile([.6, .6, .3, .3], (3, 1)), atol=2e-16)
    assert torch.equal(rng_a.get_state(), rng_b.get_state())
    assert before == state_fingerprint(model)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_random_original_path_replays_actual_model_exactly():
    model = Phi4B2QCR(1.2, hidden=4).eval().requires_grad_(False)
    field = torch.randn((1, 6, 8, 8, 1), dtype=torch.double)
    rng_a, rng_b, rng_reference = [torch.Generator().manual_seed(525) for _ in range(3)]
    original, _, _ = diagnostic.paired_paths(model, field, rng_a, rng_b)
    with torch.no_grad():
        reference = model.coarse_grain(field, rng_reference)
    for actual, expected in zip(original, reference):
        assert torch.equal(actual, expected)
    assert torch.equal(rng_a.get_state(), rng_reference.get_state())


def test_wrong_noise_pairing_is_not_silently_accepted():
    model = Phi4B2QCR(1., hidden=3).eval().requires_grad_(False)
    field = torch.zeros((1, 4, 8, 8, 1), dtype=torch.double)
    with pytest.raises(RuntimeError, match="noise identity"):
        diagnostic.paired_paths(model, field, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))


def test_softbin_mirror_statistic_and_endpoint_mass():
    model = Phi4B2QCR(1., hidden=3).eval().requires_grad_(False)
    positives = torch.ones((1, 2, 8, 8, 1), dtype=torch.double)
    field = torch.cat((positives, -positives), dim=1)
    g, target, mirror, endpoint = diagnostic.feature_statistics(model, field)
    assert g.shape == (1,)
    assert target.shape == (105,)
    assert float(mirror) < 1e-15
    assert 0 <= float(endpoint) <= 1


def test_replay_checks_all_scales_and_features():
    dataset = SimpleNamespace(splits=np.array(["train", "test"]), cell_ids=np.array([0, 1]),
                              labels=np.array([[0., .2, 0.], [.1, .2, 0.]]))
    arrays = dict(original_direct=np.zeros((2, 4, 1)), original_feature_target=np.zeros((2, 4, 105)))
    saved = {}
    for index, split in enumerate(("train", "test")):
        for key in ("direct", "feature_target"):
            saved[f"all_start_{split}_{key}"] = arrays[f"original_{key}"][index:index+1].copy()
        saved[f"all_start_{split}_labels"] = dataset.labels[index:index+1]
        saved[f"all_start_{split}_cell_ids"] = dataset.cell_ids[index:index+1]
    assert diagnostic.compare_replay(arrays, dataset, saved)["verified"]
    saved["all_start_test_feature_target"][0, 3, 104] += 1e-8
    with pytest.raises(ValueError, match="replay differs"):
        diagnostic.compare_replay(arrays, dataset, saved)


@pytest.fixture
def completed_source(tmp_path):
    root = tmp_path / "source"
    data, training = root / "data", root / "training"
    data.mkdir(parents=True)
    training.mkdir()
    opts = dict(dimension=2, sizes=[32], samples=4, chains=2, r_values=[-.5, -.1, .3],
                u_values=[.2], h=0., test_every=3, test_offset=1)
    shards = []
    for cell, r in enumerate(opts["r_values"]):
        directory = data / f"cell{cell}"
        directory.mkdir()
        fields = np.random.default_rng(751+cell).normal(size=(2, 4, 32, 32, 1))
        np.savez(directory / "data.npz", configurations=fields)
        split = "test" if cell == 1 else "train"
        data_sha = sha256_file(directory / "data.npz")
        write_json(directory / "sidecar.json", dict(status="complete", size=32, dimension=2,
            cell_index=cell, split=split, r=r, u=.2, h=0., data_sha256=data_sha))
        shards.append(dict(cell_index=cell, size=32, split=split, path=f"cell{cell}/data.npz",
            sha256=data_sha, sidecar=f"cell{cell}/sidecar.json", sidecar_sha256=sha256_file(directory / "sidecar.json")))
    write_json(data / "manifest.json", dict(status="complete", options=opts, shards=shards,
        warning_count=3, qualification="Synthetic data, not scientific sampling"))
    dataset = source.load_dataset(data)
    torch.manual_seed(5)
    model = Phi4B2QCR(dataset.scale, hidden=4).eval().requires_grad_(False)
    arrays = source.evaluate(model, dataset, torch.device("cpu"), lambda: None)
    np.savez(training / "evaluation_arrays.npz", **{f"all_start_{k}": v for k, v in arrays.items()})
    import recurrent_rg.phi4_b2_qcr as model_module
    write_json(training / "manifest.json", dict(recipe=source.RECIPE, theory_dim=1,
        lattice_sizes=[32, 16, 8, 4], scale=dataset.scale, data_root=str(data),
        data_manifest_sha256=dataset.manifest_sha256, data_shards=dataset.receipts,
        source_sha256={str(model_module.__file__): sha256_file(model_module.__file__)}))
    modules = module_fingerprints(model)
    write_json(training / "summary.json", dict(recipe=source.RECIPE, status="complete", actual_steps=3000,
        C_Q_pairing_verified=True, arms={name: dict(final_module_sha256=modules) for name in ("all_start", "legacy")}))
    write_json(training / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        **{f"{name}_sha256": sha256_file(training / f"{name}.json") for name in ("manifest", "summary")},
        evaluation_arrays_sha256=sha256_file(training / "evaluation_arrays.npz")))
    checkpoint = training / "all_start" / "training_step_3000.pt"
    checkpoint.parent.mkdir()
    torch.save(dict(recipe=source.RECIPE, step=3000, closure_objective="all_start_anchor", model=model.state_dict()), checkpoint)
    return root


def test_checkpoint_sha_and_source_inputs_are_pinned_before_load(completed_source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong checkpoint should not be loaded")
    monkeypatch.setattr(diagnostic.torch, "load", forbidden)
    with pytest.raises(ValueError, match="pinned identity"):
        diagnostic.load_source(completed_source, "0" * 64)


def test_weights_only_policy_and_endpoint_fingerprint(completed_source, monkeypatch):
    actual_load = torch.load
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs == dict(map_location="cpu", weights_only=True)
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(diagnostic.torch, "load", checked)
    model, _, _, _, pins = diagnostic.load_source(completed_source)
    assert calls and all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert any(path.endswith("training_step_3000.pt") for path in pins)


def test_source_public_tampering_rejected(completed_source):
    path = completed_source / "training" / "summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="receipt mismatch"):
        diagnostic.load_source(completed_source)


def test_full_synthetic_cpu_diagnostic_preserves_all_source_files(completed_source, tmp_path, monkeypatch):
    before = {str(path): sha256_file(path) for path in completed_source.rglob("*") if path.is_file()}
    monkeypatch.setattr(diagnostic, "_plot", lambda arrays, output: None)
    output = tmp_path / "diagnostic"
    result = diagnostic.run(diagnostic.Options(str(completed_source), str(output), device="cpu"))
    assert result["status"] == "complete"
    assert result["source_replay"]["verified"]
    assert result["paired_noise_identical"]
    assert result["model_updates"] == 0
    assert result["gpu_seconds"] == 0
    assert result["thermal_nu"] is None
    assert before == {str(path): sha256_file(path) for path in completed_source.rglob("*") if path.is_file()}
    with np.load(output / "diagnostic_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["original_raw_stats"].shape == (6, 4, 6)
        assert arrays["original_input_odd_residual"].shape == (6, 3, 4)
        assert arrays["counterfactual_feature_target"].shape == (6, 4, 105)
        assert all("configuration" not in key and "field" not in key for key in arrays.files)
        np.testing.assert_array_equal(arrays["original_direct"][:, 0], arrays["counterfactual_direct"][:, 0])
    status = json.loads((output / "status.json").read_text())
    assert status["workflow_complete"] and status["returncode"] == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["initial_model_state_sha256"] == receipt["final_model_state_sha256"]
    with pytest.raises(FileExistsError):
        diagnostic.run(diagnostic.Options(str(completed_source), str(output), device="cpu"))


def test_budget_hold_records_no_completion_or_automatic_retry(completed_source, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_plot", lambda arrays, output: None)
    output = tmp_path / "hold"
    result = diagnostic.run(diagnostic.Options(str(completed_source), str(output), device="cpu", cpu_budget_seconds=1e-12))
    assert result["status"] == "budget_hold"
    status = json.loads((output / "status.json").read_text())
    assert not status["workflow_complete"] and status["returncode"] == 3
    assert not (output / "diagnostic_arrays.npz").exists()


def test_chinese_derived_plots_are_written_from_public_statistics_only(tmp_path):
    arrays = dict(cell_ids=np.array([0, 1]), labels=np.array([[-.5, .2, 0.], [.3, .2, 0.]]),
                  train_rms=np.asarray(1.))
    for path in ("original", "counterfactual"):
        arrays[f"{path}_raw_stats"] = np.zeros((2, 4, 6))
        arrays[f"{path}_mirror_softbin_L1"] = np.zeros((2, 4))
        arrays[f"{path}_direct"] = np.zeros((2, 4, 1))
    diagnostic._plot(arrays, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "amplitude_by_scale.png", "symmetry_by_scale.png", "coupling_by_scale.png"}
    assert all(path.stat().st_size > 1000 for path in tmp_path.iterdir())


@pytest.mark.parametrize("values,error", [
    ({"device": "cuda:0", "expected_gpu_uuid": "GPU-x"}, "fixed cuda:2"),
    ({"device": "cuda:2"}, "UUID"),
    ({"device": "cpu", "cpu_budget_seconds": 0}, "cpu_budget"),
])
def test_invalid_options(tmp_path, values, error):
    with pytest.raises(ValueError, match=error):
        diagnostic.Options(str(tmp_path / "source"), str(tmp_path / "output"), **values).validate()
