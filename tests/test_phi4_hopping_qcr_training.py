"""Aligned runner checks on synthetic tiny CPU fields, never physical science."""
import copy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from experiments import phi4_hopping_qcr as training
from recurrent_rg.data import phi4_hopping_loader as loader
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, aligned_objectives


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "physical"
    root.mkdir()
    semantics = dict(schema=loader.SCHEMA, parameterization=loader.PARAMETERIZATION,
        evaluation_label_columns=list(loader.EVALUATION_LABEL_COLUMNS),
        field_representation="phi", labels_used_in_training=False)
    options = dict(dimension=2, sizes=[9], chains=4, samples=4,
        kappa_values=[.58, .68, .78], quartic_lambda=1., h=0., test_every=3, test_offset=1)
    pins = {"synthetic_fixture_only.py": "a" * 64}
    training.write_json(root / "plan.json", dict(**semantics, schema_version=5,
        options=options, source_pins=pins, train_cell_indices=[0, 2], test_cell_indices=[1]))
    shards = []
    for cell, kappa in enumerate(options["kappa_values"]):
        directory = root / f"c{cell}"
        directory.mkdir()
        fields = np.random.default_rng(1851 + cell).normal(size=(4, 4, 9, 9, 1))
        np.savez(directory / "data.npz", configurations=fields,
            unused_diagnostic=np.array([{"not_unpickled": True}], dtype=object))
        digest = training.sha256_file(directory / "data.npz")
        split = "test" if cell == 1 else "train"
        training.write_json(directory / "sidecar.json", dict(**semantics, status="complete",
            cell_index=cell, size=9, dimension=2, kappa=kappa, quartic_lambda=1., h=0.,
            split=split, source_pins=pins, independent_chains=4, samples_per_chain=4,
            data_sha256=digest))
        shards.append(dict(cell_index=cell, size=9, split=split,
            path=f"c{cell}/data.npz", sha256=digest, sidecar=f"c{cell}/sidecar.json",
            sidecar_sha256=training.sha256_file(directory / "sidecar.json")))
    training.write_json(root / "manifest.json", dict(**semantics, status="complete",
        options=options, shards=shards, source_pins=pins,
        plan_sha256=training.sha256_file(root / "plan.json"), warning_count=3,
        qualification="Synthetic implementation fixture; not physical data"))
    training.write_json(root / "status.json", dict(schema=loader.SCHEMA, status="complete",
        workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        manifest_sha256=training.sha256_file(root / "manifest.json")))
    return root


def _options(tmp_path, **changes):
    options = training.TrainingOptions(data=str(tmp_path / "physical"),
        output=str(tmp_path / "training"))
    return replace(options, **changes)


def _optimizers(model):
    return {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
        for name in ("Q", "C", "R")}


def test_current_defaults_are_explicit_and_dry(tmp_path, monkeypatch, capsys):
    options = _options(tmp_path)
    assert (options.steps, options.system_batch, options.sample_batch, options.seed) == (3000, 32, 64, 91)
    assert options.threads == 1
    monkeypatch.setattr(training, "load_dataset", lambda *_: pytest.fail("Dry plan loaded data"))
    monkeypatch.setattr(training, "run_training", lambda *_: pytest.fail("Dry plan executed"))
    before = torch.get_rng_state().clone()
    plan = training.main(["--data", options.data, "--output", options.output])
    assert torch.equal(before, torch.get_rng_state())
    assert plan["execution_requested"] is False and plan["formal_training_started"] is False
    assert plan["loss"]["closure_objective"] == "all_start_anchor"
    assert plan["model"]["block_size"] == plan["model"]["stride"] == 3
    assert not (tmp_path / "training").exists()
    assert json.loads(capsys.readouterr().out)["recipe"] == training.RECIPE


@pytest.mark.parametrize("changes", [
    {"steps": True}, {"steps": 1.5}, {"steps": 0}, {"seed": -1}, {"seed": False},
    {"sample_batch": 3}, {"sample_batch": 5}, {"sample_batch": 4.0},
    {"system_batch": 0}, {"threads": 2}, {"information_pairs": 1},
    {"max_training_seconds": float("nan")}, {"max_training_seconds": True},
    {"max_cuda_bytes": 0}, {"device": "cuda"}, {"device": "auto"},
    {"device": "cpu", "expected_gpu_uuid": "abc"}, {"output": ""}, {"data": ""},
])
def test_invalid_options_fail_before_output_creation(tmp_path, changes):
    with pytest.raises((ValueError, FileExistsError)):
        _options(tmp_path, **changes).validate()
    assert not (tmp_path / "training").exists()


def test_existing_output_never_overwritten(tmp_path):
    options = _options(tmp_path)
    path = tmp_path / "training"
    path.mkdir()
    sentinel = path / "old.json"
    sentinel.write_text("old")
    with pytest.raises(FileExistsError):
        training.run_training(options)
    assert sentinel.read_text() == "old"


def test_strict_new_config_and_cli_override(tmp_path, capsys):
    config = tmp_path / "aligned.json"
    training.write_json(config, dict(schema_version=5, recipe=training.RECIPE,
        data=str(tmp_path / "physical"), output=str(tmp_path / "training"),
        model=training.MODEL_POLICY, loss=training.LOSS_POLICY,
        training=dict(system_batch=32, sample_batch=64, steps=3000)))
    plan = training.main(["--config", str(config), "--device", "cpu"])
    assert plan["options"]["device"] == "cpu"
    assert plan["options"]["steps"] == 3000
    capsys.readouterr()
    value = json.loads(config.read_text())
    for target, change in (("recipe", "historical"), ("schema_version", 4),
            ("model", dict(training.MODEL_POLICY, block_size=2)),
            ("loss", dict(training.LOSS_POLICY, closure_objective="one_step_and_rollout")),
            ("training", {"allow_nondefault_recipe": True})):
        training.write_json(config, dict(value, **{target: change}))
        with pytest.raises(ValueError):
            training.load_config(config)


def test_initialization_caps_train_only_and_preserves_rng_and_modules():
    torch.set_num_threads(1)
    torch.manual_seed(417)
    model = AlignedPhi4QCR(1., hidden=4, local_features=3)
    reference = copy.deepcopy(model)
    configurations = torch.randn(42, 4, 9, 9, 1, dtype=torch.float64)
    indices = np.arange(1, 41, dtype=np.int64)
    before = torch.get_rng_state().clone()
    receipt = training.initialize_q_decoder(model, configurations, indices, seed=51)
    with torch.no_grad():
        levels = reference.coarse_grain(configurations[indices[:32]],
            generator=torch.Generator().manual_seed(9051))
        expected = torch.stack([reference.features(level).mean(0) for level in levels]).mean(0)
    torch.testing.assert_close(model.Q.decoder.bias, expected, rtol=0., atol=0.)
    assert torch.equal(before, torch.get_rng_state())
    assert receipt["selected_train_indices"] == list(range(1, 33))
    assert receipt["physical_ensemble_count"] == 32
    assert receipt["labels_used"] is receipt["test_data_used"] is False
    for name in ("C", "R"):
        assert training.state_fingerprint(getattr(model, name)) == training.state_fingerprint(getattr(reference, name))
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [43], [0.5]])
def test_invalid_initialization_indices_rejected(indices):
    model = AlignedPhi4QCR(1., hidden=4, local_features=3)
    with pytest.raises(ValueError):
        training.initialize_q_decoder(model,
            torch.ones(2, 4, 9, 9, 1, dtype=torch.float64), indices, seed=1)


def test_independent_step_matches_all_gradients_before_any_update():
    torch.set_num_threads(1)
    torch.manual_seed(164)
    model = AlignedPhi4QCR(1., hidden=4, local_features=3)
    reference = copy.deepcopy(model)
    batch = torch.randn(2, 4, 9, 9, 1, dtype=torch.float64)
    optimizers, other = _optimizers(model), _optimizers(reference)
    row = training.independent_step(model, batch, optimizers,
        generator=torch.Generator().manual_seed(718), information_pairs=4)
    losses = aligned_objectives(reference, batch,
        generator=torch.Generator().manual_seed(718), information_pairs=4)[:3]
    gradients = {name: torch.autograd.grad(loss, tuple(getattr(reference, name).parameters()))
        for name, loss in zip(("Q", "C", "R"), losses)}
    for name in ("Q", "C", "R"):
        for parameter, gradient in zip(getattr(reference, name).parameters(), gradients[name]):
            parameter.grad = gradient
        other[name].step()
    reference.Q.normalize_decoder_()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0., atol=0.)
    assert all(np.isfinite(value) for value in row.values())
    assert all(clock["min"] == clock["max"] == 1
        for clock in training.optimizer_step_counts(optimizers, 1).values())


def test_evaluation_has_new_joint_targets_and_train_only_flow_bounds(data_root):
    torch.set_num_threads(1)
    dataset = loader.load_dataset(data_root)
    torch.manual_seed(639)
    model = AlignedPhi4QCR(dataset.scale, hidden=4, local_features=3)
    before, rng = training.state_fingerprint(model), torch.get_rng_state().clone()
    arrays = training.evaluate_model(model, dataset, torch.device("cpu"))
    again = training.evaluate_model(model, dataset, torch.device("cpu"))
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], again[name])
    assert torch.equal(rng, torch.get_rng_state())
    assert training.state_fingerprint(model) == before
    assert arrays["direct_g"].shape == arrays["rollout_g"].shape == (12, 3, 1)
    assert arrays["one_step_g"].shape == (12, 2, 1)
    assert arrays["target_features"].shape == arrays["reconstructed_features"].shape == (12, 3, 625)
    np.testing.assert_allclose(arrays["target_features"].sum(-1), 1., atol=1e-12)
    np.testing.assert_array_equal(arrays["rollout_g"][:, 0], arrays["direct_g"][:, 0])
    values = arrays["direct_g"][dataset.train_indices, :-1].reshape(-1)
    padding = .1 * max(np.ptp(values), 1e-6)
    np.testing.assert_allclose(arrays["flow_grid"][[0, -1], 0], [values.min() - padding, values.max() + padding])
    with torch.no_grad():
        np.testing.assert_array_equal(arrays["flow_next"],
            model.advance(torch.from_numpy(arrays["flow_grid"])).numpy())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_labels_do_not_affect_any_model_array(data_root):
    dataset = loader.load_dataset(data_root)
    torch.manual_seed(921)
    model = AlignedPhi4QCR(dataset.scale, hidden=4, local_features=3)
    first = training.evaluate_model(model, dataset, torch.device("cpu"))
    dataset.labels = np.full_like(dataset.labels, np.nan)
    # Nonfinite metadata is caught at public evaluation boundary, not model input.
    with pytest.raises(FloatingPointError, match="labels"):
        training.evaluate_model(model, dataset, torch.device("cpu"))
    dataset.labels = np.ones_like(dataset.labels) * 911.
    second = training.evaluate_model(model, dataset, torch.device("cpu"))
    for name in set(first) - {"labels"}:
        np.testing.assert_array_equal(first[name], second[name])


def test_two_step_cpu_execution_and_report_contract(data_root, tmp_path, monkeypatch):
    import scripts.report_phi4_hopping_qcr as reporter
    called = []
    def report(run):
        status = json.loads((run / "status.json").read_text())
        assert status["status"] == "reporting" and status["workflow_complete"] is False
        manifest, summary, history, arrays, _ = reporter.load_public_run(run)
        called.append((manifest, summary, history, arrays))
        (run / "report").mkdir()
        training.write_json(run / "report" / "report.json", {"status": "complete"})
    monkeypatch.setattr(reporter, "render_report", report)
    options = _options(tmp_path, data=str(data_root), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, log_every=1,
        checkpoint_every=2, max_training_seconds=120.)
    result = training.run_training(options)
    status = json.loads((tmp_path / "training" / "status.json").read_text())
    assert status["status"] == result["status"] == "complete"
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert result["actual_steps"] == 2 and result["thermal_nu"] is None
    assert all(clock["min"] == clock["max"] == 2 for clock in result["optimizer_steps"].values())
    assert called[0][0]["evaluation_label_columns"] == ["kappa", "quartic_lambda", "h"]
    assert called[0][0]["labels_used_in_training"] is False
    assert called[0][0]["exact_binary_recipe"] is False
    assert len(called[0][2]) == 2 and (tmp_path / "training" / "training_step_2.pt").is_file()
    assert any(item["path"] == "evaluation_arrays.npz" for item in status["public_files"])
    assert not any(item["path"].endswith(".pt") for item in status["public_files"])
    for item in status["public_files"]:
        assert item["sha256"] == training.sha256_file(tmp_path / "training" / item["path"])


@pytest.mark.parametrize("kind", ["ordinary", "budget"])
def test_failures_preserve_output_and_distinguish_budget(data_root, tmp_path, monkeypatch, kind):
    exception = RuntimeError("test failure") if kind == "ordinary" else training.BudgetHold("test budget")
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(training, "independent_step", fail)
    options = _options(tmp_path, data=str(data_root), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, max_training_seconds=120.)
    with pytest.raises(type(exception)):
        training.run_training(options)
    status = json.loads((tmp_path / "training" / "status.json").read_text())
    assert status["status"] == ("failed" if kind == "ordinary" else "budget_hold")
    assert status["returncode"] == (1 if kind == "ordinary" else 3)
    assert status["actual_steps"] == 0 and status["failed_stage"] == "training"
    assert status["requires_user_action"] is True and status["workflow_complete"] is False
    assert (tmp_path / "training" / "manifest.json").is_file()


def test_nrmse_uses_train_population_target_std():
    arrays = dict(direct_g=np.array([[[0.], [1.], [2.]], [[0.], [3.], [2.]], [[0.], [2.], [2.]]]),
        one_step_g=np.array([[[1.], [2.]], [[3.], [2.]], [[3.], [4.]]]),
        rollout_g=np.array([[[0.], [1.], [2.]], [[0.], [3.], [2.]], [[0.], [3.], [4.]]]),
        splits=np.array(["train", "train", "test"]))
    metric = training.closure_metrics(arrays)
    assert metric["test"]["one_step"][0]["rmse"] == metric["test"]["one_step"][0]["nrmse"] == 1.
    assert metric["test"]["one_step"][1]["rmse"] == 2.
    assert metric["test"]["one_step"][1]["nrmse"] is None


@pytest.mark.parametrize("path", ["manifest.json", "plan.json", "c0/data.npz", "c0/sidecar.json"])
def test_recheck_detects_new_data_changes_after_loading(data_root, path):
    dataset = loader.load_dataset(data_root)
    training.verify_dataset_unchanged(data_root, dataset)
    with (data_root / path).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        training.verify_dataset_unchanged(data_root, dataset)


def test_tiny_cpu_execution_generates_actual_chinese_report(data_root, tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try:
        select_chinese_font()
    except RuntimeError:
        pytest.skip("No existing Chinese font; never install fonts in tests")
    options = _options(tmp_path, data=str(data_root), device="cpu", steps=2,
        system_batch=2, sample_batch=4, information_pairs=4, log_every=1,
        checkpoint_every=2, max_training_seconds=120.)
    training.run_training(options)
    directory = tmp_path / "training"
    for name in ("loss_curves.png", "R_vs_g.png", "coupling_by_scale.png", "report.json"):
        assert (directory / "report" / name).is_file()
    status = json.loads((directory / "status.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert status["workflow_complete"] is True and status["report_seconds"] > 0.
    assert status["workflow_wall_seconds"] >= summary["timing"]["wall_seconds"]
    assert "excludes report" in summary["timing_scope"]
