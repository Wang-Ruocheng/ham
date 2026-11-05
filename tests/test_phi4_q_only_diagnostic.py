"""Small CPU-only checks for the isolated continuous Q diagnostics."""
import copy
import json
import warnings

import numpy as np
import pytest
import torch

from experiments import phi4_q_only_diagnostic as diagnostic
from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_qcr import FEATURE_DIM, Phi4QCR


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(527)
    source = Phi4QCR(.8, hidden=4)
    run = tmp_path / "source"
    run.mkdir()
    manifest = dict(status="complete", theory_dim=1, hidden=4, scale=.8)
    write_json(run / "manifest.json", manifest)
    write_json(run / "summary.json", dict(status="complete", actual_steps=3000,
        latest_checkpoint_step=3000, final_state_sha256=state_fingerprint(source),
        equilibrium_certified=False, data_warning_count=11))
    torch.save(dict(step=3000, model=source.state_dict()), run / "training_step_3000.pt")
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(947)
    train = rng.normal(.01, .003, size=(2, 8, 3, 12, FEATURE_DIM))
    evaluation_train = rng.normal(.01, .003, size=(8, 3, 12, FEATURE_DIM))
    evaluation_test = rng.normal(.01, .003, size=(4, 3, 12, FEATURE_DIM))
    arrays = dict(train_per_sample=train)
    for name, value in (("train", evaluation_train), ("test", evaluation_test)):
        arrays[f"eval_{name}_first"] = value[:, :, :6].mean(2)
        arrays[f"eval_{name}_second"] = value[:, :, 6:].mean(2)
        arrays[f"eval_{name}_labels"] = np.full((len(value), 3), 1e100)
        arrays[f"eval_{name}_cell_ids"] = np.arange(len(value)) // 2
        arrays[f"eval_{name}_chain_ids"] = np.arange(len(value))
    np.savez(cache / "arrays.npz", **arrays)
    receipt = dict(source_run=str(run), scale=.8,
        source_checkpoint_sha256=sha256_file(run / "training_step_3000.pt"),
        source_state_sha256=state_fingerprint(source),
        source_manifest_sha256=sha256_file(run / "manifest.json"),
        arrays_sha256=sha256_file(cache / "arrays.npz"))
    write_json(cache / "receipt.json", receipt)
    return source, run, cache, arrays


def test_matched_initializations_and_frozen_original(inputs):
    source = inputs[0]
    before = state_fingerprint(source)
    source.requires_grad_(False)
    mean = torch.linspace(.001, .02, FEATURE_DIM, dtype=torch.float64)
    arms = diagnostic.make_arms(source, mean, 91)
    torch.manual_seed(91)
    direct = Phi4QCR(.8, hidden=4)
    assert state_fingerprint(arms["B"]) == state_fingerprint(direct.Q)
    assert state_fingerprint(arms["A"]) == state_fingerprint(source.Q)
    assert state_fingerprint(arms["C"]) == state_fingerprint(arms["E"])
    for key, value in arms["B"].state_dict().items():
        if key != "decoder.bias":
            torch.testing.assert_close(value, arms["C"].state_dict()[key], atol=0., rtol=0.)
    torch.testing.assert_close(arms["C"].decoder.bias, mean)
    for key, value in arms["C"].decoder.state_dict().items():
        torch.testing.assert_close(value, arms["D"].decoder.state_dict()[key], atol=0., rtol=0.)
    assert state_fingerprint(source) == before
    assert all(not parameter.requires_grad for parameter in source.parameters())
    assert all(parameter.requires_grad for encoder in arms.values() for parameter in encoder.parameters())


def test_objective_matches_original_and_routes_only_to_Q(inputs):
    source = inputs[0].requires_grad_(False)
    arms = diagnostic.make_arms(source, torch.zeros(FEATURE_DIM, dtype=torch.float64))
    first = torch.randn(8, 3, FEATURE_DIM, dtype=torch.float64, requires_grad=True)
    second = torch.randn(8, 3, FEATURE_DIM, dtype=torch.float64, requires_grad=True)
    loss, terms = diagnostic.q_objective(arms["C"], first, second)
    ga, gb = arms["C"].from_features(first), arms["C"].from_features(second)
    reconstruction = (arms["C"].decode(.5 * (ga + gb)) - .5 * (first + second)).square().sum(-1).mean(0)
    subset = (ga - gb).square().sum(-1).mean(0)
    expected = 10 * reconstruction.mean() + 20 * subset[0] + 5 * subset[1:].mean()
    torch.testing.assert_close(loss, expected, atol=1e-13, rtol=1e-13)
    loss.backward()
    assert first.grad is None and second.grad is None
    assert all(parameter.grad is None for parameter in source.parameters())
    assert all(parameter.grad is not None for parameter in arms["C"].parameters())
    torch.testing.assert_close(terms["q_full_objective"], loss)


def test_E_reports_full_loss_while_optimizing_reconstruction_only(inputs):
    q = inputs[0].Q
    first = torch.randn(8, 3, FEATURE_DIM, dtype=torch.float64)
    second = first + .3
    optimized, terms = diagnostic.q_objective(q, first, second, reconstruction_only=True)
    assert terms["q_full_objective"] > optimized
    torch.testing.assert_close(optimized, 10 * terms["reconstruction_l2"])
    np.testing.assert_allclose(terms["optimization_objective"].detach().numpy(), optimized.detach().numpy())


def test_linear_encoder_half_mean_equals_pooled_exactly(inputs):
    q = diagnostic.make_arms(inputs[0], torch.zeros(FEATURE_DIM, dtype=torch.float64))["D"]
    first = torch.randn(8, 3, FEATURE_DIM, dtype=torch.float64)
    second = torch.randn_like(first)
    torch.testing.assert_close(.5 * (q(first) + q(second)), q(.5 * (first + second)),
                               atol=1e-15, rtol=1e-14)


def test_cache_loader_never_accesses_evaluation_labels(inputs, monkeypatch):
    original_load = np.load

    class StatisticsOnly:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key in diagnostic.ARRAY_KEYS
            return self.archive[key]

    monkeypatch.setattr(diagnostic.np, "load", StatisticsOnly)
    features, _ = diagnostic.load_cache(inputs[2])
    assert set(features) == set(diagnostic.ARRAY_KEYS)


def test_paired_schedule_reproducible_and_half_features_exact(inputs):
    features, _ = diagnostic.load_cache(inputs[2])
    options = diagnostic.Options("unused", "unused", "unused", steps=20, sample_batch=8)
    a, b = (diagnostic.schedule(features["train_per_sample"], options) for _ in range(2))
    assert a[0] == b[0]
    torch.testing.assert_close(a[1], b[1], atol=0., rtol=0.)
    torch.testing.assert_close(a[2], b[2], atol=0., rtol=0.)
    first, second = diagnostic.statistic_halves(features["train_per_sample"], a[0][0], a[1][0], a[2][0])
    expected = features["train_per_sample"][a[0][0], a[1][0]][:, :, a[2][0]].mean(2)
    torch.testing.assert_close(.5 * (first + second), expected, atol=1e-17, rtol=1e-14)


def test_small_run_fixed_steps_source_unchanged_and_full_evaluation(inputs, tmp_path):
    source, run, cache, _ = inputs
    source_before = sha256_file(run / "training_step_3000.pt")
    output = tmp_path / "out"
    summary = diagnostic.train(diagnostic.Options(str(run), str(cache), str(output),
        steps=20, sample_batch=8, log_every=10, cpu_budget_seconds=30))
    assert summary["status"] == "complete"
    assert summary["actual_steps_by_arm"] == dict.fromkeys(diagnostic.ARM_NAMES, 20)
    assert all(value["min"] == value["max"] == 20 for value in summary["optimizer_steps"].values())
    assert summary["frozen_source_before"] == summary["frozen_source_after"]
    assert summary["source_gradients_none"] and summary["labels_used_in_training"] is False
    assert sha256_file(run / "training_step_3000.pt") == source_before
    status = json.loads((output / "status.json").read_text())
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert status["history_sha256"] == sha256_file(output / "history_progress.json")
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["A_train_direct"].shape == (8, 3, 1)
        np.testing.assert_array_equal(arrays["eval_test_labels"], np.full((4, 3), 1e100))
        std = arrays["train_std_by_scale"]
        for name in ("source", *diagnostic.ARM_NAMES):
            for split in ("train", "test"):
                rmse = np.sqrt(np.mean((arrays[f"{name}_{split}_reconstruction"] -
                    arrays[f"{split}_target"]) ** 2, axis=(0, 2)))
                metric = summary["arms"][name]["metrics"][split]
                np.testing.assert_allclose(metric["reconstruction_rmse_by_scale"], rmse, rtol=1e-13)
                np.testing.assert_allclose(metric["reconstruction_nrmse_by_scale"], rmse / std, rtol=1e-13)
        assert np.max(np.abs(arrays["D_train_direct"] - arrays["D_train_direct_full"])) < 1e-15
    history = json.loads((output / "history_progress.json").read_text())["history"]
    assert [row["step"] for row in history] == [0, 1, 10, 20]
    assert history[0]["arms"]["C"]["train"] == history[0]["arms"]["E"]["train"] | {
        "optimization_objective": history[0]["arms"]["C"]["train"]["optimization_objective"]}
    assert all((output / name / "Q_step_20.pt").exists() for name in diagnostic.ARM_NAMES)


def test_budget_hold_never_shortens_steps_or_mutates_source(inputs, tmp_path):
    _, run, cache, _ = inputs
    fingerprint = sha256_file(run / "training_step_3000.pt")
    output = tmp_path / "hold"
    result = diagnostic.train(diagnostic.Options(str(run), str(cache), str(output),
        steps=20, sample_batch=8, cpu_budget_seconds=1e-9))
    assert result["status"] == "budget_hold"
    status = json.loads((output / "status.json").read_text())
    assert status["requires_user_action"] and not status["workflow_complete"]
    assert status["actual_steps_by_arm"] == dict.fromkeys(diagnostic.ARM_NAMES, 0)
    assert sha256_file(run / "training_step_3000.pt") == fingerprint
    assert not (output / "summary.json").exists()


def test_trainer_public_schema_integrates_with_read_only_report(inputs, tmp_path):
    from PIL import Image
    from scripts.report_phi4_q_diagnostic import load_diagnostic, write_report
    _, run, cache, original_arrays = inputs
    arrays = copy.deepcopy(original_arrays)
    for split, offset in (("train", 0), ("test", 10)):
        ids = arrays[f"eval_{split}_cell_ids"] + offset
        arrays[f"eval_{split}_cell_ids"] = ids
        arrays[f"eval_{split}_labels"] = np.column_stack((ids / 10.,
            np.full(len(ids), .2), np.zeros(len(ids))))
    np.savez(cache / "arrays.npz", **arrays)
    receipt = json.loads((cache / "receipt.json").read_text())
    receipt["arrays_sha256"] = sha256_file(cache / "arrays.npz")
    write_json(cache / "receipt.json", receipt)
    output = tmp_path / "integration"
    diagnostic.train(diagnostic.Options(str(run), str(cache), str(output),
        steps=20, sample_batch=8, log_every=10, cpu_budget_seconds=30))
    evidence = load_diagnostic(output)
    assert set(evidence["metrics"]) == {"source", *diagnostic.ARM_NAMES}
    assert {"constant", "PCA1"}.issubset(evidence["baselines"])
    assert evidence["summary"]["actual_steps_by_arm"] == dict.fromkeys(diagnostic.ARM_NAMES, 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        receipt, metrics = write_report(output, tmp_path / "report")
    assert receipt["status"] == "complete" and metrics["actual_steps"] == 20
    assert not any("Glyph" in str(item.message) and "missing" in str(item.message) for item in caught)
    pngs = list((tmp_path / "report").glob("*.png"))
    assert len(pngs) == 4
    for path in pngs:
        with Image.open(path) as image:
            assert image.width >= 1000 and image.height >= 500
            image.verify()


def test_cache_tampering_fails(inputs):
    with (inputs[2] / "arrays.npz").open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="SHA256"):
        diagnostic.load_cache(inputs[2])


def test_output_never_overwrites_existing_directory(inputs, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        diagnostic.train(diagnostic.Options(str(inputs[1]), str(inputs[2]), str(output), steps=1))
