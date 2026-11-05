"""Exact-resume CPU contracts; tiny synthetic fields are not physics results."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_qcr as original
from experiments import phi4_qcr_continue as training


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    """Two genuine updates and optimizer moments, with a synthetic source pin."""
    root = tmp_path / "source"
    run, data = root / "run", root / "data"
    run.mkdir(parents=True)
    data.mkdir()
    original.write_json(data / "manifest.json", {"status": "complete", "synthetic": True})
    data_sha = original.sha256_file(data / "manifest.json")
    dataset = original.FieldDataset(
        configurations=torch.randn(12, 64, 9, 9, 1, dtype=torch.float64,
                                   generator=torch.Generator().manual_seed(791)),
        splits=np.array(["train"] * 8 + ["test"] * 4),
        cell_ids=np.repeat(np.arange(3), 4),
        labels=np.column_stack((np.repeat([-2., -1., 0.], 4), np.ones(12), np.zeros(12))),
        manifest_sha256=data_sha, receipts=[], warning_count=2,
        qualification="Synthetic fields, not equilibrium evidence")
    options = dict(steps=2, seed=91, system_batch=8, sample_batch=64,
                   information_pairs=128, threads=1, log_every=1,
                   checkpoint_every=1000)
    manifest = dict(schema_version=5, recipe=original.RECIPE, options=options,
        data_root=str(data), data_manifest_sha256=data_sha, scale=dataset.scale,
        theory_dim=1, hidden=4, lattice_sizes=[9, 3, 1], block_size=3, stride=3,
        dtype="float64", learning_rate=.001, closure_objective="all_start_anchor",
        closure_scale_weights=None, labels_used_in_training=False,
        source_sha256={str(original.__file__): training.SOURCE_TRAINER_SHA,
                       str(training.model_module.__file__): training.SOURCE_MODEL_SHA})
    torch.manual_seed(91)
    model = original.Phi4QCR(dataset.scale, hidden=4)
    optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                  for name in ("Q", "C", "R")}
    batch_rng = torch.Generator().manual_seed(192)
    field_rng = torch.Generator().manual_seed(293)
    history = []
    for step in (1, 2):
        metrics = training.training_update(model, dataset, optimizers, batch_rng,
                                           field_rng, torch.device("cpu"), options)
        history.append(dict(step=step, elapsed_seconds=step*.1, **metrics))
    checkpoint = copy.deepcopy(training.checkpoint_payload(model, optimizers, batch_rng,
                                                           field_rng, 2, manifest, {}))
    torch.save(checkpoint, run / "training_step_2.pt")
    summary = dict(status="complete", actual_steps=2,
        final_state_sha256=original.state_fingerprint(model),
        final_module_sha256=original.module_fingerprints(model),
        data_manifest_sha256=data_sha, optimizer_steps=original.optimizer_step_counts(optimizers, 2))
    original.write_json(run / "manifest.json", manifest)
    original.write_json(run / "summary.json", summary)
    original.write_json(run / "history_progress.json", dict(history=history, latest_step=2,
                                                            latest_checkpoint_step=2))
    complete = dict(status="complete", workflow_complete=True, error=None,
                    requires_user_action=False, returncode=0)
    original.write_json(root / "status.json", complete)
    original.write_json(root / "job.json", dict(**complete, stages=[dict(
        name="synthetic", status="complete", returncode=0)]))
    original.write_json(run / "status.json", dict(**complete, actual_steps=2,
        latest_checkpoint_step=2, summary_sha256=original.sha256_file(run / "summary.json")))
    for name, value in dict(SOURCE_ROOT=str(root), SOURCE_STEPS=2, ADDITIONAL_STEPS=2,
                            PROFILE_STEPS=2, SOURCE_DATA_SHA=data_sha, HIDDEN=4,
                            LATTICE_SIZES=[9, 3, 1],
                            SOURCE_CHECKPOINT_SHA=original.sha256_file(run / "training_step_2.pt")).items():
        monkeypatch.setattr(training, name, value)
    monkeypatch.setattr(original, "load_dataset", lambda _: dataset)
    return dict(root=root, dataset=dataset, checkpoint=checkpoint, manifest=manifest,
                model=model, optimizers=optimizers, batch_rng=batch_rng, field_rng=field_rng)


def options(source, output, *, profile=False):
    return training.ContinueOptions(source=str(source), output=str(output), device="cpu",
        expected_gpu_uuid=None, additional_steps=2, profile_steps=2,
        profile_only=profile, max_training_seconds=60.)


def assert_tree_equal(actual, expected):
    assert training.state_tree_fingerprint(actual) == training.state_tree_fingerprint(expected)


def test_two_plus_two_matches_four_exactly_including_adam_and_rng(tiny_source):
    source = tiny_source
    global_rng = torch.get_rng_state().clone()
    model, optimizers, batch_rng, field_rng = training.restore_training(
        source["checkpoint"], source["manifest"], source["dataset"],
        torch.device("cpu"), expected_step=2)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert_tree_equal(model.state_dict(), source["checkpoint"]["model"])
    for _ in range(2):
        training.training_update(source["model"], source["dataset"], source["optimizers"],
            source["batch_rng"], source["field_rng"], torch.device("cpu"), source["manifest"]["options"])
        training.training_update(model, source["dataset"], optimizers, batch_rng,
            field_rng, torch.device("cpu"), source["manifest"]["options"])
    assert_tree_equal(model.state_dict(), source["model"].state_dict())
    assert_tree_equal({name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                      {name: optimizer.state_dict() for name, optimizer in source["optimizers"].items()})
    assert torch.equal(batch_rng.get_state(), source["batch_rng"].get_state())
    assert torch.equal(field_rng.get_state(), source["field_rng"].get_state())
    assert all(row["min"] == row["max"] == 4 for row in original.optimizer_step_counts(optimizers, 4).values())


def test_full_continuation_no_reinitialization_and_source_immutable(tiny_source, tmp_path, monkeypatch):
    root, output = tiny_source["root"], tmp_path / "continued"
    before = {str(path): original.sha256_file(path) for path in root.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Q initialization must not rerun during exact continuation")

    monkeypatch.setattr(original, "initialize_q_decoder", forbidden)
    summary = training.run_continuation(options(root, output))
    assert summary["actual_steps"] == 4
    assert summary["source_steps"] == summary["additional_steps"] == 2
    assert summary["fresh_adam"] is summary["q_reinitialized"] is False
    assert summary["data_resampled"] is summary["loss_changed"] is False
    assert summary["frozen_evaluation_verified"] is True
    assert all(row["min"] == row["max"] == 4 for row in summary["optimizer_steps"].values())
    assert (output / "training_step_4.pt").is_file()
    endpoint = torch.load(output / "training_step_4.pt", map_location="cpu", weights_only=False)
    assert endpoint["step"] == 4 and set(endpoint["optimizers"]) == {"Q", "C", "R"}
    assert endpoint["manifest"]["source_steps"] == 2
    history = read(output / "history_progress.json")
    assert [row["step"] for row in history["history"]] == [1, 2, 3, 4]
    assert history["history"][:2] == read(root / "run/history_progress.json")["history"]
    status = read(output / "status.json")
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert status["evaluation_arrays_sha256"] == original.sha256_file(output / "evaluation_arrays.npz")
    after = {str(path): original.sha256_file(path) for path in root.rglob("*") if path.is_file()}
    assert before == after
    # The ordinary report loader accepts the cumulative 4-update endpoint.
    from scripts.report_phi4_qcr import load_public_run
    public = load_public_run(output)
    assert public.manifest["options"]["steps"] == 4


def test_profile_restores_source_then_discards_without_evaluation(tiny_source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Short profile must not do scientific evaluation")

    monkeypatch.setattr(original, "evaluate_model", forbidden)
    output = tmp_path / "profile"
    summary = training.run_continuation(options(tiny_source["root"], output, profile=True))
    assert summary["actual_steps"] == 4 and summary["scientific_result"] is False
    assert summary["timing"]["seconds_per_update"] > 0
    assert not list(output.glob("*.pt")) and not (output / "evaluation_arrays.npz").exists()
    status = read(output / "status.json")
    assert status["profile_complete"] is True and status["workflow_complete"] is False


@pytest.mark.parametrize("mutation,match", [
    ("checkpoint", "checkpoint SHA256"), ("data", "data manifest SHA256"),
    ("architecture", "architecture"), ("counter", "optimizer parameter clocks"),
    ("failed_stage", "Source stages"), ("prefix", "ordered complete prefix")])
def test_source_contract_rejects_changes(tiny_source, mutation, match):
    root = tiny_source["root"]
    if mutation == "checkpoint":
        with (root / "run/training_step_2.pt").open("ab") as handle:
            handle.write(b"tamper")
    elif mutation == "data":
        original.write_json(root / "data/manifest.json", {"status": "complete", "changed": True})
    elif mutation == "architecture":
        value = read(root / "run/manifest.json")
        value["hidden"] += 1
        original.write_json(root / "run/manifest.json", value)
    elif mutation == "failed_stage":
        value = read(root / "job.json")
        value["stages"][0]["returncode"] = 1
        original.write_json(root / "job.json", value)
    elif mutation == "prefix":
        value = read(root / "run/history_progress.json")
        value["history"][0]["step"] = 3
        original.write_json(root / "run/history_progress.json", value)
    else:
        value = read(root / "run/summary.json")
        value["optimizer_steps"]["C"]["min"] = 0
        original.write_json(root / "run/summary.json", value)
        status = read(root / "run/status.json")
        status["summary_sha256"] = original.sha256_file(root / "run/summary.json")
        original.write_json(root / "run/status.json", status)
    with pytest.raises(ValueError, match=match):
        training.source_receipt(root)


@pytest.mark.parametrize("mutation,match", [
    ("step", "recipe/step"), ("adam", "all three Adam"),
    ("lr", "hyperparameters"), ("dtype", "float64"), ("clock", "step counters")])
def test_checkpoint_restore_rejects_incompatible_training_state(tiny_source, mutation, match):
    source = tiny_source
    checkpoint = copy.deepcopy(source["checkpoint"])
    if mutation == "step":
        checkpoint["step"] = 0
    elif mutation == "adam":
        del checkpoint["optimizers"]["C"]
    elif mutation == "lr":
        checkpoint["optimizers"]["Q"]["param_groups"][0]["lr"] = .01
    elif mutation == "dtype":
        key = next(iter(checkpoint["model"]))
        checkpoint["model"][key] = checkpoint["model"][key].float()
    else:
        key = next(iter(checkpoint["optimizers"]["R"]["state"]))
        checkpoint["optimizers"]["R"]["state"][key]["step"].zero_()
    with pytest.raises((ValueError, RuntimeError), match=match):
        training.restore_training(checkpoint, source["manifest"], source["dataset"],
                                  torch.device("cpu"), expected_step=2)


def test_zero_steps_no_overwrite_and_independent_output_required(tiny_source, tmp_path):
    value = options(tiny_source["root"], tmp_path / "unused")
    value.additional_steps = 0
    with pytest.raises(ValueError, match="exactly"):
        value.validate()
    value = options(tiny_source["root"], tiny_source["root"] / "new_run")
    with pytest.raises(ValueError, match="independent"):
        value.validate()
    value = options(tiny_source["root"], tiny_source["root"])
    with pytest.raises(FileExistsError, match="Fresh"):
        value.validate()
