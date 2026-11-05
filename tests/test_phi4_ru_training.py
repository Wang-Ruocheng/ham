"""Tiny synthetic CPU runs; never touches scientific physical data or GPUs."""
import json

import numpy as np
import pytest
import torch

from experiments import phi4_qcr as original
from experiments import phi4_ru_qcr as train
from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_data(root):
    root.mkdir()
    options = dict(dimension=2, sizes=[9], chains=2, samples=4,
        r_values=[-.9, -.5], u_values=[.5, 1.], h=0., test_every=3, test_offset=1)
    shards = []
    for cell in range(4):
        relative = f"c{cell}"
        directory = root / relative
        directory.mkdir()
        rng = np.random.default_rng(100 + cell)
        np.savez(directory / "data.npz", configurations=rng.normal(size=(2, 4, 9, 9, 1)))
        digest = original.sha256_file(directory / "data.npz")
        split = "test" if cell % 3 == 1 else "train"
        original.write_json(directory / "sidecar.json", dict(status="complete", dimension=2,
            cell_index=cell, size=9, r=options["r_values"][cell // 2],
            u=options["u_values"][cell % 2], h=0., split=split, data_sha256=digest))
        shards.append(dict(cell_index=cell, size=9, split=split, path=f"{relative}/data.npz",
            sha256=digest, sidecar=f"{relative}/sidecar.json",
            sidecar_sha256=original.sha256_file(directory / "sidecar.json")))
    original.write_json(root / "manifest.json", dict(status="complete", options=options,
        shards=shards, warning_count=2, qualification="Synthetic nonphysical unit test"))


@pytest.mark.parametrize("branch", ["Q", "C", "R"])
def test_dim_two_loss_routing_and_unchanged_coefficients(branch):
    torch.manual_seed(13)
    model = Phi4QCR(1., hidden=4, theory_dim=2)
    batch = torch.randn(2, 4, 9, 9, 1, dtype=torch.float64)
    q, c, r, m = continuous_objectives(model, batch, torch.Generator().manual_seed(8), 4)
    assert q.item() == pytest.approx(10*m["reconstruction"] + 20*m["subset_micro"] + 5*m["subset_coarse"])
    assert c.item() == pytest.approx(-2*m["information_proxy"] + 2*m["variance_floor"] + m["uniform_amplitude"])
    assert r.item() == pytest.approx(5*m["closure_one_step"] + 5*m["closure_rollout"])
    {"Q": q, "C": c, "R": r}[branch].backward()
    for name in ("Q", "C", "R"):
        gradients = [parameter.grad for parameter in getattr(model, name).parameters()]
        assert any(value is not None for value in gradients) if name == branch else all(value is None for value in gradients)


def test_tiny_training_clocks_checkpoint_public_arrays_and_frozen_evaluation(tmp_path):
    data, output = tmp_path / "data", tmp_path / "run"
    tiny_data(data)
    summary = train.run_training(train.TrainingOptions(str(data), str(output), device="cpu",
        steps=2, system_batch=2, sample_batch=4, information_pairs=4, log_every=1,
        checkpoint_every=1, max_training_seconds=120.))
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    checkpoint = torch.load(output / "training_step_2.pt", map_location="cpu", weights_only=False)
    assert manifest["schema_version"] == 5 and manifest["theory_dim"] == 2
    assert manifest["closure_scale_weights"] is None and manifest["closure_objective"] == "all_start_anchor"
    assert not manifest["labels_used_in_training"] and manifest["model_input_members"] == ["configurations"]
    assert not manifest["continuous_recipe"]["discrete_histogram_MI_or_color_NLL"]
    assert manifest["q_initialization_receipt"]["C_trainable"]
    assert manifest["q_initialization_receipt"]["physical_ensemble_count"] == 6
    assert checkpoint["step"] == 2 and checkpoint["manifest"] == manifest
    assert checkpoint["batch_rng"].device.type == checkpoint["field_rng"].device.type == "cpu"
    assert summary["frozen_evaluation_verified"] and summary["thermal_nu"] is None
    assert all(row["min"] == row["max"] == 2 for row in summary["optimizer_steps"].values())
    assert status["status"] == "complete" and status["returncode"] == 0
    assert original.sha256_file(output / "summary.json") == status["summary_sha256"]
    assert original.sha256_file(output / "evaluation_arrays.npz") == status["evaluation_arrays_sha256"]
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["train_direct"].shape == (6, 3, 2)
        assert arrays["test_direct"].shape == (2, 3, 2)
        assert arrays["train_feature_target"].shape == (6, 3, 105)
        assert arrays["flow_g"].shape == (1089, 2)
        np.testing.assert_allclose(arrays["flow_displacement"], arrays["flow_R"] - arrays["flow_g"])
        assert arrays["grid_R_of_mean"].shape == arrays["grid_mean_R"].shape == (4, 2, 2)
    with pytest.raises(FileExistsError):
        train.run_training(train.TrainingOptions(str(data), str(output), device="cpu"))


def test_discarded_profile_has_independent_seed_no_checkpoint_or_final_science(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "profile"
    tiny_data(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Discarded profile must not run scientific final evaluation")

    monkeypatch.setattr(original, "evaluate_model", forbidden)
    summary = train.run_training(train.TrainingOptions(str(data), str(output), device="cpu",
        steps=3000, seed=191, system_batch=2, sample_batch=4, information_pairs=4,
        log_every=1, profile_only=True, profile_steps=2, max_training_seconds=120.))
    assert summary["actual_steps"] == 2 and summary["profile_endpoint_discarded"]
    assert not summary["frozen_evaluation_verified"] and summary["evaluation_skipped_for_profile"]
    assert not list(output.glob("*.pt")) and not (output / "evaluation_arrays.npz").exists()
    assert summary["timing"]["initialization_seconds"] > 0
    assert len(summary["timing"]["update_seconds"]) == 2
    assert summary["optimizer_steps"]["R"]["min"] == 2


def test_profile_seed_and_budget_and_constructor_mismatch_fail_without_updates(tmp_path):
    with pytest.raises(ValueError, match="seed 191"):
        train.TrainingOptions("unused", str(tmp_path / "x"), profile_only=True).validate()
    data = tmp_path / "data"
    tiny_data(data)
    output = tmp_path / "budget"
    with pytest.raises(original.BudgetHold):
        train.run_training(train.TrainingOptions(str(data), str(output), device="cpu",
            system_batch=2, sample_batch=4, information_pairs=4, max_training_seconds=1e-12))
    assert json.loads((output / "status.json").read_text())["actual_steps"] == 0
    assert not list(output.glob("*.pt"))
    with pytest.raises(RuntimeError, match="construction SHA256"):
        train.run_training(train.TrainingOptions(str(data), str(tmp_path / "mismatch"), device="cpu",
            system_batch=2, sample_batch=4, information_pairs=4,
            expected_random_construction_sha256="0" * 64))
