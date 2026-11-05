"""Tiny CPU-only stopping checks, not physical sampling or experiment results."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_hopping_qcr as training
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR
from test_phi4_hopping_qcr_training import data_root


POLICY = "train_closure_100_200_patience20_v1"


def _b2_data_root(data_root):
    """Rebind only the tiny random-field fixture to 24 -> 12 -> 6 -> 3."""
    plan = json.loads((data_root / "plan.json").read_text())
    manifest = json.loads((data_root / "manifest.json").read_text())
    for value in (plan, manifest):
        value["options"]["sizes"] = [24]
    training.write_json(data_root / "plan.json", plan)
    for cell, shard in enumerate(manifest["shards"]):
        shard["size"] = 24
        path = data_root / shard["path"]
        np.savez(path, configurations=np.random.default_rng(102 + cell).normal(
            size=(4, 4, 24, 24, 1)))
        shard["sha256"] = training.sha256_file(path)
        sidepath = data_root / shard["sidecar"]
        side = json.loads(sidepath.read_text())
        side.update(size=24, data_sha256=shard["sha256"])
        training.write_json(sidepath, side)
        shard["sidecar_sha256"] = training.sha256_file(sidepath)
    manifest["plan_sha256"] = training.sha256_file(data_root / "plan.json")
    training.write_json(data_root / "manifest.json", manifest)
    status = json.loads((data_root / "status.json").read_text())
    status["manifest_sha256"] = training.sha256_file(data_root / "manifest.json")
    training.write_json(data_root / "status.json", status)
    return data_root


def test_probe_is_fixed_train_only_uniform_ranks_and_all_start(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(241)
    model = AlignedPhi4QCR(1., hidden=4, local_features=3, block_size=2,
        steps=3, patch_chunk_size=64)
    fields = torch.randn(28, 10, 24, 24, 1, dtype=torch.float64,
        generator=torch.Generator().manual_seed(812))
    train_indices = np.arange(2, 26, dtype=np.int64)
    expected_indices = train_indices[np.linspace(0, len(train_indices) - 1, 8, dtype=int)]
    assert expected_indices[-1] == train_indices[-1]
    assert expected_indices.tolist() != train_indices[:8].tolist()
    original_coarse_grain = model.coarse_grain
    batches = []

    def coarse_grain(batch, generator=None):
        batches.append(batch.detach().clone())
        torch.testing.assert_close(batch, fields[expected_indices, :8], rtol=0., atol=0.)
        return original_coarse_grain(batch, generator=generator)

    monkeypatch.setattr(model, "coarse_grain", coarse_grain)
    # Existing mixed None/non-None gradients must not be erased by a probe.
    for index, parameter in enumerate(model.parameters()):
        if index % 2:
            parameter.grad = torch.full_like(parameter, .125)
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    state_before = training.state_fingerprint(model)
    cpu_rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        levels = original_coarse_grain(fields[expected_indices, :8],
            generator=torch.Generator().manual_seed(91 + 303))
        direct = torch.stack([.5 * (model.Q(level[:, :4]) + model.Q(level[:, 4:]))
            for level in levels], dim=1)
        expected = 10. * all_start_anchor_closure_loss(model, direct,
            coordinate_reduction="sum")[0].item()
    first = training.fixed_train_closure_probe(model, fields, train_indices, seed=91)
    # TEST chains and the unselected tail fields do not enter the selector.
    other_fields = fields.clone()
    other_fields[[0, 1, 26, 27]] = float("nan")
    other_fields[:, 8:] = float("nan")
    second = training.fixed_train_closure_probe(model, other_fields, train_indices, seed=91)
    assert first == second == expected
    assert len(batches) == 2 and batches[0].shape[:2] == (8, 8)
    assert torch.equal(cpu_rng_before, torch.get_rng_state())
    assert training.state_fingerprint(model) == state_before
    assert model.training is True
    for previous, parameter in zip(gradients, model.parameters()):
        if previous is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, previous, rtol=0., atol=0.)


@pytest.mark.parametrize("changes", [
    dict(steps=199), dict(steps=201), dict(block_size=3, depth=2),
    dict(early_stopping="heldout_minimum"),
])
def test_early_stopping_is_explicit_fixed_b2_policy(tmp_path, changes):
    values = dict(data=str(tmp_path / "data"), output=str(tmp_path / "out"),
        device="cpu", steps=200, block_size=2, depth=3, early_stopping=POLICY)
    values.update(changes)
    with pytest.raises(ValueError):
        training.TrainingOptions(**values).validate()
    assert not (tmp_path / "out").exists()


@pytest.fixture
def completed_early_run(data_root, tmp_path, monkeypatch):
    import scripts.report_phi4_hopping_qcr as reporter
    root = _b2_data_root(data_root)
    output = tmp_path / "early_training"
    seen = dict(update=0, probes=[], report_calls=0, model_at_steps={})
    real_evaluation = training.evaluate_model
    probe_values = {100: 3., 110: 2., 120: 2.5, 130: 3.}

    def cheap_step(model, batch, optimizers, *, generator, information_pairs):
        seen["update"] += 1
        # Exercise all real Adam parameter clocks without costly objectives.
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
        # A tiny synthetic sentinel makes checkpoint restoration observable.
        # This is a mocked implementation fixture, never a scientific update.
        with torch.no_grad():
            model.R[-1].bias.add_(1e-5)
        if seen["update"] in (110, 130):
            seen["model_at_steps"][seen["update"]] = training.state_fingerprint(model)
        return dict(q_loss=.5, c_loss=.25, r_loss=.125)

    def fixed_probe(model, configurations, train_indices, *, seed):
        # The selector API receives only fields and TRAIN indices, never labels,
        # a TEST metric, or evaluator output. The real probe is tested above.
        assert seed == 91
        assert np.asarray(train_indices).tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
        seen["probes"].append(seen["update"])
        return probe_values[seen["update"]]

    def evaluate(model, dataset, device, *, budget_check=None):
        assert seen["probes"] == [100, 110, 120, 130]
        seen["evaluation_before"] = training.state_fingerprint(model)
        assert seen["evaluation_before"] == seen["model_at_steps"][110]
        assert seen["evaluation_before"] != seen["model_at_steps"][130]
        assert all(parameter.grad is None for parameter in model.parameters())
        arrays = real_evaluation(model, dataset, device, budget_check=budget_check)
        seen["evaluation_after"] = training.state_fingerprint(model)
        return arrays

    def report(run):
        status = json.loads((run / "status.json").read_text())
        assert status["status"] == "reporting" and status["workflow_complete"] is False
        manifest, summary, history, arrays, _ = reporter.load_public_run(run)
        assert manifest["options"]["steps"] == 200
        assert summary["actual_steps"] == history[-1]["step"] == 130
        assert summary["evaluated_checkpoint_step"] == 110
        assert arrays["direct_g"].shape == (12, 4, 1)
        seen["report_calls"] += 1
        (run / "report").mkdir()
        training.write_json(run / "report" / "report.json", dict(status="complete"))

    monkeypatch.setattr(training, "independent_step", cheap_step)
    monkeypatch.setattr(training, "fixed_train_closure_probe", fixed_probe)
    monkeypatch.setattr(training, "evaluate_model", evaluate)
    monkeypatch.setattr(reporter, "render_report", report)
    summary = training.run_training(training.TrainingOptions(data=str(root), output=str(output),
        device="cpu", steps=200, block_size=2, depth=3, early_stopping=POLICY,
        system_batch=2, sample_batch=4, information_pairs=4, log_every=10,
        checkpoint_every=100, max_training_seconds=300.))
    return output, summary, seen


def test_workflow_distinguishes_actual_stop_selected_checkpoint_and_maximum(completed_early_run):
    output, summary, seen = completed_early_run
    assert seen["update"] == summary["actual_steps"] == 130
    assert seen["probes"] == [100, 110, 120, 130]
    assert seen["report_calls"] == 1
    assert summary["evaluated_checkpoint_step"] == 110
    assert summary["final_state_sha256"] == seen["evaluation_before"] == seen["evaluation_after"]
    for clock in summary["optimizer_steps"].values():
        assert clock["min"] == clock["max"] == 130 and clock["parameters"] > 0
    for clock in summary["selected_optimizer_steps"].values():
        assert clock["min"] == clock["max"] == 110 and clock["parameters"] > 0
    receipt = json.loads((output / "early_stopping.json").read_text())
    assert receipt == summary["early_stopping"]
    assert receipt["actual_steps"] == 130 and receipt["best_step"] == 110
    assert receipt["best_value"] == 2. and receipt["stop_reason"] == "patience"
    assert receipt["test_data_used"] is receipt["labels_used"] is False
    assert receipt["selected_checkpoint_sha256"] == training.sha256_file(output / "training_step_110.pt")
    for step in (100, 110, 120, 130):
        checkpoint = torch.load(output / f"training_step_{step}.pt", map_location="cpu", weights_only=False)
        assert checkpoint["step"] == step
        for state in checkpoint["optimizers"].values():
            assert all(int(value["step"].item()) == step for value in state["state"].values())
    assert not (output / "training_step_200.pt").exists()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"] is True
    assert status["returncode"] == 0 and status["error"] is None
    assert status["actual_steps"] == status["latest_checkpoint_step"] == 130
    assert status["evaluated_checkpoint_step"] == 110
    assert status["early_stopping"] == receipt
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["early_stopping"]["selected_train_indices"] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert manifest["early_stopping"]["generator_seed"] == 394
    assert manifest["early_stopping"]["test_data_used"] is False
    assert manifest["early_stopping"]["labels_used"] is False
    assert all(not row["path"].endswith(".pt") for row in status["public_files"])
    for row in status["public_files"]:
        assert training.sha256_file(output / row["path"]) == row["sha256"]


@pytest.mark.parametrize("tamper", ["selected_step", "actual_steps", "selected_clock", "observations", "test_selection"])
def test_report_rejects_inconsistent_or_test_selected_receipt(completed_early_run, tamper):
    import scripts.report_phi4_hopping_qcr as reporter
    output, summary, _ = completed_early_run
    invalid = copy.deepcopy(summary)
    if tamper == "selected_step":
        invalid["evaluated_checkpoint_step"] = 120
    elif tamper == "actual_steps":
        invalid["actual_steps"] = 200
    elif tamper == "selected_clock":
        invalid["selected_optimizer_steps"]["Q"]["max"] = 130
    elif tamper == "observations":
        invalid["early_stopping"]["observations"][1]["value"] = 10.
    else:
        invalid["early_stopping"]["test_data_used"] = True
    training.write_json(output / "summary.json", invalid)
    with pytest.raises(ValueError):
        reporter.load_public_run(output)
