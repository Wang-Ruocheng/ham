"""Leakage, repeat averaging and optimizer-selection contracts on tiny data."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments import phi4_ru_r_validation as experiment
from experiments.phi4_qcr import state_fingerprint
from recurrent_rg.phi4_r_calibration import fit_input_normalizer


def tiny_encodings():
    rng = np.random.default_rng(203)
    start = rng.normal(0, .08, (4, 8, 2))
    first = .83*start + np.array([.02, -.03])
    second = .83*first + np.array([.02, -.03])
    fit = torch.tensor(np.stack((start, first, second), axis=-2), dtype=torch.float64)
    val = fit.clone()+torch.tensor([.01, -.005], dtype=torch.float64)
    return fit, val


def tiny_schedule():
    return experiment.Schedule(adam_steps=6, checkpoint_every=2,
        lbfgs_block_iterations=2, lbfgs_max_iterations=6, patience=3)


def test_condition_split_is_fixed_whole_group_and_chain_order_invariant():
    ids = np.repeat(np.arange(34), 4)
    split = experiment.split_conditions(ids)
    shuffled = np.random.default_rng(20261030).permutation(np.arange(34))
    np.testing.assert_array_equal(split["validation_cells"], np.sort(shuffled[:8]))
    assert len(split["fit_cells"]) == 26
    assert split["fit_mask"].sum() == 104 and split["validation_mask"].sum() == 32
    assert not set(split["fit_cells"]) & set(split["validation_cells"])
    reverse = experiment.split_conditions(ids[::-1])
    np.testing.assert_array_equal(reverse["validation_cells"], split["validation_cells"])
    with pytest.raises(ValueError, match="condition groups"):
        experiment.split_conditions(ids[:-1])


def test_lazy_loader_excludes_TEST_labels_decoder_and_old_R(monkeypatch):
    access = []
    ids = np.repeat(np.arange(34), 4)
    values = dict(train_cell_ids=ids,
        fit_direct=np.broadcast_to(ids[None, :, None, None], (4, 136, 3, 2)).copy(),
        eval_train_direct=np.broadcast_to(1000+ids[None, :, None, None], (4, 136, 3, 2)).copy())
    class LazyArchive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __getitem__(self, name):
            access.append(name)
            assert name in values, "Forbidden member decompression before endpoint selection"
            return values[name]
    monkeypatch.setattr(experiment.np, "load", lambda *args, **kwargs: LazyArchive())
    fit, val, split, actual_ids = experiment.load_r_fit_and_validation("unused")
    assert access == ["train_cell_ids", "fit_direct", "eval_train_direct"]
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(fit[0, :, 0, 0].numpy(), ids[split["fit_mask"]])
    np.testing.assert_array_equal(val[0, :, 0, 0].numpy(), 1000+ids[split["validation_mask"]])
    assert fit.shape == (4, 104, 3, 2) and val.shape == (4, 32, 3, 2)


def test_fresh_initialization_and_normalizer_use_only_fit_input_scales():
    fit, val = tiny_encodings()
    model = experiment.initialize_model(fit, 91, hidden=5)
    same = experiment.initialize_model(fit, 91, hidden=5)
    other = experiment.initialize_model(fit, 92, hidden=5)
    assert state_fingerprint(model) == state_fingerprint(same) != state_fingerprint(other)
    center, scale = fit_input_normalizer(fit)
    assert torch.equal(model.center, center) and torch.equal(model.scale, scale)
    fit[..., 2, :] += 1000
    target_shifted = experiment.initialize_model(fit, 91, hidden=5)
    assert state_fingerprint(target_shifted) == state_fingerprint(model)
    val += 1e9
    assert state_fingerprint(experiment.initialize_model(fit, 91, hidden=5)) == state_fingerprint(model)


def test_validation_averages_residuals_after_nonlinear_prediction():
    class Square:
        def advance(self, g):
            return g.square()
    g = torch.zeros((2, 3, 3, 2), dtype=torch.float64)
    g[0, :, 0] = -1
    g[1, :, 0] = 1
    # The two stochastic repeats predict 1, while R(mean g0)=0.
    g[..., 1, :] = 1
    g[..., 2, :] = 1
    metrics = experiment.validation_metrics(Square(), g)
    assert metrics["chain_mean_loss"] == metrics["repeat_mean_loss"] == 0
    assert metrics["physical_chains"] == 3 and metrics["encoding_repeats"] == 2
    g[0, :, 1, :] = 0
    g[1, :, 1, :] = 2
    metrics = experiment.validation_metrics(Square(), g)
    assert metrics["chain_mean_terms"]["e01"] == 0
    assert metrics["repeat_terms"]["e01"] == [2., 2.]
    assert metrics["repeat_mean_loss"] > metrics["chain_mean_loss"]


def test_patience_has_separate_significant_reference_and_strict_best():
    tracker = experiment.ImprovementTracker(1., patience=3, relative_improvement=.001)
    first = tracker.update(.9997)
    assert first["strict_improvement"] and not first["significant_improvement"]
    assert first["bad_blocks"] == 1 and tracker.best == .9997 and tracker.reference == 1.
    second = tracker.update(.9989)
    assert second["significant_improvement"] and second["bad_blocks"] == 0
    assert not tracker.update(1.1)["should_stop"]
    assert not tracker.update(1.2)["should_stop"]
    assert tracker.update(1.3)["should_stop"]


def test_selection_restores_best_adam_before_fresh_persistent_lbfgs():
    fit, val = tiny_encodings()
    checkpoints = []
    def save(stage, step, model, row):
        checkpoints.append((stage, step, state_fingerprint(model)))
    models, record = experiment.train_seed(fit, val, 91, hidden=5,
        schedule=tiny_schedule(), on_checkpoint=save)
    assert record["adam_updates"] == record["adam_parameter_clock_min"] == record["adam_parameter_clock_max"] == 6
    assert record["gradient_calls"]["adam"] == 6
    assert record["lbfgs_start_matches_adam_best"]
    assert record["lbfgs_start_R_sha256"] == state_fingerprint(models["adam_best"])
    assert record["lbfgs_optimizer_history_retained_across_blocks"]
    adam_rows, lbfgs_rows = record["history"]["adam"], record["history"]["lbfgs"]
    assert [row["step"] for row in adam_rows] == [0, 2, 4, 6]
    losses = [row["validation"]["chain_mean_loss"] for row in adam_rows+lbfgs_rows]
    assert record["selected_validation_loss"] == min(losses)
    selected_rows = [row for row in adam_rows+lbfgs_rows
                     if row["R_sha256"] == state_fingerprint(models["selected"])]
    assert selected_rows
    assert selected_rows[0]["validation"]["chain_mean_loss"] == min(losses)
    assert lbfgs_rows[-1]["optimizer_iterations"] == record["lbfgs_optimizer_iterations"] <= 6
    calls = [row["gradient_calls"]["lbfgs"] for row in lbfgs_rows]
    assert calls == sorted(calls)
    assert checkpoints[0][0:2] == ("adam", 0)
    assert all(p.grad is None for model in models.values() for p in model.parameters())
    assert all(torch.equal(model.center, models["initial"].center) for model in models.values())


def test_validation_changes_do_not_change_adam_fit_gradients_or_final_state():
    fit, val = tiny_encodings()
    untouched_fit, untouched_val = fit.clone(), val.clone()
    first, _ = experiment.train_seed(fit, val, 92, hidden=5, schedule=tiny_schedule())
    second, _ = experiment.train_seed(fit, val+10., 92, hidden=5, schedule=tiny_schedule())
    assert state_fingerprint(first["adam6000"]) == state_fingerprint(second["adam6000"])
    assert torch.equal(fit, untouched_fit) and torch.equal(val, untouched_val)


def test_validation_does_not_mutate_state_or_allocate_gradients():
    fit, val = tiny_encodings()
    model = experiment.initialize_model(fit, 93, hidden=5)
    before = state_fingerprint(model)
    experiment.validation_metrics(model, val)
    assert before == state_fingerprint(model)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="frozen"):
        experiment.validation_metrics(model, val.requires_grad_())


def test_budget_interruption_keeps_inputs_and_does_not_return_partial_endpoint():
    fit, val = tiny_encodings()
    before_fit, before_val = fit.clone(), val.clone()
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise experiment.BudgetHold("synthetic limit")
    with pytest.raises(experiment.BudgetHold, match="synthetic limit"):
        experiment.train_seed(fit, val, 91, hidden=5, schedule=tiny_schedule(), guard=guard)
    assert torch.equal(fit, before_fit) and torch.equal(val, before_val)
