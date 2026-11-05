"""Optimizer contracts on tiny synthetic frozen-g problems only."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_ru_r_optimizer_probe as probe
from experiments.phi4_qcr import state_fingerprint
from recurrent_rg.phi4_r_calibration import FrozenResidualMap, fit_input_normalizer


def fixture():
    torch.manual_seed(318)
    x = torch.randn(2, 8, 2, dtype=torch.float64) * .08
    g1 = .85 * x + torch.tensor([.03, -.01])
    direct = torch.stack((x, g1, .85*g1 + torch.tensor([.03, -.01])), dim=-2)
    center, scale = fit_input_normalizer(direct)
    return FrozenResidualMap(2, 5, center=center, scale=scale), direct


def test_paired_starts_exact_and_noopt_is_immutable():
    source, direct = fixture()
    fingerprint = state_fingerprint(source)
    noopt, baseline = probe.optimizer_arm(source, direct, "noopt")
    adam, result = probe.optimizer_arm(source, direct, "adam", adam_steps=3)
    assert baseline["initial_R_sha256"] == result["initial_R_sha256"] == fingerprint
    assert baseline["final_R_sha256"] == fingerprint == state_fingerprint(source)
    assert baseline["gradient_calls"] == baseline["optimizer_iterations"] == 0
    assert result["gradient_calls"] == result["optimizer_iterations"] == 3
    assert state_fingerprint(adam) != fingerprint
    for name in ("center", "scale"):
        assert torch.equal(adam.state_dict()[name], noopt.state_dict()[name])
    assert all(p.grad is None for p in adam.parameters())


def test_lbfgs_actual_evaluations_and_improvement_on_synthetic_closure():
    source, direct = fixture()
    _, initial = probe.optimizer_arm(source, direct, "noopt")
    endpoint, result = probe.optimizer_arm(source, direct, "lbfgs", lbfgs_max_iter=8,
                                           lbfgs_max_eval=20)
    assert 0 < result["optimizer_iterations"] <= 8
    assert result["gradient_calls"] >= result["optimizer_iterations"]
    assert result["objective_forward_evaluations"] == result["gradient_calls"] + 2
    assert result["endpoint_train_loss"] < initial["endpoint_train_loss"]
    assert result["max_eval_is_soft_under_strong_wolfe"]
    assert all(p.grad is None for p in endpoint.parameters())


def test_budget_interrupt_does_not_modify_source_or_return_fake_endpoint():
    source, direct = fixture()
    state = state_fingerprint(source)
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise probe.BudgetHold("synthetic budget")
    with pytest.raises(probe.BudgetHold, match="synthetic budget"):
        probe.optimizer_arm(source, direct, "lbfgs", guard=guard,
                            lbfgs_max_iter=8, lbfgs_max_eval=20)
    assert state_fingerprint(source) == state
    assert all(p.grad is None for p in source.parameters())


def test_fit_loader_does_not_access_labels_or_eval_members(monkeypatch):
    accesses = []
    class LazyArchive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __getitem__(self, name):
            accesses.append(name)
            if name != "fit_direct":
                raise AssertionError("Evaluation or label access before optimization")
            return np.zeros((4, 136, 3, 2))
    monkeypatch.setattr(probe.np, "load", lambda *args, **kwargs: LazyArchive())
    direct = probe.load_fit_only("unused")
    assert accesses == ["fit_direct"]
    assert direct.shape == (4, 136, 3, 2)
    assert not direct.requires_grad


def test_endpoint_restore_checks_sha_and_preserves_fixed_normalizer(tmp_path):
    torch.manual_seed(1)
    fit = torch.randn(4, 136, 3, 2, dtype=torch.float64)
    center, scale = fit_input_normalizer(fit)
    models = {seed: FrozenResidualMap(2, 96, center=center, scale=scale) for seed in probe.SEEDS}
    arrays = {f"scaled96_seed{seed}.{name}": value.numpy()
              for seed, model in models.items() for name, value in model.state_dict().items()}
    np.savez(tmp_path / "R_parameters.npz", **arrays)
    summary = dict(actual_steps_by_model={f"scaled96_seed{s}": 6000 for s in probe.SEEDS},
        final_R_sha256={f"scaled96_seed{s}": state_fingerprint(m) for s, m in models.items()})
    restored = probe.restore_endpoints(tmp_path, summary, fit)
    for seed in probe.SEEDS:
        assert state_fingerprint(restored[seed]) == state_fingerprint(models[seed])
        assert all(p.requires_grad for p in restored[seed].parameters())
    corrupted = deepcopy(summary)
    corrupted["final_R_sha256"]["scaled96_seed92"] = "wrong"
    with pytest.raises(ValueError, match="endpoint SHA"):
        probe.restore_endpoints(tmp_path, corrupted, fit)


def test_direction_metrics_average_repeats_then_chains_and_keep_condition_axes():
    # Repetition noise cancels; six physical chains grouped in two conditions.
    rng = np.random.default_rng(119)
    direct = rng.normal(size=(4, 6, 3, 2))
    empirical = direct[..., 1:, :] - direct[..., :2, :]
    prediction = direct[..., :2, :] - empirical
    cells = np.array([1, 2, 1, 2, 1, 2])
    rows = probe.direction_metrics(direct, {"one_step": prediction}, cells)
    for row in rows:
        assert row["physical_chains"]["opposite_fraction"] == 1.
        assert row["parameter_conditions"]["opposite_fraction"] == 1.
        assert row["physical_chains"]["vector_count"] == 6
        assert row["parameter_conditions"]["vector_count"] == 2
        assert row["cell_opposite"] == [True, True]
        assert row["encoding_repeats_averaged_first"]
