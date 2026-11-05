"""Toy-only checks: never load a real model, cache or physical sample."""
import copy

import numpy as np
import pytest
import torch

from scripts import run_phi4_hopping_macro_R_ablation as runner


def toy_r():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        return torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)).double()


def test_pair_weights_and_source_shares():
    equal, macro = [runner.pair_weights(arm) for arm in runner.ARMS]
    assert equal.sum() == pytest.approx(10)
    assert macro.sum() == pytest.approx(10)
    shares = lambda weights: [sum(w for w, (s, _) in zip(weights, runner.PAIRS) if s == start) / 10 for start in range(3)]
    assert shares(equal) == pytest.approx([11/18, 5/18, 1/9])
    assert shares(macro) == pytest.approx([1/6, 2/6, 3/6])
    assert [sum(w for w, (_, t) in zip(equal, runner.PAIRS) if t == target) for target in (1, 2, 3)] == pytest.approx([10/3]*3)


@pytest.mark.parametrize("arm", runner.ARMS)
def test_objective_matches_explicit_form_and_detaches_anchors(arm):
    model = toy_r()
    direct = torch.linspace(-.5, .5, 24, dtype=torch.float64).reshape(6, 4, 1).requires_grad_()
    prediction = runner.predict_pairs(model, direct)
    assert prediction.shape == (6, 6, 1)
    explicit = sum(float(weight) * (prediction[:, i] - direct[:, target].detach()).square().mean()
        for i, ((_, target), weight) in enumerate(zip(runner.PAIRS, runner.pair_weights(arm))))
    actual = runner.objective(model, direct, arm)
    assert torch.equal(actual, explicit) or torch.allclose(actual, explicit, atol=1e-15, rtol=1e-15)
    actual.backward()
    assert direct.grad is None
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_recurrence_keeps_gradients_through_intermediate_calls():
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(.5)
    direct = torch.ones(1, 4, 1, dtype=torch.float64)
    runner.predict_pairs(model, direct)[:, 3].sum().backward()  # (0,3)
    assert model.weight.grad.item() == pytest.approx(3 * 1.5**2)


def test_updates_paired_and_independent_without_label_arguments():
    model = toy_r()
    models = {arm: copy.deepcopy(model) for arm in runner.ARMS}
    optimizers = {arm: torch.optim.Adam(models[arm].parameters(), lr=.001) for arm in runner.ARMS}
    assert next(models[runner.ARMS[0]].parameters()).data_ptr() != next(models[runner.ARMS[1]].parameters()).data_ptr()
    anchors = torch.linspace(-.3, .6, 160, dtype=torch.float64).reshape(40, 4, 1)
    seed = torch.Generator().manual_seed(91)
    expected = torch.randint(40, (32,), generator=torch.Generator().manual_seed(91))
    actual = runner.train_update(models, optimizers, anchors, seed)
    assert torch.equal(actual, expected)
    assert all(runner.optimizer_counts(optimizers[arm], models[arm])["min"] == 1 for arm in runner.ARMS)
    assert runner.state_sha(models[runner.ARMS[0]].state_dict()) != runner.state_sha(models[runner.ARMS[1]].state_dict())


def test_make_r_preserves_global_rng_and_matches_state():
    with torch.random.fork_rng(devices=[]):
        initial = torch.nn.Sequential(torch.nn.Linear(1, 96), torch.nn.Tanh(), torch.nn.Linear(96, 96),
            torch.nn.Tanh(), torch.nn.Linear(96, 1)).double()
        before = torch.get_rng_state().clone()
        loaded = runner.make_r(initial.state_dict())
        assert torch.equal(before, torch.get_rng_state())
        assert runner.state_sha(initial.state_dict()) == runner.state_sha(loaded.state_dict())


def test_metrics_population_target_std_and_identity():
    direct = np.arange(24, dtype=np.float64).reshape(6, 4, 1)
    prediction = direct[:, [start for start, _ in runner.PAIRS]]
    std = direct.std(axis=0, ddof=0)[:, 0]
    means = direct.mean(axis=0)[:, 0]
    result = runner.calculate_metrics(direct, prediction, std, means)
    for row in result["pairs"]:
        assert row["rmse"] == row["identity_rmse"]
        assert row["nrmse"] == row["identity_nrmse"]
        assert row["constant_train_mean_nrmse"] == pytest.approx(1)
        assert row["constant_train_mean_rmse"] == pytest.approx(std[row["target"]])
    for arm, objective_name in zip(runner.ARMS, ("equal_target_objective", "macro_source_objective")):
        assert sum(result["source_group_contributions"][arm].values()) == pytest.approx(result[objective_name])


def test_source_cache_split_geometry_rejects_leakage():
    direct = np.linspace(-1, 1, 240).reshape(60, 4, 1)
    cells = np.repeat(np.arange(15), 4)
    splits = np.where(cells % 3 == 1, "test", "train")
    arrays = dict(direct_g=direct, direct_full_g=direct.copy(), cell_ids=cells, splits=splits,
        labels=np.repeat(np.arange(45).reshape(15, 3), 4, axis=0), all_start_pairs=np.asarray(runner.PAIRS))
    runner.validate_cache(arrays)
    arrays["splits"] = splits.copy()
    arrays["splits"][[0, 4]] = arrays["splits"][[4, 0]]
    with pytest.raises(ValueError, match="split leaks"):
        runner.validate_cache(arrays)
