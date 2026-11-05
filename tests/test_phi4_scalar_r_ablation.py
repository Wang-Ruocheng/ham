from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiments import phi4_scalar_r_ablation as ablation
from experiments.phi4_qcr import sha256_file, state_fingerprint
from experiments.phi4_ru_r_validation import Schedule
from experiments.phi4_scalar_r_validation import split_conditions
from recurrent_rg.phi4_r_calibration import all_start_closure_loss


def repeated_data(chains=32):
    rng = np.random.default_rng(19)
    first = rng.normal(0, .04, (4, chains, 1))
    second = .8*first+.006
    third = .8*second+.006
    return np.stack((first, second, third), axis=-2).astype(np.float64)


def make_cache(root):
    root.mkdir()
    train, test = repeated_data(), repeated_data(20)
    train_ids = np.repeat([1, 2, 4, 5, 7, 8, 10, 11], 4)
    test_ids = np.repeat([0, 3, 6, 9, 12], 4)
    np.savez(root / "cache.npz", fit_direct=train, eval_train_direct=train+.0001,
        eval_test_direct=test, train_cell_ids=train_ids, test_cell_ids=test_ids,
        train_labels=np.column_stack((train_ids, np.ones(32), np.zeros(32))),
        test_labels=np.column_stack((test_ids, np.ones(20), np.zeros(20))))
    manifest = dict(target_stds=[.04, .032, .026], source_state_sha256="a"*64,
        source_module_sha256={key: letter*64 for key, letter in zip(("Q", "C", "R"), "bcd")})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "status.json").write_text(json.dumps(dict(status="complete",
        workflow_complete=True, returncode=0, error=None)))
    (root / "public_files.json").write_text(json.dumps({name: sha256_file(root / name)
        for name in ("manifest.json", "cache.npz")}))


def test_four_condition_folds_cover_once_and_match_original_fold_zero():
    ids = np.repeat([1, 2, 4, 5, 7, 8, 10, 11], 4)
    folds = ablation.condition_folds(ids)
    assert len(folds) == 4
    assert np.array_equal(folds[0]["validation_cells"], split_conditions(ids)["validation_cells"])
    assert np.array_equal(sum((s["validation_mask"].astype(int) for s in folds)), np.ones(32))
    for i, split in enumerate(folds):
        assert split["fold"] == i
        assert split["fit_mask"].sum() == 24 and split["validation_mask"].sum() == 8
        assert not set(split["fit_cells"]) & set(split["validation_cells"])
        assert set(split["fit_cells"]) | set(split["validation_cells"]) == set(ids)
    with pytest.raises(ValueError):
        ablation.condition_folds(ids[:-1])


def test_preselection_loader_never_decompresses_test_or_labels(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    make_cache(cache)
    original, reads = np.load, []

    class Tracked:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            reads.append(key)
            assert key in ("train_cell_ids", "fit_direct", "eval_train_direct")
            return self.archive[key]

    monkeypatch.setattr(ablation.np, "load", lambda *a, **kw: Tracked(original(*a, **kw)))
    fit, evaluation, folds, ids = ablation.load_train_only(cache)
    assert reads == ["train_cell_ids", "fit_direct", "eval_train_direct"]
    assert fit.shape == evaluation.shape == (4, 32, 3, 1)
    assert len(folds) == 4 and len(ids) == 32


def test_all_arms_and_heads_share_fit_only_input_normalizer():
    fit = torch.tensor(repeated_data(6))
    # Only target scale 2 is modified: neither center nor scale may move.
    changed = fit.clone()
    changed[..., 2, :] += 1000
    inputs = fit[..., :2, :].reshape(-1, 1)
    for spec in ablation.ARMS.values():
        a = ablation.initialize_model(fit, 91, hidden=spec["hidden"], kind=spec["kind"])
        b = ablation.initialize_model(changed, 91, hidden=spec["hidden"], kind=spec["kind"])
        for head, other in zip(a.heads, b.heads):
            assert torch.equal(head.center, inputs.mean(0))
            assert torch.equal(head.scale, inputs.std(0, unbiased=False))
            assert torch.equal(head.center, other.center) and torch.equal(head.scale, other.scale)
        assert len(a.heads) == spec["heads"]
    shared136 = ablation.initialize_model(fit, 91, hidden=136)
    separate96 = ablation.initialize_model(fit, 91, hidden=96, kind="separate")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert abs(count(shared136)/count(separate96)-1) < .02


def test_shared_head_exactly_recovers_existing_all_start_objective():
    fit = torch.tensor(repeated_data(6))
    model = ablation.initialize_model(fit, 91, hidden=4)
    new_loss, new_terms = ablation.closure_loss(model, fit)
    old_loss, old_terms = all_start_closure_loss(model.heads[0], fit)
    assert torch.equal(new_loss, old_loss)
    assert all(torch.equal(new_terms[k], old_terms[k]) for k in old_terms)
    with pytest.raises(ValueError, match="frozen"):
        ablation.closure_loss(model, fit.clone().requires_grad_())


def test_two_head_rollout_retains_both_gradient_paths_and_order():
    fit = torch.tensor(repeated_data(6))
    model = ablation.initialize_model(fit, 91, hidden=4, kind="separate")
    _, terms = ablation.closure_loss(model, fit)
    grads = torch.autograd.grad(terms["e02"], tuple(model.parameters()))
    split = len(tuple(model.heads[0].parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads[:split]) > 0
    assert sum(float(g.abs().sum()) for g in grads[split:]) > 0
    prediction = ablation.predict(model, fit.numpy())
    with torch.no_grad():
        expected = model.step(model.step(fit[..., 0, :], 0), 1).numpy()
    assert np.array_equal(prediction["rollout"][..., 2, :], expected)
    for invalid in (-1, 2, True, .5):
        with pytest.raises(ValueError):
            model.step(fit[..., 0, :], invalid)


def test_validation_averages_paired_residual_before_square():
    model = ablation.ScalarClosureMaps(4, "separate")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    direct = torch.zeros((4, 2, 3, 1), dtype=torch.float64)
    direct[:, :, 1, 0] = torch.tensor([-1., 1., -1., 1.])[:, None]
    direct[:, :, 2, 0] = direct[:, :, 1, 0]
    before = state_fingerprint(model)
    scores = ablation.validation_metrics(model, direct)
    assert scores["chain_mean_loss"] == 0 and scores["repeat_mean_loss"] > 0
    assert before == state_fingerprint(model)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("kind", ["shared", "separate"])
def test_tiny_schedule_parameter_clocks_selection_and_frozen_units(kind):
    torch.set_num_threads(1)
    fit, validation = torch.tensor(repeated_data(6)), torch.tensor(repeated_data(2))
    tiny = Schedule(adam_steps=4, checkpoint_every=2, lbfgs_block_iterations=2,
        lbfgs_max_iterations=4, patience=3, relative_improvement=.001)
    checkpoints = []
    models, record = ablation.train_seed(fit, validation, 91, hidden=4, kind=kind, schedule=tiny,
        on_checkpoint=lambda stage, step, model, row: checkpoints.append((stage, step)))
    assert record["adam_updates"] == record["adam_parameter_clock_min"] == record["adam_parameter_clock_max"] == 4
    assert record["adam_parameter_count"] == len(list(models["selected"].parameters()))
    assert record["no_validation_gradient"] and record["normalizer_unchanged"]
    assert record["lbfgs_start_matches_adam_best"]
    assert ("adam", 0) in checkpoints and ("adam", 4) in checkpoints
    assert set(models) == {"selected", "adam6000"}
    score = ablation.validation_metrics(models["selected"], validation)["chain_mean_loss"]
    assert score == pytest.approx(record["selected_validation_loss"])
    for model in models.values():
        assert all(p.grad is None for p in model.parameters())
        for head in model.heads:
            assert torch.equal(head.center, models["selected"].heads[0].center)
            assert torch.equal(head.scale, models["selected"].heads[0].scale)


def test_budget_guard_interrupts_without_continuing_optimization():
    fit = torch.tensor(repeated_data(2))
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise ablation.BudgetHold("test budget")
    with pytest.raises(ablation.BudgetHold):
        ablation.train_seed(fit, fit, 91, hidden=4, guard=guard)
    assert len(calls) == 2


def test_run_selects_all_48_before_test_and_writes_frozen_public_endpoints(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "run"
    make_cache(cache)
    selected = []
    loader = ablation.load_evaluation_after_selection

    def train(fit, validation, seed, **kwargs):
        selected.append((kwargs["kind"], kwargs["hidden"], seed))
        model = ablation.initialize_model(fit, seed, hidden=2, kind=kwargs["kind"]).eval()
        for p in model.parameters():
            p.grad = None
        return {"selected": model, "adam6000": deepcopy(model)}, dict(
            selected_stage="adam", selected_step=6000, selected_validation_loss=.1,
            selected_R_sha256=state_fingerprint(model), lbfgs_stopping="validation_patience", cpu_seconds=0)

    def evaluation(*args, **kwargs):
        assert len(selected) == 48
        return loader(*args, **kwargs)

    monkeypatch.setattr(ablation, "train_seed", train)
    monkeypatch.setattr(ablation, "load_evaluation_after_selection", evaluation)
    monkeypatch.setattr(ablation.torch, "set_num_interop_threads", lambda _: None)
    result = ablation.run(ablation.Options(str(cache), str(output)))
    assert result["status"] == "complete" and result["completed_runs"] == 48
    assert result["all_runs_selected_before_TEST_and_labels_loaded"]
    assert len(result["final_R_sha256"]) == 96
    assert result["Q_updates"] == result["C_updates"] == result["GPU_seconds"] == 0
    status = json.loads((output / "status.json").read_text())
    assert status["returncode"] == 0 and status["workflow_complete"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["splits"]) == 4 and len(manifest["normalizers"]) == 4
    assert manifest["arms"] == ablation.ARMS
    inventory = json.loads((output / "public_files.json").read_text())
    assert {"R_parameters.npz", "evaluation_arrays.npz", "summary.json", "status.json"} <= set(inventory)
    assert all(sha256_file(output / name) == digest for name, digest in inventory.items())
    assert not list(output.glob("*.pt")) and not list(output.glob("checkpoint*"))
    with np.load(output / "R_parameters.npz", allow_pickle=False) as archive:
        assert "fold0_A_shared96_seed91_selected.heads.0.center" in archive.files
        assert "fold0_B_separate96_seed91_selected.heads.1.center" in archive.files
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as archive:
        assert "eval_fold3_D_shared16_seed93_selected_test_one_step" in archive.files
        assert "fit_direct" not in archive.files
    with pytest.raises(FileExistsError):
        ablation.run(ablation.Options(str(cache), str(output)))


@pytest.mark.parametrize("budget", [0., -1., 7200.001, float("inf"), float("nan")])
def test_invalid_budget_fails_before_output(tmp_path, budget):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="7200"):
        ablation.run(ablation.Options(str(tmp_path / "cache"), str(output), budget))
    assert not output.exists()
