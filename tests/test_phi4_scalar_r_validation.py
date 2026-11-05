import json

import numpy as np
import pytest
import torch

from experiments import phi4_scalar_r_validation as scalar
from experiments.phi4_qcr import sha256_file, state_fingerprint
from experiments.phi4_ru_r_validation import Schedule


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
    return manifest


def test_split_is_condition_grouped_fixed_and_complete():
    ids = np.repeat([1, 2, 4, 5, 7, 8, 10, 11], 4)
    a, b = scalar.split_conditions(ids), scalar.split_conditions(ids)
    assert np.array_equal(a["validation_cells"], b["validation_cells"])
    assert a["fit_mask"].sum() == 24 and a["validation_mask"].sum() == 8
    assert len(a["fit_cells"]) == 6 and len(a["validation_cells"]) == 2
    assert not set(a["fit_cells"]) & set(a["validation_cells"])
    with pytest.raises(ValueError):
        scalar.split_conditions(ids[:-1])
    ids[0] = 99
    with pytest.raises(ValueError):
        scalar.split_conditions(ids)


def test_training_loader_never_decompresses_test_or_labels(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    make_cache(cache)
    original = np.load
    reads = []

    class TrackedArchive:
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

    monkeypatch.setattr(scalar.np, "load", lambda *a, **kw: TrackedArchive(original(*a, **kw)))
    fit, validation, split, _ = scalar.load_r_fit_and_validation(cache)
    assert reads == ["train_cell_ids", "fit_direct", "eval_train_direct"]
    assert fit.shape == (4, 24, 3, 1) and validation.shape == (4, 8, 3, 1)
    model = scalar.initialize_model(fit, 91)
    inputs = fit[..., :2, :].reshape(-1, 1)
    assert torch.equal(model.center, inputs.mean(0))
    assert torch.equal(model.scale, inputs.std(0, unbiased=False))
    assert len(split["validation_cells"]) == 2


def test_validation_averages_paired_residual_before_square():
    model = scalar.FrozenResidualMap(1, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    direct = torch.zeros((4, 2, 3, 1), dtype=torch.float64)
    direct[:, :, 1, 0] = torch.tensor([-1., 1., -1., 1.])[:, None]
    direct[:, :, 2, 0] = direct[:, :, 1, 0]
    before = state_fingerprint(model)
    scores = scalar.validation_metrics(model, direct)
    assert scores["chain_mean_loss"] == 0
    assert scores["repeat_mean_loss"] > 0
    assert state_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_endpoint_metrics_use_fixed_target_std_and_both_error_orders():
    direct = np.zeros((4, 2, 3, 1), dtype=np.float64)
    one = np.ones((4, 2, 2, 1), dtype=np.float64)
    one[[1, 3]] *= -1
    rollout = np.zeros_like(direct)
    rollout[..., 2, :] = one[..., 0, :]
    metrics = scalar.endpoint_metrics(direct, dict(one_step=one, rollout=rollout), [2., 4., 5.])
    assert [row["train_target_std"] for row in metrics] == [4., 5., 5.]
    assert [row["raw_repeat_rmse"] for row in metrics] == [1., 1., 1.]
    assert [row["chain_mean_rmse"] for row in metrics] == [0., 0., 0.]
    assert metrics[0]["raw_repeat_nrmse"] == .25
    json.dumps(metrics, allow_nan=False)


def test_train_seed_schedule_selection_and_normalizer_are_fixed():
    torch.set_num_threads(1)
    fit = torch.tensor(repeated_data(6))
    validation = torch.tensor(repeated_data(2))
    tiny = Schedule(adam_steps=4, checkpoint_every=2, lbfgs_block_iterations=2,
        lbfgs_max_iterations=4, patience=3, relative_improvement=.001)
    checkpoints = []
    models, record = scalar.train_seed(fit, validation, 91, hidden=4, schedule=tiny,
        on_checkpoint=lambda stage, step, model, row: checkpoints.append((stage, step)))
    assert record["adam_updates"] == record["adam_parameter_clock_min"] == 4
    assert record["adam_parameter_clock_max"] == 4
    assert record["no_validation_gradient"] and record["normalizer_unchanged"]
    assert record["lbfgs_start_matches_adam_best"]
    assert ("adam", 0) in checkpoints and ("adam", 4) in checkpoints
    assert set(models) == {"initial", "adam6000", "adam_best", "lbfgs_final", "lbfgs_best", "selected"}
    selected = scalar.validation_metrics(models["selected"], validation)["chain_mean_loss"]
    assert selected == pytest.approx(record["selected_validation_loss"])
    assert selected <= record["adam_best_validation_loss"]
    for model in models.values():
        assert model.theory_dim == 1
        assert torch.equal(model.center, models["initial"].center)
        assert torch.equal(model.scale, models["initial"].scale)
        assert all(parameter.grad is None for parameter in model.parameters())


def test_verify_cache_pins_without_array_decompression_and_rejects_tamper(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    expected = make_cache(cache)
    monkeypatch.setattr(scalar.np, "load", lambda *a, **kw: pytest.fail("container verification decompressed NPZ"))
    manifest, hashes = scalar.verify_cache(cache)
    assert manifest == expected and len(hashes) == 4
    (cache / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        scalar.verify_cache(cache)


def test_evaluation_keeps_original_train_and_test_groups(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    _, _, split, ids = scalar.load_r_fit_and_validation(cache)
    saved, source = scalar.load_evaluation_after_selection(cache, split, ids)
    assert not source
    assert saved["train_direct"].shape == (4, 32, 3, 1)
    assert saved["fit_direct"].shape == (4, 24, 3, 1)
    assert saved["validation_direct"].shape == (4, 8, 3, 1)
    assert saved["test_direct"].shape == (4, 20, 3, 1)
    assert not set(saved["train_cell_ids"]) & set(saved["test_cell_ids"])


def test_run_selects_every_seed_before_test_read_and_writes_public_endpoints(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "run"
    make_cache(cache)
    selected = []
    original_loader = scalar.load_evaluation_after_selection

    def train(fit, validation, seed, **kwargs):
        selected.append(seed)
        model = scalar.initialize_model(fit, seed, hidden=4).eval()
        for parameter in model.parameters():
            parameter.grad = None
        return {"selected": model, "adam6000": deepcopy_model(model)}, dict(
            selected_stage="adam", selected_step=6000, selected_validation_loss=.1,
            lbfgs_stopping="validation_patience", cpu_seconds=0)

    def evaluation(*args, **kwargs):
        assert selected == [91, 92, 93]
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(scalar, "train_seed", train)
    monkeypatch.setattr(scalar, "load_evaluation_after_selection", evaluation)
    monkeypatch.setattr(scalar.torch, "set_num_interop_threads", lambda _: None)
    result = scalar.run(scalar.Options(str(cache), str(output)))
    assert result["status"] == "complete" and result["all_seeds_selected_before_TEST_and_labels_loaded"]
    assert len(result["final_R_sha256"]) == 6
    assert result["Q_updates"] == result["C_updates"] == result["GPU_seconds"] == 0
    status = json.loads((output / "status.json").read_text())
    assert status["returncode"] == 0 and status["workflow_complete"]
    inventory = json.loads((output / "public_files.json").read_text())
    assert {"R_parameters.npz", "evaluation_arrays.npz", "summary.json", "status.json"} <= set(inventory)
    assert all(sha256_file(output / name) == digest for name, digest in inventory.items())
    assert not list(output.glob("*.pt"))
    with np.load(output / "R_parameters.npz", allow_pickle=False) as archive:
        assert "selected_seed91.center" in archive.files


def deepcopy_model(model):
    from copy import deepcopy
    return deepcopy(model)


@pytest.mark.parametrize("budget", [0., -1., 1200.001, float("inf"), float("nan")])
def test_run_rejects_invalid_or_expanded_cpu_budget_without_output(tmp_path, budget):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="1200"):
        scalar.run(scalar.Options(str(tmp_path / "cache"), str(output), budget))
    assert not output.exists()
