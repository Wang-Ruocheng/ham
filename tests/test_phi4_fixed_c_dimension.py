import json

import numpy as np
import pytest
import torch

from experiments import phi4_fixed_c_dimension as dimension
from experiments.phi4_qcr import sha256_file, state_fingerprint
from recurrent_rg.phi4_r_calibration import FrozenResidualMap, fit_input_normalizer


def features(chains=32, seed=17):
    rng = np.random.default_rng(seed)
    value = rng.uniform(.01, 1., (4, chains, 3, 2, 105))
    return (value/value.sum(-1, keepdims=True)).astype(np.float64)


def make_cache(root):
    root.mkdir()
    fit, evaluation, test = features(), features(seed=19), features(20, 23)
    ids = np.repeat([1, 2, 4, 5, 7, 8, 10, 11], 4)
    test_ids = np.repeat([0, 3, 6, 9, 12], 4)
    np.savez(root / "features.npz", fit_half_features=fit, eval_train_half_features=evaluation,
        eval_test_half_features=test, eval_train_feature_target=evaluation.mean(-2),
        eval_test_feature_target=test.mean(-2), train_cell_ids=ids, test_cell_ids=test_ids,
        train_labels=np.column_stack((ids, np.ones(32), np.zeros(32))),
        test_labels=np.column_stack((test_ids, np.ones(20), np.zeros(20))))
    manifest = dict(recipe="phi4_u1_frozen_C_paired_half_features_v1", lattice_sizes=[36, 12, 4],
        feature_dim=105, train_chains=32, test_chains=20, fit_repeats=4, evaluation_repeats=4,
        source_training_steps=3000, source_module_sha256={"C": "c"*64}, data_warning_count=13)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False)))
    (root / "receipt.json").write_text(json.dumps(dict(source_unchanged=True,
        features_sha256=sha256_file(root / "features.npz"))))
    inventory = {name: sha256_file(root / name) for name in ("manifest.json", "status.json", "receipt.json", "features.npz")}
    (root / "public_files.json").write_text(json.dumps(inventory))
    return manifest


def test_container_pins_reject_tamper_without_loading_arrays(tmp_path, monkeypatch):
    cache = tmp_path / "features"
    expected = make_cache(cache)
    monkeypatch.setattr(dimension.np, "load", lambda *a, **kw: pytest.fail("verification decompressed NPZ"))
    actual, pins = dimension.verify_features(cache)
    assert actual == expected and len(pins) == 5
    (cache / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        dimension.verify_features(cache)


def test_train_loader_reads_no_labels_test_or_source_q_encodings(tmp_path, monkeypatch):
    cache = tmp_path / "features"
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
            assert key in ("train_cell_ids", "fit_half_features", "eval_train_half_features")
            return self.archive[key]

    monkeypatch.setattr(dimension.np, "load", lambda *a, **kw: Tracked(original(*a, **kw)))
    fit, evaluation, ids, split = dimension.load_train_only(cache)
    assert reads == ["train_cell_ids", "fit_half_features", "eval_train_half_features"]
    assert fit.shape == evaluation.shape == (4, 32, 3, 2, 105)
    assert len(ids) == 32 and split["fit_mask"].sum() == 24 and split["validation_mask"].sum() == 8


def test_paired_q_initialization_units_and_private_rng():
    fit = torch.tensor(features())
    rng = torch.get_rng_state().clone()
    one, two = dimension.initialize_paired_q(fit, 91).values()
    assert torch.equal(rng, torch.get_rng_state())
    for index in (0, 2, 4):
        assert torch.equal(one.net[index].weight, two.net[index].weight)
        assert torch.equal(one.net[index].bias, two.net[index].bias)
    assert torch.equal(one.net[6].weight, two.net[6].weight[:1])
    assert torch.equal(one.net[6].bias, two.net[6].bias[:1])
    assert torch.allclose(one.decoder.weight[:, 0], two.decoder.weight[:, 0], atol=1e-15, rtol=0.)
    assert torch.equal(one.decoder.bias, fit.mean((0, 1, 2, 3)))
    assert torch.equal(one.decoder.bias, two.decoder.bias)
    assert torch.allclose(one.from_features(fit), two.from_features(fit)[..., :1], atol=1e-15, rtol=0.)
    for model in (one, two):
        assert torch.allclose(model.decoder.weight.norm(dim=0), torch.ones(model.decoder.in_features, dtype=torch.float64))


def test_q_objective_exact_current_weights_and_no_input_gradients():
    fit = torch.tensor(features())[:1, :3]
    q = dimension.initialize_paired_q(fit, 91)[2]
    loss, terms = dimension.q_objective(q, fit)
    encoded = q.from_features(fit)
    direct, target = encoded.mean(-2), fit.mean(-2)
    recon = (q.decode(direct)-target).square().sum(-1).mean()
    subset = (encoded[..., 0, :]-encoded[..., 1, :]).square().sum(-1)
    expected = 10*recon+20*subset[..., 0].mean()+5*subset[..., 1:].mean()
    assert torch.equal(loss, expected)
    assert torch.equal(terms["reconstruction"], recon)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
    assert fit.grad is None
    with pytest.raises(ValueError, match="frozen"):
        dimension.q_objective(q, fit.clone().requires_grad_())


def test_encoding_is_average_of_two_q_values_not_q_of_average():
    fit = torch.tensor(features())
    q = dimension.initialize_paired_q(fit, 91)[1]
    actual = dimension.encode(q, fit)
    with torch.no_grad():
        expected = .5*(q.from_features(fit[..., 0, :])+q.from_features(fit[..., 1, :]))
        wrong = q.from_features(fit.mean(-2))
    assert torch.equal(actual, expected)
    assert not torch.allclose(actual, wrong, atol=1e-12, rtol=0.)
    assert not actual.requires_grad


def test_tiny_q_training_identical_batches_exact_clocks_and_column_normalization():
    torch.set_num_threads(1)
    fit = torch.tensor(features())
    qs = dimension.initialize_paired_q(fit, 91)
    global_rng = torch.get_rng_state().clone()
    records = {dim: dimension.train_q(q, fit, 91, steps=4, checkpoint_every=2) for dim, q in qs.items()}
    assert torch.equal(global_rng, torch.get_rng_state())
    assert records[1]["batch_stream_sha256"] == records[2]["batch_stream_sha256"]
    for dim, record in records.items():
        assert record["actual_steps"] == record["adam_parameter_clock_min"] == record["adam_parameter_clock_max"] == 4
        assert [row["step"] for row in record["history"]] == [0, 2, 4]
        assert record["C_updates"] == record["R_updates"] == 0
        assert all(not p.requires_grad and p.grad is None for p in qs[dim].parameters())
        assert torch.allclose(qs[dim].decoder.weight.norm(dim=0), torch.ones(dim, dtype=torch.float64), atol=1e-14, rtol=0.)


def test_common_feature_metrics_use_identical_targets_and_two_error_orders():
    fit = torch.tensor(features(2))
    q = dimension.initialize_paired_q(fit, 91)[1].eval().requires_grad_(False)
    direct = dimension.encode(q, fit)
    identity = dict(one_step=np.stack((direct[..., 0, :].numpy(), direct[..., 1, :].numpy()), -2),
                    rollout=np.stack((direct[..., 0, :].numpy(), direct[..., 0, :].numpy(), direct[..., 0, :].numpy()), -2))
    target = fit.mean(-2).numpy()
    stds = np.ones((3, 1))
    feature_rows, latent_rows = dimension.endpoint_metrics(q, direct, identity, target, stds)
    assert len(feature_rows) == len(latent_rows) == 3
    for row in feature_rows:
        assert row["directly_comparable_across_dg"]
        assert row["prediction"] == row["identity_decoded"]
        assert row["feature_dim"] == 105
    assert all(not row["directly_comparable_across_dg"] for row in latent_rows)
    residual = np.ones((4, 2, 105))
    residual[[1, 3]] *= -1
    summary = dimension.residual_metrics(residual)
    assert summary["raw_repeat_rmse"] == 1 and summary["chain_mean_rmse"] == 0


def test_run_finishes_both_q_and_all_six_r_selections_before_test(tmp_path, monkeypatch):
    cache, output = tmp_path / "features", tmp_path / "run"
    make_cache(cache)
    events = []
    original_post = dimension.load_postselection

    def q_fit(q, fit, seed, **kwargs):
        dim = q.decoder.in_features
        events.append(f"Q{dim}")
        q.eval().requires_grad_(False)
        for parameter in q.parameters():
            parameter.grad = None
        return dict(batch_stream_sha256="a"*64, final_Q_sha256=state_fingerprint(q))

    def r_fit(fit, validation, seed, **kwargs):
        assert events[:2] == ["Q1", "Q2"]
        dim = fit.shape[-1]
        events.append(f"R{dim}_{seed}")
        center, scale = fit_input_normalizer(fit)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = FrozenResidualMap(dim, 4, center=center, scale=scale).eval()
        return dict(selected=model, adam6000=model), dict(seed=seed, selected_stage="adam", selected_step=6000)

    def post(*args, **kwargs):
        assert events == ["Q1", "Q2", "R1_91", "R1_92", "R1_93", "R2_91", "R2_92", "R2_93"]
        return original_post(*args, **kwargs)

    monkeypatch.setattr(dimension, "train_q", q_fit)
    monkeypatch.setattr(dimension, "train_scalar_r", r_fit)
    monkeypatch.setattr(dimension, "train_vector_r", r_fit)
    monkeypatch.setattr(dimension, "load_postselection", post)
    monkeypatch.setattr(dimension.torch, "set_num_interop_threads", lambda _: None)
    result = dimension.run(dimension.Options(str(cache), str(output)))
    assert result["status"] == "complete" and result["all_Q_fits_and_R_selections_before_TEST_and_labels"]
    assert result["common_feature_targets_identical"] and result["Q_frozen_during_R"]
    assert len(result["final_R_sha256"]) == 12 and len(result["final_Q_sha256"]) == 2
    assert result["C_updates"] == result["GPU_seconds"] == 0
    assert set(result["latent_TRAIN_coordinate_stds"]) == {"1", "2"}
    status = json.loads((output / "status.json").read_text())
    assert status["returncode"] == 0 and status["workflow_complete"]
    inventory = json.loads((output / "public_files.json").read_text())
    assert {"Q_parameters.npz", "R_parameters.npz", "evaluation_arrays.npz", "summary.json", "status.json"} <= set(inventory)
    assert all(sha256_file(output / name) == sha for name, sha in inventory.items())
    assert not list(output.glob("*.pt"))
    with np.load(output / "Q_parameters.npz", allow_pickle=False) as archive:
        assert "dg2.decoder.weight" in archive.files
    with np.load(output / "R_parameters.npz", allow_pickle=False) as archive:
        assert "dg1_seed91_selected.center" in archive.files
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as archive:
        assert archive["dg2_train_direct"].shape == (4, 32, 3, 2)
        assert archive["train_feature_target"].shape == (4, 32, 3, 105)
    with pytest.raises(FileExistsError):
        dimension.run(dimension.Options(str(cache), str(output)))


@pytest.mark.parametrize("budget", [0., -1., 3600.001, float("inf"), float("nan")])
def test_budget_rejected_before_output(tmp_path, budget):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="3600"):
        dimension.run(dimension.Options(str(tmp_path / "features"), str(output), cpu_budget_seconds=budget))
    assert not output.exists()
