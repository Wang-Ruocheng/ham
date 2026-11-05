"""Synthetic-only contracts; never read actual physical fields or checkpoints."""
import json

import numpy as np
import pytest
import torch

from experiments import phi4_hopping_r_finefit as fit
from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, conditioned_copy, fit_input_normalizer,
)


def synthetic_g():
    x = np.linspace(.3, .9, 60)
    return np.stack((x, x + .03, x + .05), axis=1)[..., None].astype(np.float64)


def synthetic_splits():
    cells = np.repeat(np.arange(15), 4)
    return np.where(cells % 3 == 1, "test", "train"), cells


def test_plan_is_dry_fixed_and_label_free(tmp_path, capsys):
    output = tmp_path / "must_not_exist"
    assert fit.main(["--source-root", str(tmp_path / "absent_source"), "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_plan" and not output.exists()
    assert result["objective"] == "5*e01 + 2.5*e12 + 2.5*e02"
    assert result["optimizer"]["adam_steps"] == 10000
    assert result["optimizer"]["lbfgs_max_iter"] == 2000
    assert result["optimizer"]["lbfgs_max_eval"] == 8000
    assert result["CPU_threads"] == 1 and result["nice"] == 10
    for key in ("Q_constructed", "C_constructed", "GPU_used", "labels_used_by_optimizer", "TEST_used_by_optimizer"):
        assert result[key] is False
    assert result["cpu_budget_seconds"] == 600
    assert result["output_budget_bytes"] == 64 * 1024**2


def test_only_r_six_tensors_are_extracted_and_source_unchanged():
    raw = FrozenResidualMap(theory_dim=1, hidden=96)
    state = {f"R.{k}": v.clone() for k, v in raw.net.state_dict().items()}
    state["Q.placeholder"] = torch.tensor([17.], dtype=torch.float64)
    state["C.placeholder"] = torch.tensor([23.], dtype=torch.float64)
    before = fit.state_fingerprint(state)
    recovered = fit.extract_raw_r(state)
    assert fit.state_fingerprint(state) == before
    assert len(list(recovered.parameters())) == 6
    x = torch.tensor([[.3], [.8]], dtype=torch.float64)
    torch.testing.assert_close(recovered(x), raw(x), rtol=0, atol=0)
    assert not any(k.startswith(("Q", "C")) for k in recovered.state_dict())


@pytest.mark.parametrize("corruption", ["extra", "shape", "dtype", "nan"])
def test_extraction_fails_closed(corruption):
    raw = FrozenResidualMap(theory_dim=1, hidden=96)
    state = {f"R.{k}": v.clone() for k, v in raw.net.state_dict().items()}
    if corruption == "extra":
        state["R.surprise"] = torch.ones(1)
    elif corruption == "shape":
        state["R.0.weight"] = torch.ones(1, dtype=torch.float64)
    elif corruption == "dtype":
        state["R.0.weight"] = state["R.0.weight"].float()
    else:
        state["R.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit.extract_raw_r(state)


def test_split_normalization_never_uses_test_or_target_scale():
    direct = synthetic_g()
    splits, cells = synthetic_splits()
    train, mask = fit.split_frozen_g(direct, splits, cells)
    assert train.shape == (40, 3, 1)
    initial = fit_input_normalizer(train)
    direct[~mask] = 1e9
    direct[:, 2] = -1e8
    changed, _ = fit.split_frozen_g(direct, splits, cells)
    for a, b in zip(initial, fit_input_normalizer(changed)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not train.requires_grad


def test_split_rejects_condition_leakage():
    direct, (splits, cells) = synthetic_g(), synthetic_splits()
    splits[0], splits[4] = splits[4], splits[0]
    with pytest.raises(ValueError, match="whole-condition"):
        fit.split_frozen_g(direct, splits, cells)


def test_initial_function_and_full_derivative_preserved():
    train = torch.tensor(synthetic_g()[:40])
    raw = FrozenResidualMap(theory_dim=1, hidden=96)
    center, scale = fit_input_normalizer(train)
    model = conditioned_copy(raw, center, scale)
    receipt = fit.preservation_receipt(raw, model, train)
    assert max(receipt.values()) < 1e-12
    assert not model.center.requires_grad and not model.scale.requires_grad


def test_exact_allstart_weights_and_two_call_gradient():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))
        def advance(self, x):
            return self.a * x
    model, direct = Linear(), torch.tensor(synthetic_g()[:4])
    loss, terms = all_start_closure_loss(model, direct)
    torch.testing.assert_close(loss, 5*terms["e01"] + 2.5*terms["e12"] + 2.5*terms["e02"])
    terms["e02"].backward()
    expected = (4 * (model.a.detach()**2*direct[:, 0] - direct[:, 2])
                * model.a.detach() * direct[:, 0]).mean()
    torch.testing.assert_close(model.a.grad, expected)
    assert direct.grad is None


def test_metrics_use_common_train_target_std_and_identity():
    direct = synthetic_g()
    splits, _ = synthetic_splits()
    mask = splits == "train"
    predictions = np.stack((direct[:, 1]+.01, direct[:, 2]+.02, direct[:, 2]-.03), axis=1)
    result = fit.endpoint_metrics(direct, predictions, mask, [.1, .2])
    assert result["test"]["0_to_1"]["rmse"] == pytest.approx(.01)
    assert result["test"]["0_to_1"]["nrmse"] == pytest.approx(.1)
    assert result["test"]["1_to_2"]["nrmse"] == pytest.approx(.1)
    assert result["test"]["0_to_2"]["identity_rmse"] == pytest.approx(.05)


def test_short_synthetic_optimizer_full_train_fixed_endpoints():
    torch.set_num_threads(1)
    torch.manual_seed(172)
    direct = torch.tensor(synthetic_g()[:8])
    raw = FrozenResidualMap(theory_dim=1, hidden=96)
    center, scale = fit_input_normalizer(direct)
    model = conditioned_copy(raw, center, scale)
    before = all_start_closure_loss(model, direct)[0].item()
    rows, saved = [], []
    result = fit.train_full_batch(model, direct, budget=lambda: None,
        record=rows.append, checkpoint=lambda name, actual, opt: saved.append((name, actual.copy())),
        adam_steps=5, lbfgs_iterations=8, lbfgs_evaluations=20)
    assert result["adam_steps"] == 5 and result["adam_parameter_steps"] == [5]*6
    assert [name for name, _ in saved] == ["adam10000", "lbfgs"]
    assert result["lbfgs_closure_calls"] >= 1
    assert all_start_closure_loss(model, direct)[0].item() < before
    assert rows[-1]["phase"] == "lbfgs_endpoint"


def test_budget_stops_before_next_update_without_test_callback():
    direct = torch.tensor(synthetic_g()[:4])
    model = FrozenResidualMap(theory_dim=1, hidden=96)
    before = fit.state_fingerprint(model.state_dict())
    def stop():
        raise fit.BudgetHold("synthetic budget")
    with pytest.raises(fit.BudgetHold):
        fit.train_full_batch(model, direct, budget=stop, record=lambda row: None,
                             checkpoint=lambda *args: None, adam_steps=2)
    assert fit.state_fingerprint(model.state_dict()) == before


def test_lbfgs_budget_trial_is_rolled_back_to_accepted_adam():
    direct = torch.tensor(synthetic_g()[:6])
    model = FrozenResidualMap(theory_dim=1, hidden=96)
    phase = {"adam_done": False, "calls": 0}
    accepted = {}
    def save(name, actual, optimizer):
        if name == "adam10000":
            accepted.update({key: value.clone() for key, value in model.state_dict().items()})
            phase["adam_done"] = True
    def budget():
        if phase["adam_done"]:
            phase["calls"] += 1
            if phase["calls"] == 2:
                raise fit.BudgetHold("trial guard")
    with pytest.raises(fit.BudgetHold) as caught:
        fit.train_full_batch(model, direct, budget=budget, record=lambda row: None,
            checkpoint=save, adam_steps=1, lbfgs_iterations=8, lbfgs_evaluations=20)
    assert caught.value.retained_endpoint == "adam10000"
    assert caught.value.trial_parameters_rolled_back is True
    assert caught.value.actual["adam_steps"] == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, accepted[name], rtol=0, atol=0)


def make_mock_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run = source / "training"
    (run / "report").mkdir(parents=True)
    raw = FrozenResidualMap(theory_dim=1, hidden=96)
    state = {f"R.{key}": value.clone() for key, value in raw.net.state_dict().items()}
    state.update({"Q.mock": torch.ones(2, dtype=torch.float64), "C.mock": torch.zeros(3, dtype=torch.float64)})
    fingerprint = fit.state_fingerprint(state)
    modules = {name: fit.state_fingerprint({key[2:]: value for key, value in state.items() if key.startswith(name + ".")})
               for name in ("Q", "C", "R")}
    monkeypatch.setattr(fit, "SOURCE_STATE", fingerprint)
    monkeypatch.setattr(fit, "SOURCE_MODULES", modules)
    manifest = dict(schema_version=5, recipe=fit.SOURCE_RECIPE, options=dict(steps=3000),
        closure_objective="all_start_anchor", closure_scale_weights=None,
        labels_used_in_training=False, theory_dim=1, hidden=96, dtype="float64",
        evaluation_encoding="mean of Q(first half fields) and Q(second half fields); direct_full_g separately encodes pooled fields")
    summary = dict(recipe=fit.SOURCE_RECIPE, status="complete", actual_steps=3000,
        latest_checkpoint_step=3000, labels_used_in_training=False, final_state_sha256=fingerprint,
        final_module_sha256=modules, frozen_evaluation_verified=True, data_warning_count=12,
        optimizer_steps={name: dict(min=3000, max=3000) for name in ("Q", "C", "R")})
    complete = dict(status="complete", workflow_complete=True, returncode=0)
    fit.write_json(source / "job.json", dict(**complete, stages=[dict(status="complete", returncode=0)]))
    fit.write_json(run / "status.json", complete)
    fit.write_json(run / "manifest.json", manifest)
    fit.write_json(run / "summary.json", summary)
    direct = synthetic_g()
    splits, cells = synthetic_splits()
    grid = np.linspace(.2, 1., 501)[:, None]
    predictions = fit.predict(raw, direct)
    np.savez(run / "evaluation_arrays.npz", direct_g=direct, splits=splits, cell_ids=cells,
             flow_grid=grid, flow_next=raw(torch.tensor(grid)).detach().numpy(),
             one_step_g=predictions[:, :2],
             rollout_g=np.stack((direct[:, 0], predictions[:, 0], predictions[:, 2]), axis=1))
    (run / "report/coupling_by_scale.png").write_bytes(b"synthetic image fingerprint reference only")
    torch.save(dict(recipe=fit.SOURCE_RECIPE, step=3000, model=state, manifest=manifest), run / "training_step_3000.pt")
    files = ("training/manifest.json", "training/summary.json", "training/evaluation_arrays.npz", "training/report/coupling_by_scale.png")
    fit.write_json(source / "public_files.json", {name: fit.sha256_file(source / name) for name in files})
    return source, raw


def test_mock_checkpoint_manifest_module_and_prediction_verification(tmp_path, monkeypatch):
    source, raw = make_mock_source(tmp_path, monkeypatch)
    extracted, train, mask, arrays, manifest, summary, hashes, loading = fit.load_source(source)
    assert train.shape == (40, 3, 1) and mask.sum() == 40
    assert fit.state_fingerprint(extracted.net.state_dict()) == fit.SOURCE_MODULES["R"]
    assert fit.sha256_file(source / "training/training_step_3000.pt") == hashes["training/training_step_3000.pt"]
    assert "weights_only" in loading
    torch.testing.assert_close(raw(train[:, 0]), extracted(train[:, 0]))
    # A public-array substitution is detected before any optimizer runs.
    with (source / "training/evaluation_arrays.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Public source SHA"):
        fit.load_source(source)


def test_mock_whole_run_defers_test_evaluation_and_exports_only_safe_arrays(tmp_path, monkeypatch):
    source, _ = make_mock_source(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(fit.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(fit, "render_plots", lambda *args: None)
    optimized = {"finished": False}
    original_metrics = fit.endpoint_metrics
    def checked_metrics(*args):
        assert optimized["finished"], "TEST metrics must be deferred until optimization has finished"
        return original_metrics(*args)
    monkeypatch.setattr(fit, "endpoint_metrics", checked_metrics)
    def short_fit(model, train, *, budget, record, checkpoint):
        assert train.shape == (40, 3, 1) and not train.requires_grad
        record(dict(phase="adam", index=0, loss=1., e01=.1, e12=.1, e02=.1,
                    adam_steps=0, lbfgs_iterations=0, lbfgs_closure_calls=0))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        checkpoint("adam10000", dict(adam_steps=10000, lbfgs_iterations=0, lbfgs_closure_calls=0), optimizer)
        checkpoint("lbfgs", dict(adam_steps=10000, lbfgs_iterations=2, lbfgs_closure_calls=3), optimizer)
        optimized["finished"] = True
        return dict(adam_steps=10000, lbfgs_iterations=2, lbfgs_closure_calls=3,
                    stopping_reason="synthetic", gradient_converged=False)
    monkeypatch.setattr(fit, "train_full_batch", short_fit)
    output = tmp_path / "new_output"
    assert fit.run(fit.Options(str(source), str(output))) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["workflow_complete"] is True and status["returncode"] == 0
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["endpoint_metrics"]) == {"source", "adam10000", "lbfgs"}
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        assert "labels" not in arrays.files and "target_features" not in arrays.files
        assert arrays["direct_g"].shape == (60, 3, 1)
    public = json.loads((output / "public_files.json").read_text())
    assert not any(name.endswith(".pt") for name in public)
    assert all(fit.sha256_file(output / name) == digest for name, digest in public.items())


def test_output_cannot_nest_in_source(tmp_path):
    source = tmp_path / "immutable"
    source.mkdir()
    with pytest.raises(ValueError, match="independent"):
        fit.run(fit.Options(str(source), str(source / "overwrite")))


def test_implementation_does_not_import_qc_or_sampling_modules():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(fit.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "recurrent_rg.phi4_r_calibration" in imports
    assert not any(name and ("aligned_qcr" in name or "hopping_loader" in name or "phi4_qcr" in name) for name in imports)
