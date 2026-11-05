"""Independent CPU checks of the explicitly nondefault scalar-closure ablation.

These fixtures contain statistics only: no physical configurations, coupling
labels, sampler calls, model inference on external data, or CUDA admission.
"""
import json
import shutil

import numpy as np
import pytest
import torch

from experiments import phi4_scalar_closure_ablation as ablation
from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_qcr import FEATURE_DIM, Phi4QCR


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5721)
        model = Phi4QCR(.8, hidden=4)
        first = torch.randn(8, 3, FEATURE_DIM, dtype=torch.float64) * .2
        second = first + .05 * torch.randn_like(first)
        guard = torch.randn(2, 12, 3, FEATURE_DIM, dtype=torch.float64) * .3
    reference = torch.tensor([.3, .4, .5], dtype=torch.float64)
    return model, first, second, guard, reference


def manual_closure(model, g):
    """Independent expansion of uniform target/all-earlier-start weights."""
    e01 = (model.advance(g[:, 0]) - g[:, 1]).square().sum(-1).mean()
    e12 = (model.advance(g[:, 1]) - g[:, 2]).square().sum(-1).mean()
    e02 = (model.advance(model.advance(g[:, 0])) - g[:, 2]).square().sum(-1).mean()
    return 5. * e01 + 2.5 * e12 + 2.5 * e02


def manual_q_base(model, first, second):
    ga, gb = model.Q.from_features(first.detach()), model.Q.from_features(second.detach())
    g = .5 * (ga + gb)
    target = .5 * (first.detach() + second.detach())
    reconstruction = (model.Q.decode(g) - target).square().sum(-1).mean()
    subset = (ga - gb).square().sum(-1).mean(0)
    return 10. * reconstruction + 20. * subset[0] + 5. * subset[1:].mean(), g


def manual_guard(model, guard_features, reference_std, guard_scale):
    # Both fixed draws enter the frozen TRAIN guard.  Physical systems are
    # not pooled into a single latent and scales keep separate standard deviations.
    g = model.Q.from_features(guard_features.detach().reshape(-1, 3, FEATURE_DIM))
    current_std = g.std(0, correction=0).squeeze(-1)
    return guard_scale * torch.relu(1. - current_std / (.5 * reference_std)).square().mean()


def test_closure_is_exact_all_start_anchor_not_prediction_prediction(inputs):
    model = inputs[0]
    g = torch.tensor([[[-.7], [.4], [.9]], [[.1], [-.2], [.3]]], dtype=torch.float64)
    actual = ablation.closure_objective(model, g)
    torch.testing.assert_close(actual, manual_closure(model, g), atol=1e-14, rtol=1e-14)


def test_two_step_closure_retains_both_R_parameter_derivatives(inputs):
    model = inputs[0]
    model.R = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.R.weight.fill_(.2)
    g = torch.tensor([[[.7], [.3], [.1]]], dtype=torch.float64)
    actual = ablation.closure_objective(model, g)
    actual.backward()
    factor = 1.2
    expected = (10. * (factor * .7 - .3) * .7
                + 5. * (factor * .3 - .1) * .3
                + 5. * (factor ** 2 * .7 - .1) * (2. * factor * .7))
    assert model.R.weight.grad.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_frozen_R_keeps_input_and_both_endpoint_gradients(inputs):
    model = inputs[0]
    model.R = torch.nn.Linear(1, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.R.weight.fill_(.2)
        model.R.bias.fill_(.05)
    original_flags = [parameter.requires_grad for parameter in model.R.parameters()]
    before = state_fingerprint(model)
    g = torch.tensor([[[.7], [.3], [.1]], [[-.2], [.1], [.4]]],
                     dtype=torch.float64, requires_grad=True)
    expected_g = g.detach().clone().requires_grad_()
    expected = manual_closure(model, expected_g)
    expected_gradient, = torch.autograd.grad(expected, expected_g)
    actual = ablation.closure_objective(model, g, frozen_R=True)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    actual.backward()
    torch.testing.assert_close(g.grad, expected_gradient, atol=1e-13, rtol=1e-13)
    assert torch.all(g.grad.abs().sum((0, 2)) > 0), "All three scale endpoints must receive Q feedback"
    assert all(parameter.grad is None for parameter in model.R.parameters())
    assert [parameter.requires_grad for parameter in model.R.parameters()] == original_flags
    assert state_fingerprint(model) == before


def test_frozen_R_never_changes_preexisting_requires_grad_policy(inputs):
    model = inputs[0]
    next(model.R.parameters()).requires_grad_(False)
    flags = [parameter.requires_grad for parameter in model.R.parameters()]
    g = torch.randn(4, 3, 1, dtype=torch.float64, requires_grad=True)
    ablation.closure_objective(model, g, frozen_R=True).backward()
    assert [parameter.requires_grad for parameter in model.R.parameters()] == flags
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert all(parameter.grad is None for parameter in model.R.parameters())


@pytest.mark.parametrize("closure_enabled", [False, True])
def test_paired_objective_exact_original_Q_plus_named_ablation_terms(inputs, closure_enabled):
    model, first, second, guard, reference = inputs
    guard_scale, alpha = 1.7, .8
    q, r, metrics = ablation.paired_objectives(model, first, second, reference,
        guard_scale, alpha, closure_enabled, guard_features=guard)
    base, g = manual_q_base(model, first, second)
    expected = base + manual_guard(model, guard, reference, guard_scale)
    if closure_enabled:
        expected = expected + alpha * manual_closure(model, g)
    torch.testing.assert_close(q, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(r, manual_closure(model, g.detach()), atol=1e-13, rtol=1e-13)
    assert metrics and all(isinstance(value, torch.Tensor) for value in metrics.values())


@pytest.mark.parametrize("branch", ["Q", "R"])
@pytest.mark.parametrize("closure_enabled", [False, True])
def test_paired_gradients_are_independent_and_statistics_are_detached(inputs, monkeypatch,
                                                                    branch, closure_enabled):
    model, first, second, guard, reference = inputs
    first.requires_grad_()
    second.requires_grad_()
    guard.requires_grad_()
    before = state_fingerprint(model)

    def forbidden_C(*args, **kwargs):
        raise AssertionError("Frozen-C diagnostic must never invoke C")

    monkeypatch.setattr(model.C, "forward", forbidden_C)
    q, r, _ = ablation.paired_objectives(model, first, second, reference,
        1.7, .8, closure_enabled, guard_features=guard)
    {"Q": q, "R": r}[branch].backward()
    for name in ("Q", "C", "R"):
        grads = [parameter.grad for parameter in getattr(model, name).parameters()]
        if name == branch:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in grads)
            assert all(gradient is None or torch.isfinite(gradient).all() for gradient in grads)
        else:
            assert all(gradient is None for gradient in grads)
    assert first.grad is None and second.grad is None and guard.grad is None
    assert state_fingerprint(model) == before


def test_R_loss_is_unchanged_when_Q_closure_is_enabled(inputs):
    model, first, second, guard, reference = inputs
    a = ablation.paired_objectives(model, first, second, reference, 1.7, .8, False,
                                  guard_features=guard)
    b = ablation.paired_objectives(model, first, second, reference, 1.7, .8, True,
                                  guard_features=guard)
    torch.testing.assert_close(a[1], b[1], atol=0., rtol=0.)
    grads_a = torch.autograd.grad(a[1], tuple(model.R.parameters()))
    grads_b = torch.autograd.grad(b[1], tuple(model.R.parameters()))
    for ga, gb in zip(grads_a, grads_b):
        torch.testing.assert_close(ga, gb, atol=0., rtol=0.)


def test_alpha_zero_is_exact_paired_control_in_loss_and_gradients(inputs):
    model, first, second, guard, reference = inputs
    a = ablation.paired_objectives(model, first, second, reference, 1.7, 0., False,
                                  guard_features=guard)
    b = ablation.paired_objectives(model, first, second, reference, 1.7, 0., True,
                                  guard_features=guard)
    torch.testing.assert_close(a[0], b[0], atol=0., rtol=0.)
    grads_a = torch.autograd.grad(a[0], tuple(model.Q.parameters()))
    grads_b = torch.autograd.grad(b[0], tuple(model.Q.parameters()))
    for ga, gb in zip(grads_a, grads_b):
        torch.testing.assert_close(ga, gb, atol=0., rtol=0.)


def test_guard_uses_external_full_TRAIN_per_scale_population_std(inputs):
    model, first, second, guard, _ = inputs
    with torch.no_grad():
        current = model.Q.from_features(guard.reshape(-1, 3, FEATURE_DIM)).std(0, correction=0).squeeze(-1)
    # Ratios to the floor are 1/4, 1/2 and 2.  Only the first two scales pay.
    reference = current * torch.tensor([8., 4., 1.], dtype=torch.float64)
    actual, _, _ = ablation.paired_objectives(model, first, second, reference,
        2.4, 0., False, guard_features=guard)
    base, _ = manual_q_base(model, first, second)
    expected_penalty = 2.4 * (.75 ** 2 + .5 ** 2) / 3.
    torch.testing.assert_close(actual - base, actual.new_tensor(expected_penalty),
                               atol=1e-13, rtol=1e-13)
    # Changing minibatch diversity cannot change a guard evaluated on all TRAIN.
    identical_first = first[:1].expand_as(first)
    identical_second = second[:1].expand_as(second)
    other, _, _ = ablation.paired_objectives(model, identical_first, identical_second,
        reference, 2.4, 0., False, guard_features=guard)
    other_base, _ = manual_q_base(model, identical_first, identical_second)
    torch.testing.assert_close(other - other_base, actual - base, atol=1e-13, rtol=1e-13)


def test_guard_is_same_in_both_arms_and_zero_above_floor(inputs):
    model, first, second, guard, _ = inputs
    with torch.no_grad():
        reference = model.Q.from_features(guard.reshape(-1, 3, FEATURE_DIM)).std(0, correction=0).squeeze(-1)
    a, _, _ = ablation.paired_objectives(model, first, second, reference, 3., .8, False,
                                        guard_features=guard)
    b, _, _ = ablation.paired_objectives(model, first, second, reference, 3., .8, True,
                                        guard_features=guard)
    base, g = manual_q_base(model, first, second)
    torch.testing.assert_close(a, base, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(b - a, .8 * manual_closure(model, g), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_paired_objective_rejects_nonfinite_statistics(inputs, bad):
    model, first, second, guard, reference = inputs
    first = first.clone()
    first[0, 0, 0] = bad
    with pytest.raises((ValueError, FloatingPointError), match="[Ff]inite|[Nn]an|[Ii]nf"):
        ablation.paired_objectives(model, first, second, reference, 1., .8, True,
                                  guard_features=guard)


@pytest.mark.parametrize("which", ["reference", "guard", "alpha"])
def test_paired_objective_rejects_nonfinite_frozen_calibration(inputs, which):
    model, first, second, guard, reference = inputs
    guard_scale, alpha = 1., .8
    if which == "reference":
        reference = reference.clone()
        reference[0] = float("nan")
    elif which == "guard":
        guard = guard.clone()
        guard[0, 0, 0, 0] = float("inf")
    else:
        alpha = float("nan")
    with pytest.raises((ValueError, FloatingPointError), match="[Ff]inite|[Nn]an|[Ii]nf"):
        ablation.paired_objectives(model, first, second, reference, guard_scale, alpha, True,
                                  guard_features=guard)


@pytest.mark.parametrize("bad_shape", [(8, 2, FEATURE_DIM), (8, 3, FEATURE_DIM - 1),
                                        (3, FEATURE_DIM), (0, 3, FEATURE_DIM)])
def test_paired_objective_rejects_incompatible_half_statistics(inputs, bad_shape):
    model, _, _, guard, reference = inputs
    first = torch.zeros(bad_shape, dtype=torch.float64)
    with pytest.raises(ValueError):
        ablation.paired_objectives(model, first, first, reference, 1., .8, True,
                                  guard_features=guard)


@pytest.fixture
def completed_statistics_fixture(tmp_path):
    """Tiny synthetic file contract, not a claimed physical/model result."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(928)
        model = Phi4QCR(.8, hidden=4)
    run = tmp_path / "completed_source_fixture"
    run.mkdir()
    manifest = dict(theory_dim=1, hidden=4, scale=.8, lattice_sizes=[36, 12, 4],
                    data_manifest_sha256="synthetic-statistics-fixture")
    write_json(run / "manifest.json", manifest)
    torch.save(dict(step=3000, model=model.state_dict()), run / "training_step_3000.pt")
    rng = np.random.default_rng(782)
    train = rng.normal(.01, .02, size=(2, 8, 3, 12, FEATURE_DIM))
    arrays = dict(train_per_sample=train)
    source_evaluation = dict(flow_g=np.linspace(-1., 1., 8)[:, None],
        flow_R=np.linspace(-.9, 1.1, 8)[:, None], flow_displacement=np.full((8, 1), .1))
    for split, size, offset in (("train", 8, 0), ("test", 4, 10)):
        first = rng.normal(.01, .02, size=(size, 3, FEATURE_DIM))
        second = first + rng.normal(0., .001, size=first.shape)
        arrays[f"eval_{split}_first"] = first
        arrays[f"eval_{split}_second"] = second
        ids = np.arange(size) // 2 + offset
        arrays[f"eval_{split}_labels"] = np.column_stack((ids / 100.,
            np.full(size, .2), np.zeros(size)))
        arrays[f"eval_{split}_cell_ids"] = ids
        arrays[f"eval_{split}_chain_ids"] = np.arange(size)
        source_evaluation[f"{split}_feature_target"] = .5 * (first + second)
    np.savez(run / "evaluation_arrays.npz", **source_evaluation)
    write_json(run / "summary.json", dict(status="complete", actual_steps=3000,
        final_state_sha256=state_fingerprint(model), numerical_fixed_points=[],
        equilibrium_certified=False, data_warning_count=11))
    write_json(run / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        summary_sha256=sha256_file(run / "summary.json"),
        evaluation_arrays_sha256=sha256_file(run / "evaluation_arrays.npz")))
    cache = tmp_path / "completed_cache_fixture"
    cache.mkdir()
    np.savez(cache / "arrays.npz", **arrays)
    receipt = dict(source_run=str(run), scale=.8,
        source_checkpoint_sha256=sha256_file(run / "training_step_3000.pt"),
        source_state_sha256=state_fingerprint(model),
        source_manifest_sha256=sha256_file(run / "manifest.json"),
        source_data_manifest_sha256=manifest["data_manifest_sha256"],
        arrays_sha256=sha256_file(cache / "arrays.npz"))
    write_json(cache / "receipt.json", receipt)
    write_json(cache / "status.json", dict(status="complete",
        receipt_sha256=sha256_file(cache / "receipt.json")))
    return run, cache


def test_small_CPU_run_has_paired_initialization_fixed_steps_and_immutable_source(
        completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    protected = {path: sha256_file(path) for root in (run, cache) for path in root.iterdir()}
    output = tmp_path / "scalar_closure_smoke"
    summary = ablation.train(ablation.Options(str(run), str(cache), str(output), steps=3,
        sample_batch=8, log_every=2, cpu_budget_seconds=30.))
    assert summary["status"] == "complete"
    assert summary["actual_steps_by_arm"] == {"A": 3, "B": 3}
    assert summary["frozen_source_verified"] and summary["source_gradients_none"]
    assert summary["source_state_sha256_before"] == summary["source_state_sha256_after"]
    assert summary["labels_used_in_training"] is False
    assert summary["C_updates"] == summary["physical_sampling_calls"] == summary["MI_calls"] == 0
    assert summary["GPU_used"] is False and summary["default_recipe_modified"] is False
    assert summary["thermal_nu"] is None and summary["critical_r"] is None
    for name in ("A", "B"):
        arm = summary["arms"][name]
        assert arm["actual_steps"] == 3 and arm["source_actual_steps"] == 3000
        for optimizer in ("Q", "R"):
            assert arm["optimizer_steps"][optimizer]["min"] == 3
            assert arm["optimizer_steps"][optimizer]["max"] == 3
        assert arm["initial_module_sha256"] == summary["arms"]["source"]["initial_module_sha256"]
        assert arm["initial_module_sha256"]["C"] == arm["final_module_sha256"]["C"]
        assert (output / name / "training_step_3.pt").exists()
    for path, digest in protected.items():
        assert sha256_file(path) == digest
    status = json.loads((output / "status.json").read_text())
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert status["summary_sha256"] == sha256_file(output / "summary.json")
    assert status["evaluation_arrays_sha256"] == sha256_file(output / "evaluation_arrays.npz")
    history = json.loads((output / "history_progress.json").read_text())
    assert history["latest_checkpoint_step"] == 3
    assert [row["step"] for row in history["history"]] == [0, 1, 2, 3]
    initial = history["history"][0]["arms"]
    for key in ("q_base", "q_guard", "q_compatible_closure", "r_loss"):
        assert initial["A"][key] == initial["B"][key]
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["A_test_direct"].shape == (4, 3, 1)
        np.testing.assert_array_equal(arrays["A_train_feature_target"], arrays["B_train_feature_target"])
        np.testing.assert_array_equal(arrays["source_flow_g"], np.linspace(-1., 1., 8)[:, None])


def test_existing_output_is_never_overwritten(completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        ablation.train(ablation.Options(str(run), str(cache), str(output), steps=1, sample_batch=8))


def test_budget_hold_never_reduces_fixed_steps_or_changes_source(
        completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    original = sha256_file(run / "training_step_3000.pt")
    output = tmp_path / "held"
    result = ablation.train(ablation.Options(str(run), str(cache), str(output), steps=3,
        sample_batch=8, cpu_budget_seconds=1e-12))
    assert result["status"] == "budget_hold"
    status = json.loads((output / "status.json").read_text())
    assert status["requires_user_action"] and not status["workflow_complete"]
    assert status["actual_steps_by_arm"] == {"A": 0, "B": 0}
    assert sha256_file(run / "training_step_3000.pt") == original
    assert not (output / "summary.json").exists()


def test_coupling_metadata_is_opened_only_after_all_updates(
        completed_statistics_fixture, tmp_path, monkeypatch):
    run, cache = completed_statistics_fixture
    original_load, original_step = np.load, torch.optim.Adam.step
    completed_updates = 0
    metadata_reads = []

    def counted_step(optimizer, *args, **kwargs):
        nonlocal completed_updates
        result = original_step(optimizer, *args, **kwargs)
        completed_updates += 1
        return result

    class TimingCheckedArchive:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            if key in ablation.EVALUATION_METADATA:
                assert completed_updates == 12  # 3 steps x 2 arms x Q/R.
                metadata_reads.append(key)
            return self.archive[key]

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    monkeypatch.setattr(ablation.np, "load", TimingCheckedArchive)
    result = ablation.train(ablation.Options(str(run), str(cache), str(tmp_path / "labels_late"),
        steps=3, sample_batch=8, log_every=3, cpu_budget_seconds=30.))
    assert result["status"] == "complete" and completed_updates == 12
    assert set(metadata_reads) == set(ablation.EVALUATION_METADATA)


def _fork_synthetic_source(run, destination, mutation):
    """Make a separate synthetic completed endpoint for cache-identity tests."""
    shutil.copytree(run, destination)
    saved = torch.load(destination / "training_step_3000.pt", map_location="cpu", weights_only=True)
    model = Phi4QCR(.8, hidden=4)
    model.load_state_dict(saved["model"])
    with torch.no_grad():
        model.Q.net[-1].bias.add_(.1)
        model.R[-1].bias.add_(.2)
        if mutation == "C":
            model.C.net[-1].bias.add_(.01)
    saved["model"] = model.state_dict()
    torch.save(saved, destination / "training_step_3000.pt")
    summary = json.loads((destination / "summary.json").read_text())
    summary["final_state_sha256"] = state_fingerprint(model)
    write_json(destination / "summary.json", summary)
    if mutation == "data":
        manifest = json.loads((destination / "manifest.json").read_text())
        manifest["data_manifest_sha256"] = "other-synthetic-data"
        write_json(destination / "manifest.json", manifest)
    status = json.loads((destination / "status.json").read_text())
    status["summary_sha256"] = sha256_file(destination / "summary.json")
    write_json(destination / "status.json", status)
    return model


def test_cache_can_be_reused_for_different_QR_only_with_exact_same_C_and_targets(
        completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    latest = tmp_path / "different_QR_same_C"
    expected = _fork_synthetic_source(run, latest, "QR")
    features, receipt = ablation.load_cache(cache)
    actual, _, _, protected = ablation.load_source_for_cache(latest, cache, features, receipt)
    assert state_fingerprint(actual) == state_fingerprint(expected)
    assert str(run / "training_step_3000.pt") in protected
    assert str(latest / "training_step_3000.pt") in protected
    assert str(cache / "arrays.npz") in protected
    assert all(not parameter.requires_grad for parameter in actual.parameters())


@pytest.mark.parametrize("mutation", ["C", "data"])
def test_cache_reuse_rejects_changed_C_or_physical_data_contract(
        completed_statistics_fixture, tmp_path, mutation):
    run, cache = completed_statistics_fixture
    latest = tmp_path / f"different_{mutation}"
    _fork_synthetic_source(run, latest, mutation)
    features, receipt = ablation.load_cache(cache)
    with pytest.raises(ValueError, match="Cached C|physical data"):
        ablation.load_source_for_cache(latest, cache, features, receipt)


def test_cache_reuse_rejects_unpaired_evaluation_targets(completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    latest = tmp_path / "unpaired_latest"
    _fork_synthetic_source(run, latest, "QR")
    with np.load(latest / "evaluation_arrays.npz", allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values["test_feature_target"][0, 0, 0] += 1e-9
    np.savez(latest / "evaluation_arrays.npz", **values)
    status = json.loads((latest / "status.json").read_text())
    status["evaluation_arrays_sha256"] = sha256_file(latest / "evaluation_arrays.npz")
    write_json(latest / "status.json", status)
    features, receipt = ablation.load_cache(cache)
    with pytest.raises(ValueError, match="exactly paired"):
        ablation.load_source_for_cache(latest, cache, features, receipt)


def test_archive_tampering_is_rejected_before_updates(completed_statistics_fixture, tmp_path):
    run, cache = completed_statistics_fixture
    original = sha256_file(run / "training_step_3000.pt")
    with (cache / "arrays.npz").open("ab") as handle:
        handle.write(b"synthetic-tamper")
    output = tmp_path / "tampered"
    with pytest.raises(ValueError, match="SHA256"):
        ablation.train(ablation.Options(str(run), str(cache), str(output), steps=3, sample_batch=8))
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["actual_steps_by_arm"] == {"A": 0, "B": 0}
    assert sha256_file(run / "training_step_3000.pt") == original
